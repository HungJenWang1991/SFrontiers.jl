# Regression tests: predict(:marginal) and (:all) for :MLE Exponential fits.
#
# predict(what = :marginal) raised "the kernel path for `ineff=:Exponential` supports the MCI /
# MSLE backends only; got backend `:MLE`" for a cross-sectional Normal + Exponential fit made with
# method = :MLE, and what = :all raised the same error when σᵤ² is heteroscedastic, although the
# fit itself reports its marginal effects in result.marginal. _predict_marginal_kernel now serves
# that fit with the fit's own E(u) = σᵤ (MLE_Backend's marg_expo, the function get_marg
# differentiates for result.marginal). On the fit's zvar the non-constant columns of
# predict(:marginal) equal result.marginal bit for bit, also when zvar has no constant
# column; at any z they equal ∂σᵤ/∂z = δ·σᵤ/2
# (512-bit references), and zero where σᵤ² = exp(z'δ) is held at its clamp [1e-12, 1e12];
# what = :all returns the six targets. A homoscedastic fit takes a one-column zvar, as under
# :MCI and :MSLE, and its what = :all keeps its five targets.

using Test
using SFrontiers
using Random
using Statistics: mean
using Optim

@testset "predict(:marginal) and (:all) for :MLE Exponential fits" begin
    quiet(f) = redirect_stdout(f, devnull)
    # predict, or `nothing` when it throws, so that the package before the fix fails the asserts
    tryp(r; kw...) = try
        quiet(() -> predict(r; kw...))
    catch
        nothing
    end
    errtext(f) = try
        quiet(f); ""
    catch e
        sprint(showerror, e)
    end
    fitq(sp; kw...) = quiet(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE),
                                              show_table = false, verbose = false; kw...))
    # δ, the ln σᵤ² coefficients, at the positions predict reads them from
    sdelta(r, sp) = r.coeff[sp.mci_spec.idx.ineff.sigma_sq]
    # 512-bit reference of ∂σᵤ/∂z = δ·σᵤ/2 with σᵤ = √exp(z'δ); zero where σᵤ² is clamped.
    # (Its variables are declared local: a function defined in a testset shares the
    # testset's variables, and `n` is one of them.)
    function ref_marg(δ, Zr)
        setprecision(BigFloat, 512) do
            local nr, nc = size(Zr)
            local out = zeros(nr, nc)
            for i in 1:nr
                local w = sum(big(δ[j]) * big(Float64(Zr[i, j])) for j in 1:nc)
                local s2 = exp(w)
                (big(1e-12) <= s2 <= big(1e12)) || continue
                out[i, :] .= Float64.(big.(δ) .* sqrt(s2) ./ 2)
            end
            out
        end
    end
    relerr(a, b) = maximum(abs(a[k] - b[k]) / (b[k] == 0 ? 1.0 : abs(b[k])) for k in eachindex(b))
    ALL6 = (:frontier, :residuals, :jlms, :bc, :response, :marginal)
    ALL5 = (:frontier, :residuals, :jlms, :bc, :response)

    rng = Xoshiro(38)
    n = 300
    x = randn(rng, n); z1 = randn(rng, n); z2 = rand(rng, n)
    X = hcat(ones(n), x)
    Z = hcat(ones(n), z1)
    y = X * [1.0, 0.5] .+ 0.25 .* randn(rng, n) .- exp.(0.5 .* (-1.2 .+ 0.6 .* z1)) .* randexp(rng, n)

    @testset "heteroscedastic σᵤ², one z: result.marginal bit for bit, and :all" begin
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal, ineff = :Exponential,
                          hetero = [:sigma_u_sq], type = :prod)
        r = fitq(sp)
        M = Matrix{Float64}(r.marginal)
        pm = tryp(r; zvar = Z, what = :marginal)
        @test pm isa Matrix{Float64} && size(pm) == size(Z)
        @test pm !== nothing && isequal(pm[:, 2:end], M)
        @test pm !== nothing && round(mean(pm[:, 2]); digits = 5) == r.marginal_mean[1]
        @test pm !== nothing && relerr(pm, ref_marg(sdelta(r, sp), Z)) < 1e-13
        pa = tryp(r; frontier = X, depvar = y, zvar = Z, what = :all)
        @test pa isa NamedTuple && keys(pa) == ALL6
        @test pa !== nothing && isequal(pa.marginal, pm)
        @test pa !== nothing && isequal(pa.jlms, vec(r.jlms)) && isequal(pa.bc, vec(r.bc))
        @test pa !== nothing &&
              isequal(pa.frontier, predict(r; frontier = X, what = :frontier)) &&
              isequal(pa.residuals, predict(r; frontier = X, depvar = y, what = :residuals)) &&
              isequal(pa.response, predict(r; frontier = X, depvar = y, zvar = Z, what = :response))
        # new data: the reference, and the zvar rule of :MCI and :MSLE
        g = collect(range(-3.0, 3.0; length = 25))
        Zg = hcat(ones(25), g)
        pg = tryp(r; zvar = Zg, what = :marginal)
        @test pg !== nothing && size(pg) == (25, 2) && relerr(pg, ref_marg(sdelta(r, sp), Zg)) < 1e-13
        @test occursin("has 1 columns but model was fitted with 2",
                       errtext(() -> predict(r; zvar = reshape(g, :, 1), what = :marginal)))
    end

    @testset "heteroscedastic σᵤ², two z, cost frontier, held coefficients, rows at the clamp" begin
        Z2 = hcat(ones(n), z1, z2)
        Z2[1, 2] = 37.0      # z'δ from 28.0 to 28.6: σᵤ² above 1e12 (ln 1e12 = 27.63), held there
        Z2[2, 2] = 40.0      # z'δ from 30.4 to 31.0
        Z2[3, 2] = -34.0     # z'δ from -28.8 to -28.2: σᵤ² below 1e-12, held there
        δ = [-1.0, 0.8, -0.6]
        yc = X * [0.5, 0.8] .+ 0.25 .* randn(rng, n) .+ 0.4 .* randexp(rng, n)
        sp = sfmodel_spec(depvar = yc, frontier = X, zvar = Z2, noise = :Normal, ineff = :Exponential,
                          hetero = [:sigma_u_sq], type = :cost)
        ini = sfmodel_init(spec = sp, frontier = [0.5, 0.8], ln_sigma_u_sq = δ, ln_sigma_v_sq = log(0.0625))
        hold = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))   # no warmstart
        r = fitq(sp; init = ini, optim_options = hold)
        @test isequal(sdelta(r, sp), δ)               # the fit held the coefficients
        M = Matrix{Float64}(r.marginal)
        pm = tryp(r; zvar = Z2, what = :marginal)
        @test pm isa Matrix{Float64} && size(pm) == (n, 3)
        @test pm !== nothing && isequal(pm[:, 2:end], M)
        @test pm !== nothing && all(iszero, pm[1:3, :]) && all(iszero, M[1:3, :])
        @test pm !== nothing && all(!iszero, pm[4:end, :])
        @test pm !== nothing && relerr(pm, ref_marg(δ, Z2)) < 1e-13
        pa = tryp(r; frontier = X, depvar = yc, zvar = Z2, what = :all)
        @test pa isa NamedTuple && keys(pa) == ALL6 && isequal(pa.marginal, pm)
    end

    @testset "homoscedastic σᵤ²: one column, as under :MCI and :MSLE; :all keeps five targets" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :Exponential, type = :prod)
        r = fitq(sp)
        on = ones(n, 1)
        pm = tryp(r; zvar = on, what = :marginal)
        @test pm isa Matrix{Float64} && size(pm) == (n, 1)
        # the derivative with respect to the constant column, δ·σᵤ/2, the same in every row
        @test pm !== nothing && relerr(pm, ref_marg(sdelta(r, sp), on)) < 1e-13
        @test occursin("has 2 columns but model was fitted with 1",
                       errtext(() -> predict(r; zvar = Z, what = :marginal)))
        p5 = tryp(r; frontier = X, depvar = y, zvar = Z, what = :all)
        @test p5 isa NamedTuple && keys(p5) == ALL5
        p5b = tryp(r; frontier = X, depvar = y, what = :all)
        @test p5b isa NamedTuple && keys(p5b) == ALL5
    end

    @testset "Float32 data: result.marginal bit for bit" begin
        sp = sfmodel_spec(depvar = Float32.(y), frontier = Float32.(X), zvar = Float32.(Z),
                          noise = :Normal, ineff = :Exponential, hetero = [:sigma_u_sq], type = :prod)
        r = fitq(sp)
        pm = tryp(r; zvar = Float32.(Z), what = :marginal)
        @test pm isa Matrix{Float64} && isequal(pm[:, 2:end], Matrix{Float64}(r.marginal))
        pa = tryp(r; frontier = Float32.(X), depvar = Float32.(y), zvar = Float32.(Z), what = :all)
        @test pa isa NamedTuple && keys(pa) == ALL6 && isequal(pa.marginal, pm)
    end

    @testset "heteroscedastic σᵤ², one z and no constant column, held coefficients" begin
        # σᵤᵢ² = exp(zᵢδ) with zvar = z alone: every column of predict(:marginal) is a column
        # of result.marginal. Only :marginal and the keys of :all are checked here. This
        # layout's :jlms, :bc and :response (and so those fields of :all) ignored z in an
        # earlier form of predict; test_mle_expo_onecol_predict.jl tests them.
        Z1 = reshape(z1, :, 1)
        δ1 = [0.7]
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = Z1, noise = :Normal, ineff = :Exponential,
                          hetero = [:sigma_u_sq], type = :prod)
        ini = sfmodel_init(spec = sp, frontier = [1.0, 0.5], ln_sigma_u_sq = δ1, ln_sigma_v_sq = log(0.0625))
        hold = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))   # no warmstart
        r = fitq(sp; init = ini, optim_options = hold)
        @test isequal([sdelta(r, sp);], δ1)           # the fit held the coefficients
        pm = tryp(r; zvar = Z1, what = :marginal)
        @test pm isa Matrix{Float64} && size(pm) == (n, 1)
        @test pm !== nothing && isequal(pm, Matrix{Float64}(r.marginal))
        @test pm !== nothing && relerr(pm, ref_marg(δ1, Z1)) < 1e-13
        pa = tryp(r; frontier = X, depvar = y, zvar = Z1, what = :all)
        @test pa isa NamedTuple && keys(pa) == ALL6 && isequal(pa.marginal, pm)
    end
end
