# Regression tests: predict for :MLE Exponential with a one-column zvar without a constant.
#
# For a cross-sectional Normal + Exponential fit made with method = :MLE whose σᵤ² depends on a
# single zvar column without a constant (hetero = [:sigma_u_sq], zvar = z, so σᵤᵢ² = exp(zᵢδ)),
# predict's closed form (_predict_eff_mle_exp) chose its homoscedastic branch by the number of
# ln σᵤ² coefficients. This model also has one, so predict ignored zvar and used σᵤ² = exp(δ) in
# every row: :jlms, :bc, :response and those entries of :all differed from result.jlms /
# result.bc in every row, and :jlms without zvar was accepted. The branch is now chosen by the
# model (`:sigma_u_sq in hetero`), as for the half normal and truncated normal. Checked here: on
# the training data the four targets equal the fit's values bit for bit (production and cost
# frontiers); on new data they match the closed form at 512 bits; without zvar they raise the
# error the other heteroscedastic models give; the layouts whose branch did not change
# (homoscedastic, zvar with a constant, the half normal) still reproduce the fit.

using Test
using SFrontiers
using Random
using Optim
using SpecialFunctions: erfc

@testset "predict for :MLE Exponential with a one-column zvar without a constant" begin
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
    # Normal + Exponential JLMS and BC at 512 bits (production convention ε = v - u), with
    # σᵤᵢ = √exp(zᵢδ) per row; coefficients in the MLE order (β, δ, ln σᵥ²) of this model.
    function ref_expo(c, Xr, yr, zr, sg)
        setprecision(BigFloat, 512) do
            local b = big.(c)
            local sv = sqrt(exp(b[end]))
            local nr = length(yr)
            local jl = zeros(nr); local bcv = zeros(nr)
            for i in 1:nr
                local ep = sg * (big(yr[i]) - sum(big(Xr[i, j]) * b[j] for j in 1:size(Xr, 2)))
                local su = sqrt(exp(big(zr[i]) * b[end-1]))
                local mus = -ep - sv^2 / su
                local a = mus / sv
                local Pa = erfc(-a / sqrt(big(2))) / 2
                local pa = exp(-a^2 / 2) / sqrt(2 * big(pi))
                jl[i] = Float64(sv * pa / Pa + mus)
                bcv[i] = Float64(exp(-mus + sv^2 / 2) * (erfc(-(a - sv) / sqrt(big(2))) / 2) / Pa)
            end
            jl, bcv
        end
    end
    relerr(a, b) = maximum(abs(a[k] - b[k]) / abs(b[k]) for k in eachindex(b))
    ALL6 = (:frontier, :residuals, :jlms, :bc, :response, :marginal)

    rng = Xoshiro(47)
    n = 300
    x = randn(rng, n); z = 1.2 .* randn(rng, n)
    X = hcat(ones(n), x)
    Z1 = reshape(z, :, 1)
    v = 0.3 .* randn(rng, n); u = exp.(0.5 .* (1.1 .* z)) .* randexp(rng, n)
    y = X * [1.0, 0.5] .+ v .- u
    # new data
    m = 40
    xn = randn(rng, m); zn = collect(range(-2.5, 2.5; length = m))
    Xn = hcat(ones(m), xn)
    yn = Xn * [1.0, 0.5] .+ 0.3 .* randn(rng, m) .- exp.(0.5 .* (1.1 .* zn)) .* randexp(rng, m)

    sp = sfmodel_spec(depvar = y, frontier = X, zvar = Z1, noise = :Normal, ineff = :Exponential,
                      hetero = [:sigma_u_sq], type = :prod)
    r = fitq(sp)
    jl = vec(r.jlms); bcf = vec(r.bc)

    @testset "training data: :jlms, :bc, :response and :all equal the fit bit for bit" begin
        @test r.converged
        pj = tryp(r; frontier = X, depvar = y, zvar = Z1, what = :jlms)
        pb = tryp(r; frontier = X, depvar = y, zvar = Z1, what = :bc)
        pr = tryp(r; frontier = X, depvar = y, zvar = Z1, what = :response)
        @test pj !== nothing && isequal(pj, jl)
        @test pb !== nothing && isequal(pb, bcf)
        @test pr !== nothing && isequal(pr, predict(r; frontier = X, what = :frontier) .- jl)
        pa = tryp(r; frontier = X, depvar = y, zvar = Z1, what = :all)
        @test pa isa NamedTuple && keys(pa) == ALL6
        @test pa !== nothing && isequal(pa.jlms, jl) && isequal(pa.bc, bcf)
        @test pa !== nothing && isequal(pa.response, predict(r; frontier = X, what = :frontier) .- jl)
        @test pa !== nothing && isequal(pa.marginal, Matrix{Float64}(r.marginal))
        # the first 17 rows, as a vector zvar (read as the one-column matrix)
        pj17 = tryp(r; frontier = X[1:17, :], depvar = y[1:17], zvar = z[1:17], what = :jlms)
        @test pj17 !== nothing && isequal(pj17, jl[1:17])
        # the fit's own values are the model's (σᵤᵢ per row)
        rj, rb = ref_expo(r.coeff, X, y, z, 1)
        @test relerr(jl, rj) < 1e-12 && relerr(bcf, rb) < 1e-12
    end

    @testset "new data: the closed form with σᵤᵢ per row (512-bit reference)" begin
        rj, rb = ref_expo(r.coeff, Xn, yn, zn, 1)
        pj = tryp(r; frontier = Xn, depvar = yn, zvar = reshape(zn, :, 1), what = :jlms)
        pb = tryp(r; frontier = Xn, depvar = yn, zvar = reshape(zn, :, 1), what = :bc)
        @test pj !== nothing && relerr(pj, rj) < 1e-12
        @test pb !== nothing && relerr(pb, rb) < 1e-12
    end

    @testset "without zvar: the error of the other heteroscedastic models" begin
        sph = sfmodel_spec(depvar = y, frontier = X, zvar = Z1, noise = :Normal, ineff = :HalfNormal,
                           hetero = [:sigma_u_sq], type = :prod)
        rh = fitq(sph)
        msg_hn = errtext(() -> predict(rh; frontier = X, depvar = y, what = :jlms))
        @test occursin("keyword `zvar` is required", msg_hn)
        for w in (:jlms, :bc, :response, :all)
            @test errtext(() -> predict(r; frontier = X, depvar = y, what = w)) == msg_hn
        end
        @test occursin("has 2 columns but model expects 1",
                       errtext(() -> predict(r; frontier = X, depvar = y, zvar = hcat(ones(n), z),
                                             what = :jlms)))
        # :frontier and :residuals need no zvar, as before
        @test isequal(predict(r; frontier = X, depvar = y, what = :residuals),
                      y .- X * r.coeff[1:2])
    end

    @testset "cost frontier, held coefficients" begin
        yc = X * [1.0, 0.5] .+ v .+ u
        spc = sfmodel_spec(depvar = yc, frontier = X, zvar = Z1, noise = :Normal, ineff = :Exponential,
                           hetero = [:sigma_u_sq], type = :cost)
        ini = sfmodel_init(spec = spc, frontier = [1.0, 0.5], ln_sigma_u_sq = [1.1], ln_sigma_v_sq = log(0.09))
        hold = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))   # no warmstart
        rc = fitq(spc; init = ini, optim_options = hold)
        @test isequal(rc.coeff, [1.0, 0.5, 1.1, log(0.09)])   # the fit held the coefficients
        jc = vec(rc.jlms)
        pj = tryp(rc; frontier = X, depvar = yc, zvar = Z1, what = :jlms)
        pb = tryp(rc; frontier = X, depvar = yc, zvar = Z1, what = :bc)
        pr = tryp(rc; frontier = X, depvar = yc, zvar = Z1, what = :response)
        @test pj !== nothing && isequal(pj, jc)
        @test pb !== nothing && isequal(pb, vec(rc.bc))
        @test pr !== nothing && isequal(pr, predict(rc; frontier = X, what = :frontier) .+ jc)
        rj, rb = ref_expo(rc.coeff, X, yc, z, -1)
        @test pj !== nothing && relerr(pj, rj) < 1e-12 && relerr(pb, rb) < 1e-12
    end

    @testset "layouts whose branch did not change still reproduce the fit" begin
        # homoscedastic: no zvar at all
        sp0 = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :Exponential, type = :prod)
        r0 = fitq(sp0)
        @test isequal(predict(r0; frontier = X, depvar = y, what = :jlms), vec(r0.jlms))
        @test isequal(predict(r0; frontier = X, depvar = y, what = :bc), vec(r0.bc))
        # heteroscedastic with a constant column
        Zc = hcat(ones(n), z)
        spc = sfmodel_spec(depvar = y, frontier = X, zvar = Zc, noise = :Normal, ineff = :Exponential,
                           hetero = [:sigma_u_sq], type = :prod)
        rc = fitq(spc)
        @test isequal(predict(rc; frontier = X, depvar = y, zvar = Zc, what = :jlms), vec(rc.jlms))
        @test isequal(predict(rc; frontier = X, depvar = y, zvar = Zc, what = :bc), vec(rc.bc))
        # the half normal with the same one-column zvar (decided by the model already)
        sph = sfmodel_spec(depvar = y, frontier = X, zvar = Z1, noise = :Normal, ineff = :HalfNormal,
                           hetero = [:sigma_u_sq], type = :prod)
        rh = fitq(sph)
        @test isequal(predict(rh; frontier = X, depvar = y, zvar = Z1, what = :jlms), vec(rh.jlms))
    end
end
