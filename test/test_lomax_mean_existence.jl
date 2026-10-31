# Marginal effects on E(u) of Lomax fits where the Lomax mean λ/(α − 1) does not exist (α ≤ 1): they are NaN,
# and the fit and predict(what = :marginal) each print one warning, also with show_table = false and
# verbose = false; `shape_warning = false` turns the fit's warning off. Every fit is held at its start values
# (no warmstart, no Newton iteration), so the data only define the model.

using Test
using SFrontiers
using Random
using Optim: Newton

@testset "Lomax marginal effects where the mean does not exist" begin
    rng = Xoshiro(3801)
    N = 200
    x = randn(rng, N)
    z = randn(rng, N)
    X = hcat(ones(N), x)
    Z = hcat(ones(N), z)
    y = X * [1.0, 0.5] .+ 0.5 .* randn(rng, N) .- abs.(0.6 .* randn(rng, N))
    hold = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))
    phrase = "exists only for α > 1"

    # The value of f() and the text it printed.
    function _lme_capture(f)
        path, io = mktemp()
        ret = redirect_stdout(f, io)
        close(io)
        out = read(path, String)
        rm(path)
        return ret, out
    end
    _lme_fit(sp, m, p) = _lme_capture(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = m, n_draws = 64),
        init = p, optim_options = hold, marginal = true, show_table = false, verbose = false))

    @testset "homoscedastic α, $m" for m in (:MCI, :MSLE)
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal, ineff = :Lomax, hetero = [:lambda],
                          type = :prod)
        p = sfmodel_init(spec = sp, frontier = X \ y, ln_lambda = [-1.0, 0.5], ln_alpha = log(0.8),
                         ln_sigma_v_sq = log(0.25))
        r, out = _lme_fit(sp, m, p)
        @test all(isnan, Matrix(r.marginal))
        @test count(phrase, out) == 1
        @test occursin("(all 200 observations)", out)
        @test occursin("the marginal effects are NaN there either way", out)
        r0, out0 = _lme_capture(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = m, n_draws = 64),
            init = p, optim_options = hold, marginal = true, shape_warning = false, show_table = false,
            verbose = false))
        @test all(isnan, Matrix(r0.marginal))
        @test !occursin("Warning", out0)
        mp, outp = _lme_capture(() -> predict(r; zvar = Z, what = :marginal))
        @test all(isnan, mp)
        @test count(phrase, outp) == 1
        @test !occursin("shape_warning", outp)
        _, outa = _lme_capture(() -> predict(r; frontier = X, zvar = Z, depvar = y, what = :all))
        @test count(phrase, outa) == 1
        # a frame of one row: the count reads "(1 observation)"
        m1, out1 = _lme_capture(() -> predict(r; zvar = Z[1:1, :], what = :marginal))
        @test all(isnan, m1)
        @test occursin("(1 observation)", out1)

        # α = 3: the mean exists, the marginal effects are finite and no warning about it is printed.
        p3 = sfmodel_init(spec = sp, frontier = X \ y, ln_lambda = [-1.0, 0.5], ln_alpha = log(3.0),
                          ln_sigma_v_sq = log(0.25))
        r3, out3 = _lme_fit(sp, m, p3)
        @test all(isfinite, Matrix(r3.marginal))
        @test !occursin(phrase, out3)
        _, outp3 = _lme_capture(() -> predict(r3; zvar = Z, what = :marginal))
        @test !occursin(phrase, outp3)
    end

    @testset "heteroscedastic α, $m" for m in (:MCI, :MSLE)
        spA = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal, ineff = :Lomax, hetero = [:alpha],
                           type = :prod)
        pA = sfmodel_init(spec = spA, frontier = X \ y, ln_lambda = -1.0, ln_alpha = [log(1.2), 0.4],
                          ln_sigma_v_sq = log(0.25))
        rows = findall(<=(1.0), exp.(log(1.2) .+ 0.4 .* z))      # α_i ≤ 1 where z_i ≤ −0.4558
        r, out = _lme_fit(spA, m, pA)
        M = Matrix(r.marginal)
        @test findall(i -> any(isnan, view(M, i, :)), axes(M, 1)) == rows
        @test occursin("($(length(rows)) of 200 observations)", out)
    end

    @testset "scaling property, $m" for m in (:MCI, :MSLE)
        z1 = reshape(z, :, 1)
        spS = sfmodel_spec(depvar = y, frontier = X, zvar = z1, noise = :Normal, ineff = :Lomax, hetero = :scaling,
                           type = :prod)
        pS = sfmodel_init(spec = spS, frontier = X \ y, scaling = [0.3], ln_lambda = -1.0, ln_alpha = log(0.8),
                          ln_sigma_v_sq = log(0.25))
        r, out = _lme_fit(spS, m, pS)
        @test all(isnan, Matrix(r.marginal))
        @test count(phrase, out) == 1
        mp, outp = _lme_capture(() -> predict(r; zvar = z1, what = :marginal))
        @test all(isnan, mp)
        @test count(phrase, outp) == 1
    end

    @testset "panel_TFE, MCI" begin
        rngp = Xoshiro(3802)
        nf = 50
        T = 4
        NT = nf * T
        id = repeat(1:nf, inner = T)
        xp = randn(rngp, NT)
        zp = randn(rngp, NT)
        yp = 0.5 .* xp .+ 0.3 .* randn(rngp, NT) .- abs.(0.5 .* randn(rngp, NT)) .* exp.(0.3 .* zp)
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = reshape(xp, :, 1), zvar = reshape(zp, :, 1),
                           id = id, noise = :Normal, ineff = :Lomax)
        pp = sfmodel_init(spec = spp, frontier = [0.5], scaling = [0.3], ln_lambda = -1.0, ln_alpha = log(0.8),
                          ln_sigma_v_sq = log(0.09))
        r, out = _lme_fit(spp, :MCI, pp)
        @test all(isnan, Matrix(r.marginal))
        @test count(phrase, out) == 1
        @test occursin("(all 200 observations)", out)
    end
end
