# Regression tests: the printed header of scaling models and DSL coefficient names under :MLE.
#
# (a) The printed line "Number of Z columns (L)" read 1 for every cross-sectional model with
#     hetero = :scaling (the truncated normal under :MLE; any inefficiency distribution under :MCI
#     and :MSLE), because each backend pins its internal L to 1 there (the inefficiency
#     parameters are constants). It now gives the number of scaling variables. A model without
#     zvar still prints 1.
# (b) :MCI and :MSLE printed "Homoscedastic model (no heteroscedasticity)" for these models, and
#     :MLE for panel_TFE; both now print "Heteroscedastic parameters: :scaling", as :MLE did for
#     the cross-sectional scaling model.
# (c) Under :MLE, the DSL form put the @zvar names on an equation only when it had more than one
#     column: a single scaling variable was named hscale_var1 (cross-section and panel_TFE), and a
#     one-column heteroscedastic mu or sigma_u^2 equation mu_var1 / sigma_u^2_var1, where :MCI,
#     :MSLE and the marginal-effect column (when there is one) use the column name. Such a
#     coefficient now carries the column name in the printed table, result.table,
#     result.list[:varnames], result.varnames and coeftable. Fits in the keyword form keep their
#     names.
# The :MLE header is checked on fits. For :MCI and :MSLE the test assembles the spec as the fit
# does (32 draws) and calls the spec-based print_table that the fit calls, so the arguments that
# method passes on (L_scaling among them) are checked too; this saves compiling two simulation
# fits. Every :MLE fit below has 7 parameters (the frontier has 2, 3 or 4 columns to make it so),
# so ForwardDiff compiles one Dual width only; this keeps the test near half a minute. Printed
# lines and stored names only: a DSL fit's estimates still equal, bit for bit, those of the same
# model in the keyword form, whose code path the fix does not touch.

using Test
using SFrontiers
using Random
using Optim
using DataFrames
using LinearAlgebra

@testset "printed header of scaling models and DSL coefficient names under :MLE" begin
    # stdout goes to a file; a wide display keeps every column of the printed table, whatever
    # COLUMNS the test runs under
    function capture(f)
        path, io = mktemp()
        r = withenv(() -> redirect_stdout(f, io), "COLUMNS" => "300")
        close(io)
        txt = read(path, String); rm(path; force = true)
        return r, txt
    end
    lines_with(txt, pats...) = [String(strip(l)) for l in split(txt, '\n') if any(p -> occursin(p, l), pats)]
    hetline(txt) = only(lines_with(txt, "Heteroscedastic parameters", "Homoscedastic model"))
    Lline(txt) = only(lines_with(txt, "Number of Z columns (L)"))
    # the Var. column of the printed estimation table (rows with at least 7 fields; the
    # auxiliary table of converted variances has fewer)
    function printed_names(txt)
        out = String[]
        for l in split(txt, '\n')
            occursin('│', l) || continue
            f = strip.(split(l, '│'))
            length(f) >= 7 && f[3] != "Var." && push!(out, String(f[3]))
        end
        return out
    end
    opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 20, time_limit = 60.0),
                      main_solver = Newton(), main_opt = (iterations = 40, time_limit = 60.0))
    function fitc(sp)
        r, txt = capture(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), optim_options = opt,
                                           marginal = false, show_table = true, verbose = false))
        @test length(r.coeff) == 7                     # one Dual width for the whole test
        return r, txt
    end
    # the header of :MCI / :MSLE: the spec assembled as sfmodel_MCI_fit / sfmodel_MSLE_fit
    # assemble it (32 draws), and the spec-based print_table those fits call with it
    function sim_header(B, ms)
        mci = B === SFrontiers.MCI_Backend
        isp = (mci ? B._assemble_MCI_spec : B._assemble_MSLE_spec)(ms,
                  B.sfmodel_method(method = mci ? :MCI : :MSLE, n_draws = 32, multiRand = false))
        np = length(isp.varnames)
        return capture(() -> B.print_table(isp, zeros(np), Matrix(1.0I, np, np)))[2]
    end

    rng = Xoshiro(3737); n = 200
    x1 = randn(rng, n); x2 = randn(rng, n); x3 = randn(rng, n); z1 = randn(rng, n); z2 = randn(rng, n)
    X2 = hcat(ones(n), x1); X3 = hcat(X2, x2); X4 = hcat(X3, x3)
    y_sc = X2 * [1.0, 0.5] .+ 0.25 .* randn(rng, n) .- exp.(0.4 .* z1 .- 0.3 .* z2) .* abs.(0.25 .+ 0.5 .* randn(rng, n))
    y_hz = X2 * [1.0, 0.5] .+ 0.25 .* randn(rng, n) .- exp.(0.5 .* z1) .* abs.(0.5 .* randn(rng, n))
    nf, T = 40, 5; np_ = nf * T
    idp = repeat(1:nf, inner = T)
    XP = hcat(randn(rng, np_), randn(rng, np_), randn(rng, np_)); ZP = hcat(randn(rng, np_), randn(rng, np_))
    tn = Float64[]
    while length(tn) < nf
        d = 0.3 + 0.5 * randn(rng); d > 0 && push!(tn, d)
    end
    yp = repeat(0.5 .* randn(rng, nf), inner = T) .+ XP[:, 1:2] * [0.6, -0.4] .+ 0.3 .* randn(rng, np_) .-
         exp.(0.3 .* ZP[:, 1] .- 0.2 .* ZP[:, 2]) .* repeat(tn, inner = T)

    @testset "(a, b) header lines" begin
        sc2 = sfmodel_spec(depvar = y_sc, frontier = X2, zvar = hcat(z1, z2), noise = :Normal,
                           ineff = :TruncatedNormal, hetero = :scaling)
        _, txt = fitc(sc2)
        @test hetline(txt) == "Heteroscedastic parameters: :scaling"
        @test Lline(txt) == "Number of Z columns (L): 2"                   # was 1
        for (B, ms) in ((SFrontiers.MCI_Backend, sc2.mci_spec), (SFrontiers.MSLE_Backend, sc2.msle_spec))
            @test ms.L == 1 && ms.L_scaling == 2 && isempty(ms.hetero)   # what the backends store
            txt = sim_header(B, ms)
            @test hetline(txt) == "Heteroscedastic parameters: :scaling"   # was "Homoscedastic model"
            @test Lline(txt) == "Number of Z columns (L): 2"               # was 1
        end
        # :MCI and :MSLE have the scaling form for every inefficiency distribution (:MLE for the
        # truncated normal only), and print it alike
        hns = sfmodel_spec(depvar = y_sc, frontier = X2, zvar = hcat(z1, z2), noise = :Normal,
                           ineff = :HalfNormal, hetero = :scaling)
        @test hns.mle_spec === nothing
        for (B, ms) in ((SFrontiers.MCI_Backend, hns.mci_spec), (SFrontiers.MSLE_Backend, hns.msle_spec))
            txt = sim_header(B, ms)
            @test hetline(txt) == "Heteroscedastic parameters: :scaling"   # was "Homoscedastic model"
            @test Lline(txt) == "Number of Z columns (L): 2"               # was 1
        end
        ptfe = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XP[:, 1:2], zvar = ZP, id = idp,
                            noise = :Normal, ineff = :TruncatedNormal)
        _, txt = fitc(ptfe)
        @test hetline(txt) == "Heteroscedastic parameters: :scaling"       # was "Homoscedastic model"
        @test Lline(txt) == "Number of Z columns (L): 2"
        # unchanged: a model without zvar prints L = 1; a heteroscedastic model its zvar columns
        hom = sfmodel_spec(depvar = y_sc, frontier = X4, noise = :Normal, ineff = :TruncatedNormal)
        _, txt = fitc(hom)
        @test hetline(txt) == "Homoscedastic model (no heteroscedasticity)"
        @test Lline(txt) == "Number of Z columns (L): 1"
        for (B, ms) in ((SFrontiers.MCI_Backend, hom.mci_spec), (SFrontiers.MSLE_Backend, hom.msle_spec))
            txt = sim_header(B, ms)
            @test hetline(txt) == "Homoscedastic model (no heteroscedasticity)"
            @test Lline(txt) == "Number of Z columns (L): 1"
        end
        het = sfmodel_spec(depvar = y_hz, frontier = X2, zvar = hcat(ones(n), z1), noise = :Normal,
                           ineff = :TruncatedNormal, hetero = [:mu, :sigma_u_sq])
        _, txt = fitc(het)
        @test hetline(txt) == "Heteroscedastic parameters: [:mu, :sigma_u_sq]"
        @test Lline(txt) == "Number of Z columns (L): 2"
        txt = sim_header(SFrontiers.MCI_Backend, het.mci_spec)
        @test hetline(txt) == "Heteroscedastic parameters: [:mu, :sigma_u_sq]"
        @test Lline(txt) == "Number of Z columns (L): 2"
    end

    @testset "(c) DSL names under :MLE" begin
        df = DataFrame(y = y_sc, yh = y_hz, c = ones(n), x1 = x1, x2 = x2, x3 = x3, za = z1, zb = z2)
        dfp = DataFrame(y = yp, xa = XP[:, 1], xb = XP[:, 2], xc = XP[:, 3], zq = ZP[:, 1], firm = idp)

        # one scaling variable, cross-section: the column name everywhere, as under :MCI and :MSLE
        sp = sfmodel_spec(@useData(df), @depvar(y), @frontier(c, x1, x2), @zvar(za), noise = :Normal,
                          ineff = :TruncatedNormal, hetero = :scaling)
        r, txt = fitc(sp)
        @test coeftable(r).rownms == ["c", "x1", "x2", "za", "_cons", "_cons", "_cons"]   # was hscale_var1
        @test r.varnames == ["c", "x1", "x2", "za", "_cons", "_cons", "_cons"]
        @test r.list[:varnames] == ["c", "x1", "x2", "_cons", "za", "_cons", "_cons"]      # analytic order
        @test String.(r.table[2:end, 2]) == r.list[:varnames]
        @test printed_names(txt) == r.list[:varnames]
        @test sp.mci_spec.varnames[4] == sp.msle_spec.varnames[4] == "za"                  # :MCI / :MSLE rows
        # the same model in the keyword form: its own names, and the same estimates bit for bit
        rk, _ = fitc(sfmodel_spec(depvar = y_sc, frontier = X3, zvar = reshape(z1, n, 1), noise = :Normal,
                                  ineff = :TruncatedNormal, hetero = :scaling))
        @test coeftable(rk).rownms == ["frontier_var1", "frontier_var2", "frontier_var3", "hscale_var1",
                                       "_cons", "_cons", "_cons"]
        @test collect(r.coeff) == collect(rk.coeff) && collect(r.std_err) == collect(rk.std_err)
        @test r.loglikelihood == rk.loglikelihood

        # two scaling variables: unchanged
        r, _ = fitc(sfmodel_spec(@useData(df), @depvar(y), @frontier(c, x1), @zvar(za, zb), noise = :Normal,
                                 ineff = :TruncatedNormal, hetero = :scaling))
        @test coeftable(r).rownms == ["c", "x1", "za", "zb", "_cons", "_cons", "_cons"]

        # panel_TFE with one scaling variable
        r, txt = fitc(sfmodel_spec(@useData(dfp), @depvar(y), @frontier(xa, xb, xc), @zvar(zq), @id(firm),
                                   noise = :Normal, ineff = :TruncatedNormal, datatype = :panel_TFE))
        @test coeftable(r).rownms == ["xa", "xb", "xc", "zq", "_cons", "_cons", "_cons"]   # was hscale_var1
        @test r.list[:varnames] == ["xa", "xb", "xc", "_cons", "zq", "_cons", "_cons"]
        @test printed_names(txt) == r.list[:varnames]

        # one-column heteroscedastic mu and sigma_u^2 equations (the sigma_u^2 loop is the one
        # that the half normal and the exponential use as well)
        r, txt = fitc(sfmodel_spec(@useData(df), @depvar(yh), @frontier(c, x1, x2, x3), @zvar(za), noise = :Normal,
                                   ineff = :TruncatedNormal, hetero = [:mu, :sigma_u_sq]))
        @test coeftable(r).rownms == ["c", "x1", "x2", "x3", "za", "za", "_cons"]          # was μ_var1, σᵤ²_var1
        @test printed_names(txt) == ["c", "x1", "x2", "x3", "za", "za", "_cons"]
        # a homoscedastic sigma_u^2 next to a heteroscedastic mu keeps `_cons` (unchanged)
        r, _ = fitc(sfmodel_spec(@useData(df), @depvar(yh), @frontier(c, x1, x2), @zvar(c, za), noise = :Normal,
                                 ineff = :TruncatedNormal, hetero = [:mu]))
        @test coeftable(r).rownms == ["c", "x1", "x2", "c", "za", "_cons", "_cons"]
    end
end
