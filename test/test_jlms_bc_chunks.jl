# The efficiency indices computed per group of observations or firms equal the indices computed at once.

using Test
using SFrontiers
using Optim
using Random

isdefined(@__MODULE__, :_ref_panel_nll) || include(joinpath(@__DIR__, "draw_memory_reference.jl"))

@testset "efficiency indices per group of observations or firms" begin
    PB = SFrontiers.Panel_Backend
    MC = SFrontiers.MCI_Backend
    MS = SFrontiers.MSLE_Backend

    # the data generator of test_logsumexp_temporary.jl: y, X = [1 x], Z = [1 z], Zs = z
    function _jbc_data(seed, n; T = Float64)
        rng = Xoshiro(seed)
        x = randn(rng, n); z = randn(rng, n)
        u = exp.(-1.0 .+ 0.3 .* z .+ 0.5 .* randn(rng, n))
        y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(rng, n) .- u
        return T.(y), T.(hcat(ones(n), x)), T.(hcat(ones(n), z)), T.(reshape(z, n, 1))
    end
    # internal specifications as test_logsumexp_temporary.jl builds them; no zvar (and no hetero) when zv === nothing
    function _jbc_spec(backend, y, X, zv, ineff, het, copula; D = 256)
        us = zv === nothing ?
            sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = ineff, copula = copula, type = :prod) :
            sfmodel_spec(depvar = y, frontier = X, zvar = zv, noise = :Normal, ineff = ineff, hetero = het,
                         copula = copula, type = :prod)
        m = sfmodel_method(method = backend, n_draws = D, chunks = 10)
        return backend === :MCI ? MC._assemble_MCI_spec(us.mci_spec, m) : MS._assemble_MSLE_spec(us.msle_spec, m)
    end
    # the OLS frontier coefficients, then 0.1 + 0.05 sin(j) for the j-th remaining parameter
    function _jbc_point(B, sp)
        K = B.plen(sp.model, sp.K, sp.L, sp.hetero; L_scaling = sp.L_scaling)
        b = Float64.(Array(sp.frontier)) \ Float64.(Array(sp.depvar))
        return vcat(b, [0.1 + 0.05 * sin(j) for j in 1:(K - sp.K)])
    end

    @testset "cross-sectional indices per group equal the indices computed at once" begin
        y, X, Z, Zs = _jbc_data(400, 400)
        cases = [(:HalfNormal, nothing, nothing, :None), (:TruncatedNormal, Z, [:mu], :None),
                 (:Lognormal, nothing, nothing, :None), (:HalfNormal, nothing, nothing, :Gaussian),
                 (:TruncatedNormal, Zs, :scaling, :None)]
        for b in (:MCI, :MSLE), (ineff, zv, het, cop) in cases
            B = b === :MCI ? MC : MS
            sp = _jbc_spec(b, y, X, zv, ineff, het, cop)
            θ = _jbc_point(B, sp)
            r1 = B.jlms_bc_indices(sp, θ; chunks = 1)
            title = "$b $ineff" * (het === nothing ? "" : " hetero=$het") * (cop === :None ? "" : " copula=$cop")
            @testset "$title chunks=$k" for k in (7, 10, 20)
                rk = B.jlms_bc_indices(sp, θ; chunks = k)
                @test rk.jlms == r1.jlms
                @test rk.bc == r1.bc
                @test rk.likelihood == r1.likelihood
            end
        end
    end

    @testset "an observation alone in its group agrees to rounding" begin
        y, X, _, _ = _jbc_data(7, 7)
        for b in (:MCI, :MSLE)
            B = b === :MCI ? MC : MS
            sp = _jbc_spec(b, y, X, nothing, :HalfNormal, nothing, :None)
            θ = _jbc_point(B, sp)
            r1 = B.jlms_bc_indices(sp, θ; chunks = 1)
            r4 = B.jlms_bc_indices(sp, θ; chunks = 4)          # groups of 2, 2, 2 and 1 observations
            @test isapprox(r4.jlms, r1.jlms; rtol = 1e-14)
            @test isapprox(r4.bc, r1.bc; rtol = 1e-14)
            @test isapprox(r4.likelihood, r1.likelihood; rtol = 1e-14)
        end
    end

    @testset "panel indices per group equal the function before the rewrite" begin
        d = _ref_panel_data(400, 5)
        for (meth, ineff) in ((:MSLE, :HalfNormal), (:MCI, :HalfNormal), (:MCI, :Weibull)), mr in (true, false)
            isp = _ref_panel_spec(d, ineff; method = meth, D = 256, multiRand = mr)
            θ = _ref_theta(d, ineff)
            ref = _ref_panel_jlms_bc(isp, θ)
            @testset "$meth $ineff multiRand=$mr chunks=$k" for k in (1, 7, 10, 20)
                out = PB.panel_jlms_bc(isp, θ; chunks = k)
                @test out.jlms == ref.jlms
                @test out.bc == ref.bc
            end
        end
    end

    @testset "the panel indices never read the whole draw block at once" begin
        d = _ref_panel_data(400, 5)
        isp = _ref_panel_spec(d, :HalfNormal; method = :MCI, D = 256, multiRand = true)
        θ = _ref_theta(d, :HalfNormal)
        w = _RowLog(isp.draws, Int[])
        isp_w = PB._PanelInternalSpec{Float64}((f === :draws ? w : getfield(isp, f) for f in fieldnames(typeof(isp)))...)
        @test PB.panel_jlms_bc(isp_w, θ; chunks = 10) == PB.panel_jlms_bc(isp, θ; chunks = 10)
        @test sum(w.spans) == 400
        @test maximum(w.spans) == cld(400, 10)
    end

    @testset "fitted indices equal the indices computed at once, and predict reproduces them" begin
        og = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 15,),
                         main_solver = NelderMead(), main_opt = (iterations = 200,))
        y, X, _, _ = _jbc_data(400, 400)
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod)
        for (b, B) in ((:MCI, MC), (:MSLE, MS))
            r = sfmodel_fit(spec = sp, method = sfmodel_method(method = b, n_draws = 256, chunks = 10),
                            optim_options = og, marginal = false, show_table = false, verbose = false)
            r1 = B.jlms_bc_indices(r.model, r.coeff; chunks = 1)
            @test r.jlms == r1.jlms
            @test r.bc == r1.bc
            @test predict(r; frontier = X, depvar = y, what = :jlms) == r.jlms
        end
        d = _ref_panel_data(120, 9)
        rp = sfmodel_fit(spec = _ref_unified_spec(d, :HalfNormal),
                         method = sfmodel_method(method = :MCI, n_draws = 128, chunks = 10),
                         optim_options = og, marginal = false, show_table = false, verbose = false)
        r1 = PB.panel_jlms_bc(rp.model, rp.coeff; chunks = 1)
        @test rp.jlms == r1.jlms
        @test rp.bc == r1.bc
        @test predict(rp; frontier = d.X, zvar = reshape(d.z, :, 1), depvar = d.y, id = d.id, what = :jlms) == rp.jlms
    end
end
