# Regression tests: one log-sum-exp temporary per evaluation (MCI, MSLE).
#
# The chunked MCI and MSLE likelihoods (MCI_Backend.MCI_nll_mci_T; the two MSLE_Backend.qmc_nll
# methods) called logsumexp_rows once per chunk, and every call allocated a new temporary
# exp.(A .- max) of the chunk's size: about N x D elements per evaluation, most of the bytes an
# evaluation allocates when chunks > 1 (device memory on the GPU). They now allocate one temporary
# per evaluation (plus one of exact size for a shorter last chunk), which _logsumexp_rows_tmp!
# fills with the same expression. Checked here: (1) _logsumexp_rows_tmp! equals logsumexp_rows bit
# for bit, whatever an earlier call left in the temporary; (2) the chunked likelihoods equal the
# unchunked likelihood (chunks = 1) to rounding: values, ForwardDiff gradients and Hessians, chunks 7
# (shorter last chunk), 10 and 20; (3) the bytes allocated per evaluation.
# On the code before the fix the helper is missing and the allocation asserts fail.

using Test
using SFrontiers
using ForwardDiff
using Random

const _B40C = SFrontiers.MCI_Backend
const _B40S = SFrontiers.MSLE_Backend

# ---- bitwise comparison of numbers, ForwardDiff Duals (nested or not) and arrays of them ------
_b40_bits!(v, x::Float64) = push!(v, reinterpret(UInt64, x))
_b40_bits!(v, x::Float32) = push!(v, UInt64(reinterpret(UInt32, x)))
function _b40_bits!(v, x::ForwardDiff.Dual)
    _b40_bits!(v, ForwardDiff.value(x))
    for q in ForwardDiff.partials(x)
        _b40_bits!(v, q)
    end
    return v
end
function _b40_bits(xs)
    v = UInt64[]
    for x in xs
        _b40_bits!(v, x)
    end
    return v
end
_b40_same(a, b) = typeof(a) == typeof(b) && size(a) == size(b) && _b40_bits(a) == _b40_bits(b)

# ---- data and specifications ---------------------------------------------------------------------
function _b40_data(seed, n; T = Float64)
    rng = Xoshiro(seed)
    x = randn(rng, n); z = randn(rng, n)
    u = exp.(-1.0 .+ 0.3 .* z .+ 0.5 .* randn(rng, n))
    y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(rng, n) .- u
    return T.(y), T.(hcat(ones(n), x)), T.(hcat(ones(n), z)), T.(reshape(z, n, 1))
end
function _b40_spec(backend, y, X, Z, ineff, het; D = 256, multiRand = true)
    us = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal, ineff = ineff,
                      hetero = het, type = :prod)
    m = sfmodel_method(method = backend, n_draws = D, multiRand = multiRand, chunks = 10)
    return backend === :MCI ? _B40C._assemble_MCI_spec(us.mci_spec, m) :
                              _B40S._assemble_MSLE_spec(us.msle_spec, m)
end
function _b40_point(B, sp)
    K = B.plen(sp.model, sp.K, sp.L, sp.hetero; L_scaling = sp.L_scaling)
    b = Float64.(Array(sp.frontier)) \ Float64.(Array(sp.depvar))
    return vcat(b, [0.1 + 0.05 * sin(j) for j in 1:(K - sp.K)])
end

# gradients and Hessians with one tag (nothing) for every closure, so that the likelihood is
# compiled once per element type rather than once per closure
_b40_grad(f, p) = ForwardDiff.gradient(f, p, ForwardDiff.GradientConfig(nothing, p), Val{false}())
_b40_hess(f, p) = ForwardDiff.hessian(f, p, ForwardDiff.HessianConfig(nothing, p), Val{false}())

# bytes allocated by one evaluation, after two warm-up calls
_b40_alloc(f) = (f(); f(); @allocated f())

@testset "one log-sum-exp temporary per evaluation (MCI, MSLE)" begin

    @testset "the helper equals logsumexp_rows bit for bit" begin
        @test isdefined(_B40C, :_logsumexp_rows_tmp!)
        @test isdefined(_B40S, :_logsumexp_rows_tmp!)
        if isdefined(_B40C, :_logsumexp_rows_tmp!) && isdefined(_B40S, :_logsumexp_rows_tmp!)
            P1 = eltype(ForwardDiff.GradientConfig(nothing, zeros(3)).duals)
            P2 = eltype(ForwardDiff.HessianConfig(nothing, zeros(3)).gradient_config.duals)
            rng = Xoshiro(4040)
            mk(::Type{T}, v) where {T<:AbstractFloat} = T(v)
            mk(::Type{T}, v) where {T<:ForwardDiff.Dual} = T(mk(ForwardDiff.valtype(T), v),
                ForwardDiff.Partials(ntuple(_ -> mk(ForwardDiff.valtype(T), randn(rng)), 3)))
            D = 128
            for T in (Float64, Float32, P1, P2), rows in (61, 100), B in (_B40C, _B40S)
                big = [mk(T, -5 + 3 * randn(rng)) for i in 1:rows+3, j in 1:D]   # chunk buffer
                A = @view big[1:rows, :]                                          # chunk view
                tmp = Matrix{T}(undef, rows, D)
                fill!(tmp, mk(T, NaN))                                            # stale content
                @test _b40_same(B._logsumexp_rows_tmp!(tmp, A), B.logsumexp_rows(A))
                big2 = [mk(T, -20 + 9 * randn(rng)) for i in 1:rows+3, j in 1:D]
                A2 = @view big2[1:rows, :]                                        # reuse
                @test _b40_same(B._logsumexp_rows_tmp!(tmp, A2), B.logsumexp_rows(A2))
            end
            # a temporary of another size is refused (one row would otherwise broadcast silently)
            for B in (_B40C, _B40S)
                @test_throws DimensionMismatch B._logsumexp_rows_tmp!(zeros(5, 16), randn(1, 16))
                @test_throws DimensionMismatch B._logsumexp_rows_tmp!(zeros(5, 16), randn(4, 16))
            end
        end
    end

    n = 400       # chunks 10 and 20 divide n; chunks 7 leaves a shorter last chunk (6 x 58 + 52)
    yd, Xd, Zd, _ = _b40_data(400, n)
    yf, Xf, Zf, _ = _b40_data(401, n; T = Float32)

    # Per backend: value, gradient and Hessian of one specification at chunks 7, 10 and 20 against
    # chunks = 1, where the likelihood takes its unchunked path; for MSLE also the raw-data method, and
    # values for Float32 data with shared draws and Float32 parameters, which make the chunk buffers and
    # the temporary Float32. The chunks sum the observations' log-likelihoods in another order, so the
    # results agree to rounding: relative 1e-12 (below 1e-15 measured), for Float32 1e-5 (2.1e-7 measured).
    @testset "chunked MCI likelihood against chunks = 1, to rounding" begin
        sp = _b40_spec(:MCI, yd, Xd, Zd, :Lognormal, [:mu, :sigma_u_sq])
        p = _b40_point(_B40C, sp)
        f1 = q -> _B40C.MCI_nll(sp, q; chunks = 1)
        v1, g1, h1 = f1(p), _b40_grad(f1, p), _b40_hess(f1, p)
        @testset "Lognormal [mu, sigma_u_sq], multiRand, chunks $ch" for ch in (7, 10, 20)
            fk = q -> _B40C.MCI_nll(sp, q; chunks = ch)
            @test isapprox(fk(p), v1; rtol = 1e-12)
            @test isapprox(_b40_grad(fk, p), g1; rtol = 1e-12)
            @test isapprox(_b40_hess(fk, p), h1; rtol = 1e-12)
        end
    end

    @testset "chunked MSLE likelihood against chunks = 1, to rounding" begin
        sp = _b40_spec(:MSLE, yd, Xd, Zd, :Lognormal, [:mu, :sigma_u_sq])
        p = _b40_point(_B40S, sp)
        f1 = q -> _B40S.qmc_nll(sp, q; chunks = 1)
        v1, g1, h1 = f1(p), _b40_grad(f1, p), _b40_hess(f1, p)
        @testset "Lognormal [mu, sigma_u_sq], multiRand, chunks $ch" for ch in (7, 10, 20)
            fk = q -> _B40S.qmc_nll(sp, q; chunks = ch)
            @test isapprox(fk(p), v1; rtol = 1e-12)
            @test isapprox(_b40_grad(fk, p), g1; rtol = 1e-12)
            @test isapprox(_b40_hess(fk, p), h1; rtol = 1e-12)
        end
        # the raw-data method qmc_nll(Y, X, Z, p, draws; ...), same data and draws
        rawf(ch) = q -> _B40S.qmc_nll(yd, Xd, Zd, q, sp.draws_2D; noise = :Normal, ineff = :Lognormal,
                                      hetero = [:mu, :sigma_u_sq], chunks = ch)
        r1 = rawf(1)
        rv1, rg1 = r1(p), _b40_grad(r1, p)
        @testset "raw-data method, chunks $ch" for ch in (7, 10)
            fk = rawf(ch)
            @test isapprox(fk(p), rv1; rtol = 1e-12)
            @test isapprox(_b40_grad(fk, p), rg1; rtol = 1e-12)
        end
        @testset "Weibull [lambda, k], Float32 data and parameters, shared draws" begin
            s32 = _b40_spec(:MSLE, yf, Xf, Zf, :Weibull, [:lambda, :k]; multiRand = false)
            q32 = Float32.(_b40_point(_B40S, s32))
            w1 = _B40S.qmc_nll(s32, q32; chunks = 1)
            for ch in (7, 10)
                w = _B40S.qmc_nll(s32, q32; chunks = ch)
                @test w isa Float32 && isapprox(w, w1; rtol = 1e-5)
            end
        end
    end

    @testset "bytes allocated per evaluation" begin
        # Before the fix an evaluation allocated about (2 [MCI] or 1 [MSLE] chunk buffers + N) x D
        # elements, i.e. 1.05-1.2 x N x D at chunks 10-20; now (3 or 2 chunk buffers) x D, 0.1-0.3 x
        # N x D, plus one temporary for a shorter last chunk (0.42-0.57 x N x D at chunks 7).
        D = 256
        P1 = eltype(ForwardDiff.GradientConfig(nothing, zeros(7)).duals)
        for (B, sp, f) in ((_B40C, _b40_spec(:MCI, yd, Xd, Zd, :Lognormal, [:mu, :sigma_u_sq]; D = D), _B40C.MCI_nll),
                           (_B40S, _b40_spec(:MSLE, yd, Xd, Zd, :Lognormal, [:mu, :sigma_u_sq]; D = D), _B40S.qmc_nll))
            p = _b40_point(B, sp)
            pd = [P1(p[i], ForwardDiff.Partials(ntuple(j -> Float64(i == j), 7))) for i in eachindex(p)]
            for q in (p, pd)
                full = n * D * sizeof(eltype(q))
                for (ch, bound) in ((10, 0.6), (20, 0.6), (7, 0.8))
                    @test _b40_alloc(() -> f(sp, q; chunks = ch)) < bound * full
                end
            end
        end
    end
end
