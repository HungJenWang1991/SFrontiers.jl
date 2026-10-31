# Regression tests: behaviour the manual documents.
#
# Each one was a case where the manual documented behaviour the code did not have. They are
# repaired rather than documented away, because in each case the manual describes what the
# function is for and the code simply failed to do it.
#
#  1. `sfmodel_ChiSquareTable` is documented in full in the manual's Section 6.6 and threw
#     `UndefVarError: sf_table` on every call, having never assigned that variable; it also
#     had no `return`, so it could not have produced the row of critical values the manual
#     promises.
#  2. `sfmodel_MixTable` called `sfmodel_opt()` when `_dicOPT` was undefined. That call
#     also marks the options as the user's own, which makes a later bare `sfmodel_fit`
#     skip its reset. Both table functions now read the table format and write nothing.
#  3. The list form `frontier = [v1, v2, ...]`, documented in the manual's `sfmodel_spec()`
#     table, worked for
#     every distribution except the three the analytic backend supports — that is, it failed
#     for exactly the models most readers start with.
#  4. `depvar`, `frontier` and `zvar` were tested with `isa Vector`, so any `AbstractVector`
#     that is not a `Vector` was refused. `CSV.read` returns one (a `ChainedVector`) for any
#     file above its chunking threshold, so a large data file was refused with
#     "`depvar` must be a Vector or Matrix." The test below uses a `view`, which is the same
#     `isa Vector`/`isa AbstractVector` distinction without depending on a file large enough
#     to make CSV chunk.
#  5. After a fit made with `jlms_bc_index = false`, `efficiency_summary` and
#     `residuals(r; type = :u)` died on `MethodError: no method matching length(::Nothing)`,
#     which names neither the field nor the argument that switched it off.
using Test
using SFrontiers
using Random
using Optim

@testset "behaviour the manual documents" begin

    Random.seed!(20260822)
    n = 150
    c = ones(n); x1 = randn(n); x2 = randn(n)
    X = hcat(c, x1, x2)
    Z = hcat(ones(n), randn(n))
    y = X * [1.0, 0.5, -0.3] .- abs.(0.4 .* randn(n)) .+ 0.3 .* randn(n)

    optN = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 500, g_abstol = 1e-8))
    MCI  = sfmodel_method(method = :MCI, n_draws = 64)
    MLE  = sfmodel_method(method = :MLE)

    # ------------------------------------------------------------------ 1. ChiSquareTable
    @testset "sfmodel_ChiSquareTable works and returns the row" begin
        r1 = sfmodel_ChiSquareTable(1)
        @test r1 !== nothing
        @test length(r1) == 5
        @test r1[1] == 1.0                       # leading dof column, as sfmodel_MixTable
        # χ²(1) critical values, to three decimals as the printed table shows them
        @test isapprox(r1[2],  2.706; atol = 5e-4)
        @test isapprox(r1[3],  3.841; atol = 5e-4)
        @test isapprox(r1[4],  5.024; atol = 5e-4)
        @test isapprox(r1[5],  6.635; atol = 5e-4)
        r3 = sfmodel_ChiSquareTable(3)
        @test r3[1] == 3.0
        @test isapprox(r3[3], 7.815; atol = 5e-4)
        # the default argument still works, and the informative non-integer error survives
        @test length(sfmodel_ChiSquareTable()) == 5
        @test_throws "has to be an integer" sfmodel_ChiSquareTable(1.5)
        # its working sibling is unchanged
        @test vec(collect(sfmodel_MixTable(1))) ≈ [1.0, 1.642, 2.705, 3.841, 5.412] atol = 5e-4
        @test size(sfmodel_MixTable()) == (40, 5)
    end

    # ------------------------------------------- 2. the table utilities write no module state
    @testset "the table utilities leave the optimizer options alone" begin
        M = SFrontiers.MLE_Backend.SFmle
        before = isdefined(M, :_dicOPT) ? copy(getfield(M, :_dicOPT)) : nothing
        sfmodel_MixTable(2); sfmodel_ChiSquareTable(2); sfmodel_MixTable()
        after  = isdefined(M, :_dicOPT) ? copy(getfield(M, :_dicOPT)) : nothing
        @test before == after
        # in particular, calling a table does not claim the options as the user's own
        if after !== nothing
            @test get(after, :_user_set, false) == get(something(before, Dict()), :_user_set, false)
        end
    end

    # ------------------------------------------------------------ 3. the list form of frontier
    @testset "frontier accepts the documented list form" begin
        for ineff in (:HalfNormal, :TruncatedNormal, :Exponential,
                      :Weibull, :Lognormal, :Rayleigh, :Lomax)
            sp = sfmodel_spec(depvar = y, frontier = [c, x1, x2], noise = :Normal, ineff = ineff)
            @test sp !== nothing
        end
        # the three the analytic backend supports must now build an MLE spec from the list
        # form too — that is precisely the set that used to fail
        for ineff in (:HalfNormal, :TruncatedNormal, :Exponential)
            sp = sfmodel_spec(depvar = y, frontier = [c, x1, x2], noise = :Normal, ineff = ineff)
            @test sp.mle_spec !== nothing
        end
        # and the list form fits to the same answer as the matrix form
        sp_list = sfmodel_spec(depvar = y, frontier = [c, x1, x2], noise = :Normal,
                               ineff = :HalfNormal)
        sp_mat  = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        for m in (MLE, MCI)
            a = sfmodel_fit(spec = sp_list, method = m, optim_options = optN,
                            marginal = false, show_table = false, verbose = false)
            b = sfmodel_fit(spec = sp_mat, method = m, optim_options = optN,
                            marginal = false, show_table = false, verbose = false)
            @test a.loglikelihood == b.loglikelihood
            @test collect(a.coeff) == collect(b.coeff)
        end
        # zvar takes the list form as well
        z1 = Z[:, 1]; z2 = Z[:, 2]
        @test sfmodel_spec(depvar = y, frontier = [c, x1, x2], zvar = [z1, z2],
                           noise = :Normal, ineff = :HalfNormal,
                           hetero = [:sigma_u_sq]).mle_spec !== nothing
    end

    # ---------------------------------------- 4. an AbstractVector that is not a Vector
    @testset "depvar and frontier accept any AbstractVector" begin
        yv = view(y, :)                       # SubArray: AbstractVector, not Vector
        @test !(yv isa Vector)
        @test yv isa AbstractVector
        sp_view = sfmodel_spec(depvar = yv, frontier = X, noise = :Normal, ineff = :HalfNormal)
        @test sp_view.mle_spec !== nothing
        sp_plain = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        a = sfmodel_fit(spec = sp_view,  method = MLE, optim_options = optN,
                        marginal = false, show_table = false, verbose = false)
        b = sfmodel_fit(spec = sp_plain, method = MLE, optim_options = optN,
                        marginal = false, show_table = false, verbose = false)
        @test a.loglikelihood == b.loglikelihood
        # a genuinely wrong type is still refused (the simulation backend gets there
        # first, so the message is its own; what matters is that it does not build)
        @test_throws Exception sfmodel_spec(
            depvar = "not data", frontier = X, noise = :Normal, ineff = :HalfNormal)
        @test_throws Exception sfmodel_spec(
            depvar = y, frontier = "not data", noise = :Normal, ineff = :HalfNormal)
    end

    # ------------------------------- 5. a fit without the indices says so when asked for them
    @testset "jlms_bc_index = false gives a message that names the cause" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        r  = sfmodel_fit(spec = sp, method = MCI, optim_options = optN,
                         jlms_bc_index = false, marginal = false,
                         show_table = false, verbose = false)
        @test r.jlms === nothing
        for f in (() -> efficiency_summary(r),
                  () -> residuals(r; type = :u),
                  () -> residuals(r; type = :v),
                  () -> fitted(r; type = :response))
            e = try (f(); "") catch err; sprint(showerror, err) end
            @test occursin("jlms_bc_index", e)
            @test !occursin("no method matching length", e)
        end
        # the parts that do not need the indices still work
        @test length(residuals(r; type = :composed)) == n
        @test length(coef(r)) == 5
    end
end
