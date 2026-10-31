# test_optim_options_isolation.jl — optimizer options must not leak between fits.
#
# `sfmodel_fit` documents `optim_options = nothing` as "uses defaults". On the MLE backend a
# defined `_dicOPT` was kept instead, so a fit that omitted the options silently inherited the
# solver, iteration cap and tolerance of the previous fit in the session. Measured before the
# repair: three fits on one dataset gave ll = -79.166128 (converged), then -183.867398 from
# deliberately crippled options, then -183.867398 again from a fit that asked for defaults. The
# simulation backends never leaked. This is the same family as the spec-isolation defect
# (test_mle_spec_isolation.jl): module state outliving the call that wrote it.
#
# The reset cannot be unconditional. The legacy workflow is `sfmodel_opt(...)` followed by a bare
# `sfmodel_fit()`, where the standing `_dicOPT` IS the user's intent. So `sfmodel_opt` marks what
# it writes as user-set, `sfmodel_fit` clears that mark on options it writes from its own
# argument, and only the latter are discarded. Both halves are pinned below.
using Test
using SFrontiers
using Random
using Optim: NelderMead, Newton

@testset "optimizer options do not leak between fits" begin

    Random.seed!(31)
    n = 150
    X = hcat(ones(n), randn(n), randn(n))
    y = X * [1.0, 0.5, -0.3] .- abs.(0.4 .* randn(n)) .+ 0.3 .* randn(n)
    sp = sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:HalfNormal)

    # three iterations of Nelder-Mead cannot reach the optimum, so a fit that runs under these
    # is unmistakable: the two variance parameters never move off their 0.1 starting values
    crippled() = sfmodel_opt(main_solver=NelderMead(), main_opt=(iterations=3, g_abstol=0.1))
    co(r) = collect(float.(r.coeff))

    function run(m, opt)
        r = sfmodel_fit(spec=sp, method = (m === :MLE ? sfmodel_method(method = m) :
                                                        sfmodel_method(method = m, n_draws = 1024)), init=nothing,
                        optim_options=opt, marginal=false, show_table=false, verbose=false)
        return co(r), r.converged
    end

    @testset "a fit asking for defaults gets defaults, on every backend" begin
        for m in (:MLE, :MCI, :MSLE)
            meth = m === :MLE ? m : m
            first_default, conv1 = run(meth, nothing)
            crippled_fit, conv2  = run(meth, crippled())
            again_default, conv3 = run(meth, nothing)

            # the crippled options really are distinguishable, or the test proves nothing
            @test !isapprox(first_default, crippled_fit; rtol=1e-3)
            @test conv1 == true
            @test conv2 == false

            @test again_default == first_default      # bit-for-bit, not merely close
            @test conv3 == conv1
            @test again_default != crippled_fit
        end
    end

    @testset "options given explicitly are still honoured" begin
        a, _ = run(:MLE, crippled())
        b, _ = run(:MLE, crippled())
        @test a == b                                   # deterministic under the same options
        c, conv = run(:MLE, nothing)
        @test conv == true
        @test c != a
    end

    # ---------------------------------------------------------------- the legacy half
    # `sfmodel_opt(...)` followed by a bare `sfmodel_fit()` must still use the user's options,
    # and must keep using them for a second bare fit. Reached only by module-qualified call.
    @testset "the legacy sfmodel_opt then bare fit workflow is unchanged" begin
        S = SFrontiers.MLE_Backend.SFmle
        S.sfmodel_opt(main_solver=NelderMead(), main_opt=(iterations=3, g_abstol=0.1))
        @test S._dicOPT[:_user_set] == true
        r1 = S.sfmodel_fit(show_table=false, verbose=false, marginal=false)
        @test S._dicOPT[:main_maxIT] == 3              # the user's cap, not the backend's default 2000
        @test r1.converged == false
        r2 = S.sfmodel_fit(show_table=false, verbose=false, marginal=false)
        @test S._dicOPT[:main_maxIT] == 3              # and still there for a second bare fit
        @test co(r2) == co(r1)

        # a fit that supplies its own options clears the mark, so the next default fit resets
        run(:MLE, crippled())
        @test S._dicOPT[:_user_set] == false
        run(:MLE, nothing)
        # the unified default every method shares (Newton 200), not the analytic backend's own
        # default of 2000
        @test S._dicOPT[:main_maxIT] == 200
    end
end
