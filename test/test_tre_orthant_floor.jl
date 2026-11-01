# Regression tests for the panel_TRE orthant probability, formed in log space instead of through
# a 1e-15 floor.
#
# Both true-random-effect likelihoods and both of their jlms/bc indices form
#
#     P = int phi(u) prod_t Phi((B_t - rho u)/gamma) du,
#
# a T-dimensional orthant probability written as one-dimensional quadrature over the common
# random factor, and then took its logarithm through log(max(P, 1e-15)).  Phi underflows to
# exactly 0 below about -38 and P reaches 1e-15 long before that, after which the floor
# supplied log(1e-15) = -34.539 whatever the truth was: 49.5 nats out at B/gamma = -8, 449 at
# -20 and 1.18e10 at -1e5, all against a 256-bit reference.  Two further consequences:
#
#   * On the flat side of the floor d log P/dB is exactly 0, so the optimiser was told the
#     objective did not depend on the parameters.  In a comparison of truncated-normal fits
#     (NelderMead, 300 iterations) all eight cells settled on fabricated optima
#     -- 11167 at N = 4, T = 10, 1989 at N = 3, T = 20 -- each reporting sum(jlms) exactly 0
#     and every bc exactly 1, that is, every firm perfectly efficient.
#   * jlmsbc uses the same probability three ways per firm and the floor reaches all three.
#     Before MCDF1 itself floors, MCDF2 -- built on the more negative B2 -- floors first, and
#     bc = h2/h1 is reported ABOVE 1: 2.856 for a true 0.9064 on a T = 5 firm four units on
#     the inefficient side of the frontier, 7.419 for 0.9296 two units further out, and 27.54
#     for 0.9536 at ten.  E[exp(-u)] with u >= 0 cannot exceed 1.
#
# The same line of LL_T(PTRET) carried a second floor, on the truncated normal's normalising
# constant sigma_u*Phi(mu/sigma_u).  It bites at mu/sigma_u = -7.70 for the sigma_u = 0.152 an
# N = 4, T = 10 fit settles at, and that fit sat at -7.61 -- one step away.
#
# The repair computes log P directly, scales each side of the peak by its own one-nat width,
# and takes every difference of large logarithms analytically through erfcx; see
# _tre_logorthant in src/sf_MLE/SFloglikefun.jl.  The numbers pinned below were computed with
# this code and checked against three independent quadrature rules over the random effect in
# 256-bit arithmetic.
#
# Only CSV and Test are test dependencies, so nothing here loads Logging; log records are
# swallowed with Test.collect_test_logs, per test_predict_draw_alignment.jl.

using SFrontiers
using Test
using Optim
using LinearAlgebra
using QuadGK
using StatsFuns
using SpecialFunctions
using ForwardDiff

const _SFO = SFrontiers.MLE_Backend.SFmle

_tre_flr_poH() = (begx=1, endx=2, begz=0, endz=0, begq=3, endq=3, begw=4, endw=4, begv=5, endv=5)
_tre_flr_poT() = (begx=1, endx=2, begz=3, endz=3, begq=4, endq=4, begw=5, endw=5, begv=6, endv=6)

function _tre_flr_data(nf, nt)
    n  = nf * nt
    id = repeat(1:nf, inner = nt)
    q  = collect(range(-1, 1; length = n))
    X  = hcat(ones(n), q)
    y  = X * [0.62, -0.24] .- abs.(0.18 .* sin.((1:n) .* 0.81)) .+ 0.035 .* cos.((1:n) .* 1.09)
    idt = Matrix{Any}(undef, nf, 2)
    for i in 1:nf
        idt[i, 1] = collect(((i-1)*nt+1):(i*nt))
        idt[i, 2] = nt
    end
    (y, X, id, idt, n)
end

# ---------------------------------------------------------------------------------------
# A 256-bit reference that shares no code with the package.
#
# Phi(x) = erfcx(-x/sqrt2)*exp(-x^2/2)/2, and erfcx cannot underflow, so log Phi stays finite
# for every finite x; for x >= 0 erfcx(-x/sqrt2) would OVERFLOW instead, so that side goes
# through log1p.  Taking the logarithm this way is what lets the reference reach the depths
# the floor was hiding: a reference that formed P first would hit MPFR's own exponent limit
# and return -Inf for everything past log P = -7.4e8.
# ---------------------------------------------------------------------------------------
function _tre_flr_logphi(x::BigFloat)
    s2 = sqrt(BigFloat(2))
    x < 0 ? log(erfcx(-x/s2)) - x*x/2 - log(BigFloat(2)) : log1p(-erfc(x/s2)/2)
end

# log P by a uniform trapezoid over a window supplied here rather than by the package: a
# different quadrature rule and an independently given peak, so only the mathematics is
# shared.  The integrand's logarithm is strictly concave, so beyond `span` widths it has
# fallen at least `span` nats and what is left outside is bounded by exp(-span).
function _tre_flr_ref(B, rho, gam, u0, w; n = 2401, span = 60)
    bB, br, bg = BigFloat.(B), BigFloat(rho), BigFloat(gam)
    h(u) = -u*u/2 - log(sqrt(2*BigFloat(pi))) + sum(_tre_flr_logphi((b - br*u)/bg) for b in bB)
    bu, bw = BigFloat(u0), BigFloat(w)
    M = h(bu)
    lo, hi = bu - span*bw, bu + span*bw
    step = (hi - lo)/(n - 1)
    acc = BigFloat(0)
    for k in 0:(n-1)
        acc += ((k == 0 || k == n-1) ? BigFloat(0.5) : BigFloat(1)) * exp(h(lo + k*step) - M)
    end
    M + log(acc*step)
end

@testset "panel_TRE orthant probability without the 1e-15 floor" begin

    # ------------------------------------------------------------------
    # 1. log P against a 256-bit reference over the range where the floor invented the answer.
    #    Columns: B/gamma (equal across the three components), the peak and the wider of the
    #    two one-nat widths handed to the reference, the true log P, and the nats the floor
    #    invented.  rho = 0.3, gamma = 1 throughout.  The package integrates to rtol 1e-10, so
    #    log P is held to 1e-9 nats (atol) or 1e-12 of its value, whichever is larger.
    # ------------------------------------------------------------------
    @testset "log P matches a 256-bit reference where the floor invented it" begin
        cases = ((-0.5,     -0.861902,     1.4167, -3.17060882206,    0.0),
                 (-2.0,     -1.730228,     1.3911, -9.60729566953,    0.0),
                 (-4.331,   -3.253395,     1.3774, -28.9011430574,    0.0),
                 (-8.0,     -5.777235,     1.3718, -84.0497578076,    49.511),
                 (-20.0,    -14.217910,    1.3691, -483.597705900,    449.059),
                 (-100.0,   -70.875139,    1.3686, -11826.9988211,    11792.5),
                 (-1000.0,  -708.662317,   1.3685, -1181125.24475,    1.18109e6),
                 (-1.0e5,   -70866.141740, 1.3685, -1.1811023658745e10, 1.1811e10))
        setprecision(BigFloat, 256) do
            for (a, u0, w, want, _) in cases
                B   = fill(a, 3)
                got = _SFO._tre_logorthant(B, 0.3, 1.0)
                ref = Float64(_tre_flr_ref(B, 0.3, 1.0, u0, w))
                @test isfinite(got)
                @test isapprox(got, ref;  rtol = 1e-12, atol = 1e-9)
                @test isapprox(got, want; rtol = 1e-11, atol = 1e-9)
            end
        end
    end

    # ------------------------------------------------------------------
    # 2. Unequal components, positive arguments, T = 1 and T = 20, a peak nowhere near zero,
    #    and the one-sided peak the N = 4, T = 10 fit walks into, where the curvature at the
    #    maximum is no guide to the width: h falls 112 nats one curvature unit to the RIGHT
    #    of u* and 0.63 nats forty units to the LEFT.  A single curvature-based width put
    #    log P 0.61 nats out there.
    # ------------------------------------------------------------------
    @testset "unequal components, both tails, and the one-sided peak" begin
        cases = ((collect(-3.0:1.0:1.0),      0.7,    1.3,    -2.018283,  1.1648, -6.28755417359,      true),
                 ([2.0, 1.5, 3.0],            0.4,    0.9,    -0.059258,  1.5045, -0.0869190945573,    true),
                 ([-30.0, -1.0, 0.5, -12.0],  0.9,    0.6,    -23.106760, 0.8567, -388.858314458,      true),
                 ([0.0, 0.0],                 0.5,    1.0,    -0.613426,  1.3620, -1.26568117084,      true),
                 ([-40.0],                    0.2,    1.0,    -7.697301,  1.5123, -773.819625809,      true),
                 (fill(-3.0, 20),             0.5,    1.0,    -6.489321,  0.7810, -32.0624207867,      true),
                 (fill(-1.0e5, 10),           1.0e-3, 1.0,    -999.99000, 1.5422, -4.9999500129319e10, true))
        setprecision(BigFloat, 256) do
            for (B, rho, gam, u0, w, want, refable) in cases
                got = _SFO._tre_logorthant(B, rho, gam)
                @test isfinite(got)
                @test isapprox(got, want; rtol = 1e-11, atol = 1e-9)
                if refable
                    @test isapprox(got, Float64(_tre_flr_ref(B, rho, gam, u0, w)); rtol = 1e-12, atol = 1e-9)
                end
            end
        end
        # The drift point's two widths differ by a factor of 279, so no single-scale reference
        # window resolves it; the value below is pinned against three separate quadrature rules
        # in 256-bit arithmetic, which agree to 5e-8 nats out of 349.
        drift = [-1.2655801484823, -1.2492066548288, -1.2328331611754, -1.2164596675219,
                 -1.2000861738685, -1.1837126802150, -1.1673391865616, -1.1509656929081,
                 -1.1345921992547, -1.1182187056012]
        @test isapprox(_SFO._tre_logorthant(drift, 0.04818642, 1.5794807e-6),
                       -349.093987285; rtol = 1e-9)
        u0, sL, sR = _SFO._tre_orthant_peak(drift, 0.04818642, 1.5794807e-6)
        @test isapprox(u0, -26.2643642; rtol = 1e-6)
        @test sL > 250*sR                                     # 0.0382 against 0.000137
        @test isapprox(sL, 0.03819085;   rtol = 1e-4)
        @test isapprox(sR, 0.0001368014; rtol = 1e-4)
    end

    # ------------------------------------------------------------------
    # 3. The slope.  On the flat side of the floor the reported log P did not depend on B at
    #    all, which is what let the optimiser walk into a fabricated optimum.
    # ------------------------------------------------------------------
    @testset "the slope is no longer identically zero" begin
        for (a, want) in ((-8.0, 6.41932633), (-20.0, 15.79769146), (-100.0, 78.75015471))
            B = fill(a, 3)
            g = ForwardDiff.gradient(b -> _SFO._tre_logorthant(b, 0.3, 1.0), B)
            @test all(isfinite, g)
            @test all(>(0), g)
            @test isapprox(g[1], want; rtol = 1e-6)
            # a central difference on the repaired form, at a step it actually varies over
            for k in 1:3
                hstep = 1e-5
                bp = copy(B); bp[k] += hstep
                bm = copy(B); bm[k] -= hstep
                fd = (_SFO._tre_logorthant(bp, 0.3, 1.0) -
                      _SFO._tre_logorthant(bm, 0.3, 1.0))/(2*hstep)
                @test isapprox(g[k], fd; rtol = 1e-6)
            end
        end
        # u*, the two widths and h(u*) are held fixed under differentiation, which is exact
        # because log P = h(u*) + log(sL*int + sR*int) holds for ANY such constants.  The
        # Hessian is taken by nesting ForwardDiff over itself, so check that path too.
        f(b) = _SFO._tre_logorthant(b, 0.3, 1.0)
        H  = ForwardDiff.hessian(f, fill(-20.0, 3))
        Hf = similar(H)
        for k in 1:3
            bp = fill(-20.0, 3); bp[k] += 1e-4
            bm = fill(-20.0, 3); bm[k] -= 1e-4
            Hf[:, k] = (ForwardDiff.gradient(f, bp) .- ForwardDiff.gradient(f, bm))./2e-4
        end
        @test all(isfinite, H)
        @test isapprox(H, Hf; rtol = 1e-5)
    end

    # ------------------------------------------------------------------
    # 4. bc is a conditional expectation of exp(-u) with u >= 0, so it lies in (0, 1].  The
    #    floor put it above 1 well before MCDF1 itself floored, because MCDF2 -- formed on the
    #    more negative B2 -- reaches the floor first.  The `was` column is what this same call
    #    returned on the pre-repair tree.
    # ------------------------------------------------------------------
    @testset "bc stays in (0, 1] and jlms stays positive" begin
        T   = 5
        pos = (endx = 2, begq = 3, begw = 4, begv = 5)
        idt = Matrix{Any}(undef, 1, 2); idt[1, 1] = collect(1:T); idt[1, 2] = T
        X   = hcat(ones(T), zeros(T))
        coef = [0.0, 0.0, log(0.09), log(0.25), log(0.0625)]   # sigma_a2, sigma_u2, sigma_v2
        cases = ((0.0,  0.27855042,  0.76939373, 0.76939373),
                 (1.0,  0.20039839,  0.82730886, 0.82730886),
                 (2.0,  0.15370363,  0.86393554, 0.86393554),
                 (3.0,  0.12333101,  0.88869690, 0.88869690),
                 (4.0,  0.10227421,  0.90635621, 2.8557922),
                 (6.0,  0.07540045,  0.92956333, 7.4187931),
                 (10.0, 0.04859910,  0.95358273, 27.535933))
        for (shift, jl, bcv, was) in cases
            Y = reshape(fill(shift, T), T, 1)
            jlms, bc = _SFO.jlmsbc(_SFO.PTREH, 1, pos, coef, Y, X, (), (), (), (), idt)
            @test all(isfinite, jlms) && all(isfinite, bc)
            @test all(>(0.0), vec(jlms))
            @test all(b -> 0.0 < b <= 1.0, vec(bc))
            @test isapprox(jlms[1], jl;  rtol = 1e-6)
            @test isapprox(bc[1],   bcv; rtol = 1e-6)
            was > 1.0 && @test bc[1] < 1.0 < was      # what the floor used to report
        end
    end

    # ------------------------------------------------------------------
    # 5. The likelihood at the parameter vector the old N = 4, T = 10 NelderMead(300) fit
    #    settled on.  It reported 11167.105 there; both floors were binding at once -- every
    #    firm's orthant probability replaced by log(1e-15), and the truncation constant one
    #    step from its own floor at mu/sigma_u = -7.61.
    # ------------------------------------------------------------------
    @testset "the fabricated optimum is gone" begin
        y, X, id, idt, n = _tre_flr_data(4, 10)
        Y = reshape(y, n, 1); one1 = ones(n, 1)
        rho = [0.22626705943151978, -0.0771724675011902, -1.15615238463842,
               0.3854242870505573, -3.7611916868682758, -26.716835948841837]
        got = -_SFO.LL_T(_SFO.PTRET, Y, X, one1, one1, one1, one1, 1, n,
                         _tre_flr_poT(), rho, idt, nothing)
        @test isfinite(got)
        @test !isapprox(got, 11167.10501226831; rtol = 1e-6)
        # A direct 256-bit integral over the random effect, sharing no algebra with the closed
        # form, gives -133.5363993, 1.4e-3 from the 512-bit reference below that the closed
        # form matches.
        # With only the orthant floor repaired this line returned +9882.87, because the three
        # quadratic forms were still being kept apart -- see the note in LL_T(PTREH).  Formed
        # apart, log P and the truncation constant put the value 3.3e-5 higher at this
        # sigma_v^2-clamp point (their separate quantisation); they are now combined.
        # sigma_u/sigma_v is about 1e5 here, where a single quadrature of the orthant integral
        # and the cancellation in B put the value 5.4e-6 off (-133.53501518928948); integrated
        # piecewise from a B formed without cancellation, it matches the 512-bit reference
        # -133.53500975118723.
        @test isapprox(got, -133.53500975118723; rtol = 1e-9)
        @test abs(got - (-133.5363993)) < 0.01
        # The THIRD cancellation, at the parameter vector a NelderMead(300) fit of this panel
        # settled on: sigma_u^2 at its 1e-12 clamp with mu = -12.004, so mu/sigma_u = -1.2e7
        # and every orthant argument tends to it.  log P and the truncation constant
        # -T log Phi(mu/sigma_u) are each about -5.9e14 while their sum is O(1); formed
        # apart, the Float64 difference is quantised at 0.06, the likelihood sat 2.0 nats
        # above a direct 256-bit integral over the random effect (46.45774984), and bc came
        # out 1.101 for three of forty observations.
        # Combined before either is formed (_tre_logorthant_ratio for the likelihood,
        # _tre_logorthant_diff for bc's log-probability difference), the closed form
        # agrees with the direct integral to 3e-11 here and max bc = 1 - 7.0e-14.
        rho4 = [0.5174730927622206, -0.31766953286416344, -12.004456321841776,
                -3.3915792532218925, -72.88297174248268, -5.565242008642048]
        got4 = -_SFO.LL_T(_SFO.PTRET, Y, X, one1, one1, one1, one1, 1, n,
                          _tre_flr_poT(), rho4, idt, nothing)
        @test abs(got4 - 46.45774984) < 1e-7
        pos4 = (endx = 2, begz = 3, endz = 3, begq = 4, begw = 5, begv = 6)
        jl4, bc4 = _SFO.jlmsbc(_SFO.PTRET, 1, pos4, rho4, Y, X, one1, one1, one1, one1, idt)
        @test all(b -> 0.0 < b <= 1.0, vec(bc4))
        @test all(x -> isfinite(x) && x > 0.0, vec(jl4))
        # and the fit itself no longer reports every firm perfectly efficient
        sp = sfmodel_spec(datatype = :panel_TRE, depvar = y, frontier = X, id = id,
                          type = :production, noise = :Normal, ineff = :TruncatedNormal)
        ini = sfmodel_init(spec = sp, frontier = [0.44, -0.16], mu = 0.08, sigma_a_sq = 0.12,
                           ln_sigma_u_sq = -1.2, ln_sigma_v_sq = -2.2)
        o = sfmodel_opt(warmstart_solver = nothing, main_solver = NelderMead(),
                        main_opt = (iterations = 300, g_abstol = 1e-8))
        r = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MLE), init = ini, optim_options = o,
                jlms_bc_index = true, marginal = false, show_table = false, verbose = false)))
        @test isfinite(r.loglikelihood)
        @test Float64(r.loglikelihood) < 1000.0            # was 11167.105
        @test sum(Float64.(vec(r.jlms))) > 0.0             # was exactly 0.0
        @test all(isfinite, Float64.(vec(r.bc)))
        @test all(>(0.0), Float64.(vec(r.bc)))
        # With log P and
        # the truncation constant combined, the likelihood is honest at the sigma_u^2
        # clamp and this fit now walks deeper still (mu = -41, ln sigma_a^2 = -12: the
        # truncated-normal likelihood on this panel genuinely rises toward the
        # no-inefficiency boundary, and its value at the fit's own optimum agrees with the
        # direct 256-bit integral to 2e-9).  Every bc is 1 to double
        # precision there: at mu/sigma_u = -4e7 the true 1 - bc is about 2e-14, BELOW the
        # bc exponent's own rounding (one ulp of B_j = -41 is 7e-15), so a few ulps above
        # 1.0 can and do appear -- 1 + 2.7e-15 on two of forty observations, the honest
        # double-precision reading of a number that is 1 - 2e-14.  Anything materially
        # above 1 -- the 1.101 the quantised difference used to report here -- fails this
        # bound (the exact 1.0-for-all-firms the floors fabricated before that fails the jlms
        # sum above). Where the fit ends depends on NelderMead's path, so the bound leaves
        # room for rounding: 1e-10, not a count of ulps.
        @test maximum(Float64.(vec(r.bc))) <= 1.0 + 1e-10      # was 1.101
    end

    # ------------------------------------------------------------------
    # 6. Where the floor never bound, nothing moves.  The half-normal fit of a seven-firm,
    #    three-period panel (_tre_flr_data(7, 3)) is unchanged, and its jlms/bc sums with it:
    #    to 1e-8, as the integrals are computed to rtol 1e-10 and their rounding can differ
    #    between machines.
    # ------------------------------------------------------------------
    @testset "inert where the floor never bound" begin
        y, X, id, idt, n = _tre_flr_data(7, 3)
        Y = reshape(y, n, 1); one1 = ones(n, 1)
        for (la, lu, lv) in ((0.0, -1.0, -2.3), (0.0, -1.0, -4.6), (-1.0, -2.0, -6.9),
                             (0.0, 0.0, 0.0))
            got = -_SFO.LL_T(_SFO.PTREH, Y, X, (), one1, one1, one1, 1, n,
                             _tre_flr_poH(), [0.62, -0.24, la, lu, lv], idt, nothing)
            @test isfinite(got)
        end
        sp = sfmodel_spec(datatype = :panel_TRE, depvar = y, frontier = X, id = id,
                          type = :production, noise = :Normal, ineff = :HalfNormal)
        ini = sfmodel_init(spec = sp, frontier = [0.44, -0.16], sigma_a_sq = 0.12,
                           ln_sigma_u_sq = -1.2, ln_sigma_v_sq = -2.2)
        o = sfmodel_opt(warmstart_solver = nothing, main_solver = NelderMead(),
                        main_opt = (iterations = 8, g_abstol = 1e-8))
        r = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MLE), init = ini, optim_options = o,
                jlms_bc_index = true, marginal = false, show_table = false, verbose = false)))
        # the pre-repair tree reported -0.9849289654764188, 3.933890663179996, 17.53930923897809
        @test isapprox(Float64(r.loglikelihood), -0.9849289654764188; rtol = 1e-8)
        @test isapprox(sum(Float64.(vec(r.jlms))),  3.933890663179996; rtol = 1e-8)
        @test isapprox(sum(Float64.(vec(r.bc))),   17.539309238978090; rtol = 1e-8)
    end
end
