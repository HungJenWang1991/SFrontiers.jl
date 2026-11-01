# Regression test: datatype =
# :panel_TRE where σᵥ is very small next to σᵤ.
#
# Each firm's likelihood contains P = ∫ φ(u) ∏ₜ Φ(aₜ(u)) du, aₜ(u) = (Bₜ - ρu)/γ, integrated by
# quadgk on each side of the peak over sixty one-nat widths. When σᵤ/σᵥ is large every factor
# turns from 1 to its Gaussian tail within a few 1/r of u (r = ρ/γ), far less than the widths,
# and one adaptive rule per side stepped over that turn: at σᵤ/σᵥ = 1000 to 1e6 the Hessians of
# LL_T(PTREH) and LL_T(PTRET) came out wrong by up to 1.9 times their largest entry (24 times
# for PTRET at 1e4), gradients by up to 1.6e-3; LL_T(PTRET), whose B is formed from
# 1/σᵥ²-sized terms that cancel, was off by 3e4 nats at 1e6; jlms and bc by up to 9e-4.
# The fix: where z = r·max(sL, sR) > 160, _tre_logorthant (LL_T(PTREH)) integrates piecewise,
# with breakpoints at the turns of the factors, through TRE-only copies of the integrand
# differences and from a cancellation-free B; LL_T(PTRET)'s helper does so from D and m (no
# cancellation) wherever r·√T > 100, and so does jlmsbc of both models, per firm, where PTREH's
# bc then takes its log-probability difference in one piece; a Dual method of
# _tre_invmills below a = -1e3. Below the switches every number is unchanged, apart
# from Hessians that meet an argument below -1e3.
#
# References: 512 bits, in log space. The closed form (B, G, ρ, γ, the determinants, the
# quadratic form) is evaluated at 512 bits; log P is a composite 16-node (24-node for the
# likelihood at σᵤ/σᵥ = 4e5, below) Gauss-Legendre rule on panels graded geometrically
# (factor 2) away from the peak and placed at (aₜ(u*) + j)/r, j = 0, ±1, ±3, ±10, ±30, around
# every turn, skipping panels where h is more than 100 nats down; log Φ is log(erfc(-a/√2)/2),
# or its asymptotic series below -30.
# The panels are placed from Float64 numbers (the package's peak and widths only place them),
# so ForwardDiff over BigFloat differentiates the discrete sum exactly. The Hessian blocks are
# central differences of that 512-bit gradient, with every firm's panels placed once, at the
# point itself, and used at both shifted points. The rule's own error moves with the turns,
# which move by r per unit of the parameters, so it enters the Hessian about r² times larger
# than the value: with 16 nodes (as in the first version of this test) 6.48e-22 and 8.26e-22
# in the value put 8.95e-11 and 1.03e-10 into the Hessian blocks at σᵤ/σᵥ = 4e5. With 24 nodes
# the blocks there agree with an independent reference to 2.60e-24 of their
# largest entry, and 32 nodes or panels of half the width move them by at most 5.26e-20; at
# 1e3 and 2e3 the 16-node blocks agree with that reference to 1.14e-14. The data are fixed
# numbers, not random draws.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using StatsFuns
using QuadGK
using LinearAlgebra

const _SF43 = SFrontiers.MLE_Backend.SFmle
const _B43_DR = ForwardDiff.DiffResults

_b43_val(x::Real) = Float64(x)
_b43_val(x::ForwardDiff.Dual) = _b43_val(ForwardDiff.value(x))
_b43_rel(A, R) = maximum(abs.(A .- R)) / maximum(abs.(R))

# a 2-firm, T = 5 panel (no random numbers): from σᵤ/σᵥ = 1000 up, firm 1's peak is one-sided
# (narrow/wide width 0.012 and less) and firm 2's peak sits at u = 0 with every Φ = 1 and a turn
# about one width to its right (narrow/wide 0.15 under PTREH, 0.71 under PTRET)
function _b43_panel()
    nf, nt = 2, 5
    n  = nf*nt
    X  = hcat(ones(n), collect(range(-1, 1; length = n)))
    id = repeat(1:nf, inner = nt)
    y  = X*[1.0, 0.5] .+ [0.3*cos(1.7i) for i in id] .+ 1e-3 .* cos.(1.31 .* (1:n)) .-
         abs.(0.6 .* sin.(0.83 .* (1:n)))
    idt = Matrix{Any}(undef, nf, 2)
    for i in 1:nf
        idt[i, 1] = collect(((i-1)*nt+1):(i*nt)); idt[i, 2] = nt
    end
    return reshape(y, n, 1), X, ones(n, 1), idt, n
end
const _B43_POH = (begx=1, endx=2, begz=0, endz=0, begq=3, endq=3, begw=4, endw=4, begv=5, endv=5)
const _B43_POT = (begx=1, endx=2, begz=3, endz=3, begq=4, endq=4, begw=5, endw=5, begv=6, endv=6)
# σₐ² = 0.09 and σᵤ² = 0.25; σᵥ² = 0.25/ratio² stays inside the package's bound [1e-12, 1e12]
# up to ratio = 5e5, so the largest ratio below is 4e5
_b43_θ(model, ratio) = model === :H ? [1.0, 0.5, log(0.09), log(0.25), log(0.25/ratio^2)] :
                                      [1.0, 0.5, 0.2, log(0.09), log(0.25), log(0.25/ratio^2)]
# a one-firm panel: T = 15, PTRET at
# σᵤ/σᵥ = 1000, μ/σᵤ = 0.6, σₐ²/σᵤ² = 4, data drawn with σᵥ = σᵤ/2 (what a fit running towards
# σᵥ → 0 meets). z = 83.7 lies below the layout switch, r·√T = 991.8 above jlmsbc's switch; on
# the old layout one quadgk rule per side mis-integrated the jlms numerator's wide side, and
# jlms was 1.4e-5 off.
function _b43_s1_firm()
    y = [-1.0184987270766965, -0.9669823554868817, -0.9540953128595192, -2.0185937158209017, 2.698495628432862,
         1.793839919447492, 1.4941393944634982, 0.07717707279743724, 0.4468234825417947, -0.5590230494515114,
         -1.9620034250165954, 1.9830942211997158, 2.7435002046829164, 0.2666858476874102, -1.7761525835933605]
    x = [-0.2754902842712705, -0.8341797468599144, -2.279400392932779, -0.890244395405134, 1.521987365267431,
         0.7413773145396982, 0.46131145098801374, 0.45898463994505817, -0.06511660219118143, 0.7780381633880828,
         -0.6922560061734341, 1.4877778619200495, 0.648118582842652, -0.06870148284216908, -1.4439314204086]
    θ = [0.5164600270202946, 1.0202020503943092, 0.848528137423857, 2.0794415416798357, 0.6931471805599453,
         -13.122363377404328]
    idt = Matrix{Any}(undef, 1, 2); idt[1, 1] = collect(1:15); idt[1, 2] = 15
    return reshape(y, 15, 1), hcat(ones(15), x), ones(15, 1), idt, θ
end
# The functions ForwardDiff differentiates, as one callable type per model: ForwardDiff tags its
# Duals by the function's type, so one type per model lets every check below share the compiled
# code. role 0: the likelihood LL_T; role 1: firm i's log orthant term through the package's
# helper; role 2: the same through the pre-fix copy of the helper (defined below).
struct _B43F{M}
    role::Int; i::Int; Y::Matrix{Float64}; X::Matrix{Float64}; Z::Matrix{Float64}; idt::Matrix{Any}; n::Int
end
_B43LL(model, Y, X, Z, idt, n) = _B43F{model}(0, 0, Y, X, Z, idt, n)
function (f::_B43F{M})(t) where {M}
    if f.role == 0
        return M === :H ?
            -_SF43.LL_T(_SF43.PTREH, f.Y, f.X, (), f.Z, f.Z, f.Z, 1, f.n, _B43_POH, t, f.idt, nothing) :
            -_SF43.LL_T(_SF43.PTRET, f.Y, f.X, f.Z, f.Z, f.Z, f.Z, 1, f.n, _B43_POT, t, f.idt, nothing)
    end
    p = _b43_parts(t, M, f.Y, f.X, f.idt, f.i)
    if M === :H
        return f.role == 1 ? _SF43._tre_logorthant(p.B, p.ρ, p.γ) : _b43_old_logorthant(p.B, p.ρ, p.γ)
    else
        return f.role == 1 ? _SF43._tre_logorthant_ratio(p.B, p.ρ, p.γ, p.m, p.D) : _b43_old_ratio(p.B, p.ρ, p.γ, p.m, p.D)
    end
end
_b43_jlmsbc(model, θ, Y, X, Z, idt) = model === :H ?
    _SF43.jlmsbc(_SF43.PTREH, 1, (endx = 2, begq = 3, begw = 4, begv = 5), θ, Y, X, (), (), (), (), idt) :
    _SF43.jlmsbc(_SF43.PTRET, 1, (endx = 2, begz = 3, endz = 3, begq = 4, begw = 5, begv = 6), θ, Y, X, Z, Z, Z, Z, idt)

# ---------------------------------------------------------------------------------------------
# the 512-bit reference
# ---------------------------------------------------------------------------------------------
function _b43_logPhi(a)
    a >= -30 && return log(erfc(-a/sqrt(big(2)))/2)
    x = -a; v = 1/(x*x); s = one(x); w = one(x); k = 1
    while true
        w *= -(2k - 1)*v
        abs(_b43_val(w)) < 1e-157 && break
        s += w; k += 1
    end
    return -x*x/2 - log(x) - log(sqrt(2*big(pi))) + log(s)
end
const _B43_GL = Ref{Any}(nothing)
function _b43_grid(Bv, ρv, γv)
    u0, sL, sR = _SF43._tre_orthant_peak(Bv, ρv, γv)
    r  = ρv/γv
    a0 = (Bv .- ρv*u0)./γv
    lo, hi = -60sL, 60sR
    pts = Float64[lo, 0.0, hi]
    e0 = min(sL, sR, 1/r)/32
    for (sg, w) in ((-1.0, sL), (1.0, sR))
        d = e0
        while d < 60w; push!(pts, sg*d); d *= 2; end
    end
    for a in a0, j in (0, 1, -1, 3, -3, 10, -10, 30, -30)
        p = (a + j)/r
        lo < p < hi && push!(pts, p)
    end
    sort!(unique!(pts))
    drop(du) = -_SF43._tre_dh(a0, r, u0, du)               # placement only (Float64)
    keep = [k for k in 1:length(pts)-1 if min(drop(pts[k]), drop(pts[k+1])) < 100]
    return u0, [(pts[k], pts[k+1]) for k in keep]
end
# B, ρ, γ at 512 bits (or Duals); the grid is placed from their Float64 values unless given
function _b43_logP(B, ρ, γ, grid = _b43_grid(_b43_val.(B), _b43_val(ρ), _b43_val(γ)))
    u0, pieces = grid
    x, w = _B43_GL[]
    U0 = big(u0)
    h(du) = -(U0 + du)^2/2 + sum(_b43_logPhi((B[k] - ρ*(U0 + du))/γ) for k in eachindex(B))
    h0 = h(zero(U0))
    acc = zero(h0)
    for (A, C) in pieces
        half = (big(C) - big(A))/2; mid = (big(C) + big(A))/2
        for j in eachindex(x)
            acc += w[j]*half*exp(h(mid + half*x[j]) - h0)
        end
    end
    return h0 - log(sqrt(2*big(pi))) + log(acc)
end
# a firm's closed form at 512 bits. Every matrix here is aI + b11', whose inverse is
# (1/a)I - (b/(a(a + Tb)))11' and whose log determinant is (T-1)log a + log(a + Tb); at 512 bits
# the cancellations that the Float64 code has to avoid are harmless.
_b43_inv(a, b, T) = (1/a, -b/(a*(a + T*b)))                     # (a', b') of the inverse
_b43_logdet(a, b, T) = (T - 1)*log(a) + log(a + T*b)
_b43_mul(a, b, v) = a .* v .+ b*sum(v)                            # (aI + b11')v
function _b43_firm(θ, model, y, Xi)
    T  = length(y)
    β  = θ[1:2]
    δ  = model === :H ? zero(θ[1]) : θ[3]
    σa2, σu2, σv2 = exp(θ[end-2]), exp(θ[end-1]), exp(θ[end])
    ε  = BigFloat.(y) .- BigFloat.(Xi)*β
    μ  = fill(δ, T)
    ia, ib = _b43_inv(σv2, σa2, T)                                # Σ = σv2 I + σa2 11'
    ga, gb = _b43_inv(ia + 1/σu2, ib, T)                          # G = (Σ⁻¹ + Σu⁻¹)⁻¹
    B  = _b43_mul(ga, gb, μ ./ σu2 .- _b43_mul(ia, ib, ε))
    G  = ga .* Matrix{eltype(B)}(I, T, T) .+ gb
    sa, sb = _b43_inv(σv2 + σu2, σa2, T)                          # (Σ + Σu)⁻¹
    z  = ε .+ μ
    quad = -sum(z .* _b43_mul(sa, sb, z))/2
    σu = sqrt(σu2)
    return (B = B, G = G, ρ = sqrt(gb), γ = sqrt(ga),
            rest = _b43_logdet(ga, gb, T)/2 - _b43_logdet(σv2, σa2, T)/2 - T*log(σu) -
                   sum(_b43_logPhi(μ[t]/σu) for t in 1:T) + quad - T*log(2*big(pi))/2)
end
# every firm's grid at θ, placed from Float64 numbers as _b43_logP places it
_b43_ref_grids(θ, model, Y, X, idt) =
    [(f = _b43_firm(θ, model, Y[idt[i, 1]], X[idt[i, 1], :]); _b43_grid(_b43_val.(f.B), _b43_val(f.ρ), _b43_val(f.γ)))
     for i in 1:size(idt, 1)]
function _b43_ref_ll(θ, model, Y, X, idt, grids)
    tot = zero(θ[1])
    for i in 1:size(idt, 1)
        ind = idt[i, 1]
        f = _b43_firm(θ, model, Y[ind], X[ind, :])
        tot += _b43_logP(f.B, f.ρ, f.γ, grids[i]) + f.rest
    end
    return tot
end
struct _B43Ref
    model::Symbol; Y::Matrix{Float64}; X::Matrix{Float64}; idt::Matrix{Any}; grids::Vector{Any}
end
(r::_B43Ref)(s) = _b43_ref_ll(s, r.model, r.Y, r.X, r.idt, r.grids)
# the Hessian block blk: central differences (step 1e-20) of the 512-bit AD gradient, on the
# grids placed at θb (so that a Float64 rounding that differs between θb + h and θb - h cannot
# move a panel, whose effect the difference would divide by 2h); the truncation error is of
# order 1e-40 and the rounding about 1e-134
function _b43_ref_hess(θb, blk, model, Y, X, idt, grids)
    g(t) = ForwardDiff.gradient(_B43Ref(model, Y, X, idt, grids), t)
    H = zeros(BigFloat, length(blk), length(blk))
    for (c, k) in enumerate(blk)
        h = big(1e-20)*max(abs(θb[k]), 1)
        tp = copy(θb); tp[k] += h; tm = copy(θb); tm[k] -= h
        H[:, c] = (g(tp)[blk] .- g(tm)[blk]) ./ (2h)
    end
    return H
end
#   jlms_j = γ ∫ φ(u) ∏ Φ(aₜ) m(a_j) du / ∫ φ(u) ∏ Φ(aₜ) du,  m(a) = a + φ(a)/Φ(a);
#   bc_j   = exp(log P(B - G e_j) - log P(B) - B_j + G_jj/2)
function _b43_ref_jlmsbc(θ, model, Y, X, idt)
    jl = BigFloat[]; bc = BigFloat[]
    for i in 1:size(idt, 1)
        ind = idt[i, 1]; T = idt[i, 2]
        f = _b43_firm(θ, model, Y[ind], X[ind, :])
        u0, pieces = _b43_grid(Float64.(f.B), Float64(f.ρ), Float64(f.γ))
        x, w = _B43_GL[]
        U0 = big(u0)
        a(du, k) = (f.B[k] - f.ρ*(U0 + du))/f.γ
        h(du) = -(U0 + du)^2/2 + sum(_b43_logPhi(a(du, k)) for k in 1:T)
        h0 = h(big(0.0))
        num = zeros(BigFloat, T); den = big(0.0)
        for (A, C) in pieces
            half = (big(C) - big(A))/2; mid = (big(C) + big(A))/2
            for jj in eachindex(x)
                du = mid + half*x[jj]
                e = w[jj]*half*exp(h(du) - h0)
                den += e
                for j in 1:T
                    aj = a(du, j)
                    num[j] += e*(aj + exp(-aj^2/2 - log(sqrt(2*big(pi))) - _b43_logPhi(aj)))
                end
            end
        end
        append!(jl, f.γ .* num ./ den)
        lp1 = _b43_logP(f.B, f.ρ, f.γ)
        append!(bc, [exp(_b43_logP(f.B .- f.G[:, j], f.ρ, f.γ) - lp1 - f.B[j] + f.G[j, j]/2) for j in 1:T])
    end
    return jl, bc
end

# ---------------------------------------------------------------------------------------------
# the pre-fix helpers and jlmsbc, statement for statement, for the bit-identity asserts
# ---------------------------------------------------------------------------------------------
function _b43_old_two(f, sL, sR)
    sL*quadgk(t -> f(-sL*t), 0, _SF43._TRE_SPAN, rtol = 1e-10)[1] +
    sR*quadgk(t -> f( sR*t), 0, _SF43._TRE_SPAN, rtol = 1e-10)[1]
end
function _b43_old_logorthant(B, ρ, γ)
    u0, sL, sR = _SF43._tre_orthant_peak(B, ρ, γ)
    r   = ρ/γ
    a₀  = (B .- ρ*u0)./γ
    hu0 = normlogpdf(u0) + sum(_SF43._stable_normlogcdf, a₀)
    isfinite(_b43_val(hu0)) || return hu0
    I = sL*quadgk(t -> exp(_SF43._tre_dh(a₀, r, u0, -sL*t)), 0, _SF43._TRE_SPAN, rtol = 1e-10)[1] +
        sR*quadgk(t -> exp(_SF43._tre_dh(a₀, r, u0,  sR*t)), 0, _SF43._TRE_SPAN, rtol = 1e-10)[1]
    return hu0 + log(I)
end
function _b43_old_ratio(B, ρ, γ, m, D)
    u0, sL, sR = _SF43._tre_orthant_peak(B, ρ, γ)
    r   = ρ/γ
    a₀  = (B .- ρ*u0)./γ
    hr0 = normlogpdf(u0) + sum(_SF43._tre_dlogcdf(m[k], (D[k] - ρ*u0)/γ) for k in eachindex(m))
    isfinite(_b43_val(hr0)) || return hr0
    I = sL*quadgk(t -> exp(_SF43._tre_dh(a₀, r, u0, -sL*t)), 0, _SF43._TRE_SPAN, rtol = 1e-10)[1] +
        sR*quadgk(t -> exp(_SF43._tre_dh(a₀, r, u0,  sR*t)), 0, _SF43._TRE_SPAN, rtol = 1e-10)[1]
    return hr0 + log(I)
end
function _b43_old_diff(B, ρ, γ, S)
    u01, sL1, sR1 = _SF43._tre_orthant_peak(B, ρ, γ)
    B2 = B .+ S
    u02, sL2, sR2 = _SF43._tre_orthant_peak(B2, ρ, γ)
    r   = ρ/γ
    a₀1 = (B  .- ρ*u01)./γ
    a₀2 = (B2 .- ρ*u02)./γ
    hd  = -(u02 - u01)*(u02 + u01)/2 +
          sum(_SF43._tre_dlogcdf(a₀1[k], (S[k] - ρ*(u02 - u01))/γ) for k in eachindex(S))
    isfinite(_b43_val(hd)) || return hd
    I1 = _b43_old_two(du -> exp(_SF43._tre_dh(a₀1, r, u01, du)), sL1, sR1)
    I2 = _b43_old_two(du -> exp(_SF43._tre_dh(a₀2, r, u02, du)), sL2, sR2)
    return hd + log(I2/I1)
end
function _b43_old_jlms(B, ρ, γ)
    u0, sL, sR = _SF43._tre_orthant_peak(B, ρ, γ)
    r  = ρ/γ
    a₀ = (B .- ρ*u0)./γ
    D = _b43_old_two(du -> exp(_SF43._tre_dh(a₀, r, u0, du)), sL, sR)
    return [γ*_b43_old_two(du -> exp(_SF43._tre_dh(a₀, r, u0, du))*_SF43._tre_millsm(a₀[j] - r*du), sL, sR)/D
            for j in eachindex(B)]
end
function _b43_old_jlmsbc(model, coef, Y, X, Z, idt)
    nofobs = length(Y)
    β   = coef[1:2]
    σₐ² = clamp(exp(coef[end-2]), 1e-12, 1e12)
    σᵤ² = clamp(exp(coef[end-1]), 1e-12, 1e12)
    σᵥ² = clamp(exp(coef[end]),   1e-12, 1e12)
    ε   = 1*(Y - X * β)
    μ   = model === :H ? zeros(nofobs, 1) : (Z * coef[3:3])
    jlms = zeros(nofobs, 1)
    bc   = zeros(nofobs, 1)
    for i in 1:size(idt, 1)
        @views ind  = idt[i, 1]
        @views T    = idt[i, 2]
              invΣ  = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
              invΣᵤ = (1/σᵤ²) * Matrix(I,T,T)
        @views K    = ε[ind]'*invΣ - μ[ind]'*invΣᵤ
        c    = 1/σᵤ² + 1/σᵥ²
        cTd  = 1/σᵤ² + 1/(σᵥ² + T*σₐ²)
        γ²   = 1/c
        ρ²   = σₐ²/(σᵥ²*(σᵥ² + T*σₐ²)) / (c*cTd)
        G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)
        B    = (-K*G)'
        γ    = sqrt(γ²)
        ρ    = sqrt(ρ²)
        if model === :H
            logMCDF1 = _b43_old_logorthant(B, ρ, γ)
            jlms[ind] = _b43_old_jlms(B, ρ, γ)
            bc_i = zeros(T)
            for j in 1:T
               l_T    = zeros(T)
               l_T[j] = 1.0
               K2 = K + l_T'
               B2 = (-K2*G)'
               bc_i[j] = exp(_b43_old_logorthant(B2, ρ, γ) - logMCDF1 - B[j] + 0.5*G[j,j])
            end
            bc[ind] = bc_i
        else
            jlms[ind] = _b43_old_jlms(B, ρ, γ)
            bc[ind] = [exp(_b43_old_diff(B, ρ, γ, -G[:, j]) - B[j] + 0.5*G[j,j]) for j in 1:T]
        end
    end
    return jlms, bc
end
# a firm's B, ρ, γ (and m, D for PTRET), as LL_T forms them
function _b43_parts(θ, model, Y, X, idt, i)
    β   = θ[1:2]
    σₐ² = exp(θ[end-2]); σᵤ² = exp(θ[end-1]); σᵥ² = exp(θ[end]); σᵤ = sqrt(σᵤ²)
    ind = idt[i, 1]; T = idt[i, 2]
    ε   = (Y - X*β)[ind]
    μ   = model === :H ? zeros(eltype(ε), T) : fill(θ[3], T)
    invΣ  = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
    invΣᵤ = (1/σᵤ²) * Matrix(I,T,T)
    K    = ε'*invΣ - μ'*invΣᵤ
    c    = 1/σᵤ² + 1/σᵥ²
    cTd  = 1/σᵤ² + 1/(σᵥ² + T*σₐ²)
    γ²   = 1/c
    ρ²   = σₐ²/(σᵥ²*(σᵥ² + T*σₐ²)) / (c*cTd)
    G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)
    B    = (-K*G)'
    m    = μ ./ σᵤ
    q    = σᵤ²/σᵥ²
    εbar = sum(ε)/T
    D    = (-q/((1+q)*(1+sqrt(1+q)))) .* μ .+ (ρ²/σᵤ²)*sum(μ) .-
           (γ²/σᵥ²).*(ε .- (T*σₐ²/(σᵥ² + T*σₐ²))*εbar) .- ρ²*T*εbar/(σᵥ² + T*σₐ²)
    return (B = B, ρ = sqrt(ρ²), γ = sqrt(γ²), G = G, m = m, D = D)
end

@testset "panel_TRE where sigma_v is very small next to sigma_u" begin
    Y, X, Z, idt, n = _b43_panel()

    # -----------------------------------------------------------------------------------------
    # 1. LL_T against 512 bits: value, gradient and the Hessian block of (ln sigma_u^2,
    #    ln sigma_v^2), at sigma_u/sigma_v = 1e3 and 4e5 (PTRET) and 2e3 and 4e5 (PTREH; on this
    #    panel the old layout happens to be right for PTREH at 1e3). At 4e5, sigma_v^2 = 1.5625e-12
    #    lies just inside the package's bound. Measured after the fix:
    #    PTREH at 2e3 within 3.77e-15, 1.87e-15 and 8.14e-13 (before the fix
    #    1.42e-7, 1.86e-4 and 0.330; without the cancellation-free B in LL_T(PTREH) 5.19e-10,
    #    3.96e-10 and 9.57e-10); at 4e5 within 1.13e-14, 3.88e-14 and 1.14e-8 (PTREH) and 1.27e-14,
    #    6.09e-13 and 1.22e-8 (PTRET), where the package was off by 5.07e-5, 1.92e-5 and 0.328
    #    (PTREH) and 2.68e-5, 7.06e-2 and 4.17e3 (PTRET) before the fix. Each tolerance is at least
    #    about 80 times the error measured, so that rounding and the quadrature's subdivision (the
    #    package integrates to rtol 1e-10) leave room.
    # -----------------------------------------------------------------------------------------
    setprecision(BigFloat, 512) do
        for (model, ratio, tv, tg, tH) in ((:T, 1e3, 1e-11, 1e-10, 1e-8), (:T, 4e5, 1e-12, 1e-10, 1e-6),
                                           (:H, 2e3, 1e-11, 1e-11, 1e-10), (:H, 4e5, 1e-12, 1e-10, 1e-6))
            @testset "LL_T(PTRE$(model)) at sigma_u/sigma_v = $(ratio)" begin
                _B43_GL[] = QuadGK.gauss(BigFloat, ratio > 1e5 ? 24 : 16)
                θ = _b43_θ(model, ratio)
                f = _B43LL(model, Y, X, Z, idt, n)
                blk = model === :H ? [4, 5] : [5, 6]
                θb = BigFloat.(θ)
                grids = _b43_ref_grids(θb, model, Y, X, idt)
                vr = Float64(_b43_ref_ll(θb, model, Y, X, idt, grids))
                gr = Float64.(ForwardDiff.gradient(_B43Ref(model, Y, X, idt, grids), θb))
                Hr = Float64.(_b43_ref_hess(θb, blk, model, Y, X, idt, grids))
                @test abs(f(θ) - vr) < tv
                @test _b43_rel(ForwardDiff.gradient(f, θ), gr) < tg
                @test _b43_rel(ForwardDiff.hessian(f, θ)[blk, blk], Hr) < tH
            end
        end

        # -------------------------------------------------------------------------------------
        # 2. jlms and bc past jlmsbc's switch, against 512 bits (PTREH's bc as one log-probability difference): at 4e5,
        #    where they were off by up to 1.78e-5 before the fix, and the T = 15 firm of
        #    _b43_s1_firm (PTRET, 1000), whose jlms was 1.40e-5 off before the fix and in its
        #    first version. Measured after the fix: at most 8.88e-16 at 4e5 and 5.23e-16 for the
        #    T = 15 firm. jlms and bc are ratios of integrals and take no derivative, so the
        #    references use 16 nodes here (24 nodes give the same Float64 numbers).
        # -------------------------------------------------------------------------------------
        _B43_GL[] = QuadGK.gauss(BigFloat, 16)
        @testset "jlms and bc at sigma_u/sigma_v = 4e5" begin
            for model in (:H, :T)
                θ = _b43_θ(model, 4e5)
                jl, bc = _b43_jlmsbc(model, θ, Y, X, Z, idt)
                jr, br = _b43_ref_jlmsbc(BigFloat.(θ), model, Y, X, idt)
                @test maximum(abs.(vec(jl) .- Float64.(jr)) ./ Float64.(jr)) < 1e-13
                @test maximum(abs.(vec(bc) .- Float64.(br)) ./ Float64.(br)) < 1e-13
                @test all(b -> 0.0 < b <= 1.0, vec(bc))
            end
        end
        @testset "jlms and bc of a firm below the layout switch" begin
            Y1, X1, Z1, idt1, θ1 = _b43_s1_firm()
            jl, bc = _b43_jlmsbc(:T, θ1, Y1, X1, Z1, idt1)
            jr, br = _b43_ref_jlmsbc(BigFloat.(θ1), :T, Y1, X1, idt1)
            @test maximum(abs.(vec(jl) .- Float64.(jr)) ./ Float64.(jr)) < 1e-13
            @test maximum(abs.(vec(bc) .- Float64.(br)) ./ Float64.(br)) < 1e-13
        end
    end

    # -----------------------------------------------------------------------------------------
    # 3. Below the switches nothing moves: the helpers equal their pre-fix copies bit for bit
    #    (value ===, gradient and Hessian ==) at sigma_u/sigma_v = 100 (z at most 57,
    #    r*sqrt(T) = 80) and, for the layout switch, 250 (z at most 127); jlmsbc equals the
    #    pre-fix jlmsbc at 100 and at 120 (r*sqrt(T) = 96.2, just below jlmsbc's switch).
    # -----------------------------------------------------------------------------------------
    @testset "below the switches, bit for bit" begin
        for (model, ratio, hess) in ((:H, 100.0, false), (:H, 250.0, true), (:T, 100.0, true))
            θ = _b43_θ(model, ratio)
            for i in 1:2
                fnew = _B43F{model}(1, i, Y, X, Z, idt, n); fold = _B43F{model}(2, i, Y, X, Z, idt, n)
                @test fnew(θ) === fold(θ)
                @test ForwardDiff.gradient(fnew, θ) == ForwardDiff.gradient(fold, θ)
                hess && @test ForwardDiff.hessian(fnew, θ) == ForwardDiff.hessian(fold, θ)
                p = _b43_parts(θ, model, Y, X, idt, i)
                S = -p.G[:, 2]
                @test _SF43._tre_logorthant_diff(p.B, p.ρ, p.γ, S) === _b43_old_diff(p.B, p.ρ, p.γ, S)
                @test _SF43._tre_orthant_jlms(p.B, p.ρ, p.γ) == _b43_old_jlms(p.B, p.ρ, p.γ)
            end
        end
        for model in (:H, :T), ratio in (100.0, 120.0)
            θ = _b43_θ(model, ratio)
            jl, bc = _b43_jlmsbc(model, θ, Y, X, Z, idt)
            jo, bo = _b43_old_jlmsbc(model, θ, Y, X, Z, idt)
            @test jl == jo
            @test bc == bo
        end
    end

    # -----------------------------------------------------------------------------------------
    # 4. d lambda/da of _tre_invmills on a Dual against the continued fraction at 512
    #    bits (lambda' = -lambda*(a + lambda)); above -1e3 the generic method, bit for bit.
    # -----------------------------------------------------------------------------------------
    @testset "_tre_invmills on Duals" begin
        setprecision(BigFloat, 512) do
            for a in (-1e4, -1e6, -1e8)
                A = big(a); t = -A
                for k in 4000:-1:1
                    t = -A + k/t
                end
                @test abs(ForwardDiff.derivative(x -> _SF43._tre_invmills(x), a) - Float64(-t*(A + t)))/Float64(t*(A + t)) < 1e-14
            end
        end
        # the generic method (the pre-fix expression), called on the Dual explicitly
        old(x) = invoke(_SF43._tre_invmills, Tuple{Any}, x)
        for a in (-999.0, -40.0, -1.0, 2.0)
            @test ForwardDiff.derivative(x -> _SF43._tre_invmills(x), a) === ForwardDiff.derivative(old, a)
        end
    end

    # -----------------------------------------------------------------------------------------
    # 5. The values of LL_T(PTRET) and LL_T(PTREH) no longer depend on the number type they are
    #    evaluated with (a Float64 pass against the value part of a gradient and of a Hessian
    #    pass), at 1e4 and 4e5; before the fix they differed by up to 3.12e-6 (PTRET) and
    #    4.67e-6 (PTREH) at 4e5; after it, not at all.
    # -----------------------------------------------------------------------------------------
    @testset "PTRE$(model) value independent of the number type" for model in (:T, :H)
        for ratio in (1e4, 4e5)
            θ = _b43_θ(model, ratio)
            f = _B43LL(model, Y, X, Z, idt, n)
            v = f(θ)
            gr = _B43_DR.GradientResult(θ); ForwardDiff.gradient!(gr, f, θ)
            hr = _B43_DR.HessianResult(θ);  ForwardDiff.hessian!(hr, f, θ)
            @test abs(_B43_DR.value(gr) - v) < 1e-12
            @test abs(_B43_DR.value(hr) - v) < 1e-12
        end
    end
end
