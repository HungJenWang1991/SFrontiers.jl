# Regression tests: `redflag` also
# requires the Hessian of the negative log-likelihood at the estimate to be positive definite (the
# second-order condition for a strict local maximum of the log-likelihood).
#
# Before the fix SFRedflag.sf_classify_hessian, the Hessian part of redflag shared by the MLE, MCI,
# MSLE and panel backends, tested only that H can be inverted, that its 1-norm reciprocal condition
# number is at least eps, and that diag(inv H) > 0. An indefinite H whose inverse has a positive
# diagonal (a saddle point, or a ridge along which the likelihood does not fall), such as
# [-1 2; 2 -1], read 0. Now the Cholesky factorization of (H + H')/2 must succeed as well; if it does
# not, hessian_flag (and redflag) is 1, the var-cov matrix is kept as computed, and one new message is
# printed. Everything else is as before; the testset "unchanged elsewhere" compares the new function
# with an in-test copy of the old one.

using Test
using SFrontiers
using LinearAlgebra
using Random

const _RF46 = SFrontiers.SFRedflag

# an in-test copy of the earlier classifier (src/sf_redflag.jl), messages left out
function _bf46_old_classify(H::AbstractMatrix{T}; at_clamp::Bool = false) where {T<:Real}
    F = float(T)
    V = try
        inv(H)
    catch
        nothing
    end
    V === nothing && return (var_cov_matrix = fill!(similar(H, F), F(NaN)), redflag = 1, hessian_flag = 1)
    rcond = inv(opnorm(H, 1) * opnorm(V, 1))
    !(rcond >= eps(F)) && return (var_cov_matrix = fill!(similar(H, F), F(NaN)), redflag = 1, hessian_flag = 1)
    hflag = all(diag(V) .> 0) ? 0 : 2
    return (var_cov_matrix = V, redflag = at_clamp ? 1 : hflag, hessian_flag = hflag)
end

function _bf46_capture(f)
    path, io = mktemp()
    r = redirect_stdout(f, io)
    close(io)
    txt = read(path, String); rm(path; force = true)
    return r, txt
end

@testset "redflag requires a positive-definite Hessian" begin
    cl(H; kw...) = _RF46.sf_classify_hessian(H; message = false, kw...)
    S = [-1.0 2.0; 2.0 -1.0]          # eigenvalues 1 and -3; inv(S) = [1 2; 2 1]/3, a positive diagonal
    S3 = [-1.0 2.0 0.0; 2.0 -1.0 0.0; 0.0 0.0 4.0]

    @testset "saddle points with a positive diag(inv H): 1, the matrix kept" begin
        for H in (S, S3)
            r = cl(H)
            @test r.redflag == 1
            @test r.hessian_flag == 1
            @test isequal(r.var_cov_matrix, inv(H))       # kept as computed: no NaN standard errors
        end
        # a 4 x 4 saddle with no zero entry: eigenvalues 5, 3, 2 and -40, every diagonal element of
        # its inverse (1/5 + 1/3 + 1/2 - 1/40)/4 > 0
        Q = [1.0 1.0 1.0 1.0; 1.0 -1.0 1.0 -1.0; 1.0 1.0 -1.0 -1.0; 1.0 -1.0 -1.0 1.0] ./ 2
        H4 = Matrix(Symmetric(Q * Diagonal([5.0, 3.0, 2.0, -40.0]) * Q'))
        r = cl(H4)
        @test r.redflag == 1 && isequal(r.var_cov_matrix, inv(H4))
        # the test is strict: two nearly flat directions of opposite curvature (eigenvalues 1, 0.5,
        # 5e-15 and -1e-14) pass the old tests (reciprocal condition number about 5e-15), and would
        # pass a tolerant test "lambda_min < -tau * lambda_max" with tau = 1e-12 or sqrt(eps)
        H5 = Matrix(Symmetric(Q * Diagonal([1.0, 0.5, 0.5e-14, -1e-14]) * Q'))
        @test cl(H5).redflag == 1
        # a ForwardDiff Hessian is symmetric only to rounding: the symmetric part decides
        Sa = copy(S); Sa[1, 2] += 4eps()
        @test cl(Sa).redflag == 1
        Pa = [2.0 0.5; 0.5 1.0]; Pa[2, 1] += 4eps()
        @test cl(Pa).redflag == 0
    end

    @testset "unchanged elsewhere" begin
        P = [2.0 0.5; 0.5 1.0]
        @test cl(-P).redflag == 2                            # negative definite: 2 (wins over 1)
        @test all(isnan, cl([1.0 0.0; 0.0 1e-17]).var_cov_matrix)   # rcond below eps: all NaN
        # the boundary override: 1 whatever the Hessian; hessian_flag says what the Hessian gave
        r = cl(S; at_clamp = true)
        @test r.redflag == 1 && r.hessian_flag == 1 && isequal(r.var_cov_matrix, inv(S))
        # random Hessians: the same matrix, bit for bit, and the same flag as the old classifier,
        # except that 0 becomes 1 exactly when the Cholesky factorization of (H + H')/2 fails
        rng = Xoshiro(4646)
        nnew = 0; nsame = 0
        for _ in 1:400
            k = rand(rng, 2:6)
            A = randn(rng, k, k)
            H = rand(rng) < 0.5 ? A * A' + 0.1I : (A + A') / 2
            H[1, k] += 1e-15 * randn(rng)                    # not exactly symmetric, as ForwardDiff's
            for ac in (false, true)
                o = _bf46_old_classify(H; at_clamp = ac); n = cl(H; at_clamp = ac)
                pd = isposdef(Symmetric((H + H') / 2))
                same = isequal(o.var_cov_matrix, n.var_cov_matrix) &&
                       (o.hessian_flag == 0 && !pd ? (n.hessian_flag == 1 && n.redflag == 1) :
                                                     (n.hessian_flag == o.hessian_flag && n.redflag == o.redflag))
                same && (nsame += 1)
                (o.hessian_flag == 0 && !pd && !ac) && (nnew += 1)
            end
        end
        @test nsame == 800
        @test nnew > 0                                       # the draw contains saddles that the old rule passed
    end

    @testset "Float32 Hessians" begin
        r = cl(Float32.(S))
        @test r.redflag == 1 && r.hessian_flag == 1
        @test eltype(r.var_cov_matrix) == Float32 && isequal(r.var_cov_matrix, inv(Float32.(S)))
        r = cl(Float32[2 0.5; 0.5 1])
        @test r.redflag == 0 && eltype(r.var_cov_matrix) == Float32
        @test cl(Float32[1 0; 0 -1]).redflag == 2
        r = cl(Float32[1 1; 1 1])
        @test r.redflag == 1 && eltype(r.var_cov_matrix) == Float32 && all(isnan, r.var_cov_matrix)
    end

    @testset "the backends' entry points apply the rule" begin
        @test SFrontiers.MCI_Backend._invert_hessian(S; message = false).redflag == 1
        @test SFrontiers.MSLE_Backend._invert_hessian(S; message = false).redflag == 1
        @test SFrontiers.Panel_Backend._panel_invert_hessian(S; message = false).redflag == 1
        @test SFrontiers.MCI_Backend.var_cov_mat(S; message = false).redflag == 1
        @test SFrontiers.MSLE_Backend.var_cov_mat(S; message = false).redflag == 1
        @test SFrontiers.MCI_Backend._invert_hessian([2.0 0.5; 0.5 1.0]; message = false).redflag == 0
    end

    @testset "messages" begin
        msg(H; kw...) = _bf46_capture(() -> _RF46.sf_classify_hessian(H; message = true, kw...))[2]
        t = msg(S)
        @test occursin("The Hessian matrix is not positive definite", t)
        @test occursin("not a strict local maximum of the log-likelihood", t)
        @test count("Hessian", t) == 1                       # one message about the Hessian
        @test !occursin("non-positive", t) && !occursin("not invertible", t)
        t = msg([1.0 0.0; 0.0 -1.0])                         # 2: the diagonal message only
        @test occursin("non-positive", t) && !occursin("not positive definite", t)
        t = msg([1.0 1.0; 1.0 1.0])                          # not invertible: that message only
        @test occursin("not invertible", t) && !occursin("not positive definite", t)
        @test isempty(msg([2.0 0.5; 0.5 1.0]))               # positive definite: nothing printed
        @test isempty(_bf46_capture(() -> _RF46.sf_classify_hessian(S; message = false))[2])
        t = msg(S; at_clamp = true)                          # the boundary message, then this one
        @test occursin("boundary solution", t) && occursin("not positive definite", t)
    end
end
