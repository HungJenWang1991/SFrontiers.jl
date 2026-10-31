# A fit stores its draws once, and uses a user's draw array without a copy when its element type and
# device already match.

using Test
using SFrontiers
using Optim
using Random

isdefined(@__MODULE__, :_ref_panel_nll) || include(joinpath(@__DIR__, "draw_memory_reference.jl"))

@testset "draw storage without copies" begin
    PB = SFrontiers.Panel_Backend

    # Cross-sectional data, y = 1 + 0.5x - u + v, in the element type of the case.
    function _dst_data(seed, N; T = Float64)
        rng = Xoshiro(seed)
        x = randn(rng, N)
        X = [ones(N) x]
        y = X * [1.0, 0.5] .- abs.(0.5 .* randn(rng, N)) .+ 0.2 .* randn(rng, N)
        return T.(X), T.(y)
    end
    _dst_spec(X, y) = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod)
    # Loose options: the tests look at the stored draws and compare fits made from equal draws, so the
    # quality of the optimum does not matter.
    _dst_opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 60,),
                             main_solver = NelderMead(), main_opt = (iterations = 400,))
    _dst_fit(sp, meth) = sfmodel_fit(spec = sp, method = meth, optim_options = _dst_opt(),
                                     marginal = false, show_table = false, verbose = false)

    @testset "a host source for a host target is used as it is" begin
        y = rand(10)
        M = rand(10, 4)
        v = rand(4)
        @test SFrontiers.MCI_Backend._to_device_array(y, M) === M
        @test SFrontiers.MCI_Backend._to_device_array(y, v) === v
        @test SFrontiers.MSLE_Backend._to_device_array(y, M) === M
    end

    @testset "one draw block per specification and per result" begin
        X, y = _dst_data(4, 50)
        us = _dst_spec(X, y)
        meth = sfmodel_method(method = :MCI, n_draws = 64, chunks = 4)
        spec = SFrontiers.MCI_Backend._assemble_MCI_spec(us.mci_spec, meth)
        @test pointer(spec.draws) == pointer(spec.draws_2D)        # the flat draws are the block itself
        r = _dst_fit(us, meth)
        @test r.model.draws_2D === r.list[:model].draws_2D
    end

    @testset "a user's draw array is used without a copy" begin
        X, y = _dst_data(4, 50)
        sp = _dst_spec(X, y)
        Ud = rand(Xoshiro(1), 50, 64)          # one row of draws per observation
        uv = rand(Xoshiro(2), 64)              # one sequence shared by every observation
        @testset "$b cross-section" for b in (:MCI, :MSLE)
            r = _dst_fit(sp, sfmodel_method(method = b, draws = Ud, n_draws = 64))
            @test r.model.draws_2D === Ud
            r = _dst_fit(sp, sfmodel_method(method = b, draws = uv, n_draws = 64))
            @test pointer(r.model.draws_2D) == pointer(uv)      # kept as a 1 × D matrix over the same memory
        end
        @testset "MCI panel" begin
            d = _ref_panel_data(12, 4)         # 12 firms
            Upd = rand(Xoshiro(3), 12, 64)     # one row of draws per firm
            upv = rand(Xoshiro(4), 64)         # one sequence shared by every firm
            psp = _ref_unified_spec(d, :HalfNormal)
            r = _dst_fit(psp, sfmodel_method(method = :MCI, draws = Upd, n_draws = 64))
            @test r.model.draws === Upd
            r = _dst_fit(psp, sfmodel_method(method = :MCI, draws = upv, n_draws = 64))
            @test r.model.draws === upv
        end
    end

    @testset "a user's draw array is converted when its element type or kind differs" begin
        Ud = rand(Xoshiro(1), 50, 64)
        X32, y32 = _dst_data(4, 50; T = Float32)
        X, y = _dst_data(4, 50)
        @testset "$b cross-section" for b in (:MCI, :MSLE)
            # Float64 draws for Float32 data are narrowed to Float32
            r = _dst_fit(_dst_spec(X32, y32), sfmodel_method(method = b, draws = Ud, n_draws = 64))
            @test eltype(r.model.draws_2D) == Float32
            @test r.model.draws_2D !== Ud
            # a view is not an Array: the fit stores the draws as a Matrix
            r = _dst_fit(_dst_spec(X, y), sfmodel_method(method = b, draws = view(Ud, 1:50, :), n_draws = 64))
            @test r.model.draws_2D isa Matrix{Float64}
        end
    end

    @testset "the user's array and a copy of it give the same fit" begin
        X, y = _dst_data(4, 50)
        sp = _dst_spec(X, y)
        Ud = rand(Xoshiro(1), 50, 64)
        @testset "$b cross-section" for b in (:MCI, :MSLE)
            r1 = _dst_fit(sp, sfmodel_method(method = b, draws = Ud, n_draws = 64))
            r2 = _dst_fit(sp, sfmodel_method(method = b, draws = copy(Ud), n_draws = 64))
            @test r1.coeff == r2.coeff
            @test r1.loglikelihood == r2.loglikelihood
            @test r1.jlms == r2.jlms
        end
    end
end
