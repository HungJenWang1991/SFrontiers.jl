# predict reads the draws stored on the result, in groups of the fit's `chunks`, and gives new panel
# firms the estimation's draws.

using Test
using SFrontiers
using Optim
using Random

isdefined(@__MODULE__, :_ref_panel_nll) || include(joinpath(@__DIR__, "draw_memory_reference.jl"))

@testset "predict reuses the draws stored on the result" begin
    PB = SFrontiers.Panel_Backend

    # Cross-sectional data, y = 1 + 0.5x - u + v, in the element type of the case.
    function _pdr_data(seed, N; T = Float64)
        rng = Xoshiro(seed)
        x = randn(rng, N)
        X = [ones(N) x]
        y = X * [1.0, 0.5] .- abs.(0.5 .* randn(rng, N)) .+ 0.2 .* randn(rng, N)
        return T.(X), T.(y)
    end
    _pdr_spec(X, y) = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod)
    # Loose options: predict is compared with the fit at the same coefficients, so the quality of the
    # optimum does not matter.
    _pdr_opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 60,),
                             main_solver = NelderMead(), main_opt = (iterations = 400,))
    _pdr_fit(sp, meth) = sfmodel_fit(spec = sp, method = meth, optim_options = _pdr_opt(),
                                     marginal = false, show_table = false, verbose = false)
    # predict on a panel frame, with the single zvar column as a matrix
    _pdr_ppredict(r, X, z, y, id; kw...) =
        predict(r; frontier = X, zvar = reshape(z, :, 1), depvar = y, id = id, kw...)
    # Halve row 1 of the stored block `blk`: the entries `hit` that use the row move by far more than
    # rounding, all other entries stay as fitted, and restoring the row restores every entry. A
    # permutation of the row would not do: the simulated integrals are sums over the draws, so a
    # permuted row changes them only by rounding, and a panel firm's not at all.
    function _pdr_halve_row1(r, blk, pred, hit)
        rest = setdiff(eachindex(r.jlms), hit)
        saved = blk[1, :]
        blk[1, :] .= saved ./ 2
        p = pred()
        @test all(abs.(p[hit] .- r.jlms[hit]) .> 1e-6 .* abs.(r.jlms[hit]))
        @test p[rest] == r.jlms[rest]
        blk[1, :] .= saved
        @test pred() == r.jlms
    end

    @testset "predict reads the draw block stored on the result" begin
        X, y = _pdr_data(1, 200)
        sp = _pdr_spec(X, y)
        @testset "$b cross-section" for b in (:MCI, :MSLE)
            r = _pdr_fit(sp, sfmodel_method(method = b, n_draws = 128, chunks = 10))
            _pdr_halve_row1(r, r.model.draws_2D, () -> predict(r; frontier = X, depvar = y, what = :jlms), [1])
        end
        @testset "MCI panel" begin
            d = _ref_panel_data(12, 4)
            r = _pdr_fit(_ref_unified_spec(d, :HalfNormal),
                         sfmodel_method(method = :MCI, n_draws = 64, chunks = 10))
            _pdr_halve_row1(r, r.model.draws, () -> _pdr_ppredict(r, d.X, d.z, d.y, d.id; what = :jlms),
                            findall(==(1), d.id))          # every observation of firm 1
        end
    end

    @testset "predict neither regenerates nor copies the draw block" begin
        X, y = _pdr_data(2, 400)
        r = _pdr_fit(_pdr_spec(X, y), sfmodel_method(method = :MCI, n_draws = 1024, chunks = 10))
        # The block holds N*S*8 bytes. predict reads it where it is and allocates only temporaries for
        # one group of rows at a time; regenerating the draws would add about three blocks.
        @test _alloc(() -> predict(r; frontier = X, depvar = y, what = :jlms)) < 6 * 400 * 1024 * 8
    end

    @testset "a Float32 draw block is read in groups of Float64 rows" begin
        X, y = _pdr_data(3, 200; T = Float32)
        sp = _pdr_spec(X, y)
        # 230 rows: the 200 estimation rows, then rows 1-30 again. predict uses groups of 23 rows, and
        # the ninth group (predict rows 185-207) reads the draws of estimation rows 185-200 and 1-7.
        rows = vcat(1:200, 1:30)
        @testset "$b cross-section" for b in (:MCI, :MSLE)
            r = _pdr_fit(sp, sfmodel_method(method = b, n_draws = 128, chunks = 10))
            W = Float64.(Array(r.model.draws_2D))
            for (Xf, yf) in ((X, y), (X[rows, :], y[rows])), what in (:jlms, :bc)
                p = predict(r; frontier = Xf, depvar = yf, what = what)
                @test p == predict(r; frontier = Xf, depvar = yf, draws = W, what = what)
                @test all(isfinite, p)
            end
        end
        @testset "MCI panel, with a new firm" begin
            d = _ref_panel_data(12, 4)
            r = _pdr_fit(_ref_unified_spec(d, :HalfNormal; T = Float32),
                         sfmodel_method(method = :MCI, n_draws = 64, chunks = 10))
            W = Float64.(Array(r.model.draws))
            Xp, zp, yp = Float32.(d.X), Float32.(d.z), Float32.(d.y)
            # the estimation frame, and the same frame followed by firm 1's four rows under the new id 13
            f1 = findall(==(1), d.id)
            frames = ((Xp, zp, yp, d.id),
                      (vcat(Xp, Xp[f1, :]), vcat(zp, zp[f1]), vcat(yp, yp[f1]), vcat(d.id, fill(13, length(f1)))))
            for (Xf, zf, yf, idf) in frames, what in (:jlms, :bc)
                p = _pdr_ppredict(r, Xf, zf, yf, idf; what = what)
                @test p == _pdr_ppredict(r, Xf, zf, yf, idf; draws = W, what = what)
                @test all(isfinite, p)
            end
        end
    end

    @testset "a fit given its own draws predicts from them when draws are omitted" begin
        d = _ref_panel_data(12, 4)
        ud = rand(Xoshiro(3), 12, 64)
        r = _pdr_fit(_ref_unified_spec(d, :HalfNormal),
                     sfmodel_method(method = :MCI, draws = ud, n_draws = 64, chunks = 10))
        for what in (:jlms, :bc)
            @test _pdr_ppredict(r, d.X, d.z, d.y, d.id; what = what) ==
                  _pdr_ppredict(r, d.X, d.z, d.y, d.id; draws = ud, what = what)
        end
    end

    @testset "row maps and draw sources" begin
        # recycled rows are a view of the block, not a copy
        dtr = Float64[10i + j for i in 1:5, j in 1:3]
        @test SFrontiers._align_draws(dtr, 7) isa SubArray
        @test parent(SFrontiers._align_draws(dtr, 7)) === dtr
        # a Float32 block: a row source when the kernels read groups (chunks >= 2), else one Float64 matrix
        b32 = rand(Float32, 5, 8)
        @test SFrontiers._predict_draw_source(b32, 5, 2) isa SFrontiers.SFDraws.DrawRows
        @test SFrontiers._predict_draw_source(b32, 5, 1) == Float64.(b32)
        # predict rows 4-7 of a 7-row frame over a 5-row block use rows 4, 5, 1 and 2
        @test SFrontiers.SFDraws.host_rows(SFrontiers._predict_draw_source(b32, 7, 2), 4, 7) ==
              Float64.(b32[[4, 5, 1, 2], :])
        # a host Float64 block for the estimation frame is the block itself
        b64 = rand(5, 8)
        @test SFrontiers._predict_draw_source(b64, 5, 2) === b64
        # shared draws (one row) are widened once
        sh = SFrontiers._predict_draw_source(rand(Float32, 1, 8), 5, 2)
        @test sh isa Matrix{Float64} && size(sh) == (1, 8)
        # a result without `model` has no draws to read
        @test_throws "does not carry the draws" SFrontiers._fit_draw_block(SFrontiers.SFResult((coeff = [1.0],), :MCI))
    end

    @testset "a frame with no rows" begin
        # rows a:b with a > b of a row source: no rows, the block's number of columns
        src = SFrontiers.SFDraws.DrawRows(rand(Float32, 5, 8), collect(1:5))
        @test SFrontiers.SFDraws.host_rows(src, 1, 0) isa Matrix{Float64}
        @test size(SFrontiers.SFDraws.host_rows(src, 1, 0)) == (0, 8)
        @test size(SFrontiers.SFDraws.host_rows(src, 4, 3)) == (0, 8)
        # with chunks = 1 a Float32 block is read as one Float64 matrix, here without rows
        @test size(SFrontiers._predict_draw_source(rand(Float32, 5, 8), 0, 1)) == (0, 8)
        X, y = _pdr_data(6, 200; T = Float32)
        sp = _pdr_spec(X, y)
        @testset "$b cross-section, chunks = 1" for b in (:MCI, :MSLE)
            r = _pdr_fit(sp, sfmodel_method(method = b, n_draws = 128, chunks = 1))
            for what in (:jlms, :bc)
                @test isempty(predict(r; frontier = X[1:0, :], depvar = y[1:0], what = what))
            end
        end
    end
end
