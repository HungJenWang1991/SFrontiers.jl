# ---------------------------------------------------------------------------
# GC-01 (09_gap_closure_findings.md) — bare-Symbol `hetero` on the DSL ctor
# paths must fail with the SAME clean message as the keyword ctor's R-13 guard,
# not a cryptic MethodError (`_validate_hetero(::HalfNormal_MCI, ::Symbol)` on
# the MCI ctors; `iterate(::Symbol)` on the MSLE 3-arg ctor's varnames
# ternaries). Guards added to: MCI 4-arg (with ZvarSpec), MCI 3-arg, MSLE
# 3-arg; the MSLE 4-arg guard's message was aligned to the same text.
# ---------------------------------------------------------------------------
@testset "GC-01 DSL bare-Symbol hetero errors cleanly (R-13 parity)" begin
    Random.seed!(2031)
    n = 80
    df = DataFrame(x1 = randn(n), x2 = randn(n), z1 = abs.(randn(n)) .+ 0.5)
    df.cons = ones(n)
    df.y = 1.0 .+ 0.5 .* df.x1 .- 0.3 .* df.x2 .+ 0.3 .* randn(n) .-
           abs.(0.4 .* randn(n))

    clean_msg(f) = begin   # the guard's message, not a MethodError
        err = try f(); nothing catch e; e end
        err isa ErrorException && occursin("Invalid `hetero`", err.msg) &&
            occursin(":scaling for the scaling property model", err.msg)
    end

    # Unified DSL path, 4-arg backend ctors (with @zvar) — previously
    # MethodError: _validate_hetero(::HalfNormal_MCI, ::Symbol)
    @test clean_msg(() -> sfmodel_spec(@useData(df), @depvar(y),
                                       @frontier(cons, x1, x2), @zvar(cons, z1);
                                       noise=:Normal, ineff=:HalfNormal,
                                       hetero=:sigma_sq))
    # Unified DSL path, 3-arg backend ctors (no @zvar)
    @test clean_msg(() -> sfmodel_spec(@useData(df), @depvar(y),
                                       @frontier(cons, x1, x2);
                                       noise=:Normal, ineff=:HalfNormal,
                                       hetero=:sigma_sq))
    # Direct MSLE backend DSL ctors (public API, unreachable via the unified
    # layer which dies in the MCI ctor first) — 3-arg previously threw
    # MethodError: iterate(::Symbol); 4-arg had a terser message.
    @test clean_msg(() -> SFrontiers.MSLE_Backend.sfmodel_spec(
        SFrontiers.MSLE_Backend.UseDataSpec(df),
        SFrontiers.MSLE_Backend.DepvarSpec(:y),
        SFrontiers.MSLE_Backend.FrontierSpec([:cons, :x1, :x2]);
        noise=:Normal, ineff=:HalfNormal, hetero=:sigma_sq))
    @test clean_msg(() -> SFrontiers.MSLE_Backend.sfmodel_spec(
        SFrontiers.MSLE_Backend.UseDataSpec(df),
        SFrontiers.MSLE_Backend.DepvarSpec(:y),
        SFrontiers.MSLE_Backend.FrontierSpec([:cons, :x1, :x2]),
        SFrontiers.MSLE_Backend.ZvarSpec([:cons, :z1]);
        noise=:Normal, ineff=:HalfNormal, hetero=:sigma_sq))
    # Direct MCI backend DSL ctors, both arities
    @test clean_msg(() -> SFrontiers.MCI_Backend.sfmodel_spec(
        SFrontiers.MCI_Backend.UseDataSpec(df),
        SFrontiers.MCI_Backend.DepvarSpec(:y),
        SFrontiers.MCI_Backend.FrontierSpec([:cons, :x1, :x2]);
        noise=:Normal, ineff=:HalfNormal, hetero=:sigma_sq))
    @test clean_msg(() -> SFrontiers.MCI_Backend.sfmodel_spec(
        SFrontiers.MCI_Backend.UseDataSpec(df),
        SFrontiers.MCI_Backend.DepvarSpec(:y),
        SFrontiers.MCI_Backend.FrontierSpec([:cons, :x1, :x2]),
        SFrontiers.MCI_Backend.ZvarSpec([:cons, :z1]);
        noise=:Normal, ineff=:HalfNormal, hetero=:sigma_sq))

    # Keyword path unchanged (same guard text, R-13)
    @test clean_msg(() -> sfmodel_spec(depvar=Vector(df.y),
                                       frontier=Matrix{Float64}(df[!, [:cons, :x1, :x2]]),
                                       zvar=Matrix{Float64}(df[!, [:cons, :z1]]),
                                       noise=:Normal, ineff=:HalfNormal,
                                       hetero=:sigma_sq))

    # :scaling DSL paths unaffected: builds with @zvar, errors (scaling-specific
    # message, not the GC-01 guard) without @zvar
    sp = sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2), @zvar(z1);
                      noise=:Normal, ineff=:TruncatedNormal, hetero=:scaling)
    @test sp isa SFrontiers.UnifiedSpec
    err = try
        sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2);
                     noise=:Normal, ineff=:TruncatedNormal, hetero=:scaling)
        nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("scaling", err.msg) &&
          !occursin("Invalid `hetero`", err.msg)
    # Vector form still builds through the DSL
    spv = sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2),
                       @zvar(cons, z1);
                       noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_sq])
    @test spv isa SFrontiers.UnifiedSpec
end
