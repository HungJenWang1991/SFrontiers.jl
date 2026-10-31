# Laplace noise under MSLE: the density evaluates |z| as sqrt(z^2 + 1e-16), as the MCI density does, so that both
# methods maximize the same objective. The log densities of the two backends are equal.

using Test
using SFrontiers

@testset "Laplace noise: the MSLE density uses the smoothed absolute value of MCI" begin
    MS = SFrontiers.MSLE_Backend
    MC = SFrontiers.MCI_Backend

    @testset "the Laplace kernels of MSLE and MCI" begin
        z = [0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-8, -1e-8, 1e-6, -1e-6, 1e-3, -1e-3, 0.3, -0.3, 1.0, -1.0,
             30.0, -30.0]
        cs = MS.make_constants(MS._build_model(:Laplace, :HalfNormal), Float64)
        cc = MC.make_constants(MC._build_model(:Laplace, :HalfNormal), Float64)
        for b in (0.01, 0.25, 1.0)
            nv = (inv_b = 1 / b, inv_2b = 1 / (2b))
            ls = similar(z)
            lc = similar(z)
            MS.log_noise_pdf!(MS.LaplaceNoise_MSLE(), ls, z, nv, cs)
            MC.log_noise_pdf!(MC.LaplaceNoise_MCI(), lc, z, nv, cc)
            @test isequal(ls, lc)
            fast = MS.myLaplacePDF!(similar(z); z = z, inv_b = nv.inv_b, inv_2b = nv.inv_2b)
            slow = MS.myLaplacePDF!(similar(z); z = z, b = b)
            plain = MS.myLaplacePDF(; z = z, b = b)
            for f in (fast, slow, plain)
                @test all(isapprox.(f, exp.(ls); rtol = 1e-13))
            end
        end
    end
end
