using Statistics
using LinearAlgebra, Random, Test

struct EmptyCovarianceEstimator <: CovarianceEstimator end

@testset "Covariance" begin
weight_funcs = (weights, aweights, fweights, pweights)

@testset "$f" for f in weight_funcs
    X = randn(3, 8)

    Z1 = X .- mean(X, dims = 1)
    Z2 = X .- mean(X, dims = 2)

    w1 = rand(3)
    w2 = rand(8)

    # varcorrection is negative if sum of weights is smaller than 1
    if f === fweights
        w1[1] += 1
        w2[1] += 1
    end

    wv1 = f(w1)
    wv2 = f(w2)

    Z1w = X .- mean(X, weights=wv1, dims=1)
    Z2w = X .- mean(X, weights=wv2, dims=2)

    ## reference results

    S1 = Z1'Z1
    S2 = Z2 * Z2'

    Sz1 = X'X
    Sz2 = X * X'

    S1w = Z1w' * Matrix(Diagonal(w1)) * Z1w
    S2w = Z2w * Matrix(Diagonal(w2)) * Z2w'

    Sz1w = X' * Matrix(Diagonal(w1)) * X
    Sz2w = X * Matrix(Diagonal(w2)) * X'

    @testset "Scattermat" begin
        @test scattermat(X)         ≈ S1
        @test scattermat(X, dims=2) ≈ S2

        @test scattermat(X, mean=0)         ≈ Sz1
        @test scattermat(X, mean=0, dims=2) ≈ Sz2

        @test scattermat(X, mean=mean(X, dims=1))         ≈ S1
        @test scattermat(X, mean=mean(X, dims=2), dims=2) ≈ S2

        @test scattermat(X, mean=zeros(1,8))       ≈ Sz1
        @test scattermat(X, mean=zeros(3), dims=2) ≈ Sz2

        @testset "Weighted" begin
            @test scattermat(X, weights=wv1)         ≈ S1w
            @test scattermat(X, weights=wv2, dims=2) ≈ S2w

            @test scattermat(X, weights=wv1, mean=0)         ≈ Sz1w
            @test scattermat(X, weights=wv2, mean=0, dims=2) ≈ Sz2w

            @test scattermat(X, weights=wv1, mean=mean(X, weights=wv1, dims=1))         ≈ S1w
            @test scattermat(X, weights=wv2, mean=mean(X, weights=wv2, dims=2), dims=2) ≈ S2w

            @test scattermat(X, weights=wv1, mean=zeros(1,8))       ≈ Sz1w
            @test scattermat(X, weights=wv2, mean=zeros(3), dims=2) ≈ Sz2w
        end
    end

    @testset "Uncorrected" begin
        @testset "Weighted Covariance" begin
            @test cov(X, weights=wv1; corrected=false)    ≈ S1w ./ sum(wv1)
            @test cov(X, weights=wv2, dims=2; corrected=false) ≈ S2w ./ sum(wv2)
        end
        @testset "Conversions" begin
            std1 = std(X, weights=wv1, dims=1; corrected=false)
            std2 = std(X, weights=wv2, dims=2; corrected=false)

            cov1 = cov(X, weights=wv1, dims=1; corrected=false)
            cov2 = cov(X, weights=wv2, dims=2; corrected=false)

            cor1 = cor(X, weights=wv1, dims=1)
            cor2 = cor(X, weights=wv2, dims=2)

            @testset "cov2cor" begin
                @test cov2cor(cov(X, dims = 1), std(X, dims = 1)) ≈ cor(X, dims = 1)
                @test cov2cor(cov(X, dims = 2), std(X, dims = 2)) ≈ cor(X, dims = 2)
                @test cov2cor(cov1, std1) ≈ cor1
                @test cov2cor(cov2, std2) ≈ cor2
            end
            @testset "cor2cov" begin
                @test cor2cov(cor(X, dims = 1), std(X, dims = 1)) ≈ cov(X, dims = 1)
                @test cor2cov(cor(X, dims = 2), std(X, dims = 2)) ≈ cov(X, dims = 2)
                @test cor2cov(cor1, std1) ≈ cov1
                @test cor2cov(cor2, std2) ≈ cov2
            end
        end
    end

    @testset "Corrected" begin
        @testset "Weighted Covariance" begin
            if isa(wv1, Weights)
                @test_throws ArgumentError cov(X, weights=wv1, corrected=true)
            else
                var_corr1 = Statistics.varcorrection(wv1, true)
                var_corr2 = Statistics.varcorrection(wv2, true)

                @test cov(X, weights=wv1, corrected=true)    ≈ S1w .* var_corr1
                @test cov(X, weights=wv2, dims=2, corrected=true) ≈ S2w .* var_corr2
            end
        end
        @testset "Conversions" begin
            if !isa(wv1, Weights)
                std1 = std(X, weights=wv1, dims=1; corrected=true)
                std2 = std(X, weights=wv2, dims=2; corrected=true)

                cov1 = cov(X, weights=wv1, dims=1; corrected=true)
                cov2 = cov(X, weights=wv2, dims=2; corrected=true)

                cor1 = cor(X, weights=wv1, dims=1)
                cor2 = cor(X, weights=wv2, dims=2)

                @testset "cov2cor" begin
                    @test cov2cor(cov(X, dims = 1), std(X, dims = 1)) ≈ cor(X, dims = 1)
                    @test cov2cor(cov(X, dims = 2), std(X, dims = 2)) ≈ cor(X, dims = 2)
                    @test cov2cor(cov1, std1) ≈ cor1
                    @test cov2cor(cov2, std2) ≈ cor2
                end

                @testset "cov2cor!" begin
                    tmp_cov1 = copy(cov1)
                    @test !(tmp_cov1 ≈ cor1)
                    Statistics.cov2cor!(tmp_cov1, std1)
                    @test tmp_cov1 ≈ cor1

                    tmp_cov2 = copy(cov2)
                    @test !(tmp_cov2 ≈ cor2)
                    Statistics.cov2cor!(tmp_cov2, std2)
                    @test tmp_cov2 ≈ cor2
                end

                @testset "cor2cov" begin
                    @test cor2cov(cor(X, dims = 1), std(X, dims = 1)) ≈ cov(X, dims = 1)
                    @test cor2cov(cor(X, dims = 2), std(X, dims = 2)) ≈ cov(X, dims = 2)
                    @test cor2cov(cor1, std1) ≈ cov1
                    @test cor2cov(cor2, std2) ≈ cov2
                end

                @testset "cor2cov!" begin
                    tmp_cor1 = copy(cor1)
                    @test !(tmp_cor1 ≈ cov1)
                    Statistics.cor2cov!(tmp_cor1, std1)
                    @test tmp_cor1 ≈ cov1

                    tmp_cor2 = copy(cor2)
                    @test !(tmp_cor2 ≈ cov2)
                    Statistics.cor2cov!(tmp_cor2, std2)
                    @test tmp_cor2 ≈ cov2
                end
            end
        end
    end

    @testset "Correlation" begin
        @test cor(X, weights=f(ones(3)), dims=1) ≈ cor(X, dims = 1)
        @test cor(X, weights=f(ones(8)), dims=2) ≈ cor(X, dims = 2)

        cov1 = cov(X, weights=wv1, dims=1, corrected=false)
        std1 = std(X, weights=wv1, dims=1, corrected=false)
        cov2 = cov(X, weights=wv2, dims=2, corrected=false)
        std2 = std(X, weights=wv2, dims=2, corrected=false)
        expected_cor1 = Statistics.cov2cor!(cov1, std1)
        expected_cor2 = Statistics.cov2cor!(cov2, std2)

        @test cor(X, weights=wv1, dims=1) ≈ expected_cor1
        @test cor(X, weights=wv2, dims=2) ≈ expected_cor2
    end

    @testset "Abstract covariance estimation" begin
        Xm1 = mean(X, dims=1)
        Xm2 = mean(X, dims=2)

        for corrected ∈ (false, true)
            scc = SimpleCovariance(corrected=corrected)
            @test_throws ArgumentError cov(scc, X, dims=0)
            @test_throws ArgumentError cov(scc, X, weights=wv1, dims=0)
            @test cov(scc, X) ≈ cov(X, corrected=corrected)
            @test cov(scc, X, mean=Xm1) ≈ Statistics.covm(X, Xm1, nothing, corrected=corrected)
            @test cov(scc, X, mean=Xm2, dims=2) ≈ Statistics.covm(X, Xm2, nothing, 2, corrected=corrected)
            if f !== weights || corrected === false
                @test cov(scc, X, weights=wv1, dims=1) ≈
                    cov(X, weights=wv1, dims=1, corrected=corrected)
                @test cov(scc, X, weights=wv2, dims=2) ≈
                    cov(X, weights=wv2, dims=2, corrected=corrected)
                @test cov(scc, X, weights=wv1, mean=Xm1) ≈
                    Statistics.covm(X, Xm1, wv1, corrected=corrected)
                @test cov(scc, X, weights=wv2, mean=Xm2, dims=2) ≈
                    Statistics.covm(X, Xm2, wv2, 2, corrected=corrected)
            end
        end
    end
end

@testset "Abstract covariance estimation" begin
    est = EmptyCovarianceEstimator()
    wv = fweights(rand(2))
    @test_throws ErrorException cov(est, [1.0 2.0; 3.0 4.0])
    @test_throws ErrorException cov(est, [1.0 2.0; 3.0 4.0], weights=wv)
    @test_throws ErrorException cov(est, [1.0 2.0; 3.0 4.0], dims = 2)
    @test_throws ErrorException cov(est, [1.0 2.0; 3.0 4.0], weights=wv, dims = 2)
    @test_throws ErrorException cov(est, [1.0 2.0; 3.0 4.0], mean = nothing)
    @test_throws ErrorException cov(est, [1.0 2.0; 3.0 4.0], weights=wv, mean = nothing)
    @test_throws ErrorException cov(est, [1.0 2.0; 3.0 4.0], dims = 2, mean = nothing)
    @test_throws ErrorException cov(est, [1.0 2.0; 3.0 4.0], weights=wv, dims = 2, mean = nothing)
    @test_throws ErrorException cov(est, [1.0, 2.0], [3.0, 4.0])
    @test_throws ErrorException cov(est, [1.0, 2.0])

    x = rand(8)
    y = rand(8)

    for corrected ∈ (false, true)
        @test_throws MethodError SimpleCovariance(corrected)
        scc = SimpleCovariance(corrected=corrected)
        @test cov(scc, x) ≈ cov(x; corrected=corrected)
        @test cov(scc, x, y) ≈ cov(x, y; corrected=corrected)
    end
end
end # @testset "Covariance"
