using Statistics
using Test

@testset "Normalizations" begin
    # matrix
    X = rand(5, 8)
    X_ = copy(X)

    t = fit(ZScoreNormalization, X, dims=1, center=false, scale=false)
    Y = normalize(t, X)
    @test isa(t, AbstractNormalization)
    @test isempty(t.mean)
    @test isempty(t.scale)
    @test isequal(X, Y)
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(ZScoreNormalization, X, dims=1, center=false)
    Y = normalize(t, X)
    @test isempty(t.mean)
    @test length(t.scale) == 8
    @test Y ≈ X ./ std(X, dims=1)
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(ZScoreNormalization, X, dims=1, scale=false)
    Y = normalize(t, X)
    @test length(t.mean) == 8
    @test isempty(t.scale)
    @test Y ≈ X .- mean(X, dims=1)
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(ZScoreNormalization, X, dims=1)
    Y = normalize(t, X)
    @test length(t.mean) == 8
    @test length(t.scale) == 8
    @test Y ≈ (X .- mean(X, dims=1)) ./ std(X, dims=1)
    @test unnormalize(t, Y) ≈ X
    @test Y ≈ normalize(ZScoreNormalization, X, dims=1)
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(ZScoreNormalization, X, dims=2)
    Y = normalize(t, X)
    @test length(t.mean) == 5
    @test length(t.scale) == 5
    @test Y ≈ (X .- mean(X, dims=2)) ./ std(X, dims=2)
    @test unnormalize(t, Y) ≈ X
    @test Y ≈ normalize(ZScoreNormalization, X, dims=2)
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(MinMaxNormalization, X, dims=1, zero=false)
    Y = normalize(t, X)
    @test length(t.min) == 8
    @test length(t.scale) == 8
    @test Y ≈ X ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(MinMaxNormalization, X, dims=1)
    Y = normalize(t, X)
    @test isa(t, AbstractNormalization)
    @test length(t.min) == 8
    @test length(t.scale) == 8
    @test Y ≈ (X .- minimum(X, dims=1)) ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    @test unnormalize(t, Y) ≈ X
    @test Y ≈ normalize(MinMaxNormalization, X, dims=1)
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(MinMaxNormalization, X, dims=2)
    Y = normalize(t, X)
    @test isa(t, AbstractNormalization)
    @test length(t.min) == 5
    @test length(t.scale) == 5
    @test Y ≈ (X .- minimum(X, dims=2)) ./ (maximum(X, dims=2) .- minimum(X, dims=2))
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    # vector
    X = rand(10)
    X_ = copy(X)

    t = fit(ZScoreNormalization, X, dims=1, center=false, scale=false)
    Y = normalize(t, X)
    @test normalize(t, X) ≈ Y
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(ZScoreNormalization, X, dims=1, center=false)
    Y = normalize(t, X)
    @test Y ≈ X ./ std(X, dims=1)
    @test normalize(t, X) ≈ Y
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(ZScoreNormalization, X, dims=1, scale=false)
    Y = normalize(t, X)
    @test Y ≈ X .- mean(X, dims=1)
    @test normalize(t, X) ≈ Y
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(ZScoreNormalization, X, dims=1)
    Y = normalize(t, X)
    @test Y ≈ (X .- mean(X, dims=1)) ./ std(X, dims=1)
    @test normalize(t, X) ≈ Y
    @test unnormalize(t, Y) ≈ X
    @test Y ≈ normalize(ZScoreNormalization, X, dims=1)
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(MinMaxNormalization, X, dims=1)
    Y = normalize(t, X)
    @test Y ≈ (X .- minimum(X, dims=1)) ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    @test normalize(t, X) ≈ Y
    @test unnormalize(t, Y) ≈ X
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

    X = copy(X_)
    t = fit(MinMaxNormalization, X, dims=1, zero=false)
    Y = normalize(t, X)
    @test Y ≈ X ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    @test normalize(t, X) ≈ Y
    @test unnormalize(t, Y) ≈ X
    @test Y ≈ normalize(MinMaxNormalization, X, dims=1, zero=false)
    @test normalize!(t, X) === X
    @test isequal(X, Y)
    @test unnormalize!(t, Y) === Y
    @test Y ≈ X_

end
