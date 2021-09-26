using Random
using Statistics: wsum, wsum!

@testset "weighted sum" begin
    wts = ([1.4, 2.5, 10.1], [1.4f0, 2.5f0, 10.1f0], [0.0, 2.3, 5.6],
           [NaN, 2.3, 5.6], [Inf, 2.3, 5.6],
           [2, 1, 3], Int8[1, 2, 3], [1, 1, 1])
    for a in (rand(3), rand(Int, 3), rand(Int8, 3))
        for w in wts
            res = @inferred wsum(a, weights=w)
            expected = sum(a.*w)
            if isfinite(res)
                @test res ≈ expected
            else
                @test isequal(res, expected)
            end
            @test typeof(res) == typeof(expected)
        end
    end
    for a in (rand(3, 5), rand(Float32, 3, 5), rand(Int, 3, 5), rand(Int8, 3, 5))
        for w in wts
            wr = repeat(w, outer=(1, 5))
            res = @inferred wsum(a, weights=wr)
            expected = sum(a.*wr)
            if isfinite(res)
                @test res ≈ expected
            else
                @test isequal(res, expected)
            end
            @test typeof(res) == typeof(expected)
        end
    end
end

@testset "weighted sum over dimensions" begin
    wts = ([1.4, 2.5, 10.1], [1.4f0, 2.5f0, 10.1f0], [0.0, 2.3, 5.6],
           [NaN, 2.3, 5.6], [Inf, 2.3, 5.6],
           [2, 1, 3], Int8[1, 2, 3], [1, 1, 1])

    ainf = rand(3)
    ainf[1] = Inf
    anan = rand(3)
    anan[1] = NaN
    for a in (rand(3), rand(Float32, 3), ainf, anan,
              rand(Int, 3), rand(Int8, 3),
              view(rand(5), 2:4))
        for w in wts
            if all(isfinite, a) && all(isfinite, w)
                expected = sum(a.*w, dims=1)
                res = @inferred wsum(a, weights=w, dims=1)
                @test res ≈ expected
                @test typeof(res) == typeof(expected)
                x = rand!(similar(expected))
                y = copy(x)
                @inferred wsum!(y, a, weights=w)
                @test y ≈ expected
                y = copy(x)
                @inferred wsum!(y, a, weights=w, init=false)
                @test y ≈ x + expected
            else
                expected = sum(a.*w, dims=1)
                res = @inferred wsum(a, weights=w, dims=1)
                @test isfinite.(res) == isfinite.(expected)
                @test typeof(res) == typeof(expected)
                x = rand!(similar(expected))
                y = copy(x)
                @inferred wsum!(y, a, weights=w)
                @test isfinite.(y) == isfinite.(expected)
                y = copy(x)
                @inferred wsum!(y, a, weights=w, init=false)
                @test isfinite.(y) == isfinite.(expected)
            end
        end
    end

    ainf = rand(3, 3, 3)
    ainf[1] = Inf
    anan = rand(3, 3, 3)
    anan[1] = NaN
    for a in (rand(3, 3, 3), rand(Float32, 3, 3, 3), ainf, anan,
              rand(Int, 3, 3, 3), rand(Int8, 3, 3, 3),
              view(rand(3, 3, 5), :, :, 2:4))
        for w in wts
            for (d, rw) in ((1, reshape(w, :, 1, 1)),
                            (2, reshape(w, 1, :, 1)),
                            (3, reshape(w, 1, 1, :)))
                if all(isfinite, a) && all(isfinite, w)
                    expected = sum(a.*rw, dims=d)
                    res = @inferred wsum(a, weights=w, dims=d)
                    @test res ≈ expected
                    @test typeof(res) == typeof(expected)
                    x = rand!(similar(expected))
                    y = copy(x)
                    @inferred wsum!(y, a, weights=w)
                    @test y ≈ expected
                    y = copy(x)
                    @inferred wsum!(y, a, weights=w, init=false)
                    @test y ≈ x + expected
                else
                    expected = sum(a.*rw, dims=d)
                    res = @inferred wsum(a, weights=w, dims=d)
                    @test isfinite.(res) == isfinite.(expected)
                    @test typeof(res) == typeof(expected)
                    x = rand!(similar(expected))
                    y = copy(x)
                    @inferred wsum!(y, a, weights=w)
                    @test isfinite.(y) == isfinite.(expected)
                    y = copy(x)
                    @inferred wsum!(y, a, weights=w, init=false)
                    @test isfinite.(y) == isfinite.(expected)
                end
            end

            @test_throws DimensionMismatch wsum(a, weights=w, dims=4)
        end
    end

    # Corner case with a single row
    @test wsum([1 2], weights=[2], dims=1) == [2 4]
end
