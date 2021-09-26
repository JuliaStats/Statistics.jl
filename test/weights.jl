using Statistics
using LinearAlgebra, Random, SparseArrays, Test, Dates
using Statistics: wsum, wsum!

@testset "Weights" begin
weight_funcs = (weights, aweights, fweights, pweights)

## Construction

@testset "$f" for f in weight_funcs
    @test isa(f([1, 2, 3]), AbstractWeights{Int})
    @test isa(f([1., 2., 3.]), AbstractWeights{Float64})
    @test isa(f([1 2 3; 4 5 6]), AbstractWeights{Int})

    @test isempty(f(Float64[]))
    @test size(f([1, 2, 3])) == (3,)

    w  = [1., 2., 3.]
    wv = f(w)
    @test eltype(wv) === Float64
    @test length(wv) === 3
    @test wv ==  w
    @test sum(wv) === 6.0
    @test !isempty(wv)

    b  = trues(3)
    bv = f(b)
    @test eltype(bv) === Bool
    @test length(bv) === 3
    @test convert(Vector, bv) ==  b
    @test sum(bv)    === 3
    @test !isempty(bv)
end

@testset "$f, setindex!" for f in weight_funcs
    w = [1., 2., 3.]
    wv = f(w)

    # Check getindex & sum
    @test wv[1] === 1.
    @test sum(wv) === 6.
    @test wv == w

    # Test setindex! success
    @test (wv[1] = 4) === 4             # setindex! returns original val
    @test wv[1] === 4.                  # value correctly converted and set
    @test sum(wv) === 9.                # sum updated
    @test wv == [4., 2., 3.]    # Test state of all values

    # Test mulivalue setindex!
    wv[1:2] = [3., 5.]
    @test wv[1] === 3.
    @test wv[2] === 5.
    @test sum(wv) === 11.
    @test wv == [3., 5., 3.]   # Test state of all values

    # Test failed setindex! due to conversion error
    w = [1, 2, 3]
    wv = f(w)

    @test_throws InexactError wv[1] = 1.5   # Returns original value
    @test wv[1] === 1                       # value not updated
    @test sum(wv) === 6                     # sum not corrupted
    @test wv == [1, 2, 3]           # Test state of all values
end

@testset "$f, isequal and ==" for f in weight_funcs
    x = f([1, 2, 3])

    y = f([1, 2, 3]) # same values, type and parameters
    @test isequal(x, y)
    @test x == y

    y = f([1.0, 2.0, 3.0]) # same values and type, different parameters
    @test isequal(x, y)
    @test x == y

    if f != fweights # same values and parameters, different types
        y = fweights([1, 2, 3])
        @test !isequal(x, y)
        @test x != y
    end

    x = f([1, 2, NaN]) # isequal and == treat NaN differently
    y = f([1, 2, NaN])
    @test isequal(x, y)
    @test x != y

    x = f([1.0, 2.0, 0.0]) # isequal and == treat ±0.0 differently
    y = f([1.0, 2.0, -0.0])
    @test !isequal(x, y)
    @test x == y
end

@testset "Unit weights" begin
    wv = uweights(Float64, 3)
    @test wv[1] === 1.
    @test wv[1:3] == fill(1.0, 3)
    @test wv[:] == fill(1.0, 3)
    @test !isempty(wv)
    @test length(wv) === 3
    @test size(wv) === (3,)
    @test sum(wv) === 3.
    @test wv == fill(1.0, 3)
    @test Statistics.varcorrection(wv) == 1/3
    @test !isequal(wv, fweights(fill(1.0, 3)))
    @test isequal(wv, uweights(3))
    @test wv != fweights(fill(1.0, 3))
    @test wv == uweights(3)
    @test wv[[true, false, false]] == uweights(Float64, 1)
end

## wsum

@testset "wsum" begin
    x = [6., 8., 9.]
    w = [2., 3., 4.]
    p = [1. 2. ; 3. 4.]
    q = [1., 2., 3., 4.]

    @test wsum(Float64[], weights=Float64[]) === 0.0
    @test wsum(x, weights=w) === 72.0
    @test wsum(p, weights=q) === 29.0

    ## wsum along dimension

    @test wsum(x, weights=w, dims=1) == [72.0]

    x  = rand(6, 8)
    w1 = rand(6)
    w2 = rand(8)

    @test size(wsum(x, weights=w1, dims=1)) == (1, 8)
    @test size(wsum(x, weights=w2, dims=2)) == (6, 1)

    @test wsum(x, weights=w1, dims=1) ≈ sum(x .* w1, dims=1)
    @test wsum(x, weights=w2, dims=2) ≈ sum(x .* w2', dims=2)

    x = rand(6, 5, 4)
    w1 = rand(6)
    w2 = rand(5)
    w3 = rand(4)

    @test size(wsum(x, weights=w1, dims=1)) == (1, 5, 4)
    @test size(wsum(x, weights=w2, dims=2)) == (6, 1, 4)
    @test size(wsum(x, weights=w3, dims=3)) == (6, 5, 1)

    @test wsum(x, weights=w1, dims=1) ≈ sum(x .* w1, dims=1)
    @test wsum(x, weights=w2, dims=2) ≈ sum(x .* w2', dims=2)
    @test wsum(x, weights=w3, dims=3) ≈ sum(x .* reshape(w3, 1, 1, 4), dims=3)

    v = view(x, 2:4, :, :)

    @test wsum(v, weights=w1[1:3], dims=1) ≈ sum(v .* w1[1:3], dims=1)
    @test wsum(v, weights=w2, dims=2)      ≈ sum(v .* w2', dims=2)
    @test wsum(v, weights=w3, dims=3)      ≈ sum(v .* reshape(w3, 1, 1, 4), dims=3)

    ## wsum for Arrays with non-BlasReal elements

    x = rand(1:100, 6, 8)
    w1 = rand(6)
    w2 = rand(8)

    @test wsum(x, weights=w1, dims=1) ≈ sum(x .* w1, dims=1)
    @test wsum(x, weights=w2, dims=2) ≈ sum(x .* w2', dims=2)

    ## wsum!

    x = rand(6)
    w = rand(6)

    r = ones(1)
    @test wsum!(r, x, weights=w, init=true) === r
    @test r ≈ [dot(x, w)]

    r = ones(1)
    @test wsum!(r, x, weights=w, init=false) === r
    @test r ≈ [dot(x, w) + 1.0]

    x = rand(6, 8)
    w1 = rand(6)
    w2 = rand(8)

    r = ones(1, 8)
    @test wsum!(r, x, weights=w1, init=true) === r
    @test r ≈ sum(x .* w1, dims=1)

    r = ones(1, 8)
    @test wsum!(r, x, weights=w1, init=false) === r
    @test r ≈ sum(x .* w1, dims=1) .+ 1.0

    r = ones(6, 1)
    @test wsum!(r, x, weights=w2, init=true) === r
    @test r ≈ sum(x .* w2', dims=2)

    r = ones(6, 1)
    @test wsum!(r, x, weights=w2, init=false) === r
    @test r ≈ sum(x .* w2', dims=2) .+ 1.0

    x = rand(8, 6, 5)
    w1 = rand(8)
    w2 = rand(6)
    w3 = rand(5)

    r = ones(1, 6, 5)
    @test wsum!(r, x, weights=w1, init=true) === r
    @test r ≈ sum(x .* w1, dims=1)

    r = ones(1, 6, 5)
    @test wsum!(r, x, weights=w1, init=false) === r
    @test r ≈ sum(x .* w1, dims=1) .+ 1.0

    r = ones(8, 1, 5)
    @test wsum!(r, x, weights=w2, init=true) === r
    @test r ≈ sum(x .* w2', dims=2)

    r = ones(8, 1, 5)
    @test wsum!(r, x, weights=w2, init=false) === r
    @test r ≈ sum(x .* w2', dims=2) .+ 1.0

    r = ones(8, 6, 1)
    @test wsum!(r, x, weights=w3, init=true) === r
    @test r ≈ sum(x .* reshape(w3, (1, 1, 5)), dims=3)

    r = ones(8, 6, 1)
    @test wsum!(r, x, weights=w3, init=false) === r
    @test r ≈ sum(x .* reshape(w3, (1, 1, 5)), dims=3) .+ 1.0

    # additional tests
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

# sum, mean and quantile

a = reshape(1.0:27.0, 3, 3, 3)

@testset "Sum $f" for f in weight_funcs
    @test wsum([1.0, 2.0, 3.0], weights=f([1.0, 0.5, 0.5])) ≈ 3.5
    @test wsum(1:3, weights=f([1.0, 1.0, 0.5]))             ≈ 4.5

    for wt in ([1.0, 1.0, 1.0], [1.0, 0.2, 0.0], [0.2, 0.0, 1.0])
        @test wsum(a, weights=f(wt), dims=1)  ≈ sum(a.*reshape(wt, length(wt), 1, 1), dims=1)
        @test wsum(a, weights=f(wt), dims=2)  ≈ sum(a.*reshape(wt, 1, length(wt), 1), dims=2)
        @test wsum(a, weights=f(wt), dims=3)  ≈ sum(a.*reshape(wt, 1, 1, length(wt)), dims=3)
    end
end

@testset "Mean $f" for f in weight_funcs
    @test mean([1:3;], weights=f([1.0, 1.0, 0.5])) ≈ 1.8
    @test mean(1:3, weights=f([1.0, 1.0, 0.5]))    ≈ 1.8

    a = reshape(1.0:27.0, 3, 3, 3)
    for wt in ([1.0, 1.0, 1.0], [1.0, 0.2, 0.0], [0.2, 0.0, 1.0])
        @test mean(a, weights=f(wt), dims=1) ≈
            sum(a.*reshape(wt, :, 1, 1), dims=1)/sum(wt)
        @test mean(a, weights=f(wt), dims=2) ≈
            sum(a.*reshape(wt, 1, :, 1), dims=2)/sum(wt)
        @test mean(a, weights=f(wt), dims=3) ≈
            sum(a.*reshape(wt, 1, 1, :), dims=3)/sum(wt)
        @test_throws DimensionMismatch mean(a, weights=f(wt), dims=4)
    end
end

@testset "Quantile fweights" begin
    data = (
        [7, 1, 2, 4, 10],
        [7, 1, 2, 4, 10],
        [7, 1, 2, 4, 10, 15],
        [1, 2, 4, 7, 10, 15],
        [0, 10, 20, 30],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [30, 40, 50, 60, 35],
        [2, 0.6, 1.3, 0.3, 0.3, 1.7, 0.7, 1.7],
        [1, 2, 2],
        [3.7, 3.3, 3.5, 2.8],
        [100, 125, 123, 60, 45, 56, 66],
        [2, 2, 2, 2, 2, 2],
        [2.3],
        [-2, -3, 1, 2, -10],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [-2, 2, -1, 3, 6],
        [-10, 1, 1, -10, -10],
    )
    wt = (
        [3, 1, 1, 1, 3],
        [1, 1, 1, 1, 1],
        [3, 1, 1, 1, 3, 3],
        [1, 1, 1, 3, 3, 3],
        [30, 191, 9, 0],
        [10, 1, 1, 1, 9],
        [10, 1, 1, 1, 900],
        [1, 3, 5, 4, 2],
        [2, 2, 5, 0, 2, 2, 1, 6],
        [1, 1, 8],
        [5, 5, 4, 1],
        [30, 56, 144, 24, 55, 43, 67],
        [1, 2, 3, 4, 5, 6],
        [12],
        [7, 1, 1, 1, 6],
        [1, 0, 0, 0, 2],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 2, 1],
        [0, 1, 1, 1, 1],
    )
    p = [0.0, 0.25, 0.5, 0.75, 1.0]
    function _rep(x::AbstractVector, lengths::AbstractVector{Int})
        res = similar(x, sum(lengths))
        i = 1
        for idx in 1:length(x)
            tmp = x[idx]
            for kdx in 1:lengths[idx]
                res[i] = tmp
                i += 1
            end
        end
        return res
    end
    # quantile with fweights is the same as repeated vectors
    for i = 1:length(data)
        @test quantile(data[i], p, weights=fweights(wt[i])) ≈
            quantile(_rep(data[i], wt[i]), p)
    end
    # quantile with fweights = 1  is the same as quantile
    for i = 1:length(data)
        @test quantile(data[i], p, weights=fweights(fill!(similar(wt[i]), 1))) ≈ quantile(data[i], p)
    end

    # Issue JuliaStats/StatsBase#313
    @test quantile([1, 2, 3, 4, 5], p, weights=fweights([0,1,2,1,0])) ≈
        quantile([2, 3, 3, 4], p)
    @test quantile([1, 2], 0.25, weights=fweights([1, 1])) ≈ 1.25
    @test quantile([1, 2], 0.25, weights=fweights([2, 2])) ≈ 1.0

    # test non integer frequency weights
    quantile([1, 2], 0.25, weights=fweights([1.0, 2.0])) ==
        quantile([1, 2], 0.25, weights=fweights([1, 2]))
    @test_throws ArgumentError quantile([1, 2], 0.25, weights=fweights([1.5, 2.0]))

    @test_throws ArgumentError quantile([1, 2], nextfloat(1.0), weights=fweights([1, 2]))
    @test_throws ArgumentError quantile([1, 2], prevfloat(0.0), weights=fweights([1, 2]))
end

@testset "Quantile aweights, pweights and weights" for f in (aweights, pweights, weights)
    data = (
        [7, 1, 2, 4, 10],
        [7, 1, 2, 4, 10],
        [7, 1, 2, 4, 10, 15],
        [1, 2, 4, 7, 10, 15],
        [0, 10, 20, 30],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [30, 40, 50, 60, 35],
        [2, 0.6, 1.3, 0.3, 0.3, 1.7, 0.7, 1.7],
        [1, 2, 2],
        [3.7, 3.3, 3.5, 2.8],
        [100, 125, 123, 60, 45, 56, 66],
        [2, 2, 2, 2, 2, 2],
        [2.3],
        [-2, -3, 1, 2, -10],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [-2, 2, -1, 3, 6],
        [-10, 1, 1, -10, -10],
    )
    wt = (
        [1, 1/3, 1/3, 1/3, 1],
        [1, 1, 1, 1, 1],
        [1, 1/3, 1/3, 1/3, 1, 1],
        [1/3, 1/3, 1/3, 1, 1, 1],
        [30, 191, 9, 0],
        [10, 1, 1, 1, 9],
        [10, 1, 1, 1, 900],
        [1, 3, 5, 4, 2],
        [2, 2, 5, 1, 2, 2, 1, 6],
        [0.1, 0.1, 0.8],
        [5, 5, 4, 1],
        [30, 56, 144, 24, 55, 43, 67],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [12],
        [7, 1, 1, 1, 6],
        [1, 0, 0, 0, 2],
        [1, 2, 3, 4, 5],
        [0.1, 0.2, 0.3, 0.2, 0.1],
        [1, 1, 1, 1, 1],
    )
    quantile_answers = (
        [1.0, 4.0, 6.0, 8.0, 10.0],
        [1.0, 2.0, 4.0, 7.0, 10.0],
        [1.0, 4.75, 7.5, 10.4166667, 15.0],
        [1.0, 4.75, 7.5, 10.4166667, 15.0],
        [0.0, 2.6178010, 5.2356021, 7.8534031, 20.0],
        [1.0, 4.0, 4.3333333, 4.6666667, 5.0],
        [1.0, 4.2475, 4.4983333, 4.7491667, 5.0],
        [30.0, 37.5, 44.0, 51.25, 60.0],
        [0.3, 0.7, 1.3, 1.7, 2.0],
        [1.0, 2.0, 2.0, 2.0, 2.0],
        [2.8, 3.15, 3.4, 3.56, 3.7],
        [45.0, 62.149253, 102.875, 117.4097222, 125.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [2.3, 2.3, 2.3, 2.3, 2.3],
        [-10.0, -2.7857143, -2.4285714, -2.0714286, 2.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 1.625, 2.3333333, 3.25, 5.0],
        [-2.0, -1.3333333, 0.5, 2.5, 6.0],
        [-10.0, -10.0, -10.0, 1.0, 1.0]
    )
    p = [0.0, 0.25, 0.5, 0.75, 1.0]

    Random.seed!(10)
    for i = 1:length(data)
        @test quantile(data[i], p, weights=f(wt[i])) ≈ quantile_answers[i] atol = 1e-5
        for j = 1:10
            # order of p does not matter
            reorder = sortperm(rand(length(p)))
            @test quantile(data[i], p[reorder], weights=f(wt[i])) ≈
                quantile_answers[i][reorder] atol = 1e-5
        end
        for j = 1:10
            # order of w does not matter
            reorder = sortperm(rand(length(data[i])))
            @test quantile(data[i][reorder], p, weights=f(wt[i][reorder])) ≈
                quantile_answers[i] atol = 1e-5
        end
    end
    # All equal weights corresponds to base quantile
    for v in (1, 2, 345)
        for i = 1:length(data)
            w = f(fill(v, length(data[i])))
            @test quantile(data[i], p, weights=w) ≈ quantile(data[i], p) atol = 1e-5
            for j = 1:10
                prandom = rand(4)
                @test quantile(data[i], prandom, weights=w) ≈
                    quantile(data[i], prandom) atol = 1e-5
            end
        end
    end
    # test zeros are removed
    for i = 1:length(data)
        @test quantile(vcat(1.0, data[i]), p, weights=f(vcat(0.0, wt[i]))) ≈
            quantile_answers[i] atol = 1e-5
    end
    # Syntax
    v = [7, 1, 2, 4, 10]
    w = [1, 1/3, 1/3, 1/3, 1]
    answer = 6.0
    @test quantile(data[1], 0.5, weights=f(w)) ≈  answer atol = 1e-5
    # alpha and beta not supported
    @test_throws ArgumentError quantile(1:4, 0.1, weights=f(1:4), alpha=2)
    @test_throws ArgumentError quantile(1:4, 0.1, weights=f(1:4), beta=2)
    @test_throws ArgumentError quantile(1:4, 0.1, weights=f(1:4), alpha=2, beta=2)
end

@testset "Median $f" for f in weight_funcs
    data = [4, 3, 2, 1]
    wt = [0, 0, 0, 0]
    @test_throws ArgumentError median(data, weights=f(wt))
    @test_throws ArgumentError median(Float64[], weights=f(Float64[]))
    wt = [1, 2, 3, 4, 5]
    @test_throws ArgumentError median(data, weights=f(wt))
    @test_throws ArgumentError median([4 3 2 1 0], weights=f(wt))
    @test_throws ArgumentError median([1 2; 4 5; 7 8; 10 11; 13 14],
                                      weights=f(wt))
    data = [1, 3, 2, NaN, 2]
    @test isnan(median(data, weights=f(wt)))
    wt = [1, 2, NaN, 4, 5]
    @test_throws ArgumentError median(data, weights=f(wt))
    data = [1, 3, 2, 1, 2]
    @test_throws ArgumentError median(data, weights=f(wt))
    wt = [-1, -1, -1, -1, -1]
    @test_throws ArgumentError median(data, weights=f(wt))
    wt = [-1, -1, -1, 0, 0]
    @test_throws ArgumentError median(data, weights=f(wt))

    data = [4, 3, 2, 1]
    wt = [1, 2, 3, 4]
    @test median(data, weights=f(wt)) ≈
        quantile(data, 0.5, weights=f(wt)) atol = 1e-5
end

@testset "Mismatched eltypes" begin
    @test round(mean(Union{Int,Missing}[1,2], weights=weights([1,2])), digits=3) ≈ 1.667
end

@testset "Sum, mean, quantiles and variance for unit weights" begin
    wt = uweights(Float64, 3)

    @test wsum([1.0, 2.0, 3.0], weights=wt) ≈ 6.0
    @test mean([1.0, 2.0, 3.0], weights=wt) ≈ 2.0

    @test wsum(a, weights=wt, dims=1) ≈ sum(a, dims=1)
    @test wsum(a, weights=wt, dims=2) ≈ sum(a, dims=2)
    @test wsum(a, weights=wt, dims=3) ≈ sum(a, dims=3)

    @test wsum(a, weights=wt, dims=1) ≈ sum(a, dims=1)
    @test wsum(a, weights=wt, dims=2) ≈ sum(a, dims=2)
    @test wsum(a, weights=wt, dims=3) ≈ sum(a, dims=3)

    @test mean(a, weights=wt, dims=1) ≈ mean(a, dims=1)
    @test mean(a, weights=wt, dims=2) ≈ mean(a, dims=2)
    @test mean(a, weights=wt, dims=3) ≈ mean(a, dims=3)

    @test_throws DimensionMismatch wsum(a, weights=wt)
    @test_throws DimensionMismatch wsum(a, weights=wt, dims=4)
    @test_throws DimensionMismatch wsum(a, weights=wt, dims=4)
    @test_throws DimensionMismatch mean(a, weights=wt, dims=4)

    @test quantile([1.0, 4.0, 6.0, 8.0, 10.0], [0.5], weights=uweights(5)) ≈ [6.0]
    @test quantile([1.0, 4.0, 6.0, 8.0, 10.0], 0.5, weights=uweights(5)) ≈ 6.0
    @test median([1.0, 4.0, 6.0, 8.0, 10.0], weights=uweights(5)) ≈ 6.0

    @test_throws DimensionMismatch var(a, weights=uweights(Float64, 27))
end

@testset "Exponential Weights" begin
    @testset "Usage" begin
        θ = 5.25
        λ = 1 - exp(-1 / θ)     # simple conversion for the more common/readable method
        v = [λ*(1-λ)^(1-i) for i = 1:4]
        w = Weights(v)

        @test round.(w, digits=4) == [0.1734, 0.2098, 0.2539, 0.3071]

        @testset "basic" begin
            @test eweights(1:4, λ) ≈ w
        end

        @testset "1:n" begin
            @test eweights(4, λ) ≈ w
        end

        @testset "indexin" begin
            v = [λ*(1-λ)^(1-i) for i = 1:10]

            # Test that we should be able to skip indices easily
            @test eweights([1, 3, 5, 7], 1:10, λ) ≈ Weights(v[[1, 3, 5, 7]])

            # This should also work with actual time types
            t1 = DateTime(2019, 1, 1, 1)
            tx = t1 + Hour(7)
            tn = DateTime(2019, 1, 2, 1)

            @test eweights(t1:Hour(2):tx, t1:Hour(1):tn, λ) ≈ Weights(v[[1, 3, 5, 7]])
        end
    end

    @testset "Empty" begin
        @test eweights(0, 0.3) == Weights(Float64[])
        @test eweights(1:0, 0.3) == Weights(Float64[])
        @test eweights(Int[], 1:10, 0.4) == Weights(Float64[])
    end

    @testset "Failure Conditions" begin
        # λ > 1.0
        @test_throws ArgumentError eweights(1, 1.1)

        # time indices are not all positive non-zero integers
        @test_throws ArgumentError eweights([0, 1, 2, 3], 0.3)

        # Passing in an array of bools will work because Bool <: Integer,
        # but any `false` values will trigger the same argument error as 0.0
        @test_throws ArgumentError eweights([true, false, true, true], 0.3)
    end
end

end # @testset Weights
