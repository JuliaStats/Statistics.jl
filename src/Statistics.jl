# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    Statistics

Standard library module for basic statistics functionality.
"""
module Statistics

using LinearAlgebra, SparseArrays
using LinearAlgebra: BlasReal

using Base: has_offset_axes, require_one_based_indexing

using Printf: @printf

export std, stdm, var, varm, mean!, mean,
    median!, median, middle, quantile!, quantile,
    # moments.jl
    skewness, kurtosis,
    # weights.jl
    AbstractWeights, Weights, AnalyticWeights, FrequencyWeights, ProbabilityWeights, UnitWeights,
    weights, aweights, eweights, fweights, pweights, uweights,
    # scalarstats.jl
    geomean, harmmean, genmean, mode, modes, percentile, span, variation, sem, mad, mad!,
    iqr, genvar, totalvar, entropy, renyientropy, crossentropy, kldivergence, describe,
    zscore, zscore!,
    # cov.jl
    cor, cov, scattermat, cov2cor, cor2cov, CovarianceEstimator, SimpleCovariance,
    # partialcor.jl
    partialcor,
    # signalcorr.jl
    autocov!, autocov, autocor!, autocor, crosscov!, crosscov, crosscor!, crosscor,
    pacf!, pacf,
    # robust.jl
    trim, trim!, trimvar, winsor, winsor!,
    # ranking.jl
    ordinalrank, competerank, denserank, tiedrank,
    # rankcorr.jl
    corkendall, corspearman,
    # empirical.jl
    ecdf, ECDF,
    # hist.jl
    fit, AbstractHistogram, Histogram, midpoints, norm, normalize, normalize!,
    # transformations
    unnormalize, unnormalize!,
    AbstractNormalization, MinMaxNormalization, ZScoreNormalization,
    # reliability.jl
    cronbachalpha, CronbachAlpha

include("common.jl")
include("weights.jl")
include("moments.jl")
include("scalarstats.jl")
include("cov.jl")
include("partialcor.jl")
include("toeplitzsolvers.jl")
include("signalcorr.jl")
include("robust.jl")
include("ranking.jl")
include("rankcorr.jl")
include("empirical.jl")
include("hist.jl")
include("transformations.jl")
include("reliability.jl")

##### mean #####

"""
    mean(itr)

Compute the mean of all elements in a collection.

!!! note
    If `itr` contains `NaN` or [`missing`](@ref) values, the result is also
    `NaN` or `missing` (`missing` takes precedence if array contains both).
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    mean of non-missing values.

# Examples
```jldoctest
julia> using Statistics

julia> mean(1:20)
10.5

julia> mean([1, missing, 3])
missing

julia> mean(skipmissing([1, missing, 3]))
2.0
```
"""
mean(itr) = mean(identity, itr)

"""
    mean(f::Function, itr)

Apply the function `f` to each element of collection `itr` and take the mean.

```jldoctest
julia> using Statistics

julia> mean(√, [1, 2, 3])
1.3820881233139908

julia> mean([√1, √2, √3])
1.3820881233139908
```
"""
function mean(f, itr)
    y = iterate(itr)
    if y === nothing
        return Base.mapreduce_empty_iter(f, +, itr,
                                         Base.IteratorEltype(itr)) / 0
    end
    count = 1
    value, state = y
    f_value = f(value)/1
    total = Base.reduce_first(+, f_value)
    y = iterate(itr, state)
    while y !== nothing
        value, state = y
        total += _mean_promote(total, f(value))
        count += 1
        y = iterate(itr, state)
    end
    return total/count
end

"""
    mean(f::Function, A::AbstractArray; dims)

Apply the function `f` to each element of array `A` and take the mean over dimensions `dims`.

!!! compat "Julia 1.3"
    This method requires at least Julia 1.3.

```jldoctest
julia> using Statistics

julia> mean(√, [1, 2, 3])
1.3820881233139908

julia> mean([√1, √2, √3])
1.3820881233139908

julia> mean(√, [1 2 3; 4 5 6], dims=2)
2×1 Matrix{Float64}:
 1.3820881233139908
 2.2285192400943226
```
"""
mean(f, A::AbstractArray; dims=:) = _mean(f, A, dims)

"""
    mean!(r, v; [weights::AbstractVector])

Compute the mean of `v` over the singleton dimensions of `r`, and write results to `r`.
If `r` has only one singleton dimension `i`, `weights` can be a vector of length
`size(v, i)` to compute the weighted mean.

!!! compat "Julia 1.3"
    The `weights`  argument requires at least Julia 1.3.

# Examples
```jldoctest
julia> using Statistics

julia> v = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> mean!([1., 1.], v)
2-element Vector{Float64}:
 1.5
 3.5

julia> mean!([1. 1.], v)
1×2 Matrix{Float64}:
 2.0  3.0
```
"""
mean!(R::AbstractArray, A::AbstractArray;
      weights::Union{AbstractArray,Nothing}=nothing) =
    _mean!(R, A, weights)

function _mean!(R::AbstractArray, A::AbstractArray, weights::Nothing)
    sum!(R, A; init=true)
    x = max(1, length(R)) // length(A)
    R .= R .* x
    return R
end

"""
    mean(A::AbstractArray; [dims], [weights::AbstractArray])

Compute the mean of array `A`.
If `dims` is provided, return an array of means over these dimensions.
If `weights` is provided, return the weighted mean(s). `weights` must be
either an array of the same size as `A` if `dims` is omitted,
or a vector with the same length as `size(A, dims)` if `dims` is provided.

!!! compat "Julia 1.1"
    `mean` for empty arrays requires at least Julia 1.1.

!!! compat "Julia 1.3"
    The `weights` keyword argument requires at least Julia 1.3.

# Examples
```jldoctest
julia> using Statistics

julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> mean(A, dims=1)
1×2 Matrix{Float64}:
 2.0  3.0

julia> mean(A, dims=2)
2×1 Matrix{Float64}:
 1.5
 3.5

julia> mean(A, weights=[2 1; 2 1])
2.3333333333333335

julia> mean(A, weights=[2, 1], dims=1)
1×2 Array{Float64,2}:
 1.66667  2.66667
```
"""
mean(A::AbstractArray; dims=:, weights::Union{AbstractArray, Nothing}=nothing) =
    _mean(identity, A, dims, weights)

_mean_promote(x::T, y::S) where {T,S} = convert(promote_type(T, S), y)

# ::Dims is there to force specializing on Colon (as it is a Function)
function _mean(f, A::AbstractArray, dims=:, weights::Nothing=nothing) where Dims
    isempty(A) && return sum(f, A, dims=dims)/0
    if dims === (:)
        n = length(A)
    else
        n = mapreduce(i -> size(A, i), *, unique(dims); init=1)
    end
    x1 = f(first(A)) / 1
    result = sum(x -> _mean_promote(x1, f(x)), A, dims=dims)
    if dims === (:)
        return result / n
    else
        return result ./= n
    end
end

function _mean(::typeof(identity), r::AbstractRange{<:Real}, dims::Colon, weights::Nothing)
    isempty(r) && return oftype((first(r) + last(r)) / 2, NaN)
    (first(r) + last(r)) / 2
end

##### variances #####

# faster computation of real(conj(x)*y)
realXcY(x::Real, y::Real) = x*y
realXcY(x::Complex, y::Complex) = real(x)*real(y) + imag(x)*imag(y)

function var(iterable; corrected::Bool=true, mean=nothing)
    s, count = _sumsq(iterable, mean)
    s / (count - Int(corrected))
end

function _sumsq(iterable, mean)
    y = iterate(iterable)
    if y === nothing
        T = eltype(iterable)
        return oftype((abs2(zero(T)) + abs2(zero(T)))/2, NaN), 0
    end
    count = 1
    value, state = y
    y = iterate(iterable, state)
    if mean === nothing
        # Use Welford algorithm as seen in (among other places)
        # Knuth's TAOCP, Vol 2, page 232, 3rd edition.
        M = value / 1
        S = real(zero(M))
        while y !== nothing
            value, state = y
            y = iterate(iterable, state)
            count += 1
            new_M = M + (value - M) / count
            S = S + realXcY(value - M, value - new_M)
            M = new_M
        end
        return S, count
    elseif isa(mean, Number) # mean provided
        # Cannot use a compensated version, e.g. the one from
        # "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances."
        # by Chan, Golub, and LeVeque, Technical Report STAN-CS-79-773,
        # Department of Computer Science, Stanford University,
        # because user can provide mean value that is different to mean(iterable)
        sum2 = abs2(value - mean::Number)
        while y !== nothing
            value, state = y
            y = iterate(iterable, state)
            count += 1
            sum2 += abs2(value - mean)
        end
        return sum2, count
    else
        throw(ArgumentError("invalid value of mean, $(mean)::$(typeof(mean))"))
    end
end

centralize_sumabs2(A::AbstractArray, m) =
    mapreduce(x -> abs2.(x - m), +, A)
centralize_sumabs2(A::AbstractArray, m, ifirst::Int, ilast::Int) =
    Base.mapreduce_impl(x -> abs2.(x - m), +, A, ifirst, ilast)

function centralize_sumabs2!(R::AbstractArray{S}, A::AbstractArray, means::AbstractArray,
                             w::Union{AbstractArray, Nothing}=nothing) where S
    # following the implementation of _mapreducedim! at base/reducedim.jl
    lsiz = Base.check_reducedims(R,A)
    for i in 1:max(ndims(R), ndims(means))
        if axes(means, i) != axes(R, i)
            throw(DimensionMismatch("dimension $i of `mean` should have indices $(axes(R, i)), but got $(axes(means, i))"))
        end
    end
    isempty(R) || fill!(R, zero(S))
    isempty(A) && return R

    if w === nothing && Base.has_fast_linear_indexing(A) && lsiz > 16 && !has_offset_axes(R, means)
        nslices = div(length(A), lsiz)
        ibase = first(LinearIndices(A))-1
        for i = 1:nslices
            @inbounds R[i] = centralize_sumabs2(A, means[i], ibase+1, ibase+lsiz)
            ibase += lsiz
        end
        return R
    end
    indsAt, indsRt = Base.safe_tail(axes(A)), Base.safe_tail(axes(R)) # handle d=1 manually
    keep, Idefault = Broadcast.shapeindexer(indsRt)
    if Base.reducedim1(R, A)
        i1 = first(Base.axes1(R))
        @inbounds for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            r = R[i1,IR]
            m = means[i1,IR]
            @simd for i in axes(A, 1)
                if w === nothing
                    r += abs2(A[i,IA] - m)
                else
                    r += abs2(A[i,IA] - m) * w[i]
                end
            end
            R[i1,IR] = r
        end
    else
        @inbounds for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            if w !== nothing
                wi = w[IA]
            end
            @simd for i in axes(A, 1)
                if w === nothing
                    R[i,IR] += abs2(A[i,IA] - means[i,IR])
                else
                    R[i,IR] += abs2(A[i,IA] - means[i,IR]) * wi
                end
            end
        end
    end
    return R
end

function varm!(R::AbstractArray{S}, A::AbstractArray, m::AbstractArray, w::Nothing;
               corrected::Bool=true) where S
    if isempty(A)
        fill!(R, convert(S, NaN))
    else
        rn = div(length(A), length(R)) - Int(corrected)
        centralize_sumabs2!(R, A, m)
        R .= R .* (1 // rn)
    end
    return R
end

function varm!(R::AbstractArray, A::AbstractArray, m::AbstractArray, w::AbstractArray;
               corrected::Bool=true)
    rmul!(centralize_sumabs2!(R, A, m, values(w)),
          varcorrection(w, corrected))
end

"""
    varm(itr, mean; dims, corrected::Bool=true)

Compute the sample variance of collection `itr`, with known mean(s) `mean`.

The algorithm returns an estimator of the generative distribution's variance
under the assumption that each entry of `itr` is a sample drawn from the same
unknown distribution, with the samples uncorrelated.
For arrays, this computation is equivalent to calculating
`sum((itr .- mean(itr)).^2) / (length(itr) - 1)`.
If `corrected` is `true`, then the sum is scaled with `n-1`,
whereas the sum is scaled with `n` if `corrected` is
`false` with `n` the number of elements in `itr`.

If `itr` is an `AbstractArray`, `dims` can be provided to compute the variance
over dimensions. In that case, `mean` must be an array with the same shape as
`mean(itr, dims=dims)` (additional trailing singleton dimensions are allowed).

!!! note
    If array contains `NaN` or [`missing`](@ref) values, the result is also
    `NaN` or `missing` (`missing` takes precedence if array contains both).
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    variance of non-missing values.
"""
varm(A::AbstractArray, m; corrected::Bool=true, dims=:,
     weights::Union{AbstractWeights, Nothing}=nothing) =
    _varm(A, m, corrected, dims, weights)

varm(iterable, m; corrected::Bool=true) =
    var(iterable, mean=m, corrected=corrected)

_varm(A::AbstractArray, m, corrected::Bool, dims, w::Nothing) =
    varm!(Base.reducedim_init(t -> abs2(t)/2, +, A, dims), A, m, w, corrected=corrected)

_varm(A::AbstractArray, m, corrected::Bool, dims, w::AbstractWeights{T}) where {T<:Real} =
    varm!(Base.reducedim_init(t -> (abs2(t)*zero(T))/2, +, A, dims), A, m, w,
          corrected=corrected)

function _varm(A::AbstractArray{T}, m, corrected::Bool, dims::Colon, w::Nothing) where T
    n = length(A)
    n == 0 && return oftype((abs2(zero(T)) + abs2(zero(T)))/2, NaN)
    return centralize_sumabs2(A, m) / (n - Int(corrected))
end

function _varm(A::AbstractArray{T}, m, corrected::Bool, dims::Colon,
               w::AbstractWeights) where T
    s = (zero(T) - zero(m))^2 * zero(eltype(w))
    @inbounds @simd for i in eachindex(A, w)
        z = A[i] - m
        s += (z * z) * w[i]
    end

    varcorrection(w, corrected) * s
end

"""
    var(itr; corrected::Bool=true, [weights::AbstractWeights], mean=nothing[, dims])

Compute the sample variance of collection `itr`.

The algorithm returns an estimator of the generative distribution's variance
under the assumption that each entry of `itr` is a sample drawn from the same
unknown distribution, with the samples uncorrelated.
For arrays, this computation is equivalent to calculating
`sum((itr .- mean(itr)).^2) / (length(itr) - 1))`.
If `corrected` is `true`, then the sum is scaled with `n-1`,
whereas the sum is scaled with `n` if `corrected` is
`false` where `n` is the number of elements in `itr`.

If `itr` is an `AbstractArray`, `dims` can be provided to compute the variance
over dimensions.

A pre-computed `mean` may be provided. When `dims` is specified, `mean` must be
an array with the same shape as `mean(itr, dims=dims)` (additional trailing
singleton dimensions are allowed).

If `itr` is an `AbstractArray`, `weights` can be provided to compute the weighted
variance. `weights` must be either an array of the same size
as `A` if `dims` is omitted, or a vector with the same length as `size(A, dims)`
if `dims` is provided.
The weighted uncorrected (when `corrected=false`) sample variance
is defined as:
```math
\\frac{1}{\\sum{w}} \\sum_{i=1}^n {w_i\\left({x_i - μ}\\right)^2 }
```
where ``n`` is the length of the input and ``μ`` is the mean.
The unbiased estimate (when `corrected=true`) of the population variance is
computed by replacing ``\\frac{1}{\\sum{w}}`` with a factor dependent on the type of
weights used:
* [`AnalyticWeights`](@ref): ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* [`FrequencyWeights`](@ref): ``\\frac{1}{\\sum{w} - 1}``
* [`ProbabilityWeights`](@ref): ``\\frac{n}{(n - 1) \\sum w}`` where ``n``
  equals `count(!iszero, w)`
* [`Weights`](@ref): `ArgumentError` (bias correction not supported)

!!! note
    If array contains `NaN` or [`missing`](@ref) values, the result is also
    `NaN` or `missing` (`missing` takes precedence if array contains both).
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    variance of non-missing values.
"""
var(A::AbstractArray;
    corrected::Bool=true, mean=nothing, dims=:,
    weights::Union{AbstractWeights, Nothing}=nothing) =
    _var(A, corrected, mean, dims, weights)

function _var(A::AbstractArray, corrected::Bool, mean, dims,
              w::Union{AbstractWeights, Nothing})
  if mean === nothing
      mean = Statistics.mean(A, dims=dims, weights=w)
  end
  return varm(A, mean; corrected=corrected, dims=dims, weights=w)
end

function _var(A::AbstractArray, corrected::Bool, mean, ::Colon,
              w::Union{AbstractWeights, Nothing})
  if mean === nothing
      mean = Statistics.mean(A, weights=w)
  end
  return real(varm(A, mean; corrected=corrected, weights=w))
end

## variances over ranges

varm(v::AbstractRange, m::AbstractArray) = range_varm(v, m)
varm(v::AbstractRange, m) = range_varm(v, m)

function range_varm(v::AbstractRange, m)
    f  = first(v) - m
    s  = step(v)
    l  = length(v)
    vv = f^2 * l / (l - 1) + f * s * l + s^2 * l * (2 * l - 1) / 6
    if l == 0 || l == 1
        return typeof(vv)(NaN)
    end
    return vv
end

function var(v::AbstractRange)
    s  = step(v)
    l  = length(v)
    vv = abs2(s) * (l + 1) * l / 12
    if l == 0 || l == 1
        return typeof(vv)(NaN)
    end
    return vv
end


##### standard deviation #####

function sqrt!(A::AbstractArray)
    for i in eachindex(A)
        @inbounds A[i] = sqrt(A[i])
    end
    A
end

stdm(A::AbstractArray, m; corrected::Bool=true) =
    sqrt.(varm(A, m; corrected=corrected))

"""
    std(itr; corrected::Bool=true, mean=nothing, [weights::AbstractWeights], [dims])

The algorithm returns an estimator of the generative distribution's standard
deviation under the assumption that each entry of `itr` is a sample drawn from
the same unknown distribution, with the samples uncorrelated.
For arrays, this computation is equivalent to calculating
`sqrt(sum((itr .- mean(itr)).^2) / (length(itr) - 1))`.
If `corrected` is `true`, then the sum is scaled with `n-1`,
whereas the sum is scaled with `n` if `corrected` is
`false` with `n` the number of elements in `itr`.

If `itr` is an `AbstractArray`, `dims` can be provided to compute the standard deviation
over dimensions, and `mean` may contain means for each dimension of `itr`.

If `itr` is an `AbstractArray`, `weights` can be provided to compute the weighted
standard deviation. `weights` must be either an array of the same size
as `A` if `dims` is omitted, or a vector with the same length as `size(A, dims)`
if `dims` is provided.
The weighted uncorrected (when `corrected=false`) sample standard deviation
is defined as:
```math
\\sqrt{\\frac{1}{\\sum{w}} \\sum_{i=1}^n {w_i\\left({x_i - μ}\\right)^2 }}
```
where ``n`` is the length of the input and ``μ`` is the mean.
The unbiased estimate (when `corrected=true`) of the population standard deviation is
computed by replacing ``\\frac{1}{\\sum{w}}`` with a factor dependent on the type of
weights used:
* [`AnalyticWeights`](@ref): ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* [`FrequencyWeights`](@ref): ``\\frac{1}{\\sum{w} - 1}``
* [`ProbabilityWeights`](@ref): ``\\frac{n}{(n - 1) \\sum w}`` where ``n``
  equals `count(!iszero, w)`
* [`Weights`](@ref): `ArgumentError` (bias correction not supported)

A pre-computed `mean` may be provided. When `dims` is specified, `mean` must be
an array with the same shape as `mean(itr, dims=dims)` (additional trailing
singleton dimensions are allowed).

!!! note
    If array contains `NaN` or [`missing`](@ref) values, the result is also
    `NaN` or `missing` (`missing` takes precedence if array contains both).
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    standard deviation of non-missing values.

!!! compat "Julia 1.3"
    The `weights` keyword argument requires at least Julia 1.3.
"""
std(A::AbstractArray;
    corrected::Bool=true, mean=nothing, dims=:,
    weights::Union{AbstractWeights, Nothing}=nothing) =
    _std(A, corrected, mean, dims, weights)

_std(A::AbstractArray, corrected::Bool, mean, dims,
     weights::Union{AbstractWeights, Nothing}) =
    sqrt.(var(A; corrected=corrected, mean=mean, dims=dims, weights=weights))

_std(A::AbstractArray, corrected::Bool, mean, ::Colon, w::Union{AbstractWeights, Nothing}) =
    sqrt.(var(A; corrected=corrected, mean=mean, weights=w))

_std(A::AbstractArray{<:AbstractFloat}, corrected::Bool, mean, dims,
     w::Union{AbstractWeights, Nothing}) =
    sqrt!(var(A; corrected=corrected, mean=mean, dims=dims, weights=w))

_std(A::AbstractArray{<:AbstractFloat}, corrected::Bool, mean, ::Colon,
     w::Union{AbstractWeights, Nothing}) =
    sqrt.(var(A; corrected=corrected, mean=mean, weights=w))

std(iterable; corrected::Bool=true, mean=nothing) =
    sqrt(var(iterable, corrected=corrected, mean=mean))

"""
    stdm(itr, mean; corrected::Bool=true)

Compute the sample standard deviation of collection `itr`, with known mean(s) `mean`.

The algorithm returns an estimator of the generative distribution's standard
deviation under the assumption that each entry of `itr` is a sample drawn from
the same unknown distribution, with the samples uncorrelated.
For arrays, this computation is equivalent to calculating
`sqrt(sum((itr .- mean(itr)).^2) / (length(itr) - 1))`.
If `corrected` is `true`, then the sum is scaled with `n-1`,
whereas the sum is scaled with `n` if `corrected` is
`false` with `n` the number of elements in `itr`.

If `itr` is an `AbstractArray`, `dims` can be provided to compute the standard deviation
over dimensions. In that case, `mean` must be an array with the same shape as
`mean(itr, dims=dims)` (additional trailing singleton dimensions are allowed).

!!! note
    If array contains `NaN` or [`missing`](@ref) values, the result is also
    `NaN` or `missing` (`missing` takes precedence if array contains both).
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    standard deviation of non-missing values.
"""
stdm(iterable, m; corrected::Bool=true) =
    std(iterable, corrected=corrected, mean=m)


###### covariance ######

# auxiliary functions

_conj(x::AbstractArray{<:Real}) = x
_conj(x::AbstractArray) = conj(x)

_getnobs(x::AbstractVector, vardim::Int) = length(x)
_getnobs(x::AbstractMatrix, vardim::Int) = size(x, vardim)

function _getnobs(x::AbstractVecOrMat, y::AbstractVecOrMat, vardim::Int)
    n = _getnobs(x, vardim)
    _getnobs(y, vardim) == n || throw(DimensionMismatch("dimensions of x and y mismatch"))
    return n
end

_vmean(x::AbstractVector, vardim::Int, w::Union{AbstractWeights, Nothing}=nothing) =
    mean(x, weights=w)
_vmean(x::AbstractMatrix, vardim::Int, w::Union{AbstractWeights, Nothing}=nothing) =
    mean(x, dims=vardim, weights=w)

# core functions

unscaled_covzm(x::AbstractVector{<:Number})    = sum(abs2, x)
unscaled_covzm(x::AbstractVector)              = sum(t -> t*t', x)
unscaled_covzm(x::AbstractMatrix, vardim::Int) = (vardim == 1 ? _conj(x'x) : x * x')

unscaled_covzm(x::AbstractVector, y::AbstractVector) = sum(conj(y[i])*x[i] for i in eachindex(y, x))
unscaled_covzm(x::AbstractVector, y::AbstractMatrix, vardim::Int) =
    (vardim == 1 ? *(transpose(x), _conj(y)) : *(transpose(x), transpose(_conj(y))))
unscaled_covzm(x::AbstractMatrix, y::AbstractVector, vardim::Int) =
    (c = vardim == 1 ? *(transpose(x), _conj(y)) :  x * _conj(y); reshape(c, length(c), 1))
unscaled_covzm(x::AbstractMatrix, y::AbstractMatrix, vardim::Int) =
    (vardim == 1 ? *(transpose(x), _conj(y)) : *(x, adjoint(y)))

# covzm (with centered data)

covzm(x::AbstractVector; corrected::Bool=true) = unscaled_covzm(x) / (length(x) - Int(corrected))
function covzm(x::AbstractMatrix, vardim::Int=1; corrected::Bool=true)
    C = unscaled_covzm(x, vardim)
    T = promote_type(typeof(first(C) / 1), eltype(C))
    A = convert(AbstractMatrix{T}, C)
    b = 1//(size(x, vardim) - corrected)
    A .= A .* b
    return A
end
covzm(x::AbstractVector, y::AbstractVector; corrected::Bool=true) =
    unscaled_covzm(x, y) / (length(x) - Int(corrected))
function covzm(x::AbstractVecOrMat, y::AbstractVecOrMat, vardim::Int=1; corrected::Bool=true)
    C = unscaled_covzm(x, y, vardim)
    T = promote_type(typeof(first(C) / 1), eltype(C))
    A = convert(AbstractArray{T}, C)
    b = 1//(_getnobs(x, y, vardim) - corrected)
    A .= A .* b
    return A
end

# covm (with provided mean)
## Use map(t -> t - xmean, x) instead of x .- xmean to allow for Vector{Vector}
## which can't be handled by broadcast
covm(x::AbstractVector, xmean; corrected::Bool=true) =
    covzm(map(t -> t - xmean, x); corrected=corrected)
covm(x::AbstractMatrix, xmean, weights::Nothing=nothing, vardim::Int=1; corrected::Bool=true) =
    covzm(x .- xmean, vardim; corrected=corrected)
covm(x::AbstractVector, xmean, y::AbstractVector, ymean; corrected::Bool=true) =
    covzm(map(t -> t - xmean, x), map(t -> t - ymean, y); corrected=corrected)
covm(x::AbstractVecOrMat, xmean, y::AbstractVecOrMat, ymean, vardim::Int=1; corrected::Bool=true) =
    covzm(x .- xmean, y .- ymean, vardim; corrected=corrected)

# cov (API)
"""
    cov(x::AbstractVector; corrected::Bool=true)

Compute the variance of the vector `x`. If `corrected` is `true` (the default) then the sum
is scaled with `n-1`, whereas the sum is scaled with `n` if `corrected` is `false` where `n = length(x)`.
"""
cov(x::AbstractVector; corrected::Bool=true) = covm(x, mean(x); corrected=corrected)

"""
    cov(X::AbstractMatrix; dims::Int=1, corrected::Bool=true[, weights::AbstractWeights])

Compute the covariance matrix of the matrix `X` along the dimension `dims`. If `corrected`
is `true` (the default) then the sum is scaled with `n-1`, whereas the sum is scaled with `n`
if `corrected` is `false` where `n = size(X, dims)`.

If `weights` is provided, the biased covariance matrix (`corrected=false`)
is computed by multiplying `scattermat(X, w)` by
``\\frac{1}{\\sum{w}}`` to normalize. However, the unbiased covariance matrix
(`corrected=true`) is dependent on the type of weights used:
* `AnalyticWeights`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* `FrequencyWeights`: ``\\frac{1}{\\sum{w} - 1}``
* `ProbabilityWeights`: ``\\frac{n}{(n - 1) \\sum w}`` where ``n`` equals `count(!iszero, w)`
* `Weights`: `ArgumentError` (bias correction not supported)
"""
cov(X::AbstractMatrix; dims::Int=1, corrected::Bool=true,
    weights::Union{AbstractWeights, Nothing}=nothing) =
    covm(X, _vmean(X, dims, weights), weights, dims; corrected=corrected)

"""
    cov(x::AbstractVector, y::AbstractVector; corrected::Bool=true)

Compute the covariance between the vectors `x` and `y`. If `corrected` is `true` (the
default), computes ``\\frac{1}{n-1}\\sum_{i=1}^n (x_i-\\bar x) (y_i-\\bar y)^*`` where
``*`` denotes the complex conjugate and `n = length(x) = length(y)`. If `corrected` is
`false`, computes ``\\frac{1}{n}\\sum_{i=1}^n (x_i-\\bar x) (y_i-\\bar y)^*``.
"""
cov(x::AbstractVector, y::AbstractVector; corrected::Bool=true) =
    covm(x, mean(x), y, mean(y); corrected=corrected)

"""
    cov(X::AbstractVecOrMat, Y::AbstractVecOrMat; dims::Int=1, corrected::Bool=true)

Compute the covariance between the vectors or matrices `X` and `Y` along the dimension
`dims`. If `corrected` is `true` (the default) then the sum is scaled with `n-1`, whereas
the sum is scaled with `n` if `corrected` is `false` where `n = size(X, dims) = size(Y, dims)`.
"""
cov(X::AbstractVecOrMat, Y::AbstractVecOrMat; dims::Int=1, corrected::Bool=true) =
    covm(X, _vmean(X, dims), Y, _vmean(Y, dims), dims; corrected=corrected)

##### correlation #####

"""
    clampcor(x)

Clamp a real correlation to between -1 and 1, leaving complex correlations unchanged
"""
clampcor(x::Real) = clamp(x, -1, 1)
clampcor(x) = x

# cov2cor!

function cov2cor!(C::AbstractMatrix{T}, xsd::AbstractArray) where T
    require_one_based_indexing(C, xsd)
    nx = length(xsd)
    size(C) == (nx, nx) || throw(DimensionMismatch("inconsistent dimensions"))
    for j = 1:nx
        for i = 1:j-1
            C[i,j] = adjoint(C[j,i])
        end
        C[j,j] = oneunit(T)
        for i = j+1:nx
            C[i,j] = clampcor(C[i,j] / (xsd[i] * xsd[j]))
        end
    end
    return C
end
function cov2cor!(C::AbstractMatrix, xsd, ysd::AbstractArray)
    require_one_based_indexing(C, ysd)
    nx, ny = size(C)
    length(ysd) == ny || throw(DimensionMismatch("inconsistent dimensions"))
    for (j, y) in enumerate(ysd)   # fixme (iter): here and in all `cov2cor!` we assume that `C` is efficiently indexed by integers
        for i in 1:nx
            C[i,j] = clampcor(C[i, j] / (xsd * y))
        end
    end
    return C
end
function cov2cor!(C::AbstractMatrix, xsd::AbstractArray, ysd)
    require_one_based_indexing(C, xsd)
    nx, ny = size(C)
    length(xsd) == nx || throw(DimensionMismatch("inconsistent dimensions"))
    for j in 1:ny
        for (i, x) in enumerate(xsd)
            C[i,j] = clampcor(C[i,j] / (x * ysd))
        end
    end
    return C
end
function cov2cor!(C::AbstractMatrix, xsd::AbstractArray, ysd::AbstractArray)
    require_one_based_indexing(C, xsd, ysd)
    nx, ny = size(C)
    (length(xsd) == nx && length(ysd) == ny) ||
        throw(DimensionMismatch("inconsistent dimensions"))
    for (i, x) in enumerate(xsd)
        for (j, y) in enumerate(ysd)
            C[i,j] = clampcor(C[i,j] / (x * y))
        end
    end
    return C
end

# corzm (non-exported, with centered data)

corzm(x::AbstractVector{T}) where {T} =
    T === Missing ? missing : one(float(nonmissingtype(T)))
function corzm(x::AbstractMatrix, vardim::Int=1)
    c = unscaled_covzm(x, vardim)
    return cov2cor!(c, collect(sqrt(c[i,i]) for i in 1:min(size(c)...)))
end
corzm(x::AbstractVector, y::AbstractMatrix, vardim::Int=1) =
    cov2cor!(unscaled_covzm(x, y, vardim), sqrt(sum(abs2, x)), sqrt!(sum(abs2, y, dims=vardim)))
corzm(x::AbstractMatrix, y::AbstractVector, vardim::Int=1) =
    cov2cor!(unscaled_covzm(x, y, vardim), sqrt!(sum(abs2, x, dims=vardim)), sqrt(sum(abs2, y)))
corzm(x::AbstractMatrix, y::AbstractMatrix, vardim::Int=1) =
    cov2cor!(unscaled_covzm(x, y, vardim), sqrt!(sum(abs2, x, dims=vardim)), sqrt!(sum(abs2, y, dims=vardim)))

# corm

corm(x::AbstractVector{T}, xmean) where {T} =
    T === Missing ? missing : one(float(nonmissingtype(T)))
corm(x::AbstractMatrix, xmean, weights::Nothing=nothing, vardim::Int=1) =
    corzm(x .- xmean, vardim)
function corm(x::AbstractVector, mx, y::AbstractVector, my)
    require_one_based_indexing(x, y)
    n = length(x)
    length(y) == n || throw(DimensionMismatch("inconsistent lengths"))
    n > 0 || throw(ArgumentError("correlation only defined for non-empty vectors"))

    @inbounds begin
        # Initialize the accumulators
        xx = zero(sqrt(abs2(x[1])))
        yy = zero(sqrt(abs2(y[1])))
        xy = zero(x[1] * y[1]')

        @simd for i in eachindex(x, y)
            xi = x[i] - mx
            yi = y[i] - my
            xx += abs2(xi)
            yy += abs2(yi)
            xy += xi * yi'
        end
    end
    return clampcor(xy / max(xx, yy) / sqrt(min(xx, yy) / max(xx, yy)))
end

corm(x::AbstractVecOrMat, xmean, y::AbstractVecOrMat, ymean, vardim::Int=1) =
    corzm(x .- xmean, y .- ymean, vardim)

# cor
"""
    cor(x::AbstractVector)

Return the number one.
"""
cor(x::AbstractVector{T}) where {T} =
    T === Missing ? missing : one(float(nonmissingtype(T)))

"""
    cor(X::AbstractMatrix; dims::Int=1[, weights::AbstractWeights])

Compute the Pearson correlation matrix of the matrix `X` along the dimension `dims`.
The weighted correlation is computed if `weights` is provided.
"""
cor(X::AbstractMatrix; dims::Int=1, weights::Union{AbstractWeights, Nothing}=nothing) =
    corm(X, _vmean(X, dims, weights), weights, dims)

"""
    cor(x::AbstractVector, y::AbstractVector)

Compute the Pearson correlation between the vectors `x` and `y`.
"""
cor(x::AbstractVector, y::AbstractVector) = corm(x, mean(x), y, mean(y))

"""
    cor(X::AbstractVecOrMat, Y::AbstractVecOrMat; dims=1)

Compute the Pearson correlation between the vectors or matrices `X` and `Y` along the dimension `dims`.
"""
cor(x::AbstractVecOrMat, y::AbstractVecOrMat; dims::Int=1) =
    corm(x, _vmean(x, dims), y, _vmean(y, dims), dims)

##### middle, median & quantiles #####

"""
    middle(x)

Compute the middle of a scalar value, which is equivalent to `x` itself, but of the type of `middle(x, x)` for consistency.
"""
middle(x::Union{Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128}) = Float64(x)
# Specialized functions for number types allow for improved performance
middle(x::AbstractFloat) = x
middle(x::Number) = (x + zero(x)) / 1

"""
    middle(x, y)

Compute the middle of two numbers `x` and `y`, which is
equivalent in both value and type to computing their mean (`(x + y) / 2`).
"""
middle(x::Number, y::Number) = x/2 + y/2

"""
    middle(range)

Compute the middle of a range, which consists of computing the mean of its extrema.
Since a range is sorted, the mean is performed with the first and last element.

```jldoctest
julia> using Statistics

julia> middle(1:10)
5.5
```
"""
middle(a::AbstractRange) = middle(a[1], a[end])

"""
    middle(a)

Compute the middle of an array `a`, which consists of finding its
extrema and then computing their mean.

```jldoctest
julia> using Statistics

julia> a = [1,2,3.6,10.9]
4-element Vector{Float64}:
  1.0
  2.0
  3.6
 10.9

julia> middle(a)
5.95
```
"""
middle(a::AbstractArray) = ((v1, v2) = extrema(a); middle(v1, v2))

"""
    median!(v)

Like [`median`](@ref), but may overwrite the input vector.
"""
function median!(v::AbstractVector)
    isempty(v) && throw(ArgumentError("median of an empty array is undefined, $(repr(v))"))
    eltype(v)>:Missing && any(ismissing, v) && return missing
    any(x -> x isa Number && isnan(x), v) && return convert(eltype(v), NaN)
    inds = axes(v, 1)
    n = length(inds)
    mid = div(first(inds)+last(inds),2)
    if isodd(n)
        return middle(partialsort!(v,mid))
    else
        m = partialsort!(v, mid:mid+1)
        return middle(m[1], m[2])
    end
end
median!(v::AbstractArray) = median!(vec(v))

"""
    median(itr)

Compute the median of all elements in a collection.
For an even number of elements no exact median element exists, so the result is
equivalent to calculating mean of two median elements.

!!! note
    If `itr` contains `NaN` or [`missing`](@ref) values, the result is also
    `NaN` or `missing` (`missing` takes precedence if `itr` contains both).
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    median of non-missing values.

# Examples
```jldoctest
julia> using Statistics

julia> median([1, 2, 3])
2.0

julia> median([1, 2, 3, 4])
2.5

julia> median([1, 2, missing, 4])
missing

julia> median(skipmissing([1, 2, missing, 4]))
2.0
```
"""
median(itr) = median!(collect(itr))

"""
    median(A::AbstractArray; [dims], [weights::AbstractArray])

Compute the median of array `A`.
If `dims` is provided, return an array of median over these dimensions.
If `weights` is provided, return the weighted median(s). `weights` must be
either an array of the same size as `A`. `dims` and `weights` cannot be specified
at the same time.

See the documentation for [`quantile`](@ref) for more details.

!!! compat "Julia 1.3"
    The `weights` keyword argument requires at least Julia 1.3c.

# Examples
```jl
julia> using Statistics

julia> median([1 2; 3 4], dims=1)
1×2 Matrix{Float64}:
 2.0  3.0

julia> median([1 2; 3 4], weights=fweights([1 1; 2 1]))
3.0
```
"""
median(A::AbstractArray; dims=:, weights::Union{AbstractArray, Nothing}=nothing) =
    _median(A, dims, weights)

_median(r::AbstractRange{<:Real}, dims::Colon, w::Nothing) = mean(r)

_median(A::AbstractArray, dims, w::Nothing) = mapslices(median!, A, dims = dims)

_median(A::AbstractArray{T}, dims::Colon, w::Nothing) where {T} =
    median!(copyto!(Array{T,1}(undef, length(A)), A))

"""
    quantile!([q::AbstractArray, ] v::AbstractVector, p; sorted=false, alpha::Real=1.0, beta::Real=alpha)

Compute the quantile(s) of a vector `v` at a specified probability or vector or tuple of
probabilities `p` on the interval [0,1]. If `p` is a vector, an optional
output array `q` may also be specified. (If not provided, a new output array is created.)
The keyword argument `sorted` indicates whether `v` can be assumed to be sorted; if
`false` (the default), then the elements of `v` will be partially sorted in-place.

By default (`alpha = beta = 1`), quantiles are computed via linear interpolation between the points
`((k-1)/(n-1), v[k])`, for `k = 1:n` where `n = length(v)`. This corresponds to Definition 7
of Hyndman and Fan (1996), and is the same as the R and NumPy default.

The keyword arguments `alpha` and `beta` correspond to the same parameters in Hyndman and Fan,
setting them to different values allows to calculate quantiles with any of the methods 4-9
defined in this paper:
- Def. 4: `alpha=0`, `beta=1`
- Def. 5: `alpha=0.5`, `beta=0.5`
- Def. 6: `alpha=0`, `beta=0` (Excel `PERCENTILE.EXC`, Python default, Stata `altdef`)
- Def. 7: `alpha=1`, `beta=1` (Julia, R and NumPy default, Excel `PERCENTILE` and `PERCENTILE.INC`, Python `'inclusive'`)
- Def. 8: `alpha=1/3`, `beta=1/3`
- Def. 9: `alpha=3/8`, `beta=3/8`

!!! note
    An `ArgumentError` is thrown if `v` contains `NaN` or [`missing`](@ref) values.

# References
- Hyndman, R.J and Fan, Y. (1996) "Sample Quantiles in Statistical Packages",
  *The American Statistician*, Vol. 50, No. 4, pp. 361-365

- [Quantile on Wikipedia](https://en.m.wikipedia.org/wiki/Quantile) details the different quantile definitions

# Examples
```jldoctest
julia> using Statistics

julia> x = [3, 2, 1];

julia> quantile!(x, 0.5)
2.0

julia> x
3-element Vector{Int64}:
 1
 2
 3

julia> y = zeros(3);

julia> quantile!(y, x, [0.1, 0.5, 0.9]) === y
true

julia> y
3-element Vector{Float64}:
 1.2000000000000002
 2.0
 2.8000000000000003
```
"""
function quantile!(q::AbstractArray, v::AbstractVector, p::AbstractArray;
                   sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha)
    require_one_based_indexing(q, v, p)
    if size(p) != size(q)
        throw(DimensionMismatch("size of p, $(size(p)), must equal size of q, $(size(q))"))
    end
    isempty(q) && return q

    minp, maxp = extrema(p)
    _quantilesort!(v, sorted, minp, maxp)

    for (i, j) in zip(eachindex(p), eachindex(q))
        @inbounds q[j] = _quantile(v,p[i], alpha=alpha, beta=beta)
    end
    return q
end

function quantile!(v::AbstractVector, p::Union{AbstractArray, Tuple{Vararg{Real}}};
                   sorted::Bool=false, alpha::Real=1., beta::Real=alpha)
    if !isempty(p)
        minp, maxp = extrema(p)
        _quantilesort!(v, sorted, minp, maxp)
    end
    return map(x->_quantile(v, x, alpha=alpha, beta=beta), p)
end

quantile!(v::AbstractVector, p::Real; sorted::Bool=false, alpha::Real=1., beta::Real=alpha) =
    _quantile(_quantilesort!(v, sorted, p, p), p, alpha=alpha, beta=beta)

# Function to perform partial sort of v for quantiles in given range
function _quantilesort!(v::AbstractArray, sorted::Bool, minp::Real, maxp::Real)
    isempty(v) && throw(ArgumentError("empty data vector"))
    require_one_based_indexing(v)

    if !sorted
        lv = length(v)
        lo = floor(Int,minp*(lv))
        hi = ceil(Int,1+maxp*(lv))

        # only need to perform partial sort
        sort!(v, 1, lv, Base.Sort.PartialQuickSort(lo:hi), Base.Sort.Forward)
    end
    if (sorted && (ismissing(v[end]) || (v[end] isa Number && isnan(v[end])))) ||
        any(x -> ismissing(x) || (x isa Number && isnan(x)), v)
        throw(ArgumentError("quantiles are undefined in presence of NaNs or missing values"))
    end
    return v
end

# Core quantile lookup function: assumes `v` sorted
@inline function _quantile(v::AbstractVector, p::Real; alpha::Real=1.0, beta::Real=alpha)
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))
    0 <= alpha <= 1 || throw(ArgumentError("alpha parameter out of [0,1] range"))
    0 <= beta <= 1 || throw(ArgumentError("beta parameter out of [0,1] range"))
    require_one_based_indexing(v)

    n = length(v)
    
    @assert n > 0 # this case should never happen here
    
    m = alpha + p * (one(alpha) - alpha - beta)
    aleph = n*p + oftype(p, m)
    j = clamp(trunc(Int, aleph), 1, n-1)
    γ = clamp(aleph - j, 0, 1)

    if n == 1
        a = v[1]
        b = v[1]
    else
        a = v[j]
        b = v[j + 1]
    end
    
    if isfinite(a) && isfinite(b)
        return a + γ*(b-a)
    else
        return (1-γ)*a + γ*b
    end
end

"""
    quantile(itr, p; sorted=false, alpha::Real=1.0, beta::Real=alpha, [weights::AbstractWeights])

Compute the quantile(s) of a collection `itr` at a specified probability or vector or tuple of
probabilities `p` on the interval [0,1]. The keyword argument `sorted` indicates whether
`itr` can be assumed to be sorted.

Samples quantile are defined by `Q(p) = (1-γ)*x[j] + γ*x[j+1]`,
where ``x[j]`` is the j-th order statistic, and `γ` is a function of
`j = floor(n*p + m)`, `m = alpha + p*(1 - alpha - beta)` and
`g = n*p + m - j`.

By default (`alpha = beta = 1`), quantiles are computed via linear interpolation between the points
`((k-1)/(n-1), v[k])`, for `k = 1:n` where `n = length(itr)`. This corresponds to Definition 7
of Hyndman and Fan (1996), and is the same as the R and NumPy default.

The keyword arguments `alpha` and `beta` correspond to the same parameters in Hyndman and Fan,
setting them to different values allows to calculate quantiles with any of the methods 4-9
defined in this paper:
- Def. 4: `alpha=0`, `beta=1`
- Def. 5: `alpha=0.5`, `beta=0.5`
- Def. 6: `alpha=0`, `beta=0` (Excel `PERCENTILE.EXC`, Python default, Stata `altdef`)
- Def. 7: `alpha=1`, `beta=1` (Julia, R and NumPy default, Excel `PERCENTILE` and `PERCENTILE.INC`, Python `'inclusive'`)
- Def. 8: `alpha=1/3`, `beta=1/3`
- Def. 9: `alpha=3/8`, `beta=3/8`

If `itr` is an `AbstractArray`, `weights` can be specified to compute weighted quantiles.
Weights must not be negative and must have the same length as the data.
With [`FrequencyWeights`](@ref), the function returns the same result as
`quantile` for a vector with repeated values. Weights must be integers.
With non `FrequencyWeights`,  denote ``N`` the length of the vector, ``w`` the vector of weights,
``h = p (\\sum_{i<= N} w_i - w_1) + w_1`` the cumulative weight corresponding to the
probability ``p`` and ``S_k = \\sum_{i<=k} w_i`` the cumulative weight for each
observation, define ``v_{k+1}`` the smallest element of `v` such that ``S_{k+1}``
is strictly superior to ``h``. The weighted ``p`` quantile is given by ``v_k + \\gamma (v_{k+1} - v_k)``
with  ``\\gamma = (h - S_k)/(S_{k+1} - S_k)``. In particular, when all weights are equal,
the function returns the same result as the unweighted `quantile`.

!!! note
    An `ArgumentError` is thrown if `v` contains `NaN` or [`missing`](@ref) values.
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    quantiles of non-missing values.

# References
- Hyndman, R.J and Fan, Y. (1996) "Sample Quantiles in Statistical Packages",
  *The American Statistician*, Vol. 50, No. 4, pp. 361-365

- [Quantile on Wikipedia](https://en.m.wikipedia.org/wiki/Quantile) details the different quantile definitions

# Examples
```jldoctest
julia> using Statistics

julia> quantile(0:20, 0.5)
10.0

julia> quantile(0:20, [0.1, 0.5, 0.9])
3-element Vector{Float64}:
  2.0
 10.0
 18.000000000000004

julia> quantile(skipmissing([1, 10, missing]), 0.5)
5.5
```
"""
quantile(itr, p; sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha,
         weights::Union{AbstractArray,Nothing}=nothing) =
    _quantile(itr, p, sorted, alpha, beta, weights)

_quantile(itr, p, sorted::Bool, alpha::Real, beta::Real, weights::Nothing) =
    quantile!(collect(itr), p, sorted=sorted, alpha=alpha, beta=beta)

_quantile(itr::AbstractArray, p, sorted::Bool, weights::Nothing) =
    quantile!(sorted ? itr : Base.copymutable(itr), p; sorted=sorted,
              alpha=alpha, beta=beta)

"""
    quantile(x, n::Integer)

Return the n-quantiles of collection `x`, i.e. the values which
partition `v` into `n` subsets of nearly equal size.
Equivalent to `quantile(x, [0:n]/n)`. For example, `quantile(x, 5)`
returns a vector of quantiles, respectively at `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
"""
quantile(x, n::Integer) = quantile(x, (0:n)/n)

"""
    percentile(x, p)

Return the `p`th percentile of a collection `x`, i.e. `quantile(x, p / 100)`.
"""
percentile(x, p) = quantile(x, p * 0.01)

##### SparseArrays optimizations #####

function cov(X::SparseMatrixCSC; dims::Int=1, corrected::Bool=true)
    vardim = dims
    a, b = size(X)
    n, p = vardim == 1 ? (a, b) : (b, a)

    # The covariance can be decomposed into two terms
    # 1/(n - 1) ∑ (x_i - x̄)*(x_i - x̄)' = 1/(n - 1) (∑ x_i*x_i' - n*x̄*x̄')
    # which can be evaluated via a sparse matrix-matrix product

    # Compute ∑ x_i*x_i' = X'X using sparse matrix-matrix product
    out = Matrix(unscaled_covzm(X, vardim))

    # Compute x̄
    x̄ᵀ = mean(X, dims=vardim)

    # Subtract n*x̄*x̄' from X'X
    @inbounds for j in 1:p, i in 1:p
        out[i,j] -= x̄ᵀ[i] * x̄ᵀ[j]' * n
    end

    # scale with the sample size n or the corrected sample size n - 1
    return rmul!(out, inv(n - corrected))
end

# This is the function that does the reduction underlying var/std
function centralize_sumabs2!(R::AbstractArray{S}, A::SparseMatrixCSC{Tv,Ti}, means::AbstractArray,
                             w::Nothing) where {S,Tv,Ti}
    require_one_based_indexing(R, A, means)
    lsiz = Base.check_reducedims(R,A)
    for i in 1:max(ndims(R), ndims(means))
        if axes(means, i) != axes(R, i)
            throw(DimensionMismatch("dimension $i of `mean` should have indices $(axes(R, i)), but got $(axes(means, i))"))
        end
    end
    isempty(R) || fill!(R, zero(S))
    isempty(A) && return R

    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    m = size(A, 1)
    n = size(A, 2)

    if size(R, 1) == size(R, 2) == 1
        # Reduction along both columns and rows
        R[1, 1] = centralize_sumabs2(A, means[1])
    elseif size(R, 1) == 1
        # Reduction along rows
        @inbounds for col = 1:n
            mu = means[col]
            r = convert(S, (m-colptr[col+1]+colptr[col])*abs2(mu))
            @simd for j = colptr[col]:colptr[col+1]-1
                r += abs2(nzval[j] - mu)
            end
            R[1, col] = r
        end
    elseif size(R, 2) == 1
        # Reduction along columns
        rownz = fill(convert(Ti, n), m)
        @inbounds for col = 1:n
            @simd for j = colptr[col]:colptr[col+1]-1
                row = rowval[j]
                R[row, 1] += abs2(nzval[j] - means[row])
                rownz[row] -= 1
            end
        end
        for i = 1:m
            R[i, 1] += rownz[i]*abs2(means[i])
        end
    else
        # Reduction along a dimension > 2
        @inbounds for col = 1:n
            lastrow = 0
            @simd for j = colptr[col]:colptr[col+1]-1
                row = rowval[j]
                for i = lastrow+1:row-1
                    R[i, col] = abs2(means[i, col])
                end
                R[row, col] = abs2(nzval[j] - means[row, col])
                lastrow = row
            end
            for i = lastrow+1:m
                R[i, col] = abs2(means[i, col])
            end
        end
    end
    return R
end

end # module
