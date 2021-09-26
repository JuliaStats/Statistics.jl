##### Weight vector #####

"""
    AbstractWeights <: AbstractVector

The abstract supertype of all vectors of statistical weights.

Object of this type behave like other `AbstractVector`s, but
they store the sum of their values internally for efficiency.
Concrete `AbstractWeights` type indicates what correction
has to be applied when computing statistics which depend on the
meaning of weights.

!!! compat "Julia 1.3"
    This type requires at least Julia 1.3.
"""
abstract type AbstractWeights{S<:Real, T<:Real, V<:AbstractVector{T}} <: AbstractVector{T} end

"""
    @weights name

Generates a new generic weight type with specified `name`, which subtypes `AbstractWeights`
and stores the `values` (`V<:RealVector`) and `sum` (`S<:Real`).
"""
macro weights(name)
    return quote
        mutable struct $name{S<:Real, T<:Real, V<:AbstractVector{T}} <: AbstractWeights{S, T, V}
            values::V
            sum::S
        end
        $(esc(name))(values::AbstractVector{<:Real}) = $(esc(name))(values, sum(values))
    end
end

Base.length(wv::AbstractWeights) = length(wv.values)
Base.sum(wv::AbstractWeights) = wv.sum
Base.isempty(wv::AbstractWeights) = isempty(wv.values)
Base.size(wv::AbstractWeights) = size(wv.values)

Base.convert(::Type{Vector}, wv::AbstractWeights) = convert(Vector, wv.values)

Base.@propagate_inbounds function Base.getindex(wv::AbstractWeights, i::Integer)
    @boundscheck checkbounds(wv, i)
    @inbounds wv.values[i]
end

Base.@propagate_inbounds function Base.getindex(wv::W, i::AbstractArray) where W <: AbstractWeights
    @boundscheck checkbounds(wv, i)
    @inbounds v = wv.values[i]
    W(v, sum(v))
end

Base.getindex(wv::W, ::Colon) where {W <: AbstractWeights} = W(copy(wv.values), sum(wv))

Base.@propagate_inbounds function Base.setindex!(wv::AbstractWeights, v::Real, i::Int)
    s = v - wv[i]
    wv.values[i] = v
    wv.sum += s
    v
end

"""
    varcorrection(n::Integer, corrected=false)

Compute a bias correction factor for calculating `var`, `std` and `cov` with
`n` observations. Returns ``\\frac{1}{n - 1}`` when `corrected=true`
(i.e. [Bessel's correction](https://en.wikipedia.org/wiki/Bessel's_correction)),
otherwise returns ``\\frac{1}{n}`` (i.e. no correction).
"""
@inline varcorrection(n::Integer, corrected::Bool=false) = 1 / (n - Int(corrected))

@weights Weights

@doc """
    Weights(vs, wsum=sum(vs))

Construct a `Weights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.

The `Weights` type describes a generic weights vector which does not support
all operations possible for [`FrequencyWeights`](@ref), [`AnalyticWeights`](@ref)
and [`ProbabilityWeights`](@ref).

!!! compat "Julia 1.3"
    This type requires at least Julia 1.3.
""" Weights

"""
    weights(vs)

Construct a `Weights` vector from array `vs`.
See the documentation for [`Weights`](@ref) for more details.
"""
weights(vs::RealVector) = Weights(vs)
weights(vs::RealArray) = Weights(vec(vs))

"""
    varcorrection(w::Weights, corrected=false)

Returns ``\\frac{1}{\\sum w}`` when `corrected=false` and throws an `ArgumentError`
if `corrected=true`.
"""
@inline function varcorrection(w::Weights, corrected::Bool=false)
    corrected && throw(ArgumentError("Weights type does not support bias correction: " *
                                     "use FrequencyWeights, AnalyticWeights or ProbabilityWeights if applicable."))
    1 / w.sum
end

@weights AnalyticWeights

@doc """
    AnalyticWeights(vs, wsum=sum(vs))

Construct an `AnalyticWeights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.

Analytic weights describe a non-random relative importance (usually between 0 and 1)
for each observation. These weights may also be referred to as reliability weights,
precision weights or inverse variance weights. These are typically used when the observations
being weighted are aggregate values (e.g., averages) with differing variances.

!!! compat "Julia 1.3"
    This type requires at least Julia 1.3.
""" AnalyticWeights

"""
    aweights(vs)

Construct an `AnalyticWeights` vector from array `vs`.
See the documentation for [`AnalyticWeights`](@ref) for more details.

!!! compat "Julia 1.3"
    This function requires at least Julia 1.3.
"""
aweights(vs::RealVector) = AnalyticWeights(vs)
aweights(vs::RealArray) = AnalyticWeights(vec(vs))

"""
    varcorrection(w::AnalyticWeights, corrected=false)

* `corrected=true`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* `corrected=false`: ``\\frac{1}{\\sum w}``
"""
@inline function varcorrection(w::AnalyticWeights, corrected::Bool=false)
    s = w.sum

    if corrected
        sum_sn = sum(x -> (x / s) ^ 2, w)
        1 / (s * (1 - sum_sn))
    else
        1 / s
    end
end

@weights FrequencyWeights

@doc """
    FrequencyWeights(vs, wsum=sum(vs))

Construct a `FrequencyWeights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.

Frequency weights describe the number of times (or frequency) each observation
was observed. These weights may also be referred to as case weights or repeat weights.

!!! compat "Julia 1.3"
    This type requires at least Julia 1.3.
""" FrequencyWeights

"""
    fweights(vs)

Construct a `FrequencyWeights` vector from a given array.
See the documentation for [`FrequencyWeights`](@ref) for more details.

!!! compat "Julia 1.3"
    This function requires at least Julia 1.3.
"""
fweights(vs::RealVector) = FrequencyWeights(vs)
fweights(vs::RealArray) = FrequencyWeights(vec(vs))

"""
    varcorrection(w::FrequencyWeights, corrected=false)

* `corrected=true`: ``\\frac{1}{\\sum{w} - 1}``
* `corrected=false`: ``\\frac{1}{\\sum w}``
"""
@inline function varcorrection(w::FrequencyWeights, corrected::Bool=false)
    s = w.sum

    if corrected
        1 / (s - 1)
    else
        1 / s
    end
end

@weights ProbabilityWeights

@doc """
    ProbabilityWeights(vs, wsum=sum(vs))

Construct a `ProbabilityWeights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.

Probability weights represent the inverse of the sampling probability for each observation,
providing a correction mechanism for under- or over-sampling certain population groups.
These weights may also be referred to as sampling weights.

!!! compat "Julia 1.3"
    This type requires at least Julia 1.3.
""" ProbabilityWeights

"""
    pweights(vs)

Construct a `ProbabilityWeights` vector from a given array.
See the documentation for [`ProbabilityWeights`](@ref) for more details.

!!! compat "Julia 1.3"
    This function requires at least Julia 1.3.
"""
pweights(vs::RealVector) = ProbabilityWeights(vs)
pweights(vs::RealArray) = ProbabilityWeights(vec(vs))

"""
    varcorrection(w::ProbabilityWeights, corrected=false)

* `corrected=true`: ``\\frac{n}{(n - 1) \\sum w}``, where ``n`` equals `count(!iszero, w)`
* `corrected=false`: ``\\frac{1}{\\sum w}``
"""
@inline function varcorrection(w::ProbabilityWeights, corrected::Bool=false)
    s = w.sum

    if corrected
        n = count(!iszero, w)
        n / (s * (n - 1))
    else
        1 / s
    end
end

"""
    eweights(t::AbstractVector{<:Integer}, λ::Real)
    eweights(t::AbstractVector{T}, r::StepRange{T}, λ::Real) where T
    eweights(n::Integer, λ::Real)

Construct a [`Weights`](@ref) vector which assigns exponentially decreasing weights to past
observations, which in this case corresponds to larger integer values `i` in `t`.
If an integer `n` is provided, weights are generated for values from 1 to `n`
(equivalent to `t = 1:n`).

For each element `i` in `t` the weight value is computed as:

``λ (1 - λ)^{1 - i}``

# Arguments

- `t::AbstractVector`: temporal indices or timestamps
- `r::StepRange`: a larger range to use when constructing weights from a subset of timestamps
- `n::Integer`: if provided instead of `t`, temporal indices are taken to be `1:n`
- `λ::Real`: a smoothing factor or rate parameter such that ``0 < λ ≤ 1``.
  As this value approaches 0, the resulting weights will be almost equal,
  while values closer to 1 will put greater weight on the tail elements of the vector.

# Examples
```julia-repl
julia> eweights(1:10, 0.3)
10-element Weights{Float64,Float64,Array{Float64,1}}:
 0.3
 0.42857142857142855
 0.6122448979591837
 0.8746355685131197
 1.249479383590171
 1.7849705479859588
 2.549957925694227
 3.642797036706039
 5.203995766722913
 7.434279666747019
```
"""
function eweights(t::AbstractVector{T}, λ::Real) where T<:Integer
    0 < λ <= 1 || throw(ArgumentError("Smoothing factor must be between 0 and 1"))

    w0 = map(t) do i
        i > 0 || throw(ArgumentError("Time indices must be non-zero positive integers"))
        λ * (1 - λ)^(1 - i)
    end

    s = sum(w0)
    Weights(w0, s)
end

eweights(n::Integer, λ::Real) = eweights(1:n, λ)
eweights(t::AbstractVector, r::AbstractRange, λ::Real) =
    eweights(something.(indexin(t, r)), λ)

# NOTE: no variance correction is implemented for exponential weights

struct UnitWeights{T<:Real} <: AbstractWeights{Int, T, V where V<:Vector{T}}
    len::Int
end

@doc """
    UnitWeights{T}(s)

Construct a `UnitWeights` vector with length `s` and weight elements of type `T`.
All weight elements are identically one.
""" UnitWeights

Base.sum(wv::UnitWeights{T}) where T = convert(T, length(wv))
Base.isempty(wv::UnitWeights) = iszero(wv.len)
Base.length(wv::UnitWeights) = wv.len
Base.size(wv::UnitWeights) = tuple(length(wv))

Base.convert(::Type{Vector}, wv::UnitWeights{T}) where {T} = ones(T, length(wv))

Base.@propagate_inbounds function Base.getindex(wv::UnitWeights{T}, i::Integer) where T
    @boundscheck checkbounds(wv, i)
    one(T)
end

Base.@propagate_inbounds function Base.getindex(wv::UnitWeights{T}, i::AbstractArray{<:Int}) where T
    @boundscheck checkbounds(wv, i)
    UnitWeights{T}(length(i))
end

function Base.getindex(wv::UnitWeights{T}, i::AbstractArray{Bool}) where T
   length(wv) == length(i) || throw(DimensionMismatch())
   UnitWeights{T}(count(i))
end

Base.getindex(wv::UnitWeights{T}, ::Colon) where {T} = UnitWeights{T}(wv.len)

"""
    uweights(s::Integer)
    uweights(::Type{T}, s::Integer) where T<:Real

Construct a `UnitWeights` vector with length `s` and weight elements of type `T`.
All weight elements are identically one.

# Examples
```julia-repl
julia> uweights(3)
3-element UnitWeights{Int64}:
 1
 1
 1
 
julia> uweights(Float64, 3)
3-element UnitWeights{Float64}:
 1.0
 1.0
 1.0
```
"""
uweights(s::Int)                            = UnitWeights{Int}(s)
uweights(::Type{T}, s::Int) where {T<:Real} = UnitWeights{T}(s)

"""
    varcorrection(w::UnitWeights, corrected=false)

* `corrected=true`: ``\\frac{n}{n - 1}``, where ``n`` is the length of the weight vector
* `corrected=false`: ``\\frac{1}{n}``, where ``n`` is the length of the weight vector

This definition is equivalent to the correction applied to unweighted data.
"""
@inline function varcorrection(w::UnitWeights, corrected::Bool=false)
    corrected ? (1 / (w.len - 1)) : (1 / w.len)
end

#### Equality tests #####

for w in (AnalyticWeights, FrequencyWeights, ProbabilityWeights, Weights)
    @eval begin
        Base.isequal(x::$w, y::$w) = isequal(x.sum, y.sum) && isequal(x.values, y.values)
        Base.:(==)(x::$w, y::$w)   = (x.sum == y.sum) && (x.values == y.values)
    end
end

Base.isequal(x::UnitWeights, y::UnitWeights) = isequal(x.len, y.len)
Base.:(==)(x::UnitWeights, y::UnitWeights)   = (x.len == y.len)

Base.isequal(x::AbstractWeights, y::AbstractWeights) = false
Base.:(==)(x::AbstractWeights, y::AbstractWeights)   = false

##### Weighted sum #####

## weighted sum over vectors

"""
    wsum(v; weights::AbstractVector[, dims])

Compute the weighted sum of an array `v` with weights `weights`,
optionally over the dimension `dim`.
"""
wsum(A::AbstractArray; dims=:, weights::AbstractArray) =
    _wsum(A, dims, weights)

# Optimized method for weighted sum with BlasReal
# dot cannot be used for other types as it uses + rather than add_sum for accumulation,
# and therefore does not return the correct type
_wsum(A::AbstractArray{<:BlasReal}, dims::Colon, w::AbstractArray{<:BlasReal}) =
    dot(vec(A), vec(w))

_wsum(A::AbstractArray, dims, w::AbstractArray{<:Real}) =
    _wsum!(Base.reducedim_init(t -> t*zero(eltype(w)), Base.add_sum, A, dims), A, w)

function _wsum(A::AbstractArray, dims::Colon, w::AbstractArray{<:Real})
    sw = size(w)
    sA = size(A)
    if sw != sA
        throw(DimensionMismatch("weights must have the same dimension as data (got $sw and $sA)."))
    end
    s0 = zero(eltype(A)) * zero(eltype(w))
    s = Base.add_sum(s0, s0)
    @inbounds @simd for i in eachindex(A, w)
        s = Base.add_sum(s, A[i] * w[i])
    end
    s
end

wsum!(r::AbstractArray, A::AbstractArray;
      init::Bool=true, weights::AbstractArray) =
    _wsum!(r, A, weights; init=init)

## wsum along dimension
#
#  Brief explanation of the algorithm:
#  ------------------------------------
#
#  1. _wsum! provides the core implementation, which assumes that
#     the dimensions of all input arguments are consistent, and no
#     dimension checking is performed therein.
#
#     wsum and wsum! perform argument checking and call _wsum!
#     internally.
#
#  2. _wsum! adopt a Cartesian based implementation for general
#     sub types of AbstractArray. Particularly, a faster routine
#     that keeps a local accumulator will be used when dim = 1.
#
#     The internal function that implements this is _wsum_general!
#
#  3. _wsum! is specialized for following cases:
#     (a) A is a vector: we invoke the vector version wsum above.
#         The internal function that implements this is _wsum1!
#
#     (b) A is a dense matrix with eltype <: BlasReal: we call gemv!
#         The internal function that implements this is _wsum2_blas!
#
#     (c) A is a contiguous array with eltype <: BlasReal:
#         dim == 1: treat A like a matrix of size (d1, d2 x ... x dN)
#         dim == N: treat A like a matrix of size (d1 x ... x d(N-1), dN)
#         otherwise: decompose A into multiple pages, and apply _wsum2_blas!
#         for each
#         The internal function that implements this is _wsumN!
#
#     (d) A is a general dense array with eltype <: BlasReal:
#         dim <= 2: delegate to (a) and (b)
#         otherwise, decompose A into multiple pages
#         The internal function that implements this is _wsumN!

function _wsum1!(R::AbstractArray, A::AbstractVector, w::AbstractVector, init::Bool)
    r = _wsum(A, :, w)
    if init
        R[1] = r
    else
        R[1] += r
    end
    return R
end

function _wsum2_blas!(R::StridedVector{T}, A::StridedMatrix{T}, w::StridedVector{T}, dim::Int, init::Bool) where T<:BlasReal
    beta = ifelse(init, zero(T), one(T))
    trans = dim == 1 ? 'T' : 'N'
    BLAS.gemv!(trans, one(T), A, w, beta, R)
    return R
end

function _wsumN!(R::StridedArray{T}, A::StridedArray{T,N}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal,N}
    if dim == 1
        m = size(A, 1)
        n = div(length(A), m)
        _wsum2_blas!(view(R,:), reshape(A, (m, n)), w, 1, init)
    elseif dim == N
        n = size(A, N)
        m = div(length(A), n)
        _wsum2_blas!(view(R,:), reshape(A, (m, n)), w, 2, init)
    else # 1 < dim < N
        m = 1
        for i = 1:dim-1
            m *= size(A, i)
        end
        n = size(A, dim)
        k = 1
        for i = dim+1:N
            k *= size(A, i)
        end
        Av = reshape(A, (m, n, k))
        Rv = reshape(R, (m, k))
        for i = 1:k
            _wsum2_blas!(view(Rv,:,i), view(Av,:,:,i), w, 2, init)
        end
    end
    return R
end

function _wsumN!(R::StridedArray{T}, A::DenseArray{T,N}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal,N}
    @assert N >= 3
    if dim <= 2
        m = size(A, 1)
        n = size(A, 2)
        npages = 1
        for i = 3:N
            npages *= size(A, i)
        end
        rlen = ifelse(dim == 1, n, m)
        Rv = reshape(R, (rlen, npages))
        for i = 1:npages
            _wsum2_blas!(view(Rv,:,i), view(A,:,:,i), w, dim, init)
        end
    else
        _wsum_general!(R, A, w, dim, init)
    end
    return R
end

## general Cartesian-based weighted sum across dimensions

function _wsum_general!(R::AbstractArray{S}, A::AbstractArray, w::AbstractVector, dim::Int, init::Bool) where {S}
    # following the implementation of _mapreducedim!
    lsiz = Base.check_reducedims(R,A)
    !isempty(R) && init && fill!(R, zero(S))
    isempty(A) && return R

    indsAt, indsRt = Base.safe_tail(axes(A)), Base.safe_tail(axes(R)) # handle d=1 manually
    keep, Idefault = Broadcast.shapeindexer(indsRt)
    if Base.reducedim1(R, A)
        i1 = first(Base.axes1(R))
        for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            r = R[i1,IR]
            @inbounds @simd for i in axes(A, 1)
                r += A[i,IA] * w[dim > 1 ? IA[dim-1] : i]
            end
            R[i1,IR] = r
        end
    else
        for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            @inbounds @simd for i in axes(A, 1)
                R[i,IR] += A[i,IA] * w[dim > 1 ? IA[dim-1] : i]
            end
        end
    end
    return R
end

# N = 1
_wsum!(R::StridedArray{T}, A::DenseArray{T,1}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal} =
    _wsum1!(R, A, w, init)

_wsum!(R::AbstractArray, A::AbstractVector, w::AbstractVector, dim::Int, init::Bool) =
    _wsum1!(R, A, w, init)

# N = 2
_wsum!(R::StridedArray{T}, A::DenseArray{T,2}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal} =
    (_wsum2_blas!(view(R,:), A, w, dim, init); R)

# N >= 3
_wsum!(R::StridedArray{T}, A::DenseArray{T,N}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal,N} =
    _wsumN!(R, A, w, dim, init)

_wsum!(R::AbstractArray, A::AbstractArray, w::AbstractVector, dim::Int, init::Bool) =
    _wsum_general!(R, A, w, dim, init)

function _wsum!(R::AbstractArray, A::AbstractArray{T,N}, w::AbstractArray; init::Bool=true) where {T,N}
    w isa AbstractVector || throw(ArgumentError("Only vector `weights` are supported"))

    Base.check_reducedims(R,A)
    reddims = size(R) .!= size(A)
    dim = something(findfirst(reddims), ndims(R)+1)
    if dim > N
        dim1 = findfirst(==(1), size(A))
        if dim1 !== nothing
            dim = dim1
        end
    end
    if findnext(reddims, dim+1) !== nothing
        throw(ArgumentError("reducing over more than one dimension is not supported with weights"))
    end
    lw = length(w)
    ldim = size(A, dim)
    if lw != ldim
        throw(DimensionMismatch("weights must have the same length as the dimension " *
                                "over which reduction is performed (got $lw and $ldim)."))
    end
    _wsum!(R, A, w, dim, init)
end

function _wsum(A::AbstractArray, dims, w::UnitWeights)
    size(A, dims) != length(w) && throw(DimensionMismatch("Inconsistent array dimension."))
    return sum(A, dims=dims)
end

function _wsum(A::AbstractArray, dims::Colon, w::UnitWeights)
    length(A) != length(w) && throw(DimensionMismatch("Inconsistent array dimension."))
    return sum(A)
end

# To fix ambiguity
function _wsum(A::AbstractArray{<:BlasReal}, dims::Colon, w::UnitWeights)
    length(A) != length(w) && throw(DimensionMismatch("Inconsistent array dimension."))
    return sum(A)
end

##### Weighted means #####

# Note: weighted mean currently does not use _mean_promote to avoid overflow
# contrary non-weighted method

_mean!(R::AbstractArray, A::AbstractArray, w::AbstractArray) =
    rmul!(wsum!(R, A, weights=w), inv(sum(w)))

_mean(::typeof(identity), A::AbstractArray, dims::Colon, w::AbstractArray) =
    wsum(A, weights=w) / sum(w)

_mean(::typeof(identity), A::AbstractArray, dims, w::AbstractArray) =
    _mean!(Base.reducedim_init(t -> (t*zero(eltype(w)))/2, Base.add_sum, A, dims), A, w)

function _mean(::typeof(identity), A::AbstractArray, dims, w::UnitWeights)
    size(A, dims) != length(w) && throw(DimensionMismatch("Inconsistent array dimension."))
    return mean(A, dims=dims)
end

function _mean(::typeof(identity), A::AbstractArray, dims::Colon, w::UnitWeights)
    length(A) != length(w) && throw(DimensionMismatch("Inconsistent array dimension."))
    return mean(A)
end

##### Weighted quantile #####

function _quantile(v::AbstractArray{V}, p, sorted::Bool, alpha::Real, beta::Real,
                   w::AbstractArray{W}) where {V,W}
    # checks
    alpha == beta == 1 || throw(ArgumentError("only alpha == beta == 1 is supported " *
                                              "when weights are provided"))
    isempty(v) && throw(ArgumentError("quantile of an empty array is undefined"))
    isempty(p) && throw(ArgumentError("empty quantile array"))
    all(x -> 0 <= x <= 1, p) || throw(ArgumentError("input probability out of [0,1] range"))

    wsum = sum(w)
    wsum == 0 && throw(ArgumentError("weight vector cannot sum to zero"))
    size(v) == size(w) || throw(ArgumentError("weights must have the same dimension as data " *
                                              "(got $(size(v)) and $(size(w)))"))
    for x in w
        isnan(x) && throw(ArgumentError("weight vector cannot contain NaN entries"))
        x < 0 && throw(ArgumentError("weight vector cannot contain negative entries"))
    end

    isa(w, FrequencyWeights) && !(eltype(w) <: Integer) && any(!isinteger, w) &&
        throw(ArgumentError("The values of the vector of `FrequencyWeights` must be numerically" *
                            "equal to integers. Use `ProbabilityWeights` or `AnalyticWeights` instead."))

    # remove zeros weights and sort
    nz = .!iszero.(w)
    vw = sort!(collect(zip(view(v, nz), view(w, nz))))
    N = length(vw)

    # prepare percentiles
    ppermute = sortperm(p)
    p = p[ppermute]

    # prepare out vector
    out = Vector{typeof(zero(V)/1)}(undef, length(p))
    fill!(out, vw[end][1])

    @inbounds for x in v
        isnan(x) && return fill!(out, x)
    end

    # loop on quantiles
    Sk, Skold = zero(W), zero(W)
    vk, vkold = zero(V), zero(V)
    k = 0

    w1 = vw[1][2]
    for i in 1:length(p)
        if isa(w, FrequencyWeights)
            h = p[i] * (wsum - 1) + 1
        else
            h = p[i] * (wsum - w1) + w1
        end
        while Sk <= h
            k += 1
            if k > N
               # out was initialized with maximum v
               return out
            end
            Skold, vkold = Sk, vk
            vk, wk = vw[k]
            Sk += wk
        end
        if isa(w, FrequencyWeights)
            out[ppermute[i]] = vkold + min(h - Skold, 1) * (vk - vkold)
        else
            out[ppermute[i]] = vkold + (h - Skold) / (Sk - Skold) * (vk - vkold)
        end
    end
    return out
end

function _quantile(v::AbstractArray, p, sorted::Bool,
                   alpha::Real, beta::Real, w::UnitWeights)
    length(v) != length(w) && throw(DimensionMismatch("Inconsistent array dimension."))
    return quantile(v, p)
end

function _quantile(v::AbstractArray, p::Real, sorted::Bool,
                   alpha::Real, beta::Real, w::UnitWeights)
    length(v) != length(w) && throw(DimensionMismatch("Inconsistent array dimension."))
    return quantile(v, p)
end

_quantile(v::AbstractArray, p::Real, sorted::Bool, alpha::Real, beta::Real,
          w::AbstractArray) =
    _quantile(v, [p], sorted, alpha, beta, w)[1]

_quantile(itr, p, sorted::Bool, alpha::Real, beta::Real, weights) =
    throw(ArgumentError("weights are only supported with AbstractArrays inputs"))

##### Weighted median #####

_median(v::AbstractArray, dims::Colon, w::AbstractArray) = quantile(v, 0.5, weights=w)

_median(A::AbstractArray, dims, w::AbstractArray) =
    throw(ArgumentError("weights and dims cannot be specified at the same time"))