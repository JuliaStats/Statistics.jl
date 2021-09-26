###### Weights array #####

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

Generate a new generic weight type with specified `name`, which subtypes `AbstractWeights`
and stores the `values` (`V<:AbstractVector{<:Real}`) and `sum` (`S<:Real`).
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
weights(vs::AbstractVector{<:Real}) = Weights(vs)
weights(vs::AbstractArray{<:Real}) = Weights(vec(vs))

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
aweights(vs::AbstractVector{<:Real}) = AnalyticWeights(vs)
aweights(vs::AbstractArray{<:Real}) = AnalyticWeights(vec(vs))

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
fweights(vs::AbstractVector{<:Real}) = FrequencyWeights(vs)
fweights(vs::AbstractArray{<:Real}) = FrequencyWeights(vec(vs))

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
pweights(vs::AbstractVector{<:Real}) = ProbabilityWeights(vs)
pweights(vs::AbstractArray{<:Real}) = ProbabilityWeights(vec(vs))

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
