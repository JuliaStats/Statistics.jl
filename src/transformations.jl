### Normalizations

abstract type AbstractNormalization end

# apply the normalization
"""
    normalize!(t::AbstractNormalization, x)

Apply normalization `t` to vector or matrix `x` in place.
"""
LinearAlgebra.normalize!(t::AbstractNormalization, x::AbstractMatrix{<:Real}) =
    normalize!(x, t, x)
LinearAlgebra.normalize!(t::AbstractNormalization, x::AbstractVector{<:Real}) =
    (normalize!(t, reshape(x, :, 1)); x)

"""
    normalize(t::AbstractNormalization, x)

Return a standardized copy of vector or matrix `x` using normalization `t`.
"""
LinearAlgebra.normalize(t::AbstractNormalization, x::AbstractMatrix{<:Real}) =
    normalize!(similar(x), t, x)
LinearAlgebra.normalize(t::AbstractNormalization, x::AbstractVector{<:Real}) =
    vec(normalize(t, reshape(x, :, 1)))

# unnormalize the original data from normalized values
"""
    unnormalize(t::AbstractNormalization, y)

Perform an in-place unnormalizeion into an original data scale from
vector or matrix `y` transformed using normalization `t`.
"""
unnormalize!(t::AbstractNormalization, y::AbstractMatrix{<:Real}) =
    unnormalize!(y, t, y)
unnormalize!(t::AbstractNormalization, y::AbstractVector{<:Real}) =
    (unnormalize!(t, reshape(y, :, 1)); y)

"""
    unnormalize(t::AbstractNormalization, y)

Return a unnormalizeion of an originally scaled data from a vector
or matrix `y` transformed using normalization `t`.
"""
unnormalize(t::AbstractNormalization, y::AbstractMatrix{<:Real}) =
    unnormalize!(similar(y), t, y)
unnormalize(t::AbstractNormalization, y::AbstractVector{<:Real}) =
    vec(unnormalize(t, reshape(y, :, 1)))

"""
Standardization (Z-score normalization)
"""
struct ZScoreNormalization{T<:Real, U<:AbstractVector{T}} <: AbstractNormalization
    len::Int
    dims::Int
    mean::U
    scale::U

    function ZScoreNormalization(l::Int, dims::Int, m::U, s::U) where {T<:Real, U<:AbstractVector{T}}
        lenm = length(m)
        lens = length(s)
        lenm == l || lenm == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        lens == l || lens == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new{T, U}(l, dims, m, s)
    end
end

"""
    fit(ZScoreNormalization, X; dims, center=true, scale=true)

Fit standardization parameters to vector or matrix `X`
and return a `ZScoreNormalization` object.

# Keyword arguments

* `dims`: if `1` fit standardization parameters in column-wise fashion;
  if `2` fit in row-wise fashion.

* `center`: if `true` (the default) center data so that its mean is zero.

* `scale`: if `true` (the default) scale the data so that its variance is equal to one.

# Examples

```jldoctest
julia> using Statistics

julia> X = [0.0 -0.5 0.5; 0.0 1.0 2.0]
2×3 Matrix{Float64}:
 0.0  -0.5  0.5
 0.0   1.0  2.0

julia> dt = fit(ZScoreNormalization, X, dims=2)
ZScoreNormalization{Float64, Vector{Float64}}(2, 2, [0.0, 1.0], [0.5, 1.0])

julia> normalize(dt, X)
2×3 Matrix{Float64}:
  0.0  -1.0  1.0
 -1.0   0.0  1.0
```
"""
function fit(::Type{ZScoreNormalization}, X::AbstractMatrix{<:Real};
             dims::Integer, center::Bool=true, scale::Bool=true)
    if dims == 1
        n, l = size(X)
        n >= 2 || error("X must contain at least two rows.")
    elseif dims == 2
        l, n = size(X)
        n >= 2 || error("X must contain at least two columns.")
    else
        throw(DomainError(dims, "fit only accept dims to be 1 or 2."))
    end
    m = mean(X, dims=dims)
    s = std(X, mean=m, dims=dims)
    return ZScoreNormalization(l, dims, (center ? vec(m) : similar(m, 0)),
                                    (scale ? vec(s) : similar(s, 0)))
end

function fit(::Type{ZScoreNormalization}, X::AbstractVector{<:Real};
             dims::Integer=1, center::Bool=true, scale::Bool=true)
    if dims != 1
        throw(DomainError(dims, "fit only accepts dims=1 over a vector. Try fit(t, x, dims=1)."))
    end

    return fit(ZScoreNormalization, reshape(X, :, 1); dims=dims, center=center, scale=scale)
end

function LinearAlgebra.normalize!(y::AbstractMatrix{<:Real},
                                  t::ZScoreNormalization,
                                  x::AbstractMatrix{<:Real})
    if t.dims == 1
        l = t.len
        size(x,2) == size(y,2) == l || throw(DimensionMismatch("Inconsistent dimensions."))
        n = size(y,1)
        size(x,1) == n || throw(DimensionMismatch("Inconsistent dimensions."))

        m = t.mean
        s = t.scale

        if isempty(m)
            if isempty(s)
                if x !== y
                    copyto!(y, x)
                end
            else
                broadcast!(/, y, x, s')
            end
        else
            if isempty(s)
                broadcast!(-, y, x, m')
            else
                broadcast!((x,m,s)->(x-m)/s, y, x, m', s')
            end
        end
    elseif t.dims == 2
        t_ = ZScoreNormalization(t.len, 1, t.mean, t.scale)
        normalize!(y', t_, x')
    end
    return y
end

function unnormalize!(x::AbstractMatrix{<:Real}, t::ZScoreNormalization, y::AbstractMatrix{<:Real})
    if t.dims == 1
        l = t.len
        size(x,2) == size(y,2) == l || throw(DimensionMismatch("Inconsistent dimensions."))
        n = size(y,1)
        size(x,1) == n || throw(DimensionMismatch("Inconsistent dimensions."))

        m = t.mean
        s = t.scale

        if isempty(m)
            if isempty(s)
                if y !== x
                    copyto!(x, y)
                end
            else
                broadcast!(*, x, y, s')
            end
        else
            if isempty(s)
                broadcast!(+, x, y, m')
            else
                broadcast!((y,m,s)->y*s+m, x, y, m', s')
            end
        end
    elseif t.dims == 2
        t_ = ZScoreNormalization(t.len, 1, t.mean, t.scale)
        unnormalize!(x', t_, y')
    end
    return x
end

"""
Min-max normalization
"""
struct MinMaxNormalization{T<:Real, U<:AbstractVector}  <: AbstractNormalization
    len::Int
    dims::Int
    zero::Bool
    min::U
    scale::U

    function MinMaxNormalization(l::Int, dims::Int, zero::Bool, min::U, max::U) where {T, U<:AbstractVector{T}}
        lenmin = length(min)
        lenmax = length(max)
        lenmin == l || lenmin == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        lenmax == l || lenmax == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new{T, U}(l, dims, zero, min, max)
    end
end

# fit a min-max normalization
"""
    fit(MinMaxNormalization, X; dims, zero=true)

Fit a scaling parameters to vector or matrix `X`
and return a `MinMaxNormalization` object.

# Keyword arguments

* `dims`: if `1` fit standardization parameters in column-wise fashion;
 if `2` fit in row-wise fashion.

* `zero`: if `true` (the default) shift the minimum data to zero.

# Examples

```jldoctest
julia> using Statistics

julia> X = [0.0 -0.5 0.5; 0.0 1.0 2.0]
2×3 Matrix{Float64}:
 0.0  -0.5  0.5
 0.0   1.0  2.0

julia> dt = fit(MinMaxNormalization, X, dims=2)
MinMaxNormalization{Float64, Vector{Float64}}(2, 2, true, [-0.5, 0.0], [1.0, 0.5])

julia> normalize(dt, X)
2×3 Matrix{Float64}:
 0.5  0.0  1.0
 0.0  0.5  1.0
```
"""
function fit(::Type{MinMaxNormalization}, X::AbstractMatrix{<:Real};
             dims::Integer, zero::Bool=true)
    dims ∈ (1, 2) || throw(DomainError(dims, "fit only accept dims to be 1 or 2."))
    tmin, tmax = _compute_extrema(X, dims)
    @. tmax = 1 / (tmax - tmin)
    l = length(tmin)
    return MinMaxNormalization(l, dims, zero, tmin, tmax)
end

function _compute_extrema(X::AbstractMatrix, dims::Integer)
    dims == 2 && return _compute_extrema(X', 1)
    l = size(X, 2)
    tmin = similar(X, l)
    tmax = similar(X, l)
    for i in 1:l
        @inbounds tmin[i], tmax[i] = extrema(@view(X[:, i]))
    end
    return tmin, tmax
end

function fit(::Type{MinMaxNormalization}, X::AbstractVector{<:Real};
             dims::Integer=1, zero::Bool=true)
    if dims != 1
        throw(DomainError(dims, "fit only accept dims=1 over a vector. Try fit(t, x, dims=1)."))
    end
    tmin, tmax = extrema(X)
    tmax = 1 / (tmax - tmin)
    return MinMaxNormalization(1, dims, zero, [tmin], [tmax])
end

function LinearAlgebra.normalize!(y::AbstractMatrix{<:Real},
                                  t::MinMaxNormalization,
                                  x::AbstractMatrix{<:Real})
    if t.dims == 1
        l = t.len
        size(x,2) == size(y,2) == l || throw(DimensionMismatch("Inconsistent dimensions."))
        n = size(x,1)
        size(y,1) == n || throw(DimensionMismatch("Inconsistent dimensions."))

        tmin = t.min
        tscale = t.scale

        if t.zero
            broadcast!((x,s,m)->(x-m)*s, y, x, tscale', tmin')
        else
            broadcast!(*, y, x, tscale')
        end
    elseif t.dims == 2
        t_ = MinMaxNormalization(t.len, 1, t.zero, t.min, t.scale)
        normalize!(y', t_, x')
    end
    return y
end

function unnormalize!(x::AbstractMatrix{<:Real}, t::MinMaxNormalization, y::AbstractMatrix{<:Real})
    if t.dims == 1
        l = t.len
        size(x,2) == size(y,2) == l || throw(DimensionMismatch("Inconsistent dimensions."))
        n = size(y,1)
        size(x,1) == n || throw(DimensionMismatch("Inconsistent dimensions."))

        tmin = t.min
        tscale = t.scale

        if t.zero
            broadcast!((y,s,m)->y/s+m, x, y, tscale', tmin')
        else
            broadcast!(/, x, y, tscale')
        end
    elseif t.dims == 2
        t_ = MinMaxNormalization(t.len, 1, t.zero, t.min, t.scale)
        unnormalize!(x', t_, y')
    end
    return x
end

"""
    normalize(DT, X; dims=nothing, kwargs...)

 Return a normalized copy of vector or matrix `X` along dimensions `dims`
 using normalization `DT` which is a subtype of `AbstractNormalization`:

- `ZScoreNormalization`
- `MinMaxNormalization`

# Example

```jldoctest
julia> using Statistics

julia> normalize(ZScoreNormalization, [0.0 -0.5 0.5; 0.0 1.0 2.0], dims=2)
2×3 Matrix{Float64}:
  0.0  -1.0  1.0
 -1.0   0.0  1.0

julia> normalize(MinMaxNormalization, [0.0 -0.5 0.5; 0.0 1.0 2.0], dims=2)
2×3 Matrix{Float64}:
 0.5  0.0  1.0
 0.0  0.5  1.0
```
"""
LinearAlgebra.normalize(::Type{DT}, X::AbstractVecOrMat{<:Real}; kwargs...) where
    {DT <: AbstractNormalization} =
    normalize(fit(DT, X; kwargs...), X)
