# Empirical Estimation of Distributions

## Histograms

```@docs
Histogram
```

Histograms can be fitted to data using the `fit` method.

```@docs
fit(::Type{Histogram}, args...; kwargs...)
```

Additional methods
```@docs
merge!
merge
midpoints
norm
normalize(h::Histogram{T,N}) where {T<:AbstractFloat,N}
normalize(h::Histogram{T,N}, aux_weights::Array{T,N}...) where {T<:AbstractFloat,N}
normalize!(h::Histogram{T,N}, aux_weights::Array{T,N}...) where {T<:AbstractFloat,N}
zero
```

## Empirical Cumulative Distribution Function

```@docs
ecdf
```
