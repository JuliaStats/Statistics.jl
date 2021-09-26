using Base: add_sum, reducedim_init, check_reducedims, safe_tail, reducedim1, axes1
using LinearAlgebra: BlasReal

wsum(A::AbstractArray; dims=:, weights::AbstractArray) =
    _wsum(A, dims, weights)

_wsum(A::AbstractArray, dims, weights::AbstractArray) =
    _wsum!(reducedim_init(t -> t*zero(eltype(weights)), add_sum, A, dims), A, weights)

function _wsum(A::AbstractArray, dims::Colon, w::AbstractArray{<:Real})
    sw = size(w)
    sA = size(A)
    if sw != sA
        throw(DimensionMismatch("weights must have the same dimension as data (got $sw and $sA)."))
    end
    s0 = zero(eltype(A)) * zero(eltype(w))
    s = add_sum(s0, s0)
    @inbounds @simd for i in eachindex(A, w)
        s = add_sum(s, A[i] * w[i])
    end
    s
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

wsum!(r::AbstractArray, A::AbstractArray;
      init::Bool=true, weights::AbstractArray) =
    _wsum!(r, A, weights; init=init)

# Weighted sum over dimensions
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
#         (in LinearAlgebra/src/wsum.jl)
#
#     (c) A is a contiguous array with eltype <: BlasReal:
#         dim == 1: treat A like a matrix of size (d1, d2 x ... x dN)
#         dim == N: treat A like a matrix of size (d1 x ... x d(N-1), dN)
#         otherwise: decompose A into multiple pages, and apply _wsum2_blas!
#         for each
#         The internal function that implements this is _wsumN!
#         (in LinearAlgebra/src/wsum.jl)
#
#     (d) A is a general dense array with eltype <: BlasReal:
#         dim <= 2: delegate to (a) and (b)
#         otherwise, decompose A into multiple pages
#         The internal function that implements this is _wsumN!
#         (in LinearAlgebra/src/wsum.jl)

function _wsum1!(R::AbstractArray, A::AbstractVector, w::AbstractVector, init::Bool)
    r = _wsum(A, :, w)
    if init
        R[1] = r
    else
        R[1] += r
    end
    return R
end

function _wsum_general!(R::AbstractArray{S}, A::AbstractArray, w::AbstractVector,
                        dim::Int, init::Bool) where {S}
    # following the implementation of _mapreducedim!
    lsiz = check_reducedims(R,A)
    !isempty(R) && init && fill!(R, zero(S))
    isempty(A) && return R

    indsAt, indsRt = safe_tail(axes(A)), safe_tail(axes(R)) # handle d=1 manually
    keep, Idefault = Broadcast.shapeindexer(indsRt)
    if reducedim1(R, A)
        i1 = first(axes1(R))
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

_wsum!(R::AbstractArray, A::AbstractVector, w::AbstractVector,
    dim::Int, init::Bool) =
    _wsum1!(R, A, w, init)

_wsum!(R::AbstractArray, A::AbstractArray, w::AbstractVector,
       dim::Int, init::Bool) =
    _wsum_general!(R, A, w, dim, init)

function _wsum!(R::AbstractArray, A::AbstractArray{T,N}, w::AbstractArray;
                init::Bool=true) where {T,N}
    w isa AbstractVector || throw(ArgumentError("Only vector `weights` are supported"))

    check_reducedims(R,A)
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

# Optimized method for weighted sum with BlasReal
# dot cannot be used for other types as it uses + rather than add_sum for accumulation,
# and therefore does not return the correct type
_wsum(A::AbstractArray{<:BlasReal}, dims::Colon, w::AbstractArray{<:BlasReal}) =
    dot(vec(A), vec(w))

# Optimized methods for weighted sum over dimensions with BlasReal
# (generic method is defined in base/reducedim.jl)
#
#  _wsum! is specialized for following cases:
#     (a) A is a dense matrix with eltype <: BlasReal: we call gemv!
#         The internal function that implements this is _wsum2_blas!
#
#     (b) A is a contiguous array with eltype <: BlasReal:
#         dim == 1: treat A like a matrix of size (d1, d2 x ... x dN)
#         dim == N: treat A like a matrix of size (d1 x ... x d(N-1), dN)
#         otherwise: decompose A into multiple pages, and apply _wsum2_blas!
#         for each
#         The internal function that implements this is _wsumN!
#
#     (c) A is a general dense array with eltype <: BlasReal:
#         dim <= 2: delegate to (a) and (b)
#         otherwise, decompose A into multiple pages
#         The internal function that implements this is _wsumN!

function _wsum2_blas!(R::StridedVector{T}, A::StridedMatrix{T}, w::StridedVector{T},
                      dim::Int, init::Bool) where T<:BlasReal
    beta = ifelse(init, zero(T), one(T))
    trans = dim == 1 ? 'T' : 'N'
    BLAS.gemv!(trans, one(T), A, w, beta, R)
    return R
end

function _wsumN!(R::StridedArray{T}, A::StridedArray{T,N}, w::StridedVector{T},
                 dim::Int, init::Bool) where {T<:BlasReal,N}
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

function _wsumN!(R::StridedArray{T}, A::DenseArray{T,N}, w::StridedVector{T},
                 dim::Int, init::Bool) where {T<:BlasReal,N}
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

_wsum!(R::StridedArray{T}, A::DenseMatrix{T}, w::StridedVector{T},
       dim::Int, init::Bool) where {T<:BlasReal} =
    _wsum2_blas!(view(R,:), A, w, dim, init)

_wsum!(R::StridedArray{T}, A::DenseArray{T}, w::StridedVector{T},
       dim::Int, init::Bool) where {T<:BlasReal} =
    _wsumN!(R, A, w, dim, init)

_wsum!(R::StridedVector{T}, A::DenseArray{T}, w::StridedVector{T},
       dim::Int, init::Bool) where {T<:BlasReal} =
    _wsum1!(R, A, w, init)