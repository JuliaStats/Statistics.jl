module StatisticsSparseArraysExt

##### SparseArrays optimizations #####

import SparseArrays: SparseMatrixCSC
using SparseArrays: AbstractSparseMatrix, rowvals, nonzeros, nzrange

using Statistics
import Statistics: cov, centralize_sumabs2!, centralize_sumabs2
using Statistics: unscaled_covzm

using LinearAlgebra
using LinearAlgebra: require_one_based_indexing

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
function centralize_sumabs2!(R::AbstractArray{S}, A::SparseMatrixCSC{Tv,Ti}, means::AbstractArray) where {S,Tv,Ti}
    require_one_based_indexing(R, A, means)
    lsiz = Base.check_reducedims(R,A)
    for i in 1:max(ndims(R), ndims(means))
        if axes(means, i) != axes(R, i)
            throw(DimensionMismatch("dimension $i of `mean` should have indices $(axes(R, i)), but got $(axes(means, i))"))
        end
    end
    isempty(R) || fill!(R, zero(S))
    isempty(A) && return R

    rowval = rowvals(A)
    nzval = nonzeros(A)
    m = size(A, 1)
    n = size(A, 2)

    if size(R, 1) == size(R, 2) == 1
        # Reduction along both columns and rows
        R[1, 1] = centralize_sumabs2(A, means[1])
    elseif size(R, 1) == 1
        # Reduction along rows
        @inbounds for col = 1:n
            mu = means[col]
            r = convert(S, (m - length(nzrange(A, col)))*abs2(mu))
            @simd for j = nzrange(A, col)
                r += abs2(nzval[j] - mu)
            end
            R[1, col] = r
        end
    elseif size(R, 2) == 1
        # Reduction along columns
        rownz = fill(convert(Ti, n), m)
        @inbounds for col = 1:n
            @simd for j = nzrange(A, col)
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
            @simd for j = nzrange(A, col)
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
