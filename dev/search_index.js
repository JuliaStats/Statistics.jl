var documenterSearchIndex = {"docs":
[{"location":"#Statistics-1","page":"Statistics","title":"Statistics","text":"","category":"section"},{"location":"#","page":"Statistics","title":"Statistics","text":"The Statistics standard library module contains basic statistics functionality.","category":"page"},{"location":"#","page":"Statistics","title":"Statistics","text":"std\nstdm\nvar\nvarm\ncor\ncov\nmean!\nmean\nmedian!\nmedian\nmiddle\nquantile!\nquantile","category":"page"},{"location":"#Statistics.std","page":"Statistics","title":"Statistics.std","text":"std(itr; corrected::Bool=true, mean=nothing[, dims])\n\nCompute the sample standard deviation of collection itr.\n\nThe algorithm returns an estimator of the generative distribution's standard deviation under the assumption that each entry of itr is a sample drawn from the same unknown distribution, with the samples uncorrelated. For arrays, this computation is equivalent to calculating sqrt(sum((itr .- mean(itr)).^2) / (length(itr) - 1)). If corrected is true, then the sum is scaled with n-1, whereas the sum is scaled with n if corrected is false with n the number of elements in itr.\n\nIf itr is an AbstractArray, dims can be provided to compute the standard deviation over dimensions.\n\nA pre-computed mean may be provided. When dims is specified, mean must be an array with the same shape as mean(itr, dims=dims) (additional trailing singleton dimensions are allowed).\n\nnote: Note\nIf array contains NaN or missing values, the result is also NaN or missing (missing takes precedence if array contains both). Use the skipmissing function to omit missing entries and compute the standard deviation of non-missing values.\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.stdm","page":"Statistics","title":"Statistics.stdm","text":"stdm(itr, mean; corrected::Bool=true[, dims])\n\nCompute the sample standard deviation of collection itr, with known mean(s) mean.\n\nThe algorithm returns an estimator of the generative distribution's standard deviation under the assumption that each entry of itr is a sample drawn from the same unknown distribution, with the samples uncorrelated. For arrays, this computation is equivalent to calculating sqrt(sum((itr .- mean(itr)).^2) / (length(itr) - 1)). If corrected is true, then the sum is scaled with n-1, whereas the sum is scaled with n if corrected is false with n the number of elements in itr.\n\nIf itr is an AbstractArray, dims can be provided to compute the standard deviation over dimensions. In that case, mean must be an array with the same shape as mean(itr, dims=dims) (additional trailing singleton dimensions are allowed).\n\nnote: Note\nIf array contains NaN or missing values, the result is also NaN or missing (missing takes precedence if array contains both). Use the skipmissing function to omit missing entries and compute the standard deviation of non-missing values.\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.var","page":"Statistics","title":"Statistics.var","text":"var(itr; corrected::Bool=true, mean=nothing[, dims])\n\nCompute the sample variance of collection itr.\n\nThe algorithm returns an estimator of the generative distribution's variance under the assumption that each entry of itr is a sample drawn from the same unknown distribution, with the samples uncorrelated. For arrays, this computation is equivalent to calculating sum((itr .- mean(itr)).^2) / (length(itr) - 1). If corrected is true, then the sum is scaled with n-1, whereas the sum is scaled with n if corrected is false where n is the number of elements in itr.\n\nIf itr is an AbstractArray, dims can be provided to compute the variance over dimensions.\n\nA pre-computed mean may be provided. When dims is specified, mean must be an array with the same shape as mean(itr, dims=dims) (additional trailing singleton dimensions are allowed).\n\nnote: Note\nIf array contains NaN or missing values, the result is also NaN or missing (missing takes precedence if array contains both). Use the skipmissing function to omit missing entries and compute the variance of non-missing values.\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.varm","page":"Statistics","title":"Statistics.varm","text":"varm(itr, mean; dims, corrected::Bool=true)\n\nCompute the sample variance of collection itr, with known mean(s) mean.\n\nThe algorithm returns an estimator of the generative distribution's variance under the assumption that each entry of itr is a sample drawn from the same unknown distribution, with the samples uncorrelated. For arrays, this computation is equivalent to calculating sum((itr .- mean(itr)).^2) / (length(itr) - 1). If corrected is true, then the sum is scaled with n-1, whereas the sum is scaled with n if corrected is false with n the number of elements in itr.\n\nIf itr is an AbstractArray, dims can be provided to compute the variance over dimensions. In that case, mean must be an array with the same shape as mean(itr, dims=dims) (additional trailing singleton dimensions are allowed).\n\nnote: Note\nIf array contains NaN or missing values, the result is also NaN or missing (missing takes precedence if array contains both). Use the skipmissing function to omit missing entries and compute the variance of non-missing values.\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.cor","page":"Statistics","title":"Statistics.cor","text":"cor(x::AbstractVector)\n\nReturn the number one.\n\n\n\n\n\ncor(X::AbstractMatrix; dims::Int=1)\n\nCompute the Pearson correlation matrix of the matrix X along the dimension dims.\n\n\n\n\n\ncor(x::AbstractVector, y::AbstractVector)\n\nCompute the Pearson correlation between the vectors x and y.\n\n\n\n\n\ncor(X::AbstractVecOrMat, Y::AbstractVecOrMat; dims=1)\n\nCompute the Pearson correlation between the vectors or matrices X and Y along the dimension dims.\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.cov","page":"Statistics","title":"Statistics.cov","text":"cov(x::AbstractVector; corrected::Bool=true)\n\nCompute the variance of the vector x. If corrected is true (the default) then the sum is scaled with n-1, whereas the sum is scaled with n if corrected is false where n = length(x).\n\n\n\n\n\ncov(X::AbstractMatrix; dims::Int=1, corrected::Bool=true)\n\nCompute the covariance matrix of the matrix X along the dimension dims. If corrected is true (the default) then the sum is scaled with n-1, whereas the sum is scaled with n if corrected is false where n = size(X, dims).\n\n\n\n\n\ncov(x::AbstractVector, y::AbstractVector; corrected::Bool=true)\n\nCompute the covariance between the vectors x and y. If corrected is true (the default), computes frac1n-1sum_i=1^n (x_i-bar x) (y_i-bar y)^* where * denotes the complex conjugate and n = length(x) = length(y). If corrected is false, computes frac1nsum_i=1^n (x_i-bar x) (y_i-bar y)^*.\n\n\n\n\n\ncov(X::AbstractVecOrMat, Y::AbstractVecOrMat; dims::Int=1, corrected::Bool=true)\n\nCompute the covariance between the vectors or matrices X and Y along the dimension dims. If corrected is true (the default) then the sum is scaled with n-1, whereas the sum is scaled with n if corrected is false where n = size(X, dims) = size(Y, dims).\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.mean!","page":"Statistics","title":"Statistics.mean!","text":"mean!(r, v)\n\nCompute the mean of v over the singleton dimensions of r, and write results to r.\n\nExamples\n\njulia> using Statistics\n\njulia> v = [1 2; 3 4]\n2×2 Matrix{Int64}:\n 1  2\n 3  4\n\njulia> mean!([1., 1.], v)\n2-element Vector{Float64}:\n 1.5\n 3.5\n\njulia> mean!([1. 1.], v)\n1×2 Matrix{Float64}:\n 2.0  3.0\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.mean","page":"Statistics","title":"Statistics.mean","text":"mean(itr)\n\nCompute the mean of all elements in a collection.\n\nnote: Note\nIf itr contains NaN or missing values, the result is also NaN or missing (missing takes precedence if array contains both). Use the skipmissing function to omit missing entries and compute the mean of non-missing values.\n\nExamples\n\njulia> using Statistics\n\njulia> mean(1:20)\n10.5\n\njulia> mean([1, missing, 3])\nmissing\n\njulia> mean(skipmissing([1, missing, 3]))\n2.0\n\n\n\n\n\nmean(f, itr)\n\nApply the function f to each element of collection itr and take the mean.\n\njulia> using Statistics\n\njulia> mean(√, [1, 2, 3])\n1.3820881233139908\n\njulia> mean([√1, √2, √3])\n1.3820881233139908\n\n\n\n\n\nmean(f, A::AbstractArray; dims)\n\nApply the function f to each element of array A and take the mean over dimensions dims.\n\ncompat: Julia 1.3\nThis method requires at least Julia 1.3.\n\njulia> using Statistics\n\njulia> mean(√, [1, 2, 3])\n1.3820881233139908\n\njulia> mean([√1, √2, √3])\n1.3820881233139908\n\njulia> mean(√, [1 2 3; 4 5 6], dims=2)\n2×1 Matrix{Float64}:\n 1.3820881233139908\n 2.2285192400943226\n\n\n\n\n\nmean(A::AbstractArray; dims)\n\nCompute the mean of an array over the given dimensions.\n\ncompat: Julia 1.1\nmean for empty arrays requires at least Julia 1.1.\n\nExamples\n\njulia> using Statistics\n\njulia> A = [1 2; 3 4]\n2×2 Matrix{Int64}:\n 1  2\n 3  4\n\njulia> mean(A, dims=1)\n1×2 Matrix{Float64}:\n 2.0  3.0\n\njulia> mean(A, dims=2)\n2×1 Matrix{Float64}:\n 1.5\n 3.5\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.median!","page":"Statistics","title":"Statistics.median!","text":"median!(v)\n\nLike median, but may overwrite the input vector.\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.median","page":"Statistics","title":"Statistics.median","text":"median(itr)\n\nCompute the median of all elements in a collection. For an even number of elements no exact median element exists, so the result is equivalent to calculating mean of two median elements.\n\nnote: Note\nIf itr contains NaN or missing values, the result is also NaN or missing (missing takes precedence if itr contains both). Use the skipmissing function to omit missing entries and compute the median of non-missing values.\n\nExamples\n\njulia> using Statistics\n\njulia> median([1, 2, 3])\n2.0\n\njulia> median([1, 2, 3, 4])\n2.5\n\njulia> median([1, 2, missing, 4])\nmissing\n\njulia> median(skipmissing([1, 2, missing, 4]))\n2.0\n\n\n\n\n\nmedian(A::AbstractArray; dims)\n\nCompute the median of an array along the given dimensions.\n\nExamples\n\njulia> using Statistics\n\njulia> median([1 2; 3 4], dims=1)\n1×2 Matrix{Float64}:\n 2.0  3.0\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.middle","page":"Statistics","title":"Statistics.middle","text":"middle(x)\n\nCompute the middle of a scalar value, which is equivalent to x itself, but of the type of middle(x, x) for consistency.\n\n\n\n\n\nmiddle(x, y)\n\nCompute the middle of two numbers x and y, which is equivalent in both value and type to computing their mean ((x + y) / 2).\n\n\n\n\n\nmiddle(a::AbstractArray)\n\nCompute the middle of an array a, which consists of finding its extrema and then computing their mean.\n\njulia> using Statistics\n\njulia> middle(1:10)\n5.5\n\njulia> a = [1,2,3.6,10.9]\n4-element Vector{Float64}:\n  1.0\n  2.0\n  3.6\n 10.9\n\njulia> middle(a)\n5.95\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.quantile!","page":"Statistics","title":"Statistics.quantile!","text":"quantile!([q::AbstractArray, ] v::AbstractVector, p; sorted=false, alpha::Real=1.0, beta::Real=alpha)\n\nCompute the quantile(s) of a vector v at a specified probability or vector or tuple of probabilities p on the interval [0,1]. If p is a vector, an optional output array q may also be specified. (If not provided, a new output array is created.) The keyword argument sorted indicates whether v can be assumed to be sorted; if false (the default), then the elements of v will be partially sorted in-place.\n\nSamples quantile are defined by Q(p) = (1-γ)*x[j] + γ*x[j+1], where x[j] is the j-th order statistic of v, j = floor(n*p + m), m = alpha + p*(1 - alpha - beta) and γ = n*p + m - j.\n\nBy default (alpha = beta = 1), quantiles are computed via linear interpolation between the points ((k-1)/(n-1), x[k]), for k = 1:n where n = length(v). This corresponds to Definition 7 of Hyndman and Fan (1996), and is the same as the R and NumPy default.\n\nThe keyword arguments alpha and beta correspond to the same parameters in Hyndman and Fan, setting them to different values allows to calculate quantiles with any of the methods 4-9 defined in this paper:\n\nDef. 4: alpha=0, beta=1\nDef. 5: alpha=0.5, beta=0.5\nDef. 6: alpha=0, beta=0 (Excel PERCENTILE.EXC, Python default, Stata altdef)\nDef. 7: alpha=1, beta=1 (Julia, R and NumPy default, Excel PERCENTILE and PERCENTILE.INC, Python 'inclusive')\nDef. 8: alpha=1/3, beta=1/3\nDef. 9: alpha=3/8, beta=3/8\n\nnote: Note\nAn ArgumentError is thrown if v contains NaN or missing values.\n\nReferences\n\nHyndman, R.J and Fan, Y. (1996) \"Sample Quantiles in Statistical Packages\", The American Statistician, Vol. 50, No. 4, pp. 361-365\nQuantile on Wikipedia details the different quantile definitions\n\nExamples\n\njulia> using Statistics\n\njulia> x = [3, 2, 1];\n\njulia> quantile!(x, 0.5)\n2.0\n\njulia> x\n3-element Vector{Int64}:\n 1\n 2\n 3\n\njulia> y = zeros(3);\n\njulia> quantile!(y, x, [0.1, 0.5, 0.9]) === y\ntrue\n\njulia> y\n3-element Vector{Float64}:\n 1.2000000000000002\n 2.0\n 2.8000000000000003\n\n\n\n\n\n","category":"function"},{"location":"#Statistics.quantile","page":"Statistics","title":"Statistics.quantile","text":"quantile(itr, p; sorted=false, alpha::Real=1.0, beta::Real=alpha)\n\nCompute the quantile(s) of a collection itr at a specified probability or vector or tuple of probabilities p on the interval [0,1]. The keyword argument sorted indicates whether itr can be assumed to be sorted.\n\nSamples quantile are defined by Q(p) = (1-γ)*x[j] + γ*x[j+1], where x[j] is the j-th order statistic of itr, j = floor(n*p + m), m = alpha + p*(1 - alpha - beta) and γ = n*p + m - j.\n\nBy default (alpha = beta = 1), quantiles are computed via linear interpolation between the points ((k-1)/(n-1), x[k]), for k = 1:n where n = length(itr). This corresponds to Definition 7 of Hyndman and Fan (1996), and is the same as the R and NumPy default.\n\nThe keyword arguments alpha and beta correspond to the same parameters in Hyndman and Fan, setting them to different values allows to calculate quantiles with any of the methods 4-9 defined in this paper:\n\nDef. 4: alpha=0, beta=1\nDef. 5: alpha=0.5, beta=0.5\nDef. 6: alpha=0, beta=0 (Excel PERCENTILE.EXC, Python default, Stata altdef)\nDef. 7: alpha=1, beta=1 (Julia, R and NumPy default, Excel PERCENTILE and PERCENTILE.INC, Python 'inclusive')\nDef. 8: alpha=1/3, beta=1/3\nDef. 9: alpha=3/8, beta=3/8\n\nnote: Note\nAn ArgumentError is thrown if v contains NaN or missing values. Use the skipmissing function to omit missing entries and compute the quantiles of non-missing values.\n\nReferences\n\nHyndman, R.J and Fan, Y. (1996) \"Sample Quantiles in Statistical Packages\", The American Statistician, Vol. 50, No. 4, pp. 361-365\nQuantile on Wikipedia details the different quantile definitions\n\nExamples\n\njulia> using Statistics\n\njulia> quantile(0:20, 0.5)\n10.0\n\njulia> quantile(0:20, [0.1, 0.5, 0.9])\n3-element Vector{Float64}:\n  2.0\n 10.0\n 18.000000000000004\n\njulia> quantile(skipmissing([1, 10, missing]), 0.5)\n5.5\n\n\n\n\n\n","category":"function"}]
}
