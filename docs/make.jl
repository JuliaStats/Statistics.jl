using Documenter, Statistics, Random

# Workaround for JuliaLang/julia/pull/28625
if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

makedocs(
    sitename = "Statistics.jl",
    modules = [Statistics],
    pages = ["index.md",
             "weights.md",
             "scalarstats.md",
             "cov.md",
             "robust.md",
             "ranking.md",
             "empirical.md",
             "transformations.md"]
)

deploydocs(
    repo = "github.com/JuliaLang/Statistics.jl.git"
)
