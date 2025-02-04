using Documenter, Statistics

# Workaround for JuliaLang/julia/pull/28625
if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

makedocs(
    modules = [Statistics],
    sitename = "Statistics",
    warnonly = [:missing_docs, :cross_references]
    pages = Any[
        "Statistics" => "index.md"
        ]
    )

deploydocs(repo = "github.com/JuliaStats/Statistics.jl.git")
