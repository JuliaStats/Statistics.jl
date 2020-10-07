# Statistics.jl

[![Travis CI Build Status][travis-img]][travis-url]

Development repository for the Statistics standard library (stdlib) that ships with Julia. 

#### Using the development version of Statistics.jl

If you want to develop this package, do the following steps:
- Clone the repo anywhere.
- In line 2 of the `Project.toml` file (the line that begins with `uuid = ...`), modify the UUID, e.g. change the `107` to `207`.
- Change the current directory to the Statistics repo you just cloned and start julia with `julia --project`.
- `import Statistics` will now load the files in the cloned repo instead of the Statistics stdlib.
- To test your changes, simply do `include("test/runtests.jl")`.

If you need to build Julia from source with a git checkout of Statistics, then instead use `make DEPS_GIT=Statistics` when building Julia. The `Statistics` repo is in `stdlib/Statistics`, and created initially with a detached `HEAD`. If you're doing this from a pre-existing Julia repository, you may need to `make clean` beforehand.

[travis-img]: https://travis-ci.com/JuliaLang/Statistics.jl.svg?branch=master
[travis-url]: https://travis-ci.com/JuliaLang/Statistics.jl
