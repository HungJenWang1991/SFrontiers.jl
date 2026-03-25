using Documenter
using SFrontiers

makedocs(
    sitename = "SFrontiers.jl",
    modules = [SFrontiers],
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(
    repo = "github.com/HungJenWang/SFrontiers.jl.git",
)
