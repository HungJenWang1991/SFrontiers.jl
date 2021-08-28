using Documenter, SFrontiers
makedocs(sitename="Stochastic Frontier Analysis using Julia")

push!(LOAD_PATH,"../src/")


makedocs(
  sitename="Stochastic Frontier Analysis using Julia",
  authors = "Hung-Jen Wang",
  format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        # canonical = "url_of_my_web_page/stable/",
        # assets = ["assets/detailed_example.zip"],
        analytics = "UA-134239283-1",
  ),
  pages = [
        "Home of SFrontiers.jl" => "index.md",
        "User Guide" => Any[
            "Installation" =>  "installation.md",
            "Estimation Overview" => "overview.md",
            "A Detailed Example" => "ex_detail.md",
            "Other Examples" => Any[
                    "cross-sectional models" => "ex_cross.md",
                    "panel models" => "ex_panel.md",
            ],
            "API Reference" => "api.md",
        ],
    ]
)