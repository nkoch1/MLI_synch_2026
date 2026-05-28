# Code for Koch & Khadra 2026


This repository contains the data and code for reproducing the figures in the following publication:


Koch, N. A. & Khadra, A. (2026). Gap junctional coupling of molecular layer interneurons enables transient NMDA driven synchronization. bioRxiv. doi:[doi](https://doi.org/doi)



## Model simulation and figures

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named ```MLI_synch_2026```

To reproduce the simulations and figures, do the following:

1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```
   This will install all necessary packages for you to be able to run the scripts and everything should work out of the box, including correctly finding local paths.

3. Run the Julia (.jl) scripts in the ```./scripts/``` directory to run the model simulations
4. Run the Julia (.jl) scripts in the ```./scripts/Figures/``` directory to reproduce the publication figures


You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "MLI_synch_2026"
```
which auto-activate the project and enable local path handling from DrWatson.


## License

[<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by.png" style="width: 150px" >](https://creativecommons.org/licenses/by/4.0/)

This repository is licensed under a Creative Commons Attribution 4.0 International License.
