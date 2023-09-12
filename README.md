# Deal.II with Julia

This repository is an example of how to embed Julia code in C++. Specifically,
it demonstrates how to use the deal.II finite element library with parts of the
program, the material law, is using Julia.

This demonstration will be presented at the deal.II Users and Developers
Workshop, September 11-15, 2023, in Hannover.

## Build and run

1. Install deal.II
2. Install Julia and make sure it is available in `PATH`.
3. Install the Julia package
   [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl):
   ```
   julia -e 'using Pkg; Pkg.add("Tensors")'
   ```
4. Run `cmake` and `make`:
   ```
   cmake -DDEAL_II_DIR=/path/to/deal.II .
   make
   ```
5. Run the executable
   ```
   ./hyperelasticity
   ```
