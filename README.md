# Deal.II with Julia

This repository is an example of how to embed Julia code in C++. Specifically,
it demonstrates how to use the deal.II finite element library with parts of the
program, the material law, is using Julia.

This demonstration will be presented at the deal.II Users and Developers
Workshop, September 11-15, 2023, in Hannover.

## Build and run

1. Install deal.II
2. Install Julia and make sure it is available in `PATH`.
3. Run `cmake` and `make`:
   ```
   cmake -DDEAL_II_DIR=/path/to/deal.II .
   make
   ```
4. Run the executable

   ```
   ./hyperelasticity
   ```
