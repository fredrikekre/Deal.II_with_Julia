#pragma once

#include "Time.h"

/* #include <deal.II/base/parameter_handler.h> */
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/base/quadrature_lib.h>
#include <string>
#include <vector>

namespace HyperelasticityNS {

struct NeoHooke {
  double mu;
  double lambda;
};

struct QuadraturePointData;

template <int dim> class HyperelasticitySim {
public:
  /* using FuncType = void(*)(double*, double*, std::array<double, dim*dim>, std::array<double, dim>*, std::array<double, dim*dim>*, int, double, NeoHooke); */
  /* using FuncType = void(*)(double*, double*, double*, std::array<double, dim*dim>, std::array<double, dim>*, std::array<double, dim*dim>*, int, double, NeoHooke); */
  using FuncType = void(*)(double*, double*, QuadraturePointData*, std::array<double, dim*dim>, std::array<double, dim>*, std::array<double, dim*dim>*, int, double, NeoHooke);

  explicit HyperelasticitySim();
  virtual ~HyperelasticitySim();
  void run();

private:
  struct Errors {
    Errors() : norm(1.0), u(1.0) {}
    void reset() {
      norm = 1.0;
      u = 1.0;
    }
    void normalise(const Errors &rhs) {
      if (rhs.norm != 0.0)
        norm /= rhs.norm;
      if (rhs.u != 0.0)
        u /= rhs.u;
    }
    double norm, u;
  };

  void get_error_residual();
  void get_error_update(const dealii::Vector<double> &newton_update);

  void make_grid();
  void system_setup();
  void make_dirichlet_constraints();
  void solve_nonlinear_timestep(dealii::Vector<double> &solution_delta);
  std::pair<unsigned int, double>
  solve_linear_system(dealii::Vector<double> &newton_update);
  dealii::Vector<double>
  get_total_solution(const dealii::Vector<double> &solution_delta) const;
  void output_results();
  static void print_conv_header();
  void print_conv_footer();
  void setup_quadrature_point_data();

  void assemble_system(const dealii::Vector<double> &solution_delta);
  FuncType compute_jl;
  dealii::Triangulation<dim> triangulation;
  const unsigned int degree;
  const dealii::FESystem<dim> fe;
  const unsigned int dofs_per_cell;
  dealii::DoFHandler<dim> dof_handler;
  dealii::AffineConstraints<double> dirichlet_constraints, newton_constraints;
  dealii::SparsityPattern sparsity_pattern;
  dealii::SparseMatrix<double> tangent_matrix;
  dealii::Vector<double> solution_n;
  dealii::Vector<double> system_rhs;

  dealii::TimerOutput timer;
  const dealii::QGauss<dim> q_cell;
  const unsigned int n_q_points;
  std::vector<QuadraturePointData> quadrature_point_data;

  Time time;
  Errors error_residual, error_residual_0, error_residual_norm, error_update,
      error_update_0, error_update_norm;
};

} // namespace HyperelasticityNS
