// MIT License
// Copyright (c) 2023 Kristoffer Carlsson, Fredrik Ekre

#include <fstream>
#include <iostream>
#include <vector>

#include <julia.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "hyperelasticity.h"

#ifndef JULIA_SOURCE_FILE
#define JULIA_SOURCE_FILE "./src/hyperelasticity.jl"
#endif

namespace HyperelasticityNS {

using namespace dealii;

// Material parameters.
// Maps directly to the corresponding struct in hyperelasticity.jl.
struct NeoHooke {
  double mu;
  double lambda;
};

// Material state.
// Maps directly to the corresponding struct in hyperelasticity.jl. This struct
// is allocated on the C side, but only used on the Julia side so the stress is
// stored as an std::array<T, N> (aka Tensor{2} in Julia) and not a
// dealii:Tensor.
template <int dim> struct MaterialState {
  std::array<double, dim * dim> cauchy;
};

// Quadrature point data.
// Stores the previous (from the last converged timestep) and new material
// state, together with the integration weight for computing the cell-wise
// averaged von Mise stress in postprocessing.
template <int dim> struct QuadraturePointData {
  MaterialState<dim> prev_state;
  MaterialState<dim> new_state;
  double JxW;
};

// dealii::Tensor<1, dim> to std::array<T, dim> conversion
template <int dim>
std::array<double, dim> convert_tensor_to_array(dealii::Tensor<1, dim> t) {
  std::array<double, dim> a;
  for (int i = 0; i < dim; ++i)
    a[i] = t[i];
  return a;
}

// dealii::Tensor<2, dim> to std::array<T, dim * dim> conversion
template <int dim>
std::array<double, dim * dim>
convert_tensor_to_array(dealii::Tensor<2, dim> t) {
  std::array<double, dim * dim> a;
  // Note: Tensors.jl's Tensor type is column major so the row-loop is the
  // innermost loop
  int i = 0;
  for (int col = 0; col < dim; ++col) {
    for (int row = 0; row < dim; ++row)
      a[i++] = t[row][col];
  }
  return a;
}

// Boundary values
template <int dim> class BoundaryValues : public Function<dim> {
public:
  BoundaryValues(const double time = 0.);

  double get_time() const;

  virtual void vector_value(const Point<dim>& p,
                            Vector<double>& values) const override;

  virtual void
  vector_value_list(const std::vector<Point<dim>>& points,
                    std::vector<Vector<double>>& value_list) const override;

private:
  const double time;
};

template <int dim> double BoundaryValues<dim>::get_time() const { return time; }

template <int dim>
BoundaryValues<dim>::BoundaryValues(const double time)
    : Function<dim>(dim), time(time) {}

template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim>& p,
                                       Vector<double>& values) const {
  auto t = get_time();
  values(0) = 0.01 * (p(1) + p(2));
  values(1) = 0.01 * (p(2) + p(0));
  if (dim == 3)
    values(2) = 0.01 * (p(0) + p(1));
  values *= t;
}

template <int dim>
void BoundaryValues<dim>::vector_value_list(
    const std::vector<Point<dim>>& points,
    std::vector<Vector<double>>& value_list) const {
  const unsigned int n_points = points.size();
  Assert(value_list.size() == n_points,
         ExcDimensionMismatch(value_list.size(), n_points));
  for (unsigned int p = 0; p < n_points; ++p)
    BoundaryValues<dim>::vector_value(points[p], value_list[p]);
}

template <int dim>
HyperelasticitySim<dim>::HyperelasticitySim()
    : degree(1), fe(FE_Q<dim>(1), dim), dofs_per_cell(fe.dofs_per_cell),
      dof_handler(triangulation),
      timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
      q_cell(2), n_q_points(q_cell.size()), time(1.0, 0.1) {
  jl_assemble =
      reinterpret_cast<jl_assemble_t>(jl_unbox_voidpointer(jl_eval_string(
          "@cfunction(do_assemble!, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, "
          "Ptr{MaterialState}, MaterialState, Tensor{2, 3, Cdouble, 9}, "
          "Ptr{Vec{3, Cdouble}}, Ptr{Tensor{2, 3, Cdouble, 9}}, Cint, Cdouble, "
          "NeoHooke))")));

  jl_compute_mise = reinterpret_cast<jl_compute_mise_t>(jl_unbox_voidpointer(
      jl_eval_string("@cfunction(compute_mise, Cdouble, (MaterialState, ))")));
}

template <int dim> HyperelasticitySim<dim>::~HyperelasticitySim() {
  dof_handler.clear();
}

template <int dim> void HyperelasticitySim<dim>::run() {
  {
    TimerOutput::Scope timer_section(timer, "Making grid");
    make_grid();
  }

  {
    TimerOutput::Scope timer_section(timer, "Setup");
    system_setup();
  }

  Vector<double> solution_delta(dof_handler.n_dofs());
  time.increment();

  while (time.current() < time.end()) {
    solution_delta = 0.0;
    solve_nonlinear_timestep(solution_delta);
    // Accumulate increment to total solution
    solution_n += solution_delta;
    update_quadrature_point_data();
    output_results();
    time.increment();
  }
}

template <int dim> void HyperelasticitySim<dim>::make_grid() {
  deallog << "Creating mesh" << std::endl << std::flush;

  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

  GridTools::scale(1.0, triangulation);
  triangulation.refine_global(3);
}

/*
 * Creates the dofs for the fe systems,
 * Reorder the dofs,
 * Creates the sparsity pattern,
 * Computes dirichlet contrains
 */
template <int dim> void HyperelasticitySim<dim>::system_setup() {

  setup_quadrature_point_data();
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  std::cout << "Triangulation:"
            << "\n\t Number of active cells: " << triangulation.n_active_cells()
            << "\n\t Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  newton_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, newton_constraints);
  VectorTools::interpolate_boundary_values(
      dof_handler, 0, Functions::ZeroFunction<dim>(dim), newton_constraints);
  newton_constraints.close();

  tangent_matrix.clear();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, newton_constraints);
  sparsity_pattern.copy_from(dsp);

  tangent_matrix.reinit(sparsity_pattern);
  solution_n.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim> void HyperelasticitySim<dim>::output_results() {
  static int n_output = 0;
  DataOut<dim> data_out;
  std::vector<std::string> solution_names(dim, "displacements");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_n, solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  Vector<double> stress_out(triangulation.n_active_cells());

  unsigned int index = 0;
  for (const auto& cell : triangulation.active_cell_iterators()) {
    double weighted_stress = 0.0;
    double volume = 0.0;
    for (unsigned int q = 0; q < n_q_points; ++q) {
      auto& data =
          reinterpret_cast<QuadraturePointData<dim>*>(cell->user_pointer())[q];
      auto m = jl_compute_mise(data.prev_state);
      weighted_stress += m * data.JxW;
      volume += data.JxW;
    }
    stress_out[index] = weighted_stress / volume;
    index++;
  }

  Assert(index == triangulation.n_active_cells(), ExcInternalError());
  data_out.add_data_vector(stress_out, "vonmises");
  data_out.build_patches();
  std::ofstream out("solution" + std::to_string(n_output) + ".vtu");
  n_output++;
  data_out.write_vtu(out);
}

template <int dim> void HyperelasticitySim<dim>::get_error_residual() {
  Vector<double> error_res(dof_handler.n_dofs());
  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    if (!dirichlet_constraints.is_constrained(i))
      error_res(i) = system_rhs(i);
  error_residual.norm = error_res.l2_norm();
  error_residual.u = error_res.l2_norm();
}

template <int dim>
void HyperelasticitySim<dim>::get_error_update(
    const Vector<double>& newton_update) {
  Vector<double> error_ud(dof_handler.n_dofs());
  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    if (!dirichlet_constraints.is_constrained(i))
      error_ud(i) = newton_update(i);
  error_update.norm = error_ud.l2_norm();
  // TODO: Fix this when more fields
  error_update.u = error_ud.l2_norm();
}

template <int dim>
Vector<double> HyperelasticitySim<dim>::get_total_solution(
    const Vector<double>& solution_delta) const {
  Vector<double> solution_total(solution_n);
  solution_total += solution_delta;
  return solution_total;
}

/*
 * ********
 * SOLVERS
 * ********
 */

template <int dim>
void HyperelasticitySim<dim>::solve_nonlinear_timestep(
    Vector<double>& solution_delta) {
  std::cout << std::endl
            << "Timestep " << time.get_timestep() << " @ " << time.current()
            << "s" << std::endl;
  Vector<double> newton_update(dof_handler.n_dofs());
  error_residual.reset();
  error_residual_0.reset();
  error_residual_norm.reset();
  error_update.reset();
  error_update_0.reset();
  error_update_norm.reset();
  /*
   * We make the dirichlet constrains before the newton iterations
   * and distribute it to the full solution vector.
   * We will then use zero boundary conditions for the newton updats
   * so the distributed values will be never change:
   */
  make_dirichlet_constraints();
  dirichlet_constraints.distribute(solution_n);
  print_conv_header();
  unsigned int newton_iteration = 1;
  for (; newton_iteration <= 10; ++newton_iteration) {
    std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;
    tangent_matrix = 0.0;
    system_rhs = 0.0;

    if (newton_iteration == 1) {
      error_residual_0 = error_residual;
    } else {
    }

    assemble_system(solution_delta);
    get_error_residual();
    error_residual_norm = error_residual;
    error_residual_norm.normalise(error_residual_0);

    if (newton_iteration > 1 && error_update_norm.u <= 1e-6 &&
        error_residual_norm.u <= 1e-6) {
      std::cout << " CONVERGED! " << std::endl;
      // We converged, so update the gauss points with the new quadrature
      // points.
      print_conv_footer();
      break;
    }

    const std::pair<unsigned int, double> lin_solver_output =
        solve_linear_system(newton_update);
    get_error_update(newton_update);
    if (newton_iteration == 1)
      error_update_0 = error_update;
    error_update_norm = error_update;
    error_update_norm.normalise(error_update_0);
    solution_delta -= newton_update;
    std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
              << std::scientific << lin_solver_output.first << "  "
              << lin_solver_output.second << "  " << error_residual_norm.norm
              << "  " << error_residual_norm.u << "  " << error_update_norm.norm
              << "  " << error_update_norm.u << std::endl;
  }
  AssertThrow(newton_iteration <= 10,
              ExcMessage("No convergence in nonlinear solver!"));
}

template <int dim>
std::pair<unsigned int, double>
HyperelasticitySim<dim>::solve_linear_system(Vector<double>& newton_update) {
  timer.enter_subsection("Solving linear system");
  std::cout << " SLV " << std::flush;
  unsigned int lin_it = 0;
  double lin_res = 0.0;

  SparseDirectUMFPACK A_direct;
  A_direct.initialize(tangent_matrix);
  A_direct.vmult(newton_update, system_rhs);

  newton_constraints.distribute(newton_update);
  timer.leave_subsection();
  return std::make_pair(lin_it, lin_res);
}

template <int dim> void HyperelasticitySim<dim>::print_conv_header() {
  static const unsigned int l_width = 100;
  for (unsigned int i = 0; i < l_width; ++i)
    std::cout << "_";
  std::cout << std::endl;
  std::cout << "  SOLVER STEP "
            << " |  LIN_IT   LIN_RES    RES_NORM    "
            << " RES_U  "
            << "    NU_NORM     NU_U" << std::endl;
  for (unsigned int i = 0; i < l_width; ++i)
    std::cout << "_";
  std::cout << std::endl;
}

template <int dim> void HyperelasticitySim<dim>::print_conv_footer() {
  static const unsigned int l_width = 155;
  for (unsigned int i = 0; i < l_width; ++i)
    std::cout << "_";
  std::cout << std::endl;
  std::cout << "Relative errors:" << std::endl
            << "Displacement:\t" << error_update.u / error_update_0.u
            << std::endl
            << "Force: \t\t" << error_residual.u / error_residual_0.u
            << std::endl;
}

template <int dim>
void HyperelasticitySim<dim>::assemble_system(
    const Vector<double>& solution_delta) {
  timer.enter_subsection("Assembling");
  std::cout << " ASM " << std::flush;

  timer.enter_subsection("Preamble");

  FEValues<dim> fe_values(fe, q_cell,
                          update_values | update_gradients | update_JxW_values |
                              update_quadrature_points);
  auto total_solution = get_total_solution(solution_delta);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Vector displacement(0);
  tangent_matrix = 0;
  system_rhs = 0;

  auto E = 10.0;
  auto ν = 0.3;
  auto μ = E / (2 * (1 + ν));
  auto λ = (E * ν) / ((1 + ν) * (1 - 2 * ν));
  NeoHooke mp{μ, λ};

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<double> cell_matrix_raw(dofs_per_cell * dofs_per_cell);
  std::vector<double> cell_rhs_raw(dofs_per_cell);

  std::vector<std::array<double, dim>> δui(dofs_per_cell);
  std::vector<std::array<double, dim * dim>> grad_δui(dofs_per_cell);

  std::vector<Tensor<2, dim>> grad_u(n_q_points);
  timer.leave_subsection();

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    timer.enter_subsection("Loop cells");

    cell_matrix = 0;
    cell_rhs = 0;
    std::fill(cell_matrix_raw.begin(), cell_matrix_raw.end(), 0.0);
    std::fill(cell_rhs_raw.begin(), cell_rhs_raw.end(), 0.0);

    fe_values.reinit(cell);

    fe_values[displacement].get_function_gradients(total_solution, grad_u);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      auto grad_u_q = grad_u[q_point];
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        δui[i] =
            convert_tensor_to_array(fe_values[displacement].value(i, q_point));
        grad_δui[i] = convert_tensor_to_array(
            fe_values[displacement].gradient(i, q_point));
      }
      auto dΩ = fe_values.JxW(q_point);
      {
        timer.enter_subsection("Compute jl");
        auto& data = reinterpret_cast<QuadraturePointData<dim>*>(
            cell->user_pointer())[q_point];
        data.JxW = dΩ;
        jl_assemble(cell_rhs_raw.data(), cell_matrix_raw.data(),
                    &(data.new_state), data.prev_state,
                    convert_tensor_to_array(grad_u_q), δui.data(),
                    grad_δui.data(), dofs_per_cell, dΩ, mp);
        timer.leave_subsection();
      };
    }

    // Loop over cell_rhs_raw and copy to cell_rhs
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      cell_rhs[i] = cell_rhs_raw[i];

    int m = 0;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        cell_matrix[i][j] = cell_matrix_raw[m];
        m++;
      }

    cell->get_dof_indices(local_dof_indices);
    newton_constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, tangent_matrix, system_rhs);
    timer.leave_subsection();
  }
  timer.leave_subsection();
}

template <int dim> void HyperelasticitySim<dim>::make_dirichlet_constraints() {
  std::cout << " CST " << std::endl << std::flush;
  dirichlet_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, dirichlet_constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 0,
                                           BoundaryValues<dim>(time.current()),
                                           dirichlet_constraints);
  dirichlet_constraints.close();
}

template <int dim> void HyperelasticitySim<dim>::setup_quadrature_point_data() {
  deallog << "Setting up quadrature point history" << std::endl << std::flush;
  triangulation.clear_user_data();
  {
    std::vector<QuadraturePointData<dim>> tmp;
    tmp.swap(quadrature_point_data);
  }
  quadrature_point_data.resize(triangulation.n_active_cells() * n_q_points);
  unsigned int qp_index = 0;
  for (auto cell : dof_handler.active_cell_iterators()) {
    cell->set_user_pointer(&quadrature_point_data[qp_index]);
    qp_index += n_q_points;
  }
  Assert(qp_index == quadrature_point_data.size(), ExcInternalError());
  deallog << "Set up a total of " << qp_index << " qphs on "
          << triangulation.n_active_cells() << " cells" << std::endl
          << std::flush;
}

template <int dim>
void HyperelasticitySim<dim>::update_quadrature_point_data() {
  for (auto& data : quadrature_point_data) {
    std::swap(data.prev_state, data.new_state);
  }
}

} // namespace HyperelasticityNS

jl_value_t* checked_eval_string(const char* code) {
  jl_value_t* result = jl_eval_string(code);
  if (jl_exception_occurred()) {
    jl_call2(jl_get_function(jl_base_module, "showerror"), jl_stderr_obj(),
             jl_exception_occurred());
    jl_printf(jl_stderr_stream(), "\n");
    jl_atexit_hook(1);
    exit(1);
  }
  assert(result && "Missing return value but no exception occurred!");
  return result;
}

int main() {
  using dealii::deallog;
  using HyperelasticityNS::HyperelasticitySim;

  try {
    deallog.depth_console(0);
    std::ofstream log_out("hyperelasticity.log");
    deallog.attach(log_out);

    // Setup the julia context
    jl_init();

    // Load the julia source file
    auto str = std::string("include(\"") + std::string(JULIA_SOURCE_FILE) +
               std::string("\")");
    checked_eval_string(str.c_str());

    // Run the simulation
    HyperelasticitySim<3> sim;
    sim.run();
  } catch (std::exception& exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    jl_atexit_hook(1);
    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    jl_atexit_hook(1);
    return 1;
  }
  // Shut down julia and return
  jl_atexit_hook(0);
  return 0;
}
