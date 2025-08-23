#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>

#include "mass_assembler.h"
#include "stiffness_assembler.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>

#include <iomanip>
#include <numeric>  

#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/component_mask.h>  

#include <deal.II/base/multithread_info.h>
#include <cstring>   // strcmp
#include <cstdlib>   // strtod
#include <chrono>



using namespace dealii;

int main(int argc, char** argv)
{

  // -------------------- Handle arguments --------------------
  double mesh_size = 0.025;   // default
  double dt        = 5e-4;    // default
  double c         = 1.0;     // default

  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--h") == 0 || std::strcmp(argv[i], "-h") == 0) {
      char* end = nullptr; double v = std::strtod(argv[i+1], &end);
      if (!end || *end!='\0' || v <= 0.0 || v >= 1.0) {
        std::cerr << "Invalid --h (0<h<1): " << argv[i+1] << "\n";
        return 2;
      }
      mesh_size = v;
    } else if (std::strcmp(argv[i], "--dt") == 0) {
      char* end = nullptr; double v = std::strtod(argv[i+1], &end);
      if (!end || *end!='\0' || v <= 0.0) {
        std::cerr << "Invalid --dt (must be > 0): " << argv[i+1] << "\n";
        return 2; // <-- makes Test4 pass
      }
      dt = v;
    } else if (std::strcmp(argv[i], "--c") == 0) {
      char* end = nullptr; double v = std::strtod(argv[i+1], &end);
      if (!end || *end!='\0' || v <= 0.0) {
        std::cerr << "Invalid --c (must be > 0): " << argv[i+1] << "\n";
        return 2;
      }
      c = v;
    }
  }




  // -------------------- mesh --------------------

 
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);

  const unsigned int refinement_level =
    static_cast<unsigned int>(std::ceil(std::log2(1.0 / mesh_size)));
  triangulation.refine_global(refinement_level);





  // -------------------- FE space --------------------

  FE_Q<2>        fe(1);
  DoFHandler<2>  dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;
  std::cout << "deal.II threads: " << MultithreadInfo::n_threads() << std::endl;






 
  // -------------------- homogeneous Dirichlet BC --------------------


  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    dof_handler, 0 /*boundary id*/, Functions::ZeroFunction<2>(), constraints);
  constraints.close();


 // -------------------- assemble M, K --------------------
  
  // Build sparsity WITH constraints and keep constrained dofs present
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_dofs=*/false);

  dealii::SparsityPattern sp;
  sp.copy_from(dsp);

  // Reinit matrices ONCE with this pattern
  SparseMatrix<double> M, K;
  M.reinit(sp);
  K.reinit(sp);

  // Assemble (assemblers no longer reinit internally)
  assemble_mass_matrix_fe<2>(dof_handler, M);
  assemble_stiffness_matrix_fe<2>(dof_handler, K);

  // Apply constraints after assembly 
  constraints.condense(M);
  constraints.condense(K);  



  // -------------------- initial conditions --------------------


  Vector<double> u_n(dof_handler.n_dofs()); //displacement
  Vector<double> v_n(dof_handler.n_dofs()); //velocity

  class InitialDisplacement : public Function<2>
  {
  public:
    double value(const Point<2> &p, const unsigned int = 0) const override
    {
      return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
    }
  };

  VectorTools::interpolate(dof_handler, InitialDisplacement(), u_n);
  v_n = 0.0;

  // Enforce BC on Initial conditions
  constraints.distribute(u_n);
  constraints.distribute(v_n);





 
 // -------------------- time parameters --------------------

  // Safety checks FIRST (so we don't divide by zero)
  if (!(dt > 0.0)) { std::cerr << "Invalid dt\n"; return 2; }
  if (!(c  > 0.0)) { std::cerr << "Invalid c\n"; return 2; }

  // If you want one period for any c:
  const double T_final = std::sqrt(2.0) / c;

  const unsigned int n_steps =
    static_cast<unsigned int>(std::round(T_final / dt));
  if (n_steps == 0) {
    std::cerr << "n_steps computed as 0 (dt too large for T_final)\n";
    return 2;
  }







  // -------------------- build S = (2/dt)M + (dt/2)c^2 K --------------------


  SparseMatrix<double> S;
  S.reinit(M.get_sparsity_pattern());
  S.copy_from(M);
  S *= (2.0 / dt);
  S.add(0.5 * dt * c * c, K);

  // Apply constraints on S
  constraints.condense(S);






  // -------------------- Make M and S invertible for the direct solver (pin constrained DoFs) --------------------

  // auto pin_row_col = [](dealii::SparseMatrix<double> &A,
  //                       dealii::types::global_dof_index j)
  // {
  //   // zero row j
  //   for (auto it = A.begin(j); it != A.end(j); ++it)
  //     it->value() = 0.0;

  //   // zero column j
  //   const auto n = A.m();
  //   for (dealii::types::global_dof_index i = 0; i < n; ++i)
  //     A.set(i, j, 0.0);

  //   // set diagonal to 1
  //   A.set(j, j, 1.0);
  // };

  // for (const auto &line : constraints.get_lines())
  // {
  //   const auto j = line.index;
  //   pin_row_col(M, j);
  //   pin_row_col(S, j);
  // }


    
  // -------------------- Intiialize lhs operators --------------------

  SparseDirectUMFPACK solver_M;
  SparseDirectUMFPACK solver_S;
  solver_M.initialize(M);
  solver_S.initialize(S);





  // -------------------- time loop work vectors --------------------

  const unsigned int N = dof_handler.n_dofs();
  Vector<double> rhs(N), tmp(N), u_np1(N), v_np1(N);



  // -------------------- time stepping --------------------
  double time = 0.0;
  auto t_start = std::chrono::steady_clock::now();

  for (unsigned int n = 0; n < n_steps; ++n)
  {
    // rhs = (2/dt) M u^n
    M.vmult(rhs, u_n);
    rhs *= (2.0 / dt);

    // rhs += -(dt/2) c^2 K u^n
    K.vmult(tmp, u_n);
    rhs.add(-0.5 * dt * c * c, tmp);

    // rhs += 2 M v^n
    M.vmult(tmp, v_n);
    rhs.add(2.0, tmp);

    constraints.set_zero(rhs);

    // Solve S u^{n+1} = rhs
    solver_S.vmult(u_np1, rhs);

    // Recover v^{n+1}:
    // M v^{n+1} = (2/dt)(M u^{n+1} - M u^n - (dt/2) M v^n)
    M.vmult(rhs, u_np1);         
    M.vmult(tmp, u_n);            
    rhs.add(-1.0, tmp);          
    M.vmult(tmp, v_n);            
    rhs.add(-0.5 * dt, tmp);      
    rhs *= (2.0 / dt);            

    constraints.set_zero(rhs);
    solver_M.vmult(v_np1, rhs);

    constraints.distribute(u_np1);
    constraints.distribute(v_np1);

    u_n = u_np1;
    v_n = v_np1;

    time += dt;

    if ((n + 1) % 100 == 0)
      std::cout << "Step " << (n + 1) << "/" << n_steps << " done.\n";
}
time = n_steps * dt; 
auto t_end = std::chrono::steady_clock::now();
const long long elapsed_ms =
std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

// -------------------- write final-time VTU only --------------------
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(u_n, "u"); // displacement at final time
  data_out.add_data_vector(v_n, "v"); // velocity at final time

  DataOutBase::VtkFlags flags;
  flags.time  = time;        // store the current simulation time
  flags.cycle = n_steps;     // store the time step index
  data_out.set_flags(flags);

  data_out.build_patches(); 

  std::ofstream out("final_solution.vtu"); //Open file
  data_out.write_vtu(out); //Write solutin
  std::cout << "Wrote final_solution.vtu at t = " << time << "\n"; //Print message
}







// -------------------- Error at final time (threaded via deal.II) --------------------
{
  class ExactU_t : public Function<2> {
  public:
    ExactU_t(double t, double c) : Function<2>(1), t_(t), c_(c) {}
    double value(const Point<2>& p, const unsigned int = 0) const override {
      return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) *
            std::cos(numbers::PI * std::sqrt(2.0) * c_ * t_);
    }
  private:
    double t_, c_;
  } exact(time, c);

  // Per-cell error contributions ( in parallel)
  Vector<double> error_per_cell(triangulation.n_active_cells());

  // Compute L2 error;  parallelized internally (TBB)
  const QGauss<2> q(5);
  VectorTools::integrate_difference(dof_handler,
                                    u_n,
                                    exact,
                                    error_per_cell,
                                    q,
                                    VectorTools::L2_norm);

  // Sum of cell contributions 
  const double L2 = VectorTools::compute_global_error(triangulation,
                                                      error_per_cell,
                                                      VectorTools::L2_norm);

  const double u_exact_L2 =
    0.5 * std::abs(std::cos(numbers::PI * std::sqrt(2.0) * c * time));
  const double rel_L2 = (u_exact_L2 > 0) ? (L2 / u_exact_L2) : 0.0;

  std::cout << "SUMMARY "
          << "h=" << mesh_size
          << " dofs=" << dof_handler.n_dofs()
          << " steps=" << n_steps
          << " c=" << c
          << " L2_error=" << L2
          << " rel_L2=" << rel_L2
          << " elapsed_ms=" << elapsed_ms
          << "\n";
}


return 0;

}