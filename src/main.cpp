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



using namespace dealii;

int main()
{



  // -------------------- mesh --------------------

  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);

  const double       mesh_size        = 0.05;
  const unsigned int refinement_level =
    static_cast<unsigned int>(std::ceil(std::log2(1.0 / mesh_size)));
  triangulation.refine_global(refinement_level);





  // -------------------- FE space --------------------

  FE_Q<2>        fe(1);
  DoFHandler<2>  dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;





 
  // -------------------- homogeneous Dirichlet BC --------------------


  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    dof_handler, 0 /*boundary id*/, Functions::ZeroFunction<2>(), constraints);
  constraints.close();


 // -------------------- assemble M, K --------------------
  
  // Build sparsity WITH constraints and keep constrained dofs present
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_dofs=*/true);

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

  const double       c       = 1.0;
  const double       dt      = 5e-5;
  const unsigned int n_steps = 1000;





  // -------------------- build S = (2/dt)M + (dt/2)c^2 K --------------------


  SparseMatrix<double> S;
  S.reinit(M.get_sparsity_pattern());
  S.copy_from(M);
  S *= (2.0 / dt);
  S.add(0.5 * dt * c * c, K);

  // Apply constraints on S
  constraints.condense(S);






  // -------------------- Make M and S invertible for the direct solver (pin constrained DoFs) --------------------

  auto pin_row_col = [](dealii::SparseMatrix<double> &A,
                        dealii::types::global_dof_index j)
  {
    // zero row j
    for (auto it = A.begin(j); it != A.end(j); ++it)
      it->value() = 0.0;

    // zero column j
    const auto n = A.m();
    for (dealii::types::global_dof_index i = 0; i < n; ++i)
      A.set(i, j, 0.0);

    // set diagonal to 1
    A.set(j, j, 1.0);
  };

  for (const auto &line : constraints.get_lines())
  {
    const auto j = line.index;
    pin_row_col(M, j);
    pin_row_col(S, j);
  }


    
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

return 0;
}