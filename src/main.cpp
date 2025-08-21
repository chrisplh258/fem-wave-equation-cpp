#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/fe/fe_q.h>             
#include <deal.II/dofs/dof_handler.h>     
#include <deal.II/dofs/dof_tools.h> 
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>

#include <cmath> 
#include <fstream>
#include <iostream>

#include "mass_assembler.h"
#include "stiffness_assembler.h"

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_direct.h>


using namespace dealii;


int main()
{

                                                                        /////// Create the mesh ////////

                                
// Set Dimensions - 2
Triangulation<2> triangulation;  

// Set type of domain - Unit Square
GridGenerator::hyper_cube(triangulation, 0, 1); 

// Define the mesh step
const double mesh_size = 0.05;

//Refinement level, given the mesh step
const unsigned int refinement_level = static_cast<unsigned int>(
        std::ceil(std::log2(1.0 / mesh_size)));

//Refine
triangulation.refine_global(refinement_level);



                                                                        ////////// Define Finite Element Space //////////



    // Type of basis functions - 1
    FE_Q<2> fe(1);

    // Link DoFHandler to mesh
    DoFHandler<2> dof_handler(triangulation);
    
    // Assign DoFs
    dof_handler.distribute_dofs(fe);

    std::cout << "Number of degrees of freedom: " 
              << dof_handler.n_dofs() << std::endl;





                                                                        ////////// Assemble global matrices //////////

SparseMatrix<double> M, K;
assemble_mass_matrix_fe<2>(dof_handler, M);    
assemble_stiffness_matrix_fe<2>(dof_handler, K);


                                                                        ////////// Apply Homogeneous Dirichlet BC //////////

AffineConstraints<double> constraints;
VectorTools::interpolate_boundary_values(
    dof_handler, 0, Functions::ZeroFunction<2>(), constraints);
constraints.close();

constraints.condense(M);
constraints.condense(K);


}




                                                                        ////////// Create solution vectors and Apply initial conditions //////////

// Two solution vectors: displacement u and velocity v
Vector<double> u_n(dof_handler.n_dofs()); 
Vector<double> v_n(dof_handler.n_dofs()); 

// Initial displacement 
class InitialDisplacement : public Function<2>
{
public:
  double value(const Point<2> &p, const unsigned int = 0) const override
  {
    // u0(x,y) = sin(pi x) sin(pi y)
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }
};

//Interpolate
VectorTools::interpolate(dof_handler, InitialDisplacement(), u_n);


// Initial velocity
v_n = 0.0;

// Enforce Dirichlet BC
constraints.distribute(u_n);
constraints.distribute(v_n);





                                                                            ////////// Parameters //////////
const double c  = 1.0;      // wave speed
const double dt = 5e-4;     // time step 
const unsigned int n_steps = 1000;





                                                                            ////////// Create Block Matrix System //////////


