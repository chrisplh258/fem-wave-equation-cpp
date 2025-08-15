#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/fe/fe_q.h>              // For FE_Q
#include <deal.II/dofs/dof_handler.h>     // For DoFHandler
#include <deal.II/dofs/dof_tools.h> 

#include <cmath> 
#include <fstream>
#include <iostream>

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







}