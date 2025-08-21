#include "stiffness_assembler.h"

#include <deal.II/base/quadrature_lib.h>           
#include <deal.II/fe/fe_values.h>                     
#include <deal.II/dofs/dof_tools.h>                   
#include <deal.II/lac/dynamic_sparsity_pattern.h>     
#include <deal.II/lac/full_matrix.h>                  
#include <vector>                                      



template <int dim>
void assemble_stiffness_matrix_fe(const dealii::DoFHandler<dim> &dof_handler,
                                  dealii::SparseMatrix<double>  &K,
                                  const double c2)
{
  const auto &fe = dof_handler.get_fe();

  //gauss points and weights
  dealii::QGauss<dim> quad(fe.degree + 1);

  
  //update_gradients: we need ∇φ_i(x_q), ∇φ_j(x_q)
  //update_JxW_values: we need |detJ| * w_q (volume element)
  dealii::FEValues<dim> fe_values(
      fe, quad,
      dealii::update_gradients |
      dealii::update_JxW_values);

  //dofs per cell
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  //Total number of quadrature points per element
  const unsigned int n_q = quad.size();
  //total dofs
  const unsigned int n_dofs = dof_handler.n_dofs();

  // Create sparsity pattern for the mass matrix based on mesh connectivity
  dealii::DynamicSparsityPattern dsp(n_dofs);
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);

  // Initialize the sparse mass matrix using this pattern
  K.reinit(dsp);

  dealii::FullMatrix<double> cell(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);

  // Loop over elements
  for (const auto &cell_it : dof_handler.active_cell_iterators())
  {
    // Update data for each element
    fe_values.reinit(cell_it);
    cell = 0.0;

    ///////////////// Build local mass matrix ///////////////
    //Loop over quadrature (Gauss) points
    for (unsigned int q = 0; q < n_q; ++q)
    {
      //Loop over local row DOFs (test functions)
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        // Gradient of TEST function i at quadrature point q
        const dealii::Tensor<1,dim> grad_phi_i = fe_values.shape_grad(i, q);

        // Loop over local column DOFs (trial functions)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // Gradient of TRIAL function j at quadrature point q
          const dealii::Tensor<1,dim> grad_phi_j = fe_values.shape_grad(j, q);
          cell(i, j) += c2 *
                        (grad_phi_i * grad_phi_j) *
                        fe_values.JxW(q);
        }
      }
    }

    cell_it->get_dof_indices(dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        K.add(dof_indices[i], dof_indices[j], cell(i, j));
  }

  K.compress();
}

// Compile for 2D
template void assemble_stiffness_matrix_fe<2>(const dealii::DoFHandler<2>&,
                                              dealii::SparseMatrix<double>&,
                                              const double);