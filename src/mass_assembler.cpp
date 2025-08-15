#include "mass_assembler.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

//Dimension
template <int dim>


void assemble_mass_matrix_fe(const dealii::DoFHandler<dim> &dof_handler,
                             dealii::SparseMatrix<double>  &M,
                             const double rho)
{
  const auto &fe = dof_handler.get_fe();

  //gauss points and weights
  dealii::QGauss<dim> quad(fe.degree + 1);

  //Shape function values at Gauss points and |detJ|*weight from quad
  dealii::FEValues<dim> fe_values(fe, quad,
                                  dealii::update_values | dealii::update_JxW_values);

  //dofs per cell
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  //Total number pf quadrature points per element
  const unsigned int n_q = quad.size();
  //total dofs
  const unsigned int n_dofs = dof_handler.n_dofs();

  // Create sparsity pattern for the mass matrix based on mesh connectivity
  dealii::DynamicSparsityPattern sparsity_pattern(n_dofs);
  dealii::DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);

  // Initialize the sparse mass matrix using this pattern
  M.reinit(sparsity_pattern);

  dealii::FullMatrix<double> cell(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);

  //Loop over elements
  for (const auto &cell_it : dof_handler.active_cell_iterators())
  {

    // Update data for each element
    fe_values.reinit(cell_it);
    cell = 0.0;

    ///////////////// Build local mass matrix ///////////////
    //Loop over quadrature (Gauss) points
    for (unsigned int q=0; q<n_q; ++q)
      //Loop over local row DOFs (test functions)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      // Loop over local column DOFs (trial functions)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          cell(i,j) += rho *
                       fe_values.shape_value(i,q) *
                       fe_values.shape_value(j,q) *
                       fe_values.JxW(q);

    cell_it->get_dof_indices(dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        M.add(dof_indices[i], dof_indices[j], cell(i,j));
  }

  M.compress();
}

// Compile for 2D
template void assemble_mass_matrix_fe<2>(const dealii::DoFHandler<2>&,
                                         dealii::SparseMatrix<double>&,
                                         const double);