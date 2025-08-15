#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>


template <int dim>

// rho : constant density (default 1.0).
void assemble_mass_matrix_fe(const dealii::DoFHandler<dim> &dof_handler,
                             dealii::SparseMatrix<double>  &M,
                             const double rho = 1.0);