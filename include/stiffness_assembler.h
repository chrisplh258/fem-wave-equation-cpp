#pragma once
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>

template <int dim>
void assemble_stiffness_matrix_fe(const dealii::DoFHandler<dim> &dof_handler,
                                  dealii::SparseMatrix<double>  &K);
