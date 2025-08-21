#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>

void assemble_stiffness_matrix_fe(const dealii::DoFHandler<dim>&,
                                  dealii::SparseMatrix<double>&,
                                  const double c2 = 1.0);