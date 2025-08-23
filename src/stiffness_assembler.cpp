#include "stiffness_assembler.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/base/tensor.h>

#include <vector>

using namespace dealii;

template <int dim>
void assemble_stiffness_matrix_fe(const DoFHandler<dim> &dof_handler,
                                  SparseMatrix<double>  &K)
{
  const auto &fe = dof_handler.get_fe();
  QGauss<dim> quad(fe.degree + 1);
  const UpdateFlags flags = update_gradients | update_JxW_values;

  // Thread-local scratch data
  struct Scratch {
    FEValues<dim> fe_values;
    Scratch(const FiniteElement<dim> &fe,
            const Quadrature<dim>    &q,
            const UpdateFlags         f)
      : fe_values(fe, q, f) {}
    Scratch(const Scratch &s)
      : fe_values(s.fe_values.get_fe(),
                  s.fe_values.get_quadrature(),
                  s.fe_values.get_update_flags()) {}
  };

  // Per-cell buffer for serial combine
  struct Copy {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> dof_indices;
    explicit Copy(unsigned int dofs_per_cell)
      : cell_matrix(dofs_per_cell, dofs_per_cell),
        dof_indices(dofs_per_cell) {}
  };

  Scratch scratch(fe, quad, flags);
  Copy    copy(fe.n_dofs_per_cell());

  // Parallel worker on cells
  auto worker = [&](const auto &cell, Scratch &scratch, Copy &copy)
  {
    auto &fe_values = scratch.fe_values;
    fe_values.reinit(cell);

    const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    const unsigned int n_q           = fe_values.n_quadrature_points;

    copy.cell_matrix = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      const double JxW = fe_values.JxW(q);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const Tensor<1,dim> grad_i = fe_values.shape_grad(i, q);
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const Tensor<1,dim> grad_j = fe_values.shape_grad(j, q);
          copy.cell_matrix(i, j) += (grad_i * grad_j) * JxW;
        }
      }
    }

    cell->get_dof_indices(copy.dof_indices);
  };

  // Serial copier: safe to write to global K
  auto copier = [&](const Copy &copy)
  {
    const auto &I = copy.dof_indices;
    for (unsigned int i = 0; i < copy.cell_matrix.m(); ++i)
      for (unsigned int j = 0; j < copy.cell_matrix.n(); ++j)
        K.add(I[i], I[j], copy.cell_matrix(i, j));
  };

  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        worker,
                        copier,
                        scratch,
                        copy,
                        MeshWorker::assemble_own_cells);

  K.compress(VectorOperation::add);
}

// Explicit instantiation for 2D
template void assemble_stiffness_matrix_fe<2>(const DoFHandler<2>&,
                                              SparseMatrix<double>&);