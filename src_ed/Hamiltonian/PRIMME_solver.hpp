/*
Interface to use PRIMME eigensolver for Exact Diagonalisation
*/

#ifndef PRIMME_solver_h
#define PRIMME_solver_h

#include <iostream>
#include <vector>
#include <algorithm>
#include "global_parameter.hpp"
#include "primme.h"
#include <Eigen/SparseCore>

/**
 Wrapper around dprimme and zprimme routine for type double and complex
 */
template<typename HilbertField>
int call_primme(double* evals, HilbertField* evecs, double* rnorm, primme_params* primme);

/**
 Matrix vector multiplication for Eigen Hamiltonian type
 */
template<typename HilbertField>
void PRIMME_Eigen_matmul(
    void *x, 
    int64_t *ldx,
    void *y,
    int64_t *ldy,
    int *blockSize,
    primme_params *primme,
    int *ierr
) {
     const Eigen::Map< Eigen::Matrix<HilbertField,Eigen::Dynamic,1> > xe((HilbertField*) x, primme->n);
     Eigen::Map< Eigen::Matrix<HilbertField,Eigen::Dynamic,1> > ye((HilbertField*) y, primme->n);
     ye = *((Eigen::SparseMatrix<HilbertField,Eigen::RowMajor>*) primme->matrix) * xe;
     *ierr = 0;
}

/**
 Jacobi preconditionner for Eigen Hamiltonian type
 Perform: Y = M^(-1) X when M = diag(A)
 */
template<typename HilbertField>
void PRIMME_Eigen_Jacobi_preconditionner(
    void *x, 
    int64_t *ldx,
    void *y,
    int64_t *ldy,
    int *blockSize,
    primme_params *primme,
    int *ierr
) {
     const Eigen::Map< Eigen::Array<HilbertField,Eigen::Dynamic,1> > xe((HilbertField*) x, primme->n);
     Eigen::Map< Eigen::Array<HilbertField,Eigen::Dynamic,1> > ye((HilbertField*) y, primme->n);
     const Eigen::Array<HilbertField,Eigen::Dynamic,1> diag = ((Eigen::SparseMatrix<HilbertField,Eigen::RowMajor>*) primme->matrix)->diagonal();
     for (size_t i=0; i<diag.size(); i++) ye[i] = (diag[i] == 0.) ? xe[i] / diag[i] : xe[i];
     *ierr = 0;
}

/**
 Matrix vector multiplication for CSR Hamiltonian type
 */
template<typename HilbertField>
void PRIMME_CSR_matmul(
    void *x, 
    int64_t *ldx,
    void *y,
    int64_t *ldy,
    int *blockSize,
    primme_params *primme,
    int *ierr
) {
    //TODO
}

/**
 Compute the lowest eigenpair(s) of the Hamiltonian using PRIMME eigensolver.
 The number of requested eigenpairs is given by numEvals.
 evals and evecs must be pre-sized to numEvals; evecs[0] is used as the initial guess.
 */
template<typename T, typename HilbertField>
void PRIMME_state_solver(
    T* H, //hamiltonian in its different format
    const size_t &dim,
    const size_t numEvals, //number of wanted eigenpairs
    std::vector<double> &evals, //the returned eigenvalues (pre-sized to numEvals)
    std::vector<std::vector<HilbertField>> &evecs, //the returned eigenvectors (pre-sized to numEvals)
    const bool verb=false
) {
    /* Solver arrays and parameters */
    std::vector<double> rnorms(numEvals);   /* Array with the computed eigenpairs residual norms */
    primme_params primme; /* PRIMME configuration struct */

    /* Other miscellaneous items */
    int ret;

    /* Set default values in PRIMME configuration struct */
    primme_initialize(&primme);

   /* Set problem parameters */
   primme.n = (long int) dim; /* set problem dimension */
   primme.numEvals = (int) numEvals;   /* Number of wanted eigenpairs */
   primme.eps = global_double("accur_lanczos"); /* ||r|| <= eps * ||matrix|| */
   primme.target = primme_smallest;  /* Wanted the smallest eigenvalues */
   primme.maxMatvecs = global_int("max_iter_lanczos");
   if (verb) {
       primme.outputFile = stdout;
       primme.printLevel = 5;
   }
   /* Scale basis/block sizes with the number of requested eigenpairs */
   primme.minRestartSize = (int) std::max((size_t) 4, numEvals + 2);
   primme.maxBasisSize   = (int) std::max((size_t) 14, 2*numEvals + 4);
   primme.maxBlockSize   = (int) std::max((size_t) 1, numEvals);

   //The initial vector has already been set to random or previous values, so:
   primme.initSize = 1;

   /* PRIMME expects eigenvectors as a contiguous column-major (n x numEvals) buffer */
   std::vector<HilbertField> evec_buffer(dim * numEvals, HilbertField(0));
   /* Copy the initial guess (in evecs[0]) into the first column */
   std::copy(evecs[0].begin(), evecs[0].end(), evec_buffer.begin());

    /* Set problem matrix */
    if (Hamiltonian_format == H_format_eigen) {
        primme.matrix = H->H_ptr;
        primme.matrixMatvec = PRIMME_Eigen_matmul<HilbertField>;
    }
    else {
        qcm_ED_throw("Diagonalisation of Hamiltonian of chosen type not implemented with PRIMME eigensolver");
    }
    
    /* Set preconditioner (optional) */
    if (global_int("PRIMME_preconditionning") == 0) {}
    else if (global_int("PRIMME_preconditionning") == 1) {
        primme.applyPreconditioner = PRIMME_Eigen_Jacobi_preconditionner<HilbertField>;
        primme.correctionParams.precondition = 1;
    }
    else {
        qcm_ED_throw("Unknown preconditionner");
    }

    /* Set method to solve the problem */
    primme_set_method((primme_preset_method) global_int("PRIMME_algorithm"), &primme);

    /* Call primme */
    ret = call_primme(evals.data(), evec_buffer.data(), rnorms.data(), &primme);

    /* Copy the converged eigenvectors out of the contiguous buffer */
    for (size_t k = 0; k < numEvals; k++) {
        evecs[k].assign(
            evec_buffer.begin() + k*dim,
            evec_buffer.begin() + (k+1)*dim
        );
    }

   /* Reporting (optional) */
   if (verb) {
       for (size_t k = 0; k < numEvals; k++) {
           std::cout << "Eval[" << k << "]: " << evals[k] << ", rnorm: " << rnorms[k] << std::endl;
       }
       std::cout << "Tolerance : " << primme.aNorm*primme.eps << std::endl;
       std::cout << "Iterations : " << primme.stats.numOuterIterations << std::endl;
       std::cout << "Restarts : " << primme.stats.numRestarts << std::endl;
       std::cout << "Matvecs : " << primme.stats.numMatvecs << std::endl;
       std::cout << "Preconds : " << primme.stats.numPreconds << std::endl;
       if (primme.stats.lockingIssue) {
           std::cout << "A locking problem has occurred" << std::endl;
           std::cout << "Some eigenpairs do not have a residual norm less than the tolerance." << std::endl;
           std::cout << "However, the subspace of evecs is accurate to the required tolerance." << std::endl;
       }
   }
   
   primme_free(&primme);

   if (ret != 0) {
      std::string explanation;
      switch (ret) {
         case -1: explanation = "unexpected internal error"; break;
         case -2: explanation = "memory allocation failure"; break;
         case -3: explanation = "reached maxOuterIterations or maxMatvecs (" + std::to_string(global_int("max_iter_lanczos")) + ") before convergence"; break;
         case -4: explanation = "argument primme is NULL"; break;
         case -5: explanation = "primme.n <= 0 or primme.nLocal <= 0 or primme.nLocal > primme.n"; break;
         case -6: explanation = "primme.numProcs < 1"; break;
         case -7: explanation = "primme.matrixMatvec is NULL"; break;
         case -8: explanation = "primme.applyPreconditioner is NULL and precondition is requested"; break;
         case -10: explanation = "primme.numEvals > primme.n"; break;
         case -11: explanation = "primme.numEvals < 0"; break;
         case -17: explanation = "maxBasisSize < 2"; break;
         default:  explanation = "see PRIMME documentation"; break;
      }
      qcm_ED_throw("PRIMME eigensolver failed (exit status " + to_string(ret) + "): " + explanation);
   }
   
}


#endif
