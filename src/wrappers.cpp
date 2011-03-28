/*! \file
  This file contains C wrappers for the SciGPU-GEMM routines.
  A CUDA context <strong>must exist</strong> before any of these routines are called.
  However, no further initialisation is required.

  \section Copyright

  SciGPU-GEMM is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  SciGPU-GEMM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with SciGPU-GEMM.  If not, see <http://www.gnu.org/licenses/>.

  &copy; Copyright 2009 President and Fellos of Harvard College
*/

#include <complex>
using namespace std;

#include "scigpugemm.hpp"
using namespace SciGPUgemm;


#include "scigpugemm_wrappers.h"


// ============================================================

void sgemm_cleaver( const char transA, const char transB,
		    const int m, const int n, const int k,
		    const float alpha,
		    const float *A, const int lda,
		    const float *B, const int ldb,
		    const float beta,
		    float *C, const int ldc ) {
  /*!
    This is a C wrapper for SciGPUgemm::GEMMcleaver::sgemm.
    Its arguments follow those of the \c GEMM routines from BLAS, and assumes
    column-major ordering.
    It performs the operation
    \f[
    C \leftarrow \alpha \cdot o(A) \cdot o(B) + \beta C
    \f]
    where \f$o(X)\f$ is either \f$X\f$ or its transpose.
    This is set by the corresponding tranposition argument, which can be '\c N' or '\c n'
    for no transposition, or '\c T', '\c t', '\c C' or '\c c' for transposition.
    The matrix \f$o(A)\f$ is of size \f$m\times k\f$,
    \f$o(B)\f$ is a \f$k\times n\f$ matrix, and
    \f$C\f$ is an \f$m\times n\f$ matrix.
    The leading dimension of each matrix is specificed by \a ld{a|b|c}.
    These are not redundant with \a m, \a n and \a k since the input matrices may not be
    fully dense (this allows in-place slicing).
    However, the leading dimensions of a matrix must be at least as great as its height
    (recall that we are using column-major ordering).
      
    @see SciGPUgemm::GEMMcleaver::sgemm
    @param[in] transA Whether to transpose matrix \a A
    @param[in] transB Whether to transpose matrix \a B
    @param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
    @param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
    @param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
    @param[in] alpha The value of \f$\alpha\f$
    @param[in] A Pointer to the first element of matrix \a A
    @param[in] lda The size of the leading dimension of \a A
    @param[in] B Pointer to the first element of matrix \a B
    @param[in] ldb The size of the leading dimension of \a B
    @param[in] beta The value of \f$\beta\f$
    @param[in,out] C Pointer to the first element of matrix \a C
    @param[in] ldc The size of the leading dimension of \a C
  */

  GEMMcleaver myCleaver;

  myCleaver.sgemm( transA, transB,
		   m, n, k,
		   alpha, A, lda, B, ldb,
		   beta, C, ldc );
}




// ============================================================

void dgemm_cleaver( const char transA, const char transB,
		    const int m, const int n, const int k,
		    const double alpha,
		    const double *A, const int lda,
		    const double *B, const int ldb,
		    const double beta,
		    double *C, const int ldc ) {
  /*!
    This is a C wrapper for SciGPUgemm::GEMMcleaver::dgemm.
    Its arguments follow those of the \c GEMM routines from BLAS, and assumes
    column-major ordering.
    It performs the operation
    \f[
    C \leftarrow \alpha \cdot o(A) \cdot o(B) + \beta C
    \f]
    where \f$o(X)\f$ is either \f$X\f$ or its transpose.
    This is set by the corresponding tranposition argument, which can be '\c N' or '\c n'
    for no transposition, or '\c T', '\c t', '\c C' or '\c c' for transposition.
    The matrix \f$o(A)\f$ is of size \f$m\times k\f$,
    \f$o(B)\f$ is a \f$k\times n\f$ matrix, and
    \f$C\f$ is an \f$m\times n\f$ matrix.
    The leading dimension of each matrix is specificed by \a ld{a|b|c}.
    These are not redundant with \a m, \a n and \a k since the input matrices may not be
    fully dense (this allows in-place slicing).
    However, the leading dimensions of a matrix must be at least as great as its height
    (recall that we are using column-major ordering).
      
    @see SciGPUgemm::GEMMcleaver::dgemm
    @param[in] transA Whether to transpose matrix \a A
    @param[in] transB Whether to transpose matrix \a B
    @param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
    @param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
    @param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
    @param[in] alpha The value of \f$\alpha\f$
    @param[in] A Pointer to the first element of matrix \a A
    @param[in] lda The size of the leading dimension of \a A
    @param[in] B Pointer to the first element of matrix \a B
    @param[in] ldb The size of the leading dimension of \a B
    @param[in] beta The value of \f$\beta\f$
    @param[in,out] C Pointer to the first element of matrix \a C
    @param[in] ldc The size of the leading dimension of \a C
  */

  GEMMcleaver myCleaver;

  myCleaver.dgemm( transA, transB,
		   m, n, k,
		   alpha, A, lda, B, ldb,
		   beta, C, ldc );
}


// ============================================================

void cgemm_cleaver( const char transA, const char transB,
		    const int m, const int n, const int k,
		    const void *alpha,
		    const void *A, const int lda,
		    const void *B, const int ldb,
		    const void *beta,
		    void *C, const int ldc ) {
  /*!
    This is a C wrapper for SciGPUgemm::GEMMcleaver::cgemm.
    Its arguments follow those of the \c GEMM routines from BLAS, and assumes
    column-major ordering.
    It performs the operation
    \f[
    C \leftarrow \alpha \cdot o(A) \cdot o(B) + \beta C
    \f]
    where \f$o(X)\f$ is either \f$X\f$, \f$X^{T}\f$ or \f$X^{\dagger}\f$.
    This is set by the corresponding tranposition argument, which can be '\c N' or '\c n'
    for no transposition; or '\c T' for '\c t' transposition, or  '\c C' or '\c c'
    for Hermitian conjugation.
    The matrix \f$o(A)\f$ is of size \f$m\times k\f$,
    \f$o(B)\f$ is a \f$k\times n\f$ matrix, and
    \f$C\f$ is an \f$m\times n\f$ matrix.
    The leading dimension of each matrix is specificed by \a ld{a|b|c}.
    These are not redundant with \a m, \a n and \a k since the input matrices may not be
    fully dense (this allows in-place slicing).
    However, the leading dimensions of a matrix must be at least as great as its height
    (recall that we are using column-major ordering).
      
    @see SciGPUgemm::GEMMcleaver::cgemm
    @param[in] transA Whether to transpose matrix \a A
    @param[in] transB Whether to transpose matrix \a B
    @param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
    @param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
    @param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
    @param[in] alpha The value of \f$\alpha\f$
    @param[in] A Pointer to the first element of matrix \a A
    @param[in] lda The size of the leading dimension of \a A
    @param[in] B Pointer to the first element of matrix \a B
    @param[in] ldb The size of the leading dimension of \a B
    @param[in] beta The value of \f$\beta\f$
    @param[in,out] C Pointer to the first element of matrix \a C
    @param[in] ldc The size of the leading dimension of \a C
  */

  GEMMcleaver myCleaver;

  myCleaver.cgemm( transA, transB,
		   m, n, k,
		   *reinterpret_cast< const complex<float>* >(alpha),
		   reinterpret_cast< const complex<float>* >(A), lda,
		   reinterpret_cast< const complex<float>* >(B), ldb,
		   *reinterpret_cast< const complex<float>* >(beta),
		   reinterpret_cast< complex<float>* >(C), ldc );
}



// ============================================================

void zgemm_cleaver( const char transA, const char transB,
		    const int m, const int n, const int k,
		    const void *alpha,
		    const void *A, const int lda,
		    const void *B, const int ldb,
		    const void *beta,
		    void *C, const int ldc ) {
  /*!
    This is a C wrapper for SciGPUgemm::GEMMcleaver::zgemm.
    Its arguments follow those of the \c GEMM routines from BLAS, and assumes
    column-major ordering.
    It performs the operation
    \f[
    C \leftarrow \alpha \cdot o(A) \cdot o(B) + \beta C
    \f]
    where \f$o(X)\f$ is either \f$X\f$, \f$X^{T}\f$ or \f$X^{\dagger}\f$.
    This is set by the corresponding tranposition argument, which can be '\c N' or '\c n'
    for no transposition; or '\c T' for '\c t' transposition, or  '\c C' or '\c c'
    for Hermitian conjugation.
    The matrix \f$o(A)\f$ is of size \f$m\times k\f$,
    \f$o(B)\f$ is a \f$k\times n\f$ matrix, and
    \f$C\f$ is an \f$m\times n\f$ matrix.
    The leading dimension of each matrix is specificed by \a ld{a|b|c}.
    These are not redundant with \a m, \a n and \a k since the input matrices may not be
    fully dense (this allows in-place slicing).
    However, the leading dimensions of a matrix must be at least as great as its height
    (recall that we are using column-major ordering).
      
    @see SciGPUgemm::GEMMcleaver::zgemm
    @param[in] transA Whether to transpose matrix \a A
    @param[in] transB Whether to transpose matrix \a B
    @param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
    @param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
    @param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
    @param[in] alpha The value of \f$\alpha\f$
    @param[in] A Pointer to the first element of matrix \a A
    @param[in] lda The size of the leading dimension of \a A
    @param[in] B Pointer to the first element of matrix \a B
    @param[in] ldb The size of the leading dimension of \a B
    @param[in] beta The value of \f$\beta\f$
    @param[in,out] C Pointer to the first element of matrix \a C
    @param[in] ldc The size of the leading dimension of \a C
  */

  GEMMcleaver myCleaver;

  myCleaver.zgemm( transA, transB,
		   m, n, k,
		   *reinterpret_cast< const complex<double>* >(alpha),
		   reinterpret_cast< const complex<double>* >(A), lda,
		   reinterpret_cast< const complex<double>* >(B), ldb,
		   *reinterpret_cast< const complex<double>* >(beta),
		   reinterpret_cast< complex<double>* >(C), ldc );
}




// ============================================================

void mgemm( const char transA, const char transB,
	    const int m, const int n, const int k,
	    const double alpha,
	    const double *A, const int lda,
	    const double *B, const int ldb,
	    const double beta,
	    double *C, const int ldc,
	    const double cutOff ) {
  /*!
    This is a C wrapper for SciGPUgemm::MGEMM::mgemm. 
    Its arguments follow those of the \c GEMM routines from BLAS, and assumes
    column-major ordering.
    It performs the operation
    \f[
    C \leftarrow \alpha \cdot o(A) \cdot o(B) + \beta C
    \f]
    where \f$o(X)\f$ is either \f$X\f$ or its transpose.
    This is set by the corresponding tranposition argument, which can be '\c N' or '\c n'
    for no transposition, or '\c T', '\c t', '\c C' or '\c c' for transposition.
    The matrix \f$o(A)\f$ is of size \f$m\times k\f$,
    \f$o(B)\f$ is a \f$k\times n\f$ matrix, and
    \f$C\f$ is an \f$m\times n\f$ matrix.
    The leading dimension of each matrix is specificed by \a ld{a|b|c}.
    These are not redundant with \a m, \a n and \a k since the input matrices may not be
    fully dense (this allows in-place slicing).
    However, the leading dimensions of a matrix must be at least as great as its height
    (recall that we are using column-major ordering).
    
    The input matrices are split into `small' and `large' portions.
    The `small' portions are multiplied in single precision on the GPU, while the `large'
    portions are handled on the GPU in double precision.
    The split is made by comparing the absolute value of each element to \a cutOff

    @see SciGPUgemm::MGEMM::mgemm
    @param[in] transA Whether to transpose matrix \a A
    @param[in] transB Whether to transpose matrix \a B
    @param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
    @param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
    @param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
    @param[in] alpha The value of \f$\alpha\f$
    @param[in] A Pointer to the first element of matrix \a A
    @param[in] lda The size of the leading dimension of \a A
    @param[in] B Pointer to the first element of matrix \a B
    @param[in] ldb The size of the leading dimension of \a B
    @param[in] beta The value of \f$\beta\f$
    @param[in,out] C Pointer to the first element of matrix \a C
    @param[in] ldc The size of the leading dimension of \a C
    @param[in] cutOff The value used to determine whether an element is `large' or `small'
    
  */
  
  MGEMM myMulti;

  myMulti.mgemm( transA, transB,
		 m, n, k,
		 alpha, A, lda, B, ldb,
		 beta, C, ldc,
		 cutOff );
}




