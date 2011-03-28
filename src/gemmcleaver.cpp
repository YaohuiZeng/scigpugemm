/*! \file
  File containing implementation of SciGPUgemm::GEMMcleaver functions

  \section Copyright

  This file is part of the SciGPU-GEMM Library

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

  &copy; Copyright 2009 President and Fellows of Harvard College
*/

#include <iostream>
#include <cstdio>

#include <cuda.h>

#include "gpuerror.hpp"
#include "densematrix.hpp"
#include "gemmcleaver.hpp"


namespace SciGPUgemm {

  // =========================================================
  // SGEMM
  
  void GEMMcleaver::sgemm( const char transA, const char transB,
			   const int m, const int n, const int k,
			   const float alpha,
			   const float *A, const int lda,
			   const float *B, const int ldb,
			   const float beta,
			   float *C, const int ldc ) const {
    /*!
      This is the matrix multiplier wrapper routine.
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
      \internal It is a wrapper for the templated method GEMMcleaver::gemm.
      
      \internal @see GEMMcleaver::gemm
      @see \ref cleaving
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
    
    this->gemm<float>( transA, transB, m, n, k,
		       alpha, A, lda, B, ldb,
		       beta, C, ldc );
  }
  
  
  // =========================================================
  // DGEMM
  
  void GEMMcleaver::dgemm( const char transA, const char transB,
			   const int m, const int n, const int k,
			   const double alpha,
			   const double *A, const int lda,
			   const double *B, const int ldb,
			   const double beta,
			   double *C, const int ldc ) const {
    /*!
      This is the matrix multiplier wrapper routine.
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
      \internal It is a wrapper for the templated method GEMMcleaver::gemm.
      
      \internal @see GEMMcleaver::gemm
      @see \ref cleaving
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
    

    this->gemm<double>( transA, transB, m, n, k,
			alpha, A, lda, B, ldb,
			beta, C, ldc );
  }

  // =========================================================
  // CGEMM
  
  void GEMMcleaver::cgemm( const char transA, const char transB,
			   const int m, const int n, const int k,
			   const std::complex<float> alpha,
			   const std::complex<float> *A, const int lda,
			   const std::complex<float> *B, const int ldb,
			   const std::complex<float> beta,
			   std::complex<float> *C, const int ldc ) const {
    /*!
      This is the matrix multiplier wrapper routine.
      Its arguments follow those of the \c GEMM routines from BLAS, and assumes
      column-major ordering.
      It performs the operation
      \f[
      C \leftarrow \alpha \cdot o(A) \cdot o(B) + \beta C
      \f]
      where \f$o(X)\f$ is either \f$X\f$, \f$X^{T}\f$ or \f$X^{\dagger}\f$.
      This is set by the corresponding tranposition argument, which can be '\c N' or '\c n'
      for no transposition; '\c T' or '\c t' for transposition or '\c C' or '\c c' for
      Hermitian conjugation.
      The matrix \f$o(A)\f$ is of size \f$m\times k\f$,
      \f$o(B)\f$ is a \f$k\times n\f$ matrix, and
      \f$C\f$ is an \f$m\times n\f$ matrix.
      The leading dimension of each matrix is specificed by \a ld{a|b|c}.
      These are not redundant with \a m, \a n and \a k since the input matrices may not be
      fully dense (this allows in-place slicing).
      However, the leading dimensions of a matrix must be at least as great as its height
      (recall that we are using column-major ordering).
      \internal It is a wrapper for the templated method GEMMcleaver::gemm.
      
      \internal @see GEMMcleaver::gemm
      @see \ref cleaving
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
    
    this->gemm< std::complex<float> >( transA, transB, m, n, k,
				       alpha, A, lda, B, ldb,
				       beta, C, ldc );
  }

  
  // =========================================================
  // ZGEMM
  
  void GEMMcleaver::zgemm( const char transA, const char transB,
			   const int m, const int n, const int k,
			   const std::complex<double> alpha,
			   const std::complex<double> *A, const int lda,
			   const std::complex<double> *B, const int ldb,
			   const std::complex<double> beta,
			   std::complex<double> *C, const int ldc ) const {
    /*!
      This is the matrix multiplier wrapper routine.
      Its arguments follow those of the \c GEMM routines from BLAS, and assumes
      column-major ordering.
      It performs the operation
      \f[
      C \leftarrow \alpha \cdot o(A) \cdot o(B) + \beta C
      \f]
      where \f$o(X)\f$ is either \f$X\f$, \f$X^{T}\f$ or \f$X^{\dagger}\f$.
      This is set by the corresponding tranposition argument, which can be '\c N' or '\c n'
      for no transposition; '\c T' or '\c t' for transposition or '\c C' or '\c c' for
      Hermitian conjugation.
      The matrix \f$o(A)\f$ is of size \f$m\times k\f$,
      \f$o(B)\f$ is a \f$k\times n\f$ matrix, and
      \f$C\f$ is an \f$m\times n\f$ matrix.
      The leading dimension of each matrix is specificed by \a ld{a|b|c}.
      These are not redundant with \a m, \a n and \a k since the input matrices may not be
      fully dense (this allows in-place slicing).
      However, the leading dimensions of a matrix must be at least as great as its height
      (recall that we are using column-major ordering).
      \internal It is a wrapper for the templated method GEMMcleaver::gemm.
      
      \internal @see GEMMcleaver::gemm
      @see \ref cleaving
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
    
    this->gemm< std::complex<double> >( transA, transB, m, n, k,
					alpha, A, lda, B, ldb,
					beta, C, ldc );
  }
  
  

  // ###########################################################################################

  unsigned long GEMMcleaver::getindex( const unsigned int i, const unsigned int j,
				       const unsigned int ld ) const {
    /*!
      A simple index calculator.
      Returns the 1D index of the element \f$(i,j)\f$ given the size of the leading
      dimension.
      Assumes column-major ordering
      @param[in] i The row of the desired element
      @param[in] j The column of the desired element
      @param[in] ld The leading dimension of the matrix
    */
    return( i+(j*ld) );
  }
  
}
