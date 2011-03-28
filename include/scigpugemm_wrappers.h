/*! \file
  This is a C header file for the SciGPU-GEMM library.
  It describes the wrapper functions, which are callable from C code.
  A CUDA context <strong>must exist</strong> prior to calling any of these routines.

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

#ifndef SciGPU_GEMM_WRAPPERS_H
#define SciGPU_GEMM_WRAPPERS_H

#if defined(__cplusplus)
extern "C" {
#endif


  //! Wrapper for SciGPUgemm::GEMMcleaver::sgemm
  void sgemm_cleaver( const char transA, const char transB,
		      const int m, const int n, const int k,
		      const float alpha,
		      const float *A, const int lda,
		      const float *B, const int ldb,
		      const float beta,
		      float *C, const int ldc );

  //! Wrapper for SciGPUgemm::GEMMcleaver::dgemm
  void dgemm_cleaver( const char transA, const char transB,
		      const int m, const int n, const int k,
		      const double alpha,
		      const double *A, const int lda,
		      const double *B, const int ldb,
		      const double beta,
		      double *C, const int ldc );

  //! Wrapper for SciGPUgemm::GEMMcleaver::cgemm
  void cgemm_cleaver( const char transA, const char transB,
		      const int m, const int n, const int k,
		      const void *alpha,
		      const void *A, const int lda,
		      const void *B, const int ldb,
		      const void *beta,
		      void *C, const int ldc );

  //! Wrapper for SciGPUgemm::GEMMcleaver::zgemm
  void zgemm_cleaver( const char transA, const char transB,
		      const int m, const int n, const int k,
		      const void *alpha,
		      const void *A, const int lda,
		      const void *B, const int ldb,
		      const void *beta,
		      void *C, const int ldc );

  //! Wrapper for SciGPUgemm::MGEMM::mgemm
  void mgemm( const char transA, const char transB,
	      const int m, const int n, const int k,
	      const double alpha,
	      const double *A, const int lda,
	      const double *B, const int ldb,
	      const double beta,
	      double *C, const int ldc,
	      const double cutOff );

#if defined(__cplusplus)
};
#endif



#endif
