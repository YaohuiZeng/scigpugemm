// 'Gold Standard' for GEMM
// Uses ATLAS, and assumes column major ordering

/*
  Copyright

  (c) Copyright 2009 President and Fellows of Harvard College

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
*/

#include <complex>
using namespace std;

#include "gemm_gold.hpp"



// ===============================================================

// Specialisations of the RunGEMM function

template<> void RunGEMM<double>( const enum CBLAS_TRANSPOSE TransA,
				 const enum CBLAS_TRANSPOSE TransB,
				 const int M, const int N, const int K,
				 const double alpha,
				 const double *A, const int lda,
				 const double *B, const int ldb,
				 const double beta,
				 double *C, const int ldc ) {
  // Call DGEMM
  cblas_dgemm( CblasColMajor, TransA, TransB,
	       M, N, K,
	       alpha,
	       A, lda, B, ldb,
	       beta,
	       C, ldc );
}



template<> void RunGEMM<float>( const enum CBLAS_TRANSPOSE TransA,
				const enum CBLAS_TRANSPOSE TransB,
				const int M, const int N, const int K,
				const float alpha,
				const float *A, const int lda,
				const float *B, const int ldb,
				const float beta,
				float *C, const int ldc ) {
  // Call DGEMM
  cblas_sgemm( CblasColMajor, TransA, TransB,
	       M, N, K,
	       alpha,
	       A, lda, B, ldb,
	       beta,
	       C, ldc );
}



template<> void RunGEMM< complex<float> >( const enum CBLAS_TRANSPOSE TransA,
					   const enum CBLAS_TRANSPOSE TransB,
					   const int M, const int N, const int K,
					   const complex<float> alpha,
					   const complex<float> *A, const int lda,
					   const complex<float> *B, const int ldb,
					   const complex<float> beta,
					   complex<float> *C, const int ldc ) {
  // Call CGEMM
  cblas_cgemm( CblasColMajor, TransA, TransB,
	       M, N, K,
	       &alpha,
	       A, lda, B, ldb,
	       &beta,
	       C, ldc );
}


template<> void RunGEMM< complex<double> >( const enum CBLAS_TRANSPOSE TransA,
					    const enum CBLAS_TRANSPOSE TransB,
					    const int M, const int N, const int K,
					    const complex<double> alpha,
					    const complex<double> *A, const int lda,
					    const complex<double> *B, const int ldb,
					    const complex<double> beta,
					    complex<double> *C, const int ldc ) {
  // Call ZGEMM
  cblas_zgemm( CblasColMajor, TransA, TransB,
	       M, N, K,
	       &alpha,
	       A, lda, B, ldb,
	       &beta,
	       C, ldc );
}
