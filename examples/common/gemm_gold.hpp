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

#ifndef GEMM_GOLD_H
#define GEMM_GOLD_H

#include <iostream>
#include <cstdlib>
#include <typeinfo>
#include <complex>

// Use CBLAS from ATLAS
// Have to warn compiler that ATLAS is in C
extern "C" {
#include <cblas.h>
};




// Dispatch routine for GEMM
// Assumes column major ordering
// This is specialised in the corresponding C++ file as well
template<typename T> void RunGEMM( const enum CBLAS_TRANSPOSE TransA,
				   const enum CBLAS_TRANSPOSE TransB,
				   const int M, const int N, const int K,
				   const T alpha,
				   const T *A, const int lda,
				   const T *B, const int ldb,
				   const T beta,
				   T *C, const int ldc ) {
  // Catch unimplemented types
  std::cerr << __PRETTY_FUNCTION__ << ": Unimplemented GEMM call for type " << typeid(alpha).name() << std::endl;
  exit( EXIT_FAILURE );
}


// Function declarations of specialised template
template<> void RunGEMM<double>( const enum CBLAS_TRANSPOSE TransA,
				 const enum CBLAS_TRANSPOSE TransB,
				 const int M, const int N, const int K,
				 const double alpha,
				 const double *A, const int lda,
				 const double *B, const int ldb,
				 const double beta,
				 double *C, const int ldc );

template<> void RunGEMM<float>( const enum CBLAS_TRANSPOSE TransA,
				const enum CBLAS_TRANSPOSE TransB,
				const int M, const int N, const int K,
				const float alpha,
				const float *A, const int lda,
				const float *B, const int ldb,
				const float beta,
				float *C, const int ldc );

template<> void RunGEMM< std::complex<float> >( const enum CBLAS_TRANSPOSE TransA,
						const enum CBLAS_TRANSPOSE TransB,
						const int M, const int N, const int K,
						const std::complex<float> alpha,
						const std::complex<float> *A, const int lda,
						const std::complex<float> *B, const int ldb,
						const std::complex<float> beta,
						std::complex<float> *C, const int ldc );



template<> void RunGEMM< std::complex<double> >( const enum CBLAS_TRANSPOSE TransA,
						 const enum CBLAS_TRANSPOSE TransB,
						 const int M, const int N, const int K,
						 const std::complex<double> alpha,
						 const std::complex<double> *A, const int lda,
						 const std::complex<double> *B, const int ldb,
						 const std::complex<double> beta,
						 std::complex<double> *C, const int ldc );



template<typename T> void gemm_gold( const T *A, const T *B, T *C,
				     const T alpha, const T beta,
				     const char transA, const char transB,
				     const int m, const int n, const int k,
				     const int lda, const int ldb, const int ldc ) {
  // Wrapper function for running gemm
  // Assumes column major ordering
  
  enum CBLAS_TRANSPOSE tA, tB;

  switch( transA ) {
  case 'N':
  case 'n':
    tA = CblasNoTrans;
    break;

  case 'T':
  case 't':
    tA = CblasTrans;
    break;

  case 'C':
  case 'c':
    tA = CblasConjTrans;
    break;

  default:
    std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transA" << std::endl;
    exit( EXIT_FAILURE );
  }

  switch( transB ) {
  case 'N':
  case 'n':
    tB = CblasNoTrans;
    break;

  case 'T':
  case 't':
    tB = CblasTrans;
    break;

  case 'C':
  case 'c':
    tB = CblasConjTrans;
    break;

  default:
    std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transB" << std::endl;
    exit( EXIT_FAILURE );
  }


  RunGEMM( tA, tB,
	   m, n, k,
	   alpha, A, lda, B, ldb,
	   beta, C, ldc );
}



#endif
