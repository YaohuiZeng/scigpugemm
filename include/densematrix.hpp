/*! \file
  Templated Dense Matrix class, used by SciGPUgemm::GEMMcleaver

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

#ifndef DenseMatrix_H
#define DenseMatrix_H


#include <cstdlib>
#include <iostream>
#include <exception>
#include <typeinfo>
#include <complex>

#include <cuda_runtime.h>
#include <cublas.h>



#include "gpuerror.hpp"


namespace SciGPUgemm {
  //! The DenseMatrix class  holds a full dense matrix in page-locked memory and mirrors it on the GPU
  /*!
    \internal
    This is a class for use by GEMMcleaver.
    It holds a dense matrix (column-major, as CUBLAS assumes) in page-locked
    RAM, and mirrored on the GPU.
    Although memory is allocated simultaneously on the host and GPU, transfers only
    occur when required (via the DenseMatrix::send and DenseMatrix::receive methods).
    It is templated to work with multiple data types, specifically float and double.
    @tparam T The data type for this matrix
  */
  template<typename T>
  class DenseMatrix {

  public:  
    // ---------------------------------------------------------
    
    //! Default constructor creates a null matrix
    DenseMatrix( void ) : h_matrix(NULL),
			  d_matrix(NULL),
			  nRows(0),
			  nCols(0) {
      /*!
	Creates a null dense matrix
      */
#ifdef _DEBUG
      std::cout << __FUNCTION__ << std::endl;
#endif
    }
    


    //! Copy constructor
    DenseMatrix( const DenseMatrix &src ) : nRows(src.nRows),
					    nCols(src.nCols) {
      /*!
	Copies the input dense matrix via CUDA API calls.
	This compiles, but is not tested (since it is not used).
	It exists to keep the compiler happy, and in case future
	extensions to SciGPU-GEMM require it
      */
#ifdef _DEBUG
      std::cout << __FUNCTION__ << std::endl;
#endif

      SciGPU_TRY_CUDA_RUNTIME( cudaMallocHost( (void**)&h_matrix, size() ),
			       GPUexception::eMallocHost );
      
      SciGPU_TRY_CUDA_RUNTIME( cudaMalloc( (void**)&d_matrix, size() ),
			       GPUexception::eMallocGPU );

      SciGPU_TRY_CUDA_RUNTIME( cudaMemcpy( h_matrix, src.h_matrix, size(), cudaMemcpyHostToHost ),
			       GPUexception::eMiscCUDAruntime );

      SciGPU_TRY_CUDA_RUNTIME( cudaMemcpy( d_matrix, src.d_matrix, size(), cudaMemcpyDeviceToDevice ),
			       GPUexception::eMiscCUDAruntime );
    }

    
    // ---------------------------------------------------------
    
    //! Destructor releases any memory associated with this instance
    ~DenseMatrix( void ) throw( GPUexception ) {
      /*!
	Releases the storage allocated both in host pinned memory
	and on the GPU
      */
#ifdef _DEBUG
      std::cout << __FUNCTION__ << std::endl;
#endif
      
      nRows = 0;
      nCols = 0;
      
      // Have to check host and GPU memory separately in case an exception has been thrown
      
      // First pinned host memory
      if( this->h_matrix != NULL ) {
	SciGPU_TRY_CUDA_RUNTIME( cudaFreeHost( this->h_matrix ),
				 GPUexception::eFreeHost );
      }
      
      // Then, GPU memory
      if( this->d_matrix != NULL ) {
	SciGPU_TRY_CUDA_RUNTIME( cudaFree( this->d_matrix ),
				 GPUexception::eFreeGPU );
      }
    }

    
    // ---------------------------------------------------------

    //! Assignment operator
    DenseMatrix& operator=( const DenseMatrix &src ) {
      /*!
	Copies one DenseMatrix to another, with a guard against
	self-assignment.
	This compiles, but is not tested (since it is not used).
	It exists to keep the compiler happy, and in case future
	extensions to SciGPU-GEMM require it
      */
#ifdef _DEBUG
      std::cout << __FUNCTION__ << std::endl;
#endif
      if( this != &src ) {
	// Acquire new memory
	T *h_new, *d_new;

	SciGPU_TRY_CUDA_RUNTIME( cudaMallocHost( (void**)&h_new, src.size() ),
				 GPUexception::eMallocHost );
      
	SciGPU_TRY_CUDA_RUNTIME( cudaMalloc( (void**)&d_new, src.size() ),
				 GPUexception::eMallocGPU );
	
	// Copy values
	SciGPU_TRY_CUDA_RUNTIME( cudaMemcpy( h_new, src.h_matrix, src.size(), cudaMemcpyHostToHost ),
			       GPUexception::eMiscCUDAruntime );

	SciGPU_TRY_CUDA_RUNTIME( cudaMemcpy( d_new, src.d_matrix, src.size(), cudaMemcpyDeviceToDevice ),
			       GPUexception::eMiscCUDAruntime );

	// Release old memory
	SciGPU_TRY_CUDA_RUNTIME( cudaFreeHost( h_matrix ),
				 GPUexception::eFreeHost );
	SciGPU_TRY_CUDA_RUNTIME( cudaFree( d_matrix ),
				 GPUexception::eFreeGPU );

	// Assign to new object
	nRows = src.nRows;
	nCols = src.nCols;

	h_matrix = h_new;
	d_matrix = d_new;

      }

      return *this;
    }
	
   
    
    // ---------------------------------------------------------
    
    //! Creates a dense matrix from a BLAS style matrix
    void compact( const T *A, const unsigned int ld,
		  const unsigned int m, const unsigned int n ) throw( GPUexception ) {
      /*!
	Extracts a dense matrix from the given input BLAS-style matrix.
	A submatrix with of size \f$m \times n\f$ is extracted from the
	matrix \a A.
	No range checking can be performed, so pass \a A carefully
	
	@param[in] A Pointer to the first element of the input matrix
	@param[in] ld The size of the leading dimension of \a A
	@param[in] m The number of rows in the output matrix
	@param[in] n The number of columns in the output matrix
      */
      
      
      // Sanity check
      if( ld < m ) {
	std::cerr << __PRETTY_FUNCTION__ << ": ld < m" << std::endl;
      }
      
      // Allocate our memory
      this->allocate( m, n );
      
      
      // Extract the matrix
      // Recall that we're column major
      for( unsigned int j=0; j<n; j++ ) {
	for( unsigned int i=0; i<m; i++ ) {
	  this->h_matrix[i+(j*m)] = A[i+(j*ld)];
	}
      }
    }
    
    
    // ---------------------------------------------------------
    
    //! Expands a dense matrix to a BLAS style matrix
    void expand( T *A, const unsigned int ld, const T beta ) const throw() {
      /*!
	Converts a dense matrix back to a BLAS-style matrix.
	The mathematical operation performed is
	\f[
	A_{ij} \leftarrow A^{\textrm{dense}}_{ij} + \beta A_{ij}
	\f]
	which follows BLAS conventions.
	Programatically, an expansion back to the leading-dimension form
	of BLAS storage is also performed.
	Again, there's no bounds checking possible here.
	
	@param[in,out] A Pointer to the first element of the matrix \a A
	@param[in] ld The size of the leading dimension of \a A
	@param[in] beta The value of \f$\beta\f$
      */
      
      // Sanity check
      if( ld < nRows ) {
	std::cerr << __PRETTY_FUNCTION__ << ": ld < nRows" << std::endl;
      }
      
      // Do the expansion, special casing beta==0
      // This can avoid a memory load, and also prevent trouble with
      // an uninitialised A containing nan
      if( beta != static_cast<T>(0) ) {
	for( unsigned int j=0; j<nCols; j++ ) {
	  for( unsigned int i=0; i<nRows; i++ ) {
	    A[i+(j*ld)] = this->h_matrix[i+(j*nRows)] + (beta*A[i+(j*ld)]);
	  }
	}
      } else {
	for( unsigned int j=0; j<nCols; j++ ) {
	  for( unsigned int i=0; i<nRows; i++ ) {
	    A[i+(j*ld)] = this->h_matrix[i+(j*nRows)];
	  }
	}
      }
    }
    
    
    // ---------------------------------------------------------
    
    //! Templated dispatch routine to catch unimplemented GEMM calls
    template<typename U> void GEMMDispatch( const char transA,
					    const char transB,
					    const int m, const int n, const int k,
					    const U alpha,
					    const U *A, const int lda,
					    const U *B, const int ldb,
					    const U beta,
					    U *C, const int ldc ) {
      /*!
	This is a dispatch routine for CUBLAS GEMM calls.
	In the unspecialised form, it simply aborts

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
      std::cerr << __PRETTY_FUNCTION__ << std::endl;
      std::cerr << "Unimplemented GEMM call for type ";
      std::cerr << typeid(alpha).name() << std::endl;
      exit( EXIT_FAILURE );
    }

    //! Dispatches SGEMM call on GPU
    void GEMMDispatch( const char transA,
		       const char transB,
		       const int m, const int n, const int k,
		       const float alpha,
		       const float *A, const int lda,
		       const float *B, const int ldb,
		       const float beta,
		       float *C, const int ldc ) {
      /*!
	This overloaded implementation of the GEMMDispatch method
	calls CUBLAS for single precision matrices.
	We use an overload rather than a specialisation in order to
	stop things getting too confusing
      */
      cublasSgemm( transA, transB,
		   m, n, k,
		   alpha,
		   A, lda,
		   B, ldb,
		   beta,
		   C, ldc );
    }


    //! Dispatches CGEMM call on GPU
    void GEMMDispatch( const char transA,
		       const char transB,
		       const int m, const int n, const int k,
		       const std::complex<float> alpha,
		       const std::complex<float> *A, const int lda,
		       const std::complex<float> *B, const int ldb,
		       const std::complex<float> beta,
		       std::complex<float> *C, const int ldc ) {
      /*!
	This overloaded implementation of the GEMMDispatch method
	calls CUBLAS for single precision complex matrices.
	We use an overload rather than a specialisation in order to
	stop things getting too confusing
      */
      const cuFloatComplex localAlpha = make_cuFloatComplex( std::real(alpha), std::imag(alpha) );
      const cuFloatComplex localBeta = make_cuFloatComplex( std::real(beta), std::imag(beta) );

      cublasCgemm( transA, transB,
		   m, n, k,
		   localAlpha,
		   reinterpret_cast<const cuFloatComplex*>(A), lda,
		   reinterpret_cast<const cuFloatComplex*>(B), ldb,
		   localBeta,
		   reinterpret_cast<cuFloatComplex*>(C), ldc );
    }

    
    //! Dispatches DGEMM call on GPU
    void GEMMDispatch( const char transA,
		       const char transB,
		       const int m, const int n, const int k,
		       const double alpha,
		       const double *A, const int lda,
		       const double *B, const int ldb,
		       const double beta,
		       double *C, const int ldc ){
      /*!
	This overloaded implementation of the GEMMDispatch method
	calls CUBLAS for double precision matrices.
	We use an overload rather than a specialisation in order to
	stop things getting too confusing
      */
      cublasDgemm( transA, transB,
		   m, n, k,
		   alpha,
		   A, lda,
		   B, ldb,
		   beta,
		   C, ldc );
    }
    

    //! Dispatches ZGEMM call on GPU
    void GEMMDispatch( const char transA,
		       const char transB,
		       const int m, const int n, const int k,
		       const std::complex<double> alpha,
		       const std::complex<double> *A, const int lda,
		       const std::complex<double> *B, const int ldb,
		       const std::complex<double> beta,
		       std::complex<double> *C, const int ldc ) {
      /*!
	This overloaded implementation of the GEMMDispatch method
	calls CUBLAS for single precision matrices.
	We use an overload rather than a specialisation in order to
	stop things getting too confusing
      */
      const cuDoubleComplex localAlpha = make_cuDoubleComplex( std::real(alpha), std::imag(alpha) );
      const cuDoubleComplex localBeta = make_cuDoubleComplex( std::real(beta), std::imag(beta) );

      cublasZgemm( transA, transB,
		   m, n, k,
		   localAlpha,
		   reinterpret_cast<const cuDoubleComplex*>(A), lda,
		   reinterpret_cast<const cuDoubleComplex*>(B), ldb,
		   localBeta,
		   reinterpret_cast<cuDoubleComplex*>(C), ldc );
    }
    

    //! Performs a matrix multiply
    void multiply( const char transA, const char transB,
		   const T alpha,
		   const DenseMatrix &A, const DenseMatrix &B ) throw( GPUexception ) {
      /*!
	Uses CUBLAS to perform \f$C = \alpha A \cdot B\f$ where C is the matrix on which the
	method is called.
	The input matrices are assumed to be resident on the GPU, and the answer is left there.
	To avoid creating new objects, the input matrices are passed by reference
	
	@param[in] transA Whether to transpose matrix \a A
	@param[in] transB Whether to transpose matrix \a B
	@param[in] alpha The value of \f$\alpha\f$
	@param[in] A The matrix A
	@param[in] B The matrix B
      */
      
      int m, n, k;
      m = n = k = -1;
      
      // Determine the sizes, first for A
      switch( transA ) {
      case 'n':
      case 'N':
	m = A.nRows;
	k = A.nCols;
	break;
	
      case 't':
      case 'T':
      case 'C':
      case 'c':
	m = A.nCols;
	k = A.nRows;
	break;
	
      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transA" << std::endl;
      }
      
      // Second for B
      switch( transB ) {
      case 'n':
      case 'N':
	n = B.nCols;
	if( static_cast<unsigned int>(k) != B.nRows ) {
	  std::cerr << __PRETTY_FUNCTION__ << ": Matrices not conformable" << std::endl;
	}
	break;
	
      case 't':
      case 'T':
      case 'C':
      case 'c':
	n = B.nRows;
	if( static_cast<unsigned int>(k) != B.nCols ) {
	  std::cerr << __PRETTY_FUNCTION__ << ": Matrices not conformable" << std::endl;
	}
	break;
	
      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transB" << std::endl;
      }
      
      // Allocate space for the result
      this->allocate( m, n );
      
      // Run GEMM on the GPU
      GEMMDispatch( transA, transB,
		    m, n, k,
		    alpha,
		    A.d_matrix, A.nRows,
		    B.d_matrix, B.nRows,
		    static_cast<T>(0), // beta
		    this->d_matrix, this->nRows );
      cudaThreadSynchronize();

      // Check up on errors
      cublasStatus cublasErr = cublasGetError();
      if( cublasErr != CUBLAS_STATUS_SUCCESS ) {
#ifdef _DEBUG
	std::cerr << __FUNCTION__ << ": CUBLAS GEMM call failed" << std::endl;
	std::cerr << "Error code was " << cublasErr;
#endif
	SciGPU_GPUexception_THROW( GPUexception::eCUBLAS, cublasErr );
      }
      
    }
    
    
    // -------------------------------------------------
    
    //! Sends the matrix to the GPU
    void send( void ) throw( GPUexception ) {
    /*!
      Uses the CUDA library functions to copy the matrix
      from the CPU to GPU
    */
      SciGPU_TRY_CUDA_RUNTIME( cudaMemcpy( d_matrix,
					   h_matrix,
					   size(),
					   cudaMemcpyHostToDevice ),
			       GPUexception::eTransfer );
    }
    
    // -------------------------------------------------
    
    //! Receives the matrix from the GPU
    void receive( void ) throw( GPUexception ) {
      /*!
	Uses the CUDA library functions to copy the matrix
	from the GPU back to the CPU
      */
      
      SciGPU_TRY_CUDA_RUNTIME( cudaMemcpy( h_matrix,
					   d_matrix,
					   size(),
					   cudaMemcpyDeviceToHost ),
			       GPUexception::eTransfer );
    }
    
    // ###########################################################################################################
  private:
    // Methods
    
    //! Does the actual allocation of memory
    void allocate( const unsigned int rows, const unsigned int cols ) throw( GPUexception ) {
      /*!
	Does the actual work of allocating a dense matrix.
	If the matrix is already allocated, this method checks to see if it
	is the correct size.
	If not, the old allocation is released, and new memory allocated
	
	@param[in] rows Number of rows in the matrix
	@param[in] cols Number of columns in the matrix
      */
      
      //std::cout << __FUNCTION__ << ": Allocating ...";
      
      // See if the matrix is already allocated
      if( h_matrix != NULL ) {
	
	// Is the allocated matrix the right size?
	if( (nRows==rows) && (nCols==cols) ) {
	  // Allocated matrix has right side, so we can just return
	  //std::cout << "Reusing" << std::endl;
	  return;
	}
	
	//std::cout << "Reallocating ... " << std::endl;
	
	// We're going to have to allocate new memory, so release the current
	SciGPU_TRY_CUDA_RUNTIME( cudaFreeHost( h_matrix ),
				 GPUexception::eFreeHost );
	SciGPU_TRY_CUDA_RUNTIME( cudaFree( d_matrix ),
				 GPUexception::eFreeGPU );
      }
      
      nRows = rows;
      nCols = cols;
      
      SciGPU_TRY_CUDA_RUNTIME( cudaMallocHost( (void**)&h_matrix, size() ),
			       GPUexception::eMallocHost );
      
      SciGPU_TRY_CUDA_RUNTIME( cudaMalloc( (void**)&d_matrix, size() ),
			       GPUexception::eMallocGPU );
      
      //std::cout << "Done" << std::endl;
    }
    
    // ------------------------------------------------------
    
    //! Returns the size of the matrix in bytes
    size_t size( void ) const {
      /*!
	Computes the size in bytes from the number of rows and columns
      */
      
      return( nRows *nCols *sizeof(T) );
    }
    
    
    // =======================================================
    // Variables
    
    //! Holds the matrix in page-locked host memory
    T *h_matrix;
    //! Holds the matrix on the GPU
    T *d_matrix;
    
    //! Number of rows in this matrix
    unsigned int nRows;
    
    //! Number of columns in this matrix
    unsigned int nCols;
    
    
  };
  
}


#endif
