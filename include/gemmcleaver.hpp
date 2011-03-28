/*! \file
  Header file for SciGPUgemm::GEMMcleaver class

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


#ifndef GEMMcleaver_H
#define GEMMcleaver_H


#include <complex>

#include <cuda.h>

#include "gpuerror.hpp"
#include "gpugemm.hpp"
#include "densematrix.hpp"



namespace SciGPUgemm {

  //! The GEMMcleaver class divides large GEMM calls up for running on a limited-memory GPU
  /*!
    GEMMcleaver provides a means of using CUBLAS acceleration for GEMM calls on a limited
    memory GPU.
    If all the required data do not fit into the GPU memory, then the matrices are chopped
    into smaller pieces which fit onto the GPU.
    These are then staged through the GPU, and assembled into the final result.
    
    
    \section cleaving Matrix Cleaving
    
    Consider the matrix multiplication
    \f[
    C = A \cdot B
    \f]
    where \f$A\f$ is an \f$(m \times k)\f$ matrix and \f$B\f$ is an \f$(k \times n)\f$ matrix,
    making  \f$C\f$ an \f$(m \times n)\f$ matrix.
    We can divide \f$A\f$ into a column vector of matrices
    \f[
    A = \left (
    \begin{array}{c} A_0 \\ A_1 \\ \vdots \\  A_r \end{array}
    \right )
    \f]
    where each entry \f$A_i\f$ is an \f$( p_i \times k )\f$ matrix, for some \f$\sum p_i = m\f$.
    In practice, all the \f$p_i\f$ will be the same, with the possible exception of \f$p_r\f$,
    which will be an edge case.
    In a similar manner, we can divide \f$B\f$ into a row vector of matrices
    \f[
    B = \left (
    \begin{array}{cccc} B_0 & B_1 & \cdots & B_s \end{array}
    \right )
    \f]
    where each \f$B_j\f$ is an \f$( k \times q_j )\f$ matrix and \f$\sum q_j = n\f$.
    Again all the \f$q_j\f$ will be the same, with the possible exception of \f$q_s\f$.
    We then form the outer product of these two vectors
    \f[
    C = 
    \left (
    \begin{array}{c} A_0 \\ A_1 \\ \vdots \\  A_r \end{array}
    \right )
    \cdot
    \left (
    \begin{array}{cccc} B_0 & B_1 & \cdots & B_s \end{array}
    \right )
    =
    \left (
    \begin{array}{cccc}
    A_0 B_0 & A_0 B_1 & \cdots & A_0 B_s \\
    A_1 B_0 & A_1 B_1 &        & A_1 B_s \\
    \vdots  &         & \ddots &         \\
    A_r B_0 &         &        & A_r B_s
    \end{array}
    \right )
    \f]
    Each individual \f$C_{ij} = A_i B_j\f$ is an \f$( p_i \times q_j )\f$ matrix, and can be
    computed independently of all the others.



    \section cleaveimplement Cleaving Implementation

    The GEMMcleaver class provides a full implementation of the BLAS routines
    \c SGEMM, \c DGEMM, \c CGEMM and \c ZGEMM.
    This introduces a few extra subtleties over the basic strategy described above.
    In particular, transposal (or hermitian conjugation) of the input matrices may
    be required.
    This is achieved by cleaving the untransposed input matrix, and the instructing
    the appropriate CUBLAS call to transpose the submatrices for each individual
    sub-multiplication.

    @warning A valid GPU context must exist before any routines are called
    @warning If a call fails, the entire program will abort
    @note No destructor is required, since no memory is allocated by this class
  */
  class GEMMcleaver : public GPUgemm {

  public:

    //! Default constructor
    GEMMcleaver( void ) {
      /*!
	Default constructor sets maxGPUmemory to zero,
	to indicated that this instance should use all
	available GPU memory.
	The internal matrix sizes are zeroed
	
	\internal
	This is actually null, since the GPUgemm constructor
	handles everything
      */
    }

    //! Constructor with limit on GPU memory usage
    GEMMcleaver( const size_t maxMem ) : GPUgemm( maxMem ) {
       /*!
	This constructor sets the maximum memory usage of
	the current instance to \a maxMem.
	It does not check this is sensible for the current GPU.
	The internal matrix sizes are zeroed

	\internal
	This is actually null, since the GPUgemm constructor
	handles everything

	@param[in] maxMem The amount of memory to use on the GPU
      */
    }


    //! Wrapper for SGEMM
    void sgemm( const char transA, const char transB,
		const int m, const int n, const int k,
		const float alpha,
		const float *A, const int lda,
		const float *B, const int ldb,
		const float beta,
		float *C, const int ldc ) const;
    
    //! Wrapper for DGEMM
    void dgemm( const char transA, const char transB,
		const int m, const int n, const int k,
		const double alpha,
		const double *A, const int lda,
		const double *B, const int ldb,
		const double beta,
		double *C, const int ldc ) const;
    
    //! Wrapper for CGEMM
    void cgemm( const char transA, const char transB,
		const int m, const int n, const int k,
		const std::complex<float> alpha,
		const std::complex<float> *A, const int lda,
		const std::complex<float> *B, const int ldb,
		const std::complex<float> beta,
		std::complex<float> *C, const int ldc ) const;

    //! Wrapper for ZGEMM
    void zgemm( const char transA, const char transB,
		const int m, const int n, const int k,
		const std::complex<double> alpha,
		const std::complex<double> *A, const int lda,
		const std::complex<double> *B, const int ldb,
		const std::complex<double> beta,
		std::complex<double> *C, const int ldc ) const;


    //! Templated automatic wrapper
    template<typename T>
    void autogemm( const char transA, const char transB,
		   const int m, const int n, const int k,
		   const T alpha,
		   const T *A, const int lda,
		   const T *B, const int ldb,
		   const T beta,
		   T *C, const int ldc ) const {
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

	This version of the routine automatically selects the appropriate method based
	on the input arguments.

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
      this->gemm<T>( transA, transB, m, n, k,
		     alpha, A, lda, B, ldb,
		     beta, C, ldc );
    }

    
    // ##################################################
    
  private:
    
    //! Enumeration to select cleaving directions
    /*!
      Used within the cleaver, to decide which direction
      to cleave a matrix
    */
    enum CleaveDirection{
      CleaveRows, //!< Cleave along the rows (i.e. split into column vector)
      CleaveCols //!< Cleave along columns (i.e. split into row vector)
    };
    
    //! Initial cleave size
    const static unsigned long cFirstCleave = 1048576;
    /*!<
      This is the cleaving size which will be tried first - the initial guess
      for \f$p_i\f$ and \f$q_j\f$ defined in section \ref cleaving.
      It is large enough to ensure that matrices which will fit into GPU
      memory will not be cloven.
    */
    
    // =====================================================
    
    //! Templated GEMM implementation
    template<typename T>
    void gemm( const char transA, const char transB,
	       const int m, const int n, const int k,
	       const T alpha,
	       const T *A, const int lda,
	       const T *B, const int ldb,
	       const T beta,
	       T *C, const int ldc ) const throw() {
      /*!
	This template implements the cloven matrix-matrix multiplication.
	Its arguments follow those of the \c GEMM routines from BLAS, and assumes
	column-major ordering.
	
	@see GEMMcleaver::sgemm GEMMcleaver::dgemm
	@see \ref cleaving
	@tparam T The kind of matrices to be multiplied. Must be float or double
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
	@todo Consider adding a failover to CPU if we can't cleave on the GPU
      */
      
      // Dense matrix placeholders
      DenseMatrix<T> Aprime, Bprime, Cprime;
      
      unsigned int nAblocks, nBblocks;
      
      unsigned int cleaveSizeA, cleaveSizeB;
      
      // Verify Inputs using method inherited from GPUgemm
      // This will also stash sizes in h{A|B|C} and w{A|B|C}
      VerifyAndExtract( transA, transB, m, n, k, lda, ldb, ldc );
      
      try {
	// Work out sizes
	GetCleaveSizes<T>( m, n, k, cleaveSizeA, cleaveSizeB );
	if( (cleaveSizeA==0) || (cleaveSizeB==0) ) {
	  // Insufficient GPU memory
	  std::cerr << __PRETTY_FUNCTION__ << ": Could not cleave on GPU" << std::endl;
	  exit( EXIT_FAILURE );
	}
	
	// Compute how many blocks each matrix will be split
	nAblocks = m / cleaveSizeA;
	if( (m%cleaveSizeA) != 0 ) {
	  nAblocks++;
	}
	
	nBblocks = n / cleaveSizeB;
	if( (n%cleaveSizeB) != 0 ) {
	  nBblocks++;
	}
	
	// Loop over all the pairs of blocks
	for( unsigned int iAblock=0; iAblock<nAblocks; iAblock++ ) {
	  
	  // Extract the next A sub-matrix
	  switch( transA ) {
	  case 'N':
	  case 'n':
	    cleave( Aprime, iAblock, cleaveSizeA, CleaveRows, A, lda, m, k );
	    break;
	    
	  case 'T':
	  case 't':
	  case 'C':
	  case 'c':
	    cleave( Aprime, iAblock, cleaveSizeA, CleaveCols, A, lda, k, m );
	    break;
	    
	  default:
	    std::cerr << __FUNCTION__ << ": Bad transA when cleaving A" << std::endl;
	    exit( EXIT_FAILURE );
	  }
	  
	  // Place it on the GPU
	  Aprime.send();
	  
	  for( unsigned int iBblock=0; iBblock<nBblocks; iBblock++ ) {
	    
	    // Extract the next B sub-matrix
	    switch( transB ) {
	    case 'N':
	    case 'n':
	      cleave( Bprime, iBblock, cleaveSizeB, CleaveCols, B, ldb, k, n );
	      break;
	      
	    case 'T':
	    case 't':
	    case 'C':
	    case 'c':
	      cleave( Bprime, iBblock, cleaveSizeB, CleaveRows, B, ldb, n, k );
	      break;
	      
	    default:
	      std::cerr << __FUNCTION__ << ": Bad transB when cleaving B" << std::endl;
	      exit( EXIT_FAILURE );
	    }
	    
	    // Place it on the GPU
	    Bprime.send();
	    
	    // Compute the matrix multiplication of the current two sub-matrices
	    Cprime.multiply( transA, transB, alpha, Aprime, Bprime );
	    
	    // Retrieve the results
	    Cprime.receive();
	    
	    // Expand into the result matrix, taking care of the beta*C term
	    Cprime.expand( &( C[getindex( iAblock*cleaveSizeA, iBblock*cleaveSizeB, ldc)] ),
			   ldc,
			   beta );
	  }
	}
      }
      catch( GPUexception &e ) {
	e.printout();
	exit( EXIT_FAILURE );
      }
      
    }
    
    
    
    //! Index calculator
    unsigned long getindex( const unsigned int i, const unsigned int j,
			    const unsigned int ld ) const;
    
    // ----------------------------------------------
    
    //! Routine to cleave a matrix into a DenseMatrix
    template<typename T>
    void cleave( DenseMatrix<T> &myBlock,
		 const unsigned int iBlock,
		 const unsigned int cleaveSize,
		 const enum CleaveDirection direc,
		 const T *A, const int ld,
		 const int m, const int n ) const {
      /*!
	Extracts the specified block of the input matrix \a A, cleaved in the specified
	direction.
	The matrix \a A is of size \f$( m \times n )\f$.
	The product is a DenseMatrix containing the desired block
	
	If the cleaving direction is ::CleaveRows, then the matrix is being
	split into a column vector (where every element is a matrix) and a
	\f$( \textrm{cleaveSize} \times n)\f$ DenseMatrix is returned.
	Similarly, if the cleaving direction is ::CleaveCols, then the matrix
	is being split into a row vector, and a
	\f$( m \times \textrm{cleaveSize} )\f$ DenseMatrix will be returned.
	In both cases, the size of the block will be trimmed if necessary,
	to prevent the array bounds being exceeded
	
	@tparam T The type of matrices to be cloven
	@param[out] myBlock DenseMatrix in which the block should be stored
	@param[in] iBlock Which block is wanted
	@param[in] cleaveSize The size of the block to be taken
	@param[in] direc The direction along which to cleave
	@param[in] A The matrix to be cleaved
	@param[in] ld The size of the leading dimension of \a A
	@param[in] m The number of rows in \a A
	@param[in] n The number of columns in \a A
      */
      
      unsigned int takeRows, takeCols, index;
      
      switch( direc ) {
      case CleaveRows:
	// We're splitting into a column vector
	
	// Sanity check the input
	// This error should never appear if the routine is called correctly
	if( (iBlock*cleaveSize) > static_cast<unsigned int>(m) ) {
	  std::cerr << __PRETTY_FUNCTION__ << ": Out of bounds in CleaveRows" << std::endl;
	  exit( EXIT_FAILURE );
	}
	
	index = getindex( iBlock*cleaveSize, 0, ld );
	takeRows = min( cleaveSize, m-(iBlock*cleaveSize) );
	
	myBlock.compact(  &(A[index]), ld, takeRows, n );
	break;
	
	
      case CleaveCols:
	// We're splitting into a row vector
	if( (iBlock*cleaveSize) > static_cast<unsigned int>(n) ) {
	  std::cerr << __PRETTY_FUNCTION__ << ": Out of bounds in CleaveCols" << std::endl;
	  exit( EXIT_FAILURE );
	}
	
	index = getindex( 0, iBlock*cleaveSize, ld );
	takeCols = min( cleaveSize, n-(iBlock*cleaveSize) );
	
	myBlock.compact( &(A[index]), ld, m, takeCols );
	break;
	
      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Can't get here" << std::endl;
	exit( EXIT_FAILURE );
      }
    }
    
    // ----------------------------------------------
    
    //! Minimum function for index calculations
    template<typename T>
    T min( const T a, const T b ) const {
      if( a < b ) {
	return a;
      } else {
	return b;
      }
    }
    
    // ----------------------------------------------
    
    //! Memory consumption calculator
    template<typename T>
    size_t CalculateMemory( const int m, const int n, const int k ) const {
      /*!
	Calculates the amount of memory required on the GPU to perform the
	GEMM multiplication defined by m, n and k.
	Since GEMMcleaver makes the matrices on the GPU dense, we don't need
	the leading dimensions
	@tparam T The type of matrices we're handling (float, double etc.)
	@param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
	@param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
	@param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
	@retval CalculateMemory The number of bytes which will be required
      */
      return( sizeof(T) * ( (m*k) + (k*n) + (m*n) ) );
    }
    
    
    
    // ----------------------------------------------
    
    //! Routine to compute where to cleave matrices
    template<typename T>
    void GetCleaveSizes( const int m, const int n, const int k,
			 unsigned int &cleaveSizeA, unsigned int &cleaveSizeB ) const throw( GPUexception ) {
      /*!
	Calculates the required cleaving sizes for the A and B matrices.
	The initial guesses are set to GEMMcleaver::cFirstCleave, and reduced by a factor of
	two until the required memory is less than GEMMcleaver::maxGPUmemory
	
	@tparam T The type of matrices being handled
	@param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
	@param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
	@param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
	@param[out] cleaveSizeA The number of rows to include in an A sub-matrix
	@param[out] cleaveSizeB The number of columns to include in a B submatrix
	@warning Since we can't lock the GPU, there is a race condition on the size calculations. This can cause out of memory exceptions
      */
      size_t maxMem;
      
      // Initialise guesses
      cleaveSizeA = cleaveSizeB = cFirstCleave;
      
      // Get the maximum allowed amount of memory
      maxMem = this->maxGPUmemory;
      if( maxMem == 0 ) {
	unsigned int free, total;
	
	/*
	  This call is in the driver API so the GPU context must be 
	  initialised prior to this function being called
	*/
	SciGPU_TRY_CUDA_DRIVER( cuMemGetInfo(&free, &total) );
	
	maxMem = free;
      }
      
      
#ifdef _DEBUG
      std::cout << __FUNCTION__ << ": maxMem = " << maxMem << std::endl;
#endif
      
      // Catch case where everything fits in GPU memory
      if( CalculateMemory<T>( m, n, k ) < maxMem ) {
	/*
	  With the cleaveSizes set to be so large, no actual
	  cleaving will take place
	  We can just return
	*/
#ifdef _DEBUG
	std::cout << __FUNCTION__ << ": No need to cleave" << std::endl;
#endif    
	
	return;
      }
      
      // Knock down our initial guesses for cleaving
      while( cleaveSizeA > static_cast<unsigned int>(m) ) {
	cleaveSizeA /= 2;
      }
      while( cleaveSizeB > static_cast<unsigned int>(n) ) {
	cleaveSizeB /= 2;
      }
      
      // Try progressively smaller sizes
      bool changeA = true;
      while( CalculateMemory<T>( cleaveSizeA, cleaveSizeB, k ) > maxMem ) {
	
	// Have we gone as small as we can?
	if ( (cleaveSizeA==1) && (cleaveSizeB==1) ) {
	  // If this check is true, then the matrices can't fit in GPU memory
	  cleaveSizeA = cleaveSizeB = 0;
	  break;
	}
	
	// Alternate between reducing the cleavage of A and B
	// We also have to test if we've reach a size of 1
	if( changeA ) {
	  if( cleaveSizeA > 1 ) {
	    cleaveSizeA /= 2;
	  }
	} else {
	  if( cleaveSizeB > 1 ) {
	    cleaveSizeB /= 2;
	  }
	}
	
	// Make sure we change the other cleaveSize next time
	changeA = !changeA;
      }
      
#ifdef _DEBUG
      std::cout << __FUNCTION__ << ": Cleave Sizes are " << cleaveSizeA << " " << cleaveSizeB << std::endl;
#endif
      
    }
    
  };
  



}

#endif
