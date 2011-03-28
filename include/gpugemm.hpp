/*! \file
  Header file containing the SciGPUgemm::GPUgemm base class

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

#ifndef GPU_GEMM_H
#define GPU_GEMM_H

#include <cstdlib>
#include <iostream>

namespace SciGPUgemm {

  //! A base class for GEMM calls on the GPU
  /*!
    \internal
    This is a base class for GEMM calls on the GPU.
    It contains the basic data items required for a call,
    but no implementations.
    As such, it is not useful by itself, but only
    as a base for other classes.
    @todo An InitGPU function might be useful, for those who have minimal CUDA knowledge
   */
  class GPUgemm {
    
  protected:
    //! Default constructor
    GPUgemm( void ) : maxGPUmemory(0),
		      hA(0), wA(0),
		      hB(0), wB(0),
		      hC(0), wC(0) {
      /*!
	Default constructor sets maxGPUmemory to zero,
	indicating that all available GPU memory should
	be used.
	The internal matrix sizes are also zeroed
      */
    }

    //! Constructor with limit on GPU memory usage
    GPUgemm( const size_t maxMem ) : maxGPUmemory(maxMem),
				     hA(0), wA(0),
				     hB(0), wB(0),
				     hC(0), wC(0) {
      /*!
	This constructor sets the maximum memory usage of
	the current instance to \a maxMem.
	It does not check this is sensible for the current GPU.
	The interal matrix sizes are zeroed
	@param[in] maxMem The amount of memory to use on the GPU
      */
    }

    //! Virtual destructor, to ensure correct destructor is called
    virtual ~GPUgemm( void ) {
      /*!
	We have to provide an empty destructor
      */
    }
      
    
    
    //! Maximum amount of GPU memory to use
    size_t maxGPUmemory;
    /*!<
      This private member variable holds the maximum amount of
      memory which the current instance will use on the GPU.
      It is passed through to GEMMcleaver.
      If zero, all GPU memory may be used.
    */

    //! Number of rows in matrix A
    mutable unsigned int hA;
    //! Number of columns in matrix A
    mutable unsigned int wA;
    //! Number of rows in matrix B
    mutable unsigned int hB;
    //! Number of columns in matrix B
    mutable unsigned int wB;
    //! Number of rows in matrix C
    mutable unsigned int hC;
    //! Number of columns in matrix C
    mutable unsigned int wC;

    
    // --------------------------------------
    //! Verifies the GEMM call and stores the matrix sizes for subsequent use
    void VerifyAndExtract( const char transA, const char transB,
			   const int m, const int n, const int k,
			   const int lda, const int ldb, const int ldc ) const {
      /*!
	Verifies the input parameters for a GEMM call, and extracts
	the individual matrix sizes into the private class variables.
	The arguments follow those in a GEMM call.
	This is declared as \c const because the extracted parameters
	don't affect the GEMM call itself (and hence they are listed as
	`mutable' in the class declaration).
	@param[in] transA Whether to transpose matrix \a A
	@param[in] transB Whether to transpose matrix \a B
	@param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
	@param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
	@param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
	@param[in] lda The size of the leading dimension of \a A
	@param[in] ldb The size of the leading dimension of \a B
	@param[in] ldc The size of the leading dimension of \a C
      */
      
      // Set up the size of C
      hC = m;
      wC = n;
      if( static_cast<unsigned int>(ldc) < hC ) {
	std::cerr << __PRETTY_FUNCTION__ << ": ldc < hC" << std::endl;
	exit( EXIT_FAILURE );
      }
      
      // Sort out A
      switch( transA ) {
      case 'N':
      case 'n':
	// No transposition
	hA = m;
	wA = k;
	break;
	
      case 'T':
      case 't':
      case 'C':
      case 'c':
	// Transposing
	hA = k;
	wA = m;
	break;
	
      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transA" << std::endl;
	exit( EXIT_FAILURE );
      }
      
      // Check sizes
      if( static_cast<unsigned int>(lda) < hA ) {
	std::cerr << __PRETTY_FUNCTION__ << ": lda < hA" << std::endl;
	exit( EXIT_FAILURE );
      }
      
      // And B
      switch( transB ) {
      case 'N':
      case 'n':
	// No transposition
	hB = k;
	wB = n;
	break;

      case 'T':
      case 't':
      case 'C':
      case 'c':
	// Transposing
	hB = n;
	wB = k;
	break;
	
      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transB" << std::endl;
	exit( EXIT_FAILURE );
      }
      
      // Check sizes
      if( static_cast<unsigned int>(ldb) < hB ) {
	std::cerr << __PRETTY_FUNCTION__ << ": ldb < hB" << std::endl;
	exit( EXIT_FAILURE );
      }
      
    }

  };





}

#endif
