/*! \file
  Header file for the SciGPUgemm::MGEMM class

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

#ifndef MGEMM_H
#define MGEMM_H

#include <vector>

#include "gpugemm.hpp"

namespace SciGPUgemm {

  //! The MGEMM class performs a matrix-matrix multiplication in mixed precision
  /*!
    MGEMM provides a means of using a single-precision GPU to accelerate a
    double precision matrix-matrix multiply.
    It does this by separating the input matrices into `small' and `large'
    portions.



    \section splitting Matrix Splitting for Multi-Precision Arithmetic

    Consider the following matrix-matrix multiplication:
    \f[
    C = A \cdot B
    \f]
    We can split each matrix into `large' and `small' portions, so that
    \f[
    C = \left ( A^{l} + A^{s} \right )
            \cdot
            \left ( B^{l} + B^{s} \right ) \\
      = A \cdot B^{l} + A^{l} \cdot B^{s} + A^{s} \cdot B^{s}
    \f]
    where an `l' superscript denotes the `large' portion of a matrix,
    and an `s' superscript denotes the `small' portion.
    Note that there are only three terms in the final expansion, since
    the first term contains \f$A\f$.
    We aim to split the matrices in such a way that \f$A^{l}\f$ and
    \f$B^{l}\f$ are sparse matrices.
    The dense \f$A^{s} \cdot B^{s}\f$ term is handled in single
    precision on the GPU (via GEMMcleaver), while the other two terms are
    performed in double precision on the CPU.
    This approach can be generalised up to a full \c GEMM call.
    
    \section performance MGEMM Performance Notes

    Internally, the \f$A \cdot B^{l}\f$ and \f$A^{l} \cdot B^{s}\f$
    terms are computed as a series of scalar-vector multiplications,
    where the scalar is a single element of the \f$A^{l}\f$, and the
    vector is a row from \f$B^{s}\f$ (and similarly for the
    other multiplication).
    Traversing a matrix row is not cache friendly for a column-major
    matrix, which means that computation of these two terms becomes
    very computationally expensive when there are many `large' 
    elements present.
    See the notes for MGEMM::ABlarge and MGEMM::AlargeBsmall for
    further details.

    @see GEMMcleaver
  */
  class MGEMM : public GPUgemm {
    
  public:

    //! Default constructor
    MGEMM( void ) : nAlarge(0),
		    nBlarge(0) {
      /*!
	Default constructor sets maxGPUmemory to zero,
	to indicated that this instance should use all
	available GPU memory.
	The internal matrix sizes are zeroed.
      */
    }

    //! Constructor with limit on GPU memory usage
    MGEMM( const size_t maxMem ) : GPUgemm(maxMem),
				   nAlarge(0),
				   nBlarge(0) {
      /*!
	This constructor sets the maximum memory usage of
	the current instance to \a maxMem.
	It does not check this is sensible for the current GPU.
	The internal matrix sizes are zeroed
	
	
	@param[in] maxMem The amount of memory to use on the GPU
      */
    }

    //! Multi-precision matrix multiply
    void mgemm( const char transA, const char transB,
		const int m, const int n, const int k,
		const double alpha,
		const double *A, const int lda,
		const double *B, const int ldb,
		const double beta,
		double *C, const int ldc,
		const double cutOff ) const;

    //! Number of large elements in \a A
    mutable unsigned int nAlarge;
    //! Number of large elements in \a B
    mutable unsigned int nBlarge;


    // ##################################################

  private:
    
    //! Holds information about big, sparse matrices
    /*!
      \internal
      This class holds the information about large, compact matrices
      Each element has a double precision value, and an index
      denoting its position in the larger matrix.
      This class is private to the MGEMM class
    */
    class Element {
   
    public:
      //! The value of this element
      double value;
      //! The location of this element within the larger matrix
      int index;
    };
    
    //! Defines a sparse matrix to be a list of Element entries
    /*!
      It is not anticipated that enough `large' elements will be found
      to justify a sophisticated sparse matrix treatment.
      Accordingly, the `large' matrices are simply a list of
      individual elements.
      The Standard Template Library is used to implement this list.
     */
    typedef std::vector<Element> SparseMatrix;

    
    // ---------------------------------------------

    //! Private method to split a matrix into `small' and `large' components
    void SeparateAndCompact( const double *A, double *Asmall, SparseMatrix &Alarge,
			     const unsigned int ld,
			     const unsigned int h,
			     const unsigned int w,
			     const double cutOff ) const;

    //! Private method to multiply the two small matrices on the GPU
    void AsmallBsmall( const char transA, const char transB,
		       const int m, const int n, const int k,
		       const double alpha,
		       const double *Asmall, const double *Bsmall,
		       const double beta,
		       double *C, const int ldc ) const;

    //! Private method to compact a double precision matrix into a float matrix
    void CompactDoubleToFloat( const double *Ad, float *Af,
			       const unsigned int ld,
			       const unsigned int h,
			       const unsigned int w ) const;

    //! Private method to expand a float matrix back up to double precision
    void ExpandFloatToDouble( const float *Af,
			      double *Ad,
			      const unsigned int ld,
			      const unsigned int h,
			      const unsigned int w,
			      const double beta ) const;

    //! Implementation of daxpy for MGEMM
    void daxpy( const unsigned int n,
		const double alpha,
		const double *x, const unsigned int xStep,
		double *y, const unsigned int yStep ) const;

    //! Private method to perform \f$A \cdot B^{l}\f$ multiplication
    void ABlarge( const char transA, const char transB,
		  const double alpha,
		  const double *A, const unsigned int lda,
		  const SparseMatrix &Blarge,
		  double *C, const unsigned int ldc ) const;

    //! Private method to perform \f$A^{l} \cdot B^{s}\f$ multiplication
    void AlargeBsmall( const char transA, const char transB,
		       const double alpha,
		       const SparseMatrix &Alarge,
		       const double *Bsmall,
		       double *C, const unsigned int ldc ) const;
      
  };


}


#endif
