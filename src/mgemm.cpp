/*! \file
  File containing implementation of SciGPUgemm::MGEMM methods

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

#include "gemmcleaver.hpp"

#include "mgemm.hpp"


namespace SciGPUgemm {
  void MGEMM::mgemm( const char transA, const char transB,
		     const int m, const int n, const int k,
		     const double alpha,
		     const double *A, const int lda,
		     const double *B, const int ldb,
		     const double beta,
		     double *C, const int ldc,
		     const double cutOff ) const {
    /*!
      The multi-precision matrix multiply itself.
      Its arguments follow those of the *GEMM routines from BLAS, and assumes
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

      On exit, the member variables MGEMM::nAlarge and MGEMM::nBlarge are set to the
      number of large elements found in their respective input matrices
      
      @see \ref splitting
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
      @todo Consider multi-threading this library again

    */
  
    SparseMatrix Alarge, Blarge;
    double *Asmall, *Bsmall;

    // Verify the input parameters using inherited method
    // This will also stash sizes in  h{A|B|C} and w{A|B|C}
    VerifyAndExtract( transA, transB, m, n, k, lda, ldb, ldc );

    // Allocate memory
    Asmall = new double[hA*wA];
    Bsmall = new double[hB*wB];

    // Split the matrices
    SeparateAndCompact( A, Asmall, Alarge, lda, hA, wA, cutOff );
    SeparateAndCompact( B, Bsmall, Blarge, ldb, hB, wB, cutOff );
    nAlarge = Alarge.size();
    nBlarge = Blarge.size();

#ifdef _DEBUG
    std::cout << __FUNCTION__ << ": Number of large elements is " << nAlarge << " and " << nBlarge << std::endl;
#endif

    // Perform the 'small' portion in single precision on the GPU
    // Note that the 'result' argument is the original C
    AsmallBsmall( transA, transB, m, n, k,
		  alpha, Asmall, Bsmall,
		  beta, C, ldc );

    // Perform A * Blarge
    ABlarge( transA, transB, alpha,
	     A, static_cast<unsigned int>(lda),
	     Blarge,
	     C, static_cast<unsigned int>(ldc) );

    // Perform Alarge * Bsmall
    AlargeBsmall( transA, transB, alpha,
		  Alarge,
		  Bsmall,
		  C, static_cast<unsigned int>(ldc) );

    // Release memory
    delete[] Asmall;
    delete[] Bsmall;

  }

  // ##########################################################

  void MGEMM::SeparateAndCompact( const double *A, double *Asmall, SparseMatrix &Alarge,
				  const unsigned int ld,
				  const unsigned int h,
				  const unsigned int w,
				  const double cutOff ) const {
    /*!
      Splits the input matrix \a A into `large' and `small' components,
      based on the \a cutOff value.
      At the same time, it compacts the `small' matrix from its
      initial BLAS configuration to one which is fully dense.
      Note that \a Asmall is still of double precision type, which is required
      for the sparse matrix multiplies.
      The MGEMM::AsmallBsmall method automatically down-converts the small matrices
      for the GPU.
      Recall that we have column major matrices.

      @param[in] A Pointer to the first element of the matrix to be separated
      @param[out] Asmall Pointer to the first element of the `small' matrix (must be already allocated)
      @param[out] Alarge Refernce to the structure which will hold the (sparse) `large' matrix
      @param[in] ld The leading dimension of matrix \a A
      @param[in] h The number of rows in matrix \a A
      @param[in] w The number of columns in matrix \a A
      @param[in] cutOff The value used to divide between `small' and `large' elements
    */

    // Sanity check
    if( ld < h ) {
      std::cerr << __PRETTY_FUNCTION__ << ": ld < h" << std::endl;
      exit( EXIT_FAILURE );
    }

    for( unsigned int j=0; j<w; j++ ) {
      for( unsigned int i=0; i<h; i++ ) {

	// Assign Asmall as a compacted straight copy
	Asmall[i+(j*h)] = A[i+(j*ld)];
	
	// Check size
	if( fabs(Asmall[i+(j*h)]) > cutOff ) {
	  Element largeElement;

	  // Put into the `large' element
	  largeElement.value = Asmall[i+(j*h)];
	  largeElement.index = i+(j*h);
	  Alarge.push_back( largeElement );
	  
	  // Zero out the corresponding `small' element
	  Asmall[i+(j*h)] = 0;
	}
	
      }
    }

  }


  // ----------------------------------------------
  void MGEMM::CompactDoubleToFloat( const double *Ad, float *Af,
				    const unsigned int ld,
				    const unsigned int h,
				    const unsigned int w ) const {
    /*!
      Converts a double precision matrix into a floating
      point matrix, compacting into fully dense form
      as it goes
      @param[in] Ad Pointer to the first element of the input matrix
      @param[out] Af Pointer to the first element of the output matrix (already allocated)
      @param[in] ld Leading dimension of \a Ad
      @param[in] h Number of rows in the matrices
      @param[in] w Number of columns in the matrices
    */

    // Sanity check
    if( ld < h ) {
      std::cerr << __PRETTY_FUNCTION__ << ": ld < h" << std::endl;
      exit( EXIT_FAILURE );
    }

    for( unsigned int j=0; j<w; j++ ) {
      for( unsigned int i=0; i<h; i++ ) {
	Af[i+(j*h)] = Ad[i+(j*ld)];
      }
    }
  }

  // ------------------------------------------------
  void MGEMM::ExpandFloatToDouble( const float *Af,
				   double *Ad,
				   const unsigned int ld,
				   const unsigned int h,
				   const unsigned int w,
				   const double beta ) const {
    /*!
      Converts a dense float matrix back to a BLAS-style
      double precision matrix.
      The mathematical operation performed is
      \f[
      A^{d}_{ij} \leftarrow A^{f}_{ij} + \beta A^{d}_{ij}
      \f]
      which is similar to BLAS conventions.
      Programatically, an expansion back to BLAS-style
      leading dimension form is also performed.
      No bounds checking is possible
      
      @param[in] Af Pointer to the first element of the input float matrix
      @param[in,out] Ad Pointer to the first element of the output double precision matrix
      @param[in] ld The leading dimension of \a Ad
      @param[in] h The number of row in the matrices
      @param[in] w The number of columns in the matrices
      @param[in] beta The value of \f$\beta\f$
    */

    // Sanity check
    if( ld < h ) {
      std::cerr << __PRETTY_FUNCTION__ << ": ld < h" << std::endl;
      exit( EXIT_FAILURE );
    }

    if( beta != 0 ) {
      // General case
      for( unsigned int j=0; j<w; j++ ) {
	for( unsigned int i=0; i<h; i++ ) {
	  Ad[i+(j*ld)] = Af[i+(j*h)] + ( beta * Ad[i+(j*ld)] );
	}
      }
    } else {
      // Try to avoid a memory load if beta==0
      // Also avoids problem with Ad containing nan if input matrix uninitialised
      for( unsigned int j=0; j<w; j++ ) {
	for( unsigned int i=0; i<h; i++ ) {
	  Ad[i+(j*ld)] = Af[i+(j*h)];
	}
      }
    }
  }

  // ------------------------------------------------
  void MGEMM::daxpy( const unsigned int n,
		     const double alpha,
		     const double *x, const unsigned int xStep,
		     double *y, const unsigned int yStep ) const {
    /*!
      Implements daxpy (from BLAS) for MGEMM.
      It performs
      \f[
      y_i \leftarrow \alpha x_i + y_i
      \f]
      for two vectors \a x and \a y.
      Not a full daxpy, since the steps are assumed to be
      greater than zero.
      @param[in] n The number of elements in the vectors
      @param[in] alpha The value of \f$\alpha\f$
      @param[in] x Pointer to the first element of vector \a x
      @param[in] xStep The number of memory locations between elements of \a x
      @param[in,out] y Pointer to the first element of vector \a y
      @param[in] yStep The number of memory locations between elements of \a y
     */
    for( unsigned int i=0; i<n; i++ ) {
      y[i*yStep] += alpha * x[i*xStep];
    }
  }


  // ------------------------------------------------
  void MGEMM::AsmallBsmall( const char transA, const char transB,
			    const int m, const int n, const int k,
			    const double alpha,
			    const double *Asmall, const double *Bsmall,
			    const double beta,
			    double *C, const int ldc ) const {
    /*!
      Uses GEMMcleaver to perform the multiplication of the `small'
      matrices on the GPU.
      Note that \a Asmall and \a Bsmall are still double precision.
      This routine has to down-convert them first.
      The matrix \a C is the real matrix passed in the call to
      MGEMM::mgemm.
      The arguments are generally passed through from MGEMM::mgemm,
      but leading dimensions of \a Asmall and \a Bsmall are obtained
      from the private variables MGEMM::hA and MGEMM::hB, since they
      will be dense matrices
      @param[in] transA Whether to transpose matrix \a A
      @param[in] transB Whether to transpose matrix \a B
      @param[in] m Number of rows in \f$o(A)\f$ and \f$C\f$
      @param[in] n Number of columns in \f$o(B)\f$ and \f$C\f$
      @param[in] k Number of columns in \f$o(A)\f$ and rows in \f$o(B)\f$
      @param[in] alpha The value of \f$\alpha\f$
      @param[in] Asmall The `small' \a A matrix
      @param[in] Bsmall The `small' \a B matrix
      @param[in] beta The value of \f$\beta\f$
      @param[in] C Pointer to the first element of \a C
      @param[in] ldc The size of the leading dimension of \a C
    */
    float *Af, *Bf, *Cf;

    // Allocate our float matrices
    Af = new float[hA*wA];
    Bf = new float[hB*wB];
    Cf = new float[hC*wC];

    // Convert Asmall and Bsmall
    // These are fully dense, so the leading dimension is the height
    // We don't need to do anything with C, since we'll handle that later
    CompactDoubleToFloat( Asmall, Af, hA, hA, wA );
    CompactDoubleToFloat( Bsmall, Bf, hB, hB, wB );


    // Use GEMMcleaver
    GEMMcleaver RunGPU( maxGPUmemory );

    RunGPU.sgemm( transA, transB,
		  m, n, k,
		  static_cast<float>(alpha),
		  Af, hA, Bf, hB,
		  0, // beta = 0 for this call
		  Cf, hC );

    // Cf now contains alpha * A * B
    // Expand back up to the C matrix, dealing with beta part
    ExpandFloatToDouble( Cf, C, ldc, hC, wC, beta );
		  

    // Release the float matrices
    delete[] Af;
    delete[] Bf;
    delete[] Cf;
  }


  // ------------------------------------------------
  void MGEMM::ABlarge( const char transA, const char transB,
		       const double alpha,
		       const double *A, const unsigned int lda,
		       const SparseMatrix &Blarge,
		       double *C, const unsigned int ldc ) const {
    /*!
      Method to compute the \f$A \cdot B^{l}\f$ term. Specfically, it performs
      \f[
      C_{ik} = \alpha A_{ij} B^{l}_{jk} + C_{ik}
      \f]
      (we do not need \f$\beta\f$, since that is taken care of by MGEMM::AsmallBsmall).
      Since \f$B^{l}\f$ is a sparse matrix, all elements are zero except for
      particular \f$(j,k)\f$ values.
      The basic idea is then simple: we loop over these non-zero values.
      To the <em>k</em>th column of \a C, we add values from the <em>j</em>th column of \a A.
      The <em>k</em>th column of \a C has MGEMM::hC elements (set by the
      MGEMM::VerifyAndExtract call at the start of MGEMM::mgemm) in consecutive
      locations, starting at <tt>C[k*hC]</tt>.
      Similarly, the <em>j</em>th column of \a A (which also has <tt>hC==hA</tt> elements)
      starts at <tt>A[j*hA]</tt> and is in consecutive memory locations.

      In practice, there are some subtleties.
      Firstly, \a A and \a C can have leading dimensions greater than MGEMM::hA and MGEMM::hC.
      This means that the <em>k</em>th column of \a C actually starts at
      <tt>C[k*ldc]</tt>, and similarly for the <em>j</em>th column of \a A.
      The transposes add further complications.
      Transposal of \a B can be handled by swapping what we designate as \c j
      and \c k.
      Transposing \a A is slightly trickier.
      If \a A is being transposed, the multiplication will be going along the
      <em>j</em>th row of \a A.
      This has MGEMM::wA (now equal to MGEMM::hC) elements, starting at
      <tt>A[j]</tt> and separated by \a lda memory locations.
      
      @note The matrix \a A will be traversed in a cache-unfriendly manner
      if it is being transposed

      @param[in] transA Whether to transpose matrix \a A
      @param[in] transB Whether to transpose matrix \a B
      @param[in] alpha The value of \f$\alpha\f$
      @param[in] A Pointer to the first element of matrix \a A
      @param[in] lda The size of the leading dimension of \a A
      @param[in] Blarge The (sparse) \f$B^{l}\f$ matrix
      @param[in,out] C Pointer to the first element of \a C
      @param[in] ldc The size of the leading dimension of \a C
    */

    unsigned int j, k;

    for( unsigned int myElement=0; myElement<Blarge.size(); myElement++ ) {
      // Compute j & k values
      switch( transB ) {
      case 'n':
      case 'N':
	j = Blarge.at(myElement).index % hB;
	k = Blarge.at(myElement).index / hB;
	break;

      case 't':
      case 'T':
      case 'c':
      case 'C':
	k = Blarge.at(myElement).index % hB;
	j = Blarge.at(myElement).index / hB;
	break;

      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transB" << std::endl;
	exit( EXIT_FAILURE );
      }

      // DO DAXPY on kth column of C

      switch( transA ) {
      case 'n':
      case 'N':
	daxpy( hC, alpha*Blarge.at(myElement).value,
	       &(A[j*lda]), 1,
	       &(C[k*ldc]), 1 );
	break;
	
      case 't':
      case 'T':
      case 'c':
      case 'C':
	daxpy( hC, alpha*Blarge.at(myElement).value,
	       &(A[j]), lda,
	       &(C[k*ldc]), 1 );
	break;

      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transA" << std::endl;
	exit( EXIT_FAILURE );
      }
      
    }
  }

  
  // ------------------------------------------------
  void MGEMM::AlargeBsmall( const char transA, const char transB,
			    const double alpha,
			    const SparseMatrix &Alarge,
			    const double *Bsmall,
			    double *C, const unsigned int ldc ) const {
    /*!
      Method to compute the \f$A^{l} \cdot B^{s}\f$ term. Specfically, it performs
      \f[
      C_{ik} = \alpha A^{l}_{ij} B^{s}_{jk} + C_{ik}
      \f]
      (we do not need \f$\beta\f$, since that is taken care of by MGEMM::AsmallBsmall).
      Since \f$A^{l}\f$ is a sparse matrix, all elements are zero except for
      particular \f$(i,j)\f$ values.
      The basic idea is then simple: we loop over these non-zero values.
      To the <em>i</em>th row of \a C, we add values from the <em>j</em>th row of \a B.
      The <em>i</em>th row of \a C has MGEMM::wC elements (set by the
      MGEMM::VerifyAndExtract call at the start of MGEMM::mgemm), starting at
      <tt>C[i]</tt> and separated by MGEMM::hC elements.
      Similarly, the <em>j</em>th row of \a Bsmall has MGEMM::wB elements starting
      at <tt>Bsmall[j]</tt> and separated by MGEMM::hB locations.

      In practice, there are some subtleties.
      Firstly, \a C can have leading dimension greater than MGEMM::hC, meaning that
      elements of the <em>i</em>th row of \a C are actually separated by \a ldc
      memory locations.
      This problem does not occur for \a Bsmall, due to its construction as a
      fully dense matrix in MGEMM::SeparateAndCompact.
      The transposes add further complications.
      Transposal of \a A can be handled by swapping what we designate as \c i
      and \c j.
      Transposing \a B is slightly trickier.
      If \a B is being transposed, the multiplication will be going along the
      <em>j</em>th column of \a Bsmall.
      This starts at <tt>Bsmall[j*hB]</tt> and the elements are in consective
      memory locations
      
      @note In this routine, \a C is always traversed in a cache-unfriendly manner,
      as will \a Bsmall if untransposed

      @param[in] transA Whether to transpose matrix \a A
      @param[in] transB Whether to transpose matrix \a B
      @param[in] alpha The value of \f$\alpha\f$
      @param[in] Alarge The (sparse) \f$A^{l}\f$ matrix
      @param[in] Bsmall Pointer to the first element of \f$B^s\f$
      @param[in,out] C Pointer to the first element of \a C
      @param[in] ldc The size of the leading dimension of \a C
    */

    unsigned int i, j;

    for( unsigned int myElement=0; myElement<Alarge.size(); myElement++ ) {
      // Compute i & j
      switch( transA ) {
      case 'n':
      case 'N':
	i = Alarge.at(myElement).index % hA;
	j = Alarge.at(myElement).index / hA;
	break;

      case 't':
      case 'T':
      case 'C':
      case 'c':
	j = Alarge.at(myElement).index % hA;
	i = Alarge.at(myElement).index / hA;
	break;

      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transA" << std::endl;
	exit( EXIT_FAILURE );
      }

      // Do DAXPY on the ith row of C
      switch( transB ) {
      case 'n':
      case 'N':
	daxpy( wC, alpha*Alarge.at(myElement).value,
	       &(Bsmall[j]), hB,
	       &(C[i]), ldc );
	break;
	
      case 't':
      case 'T':
      case 'C':
      case 'c':
	daxpy( wC, alpha*Alarge.at(myElement).value,
	       &(Bsmall[j*hB]), 1,
	       &(C[i]), ldc );
	break;

      default:
	std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised transB" << std::endl;
	exit( EXIT_FAILURE );
      }
    }
  }

}
