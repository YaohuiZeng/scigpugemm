// Matrix Comparison Class

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

#ifndef MATRIX_COMPARE_H
#define MATRIX_COMPARE_H

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <typeinfo>

#include <cmath>

#include <gsl/gsl_statistics.h>





// ###############################################################

class MatrixCompare {

public:
  // Allow access to stream insertion operator
  // Takes a reference to avoid a copy
  friend ostream& operator<<( ostream& os, const MatrixCompare& myCompare );

  // ---------------------------

  //! Default constructor zero initialises everything
  MatrixCompare( void ) : maxDiff(0),
			  meanDiff(0),
			  medianDiff(0),
			  errL2(0),
			  errRMS(0) {
  }

  // ---------------------------


  // Templated compare function
  template<typename T>
  void CompareArrays( const T *A, const T *B,
		      const unsigned int ld, const unsigned int h, const unsigned int w ) {
    // Compiles various useful information about the differences between
    // the matrices A and B
    // They must both have dimensions h*w, and leading dimension ld

    T *diff;
    double err, ref;
    double normRef;
    double normErr;
    
    if( ld < h ) {
      std::cerr  << __PRETTY_FUNCTION__ << ": ld < h" << std::endl;
    }
    
    // Create array to hold the differences
    diff = new T[h*w];
    
    for( unsigned int j=0; j<w; j++ ) {
      for( unsigned int i=0; i<h; i++ ) {
	diff[i+(j*h)] = fabs( A[i+(j*ld)] - B[i+(j*ld)] );
      }
    }
    
    // Sort the array
    qsort( diff, h*w, sizeof(T), qsortcompare<T> );
    
    // Start using gsl
    GetArrayStats<T>( diff, h*w );
    
    // Compute L2 norm of the error
    err = ref = 0;
    
    for( unsigned int j=0; j<w; j++ ) {
      for( unsigned int i=0; i<h; i++ ) {
	T currDiff = A[i+(j*ld)]-B[i+(j*ld)];
	err += (currDiff*currDiff);
	ref += A[i+(j*ld)]*A[i+(j*ld)];
      }
    }

    errRMS = sqrt( err / (h*w) );
    
    normRef = sqrt( ref );
    normErr = sqrt( err );
    errL2 = normErr/normRef;
    
    delete[] diff;
  }


  // ---------------------------
  

  // Routine to return a list of the column headings
  static void GetHeaders( vector<string> &header ) {
    // Fills out a list of column header strings
    
    header.clear();

    header.push_back( "maxDiff" );
    header.push_back( "meanDiff" );
    header.push_back( "medianDiff" );
    header.push_back( "errL2" );
    header.push_back( "errRMS") ;
  }


  // =======================================================

private:
  // The result members
  double maxDiff, meanDiff, medianDiff, errL2, errRMS;

  // ---------------------------

  template<typename T>
  void GetArrayStats( const T *A, const unsigned int nElements ) {
    // Dispatch routine into GSL
    // Probably broken on other compilers
    // How do I do template specialisation within a class?

    switch( typeid(A[0]).name()[0] ) {
    case 'f':
      maxDiff = gsl_stats_float_max( reinterpret_cast<const float *>(A), 1, nElements );
      meanDiff = gsl_stats_float_mean( reinterpret_cast<const float *>(A), 1, nElements );
      medianDiff = gsl_stats_float_median_from_sorted_data( reinterpret_cast<const float *>(A), 1, nElements );
      break;

    case 'd':
      maxDiff = gsl_stats_max( reinterpret_cast<const double *>(A), 1, nElements );
      meanDiff = gsl_stats_mean( reinterpret_cast<const double *>(A), 1, nElements );
      medianDiff = gsl_stats_median_from_sorted_data( reinterpret_cast<const double *>(A), 1, nElements );
      break;

    default:
      std::cerr << __PRETTY_FUNCTION__ << ": Unimplemented call for type ";
      std::cerr << typeid(A[0]).name() << std::endl;
      exit( EXIT_FAILURE );
    }
  }
    

  // ---------------------------

  template<typename T> static int qsortcompare( const void *aptr, const void *bptr ) {
    // Routine for use by qsort
    // Have to cast the pointers to the numerical type T, then subtract
    // to compare
    // Declared 'static' to suppress the 'this' pointer
    T a, b;
    
    a = *(T*)aptr;
    b = *(T*)bptr;
    
    if( a<b ) {
      return -1;
  } else if ( a>b ) {
      return 1;
    } else {
      return 0;
    }
  }

};






ostream& operator<<( ostream& os, const MatrixCompare& myCompare ) {
  // Writes the internals out to the designated stream
  // Takes a reference to avoid a copy
  os << setw(15) << myCompare.maxDiff;
  os << setw(15) << myCompare.meanDiff;
  os << setw(15) << myCompare.medianDiff;
  os << setw(15) << myCompare.errL2;
  os << setw(15) << myCompare.errRMS;

  return( os );
}


#endif
