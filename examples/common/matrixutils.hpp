// Matrix Utilities
// ================


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


#ifndef MATRIX_UTILITIES_H
#define MATRIX_UTILITIES_H

#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <cmath>
#include <cstdlib>
#include <vector>
#include <complex>


// Note that we're going to store our matrices column-major
// This is Fortran style, matching CUBLAS

// ==========================================================================
// PrintMatrix
template<typename T> void PrintMatrix( const T *A, const int ld, const int h, const int w ) {
  // Prints the given matrix to std.out
  // ld is the leading dimension of the matrix, and we must have ld>=h
  // Templated to work with multiple datatypes
  
  if( ld < h ) {
    std::cerr << __PRETTY_FUNCTION__ << ": ld < h" << std::endl;
  }

  for( int i=0; i<h; i++ ) {
    for( int j=0; j<w; j++ ) {
      // We're going to be having column-major matrices, to match CUBLAS
      std::cout << std::setw(12) << std::setprecision(4) << A[(j*ld)+i];
    }
    std::cout << std::endl;
  }
}

// ==========================================================================

template<typename T> void RandomValue( T& result, const T min, const T max ) {
  // Returns a random number between min and max
  T randNum = std::rand() / static_cast<T>(RAND_MAX);

  result = min + ( (max-min) * randNum );
}


template<typename T> void RandomValue( std::complex<T>& result, const std::complex<T> min, const std::complex<T> max ) {
  T a, b;

  RandomValue( a, std::real(min), std::real(max) );
  RandomValue( b, std::imag(min), std::imag(max) );

  result = std::complex<T>( a, b );
}



// ==========================================================================
// RandomMatrix

template<typename T> void RandomMatrix( T *A, const int ld, const int h, const int w,
					const T min, const T max ) {
  // Fills a matrix with random numbers
  // ld is the leading dimension of the matrix, and we must have ld>=h
  // Templated to work with double and float

  if( ld < h ) {
    std::cerr << __FUNCTION__ << ": ld < h" << std::endl;
  }

  for( int j=0; j<w; j++ ) {
    for( int i=0; i<h; i++ ) {
      RandomValue( A[(j*ld)+i], min, max );
    }
  }
}

// ==========================================================================
// OrderedMatrix

template<typename T> void OrderedMatrix( T *A, const int ld, const int h, const int w ) {
  // Fills a matrix with sequential numbers
  // Templated to work with double and float
  
  if( ld < h ) {
    std::cerr << __PRETTY_FUNCTION__ << ": ld < h" << std::endl;
  }

  for( int j=0; j<w; j++ ) {
    for( int i=0; i<h; i++ ) {
      A[(j*ld)+i] = (i*w) + j;
    }
  }
}

// =========================================================================
// Salt Matrix

template<typename T> void SaltMatrix( T *A, const unsigned int ld, const unsigned int h, const unsigned int w,
				      const T saltMin, const T saltMax,
				      const unsigned int nSalt ) {
  // 'Salts' a matrix with extra random values in the range [saltMin,saltMax]
  // nSalt of these values are placed in random locations in the matrix

  unsigned int currSize;
   // List of unused indicies, created to be appropriate size using alternate constructor
  std::vector<unsigned int> indices(h*w);

  if( nSalt > h*w ) {
    std::cerr << "nSalt too large in SaltMatrix" << std::endl;
    exit( EXIT_FAILURE );
  }

  // Fill in the index list created above
  for( unsigned int i=0; i<h*w; i++ ) {
    indices.at(i) = i;
  }

  
  currSize = h*w;

  for( unsigned int iSalt=0; iSalt<nSalt; iSalt++ ) {
    // Pick a location from the first currSize elements of indices
    unsigned int randIndex = std::rand() % currSize;
    unsigned int index = indices.at(randIndex);
    unsigned int i = index % h;
    unsigned int j = index / h;

    // Delete the value from the list of available indices
    // We do this setting the currently selected index to the value at indices[currSize-1]
    // and decreasing currSize by 1
    
    indices.at(randIndex) = indices.at(currSize-1);
    currSize--;
    

    // Get a random number
    T randNum = std::rand() / static_cast<T>(RAND_MAX);

    // Salt the array
    A[(j*ld)+i] = saltMin + ( (saltMax-saltMin) * randNum );
  }
}

// =========================================================================================
// Comparison Functions


#endif
