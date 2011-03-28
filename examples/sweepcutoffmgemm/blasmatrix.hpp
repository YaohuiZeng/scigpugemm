/*! \file
  Class file for a BLAS matrix
*/

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

#ifndef BLAS_MATRIX_H
#define BLAS_MATRIX_H

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>


//! Class to hold a dense column order matrix on the CPU
/*!
  This class isn't fully defined with copy constructors and
  operators.
  As a result, the compiler will whinge a bit
*/
template<typename T>
class BlasMatrix {

public:

  //! Default constructor
  BlasMatrix( void ) : matrix(NULL),
		       nRows(0),
		       nCols(0) {
  }

  //! Construct from given file
  BlasMatrix( const std::string matrixFile ) : matrix(NULL),
					       nRows(0),
					       nCols(0) {
    // Open up the file
    std::ifstream inputFile( matrixFile.c_str() );
    if( !inputFile.good() ) {
      std::cerr << __PRETTY_FUNCTION__ << ": Failed to open file " << matrixFile << std::endl;
      exit( EXIT_FAILURE );
    }

    // Get the number of rows and columns
    inputFile >> nRows >> nCols;
    if( !inputFile.good() ) {
      std::cerr << __PRETTY_FUNCTION__ << ": Can't read matrix sizes in file " << matrixFile << std::endl;
      exit( EXIT_FAILURE );
    }
    std::cerr << matrixFile << ": nRows = " << nRows << "  nCols = " << nCols << std::endl;

    // Allocate the matrix
    allocate( nRows, nCols );

    // Loop
    for( unsigned int j=0; j<nCols; j++ ) {
      for( unsigned int i=0; i<nRows; i++ ) {
	unsigned int myI, myJ;
	inputFile >> myI >> myJ >> matrix[i+(j*nRows)];
	if( !inputFile.good() ) {
	  std::cerr << __PRETTY_FUNCTION__ << ": Bad read for " << i << " " << j << std::endl;
	  release();
	  exit( EXIT_FAILURE );
	}
	if( (i!=(myI-1)) || (j!=(myJ-1)) ) {
	  std::cerr << __PRETTY_FUNCTION__ << ": Location mismatch";
	  std::cerr << "(i,j)=( " << i << ", " << j << " )  ";
	  std::cerr << "(myI-1, myJ-1) = ( " << myI-1 << ", " << myJ-1 << " )" << std::endl;
	  release();
	  exit( EXIT_FAILURE );
	}
      }
    }

    std::cerr << matrixFile << ": Matrix read completed" << std::endl;
  }

  //! Destructor releases memory if required
  ~BlasMatrix( void ) {
    release();
  }
  

  //! Allocates the memory
  void allocate( const unsigned int rows, const unsigned int cols ) {
    
    if( matrix != NULL ) {
      release();
    }

    matrix = new T[rows*cols];
    nRows = rows;
    nCols = cols;
  }


  //! Releases the memory
  void release( void ) {
    if( matrix != NULL ) {
      delete[] matrix;
      matrix = NULL;
      nRows = 0;
      nCols = 0;
    }
  }

  //! Returns the size of the matrix in bytes
  size_t size( void ) const {
    /*!
      Computes the size in bytes from the number of rows and columns
    */
    return( nRows *nCols *sizeof(T) );
  }

  //! Returns the minimum and maximum values
  void GetMinMax( T &min, T &max ) const {
    if( matrix == NULL ) {
      std::cerr << __PRETTY_FUNCTION__ << ": Matrix not allocated" << std::endl;
      exit( EXIT_FAILURE );
    }

    min = matrix[0];
    max = matrix[0];

    for( unsigned int i=1; i<nRows*nCols; i++ ) {
      if( matrix[i] < min ) {
	min = matrix[i];
      }
      if( matrix[i] > max ) {
	max = matrix[i];
      }
    }
  }

  //! Returns the minimum and maximum values
  void GetAbsMinMax( T &min, T &max ) const {
    if( matrix == NULL ) {
      std::cerr << __PRETTY_FUNCTION__ << ": Matrix not allocated" << std::endl;
      exit( EXIT_FAILURE );
    }

    min = fabs(matrix[0]);
    max = fabs(matrix[0]);

    // Note we start from one!
    for( unsigned int i=1; i<nRows*nCols; i++ ) {
      if( fabs(matrix[i]) < min ) {
	min = fabs(matrix[i]);
      }
      if( fabs(matrix[i]) > max ) {
	max = fabs(matrix[i]);
      }
    }
  }
  

  // =============================
  // Variables

  T *matrix;
  unsigned int nRows;
  unsigned int nCols;


};




#endif
