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

#include <iostream>
#include <fstream>
#include <string>
using namespace std;



#include <cuda_runtime.h>

#include "cudacheck.hpp"

#include "blasmatrix.hpp"

#include "testrig.hpp"

// ================================================

void InitDevice( const int nDevice, ofstream &os ) {

  int deviceCount;
  float *d_temp;
  cudaDeviceProp prop;

  // Verify that there are CUDA devices available
  CUDA_SAFE_CALL( cudaGetDeviceCount( &deviceCount ) );
  if( deviceCount == 0 ) {
    cerr << "No device supporting CUDA available!" << endl;
    exit( EXIT_FAILURE );
  }

  // Check desired device is in range
  if( nDevice > deviceCount-1 ) {
    cerr << "Invalid CUDA device" << endl;
    exit( EXIT_FAILURE );
  }
  
  CUDA_SAFE_CALL( cudaGetDeviceProperties( &prop, nDevice ) );
  os << "# Device name: " << prop.name <<endl;
  
  
  // Set the device
  CUDA_SAFE_CALL( cudaSetDevice( nDevice ) );

  // Do a quick malloc/free to make sure it's there
  CUDA_SAFE_CALL( cudaMalloc( (void**)&d_temp, 1 ) );
  if( d_temp == NULL ) {
    cerr << "Failed to initialse GPU" << endl;
    exit( EXIT_FAILURE );
  }
  CUDA_SAFE_CALL( cudaFree( d_temp ) );
}


// ================================================

int main( void ) {
  string fileA, fileB, outFilename;
  ofstream outputFile;
  int iDevice;
  size_t memAllowed;
  char transA, transB;
  unsigned int nCutOffSteps, nRepeats;
  double cutMin, cutMax;

  // -----------------------------------------
  cout << "MGEMM Test" << std::endl;
  cout << "==========" << std::endl;

  cout << "Enter output filename" << endl;
  cin >> outFilename;

  outputFile.open( outFilename.c_str() );
  if( !outputFile.good() ) {
    cerr << "Failed to open output file " << outFilename << endl;
    return( EXIT_FAILURE );
  }

  cout << "Enter device number" << endl;
  cin >> iDevice;
  
  InitDevice( iDevice, outputFile );

  cout << "Enter allowed memory consumption" << endl;
  cin >> memAllowed;


  // ------------------------------------------
  // Get transpose options
  cout << "Enter transA, transB" << endl;
  cin >> transA >> transB;

  // Get the matrices
  cout << "Enter filename for matrix A" << endl;
  cin >> fileA;
  cout << "Enter filename for matrix B" << endl;
  cin >> fileB;

  BlasMatrix<double> A( fileA );
  BlasMatrix<double> B( fileB );

  double AabsMin, AabsMax;
  double BabsMin, BabsMax;

  A.GetAbsMinMax( AabsMin, AabsMax );
  B.GetAbsMinMax( BabsMin, BabsMax );
  cout << "Absolute size range of Matrix A: " << AabsMin << " " << AabsMax << endl;
  cout << "Absolute size range of Matrix B: " << BabsMin << " " << BabsMax << endl;

  // ------------------------------------------
  // Number of tests to run
  cout << "Enter cutMin, cutMax, nCutoffSteps" << endl;
  cin >> cutMin >> cutMax >> nCutOffSteps;
  if( (cutMin<0) || (cutMax<cutMin) ) {
    cerr << "Bad cutoff values" << endl;
    exit( EXIT_FAILURE );
  }

  cout << "Enter number of repeats" << endl;
  cin >> nRepeats;


  // -------------------------------------------
  
  outputFile << "# Memory allowed = " << memAllowed << endl;
  outputFile << "# transA, transB = " << transA << " " << transB << endl;
  outputFile << "# A -> " << A.nRows << "x" << A.nCols << endl;
  outputFile << "# B -> " << B.nRows << "x" << B.nCols << endl;
  outputFile << "# nRepeats " << nRepeats << endl;

  WriteHeader( outputFile );

  // ============================================
  // Run a test


  for( unsigned int iCutOff=0; iCutOff<nCutOffSteps; iCutOff++ ) {
    double currCutOff = log( cutMin ) + (iCutOff * log(cutMax/cutMin) / ( nCutOffSteps - 1 ) );
    currCutOff = exp( currCutOff );

    RunOneTest( memAllowed, transA, transB,
		A, B, nRepeats, currCutOff, outputFile );
  }


  return( EXIT_SUCCESS );
}
