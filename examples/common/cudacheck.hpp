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


/*! \file
  File to contain simple CUDA error checking macros.
  Taken from the CUDA SDK
*/


#ifndef CUDACHECK_H
#define CUDACHECK_H

#include <cstdlib>
#include <cstdio>

#define CUDA_SAFE_CALL( call ) do {					\
  cudaError err = call;							\
  if( cudaSuccess != err ) {						\
  fprintf( stderr, "CUDA Error in file '%s' on line %i : %s.\n",	\
	   __FILE__, __LINE__, cudaGetErrorString( err ) );		\
  exit( EXIT_FAILURE );							\
  } } while( 0 );

#endif
