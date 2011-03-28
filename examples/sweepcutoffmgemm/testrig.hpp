// Header file for the test rig

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

#ifndef TESTRIG_H
#define TESTRIG_H

#include "blasmatrix.hpp"

void RunOneTest( const size_t memAllowed,
		 const char transA, const char transB,
		 const BlasMatrix<double> &A, const BlasMatrix<double> &B,
		 const unsigned int nRepeats,
		 const double cutOff,
		 ostream &os );

void WriteHeader( ostream &os );

#endif
