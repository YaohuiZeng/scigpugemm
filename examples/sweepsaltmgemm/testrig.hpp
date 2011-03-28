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

void RunOneTest( const unsigned int randSeed,
		 const size_t memAllowed,
		 const char transA, const char transB,
		 const int m, const int n, const int k,
		 const int lda, const int ldb, const int ldc,
		 const double min, const double max,
		 const double minSalt, const double maxSalt,
		 const unsigned int nSalt,
		 const double alpha, const double beta,
		 const double cutOff,
		 const unsigned int nRepeats,
		 ostream &os );

void WriteHeader( ostream &os );

#endif
