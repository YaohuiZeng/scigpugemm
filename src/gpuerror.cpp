/*! \file
  File containing implementations of SciGPUgemm::GPUexception methods

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


#include "gpuerror.hpp"


namespace SciGPUgemm {
  // --------------------------------------------------------------------------------------
  
  GPUexception::GPUexception( const GPUerror myError,
			      const char* myFunction,
			      const char* myFile,
			      const unsigned long myLine,
			      const unsigned long myCUDAerr ) throw() : theError(myError),
									functionName(myFunction),
									fileName(myFile),
									lineNumber(myLine),
									cudaErrorCode(myCUDAerr) {
    /*!
      Full constructor function for a GPUexception.
      This should be used to create the exceptions the code throws
      @param[in] myError The actual error code being thrown
      @param[in] myFunction The function throwing the exception. Use the \c __PRETTY_FUNCTION__ macro
      @param[in] myFile The file throwing the exception. Use the \c __FILE__ macro
      @param[in] myLine The line throwing the exception. Use the \c __LINE__ macro
      @param[in] myCUDAerr The error (if applicable) reported by CUDA or CUBLAS
      @see SciGPU_GPUexception_THROW
      @see SciGPU_TRY_CUDA_RUNTIME
      @see SciGPU_TRY_CUDA_DRIVER
    */
  }


  // --------------------------------------------------------------------------------------

  const char* GPUexception::GetErrorString( const GPUexception::GPUerror errCode ) {
    /*!
      Routine to convert a GPUexception::GPUerror code into a human-readable string.
      This is declared static so that it may be called independently of a GPUexception object.
      If given an unrecognised code, it will abort the program
      @param[in] errCode The error we want converted
      @retval GetErrorString A character string describing the error
    */
  
    /*
      This was declared static in the class definition, but that's omitted here.
      If we try declaring the function static in this file, 'static' will mean
      'only visible in this file' (as is conventional for C).
      Don't you love thinking things through? :-/
    */
    
    switch( errCode ) {
    case GPUexception::eNoError:
      return "No error";
      break;
      
    case GPUexception::eMallocHost:
      return "Host pinned memory allocation failed";
      break;
      
    case GPUexception::eMallocGPU:
      return "GPU memory allocation failed";
      break;
      
    case GPUexception::eFreeHost:
      return "Host pinned memory free failed";
      break;
      
    case GPUexception::eFreeGPU:
      return "GPU memory free failed";
      break;
      
    case GPUexception::eTransfer:
      return "Transfer between host and GPU failed";
      break;
      
    case GPUexception::eMiscCUDAruntime:
      return "CUDA Runtime API error occurred";
      break;
      
    case GPUexception::eMiscCUDAdriver:
      return "CUDA Driver API error occurred";
      break;
      
    case GPUexception::eCUBLAS:
      return "CUBLAS Error occurred";
      break;
      
    
    default:
      std::cerr << __PRETTY_FUNCTION__ << ": Unrecognised GEMMerror value" << std::endl;
      exit( EXIT_FAILURE );
    }
  }
  


  // --------------------------------------------------------------------------------------

  void GPUexception::printout( void ) {
    /*!
      This routine prints out a full description of the
      exception encountered to std::cerr.
      This includes the throwing routine, the file containing the routine
      and the line number on which the error was detected.
      The return value from a CUDA API call is also printed out; this is only
      valid if the exception was thrown from such a call.
      Otherwise it will be undefined.
    */

    std::cerr << std::endl;
    std::cerr << "Exception : " << GetErrorString( theError ) << std::endl;
    std::cerr << "Thrown by function : " << functionName << std::endl;
    std::cerr << "Line number " << lineNumber << " of file " << fileName << std::endl;
    std::cerr << "CUDA error (if applicable) was " << cudaErrorCode << std::endl;
    std::cerr << std::endl;
    
  }

}
