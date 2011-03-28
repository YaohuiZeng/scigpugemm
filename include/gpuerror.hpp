/*! \file
  File containing errors and exceptions for the SciGPU-GEMM library

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

#ifndef GPUERROR_H
#define GPUERROR_H

#include <cstdlib>
#include <string>
#include <sstream>
#include <iostream>
#include <exception>



namespace SciGPUgemm { 
  //! Exception class for GPU routines
  /*!
    \internal
    When any of the GPU routines encounters an error,
    one of these exceptions will be thrown.
    It is not anticipated that these exceptions will ever reach
    user code
    @warning Contains std::string objects for extra information, which is not necessarily best practice
  */
  class GPUexception : public std::exception {
    
  public:  
    //! Error enumeration for GPU routines
    /*!
      This enumeration lists the error codes for GPUexception.
      
    */
    enum GPUerror {
      eNoError, //!< No error. Should not be in an exception
      eMallocHost, //!< Insufficient pinned host memory available
      eMallocGPU, //!< Insufficient GPU memory available
      eFreeHost, //!< Could not free pinned host memory
      eFreeGPU, //!< Could not free GPU memory
      eTransfer, //!< Copy between host and GPU failed
      eMiscCUDAruntime, //!< A CUDA runtime API error occurred
      eMiscCUDAdriver, //!< A CUDA driver API error occurred
      eCUBLAS //!< CUBLAS call failed
    };
    
    //! Default constructor will abort program
    GPUexception( void ) : theError(eNoError),
			   functionName(""),
			   fileName(""),
			   lineNumber(0),
			   cudaErrorCode(0) {
      /*!
	It doesn't make sense to create an exception without specifying
	an error code.
	To make the best of the situation, this will cause the whole
	program to abort.
      */
      std::cerr << __PRETTY_FUNCTION__ << ": No error specified" << std::endl;
      exit( EXIT_FAILURE );
    }
    
    //! Constructor which specifies the error
    GPUexception( const GPUerror myError,
		  const char* myFunction,
		  const char* myFile,
		  const unsigned long myLine,
		  const unsigned long myCUDAerr ) throw();
    
    
    //! Destructor to keep compiler happy
    ~GPUexception( void ) throw() {
      /*!
	This is a null destructor which can't throw exceptions.
	This exception class contains string objects, and those
	can throw exceptions.
	However, the std::exception destructor cannot.
	If a string throws an exception in its destructor,
	it will cause the program to die in this destructor
	(since this destructor cannot throw an exception itself)
      */
    };
    
    //! Routine to translate GEMMerror into a string
    static const char* GetErrorString( const GPUerror errCode );
    
    
    //! Over-ridden function to provide textual description of the exception
    virtual const char* what( void ) const throw()  {
      
      return( GetErrorString( theError ) );
    }
    
    //! Routine to print out a full description of the exception
    void printout( void );
    
    //! Accessor function for the error code
    GPUerror getError( void ) {
      return this->theError;
    }
    
  private:
    // =======================================================
    // Variables
    
    //! The error signalled by this exception
    GPUerror theError;
    //! The name of the function which threw the exception
    std::string functionName;
    //! The name of the file which threw the exception
    std::string fileName;
    //! The linenumber where the exception was created
    unsigned long lineNumber;
    //! If appropriate, the CUDA error code
    unsigned long cudaErrorCode;
    
  };
  
}


//! Macro to simplify throwing of exceptions
/*!
  This macro removes the need to include the \c __PRETTY_FUNCTION__, \c __FILE__ and \c __LINE__
  macros when throwing a GPUexception.
  This macro automatically creates an exception with the appropriate parameters set, and then
  throws it.
  @param[in] except The exception to be thrown
  @param[in] errCode The error code from CUDA or CUBLAS
*/
#define SciGPU_GPUexception_THROW( except, errCode ) do {			\
    GPUexception myException( except, __PRETTY_FUNCTION__, __FILE__, __LINE__, errCode ); \
    throw myException;							\
  } while( 0 );


//! CUDA runtime error notification macro
/*!
  This macro prints out information about a CUDA error from the runtime API,
  if the preprocessor macro _DEBUG is defined.
    Otherwise it is a null operation
    @param[in] err The error code from a CUDA runtime API call
*/
#ifdef _DEBUG
#define SciGPU_CUDA_RUNTIME_ERROR_PRINT( err ) do {				\
    std::cerr << "CUDA Error in file " << __FILE__ << " on line " << __LINE__; \
    std::cerr << " : " << cudaGetErrorString( err ) << std::endl;	\
  } while( 0 );
#else
#define SciGPU_CUDA_RUNTIME_ERROR_PRINT( err ) ;
#endif



//! CUDA driver error notification macro
/*!
  This macro prints out information about a CUDA error from the driver API,
  if the preprocessor macro _DEBUG is defined.
  Otherwise it is a null operation
  @param[in] err The error code from a CUDA driver API call
*/
#ifdef _DEBUG
#define SciGPU_CUDA_DRIVER_ERROR_PRINT( err ) do {				\
    std::cerr << "CUDA driver error in file " << __FILE__ " on line " << __LINE__; \
    std::cerr << " : " << err << std::endl;				\
  } while( 0 );
#else
#define SciGPU_CUDA_DRIVER_ERROR_PRINT( err ) ;
#endif


//! Macro to simplify checks on the CUDA runtime
/*!
  This is a wrapper macro, similar to CUDA_SAFE_CALL in the CUDA SDK.
  It runs the call, and throws an exception if the call fails. It must be supplied with
  an appropriate GEMMexception::GEMMerror error to throw as the exception
  @param[in] call The CUDA call to be tried
  @param[in] except The error to be reported. Should be a GEMMexception::GEMMerror appropriate to \a call
*/
#define SciGPU_TRY_CUDA_RUNTIME( call, except ) do {	\
    cudaError err = call;			\
    if( cudaSuccess != err ) {			    \
      SciGPU_CUDA_RUNTIME_ERROR_PRINT( err );		    \
      SciGPU_GPUexception_THROW( except, err );	    \
    } } while ( 0 );


//! Macro to simplify checks on the CUDA driver
/*!
  This is a wrapper macro, similar to CU_SAFE_CALL in the CUDA SDK
  It runs the call and throws a GEMMexception::eMiscCUDAdriver exception
  if it fails
  @param[in] call The CUDA driver call to be tried
*/
#define SciGPU_TRY_CUDA_DRIVER( call ) do { \
    CUresult err = call; \
    if( CUDA_SUCCESS != err ) { \
      SciGPU_CUDA_DRIVER_ERROR_PRINT( err ); \
      SciGPU_GPUexception_THROW( GPUexception::eMiscCUDAdriver, err ); \
    } } while ( 0 );




#endif
