/*
* Example of how to use the mxGPUArray API in a MEX file.  This example shows
* how to write a MEX function that takes a gpuArray as input and returns a
* gpuArray output for Mandelbrot Example solution, e.g. B=mexFunction(A).
*
* by Syed Alam Abbas, 5/23/2015
*/
#include <arrayfire.h>
#include <af/util.h>
#include "cuda_runtime.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"

using namespace af;
/*
* Host code
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	/* Initialize the MathWorks GPU API. */
	mxInitGPU();

	mexPrintf("Executing custom mex for computing Mandelbrot set using ArrayFire GPU accelerated library !\n");
	
	// Validate the input
	if ( nrhs < 5 || nlhs < 1 ) {
		mexErrMsgTxt("Expected 5 inputs and 1 output.");
	}

	/*Input Variables*/
	mxGPUArray const *xVal;
	mxGPUArray const *yVal;
	mxGPUArray const *Count;
	mxGPUArray*  intermediateTemp;
	int d_MaxIterations;
	int d_NumberOfElements;

	/*Device Temp Variables */
	const double* d_xVal;
	const double* d_yVal;
	const double* d_Count;
	
	/*Output Variable*/
	double* d_CountOutImage;
	
	/* Collect the input data from MATLAB MexArray RHS */
	xVal = mxGPUCreateFromMxArray(prhs[0]);
	yVal = mxGPUCreateFromMxArray(prhs[1]);   
	Count = mxGPUCreateFromMxArray(prhs[2]);
	d_MaxIterations = (size_t) mxGetScalar(prhs[3]);
	d_NumberOfElements = (size_t)mxGetScalar(prhs[4]);


	/* extract a pointer to the input data on the device.*/
	d_xVal = (double const *)(mxGPUGetDataReadOnly(xVal));
	d_yVal = (double const *)(mxGPUGetDataReadOnly(yVal));
	d_Count = (double const *)(mxGPUGetDataReadOnly(Count));

	/* Create a GPUArray to hold the result and get its underlying pointer. */
	intermediateTemp = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(Count),
		mxGPUGetDimensions(Count),
		mxGPUGetClassID(Count),
		mxGPUGetComplexity(Count),
		MX_GPU_DO_NOT_INITIALIZE);
	d_CountOutImage = (double *)(mxGPUGetData(intermediateTemp));

	/* Copy Input Value from Count */
	cudaMemcpy(d_CountOutImage, d_Count, d_NumberOfElements* d_NumberOfElements* sizeof(double), cudaMemcpyDeviceToDevice);

	/* Using ArrayFire Code for processing Now */
	array array_X(d_NumberOfElements, d_xVal);
	array array_Y(d_NumberOfElements, d_yVal);
	array CountImage(d_NumberOfElements, d_NumberOfElements, d_CountOutImage);

	array array_X_Tiled = tile(array_X, 1, d_NumberOfElements);
	array array_Y_Tiled = tile(array_Y.T(), d_NumberOfElements, 1);

	array Z_0_complex = complex(array_X_Tiled, array_Y_Tiled);
	array Solution(Z_0_complex);
	for (int ii = 0; ii < (d_MaxIterations + 1); ii++)
	{
		Solution = Solution * Solution + Z_0_complex;
		CountImage = CountImage + (abs(Solution) <= 2);
	}
	CountImage = log(CountImage);
	
	/* Copy Processed Values to Output*/
	double* d_Processed = CountImage.device<double>();
	cudaMemcpy(d_CountOutImage, d_Processed, d_NumberOfElements* d_NumberOfElements* sizeof(double), cudaMemcpyDeviceToDevice);

	/* Wrap the result up as a MATLAB gpuArray for return. */
	plhs[0] = mxGPUCreateMxArrayOnGPU(intermediateTemp);
	
	/*
	* The mxGPUArray pointers are host-side structures that refer to device
	* data. These must be destroyed before leaving the MEX function.
	*/
	mxGPUDestroyGPUArray(yVal);
	mxGPUDestroyGPUArray(xVal);
	mxGPUDestroyGPUArray(Count);

	mexPrintf("Finished processing custom CUDA mex with ArrayFire, Status = Success\n");
}
