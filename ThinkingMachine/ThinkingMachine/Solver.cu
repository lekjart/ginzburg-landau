#include <windows.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include "Solver.h"

// Run on 32x32 block grid, each with 32x32 threads
dim3 gridConf(GRIDSIZEX, GRIDSIZEY, 1);
dim3 blockConf(TILESIZE, TILESIZE, 1);

// Arrays that will keep old and new (u,v) fields
float *d_nU, *d_oU;
float *d_nV, *d_oV;
bool usePeriodic = true;

//Constant fast device memory vector for simulation parameters
float h_Para[PARACOUNT];
__constant__ float para[PARACOUNT];

// Constant fast memory vector for palette
__constant__ COLORREF anchors[4];

// Cuda related variables
cudaError_t error;
cudaEvent_t start, stop;
cudaGraphicsResource_t cudaResourceBuf;

// Flag to tell if all is properly set up
bool isReady = false;

void UpdatePalette(const COLORREF *colors)
{
	// Update device constant parameter array
	checkCudaErrors(cudaMemcpyToSymbol(anchors, colors, 4 * sizeof(COLORREF)));
	cudaDeviceSynchronize();
}

void UpdateCudaParas(void)
{
	// Update device constant parameter array
	checkCudaErrors(cudaMemcpyToSymbol(para, h_Para, PARACOUNT * sizeof(float)));
	cudaDeviceSynchronize();
}

void CreateZero(unsigned int x, unsigned int y)
{
	y = FIELDSIZEY - 1 - y;


	int x0 = x;
	int y0 = y;
	float phase;
	int bx;
	int tx;
	int by;
	int ty;
	int blockId;
	int threadId;
	float radius = 60.0f;
	float r;
	float u0, v0, un, vn;
	u0 = 0.0f;
	v0 = 1.0f;


	for (int i = 0; i < FIELDSIZEY; i++) {
		for (int j = 0; j < FIELDSIZEX; j++) {

			r = h_Para[P_DX]*sqrt((j-x0)*(j-x0)+(i-y0)*(i-y0));
			phase = 1.0f*atan2f(i-y0, j-x0);

			// Get the linear memory location
			bx = j / TILESIZE;
			tx = j % TILESIZE;
			by = i / TILESIZE;
			ty = i % TILESIZE;

			blockId = bx + by * GRIDSIZEX;
			threadId = blockId * (TILESIZE * TILESIZE)
				+ (ty * TILESIZE) + tx;

			
			un = tanhf(r)*(u0*cosf(phase) - v0*sinf(phase));
			vn = tanhf(r)*(u0*sinf(phase) + v0*cosf(phase));

			d_nU[threadId] = un*d_oU[threadId] - vn * d_oV[threadId];
			d_nV[threadId] = un*d_oV[threadId] + vn * d_oU[threadId];

			d_oU[threadId] = d_nU[threadId];
			d_oV[threadId] = d_nV[threadId];

			if (i % 2 == 3) {
				printf("%f %f\n", d_nU[threadId], d_nV[threadId]);
			}
		}
	}

	/*
	for (int j = 1; j < int(radius); j++) {
		r = j;
		for (int i = 0; i < 128; i++) {
			phase = i *2.0f*3.14159265358979323846f / 127.0f;
			x = int(x0 + r * cosf(phase)) % (FIELDSIZEX);
			y = int(y0 + r * sinf(phase)) % (FIELDSIZEY);


			bx = x / TILESIZE;
			tx = x % TILESIZE;
			by = y / TILESIZE;
			ty = y % TILESIZE;

			blockId = bx + by * GRIDSIZEX;
			threadId = blockId * (TILESIZE * TILESIZE)
				+ (ty * TILESIZE) + tx;

			d_nU[threadId] = (r/radius)*cosf(phase);
			d_nV[threadId] = (r / radius)*sinf(phase);
			d_oU[threadId] = (r / radius)*cosf(phase);
			d_oV[threadId] = (r / radius)*sinf(phase);
		}
	}

	*/


}

void ResetField(void)
{
	float noise = h_Para[P_NOISE];
	// Initialize all buffers to zero
	for (int i = 0; i < FIELDSIZEX*FIELDSIZEY; i++) {
		d_nU[i] = h_Para[P_U0] + noise*(float(rand()) / float(RAND_MAX) - 0.5f);
		d_oU[i] = h_Para[P_U0] + noise*(float(rand()) / float(RAND_MAX) - 0.5f);
		d_nV[i] = h_Para[P_V0] + noise*(float(rand()) / float(RAND_MAX) - 0.5f);
		d_oV[i] = h_Para[P_V0] + noise*(float(rand()) / float(RAND_MAX) - 0.5f);
	}

}

void InitSolver(int imageBuf,void **pixelBuffer)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	// Create GPU timers
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// Associate cuda with the GL device
	//checkCudaErrors(cudaGLSetGLDevice(0));

	// Register the GL buffer to be used for drawing
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaResourceBuf, imageBuf, cudaGraphicsRegisterFlagsNone));

	printf("Allocating memory\n");
	// Allocate managed memory for old and new (U,V) field
	void *buf;
	checkCudaErrors(cudaMallocManaged((void **)&d_nU, FIELDSIZEX * FIELDSIZEY * sizeof(float)));
	checkCudaErrors(cudaMallocManaged((void **)&d_oU, FIELDSIZEX * FIELDSIZEY * sizeof(float)));
	checkCudaErrors(cudaMallocManaged((void **)&d_nV, FIELDSIZEX * FIELDSIZEY * sizeof(float)));
	checkCudaErrors(cudaMallocManaged((void **)&d_oV, FIELDSIZEX * FIELDSIZEY * sizeof(float)));
	checkCudaErrors(cudaMallocManaged((void**)&buf, FIELDSIZEX * FIELDSIZEY * sizeof(char) * 3));
	*pixelBuffer = buf;

	printf("Initializing memory\n");
	h_Para[P_ALPHA] = 1.0f;
	h_Para[P_BETA] = 1.1f;
	h_Para[P_NU] = 0.0f;
	h_Para[P_G0] = 0.0f;
	h_Para[P_G1] = 0.0f;
	h_Para[P_G2] = 0.0f;
	h_Para[P_G3] = 0.0f;
	h_Para[P_DT] = 0.01f;
	h_Para[P_DX] = 0.85f;
	h_Para[P_CLAMP] = 1.0f;
	h_Para[P_NOISE] = 0.0f;
	h_Para[P_U0] = 0.0f;
	h_Para[P_V0] = 0.0f;

	UpdateCudaParas();

	// put a little noise at the very beginning
	h_Para[P_NOISE] = 0.1;
	ResetField();
	h_Para[P_NOISE] = 0.0;

	isReady = true;
}

void ReleaseSolver(void *pixelBuffer)
{
	cudaFree(d_oU);
	cudaFree(d_oV);
	cudaFree(d_nU);
	cudaFree(d_nV);
	cudaFree(pixelBuffer);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// Swap old and new (u,v) fields
void SwapFields(void)
{
	float *tmpSwap;
	tmpSwap = d_nU;
	d_nU = d_oU;
	d_oU = tmpSwap;

	tmpSwap = d_nV;
	d_nV = d_oV;
	d_oV = tmpSwap;
}


// Helper function that returns the proper offset into 2D array for a given thread
__device__
int getGlobalIdx_2D_2D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

// Helper function that returns the proper value at a given cell in block-structured memory
__device__
float getValueAt(const float *a, int i, int j) {

	int bx = j / TILESIZE;
	int tx = j % TILESIZE;
	int by = i / TILESIZE;
	int ty = i % TILESIZE;

	int blockId = bx + by * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (ty * blockDim.x) + tx;

	return a[threadId];
}

// Kernel function that maps the (u,v) field into a (r,g,b,a) texture buffer 
__global__
void FillBuf(unsigned int *buf, const float *u, const float *v, int currentData, void *pixelBuffer)
{
	unsigned int jj = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ii = blockIdx.y * blockDim.y + threadIdx.y;
	int id = ii*FIELDSIZEX + jj;

	unsigned char *pixel = (unsigned char *)pixelBuffer +id*3;
	unsigned char *pixel2 = (unsigned char *)&buf[id];
	float u0 = u[getGlobalIdx_2D_2D()];
	float v0 = v[getGlobalIdx_2D_2D()];
	float val = 0.0f;

	switch (currentData)
	{
	case 0:
		val = u0*u0 + v0*v0;
		val = tanhf(para[P_CLAMP] * val);
		break;
	case 1:
		val = atan2f(v0, u0)/3.14159265358979323846f;
		val = tanhf(para[P_CLAMP] * val);
		val = 0.5f*(val + 1.0f);
		break;
	case 2:
		val = tanhf(para[P_CLAMP] * u0);
		val = 0.5f*(val + 1.0f);
		break;
	case 3:
		val = tanhf(para[P_CLAMP] * v0);
		val = 0.5f*(val + 1.0f);
		break;
	}	

	// Map to current Bezier palette
	float rval, gval, bval;
	float val3 = val*val*val;
	float val2 = val*val;
	float mval = 1.0f - val;

	rval = val3*GetRValue(anchors[3]);
	gval = val3*GetGValue(anchors[3]);
	bval = val3*GetBValue(anchors[3]);

	rval += 3.0f*mval*val2*GetRValue(anchors[2]);
	gval += 3.0f*mval*val2*GetGValue(anchors[2]);
	bval += 3.0f*mval*val2*GetBValue(anchors[2]);

	mval *= mval;

	rval += 3.0f*mval*val*GetRValue(anchors[1]);
	gval += 3.0f*mval*val*GetGValue(anchors[1]);
	bval += 3.0f*mval*val*GetBValue(anchors[1]);

	mval *= (1.0f-val);

	rval += mval*GetRValue(anchors[0]);
	gval += mval*GetGValue(anchors[0]);
	bval += mval*GetBValue(anchors[0]);


	pixel[2] = (unsigned char)rval;
	pixel[1] = (unsigned char)gval;
	pixel[0] = (unsigned char)bval;

	pixel2[0] = (unsigned char)rval;
	pixel2[1] = (unsigned char)gval;
	pixel2[2] = (unsigned char)bval;
	pixel2[3] = 0;

}


// Actual kernel solver
__global__
void Advance(const float *d_ou, const float *d_ov, float *d_u, float *d_v)
{
	// Declare a shared memory buffer
	__shared__ float ou[TILESIZE][TILESIZE];
	__shared__ float ov[TILESIZE][TILESIZE];

	// Cache the tile itself into shared memory
	int globalId = getGlobalIdx_2D_2D();
	ou[threadIdx.y][threadIdx.x] = d_ou[globalId];
	ov[threadIdx.y][threadIdx.x] = d_ov[globalId];

	// Wait for all threads to have fetched the data
	//__syncthreads();

	unsigned int i = threadIdx.y;
	unsigned int j = threadIdx.x;

	// Global (i,j) coordinates
	unsigned int jj = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ii = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int iiMin = blockIdx.y * blockDim.y;
	unsigned int iiMax = iiMin + TILESIZE - 1;

	unsigned int jjMin = blockIdx.x * blockDim.x;
	unsigned int jjMax = jjMin + TILESIZE - 1;

	/*
	// Periodic boundary conditions
	unsigned int upId = (ii - 1) % FIELDSIZEY;
	unsigned int downId = (ii + 1) % FIELDSIZEY;
	unsigned int leftId = (jj - 1) % FIELDSIZEX;
	unsigned int rightId = (jj + 1) % FIELDSIZEX;
	
	// Get neighbors within shared memory or global memory depending on values
	bool withinTile;


	withinTile = (upId <= iiMax) && (upId >= iiMin) && false;
	float upU = (withinTile ? ou[threadIdx.y - 1][threadIdx.x] : getValueAt(d_ou, upId, jj));
	float upV = (withinTile ? ov[threadIdx.y - 1][threadIdx.x] : getValueAt(d_ov, upId, jj));

	withinTile = (downId <= iiMax) && (downId >= iiMin) && false;
	float downU = (withinTile ? ou[threadIdx.y + 1][threadIdx.x] : getValueAt(d_ou, downId, jj));
	float downV = (withinTile ? ov[threadIdx.y + 1][threadIdx.x] : getValueAt(d_ov, downId, jj));

	withinTile = (leftId <= jjMax) && (leftId >= jjMin) && false;
	float leftU = (withinTile ? ou[threadIdx.y][threadIdx.x - 1] : getValueAt(d_ou, ii, leftId));
	float leftV = (withinTile ? ov[threadIdx.y][threadIdx.x - 1] : getValueAt(d_ov, ii, leftId));

	withinTile = (rightId <= jjMax) && (rightId >= jjMin) && false;
	float rightU = (withinTile ? ou[threadIdx.y][threadIdx.x + 1] : getValueAt(d_ou, ii, rightId));
	float rightV = (withinTile ? ov[threadIdx.y][threadIdx.x + 1] : getValueAt(d_ov, ii, rightId));
	
	*/
	// Fixed boundary conditions

	//unsigned int upId = std::abs(int(ii) - 1);
	unsigned int upId = std::fmax(float(ii) - 1.0f, 0.0f);
	unsigned int downId = std::fmin(float(ii + 1),  float(FIELDSIZEY -1));
	//unsigned int leftId = std::abs(int(jj) - 1);
	unsigned int leftId = std::fmax(float(jj) - 1.0f, 0.0f);
	unsigned int rightId = std::fmin(float(jj + 1), float(FIELDSIZEX - 1));;

	// Get neighbors within shared memory or global memory depending on values
	bool withinTile;


	float upU = getValueAt(d_ou, upId, jj);
	float upV = getValueAt(d_ov, upId, jj);

	float downU = getValueAt(d_ou, downId, jj);
	float downV = getValueAt(d_ov, downId, jj);

	float leftU = getValueAt(d_ou, ii, leftId);
	float leftV = getValueAt(d_ov, ii, leftId);

	float rightU = getValueAt(d_ou, ii, rightId);
	float rightV = getValueAt(d_ov, ii, rightId);

	float gg2 = para[P_G2];
	float gg1 = para[P_G1];
	float gg0 = para[P_G0];
	float nu = para[P_NU];

	// Examples of ramp of parameters acrosss the field
	//float gg2 =  (para[P_G2] * (ii) / (FIELDSIZEY - 1.0));;// sqrt(para[P_G0] * (jj) / (FIELDSIZEX - 1.0));
	//float gg1 =  (para[P_G1] * (ii) / (FIELDSIZEY - 1.0));
	//float gg0 =  (para[P_G0] * (ii) / (FIELDSIZEY - 1.0));
	//float nu = (para[P_NU] * (jj) / (FIELDSIZEX - 1.0));
	//float nu =  para[P_NU] * 2.0f*(int(jj) - FIELDSIZEX / 2) / (FIELDSIZEX - 1.0) + .7*para[P_BETA];
	float u = ou[i][j];
	float v = ov[i][j];
	float u2 = u*u;
	float u3 = u2*u;
	float v2 = v*v;
	float v3 = v2*v;

	float dxx = 1.0 / (para[P_DX] * para[P_DX]);
	float dUU = dxx*(rightU + leftU + upU + downU - 4.0f*u);
	float dVV = dxx*(rightV + leftV + upV + downV - 4.0f*v);
	float a0 = (v2 - u2)*gg2 + gg0 - para[P_ALPHA] * dVV + dUU + (-2.0f * v3 - 2.0f * u2 * v)*para[P_BETA] + 2.0f * u*v2 + 2.0f * u3;
	float a1 = 2.0f * u*gg2 + gg1 + 2.0f * u*v*para[P_BETA] - v2 - 3.0f * u2 + 1.0f;
	float a2 = -nu - 2.0f * v*gg2 + (3.0f * v2 + u2)*para[P_BETA] - 2.0f * u*v;

	float b0 = 2.0f * u*v*gg2 + dVV + para[P_ALPHA]*dUU + (2.0f * u*v2 + 2.0f * u3)*para[P_BETA] + 2.0f * v3 + 2.0f * u2 * v;
	float b1 = nu - 2.0f * v*gg2 + (-v2 - 3.0f * u2)*para[P_BETA] - 2.0f * u*v;
	float b2 = -2.0f * u*gg2 - gg1 - 2.0f * u*v*para[P_BETA] - 3.0f * v2 - u2 + 1.0f;

	float det = 1.0f / ((a2*b1 - a1*b2)*para[P_DT] * para[P_DT] + (b2 + a1)*para[P_DT] - 1.0f);

	d_u[globalId] = -((a2*b0 - a0*b2)*para[P_DT] * para[P_DT] + (-u * b2 + v * a2 + a0)*para[P_DT] + u) * det;
	d_v[globalId] = ((a1*b0 - a0*b1)*para[P_DT] * para[P_DT] + (-u * b1 + v * a1 - b0)*para[P_DT] - v) * det;
}

/*
float a0 = gg0 - para[P_ALPHA] * dVV + dUU - 2.0f * v3 * para[P_BETA] + 2.0f * u3;
float a1 = gg1 + 2.0f * u * v * para[P_BETA] - v2 - 3.0f * u2 + 1.0f;
float a2 = (3.0f * v2 - u2)*para[P_BETA] - para[P_NU];

float b0 = dVV + para[P_ALPHA] * dUU + 2.0f * u3 * para[P_BETA] + 2.0f * v3;
float b1 = para[P_NU] + (v2 - 3.0f * u2)*para[P_BETA];
float b2 = -gg1 - 2.0f * u * v * para[P_BETA] - 3.0f * v2 - u2 + 1.0f;
float det = 1.0f / ((a2*b1 - a1*b2)*para[P_DT] * para[P_DT] + (b2 + a1)*para[P_DT] - 1.0f);



float a0 = gg0 - para[P_ALPHA] * dVV + dUU - 2.0f * v3 * para[P_BETA] + 2.0f * u3 + gg2*(u2-v2);
float a1 = gg1 + 2.0f * u * v * para[P_BETA] - v2 - 3.0f * u2 + 1.0 + 2.0f*gg2*u;
float a2 = (3.0f * v2 + u2)*para[P_BETA] - para[P_NU] - 2.0f*gg2* v -2*u*v;

float b0 = dVV + para[P_ALPHA] * dUU + 2.0f * u3 * para[P_BETA] + 2.0f * v3;
float b1 = para[P_NU] + (-v2 - 3.0f * u2)*para[P_BETA] - 2.0f*gg2*u -2.0f*u*v;
float b2 = -gg1 - 2.0f * u * v * para[P_BETA] - 3.0f * v2 - u2 + 1.0f-2.0*gg2*u;

float det = 1.0f / ((a2*b1 - a1*b2)*para[P_DT] * para[P_DT] + (b2 + a1)*para[P_DT] - 1.0f);
*/

bool DoStuff(float *elapsedTime, int currentData, int count, void *pixelBuffer)
{
	if (!isReady)
		return false;

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < count; i++) {
		// Swap old and new
		SwapFields();

		Advance <<<gridConf, blockConf >>>(d_oU, d_oV, d_nU, d_nV);
		
		checkCudaErrors(cudaDeviceSynchronize());
		//printf("Elapsed: %7.3f\n ms", elapsedTime);
	}

	// Map the GL texture into device buffer
	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResourceBuf, 0));
	void *devBuf;
	size_t bufLen;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&devBuf, &bufLen, cudaResourceBuf));

	// Fill buffer with new image
	FillBuf << <gridConf, blockConf >> >((unsigned int *)devBuf, d_nU, d_nV, currentData, pixelBuffer);
	checkCudaErrors(cudaDeviceSynchronize());

	// Unmap buffer
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResourceBuf, 0));

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(elapsedTime, start, stop));

	return true;
}

