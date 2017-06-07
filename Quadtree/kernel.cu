
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bitmap.h"
#include <stdio.h>
#include <math.h>
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#define GRAUWERTBILD 1
#define SOBEL 1

#define DELTA_PREFIX_CHANGE 1
#define MIN_QUAD_WIDTH 16
#define AT(x,y) ((y)*width+(x))


struct Quadrant {
	int x0,y0;
	int x1,y1;
};

struct invert
{
	__device__
	void operator()(RGBA &rgba){
		rgba.Blue = 255 - rgba.Blue;
		rgba.Green = 255 - rgba.Green;
		rgba.Red = 255 - rgba.Red;
	}
};

struct grauwert {
	__device__
	void operator()(RGBA &rgba) {
		double grauwert = 0.114 * rgba.Blue + 0.587 * rgba.Green + 0.299 * rgba.Red;
		rgba.Blue = grauwert;
		rgba.Green = grauwert;
		rgba.Red = grauwert;
	}
};

/*struct sobel
{
	RGBA *rgba;
	__device__
	void operator()(int idx) {

	}
};*/

__global__ void sobel(RGBA *deviceRgbaBild, RGBA *deviceSobelResult)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//Rand
	if (threadIdx.x == 0 ) {
		
	}
	
	int obenlinks = deviceRgbaBild[(blockIdx.x - 1) *blockDim.x + (threadIdx.x - 1)].Blue;
	int obenmitte = deviceRgbaBild[(blockIdx.x - 1) *blockDim.x + (threadIdx.x)].Blue;
	int obenrechts = deviceRgbaBild[(blockIdx.x - 1) *blockDim.x + (threadIdx.x) + 1].Blue;

	int mittelinks = deviceRgbaBild[(blockIdx.x) *blockDim.x + (threadIdx.x - 1)].Blue;
	int mittemitte = deviceRgbaBild[idx].Blue;
	int mitterechts = deviceRgbaBild[(blockIdx.x) *blockDim.x + (threadIdx.x) + 1].Blue;

	int untenlinks = deviceRgbaBild[(blockIdx.x + 1) *blockDim.x + (threadIdx.x - 1)].Blue;
	int untenmitte = deviceRgbaBild[(blockIdx.x + 1) *blockDim.x + (threadIdx.x)].Blue;
	int untenrechts = deviceRgbaBild[(blockIdx.x + 1) *blockDim.x + (threadIdx.x) + 1].Blue;

	float sobelH = 1 * obenlinks - 1 * obenrechts +
		2 * mittelinks - 2 * mitterechts +
		1 * untenlinks - 1 * untenrechts;

	float sobelV = -1 * obenlinks - 2 * obenmitte - 1 * obenrechts +
		1 * untenlinks + 2 * untenmitte + 1 * untenrechts;

	int sobelWert = (int) sqrt(sobelH*sobelH + sobelV*sobelV);
	deviceSobelResult[idx].Blue = sobelWert;
	deviceSobelResult[idx].Green = sobelWert;
	deviceSobelResult[idx].Red = sobelWert;
}

__global__ void prefixRow(RGBA *deviceRgba, int *prefixArray, int bildBreite)
{
	int idx = bildBreite * threadIdx.x;
	int sum = 0;
	for (auto i = idx; i < idx + bildBreite; i++) {
		sum += deviceRgba[i].Blue;
		prefixArray[i] = sum;
	}
}

__global__ void prefixColumn(int *prefixArray, int bildBreite, int bildHoehe)
{
	int idx = threadIdx.x;
	int sum = 0;
	for (auto i = idx; i < bildBreite * bildHoehe; i = i + bildBreite) {
		sum += prefixArray[i];
		prefixArray[i] = sum;
	}
}

void drawLine(RGBA* rgba, int bildBreite, int bildHoehe, Quadrant quadrant) {
	for (int x = quadrant.x0; x < quadrant.x1; x++) {
		//rgba[quadrant.y0 + ((quadrant.y1 - quadrant.y0)/2) * bildBreite + x].Red = 255;
		rgba[(bildHoehe - quadrant.y0) * bildBreite + x].Red = 255;
		rgba[(bildHoehe - quadrant.y1) * bildBreite + x].Red = 255;
	}

	for (int y = quadrant.y0; y < quadrant.y1; y++) {
		//rgba[y * bildBreite + quadrant.x0 + ((quadrant.x1 - quadrant.x0) / 2)].Red = 255;
		rgba[(bildHoehe - y) * bildBreite + quadrant.x0].Red = 255;
		rgba[(bildHoehe - y) * bildBreite + quadrant.x1].Red = 255;
	}
}
__global__ void integralRow(RGBA* image, int width, int height, int* output) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0;
	int y = idx;

	output[AT(x, y)] = image[AT(x, y)].Blue;

	for (x = 1; x < width; x++) {
		output[AT(x, y)] += output[AT(x - 1, y)] + image[AT(x, y)].Blue;
	}
}

__global__ void integralColumn(int* image, int width, int height, int* output) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int x = idx;
	int y = 0;

	output[AT(x, y)] = image[AT(x, y)];

	for (y = 1; y < height; y++) {
		output[AT(x, y)] += output[AT(x, y - 1)] + image[AT(x, y)];
	}
}

int* integral(int width, int height, RGBA* deviceSobelResult) {
	//--zeile
	int memSize = width * height * sizeof(int);
	int* dIntegralRow;
	cudaMalloc((void **)&dIntegralRow, memSize);

	int numBlocks = 1;
	int numThreadsPerBlock = height;
	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsPerBlock);

	integralRow << < dimGrid, dimBlock >> > (deviceSobelResult, width, height, dIntegralRow);

	//--spalte
	memSize = width * height * sizeof(int);
	int* dIntegralComplete;
	cudaMalloc((void **)&dIntegralComplete, memSize);

	numBlocks = 1;
	numThreadsPerBlock = width;
	dim3 dimGrid2(numBlocks);
	dim3 dimBlock2(numThreadsPerBlock);

	integralColumn << < dimGrid2, dimBlock2 >> > (dIntegralRow, width, height, dIntegralComplete);

	cudaFree(dIntegralRow);
	return dIntegralComplete;
}

__device__ void drawQuadrant(RGBA* imageSobel, int width, int height, Quadrant q) {
	for (int x = q.x0; x <= q.x1; x++) {
		imageSobel[AT(x, q.y0)].Red = 255;
		imageSobel[AT(x, q.y1)].Red = 255;
	}

	for (int y = q.y0; y <= q.y1; y++) {
		imageSobel[AT(q.x0, y)].Red = 255;
		imageSobel[AT(q.x1, y)].Red = 255;
	}
}

__global__ void quadtree(int* imageIntegral, RGBA* imageSobel, int width, int height, Quadrant* quadsIn, int n, int* prefixSum, Quadrant* quadrantenQutput) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	Quadrant q = quadsIn[idx];

	int change = imageIntegral[AT(q.x1, q.y1)]
		- imageIntegral[AT(q.x1, q.y0 - 1)]
		- imageIntegral[AT(q.x0 - 1, q.y1)]
		+ imageIntegral[AT(q.x0 - 1, q.y0 - 1)];

	prefixSum[idx] = (int)change > DELTA_PREFIX_CHANGE;

	for (int i = 1; i < n; i++) {
		prefixSum[i] += prefixSum[i - 1];
	}

	int posOut = (prefixSum[idx] - 1) * 4;

	if (change && q.x1 - q.x0 > MIN_QUAD_WIDTH) {
		quadrantenQutput[posOut].x0 = q.x0;
		quadrantenQutput[posOut].y0 = q.y0;
		quadrantenQutput[posOut].x1 = q.x0 + (q.x1 - q.x0) / 2;
		quadrantenQutput[posOut].y1 = q.y0 + (q.y1 - q.y0) / 2;
		drawQuadrant(imageSobel, width, height, quadrantenQutput[posOut]);

		quadrantenQutput[posOut + 1].x0 = q.x0 + (q.x1 - q.x0) / 2 + 1;
		quadrantenQutput[posOut + 1].y0 = q.y0;
		quadrantenQutput[posOut + 1].x1 = q.x1;
		quadrantenQutput[posOut + 1].y1 = q.y0 + (q.y1 - q.y0) / 2;
		drawQuadrant(imageSobel, width, height, quadrantenQutput[posOut + 1]);

		quadrantenQutput[posOut + 2].x0 = q.x0;
		quadrantenQutput[posOut + 2].y0 = q.y0 + (q.y1 - q.y0) / 2 + 1;
		quadrantenQutput[posOut + 2].x1 = q.x0 + (q.x1 - q.x0) / 2;
		quadrantenQutput[posOut + 2].y1 = q.y1;
		drawQuadrant(imageSobel, width, height, quadrantenQutput[posOut + 2]);

		quadrantenQutput[posOut + 3].x0 = q.x0 + (q.x1 - q.x0) / 2 + 1;
		quadrantenQutput[posOut + 3].y0 = q.y0 + (q.y1 - q.y0) / 2 + 1;
		quadrantenQutput[posOut + 3].x1 = q.x1;
		quadrantenQutput[posOut + 3].y1 = q.y1;
		drawQuadrant(imageSobel, width, height, quadrantenQutput[posOut + 3]);
	}
}

RGBA* startQuadTree(int bildBreite, int bildHoehe, int* deviceIntegralBild, RGBA* deviceSobelResult) {
	int anzahlQuads = 1;
	size_t memSize = anzahlQuads * sizeof(Quadrant);
	Quadrant* hostQuadrantenInput = (Quadrant*)malloc(sizeof(Quadrant));
	hostQuadrantenInput[0].x0 = 1;
	hostQuadrantenInput[0].y0 = 1;
	hostQuadrantenInput[0].x1 = bildBreite - 1;
	hostQuadrantenInput[0].y1 = bildHoehe - 1;

	Quadrant* deviceQuadrantenInput;
	cudaMalloc((void **)&deviceQuadrantenInput, memSize);
	cudaMemcpy(deviceQuadrantenInput, hostQuadrantenInput, memSize, cudaMemcpyHostToDevice);
	free(hostQuadrantenInput);

	while (anzahlQuads > 0) {
		memSize = anzahlQuads * sizeof(Quadrant);

		//Quadranten Output on Device
		Quadrant* deviceQuadrantenQutput;
		cudaMalloc((void **)&deviceQuadrantenQutput, 4 * memSize);

		//create solutions on device
		int* devicePrefixSum;
		cudaMalloc((void **)&devicePrefixSum, anzahlQuads * sizeof(int));

		dim3 dimGrid(1);
		dim3 dimBlock(anzahlQuads);
		quadtree <<< dimGrid, dimBlock >>> (deviceIntegralBild, deviceSobelResult, bildBreite, bildHoehe, deviceQuadrantenInput, anzahlQuads, devicePrefixSum, deviceQuadrantenQutput);

		//prefixsumToHost
		int* hostPrefixSum = (int*)malloc(anzahlQuads * sizeof(int));
		cudaMemcpy(hostPrefixSum, devicePrefixSum, anzahlQuads * sizeof(int), cudaMemcpyDeviceToHost);

		//init for next round
		cudaFree(deviceQuadrantenInput);
		deviceQuadrantenInput = deviceQuadrantenQutput;
		anzahlQuads = 4 * hostPrefixSum[anzahlQuads - 1];

		free(hostPrefixSum);
		cudaFree(devicePrefixSum);
	}

	cudaFree(deviceQuadrantenInput);

	return deviceSobelResult;
}


int main(){
	const char filename[64] = "q256_20.bmp";
	const char output[64] = "result.bmp";
	CBitmap *bitmap = new CBitmap(filename);
	RGBA *rgba = (RGBA*)bitmap->GetBits();

	thrust::host_vector<RGBA> hostRgba(rgba, rgba + bitmap->GetWidth() * bitmap->GetHeight());
	thrust::device_vector<RGBA> deviceGreyRgba = hostRgba;

	//Grauwert berechnen
	thrust::for_each(deviceGreyRgba.begin(), deviceGreyRgba.end(), grauwert());
	hostRgba = deviceGreyRgba;
	
	//Sobel berechnen
	unsigned int bildBreite = bitmap->GetWidth();
	unsigned int bildHoehe = bitmap->GetHeight();
	size_t memSize = bildBreite * bildHoehe * sizeof(RGBA);

	

	RGBA *hostRgbaBild = (RGBA *)malloc(memSize);
	hostRgbaBild = hostRgba.data();
	/*for (int i = 0; i < bildBreite * bildHoehe; ++i) {
		hostRgbaBild[i] = hostRgba[i];
	}*/

	RGBA *deviceRgbaBild;
	cudaMalloc((void **)&deviceRgbaBild, memSize);
	// Copy host array to device array
	cudaMemcpy(deviceRgbaBild, hostRgbaBild, memSize, cudaMemcpyHostToDevice);
	
	RGBA *deviceSobelResult;
	cudaMalloc((void **)&deviceSobelResult, memSize);

	dim3 dimGrid(bildHoehe);
	dim3 dimBlock(bildBreite);
	sobel<<<dimGrid, dimBlock>>>(deviceRgbaBild, deviceSobelResult);

	// block until the device has completed
	cudaThreadSynchronize();

	// Copy device array to host array
	cudaMemcpy(hostRgbaBild, deviceSobelResult, memSize, cudaMemcpyDeviceToHost);

	for( auto i = 0; i < bildBreite * bildHoehe; i++) {
		rgba[i] = hostRgbaBild[i];
	}

	//Integralbild erzeugen
	int memSizeIntegralBild = bildBreite * bildHoehe * sizeof(int);
	int *hostIntegralBild = (int *)malloc(memSizeIntegralBild);
	int *deviceIntegralBild;
	cudaMalloc((void **)&deviceIntegralBild, memSizeIntegralBild);
	// Copy host array to device array
	cudaMemcpy(deviceIntegralBild, hostIntegralBild, memSizeIntegralBild, cudaMemcpyHostToDevice);

	dim3 dimGridRow(1);
	dim3 dimBlockRow(bildBreite);
	prefixRow<<<dimGridRow, dimBlockRow >>>(deviceSobelResult, deviceIntegralBild, bildBreite);
	dim3 dimGridIntegralBild(1);
	dim3 dimBlockIntegralBild(bildHoehe);
	prefixColumn<<<dimGridIntegralBild, dimBlockIntegralBild >>>(deviceIntegralBild, bildBreite, bildHoehe);

	// Copy device array to host array
	cudaMemcpy(hostIntegralBild, deviceIntegralBild, memSizeIntegralBild, cudaMemcpyDeviceToHost);
	int prefixUntenRechts = hostIntegralBild[bildBreite * bildHoehe - 1];
	//std::cout << "Gesamtpräfix" << prefixUntenRechts << std::endl;

	//Quadtree
	RGBA* deviceBildQuadtree = startQuadTree(bildBreite, bildHoehe, deviceIntegralBild, deviceSobelResult);
	cudaMemcpy(rgba, deviceBildQuadtree, bildBreite * bildHoehe * sizeof(RGBA), cudaMemcpyDeviceToHost);

	bitmap->Save(output);
	delete rgba;
	cudaFree(deviceBildQuadtree);
	cudaFree(deviceSobelResult);
	cudaFree(deviceIntegralBild);

	return 0;
}
