#ifndef _FRACTAL_H_
#define _FRACTAL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <windows.h>
#include <type_traits>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

class Fractal
{
	// gets x, y (pos) and t(ime) parameters through CUDA
	__device__ virtual COLORREF* draw(int x, int y, int t) = 0;
};


template <typename FractalType>
__global__ void drawKernel(COLORREF* colormap)
{
	FractalType f; // TODO
	int index = gridDim.x * gridDim.y * threadIdx.x
		+ gridDim.x * blockIdx.y + blockIdx.x;
	colormap[index] = RGB(blockIdx.y * threadIdx.x % 256, blockIdx.x * threadIdx.x % 256, threadIdx.x % 256);
}

void drawOnConsole(HDC console, const COLORREF* colormap, int width, int height)
{
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			SetPixel(console, x, y, colormap[x + y * width]);
		}
	}
}

void drawPartOnConsole(HDC console, int fromx, int fromy, int tox, int toy)
{
	// TODO
}

template <typename FractalType>
void drawFractal(double FPS, int endtime, int width, int height)
{
	// check for type safety
	if (!std::is_base_of<Fractal, FractalType>::value)
	{
		std::string msg = "Typename not derived from Fractal: ";
		msg += typeid(FractalType).name();
		throw std::runtime_error(msg);
	}

	// get console handler
	HWND console = GetConsoleWindow();
	HDC dc = GetDC(console);

	// malloc on device
	COLORREF* dev_colormaps = nullptr;
	cudaMalloc((void**)&dev_colormaps, width * height * endtime * sizeof(COLORREF));
	// FIXME mallocPitch http://stackoverflow.com/questions/5029920/how-to-use-2d-arrays-in-cuda

	// run kernel
	dim3 blocks(width, height);
	drawKernel<FractalType><<<blocks, endtime>>>(dev_colormaps);
	cudaDeviceSynchronize();

	// copy back colormaps to host
	auto colormaps = new COLORREF[width * height * endtime];
	cudaMemcpy(colormaps, dev_colormaps, width * height * endtime * sizeof(COLORREF), cudaMemcpyDeviceToHost);
	
	// create drawer threads
	std::vector<std::thread> threads(std::thread::hardware_concurrency());
	// TODO start threads with different parts of 'image'

	// draw on console
	std::chrono::duration<int, std::ratio<1,1000>> time_between_frames ( (int) (1000.0 / FPS));
	for (int frame = 0; frame < endtime; ++frame)
	{
		drawOnConsole(dc, colormaps + frame * width * height, width, height);
		std::this_thread::sleep_for( time_between_frames );
	}

	// free all
	delete [] colormaps;
	cudaFree(dev_colormaps);
}

#endif