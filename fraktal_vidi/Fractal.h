#ifndef _FRACTAL_H_
#define _FRACTAL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <windows.h>
#include <type_traits>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>

struct cuComplex {
	
	__device__ cuComplex( float a, float b ) : r(a), i(b) {}
	__device__ ~cuComplex() {}
	
	__device__ float magnitude( void ) {
		return r * r + i * i;
	}

	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}

	float r;
	float i;
};

struct Vec
{
	Vec() : x(0), y(0) {}
	Vec(int x_, int y_) : x(x_), y(y_) {}

	int x;
	int y;
};

class Fractal
{
public:
	// gets x, y (pos) and t(ime) parameters through CUDA
	__device__ virtual COLORREF operator()(int width, int height, int x, int y, int t) = 0;
};

// TODO rename 'draw' functions
template <typename FractalType>
__global__ void drawKernel(COLORREF* colormap)
{
	FractalType f;
	int index = gridDim.x * gridDim.y * threadIdx.x
		+ gridDim.x * blockIdx.y + blockIdx.x;
	colormap[index] = f(gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, threadIdx.x);
}

void drawOnConsole(HDC console, const COLORREF* colormap, Vec size)
{
	for (int y = 0; y < size.y; ++y)
	{
		for (int x = 0; x < size.x; ++x)
		{
			SetPixel(console, x, y, colormap[x + y * size.x]);
		}
	}
}

std::atomic_flag spinlock = ATOMIC_FLAG_INIT;

void drawPartOnConsole(HWND console, const COLORREF* colormap, Vec from, Vec size, int origwidth)
{
	HDC dc = GetDC(console); // different device handlers for different threads
	for (int y = 0; y < size.y; ++y)
	{
		for (int x = 0; x < size.x; ++x)
		{
			SetPixel(dc, from.x + x, from.y + y, colormap[(from.x + x) + (y + from.y) * origwidth]);
		}
	}
	ReleaseDC(console, dc);
}

void drawOnConsoleParallel(HWND console, const COLORREF* colormap, Vec size)
{
	// create drawer threads
	auto threadnum = std::thread::hardware_concurrency();
	std::vector<std::thread> threads(threadnum);
	
	size_t i = 0;
	size.y /= threadnum; // divide height by num of threads
	for (auto& t : threads)
	{
		Vec tmpfrom(0, i * size.y);
		t = std::thread(drawPartOnConsole, console, colormap, tmpfrom, size, size.x);
		++i;
	}

	// join
	for (auto& t : threads)
		t.join();
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

	// malloc on device
	COLORREF* dev_colormaps = nullptr;
	cudaMalloc((void**)&dev_colormaps, width * height * endtime * sizeof(COLORREF));

	// run kernel
	dim3 blocks(width, height);
	drawKernel<FractalType><<<blocks, endtime>>>(dev_colormaps);
	cudaDeviceSynchronize();

	// copy back colormaps to host
	auto colormaps = new COLORREF[width * height * endtime];
	cudaMemcpy(colormaps, dev_colormaps, width * height * endtime * sizeof(COLORREF), cudaMemcpyDeviceToHost);

	// draw on console
	Vec size(width, height);
	std::chrono::duration<int, std::ratio<1,1000>> time_between_frames ( (int) (1000.0 / FPS) );
	for (int frame = 0; frame < endtime; ++frame)
	{
		auto drawtime_start = std::chrono::steady_clock::now();
		drawOnConsoleParallel(console, colormaps + frame * width * height, size);
		auto drawtime_end = std::chrono::steady_clock::now();

		// sleep taking the drawing into account
		std::chrono::duration<int, std::ratio<1,1000>> sleep_time =
			time_between_frames - std::chrono::duration_cast<std::chrono::milliseconds>(drawtime_end - drawtime_start);
		if (sleep_time.count() > 0)
			std::this_thread::sleep_for( sleep_time );
	}

	// free all
	delete [] colormaps;
	cudaFree(dev_colormaps);
}

#endif