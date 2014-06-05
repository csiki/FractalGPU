#ifndef _CSIKI_FRACTAL_H_
#define _CSIKI_FRACTAL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include "Fractal.h"

class CsikiFractal : public Fractal
{
	__device__ COLORREF* draw(int x, int y, int t)
	{
		return RGB(0,0,0);
	}
};

#endif