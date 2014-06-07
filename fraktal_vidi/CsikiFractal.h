#ifndef _CSIKI_FRACTAL_H_
#define _CSIKI_FRACTAL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include "Fractal.h"

/* Julia fractal implementation.
*/
template<int iter = 100, int level = 10>
class CsikiFractal : public Fractal
{
public:
	__device__ CsikiFractal() {}
	__device__ ~CsikiFractal() {}

	// background color
	const static COLORREF bgcolor = RGB(1, 2, 0);
	// fractal color
	const static COLORREF frontcolor = RGB(120, 210, 80);

	/* Fractal color calculator for given coordinate and time.
	*/
	__device__ COLORREF operator()(int width, int height, int x, int y, int t)
	{
		const float scale = 0.001 + 5.0 / (float) t;
		float jx = scale * (float)(width  / 2.0 - x)  / (width  / 2.0)
			+ (float) t / (iter + 5); // moves fractal every dt
		float jy = scale * (float)(height / 2.0 - y)  / (height / 2.0)
			+ (float) t / (iter + 5); // moves fractal every dt

		cuComplex c(0.285, 0.01);
		//cuComplex c(-0.8, 0.158); // change this
		cuComplex a(jx, jy);

		for (int i = 0; i < iter; i++)
		{
			a = a * a + c;
			if (a.magnitude() > 1000)
				return bgcolor * (i * (iter / level));
		}

		return frontcolor;
	}
};

#endif