
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <windows.h>
#include "Fractal.h"
#include "CsikiFractal.h"

int main(int argc, char* argv[])
{
	if (argc == 5)
	{
		int fps = atoi(argv[1]);
		int frames = atoi(argv[2]);
		int width = atoi(argv[3]);
		int height = atoi(argv[4]);

		drawFractal<CsikiFractal> (fps, frames, width, height);
	}
	else
		drawFractal<CsikiFractal> (24, 200, 400, 400);

    return 0;
}
