
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <windows.h>
#include "Fractal.h"
#include "CsikiFractal.h"

int main(int argc, char* argv[])
{
	try
	{
		if (argc == 5)
		{
			int fps = atoi(argv[1]);
			int frames = atoi(argv[2]);
			int width = atoi(argv[3]);
			int height = atoi(argv[4]);

			drawFractal<CsikiFractal<200>> (fps, frames, width, height);
		}
		else
			drawFractal<CsikiFractal<>> (24, 50, 200, 200);
	}
	catch (std::runtime_error& e)
	{
		std::cout << e.what() << std::endl;
	}

    return 0;
}
