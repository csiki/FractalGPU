#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <windows.h>
#include "Fractal.h"
#include "CsikiFractal.h"

/* Main entry point.
*/
int main(int argc, char* argv[])
{
	try
	{
		if (argc == 5)
		{
			int fps = atoi(argv[1]);
			int framenum = atoi(argv[2]);
			int width = atoi(argv[3]);
			int height = atoi(argv[4]);

			playFractalVideo<CsikiFractal<200>> (fps, framenum, width, height);
		}
		else
			playFractalVideo<CsikiFractal<100, 50>> (24, 50, 180, 180);
	}
	catch (std::runtime_error& e)
	{
		std::cout << e.what() << std::endl;
	}

    return 0;
}
