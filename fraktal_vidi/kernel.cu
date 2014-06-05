
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <windows.h>
#include "Fractal.h"
#include "CsikiFractal.h"

int main()
{
    drawFractal<CsikiFractal> (24, 20, 400, 500);
    return 0;
}
