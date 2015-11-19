#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <windows.h>
#include "Fractal.h"
#include "CsikiFractal.h"

/*#include "GL/freeglut.h"
static void RenderSceneCB()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_POINTS, 0, 1);

    glutSwapBuffers();
}


static void InitializeGlutCallbacks()
{
    glutDisplayFunc(RenderSceneCB);
}
*/

/* Main entry point.
*/
int main(int argc, char* argv[])
{
	// default values
	int fps = 24;
	int framenum = 50;
	int width = 180;
	int height = 180;

	// set values from arguments
	if (argc == 5)
	{
		fps = atoi(argv[1]);
		framenum = atoi(argv[2]);
		width = atoi(argv[3]);
		height = atoi(argv[4]);

		playFractalVideo<CsikiFractal<200>> (fps, framenum, width, height);
	}

	// draw fractal animation
	/*glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
	glutInitWindowSize(800, 600);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Tutorial 02");

	InitializeGlutCallbacks();
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glutMainLoop();*/
	try
	{
		playFractalVideo<CsikiFractal<200>> (fps, framenum, width, height);
	}
	catch (std::runtime_error& e)
	{
		std::cout << e.what() << std::endl;
	}

    return 0;
}
