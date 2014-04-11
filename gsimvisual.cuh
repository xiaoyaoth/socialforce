#ifndef GSIMVISUAL_H
#define GSIMVISUAL_H

#include "gsimlib_header.cuh"
#include "common\book.h"
#include "common\gl_helper.h"
#include "cuda_gl_interop.h"
#include "gsimcore.cuh"

class GSimVisual;

namespace visUtil{
	__global__ void paint(uchar4 *devPtr, const Continuous2D *world, int width, int height, int scale);
};

class GSimVisual{
public:
	GLuint bufferObj;
	cudaGraphicsResource *resource;
	Continuous2D *world;
	int width;
	int height;
	int scale;

	PFNGLBINDBUFFERARBPROC    glBindBuffer;
	PFNGLDELETEBUFFERSARBPROC glDeleteBuffers;
	PFNGLGENBUFFERSARBPROC    glGenBuffers;
	PFNGLBUFFERDATAARBPROC    glBufferData;

	GSimVisual(){
		if (VISUALIZE == true) {
			this->width = 256;
			this->height = 256;
			this->scale = 2;
			glBindBuffer     = NULL;
			glDeleteBuffers  = NULL;
			glGenBuffers     = NULL;
			glBufferData     = NULL;

			int c = 1;
			char *dummy = " ";
			glutInit( &c, &dummy );
			glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
			glutInitWindowSize( this->width * this->scale, this->height * this->scale );
			glutCreateWindow( "bitmap" );

			glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
			glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
			glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
			glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");
			glGenBuffers( 1, &bufferObj );
			glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
			glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, scale * scale * width * height * sizeof(uchar4),
				NULL, GL_DYNAMIC_DRAW_ARB );
			cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
			getLastCudaError("cudaGraphicsGLRegisterBuffer");

			glutDisplayFunc(drawFunc);
			glutIdleFunc(idleFunc);
			glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
		}
	}

	static void idleFunc(){
		GSimVisual vis = GSimVisual::getInstance();
		uchar4* devPtr;
		size_t size;
		cudaGraphicsMapResources(1, &vis.resource, NULL);
		getLastCudaError("cudaGraphicsMapResources");
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, vis.resource);
		getLastCudaError("cudaGraphicsResourceGetMappedPointer");
		//paint kernel here...
		cudaGraphicsUnmapResources(1, &vis.resource, NULL);
		getLastCudaError("cudaGraphicsUnmapResources");

		glutPostRedisplay();
	}

	static void drawFunc(){
		glClearColor( 1.0, 1.0, 1.0, 1.0 );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		
		GSimVisual vis = GSimVisual::getInstance();

		uchar4* devPtr;
		size_t size;
		cudaGraphicsMapResources(1, &vis.resource, NULL);
		getLastCudaError("cudaGraphicsMapResources");
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, vis.resource);
		getLastCudaError("cudaGraphicsResourceGetMappedPointer");
		cudaMemset(devPtr, 0, size);
		getLastCudaError("cudaMemset");

		glEnable(GL_TEXTURE_2D);
		int gSize = GRID_SIZE(AGENT_NO);
		visUtil::paint<<<gSize, BLOCK_SIZE>>>(devPtr, vis.world, vis.width, vis.height, vis.scale);

		cudaGraphicsUnmapResources(1, &vis.resource, NULL);
		getLastCudaError("cudaGraphicsUnmapResources");

		glDrawPixels(vis.width * vis.scale, vis.height * vis.scale, GL_RGBA,GL_UNSIGNED_BYTE, 0 );


		glutSwapBuffers();
		glutPostRedisplay();
	}

public:
	static GSimVisual& getInstance(){
		static GSimVisual instance;
		return instance;
	}

	void setWorld(Continuous2D *world){
		if (VISUALIZE == true)
			GSimVisual::getInstance().world = world;
	}

	void animate(){
		if (VISUALIZE == true)
			glutMainLoopEvent();
	}

	void stop(){
		if (VISUALIZE == true)
			glutLeaveMainLoop();
	}
};

__global__ void visUtil::paint(uchar4 *devPtr, const Continuous2D *world, int width, int height, int scale)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){
		GAgent *ag = world->allAgents[idx];
		float2d_t myLoc = ag->getLoc();
		int canvasX = (int)(myLoc.x * width / world->width);
		int canvasY = (int)(myLoc.y * height / world->height);
		for (int i = 0; i < scale; i++)
			for (int j = 0; j < scale; j++) 
			{
				int canvasXNew = canvasX * scale + j;
				int canvasYNew = canvasY * scale + i;
				int canvasIdx = canvasYNew * width * scale + canvasXNew;
				devPtr[canvasIdx].x = 0;
				devPtr[canvasIdx].y = 255;
				devPtr[canvasIdx].z = 0;
				devPtr[canvasIdx].w = 255;
			}
	}
	if (blockIdx.x == 0) {
		int segLen = height / blockDim.x;
		int canvasX = (int)(25 * width / world->width);
		int canvasY = (int)(threadIdx.x * segLen);
		float door = (float)threadIdx.x / (float)blockDim.x * 100;
		bool doorFlag = door < 48. || door > 51.;
		int canvasXNew = canvasX * scale;
		int canvasYNew = canvasY * scale;
		for (int i = 0; i < 4; i++) {
			int canvasIdx = (canvasYNew +i) * width * scale + canvasXNew;
			if (doorFlag){
				devPtr[canvasIdx].x = 255;
				devPtr[canvasIdx].y = 255;
				devPtr[canvasIdx].z = 255;
				devPtr[canvasIdx].w = 255;
			}
		}
	}
}


#endif