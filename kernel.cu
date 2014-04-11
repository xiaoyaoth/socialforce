#include "socialForce.cuh"
#include "socialForceHeader.cuh"
#include "gsimcore.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iterator>
#include <iomanip>
#ifdef _WIN32
#include <Windows.h>
#include "gsimvisual.cuh"
#else
#include <sys/time.h>
#endif
#include "cuda.h"

void initOnDevice(float *x_pos, float *y_pos){
	float *x_pos_h, *y_pos_h;
	x_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
	y_pos_h = (float*)malloc(AGENT_NO*sizeof(float));

	std::ifstream fin(dataFileName);
	std::string rec;

	char *cstr, *p;
	int i = 0;
	cstr = (char *)malloc(20 * sizeof(char));
	while (!fin.eof() && i<AGENT_NO) {
		std::getline(fin, rec);
		std::strcpy(cstr, rec.c_str());
		if(strcmp(cstr,"")==0)
			break;
		p=strtok(cstr, " ");
		x_pos_h[i] = atof(p);
		p=strtok(NULL, " ");
		y_pos_h[i] = atof(p);
		i++;
	}
	size_t floatDataSize = AGENT_NO*sizeof(float);
	cudaMemcpy(x_pos, x_pos_h, floatDataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(y_pos, y_pos_h, floatDataSize, cudaMemcpyHostToDevice);
	getLastCudaError("initOnDevice");
}

__global__ void addAgentsOnDevice(SocialForceModel *sfModel){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){ // user init step
		//Add agent here
		SocialForceAgent *ag = new SocialForceAgent(idx, sfModel);
		sfModel->agentPool->ptrArray[idx] = ag;
		sfModel->agentPool->delMark[idx] = false;
	}
	if (idx == 0) {
		//set assignment
	}
}

void readConfig(char *config_file){
	std::ifstream fin;
	fin.open(config_file);
	std::string rec;
	char *cstr, *p;
	cstr = (char *)malloc(100 * sizeof(char));

	while (!fin.eof()) {
		std::getline(fin, rec);
		std::strcpy(cstr, rec.c_str());
		if(strcmp(cstr,"")==0)
			break;
		p=strtok(cstr, "=");
		if(strcmp(p, "AGENT_NO")==0){
			p=strtok(NULL, "=");
			AGENT_NO = atoi(p);
			cudaMemcpyToSymbol(AGENT_NO_D, &AGENT_NO, sizeof(int), 0, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(GLOBAL_ID, &AGENT_NO, sizeof(int), 0, cudaMemcpyHostToDevice);
			getLastCudaError("readConfig");
		}
		if(strcmp(p, "MAX_AGENT_NO")==0){
			p=strtok(NULL, "=");
			MAX_AGENT_NO = atoi(p);
			cudaMemcpyToSymbol(MAX_AGENT_NO_D, &MAX_AGENT_NO, sizeof(int), 0, cudaMemcpyHostToDevice);
			getLastCudaError("readConfig");
		}
		if(strcmp(p, "BOARDER_L")==0){
			p=strtok(NULL, "=");
			BOARDER_L_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_L_D, &BOARDER_L_H, sizeof(int));
			getLastCudaError("readConfig");
		}
		if(strcmp(p, "BOARDER_R")==0){
			p=strtok(NULL, "=");
			BOARDER_R_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_R_D, &BOARDER_R_H, sizeof(int));
			getLastCudaError("readConfig");
		}
		if(strcmp(p, "BOARDER_U")==0){
			p=strtok(NULL, "=");
			BOARDER_U_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_U_D, &BOARDER_U_H, sizeof(int));
			getLastCudaError("readConfig");
		}
		if(strcmp(p, "BOARDER_D")==0){
			p=strtok(NULL, "=");
			BOARDER_D_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_D_D, &BOARDER_D_H, sizeof(int));
			getLastCudaError("readConfig");
		}
		if(strcmp(p, "RANGE")==0){
			p=strtok(NULL, "=");
			RANGE_H = atof(p);
			cudaMemcpyToSymbol(RANGE, &RANGE_H, sizeof(float));
			getLastCudaError("readConfig");
		}
		if(strcmp(p, "DISCRETI")==0){
			p=strtok(NULL, "=");
			DISCRETI = atoi(p);
		}
		if(strcmp(p, "STEPS")==0){
			p=strtok(NULL, "=");
			STEPS = atoi(p);
		}
		if(strcmp(p, "VERBOSE")==0){
			p=strtok(NULL, "=");
			VERBOSE = atoi(p);
		}
		if(strcmp(p, "SELECTION")==0){
			p=strtok(NULL, "=");
			SELECTION = atoi(p);
		}
		if(strcmp(p, "VISUALIZE")==0){
			p=strtok(NULL, "=");
			VISUALIZE = atoi(p);
		}
		if(strcmp(p, "FILE_GEN")==0){
			p=strtok(NULL, "=");
			FILE_GEN = atoi(p);
		}
		if(strcmp(p, "BLOCK_SIZE")==0){
			p=strtok(NULL, "=");
			BLOCK_SIZE = atoi(p);
		}
		if(strcmp(p, "HEAP_SIZE")==0){
			p=strtok(NULL, "=");
			HEAP_SIZE = atoi(p);
		}
		if(strcmp(p, "STACK_SIZE")==0){
			p=strtok(NULL, "=");
			STACK_SIZE = atoi(p);
		}
		if(strcmp(p, "DATA_FILENAME")==0){
			dataFileName = new char[20];
			p=strtok(NULL, "=");
			strcpy(dataFileName, p);
		}
	}
	free(cstr);
	fin.close();

	int CNO_PER_DIM_H = (int)pow((float)2, DISCRETI);
	cudaMemcpyToSymbol(CNO_PER_DIM, &CNO_PER_DIM_H, sizeof(int));
	getLastCudaError("readConfig");
	
	CELL_NO = CNO_PER_DIM_H * CNO_PER_DIM_H;
	cudaMemcpyToSymbol(CELL_NO_D, &CELL_NO, sizeof(int));
	getLastCudaError("readConfig");
	
	float CLEN_X_H = (float)(BOARDER_R_H-BOARDER_L_H)/CNO_PER_DIM_H;
	float CLEN_Y_H = (float)(BOARDER_D_H-BOARDER_U_H)/CNO_PER_DIM_H;
	cudaMemcpyToSymbol(CLEN_X, &CLEN_X_H, sizeof(int));
	getLastCudaError("readConfig");
	cudaMemcpyToSymbol(CLEN_Y, &CLEN_Y_H, sizeof(int));
	getLastCudaError("readConfig");

	//GRID_SIZE = AGENT_NO%BLOCK_SIZE==0 ? AGENT_NO/BLOCK_SIZE : AGENT_NO/BLOCK_SIZE + 1;
}

void oneStep(SocialForceModel *model, SocialForceModel *model_h){
	AGENT_NO = model_h->agentPoolHost->numElem;
	SocialForceAgent **poolAgentList = model_h->agentPoolHost->ptrArray;
	GAgent **schAgentList = model_h->schedulerHost->allAgents;
	cudaMemcpy(schAgentList, poolAgentList, AGENT_NO * sizeof(GAgent*), cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(AGENT_NO_D, &AGENT_NO, sizeof(int), 0, cudaMemcpyHostToDevice);
	if (model_h->worldHost != NULL) {
		GAgent **worldAgentList = model_h->worldHost->allAgents;
		cudaMemcpy(worldAgentList, poolAgentList, AGENT_NO * sizeof(GAgent*), cudaMemcpyDeviceToDevice);
	}

	GSimVisual::getInstance().animate();

	int gSize = GRID_SIZE(AGENT_NO);
	size_t sizeOfSmem = BLOCK_SIZE * (
		4*sizeof(int)
		+ sizeof(dataUnion)
		);

	getLastCudaError("before loop");
	util::genNeighbor(model_h->world, model_h->worldHost);
	getLastCudaError("end genNeighbor");
	step<<<gSize, BLOCK_SIZE, sizeOfSmem>>>(model);
	getLastCudaError("end step");	

	int scrGSize = GRID_SIZE(MAX_AGENT_NO);
	poolUtil::cleanup(model_h->agentPoolHost, model_h->agentPool);
}

void mainWork(char *config_file){
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	getLastCudaError("setting cache preference");
	readConfig(config_file);

	size_t pVal;
	cudaDeviceGetLimit(&pVal, cudaLimitMallocHeapSize);
	printf("cudaLimitMallocHeapSize: %d\n", pVal);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE);
	cudaDeviceGetLimit(&pVal, cudaLimitMallocHeapSize);
	printf("cudaLimitMallocHeapSize: %d\n", pVal);

	SocialForceModel *model = NULL;
	SocialForceModel *model_h = new SocialForceModel(BOARDER_D_H, BOARDER_D_H);
	util::copyHostToDevice(model_h, (void**)&model, sizeof(SocialForceModel));

	int gSize = GRID_SIZE(AGENT_NO);
	addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>(model);
	getLastCudaError("before going into the big loop");
	printf("steps: %d\n", STEPS);
#ifdef _WIN32
	GSimVisual::getInstance().setWorld(model_h->world);
	for (int i=0; i<STEPS; i++){
		//if ((i%(STEPS/10))==0) 
		printf("STEP:%d ", i);
		oneStep(model, model_h);
	}
	printf("finally total agent is %d\n", AGENT_NO);
	GSimVisual::getInstance().stop();
#else
	for (int i=0; i<STEPS; i++){
	 	if ((i%(STEPS/10))==0) printf("STEP:%d ", i);
		oneStep(model, model_h);
		writeRandDebug(i, devRandDebug);
	}
#endif
	getLastCudaError("finished");
}

int main(int argc, char *argv[]){
#ifndef _WIN32
	struct timeval start, end;
	gettimeofday(&start, NULL);
	mainWork(argv[1]);
	gettimeofday(&end, NULL);
	printf("%ld\n", ((end.tv_sec * 1000000 + end.tv_usec)
		  - (start.tv_sec * 1000000 + start.tv_usec)));
#else
	int start = GetTickCount();
	mainWork(argv[1]);
	int end = GetTickCount();
	int diff = end-start;
	std::cout<<"Took "<<diff<<" ms"<<std::endl;
	system("PAUSE");
#endif
}
