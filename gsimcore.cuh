#ifndef GSIMCORE_H
#define GSIMCORE_H
#include "gsimlib_header.cuh"
#include <curand_kernel.h>

//class delaration
class GAgent;
class Continuous2D;
class GScheduler;
class GModel;
class GRandom;

typedef struct iter_info_per_thread
{
	int2d_t cellCur;
	int2d_t cellUL;
	int2d_t cellDR;

	int ptr;
	int boarder;
	int count;
	float2d_t myLoc;
	int ptrInSmem;
	int id;

	float range;
} iterInfo;
extern __shared__ int smem[];

namespace util{
	__global__ void gen_hash_kernel(int *hash, Continuous2D *c2d);
	void sort_hash_kernel(int *hash, int *neighborIdx);
	__global__ void gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	void queryNeighbor(Continuous2D *c2d);
	void genNeighbor(Continuous2D *world, Continuous2D *world_h);

	void copyHostToDevice(void *hostPtr, void **devicePtr, size_t size);
};

class GAgent {
private:
	static unsigned int globalId;
protected:
	GAgentData_t *data;
	GAgentData_t *dataCopy;
	__device__ int initId();
public:
	__device__ void allocOnDevice();
	__device__ int getId() const;
	__device__ GAgentData_t *getData();
	__device__ float2d_t getLoc() const;
	__device__ void swapDataAndCopy();
	__device__ virtual void step(GModel *model) = 0;
	__device__ virtual ~GAgent() {}
	__device__ virtual void fillSharedMem(void *dataInSmem) = 0;
	int rank;
	int time;
};
unsigned int GAgent::globalId = 100;

class Continuous2D{
public:
	float width;
	float height;
	float discretization;
public:
	GAgent **allAgents;
	int *neighborIdx;
	int *cellIdxStart;
	int *cellIdxEnd;
public:
	__host__ Continuous2D(float w, float h, float disc){
		this->width = w;
		this->height = h;
		this->discretization = disc;
		size_t sizeAgArray = MAX_AGENT_NO*sizeof(int);
		size_t sizeCellArray = CELL_NO*sizeof(int);

		cudaMalloc((void**)&this->allAgents, MAX_AGENT_NO*sizeof(GAgent*));
		getLastCudaError("Continuous2D():cudaMalloc:allAgents");
		cudaMalloc((void**)&neighborIdx, sizeAgArray);
		getLastCudaError("Continuous2D():cudaMalloc:neighborIdx");
		cudaMalloc((void**)&cellIdxStart, sizeCellArray);
		getLastCudaError("Continuous2D():cudaMalloc:cellIdxStart");
		cudaMalloc((void**)&cellIdxEnd, sizeCellArray);
		getLastCudaError("Continuous2D():cudaMalloc:cellIdxEnd");
	}
	//GScheduler helper function
	__device__ const int* getNeighborIdx() const;
	//agent list manipulation
	__device__ GAgent* obtainAgentByInfoPtr(int ptr) const;
	//distance utility
	__device__ float stx(float x) const;
	__device__ float sty(float y) const;
	__device__ float tdx(float ax, float bx) const;
	__device__ float tdy(float ay, float by) const;
	__device__ float tds(float2d_t aloc, float2d_t bloc) const;
	//Neighbors related
	__device__ void nextNeighborInit2(float2d_t loc, float range, iterInfo &info) const;
	__device__ void resetNeighborInit(iterInfo &info) const;
	__device__ void calcPtrAndBoarder(iterInfo &info) const;
	template<class dataUnion> __device__ void putAgentDataIntoSharedMem(const iterInfo &info, dataUnion *elem, int tid, int lane) const;
	template<class dataUnion> __device__ dataUnion* nextAgentDataFromSharedMem(iterInfo &info) const;
	__device__ GAgentData_t *nextAgentData(iterInfo &info) const;
	//__global__ functions
	friend __global__ void util::gen_hash_kernel(int *hash, Continuous2D *c2d);
	friend __global__ void util::gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	friend void util::genNeighbor(Continuous2D *world, Continuous2D *world_h);
	friend void util::queryNeighbor(Continuous2D *c2d);

	//friend class GModel;
};
class GScheduler{
public:
	GAgent **allAgents;
	float time;
	int steps;
public:
	__device__ bool ScheduleOnce(const float time, const int rank,
		GAgent *ag);
	__device__ bool scheduleRepeating(const float time, const int rank, 
		GAgent *ag, const float interval);
	__device__ bool add(GAgent* ag, int idx);
	__device__ bool remove(GAgent *ag);
	__host__ GScheduler(){
		cudaMalloc((void**)&this->allAgents, MAX_AGENT_NO*sizeof(GAgent*));
		cudaMalloc((void**)&time, sizeof(int));
		cudaMalloc((void**)&steps, sizeof(int));
		getLastCudaError("Scheduler::allocOnDevice:cudaMalloc");
	}
};
class GModel{
public:
	GScheduler *scheduler, *schedulerHost;
	Continuous2D *world, *worldHost;
public:
	__device__ void addToScheduler(GAgent *ag, int idx);
	__device__ void foo();
	__host__ GModel(){
		world = NULL;
		worldHost = NULL;
		schedulerHost = new GScheduler();
		util::copyHostToDevice(schedulerHost, (void**)&scheduler, sizeof(GScheduler));
		getLastCudaError("GModel()");
	}
};
class GRandom {
	curandState *rState;
public:
	__device__ GRandom(int seed, int agId) {
		rState = new curandState();
		curand_init(seed, agId, 0, this->rState);
	}

	__device__ float uniform(){
		return curand_uniform(this->rState);
	}
	__device__ float gaussian(){
		return curand_normal(this->rState);
	}
};

//GAgent
__device__ int GAgent::initId() {
	return atomicInc(&GLOBAL_ID, UINT_MAX);
}
__device__ GAgentData_t *GAgent::getData(){
	return this->data;
}
__device__ float2d_t GAgent::getLoc() const{
	return this->data->loc;
}
__device__ void GAgent::swapDataAndCopy() {
	GAgentData_t *temp = this->data;
	this->data = this->dataCopy;
	this->dataCopy = temp;
}

//Continuous2D
__device__ const int* Continuous2D::getNeighborIdx() const{
	return this->neighborIdx;
}
__device__ GAgent* Continuous2D::obtainAgentByInfoPtr(int ptr) const {
	GAgent *ag = NULL;
	if (ptr < AGENT_NO_D && ptr >= 0){
		int agIdx = this->neighborIdx[ptr];
		if (agIdx < AGENT_NO_D && agIdx >=0)
			ag = this->allAgents[agIdx];
		else 
			printf("Continuous2D::obtainAgentByInfoPtr:ptr:%d\n", ptr);
	} 
	return ag;
}
__device__ float Continuous2D::stx(const float x) const{
	float res = x;
	if (x >= 0) {
		if (x >= this->width)
			res = x - this->width;
	} else
		res = x + this->width;
	if (res == this->width)
		res = 0;
	return res;
}
__device__ float Continuous2D::sty(const float y) const {
	float res = y;
	if (y >= 0) {
		if (y >= this->height)
			res = y - this->height;
	} else
		res = y + this->height;
	if (res == this->height)
		res = 0;
	return res;

}
__device__ float Continuous2D::tdx(float ax, float bx) const {
	float dx = abs(ax-bx);
	if (dx < WIDTH_D/2)
		return dx;
	else
		return WIDTH_D-dx;
}
__device__ float Continuous2D::tdy(float ay, float by) const {
	float dy = abs(ay-by);
	if (dy < HEIGHT_D/2)
		return dy;
	else
		return HEIGHT_D-dy;
}
__device__ float Continuous2D::tds(const float2d_t loc1, const float2d_t loc2) const {
	float dx = loc1.x - loc2.x;
	float dxsq = dx*dx;
	float dy = loc1.y - loc2.y;
	float dysq = dy*dy;
	float x = dxsq+dysq;
	return sqrt(x);
}
__device__ void Continuous2D::nextNeighborInit2(float2d_t agLoc, float range, iterInfo &info) const {
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	info.myLoc = agLoc;
	info.ptr = -1;
	info.boarder = -1;
	info.count = 0;
	info.range = range;
	info.ptrInSmem = 0;

	if ((agLoc.x-range)>0)	
		info.cellUL.x = (int)((agLoc.x-range)/CLEN_X);	
	else
		info.cellUL.x = 0;
	if ((agLoc.x+range)<WIDTH_D)	
		info.cellDR.x = (int)((agLoc.x+range)/CLEN_X);
	else	
		info.cellDR.x = (int)WIDTH_D/CLEN_X - 1;
	if ((agLoc.y-range)>0)	
		info.cellUL.y = (int)((agLoc.y-range)/CLEN_Y);
	else	
		info.cellUL.y = 0;
	if ((agLoc.y+range)<HEIGHT_D)	
		info.cellDR.y = (int)((agLoc.y+range)/CLEN_Y);
	else	
		info.cellDR.y = (int)HEIGHT_D/CLEN_Y - 1;

	int *cellulx = (int*)smem;
	int *celluly = (int*)&(cellulx[blockDim.x]);
	int *celldrx = (int*)&(celluly[blockDim.x]);
	int *celldry = (int*)&(celldrx[blockDim.x]);

	cellulx[tid]=info.cellUL.x;
	celluly[tid]=info.cellUL.y;
	celldrx[tid]=info.cellDR.x;
	celldry[tid]=info.cellDR.y;

	const unsigned int lane = tid&31;
	int lastFullWarp = AGENT_NO_D / warpSize;
	int totalFullWarpThreads = lastFullWarp * warpSize;
	int temp = 32;
	if (idx >= totalFullWarpThreads)
		temp = AGENT_NO_D - totalFullWarpThreads;

	for (int i=0; i<temp; i++){
#ifdef BOID_DEBUG
		;if (celluly[tid-lane+i] < 0) printf("zhongjian: y: %d, tid-lane+i: %d\n", celluly[tid-lane+i], tid-lane+i);
#endif
		info.cellUL.x = min(info.cellUL.x, cellulx[tid-lane+i]);
		info.cellUL.y = min(info.cellUL.y, celluly[tid-lane+i]);
		info.cellDR.x = max(info.cellDR.x, celldrx[tid-lane+i]);
		info.cellDR.y = max(info.cellDR.y, celldry[tid-lane+i]);
	}

	info.cellCur.x = info.cellUL.x;
	info.cellCur.y = info.cellUL.y;

#ifdef BOID_DEBUG
	if (info.cellCur.x < 0 || info.cellCur.y < 0) {
		printf("xiamian[agId :%d, loc.x: %f, loc.y: %f][xiamian: x: %d, y: %d]\n", agId, agLoc.x, agLoc.y,info.cellUL.x, info.cellUL.y);
	}
#endif

	this->calcPtrAndBoarder(info);
}
__device__ void Continuous2D::resetNeighborInit(iterInfo &info) const{
	info.ptr = -1;
	info.boarder = -1;
	info.count = 0;
	info.ptrInSmem = 0;
	info.cellCur.x = info.cellUL.x;
	info.cellCur.y = info.cellUL.y;
	this->calcPtrAndBoarder(info);
}
__device__ void Continuous2D::calcPtrAndBoarder(iterInfo &info) const {
	int hash = info.cellCur.zcode();
	if(hash < CELL_NO_D && hash>=0){
		info.ptr = this->cellIdxStart[hash];
		info.boarder = this->cellIdxEnd[hash];
	}
#ifdef DEBUG
	else {
		printf("x: %d, y: %d, hash: %d\n", info.cellCur.x, info.cellCur.y, hash);
	}
#endif
}
template<class dataUnion> __device__ void Continuous2D::putAgentDataIntoSharedMem(const iterInfo &info, dataUnion *elem, int tid, int lane) const{
	int agPtr = info.ptr + lane;
	if (agPtr <= info.boarder && agPtr >=0) {
		GAgent *ag = this->obtainAgentByInfoPtr(agPtr);
		ag->fillSharedMem(elem);
	} else
		elem->loc.x = -1;
#ifdef DEBUG
	if (agPtr < -1 || agPtr > AGENT_NO_D + 32){
		printf("Continuous2D::putAgentDataIntoSharedMem: ptr is %d, info.ptr is %d, lane is %d\n", agPtr, info.ptr, lane);
	}
#endif
}
template<class dataUnion> __device__ dataUnion *Continuous2D::nextAgentDataFromSharedMem(iterInfo &info) const {
	dataUnion *unionArray = (dataUnion*)&smem[4*blockDim.x];
	const int tid = threadIdx.x;
	const int lane = tid & 31;

	if (info.ptr>info.boarder) {
		info.ptrInSmem = 0;
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
	}

	if (info.ptrInSmem == 32)
		info.ptrInSmem = 0;

	if (info.ptrInSmem == 0) {
		dataUnion *elem = &unionArray[tid];
		this->putAgentDataIntoSharedMem(info, elem, tid, lane);
	}

	dataUnion *elem = &unionArray[tid - lane + info.ptrInSmem];
	info.ptrInSmem++;
	info.ptr++;

	while (elem->loc.x == -1 && info.cellCur.y <= info.cellDR.y) {
		info.ptrInSmem = 0;
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
		this->putAgentDataIntoSharedMem(info, &unionArray[tid], tid, lane);
		elem = &unionArray[tid - lane + info.ptrInSmem];
		info.ptrInSmem++;
		info.ptr++;
	}

	if (elem->loc.x == -1) {
		elem = NULL;
	}
	return elem;
}
__device__ GAgentData_t *Continuous2D::nextAgentData(iterInfo &info) const {

	if (info.ptr>info.boarder) {
		info.ptrInSmem = 0;
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
	}

	while (info.ptr == -1) {
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
	}

	GAgent *ag = this->obtainAgentByInfoPtr(info.ptr);
	info.ptr++;
	return ag->getData();
}

//GScheduler
__device__ bool GScheduler::ScheduleOnce(const float time, 	const int rank, GAgent *ag){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){
		ag->time = time;
		ag->rank = rank;
		allAgents[idx] = ag;
	}
	return true;
}
__device__ bool GScheduler::scheduleRepeating(const float time, const int rank, GAgent *ag, const float interval){

	return true;
}
__device__ bool GScheduler::add(GAgent *ag, int idx){
	if(idx>=MAX_AGENT_NO_D)
		return false;
	this->allAgents[idx] = ag;
	return true;
}

//GModel
__device__ void GModel::addToScheduler(GAgent *ag, int idx){
	this->scheduler->add(ag, idx);
}

//namespace continuous2D Utility
__device__ int zcode(int x, int y){
	y &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x &= 0x0000ffff;                 // x = ---- ---- ---- ---- fedc ba98 7654 3210
	y = (y ^ (y << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	y = (y ^ (y << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	y = (y ^ (y << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	y = (y ^ (y << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x | (y << 1);
}
__global__ void util::gen_hash_kernel(int *hash, Continuous2D *c2d)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D) {
		GAgent *ag = c2d->allAgents[idx];
		float2d_t myLoc = ag->getLoc();
		int xhash = (int)(myLoc.x/CLEN_X);
		int yhash = (int)(myLoc.y/CLEN_Y);
		hash[idx] = zcode(xhash, yhash);
		c2d->neighborIdx[idx] = idx;
	}
	//printf("id: %d, hash: %d, neiIdx: %d\n", idx, hash[idx], c2d->neighborIdx[idx]);
}
__global__ void util::gen_cellIdx_kernel(int *hash, Continuous2D *c2d)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D && idx > 0) {
		if (hash[idx] != hash[idx-1]) {
			c2d->cellIdxStart[hash[idx]] = idx;
			c2d->cellIdxEnd[hash[idx-1]] = idx-1;
		}
	}
	if (idx == 0) {
		c2d->cellIdxStart[hash[0]] = idx;
		c2d->cellIdxEnd[hash[AGENT_NO_D-1]] = AGENT_NO_D-1;
	}
}
void util::sort_hash_kernel(int *hash, int *neighborIdx)
{
	thrust::device_ptr<int> id_ptr(neighborIdx);
	thrust::device_ptr<int> hash_ptr(hash);
	typedef thrust::device_vector<int>::iterator Iter;
	Iter key_begin(hash_ptr);
	Iter key_end(hash_ptr + AGENT_NO);
	Iter val_begin(id_ptr);
	thrust::sort_by_key(key_begin, key_end, val_begin);
	getLastCudaError("sort_hash_kernel");
}
void util::genNeighbor(Continuous2D *world, Continuous2D *world_h)

{
	static int iterCount = 0;
	int bSize = BLOCK_SIZE;
	int gSize = GRID_SIZE(AGENT_NO);

	int *hash;
	cudaMalloc((void**)&hash, AGENT_NO*sizeof(int));
	cudaMemset(world_h->cellIdxStart, 0xff, CELL_NO*sizeof(int));
	cudaMemset(world_h->cellIdxEnd, 0xff, CELL_NO*sizeof(int));

	gen_hash_kernel<<<gSize, bSize>>>(hash, world);
	sort_hash_kernel(hash, world_h->neighborIdx);
	gen_cellIdx_kernel<<<gSize, bSize>>>(hash, world);

	//debug
	if (iterCount == SELECTION && FILE_GEN == 1){
		int *id_h, *hash_h, *cidx_h;
		id_h = new int[AGENT_NO];
		hash_h = new int[AGENT_NO];
		cidx_h = new int[CELL_NO];
		cudaMemcpy(id_h, world_h->neighborIdx, AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
		getLastCudaError("genNeighbor:cudaMemcpy(id_h");
		cudaMemcpy(hash_h, hash, AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
		getLastCudaError("genNeighbor:cudaMemcpy(hash_h");
		cudaMemcpy(cidx_h, world_h->cellIdxStart, CELL_NO * sizeof(int), cudaMemcpyDeviceToHost);
		getLastCudaError("genNeighbor:cudaMemcpy(cidx_h");
		std::fstream fout;
		char *outfname = new char[30];
		sprintf(outfname, "out_genNeighbor_%d_neighborIdx.txt", iterCount);
		fout.open(outfname, std::ios::out);
		for (int i = 0; i < AGENT_NO; i++){
			fout << id_h[i] << " " << hash_h[i] <<std::endl;
			fout.flush();
		}
		fout.close();
		sprintf(outfname, "out_genNeighbor_%d_cellIdx.txt", iterCount);
		fout.open(outfname, std::ios::out);
		for (int i = 0; i < CELL_NO; i++){
			fout << cidx_h[i] <<std::endl;
			fout.flush();
		}
		fout.close();
	}
	//~debug

	iterCount++;
	cudaFree(hash);
	getLastCudaError("genNeighbor:cudaFree:hash");
}

void util::copyHostToDevice(void *hostPtr, void **devPtr, size_t size){
	cudaMalloc(devPtr, size);
	cudaMemcpy(*devPtr, hostPtr, size, cudaMemcpyHostToDevice);
	getLastCudaError("copyHostToDevice");
}

__global__ void step(GModel *gm){
	const GScheduler *sch = gm->scheduler;
	const Continuous2D *world = gm->world;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D) {
		GAgent *ag;
		if (world != NULL)
			ag = sch->allAgents[world->neighborIdx[idx]];
		else
			ag = sch->allAgents[idx];
		ag->step(gm);
		ag->swapDataAndCopy();
	}
}

#endif