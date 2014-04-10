#ifndef SOCIAL_FORCE_CUH
#define SOCIAL_FORCE_CUH

#include "gsimcore.cuh"
#include "gsimlib_header.cuh"
#include "socialForceHeader.cuh"

#define	tao 0.5
#define	A 2000
#define	B 0.1
#define	k1 (1.2 * 100000)
#define k2 (2.4 * 100000)
#define	maxv 3

class SocialForceModel;
class SocialForceAgent;

__constant__ struct obstacleLine obsLines[10];
__constant__ int obsLineNum;

class SocialForceModel : public GModel {
public:
	Continuous2D *world, *worldHost;
	Pool<SocialForceAgent> *agentPool, *agentPoolHost;

	__host__ SocialForceModel(int envHeight, int envWidth){
		GModel::allocOnDevice();

		worldHost = new Continuous2D(envHeight, envWidth, 0);
		worldHost->allocOnDevice();
		util::copyHostToDevice(worldHost, (void**)&world, sizeof(Continuous2D));

		agentPoolHost = new Pool<SocialForceAgent>(AGENT_NO, MAX_AGENT_NO);
		util::copyHostToDevice(agentPoolHost, (void**)&agentPool, sizeof(Pool<SocialForceAgent>));

		int obsLineNumHost = 2;
		size_t obsLinesSize = sizeof(struct obstacleLine) * obsLineNumHost;
		struct obstacleLine *obsLinesHost = (struct obstacleLine *)malloc(obsLinesSize);
		obsLinesHost[0].init(20, -20, 25, 48);
		obsLinesHost[1].init(25, 51, 20,120);
		//obsLinesHost[2].init(50, 45, 60, 45);
		//obsLinesHost[3].init(60, 45, 60, 55);
		//obsLinesHost[4].init(60, 55, 50, 55);
		//obsLinesHost[5].init(50, 55, 50, 45);

		cudaMemcpyToSymbol(obsLines, obsLinesHost, obsLinesSize, 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(obsLineNum, &obsLineNumHost, sizeof(int), 0, cudaMemcpyHostToDevice);
		getLastCudaError("cudaMemcpyToSymbol:obsLines");
	}
};

__device__ float correctCrossBoader(float val, float limit)
{
	if (val > limit)
		return limit-1;
	else if (val < 0)
		return 0;
	return val;
}

class SocialForceAgent : public GAgent {
public:
	GRandom *random;
	SocialForceModel *model;

	__device__ SocialForceAgent(int id, SocialForceModel *model) {
		this->model = model;
		this->random = new GRandom(2345, id);
		SocialForceAgentData_t *data = new SocialForceAgentData_t();
		SocialForceAgentData_t *dataCopy = new SocialForceAgentData_t();
		data->goal.x = 25;
		data->goal.y = 50;
		data->loc.x = data->goal.x + (model->world->width - data->goal.x) * this->random->uniform();
		data->loc.y = (model->world->height) * this->random->uniform();
		data->velocity.x = 4 * (this->random->uniform()-0.5);
		data->velocity.y = 4 * (this->random->uniform()-0.5);
		data->v0 = 2;
		data->mass = 50;
		*dataCopy = *data;

		this->data = data;
		this->dataCopy = dataCopy;
	}

	__device__ void computeSocialForce(const SocialForceAgentData_t &myData, const dataUnion &otherData, float2d_t &fSum){
		float cMass = 100;
		//my data
		const float2d_t& loc = myData.loc;
		const float2d_t& goal = myData.goal;
		const float2d_t& velo = myData.velocity;
		const float& v0 = myData.v0;
		const float& mass = myData.mass;
		//other's data
		const float2d_t& locOther = otherData.loc;
		const float2d_t& goalOther = otherData.goal;
		const float2d_t& veloOther = otherData.velocity;
		const float& v0Other = otherData.v0;
		const float& massOther = otherData.mass;

		float d = 1e-15 + sqrt((loc.x - locOther.x) * (loc.x - locOther.x) + (loc.y - locOther.y) * (loc.y - locOther.y));
		float dDelta = mass / cMass + massOther / cMass - d;
		float fExp = A * exp(dDelta / B);
		float fKg = dDelta < 0 ? 0 : k1 *dDelta;
		float nijx = (loc.x - locOther.x) / d;
		float nijy = (loc.y - locOther.y) / d;
		float fnijx = (fExp + fKg) * nijx;
		float fnijy = (fExp + fKg) * nijy;
		float fkgx = 0;
		float fkgy = 0;
		if (dDelta > 0) {
			float tix = - nijy;
			float tiy = nijx;
			fkgx = k2 * dDelta;
			fkgy = k2 * dDelta;
			float vijDelta = (veloOther.x - velo.x) * tix + (veloOther.y - velo.y) * tiy;
			fkgx = fkgx * vijDelta * tix;
			fkgy = fkgy * vijDelta * tiy;
		}
		fSum.x += fnijx + fkgx;
		fSum.y += fnijy + fkgy;
	}

	__device__ void step(GModel *model){
		__syncthreads();
		SocialForceModel *sfModel = (SocialForceModel*)model;
		Continuous2D *world = sfModel->world;
		float width = world->width;
		float height = world->height;
		float cMass = 100;

		iterInfo info;
		
		SocialForceAgentData_t *dataLocalPtr = (SocialForceAgentData_t*)this->data;
		SocialForceAgentData_t dataLocal = *dataLocalPtr;

		const float2d_t& loc = dataLocal.loc;
		const float2d_t& goal = dataLocal.goal;
		const float2d_t& velo = dataLocal.velocity;
		const float& v0 = dataLocal.v0;
		const float& mass = dataLocal.mass;

		//compute the direction
		float2d_t dvt;	dvt.x = 0;	dvt.y = 0;
		float2d_t diff; diff.x = 0; diff.y = 0;
		float d0 = sqrt((loc.x - goal.x) * (loc.x - goal.x) + (loc.y - goal.y) * (loc.y - goal.y));
		diff.x = v0 * (goal.x - loc.x) / d0;
		diff.y = v0 * (goal.y - loc.y) / d0;
		dvt.x = (diff.x - velo.x) / tao;
		dvt.y = (diff.y - velo.y) / tao;
		
		//compute force with other agents
		float2d_t fSum; fSum.x = 0; fSum.y = 0;
		dataUnion *otherData, otherDataLocal;
		float ds = 0;

		
		world->nextNeighborInit2(loc, 10, info);
		otherData = world->nextAgentDataIntoSharedMem(info);
		while (otherData != NULL) {
			otherDataLocal = *otherData;
			ds = world->tds(otherDataLocal.loc, loc);
			if (ds < 50 && ds > 0) {
				info.count++;
				computeSocialForce(dataLocal, otherDataLocal, fSum);
			}
			otherData = world->nextAgentDataIntoSharedMem(info);
		}
		
		//compute force with wall
		for (int wallIdx = 0; wallIdx < obsLineNum; wallIdx++) {
			float diw, crx, cry;
			diw = obsLines[wallIdx].pointToLineDist(loc, crx, cry);
			float virDiw = DIST(loc.x, loc.y, crx, cry);
			float niwx = (loc.x - crx) / virDiw;
			float niwy = (loc.y - cry) / virDiw;
			float drw = mass / cMass - diw;
			float fiw1 = A * exp(drw / B);
			if (drw > 0)
				fiw1 += k1 * drw;
			float fniwx = fiw1 * niwx;
			float fniwy = fiw1 * niwy;

			float fiwKgx = 0, fiwKgy = 0;
			if (drw > 0)
			{
				float fiwKg = k2 * drw * (velo.x * (-niwy) + velo.y * niwx);
				fiwKgx = fiwKg * (-niwy);
				fiwKgy = fiwKg * niwx;
			}

			fSum.x += fniwx - fiwKgx;
			fSum.y += fniwy - fiwKgy;
		}
		

		//sum up
		dvt.x += fSum.x / mass;
		dvt.y += fSum.y / mass;
		
		float2d_t newVelo = velo;
		float2d_t newLoc = loc;
		float2d_t newGoal = goal;
		float tick = 0.1;
		newVelo.x += dvt.x * tick * (1 + this->random->gaussian() * 0.1);
		newVelo.y += dvt.y * tick * (1 + this->random->gaussian() * 0.1);
		float dv = sqrt(newVelo.x * newVelo.x + newVelo.y * newVelo.y);

		if (dv > maxv) {
			newVelo.x = newVelo.x * maxv / dv;
			newVelo.y = newVelo.y * maxv / dv;
		}

		float mint = 1;
		for (int wallIdx = 0; wallIdx < obsLineNum; wallIdx++) 
		{
			float crx, cry, tt;
			int ret = obsLines[wallIdx].intersection2LineSeg(
				loc.x, 
				loc.y, 
				loc.x + 0.5 * newVelo.x * tick,
				loc.y + 0.5 * newVelo.y * tick,
				crx,
				cry
				);
			if (ret == 1) 
			{
				if (fabs(crx - loc.x) > 0)
					tt = (crx - loc.x) / (newVelo.x * tick);
				else
					tt = (crx - loc.y) / (newVelo.y * tick + 1e-20);
				if (tt < mint)
					mint = tt;
			}
		}

		newVelo.x *= mint;
		newVelo.y *= mint;
		newLoc.x += newVelo.x * tick;
		newLoc.y += newVelo.y * tick;

		if ((newLoc.x - mass/cMass <= 25) && (newLoc.y - mass/cMass > 49) && (newLoc.y - mass/cMass < 51)) 
		{
			//int idx = threadIdx.x + blockIdx.x * blockDim.x;
			//sfModel->agentPool->remove(idx);
			newGoal.x = 0;
		}

		//newLoc.x += (this->random->uniform()-0.5) * width * 0.02 + loc.x;
		//newLoc.y += (this->random->uniform()-0.5) * height * 0.02 + loc.y;
		newLoc.x = correctCrossBoader(newLoc.x, width);
		newLoc.y = correctCrossBoader(newLoc.y, height);

		SocialForceAgentData_t *dataCopyLocalPtr = (SocialForceAgentData_t*)this->dataCopy;
		dataCopyLocalPtr->loc = newLoc;
		dataCopyLocalPtr->velocity = newVelo;
		dataCopyLocalPtr->goal = newGoal;
	}
};

#endif