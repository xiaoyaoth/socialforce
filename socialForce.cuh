#ifndef SOCIAL_FORCE_CUH
#define SOCIAL_FORCE_CUH

#include "gsimcore.cuh"
#include "gsimlib_header.cuh"
#include "socialForceHeader.cuh"

class SocialForceModel;
class SocialForceAgent;

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
	}
};

class SocialForceAgent : public GAgent {
public:
	GRandom *random;
	SocialForceModel *model;
	__device__ SocialForceAgent(int id, SocialForceModel *model) {
		this->model = model;
		this->random = new GRandom(2345, id);
		SocialForceAgentData_t *data = new SocialForceAgentData_t();
		SocialForceAgentData_t *dataCopy = new SocialForceAgentData_t();
		data->loc.x = model->world->width * this->random->uniform();
		data->loc.y = model->world->height * this->random->uniform();
		*dataCopy = *data;

		this->data = data;
		this->dataCopy = dataCopy;
	}

	__device__ void step(GModel *model){
	}
};

#endif