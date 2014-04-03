#ifndef SOCIAL_FORCE_HEADER_CUH
#define SOCIAL_FORCE_HEADER_CUH

#include "gsimlib_header.cuh"
typedef struct dataUnionStruct : public GAgentData_t {
	__device__ void addValue(GAgentData_t *data){
	}
}dataUnion;

typedef struct socialForceAgentData : public GAgentData_t {
} SocialForceAgentData_t;

#endif