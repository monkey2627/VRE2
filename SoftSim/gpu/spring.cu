#include "gpuvar.h"
#include "gpufun.h"

//���㶥��ĳ��ٶ�
extern "C" int runcalculateSTMU(float damping, float dt) {

	//ÿ��block�е��߳���
	int threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;
	calculateSTMU << <blockNum, threadNum >> > (triVertPos_d, triVertPos_old_d, triVertPos_prev_d, triVertVelocity_d, triVertExternForce_d, 
		triVertFixed_d, triVertNum_d, gravityX_d, gravityY_d, gravityZ_d, damping, dt);

	cudaDeviceSynchronize();
	printCudaError("runcalculateSTMU");
	return 0;
}


__global__ void calculateSTMU(float* positions, float* old_positions, float* prev_positions, float* velocity, 
	float* externForce, float* fixed, int vertexNum, float gravityX, float gravityY, float gravityZ, float damping, float dt) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;
	float fixflag = fixed[threadid] > 1e8 ? 0 : 1;
	//�˶�������
	velocity[indexX] *= damping * fixflag;
	velocity[indexY] *= damping * fixflag;
	velocity[indexZ] *= damping * fixflag;
	//ʩ������
	velocity[indexX] += gravityX * dt * fixflag;
	velocity[indexY] += gravityY * dt * fixflag;
	velocity[indexZ] += gravityZ * dt * fixflag;
	//ʩ����������
	velocity[indexX] += externForce[indexX] * dt * fixflag;
	velocity[indexY] += externForce[indexY] * dt * fixflag;
	velocity[indexZ] += externForce[indexZ] * dt * fixflag;

	positions[indexX] += velocity[indexX] * dt * fixflag;
	positions[indexY] += velocity[indexY] * dt * fixflag;
	positions[indexZ] += velocity[indexZ] * dt * fixflag;


	//st
	old_positions[indexX] = positions[indexX];
	old_positions[indexY] = positions[indexY];
	old_positions[indexZ] = positions[indexZ];
	prev_positions[indexX] = positions[indexX];
	prev_positions[indexY] = positions[indexY];
	prev_positions[indexZ] = positions[indexZ];

	//��������
	externForce[indexX] = 0;
	externForce[indexY] = 0;
	externForce[indexZ] = 0;
}


extern "C" int runClearCollisionMU() {
	cudaMemset(triVertForce_d, 0.0f, triVertNum_d * 3 * sizeof(float));
	cudaMemset(triVertisCollide_d, 0, triVertNum_d * sizeof(unsigned char));
	cudaMemset(triVertCollisionDiag_d, 0.0f, triVertNum_d * 3 * sizeof(float));
	cudaMemset(triVertCollisionForce_d, 0.0f, triVertNum_d * 3 * sizeof(float));
	cudaMemset(triVertInsertionDepth_d, 0.0f, triVertNum_d * sizeof(float));
	printCudaError("runClearCollisionMU");
	return 0;
}

//���㶥�������
extern "C" int runcalculateIFMU() {
	int threadNum = 512;
	int blockNum = (triEdgeNum_d + threadNum - 1) / threadNum;

	//printf("spring number: %d\n", triEdgeNum_d);
	calculateIFMU << <blockNum, threadNum >> > (triVertPos_d, triVertForce_d, // vert number
		triEdgeStiffness_d, triEdgeOrgLength_d, triEdgeIndex_d, //spring number
		triVertFixed_d,
		triEdgeNum_d);
	cudaDeviceSynchronize();
	printCudaError("runcalculateIFMU");
	return 0;
}

__global__ void calculateIFMU(float* positions, float* force, 
	float* springStiffness, float* springOrigin, unsigned int* springIndex, 
	float* triVertFixed,
	int springNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("in calculateIFMU\n");
	if (threadid >= springNum) return;

	int vIndex0 = springIndex[threadid * 2 + 0];
	int vIndex1 = springIndex[threadid * 2 + 1];

	//printf("threadid:%d, spring index:%d %d\n", threadid, vIndex0, vIndex1);
	//��ȡ�����������local��
	float pos0x = positions[vIndex0 * 3 + 0];
	float pos0y = positions[vIndex0 * 3 + 1];
	float pos0z = positions[vIndex0 * 3 + 2];
	float pos1x = positions[vIndex1 * 3 + 0];
	float pos1y = positions[vIndex1 * 3 + 1];
	float pos1z = positions[vIndex1 * 3 + 2];

#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		printf("calculateIFMU p0[%f %f %f] p1[%f %f %f]\n",
			pos0x, pos0y, pos0z, pos1x, pos1y, pos1z);
	}
#endif

	//����local��d
	float dx = pos0x - pos1x;
	float dy = pos0y - pos1y;
	float dz = pos0z - pos1z;

	float length = sqrt(dx * dx + dy * dy + dz * dz);
	if (length < springOrigin[threadid]) return;
	dx = dx * (springOrigin[threadid] / length);
	dy = dy * (springOrigin[threadid] / length);
	dz = dz * (springOrigin[threadid] / length);

	//��Ӧ�������˵������
	//����Ӧ����Ҫԭ�Ӳ���
	float tempx = dx - pos0x + pos1x;
	float tempy = dy - pos0y + pos1y;
	float tempz = dz - pos0z + pos1z;

	// �ѵ��������ڵ������˵Ķ�����
	atomicAdd(force + vIndex0 * 3 + 0, tempx * springStiffness[threadid]);
	atomicAdd(force + vIndex0 * 3 + 1, tempy * springStiffness[threadid]);
	atomicAdd(force + vIndex0 * 3 + 2, tempz * springStiffness[threadid]);

	atomicAdd(force + vIndex1 * 3 + 0, -tempx * springStiffness[threadid]);
	atomicAdd(force + vIndex1 * 3 + 1, -tempy * springStiffness[threadid]);
	atomicAdd(force + vIndex1 * 3 + 2, -tempz * springStiffness[threadid]);
}

__global__ void calculateIFMU(float* positions, float* force, 
	float* springStiffness, float* springOrigin, unsigned int* springIndex, 
	int* sortedSpringIndices, int offset, int activeElementNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("in calculateIFMU\n");
	if (threadid >= activeElementNum) return;

	int springIdx = sortedSpringIndices[offset + threadid];

	int vIndex0 = springIndex[springIdx * 2 + 0];
	int vIndex1 = springIndex[springIdx * 2 + 1];


	//��ȡ�����������local��
	float pos0x = positions[vIndex0 * 3 + 0];
	float pos0y = positions[vIndex0 * 3 + 1];
	float pos0z = positions[vIndex0 * 3 + 2];
	float pos1x = positions[vIndex1 * 3 + 0];
	float pos1y = positions[vIndex1 * 3 + 1];
	float pos1z = positions[vIndex1 * 3 + 2];


#ifdef OUTPUT_INFO
	if (springIdx == LOOK_THREAD)
	{
		printf("calculateIFMU p0[%f %f %f] p1[%f %f %f]\n",
			pos0x, pos0y, pos0z, pos1x, pos1y, pos1z);
	}
	if (threadid == 0)
	{
		printf("springIdx:%d, vert index:%d %d pos1[%f %f %f] pos2[%f %f %f]\n", springIdx, vIndex0, vIndex1,
			pos0x, pos0y, pos0z, pos1x, pos1y, pos1z);
	}
#endif

	//����local��d
	float dx = pos0x - pos1x;
	float dy = pos0y - pos1y;
	float dz = pos0z - pos1z;

	float length = sqrt(dx * dx + dy * dy + dz * dz);
	if (length < springOrigin[springIdx]) return;
	dx = dx * (springOrigin[springIdx] / length);
	dy = dy * (springOrigin[springIdx] / length);
	dz = dz * (springOrigin[springIdx] / length);

	//��Ӧ�������˵������
	//����Ӧ����Ҫԭ�Ӳ���
	float tempx = dx - pos0x + pos1x;
	float tempy = dy - pos0y + pos1y;
	float tempz = dz - pos0z + pos1z;

	//if (vIndex1 == 72990)
	//{
	//	float k = springStiffness[springIdx];
	//	printf("springIdx:%d, vert 72990 spring force[%f %f %f] stiffness:%f\n", springIdx, k * tempx, k * tempy, k * tempz, springStiffness[springIdx]);
	//}
	atomicAdd(force + vIndex0 * 3 + 0, tempx * springStiffness[springIdx]);
	atomicAdd(force + vIndex0 * 3 + 1, tempy * springStiffness[springIdx]);
	atomicAdd(force + vIndex0 * 3 + 2, tempz * springStiffness[springIdx]);

	atomicAdd(force + vIndex1 * 3 + 0, -tempx * springStiffness[springIdx]);
	atomicAdd(force + vIndex1 * 3 + 1, -tempy * springStiffness[springIdx]);
	atomicAdd(force + vIndex1 * 3 + 2, -tempz * springStiffness[springIdx]);
}

int runcalculateRestPosForceWithTetPos(float toolRadius)
{

	int threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;

	// ���ݾ����Զ�����restpos stiffness�� ��Ҫ��һ����������
	calculateRestStiffnessWithTet << <blockNum, threadNum >> > (
		toolPositionAndDirection_d, toolCollideFlag_d, 
		triVertPos_d, triVertisCollide_d, triVertfromTetStiffness_d, 
		cylinderNum_d, triVertNum_d);
	calculateRestPosWithTetPosMU << <blockNum, threadNum >> > (triVertPos_d, triVert2TetVertMapping_d,
		triVertForce_d, triVertCollisionDiag_d,
		tetVertPos_d, triVertfromTetStiffness_d,
		triVertNum_d);
	cudaDeviceSynchronize();
	printCudaError("runcalculateRestPosForceWithTetPos");
	return 0;
}

__global__ void calculateRestStiffnessWithTet(float* ballPos, unsigned char* toolCollideFlag, float* positions, unsigned char* isCollide, float* restStiffness, int toolNum, int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float base_stiffness = 100;  //���Ƹն�ϵ���Ĵ�С��Χ
	float max_stiffness = 110;

	bool flag = false;
	for (int i = 0; i < toolNum; i++)
	{
		if (toolCollideFlag[i] > 0)
			flag = true;
	}
	if (flag) { //����͹��߷�����ײ 
		switch (isCollide[threadid])
		{
		case 1: {  //��ѹ�㣬�͹���ֱ�ӷ�����ײ�Ķ��㣬���ఴ���Լ�������ȥ���Σ��ն�ϵ����С
			//restStiffness[threadid] = 1.0;
			restStiffness[threadid] = base_stiffness;
#//ifdef OUTPUT_INFO
			if (threadid == LOOK_THREAD)
				printf("mesh reststiffness in thread:%d: %f\n", threadid, restStiffness[LOOK_THREAD]);
#//endif
			break;
		}
		case 2: { //��ȡ�㣬�͹���ֱ����ײ�Ķ��㣬Ϊ���ܸ���ץǯ�ƶ����ն�ϵ���ϴ�
			restStiffness[threadid] = 1 * max_stiffness;
			break;
		}
		case 0: { //����ײ�㣬���ݶ��㵽���ߵľ�����㲻ͬ�ĸն�ϵ��
			float distance = 1e9 + 7;  //���㶥�㵽������������ľ���
			int indexX = threadid * 3 + 0;
			int indexY = threadid * 3 + 1;
			int indexZ = threadid * 3 + 2;
			float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
			for (int i = 0; i < toolNum; i++)
			{
				float dir[3] = { ballPos[0] - p[0], ballPos[1] - p[1], ballPos[2] - p[2] };
				float d = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
				if (d < distance) distance = d;
			}

			float k;
			//k = 10.0 / (1 + exp(-distance + 2));
			//k = distance + 3.0;
			//k = 0.6 / (exp2(0.3 * (-distance + 0.3)));  //����ģ��
			k = 0.6 / (exp2(0.3 * (-distance + 0.3)));
			restStiffness[threadid] = k * base_stiffness;
			if (restStiffness[threadid] > max_stiffness) restStiffness[threadid] = max_stiffness;
		}
		default:
			break;
		}
	}
	else {   //���û�к͹��߷�����ײ�������������ƶ����ն�ϵ����Ϊ���
		restStiffness[threadid] = max_stiffness;
	}
}

__global__ void calculateRestStiffnessWithTet(float* ballPos, float toolRadius,
	unsigned char* toolCollideFlag, float* positions,
	unsigned char* isCollide, float* restStiffness,
	int toolNum, int* sortedIndices, int offset, int activeElementNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	int vertIdx = sortedIndices[offset + threadid];

	float base_stiffness = 0.7;  //���Ƹն�ϵ���Ĵ�С��Χ
	float max_stiffness = 1;

	bool flag = false;
	for (int i = 0; i < toolNum; i++)
	{
		if (toolCollideFlag[i] > 0)
			flag = true;
	}
	if (flag) { //����͹��߷�����ײ 
		switch (isCollide[vertIdx])
		{
		case 1: {  //��ѹ�㣬�͹���ֱ�ӷ�����ײ�Ķ��㣬���ఴ���Լ�������ȥ���Σ��ն�ϵ����С
			//restStiffness[vertIdx] = 1.0;
			restStiffness[vertIdx] = 0.0;
#//ifdef OUTPUT_INFO
			if (vertIdx == LOOK_THREAD)
				printf("mesh reststiffness in thread:%d: %f\n", vertIdx, restStiffness[LOOK_THREAD]);
#//endif
			break;
		}
		case 2: { //��ȡ�㣬�͹���ֱ����ײ�Ķ��㣬Ϊ���ܸ���ץǯ�ƶ����ն�ϵ���ϴ�
			restStiffness[vertIdx] = 1 * max_stiffness;
			break;
		}
		case 0: { //����ײ�㣬���ݶ��㵽���ߵľ�����㲻ͬ�ĸն�ϵ��
			float distance = 1e9 + 7;  //���㶥�㵽������������ľ���
			int indexX = vertIdx * 3 + 0;
			int indexY = vertIdx * 3 + 1;
			int indexZ = vertIdx * 3 + 2;
			float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
			for (int i = 0; i < toolNum; i++)
			{
				float dir[3] = { ballPos[0] - p[0], ballPos[1] - p[1], ballPos[2] - p[2] };
				float d = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
				if (d < distance) distance = d;
			}

			float k;
			//k = 10.0 / (1 + exp(-distance + 2));
			//k = distance + 3.0;
			//k = 0.6 / (exp2(0.3 * (-distance + 0.3)));  //����ģ��

			//restStiffness[vertIdx] = max_stiffness;

			//float x = distance - 1*toolRadius;
			//float ratio = 2 / (1 + exp(-x)) - 1;
			//restStiffness[vertIdx] = ratio * max_stiffness;

			//float x = distance - 2 * toolRadius;
			//float influence_r = 2*toolRadius;
			//if (x > influence_r)
			//	x = influence_r;
			//else if (x < 0)
			//	x = 0;

			//float t = x / influence_r;
			//restStiffness[vertIdx] = t*t*t * max_stiffness;

			k = 0.6 / (exp2(0.3 * (-distance + toolRadius)));
			restStiffness[threadid] = k * base_stiffness;
			if (restStiffness[threadid] > max_stiffness) restStiffness[threadid] = max_stiffness;

		}
		default:
			break;
		}
	}
	else {   //���û�к͹��߷�����ײ�������������ƶ����ն�ϵ����Ϊ���
		restStiffness[vertIdx] = max_stiffness;
	}
	restStiffness[vertIdx] = max_stiffness;
}

__global__ void calculateRestPosWithTetPosMU(float* positions, int* skeletonIndex, float* force, float* collisionDiag, 
	float* rest_positions, float* restStiffness, int vertexNum) {
	/*˵����skeletonIndex��СΪ�������Ƕ�������x2
		ÿһ�������������񶥵��Ӧ������������ʾ����󶨵������嶥���±�
		����ڶ�������Ϊ-1�������������ֱ�Ӷ�Ӧ�ŵ�һ��������ʾ�������嶥��
		������������������񶥵����ڱ�������ϸ�ֹ��������ӵĶ��㣬restpos��Ӧ���������嶥����е㡣
	*/
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	int tet_idx1 = skeletonIndex[2 * threadid + 0]; // �����嶥��1�±�
	int tet_idx2 = skeletonIndex[2 * threadid + 1]; // �����嶥��2�±�

	//��ȡ��������
	float tri_pos0x = positions[3 * threadid + 0]; // ���涥������
	float tri_pos0y = positions[3 * threadid + 1];
	float tri_pos0z = positions[3 * threadid + 2];
	//��������
	float tempx = (rest_positions[3 * tet_idx1 + 0] + rest_positions[3 * tet_idx2 + 0]) * 0.5f - tri_pos0x;
	float tempy = (rest_positions[3 * tet_idx1 + 1] + rest_positions[3 * tet_idx2 + 1]) * 0.5f - tri_pos0y;
	float tempz = (rest_positions[3 * tet_idx1 + 2] + rest_positions[3 * tet_idx2 + 2]) * 0.5f - tri_pos0z;

#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		float tet0_x = rest_positions[3 * tet_idx1 + 0];
		float tet0_y = rest_positions[3 * tet_idx1 + 1];
		float tet0_z = rest_positions[3 * tet_idx1 + 2];

		float tet1_x = rest_positions[3 * tet_idx2 + 0];
		float tet1_y = rest_positions[3 * tet_idx2 + 1];
		float tet1_z = rest_positions[3 * tet_idx2 + 2];

		//printf("threadid:%d, vertexRestForce:[%f %f %f]\n", threadid, tempx * springStiffness, tempy * springStiffness, tempz * springStiffness);
		printf("calculateIFRestMUDefaultStiffness threadid:%d, springStiffness:%f tetidx0:%d tetidx1:%d temp[%f %f %f]\n", threadid, restStiffness[threadid], tet_idx1, tet_idx2, tempx, tempy, tempz);
		printf("\t tet0[%f %f %f] tet1[%f %f %f] tri[%f %f %f]\n",
			tet0_x, tet0_y, tet0_z,
			tet1_x, tet1_y, tet1_z,
			tri_pos0x, tri_pos0y, tri_pos0z);
	}
#endif
	atomicAdd(force + threadid * 3 + 0, tempx * restStiffness[threadid]);
	atomicAdd(force + threadid * 3 + 1, tempy * restStiffness[threadid]);
	atomicAdd(force + threadid * 3 + 2, tempz * restStiffness[threadid]);

	atomicAdd(collisionDiag + threadid * 3 + 0, restStiffness[threadid]);
	atomicAdd(collisionDiag + threadid * 3 + 1, restStiffness[threadid]);
	atomicAdd(collisionDiag + threadid * 3 + 2, restStiffness[threadid]);
}

__global__ void calculateRestPosWithTetPosMU(float* positions, int* skeletonIndex, float* force, float* collisionDiag,
	float* rest_positions, float* restStiffness, 
	int* sortedIndices, int offset, int activeElementNum) 
{
	/*˵����skeletonIndex��СΪ�������Ƕ�������x2
		ÿһ�������������񶥵��Ӧ������������ʾ����󶨵������嶥���±�
		����ڶ�������Ϊ-1�������������ֱ�Ӷ�Ӧ�ŵ�һ��������ʾ�������嶥��
		������������������񶥵����ڱ�������ϸ�ֹ��������ӵĶ��㣬restpos��Ӧ���������嶥����е㡣
	*/
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	int vertIdx = sortedIndices[offset + threadid];

	int tet_idx1 = skeletonIndex[2 * vertIdx + 0]; // �����嶥��1�±�
	int tet_idx2 = skeletonIndex[2 * vertIdx + 1]; // �����嶥��2�±�

	//��ȡ��������
	float tri_pos0x = positions[3 * vertIdx + 0]; // ���涥������
	float tri_pos0y = positions[3 * vertIdx + 1];
	float tri_pos0z = positions[3 * vertIdx + 2];
	//��������
	float tempx = (rest_positions[3 * tet_idx1 + 0] + rest_positions[3 * tet_idx2 + 0]) * 0.5f - tri_pos0x;
	float tempy = (rest_positions[3 * tet_idx1 + 1] + rest_positions[3 * tet_idx2 + 1]) * 0.5f - tri_pos0y;
	float tempz = (rest_positions[3 * tet_idx1 + 2] + rest_positions[3 * tet_idx2 + 2]) * 0.5f - tri_pos0z;

#ifdef OUTPUT_INFO
	if (vertIdx == LOOK_THREAD)
	{
		float tet0_x = rest_positions[3 * tet_idx1 + 0];
		float tet0_y = rest_positions[3 * tet_idx1 + 1];
		float tet0_z = rest_positions[3 * tet_idx1 + 2];

		float tet1_x = rest_positions[3 * tet_idx2 + 0];
		float tet1_y = rest_positions[3 * tet_idx2 + 1];
		float tet1_z = rest_positions[3 * tet_idx2 + 2];

		//printf("vertIdx:%d, vertexRestForce:[%f %f %f]\n", vertIdx, tempx * springStiffness, tempy * springStiffness, tempz * springStiffness);
		printf("calculateIFRestMUDefaultStiffness vertIdx:%d, springStiffness:%f tetidx0:%d tetidx1:%d temp[%f %f %f]\n", vertIdx, restStiffness[vertIdx], tet_idx1, tet_idx2, tempx, tempy, tempz);
		printf("\t tet0[%f %f %f] tet1[%f %f %f] tri[%f %f %f]\n",
			tet0_x, tet0_y, tet0_z,
			tet1_x, tet1_y, tet1_z,
			tri_pos0x, tri_pos0y, tri_pos0z);
	}
#endif
	atomicAdd(force + vertIdx * 3 + 0, tempx * restStiffness[vertIdx]);
	atomicAdd(force + vertIdx * 3 + 1, tempy * restStiffness[vertIdx]);
	atomicAdd(force + vertIdx * 3 + 2, tempz * restStiffness[vertIdx]);

	atomicAdd(collisionDiag + vertIdx * 3 + 0, restStiffness[vertIdx]);
	atomicAdd(collisionDiag + vertIdx * 3 + 1, restStiffness[vertIdx]);
	atomicAdd(collisionDiag + vertIdx * 3 + 2, restStiffness[vertIdx]);
}

///����ÿ�������restposԼ��
int runcalculateRestPosMU() {
	int  threadNum = 512;
	int blockNum = (tetNum_d + threadNum - 1) / threadNum;
	calculateRestPosStiffness << <blockNum, threadNum >> > (
		toolPositionAndDirection_d, toolCollideFlag_d, tetVertPos_d, tetVertisCollide_d, tetVertRestStiffness_d, 1, tetVertNum_d
		);
	calculateRestPos << <blockNum, threadNum >> > (
		tetVertPos_d, tetVertRestPos_d,
		tetVertCollisionForce_d, tetVertCollisionDiag_d,
		tetVertRestStiffness_d, tetVertNum_d);

	cudaDeviceSynchronize();
	printCudaError("runcalculateRestPos");
	return 0;
}

//�б�ѩ�����λ��
extern "C" int runcalculatePosMU(float omega, float dt) {

	int threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;
	//���м���
	
	calculatePOSMU << <blockNum, threadNum >> > (triVertPos_d,
		triVertForce_d, triVertFixed_d, triVertMass_d,
		triVertPos_next_d, triVertPos_prev_d, triVertPos_old_d,
		triEdgeDiag_d, triVertCollisionDiag_d, triVertCollisionForce_d,
		triVertNum_d, dt, omega);

	cudaDeviceSynchronize();
	printCudaError("runcalculatePosMU");
	return 0;
}


__global__ void calculatePOSMU(float* positions, float* force, float* fixed, float* mass, float* next_positions, float* prev_positions, float* old_positions, float* springDiag, float* collisionDiag, float* collisionForce, int vertexNum, float dt, float omega) {

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	if (fixed[threadid] > 1e8) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	float diagConstant = (mass[threadid] + fixed[threadid]) / (dt * dt);

	float elementX = force[indexX] + collisionForce[indexX];
	float elementY = force[indexY] + collisionForce[indexY];
	float elementZ = force[indexZ] + collisionForce[indexZ];

#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		printf("calculatePOSMU springDiag:%f collisionDiag:[%f %f %f] constantDiag:%f\n",
			springDiag[threadid], collisionDiag[indexX], collisionDiag[indexY], collisionDiag[indexZ], diagConstant);
	}
#endif
	//�൱���Ȱ������˶���ÿ��������������Ч����������
	next_positions[indexX] = (diagConstant * (old_positions[indexX] - positions[indexX]) + elementX) / (springDiag[threadid] + collisionDiag[indexX] + diagConstant) + positions[indexX];
	next_positions[indexY] = (diagConstant * (old_positions[indexY] - positions[indexY]) + elementY) / (springDiag[threadid] + collisionDiag[indexY] + diagConstant) + positions[indexY];
	next_positions[indexZ] = (diagConstant * (old_positions[indexZ] - positions[indexZ]) + elementZ) / (springDiag[threadid] + collisionDiag[indexZ] + diagConstant) + positions[indexZ];

	//under-relaxation �� �б�ѩ�����
	next_positions[indexX] = (next_positions[indexX] - positions[indexX]) * 0.6 + positions[indexX];
	next_positions[indexY] = (next_positions[indexY] - positions[indexY]) * 0.6 + positions[indexY];
	next_positions[indexZ] = (next_positions[indexZ] - positions[indexZ]) * 0.6 + positions[indexZ];

	next_positions[indexX] = omega * (next_positions[indexX] - prev_positions[indexX]) + prev_positions[indexX];
	next_positions[indexY] = omega * (next_positions[indexY] - prev_positions[indexY]) + prev_positions[indexY];
	next_positions[indexZ] = omega * (next_positions[indexZ] - prev_positions[indexZ]) + prev_positions[indexZ];

	prev_positions[indexX] = positions[indexX];
	prev_positions[indexY] = positions[indexY];
	prev_positions[indexZ] = positions[indexZ];

	positions[indexX] = next_positions[indexX];
	positions[indexY] = next_positions[indexY];
	positions[indexZ] = next_positions[indexZ];
}

__global__ void calculatePOSMU(float* positions, float* force, float* fixed, float* mass, float* next_positions, float* prev_positions,
	float* old_positions, float* springDiag, float* collisionDiag, float* collisionForce,
	int* sortedIndices, int offset, int activeElementNum,
	float dt, float omega)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;
	
	int vertIdx = sortedIndices[offset + threadid];
	int indexX = vertIdx * 3 + 0;
	int indexY = vertIdx * 3 + 1;
	int indexZ = vertIdx * 3 + 2;

	float diagConstant = (mass[vertIdx] + fixed[vertIdx]) / (dt * dt);
	//if (vertIdx == LOOK_THREAD)
	//	printf("calculatePOSMU vertIdx:0, diagConstant:%d\n", diagConstant);
	float elementX = force[indexX] + collisionForce[indexX];
	float elementY = force[indexY] + collisionForce[indexY];
	float elementZ = force[indexZ] + collisionForce[indexZ];
	//if (vertIdx == LOOK_THREAD)
	//	printf("elements:[%f %f %f]\n", elementX, elementY, elementZ);

#ifdef OUTPUT_INFO
	if (vertIdx == LOOK_THREAD)
	{
		printf("calculatePOSMU springDiag:%f collisionDiag:[%f %f %f] constantDiag:%f\n",
			springDiag[vertIdx], collisionDiag[indexX], collisionDiag[indexY], collisionDiag[indexZ], diagConstant);
	}
#endif
	//�൱���Ȱ������˶���ÿ��������������Ч����������
	next_positions[indexX] = (diagConstant * (old_positions[indexX] - positions[indexX]) + elementX) / (springDiag[vertIdx] + collisionDiag[indexX] + diagConstant) + positions[indexX];
	next_positions[indexY] = (diagConstant * (old_positions[indexY] - positions[indexY]) + elementY) / (springDiag[vertIdx] + collisionDiag[indexY] + diagConstant) + positions[indexY];
	next_positions[indexZ] = (diagConstant * (old_positions[indexZ] - positions[indexZ]) + elementZ) / (springDiag[vertIdx] + collisionDiag[indexZ] + diagConstant) + positions[indexZ];

	//under-relaxation �� �б�ѩ�����
	next_positions[indexX] = (next_positions[indexX] - positions[indexX]) * 0.6 + positions[indexX];
	next_positions[indexY] = (next_positions[indexY] - positions[indexY]) * 0.6 + positions[indexY];
	next_positions[indexZ] = (next_positions[indexZ] - positions[indexZ]) * 0.6 + positions[indexZ];

	next_positions[indexX] = omega * (next_positions[indexX] - prev_positions[indexX]) + prev_positions[indexX];
	next_positions[indexY] = omega * (next_positions[indexY] - prev_positions[indexY]) + prev_positions[indexY];
	next_positions[indexZ] = omega * (next_positions[indexZ] - prev_positions[indexZ]) + prev_positions[indexZ];

	prev_positions[indexX] = positions[indexX];
	prev_positions[indexY] = positions[indexY];
	prev_positions[indexZ] = positions[indexZ];

	positions[indexX] = next_positions[indexX];
	positions[indexY] = next_positions[indexY];
	positions[indexZ] = next_positions[indexZ];
}
//�����ٶ�
extern "C" int runcalculateVMU(float dt) {

	int threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;
	
	calculateVMU << <blockNum, threadNum >> > (triVertPos_d, triVertVelocity_d, triVertPos_old_d, triVertNum_d, dt);

	printCudaError("runcalculateVMU");
	return 0;
}

__global__ void calculateVMU(float* positions, float* velocity, float* old_positions, int vertexNum, float dt) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	velocity[threadid * 3 + 0] += (positions[threadid * 3 + 0] - old_positions[threadid * 3 + 0]) / dt;
	velocity[threadid * 3 + 1] += (positions[threadid * 3 + 1] - old_positions[threadid * 3 + 1]) / dt;
	velocity[threadid * 3 + 2] += (positions[threadid * 3 + 2] - old_positions[threadid * 3 + 2]) / dt;
#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		printf("calculateVMU v[%f %f %f]\n",
			(positions[threadid * 3 + 0] - old_positions[threadid * 3 + 0]) / dt,
			(positions[threadid * 3 + 1] - old_positions[threadid * 3 + 1]) / dt,
			(positions[threadid * 3 + 2] - old_positions[threadid * 3 + 2]) / dt);
	}
#endif // OUTPUT_INFO

}

__global__ void calculateVMU(float* positions, float* velocity, float* old_positions, 
	int* sortedIndices, int offset, int activeElementNum,
	float dt) 
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	int vertIdx = sortedIndices[offset + threadid];
	velocity[vertIdx * 3 + 0] += (positions[vertIdx * 3 + 0] - old_positions[vertIdx * 3 + 0]) / dt;
	velocity[vertIdx * 3 + 1] += (positions[vertIdx * 3 + 1] - old_positions[vertIdx * 3 + 1]) / dt;
	velocity[vertIdx * 3 + 2] += (positions[vertIdx * 3 + 2] - old_positions[vertIdx * 3 + 2]) / dt;
#ifdef OUTPUT_INFO
	if (vertIdx == LOOK_THREAD)
	{
		printf("calculateVMU v[%f %f %f]\n",
			(positions[vertIdx * 3 + 0] - old_positions[vertIdx * 3 + 0]) / dt,
			(positions[vertIdx * 3 + 1] - old_positions[vertIdx * 3 + 1]) / dt,
			(positions[vertIdx * 3 + 2] - old_positions[vertIdx * 3 + 2]) / dt);
	}
#endif // OUTPUT_INFO

}

