#include "gpuvar.h"
#include "gpufun.h"


__global__ void calculateVec3Len(float* vec, float* len, int vecNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vecNum) return;

	float x = vec[threadid * 3 + 0];
	float y = vec[threadid * 3 + 1];
	float z = vec[threadid * 3 + 2];
	float vecLen = sqrt(x * x + y * y + z * z);
	//if (vecLen > 0)
	//{
	//	printf("len>0 %f\n", vecLen);
	//}
	len[threadid] = vecLen;
}

//计算初始状态
int runcalculateST(float damping, float dt) {
	//每个block中的线程数
	int  threadNum = 512;
	//每个grid中的block数(为了保证)
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	//并行计算
	calculateST << <blockNum, threadNum >> > (tetVertPos_d, tetVertVelocity_d, 
		tetVertExternForce_d, 
		tetVertPos_old_d, tetVertPos_prev_d, tetVertPos_last_d, 
		tetVertFixed_d, 
		tetVertNum_d, gravityX_d, gravityY_d, gravityZ_d, damping, dt);
	cudaDeviceSynchronize();//cuda中核函数的执行都是异步的，加上这一步保证核函数完全执行，或者加上memcpy(cudamemcpy是同步的)
	printCudaError("runcalculateST");
	return 0;
}

__global__ void calculateST(float* positions, float* velocity, float* externForce,
	float* old_positions, float* prev_positions, float* last_Positions, float* fixed,
	int vertexNum, float gravityX, float gravityY, float gravityZ, float damping, float dt)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("calculateST vertNum:%d\n", vertexNum);
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;


	last_Positions[indexX] = positions[indexX];
	last_Positions[indexY] = positions[indexY];
	last_Positions[indexZ] = positions[indexZ];

	float fixflag = fixed[threadid] > 1e8 ? 0 : 1;
	//运动的阻尼
	velocity[indexX] *= damping * fixflag;
	velocity[indexY] *= damping * fixflag;
	velocity[indexZ] *= damping * fixflag;
	//施加重力
	velocity[indexX] += gravityX * dt * fixflag;
	velocity[indexY] += gravityY * dt * fixflag;
	velocity[indexZ] += gravityZ * dt * fixflag;
		
	//施加其他外力
	velocity[indexX] += externForce[indexX] * dt * fixflag;
	velocity[indexY] += externForce[indexY] * dt * fixflag;
	velocity[indexZ] += externForce[indexZ] * dt * fixflag;

	positions[indexX] += velocity[indexX] * dt * fixflag;
	positions[indexY] += velocity[indexY] * dt * fixflag;
	positions[indexZ] += velocity[indexZ] * dt * fixflag;


	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("%s error: %s\n", "velocity and position", cudaGetErrorString(cudaStatus));
	}
	//st
	old_positions[indexX] = positions[indexX];
	old_positions[indexY] = positions[indexY];
	old_positions[indexZ] = positions[indexZ];
	prev_positions[indexX] = positions[indexX];
	prev_positions[indexY] = positions[indexY];
	prev_positions[indexZ] = positions[indexZ];
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("%s error: %s\n", "prev and old position", cudaGetErrorString(cudaStatus));
	}
	//外力清零
	externForce[indexX] = 0.0;
	externForce[indexY] = 0.0;
	externForce[indexZ] = 0.0;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("%s error: %s\n", "externForce", cudaGetErrorString(cudaStatus));
	}
#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
		printf("calculateST p[%f %f %f] v[%f %f %f] extForce[%f %f %f]\n",
			positions[indexX], positions[indexY], positions[indexZ],
			velocity[indexX], velocity[indexY], velocity[indexZ],
			externForce[indexX], externForce[indexY], externForce[indexZ]);
#endif
}

//清空碰撞标记，和碰撞项的对角元素
int runClearCollision() {
	cudaMemset(tetVertisCollide_d, 0, tetVertNum_d * sizeof(unsigned char));
	
	cudaMemset(tetVertCollisionForce_d, 0.0f, tetVertNum_d * 3 * sizeof(float));
	cudaMemset(tetVertCollisionDiag_d, 0.0f, tetVertNum_d * 3 * sizeof(float));
	cudaMemset(tetVertInsertionDepth_d, 0.0f, tetVertNum_d * sizeof(float));

	printCudaError("runClearCollision");
	return 0;
}

int runClearForce()
{
	cudaMemset(tetVertForce_d, 0.0f, tetVertNum_d * 3 * sizeof(float));
	printCudaError("runClearForce");
	return 0;
}

int runCalculateTetEdgeSpringConstraint()
{
	int threadNum = 512;
	int blockNum = (tetSpringNum_d + threadNum - 1) / threadNum;
	//printf("tetSpringNum_d:%d\n", tetSpringNum_d);
	calculateTetEdgeSpringConstraint << <blockNum, threadNum >> > (
		tetVertPos_d,
		tetVertForce_d,
		tetSpringStiffness_d, tetSpringOrgLen_d, tetSpringIndex_d,
		tetSpringNum_d);
	cudaDeviceSynchronize();
	printCudaError("runCalculateTetEdgeSpringConstraint");
	return 0;
}

__global__ void calculateTetEdgeSpringConstraint(
	float* positions, 
	float* force, 
	float* springStiffness, float* springOrigin, int * springIndex, 
	int springNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("in calculateIFMU\n");
	if (threadid >= springNum) return;

	int vIndex0 = springIndex[threadid * 2 + 0];
	int vIndex1 = springIndex[threadid * 2 + 1];


	//printf("threadid:%d, spring index:%d %d\n", threadid, vIndex0, vIndex1);
	//获取顶点坐标计算local解
	float pos0x = positions[vIndex0 * 3 + 0];
	float pos0y = positions[vIndex0 * 3 + 1];
	float pos0z = positions[vIndex0 * 3 + 2];
	float pos1x = positions[vIndex1 * 3 + 0];
	float pos1y = positions[vIndex1 * 3 + 1];
	float pos1z = positions[vIndex1 * 3 + 2];

	//bool outputFlag = vIndex1 == 93;
	bool outputFlag = false;
	if (outputFlag)
		printf("vIndex0=%d vIndex1=%d pos1[%f %f %f]\n", vIndex0, vIndex1, pos1x, pos1y, pos1z);

#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		printf("calculateIFMU p0[%f %f %f] p1[%f %f %f]\n",
			pos0x, pos0y, pos0z, pos1x, pos1y, pos1z);
	}
#endif

	//计算local解d
	float dx = pos0x - pos1x;
	float dy = pos0y - pos1y;
	float dz = pos0z - pos1z;

	float length = sqrt(dx * dx + dy * dy + dz * dz);
	if (length < springOrigin[threadid]) return;
	dx = dx * (springOrigin[threadid] / length);
	dy = dy * (springOrigin[threadid] / length);
	dz = dz * (springOrigin[threadid] / length);

	if (outputFlag)
	{
		printf("dxyz:[%f %f %f] pos1[%f %f %f] springOrg:%f currentLen:%f\n", dx, dy, dz,
			pos1x, pos1y, pos1z,
			springOrigin[threadid], length);
	}
	//对应的两个端点的内力
	//这里应该需要原子操作
	float tempx = dx - pos0x + pos1x;
	float tempy = dy - pos0y + pos1y;
	float tempz = dz - pos0z + pos1z;

	//if (tempx>0.001)
	//{
	//	float k = springStiffness[threadid];
	//	printf("threadid:%d, vert [%d %d] \n\tpos0[%f %f %f] pos1[%f %f %f]\n\tspring force[%f %f %f] stiffness:%f \n", threadid, vIndex0, vIndex1,
	//		pos0x, pos0y, pos0z, pos1x, pos1y, pos1z,
	//		k * tempx, k * tempy, k * tempz, k);
	//}
	atomicAdd(force + vIndex0 * 3 + 0, tempx * springStiffness[threadid]);
	atomicAdd(force + vIndex0 * 3 + 1, tempy * springStiffness[threadid]);
	atomicAdd(force + vIndex0 * 3 + 2, tempz * springStiffness[threadid]);

	atomicAdd(force + vIndex1 * 3 + 0, -tempx * springStiffness[threadid]);
	atomicAdd(force + vIndex1 * 3 + 1, -tempy * springStiffness[threadid]);
	atomicAdd(force + vIndex1 * 3 + 2, -tempz * springStiffness[threadid]);
}

int runcalculateIF() {

	int  threadNum = 512;
	int blockNum = (tetNum_d + threadNum - 1) / threadNum;
	//并行计算
	calculateIF << <blockNum, threadNum >> > (tetVertPos_d, tetIndex_d,
		tetInvD3x3_d, tetInvD3x4_d,
		tetVertForce_d, tetVolume_d, tetActive_d,
		tetNum_d, tetStiffness_d);
	threadNum = 512;
	blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
	calculateVec3Len << <blockNum, threadNum >> > (tetVertForce_d, tetVertForceLen_d, tetVertNum_d);
	cudaDeviceSynchronize();
	printCudaError("runcalculateIF");
	return 0;
}

///计算每个顶点的restpos约束
int runcalculateRestPos() {
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

__global__ void calculateRestPosStiffness(float* ballPos, unsigned char* toolCollideFlag, float* positions, unsigned char* isCollide, float* reststiffness, int toolNum, int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//根据与工具的距离或者碰撞信息，计算restpos刚度系数
	if (threadid >= vertexNum) return;

	float maxStiffness = 200.0;
	float distance = 1e9 + 7;  //计算顶点到两个工具最近的距离
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
	if (isCollide[threadid]) {  //与工具发生碰撞
		switch (isCollide[threadid])
		{
		case 1: {  //按压点，和工具直接发生碰撞的顶点
			reststiffness[threadid] = 0;
			break;
		}
		case 2: { //夹取点，和工具直接碰撞的顶点
			reststiffness[threadid] = 0.0;
			break;
		}
		case 0: { //非碰撞点，根据顶点到工具的距离计算不同的刚度系数
			reststiffness[threadid] = 0.5*maxStiffness*(distance-0.5);
		}
		default:
			break;
		}
	}
	else {   //如果没有和工具发生碰撞，布料不对四面体产生约束力
		reststiffness[threadid] = maxStiffness;
	}
#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
		printf("calculateRestStiffness isCollide:%d, stiffness:%f\n", isCollide[LOOK_THREAD], reststiffness[LOOK_THREAD]);
#endif
}

__global__ void calculateRestPosStiffnessWithMesh_part(
	float* ballPos, float  ballRadius,
	unsigned char* toolCollideFlag, float* positions,
	unsigned char* isCollide, float* meshStiffness,
	int toolNum, int* sortedTetVertIndices, int startIdx, int activeElementNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//根据与工具的距离或者碰撞信息，计算restpos刚度系数
	if (threadid >= activeElementNum) return;
	int tetVertIdx = sortedTetVertIndices[startIdx + threadid];

	float maxStiffness = 1000;
	bool flag = false;
	for (int i = 0; i < toolNum; i++)
	{
		if (toolCollideFlag[i] > 0)
			flag = true;
	}
	float distance = 1e9 + 7;  //计算顶点到两个工具最近的距离
	int indexX = tetVertIdx * 3 + 0;
	int indexY = tetVertIdx * 3 + 1;
	int indexZ = tetVertIdx * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	for (int i = 0; i < toolNum; i++)
	{
		float dir[3] = { ballPos[0] - p[0], ballPos[1] - p[1], ballPos[2] - p[2] };
		float d = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
		if (d < distance) distance = d;
	}
	if (flag) {  //与工具发生碰撞
		switch (isCollide[tetVertIdx])
		{
		case 1: {  //按压点，和工具直接发生碰撞的顶点
			meshStiffness[tetVertIdx] = maxStiffness;
#ifdef OUTPUT_INFO
			if (tetVertIdx == LOOK_THREAD)
				printf("tet reststiffness with mesh in thread:%d: %f\n", meshStiffness[LOOK_THREAD]);
#endif
			break;
		}
		case 2: { //夹取点，和工具直接碰撞的顶点
			meshStiffness[tetVertIdx] = 0.0;
			break;
		}
		case 0: { //非碰撞点，根据顶点到工具的距离计算不同的刚度系数
			//float x = distance - 2*ballRadius;
			//float influence_r = 3*ballRadius;
			//if (x > influence_r)
			//	x = influence_r;
			//else if (x < 0)
			//	x = 0;
			//float t = 1 - x / influence_r;
			//meshStiffness[tetVertIdx] = t*t*t * maxStiffness;

			float k = 1 / (1 + exp(10 * distance - 5));
			meshStiffness[threadid] = k * maxStiffness;
			break;
		}
		default:
			break;
		}
	}
	else {   //如果没有和工具发生碰撞，布料不对四面体产生约束力
		meshStiffness[tetVertIdx] = 0.0;
	}
	//meshStiffness[tetVertIdx] = 0;
#ifdef OUTPUT_INFO
	//if (threadid == LOOK_THREAD)
		//printf("calculateRestStiffness isCollide:%d, stiffness:%f\n", isCollide[LOOK_THREAD], reststiffness[LOOK_THREAD]);
#endif
}

__global__ void calculateRestPosStiffnessWithMesh(float* ballPos, unsigned char* toolCollideFlag, float* positions, unsigned char* isCollide, float* meshStiffness, int toolNum, int vertexNum) 
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
//根据与工具的距离或者碰撞信息，计算restpos刚度系数
	if (threadid >= vertexNum) return;

	float maxStiffness = 50000;
	bool flag = false;
	for (int i = 0; i < toolNum; i++)
	{
		if (toolCollideFlag[i] > 0)
			flag = true;
	}
	float distance = 1e9 + 7;  //计算顶点到两个工具最近的距离
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
	if (flag) {  //与工具发生碰撞
		switch (isCollide[threadid])
		{
		case 1: {  //按压点，和工具直接发生碰撞的顶点
			meshStiffness[threadid] = maxStiffness;				
//ifdef OUTPUT_INFO
			if (threadid == LOOK_THREAD)
				printf("tet reststiffness with mesh in thread:%d: %f\n", meshStiffness[LOOK_THREAD]);
#//endif
			break;
		}
		case 2: { //夹取点，和工具直接碰撞的顶点
			meshStiffness[threadid] = 0.0;
			break;
		}
		case 0: { //非碰撞点，根据顶点到工具的距离计算不同的刚度系数
			float k = 1 / (1 + exp(10 * distance - 5));
			meshStiffness[threadid] = k * maxStiffness;
		}
		default:
			break;
		}
	}
	else {   //如果没有和工具发生碰撞，布料不对四面体产生约束力
		meshStiffness[threadid] = 0.0;
	}
#ifdef OUTPUT_INFO
	//if (threadid == LOOK_THREAD)
		//printf("calculateRestStiffness isCollide:%d, stiffness:%f\n", isCollide[LOOK_THREAD], reststiffness[LOOK_THREAD]);
#endif
}

__global__ void calculateRestPos(float* positions, float* rest_positions, float* force, float* collisionDiag, float* restStiffness, int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > vertexNum) return;

	//计算受力
	float tempx = rest_positions[3 * threadid + 0] - positions[3 * threadid + 0];
	float tempy = rest_positions[3 * threadid + 1] - positions[3 * threadid + 1];
	float tempz = rest_positions[3 * threadid + 2] - positions[3 * threadid + 2];

	atomicAdd(force + threadid * 3 + 0, tempx * restStiffness[threadid]);
	atomicAdd(force + threadid * 3 + 1, tempy * restStiffness[threadid]);
	atomicAdd(force + threadid * 3 + 2, tempz * restStiffness[threadid]);

	atomicAdd(collisionDiag + threadid * 3 + 0, restStiffness[threadid]);
	atomicAdd(collisionDiag + threadid * 3 + 1, restStiffness[threadid]);
	atomicAdd(collisionDiag + threadid * 3 + 2, restStiffness[threadid]);
}

__global__ void calculateRestPos_part(float* positions, float* rest_positions, float* force, float* collisionDiag, float* restStiffness, 
	int* sortedTetVertIndices, int offset, int activeElement)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > activeElement) return;

	int tetVertIdx = sortedTetVertIndices[threadid+offset];

	//计算受力
	float tempx = rest_positions[3 * tetVertIdx + 0] - positions[3 * tetVertIdx + 0];
	float tempy = rest_positions[3 * tetVertIdx + 1] - positions[3 * tetVertIdx + 1];
	float tempz = rest_positions[3 * tetVertIdx + 2] - positions[3 * tetVertIdx + 2];

	atomicAdd(force + tetVertIdx * 3 + 0, tempx * restStiffness[tetVertIdx]);
	atomicAdd(force + tetVertIdx * 3 + 1, tempy * restStiffness[tetVertIdx]);
	atomicAdd(force + tetVertIdx * 3 + 2, tempz * restStiffness[tetVertIdx]);

	atomicAdd(collisionDiag + tetVertIdx * 3 + 0, restStiffness[tetVertIdx]);
	atomicAdd(collisionDiag + tetVertIdx * 3 + 1, restStiffness[tetVertIdx]);
	atomicAdd(collisionDiag + tetVertIdx * 3 + 2, restStiffness[tetVertIdx]);
}

__device__ void MatrixSubstract_3_D(float* A, float* B, float* R)						//R=A-B
{
	for (int i = 0; i < 9; i++)	R[i] = A[i] - B[i];
}
__device__ void MatrixProduct_3_D(const float* A, const float* B, float* R)				//R=A*B
{
	R[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
	R[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
	R[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
	R[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
	R[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
	R[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
	R[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
	R[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
	R[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}
__device__ void MatrixProduct_D(float* A, float* B, float* R, int nx, int ny, int nz)	//R=A*B
{
	memset(R, 0, sizeof(float) * nx * nz);
	for (int i = 0; i < nx; i++)
		for (int j = 0; j < nz; j++)
			for (int k = 0; k < ny; k++)
				R[i * nz + j] += A[i * ny + k] * B[k * nz + j];
}

__global__ void calculateIF(float* positions, int* tetIndex,
	float* tetInvD3x3, float* tetInvD3x4,
	float* force, float* tetVolumn, bool* active,
	int tetNum, float* volumnStiffness) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= tetNum) return;

	if (!active[threadid]) {
		return;
	}

	//获取当前四面体的变形系数
	//volumnStiffness = tetStiffness_d[threadid];

	//计算每个四面体初始化的shape矩阵的逆
	int vIndex0 = tetIndex[threadid * 4 + 0];
	int vIndex1 = tetIndex[threadid * 4 + 1];
	int vIndex2 = tetIndex[threadid * 4 + 2];
	int vIndex3 = tetIndex[threadid * 4 + 3];

	//先计算shape矩阵
	float D[9];
	D[0] = positions[vIndex1 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[1] = positions[vIndex2 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[2] = positions[vIndex3 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[3] = positions[vIndex1 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[4] = positions[vIndex2 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[5] = positions[vIndex3 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[6] = positions[vIndex1 * 3 + 2] - positions[vIndex0 * 3 + 2];
	D[7] = positions[vIndex2 * 3 + 2] - positions[vIndex0 * 3 + 2];
	D[8] = positions[vIndex3 * 3 + 2] - positions[vIndex0 * 3 + 2];

	//计算形变梯度F
	float F[9];
	float* B = &tetInvD3x3[threadid * 9];
	MatrixProduct_3_D(D, &tetInvD3x3[threadid * 9], F);

	//从F中分解出R（直接搬运，这个算法太复杂了）
	float R[9];
	GetRotation_D((float(*)[3])F, (float(*)[3])R);//转化为数组指针，即对应二维数组的形参要求

	MatrixSubstract_3_D(R, F, R);
	//for (int i = 0; i < 9; i++)	
	//	R[i] = R[i] - F[i];

	float temp[12];
	memset(temp, 0, sizeof(float) *12);
	MatrixProduct_D(R, &tetInvD3x4[threadid * 12], temp, 3, 3, 4);

	//对应的四个点的xyz分量
	//这里应该需要原子操作
	atomicAdd(force + vIndex0 * 3 + 0, temp[0] * tetVolumn[threadid] * volumnStiffness[threadid]);
	atomicAdd(force + vIndex0 * 3 + 1, temp[4] * tetVolumn[threadid] * volumnStiffness[threadid]);
	atomicAdd(force + vIndex0 * 3 + 2, temp[8] * tetVolumn[threadid] * volumnStiffness[threadid]);

	atomicAdd(force + vIndex1 * 3 + 0, temp[1] * tetVolumn[threadid] * volumnStiffness[threadid]);
	atomicAdd(force + vIndex1 * 3 + 1, temp[5] * tetVolumn[threadid] * volumnStiffness[threadid]);
	atomicAdd(force + vIndex1 * 3 + 2, temp[9] * tetVolumn[threadid] * volumnStiffness[threadid]);

	atomicAdd(force + vIndex2 * 3 + 0, temp[2] * tetVolumn[threadid] * volumnStiffness[threadid]);
	atomicAdd(force + vIndex2 * 3 + 1, temp[6] * tetVolumn[threadid] * volumnStiffness[threadid]);
	atomicAdd(force + vIndex2 * 3 + 2, temp[10] * tetVolumn[threadid] * volumnStiffness[threadid]);

	atomicAdd(force + vIndex3 * 3 + 0, temp[3] * tetVolumn[threadid] * volumnStiffness[threadid]);
	atomicAdd(force + vIndex3 * 3 + 1, temp[7] * tetVolumn[threadid] * volumnStiffness[threadid]);
	atomicAdd(force + vIndex3 * 3 + 2, temp[11] * tetVolumn[threadid] * volumnStiffness[threadid]);
#ifdef OUTPUT_INFO
	if (vIndex0 == LOOK_THREAD)
		printf("calculateIF tetVertForce_d in calculateIF: %f %f %f\n", force[vIndex0 * 3 + 0], force[vIndex0 * 3 + 1], force[vIndex0 * 3 + 2]);
	//if (vIndex0 == 0)
	//{
	//	printf("calculateIF threadid: %d v0_temp[%f %f %f]\n",
	//		threadid, temp[0], temp[4], temp[8]);
	//	if (isnan(temp[0]) || isnan(temp[4]) || isnan(temp[8]))
	//	{
	//		unsigned int t = threadid * 12;
	//		printf("threadid: %d\n nan occured in v0, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			threadid,
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//		printf("threadid: %d\ntetInvD3x4[ %f %f %f %f \n\t%f %f %f %f\n\t %f %f %f %f]\n",
	//			threadid,
	//			tetInvD3x4[t + 0], tetInvD3x4[t + 1], tetInvD3x4[t + 2], tetInvD3x4[t + 3],
	//			tetInvD3x4[t + 4], tetInvD3x4[t + 5], tetInvD3x4[t + 6], tetInvD3x4[t + 7],
	//			tetInvD3x4[t + 8], tetInvD3x4[t + 9], tetInvD3x4[t + 10], tetInvD3x4[t + 11]);
	//	}
	//}
	//if (vIndex1 == 0)
	//{
	//	printf("calculateIF v1_temp[%f %f %f]\n",
	//		temp[1], temp[5], temp[9]);
	//	if (isnan(temp[1]) || isnan(temp[5]) || isnan(temp[9]))
	//	{
	//		printf("nan occured in v1, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
	//if (vIndex2 == 0)
	//{
	//	printf("calculateIF v2_temp[%f %f %f]\n",
	//		temp[2], temp[6], temp[10]);
	//	if (isnan(temp[2]) || isnan(temp[6]) || isnan(temp[10]))
	//	{
	//		printf("nan occured in v2, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
	//if (vIndex3 == 0)
	//{
	//	printf("calculateIF v3_temp[%f %f %f]\n",
	//		temp[3], temp[7], temp[11]);
	//	if (isnan(temp[3]) || isnan(temp[7]) || isnan(temp[11]))
	//	{
	//		printf("nan occured in v3, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
#endif
}

__global__ void calculateIF_part(float* positions, int* tetIndex,
	float* tetInvD3x3, float* tetInvD3x4,
	float* force, float* tetVolumn, float* volumnStiffness, 
	int * sortedTetIdx, int offset, int activeElementNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	//获取当前四面体的变形系数
	//volumnStiffness = tetStiffness_d[threadid];
#ifdef OUTPUT_INFO
	if (threadid == 0)
		printf("calculateIF startIdx:%d, activeTetNum:%d\n", threadid + offset, activeElementNum);
#endif
	unsigned int tetIdx = sortedTetIdx[threadid+offset];
	//计算每个四面体初始化的shape矩阵的逆
	int vIndex0 = tetIndex[tetIdx * 4 + 0];
	int vIndex1 = tetIndex[tetIdx * 4 + 1];
	int vIndex2 = tetIndex[tetIdx * 4 + 2];
	int vIndex3 = tetIndex[tetIdx * 4 + 3];

	//先计算shape矩阵
	float D[9];
	D[0] = positions[vIndex1 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[1] = positions[vIndex2 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[2] = positions[vIndex3 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[3] = positions[vIndex1 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[4] = positions[vIndex2 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[5] = positions[vIndex3 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[6] = positions[vIndex1 * 3 + 2] - positions[vIndex0 * 3 + 2];
	D[7] = positions[vIndex2 * 3 + 2] - positions[vIndex0 * 3 + 2];
	D[8] = positions[vIndex3 * 3 + 2] - positions[vIndex0 * 3 + 2];

	//计算形变梯度F
	float F[9];
	float* B = &tetInvD3x3[tetIdx * 9];
	MatrixProduct_3_D(D, &tetInvD3x3[tetIdx * 9], F);

	//从F中分解出R（直接搬运，这个算法太复杂了）
	float R[9];
	GetRotation_D((float(*)[3])F, (float(*)[3])R);//转化为数组指针，即对应二维数组的形参要求

	MatrixSubstract_3_D(R, F, R);
	//for (int i = 0; i < 9; i++)	
	//	R[i] = R[i] - F[i];

	float temp[12];
	memset(temp, 0, sizeof(float) * 12);
	MatrixProduct_D(R, &tetInvD3x4[tetIdx * 12], temp, 3, 3, 4);

	//对应的四个点的xyz分量
	//这里应该需要原子操作
	atomicAdd(force + vIndex0 * 3 + 0, temp[0] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex0 * 3 + 1, temp[4] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex0 * 3 + 2, temp[8] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);

	atomicAdd(force + vIndex1 * 3 + 0, temp[1] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex1 * 3 + 1, temp[5] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex1 * 3 + 2, temp[9] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);

	atomicAdd(force + vIndex2 * 3 + 0, temp[2] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex2 * 3 + 1, temp[6] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex2 * 3 + 2, temp[10] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);

	atomicAdd(force + vIndex3 * 3 + 0, temp[3] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex3 * 3 + 1, temp[7] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex3 * 3 + 2, temp[11] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
#ifdef OUTPUT_INFO
	if (vIndex0 == LOOK_THREAD)
		printf("calculateIF tetVertForce_d in calculateIF: %f %f %f\n", force[vIndex0 * 3 + 0], force[vIndex0 * 3 + 1], force[vIndex0 * 3 + 2]);
	//if (vIndex0 == 0)
	//{
	//	printf("calculateIF threadid: %d v0_temp[%f %f %f]\n",
	//		threadid, temp[0], temp[4], temp[8]);
	//	if (isnan(temp[0]) || isnan(temp[4]) || isnan(temp[8]))
	//	{
	//		unsigned int t = threadid * 12;
	//		printf("threadid: %d\n nan occured in v0, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			threadid,
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//		printf("threadid: %d\ntetInvD3x4[ %f %f %f %f \n\t%f %f %f %f\n\t %f %f %f %f]\n",
	//			threadid,
	//			tetInvD3x4[t + 0], tetInvD3x4[t + 1], tetInvD3x4[t + 2], tetInvD3x4[t + 3],
	//			tetInvD3x4[t + 4], tetInvD3x4[t + 5], tetInvD3x4[t + 6], tetInvD3x4[t + 7],
	//			tetInvD3x4[t + 8], tetInvD3x4[t + 9], tetInvD3x4[t + 10], tetInvD3x4[t + 11]);
	//	}
	//}
	//if (vIndex1 == 0)
	//{
	//	printf("calculateIF v1_temp[%f %f %f]\n",
	//		temp[1], temp[5], temp[9]);
	//	if (isnan(temp[1]) || isnan(temp[5]) || isnan(temp[9]))
	//	{
	//		printf("nan occured in v1, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
	//if (vIndex2 == 0)
	//{
	//	printf("calculateIF v2_temp[%f %f %f]\n",
	//		temp[2], temp[6], temp[10]);
	//	if (isnan(temp[2]) || isnan(temp[6]) || isnan(temp[10]))
	//	{
	//		printf("nan occured in v2, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
	//if (vIndex3 == 0)
	//{
	//	printf("calculateIF v3_temp[%f %f %f]\n",
	//		temp[3], temp[7], temp[11]);
	//	if (isnan(temp[3]) || isnan(temp[7]) || isnan(temp[11]))
	//	{
	//		printf("nan occured in v3, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
#endif
}
int runcalculateRestPosForceWithMeshPos(float toolRadius)
{
	int threadNum = 512;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	calculateRestPosStiffnessWithMesh << <blockNum, threadNum >> > (
		toolPositionAndDirection_d, toolCollideFlag_d,
		tetVertPos_d, tetVertisCollide_d, 
		tetVertfromTriStiffness_d, cylinderNum_d, tetVertNum_d);

	calculateRestPos << <blockNum, threadNum >> > (
		tetVertPos_d, tetVertRestPos_d,
		tetVertCollisionForce_d, tetVertCollisionDiag_d,
		tetVertfromTriStiffness_d, tetVertNum_d);
	cudaDeviceSynchronize();
	printCudaError("runcalculateRestPosForceWithMeshPos");
	return 0;
}

__global__ void calculateRestPosForceWithMeshPos(
	float* positions, int* skeletonMesh,
	float* force, float* collisionDiag,
	float* meshPositions, unsigned char* isCollide,
	float* meshStiffness, int vertexNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	int tri_Idx = skeletonMesh[threadid];
	if (tri_Idx == -1) return; // 没有与四面体顶点绑定的表面布料点不参与计算。
	if (isCollide[threadid] == 0) return;

	float deltaPos[3];
	deltaPos[0] = meshPositions[3 * tri_Idx + 0] - positions[3 * threadid + 0];
	deltaPos[1] = meshPositions[3 * tri_Idx + 1] - positions[3 * threadid + 1];
	deltaPos[2] = meshPositions[3 * tri_Idx + 2] - positions[3 * threadid + 2];
	float d = sqrt(deltaPos[0] * deltaPos[0] + deltaPos[1] * deltaPos[1] + deltaPos[2] * deltaPos[2]);
#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		printf("calculateRestPosForceWithMeshPos thread:%d deltaPos[%f %f %f]\n", threadid, deltaPos[0], deltaPos[1], deltaPos[2]);
	}
#endif
	if (d < 1e-9)
		return;
	float dir[3] = { deltaPos[0] / d, deltaPos[1] / d,deltaPos[2] / d };
	

	float forcex = deltaPos[0] * meshStiffness[threadid];
	float forcey = deltaPos[1] * meshStiffness[threadid];
	float forcez = deltaPos[2] * meshStiffness[threadid];
	force[threadid * 3 + 0] += forcex;
	force[threadid * 3 + 1] += forcey;
	force[threadid * 3 + 2] += forcez;

	collisionDiag[threadid * 3 + 0] += meshStiffness[threadid];
	collisionDiag[threadid * 3 + 1] += meshStiffness[threadid];
	collisionDiag[threadid * 3 + 2] += meshStiffness[threadid];

}

//计算position
__global__ void calculatePOS(float* positions, float* force, float* fixed, float* mass,
	float* next_positions, float* prev_positions, float* old_positions,
	float* volumnDiag, float* collisionDiag, float* collisionForce,
	int vertexNum, float dt, float omega)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	float diagConstant = (mass[threadid] + fixed[threadid]) / (dt * dt);
	float forceLen = sqrt(force[indexX] * force[indexX] + force[indexY] * force[indexY] + force[indexZ] * force[indexZ]);

	//计算每个点的shape match产生的约束部分，因为之前是按照每个四面体计算的，现在要摊到每个顶点上
	float elementX = force[indexX] + collisionForce[indexX];
	float elementY = force[indexY] + collisionForce[indexY];
	float elementZ = force[indexZ] + collisionForce[indexZ];

	//if (threadid == LOOK_THREAD)
	//{
	//	printf("calculatePOS force[%f,%f,%f] collisionForce[%f,%f,%f]\n",
	//		force[indexX], force[indexY], force[indexZ],
	//		collisionForce[indexX], collisionForce[indexY], collisionForce[indexZ]);
	//}
#ifdef OUTPUT_INFO


	if (threadid == LOOK_THREAD)
	{
		printf("calculatePOS constantDiag:%f volumeDiag:%f collisionDiag:[%f, %f %f]\n", diagConstant, volumnDiag[threadid], collisionDiag[indexX], collisionDiag[indexY], collisionDiag[indexZ]);
	}
	//if (collisionDiag[indexX] > 0)
	//{
	//	printf("threadid:%d collisionDiag[%f %f %f]\n", threadid, collisionDiag[indexX], collisionDiag[indexY], collisionDiag[indexZ]);
	//}
#endif
	//相当于先按重力运动，每次再在收重力的效果上再修正
	next_positions[indexX] = (diagConstant * (old_positions[indexX] - positions[indexX]) + elementX) / (volumnDiag[threadid] + collisionDiag[indexX] + diagConstant) + positions[indexX];
	next_positions[indexY] = (diagConstant * (old_positions[indexY] - positions[indexY]) + elementY) / (volumnDiag[threadid] + collisionDiag[indexY] + diagConstant) + positions[indexY];
	next_positions[indexZ] = (diagConstant * (old_positions[indexZ] - positions[indexZ]) + elementZ) / (volumnDiag[threadid] + collisionDiag[indexZ] + diagConstant) + positions[indexZ];


	//if (threadid== 6000) {
	//	printf("*********************\n");
	//	printf("%d:体积对角元素:%f,%f,%f\n", threadid, elementX, elementY, elementZ);
	//}
	//under-relaxation 和 切比雪夫迭代
	next_positions[indexX] = (next_positions[indexX] - positions[indexX]) * 0.6 + positions[indexX];
	next_positions[indexY] = (next_positions[indexY] - positions[indexY]) * 0.6 + positions[indexY];
	next_positions[indexZ] = (next_positions[indexZ] - positions[indexZ]) * 0.6 + positions[indexZ];

	// omega定义：omega = 4 / (4 - rho*rho*omega);
	next_positions[indexX] = omega * (next_positions[indexX] - prev_positions[indexX]) + prev_positions[indexX];
	next_positions[indexY] = omega * (next_positions[indexY] - prev_positions[indexY]) + prev_positions[indexY];
	next_positions[indexZ] = omega * (next_positions[indexZ] - prev_positions[indexZ]) + prev_positions[indexZ];

	prev_positions[indexX] = positions[indexX];
	prev_positions[indexY] = positions[indexY];
	prev_positions[indexZ] = positions[indexZ];

	positions[indexX] = next_positions[indexX];
	positions[indexY] = next_positions[indexY];
	positions[indexZ] = next_positions[indexZ];

	float deltax = positions[indexX] - prev_positions[indexX];
	float deltay = positions[indexY] - prev_positions[indexY];
	float deltaz = positions[indexZ] - prev_positions[indexZ];

	//if (threadid == LOOK_THREAD)
	//{
	//	printf("point delta x:%f %f %f\n", deltax, deltay, deltaz);
	//}
	//if (isnan(positions[indexX]))
	//{
	//	printf("nan occured in threadid %d\n", threadid);
	//}
	//if (isnan(positions[indexZ]))
	//{
	//	printf("nan occured in threadid %d\n", threadid);
	//}
	//if (isnan(positions[indexY]))
	//{
	//	printf("nan occured in threadid %d\n", threadid);
	//}

	if (forceLen > 2)
	{
		float movement = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
		//printf("%d-tetVertForce_d in calculatePOS:%f %f %f\nmovement:%f constantDiag:%f\n", threadid, force[indexX], force[indexY], force[indexZ], movement, diagConstant);

	}

	//float movement = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
	//if(movement>1e-5)
	//	printf("thread %d movement: %f\n", threadid, movement);
}

//计算更新位置
int runcalculatePOS(float omega, float dt) {
	int  threadNum = 512;

	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
	//并行计算
	calculatePOS << <blockNum, threadNum >> > (tetVertPos_d, tetVertForce_d,
		tetVertFixed_d, tetVertMass_d,
		tetVertPos_next_d, tetVertPos_prev_d, tetVertPos_old_d,
		tetVolumeDiag_d, tetVertCollisionDiag_d, tetVertCollisionForce_d,
		tetVertNum_d, dt, omega);
	cudaDeviceSynchronize();
	printCudaError("runcalculatePOS");
	return 0;
}

//计算position
__global__ void calculatePOS(float* positions, float* force, float* fixed, float* mass,
	float* next_positions, float* prev_positions, float* old_positions,
	float* volumnDiag, float* collisionDiag, float* collisionForce,
	int* sortedIndices, int offset, int activeElementNum, float dt, float omega)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;
	int vertIdx = sortedIndices[offset + threadid];

	if (vertIdx == GRABED_TETIDX)
		return;

	int indexX = vertIdx * 3 + 0;
	int indexY = vertIdx * 3 + 1;
	int indexZ = vertIdx * 3 + 2;

	float diagConstant = (mass[vertIdx] + fixed[vertIdx]) / (dt * dt);
	float forceLen = sqrt(force[indexX] * force[indexX] + force[indexY] * force[indexY] + force[indexZ] * force[indexZ]);

	float elementX = force[indexX] + collisionForce[indexX];
	float elementY = force[indexY] + collisionForce[indexY];
	float elementZ = force[indexZ] + collisionForce[indexZ];

//#ifdef OUTPUT_INFO
	//if (threadid == LOOK_THREAD)
	//{
	//	printf("calculatePOS force[%f,%f,%f] collisionForce[%f,%f,%f]\n",
	//		force[indexX], force[indexY], force[indexZ],
	//		collisionForce[indexX], collisionForce[indexY], collisionForce[indexZ]);
	//}

	//if (threadid == LOOK_THREAD)
	//{
	//	printf("calculatePOS mass:%f constantDiag:%f volumeDiag:%f collisionDiag:[%f, %f %f]\n", mass[vertIdx], diagConstant, volumnDiag[vertIdx], collisionDiag[indexX], collisionDiag[indexY], collisionDiag[indexZ]);
	//}
	//if (collisionDiag[indexX] > 0)
	//{
	//	printf("vertIdx:%d collisionDiag[%f %f %f]\n", vertIdx, collisionDiag[indexX], collisionDiag[indexY], collisionDiag[indexZ]);
	//}
//#endif
// 
// 
	//相当于先按重力运动，每次再在收重力的效果上再修正
	next_positions[indexX] = (diagConstant * (old_positions[indexX] - positions[indexX]) + elementX) / (volumnDiag[vertIdx] + collisionDiag[indexX] + diagConstant) + positions[indexX];
	next_positions[indexY] = (diagConstant * (old_positions[indexY] - positions[indexY]) + elementY) / (volumnDiag[vertIdx] + collisionDiag[indexY] + diagConstant) + positions[indexY];
	next_positions[indexZ] = (diagConstant * (old_positions[indexZ] - positions[indexZ]) + elementZ) / (volumnDiag[vertIdx] + collisionDiag[indexZ] + diagConstant) + positions[indexZ];


	//if (vertIdx== 6000) {
	//	printf("*********************\n");
	//	printf("%d:体积对角元素:%f,%f,%f\n", vertIdx, elementX, elementY, elementZ);
	//}
	//under-relaxation 和 切比雪夫迭代
	next_positions[indexX] = (next_positions[indexX] - positions[indexX]) * 0.6 + positions[indexX];
	next_positions[indexY] = (next_positions[indexY] - positions[indexY]) * 0.6 + positions[indexY];
	next_positions[indexZ] = (next_positions[indexZ] - positions[indexZ]) * 0.6 + positions[indexZ];

	// omega定义：omega = 4 / (4 - rho*rho*omega);
	next_positions[indexX] = omega * (next_positions[indexX] - prev_positions[indexX]) + prev_positions[indexX];
	next_positions[indexY] = omega * (next_positions[indexY] - prev_positions[indexY]) + prev_positions[indexY];
	next_positions[indexZ] = omega * (next_positions[indexZ] - prev_positions[indexZ]) + prev_positions[indexZ];

	prev_positions[indexX] = positions[indexX];
	prev_positions[indexY] = positions[indexY];
	prev_positions[indexZ] = positions[indexZ];

	positions[indexX] = next_positions[indexX];
	positions[indexY] = next_positions[indexY];
	positions[indexZ] = next_positions[indexZ];

	float deltax = positions[indexX] - prev_positions[indexX];
	float deltay = positions[indexY] - prev_positions[indexY];
	float deltaz = positions[indexZ] - prev_positions[indexZ];

	//if (isnan(positions[indexX]))
	//{
	//	printf("nan occured in vertIdx %d\n", vertIdx);
	//}
	//if (isnan(positions[indexZ]))
	//{
	//	printf("nan occured in vertIdx %d\n", vertIdx);
	//}
	//if (isnan(positions[indexY]))
	//{
	//	printf("nan occured in vertIdx %d\n", vertIdx);
	//}

	if (forceLen > 2)
	{
		float movement = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
		//printf("%d-tetVertForce_d in calculatePOS:%f %f %f\nmovement:%f constantDiag:%f\n", vertIdx, force[indexX], force[indexY], force[indexZ], movement, diagConstant);

	}

	//float movement = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
	//if(movement>1e-5)
	//	printf("thread %d movement: %f\n", vertIdx, movement);
}

int runcalculateV(float dt) {
	int  threadNum = 512;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
	//并行计算
	calculateV << <blockNum, threadNum >> > (tetVertPos_d, tetVertVelocity_d, tetVertPos_last_d, tetVertNum_d, dt);

	cudaDeviceSynchronize();
	printCudaError("runcalculateV");
	return 0;

}

//计算速度更新
__global__ void calculateV(float* positions, float* velocity, float* last_positions, int vertexNum, float dt) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	velocity[threadid * 3 + 0] = (positions[threadid * 3 + 0] - last_positions[threadid * 3 + 0]) / dt;
	velocity[threadid * 3 + 1] = (positions[threadid * 3 + 1] - last_positions[threadid * 3 + 1]) / dt;
	velocity[threadid * 3 + 2] = (positions[threadid * 3 + 2] - last_positions[threadid * 3 + 2]) / dt;
}

__global__ void calculateV(float* positions, float* velocity, float* last_positions, int* sortedIndices, int offset, int activeElementNum, float dt) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	int vertIdx = sortedIndices[threadid + offset];
	velocity[vertIdx * 3 + 0] = (positions[vertIdx * 3 + 0] - last_positions[vertIdx * 3 + 0]) / dt;
	velocity[vertIdx * 3 + 1] = (positions[vertIdx * 3 + 1] - last_positions[vertIdx * 3 + 1]) / dt;
	velocity[vertIdx * 3 + 2] = (positions[vertIdx * 3 + 2] - last_positions[vertIdx * 3 + 2]) / dt;
}
int runUpdateInnerTetVertDDir()
{
	int threadNum = 512;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	updateInnerTetVertDirectDirection << <blockNum, threadNum >> > (tetVertPos_d,
		tetVertBindingTetVertIndices_d, tetVertBindingTetVertWeight_d,
		tetVertNonPenetrationDir_d, tetVertNum_d);
	// no need to sychronize
	printCudaError("updateInnerTetVertDDir");
	return 0;
}
int runUpdateSurfaceTetVertDDir()
{
	int threadNum = 512;
	int blockNum = (triVertOrgNum_d + threadNum - 1) / threadNum;
	updateSurfaceTetVertDirectDirection << <blockNum, threadNum >> > (
		onSurfaceTetVertIndices_d,
		tetVert2TriVertMapping_d, triVertNorm_d,
		tetVertNonPenetrationDir_d,
		tetVertPos_d, triVertPos_d,
		triVertOrgNum_d);
	printCudaError("updateSurfaceTetVertDDir");
	return 0;
}
int runNormalizeDDir()
{
	int threadNum = 512;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
	normalizeDDir << <blockNum, threadNum >> > (tetVertNonPenetrationDir_d, tetVertNum_d);
	printCudaError("NormalizeTetVertDDir");
	return 0;
}
int runUpdateTetVertDirectDirection()
{
	runUpdateInnerTetVertDDir();
	runUpdateSurfaceTetVertDDir();
	cudaDeviceSynchronize();

	runNormalizeDDir();
	cudaDeviceSynchronize();
	printCudaError("UpdateTetVertDirectDirection");
	return 0;
}

__global__ void normalizeDDir(float* dDir, int pointNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= pointNum) return;
	
	int idxX = threadid * 3 + 0;
	int idxY = threadid * 3 + 1;
	int idxZ = threadid * 3 + 2;
	float l = sqrt(dDir[idxX] * dDir[idxX] + dDir[idxY] * dDir[idxY] + dDir[idxZ] * dDir[idxZ]);
	if(l<1e-7)
	{
		//printf("threadid %d, dDirLen=0\n", threadid);
		dDir[idxX] = 1;
		dDir[idxY] = 0;
		dDir[idxZ] = 0;
	}
	else
	{
		dDir[idxX] /= l;
		dDir[idxY] /= l;
		dDir[idxZ] /= l;
	}
	//if (threadid < 10)
	//{
	//	printf("threadid %d, DDir [%f %f %f]\n", threadid, dDir[idxX], dDir[idxY], dDir[idxZ]);
	//}
}
__device__ void GetRotation_D(float F[3][3], float R[3][3])
{
	float C[3][3];
	memset(&C[0][0], 0, sizeof(float) * 9);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				C[i][j] += F[k][i] * F[k][j];

	float C2[3][3];
	memset(&C2[0][0], 0, sizeof(float) * 9);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				C2[i][j] += C[i][k] * C[j][k];

	float det = F[0][0] * F[1][1] * F[2][2] +
		F[0][1] * F[1][2] * F[2][0] +
		F[1][0] * F[2][1] * F[0][2] -
		F[0][2] * F[1][1] * F[2][0] -
		F[0][1] * F[1][0] * F[2][2] -
		F[0][0] * F[1][2] * F[2][1];

	float I_c = C[0][0] + C[1][1] + C[2][2];
	float I_c2 = I_c * I_c;
	float II_c = 0.5 * (I_c2 - C2[0][0] - C2[1][1] - C2[2][2]);
	float III_c = det * det;
	float k = I_c2 - 3 * II_c;

	float inv_U[3][3];
	if (k < 1e-10f)
	{
		float inv_lambda = 1 / sqrt(I_c / 3);
		memset(inv_U, 0, sizeof(float) * 9);
		inv_U[0][0] = inv_lambda;
		inv_U[1][1] = inv_lambda;
		inv_U[2][2] = inv_lambda;
	}
	else
	{
		float l = I_c * (I_c * I_c - 4.5 * II_c) + 13.5 * III_c;
		float k_root = sqrt(k);
		float value = l / (k * k_root);
		if (value < -1.0) value = -1.0;
		if (value > 1.0) value = 1.0;
		float phi = acos(value);
		float lambda2 = (I_c + 2 * k_root * cos(phi / 3)) / 3.0;
		float lambda = sqrt(lambda2);

		float III_u = sqrt(III_c);
		if (det < 0)   III_u = -III_u;
		float I_u = lambda + sqrt(-lambda2 + I_c + 2 * III_u / lambda);
		float II_u = (I_u * I_u - I_c) * 0.5;

		float U[3][3];
		float inv_rate, factor;

		inv_rate = 1 / (I_u * II_u - III_u);
		factor = I_u * III_u * inv_rate;

		memset(U, 0, sizeof(float) * 9);
		U[0][0] = factor;
		U[1][1] = factor;
		U[2][2] = factor;

		factor = (I_u * I_u - II_u) * inv_rate;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				U[i][j] += factor * C[i][j] - inv_rate * C2[i][j];

		inv_rate = 1 / III_u;
		factor = II_u * inv_rate;
		memset(inv_U, 0, sizeof(float) * 9);
		inv_U[0][0] = factor;
		inv_U[1][1] = factor;
		inv_U[2][2] = factor;





		factor = -I_u * inv_rate;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				inv_U[i][j] += factor * U[i][j] + inv_rate * C[i][j];
	}




	memset(&R[0][0], 0, sizeof(float) * 9);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				R[i][j] += F[i][k] * inv_U[k][j];

	//检查，避免invert
	if (det <= 0) {
		R[0][0] = 1;
		R[0][1] = 0;
		R[0][2] = 0;
		R[1][0] = 0;
		R[1][1] = 1;
		R[1][2] = 0;
		R[2][0] = 0;
		R[2][1] = 0;
		R[2][2] = 1;
	}
}
