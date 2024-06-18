#include "gpuvar.h"
#include "gpufun.h"

// triangle
unsigned int* triIndex_d;		  // 三角网格上三角形三个顶点对应的顶点下标，3*triNum_d

__device__ float l2len(float* v0, float* v1)
{
	float d[3] = { v0[0] - v1[0],v0[1] - v1[1] ,v0[2] - v1[2] };
	return sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}
//更新mesh顶点法线
extern "C" int runUpdateMeshNormalMU() {

	int threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;
	//清除顶点法线信息，重新计算
	clearNormalMU << <blockNum, threadNum >> > (triVertNorm_d, triVertNormAccu_d, triVertNum_d);

	threadNum = 512;
	blockNum = (triNum_d + threadNum - 1) / threadNum;
	updateMeshNormalMU << <blockNum, threadNum >> > (triVertPos_d, triVertNorm_d, triVertNormAccu_d, triIndex_d, triNum_d);
	cudaDeviceSynchronize();
	printCudaError("updateMeshNormalMU");

	threadNum = 512;
	blockNum = (triVertNum_d + threadNum - 1) / threadNum;
	normalizeMeshtriVertNorm_debug << <blockNum, threadNum >> > (triVertNorm_d, triVertPos_d, triVertNormAccu_d, triVertNum_d);

	cudaDeviceSynchronize();
	printCudaError("normalizeMeshtriVertNorm");
	return 0;
}

//清除顶点法线信息，需要重新计算
__global__ void clearNormalMU(float* meshNormal, float* totAngle, int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	meshNormal[3 * threadid + 0] = 0.0f;
	meshNormal[3 * threadid + 1] = 0.0f;
	meshNormal[3 * threadid + 2] = 0.0f;

	totAngle[threadid] = 0.0f;
}

//根据顶点位置和三角面片计算面片索引
__global__ void updateMeshNormalMU(float* meshPosition, float* meshNormal, float* totAngle, unsigned int* meshTriangle, int meshTriangleNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= meshTriangleNum) return;

	//获取mesh三角形的三个顶点的索引
	unsigned int index0 = meshTriangle[threadid * 3 + 0];
	unsigned int index1 = meshTriangle[threadid * 3 + 1];
	unsigned int index2 = meshTriangle[threadid * 3 + 2];

	float vecAx = meshPosition[index1 * 3 + 0] - meshPosition[index0 * 3 + 0];
	float vecAy = meshPosition[index1 * 3 + 1] - meshPosition[index0 * 3 + 1];
	float vecAz = meshPosition[index1 * 3 + 2] - meshPosition[index0 * 3 + 2];

	float vecBx = meshPosition[index2 * 3 + 0] - meshPosition[index0 * 3 + 0];
	float vecBy = meshPosition[index2 * 3 + 1] - meshPosition[index0 * 3 + 1];
	float vecBz = meshPosition[index2 * 3 + 2] - meshPosition[index0 * 3 + 2];

	//叉乘计算三角形法线
	float crossX = vecAy * vecBz - vecBy * vecAz;
	float crossY = vecAz * vecBx - vecBz * vecAx;
	float crossZ = vecAx * vecBy - vecBx * vecAy;

	//法线单位化
	float product = crossX * crossX + crossY * crossY + crossZ * crossZ;
	product = sqrt(product);
	crossX /= product;
	crossY /= product;
	crossZ /= product;

	float len_A = sqrt(vecAx * vecAx + vecAy * vecAy + vecAz * vecAz);
	float len_B = sqrt(vecBx * vecBx + vecBy * vecBy + vecBz * vecBz);
	vecAx /= len_A; vecAy /= len_A; vecAz /= len_A;
	vecBx /= len_B; vecBy /= len_B; vecBz /= len_B;
	float angle0 = acos(vecAx * vecBx + vecAy * vecBy + vecAz * vecBz);
	//------------------------------------------------------------------
	vecAx = meshPosition[index0 * 3 + 0] - meshPosition[index1 * 3 + 0];
	vecAy = meshPosition[index0 * 3 + 1] - meshPosition[index1 * 3 + 1];
	vecAz = meshPosition[index0 * 3 + 2] - meshPosition[index1 * 3 + 2];

	vecBx = meshPosition[index2 * 3 + 0] - meshPosition[index1 * 3 + 0];
	vecBy = meshPosition[index2 * 3 + 1] - meshPosition[index1 * 3 + 1];
	vecBz = meshPosition[index2 * 3 + 2] - meshPosition[index1 * 3 + 2];
	len_A = sqrt(vecAx * vecAx + vecAy * vecAy + vecAz * vecAz);
	len_B = sqrt(vecBx * vecBx + vecBy * vecBy + vecBz * vecBz);
	vecAx /= len_A; vecAy /= len_A; vecAz /= len_A;
	vecBx /= len_B; vecBy /= len_B; vecBz /= len_B;
	float angle1 = acos(vecAx * vecBx + vecAy * vecBy + vecAz * vecBz);
	//------------------------------------------------------------------
	vecAx = meshPosition[index0 * 3 + 0] - meshPosition[index2 * 3 + 0];
	vecAy = meshPosition[index0 * 3 + 1] - meshPosition[index2 * 3 + 1];
	vecAz = meshPosition[index0 * 3 + 2] - meshPosition[index2 * 3 + 2];

	vecBx = meshPosition[index1 * 3 + 0] - meshPosition[index2 * 3 + 0];
	vecBy = meshPosition[index1 * 3 + 1] - meshPosition[index2 * 3 + 1];
	vecBz = meshPosition[index1 * 3 + 2] - meshPosition[index2 * 3 + 2];
	len_A = sqrt(vecAx * vecAx + vecAy * vecAy + vecAz * vecAz);
	len_B = sqrt(vecBx * vecBx + vecBy * vecBy + vecBz * vecBz);
	vecAx /= len_A; vecAy /= len_A; vecAz /= len_A;
	vecBx /= len_B; vecBy /= len_B; vecBz /= len_B;
	float angle2 = acos(vecAx * vecBx + vecAy * vecBy + vecAz * vecBz);
	//------------------------------------------------------------------
#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		printf("UpdateMeshNormalMU tri index: %d %d %d", index0, index1, index2);
		printf("p0[%f %f %f] p1[%f %f %f] p2[%f %f %f]\n",
			meshPosition[index0 * 3 + 0], meshPosition[index0 * 3 + 1], meshPosition[index0 * 3 + 2],
			meshPosition[index1 * 3 + 0], meshPosition[index1 * 3 + 1], meshPosition[index1 * 3 + 2],
			meshPosition[index2 * 3 + 0], meshPosition[index2 * 3 + 1], meshPosition[index2 * 3 + 2]);
	}
#endif
	//将向量累加到每个三角形面片的顶点上
	atomicAdd(meshNormal + index0 * 3 + 0, crossX * angle0);
	atomicAdd(meshNormal + index0 * 3 + 1, crossY * angle0);
	atomicAdd(meshNormal + index0 * 3 + 2, crossZ * angle0);
	atomicAdd(totAngle + index0, angle0);

	atomicAdd(meshNormal + index1 * 3 + 0, crossX * angle1);
	atomicAdd(meshNormal + index1 * 3 + 1, crossY * angle1);
	atomicAdd(meshNormal + index1 * 3 + 2, crossZ * angle1);
	atomicAdd(totAngle + index1, angle1);

	atomicAdd(meshNormal + index2 * 3 + 0, crossX * angle2);
	atomicAdd(meshNormal + index2 * 3 + 1, crossY * angle2);
	atomicAdd(meshNormal + index2 * 3 + 2, crossZ * angle2);
	atomicAdd(totAngle + index2, angle2);
}

//根据顶点位置和三角面片计算面片索引
__global__ void updateMeshNormalMU(float* meshPosition, float* meshNormal, 
	float* totAngle, unsigned int* meshTriangle, 
	int* sortedTriIndices, int offset, int activeElementNum) 
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	int triIdx = sortedTriIndices[offset + threadid];
	//获取mesh三角形的三个顶点的索引
	unsigned int index0 = meshTriangle[triIdx * 3 + 0];
	unsigned int index1 = meshTriangle[triIdx * 3 + 1];
	unsigned int index2 = meshTriangle[triIdx * 3 + 2];

	float vecAx = meshPosition[index1 * 3 + 0] - meshPosition[index0 * 3 + 0];
	float vecAy = meshPosition[index1 * 3 + 1] - meshPosition[index0 * 3 + 1];
	float vecAz = meshPosition[index1 * 3 + 2] - meshPosition[index0 * 3 + 2];

	float vecBx = meshPosition[index2 * 3 + 0] - meshPosition[index0 * 3 + 0];
	float vecBy = meshPosition[index2 * 3 + 1] - meshPosition[index0 * 3 + 1];
	float vecBz = meshPosition[index2 * 3 + 2] - meshPosition[index0 * 3 + 2];

	//叉乘计算三角形法线
	float crossX = vecAy * vecBz - vecBy * vecAz;
	float crossY = vecAz * vecBx - vecBz * vecAx;
	float crossZ = vecAx * vecBy - vecBx * vecAy;

	//法线单位化
	float product = crossX * crossX + crossY * crossY + crossZ * crossZ;
	product = sqrt(product);
	crossX /= product;
	crossY /= product;
	crossZ /= product;

	float len_A = sqrt(vecAx * vecAx + vecAy * vecAy + vecAz * vecAz);
	float len_B = sqrt(vecBx * vecBx + vecBy * vecBy + vecBz * vecBz);
	vecAx /= len_A; vecAy /= len_A; vecAz /= len_A;
	vecBx /= len_B; vecBy /= len_B; vecBz /= len_B;
	float angle0 = acos(vecAx * vecBx + vecAy * vecBy + vecAz * vecBz);
	//------------------------------------------------------------------
	vecAx = meshPosition[index0 * 3 + 0] - meshPosition[index1 * 3 + 0];
	vecAy = meshPosition[index0 * 3 + 1] - meshPosition[index1 * 3 + 1];
	vecAz = meshPosition[index0 * 3 + 2] - meshPosition[index1 * 3 + 2];

	vecBx = meshPosition[index2 * 3 + 0] - meshPosition[index1 * 3 + 0];
	vecBy = meshPosition[index2 * 3 + 1] - meshPosition[index1 * 3 + 1];
	vecBz = meshPosition[index2 * 3 + 2] - meshPosition[index1 * 3 + 2];
	len_A = sqrt(vecAx * vecAx + vecAy * vecAy + vecAz * vecAz);
	len_B = sqrt(vecBx * vecBx + vecBy * vecBy + vecBz * vecBz);
	vecAx /= len_A; vecAy /= len_A; vecAz /= len_A;
	vecBx /= len_B; vecBy /= len_B; vecBz /= len_B;
	float angle1 = acos(vecAx * vecBx + vecAy * vecBy + vecAz * vecBz);
	//------------------------------------------------------------------
	vecAx = meshPosition[index0 * 3 + 0] - meshPosition[index2 * 3 + 0];
	vecAy = meshPosition[index0 * 3 + 1] - meshPosition[index2 * 3 + 1];
	vecAz = meshPosition[index0 * 3 + 2] - meshPosition[index2 * 3 + 2];

	vecBx = meshPosition[index1 * 3 + 0] - meshPosition[index2 * 3 + 0];
	vecBy = meshPosition[index1 * 3 + 1] - meshPosition[index2 * 3 + 1];
	vecBz = meshPosition[index1 * 3 + 2] - meshPosition[index2 * 3 + 2];
	len_A = sqrt(vecAx * vecAx + vecAy * vecAy + vecAz * vecAz);
	len_B = sqrt(vecBx * vecBx + vecBy * vecBy + vecBz * vecBz);
	vecAx /= len_A; vecAy /= len_A; vecAz /= len_A;
	vecBx /= len_B; vecBy /= len_B; vecBz /= len_B;
	float angle2 = acos(vecAx * vecBx + vecAy * vecBy + vecAz * vecBz);
	//------------------------------------------------------------------
#ifdef OUTPUT_INFO
	if (triIdx == LOOK_THREAD)
	{
		printf("UpdateMeshNormalMU tri index: %d %d %d", index0, index1, index2);
		printf("p0[%f %f %f] p1[%f %f %f] p2[%f %f %f]\n",
			meshPosition[index0 * 3 + 0], meshPosition[index0 * 3 + 1], meshPosition[index0 * 3 + 2],
			meshPosition[index1 * 3 + 0], meshPosition[index1 * 3 + 1], meshPosition[index1 * 3 + 2],
			meshPosition[index2 * 3 + 0], meshPosition[index2 * 3 + 1], meshPosition[index2 * 3 + 2]);
	}
#endif
	//将向量累加到每个三角形面片的顶点上
	atomicAdd(meshNormal + index0 * 3 + 0, crossX * angle0);
	atomicAdd(meshNormal + index0 * 3 + 1, crossY * angle0);
	atomicAdd(meshNormal + index0 * 3 + 2, crossZ * angle0);
	atomicAdd(totAngle + index0, angle0);

	atomicAdd(meshNormal + index1 * 3 + 0, crossX * angle1);
	atomicAdd(meshNormal + index1 * 3 + 1, crossY * angle1);
	atomicAdd(meshNormal + index1 * 3 + 2, crossZ * angle1);
	atomicAdd(totAngle + index1, angle1);

	atomicAdd(meshNormal + index2 * 3 + 0, crossX * angle2);
	atomicAdd(meshNormal + index2 * 3 + 1, crossY * angle2);
	atomicAdd(meshNormal + index2 * 3 + 2, crossZ * angle2);
	atomicAdd(totAngle + index2, angle2);
}
//法线归一化
__global__ void normalizeMeshtriVertNorm_debug(float* meshNormal, float* meshPosition, float* totAngle, int meshVertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= meshVertexNum) return;

	float normalx = meshNormal[threadid * 3 + 0];
	float normaly = meshNormal[threadid * 3 + 1];
	float normalz = meshNormal[threadid * 3 + 2];

	// 按照三角形中角的大小对顶点法向量进行加权。共用该顶点的三角形的法向量所占权重与占用该顶点的三角形的角度成正比。
	float product = totAngle[threadid];
	normalx /= product;
	normaly /= product;
	normalz /= product;
	//法线单位化
	float sqr_len = normalx * normalx + normaly * normaly + normalz * normalz;
	float len = sqrt(sqr_len);
	normalx /= len;
	normaly /= len;
	normalz /= len;



#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		float px = meshPosition[threadid * 3 + 0];
		float py = meshPosition[threadid * 3 + 1];
		float pz = meshPosition[threadid * 3 + 2];
		printf("normalizeMeshNormalMU threadid %d: n[%f %f %f] p[%f %f %f]\n", threadid, normalx, normaly, normalz, px, py, pz);
		if (product < 1e-6)
		{
			printf("totalAngle product is too small\n");
		}
		if (len < 1e-6)
		{
			printf("length of normal is too small\n");
		}
	}
#endif
	meshNormal[threadid * 3 + 0] = normalx;
	meshNormal[threadid * 3 + 1] = normaly;
	meshNormal[threadid * 3 + 2] = normalz;
}

//法线归一化
__global__ void normalizeMeshtriVertNorm_debug(float* meshNormal, float* meshPosition, float* totAngle,
	int* sortedTriVertIndices, int offset, int activeElementNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;
	int triVertIdx = sortedTriVertIndices[offset + threadid];
	float normalx = meshNormal[triVertIdx * 3 + 0];
	float normaly = meshNormal[triVertIdx * 3 + 1];
	float normalz = meshNormal[triVertIdx * 3 + 2];

	// 按照三角形中角的大小对顶点法向量进行加权。共用该顶点的三角形的法向量所占权重与占用该顶点的三角形的角度成正比。
	float product = totAngle[triVertIdx];
	normalx /= product;
	normaly /= product;
	normalz /= product;
	//法线单位化
	float sqr_len = normalx * normalx + normaly * normaly + normalz * normalz;
	float len = sqrt(sqr_len);
	normalx /= len;
	normaly /= len;
	normalz /= len;

#ifdef OUTPUT_INFO
	if (triVertIdx == LOOK_THREAD)
	{
		float px = meshPosition[triVertIdx * 3 + 0];
		float py = meshPosition[triVertIdx * 3 + 1];
		float pz = meshPosition[triVertIdx * 3 + 2];
		printf("normalizeMeshNormalMU triVertIdx %d: n[%f %f %f] p[%f %f %f]\n", triVertIdx, normalx, normaly, normalz, px, py, pz);
		if (product < 1e-6)
		{
			printf("totalAngle product is too small\n");
		}
		if (len < 1e-6)
		{
			printf("length of normal is too small\n");
		}
	}
#endif
	meshNormal[triVertIdx * 3 + 0] = normalx;
	meshNormal[triVertIdx * 3 + 1] = normaly;
	meshNormal[triVertIdx * 3 + 2] = normalz;
}
int setDDirwithNormal()
{
	cudaMemcpy(triVertNonPenetrationDir_d, triVertNorm_d, triVertNum_d * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
	int threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;

	setNonPenetrationDirWithTriVertNormal << <blockNum, threadNum >> > (triVertNonPenetrationDir_d, triVertNorm_d, triVertNum_d);
	cudaDeviceSynchronize();
	printCudaError("runUpdateDirectDirectionMU");
	return 0;
}

__global__ void setNonPenetrationDirWithTriVertNormal(float* nonPenetrationDir, float* normal, int vertexNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	nonPenetrationDir[indexX] = -normal[indexX];
	nonPenetrationDir[indexY] = -normal[indexY];
	nonPenetrationDir[indexZ] = -normal[indexZ];
}

__global__ void updateInnerTetVertDirectDirection(
	float* tetVertPositions,
	int* bindingTetVertIndices, float* bindingWeight, 
	float* directDir, 
	int vertexNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	
	float q[3] = { tetVertPositions[threadid * 3 + 0],tetVertPositions[threadid * 3 + 1] ,tetVertPositions[threadid * 3 + 2] };
	float w0 = bindingWeight[threadid * 3 + 0];
	float w1 = bindingWeight[threadid * 3 + 1];
	float w2 = bindingWeight[threadid * 3 + 2];
	int bindingTetVertIdx0 = bindingTetVertIndices[threadid * 3 + 0];
	int bindingTetVertIdx1 = bindingTetVertIndices[threadid * 3 + 1];
	int bindingTetVertIdx2 = bindingTetVertIndices[threadid * 3 + 2];
	float p0[3] = { tetVertPositions[bindingTetVertIdx0 * 3 + 0], tetVertPositions[bindingTetVertIdx0 * 3 + 1], tetVertPositions[bindingTetVertIdx0 * 3 + 2] };
	float p1[3] = { tetVertPositions[bindingTetVertIdx1 * 3 + 0], tetVertPositions[bindingTetVertIdx1 * 3 + 1], tetVertPositions[bindingTetVertIdx1 * 3 + 2] };
	float p2[3] = { tetVertPositions[bindingTetVertIdx2 * 3 + 0], tetVertPositions[bindingTetVertIdx2 * 3 + 1], tetVertPositions[bindingTetVertIdx2 * 3 + 2] };
	float p[3];
	p[0] = p0[0] * w0 + p1[0] * w1 + p2[0] * w2;
	p[1] = p0[1] * w0 + p1[1] * w1 + p2[1] * w2;
	p[2] = p0[2] * w0 + p1[2] * w1 + p2[2] * w2;
	float dDir[3];
	dDir[0] = q[0] - p[0];
	dDir[1] = q[1] - p[1];
	dDir[2] = q[2] - p[2];
	//if (w2 > 0)
	//{// a inner point
	//	printf("q: %f %f %f\npoint0: %f %f %f point1: %f %f %f point2: %f %f %f\ndis: %f %f %f\n",
	//		q[0], q[1], q[2],
	//		p0[0], p0[1], p0[2],
	//		p1[0], p1[1], p1[2],
	//		p2[0], p2[1], p2[2],
	//		l2len(q, p0),l2len(q, p1), l2len(q, p2));
	//}
	float dDirLen = sqrt(dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2]);
	//if (dDirLen < 1e-6)
	//	printf("dDir not available. w0:%f w1:%f w2:%f\n", w0, w1, w2);
	directDir[threadid * 3 + 0] = dDir[0] / dDirLen;
	directDir[threadid * 3 + 1] = dDir[1] / dDirLen;
	directDir[threadid * 3 + 2] = dDir[2] / dDirLen;
	// 此时，dDir为对应的表面顶点指向四面体顶点的向量。
	// 如果当前四面体顶点在模型表面，dDir是有问题的。
	// 解决办法是在另一个函数中计算在表面的四面体顶点的指导向量（用对应的表面三角网格顶点的法向量的反方向定义）
	// 两个函数运行完成之后再统一标准化
}

__global__ void updateSurfaceTetVertDirectDirection(
	int* onSurfaceTetVertIndices,
	int* TetVertNearestTriVertIndices, float* triVertNorm,
	float* tetVertDDir, float* tetVertPos, float* triVertPos, int surfaceTetVertNum)
{
	//更新在表面的四面体顶点指导向量（设置为与其帮定的表面网格顶点法向量的反方向）
	// 如果表面网格顶点和四面体顶点之间有较大的偏离，那这个指导向量的方向可能不准确。
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= surfaceTetVertNum) return;

	int surfaceTetVertIdx = onSurfaceTetVertIndices[threadid];
	int idx0 = surfaceTetVertIdx * 3 + 0;
	int idx1 = surfaceTetVertIdx * 3 + 1;
	int idx2 = surfaceTetVertIdx * 3 + 2;
	int matchingTriVertIdx = TetVertNearestTriVertIndices[surfaceTetVertIdx];
	float n[3] = { triVertNorm[matchingTriVertIdx * 3 + 0],triVertNorm[matchingTriVertIdx * 3 + 1] ,triVertNorm[matchingTriVertIdx * 3 + 2] };
	tetVertDDir[idx0] = -n[0];
	tetVertDDir[idx1] = -n[1];
	tetVertDDir[idx2] = -n[2];
	float tetPos[3] = { tetVertPos[idx0], tetVertPos[idx1], tetVertPos[idx2] };
	float triPos[3] = { triVertPos[matchingTriVertIdx * 3 + 0], triVertPos[matchingTriVertIdx * 3 + 1] ,triVertPos[matchingTriVertIdx * 3 + 2] };
	float d[3] = { tetPos[0] - triPos[0],tetPos[1] - triPos[1] ,tetPos[2] - triPos[2] };
	float dist = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	//if (dist > 1e-3)
	//{
	//	printf("threadid:%d tetPoint-surfacepoint matching error! d[%f %f %f]\n", threadid, d[0], d[1], d[2]);
	//}
	//printf("SurfaceTetVertDDir: %f %f %f\n", n[0], n[1], n[2]);
}
void printCudaError() {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "error: %s\n", cudaGetErrorString(cudaStatus));
	}
}

void printCudaError(const char* funcName) {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "%s error: %s\n", funcName, cudaGetErrorString(cudaStatus));
	}
}