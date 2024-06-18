#include "gpuvar.h"
#include "gpufun.h"
float*			hapticDeformationCollisionForce_D; // UNUSED
float*			hapticDeformationInterpolatePositions_D;// UNUSED
float*			hapticDeformationExternForce_D;    // 回传反作用力，存储每个顶点上由工具施加的惩罚力。
float*			hapticDeformationExternForceTotal_D; // 累加工具施加到软体上的力，在applyForce的时候清空。
int				hapticCounter_D;// 在上一个变形帧之后经历了多少个力反馈帧的计数器。
float*			hapticDeformationPrePositions_D;	//用于插值计算
float*			hapticDeformationPositions_D;	//力反馈端进行碰撞检测的变形体粒子，需要定时同步
float*			hapticDeformationNormals_D;     // 力反馈端进行碰撞检测的四面体法向量，需要定时同步
float*			hapticCollisionZone_D;			//记录粒子碰撞产生的区域【相对于线段】，根据区域采用不同的约束方法

int* hapticContinuousFrameNumOfCollision_D;     // 记录该顶点被工具施加压力的连续帧数。

int				hapticDeformationNum_D;			//粒子数量，全体四面体顶点的数量，包括内部四面体顶点。变形中的基本单位是四面体顶点，即粒子。
int				hapticDeformationNumMem_D;



unsigned int*	hapticIsCollide_D;
float*			hapticConstraintForce_D;
float*			hapticConstraintPoints_D;	//存储最终的碰撞顶点数
float*			hapticConstraintNormals_D;
float*          hapticConstraintZone_D;

float*			hapticCylinderPos_D;
float*			hapticCylinderPhysicalPos_D;
float*			hapticCylinderDir_D;
int*			hapticIndex_D;				//存储碰撞顶点的最大索引

unsigned int*	hapticQueueIndex_D;
unsigned int*	hapticAuxSumArray_D;
int* haptic_collisionIndex_to_vertIndex_array_D; //碰撞队列下标对应的顶点下标

//方案1，使用表面三角形的连续碰撞检测
int				hapticAABBBoxNum_D;//表面三角形的数量，每个表面三角形都对应一个AABB包围盒。
float*			hapticAABBBoxs_D;
float*			hapticTriangleNormal_D;// 表面三角形的法向量
int*			hapticSurfaceIndex_D; // 表面三角形顶点在所有*包括内部的*四面体顶点中的下标。

//方案2，使用球树
int				hapticSphereNum_D;
float*			hapticSphereInfo_D;
float*			hapticSphereDirectDirection_D;	//球的指导向量
float*			hapticSphereForce_D;	//球收到的碰撞力
unsigned int*	hapticSphereIsCollide_D;
float*			hapticSphereCollisionZone_D;
int*			hapticSphereindex_D;
float*			hapticSphereConstraintPoints_D;
float*          hapticSphereConstraintZone_D;
float*			hapticSphereConstraintDirection_D;  //约束指导向量
unsigned int*	hapticSphereTetIndex_D;
float*			hapticSphereTetCoord_D;



//球树碰撞队列
unsigned int*	hapticSphereQueueIndex_D;
unsigned int*	hapticSphereAuxSumArray_D;

int				hapticSphereConstraintNumLeft;
float*			hapticSphereConstraintPosLeft;
float*			hapticSphereConstraintZoneLeft;
float*			hapticSphereConstraintDirectionLeft;

int				hapticSphereConstraintNumRight;
float*			hapticSphereConstraintPosRight;
float*			hapticSphereConstraintZoneRight;
float*			hapticSphereConstraintDirectionRight;
////////////////////////////////////////////////////////
// 点壳碰撞队列
unsigned int* hapticPointQueueIndex_D;
unsigned int* hapticPointAuxSumArray_D;

int		hapticPointConstraintNumLeft;
float*	hapticPointConstraintPosLeft;
float* hapticPointConstraintNormalLeft;
float*	hapticPointConstraintZoneLeft;

float* hapticVertexForceOrthogonalToTool_D;
//----------------------------------------
			  
int		hapticPointConstraintNumRight;
float*	hapticPointConstraintPosRight;
float*	hapticPointConstraintZoneRight;
float*	hapticPointConstraintDirectionRight;
///////////////////////////////////////////////////////////

float hapticCollisionStiffness_D;
int MAX_CONTINUOUS_FRAME_COUNT_D;
extern float* triVertCollisionDiag_d;
extern float*  triVertForce_d;

int runUpdateSurfacePointPosition(float dt, int point_num)
{
	int  threadNum = 512;
	int blockNum = (hapticDeformationNum_D + threadNum - 1) / threadNum;

	hapticUpdatePointPosition << <blockNum, threadNum >> > (\
		tetVertMass_d,
		hapticDeformationPositions_D,
		tetVertVelocity_d, 
		hapticDeformationExternForce_D,
		dt,
		point_num);
	cudaDeviceSynchronize();
	return 0;
}

//力反馈端进行碰撞检测，通过流压缩存储碰撞信息，用于之后计算虚拟工具位姿。

int runHapticCollision(float halfLength, float radius) {
	int  threadNum = 512;
	int blockNum = (hapticDeformationNum_D + threadNum - 1) / threadNum;
	//将碰撞队列清空

	cudaMemset(hapticIndex_D, -1, sizeof(int));
	
	float obj_r = 0.05f;
	float extended_radius = radius + obj_r;

	//并行计算碰撞
	////hapticCalculateCCylinder << <blockNum, threadNum >> >(hapticCylinderPos_D, hapticCylinderDir_D, halfLength, radius, hapticDeformationPositions_D, hapticIsCollide_D, hapticCollisionZone_D, hapticDeformationNum_D, hapticIndex_D);

	//printf("haptic deformationNum:%d\n", hapticDeformationNum_D);
	hapticCollision_MeshCapsule << <blockNum, threadNum >> > (
		hapticCylinderPos_D, hapticCylinderDir_D, halfLength, extended_radius,
		hapticDeformationPositions_D,
		hapticDeformationNormals_D,
		hapticIsCollide_D,
		 triVertForce_d,
		triVertCollisionDiag_d, hapticCollisionStiffness_D,
		hapticCollisionZone_D,
		hapticDeformationNum_D,
		hapticIndex_D);
	//hapticCalculateMeshCylinder <<<blockNum, threadNum >>> (
	//	hapticCylinderPos_D, hapticCylinderPhysicalPos_D,
	//	hapticCylinderDir_D, halfLength, extended_radius, 
	//	hapticDeformationPositions_D, 
	//	hapticDeformationNormals_D,
	//	hapticIsCollide_D, 
	//	hapticDeformationExternForce_D,
	//	hapticCollisionZone_D, 
	//	hapticContinuousFrameNumOfCollision_D, MAX_CONTINUOUS_FRAME_COUNT_D,
	//	hapticDeformationNum_D, 
	//	hapticIndex_D);
	//hapticCalculateContinuousCylinder << <blockNum, threadNum >> > (
	//	hapticCylinderPos_D, hapticCylinderPhysicalPos_D,
	//	hapticCylinderDir_D, halfLength, radius,
	//	hapticDeformationPositions_D,
	//	hapticDeformationNormals_D,
	//	hapticIsCollide_D,
	//	hapticDeformationExternForce_D,
	//	hapticCollisionZone_D,
	//	hapticContinuousFrameNumOfCollision_D, MAX_CONTINUOUS_FRAME_COUNT_D,
	//	hapticDeformationNum_D,
	//	hapticIndex_D);

	//得到碰撞点之后，计算前缀和得到在队列中的索引(第三个参数是共享内存大小)
	hapticCalculatePrefixSum << <blockNum, threadNum, threadNum *sizeof(unsigned int) >> > (hapticIsCollide_D, hapticQueueIndex_D, hapticAuxSumArray_D, hapticDeformationNum_D);
	//再根据索引，填写碰撞点到队列中
	//hapticAddCollisionToQueue << <blockNum, threadNum >> > (hapticIsCollide_D, hapticDeformationPositions_D, hapticDeformationNormals_D, hapticCollisionZone_D, hapticConstraintPoints_D, hapticConstraintNormals_D, hapticConstraintZone_D, hapticQueueIndex_D, hapticAuxSumArray_D, hapticDeformationNum_D);
	hapticAddCollisionToQueue_SaveMap << <blockNum, threadNum >> > (
		hapticIsCollide_D, 
		hapticDeformationPositions_D, 
		hapticDeformationNormals_D, 
		hapticCollisionZone_D, 
		hapticConstraintPoints_D, 
		hapticConstraintNormals_D, 
		hapticConstraintZone_D, 
		hapticQueueIndex_D, 
		hapticAuxSumArray_D, 
		hapticDeformationNum_D, 
		haptic_collisionIndex_to_vertIndex_array_D);

	cudaDeviceSynchronize();
	
	return 0;
}

//力反馈端的连续碰撞检测
int runHapticContinueCollision(float* start,float* end,float halfLength, float radius) {
	int  threadNum = 512;
	int blockNum = (hapticAABBBoxNum_D + threadNum - 1) / threadNum;

	//线段的碰撞检测
	hapticCalculateContinueCylinder << <blockNum, threadNum >> >(
		start[0], start[1], start[2],
		end[0], end[1], end[2],
		hapticSurfaceIndex_D, hapticDeformationPositions_D, hapticAABBBoxs_D, hapticTriangleNormal_D, hapticAABBBoxNum_D);
	
	return 0;
}


int runHapticCollisionSphere(float halfLength, float radius) {

	int  threadNum = 512;
	int blockNum = (hapticSphereNum_D + threadNum - 1) / threadNum;


	//将碰撞队列清空
	cudaMemset(hapticSphereindex_D, -1, sizeof(int));

	//圆柱和球的碰撞检测
	hapticCalculateCylinderSphere << <blockNum, threadNum >> >(hapticCylinderPos_D, hapticCylinderDir_D, halfLength, radius, hapticSphereInfo_D,hapticSphereForce_D, hapticSphereIsCollide_D,hapticSphereCollisionZone_D,hapticSphereindex_D, hapticSphereNum_D);

	//同样使用前缀和将碰撞结果放置到碰撞队列中
	//得到碰撞点之后，计算前缀和得到在队列中的索引(第三个参数是共享内存大小)
	hapticCalculatePrefixSum << <blockNum, threadNum, threadNum *sizeof(unsigned int) >> > (hapticSphereIsCollide_D, hapticSphereQueueIndex_D, hapticSphereAuxSumArray_D, hapticSphereNum_D);
	//再根据索引，填写碰撞点到队列中
	hapticAddSphereCollisionToQueue << <blockNum, threadNum >> > (hapticSphereIsCollide_D, hapticSphereInfo_D, hapticSphereCollisionZone_D, hapticSphereDirectDirection_D, hapticSphereConstraintPoints_D, hapticSphereConstraintZone_D, hapticSphereConstraintDirection_D, hapticSphereQueueIndex_D, hapticSphereAuxSumArray_D, hapticSphereNum_D);

	cudaDeviceSynchronize();

	return 0;
}

int runHapticCollisionSphere_Tri(float halfLength, float radius) {
	int threadNum = 512;
	int blockNum = (hapticSphereNum_D + threadNum - 1) / threadNum;

	//将碰撞队列清空
	cudaMemset(hapticSphereindex_D, -1, sizeof(int));

	//圆柱和球的碰撞检测
	hapticCalculateCylinderSphere_Tri<< <blockNum, threadNum>> >(hapticCylinderPos_D, hapticCylinderDir_D, halfLength, radius, hapticSphereInfo_D, hapticSphereForce_D, hapticSphereIsCollide_D, hapticSphereCollisionZone_D, hapticSphereindex_D, hapticSphereNum_D);

	//同样使用前缀和将碰撞结果放置到碰撞队列中
	//得到碰撞点之后，计算前缀和得到在队列中的索引(第三个参数是共享内存大小)
	hapticCalculatePrefixSum << <blockNum, threadNum, threadNum * sizeof(unsigned int) >> > (hapticSphereIsCollide_D, hapticSphereQueueIndex_D, hapticSphereAuxSumArray_D, hapticSphereNum_D);
	//再根据索引，填写碰撞点到队列
	hapticAddSphereCollisionToQueue_Tri << <blockNum, threadNum >> > (hapticSphereIsCollide_D, hapticSphereInfo_D, hapticSphereCollisionZone_D, hapticSphereConstraintPoints_D, hapticSphereConstraintZone_D, hapticSphereQueueIndex_D, hapticSphereAuxSumArray_D, hapticSphereNum_D);

	cudaDeviceSynchronize();
	return 0;
}
int runAccumulateExternForce(int point_num)
{
	int  threadNum = 512;
	int blockNum = (hapticDeformationNum_D + threadNum - 1) / threadNum;

	AccumulateExternForce << <blockNum, threadNum >> > (\
		hapticDeformationExternForceTotal_D,
		hapticDeformationExternForce_D,
		point_num);

	cudaDeviceSynchronize();

	hapticCounter_D++;
	return 0;
}
// 反馈力传递回变形端，作为外力施加到软体上
int runDispatchForceToTetVertex()
{
	int  threadNum = 512;
	int blockNum = (hapticSphereNum_D + threadNum - 1) / threadNum;

	dispatchForceToTetVertex << <blockNum, threadNum >> > (hapticDeformationExternForce_D, hapticVertexForceOrthogonalToTool_D, hapticIsCollide_D, hapticDeformationNum_D);
	cudaDeviceSynchronize();
	return 0;
}
//将球的受力信息传递到四面体顶点（deprecated）
int runDispatchSphereToTet() {
	int  threadNum = 512;
	int blockNum = (hapticSphereNum_D + threadNum - 1) / threadNum;

	dispatchToTet << <blockNum, threadNum >> >(hapticSphereTetIndex_D,hapticSphereTetCoord_D,hapticDeformationExternForce_D,hapticSphereForce_D, hapticSphereIsCollide_D, hapticSphereNum_D);
	cudaDeviceSynchronize();
	return 0;
}

// deprecated
__global__ void hapticCalculateSurfaceCylinder(
	float* cylinderPos, float* cylinderDir, float Length, float radius, 
	float* vertexPositions,
	unsigned int* isCollide, 
	float* zone, 
	int surfaceVertexNum, 
	int* index)// unfinished
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= surfaceVertexNum) return;
	int vertIndex0 = threadid * 3 + 0;
	int vertIndex1 = threadid * 3 + 1;
	int vertIndex2 = threadid * 3 + 2;

	float vert0_x = vertexPositions[vertIndex0 * 3 + 0];
	float vert0_y = vertexPositions[vertIndex0 * 3 + 1];
	float vert0_z = vertexPositions[vertIndex0 * 3 + 2];

	float vert1_x = vertexPositions[vertIndex1 * 3 + 0];
	float vert1_y = vertexPositions[vertIndex1 * 3 + 1];
	float vert1_z = vertexPositions[vertIndex1 * 3 + 2];

	float vert2_x = vertexPositions[vertIndex2 * 3 + 0];
	float vert2_y = vertexPositions[vertIndex2 * 3 + 1];
	float vert2_z = vertexPositions[vertIndex2 * 3 + 2];


}

__global__ void hapticCalculateContinuousCylinder(
	float* cylinderPos,
	float* hapticCylinderPos,
	float* cylinderDir, float halfLength, float radius,
	float* tetPositions,
	float* vertexNormals,
	unsigned int* isCollide,
	float* vertexForce, // 从工具指向碰撞点的反馈力
	float* zone,
	int* continuousFrameCounter, int max_continuous_frame,
	int vertexNum,
	int* index)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//重置碰撞标志位
	isCollide[threadid] = 0;
	zone[threadid] = -1;

	float nx = vertexNormals[indexX];
	float ny = vertexNormals[indexY];
	float nz = vertexNormals[indexZ];
	float len_normal = sqrt(nx * nx + ny * ny + nz * nz);
	bool isOnSurface;
	if (len_normal < 0.1)
		isOnSurface = false;
	else
	{
		isOnSurface = true;
		nx /= len_normal;
		ny /= len_normal;
		nz /= len_normal;
	}


	//if (len_normal < 0.1)// 法向量为0，该点为软体内部的顶点，不计算碰撞，跳过后面的计算。
	//	return;

	__shared__ float cylinder0[3];
	__shared__ float cylinder1[3];
	__shared__ float cylinderd[3];
	__shared__ float hapticCylinderTip[3];

	hapticCylinderTip[0] = hapticCylinderPos[0];
	hapticCylinderTip[1] = hapticCylinderPos[1];
	hapticCylinderTip[2] = hapticCylinderPos[2];

	cylinder0[0] = cylinderPos[0];
	cylinder0[1] = cylinderPos[1];
	cylinder0[2] = cylinderPos[2];

	cylinder1[0] = cylinderPos[0] + cylinderDir[0] * halfLength;
	cylinder1[1] = cylinderPos[1] + cylinderDir[1] * halfLength;
	cylinder1[2] = cylinderPos[2] + cylinderDir[2] * halfLength;

	cylinderd[0] = cylinder1[0] - cylinder0[0];
	cylinderd[1] = cylinder1[1] - cylinder0[1];
	cylinderd[2] = cylinder1[2] - cylinder0[2];
	float dx = tetPositions[indexX] - cylinder0[0];
	float dy = tetPositions[indexY] - cylinder0[1];
	float dz = tetPositions[indexZ] - cylinder0[2];
	float t = cylinderDir[0] * dx + cylinderDir[1] * dy + cylinderDir[2] * dz;

	t /= halfLength; // t是碰撞点在工具上的百分比位置，尖端为0，尾部为1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// 工具中轴上的 接触四面体的投影点->四面体位置
	// 当接触点在工具杆上的时候，该向量垂直于工具中轴线，从工具中轴上的投影点指向接触点。
	// 当接触点在工具尖端的时候，这个向量从工具中轴的尖端指向接触点。
	dx = tetPositions[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = tetPositions[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = tetPositions[indexZ] - cylinder0[2] - t * cylinderd[2];

	float sqr_distance = dx * dx + dy * dy + dz * dz;
	float distance = sqrt(sqr_distance);
	dx /= distance; dy /= distance; dz /= distance;
	// 顶点在虚拟工具中轴上的投影点
	float p0[3] = {
		cylinder0[0] + t * cylinderd[0],
		cylinder0[1] + t * cylinderd[1],
		cylinder0[2] + t * cylinderd[2] };
	// 顶点在物理工具中轴上的投影点
	float p1[3] = {
		hapticCylinderTip[0] + t * cylinderd[0],
		hapticCylinderTip[1] + t * cylinderd[1],
		hapticCylinderTip[2] + t * cylinderd[2] };
	// 从虚拟工具上的投影点指向物理工具上投影点的向量
	float v[3] = { p1[0] - p0[0], p1[1] - p0[1] ,p1[2] - p0[2] };
	float gh_distance = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

	const float GH_DISTANCE_THREASHOLD = 0.25;
	if (gh_distance < GH_DISTANCE_THREASHOLD)// 直接用图形位姿做碰撞检测
	{
		if (distance < radius)//此条件的结果需要写入碰撞队列用于计算虚拟工具位姿
		{
			if (isOnSurface)
			{
				// 计算工具在顶点上施加的外力，方向为[-nx, -ny, -nz](表面顶点法向量的反方向)
				atomicAdd(vertexForce + threadid * 3 + 0, -nx * (radius - distance));
				atomicAdd(vertexForce + threadid * 3 + 1, -ny * (radius - distance));
				atomicAdd(vertexForce + threadid * 3 + 2, -nz * (radius - distance));
			}
			else
			{
				float fx = dx * (radius - distance);
				float fy = dy * (radius - distance);
				float fz = dz * (radius - distance);
				vertexForce[indexX] += fx;
				vertexForce[indexY] += fy;
				vertexForce[indexZ] += fz;
				//printf("inner point collision:[%f %f %f] len: %f\n", fx, fy, fz, radius-distance);
			}

			isCollide[threadid] = 1;
			zone[threadid] = t;
			//队列索引加一(碰撞信息在计算前缀和的时候赋值)
			atomicAdd(index, 1);

			//// printf("虚拟工具半径范围内发生碰撞， threadid:%d counter: %d\n", threadid, continuousFrameCounter[threadid]);
			// 工具对顶点施加了力，当前顶点的碰撞连续帧数量+1
			if (continuousFrameCounter[threadid] < max_continuous_frame)
			{
				continuousFrameCounter[threadid] += 1;
			}
		}
		else
		{
			// 未施加力，当前顶点的碰撞连续帧数量-1
			if (continuousFrameCounter[threadid] > 0)
			{
				continuousFrameCounter[threadid] -= 1;
			}
		}
	}
	else
	{
		// moveDir指的是“把物理位姿对齐到虚拟工具位姿的移动向量”
		float moveDir[3] = { -hapticCylinderPos[0] + cylinderPos[0],
			-hapticCylinderPos[1] + cylinderPos[1],
		-hapticCylinderPos[2] + cylinderPos[2] };
		float moveDistance = sqrt(moveDir[0] * moveDir[0] + moveDir[1] * moveDir[1] + moveDir[2] * moveDir[2]);
		float point[3] = { tetPositions[indexX], tetPositions[indexY], tetPositions[indexZ] };
		float collisionNormal[3];
		float collisionPos[3];
		
		float k = 1;
		float middlePos[3] = { hapticCylinderPos[0] * k + cylinderPos[0] * (1 - k),
								hapticCylinderPos[1] * k + cylinderPos[1] * (1 - k),
								hapticCylinderPos[2] * k + cylinderPos[2] * (1 - k) };

		bool collided = hapticCylinderCollisionContinue(halfLength, radius,
			middlePos, cylinderPos, cylinderDir, point,
			collisionNormal, collisionPos);
		if (collided)
		{
			float fx = collisionPos[0]-point[0];
			float fy = collisionPos[1]-point[1];
			float fz = collisionPos[2]-point[2];
			float f_len = sqrt(fx * fx + fy * fy + fz * fz);
			vertexForce[indexX] += fx;
			vertexForce[indexY] += fy;
			vertexForce[indexZ] += fz;
			//printf("continuous: collisionPos:[%f %f %f] point[%f %f %f]\n", collisionPos[0], collisionPos[1], collisionPos[2], point[0], point[1], point[2]);
			isCollide[threadid] = 1;
			zone[threadid] = 1;
			//队列索引加一(碰撞信息在计算前缀和的时候赋值)
			atomicAdd(index, 1);

			// 工具对顶点施加了力，当前顶点的碰撞连续帧数量+1
			if (continuousFrameCounter[threadid] < max_continuous_frame)
			{
				continuousFrameCounter[threadid] += 1;
			}
		}
		else
		{
			// 未施加力，当前顶点的碰撞连续帧数量-1
			if (continuousFrameCounter[threadid] > 0)
			{
				continuousFrameCounter[threadid] -= 1;
			}
		}
	}
}

__device__ bool hapticCylinderCollisionContinue(
	float length, float radius,
	float* HPos, float* SPos,
	float* cylinderDir,
	float* position,
	float* collisionNormal, float* collisionPos)
{
	// moveDir指把物理工具对齐到虚拟工具需要移动的向量，是物理位姿指向虚拟位姿的向量
	float moveDir[3] = { SPos[0] - HPos[0],SPos[1] - HPos[1],SPos[2] - HPos[2] };
	float moveDistance = sqrt(moveDir[0] * moveDir[0] + moveDir[1] * moveDir[1] + moveDir[2] * moveDir[2]);
	//首先计算出运动平面的法线向量
	float normal[3];
	tetCross_D(cylinderDir, moveDir, normal);
	tetNormal_D(normal);


	//定义计算需要的变量
	float VSubO[3] = { position[0] - HPos[0] ,position[1] - HPos[1] ,position[2] - HPos[2] };//工具尖端指向可能碰撞点的向量
	float lineStart0[3] = { HPos[0] ,HPos[1] ,HPos[2] };// 当前工具尖端
	float lineStart1[3] = { SPos[0] ,SPos[1] ,SPos[2] };// 上一帧工具尖端
	float lineStart2[3] = { HPos[0] + cylinderDir[0] * length ,HPos[1] + cylinderDir[1] * length,HPos[2] + cylinderDir[2] * length };// 当前帧工具尾部


	//首先要先进行一次碰撞检测，物理工具是否发生碰撞，若不发生则不需要进行碰撞检测


	//1.计算在局部坐标系中的坐标，由于这个局部坐标不是正交的，所以不能和轴进行点积，使用高斯消元
	// 三个轴： 工具方向cylinderDir，运动方向moveDir，工具方向与运动方向张成的平面的法向量normal
	// 计算碰撞点在这三个轴组成的局部坐标系坐标[x, y, z] 用处：在这个非正交坐标系下，x就是碰撞点在工具上投影的位置，y就是在运动方向上的运动距离
	// 高斯消元：[A|I] 只使用行之间的加减，A变成I，I会变成A的逆矩阵。
	float x, y, z;
	float det = tetSolveFormula_D(cylinderDir, moveDir, normal, VSubO, &x, &y, &z);

	if (x != x || y != y || z != z) return false;

	float distance = 0.0;
	bool flag = false;
	//2.根据坐标找到顶点所在的区域
	if (x > length && y > moveDistance) {
		//计算点到点的距离
		float basePoint[3] = { SPos[0] + length * cylinderDir[0],SPos[1] + length * cylinderDir[1] , SPos[2] + length * cylinderDir[2] };
		distance = tetPointPointDistance_D(position, basePoint);
		flag = true;
	}
	else if (x > length && y < moveDistance && y>0.0) {
		//计算点到边的距离
		distance = tetPointLineDistance_D(lineStart2, moveDir, position);
	}
	else if (x > length && y < 0.0) {
		distance = tetPointPointDistance_D(position, lineStart2);
	}
	else if (x > 0.0 && x < length && y > moveDistance) {
		distance = tetPointLineDistance_D(lineStart1, cylinderDir, position);
		flag = true;
	}
	else if (x > 0.0 && x < length && y < moveDistance && y>0.0) {
		//计算点到面的距离
		distance = abs(z);
	}
	else if (x > 0.0 && x < length && y < 0.0) {
		distance = tetPointLineDistance_D(lineStart0, cylinderDir, position);
	}
	else if (x<0.0 && y > moveDistance) {
		distance = tetPointPointDistance_D(position, SPos);
		flag = true;
	}
	else if (x < 0.0 && y < moveDistance && y>0.0) {
		distance = tetPointLineDistance_D(lineStart0, moveDir, position);
	}
	else if (x < 0.0 && y < 0.0) {
		distance = tetPointPointDistance_D(position, HPos);
	}


	//3.判断距离
	if (distance > radius) return false;
	//if (flag) return false;

	//printf("x:%f,y:%f,z:%f\n", x, y, z);

	//4. 计算矫正的碰撞排除位置
	//求解二元一次方程,和两个方向的圆柱进行计算
	float lineDir[3] = { moveDir[0],moveDir[1], moveDir[2] };


	float v0[3] = { position[0] - lineStart0[0] ,position[1] - lineStart0[1] ,position[2] - lineStart0[2] };
	float v1[3] = { position[0] - lineStart1[0] ,position[1] - lineStart1[1] ,position[2] - lineStart1[2] };
	float v2[3] = { position[0] - lineStart2[0] ,position[1] - lineStart2[1] ,position[2] - lineStart2[2] };


	//和圆柱相交
	float solve00, solve01;
	float solve10, solve11;
	tetSolveInsect_D(lineDir, cylinderDir, v0, radius, &solve00, &solve01);
	tetSolveInsect_D(lineDir, moveDir, v0, radius, &solve10, &solve11);
	float solve = min(solve11, solve01);
	//tetSolveInsect_D(lineDir, cylinderDir, v1, radius, &solve00, &solve01);
	//solve = min(solve, solve01);
	//tetSolveInsect_D(lineDir, moveDir, v2, radius, &solve00, &solve01);
	//solve = min(solve, solve01);


	//和球相交
	float solve20, solve21;
	tetSolveInsectSphere_D(lineDir, v0, radius, &solve20, &solve21);
	solve = min(solve, solve21);
	//printf("%f\n", solve);
	//tetSolveInsectSphere_D(lineDir, v1, radius, &solve10, &solve11);
	//solve = min(solve, solve11);
	//tetSolveInsectSphere_D(lineDir, v2, radius, &solve10, &solve11);
	//solve = min(solve, solve11);
	//tetSolveInsectSphere_D(lineDir, VSubO, radius, &solve10, &solve11);
	//solve = min(solve, solve11);

	if (solve != solve) return false;
	//printf("x:%f,y:%f,z:%f, solve: %f\n",x,y,z, solve);

	//更新位置得到顶点排出的位置
	collisionPos[0] = position[0] - lineDir[0] * solve;
	collisionPos[1] = position[1] - lineDir[1] * solve;
	collisionPos[2] = position[2] - lineDir[2] * solve;
	printf("solve01 solve11 solve21:%f %f %f\np[%f %f %f] collisionP[%f %f %f]\n", solve01, solve11, solve21, position[0], position[1], position[2], collisionPos[0], collisionPos[1], collisionPos[2]);

	//更新顶点的碰撞法线，向工具轴线上进行投影
	float projPos[3] = { collisionPos[0] - HPos[0],collisionPos[1] - HPos[1],collisionPos[2] - HPos[2] };
	float proj = tetDot_D(projPos, cylinderDir);
	projPos[0] = collisionPos[0] - HPos[0] - cylinderDir[0] * proj;
	projPos[1] = collisionPos[1] - HPos[1] - cylinderDir[1] * proj;
	projPos[2] = collisionPos[2] - HPos[2] - cylinderDir[2] * proj;

	tetNormal_D(projPos);
	collisionNormal[0] = projPos[0];
	collisionNormal[1] = projPos[1];
	collisionNormal[2] = projPos[2];

	return true;
}

// 对四面体顶点的碰撞检测，根据虚拟工具与物理工具之间的距离较大时，采用虚拟工具与物理工具之间的扫描体做碰撞检测，以增加工具对四面体顶点施加压力的范围。.
__global__ void hapticCalculateMeshCylinder(
	float* cylinderPos,
	float* hapticCylinderPos,
	float* cylinderDir, float halfLength, float radius, 
	float* tetPositions, 
	float* vertexNormals, 
	unsigned int* isCollide, 
	float* vertexForce, // 从工具指向碰撞点的反馈力
	float* zone, 
	int* continuousFrameCounter, int max_continuous_frame,
	int vertexNum, 
	int* index)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//重置碰撞标志位
	isCollide[threadid] = 0;
	zone[threadid] = -1;

	//printf("threadid:%d p[%f %f %f]\n", threadid, tetPositions[indexX], tetPositions[indexY], tetPositions[indexZ]);

	float nx = vertexNormals[indexX];
	float ny = vertexNormals[indexY];
	float nz = vertexNormals[indexZ];
	float len_normal = sqrt(nx * nx + ny * ny + nz * nz);
	bool isOnSurface;
	//if (threadid == 500)
	//{
	//	printf("p[%f %f %f], n[%f %f %f]\n", 
	//		tetPositions[indexX], tetPositions[indexY], tetPositions[indexZ],
	//		nx, ny, nz);
	//}
	if (len_normal < 0.1)
		isOnSurface = false;
	else
	{
		isOnSurface = true;
		nx /= len_normal;
		ny /= len_normal;
		nz /= len_normal;
	}
		
	
	//if (len_normal < 0.1)// 法向量为0，该点为软体内部的顶点，不计算碰撞，跳过后面的计算。
	//	return;

	__shared__ float cylinder0[3];
	__shared__ float cylinder1[3];
	__shared__ float cylinderd[3];
	__shared__ float hapticCylinderTip[3];

	hapticCylinderTip[0] = hapticCylinderPos[0];
	hapticCylinderTip[1] = hapticCylinderPos[1];
	hapticCylinderTip[2] = hapticCylinderPos[2];

	cylinder0[0] = cylinderPos[0];
	cylinder0[1] = cylinderPos[1];
	cylinder0[2] = cylinderPos[2];

	cylinder1[0] = cylinderPos[0] + cylinderDir[0] * halfLength;
	cylinder1[1] = cylinderPos[1] + cylinderDir[1] * halfLength;
	cylinder1[2] = cylinderPos[2] + cylinderDir[2] * halfLength;

	cylinderd[0] = cylinder1[0] - cylinder0[0];
	cylinderd[1] = cylinder1[1] - cylinder0[1];
	cylinderd[2] = cylinder1[2] - cylinder0[2];
	float dx = tetPositions[indexX] - cylinder0[0];
	float dy = tetPositions[indexY] - cylinder0[1];
	float dz = tetPositions[indexZ] - cylinder0[2];
	float t = cylinderDir[0] * dx + cylinderDir[1] * dy + cylinderDir[2] * dz;

	t /= halfLength; // t是碰撞点在工具上的百分比位置，尖端为0，尾部为1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// 工具中轴上的 接触四面体的投影点->四面体位置
	// 当接触点在工具杆上的时候，该向量垂直于工具中轴线，从工具中轴上的投影点指向接触点。
	// 当接触点在工具尖端的时候，这个向量从工具中轴的尖端指向接触点。
	dx = tetPositions[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = tetPositions[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = tetPositions[indexZ] - cylinder0[2] - t * cylinderd[2];

	float sqr_distance = dx * dx + dy * dy + dz * dz;
	float distance = sqrt(sqr_distance);
	dx /= distance; dy /= distance; dz /= distance;
	// 顶点在虚拟工具中轴上的投影点
	float p0[3] = {
		cylinder0[0] + t * cylinderd[0],
		cylinder0[1] + t * cylinderd[1],
		cylinder0[2] + t * cylinderd[2] };
	// 顶点在物理工具中轴上的投影点
	float p1[3] = {
		hapticCylinderTip[0] + t * cylinderd[0],
		hapticCylinderTip[1] + t * cylinderd[1],
		hapticCylinderTip[2] + t * cylinderd[2] };
	// 从虚拟工具上的投影点指向物理工具上投影点的向量
	float v[3] = { p1[0] - p0[0], p1[1] - p0[1] ,p1[2] - p0[2] };
	float gh_distance = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

	const float GH_DISTANCE_THREASHOLD = 0.25;
	if (gh_distance < GH_DISTANCE_THREASHOLD)// 直接用图形位姿做碰撞检测
	{
		if (distance < radius)//此条件的结果需要写入碰撞队列用于计算虚拟工具位姿
		{
			if (isOnSurface)
			{
				// 计算工具在顶点上施加的外力，方向为[-nx, -ny, -nz](表面顶点法向量的反方向)
				atomicAdd(vertexForce + threadid * 3 + 0, -nx * (radius - distance));
				atomicAdd(vertexForce + threadid * 3 + 1, -ny * (radius - distance));
				atomicAdd(vertexForce + threadid * 3 + 2, -nz * (radius - distance));
			}
			else
			{
				float fx = dx * (radius - distance);
				float fy = dy * (radius - distance);
				float fz = dz * (radius - distance);
				vertexForce[indexX] += fx;
				vertexForce[indexY] += fy;
				vertexForce[indexZ] += fz;
				//printf("inner point collision:[%f %f %f]\n", fx, fy, fz);
			}

			isCollide[threadid] = 1;
			zone[threadid] = t;
			//队列索引加一(碰撞信息在计算前缀和的时候赋值)
			atomicAdd(index, 1);

			//// printf("虚拟工具半径范围内发生碰撞， threadid:%d counter: %d\n", threadid, continuousFrameCounter[threadid]);
			// 工具对顶点施加了力，当前顶点的碰撞连续帧数量+1
			if(continuousFrameCounter[threadid]<max_continuous_frame)
			{
				continuousFrameCounter[threadid] += 1;
			}
		}
		else
		{
			// 未施加力，当前顶点的碰撞连续帧数量-1
			if (continuousFrameCounter[threadid] > 0)
			{
				continuousFrameCounter[threadid] -= 1;
			}
		}
	}
	else if(gh_distance >= GH_DISTANCE_THREASHOLD) // 虚拟工具与物理工具之间拉开比较大的距离，用扫描体计算碰撞
	{
		float normal_weight = (gh_distance - distance) / gh_distance * radius;
		if (distance < radius)//此条件的结果需要写入碰撞队列用于计算虚拟工具位姿
		{
			if (isOnSurface)
			{
				// 计算工具在顶点上施加的外力，方向为[-nx, -ny, -nz](表面顶点法向量的反方向)
				atomicAdd(vertexForce + threadid * 3 + 0, -nx * normal_weight);
				atomicAdd(vertexForce + threadid * 3 + 1, -ny * normal_weight);
				atomicAdd(vertexForce + threadid * 3 + 2, -nz * normal_weight);
			}
			else
			{
				float fx = dx * normal_weight;
				float fy = dy * normal_weight;
				float fz = dz * normal_weight;
				vertexForce[indexX] += fx;
				vertexForce[indexY] += fy;
				vertexForce[indexZ] += fz;
				//printf("saomiaoti in radius f[%f %f %f]\n", fx, fy, fz);
			}
			isCollide[threadid] = 1;
			zone[threadid] = t;
			//队列索引加一(碰撞信息在计算前缀和的时候赋值)
			atomicAdd(index, 1);

			////printf("在扫描体半径范围内，threadid: %d counter:%d\n", threadid, continuousFrameCounter[threadid]);
			// 工具对顶点施加了力，当前顶点的碰撞连续帧数量+1
			if (continuousFrameCounter[threadid] < max_continuous_frame)
			{
				continuousFrameCounter[threadid] += 1;
			}
		}
		else if (distance < gh_distance)
		{
			// 工具施加压力的方向
			float dirX = v[0] / gh_distance;
			float dirY = v[1] / gh_distance;
			float dirZ = v[2] / gh_distance;
			// 虚拟工具上的投影点指向碰撞顶点的向量
			float v_g2tetPos[3] = {
				tetPositions[indexX] - cylinder0[0],
				tetPositions[indexY] - cylinder0[1],
				tetPositions[indexZ] - cylinder0[2],
			};
			float temp = v_g2tetPos[0] * dirX + v_g2tetPos[1] * dirY + v_g2tetPos[2] * dirZ;
			float k = temp / gh_distance;
			if ((k < 1) && (k > 0))// 四面体在gh连线上的投影点在gh之间
			{
				float projectedX = p0[0] + k * v[0];
				float projectedY = p0[1] + k * v[1];
				float projectedZ = p0[2] + k * v[2];
				float m[3] = {
					tetPositions[indexX] - projectedX,
					tetPositions[indexY] - projectedY,
					tetPositions[indexZ] - projectedZ
				};
				float dis = sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
				//printf("case 2.2, dis=%f\n", dis);
				if (dis < radius)
				{
					// 在力场范围内，对该顶点施加压力
					float fx = dirX * normal_weight;
					float fy = dirY * normal_weight;
					float fz = dirZ * normal_weight;

					vertexForce[indexX] += fx;
					vertexForce[indexY] += fy;
					vertexForce[indexZ] += fz;
					//printf("saomiaoti f[%f %f %f] weight: %f\n", fx, fy, fz, normal_weight);
					//// printf("在工具半径范围外扫描体内， threadid: %d counter: %d\n", threadid, continuousFrameCounter[threadid]);
					// 工具扫描体对顶点施加了力，当前顶点的碰撞连续帧数量+1
					if (continuousFrameCounter[threadid] < max_continuous_frame)
					{
						continuousFrameCounter[threadid] += 1;
					}
				}
				else
				{
					// printf("在扫描体半径范围内但没有在扫描体内\n");
					if (continuousFrameCounter[threadid] > 0)
					{
						continuousFrameCounter[threadid] -= 1;
					}
				}
				
			}
			else
			{
				//printf("顶点在扫描体所在直线上的投影点在扫描体线段之外 k: %f\n", k);
				// 未施加力，当前顶点的碰撞连续帧数量-1
				if (continuousFrameCounter[threadid] > 0)
				{
					continuousFrameCounter[threadid] -= 1;
				}
			}
		}
		else if (distance > gh_distance)
		{
			//printf("dis: %f, gh_dis:f 在扫描体最大半径外\n", distance, gh_distance);
			// 未施加力，当前顶点的碰撞连续帧数量-1
			if (continuousFrameCounter[threadid] > 0)
			{
				continuousFrameCounter[threadid] -= 1;
			}
		}
	}
}

__global__ void hapticCollision_MeshCapsule(float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* vertexPositions,
	float* vertexNormals,
	unsigned int* isCollide,
	float* vertexForce, // 从工具指向碰撞点的反馈力
	float* collisionDiag, float collisionStiffness,
	float* zone,
	int vertexNum,
	int* index)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//重置碰撞标志位
	isCollide[threadid] = 0;
	zone[threadid] = -1;

	float nx = vertexNormals[indexX];
	float ny = vertexNormals[indexY];
	float nz = vertexNormals[indexZ];
	float len_normal = nx * nx + ny * ny + nz * nz;
	if (len_normal < 0.1)// 法向量为0，该点为软体内部的顶点，不计算碰撞，跳过后面的计算。
		return;

	__shared__ float cylinder0[3];
	__shared__ float cylinder1[3];
	__shared__ float cylinderd[3];


	cylinder0[0] = cylinderPos[0];
	cylinder0[1] = cylinderPos[1];
	cylinder0[2] = cylinderPos[2];

	cylinder1[0] = cylinderPos[0] + cylinderDir[0] * halfLength;
	cylinder1[1] = cylinderPos[1] + cylinderDir[1] * halfLength;
	cylinder1[2] = cylinderPos[2] + cylinderDir[2] * halfLength;

	cylinderd[0] = cylinder1[0] - cylinder0[0];
	cylinderd[1] = cylinder1[1] - cylinder0[1];
	cylinderd[2] = cylinder1[2] - cylinder0[2];
	float dx = vertexPositions[indexX] - cylinder0[0];
	float dy = vertexPositions[indexY] - cylinder0[1];
	float dz = vertexPositions[indexZ] - cylinder0[2];
	float t = cylinderDir[0] * dx + cylinderDir[1] * dy + cylinderDir[2] * dz;

	t /= halfLength; // t是碰撞点在工具上的百分比位置，尖端为0，尾部为1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// 工具中轴上的 接触四面体的投影点->四面体位置，该向量垂直于工具
	dx = vertexPositions[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = vertexPositions[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = vertexPositions[indexZ] - cylinder0[2] - t * cylinderd[2];

	float sqr_distance = dx * dx + dy * dy + dz * dz;
	if (sqr_distance > radius * radius) return;
	float distance = sqrt(sqr_distance);

	// 单位化
	dx /= distance;
	dy /= distance;
	dz /= distance;

	// 计算反馈力，方向为[dx, dy, dz]
	atomicAdd(vertexForce + threadid * 3 + 0, dx * (radius - distance));
	atomicAdd(vertexForce + threadid * 3 + 1, dy * (radius - distance));
	atomicAdd(vertexForce + threadid * 3 + 2, dz * (radius - distance));
	collisionDiag[indexX] += dx * dx * collisionStiffness;
	collisionDiag[indexY] += dy * dy * collisionStiffness;
	collisionDiag[indexZ] += dz * dz * collisionStiffness;

	//设置标志位

	isCollide[threadid] = 1;
	zone[threadid] = t;
	//队列索引加一(碰撞信息在计算前缀和的时候赋值)
	atomicAdd(index, 1);
}

__global__ void hapticCalculateMeshCapsule(float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* tetPositions,
	float* vertexNormals,
	unsigned int* isCollide,
	float* vertexForce, // 从工具指向碰撞点的反馈力
	float* zone,
	int vertexNum,
	int* index)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//重置碰撞标志位
	isCollide[threadid] = 0;
	zone[threadid] = -1;

	float nx = vertexNormals[indexX];
	float ny = vertexNormals[indexY];
	float nz = vertexNormals[indexZ];
	float len_normal = nx * nx + ny * ny + nz * nz;
	if (len_normal < 0.1)// 法向量为0，该点为软体内部的顶点，不计算碰撞，跳过后面的计算。
		return;

	__shared__ float cylinder0[3];
	__shared__ float cylinder1[3];
	__shared__ float cylinderd[3];


	cylinder0[0] = cylinderPos[0];
	cylinder0[1] = cylinderPos[1];
	cylinder0[2] = cylinderPos[2];

	cylinder1[0] = cylinderPos[0] + cylinderDir[0] * halfLength;
	cylinder1[1] = cylinderPos[1] + cylinderDir[1] * halfLength;
	cylinder1[2] = cylinderPos[2] + cylinderDir[2] * halfLength;

	cylinderd[0] = cylinder1[0] - cylinder0[0];
	cylinderd[1] = cylinder1[1] - cylinder0[1];
	cylinderd[2] = cylinder1[2] - cylinder0[2];
	float dx = tetPositions[indexX] - cylinder0[0];
	float dy = tetPositions[indexY] - cylinder0[1];
	float dz = tetPositions[indexZ] - cylinder0[2];
	float t = cylinderDir[0] * dx + cylinderDir[1] * dy + cylinderDir[2] * dz;

	t /= halfLength; // t是碰撞点在工具上的百分比位置，尖端为0，尾部为1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// 工具中轴上的 接触四面体的投影点->四面体位置，该向量垂直于工具
	dx = tetPositions[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = tetPositions[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = tetPositions[indexZ] - cylinder0[2] - t * cylinderd[2];

	float sqr_distance = dx * dx + dy * dy + dz * dz;
	if (sqr_distance > radius * radius) return;
	float distance = sqrt(sqr_distance);

	// 单位化
	dx /= distance;
	dy /= distance;
	dz /= distance;

	// 计算反馈力，方向为[dx, dy, dz]
	atomicAdd(vertexForce + threadid * 3 + 0, dx * (radius - distance));
	atomicAdd(vertexForce + threadid * 3 + 1, dy * (radius - distance));
	atomicAdd(vertexForce + threadid * 3 + 2, dz * (radius - distance));

	//设置标志位

	isCollide[threadid] = 1;
	zone[threadid] = t;
	//队列索引加一(碰撞信息在计算前缀和的时候赋值)
	atomicAdd(index, 1);
}

__global__ void hapticCalculateCCylinder(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* tetPositions, unsigned int* isCollide, float* zone,int vertexNum, int* index) 
//float* cylinderPos, 工具尖端位置
//float* cylinderDir, 工具方向
//float halfLength, 工具长度
//float radius, 工具半径
//float* tetPositions, 四面体位置
//unsigned int* isCollide, 是否发生碰撞
//float* zone, 碰撞位置
//int vertexNum, 四面体数量
//int* index 每发生一次碰撞，这个值+1
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//重置碰撞标志位
	isCollide[threadid] = 0;
	zone[threadid] = -1;

	__shared__ float cylinder0[3];
	__shared__ float cylinder1[3];
	__shared__ float cylinderd[3];


	cylinder0[0] = cylinderPos[0];
	cylinder0[1] = cylinderPos[1];
	cylinder0[2] = cylinderPos[2];

	cylinder1[0] = cylinderPos[0] + cylinderDir[0] * halfLength;
	cylinder1[1] = cylinderPos[1] + cylinderDir[1] * halfLength;
	cylinder1[2] = cylinderPos[2] + cylinderDir[2] * halfLength;

	cylinderd[0] = cylinder1[0] - cylinder0[0];
	cylinderd[1] = cylinder1[1] - cylinder0[1];
	cylinderd[2] = cylinder1[2] - cylinder0[2];
	float dx = tetPositions[indexX] - cylinder0[0];
	float dy = tetPositions[indexY] - cylinder0[1];
	float dz = tetPositions[indexZ] - cylinder0[2];
	float t = cylinderDir[0] *dx + cylinderDir[1] *dy + cylinderDir[2] *dz;

	t /= halfLength; // t是碰撞点在工具上的百分比位置，尖端为0，尾部为1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// 工具中轴上的 接触四面体的投影点->四面体位置，该向量垂直于工具
	dx = tetPositions[indexX] - cylinder0[0] - t* cylinderd[0];
	dy = tetPositions[indexY] - cylinder0[1] - t* cylinderd[1];
	dz = tetPositions[indexZ] - cylinder0[2] - t* cylinderd[2];

	float distance = dx * dx + dy * dy + dz * dz;
	if (distance > radius*radius) return;
	//设置标志位
	isCollide[threadid] = 1;
	zone[threadid] = t;
	//队列索引加一(碰撞信息在计算前缀和的时候赋值)
	atomicAdd(index, 1);

}

//计算前缀和
__global__ void hapticCalculatePrefixSum(unsigned int* isCollide, unsigned int* queueIndex, unsigned int* auxArray, int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	//编译器这个数组的大小未知
	extern __shared__ unsigned int temp[];


	//给块内共享内存分配数据
	temp[threadIdx.x] = isCollide[threadid];

	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index < blockDim.x)
			temp[index] += temp[index - stride];//index is alway bigger than stride
		__syncthreads();
	}
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < blockDim.x)
			temp[index + stride] += temp[index];

	}
	__syncthreads();

	//更新每个block内的前缀和
	queueIndex[threadid] = temp[threadIdx.x];


	//计算整个block的和
	if (threadid % (blockDim.x - 1) == 0 && threadid != 0) {
		auxArray[blockIdx.x] = queueIndex[threadid];
	}
}

// 根据碰撞结果写入constraint缓冲区
// 参数含义
// unsigned int* isCollide, GPU上并行计算的的碰撞结果
//float* tetPositions, 四面体位置
//float* tetNormals, 四面体法向量
//float* zone, 在工具上碰撞的相对位置
//float* constraintPoints, 输出：发生碰撞的四面体位置
//float* constraintNormals, 输出： 发生碰撞的四面体法向量（朝向物体外侧）
//float* constraintZone, 输出：在工具上碰撞的相对位置
//unsigned int* queueIndex,
//unsigned int* auxArray, 
//int vertexNum
__global__ void hapticAddCollisionToQueue(unsigned int* isCollide, \
	float* tetPositions, float* tetNormals, float* zone, \
	float* constraintPoints, float* constraintNormals, float* constraintZone, \
	unsigned int* queueIndex, unsigned int* auxArray, int vertexNum) 
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	if (isCollide[threadid]) {
		int index = -1;
		//计算index
		for (int block = 0; block < blockIdx.x; block++) {
			index += auxArray[block];
		}

		index += queueIndex[threadid];
		constraintPoints[index * 3 + 0] = tetPositions[threadid * 3 + 0];
		constraintPoints[index * 3 + 1] = tetPositions[threadid * 3 + 1];
		constraintPoints[index * 3 + 2] = tetPositions[threadid * 3 + 2];
		constraintNormals[index * 3 + 2] = tetNormals[threadid * 3 + 2];
		constraintNormals[index * 3 + 0] = tetNormals[threadid * 3 + 0];
		constraintNormals[index * 3 + 1] = tetNormals[threadid * 3 + 1];
		constraintZone[index] = zone[threadid];
	}
}

__global__ void hapticAddCollisionToQueue_SaveMap(unsigned int* isCollide, \
	float* tetPositions, float* tetNormals, float* zone, \
	float* constraintPoints, float* constraintNormals, float* constraintZone, \
	unsigned int* queueIndex, unsigned int* auxArray, int vertexNum, \
	int* collisionQueueIndex_to_vertIndex_array)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	if (isCollide[threadid]) {
		int index = -1;
		//计算index
		for (int block = 0; block < blockIdx.x; block++) {
			index += auxArray[block];
		}

		index += queueIndex[threadid];
		constraintPoints[index * 3 + 0] = tetPositions[threadid * 3 + 0];
		constraintPoints[index * 3 + 1] = tetPositions[threadid * 3 + 1];
		constraintPoints[index * 3 + 2] = tetPositions[threadid * 3 + 2];
		constraintNormals[index * 3 + 2] = tetNormals[threadid * 3 + 2];
		constraintNormals[index * 3 + 0] = tetNormals[threadid * 3 + 0];
		constraintNormals[index * 3 + 1] = tetNormals[threadid * 3 + 1];
		constraintZone[index] = zone[threadid];

		int queueIndex = index;
		int vertIndex = threadid;
		collisionQueueIndex_to_vertIndex_array[index] = vertIndex;
	}
}

__global__ void hapticAddSphereCollisionToQueue(unsigned int* isCollide, float* sphereInfos, float* zone, float* directDirection, float* constraintPoints, float* constraintZone, float* constraintDirection, unsigned int* queueIndex, unsigned int* auxArray, int sphereNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= sphereNum) return;

	if (isCollide[threadid]) {
		int index = -1;
		//计算index
		for (int block = 0; block < blockIdx.x; block++) {
			index += auxArray[block];
		}

		index += queueIndex[threadid];
		//把碰撞的球的位置和半径进行保存
		constraintPoints[index * 4 + 0] = sphereInfos[threadid * 5 + 0];
		constraintPoints[index * 4 + 1] = sphereInfos[threadid * 5 + 1];
		constraintPoints[index * 4 + 2] = sphereInfos[threadid * 5 + 2];
		constraintPoints[index * 4 + 3] = sphereInfos[threadid * 5 + 3];
		constraintZone[index] = zone[threadid];
		constraintDirection[index * 3 + 0] = directDirection[threadid * 3 + 0];
		constraintDirection[index * 3 + 1] = directDirection[threadid * 3 + 1];
		constraintDirection[index * 3 + 2] = directDirection[threadid * 3 + 2];
	}
}

//将数据放置到队列中--Mesh
__global__ void hapticAddSphereCollisionToQueue_Tri(unsigned int* isCollide, float* sphereInfos, float* zone, float* constraintPoints, float* constraintZone, unsigned int* queueIndex, unsigned int* auxArray, int sphereNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= sphereNum) return;

	if (isCollide[threadid]) {
		int index = -1;
		//计算index
		for (int block = 0; block < blockIdx.x; block++) {
			index += auxArray[block];
		}


		index += queueIndex[threadid];
		//把碰撞的球的位置和半径进行保存
		constraintPoints[index * 4 + 0] = sphereInfos[threadid * 4 + 0];
		constraintPoints[index * 4 + 1] = sphereInfos[threadid * 4 + 1];
		constraintPoints[index * 4 + 2] = sphereInfos[threadid * 4 + 2];
		constraintPoints[index * 4 + 3] = sphereInfos[threadid * 4 + 3];
		constraintZone[index] = zone[threadid];
	}
}

__global__ void hapticCalculateContinueCylinder(float startx, float starty, float startz, float endx, float endy, float endz, int* index, float* positions, float* boxs, float* triangleNormal, int boxNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= boxNum) return;

	float start[3] = { startx,starty,startz };
	float end[3] = { endx,endy,endz };

	//先和包围盒进行碰撞检测
	float p0, p1;
	bool collision = hapticLineSegAABBInsect(start, end, &p0, &p1, boxs+threadid * 6);
	if (!collision) return;

	//和三角形进行碰撞检测
	int index0 = index[threadid * 3 + 0];
	int index1 = index[threadid * 3 + 1];
	int index2 = index[threadid * 3 + 2];
	float pos0[3] = { positions[index0 * 3 + 0],positions[index0 * 3 + 1], positions[index0 * 3 + 2] };
	float pos1[3] = { positions[index1 * 3 + 0],positions[index1 * 3 + 1], positions[index1 * 3 + 2] };
	float pos2[3] = { positions[index2 * 3 + 0],positions[index2 * 3 + 1], positions[index2 * 3 + 2] };
	collision = hapticLineSegTriangleInsect(start, end, pos0, pos1, pos2, triangleNormal+threadid * 3 + 0, &p0);
	if (collision) {
		//printf("%d: 被碰撞\n",threadid);
	}
}

//使用圆柱和球进行碰撞检测
__global__ void hapticCalculateCylinderSphere(float* cylinderPos, float* cylinderDir, float halfLength, float radius,float* sphereInfos,float* sphereForce,unsigned int* isCollide,float* zone, int* index,int sphereNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= sphereNum) return;
	//printf("%d\n", threadid);

	int indexX = threadid * 5 + 0;
	int indexY = threadid * 5 + 1;
	int indexZ = threadid * 5 + 2;

	float sphereRadius = sphereInfos[threadid * 5 + 3];

	//重置碰撞标志位
	isCollide[threadid] = 0;
	zone[threadid] = -1;
	//radius *= 1.2;

	__shared__ float cylinder0[3];
	__shared__ float cylinder1[3];
	__shared__ float cylinderd[3];


	cylinder0[0] = cylinderPos[0];
	cylinder0[1] = cylinderPos[1];
	cylinder0[2] = cylinderPos[2];

	cylinder1[0] = cylinderPos[0] + cylinderDir[0] * halfLength;
	cylinder1[1] = cylinderPos[1] + cylinderDir[1] * halfLength;
	cylinder1[2] = cylinderPos[2] + cylinderDir[2] * halfLength;

	cylinderd[0] = cylinder1[0] - cylinder0[0];
	cylinderd[1] = cylinder1[1] - cylinder0[1];
	cylinderd[2] = cylinder1[2] - cylinder0[2];
	float dx = sphereInfos[indexX] - cylinder0[0];
	float dy = sphereInfos[indexY] - cylinder0[1];
	float dz = sphereInfos[indexZ] - cylinder0[2];
	float t = cylinderDir[0] * dx + cylinderDir[1] * dy + cylinderDir[2] * dz;

	t /= halfLength;

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// 工具上的接触点指向接触球中心的向量。
	dx = sphereInfos[indexX] - cylinder0[0] - t* cylinderd[0];
	dy = sphereInfos[indexY] - cylinder0[1] - t* cylinderd[1];
	dz = sphereInfos[indexZ] - cylinder0[2] - t* cylinderd[2];

	float distance = dx * dx + dy * dy + dz * dz;
	if (distance > (sphereRadius+radius)*(sphereRadius+radius)) return;

	//printf("%d:球碰撞\n", threadid);
	//增加球收到的力
	// 标准化向量
	dx /= distance;
	dy /= distance;
	dz /= distance;
	// 力由嵌入深度决定，方向由标准化的指向向量决定。
	atomicAdd(sphereForce + threadid * 3 + 0, dx*(sphereRadius+radius-distance));
	atomicAdd(sphereForce + threadid * 3 + 1, dy*(sphereRadius+radius-distance));
	atomicAdd(sphereForce + threadid * 3 + 2, dz*(sphereRadius+radius-distance));

	//设置标志位
	isCollide[threadid] = 1;
	zone[threadid] = t;
	//printf("sphere index = %d\n", threadid);
	//队列索引加一(碰撞信息在计算前缀和的时候赋值)
	atomicAdd(index, 1);
}

//圆柱和球的碰撞--Mesh
__global__ void hapticCalculateCylinderSphere_Tri(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* sphereInfos, float* sphereForce, unsigned int* isCollide, float* zone, int* index, int sphereNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= sphereNum) return;
	//printf("%d\n", threadid);

	int indexX = threadid * 4 + 0;
	int indexY = threadid * 4 + 1;
	int indexZ = threadid * 4 + 2;

	float sphereRadius = sphereInfos[threadid * 4 + 3];

	//重置碰撞标志位
	isCollide[threadid] = 0;
	zone[threadid] = -1;
	radius *= 2.0;

	__shared__ float cylinder0[3];
	__shared__ float cylinder1[3];
	__shared__ float cylinderd[3];


	cylinder0[0] = cylinderPos[0];
	cylinder0[1] = cylinderPos[1];
	cylinder0[2] = cylinderPos[2];

	cylinder1[0] = cylinderPos[0] + cylinderDir[0] * halfLength;
	cylinder1[1] = cylinderPos[1] + cylinderDir[1] * halfLength;
	cylinder1[2] = cylinderPos[2] + cylinderDir[2] * halfLength;

	cylinderd[0] = cylinder1[0] - cylinder0[0];
	cylinderd[1] = cylinder1[1] - cylinder0[1];
	cylinderd[2] = cylinder1[2] - cylinder0[2];
	float dx = sphereInfos[indexX] - cylinder0[0];
	float dy = sphereInfos[indexY] - cylinder0[1];
	float dz = sphereInfos[indexZ] - cylinder0[2];
	float t = cylinderDir[0] * dx + cylinderDir[1] * dy + cylinderDir[2] * dz;

	t /= halfLength;

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	dx = sphereInfos[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = sphereInfos[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = sphereInfos[indexZ] - cylinder0[2] - t * cylinderd[2];

	float distance = dx * dx + dy * dy + dz * dz;
	if (distance > (sphereRadius + radius)*(sphereRadius + radius)) return;

	//printf("%d:球碰撞\n", threadid);
	//增加球收到的力
	dx /= distance;
	dy /= distance;
	dz /= distance;
	atomicAdd(sphereForce + threadid * 3 + 0, dx*(sphereRadius + radius - distance));
	atomicAdd(sphereForce + threadid * 3 + 1, dy*(sphereRadius + radius - distance));
	atomicAdd(sphereForce + threadid * 3 + 2, dz*(sphereRadius + radius - distance));

	//设置标志位
	isCollide[threadid] = 1;
	zone[threadid] = t;
	//队列索引加一(碰撞信息在计算前缀和的时候赋值)
	atomicAdd(index, 1);
}

//线段和AABB包围盒的求交
__device__ bool hapticLineSegAABBInsect(float* start, float* end,float* p0,float* p1, float* boxs) {

	float dir[3] = {end[0]-start[0],end[1]-start[1],end[3]-start[3]};


	//获取包围盒的交点
	float minx = boxs[0];
	float miny = boxs[1];
	float minz = boxs[2];
	float maxx = boxs[3];
	float maxy = boxs[4];
	float maxz = boxs[5];


	//获取和三组平面的交点
	float t0x = (minx - start[0]) / dir[0];
	float t1x = (maxx - start[0]) / dir[0];
	if (t0x > t1x) hapticSwap(&t0x, &t1x);
	float t0y = (miny - start[1]) / dir[1];
	float t1y = (maxy - start[1]) / dir[1];
	if (t0y > t1y) hapticSwap(&t0y, &t1y);
	float t0z = (minz - start[2]) / dir[2];
	float t1z = (maxz - start[2]) / dir[2];
	if (t0z > t1z) hapticSwap(&t0z, &t1z);

	//找到相交部分的点对
	float t0 = (t0x < t0y) ? t0y : t0x;
	float t1 = (t1x < t1y) ? t1x : t1y;
	t0 = (t0 > t0z) ? t0 : t0z;
	t1 = (t1 > t1z) ? t1z : t1;

	//获取解，但是要clamp到01之间
	*p1 = t0;
	*p0 = t1;

	if (*p0 > *p1) return false;

	//如果和包围盒相交，还需要进一步判断是否在01之间
	*p0 = hapticClamp(*p0, 0, 1);
	*p1 = hapticClamp(*p1, 0, 1);

	//如果两者相同，也是不相交的
	if (abs(*p0 - *p1) < 0.0001) return false;

	return true;
}


//线段和三角形的求交
__device__ bool hapticLineSegTriangleInsect(float* start, float* end, float* pos0, float* pos1, float* pos2, float* triangleNormal,float* ans) {
	//先计算和平面的交点
	float insectPoint[3];

	//辅助向量
	float v[3] = {pos0[0]-start[0],pos0[1] - start[1], pos0[2] - start[2]};

	float dir[3] = { end[0] - start[0],end[1] - start[1],end[3] - start[3] };
	float length = sqrt( dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] );
	dir[0] /= length;
	dir[1] /= length;
	dir[2] /= length;

	float dotNV = triangleNormal[0] * v[0] + triangleNormal[1] * v[1] + triangleNormal[2] * v[2];
	float dotND = triangleNormal[0] * dir[0] + triangleNormal[1] * dir[1] + triangleNormal[2] * dir[2];
	float t = dotNV/ dotND;


	//首先判断是否在线段内部
	if (t<0 || t> length) return false;

	float p[3] = { start[0] + t*dir[0],start[1] + t*dir[1], start[2] + t*dir[2] };

	//再判断点在不在三角形内,使用叉乘法
	float cross0[3];
	float v0[3] = { pos1[0] - pos0[0],pos1[1] - pos0[1], pos1[2] - pos0[2] };
	float v0p[3] = { p[0] - pos0[0],p[1] - pos0[1], p[2] - pos0[2] };
	hapticCross(v0, v0p, cross0);

	float cross1[3];
	float v1[3] = { pos2[0] - pos1[0],pos2[1] - pos1[1], pos2[2] - pos1[2] };
	float v1p[3] = { p[0] - pos1[0],p[1] - pos1[1], p[2] - pos1[2] };
	hapticCross(v1, v1p, cross1);

	float cross2[3];
	float v2[3] = { pos0[0] - pos2[0],pos0[1] - pos2[1], pos0[2] - pos2[2] };
	float v2p[3] = { p[0] - pos2[0],p[1] - pos2[1], p[2] - pos2[2] };
	hapticCross(v1, v1p, cross1);

	//如果出现方向反转,认为不相交
	float flag = hapticDot(cross0,cross1);
	if (flag < 0) return false;
	flag = hapticDot(cross1, cross2);
	if (flag < 0) return false;

	*ans = t;
	return true;
}

// 把碰撞信息传到
__global__ void dispatchForceToTetVertex(
	float* externForce,
	float* vertexForce,
	unsigned int* isCollide,
	int vertexNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	if (isCollide[threadid] == 0) return;

	atomicAdd(externForce + indexX, vertexForce[indexX]);
	atomicAdd(externForce + indexY, vertexForce[indexY]);
	atomicAdd(externForce + indexZ, vertexForce[indexZ]);

	// 清空顶点上的外力。
	vertexForce[indexY]	= 0.0;
	vertexForce[indexZ]	= 0.0;
	vertexForce[indexX]	= 0.0;
}
// 把碰撞球上的力作为外力施加到绑定的顶点上。
__global__ void dispatchToTet(
	unsigned int* skeletonIndex, 
	float* skeletonCoord,
	float* externForce,
	float* sphereForce, 
	unsigned int* isCollide, 
	int sphereNum) 
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= sphereNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//如果没有碰到就离开
	if (isCollide[threadid] == 0) return;

	//找四面体的顶点索引和权重
	int tetIndex0 = skeletonIndex[threadid * 4 + 0];
	int tetIndex1 = skeletonIndex[threadid * 4 + 1];
	int tetIndex2 = skeletonIndex[threadid * 4 + 2];
	int tetIndex3 = skeletonIndex[threadid * 4 + 3];

	float weight0 = skeletonCoord[threadid * 4 + 0];
	float weight1 = skeletonCoord[threadid * 4 + 1];
	float weight2 = skeletonCoord[threadid * 4 + 2];
	float weight3 = skeletonCoord[threadid * 4 + 3];

	//
	atomicAdd(externForce + tetIndex0 * 3 + 0, sphereForce[indexX] * weight0);
	atomicAdd(externForce + tetIndex0 * 3 + 1, sphereForce[indexY] * weight0);
	atomicAdd(externForce + tetIndex0 * 3 + 2, sphereForce[indexZ] * weight0);

	atomicAdd(externForce + tetIndex1 * 3 + 0, sphereForce[indexX] * weight1);
	atomicAdd(externForce + tetIndex1 * 3 + 1, sphereForce[indexY] * weight1);
	atomicAdd(externForce + tetIndex1 * 3 + 2, sphereForce[indexZ] * weight1);

	atomicAdd(externForce + tetIndex2 * 3 + 0, sphereForce[indexX] * weight2);
	atomicAdd(externForce + tetIndex2 * 3 + 1, sphereForce[indexY] * weight2);
	atomicAdd(externForce + tetIndex2 * 3 + 2, sphereForce[indexZ] * weight2);

	atomicAdd(externForce + tetIndex3 * 3 + 0, sphereForce[indexX] * weight3);
	atomicAdd(externForce + tetIndex3 * 3 + 1, sphereForce[indexY] * weight3);
	atomicAdd(externForce + tetIndex3 * 3 + 2, sphereForce[indexZ] * weight3);

}

__global__ void hapticUpdatePointPosition(
	float* mass,
	float* position,
	float* velocity,
	float* forceFromTool,
	float dt,
	int point_num)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= point_num) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	float forceX = forceFromTool[indexX];
	float forceY = forceFromTool[indexY];
	float forceZ = forceFromTool[indexZ];
	// acceleration of points due to external force
	float a_x = forceX / mass[threadid];
	float a_y = forceY / mass[threadid];
	float a_z = forceZ / mass[threadid];

	//printf("mass in thread %d: %f\tforceX: %f\n", threadid, mass[threadid], forceX);
	if (isnan(a_x))
	{
		if (isnan(forceX))
		{
			printf("%d-nan occured in force_x\n", threadid);
		}
		else if (isnan(mass[indexX]))
			printf("nan occured in mass_x\n");
	}

	//// update velocity using force and mass
	float delta_v_x = a_x * dt;
	float delta_v_y = a_y * dt;
	float delta_v_z = a_z * dt;

	float delta_pos_x = velocity[indexX] * dt + 0.5 * a_x * dt * dt;
	float delta_pos_y = velocity[indexX] * dt + 0.5 * a_y * dt * dt;
	float delta_pos_z = velocity[indexX] * dt + 0.5 * a_z * dt * dt;

	float delta_pos_len = sqrt(delta_pos_x * delta_pos_x + delta_pos_y * delta_pos_y + delta_pos_z * delta_pos_z);
	if(delta_pos_len>1)
		printf("%d-delta_pos_len:%f\n", threadid, delta_pos_len);
	atomicAdd(position + indexX, delta_pos_x);
	atomicAdd(position + indexY, delta_pos_y);
	atomicAdd(position + indexZ, delta_pos_z);
	atomicAdd(velocity + indexX, delta_v_x);
	atomicAdd(velocity + indexY, delta_v_y);
	atomicAdd(velocity + indexZ, delta_v_z);
}

__global__ void AccumulateExternForce(float* externForceTotal, float* externForce, int point_num)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= point_num) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	externForceTotal[indexX] += externForce[indexX];
	externForceTotal[indexY] += externForce[indexY];
	externForceTotal[indexZ] += externForce[indexZ];
}

//工具扫描体碰撞检测，两个工具位置张成的扫描体的碰撞检测。
__device__ bool hapticCylinderCollisionContinue(
	float length, // 工具长度 
	float moveDistance, // 定义扫描体起止点的两个工具位置之间的距离 
	float radius,// 工具半径
	float* cylinderPos, //扫描体重点的工具位置
	float* cylinderLastPos, // 扫描体起点的工具位置
	float* cylinderDir, // 工具方向
	float* moveDir, //工具起点到终点的运动方向
	float* position, // 需要进行碰撞检测的顶点位置
	float* collisionNormal, //碰撞之后顶点被排出的方向
	float* collisionPos // 顶点被排出到扫描体表面的位置
)
{
	//首先计算出运动平面的法线向量
	float normal[3];
	tetCross_D(cylinderDir, moveDir, normal);
	tetNormal_D(normal);

	//定义计算需要的变量
	float VSubO[3] = { position[0] - cylinderPos[0] ,position[1] - cylinderPos[1] ,position[2] - cylinderPos[2] };//工具尖端指向可能碰撞点的向量
	float lineStart0[3] = { cylinderPos[0] ,cylinderPos[1] ,cylinderPos[2] };// 当前工具尖端
	float lineStart1[3] = { cylinderLastPos[0] ,cylinderLastPos[1] ,cylinderLastPos[2] };// 上一帧工具尖端
	float lineStart2[3] = { cylinderPos[0] + cylinderDir[0] * length ,cylinderPos[1] + cylinderDir[1] * length,cylinderPos[2] + cylinderDir[2] * length };// 当前帧工具尾部


	//首先要先进行一次碰撞检测，物理工具是否发生碰撞，若不发生则不需要进行碰撞检测


	//1.计算在局部坐标系中的坐标，由于这个局部坐标不是正交的，所以不能和轴进行点积，使用高斯消元
	// 三个轴： 工具方向cylinderDir，运动方向moveDir，工具方向与运动方向张成的平面的法向量normal
	// 计算碰撞点在这三个轴组成的局部坐标系坐标[x, y, z] 用处：在这个非正交坐标系下，x就是碰撞点在工具上投影的位置，y就是在运动方向上的运动距离
	// 高斯消元：[A|I] 只使用行之间的加减，A变成I，I会变成A的逆矩阵。
	float x, y, z;
	float det = tetSolveFormula_D(cylinderDir, moveDir, normal, VSubO, &x, &y, &z);

	if (x != x || y != y || z != z) return false;

	float distance = 0.0;
	bool flag = false;
	//2.根据坐标找到顶点所在的区域
	if (x > length && y > moveDistance) {
		//计算点到点的距离
		float basePoint[3] = { cylinderLastPos[0] + length * cylinderDir[0],cylinderLastPos[1] + length * cylinderDir[1] , cylinderLastPos[2] + length * cylinderDir[2] };
		distance = tetPointPointDistance_D(position, basePoint);
		flag = true;
	}
	else if (x > length && y < moveDistance && y>0.0) {
		//计算点到边的距离
		distance = tetPointLineDistance_D(lineStart2, moveDir, position);
	}
	else if (x > length && y < 0.0) {
		distance = tetPointPointDistance_D(position, lineStart2);
	}
	else if (x > 0.0 && x < length && y > moveDistance) {
		distance = tetPointLineDistance_D(lineStart1, cylinderDir, position);
		flag = true;
	}
	else if (x > 0.0 && x < length && y < moveDistance && y>0.0) {
		//计算点到面的距离
		distance = abs(z);
	}
	else if (x > 0.0 && x < length && y < 0.0) {
		distance = tetPointLineDistance_D(lineStart0, cylinderDir, position);
	}
	else if (x<0.0 && y > moveDistance) {
		distance = tetPointPointDistance_D(position, cylinderLastPos);
		flag = true;
	}
	else if (x < 0.0 && y < moveDistance && y>0.0) {
		distance = tetPointLineDistance_D(lineStart0, moveDir, position);
	}
	else if (x < 0.0 && y < 0.0) {
		distance = tetPointPointDistance_D(position, cylinderPos);
	}


	//3.判断距离
	if (distance > radius) return false;
	//if (flag) return false;

	//printf("x:%f,y:%f,z:%f\n", x, y, z);

	//4. 计算矫正的碰撞排除位置
	//求解二元一次方程,和两个方向的圆柱进行计算
	float lineDir[3] = { moveDir[0],moveDir[1], moveDir[2] };


	float v0[3] = { position[0] - lineStart0[0] ,position[1] - lineStart0[1] ,position[2] - lineStart0[2] };
	float v1[3] = { position[0] - lineStart1[0] ,position[1] - lineStart1[1] ,position[2] - lineStart1[2] };
	float v2[3] = { position[0] - lineStart2[0] ,position[1] - lineStart2[1] ,position[2] - lineStart2[2] };


	//和圆柱相交
	float solve00, solve01;
	float solve10, solve11;
	tetSolveInsect_D(lineDir, cylinderDir, v0, radius, &solve00, &solve01);
	tetSolveInsect_D(lineDir, moveDir, v0, radius, &solve10, &solve11);
	float solve = min(solve11, solve01);
	//tetSolveInsect_D(lineDir, cylinderDir, v1, radius, &solve00, &solve01);
	//solve = min(solve, solve01);
	//tetSolveInsect_D(lineDir, moveDir, v2, radius, &solve00, &solve01);
	//solve = min(solve, solve01);


	//和球相交
	float solve20, solve21;
	tetSolveInsectSphere_D(lineDir, v0, radius, &solve20, &solve21);
	solve = min(solve, solve21);
	//printf("%f\n", solve);
	//tetSolveInsectSphere_D(lineDir, v1, radius, &solve10, &solve11);
	//solve = min(solve, solve11);
	//tetSolveInsectSphere_D(lineDir, v2, radius, &solve10, &solve11);
	//solve = min(solve, solve11);
	//tetSolveInsectSphere_D(lineDir, VSubO, radius, &solve10, &solve11);
	//solve = min(solve, solve11);

	if (solve != solve) return false;
	//printf("x:%f,y:%f,z:%f, solve: %f\n",x,y,z, solve);

	//更新位置得到顶点排出的位置
	collisionPos[0] = position[0] - lineDir[0] * solve;
	collisionPos[1] = position[1] - lineDir[1] * solve;
	collisionPos[2] = position[2] - lineDir[2] * solve;

	//更新顶点的碰撞法线，向工具轴线上进行投影
	float projPos[3] = { collisionPos[0] - cylinderPos[0],collisionPos[1] - cylinderPos[1],collisionPos[2] - cylinderPos[2] };
	float proj = tetDot_D(projPos, cylinderDir);
	projPos[0] = collisionPos[0] - cylinderPos[0] - cylinderDir[0] * proj;
	projPos[1] = collisionPos[1] - cylinderPos[1] - cylinderDir[1] * proj;
	projPos[2] = collisionPos[2] - cylinderPos[2] - cylinderDir[2] * proj;

	tetNormal_D(projPos);
	collisionNormal[0] = projPos[0];
	collisionNormal[1] = projPos[1];
	collisionNormal[2] = projPos[2];

	//printf("continue: x:%f,y:%f,z:%f,solve:%f\n", collisionPos[0], collisionPos[1], collisionPos[2],solve);
	//printf("continue: nx:%f,ny:%f,nz:%f\n", collisionNormal[0], collisionNormal[1], collisionNormal[2]);
	return true;
}

__device__ float ContactForceDecay(float distance, float original_radius, float max_radius)
// distance: distance between point and tool central axis
// original_radius: tool actual radius
// max_radius: the radius of exerting tool force
{
	float f1 = -1 / original_radius * distance + 1;
	float f2 = -1 / max_radius * distance + 1;
	float t = distance / max_radius;
	float scale = (1 - t) * f1 + t * f2;
	//printf("scale: %f\n", scale);
	return scale;
}

__device__ void hapticSwap(float* a, float* b) {
	float temp = *a;
	*a = *b;
	*b = temp;
}
__device__ float hapticClamp(float a, float min, float max) {
	return a<min ? min : (a>max ? max : a);
}

__device__ void hapticCross(float* a, float* b, float* c) {
	//叉乘计算三角形法线
	c[0] = a[1] * b[2] - b[1] * a[2];
	c[1] = a[2] * b[0] - b[2] * a[0];
	c[2] = a[0] * b[1] - b[0] * a[1];
}

__device__ float hapticDot(float* a, float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

