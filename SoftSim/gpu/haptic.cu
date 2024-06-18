#include "gpuvar.h"
#include "gpufun.h"
float*			hapticDeformationCollisionForce_D; // UNUSED
float*			hapticDeformationInterpolatePositions_D;// UNUSED
float*			hapticDeformationExternForce_D;    // �ش������������洢ÿ���������ɹ���ʩ�ӵĳͷ�����
float*			hapticDeformationExternForceTotal_D; // �ۼӹ���ʩ�ӵ������ϵ�������applyForce��ʱ����ա�
int				hapticCounter_D;// ����һ������֮֡�����˶��ٸ�������֡�ļ�������
float*			hapticDeformationPrePositions_D;	//���ڲ�ֵ����
float*			hapticDeformationPositions_D;	//�������˽�����ײ���ı��������ӣ���Ҫ��ʱͬ��
float*			hapticDeformationNormals_D;     // �������˽�����ײ���������巨��������Ҫ��ʱͬ��
float*			hapticCollisionZone_D;			//��¼������ײ����������������߶Ρ�������������ò�ͬ��Լ������

int* hapticContinuousFrameNumOfCollision_D;     // ��¼�ö��㱻����ʩ��ѹ��������֡����

int				hapticDeformationNum_D;			//����������ȫ�������嶥��������������ڲ������嶥�㡣�����еĻ�����λ�������嶥�㣬�����ӡ�
int				hapticDeformationNumMem_D;



unsigned int*	hapticIsCollide_D;
float*			hapticConstraintForce_D;
float*			hapticConstraintPoints_D;	//�洢���յ���ײ������
float*			hapticConstraintNormals_D;
float*          hapticConstraintZone_D;

float*			hapticCylinderPos_D;
float*			hapticCylinderPhysicalPos_D;
float*			hapticCylinderDir_D;
int*			hapticIndex_D;				//�洢��ײ������������

unsigned int*	hapticQueueIndex_D;
unsigned int*	hapticAuxSumArray_D;
int* haptic_collisionIndex_to_vertIndex_array_D; //��ײ�����±��Ӧ�Ķ����±�

//����1��ʹ�ñ��������ε�������ײ���
int				hapticAABBBoxNum_D;//���������ε�������ÿ�����������ζ���Ӧһ��AABB��Χ�С�
float*			hapticAABBBoxs_D;
float*			hapticTriangleNormal_D;// ���������εķ�����
int*			hapticSurfaceIndex_D; // ���������ζ���������*�����ڲ���*�����嶥���е��±ꡣ

//����2��ʹ������
int				hapticSphereNum_D;
float*			hapticSphereInfo_D;
float*			hapticSphereDirectDirection_D;	//���ָ������
float*			hapticSphereForce_D;	//���յ�����ײ��
unsigned int*	hapticSphereIsCollide_D;
float*			hapticSphereCollisionZone_D;
int*			hapticSphereindex_D;
float*			hapticSphereConstraintPoints_D;
float*          hapticSphereConstraintZone_D;
float*			hapticSphereConstraintDirection_D;  //Լ��ָ������
unsigned int*	hapticSphereTetIndex_D;
float*			hapticSphereTetCoord_D;



//������ײ����
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
// �����ײ����
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

//�������˽�����ײ��⣬ͨ����ѹ���洢��ײ��Ϣ������֮��������⹤��λ�ˡ�

int runHapticCollision(float halfLength, float radius) {
	int  threadNum = 512;
	int blockNum = (hapticDeformationNum_D + threadNum - 1) / threadNum;
	//����ײ�������

	cudaMemset(hapticIndex_D, -1, sizeof(int));
	
	float obj_r = 0.05f;
	float extended_radius = radius + obj_r;

	//���м�����ײ
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

	//�õ���ײ��֮�󣬼���ǰ׺�͵õ��ڶ����е�����(�����������ǹ����ڴ��С)
	hapticCalculatePrefixSum << <blockNum, threadNum, threadNum *sizeof(unsigned int) >> > (hapticIsCollide_D, hapticQueueIndex_D, hapticAuxSumArray_D, hapticDeformationNum_D);
	//�ٸ�����������д��ײ�㵽������
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

//�������˵�������ײ���
int runHapticContinueCollision(float* start,float* end,float halfLength, float radius) {
	int  threadNum = 512;
	int blockNum = (hapticAABBBoxNum_D + threadNum - 1) / threadNum;

	//�߶ε���ײ���
	hapticCalculateContinueCylinder << <blockNum, threadNum >> >(
		start[0], start[1], start[2],
		end[0], end[1], end[2],
		hapticSurfaceIndex_D, hapticDeformationPositions_D, hapticAABBBoxs_D, hapticTriangleNormal_D, hapticAABBBoxNum_D);
	
	return 0;
}


int runHapticCollisionSphere(float halfLength, float radius) {

	int  threadNum = 512;
	int blockNum = (hapticSphereNum_D + threadNum - 1) / threadNum;


	//����ײ�������
	cudaMemset(hapticSphereindex_D, -1, sizeof(int));

	//Բ���������ײ���
	hapticCalculateCylinderSphere << <blockNum, threadNum >> >(hapticCylinderPos_D, hapticCylinderDir_D, halfLength, radius, hapticSphereInfo_D,hapticSphereForce_D, hapticSphereIsCollide_D,hapticSphereCollisionZone_D,hapticSphereindex_D, hapticSphereNum_D);

	//ͬ��ʹ��ǰ׺�ͽ���ײ������õ���ײ������
	//�õ���ײ��֮�󣬼���ǰ׺�͵õ��ڶ����е�����(�����������ǹ����ڴ��С)
	hapticCalculatePrefixSum << <blockNum, threadNum, threadNum *sizeof(unsigned int) >> > (hapticSphereIsCollide_D, hapticSphereQueueIndex_D, hapticSphereAuxSumArray_D, hapticSphereNum_D);
	//�ٸ�����������д��ײ�㵽������
	hapticAddSphereCollisionToQueue << <blockNum, threadNum >> > (hapticSphereIsCollide_D, hapticSphereInfo_D, hapticSphereCollisionZone_D, hapticSphereDirectDirection_D, hapticSphereConstraintPoints_D, hapticSphereConstraintZone_D, hapticSphereConstraintDirection_D, hapticSphereQueueIndex_D, hapticSphereAuxSumArray_D, hapticSphereNum_D);

	cudaDeviceSynchronize();

	return 0;
}

int runHapticCollisionSphere_Tri(float halfLength, float radius) {
	int threadNum = 512;
	int blockNum = (hapticSphereNum_D + threadNum - 1) / threadNum;

	//����ײ�������
	cudaMemset(hapticSphereindex_D, -1, sizeof(int));

	//Բ���������ײ���
	hapticCalculateCylinderSphere_Tri<< <blockNum, threadNum>> >(hapticCylinderPos_D, hapticCylinderDir_D, halfLength, radius, hapticSphereInfo_D, hapticSphereForce_D, hapticSphereIsCollide_D, hapticSphereCollisionZone_D, hapticSphereindex_D, hapticSphereNum_D);

	//ͬ��ʹ��ǰ׺�ͽ���ײ������õ���ײ������
	//�õ���ײ��֮�󣬼���ǰ׺�͵õ��ڶ����е�����(�����������ǹ����ڴ��С)
	hapticCalculatePrefixSum << <blockNum, threadNum, threadNum * sizeof(unsigned int) >> > (hapticSphereIsCollide_D, hapticSphereQueueIndex_D, hapticSphereAuxSumArray_D, hapticSphereNum_D);
	//�ٸ�����������д��ײ�㵽����
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
// ���������ݻر��ζˣ���Ϊ����ʩ�ӵ�������
int runDispatchForceToTetVertex()
{
	int  threadNum = 512;
	int blockNum = (hapticSphereNum_D + threadNum - 1) / threadNum;

	dispatchForceToTetVertex << <blockNum, threadNum >> > (hapticDeformationExternForce_D, hapticVertexForceOrthogonalToTool_D, hapticIsCollide_D, hapticDeformationNum_D);
	cudaDeviceSynchronize();
	return 0;
}
//�����������Ϣ���ݵ������嶥�㣨deprecated��
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
	float* vertexForce, // �ӹ���ָ����ײ��ķ�����
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

	//������ײ��־λ
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


	//if (len_normal < 0.1)// ������Ϊ0���õ�Ϊ�����ڲ��Ķ��㣬��������ײ����������ļ��㡣
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

	t /= halfLength; // t����ײ���ڹ����ϵİٷֱ�λ�ã����Ϊ0��β��Ϊ1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// ���������ϵ� �Ӵ��������ͶӰ��->������λ��
	// ���Ӵ����ڹ��߸��ϵ�ʱ�򣬸�������ֱ�ڹ��������ߣ��ӹ��������ϵ�ͶӰ��ָ��Ӵ��㡣
	// ���Ӵ����ڹ��߼�˵�ʱ����������ӹ�������ļ��ָ��Ӵ��㡣
	dx = tetPositions[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = tetPositions[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = tetPositions[indexZ] - cylinder0[2] - t * cylinderd[2];

	float sqr_distance = dx * dx + dy * dy + dz * dz;
	float distance = sqrt(sqr_distance);
	dx /= distance; dy /= distance; dz /= distance;
	// ���������⹤�������ϵ�ͶӰ��
	float p0[3] = {
		cylinder0[0] + t * cylinderd[0],
		cylinder0[1] + t * cylinderd[1],
		cylinder0[2] + t * cylinderd[2] };
	// �����������������ϵ�ͶӰ��
	float p1[3] = {
		hapticCylinderTip[0] + t * cylinderd[0],
		hapticCylinderTip[1] + t * cylinderd[1],
		hapticCylinderTip[2] + t * cylinderd[2] };
	// �����⹤���ϵ�ͶӰ��ָ����������ͶӰ�������
	float v[3] = { p1[0] - p0[0], p1[1] - p0[1] ,p1[2] - p0[2] };
	float gh_distance = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

	const float GH_DISTANCE_THREASHOLD = 0.25;
	if (gh_distance < GH_DISTANCE_THREASHOLD)// ֱ����ͼ��λ������ײ���
	{
		if (distance < radius)//�������Ľ����Ҫд����ײ�������ڼ������⹤��λ��
		{
			if (isOnSurface)
			{
				// ���㹤���ڶ�����ʩ�ӵ�����������Ϊ[-nx, -ny, -nz](���涥�㷨�����ķ�����)
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
			//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
			atomicAdd(index, 1);

			//// printf("���⹤�߰뾶��Χ�ڷ�����ײ�� threadid:%d counter: %d\n", threadid, continuousFrameCounter[threadid]);
			// ���߶Զ���ʩ����������ǰ�������ײ����֡����+1
			if (continuousFrameCounter[threadid] < max_continuous_frame)
			{
				continuousFrameCounter[threadid] += 1;
			}
		}
		else
		{
			// δʩ��������ǰ�������ײ����֡����-1
			if (continuousFrameCounter[threadid] > 0)
			{
				continuousFrameCounter[threadid] -= 1;
			}
		}
	}
	else
	{
		// moveDirָ���ǡ�������λ�˶��뵽���⹤��λ�˵��ƶ�������
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
			//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
			atomicAdd(index, 1);

			// ���߶Զ���ʩ����������ǰ�������ײ����֡����+1
			if (continuousFrameCounter[threadid] < max_continuous_frame)
			{
				continuousFrameCounter[threadid] += 1;
			}
		}
		else
		{
			// δʩ��������ǰ�������ײ����֡����-1
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
	// moveDirָ�������߶��뵽���⹤����Ҫ�ƶ���������������λ��ָ������λ�˵�����
	float moveDir[3] = { SPos[0] - HPos[0],SPos[1] - HPos[1],SPos[2] - HPos[2] };
	float moveDistance = sqrt(moveDir[0] * moveDir[0] + moveDir[1] * moveDir[1] + moveDir[2] * moveDir[2]);
	//���ȼ�����˶�ƽ��ķ�������
	float normal[3];
	tetCross_D(cylinderDir, moveDir, normal);
	tetNormal_D(normal);


	//���������Ҫ�ı���
	float VSubO[3] = { position[0] - HPos[0] ,position[1] - HPos[1] ,position[2] - HPos[2] };//���߼��ָ�������ײ�������
	float lineStart0[3] = { HPos[0] ,HPos[1] ,HPos[2] };// ��ǰ���߼��
	float lineStart1[3] = { SPos[0] ,SPos[1] ,SPos[2] };// ��һ֡���߼��
	float lineStart2[3] = { HPos[0] + cylinderDir[0] * length ,HPos[1] + cylinderDir[1] * length,HPos[2] + cylinderDir[2] * length };// ��ǰ֡����β��


	//����Ҫ�Ƚ���һ����ײ��⣬�������Ƿ�����ײ��������������Ҫ������ײ���


	//1.�����ھֲ�����ϵ�е����꣬��������ֲ����겻�������ģ����Բ��ܺ�����е����ʹ�ø�˹��Ԫ
	// �����᣺ ���߷���cylinderDir���˶�����moveDir�����߷������˶������ųɵ�ƽ��ķ�����normal
	// ������ײ��������������ɵľֲ�����ϵ����[x, y, z] �ô������������������ϵ�£�x������ײ���ڹ�����ͶӰ��λ�ã�y�������˶������ϵ��˶�����
	// ��˹��Ԫ��[A|I] ֻʹ����֮��ļӼ���A���I��I����A�������
	float x, y, z;
	float det = tetSolveFormula_D(cylinderDir, moveDir, normal, VSubO, &x, &y, &z);

	if (x != x || y != y || z != z) return false;

	float distance = 0.0;
	bool flag = false;
	//2.���������ҵ��������ڵ�����
	if (x > length && y > moveDistance) {
		//����㵽��ľ���
		float basePoint[3] = { SPos[0] + length * cylinderDir[0],SPos[1] + length * cylinderDir[1] , SPos[2] + length * cylinderDir[2] };
		distance = tetPointPointDistance_D(position, basePoint);
		flag = true;
	}
	else if (x > length && y < moveDistance && y>0.0) {
		//����㵽�ߵľ���
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
		//����㵽��ľ���
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


	//3.�жϾ���
	if (distance > radius) return false;
	//if (flag) return false;

	//printf("x:%f,y:%f,z:%f\n", x, y, z);

	//4. �����������ײ�ų�λ��
	//����Ԫһ�η���,�����������Բ�����м���
	float lineDir[3] = { moveDir[0],moveDir[1], moveDir[2] };


	float v0[3] = { position[0] - lineStart0[0] ,position[1] - lineStart0[1] ,position[2] - lineStart0[2] };
	float v1[3] = { position[0] - lineStart1[0] ,position[1] - lineStart1[1] ,position[2] - lineStart1[2] };
	float v2[3] = { position[0] - lineStart2[0] ,position[1] - lineStart2[1] ,position[2] - lineStart2[2] };


	//��Բ���ཻ
	float solve00, solve01;
	float solve10, solve11;
	tetSolveInsect_D(lineDir, cylinderDir, v0, radius, &solve00, &solve01);
	tetSolveInsect_D(lineDir, moveDir, v0, radius, &solve10, &solve11);
	float solve = min(solve11, solve01);
	//tetSolveInsect_D(lineDir, cylinderDir, v1, radius, &solve00, &solve01);
	//solve = min(solve, solve01);
	//tetSolveInsect_D(lineDir, moveDir, v2, radius, &solve00, &solve01);
	//solve = min(solve, solve01);


	//�����ཻ
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

	//����λ�õõ������ų���λ��
	collisionPos[0] = position[0] - lineDir[0] * solve;
	collisionPos[1] = position[1] - lineDir[1] * solve;
	collisionPos[2] = position[2] - lineDir[2] * solve;
	printf("solve01 solve11 solve21:%f %f %f\np[%f %f %f] collisionP[%f %f %f]\n", solve01, solve11, solve21, position[0], position[1], position[2], collisionPos[0], collisionPos[1], collisionPos[2]);

	//���¶������ײ���ߣ��򹤾������Ͻ���ͶӰ
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

// �������嶥�����ײ��⣬�������⹤����������֮��ľ���ϴ�ʱ���������⹤����������֮���ɨ��������ײ��⣬�����ӹ��߶������嶥��ʩ��ѹ���ķ�Χ��.
__global__ void hapticCalculateMeshCylinder(
	float* cylinderPos,
	float* hapticCylinderPos,
	float* cylinderDir, float halfLength, float radius, 
	float* tetPositions, 
	float* vertexNormals, 
	unsigned int* isCollide, 
	float* vertexForce, // �ӹ���ָ����ײ��ķ�����
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

	//������ײ��־λ
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
		
	
	//if (len_normal < 0.1)// ������Ϊ0���õ�Ϊ�����ڲ��Ķ��㣬��������ײ����������ļ��㡣
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

	t /= halfLength; // t����ײ���ڹ����ϵİٷֱ�λ�ã����Ϊ0��β��Ϊ1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// ���������ϵ� �Ӵ��������ͶӰ��->������λ��
	// ���Ӵ����ڹ��߸��ϵ�ʱ�򣬸�������ֱ�ڹ��������ߣ��ӹ��������ϵ�ͶӰ��ָ��Ӵ��㡣
	// ���Ӵ����ڹ��߼�˵�ʱ����������ӹ�������ļ��ָ��Ӵ��㡣
	dx = tetPositions[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = tetPositions[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = tetPositions[indexZ] - cylinder0[2] - t * cylinderd[2];

	float sqr_distance = dx * dx + dy * dy + dz * dz;
	float distance = sqrt(sqr_distance);
	dx /= distance; dy /= distance; dz /= distance;
	// ���������⹤�������ϵ�ͶӰ��
	float p0[3] = {
		cylinder0[0] + t * cylinderd[0],
		cylinder0[1] + t * cylinderd[1],
		cylinder0[2] + t * cylinderd[2] };
	// �����������������ϵ�ͶӰ��
	float p1[3] = {
		hapticCylinderTip[0] + t * cylinderd[0],
		hapticCylinderTip[1] + t * cylinderd[1],
		hapticCylinderTip[2] + t * cylinderd[2] };
	// �����⹤���ϵ�ͶӰ��ָ����������ͶӰ�������
	float v[3] = { p1[0] - p0[0], p1[1] - p0[1] ,p1[2] - p0[2] };
	float gh_distance = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

	const float GH_DISTANCE_THREASHOLD = 0.25;
	if (gh_distance < GH_DISTANCE_THREASHOLD)// ֱ����ͼ��λ������ײ���
	{
		if (distance < radius)//�������Ľ����Ҫд����ײ�������ڼ������⹤��λ��
		{
			if (isOnSurface)
			{
				// ���㹤���ڶ�����ʩ�ӵ�����������Ϊ[-nx, -ny, -nz](���涥�㷨�����ķ�����)
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
			//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
			atomicAdd(index, 1);

			//// printf("���⹤�߰뾶��Χ�ڷ�����ײ�� threadid:%d counter: %d\n", threadid, continuousFrameCounter[threadid]);
			// ���߶Զ���ʩ����������ǰ�������ײ����֡����+1
			if(continuousFrameCounter[threadid]<max_continuous_frame)
			{
				continuousFrameCounter[threadid] += 1;
			}
		}
		else
		{
			// δʩ��������ǰ�������ײ����֡����-1
			if (continuousFrameCounter[threadid] > 0)
			{
				continuousFrameCounter[threadid] -= 1;
			}
		}
	}
	else if(gh_distance >= GH_DISTANCE_THREASHOLD) // ���⹤����������֮�������Ƚϴ�ľ��룬��ɨ���������ײ
	{
		float normal_weight = (gh_distance - distance) / gh_distance * radius;
		if (distance < radius)//�������Ľ����Ҫд����ײ�������ڼ������⹤��λ��
		{
			if (isOnSurface)
			{
				// ���㹤���ڶ�����ʩ�ӵ�����������Ϊ[-nx, -ny, -nz](���涥�㷨�����ķ�����)
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
			//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
			atomicAdd(index, 1);

			////printf("��ɨ����뾶��Χ�ڣ�threadid: %d counter:%d\n", threadid, continuousFrameCounter[threadid]);
			// ���߶Զ���ʩ����������ǰ�������ײ����֡����+1
			if (continuousFrameCounter[threadid] < max_continuous_frame)
			{
				continuousFrameCounter[threadid] += 1;
			}
		}
		else if (distance < gh_distance)
		{
			// ����ʩ��ѹ���ķ���
			float dirX = v[0] / gh_distance;
			float dirY = v[1] / gh_distance;
			float dirZ = v[2] / gh_distance;
			// ���⹤���ϵ�ͶӰ��ָ����ײ���������
			float v_g2tetPos[3] = {
				tetPositions[indexX] - cylinder0[0],
				tetPositions[indexY] - cylinder0[1],
				tetPositions[indexZ] - cylinder0[2],
			};
			float temp = v_g2tetPos[0] * dirX + v_g2tetPos[1] * dirY + v_g2tetPos[2] * dirZ;
			float k = temp / gh_distance;
			if ((k < 1) && (k > 0))// ��������gh�����ϵ�ͶӰ����gh֮��
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
					// ��������Χ�ڣ��Ըö���ʩ��ѹ��
					float fx = dirX * normal_weight;
					float fy = dirY * normal_weight;
					float fz = dirZ * normal_weight;

					vertexForce[indexX] += fx;
					vertexForce[indexY] += fy;
					vertexForce[indexZ] += fz;
					//printf("saomiaoti f[%f %f %f] weight: %f\n", fx, fy, fz, normal_weight);
					//// printf("�ڹ��߰뾶��Χ��ɨ�����ڣ� threadid: %d counter: %d\n", threadid, continuousFrameCounter[threadid]);
					// ����ɨ����Զ���ʩ����������ǰ�������ײ����֡����+1
					if (continuousFrameCounter[threadid] < max_continuous_frame)
					{
						continuousFrameCounter[threadid] += 1;
					}
				}
				else
				{
					// printf("��ɨ����뾶��Χ�ڵ�û����ɨ������\n");
					if (continuousFrameCounter[threadid] > 0)
					{
						continuousFrameCounter[threadid] -= 1;
					}
				}
				
			}
			else
			{
				//printf("������ɨ��������ֱ���ϵ�ͶӰ����ɨ�����߶�֮�� k: %f\n", k);
				// δʩ��������ǰ�������ײ����֡����-1
				if (continuousFrameCounter[threadid] > 0)
				{
					continuousFrameCounter[threadid] -= 1;
				}
			}
		}
		else if (distance > gh_distance)
		{
			//printf("dis: %f, gh_dis:f ��ɨ�������뾶��\n", distance, gh_distance);
			// δʩ��������ǰ�������ײ����֡����-1
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
	float* vertexForce, // �ӹ���ָ����ײ��ķ�����
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

	//������ײ��־λ
	isCollide[threadid] = 0;
	zone[threadid] = -1;

	float nx = vertexNormals[indexX];
	float ny = vertexNormals[indexY];
	float nz = vertexNormals[indexZ];
	float len_normal = nx * nx + ny * ny + nz * nz;
	if (len_normal < 0.1)// ������Ϊ0���õ�Ϊ�����ڲ��Ķ��㣬��������ײ����������ļ��㡣
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

	t /= halfLength; // t����ײ���ڹ����ϵİٷֱ�λ�ã����Ϊ0��β��Ϊ1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// ���������ϵ� �Ӵ��������ͶӰ��->������λ�ã���������ֱ�ڹ���
	dx = vertexPositions[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = vertexPositions[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = vertexPositions[indexZ] - cylinder0[2] - t * cylinderd[2];

	float sqr_distance = dx * dx + dy * dy + dz * dz;
	if (sqr_distance > radius * radius) return;
	float distance = sqrt(sqr_distance);

	// ��λ��
	dx /= distance;
	dy /= distance;
	dz /= distance;

	// ���㷴����������Ϊ[dx, dy, dz]
	atomicAdd(vertexForce + threadid * 3 + 0, dx * (radius - distance));
	atomicAdd(vertexForce + threadid * 3 + 1, dy * (radius - distance));
	atomicAdd(vertexForce + threadid * 3 + 2, dz * (radius - distance));
	collisionDiag[indexX] += dx * dx * collisionStiffness;
	collisionDiag[indexY] += dy * dy * collisionStiffness;
	collisionDiag[indexZ] += dz * dz * collisionStiffness;

	//���ñ�־λ

	isCollide[threadid] = 1;
	zone[threadid] = t;
	//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
	atomicAdd(index, 1);
}

__global__ void hapticCalculateMeshCapsule(float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* tetPositions,
	float* vertexNormals,
	unsigned int* isCollide,
	float* vertexForce, // �ӹ���ָ����ײ��ķ�����
	float* zone,
	int vertexNum,
	int* index)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//������ײ��־λ
	isCollide[threadid] = 0;
	zone[threadid] = -1;

	float nx = vertexNormals[indexX];
	float ny = vertexNormals[indexY];
	float nz = vertexNormals[indexZ];
	float len_normal = nx * nx + ny * ny + nz * nz;
	if (len_normal < 0.1)// ������Ϊ0���õ�Ϊ�����ڲ��Ķ��㣬��������ײ����������ļ��㡣
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

	t /= halfLength; // t����ײ���ڹ����ϵİٷֱ�λ�ã����Ϊ0��β��Ϊ1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// ���������ϵ� �Ӵ��������ͶӰ��->������λ�ã���������ֱ�ڹ���
	dx = tetPositions[indexX] - cylinder0[0] - t * cylinderd[0];
	dy = tetPositions[indexY] - cylinder0[1] - t * cylinderd[1];
	dz = tetPositions[indexZ] - cylinder0[2] - t * cylinderd[2];

	float sqr_distance = dx * dx + dy * dy + dz * dz;
	if (sqr_distance > radius * radius) return;
	float distance = sqrt(sqr_distance);

	// ��λ��
	dx /= distance;
	dy /= distance;
	dz /= distance;

	// ���㷴����������Ϊ[dx, dy, dz]
	atomicAdd(vertexForce + threadid * 3 + 0, dx * (radius - distance));
	atomicAdd(vertexForce + threadid * 3 + 1, dy * (radius - distance));
	atomicAdd(vertexForce + threadid * 3 + 2, dz * (radius - distance));

	//���ñ�־λ

	isCollide[threadid] = 1;
	zone[threadid] = t;
	//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
	atomicAdd(index, 1);
}

__global__ void hapticCalculateCCylinder(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* tetPositions, unsigned int* isCollide, float* zone,int vertexNum, int* index) 
//float* cylinderPos, ���߼��λ��
//float* cylinderDir, ���߷���
//float halfLength, ���߳���
//float radius, ���߰뾶
//float* tetPositions, ������λ��
//unsigned int* isCollide, �Ƿ�����ײ
//float* zone, ��ײλ��
//int vertexNum, ����������
//int* index ÿ����һ����ײ�����ֵ+1
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//������ײ��־λ
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

	t /= halfLength; // t����ײ���ڹ����ϵİٷֱ�λ�ã����Ϊ0��β��Ϊ1

	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	// ���������ϵ� �Ӵ��������ͶӰ��->������λ�ã���������ֱ�ڹ���
	dx = tetPositions[indexX] - cylinder0[0] - t* cylinderd[0];
	dy = tetPositions[indexY] - cylinder0[1] - t* cylinderd[1];
	dz = tetPositions[indexZ] - cylinder0[2] - t* cylinderd[2];

	float distance = dx * dx + dy * dy + dz * dz;
	if (distance > radius*radius) return;
	//���ñ�־λ
	isCollide[threadid] = 1;
	zone[threadid] = t;
	//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
	atomicAdd(index, 1);

}

//����ǰ׺��
__global__ void hapticCalculatePrefixSum(unsigned int* isCollide, unsigned int* queueIndex, unsigned int* auxArray, int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	//�������������Ĵ�Сδ֪
	extern __shared__ unsigned int temp[];


	//�����ڹ����ڴ��������
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

	//����ÿ��block�ڵ�ǰ׺��
	queueIndex[threadid] = temp[threadIdx.x];


	//��������block�ĺ�
	if (threadid % (blockDim.x - 1) == 0 && threadid != 0) {
		auxArray[blockIdx.x] = queueIndex[threadid];
	}
}

// ������ײ���д��constraint������
// ��������
// unsigned int* isCollide, GPU�ϲ��м���ĵ���ײ���
//float* tetPositions, ������λ��
//float* tetNormals, �����巨����
//float* zone, �ڹ�������ײ�����λ��
//float* constraintPoints, �����������ײ��������λ��
//float* constraintNormals, ����� ������ײ�������巨����������������ࣩ
//float* constraintZone, ������ڹ�������ײ�����λ��
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
		//����index
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
		//����index
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
		//����index
		for (int block = 0; block < blockIdx.x; block++) {
			index += auxArray[block];
		}

		index += queueIndex[threadid];
		//����ײ�����λ�úͰ뾶���б���
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

//�����ݷ��õ�������--Mesh
__global__ void hapticAddSphereCollisionToQueue_Tri(unsigned int* isCollide, float* sphereInfos, float* zone, float* constraintPoints, float* constraintZone, unsigned int* queueIndex, unsigned int* auxArray, int sphereNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= sphereNum) return;

	if (isCollide[threadid]) {
		int index = -1;
		//����index
		for (int block = 0; block < blockIdx.x; block++) {
			index += auxArray[block];
		}


		index += queueIndex[threadid];
		//����ײ�����λ�úͰ뾶���б���
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

	//�ȺͰ�Χ�н�����ײ���
	float p0, p1;
	bool collision = hapticLineSegAABBInsect(start, end, &p0, &p1, boxs+threadid * 6);
	if (!collision) return;

	//�������ν�����ײ���
	int index0 = index[threadid * 3 + 0];
	int index1 = index[threadid * 3 + 1];
	int index2 = index[threadid * 3 + 2];
	float pos0[3] = { positions[index0 * 3 + 0],positions[index0 * 3 + 1], positions[index0 * 3 + 2] };
	float pos1[3] = { positions[index1 * 3 + 0],positions[index1 * 3 + 1], positions[index1 * 3 + 2] };
	float pos2[3] = { positions[index2 * 3 + 0],positions[index2 * 3 + 1], positions[index2 * 3 + 2] };
	collision = hapticLineSegTriangleInsect(start, end, pos0, pos1, pos2, triangleNormal+threadid * 3 + 0, &p0);
	if (collision) {
		//printf("%d: ����ײ\n",threadid);
	}
}

//ʹ��Բ�����������ײ���
__global__ void hapticCalculateCylinderSphere(float* cylinderPos, float* cylinderDir, float halfLength, float radius,float* sphereInfos,float* sphereForce,unsigned int* isCollide,float* zone, int* index,int sphereNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= sphereNum) return;
	//printf("%d\n", threadid);

	int indexX = threadid * 5 + 0;
	int indexY = threadid * 5 + 1;
	int indexZ = threadid * 5 + 2;

	float sphereRadius = sphereInfos[threadid * 5 + 3];

	//������ײ��־λ
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

	// �����ϵĽӴ���ָ��Ӵ������ĵ�������
	dx = sphereInfos[indexX] - cylinder0[0] - t* cylinderd[0];
	dy = sphereInfos[indexY] - cylinder0[1] - t* cylinderd[1];
	dz = sphereInfos[indexZ] - cylinder0[2] - t* cylinderd[2];

	float distance = dx * dx + dy * dy + dz * dz;
	if (distance > (sphereRadius+radius)*(sphereRadius+radius)) return;

	//printf("%d:����ײ\n", threadid);
	//�������յ�����
	// ��׼������
	dx /= distance;
	dy /= distance;
	dz /= distance;
	// ����Ƕ����Ⱦ����������ɱ�׼����ָ������������
	atomicAdd(sphereForce + threadid * 3 + 0, dx*(sphereRadius+radius-distance));
	atomicAdd(sphereForce + threadid * 3 + 1, dy*(sphereRadius+radius-distance));
	atomicAdd(sphereForce + threadid * 3 + 2, dz*(sphereRadius+radius-distance));

	//���ñ�־λ
	isCollide[threadid] = 1;
	zone[threadid] = t;
	//printf("sphere index = %d\n", threadid);
	//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
	atomicAdd(index, 1);
}

//Բ���������ײ--Mesh
__global__ void hapticCalculateCylinderSphere_Tri(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* sphereInfos, float* sphereForce, unsigned int* isCollide, float* zone, int* index, int sphereNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= sphereNum) return;
	//printf("%d\n", threadid);

	int indexX = threadid * 4 + 0;
	int indexY = threadid * 4 + 1;
	int indexZ = threadid * 4 + 2;

	float sphereRadius = sphereInfos[threadid * 4 + 3];

	//������ײ��־λ
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

	//printf("%d:����ײ\n", threadid);
	//�������յ�����
	dx /= distance;
	dy /= distance;
	dz /= distance;
	atomicAdd(sphereForce + threadid * 3 + 0, dx*(sphereRadius + radius - distance));
	atomicAdd(sphereForce + threadid * 3 + 1, dy*(sphereRadius + radius - distance));
	atomicAdd(sphereForce + threadid * 3 + 2, dz*(sphereRadius + radius - distance));

	//���ñ�־λ
	isCollide[threadid] = 1;
	zone[threadid] = t;
	//����������һ(��ײ��Ϣ�ڼ���ǰ׺�͵�ʱ��ֵ)
	atomicAdd(index, 1);
}

//�߶κ�AABB��Χ�е���
__device__ bool hapticLineSegAABBInsect(float* start, float* end,float* p0,float* p1, float* boxs) {

	float dir[3] = {end[0]-start[0],end[1]-start[1],end[3]-start[3]};


	//��ȡ��Χ�еĽ���
	float minx = boxs[0];
	float miny = boxs[1];
	float minz = boxs[2];
	float maxx = boxs[3];
	float maxy = boxs[4];
	float maxz = boxs[5];


	//��ȡ������ƽ��Ľ���
	float t0x = (minx - start[0]) / dir[0];
	float t1x = (maxx - start[0]) / dir[0];
	if (t0x > t1x) hapticSwap(&t0x, &t1x);
	float t0y = (miny - start[1]) / dir[1];
	float t1y = (maxy - start[1]) / dir[1];
	if (t0y > t1y) hapticSwap(&t0y, &t1y);
	float t0z = (minz - start[2]) / dir[2];
	float t1z = (maxz - start[2]) / dir[2];
	if (t0z > t1z) hapticSwap(&t0z, &t1z);

	//�ҵ��ཻ���ֵĵ��
	float t0 = (t0x < t0y) ? t0y : t0x;
	float t1 = (t1x < t1y) ? t1x : t1y;
	t0 = (t0 > t0z) ? t0 : t0z;
	t1 = (t1 > t1z) ? t1z : t1;

	//��ȡ�⣬����Ҫclamp��01֮��
	*p1 = t0;
	*p0 = t1;

	if (*p0 > *p1) return false;

	//����Ͱ�Χ���ཻ������Ҫ��һ���ж��Ƿ���01֮��
	*p0 = hapticClamp(*p0, 0, 1);
	*p1 = hapticClamp(*p1, 0, 1);

	//���������ͬ��Ҳ�ǲ��ཻ��
	if (abs(*p0 - *p1) < 0.0001) return false;

	return true;
}


//�߶κ������ε���
__device__ bool hapticLineSegTriangleInsect(float* start, float* end, float* pos0, float* pos1, float* pos2, float* triangleNormal,float* ans) {
	//�ȼ����ƽ��Ľ���
	float insectPoint[3];

	//��������
	float v[3] = {pos0[0]-start[0],pos0[1] - start[1], pos0[2] - start[2]};

	float dir[3] = { end[0] - start[0],end[1] - start[1],end[3] - start[3] };
	float length = sqrt( dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] );
	dir[0] /= length;
	dir[1] /= length;
	dir[2] /= length;

	float dotNV = triangleNormal[0] * v[0] + triangleNormal[1] * v[1] + triangleNormal[2] * v[2];
	float dotND = triangleNormal[0] * dir[0] + triangleNormal[1] * dir[1] + triangleNormal[2] * dir[2];
	float t = dotNV/ dotND;


	//�����ж��Ƿ����߶��ڲ�
	if (t<0 || t> length) return false;

	float p[3] = { start[0] + t*dir[0],start[1] + t*dir[1], start[2] + t*dir[2] };

	//���жϵ��ڲ�����������,ʹ�ò�˷�
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

	//������ַ���ת,��Ϊ���ཻ
	float flag = hapticDot(cross0,cross1);
	if (flag < 0) return false;
	flag = hapticDot(cross1, cross2);
	if (flag < 0) return false;

	*ans = t;
	return true;
}

// ����ײ��Ϣ����
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

	// ��ն����ϵ�������
	vertexForce[indexY]	= 0.0;
	vertexForce[indexZ]	= 0.0;
	vertexForce[indexX]	= 0.0;
}
// ����ײ���ϵ�����Ϊ����ʩ�ӵ��󶨵Ķ����ϡ�
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

	//���û���������뿪
	if (isCollide[threadid] == 0) return;

	//��������Ķ���������Ȩ��
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

//����ɨ������ײ��⣬��������λ���ųɵ�ɨ�������ײ��⡣
__device__ bool hapticCylinderCollisionContinue(
	float length, // ���߳��� 
	float moveDistance, // ����ɨ������ֹ�����������λ��֮��ľ��� 
	float radius,// ���߰뾶
	float* cylinderPos, //ɨ�����ص�Ĺ���λ��
	float* cylinderLastPos, // ɨ�������Ĺ���λ��
	float* cylinderDir, // ���߷���
	float* moveDir, //������㵽�յ���˶�����
	float* position, // ��Ҫ������ײ���Ķ���λ��
	float* collisionNormal, //��ײ֮�󶥵㱻�ų��ķ���
	float* collisionPos // ���㱻�ų���ɨ��������λ��
)
{
	//���ȼ�����˶�ƽ��ķ�������
	float normal[3];
	tetCross_D(cylinderDir, moveDir, normal);
	tetNormal_D(normal);

	//���������Ҫ�ı���
	float VSubO[3] = { position[0] - cylinderPos[0] ,position[1] - cylinderPos[1] ,position[2] - cylinderPos[2] };//���߼��ָ�������ײ�������
	float lineStart0[3] = { cylinderPos[0] ,cylinderPos[1] ,cylinderPos[2] };// ��ǰ���߼��
	float lineStart1[3] = { cylinderLastPos[0] ,cylinderLastPos[1] ,cylinderLastPos[2] };// ��һ֡���߼��
	float lineStart2[3] = { cylinderPos[0] + cylinderDir[0] * length ,cylinderPos[1] + cylinderDir[1] * length,cylinderPos[2] + cylinderDir[2] * length };// ��ǰ֡����β��


	//����Ҫ�Ƚ���һ����ײ��⣬�������Ƿ�����ײ��������������Ҫ������ײ���


	//1.�����ھֲ�����ϵ�е����꣬��������ֲ����겻�������ģ����Բ��ܺ�����е����ʹ�ø�˹��Ԫ
	// �����᣺ ���߷���cylinderDir���˶�����moveDir�����߷������˶������ųɵ�ƽ��ķ�����normal
	// ������ײ��������������ɵľֲ�����ϵ����[x, y, z] �ô������������������ϵ�£�x������ײ���ڹ�����ͶӰ��λ�ã�y�������˶������ϵ��˶�����
	// ��˹��Ԫ��[A|I] ֻʹ����֮��ļӼ���A���I��I����A�������
	float x, y, z;
	float det = tetSolveFormula_D(cylinderDir, moveDir, normal, VSubO, &x, &y, &z);

	if (x != x || y != y || z != z) return false;

	float distance = 0.0;
	bool flag = false;
	//2.���������ҵ��������ڵ�����
	if (x > length && y > moveDistance) {
		//����㵽��ľ���
		float basePoint[3] = { cylinderLastPos[0] + length * cylinderDir[0],cylinderLastPos[1] + length * cylinderDir[1] , cylinderLastPos[2] + length * cylinderDir[2] };
		distance = tetPointPointDistance_D(position, basePoint);
		flag = true;
	}
	else if (x > length && y < moveDistance && y>0.0) {
		//����㵽�ߵľ���
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
		//����㵽��ľ���
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


	//3.�жϾ���
	if (distance > radius) return false;
	//if (flag) return false;

	//printf("x:%f,y:%f,z:%f\n", x, y, z);

	//4. �����������ײ�ų�λ��
	//����Ԫһ�η���,�����������Բ�����м���
	float lineDir[3] = { moveDir[0],moveDir[1], moveDir[2] };


	float v0[3] = { position[0] - lineStart0[0] ,position[1] - lineStart0[1] ,position[2] - lineStart0[2] };
	float v1[3] = { position[0] - lineStart1[0] ,position[1] - lineStart1[1] ,position[2] - lineStart1[2] };
	float v2[3] = { position[0] - lineStart2[0] ,position[1] - lineStart2[1] ,position[2] - lineStart2[2] };


	//��Բ���ཻ
	float solve00, solve01;
	float solve10, solve11;
	tetSolveInsect_D(lineDir, cylinderDir, v0, radius, &solve00, &solve01);
	tetSolveInsect_D(lineDir, moveDir, v0, radius, &solve10, &solve11);
	float solve = min(solve11, solve01);
	//tetSolveInsect_D(lineDir, cylinderDir, v1, radius, &solve00, &solve01);
	//solve = min(solve, solve01);
	//tetSolveInsect_D(lineDir, moveDir, v2, radius, &solve00, &solve01);
	//solve = min(solve, solve01);


	//�����ཻ
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

	//����λ�õõ������ų���λ��
	collisionPos[0] = position[0] - lineDir[0] * solve;
	collisionPos[1] = position[1] - lineDir[1] * solve;
	collisionPos[2] = position[2] - lineDir[2] * solve;

	//���¶������ײ���ߣ��򹤾������Ͻ���ͶӰ
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
	//��˼��������η���
	c[0] = a[1] * b[2] - b[1] * a[2];
	c[1] = a[2] * b[0] - b[2] * a[0];
	c[2] = a[0] * b[1] - b[0] * a[1];
}

__device__ float hapticDot(float* a, float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

