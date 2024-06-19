#include "gpuvar.h"
#include "gpufun.h"

//���ڴ洢ָ������

int* tetShellIdx_d;

float* directDirectionMU_D;
int* directIndexMU_D;

//��ײ������
float* planeNormal_D;
float* planePos_D;

float* toolPositionAndDirection_d;
float* toolPosePrev_d;
float* radius_d;

int* hapticCollisionNum_d;
float* toolContactDeltaPos_triVert_d;
float* totalFC_d;//���⹤�����ܵĽӴ���������Ϊ3
float* totalPartial_FC_X_d;//���⹤�����ܵĽӴ����Թ���λ�õ��ݶȣ�����Ϊ9
float* totalPartial_FC_Omega_d;//���⹤�����ܵĽӴ����Թ��߳�����ݶȣ�����Ϊ9
float* totalTC_d;// ���⹤�����ܵĽӴ���Ť�أ�����Ϊ3�����ݵ��������豸����α�ʾ����
float* totalPartial_TC_X_d;// Ť���ڹ���λ�õ��ݶȣ�����Ϊ9
float* totalPartial_TC_Omega_d;// Ť���ڹ��߳��򴦵��ݶȣ�����Ϊ9

//Բ���幤��
float* cylinderShiftMU_D;
float* cylinderLastPosMU_D;
float* cylinderPosMU_D;
float* cylinderDirZMU_D;
float* cylinderDirYMU_D;
float* cylinderDirXMU_D;
float	graperHalfLengthMU_D;

//ץǯ����ײ��Ϣ
float	grapperRadiusMU_D;
float	grapperLengthMU_D;

unsigned int* CollideFlagMU_D;	//�����Ƿ�����ײ

//ץȡ
bool	firstGrabLeftMU_D;
bool	firstGrabRightMU_D;
unsigned int* isGrabLeftMU_D;
unsigned int* isGrabRigthMU_D;
unsigned int* isGrabHalfLeftMU_D;		//ץǯ�պϹ����еıպϹ�ϵ
unsigned int* isGrabHalfRightMU_D;
float* relativePositionLeftMU_D;
float* relativePositionRightMU_D;
unsigned int* CollideFlagLeftMU_D;		//��־λ�������Ƿ�������֯������ײ
unsigned int* CollideFlagRightMU_D;



//��ײԼ����
float* triVertCollisionForce_d;
float* insertionDepthMU_D;

//����Բ������ײ
__device__ float* cylinderShift;
__device__ float* cylinderLastPos;
__device__ float* cylinderPos;
__device__ float* cylinderDirZ; // Բ���ı�������z�᷽�򣬼�Բ������ķ�����ά��
__device__ float* cylinderV;
__device__ unsigned char* cylinderCollideFlag;

//ֻ���㼷ѹʱ��ָ������������ʱ������
int runcalculateToolShift(float halfLength, float radius, int cylinderIdx) {

	//ѡȡ�����ֵĹ���
	int		cylinderButton = HAPTIC_BUTTON::normal;

	////ÿ�ζ����㣬ÿ�μ����µ�
	//if (flag == 1) {
	//	cudaMemset(cylinderShiftLeft_D, 0, 3 * sizeof(float));
	//	cylinderShift = cylinderShiftLeft_D;
	//	cylinderPos = cylinderPosLeft_D;
	//	cylinderDirZ = ;
	//	cylinderDirY = cylinderDirYLeft_D;
	//	cylinderDirX = cylinderDirXLeft_D;
	//	cylinderButton = cylinderButtonLeft_D;
	//	grapperUpDirZ = tetgrapperUpDirZLeft_D;
	//	grapperDownDirZ = tetgrapperDownDirZRight_D;
	//}
	//else {
	//	cudaMemset(cylinderShiftRight_D, 0, 3 * sizeof(float));
	//	cylinderShift = cylinderShiftRight_D;
	//	cylinderPos = cylinderPosRight_D;
	//	cylinderDirZ = cylinderDirZRight_D;
	//	cylinderDirY = cylinderDirYRight_D;
	//	cylinderDirX = cylinderDirXRight_D;
	//	cylinderButton = cylinderButtonRight_D;
	//	grapperUpDirZ = tetgrapperUpDirZRight_D;
	//	grapperDownDirZ = tetgrapperDownDirZRight_D;
	//}
	cylinderShift = &cylinderShift_d[cylinderIdx * 3];
	cudaMemset(cylinderShift, 0.0f, 3 * sizeof(float));
	cylinderPos = &cylinderPos_d[cylinderIdx * 3];
	cylinderDirZ = &cylinderDirZ_d[cylinderIdx * 3];
	cylinderV = &cylinderV_d[cylinderIdx * 3];
	cylinderCollideFlag = &toolCollideFlag_d[cylinderIdx];

	switch (cylinderButton)
	{
	case cut:
		break;
	case grab:
		break;
	case normal: 
		{
			int  threadNum = 512;
			int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
			//д��ƫ��������
			calculateToolShift << <blockNum, threadNum >> > (
				cylinderPos, cylinderDirZ, 
				tetVertNonPenetrationDir_d, 
				halfLength, radius, 
				tetVertPos_d, cylinderShift, tetVertNum_d);

			cudaDeviceSynchronize();
		}
		break;
	default:
		break;
	}
	return 0;
}

//ֻ���㼷ѹʱ��ָ������������ʱ������
int runcalculateToolShiftMU(float halfLength, float radius, int cylinderIdx) {

	//ѡȡ�����ֵĹ��� ע�⣺���밴����Ҫ��
	int		cylinderButton = HAPTIC_BUTTON::normal;

	cylinderShift = &cylinderShift_d[cylinderIdx * 3];
	cudaMemset(cylinderShift, 0.0f, 3 * sizeof(float));
	cylinderPos = &cylinderPos_d[cylinderIdx * 3];
	cylinderDirZ = &cylinderDirZ_d[cylinderIdx * 3];
	cylinderV = &cylinderV_d[cylinderIdx * 3];
	cylinderCollideFlag = &toolCollideFlag_d[cylinderIdx];

	switch (cylinderButton)
	{
	case cut:
		break;
	case grab:
		break;
	case normal:
	{
		int  threadNum = 512;
		int blockNum = (triVertNum_d + threadNum - 1) / threadNum;
		//д��ƫ��������
		calculateToolShift << <blockNum, threadNum >> > (
			cylinderPos, cylinderDirZ,
			triVertNonPenetrationDir_d,
			halfLength, radius,
			triVertPos_d, cylinderShift, triVertNum_d);

		cudaDeviceSynchronize();
	}
	break;
	default:
		break;
	}
	return 0;
}

__global__ void calculateToolShift(
	float* cylinderPos, float* cylinderDir,
	float* directDir,
	float halfLength, float radius,
	float* positions,
	float* cylinderShift,
	int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float t = 0.0;
	float solution = 0.0;

	float collisionNormal[3];
	float collisionPos[3];
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//����һ�����ߵ���ײ���
	//ָ��������ײ���ķ�����ָ�������ķ��򣬴Ӱ��������ָ�򶥵�
	float moveDir[3];
	moveDir[0] = directDir[indexX];
	moveDir[1] = directDir[indexY];
	moveDir[2] = directDir[indexZ];


	//ʹ��ָ�������������ײ���
	bool collision = cylinderRayCollisionDetection(cylinderPos, cylinderDir, positions[indexX], positions[indexY], positions[indexZ], moveDir, halfLength, radius, &t, &solution, collisionNormal, collisionPos);
	if (!collision) return;


	//�ۼӵõ�ƫ������
	atomicAdd(cylinderShift + 0, -directDir[indexX]);
	atomicAdd(cylinderShift + 1, -directDir[indexY]);
	atomicAdd(cylinderShift + 2, -directDir[indexZ]);

	//printf("ƫ����:%f,%f,%f\n", cylinderShift[0], cylinderShift[1], cylinderShift[2]);
}

int runcalculateCollisionCylinder(float halfLength, float radius, 
	float collisionStiffness, float adsorbStiffness, float frictionStiffness, 
	int idx) {
	cylinderShift = &cylinderShift_d[idx * 3];
	cylinderLastPos = &cylinderLastPos_d[idx * 3];
	cylinderPos  = &cylinderPos_d[idx * 3];
	cylinderDirZ = &cylinderDirZ_d[idx * 3];
	cylinderV = &cylinderV_d[idx * 3];
	cylinderCollideFlag = &toolCollideFlag_d[idx];

	///�˴�������Բ�������ײ
	int  threadNum = 512;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	calculateCollisionCylinderSDF << <blockNum, threadNum >> > (
		cylinderLastPos, cylinderPos, 
		cylinderDirZ, halfLength, radius, 
		tetVertPos_d, tetVertForce_d, tetVertisCollide_d, cylinderCollideFlag, 
		tetVertCollisionDiag_d, 
		tetVertNum_d,
		collisionStiffness, tetVertCollisionForce_d, tetVertNonPenetrationDir_d, cylinderShift);

	cudaDeviceSynchronize();
	return 0;
}

int runcalculateCollisionCylinderMU(float halfLength, float radius,
	float collisionStiffness, float adsorbStiffness, float frictionStiffness,
	int idx)
{
	cylinderShift = &cylinderShift_d[idx * 3];
	cylinderLastPos = &cylinderLastPos_d[idx * 3];
	cylinderPos = &cylinderPos_d[idx * 3];
	cylinderDirZ = &cylinderDirZ_d[idx * 3];
	cylinderV = &cylinderV_d[idx * 3];
	cylinderCollideFlag = &toolCollideFlag_d[idx];

	int  threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;

	calculateCollisionCylinderSDF << <blockNum, threadNum >> > (
		cylinderLastPos, cylinderPos,
		cylinderDirZ, halfLength, radius,
		triVertPos_d, triVertForce_d, triVertisCollide_d, cylinderCollideFlag,
		triVertCollisionDiag_d,
		triVertNum_d,
		collisionStiffness, triVertCollisionForce_d, triVertNonPenetrationDir_d, cylinderShift);
	cudaDeviceSynchronize();
	return 0;
}
//ʹ�û���SDF�ı�������ײ����㷨
__global__ void calculateCollisionCylinderSDF(
	float* cylinderLastPos, float* cylinderPos, float* cylinderDir, float halfLength, float radius, 
	float* positions, float* force, unsigned char* isCollide, unsigned char* collideFlag, 
	float* collisionDiag, 
	int vertexNum, 
	float collisionStiffness, float* collisionForce, 
	float* directDirection, float* cylinderShift) 
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float t = 0.0;    //�����ڹ��������ϵ�ͶӰ
	float collisionNormal[3];   //��ײ�ų�����
	float collisionPos[3];   //��ײ�ų�λ��
	int indexX = threadid * 3 + 0;  //����λ�������������е��±�
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	float directDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };

	//��ƫ��������Ϊ��λ����
	float shiftLength = sqrt(cylinderShift[0] * cylinderShift[0] + cylinderShift[1] * cylinderShift[1] + cylinderShift[2] * cylinderShift[2]);
	if (shiftLength > 0.01f) {
		cylinderShift[0] /= shiftLength;
		cylinderShift[1] /= shiftLength;
		cylinderShift[2] /= shiftLength;
	}
	float radius_extent_ratio = 0;
	//ƫ��һ���뾶���ȣ���������뾶Ϊԭ����������ʵ��ƫ�ĵ�Բ��
	float newPos[3];
	newPos[0] = cylinderPos[0] + cylinderShift[0] * radius_extent_ratio * radius;
	newPos[1] = cylinderPos[1] + cylinderShift[1] * radius_extent_ratio * radius;
	newPos[2] = cylinderPos[2] + cylinderShift[2] * radius_extent_ratio * radius;

	float tetPositions[3] = { positions[indexX], positions[indexY], positions[indexZ] };  //�����嶥��λ��
	float toolMoveDir[3] = { cylinderLastPos[0] - cylinderPos[0],cylinderLastPos[1] - cylinderPos[1], cylinderLastPos[2] - cylinderPos[2] };  //�����˶�������
	float proj = tetDot_D(toolMoveDir, cylinderDir);  //�����˶������������ڹ��������ϵ�ͶӰ
	float moveDistance = tetNormal_D(toolMoveDir);  //����֡����λ��֮���ֱ�Ӿ���
	//moveDistance = sqrt(moveDistance* moveDistance - proj * proj);  //����֡�����ڹ������ߴ�ֱ����ľ���
	
	radius *= (1+radius_extent_ratio);

	//ʹ�û���SDF��������ײ���
	if (moveDistance > 0.5) {  //�������֡����λ�˾����Զ����ʹ��������ײ���
		bool collisionSDF = cylinderCollisionContinueSDF(halfLength, moveDistance, radius, cylinderPos, cylinderLastPos, cylinderDir, toolMoveDir, tetPositions, directDir, collisionNormal, collisionPos);
		if (!collisionSDF) return;  //δ��ײֱ���˳�
	}
	else {  //�������֡����λ�˾���Ͻ�����ʹ����ɢ��ײ���
		bool collision = cylinderCollisionSDF(newPos, cylinderDir, tetPositions, directDir, halfLength, radius, &t, collisionNormal, collisionPos);
		if (!collision) return;  //δ��ײֱ���˳�
	}

	float deltaPos[3];
	deltaPos[0] = collisionPos[0] - positions[indexX];
	deltaPos[1] = collisionPos[1] - positions[indexY];
	deltaPos[2] = collisionPos[2] - positions[indexZ];

	//if(threadid==72990)
		//printf("threadid:%d collided, deltaPos[%f %f %f]\n", threadid, deltaPos[0], deltaPos[1], deltaPos[2]);

	float temp[3];  //Լ����Ӧ����ײ��
	temp[0] = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
	temp[1] = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
	temp[2] = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
	collisionForce[indexX] += temp[0];
	collisionForce[indexY] += temp[1];
	collisionForce[indexZ] += temp[2];

	//������
	force[indexX] += temp[0];
	force[indexY] += temp[1];
	force[indexZ] += temp[2];


	//����Խ�Ԫ�ض�Ӧ��ֵ
	collisionDiag[indexX] += collisionStiffness * collisionNormal[0] * collisionNormal[0];
	collisionDiag[indexY] += collisionStiffness * collisionNormal[1] * collisionNormal[1];
	collisionDiag[indexZ] += collisionStiffness * collisionNormal[2] * collisionNormal[2];


	//���ñ�־λ
	isCollide[threadid] = 1;
	collideFlag[0] = 1;
}

//ʹ�û���SDF�ı�������ײ����㷨
__global__ void calculateCollisionCylinderSDF(
	float* cylinderLastPos, float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* positions, float* force, unsigned char* isCollide, unsigned char* collideFlag,
	float* collisionDiag,
	int* sortedIndices, int offset, int activeElementNum,
	float collisionStiffness, float* collisionForce,
	float* directDirection, float* cylinderShift)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	int vertIdx = sortedIndices[offset + threadid];
	float t = 0.0;    //�����ڹ��������ϵ�ͶӰ
	float collisionNormal[3];   //��ײ�ų�����
	float collisionPos[3];   //��ײ�ų�λ��
	int indexX = vertIdx * 3 + 0;  //����λ�������������е��±�
	int indexY = vertIdx * 3 + 1;
	int indexZ = vertIdx * 3 + 2;

	float directDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };

	//��ƫ��������Ϊ��λ����
	float shiftLength = sqrt(cylinderShift[0] * cylinderShift[0] + cylinderShift[1] * cylinderShift[1] + cylinderShift[2] * cylinderShift[2]);
	if (shiftLength > 0.01f) {
		cylinderShift[0] /= shiftLength;
		cylinderShift[1] /= shiftLength;
		cylinderShift[2] /= shiftLength;
	}
	float radius_extent_ratio = 0;
	//ƫ��һ���뾶���ȣ���������뾶Ϊԭ����������ʵ��ƫ�ĵ�Բ��
	float newPos[3];
	newPos[0] = cylinderPos[0] + cylinderShift[0] * radius_extent_ratio * radius;
	newPos[1] = cylinderPos[1] + cylinderShift[1] * radius_extent_ratio * radius;
	newPos[2] = cylinderPos[2] + cylinderShift[2] * radius_extent_ratio * radius;

	float tetPositions[3] = { positions[indexX], positions[indexY], positions[indexZ] };  //�����嶥��λ��
	float toolMoveDir[3] = { cylinderLastPos[0] - cylinderPos[0],cylinderLastPos[1] - cylinderPos[1], cylinderLastPos[2] - cylinderPos[2] };  //�����˶�������
	float proj = tetDot_D(toolMoveDir, cylinderDir);  //�����˶������������ڹ��������ϵ�ͶӰ
	float moveDistance = tetNormal_D(toolMoveDir);  //����֡����λ��֮���ֱ�Ӿ���
	//moveDistance = sqrt(moveDistance* moveDistance - proj * proj);  //����֡�����ڹ������ߴ�ֱ����ľ���

	radius *= (1 + radius_extent_ratio);

	//ʹ�û���SDF��������ײ���
	if (moveDistance > 0.5) {  //�������֡����λ�˾����Զ����ʹ��������ײ���
		bool collisionSDF = cylinderCollisionContinueSDF(halfLength, moveDistance, radius, cylinderPos, cylinderLastPos, cylinderDir, toolMoveDir, tetPositions, directDir, collisionNormal, collisionPos);
		if (!collisionSDF) return;  //δ��ײֱ���˳�
	}
	else {  //�������֡����λ�˾���Ͻ�����ʹ����ɢ��ײ���
		bool collision = cylinderCollisionSDF(newPos, cylinderDir, tetPositions, directDir, halfLength, radius, &t, collisionNormal, collisionPos);
		if (!collision) return;  //δ��ײֱ���˳�
	}

	float deltaPos[3];
	deltaPos[0] = collisionPos[0] - positions[indexX];
	deltaPos[1] = collisionPos[1] - positions[indexY];
	deltaPos[2] = collisionPos[2] - positions[indexZ];

	float temp[3];  //Լ����Ӧ����ײ��
	temp[0] = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
	temp[1] = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
	temp[2] = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
	collisionForce[indexX] += temp[0];
	collisionForce[indexY] += temp[1];
	collisionForce[indexZ] += temp[2];

	//������
	force[indexX] += temp[0];
	force[indexY] += temp[1];
	force[indexZ] += temp[2];


	//����Խ�Ԫ�ض�Ӧ��ֵ
	collisionDiag[indexX] += collisionStiffness * collisionNormal[0] * collisionNormal[0];
	collisionDiag[indexY] += collisionStiffness * collisionNormal[1] * collisionNormal[1];
	collisionDiag[indexZ] += collisionStiffness * collisionNormal[2] * collisionNormal[2];


	//���ñ�־λ
	isCollide[vertIdx] = 1;
	*collideFlag = 1;
}

//����SDF��������ײ���
__device__ bool cylinderCollisionContinueSDF(float length, float moveDistance, float radius, 
	float* cylinderPos, float* cylinderLastPos, float* cylinderDir, float* moveDir, float* position, 
	float* directDir, float* collisionNormal, float* collisionPos) {
	//���ȼ�����˶�ƽ��ķ�������
	float normal[3];
	tetCross_D(cylinderDir, moveDir, normal);
	tetNormal_D(normal);

	//���������Ҫ�ı���
	float VSubO[3] = { position[0] - cylinderPos[0] ,position[1] - cylinderPos[1] ,position[2] - cylinderPos[2] };
	float lineStart0[3] = { cylinderPos[0] ,cylinderPos[1] ,cylinderPos[2] };
	float lineStart1[3] = { cylinderLastPos[0] ,cylinderLastPos[1] ,cylinderLastPos[2] };
	float lineEnd1[3] = { cylinderLastPos[0] + cylinderDir[0] * length, cylinderLastPos[1] + cylinderDir[1] * length, cylinderLastPos[2] + cylinderDir[2] * length };

	//1.�����ھֲ�����ϵ�е����꣬��������ֲ����겻�������ģ����Բ��ܺ�����е����ʹ�ø�˹��Ԫ
	float x, y, z;
	float det = tetSolveFormula_D(cylinderDir, moveDir, normal, VSubO, &x, &y, &z);

	if (x != x || y != y || z != z) return false;

	float MaxDis = 0.75 * moveDistance;  //SDF��ͬ���򻮷ֵ���ֵ
	float distance = 0.0;  //�����ڲ�ͬ������������
	int flag = 0;
	//2.���������ҵ��������ڵ�����
	if (x > 0.0 && x < length && y <= 0) {  //�ڵ�ǰ֡����Բ���İ�Բ������
		//��Բ�����߾���
		distance = tetPointLineDistance_D(lineStart0, cylinderDir, position);
		flag = 1;
	}
	else if (x > 0.0 && x < length && y < moveDistance && y > 0.0) {  //��ǰ��֡��������֮���ɨ��������
		//����ľ���
		distance = abs(z);
		if (y <= MaxDis) {
			flag = 1;
		}
		else {
			flag = 3;
		}
	}
	else if (x > 0.0 && x < length && y > moveDistance) {  //����һ֡����Բ���İ�Բ������
		//��Բ�����ߵľ���
		distance = tetPointLineDistance_D(lineStart1, cylinderDir, position);
		flag = 3;
	}
	else return false;  //δ��ײ

	//3.�жϾ���
	if (distance > radius) return false;

	//�ж��ųⷽ���Ƿ��ָ���������
	/*if (flag == 1) {
		float proDir[3] = { -moveDir[0], -moveDir[1], -moveDir[2] };
		float proA = tetDot_D(proDir, directDir);
		if (proA < -0.5) {
			flag = 3;
		}
	}
	else if (flag == 3) {
		float proA = tetDot_D(moveDir, directDir);
		if (proA < -0.5) {
			flag = 1;
		}
	}*/

	//4.���ݲ�ͬ����������ͬ�÷����ų⣬�����ų�λ��
	if (flag == 1) {  //��ǰ֡����Բ���ų�
		float lineDir[3] = { moveDir[0], moveDir[1], moveDir[2] };

		float v0[3] = { position[0] - lineStart0[0], position[1] - lineStart0[1], position[2] - lineStart0[2] };

		float solve00, solve01;
		float solve10, solve11;
		tetSolveInsect_D(lineDir, cylinderDir, v0, radius, &solve00, &solve01);
		tetSolveInsect_D(lineDir, moveDir, v0, radius, &solve10, &solve11);
		float solve = min(solve11, solve01);
		//float solve = solve11;

		if (solve != solve) return false;

		//���㶥���ų���λ��
		collisionPos[0] = position[0] - lineDir[0] * solve;
		collisionPos[1] = position[1] - lineDir[1] * solve;
		collisionPos[2] = position[2] - lineDir[2] * solve;

		//���㶥�����ײ����
		float projPos[3] = { collisionPos[0] - cylinderPos[0], collisionPos[1] - cylinderPos[1], collisionPos[2] - cylinderPos[2] };
		float proj = tetDot_D(projPos, cylinderDir);
		projPos[0] = collisionPos[0] - cylinderPos[0] - cylinderDir[0] * proj;
		projPos[1] = collisionPos[1] - cylinderPos[1] - cylinderDir[1] * proj;
		projPos[2] = collisionPos[2] - cylinderPos[2] - cylinderDir[2] * proj;

		tetNormal_D(projPos);
		collisionNormal[0] = projPos[0];
		collisionNormal[1] = projPos[1];
		collisionNormal[2] = projPos[2];
	}
	else if (flag == 2) {  //ֱ����ɨ���������ų�
		if (z >= 0.0) {
			//�����ų�����
			collisionNormal[0] = normal[0];
			collisionNormal[1] = normal[1];
			collisionNormal[2] = normal[2];
		}
		else {
			//�����ų�����
			collisionNormal[0] = -normal[0];
			collisionNormal[1] = -normal[1];
			collisionNormal[2] = -normal[2];
		}

		//�����ų�λ��
		collisionPos[0] = position[0] + collisionNormal[0] * (radius - distance);
		collisionPos[1] = position[1] + collisionNormal[1] * (radius - distance);
		collisionPos[2] = position[2] + collisionNormal[2] * (radius - distance);
	}
	else if (flag == 3) {  //����һ֡����Բ���ų�
		float lineDir[3] = { -moveDir[0], -moveDir[1], -moveDir[2] };
		float cyDir[3] = { -cylinderDir[0], -cylinderDir[1], -cylinderDir[2] };

		float v0[3] = { position[0] - lineEnd1[0], position[1] - lineEnd1[1], position[2] - lineEnd1[2] };

		//��Բ����
		float solve00, solve01;
		float solve10, solve11;
		tetSolveInsect_D(lineDir, cyDir, v0, radius, &solve00, &solve01);
		tetSolveInsect_D(lineDir, lineDir, v0, radius, &solve10, &solve11);
		float solve = min(solve11, solve01);

		if (solve != solve) return false;

		//���㶥����ų�λ��
		collisionPos[0] = position[0] - lineDir[0] * solve;
		collisionPos[1] = position[1] - lineDir[1] * solve;
		collisionPos[2] = position[2] - lineDir[2] * solve;

		//���¶������ײ���ߣ��򹤾������Ͻ���ͶӰ
		float projPos[3] = { collisionPos[0] - cylinderLastPos[0], collisionPos[1] - cylinderLastPos[1], collisionPos[2] - cylinderLastPos[2] };
		float proj = tetDot_D(projPos, cylinderDir);
		projPos[0] = collisionPos[0] - cylinderLastPos[0] - cylinderDir[0] * proj;
		projPos[1] = collisionPos[1] - cylinderLastPos[1] - cylinderDir[1] * proj;
		projPos[2] = collisionPos[2] - cylinderLastPos[2] - cylinderDir[2] * proj;

		tetNormal_D(projPos);
		collisionNormal[0] = projPos[0];
		collisionNormal[1] = projPos[1];
		collisionNormal[2] = projPos[2];

		/*float VSub1[3] = { position[0] - cylinderLastPos[0] ,position[1] - cylinderLastPos[1] ,position[2] - cylinderLastPos[2] };
		float proj = tetDot_D(VSub1, cylinderDir);
		float projPos[3] = { cylinderLastPos[0] + proj * cylinderDir[0], cylinderLastPos[1] + proj * cylinderDir[1], cylinderLastPos[2] + proj * cylinderDir[2] };

		//�����ų�����
		collisionNormal[0] = position[0] - projPos[0];
		collisionNormal[1] = position[1] - projPos[1];
		collisionNormal[2] = position[2] - projPos[2];
		tetNormal_D(collisionNormal);

		//�����ų�λ��
		collisionPos[0] = projPos[0] + collisionNormal[0] * radius;
		collisionPos[1] = projPos[1] + collisionNormal[1] * radius;
		collisionPos[2] = projPos[2] + collisionNormal[2] * radius;*/
	}

	return true;
}


//��Բ������ɢ��ײ��⣬��ָ������
__device__ bool cylinderCollisionSDF(float* pos, float* dir, float* vert, float* directDir, float length,
	float radius, float* t, float* collisionNormal, float* collisionPos) {
	float cylinder0x, cylinder0y, cylinder0z;
	cylinder0x = pos[0];
	cylinder0y = pos[1];
	cylinder0z = pos[2];
	float cylinder1x, cylinder1y, cylinder1z;
	cylinder1x = pos[0] + dir[0] * length;
	cylinder1y = pos[1] + dir[1] * length;
	cylinder1z = pos[2] + dir[2] * length;

	float cylinderdx = cylinder1x - cylinder0x;
	float cylinderdy = cylinder1y - cylinder0y;
	float cylinderdz = cylinder1z - cylinder0z;
	float dx = vert[0] - cylinder0x;
	float dy = vert[1] - cylinder0y;
	float dz = vert[2] - cylinder0z;
	*t = dir[0] * dx + dir[1] * dy + dir[2] * dz;

	*t /= length;

	if (*t < 0) {
		//return false;
		*t = 0;
	}
	else if (*t > 1) {
		//return false;
		*t = 1;
	}

	dx = vert[0] - cylinder0x - (*t) * cylinderdx;
	dy = vert[1] - cylinder0y - (*t) * cylinderdy;
	dz = vert[2] - cylinder0z - (*t) * cylinderdz;

	float distance = sqrt(dx * dx + dy * dy + dz * dz);
	if (distance > radius) return false;

	//������ײ����ͶӰ�Ľ���
	float moveLength = sqrt(directDir[0] * directDir[0] + directDir[1] * directDir[1] + directDir[2] * directDir[2]);
	directDir[0] /= moveLength;
	directDir[1] /= moveLength;
	directDir[2] /= moveLength;

	collisionNormal[0] = directDir[0];
	collisionNormal[1] = directDir[1];
	collisionNormal[2] = directDir[2];

	float projectx = cylinder0x + (*t) * cylinderdx;
	float projecty = cylinder0y + (*t) * cylinderdy;
	float projectz = cylinder0z + (*t) * cylinderdz;

	//����local��,���һ��һԪ���η���
	float solution;
	float SN = (vert[0] - projectx) * (collisionNormal[0]) + (vert[1] - projecty) * (collisionNormal[1]) + (vert[2] - projectz) * (collisionNormal[2]);
	float SS = (vert[0] - projectx) * (vert[0] - projectx) + (vert[1] - projecty) * (vert[1] - projecty) + (vert[2] - projectz) * (vert[2] - projectz);
	solution = -SN + sqrt(SN * SN - SS + radius * radius);//ֻȡ����

	if (solution != solution) return false;

	collisionPos[0] = vert[0] + collisionNormal[0] * solution;
	collisionPos[1] = vert[1] + collisionNormal[1] * solution;
	collisionPos[2] = vert[2] + collisionNormal[2] * solution;

	//�ٴ���������
	dx = collisionPos[0] - projectx;
	dy = collisionPos[1] - projecty;
	dz = collisionPos[2] - projectz;
	distance = sqrt(dx * dx + dy * dy + dz * dz);
	collisionNormal[0] = dx / distance;
	collisionNormal[1] = dy / distance;
	collisionNormal[2] = dz / distance;

	return true;
}




//ʹ��������ײ���ı�������ײ����㷨
__global__ void calculateCollisionCylinderAdvance(
	float* cylinderLastPos, float* cylinderPos,
	float* cylinderDir, float* cylinderV,
	float halfLength, float radius,
	float* positions, float* velocity, float* force,
	unsigned int* isCollide,
	float* collisionDiag,
	float* volumnDiag,
	int vertexNum, float collisionStiffness, float frictionStiffness,
	float* collisionForce, float* directDir, float* cylinderShift)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float t = 0.0;
	float solution = 0.0;

	float collisionNormal[3];
	float collisionPos[3];
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//��ƫ��������Ϊ��λ����
	float shiftLength = sqrt(cylinderShift[0] * cylinderShift[0] + cylinderShift[1] * cylinderShift[1] + cylinderShift[2] * cylinderShift[2]);
	if (shiftLength > 0.01f) {
		cylinderShift[0] /= shiftLength;
		cylinderShift[1] /= shiftLength;
		cylinderShift[2] /= shiftLength;
	}

	//ָ��������ײ���ķ���
	float moveDir[3];
	moveDir[0] = directDir[indexX];
	moveDir[1] = directDir[indexY];
	moveDir[2] = directDir[indexZ];

	float tetPosition[3] = { positions[indexX] ,positions[indexY] ,positions[indexZ] };
	float toolMoveDir[3] = { cylinderLastPos[0] - cylinderPos[0],cylinderLastPos[1] - cylinderPos[1], cylinderLastPos[2] - cylinderPos[2] };
	float moveDistance = tetNormal_D(toolMoveDir);

	float ratio = 0.0f;
	float newPos[3];
	newPos[0] = cylinderPos[0] + cylinderShift[0] * ratio * radius;
	newPos[1] = cylinderPos[1] + cylinderShift[1] * ratio * radius;
	newPos[2] = cylinderPos[2] + cylinderShift[2] * ratio * radius;
	float w = moveDistance / radius;
	float enlarged_radius = radius * (1.5 - 0.5 / w);

	if (moveDistance > 0.05) {
		//�޸�ʹ��������ײ����������������ײ
		// �ж���ָ��������������ײ���
		bool collisionContinus = cylinderCollisionContinue(halfLength, moveDistance, enlarged_radius, cylinderPos, cylinderLastPos, cylinderDir, toolMoveDir, tetPosition, &t, collisionNormal, collisionPos, moveDir);
		// �޶���ָ��������������ײ���
		//bool collisionContinus = cylinderCollisionContinue_without_directDir(halfLength, moveDistance, enlarged_radius, cylinderPos, cylinderLastPos, cylinderDir, toolMoveDir, tetPosition, &t, collisionNormal, collisionPos);
		if (!collisionContinus) return;
		//printf("lianxu\n");
	}
	else {
		//ʹ��ָ�������������ײ���
		////bool collision = cylinderRayCollisionDetection(newPos, cylinderDir, triPositions[indexX], triPositions[indexY], triPositions[indexZ], moveDir, halfLength, radius, &t, &solution, collisionNormal, collisionPos);
		//bool collision = cylinderRayCollisionDetection(cylinderPos, cylinderDir, triPositions[indexX], triPositions[indexY], triPositions[indexZ], moveDir, halfLength, radius, &t, &solution, collisionNormal, collisionPos);
		float vert[3] = { positions[indexX], positions[indexY], positions[indexZ] };
		bool collision = cylinderCollision(cylinderPos, cylinderDir, vert, halfLength, radius, &t, collisionNormal, collisionPos);
		if (!collision) return;
		//printf("---lisan\n");
	}


	////ʹ��ָ�������������ײ���
	//bool collision = cylinderRayCollisionDetection(newPos, cylinderDir, triPositions[indexX], triPositions[indexY], triPositions[indexZ], moveDir, halfLength, radius, &t, &solution, collisionNormal, collisionPos);
	//if (!collision) return;

	float deltaPos[3];
	deltaPos[0] = collisionPos[0] - positions[indexX];
	deltaPos[1] = collisionPos[1] - positions[indexY];
	deltaPos[2] = collisionPos[2] - positions[indexZ];
	//float deltaPos_length = sqrt(deltaPos[0] * deltaPos[0] + deltaPos[1] * deltaPos[1] + deltaPos[2] * deltaPos[2]);
	//if (deltaPos_length > 1e-5)
	//{
	//	printf("thread %d deltaPos %f\tqg: [%f %f %f]\tcollision pos: [%f %f %f]\n", threadid, deltaPos_length, cylinderPos[0], cylinderPos[1], cylinderPos[2], collisionPos[0], collisionPos[1], collisionPos[2]);
	//}

	// ������ײ����Ӵ�����
	float temp[3];
	temp[0] = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
	temp[1] = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
	temp[2] = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
	collisionForce[indexX] += temp[0];
	collisionForce[indexY] += temp[1];
	collisionForce[indexZ] += temp[2];


	//����Ħ����
	float friction[3];
	friction[0] = 0.0;
	friction[1] = 0.0;
	friction[2] = 0.0;
	float frictionT[3];	//����Ħ����
	float frictionN[3];	//����Ħ����
						//��������˶����ٶȣ���������ٶ�������
	float v[3];
	v[0] = velocity[indexX] - cylinderV[0];
	v[1] = velocity[indexY] - cylinderV[1];
	v[2] = velocity[indexZ] - cylinderV[2];
	float c[3];
	c[0] = collisionStiffness * collisionNormal[0] * collisionNormal[0];
	c[1] = collisionStiffness * collisionNormal[1] * collisionNormal[1];
	c[2] = collisionStiffness * collisionNormal[2] * collisionNormal[2];
	c[0] += volumnDiag[threadid];
	c[1] += volumnDiag[threadid];
	c[2] += volumnDiag[threadid];
	//b-c(x-st)������໥������
	float relatedForce[3];
	relatedForce[0] = force[indexX] + c[0] * v[0] * 0.0009;
	relatedForce[1] = force[indexY] + c[1] * v[1] * 0.0009;
	relatedForce[2] = force[indexZ] + c[2] * v[2] * 0.0009;


	//�������������Ħ����
	float product = relatedForce[0] * collisionNormal[0]\
		+ relatedForce[1] * collisionNormal[1] \
		+ relatedForce[2] * collisionNormal[2];
	//if (product < 0) {
	//	//���㴹ֱ����
	//	frictionN[0] = -product*collisionNormal[0];
	//	frictionN[1] = -product*collisionNormal[1];
	//	frictionN[2] = -product*collisionNormal[2];

	//	//�����������
	//	relatedForce[0] -= product*collisionNormal[0];
	//	relatedForce[1] -= product*collisionNormal[1];
	//	relatedForce[2] -= product*collisionNormal[2];

	//	//�Ƿ񳬹���Ħ����ֵ
	//	float relatedlength = sqrt(relatedForce[0]* relatedForce[0]+ relatedForce[1]* relatedForce[1]+ relatedForce[2]* relatedForce[2]);
	//	
	//	float frictionNlength = sqrt(frictionN[0]* frictionN[0]+ frictionN[1]* frictionN[1]+ frictionN[2]* frictionN[2]);
	//	if (relatedlength <= frictionNlength*frictionStiffness) {
	//		
	//		frictionT[0] = -relatedForce[0];
	//		frictionT[1] = -relatedForce[1];
	//		frictionT[2] = -relatedForce[2];
	//
	//	}
	//	else {
	//		frictionT[0] = -frictionStiffness*frictionNlength*(relatedForce[0]/relatedlength);
	//		frictionT[1] = -frictionStiffness*frictionNlength*(relatedForce[1]/relatedlength);
	//		frictionT[2] = -frictionStiffness*frictionNlength*(relatedForce[2]/relatedlength);
	//	}

	//	friction[0] = frictionT[0];
	//	friction[1] = frictionT[1];
	//	friction[2] = frictionT[2];
	//}

	//������
	force[indexX] += temp[0] + friction[0];
	force[indexY] += temp[1] + friction[1];
	force[indexZ] += temp[2] + friction[2];
	//triForce[indexX] += friction[0];
	//triForce[indexY] += friction[1];
	//triForce[indexZ] += friction[2];


	//����Խ�Ԫ�ض�Ӧ��ֵ
	collisionDiag[indexX] += collisionStiffness * collisionNormal[0] * collisionNormal[0];
	collisionDiag[indexY] += collisionStiffness * collisionNormal[1] * collisionNormal[1];
	collisionDiag[indexZ] += collisionStiffness * collisionNormal[2] * collisionNormal[2];

	//���ñ�־λ
	isCollide[threadid] = 1;
}



int runcalculateCollisionSphere(float ball_radius, float p_collisionStiffness, int toolIdx, bool useClusterCollision)
{
	int  threadNum = 512;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	// ���ɵ���ײ�������ĵ��������������ϵ�ָ������ͶӰ�㡱�ķ���һ�¡�
	calculateCollisionSphere << <blockNum, threadNum >> > (toolPositionAndDirection_d, ball_radius,
		tetVertPos_d, tetVertisCollide_d, toolIdx, toolCollideFlag_d, 
		tetVertNonPenetrationDir_d,
		tetVertForce_d, tetVertCollisionForce_d, tetVertCollisionDiag_d, tetVertInsertionDepth_d,
		p_collisionStiffness,
		tetVertNum_d);

	//// ���ɵ���ײ����ָ����������һ��
	//calculateCollisionSphereFollowDDir << <blockNum, threadNum >> > (toolPositionAndDirection_d, ball_radius,
	//	tetVertPos_d, tetVertisCollide_d,
	//	tetVertNonPenetrationDir_d,
	//	tetVertCollisionForce_d, tetVertCollisionDiag_d, tetVertInsertionDepth_d,
	//	p_collisionStiffness,
	//	tetVertNum_d);

	calculateVec3Len << <blockNum, threadNum >> > (tetVertCollisionForce_d, tetVertCollisionForceLen_d, tetVertNum_d);
	cudaDeviceSynchronize();
	printCudaError("runcalculateCollisionSphere");
	return 0;
}

int runcalculateCollisionSphereMU(float ball_radius, float collisionStiffness, int toolIdx)
{
	int  threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;

	calculateCollisionSphere << <blockNum, threadNum >> > (toolPositionAndDirection_d, ball_radius,
		triVertPos_d, triVertisCollide_d, toolIdx, toolCollideFlag_d,
		triVertNonPenetrationDir_d,
		triVertForce_d, triVertCollisionForce_d, triVertCollisionDiag_d, tetVertInsertionDepth_d,
		collisionStiffness,
		triVertNum_d);

	cudaDeviceSynchronize();
	printCudaError("runcalculateCollisionSphereMU");
	return 0;
}

__global__ void calculateCollisionSphere(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness, int vertexNum)
{
	// ����������嶥��֮������λ�ù�ϵ�����嶥��ָ����������ײ��⣬�Զ���ʩ��������ײԼ��
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadid:%d, vertexNum:%d\n", threadid, vertexNum);
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float d[3] = { p[0] - ballPos[0], p[1] - ballPos[1], p[2] - ballPos[2] };
	float d_square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	float distance = sqrt(d_square);
	//if(threadid<10)
	//printf("threadid %d ball[%f %f %f], p[%f %f %f] distance %f\n", threadid, 
	//	ballPos[0], ballPos[1], ballPos[2],
	//	p[0], p[1], p[2],distance);
	if (distance < radius)
	{
		insertionDepth[threadid] = radius - distance;
		// ����Χ�ڣ�����ָ����������������ײ���
		float dDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };
		float collisionNormal[3] = { directDirection[indexX],directDirection[indexY] ,directDirection[indexZ] };
		// (d+x*directDir)^2==r^2 ��x

		float a = dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2];
		float b = 2 * (d[0] * dDir[0] + d[1] * dDir[1] + d[2] * dDir[2]);
		float c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radius * radius;
		float x0 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		float x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float x = x1;
		//printf("dDir[%f,%f,%f],a:%f b:%f c:%f x0:%f x1:%f\n", dDir[0], dDir[1], dDir[2], a, b, c, x0, x1);

		// collisionPos = p+x*dDir
		float collisionPos[3] = { p[0] + dDir[0] * x, p[1] + dDir[1] * x, p[2] + dDir[2] * x };
		// calibrated collision normal(ʵ�����ǽ������������ĺ�����1.�������������� 2.����ָ������)
		collisionNormal[0] = collisionPos[0] - ballPos[0];
		collisionNormal[1] = collisionPos[1] - ballPos[1];
		collisionNormal[2] = collisionPos[2] - ballPos[2];
		float col_len = sqrt(collisionNormal[0] * collisionNormal[0] + collisionNormal[1] * collisionNormal[1] + collisionNormal[2] * collisionNormal[2]);

		collisionNormal[0] /= col_len;
		collisionNormal[1] /= col_len;
		collisionNormal[2] /= col_len;

		float deltaPos[3] = { x * dDir[0], x * dDir[1], x * dDir[2] };

		float forcex = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
		float forcey = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
		float forcez = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
		force[indexX] += forcex;
		force[indexY] += forcey;
		force[indexZ] += forcez;
		collisionForce[indexX] += forcex;
		collisionForce[indexY] += forcey;
		collisionForce[indexZ] += forcez;
		float diagx = collisionStiffness * collisionNormal[0] * collisionNormal[0];
		float diagy = collisionStiffness * collisionNormal[1] * collisionNormal[1];
		float diagz = collisionStiffness * collisionNormal[2] * collisionNormal[2];
		collisionDiag[indexX] += diagx;
		collisionDiag[indexY] += diagy;
		collisionDiag[indexZ] += diagz;
		if(threadid== LOOK_THREAD)
		{
			//printf("threadid:%d triForce[%f %f %f] diag[%f %f %f]\n", threadid,
			//	triForce[indexX], triForce[indexY], triForce[indexZ],
			//	diagx, diagy, diagz);
		}

		isCollide[threadid] = 1;
		toolCollideFlag[toolIdx] = 1;

		float forceLen = sqrt(force[indexX] * force[indexX] + force[indexY] * force[indexY] + force[indexZ] * force[indexZ]);
		if ((x1 > -1e-6) && (x0 < 0))
		{
			x = x1;
		}
		else
		{
			printf("Error: x0=%f, x1=%f a:%f b:%f c:%f\n", x0, x1, a, b, c);
			printf("collision thread:%d, p:[%f,%f,%f], ball:[%f,%f,%f], x:%f dDir:[%f,%f,%f], force:[%f,%f,%f]\n",
				threadid, p[0], p[1], p[2], ballPos[0], ballPos[1], ballPos[2], x, dDir[0], dDir[1], dDir[2],
				force[indexX], force[indexY], force[indexZ]);
		}
		
	}
	else
	{
		insertionDepth[threadid] = 0;
	}
	return;
}
int runClearFc()
{
	printCudaError("runClearFc start");
	cudaMemset(hapticCollisionNum_d, 0, sizeof(int));
	cudaMemset(toolContactDeltaPos_triVert_d, 0.0f, triVertNum_d * 3 * sizeof(float));
	cudaMemset(totalFC_d, 0.0f, 3 * sizeof(float));
	cudaMemset(totalTC_d, 0.0f, 3 * sizeof(float));
	cudaMemset(totalPartial_FC_X_d, 0.0f, 9 * sizeof(float));
	cudaMemset(totalPartial_FC_Omega_d, 0.0f, 9 * sizeof(float));
	cudaMemset(totalPartial_TC_X_d, 0.0f, 9 * sizeof(float));
	cudaMemset(totalPartial_TC_Omega_d, 0.0f, 9 * sizeof(float));
	printCudaError("runClearFc");
	// ���ö���Ĭ��ͶӰλ��Ϊ��ǰλ��
	cudaMemcpy(triVertProjectedPos_d, triVertPos_d, triVertNum_d * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
	// ������ײ���
	cudaMemset(triVertisCollide_d, 0, triVertNum_d * sizeof(unsigned char));
	return 0;
}
int runHapticCollisionSphereForTri(float toolR, float p_collisionStiffness, float kc, int toolIdx)
{
	int  threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;

	hapticCollisionSphere <<<blockNum, threadNum>>>(toolPositionAndDirection_d, toolR,
		triVertPos_d, triVertisCollide_d, toolIdx, toolCollideFlag_d,
		triVertNonPenetrationDir_d,
		triVertForce_d, triVertCollisionForce_d, triVertCollisionDiag_d, triVertInsertionDepth_d,
		p_collisionStiffness,
		toolContactDeltaPos_triVert_d, totalFC_d, totalPartial_FC_X_d, kc,
		hapticCollisionNum_d, triVertNum_d);
	cudaDeviceSynchronize();
	printCudaError("HapticCollisionSphereForTri");
	return 0;
}
int runHapticCollisionSphereForTet(float toolR, float p_collisionStiffness, float kc, int toolIdx)
{
	int  threadNum = 512;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	hapticCollisionSphere << <blockNum, threadNum >> > (toolPositionAndDirection_d, toolR,
		tetVertPos_d, tetVertisCollide_d, toolIdx, toolCollideFlag_d,
		tetVertNonPenetrationDir_d,
		tetVertForce_d, tetVertCollisionForce_d, tetVertCollisionDiag_d, tetVertInsertionDepth_d,
		p_collisionStiffness,
		toolContactDeltaPos_triVert_d, totalFC_d, totalPartial_FC_X_d, kc,
		hapticCollisionNum_d, tetVertNum_d);
	cudaDeviceSynchronize();
	printCudaError("HapticCollisionSphereForTet");
	return 0;
}
__global__ void hapticCollisionSphere(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness,
	float* toolDeltaPos, float* F_c, float* partialFc, float k_c, int* collisionNumPtr, int vertexNum)
{
	// ����������嶥��֮������λ�ù�ϵ�����嶥��ָ����������ײ��⣬�Զ���ʩ��������ײԼ��
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadid:%d, vertexNum:%d\n", threadid, vertexNum);
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float d[3] = { p[0] - ballPos[0], p[1] - ballPos[1], p[2] - ballPos[2] };
	float d_square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	float distance = sqrt(d_square);
	//if(threadid<10)
	//printf("threadid %d ball[%f %f %f], p[%f %f %f] distance %f\n", threadid, 
	//	ballPos[0], ballPos[1], ballPos[2],
	//	p[0], p[1], p[2],distance);
	if (distance < radius)
	{
		insertionDepth[threadid] = radius - distance;
		// ����Χ�ڣ�����ָ����������������ײ���
		float dDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };
		float collisionNormal[3] = { directDirection[indexX],directDirection[indexY] ,directDirection[indexZ] };
		// (d+x*directDir)^2==r^2 ��x

		
		if (isnan(dDir[0]))
		{
			printf("threadid %d, nan in dDir, dealt as no collision, return\n", threadid);
			return;
		}
		float a = dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2];
		float b = 2 * (d[0] * dDir[0] + d[1] * dDir[1] + d[2] * dDir[2]);
		float c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radius * radius;
		float x0 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		float x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float x = x1;
		//printf("dDir[%f,%f,%f],a:%f b:%f c:%f x0:%f x1:%f\n", dDir[0], dDir[1], dDir[2], a, b, c, x0, x1);

		// collisionPos = p+x*dDir
		float collisionPos[3] = { p[0] + dDir[0] * x, p[1] + dDir[1] * x, p[2] + dDir[2] * x };
		// calibrated collision normal(ʵ�����ǽ������������ĺ�����1.�������������� 2.����ָ������)
		collisionNormal[0] = collisionPos[0] - ballPos[0];
		collisionNormal[1] = collisionPos[1] - ballPos[1];
		collisionNormal[2] = collisionPos[2] - ballPos[2];
		float col_len = sqrt(collisionNormal[0] * collisionNormal[0] + collisionNormal[1] * collisionNormal[1] + collisionNormal[2] * collisionNormal[2]);

		collisionNormal[0] /= col_len;
		collisionNormal[1] /= col_len;
		collisionNormal[2] /= col_len;

		float deltaPos[3] = { x * dDir[0], x * dDir[1], x * dDir[2] };

		float forcex = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
		float forcey = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
		float forcez = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
		//triForce[indexX] += forcex;
		//triForce[indexY] += forcey;
		//triForce[indexZ] += forcez;
		collisionForce[indexX] += forcex;
		collisionForce[indexY] += forcey;
		collisionForce[indexZ] += forcez;
		float diagx = collisionStiffness * collisionNormal[0] * collisionNormal[0];
		float diagy = collisionStiffness * collisionNormal[1] * collisionNormal[1];
		float diagz = collisionStiffness * collisionNormal[2] * collisionNormal[2];
		collisionDiag[indexX] += diagx;
		collisionDiag[indexY] += diagy;
		collisionDiag[indexZ] += diagz;
		
		if (threadid == LOOK_THREAD)
		{
			printf("threadid:%d force[%f %f %f] diag[%f %f %f]\n", threadid,
				collisionForce[indexX], collisionForce[indexY], collisionForce[indexZ],
				diagx, diagy, diagz);
		}
		float forceLen = sqrt(force[indexX] * force[indexX] + force[indexY] * force[indexY] + force[indexZ] * force[indexZ]);
		if ((x1 > -1e-6) && (x0 < 0))
		{
			x = x1;
		}
		else
		{
			printf("Error: x0=%f, x1=%f a:%f b:%f c:%f\n", x0, x1, a, b, c);
			printf("collision thread:%d, p:[%f,%f,%f], ball:[%f,%f,%f], x:%f dDir:[%f,%f,%f], force:[%f,%f,%f]\n",
				threadid, p[0], p[1], p[2], ballPos[0], ballPos[1], ballPos[2], x, dDir[0], dDir[1], dDir[2],
				force[indexX], force[indexY], force[indexZ]);
		}

		float partial_F_c[9] = { -dDir[0] * dDir[0] * k_c, -dDir[0] * dDir[1] * k_c, -dDir[0] * dDir[2] * k_c,
								 -dDir[0] * dDir[1] * k_c, -dDir[1] * dDir[1] * k_c, -dDir[1] * dDir[2] * k_c,
								 -dDir[0] * dDir[2] * k_c, -dDir[1] * dDir[2] * k_c, -dDir[2] * dDir[2] * k_c };
		// toolDeltaPos ���ߵ�����˶������붥����˶������෴
		atomicAdd(toolDeltaPos + 0, -deltaPos[0]);
		atomicAdd(toolDeltaPos + 1, -deltaPos[1]);
		atomicAdd(toolDeltaPos + 2, -deltaPos[2]);
		// F_c �빤�ߵ��˶�������ͬ
		atomicAdd(F_c + 0, -deltaPos[0] * k_c);
		atomicAdd(F_c + 1, -deltaPos[1] * k_c);
		atomicAdd(F_c + 2, -deltaPos[2] * k_c);
		atomicAdd(partialFc + 0, partial_F_c[0]);
		atomicAdd(partialFc + 1, partial_F_c[1]);
		atomicAdd(partialFc + 2, partial_F_c[2]);
		atomicAdd(partialFc + 3, partial_F_c[3]);
		atomicAdd(partialFc + 4, partial_F_c[4]);
		atomicAdd(partialFc + 5, partial_F_c[5]);
		atomicAdd(partialFc + 6, partial_F_c[6]);
		atomicAdd(partialFc + 7, partial_F_c[7]);
		atomicAdd(partialFc + 8, partial_F_c[8]);
		//printf("threadid %d dDir[%f %f %f] fc[%f %f %f]\n partialFc[\n%f %f %f\n%f %f %f\n%f %f %f]\n", threadid, 
		//	dDir[0], dDir[1], dDir[2],
		//	-deltaPos[0] * k_c, -deltaPos[1] * k_c, -deltaPos[2] * k_c,
		//	partial_F_c[0], partial_F_c[1], partial_F_c[2],
		//	partial_F_c[3], partial_F_c[4], partial_F_c[5], 
		//	partial_F_c[6], partial_F_c[7], partial_F_c[8]);

		atomicAdd(collisionNumPtr, 1);

		//printf("thread%d dDir:[%f %f %f] F_c:[%f %f %f]\n", threadid, 
		//	dDir[0], dDir[1], dDir[2],
		//	- deltaPos[0] * k_c, -deltaPos[1] * k_c, -deltaPos[2] * k_c);
		isCollide[threadid] = 1;
		toolCollideFlag[toolIdx] = 1;
	}
	else
	{
		insertionDepth[threadid] = 0;
	}
	return;
}

int runHapticCollisionSphere_Merged(float toolR, float p_collisionStiffness, float kc, int toolIdx)
{
	int  threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;

	//hapticCollisionSphere_Merge<< <blockNum, threadNum >> > (toolPositionAndDirection_d, toolR,
	//	triVertPos_d, triVertisCollide_d, toolIdx, toolCollideFlag_d,
	//	triVertNonPenetrationDir_d,
	//	triVertForce_d, triVertCollisionForce_d, triVertCollisionDiag_d, triVertInsertionDepth_d,
	//	tetVertForce_d, tetVertCollisionForce_d, tetVertCollisionDiag_d, tetVertInsertionDepth_d,
	//	triVert2TetVertMapping_d,
	//	p_collisionStiffness,
	//	toolContactDeltaPos_triVert_d, totalFC_d, totalPartial_FC_X_d, kc,
	//	hapticCollisionNum_d, triVertNum_d);

	hapticCollisionSphere_Merge_with_Torque << <blockNum, threadNum >> > (toolPositionAndDirection_d, toolR,
		triVertPos_d, triVertisCollide_d, toolIdx, toolCollideFlag_d,
		triVertNonPenetrationDir_d,
		triVertForce_d, triVertCollisionForce_d, triVertCollisionDiag_d, triVertInsertionDepth_d,
		tetVertForce_d, tetVertCollisionForce_d, tetVertCollisionDiag_d, tetVertInsertionDepth_d,
		triVert2TetVertMapping_d,
		p_collisionStiffness,
		toolContactDeltaPos_triVert_d, 
		totalFC_d, totalPartial_FC_X_d, totalPartial_FC_Omega_d,
		totalTC_d, totalPartial_TC_X_d, totalPartial_TC_Omega_d,
		kc, hapticCollisionNum_d, triVertNum_d);

	cudaDeviceSynchronize();
	printCudaError("HapticCollisionSphereMerged");
	return 0;
}

int runHapticCollisionCylinder_Merged_With_Sphere(float toolR, float param_toolLength, float p_collisionStiffness, float kc, int toolIdx, float sphere_R) {
	int  threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;

	float frictionStiffness = 10;
	// ��ײ���˺���
	hapticCollisionCylinder_Merge << <blockNum, threadNum >> > (
		toolPosePrev_d, toolPositionAndDirection_d,
		param_toolLength, toolR, sphere_R,
		triVertPos_d, triVertVelocity_d, triVert2TetVertMapping_d,
		triVertForce_d, triVertCollisionForce_d, triVertCollisionDiag_d, triVertInsertionDepth_d, triVertProjectedPos_d,
		tetVertForce_d, tetVertCollisionForce_d, tetVertCollisionDiag_d, tetVertInsertionDepth_d,
		triVertisCollide_d,
		triVertNum_d, p_collisionStiffness, frictionStiffness,
		triVertNonPenetrationDir_d, cylinderShift_d, hapticCollisionNum_d);

	//cudaDeviceSynchronize();
	printCudaError("HapticCollisionCylinderMerged");
	return 0;
}

int runHapticCollisionCylinder_Merged(float toolR, float param_toolLength, float p_collisionStiffness, float kc, int toolIdx)
{
	int  threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;

	float frictionStiffness = 10;
	// ��ײ���˺���
	hapticCollisionCylinder_Merge << <blockNum, threadNum >> > (
		toolPosePrev_d, toolPositionAndDirection_d,
		param_toolLength, toolR, -1,
		triVertPos_d, triVertVelocity_d, triVert2TetVertMapping_d,
		triVertForce_d, triVertCollisionForce_d, triVertCollisionDiag_d, triVertInsertionDepth_d, triVertProjectedPos_d,
		tetVertForce_d, tetVertCollisionForce_d, tetVertCollisionDiag_d, tetVertInsertionDepth_d,
		triVertisCollide_d,
		triVertNum_d, p_collisionStiffness, frictionStiffness,
		triVertNonPenetrationDir_d, cylinderShift_d, hapticCollisionNum_d);

	//cudaDeviceSynchronize();
	printCudaError("HapticCollisionCylinderMerged");
	return 0;
}

// sphere_R���Ϊ������Ϊû������ײ��...
__global__ void hapticCollisionCylinder_Merge(
	float* cylinderLastPos, float * cylinderPose,
	float halfLength, float radius, float sphere_r, float* triPositions,
	float* velocity, int* mapping, float* triForce,
	float* triCollisionForce, float* triCollisionDiag, float* triInsertionDepth, float* triVertProjectedPos, float* tetVertForce,
	float* tetVertCollisionForce, float* tetVertCollisionDiag, float* tetInsertionDepth, unsigned char* isCollide,
	int vertexNum,
	float collisionStiffness, float frictionStiffness, float* directDir,
	 float* cylinderShift, int* collisionNumPtr)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	//if(threadid==100)
	//	printf("dir: %f %f %f\n", cylinderPose[3], cylinderPose[4], cylinderPose[5]);
	float t = 0.0;
	float depth = 0.0;
	float solution = 0.0;

	float collisionNormal[3];
	float collisionPos[3];
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;
	int tetIdx0 = mapping[threadid * 2 + 0];
	int tetIdx0x = tetIdx0 * 3 + 0;
	int tetIdx0y = tetIdx0 * 3 + 1;
	int tetIdx0z = tetIdx0 * 3 + 2;
	int tetIdx1 = mapping[threadid * 2 + 1];
	int tetIdx1x = tetIdx1 * 3 + 0;
	int tetIdx1y = tetIdx1 * 3 + 1;
	int tetIdx1z = tetIdx1 * 3 + 2;

#ifdef OUTPUT_INFO
	if (threadid == 0)
	{
		printf("vert: %f %f %f cylinder pose: %f %f %f %f %f %f\n",
			tripositions[indexx], tripositions[indexy], tripositions[indexz],
			cylinderpose[0], cylinderpose[1], cylinderpose[2],
			cylinderpose[3], cylinderpose[4], cylinderpose[5]);
		printf("vert num: %d\n", vertexnum);
	}
#endif
	//��ƫ��������Ϊ��λ����
	float shiftLength = sqrt(cylinderShift[0] * cylinderShift[0] + cylinderShift[1] * cylinderShift[1] + cylinderShift[2] * cylinderShift[2]);
	if (shiftLength > 0.01f) {
		cylinderShift[0] /= shiftLength;
		cylinderShift[1] /= shiftLength;
		cylinderShift[2] /= shiftLength;
	}

	//ָ��������ײ���ķ���
	float moveDir[3];
	moveDir[0] = directDir[indexX];
	moveDir[1] = directDir[indexY];
	moveDir[2] = directDir[indexZ];

	float tetPosition[3] = { triPositions[indexX] ,triPositions[indexY] ,triPositions[indexZ] };
	float toolMoveDir[3] = { cylinderLastPos[0] - cylinderPose[0],cylinderLastPos[1] - cylinderPose[1], cylinderLastPos[2] - cylinderPose[2] };
	float moveDistance = tetNormal_D(toolMoveDir);

	float ratio = 0.0f;
	float newPos[3];
	newPos[0] = cylinderPose[0] + cylinderShift[0] * ratio * radius;
	newPos[1] = cylinderPose[1] + cylinderShift[1] * ratio * radius;
	newPos[2] = cylinderPose[2] + cylinderShift[2] * ratio * radius;
	float w = moveDistance / radius;
	float enlarged_radius = radius * (1.5 - 0.5 / w);

	{
		// ���㶥����ײ���depth�Ͷ��㱻�ų������߱����λ��collisionPos
		float vert[3] = { triPositions[indexX], triPositions[indexY], triPositions[indexZ] };
		float distance = -1;
		bool collision = cylinderCollision_withDepth(cylinderPose, 
			vert, halfLength, radius, sphere_r,
			&t, &depth, &distance, collisionNormal, collisionPos);    // �ж�������Ƿ�������ײ. collisionPos�Ƿ�������ײ�󶥵�Ӧ��ȥ�ĵط�...
		if (!collision) return;
	}

	float deltaPos[3];
	triVertProjectedPos[indexX] = collisionPos[0];
	triVertProjectedPos[indexY] = collisionPos[1];
	triVertProjectedPos[indexZ] = collisionPos[2];
	deltaPos[0] = collisionPos[0] - triPositions[indexX];
	deltaPos[1] = collisionPos[1] - triPositions[indexY];
	deltaPos[2] = collisionPos[2] - triPositions[indexZ];
	float insertionDepth = sqrt(deltaPos[0] * deltaPos[0] + deltaPos[1] * deltaPos[1] + deltaPos[2] * deltaPos[2]);
	triInsertionDepth[threadid] = insertionDepth;    // �������..

	// ������ײ����Ӵ�����
	float temp[3];
	temp[0] = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
	temp[1] = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
	temp[2] = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);

	// �ѽӴ���ʩ�ӵ�������ײ�ı���������������������񶥵���
	triCollisionForce[indexX] += temp[0];
	triCollisionForce[indexY] += temp[1];
	triCollisionForce[indexZ] += temp[2];
	tetVertCollisionForce[tetIdx0x] += temp[0] / 2;
	tetVertCollisionForce[tetIdx0y] += temp[1] / 2;
	tetVertCollisionForce[tetIdx0z] += temp[2] / 2;
	tetVertCollisionForce[tetIdx1x] += temp[0] / 2;
	tetVertCollisionForce[tetIdx1y] += temp[1] / 2;
	tetVertCollisionForce[tetIdx1z] += temp[2] / 2;

	//����Խ�Ԫ�ض�Ӧ��ֵ
	float diagx = collisionStiffness * collisionNormal[0] * collisionNormal[0];
	float diagy = collisionStiffness * collisionNormal[1] * collisionNormal[1];
	float diagz = collisionStiffness * collisionNormal[2] * collisionNormal[2];
	triCollisionDiag[indexX] += diagx;
	triCollisionDiag[indexY] += diagy;
	triCollisionDiag[indexZ] += diagz;
	tetVertCollisionDiag[tetIdx0x] += diagx;
	tetVertCollisionDiag[tetIdx0y] += diagy;
	tetVertCollisionDiag[tetIdx0z] += diagz;
	tetVertCollisionDiag[tetIdx1x] += diagx;
	tetVertCollisionDiag[tetIdx1y] += diagy;
	tetVertCollisionDiag[tetIdx1z] += diagz;

	//���ñ�־λ
	isCollide[threadid] = 1;
	atomicAdd(collisionNumPtr, 1);
}
int runDeviceCalculateContact(float k_c)
{
	int  threadNum = 512;
	int blockNum = (triVertNum_d + threadNum - 1) / threadNum;
	
	CalculateContact<< <blockNum, threadNum >> > (triVertNonPenetrationDir_d, triVertPos_d,
		triVertProjectedPos_d, triVertInsertionDepth_d,
		toolPositionAndDirection_d, toolContactDeltaPos_triVert_d, triVertisCollide_d,
		totalFC_d, totalPartial_FC_X_d, totalPartial_FC_Omega_d,
		totalTC_d, totalPartial_TC_X_d, totalPartial_TC_Omega_d,
		k_c);
	printCudaError("runDeviceCalculateContact");
	cudaDeviceSynchronize();
	return 0;
}
__global__ void CalculateContact(float* nonPenetrationDirection, float* triVertPosition, 
	float* projectedPosition, float* insertionDepth,
	float* toolPose, float* toolDeltaPos,
	unsigned char* isCollide, float* total_FC, float* totalPartial_FC_X,
	float* totalPartial_FC_Omega, float* total_TC, float* totalPartial_TC_X,
	float* totalPartial_TC_Omega, float k_c)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (isCollide[threadid] != 1) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;
	float dDir[3] = { nonPenetrationDirection[indexX], nonPenetrationDirection[indexY], nonPenetrationDirection[indexZ] };
	
	float deltaPos[3] = { projectedPosition[indexX] - triVertPosition[indexX],projectedPosition[indexY] - triVertPosition[indexY], projectedPosition[indexZ] - triVertPosition[indexZ] };
	float d = -insertionDepth[threadid];
	float p[3] = { triVertPosition[indexX],triVertPosition[indexY],triVertPosition[indexZ] };
	float l = 1;
	float X_grasp[3] = { toolPose[0] + toolPose[3] * l,
		toolPose[1] + toolPose[4] * l,
		toolPose[2] + toolPose[5] * l };
	DeviceCalculateContact(dDir, k_c, d, p, X_grasp,
		toolDeltaPos,
		total_FC, totalPartial_FC_X, totalPartial_FC_Omega,
		total_TC, totalPartial_TC_X, totalPartial_TC_Omega, false);
}
__global__ void hapticCollisionSphere_Merge(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection,
	float* triForce, float* triCollisionForce, float* triCollisionDiag, float* triInsertionDepth,
	float* tetVertForce, float* tetVertCollisionForce, float* tetVertCollisionDiag, float* tetVertInsertionDepth,
	int* mapping,
	float collisionStiffness,
	float* toolDeltaPos, float* F_c, float* partialFc, float k_c, int* collisionNumPtr, int vertexNum)
{
	// ����������嶥��֮������λ�ù�ϵ�����嶥��ָ����������ײ��⣬�Զ���ʩ��������ײԼ��
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float d[3] = { p[0] - ballPos[0], p[1] - ballPos[1], p[2] - ballPos[2] };
	float d_square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	float distance = sqrt(d_square);
	//if(threadid<10)

	int tetIdx0 = mapping[threadid * 2 + 0];
	int tetIdx0x = tetIdx0 * 3 + 0;
	int tetIdx0y = tetIdx0 * 3 + 1;
	int tetIdx0z = tetIdx0 * 3 + 2;
	int tetIdx1 = mapping[threadid * 2 + 1];
	int tetIdx1x = tetIdx1 * 3 + 0;
	int tetIdx1y = tetIdx1 * 3 + 1;
	int tetIdx1z = tetIdx1 * 3 + 2;

	float ori_radius = radius;
	radius *= 1.05;
	if (distance < radius)
	{
		//printf("threadid %d ball[%f %f %f], p[%f %f %f] distance %f\n", threadid,
		//	ballPos[0], ballPos[1], ballPos[2],
		//	p[0], p[1], p[2], distance);
		triInsertionDepth[threadid] = radius - distance;
		// ����Χ�ڣ�����ָ����������������ײ���
		float dDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };
		float collisionNormal[3] = { directDirection[indexX],directDirection[indexY] ,directDirection[indexZ] };
		// (d+x*directDir)^2==r^2 ��x
		if (isnan(dDir[0]))
		{
			printf("threadid %d, nan in dDir, dealt as no collision, return\n", threadid);
			return;
		}
		float a = dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2];
		float b = 2 * (d[0] * dDir[0] + d[1] * dDir[1] + d[2] * dDir[2]);
		float c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radius * radius;
		float x0 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		float x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float x = x1;
		//printf("dDir[%f,%f,%f],a:%f b:%f c:%f x0:%f x1:%f\n", dDir[0], dDir[1], dDir[2], a, b, c, x0, x1);

		// collisionPos = p+x*dDir
		float collisionPos[3] = { p[0] + dDir[0] * x, p[1] + dDir[1] * x, p[2] + dDir[2] * x };
		// calibrated collision normal(ʵ�����ǽ������������ĺ�����1.�������������� 2.����ָ������)
		collisionNormal[0] = collisionPos[0] - ballPos[0];
		collisionNormal[1] = collisionPos[1] - ballPos[1];
		collisionNormal[2] = collisionPos[2] - ballPos[2];
		float col_len = sqrt(collisionNormal[0] * collisionNormal[0] + collisionNormal[1] * collisionNormal[1] + collisionNormal[2] * collisionNormal[2]);

		collisionNormal[0] /= col_len;
		collisionNormal[1] /= col_len;
		collisionNormal[2] /= col_len;

		float deltaPos[3] = { x * dDir[0], x * dDir[1], x * dDir[2] };

		float forcex = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
		float forcey = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
		float forcez = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
		//triForce[indexX] += forcex;
		//triForce[indexY] += forcey;
		//triForce[indexZ] += forcez;
		triCollisionForce[indexX] += forcex;
		triCollisionForce[indexY] += forcey;
		triCollisionForce[indexZ] += forcez;
		tetVertCollisionForce[tetIdx0x] += forcex / 2;
		tetVertCollisionForce[tetIdx0y] += forcey / 2;
		tetVertCollisionForce[tetIdx0z] += forcez / 2;
		tetVertCollisionForce[tetIdx1x] += forcex / 2;
		tetVertCollisionForce[tetIdx1y] += forcey / 2;
		tetVertCollisionForce[tetIdx1z] += forcez / 2;

		float diagx = collisionStiffness * collisionNormal[0] * collisionNormal[0];
		float diagy = collisionStiffness * collisionNormal[1] * collisionNormal[1];
		float diagz = collisionStiffness * collisionNormal[2] * collisionNormal[2];
		triCollisionDiag[indexX] += diagx;
		triCollisionDiag[indexY] += diagy;
		triCollisionDiag[indexZ] += diagz;
		tetVertCollisionDiag[tetIdx0x] += diagx;
		tetVertCollisionDiag[tetIdx0y] += diagy;
		tetVertCollisionDiag[tetIdx0z] += diagz;
		tetVertCollisionDiag[tetIdx1x] += diagx;
		tetVertCollisionDiag[tetIdx1y] += diagy;
		tetVertCollisionDiag[tetIdx1z] += diagz;

		if (threadid == LOOK_THREAD)
		{
			printf("threadid:%d force[%f %f %f] diag[%f %f %f]\n", threadid,
				triCollisionForce[indexX], triCollisionForce[indexY], triCollisionForce[indexZ],
				diagx, diagy, diagz);
		}
		float forceLen = sqrt(triForce[indexX] * triForce[indexX] + triForce[indexY] * triForce[indexY] + triForce[indexZ] * triForce[indexZ]);
		if ((x1 > -1e-6) && (x0 < 0))
		{
			x = x1;
		}
		else
		{
			printf("Error: x0=%f, x1=%f a:%f b:%f c:%f\n", x0, x1, a, b, c);
			printf("collision thread:%d, p:[%f,%f,%f], ball:[%f,%f,%f], x:%f dDir:[%f,%f,%f], force:[%f,%f,%f]\n",
				threadid, p[0], p[1], p[2], ballPos[0], ballPos[1], ballPos[2], x, dDir[0], dDir[1], dDir[2],
				triForce[indexX], triForce[indexY], triForce[indexZ]);
		}
		if (distance < ori_radius)
		{
			float partial_F_c[9] = { -dDir[0] * dDir[0] * k_c, -dDir[0] * dDir[1] * k_c, -dDir[0] * dDir[2] * k_c,
									 -dDir[0] * dDir[1] * k_c, -dDir[1] * dDir[1] * k_c, -dDir[1] * dDir[2] * k_c,
									 -dDir[0] * dDir[2] * k_c, -dDir[1] * dDir[2] * k_c, -dDir[2] * dDir[2] * k_c };
			// toolDeltaPos ���ߵ�����˶������붥����˶������෴
			atomicAdd(toolDeltaPos + 0, -deltaPos[0]);
			atomicAdd(toolDeltaPos + 1, -deltaPos[1]);
			atomicAdd(toolDeltaPos + 2, -deltaPos[2]);
			// F_c �빤�ߵ��˶�������ͬ
			atomicAdd(F_c + 0, -deltaPos[0] * k_c);
			atomicAdd(F_c + 1, -deltaPos[1] * k_c);
			atomicAdd(F_c + 2, -deltaPos[2] * k_c);
			atomicAdd(partialFc + 0, partial_F_c[0]);
			atomicAdd(partialFc + 1, partial_F_c[1]);
			atomicAdd(partialFc + 2, partial_F_c[2]);
			atomicAdd(partialFc + 3, partial_F_c[3]);
			atomicAdd(partialFc + 4, partial_F_c[4]);
			atomicAdd(partialFc + 5, partial_F_c[5]);
			atomicAdd(partialFc + 6, partial_F_c[6]);
			atomicAdd(partialFc + 7, partial_F_c[7]);
			atomicAdd(partialFc + 8, partial_F_c[8]);
			//printf("threadid %d dDir[%f %f %f] fc[%f %f %f]\n partialFc[\n%f %f %f\n%f %f %f\n%f %f %f]\n", threadid, 
			//	dDir[0], dDir[1], dDir[2],
			//	-deltaPos[0] * k_c, -deltaPos[1] * k_c, -deltaPos[2] * k_c,
			//	partial_F_c[0], partial_F_c[1], partial_F_c[2],
			//	partial_F_c[3], partial_F_c[4], partial_F_c[5], 
			//	partial_F_c[6], partial_F_c[7], partial_F_c[8]);
		}

		atomicAdd(collisionNumPtr, 1);

		//printf("thread%d dDir:[%f %f %f] F_c:[%f %f %f]\n", threadid, 
		//	dDir[0], dDir[1], dDir[2],
		//	- deltaPos[0] * k_c, -deltaPos[1] * k_c, -deltaPos[2] * k_c);
		isCollide[threadid] = 1;
		toolCollideFlag[toolIdx] = 1;
	}
	else
	{
		triInsertionDepth[threadid] = 0;
	}
	return;
}

__global__ void hapticCollisionSphere_Merge_with_Torque(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, 
	float * triForce, float * triCollisionForce, float * triCollisionDiag, float * triInsertionDepth,
	float* tetVertForce, float* tetVertCollisionForce, float* tetVertCollisionDiag, float* tetVertInsertionDepth, 
	int* mapping, 
	float collisionStiffness,
	float* toolDeltaPos, 
	float* total_FC, float* totalPartial_FC_X, float* totalPartial_FC_Omega,
	float* total_TC, float* totalPartial_TC_X, float* totalPartial_TC_Omega,
	float k_c, int* collisionNumPtr, int vertexNum)
{
	// ����������嶥��֮������λ�ù�ϵ�����嶥��ָ����������ײ��⣬�Զ���ʩ��������ײԼ��
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadid:%d, vertexNum:%d\n", threadid, vertexNum);
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float d[3] = { p[0] - ballPos[0], p[1] - ballPos[1], p[2] - ballPos[2] };
	float d_square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	float distance = sqrt(d_square);
	//if(threadid<10)

	int tetIdx0 = mapping[threadid * 2 + 0];
	int tetIdx0x = tetIdx0 * 3 + 0;
	int tetIdx0y = tetIdx0 * 3 + 1;
	int tetIdx0z = tetIdx0 * 3 + 2;
	int tetIdx1 = mapping[threadid * 2 + 1];
	int tetIdx1x = tetIdx1 * 3 + 0;
	int tetIdx1y = tetIdx1 * 3 + 1;
	int tetIdx1z = tetIdx1 * 3 + 2;

	float ori_radius = radius;
	radius *= 1.05;
	if (distance < radius)
	{
		//printf("threadid %d ball[%f %f %f], p[%f %f %f] distance %f\n", threadid,
		//	ballPos[0], ballPos[1], ballPos[2],
		//	p[0], p[1], p[2], distance);
		triInsertionDepth[threadid] = radius - distance;
		// ����Χ�ڣ�����ָ����������������ײ���
		float dDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };
		float collisionNormal[3] = { directDirection[indexX],directDirection[indexY] ,directDirection[indexZ] };
		// (d+x*directDir)^2==r^2 ��x
		if (isnan(dDir[0]))
		{
			printf("threadid %d, nan in dDir, dealt as no collision, return\n", threadid);
			return;
		}
		float a = dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2];
		float b = 2 * (d[0] * dDir[0] + d[1] * dDir[1] + d[2] * dDir[2]);
		float c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radius * radius;
		float x0 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		float x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float x = x1;
		//printf("dDir[%f,%f,%f],a:%f b:%f c:%f x0:%f x1:%f\n", dDir[0], dDir[1], dDir[2], a, b, c, x0, x1);

		// collisionPos = p+x*dDir
		float collisionPos[3] = { p[0] + dDir[0] * x, p[1] + dDir[1] * x, p[2] + dDir[2] * x };
		// calibrated collision normal(ʵ�����ǽ������������ĺ�����1.�������������� 2.����ָ������)
		collisionNormal[0] = collisionPos[0] - ballPos[0];
		collisionNormal[1] = collisionPos[1] - ballPos[1];
		collisionNormal[2] = collisionPos[2] - ballPos[2];
		float col_len = sqrt(collisionNormal[0] * collisionNormal[0] + collisionNormal[1] * collisionNormal[1] + collisionNormal[2] * collisionNormal[2]);

		collisionNormal[0] /= col_len;
		collisionNormal[1] /= col_len;
		collisionNormal[2] /= col_len;

		float deltaPos[3] = { x * dDir[0], x * dDir[1], x * dDir[2] };

		float forcex = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
		float forcey = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
		float forcez = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
		//triForce[indexX] += forcex;
		//triForce[indexY] += forcey;
		//triForce[indexZ] += forcez;
		triCollisionForce[indexX] += forcex;
		triCollisionForce[indexY] += forcey;
		triCollisionForce[indexZ] += forcez;
		tetVertCollisionForce[tetIdx0x] += forcex / 2;
		tetVertCollisionForce[tetIdx0y] += forcey / 2;
		tetVertCollisionForce[tetIdx0z] += forcez / 2;
		tetVertCollisionForce[tetIdx1x] += forcex / 2;
		tetVertCollisionForce[tetIdx1y] += forcey / 2;
		tetVertCollisionForce[tetIdx1z] += forcez / 2;

		float diagx = collisionStiffness * collisionNormal[0] * collisionNormal[0];
		float diagy = collisionStiffness * collisionNormal[1] * collisionNormal[1];
		float diagz = collisionStiffness * collisionNormal[2] * collisionNormal[2];
		triCollisionDiag[indexX] += diagx;
		triCollisionDiag[indexY] += diagy;
		triCollisionDiag[indexZ] += diagz;
		tetVertCollisionDiag[tetIdx0x] += diagx;
		tetVertCollisionDiag[tetIdx0y] += diagy;
		tetVertCollisionDiag[tetIdx0z] += diagz;
		tetVertCollisionDiag[tetIdx1x] += diagx;
		tetVertCollisionDiag[tetIdx1y] += diagy;
		tetVertCollisionDiag[tetIdx1z] += diagz;

		if (threadid == LOOK_THREAD)
		{
			printf("threadid:%d force[%f %f %f] diag[%f %f %f]\n", threadid,
				triCollisionForce[indexX], triCollisionForce[indexY], triCollisionForce[indexZ],
				diagx, diagy, diagz);
		}
		float forceLen = sqrt(triForce[indexX] * triForce[indexX] + triForce[indexY] * triForce[indexY] + triForce[indexZ] * triForce[indexZ]);
		if ((x1 > -1e-6) && (x0 < 0))
		{
			x = x1;
		}
		else
		{
			printf("check insertion depth Error: x0=%f, x1=%f a:%f b:%f c:%f\n", x0, x1, a, b, c);
			printf("collision thread:%d, p:[%f,%f,%f], ball:[%f,%f,%f], x:%f dDir:[%f,%f,%f], triForce:[%f,%f,%f]\n",
				threadid, p[0], p[1], p[2], ballPos[0], ballPos[1], ballPos[2], x, dDir[0], dDir[1], dDir[2],
				triForce[indexX], triForce[indexY], triForce[indexZ]);
		}
		if(distance<ori_radius)
		{
			bool verbose = true;
			//if (*collisionNumPtr == 0)
			//	verbose = true;
			DeviceCalculateContact(dDir, k_c, x, p, ballPos,
				toolDeltaPos, 
				total_FC, totalPartial_FC_X, totalPartial_FC_Omega,
				total_TC, totalPartial_TC_X, totalPartial_TC_Omega, verbose);
		}

		atomicAdd(collisionNumPtr, 1);

		//printf("thread%d dDir:[%f %f %f] F_c:[%f %f %f]\n", threadid, 
		//	dDir[0], dDir[1], dDir[2],
		//	- deltaPos[0] * k_c, -deltaPos[1] * k_c, -deltaPos[2] * k_c);
		isCollide[threadid] = 1;
		toolCollideFlag[toolIdx] = 1;
	}
	else
	{
		triInsertionDepth[threadid] = 0;
	}
	return;
}
__device__ void DeviceCalculateContact(
	float* dDir, float k_c, float depth, float* p, float * Xg_grasp,
	float* toolDeltaPos, 
	float* total_FC, float* totalPartial_FC_X, float* totalPartial_FC_Omega,
	float* total_TC, float* totalPartial_TC_X, 
	float* totalPartial_TC_Omega, bool printInfo)
{
	float point_F_c[3], point_TC[3];
	float r[3] = { p[0] - Xg_grasp[0],p[1] - Xg_grasp[1] ,p[2] - Xg_grasp[2] };

	float partialFCX[9], partialFCOmega[9];
	DeviceCalculateFC(k_c, depth, dDir, point_F_c, toolDeltaPos, total_FC);
	DeviceCalculatePartial_FC_X(dDir, k_c, partialFCX, totalPartial_FC_X);
	DeviceCalculatePartial_FC_Omega(dDir, r, k_c, depth, partialFCOmega, totalPartial_FC_Omega);

	float partialTCX[9], partialTCOmega[9];
	DeviceCalculateTC(point_F_c, r, point_TC, total_TC);
	DeviceCalculatePartial_TC_X(k_c, r, dDir, point_F_c, partialTCX, totalPartial_TC_X);
	DeviceCalculatePartial_TC_Omega(k_c, point_F_c, r, depth, dDir, partialTCOmega, totalPartial_TC_Omega);
	if (printInfo)
	{
		printf("FC: %f %f %f\n TC: %f %f %f\npartialFCX:\n%f %f %f\n%f %f %f\n%f %f %f\npartialFCOmega:\n%f %f %f\n%f %f %f\n%f %f %f\n", 
			point_F_c[0], point_F_c[1], point_F_c[2],
			point_TC[0], point_TC[1], point_TC[2],
			partialFCX[0], partialFCX[1], partialFCX[2],
			partialFCX[3], partialFCX[4], partialFCX[5],
			partialFCX[6], partialFCX[7], partialFCX[8],
			partialFCOmega[0], partialFCOmega[1], partialFCOmega[2],
			partialFCOmega[3], partialFCOmega[4], partialFCOmega[5],
			partialFCOmega[6], partialFCOmega[7], partialFCOmega[8]);
		//printf("partialTCX:\n%f %f %f\n%f %f %f\n%f %f %f\npartialTCOmega:\n%f %f %f\n%f %f %f\n%f %f %f\n----------------------------------\n",
		//	partialTCX[0], partialTCX[1], partialTCX[2],
		//	partialTCX[3], partialTCX[4], partialTCX[5],
		//	partialTCX[6], partialTCX[7], partialTCX[8],
		//	partialTCOmega[0], partialTCOmega[1], partialTCOmega[2],
		//	partialTCOmega[3], partialTCOmega[4], partialTCOmega[5],
		//	partialTCOmega[6], partialTCOmega[7], partialTCOmega[8]);
	}
}

__device__ void DeviceCalculateFC(
	float k_c, float* pointDeltaPos,
	float* F_c,
	float* toolDeltaPos, float * total_F_c)
{
	// deltaPos: ��������ײ��������λ�ơ�
	// ���ߵ����λ����deltaPos�෴
	atomicAdd(toolDeltaPos + 0, -pointDeltaPos[0]);
	atomicAdd(toolDeltaPos + 1, -pointDeltaPos[1]);
	atomicAdd(toolDeltaPos + 2, -pointDeltaPos[2]);

	// F_c ���㷽ʽ1�����ݶ���λ�Ƽ���Ӵ���
	F_c[0]= -pointDeltaPos[0] * k_c;
	F_c[1]= -pointDeltaPos[1] * k_c;
	F_c[2]= -pointDeltaPos[2] * k_c;

	// printf("calculate FC: pointDeltaPos[%f %f %f], kc:%f\n", pointDeltaPos[0], pointDeltaPos[1], pointDeltaPos[2], k_c);
	// F_c �빤�ߵ��˶�������ͬ
	atomicAdd(total_F_c + 0, F_c[0]);
	atomicAdd(total_F_c + 1, F_c[1]);
	atomicAdd(total_F_c + 2, F_c[2]);
}

__device__ void DeviceCalculateFC(
	float k_c, float d, float* dDir,
	float* F_c,
	float* toolDeltaPos, float* total_F_c)
{
	// deltaPos: ��������ײ��������λ�ơ�
	// ���ߵ����λ����deltaPos�෴
	float pointDeltaPos[3] = { dDir[0] * d, dDir[1] * d, dDir[2] * d };
	atomicAdd(toolDeltaPos + 0, -pointDeltaPos[0]);
	atomicAdd(toolDeltaPos + 1, -pointDeltaPos[1]);
	atomicAdd(toolDeltaPos + 2, -pointDeltaPos[2]);

	// F_c ���㷽ʽ2������Ƕ����ȺͶ���ָ����������Ӵ���
	F_c[0] = d * dDir[0] * k_c;
	F_c[1] = d * dDir[1] * k_c;
	F_c[2] = d * dDir[2] * k_c;
	// printf("calculate FC: pointDeltaPos[%f %f %f], kc:%f\n", pointDeltaPos[0], pointDeltaPos[1], pointDeltaPos[2], k_c);
	// F_c �빤�ߵ��˶�������ͬ
	atomicAdd(total_F_c + 0, F_c[0]);
	atomicAdd(total_F_c + 1, F_c[1]);
	atomicAdd(total_F_c + 2, F_c[2]);
}
__device__ void DeviceCalculatePartial_FC_X(
	float* dDir, float k_c, float* partialFCX, float * totalPartialFc)
{
	float partial_F_c[9] = { -dDir[0] * dDir[0] * k_c, -dDir[0] * dDir[1] * k_c, -dDir[0] * dDir[2] * k_c,
						 -dDir[0] * dDir[1] * k_c, -dDir[1] * dDir[1] * k_c, -dDir[1] * dDir[2] * k_c,
						 -dDir[0] * dDir[2] * k_c, -dDir[1] * dDir[2] * k_c, -dDir[2] * dDir[2] * k_c };
	partialFCX[0] = partial_F_c[0];
	partialFCX[1] = partial_F_c[1];
	partialFCX[2] = partial_F_c[2];
	partialFCX[3] = partial_F_c[3];
	partialFCX[4] = partial_F_c[4];
	partialFCX[5] = partial_F_c[5];
	partialFCX[6] = partial_F_c[6];
	partialFCX[7] = partial_F_c[7];
	partialFCX[8] = partial_F_c[8];
	// toolDeltaPos ���ߵ�����˶������붥����˶������෴
	atomicAdd(totalPartialFc + 0, partial_F_c[0]);
	atomicAdd(totalPartialFc + 1, partial_F_c[1]);
	atomicAdd(totalPartialFc + 2, partial_F_c[2]);
	atomicAdd(totalPartialFc + 3, partial_F_c[3]);
	atomicAdd(totalPartialFc + 4, partial_F_c[4]);
	atomicAdd(totalPartialFc + 5, partial_F_c[5]);
	atomicAdd(totalPartialFc + 6, partial_F_c[6]);
	atomicAdd(totalPartialFc + 7, partial_F_c[7]);
	atomicAdd(totalPartialFc + 8, partial_F_c[8]);
}

__device__ void DeviceCalculatePartial_FC_Omega(
	float* normal, float* r, float k_c, float depth,
	float* partial,
	float* totalPartial_FC_Omega
)
{
	float nnT[9], rTilde[9], nTilde[9], part0[9], part1[9];
	DeviceVec3MulVec3T(normal, normal, nnT);
	DeviceVec3toSkewSymmetricMatrix(r, rTilde);
	DeviceMat3MulMat3(nnT, rTilde, part0);
	DeviceScaleMulMat3(k_c, part0, part0);

	DeviceVec3toSkewSymmetricMatrix(normal, nTilde);
	DeviceScaleMulMat3(k_c * depth, nTilde, part1);

	DeviceMat3AddMat3(part0, part1, partial);
	
	DeviceMat3AtomicAddMat3(partial, totalPartial_FC_Omega);
}

__device__ void DeviceCalculatePartial_TC_X(
	float k_c, float* r, float* dDir, float* F_c,
	float* partialTCX,
	float* totalPartialTCX
)
{
	float nnt[9], r_tilde[9], part0[9], FC_tilde[9];
	DeviceVec3MulVec3T(dDir, dDir, nnt);
	DeviceVec3toSkewSymmetricMatrix(r, r_tilde);
	DeviceMat3MulMat3(r_tilde, nnt, part0);
	DeviceScaleMulMat3(-k_c, part0, part0);

	DeviceVec3toSkewSymmetricMatrix(F_c, FC_tilde);
	DeviceMat3AddMat3(part0, FC_tilde, partialTCX);

	DeviceMat3AtomicAddMat3(partialTCX, totalPartialTCX);
}

__device__ void DeviceCalculatePartial_TC_Omega(
	float k_c, float* F_c, float* r, float depth, float* dDir,
	float* partialTCOmega,
	float* totalPartialTCOmega)
{
	float part0[9], part1[9], part2[9], nnt[9];
	float FC_tilde[9], r_tilde[9], dDir_tilde[9];
	DeviceVec3toSkewSymmetricMatrix(r, r_tilde);
	DeviceVec3toSkewSymmetricMatrix(F_c, FC_tilde);
	DeviceMat3MulMat3(FC_tilde, r_tilde, part0);
	DeviceScaleMulMat3(-1, part0, part0);

	DeviceVec3MulVec3T(dDir, dDir, nnt);
	DeviceMat3MulMat3(nnt, r_tilde, part1);
	DeviceMat3MulMat3(r_tilde, part1, part1);
	DeviceScaleMulMat3(k_c, part1, part1);

	DeviceVec3toSkewSymmetricMatrix(dDir, dDir_tilde);
	DeviceMat3MulMat3(r_tilde, dDir_tilde, part2);
	DeviceScaleMulMat3(k_c * depth, part2, part2);

	DeviceMat3AddMat3(part0, part1, partialTCOmega);
	DeviceMat3AddMat3(part2, partialTCOmega, partialTCOmega);

	DeviceMat3AtomicAddMat3(partialTCOmega, totalPartialTCOmega);
}

// ���������о���
__device__ void DeviceVec3toSkewSymmetricMatrix(
	float* v, float* m)
{
	m[0] = 0; m[1] = -v[2]; m[2] = v[1];
	m[3] = v[2]; m[4] = 0; m[5] = -v[0];
	m[6] = -v[1]; m[7] = v[0]; m[8] = 0;
}
// ���������о���
__device__ void DeviceMatrixDotVec(
	float* m, float* v, float* result
)
{
	result[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
	result[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
	result[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
	//printf("matrix:\n %f %f %f\n%f %f %f\n%f %f %f\nvector %f %f %f\nresult %f %f %f\n", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], v[0], v[1], v[2], result[0], result[1], result[2]);
}
// ���������о���
__device__ void DeviceMat3MulMat3(
	float* m0, float* m1, float* result)
{
	result[0] = m0[0] * m1[0] + m0[1] * m1[3] + m0[2] * m1[6];
	result[1] = m0[0] * m1[1] + m0[1] * m1[4] + m0[2] * m1[7];
	result[2] = m0[0] * m1[2] + m0[1] * m1[5] + m0[2] * m1[8];
	result[3] = m0[3] * m1[0] + m0[4] * m1[3] + m0[5] * m1[6];
	result[4] = m0[3] * m1[1] + m0[4] * m1[4] + m0[5] * m1[7];
	result[5] = m0[3] * m1[2] + m0[4] * m1[5] + m0[5] * m1[8];
	result[6] = m0[6] * m1[0] + m0[7] * m1[3] + m0[8] * m1[6];
	result[7] = m0[6] * m1[1] + m0[7] * m1[4] + m0[8] * m1[7];
	result[8] = m0[6] * m1[2] + m0[7] * m1[5] + m0[8] * m1[8];
}
// ���������о���result
__device__ void DeviceVec3MulVec3T(float* v0, float* v1, float* result)
{
	result[0] = v0[0] * v1[0];
	result[1] = v0[0] * v1[1];
	result[2] = v0[0] * v1[2];
	result[3] = v0[1] * v1[0];
	result[4] = v0[1] * v1[1];
	result[5] = v0[1] * v1[2];
	result[6] = v0[2] * v1[0];
	result[7] = v0[2] * v1[1];
	result[8] = v0[2] * v1[2];
	//printf("v0:"); PrintVec3(v0);
	//printf("v1:"); PrintVec3(v1);
	//printf("result:\n"); PrintMat3(result);
}

__device__ void PrintVec3(float* vec)
{
	printf("%f %f %f\n", vec[0], vec[1], vec[2]);
}
__device__ void PrintMat3(float* mat)
{
	printf("%f %f %f\n%f %f %f\n%f %f %f\n", mat[0], mat[1], mat[2],
		mat[3], mat[4], mat[5],
		mat[6], mat[7], mat[8]);
}
__device__ void DeviceScaleMulMat3(float s, float* m, float* result)
{
	result[0] = m[0] * s;
	result[1] = m[1] * s;
	result[2] = m[2] * s;
	result[3] = m[3] * s;
	result[4] = m[4] * s;
	result[5] = m[5] * s;
	result[6] = m[6] * s;
	result[7] = m[7] * s;
	result[8] = m[8] * s;
}

__device__ void DeviceMat3AddMat3(float* m0, float* m1, float* result)
{
	result[0] = m0[0] + m1[0];
	result[1] = m0[1] + m1[1];
	result[2] = m0[2] + m1[2];
	result[3] = m0[3] + m1[3];
	result[4] = m0[4] + m1[4];
	result[5] = m0[5] + m1[5];
	result[6] = m0[6] + m1[6];
	result[7] = m0[7] + m1[7];
	result[8] = m0[8] + m1[8];
}

__device__ void DeviceMat3AtomicAddMat3(float* m, float* result)
{
	atomicAdd(result + 0, m[0]);
	atomicAdd(result + 1, m[1]);
	atomicAdd(result + 2, m[2]);
	atomicAdd(result + 3, m[3]);
	atomicAdd(result + 4, m[4]);
	atomicAdd(result + 5, m[5]);
	atomicAdd(result + 6, m[6]);
	atomicAdd(result + 7, m[7]);
	atomicAdd(result + 8, m[8]);
}

__device__ void DeviceCalculateTC(
	float* F_c, float* r,
	float* point_TC,
	float* totalTC
)
{
	float r_tilde[9];
	DeviceVec3toSkewSymmetricMatrix(r, r_tilde);
	float TC[3];
	DeviceMatrixDotVec(r_tilde, F_c, TC);
	point_TC[0] = TC[0];
	point_TC[1] = TC[1];
	point_TC[2] = TC[2];
	atomicAdd(totalTC + 0, TC[0]);
	atomicAdd(totalTC + 1, TC[1]);
	atomicAdd(totalTC + 2, TC[2]);
}




__global__ void calculateCollisionSphere(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness,
	int* sortedTetVertIndices, int offset, int activeElementNum) 
{
	// ����������嶥��֮������λ�ù�ϵ�����嶥��ָ����������ײ��⣬�Զ���ʩ��������ײԼ��
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadid:%d, vertexNum:%d\n", threadid, vertexNum);
	if (threadid >= activeElementNum) return;

	int tetVertIdx = sortedTetVertIndices[offset + threadid];
	int indexX = tetVertIdx * 3 + 0;
	int indexY = tetVertIdx * 3 + 1;
	int indexZ = tetVertIdx * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float d[3] = { p[0] - ballPos[0], p[1] - ballPos[1], p[2] - ballPos[2] };
	float d_square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	float distance = sqrt(d_square);
	//if(tetVertIdx==93)
	//	printf("tetVertIdx %d ball[%f %f %f], p[%f %f %f] distance %f\n", tetVertIdx, 
	//		ballPos[0], ballPos[1], ballPos[2],
	//		p[0], p[1], p[2],distance);
	if (distance < radius)
	{
		//printf("collided tetVertIdx:%d\n", tetVertIdx);
		if(tetVertIdx==64)
			printf("collided tetVertIdx:%d p[%f %f %f]\n", tetVertIdx, p[0], p[1], p[2]);
		insertionDepth[tetVertIdx] = radius - distance;
		// ����Χ�ڣ�����ָ����������������ײ���
		float dDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };
		float collisionNormal[3] = { directDirection[indexX],directDirection[indexY] ,directDirection[indexZ] };
		// (d+x*directDir)^2==r^2 ��x

		float a = dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2];
		float b = 2 * (d[0] * dDir[0] + d[1] * dDir[1] + d[2] * dDir[2]);
		float c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radius * radius;
		float x0 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		float x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float x = x1;
		//printf("dDir[%f,%f,%f],a:%f b:%f c:%f x0:%f x1:%f\n", dDir[0], dDir[1], dDir[2], a, b, c, x0, x1);

		// collisionPos = p+x*dDir
		float collisionPos[3] = { p[0] + dDir[0] * x, p[1] + dDir[1] * x, p[2] + dDir[2] * x };
		// calibrated collision normal(ʵ�����ǽ������������ĺ�����1.�������������� 2.����ָ������)
		collisionNormal[0] = collisionPos[0] - ballPos[0];
		collisionNormal[1] = collisionPos[1] - ballPos[1];
		collisionNormal[2] = collisionPos[2] - ballPos[2];
		float col_len = sqrt(collisionNormal[0] * collisionNormal[0] + collisionNormal[1] * collisionNormal[1] + collisionNormal[2] * collisionNormal[2]);

		collisionNormal[0] /= col_len;
		collisionNormal[1] /= col_len;
		collisionNormal[2] /= col_len;

		float deltaPos[3] = { x * dDir[0], x * dDir[1], x * dDir[2] };

		float forcex = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
		float forcey = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
		float forcez = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
		//triForce[indexX] += forcex;
		//triForce[indexY] += forcey;
		//triForce[indexZ] += forcez;
		collisionForce[indexX] += forcex;
		collisionForce[indexY] += forcey;
		collisionForce[indexZ] += forcez;
		float diagx = collisionStiffness * collisionNormal[0] * collisionNormal[0];
		float diagy = collisionStiffness * collisionNormal[1] * collisionNormal[1];
		float diagz = collisionStiffness * collisionNormal[2] * collisionNormal[2];
		collisionDiag[indexX] += diagx;
		collisionDiag[indexY] += diagy;
		collisionDiag[indexZ] += diagz;
		if (threadid == LOOK_THREAD)
		{
			//printf("threadid:%d triForce[%f %f %f] diag[%f %f %f]\n", threadid,
			//	triForce[indexX], triForce[indexY], triForce[indexZ],
			//	diagx, diagy, diagz);
		}

		isCollide[tetVertIdx] = 1;
		toolCollideFlag[toolIdx] = 1;

		float forceLen = sqrt(force[indexX] * force[indexX] + force[indexY] * force[indexY] + force[indexZ] * force[indexZ]);
		if ((x1 > -1e-6) && (x0 < 0))
		{
			x = x1;
		}
		else
		{
			printf("Error: x0=%f, x1=%f a:%f b:%f c:%f\n", x0, x1, a, b, c);
			printf("collision thread:%d, tetVertIdx:%d, p:[%f,%f,%f], ball:[%f,%f,%f], x:%f dDir:[%f,%f,%f], triForce:[%f,%f,%f]\n",
				threadid, tetVertIdx, p[0], p[1], p[2], ballPos[0], ballPos[1], ballPos[2], x, dDir[0], dDir[1], dDir[2],
				force[indexX], force[indexY], force[indexZ]);
		}

	}
	else
	{
		insertionDepth[tetVertIdx] = 0;
	}
	return;
}

__global__ void calculateCollisionSphereCluster(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness,
	int* tetIndex, int* tetVertRelatedTetInfo, int* tetVertRelatedTetIdx,
	int* sortedTetVertIndices, int offset, int activeElementNum)
{
	// ����������嶥��֮������λ�ù�ϵ�����嶥��ָ����������ײ��⣬�Զ���ʩ��������ײԼ��
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadid:%d, vertexNum:%d\n", threadid, vertexNum);
	if (threadid >= activeElementNum) return;

	int tetVertIdx = sortedTetVertIndices[offset + threadid];
	int indexX = tetVertIdx * 3 + 0;
	int indexY = tetVertIdx * 3 + 1;
	int indexZ = tetVertIdx * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float d[3] = { p[0] - ballPos[0], p[1] - ballPos[1], p[2] - ballPos[2] };
	float d_square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	float distance = sqrt(d_square);
	//if(tetVertIdx==93)
	//	printf("tetVertIdx %d ball[%f %f %f], p[%f %f %f] distance %f\n", tetVertIdx, 
	//		ballPos[0], ballPos[1], ballPos[2],
	//		p[0], p[1], p[2],distance);
	if (distance < radius)
	{
		//if(tetVertIdx==93)
		//	printf("collided tetVertIdx:%d p[%f %f %f]\n", tetVertIdx, p[0], p[1], p[2]);
		insertionDepth[tetVertIdx] = radius - distance;
		// ����Χ�ڣ�����ָ����������������ײ���
		float dDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };
		float collisionNormal[3] = { directDirection[indexX],directDirection[indexY] ,directDirection[indexZ] };
		// (d+x*directDir)^2==r^2 ��x

		float a = dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2];
		float b = 2 * (d[0] * dDir[0] + d[1] * dDir[1] + d[2] * dDir[2]);
		float c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radius * radius;
		float x0 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		float x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float x = x1;
		//printf("dDir[%f,%f,%f],a:%f b:%f c:%f x0:%f x1:%f\n", dDir[0], dDir[1], dDir[2], a, b, c, x0, x1);

		// collisionPos = p+x*dDir
		float collisionPos[3] = { p[0] + dDir[0] * x, p[1] + dDir[1] * x, p[2] + dDir[2] * x };
		// calibrated collision normal(ʵ�����ǽ������������ĺ�����1.�������������� 2.����ָ������)
		collisionNormal[0] = collisionPos[0] - ballPos[0];
		collisionNormal[1] = collisionPos[1] - ballPos[1];
		collisionNormal[2] = collisionPos[2] - ballPos[2];
		float col_len = sqrt(collisionNormal[0] * collisionNormal[0] + collisionNormal[1] * collisionNormal[1] + collisionNormal[2] * collisionNormal[2]);

		collisionNormal[0] /= col_len;
		collisionNormal[1] /= col_len;
		collisionNormal[2] /= col_len;

		float deltaPos[3] = { x * dDir[0], x * dDir[1], x * dDir[2] };

		//float forcex = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
		//float forcey = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
		//float forcez = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);

		//float diagx = collisionStiffness * collisionNormal[0] * collisionNormal[0];
		//float diagy = collisionStiffness * collisionNormal[1] * collisionNormal[1];
		//float diagz = collisionStiffness * collisionNormal[2] * collisionNormal[2];
		float forcex = collisionStiffness * deltaPos[0];
		float forcey = collisionStiffness * deltaPos[1];
		float forcez = collisionStiffness * deltaPos[2];
		float diagx = collisionStiffness * dDir[0];
		float diagy = collisionStiffness * dDir[1];
		float diagz = collisionStiffness * dDir[2];

		
		int tetInfoStart = tetVertRelatedTetInfo[tetVertIdx * 2 + 0];
		int relatedTetNum = tetVertRelatedTetInfo[tetVertIdx * 2 + 1];
		for (int i = 0; i < relatedTetNum; i++)
		{
			int idx = tetInfoStart + i;
			int tetIdx = tetVertRelatedTetIdx[idx];
			for (int v = 0; v < 4; v++)
			{
				int vIdx = tetIndex[tetIdx * 4 + v]; // ������ǰ������������ĳ�������嶥��ı��
				int iX = vIdx * 3 + 0;
				int iY = vIdx * 3 + 1;
				int iZ = vIdx * 3 + 2;
				atomicAdd(collisionForce + iX, forcex / (4 * relatedTetNum));
				atomicAdd(collisionForce + iY, forcey / (4 * relatedTetNum));
				atomicAdd(collisionForce + iZ, forcez / (4 * relatedTetNum));
				atomicAdd(collisionDiag + iX, diagx / (4 * relatedTetNum));
				atomicAdd(collisionDiag + iY, diagy / (4 * relatedTetNum));
				atomicAdd(collisionDiag + iZ, diagz / (4 * relatedTetNum));
			}
		}
		if (threadid == LOOK_THREAD)
		{
			//printf("threadid:%d triForce[%f %f %f] diag[%f %f %f]\n", threadid,
			//	triForce[indexX], triForce[indexY], triForce[indexZ],
			//	diagx, diagy, diagz);
		}

		isCollide[tetVertIdx] = 1;
		toolCollideFlag[toolIdx] = 1;

		float forceLen = sqrt(force[indexX] * force[indexX] + force[indexY] * force[indexY] + force[indexZ] * force[indexZ]);
		if ((x1 > -1e-6) && (x0 < 0))
		{
			x = x1;
		}
		else
		{
			printf("Error: x0=%f, x1=%f a:%f b:%f c:%f\n", x0, x1, a, b, c);
			printf("collision thread:%d, tetVertIdx:%d, p:[%f,%f,%f], ball:[%f,%f,%f], x:%f dDir:[%f,%f,%f], triForce:[%f,%f,%f]\n",
				threadid, tetVertIdx, p[0], p[1], p[2], ballPos[0], ballPos[1], ballPos[2], x, dDir[0], dDir[1], dDir[2],
				force[indexX], force[indexY], force[indexZ]);
		}

	}
	else
	{
		insertionDepth[tetVertIdx] = 0;
	}
	return;
}

__global__ void calculateVanillaCollisionSphere(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness,
	int* sortedTetVertIndices, int offset, int activeElementNum)
{
	// ��ײԼ�������Ǳ�ͶӰ���Ĺ��߱��淨�������������㹫ʽ��E=0.5w||p-q||_F^2
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadid:%d, vertexNum:%d\n", threadid, vertexNum);
	if (threadid >= activeElementNum) return;

	int tetVertIdx = sortedTetVertIndices[offset + threadid];
	int indexX = tetVertIdx * 3 + 0;
	int indexY = tetVertIdx * 3 + 1;
	int indexZ = tetVertIdx * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float d[3] = { p[0] - ballPos[0], p[1] - ballPos[1], p[2] - ballPos[2] };
	float d_square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	float distance = sqrt(d_square);
	//if(threadid==93)
	//printf("threadid %d ball[%f %f %f], p[%f %f %f] distance %f\n", threadid, 
	//	ballPos[0], ballPos[1], ballPos[2],
	//	p[0], p[1], p[2],distance);
	if (distance < radius)
	{
		printf("collided tetVertIdx:%d\n", tetVertIdx);
		insertionDepth[tetVertIdx] = radius - distance;
		// ����Χ�ڣ�����ָ����������������ײ���
		float dDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };
		float collisionNormal[3] = { directDirection[indexX],directDirection[indexY] ,directDirection[indexZ] };
		// (d+x*directDir)^2==r^2 ��x

		float a = dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2];
		float b = 2 * (d[0] * dDir[0] + d[1] * dDir[1] + d[2] * dDir[2]);
		float c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radius * radius;
		float x0 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		float x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float x = x1;
		//printf("dDir[%f,%f,%f],a:%f b:%f c:%f x0:%f x1:%f\n", dDir[0], dDir[1], dDir[2], a, b, c, x0, x1);

		// collisionPos = p+x*dDir
		float collisionPos[3] = { p[0] + dDir[0] * x, p[1] + dDir[1] * x, p[2] + dDir[2] * x };
		
		float deltaPos[3] = { x * dDir[0], x * dDir[1], x * dDir[2] };

		float forcex = collisionStiffness * deltaPos[0];
		float forcey = collisionStiffness * deltaPos[1];
		float forcez = collisionStiffness * deltaPos[2];
		force[indexX] += forcex;
		force[indexY] += forcey;
		force[indexZ] += forcez;
		collisionForce[indexX] += forcex;
		collisionForce[indexY] += forcey;
		collisionForce[indexZ] += forcez;
		float diagx = 0.5*collisionStiffness*deltaPos[0];
		float diagy = 0.5*collisionStiffness*deltaPos[1];
		float diagz = 0.5*collisionStiffness*deltaPos[2];
		collisionDiag[indexX] += diagx;
		collisionDiag[indexY] += diagy;
		collisionDiag[indexZ] += diagz;

		isCollide[tetVertIdx] = 1;
		toolCollideFlag[toolIdx] = 1;

		float forceLen = sqrt(force[indexX] * force[indexX] + force[indexY] * force[indexY] + force[indexZ] * force[indexZ]);
		if ((x1 > -1e-6) && (x0 < 0))
		{
			x = x1;
		}
		else
		{
			printf("Error: x0=%f, x1=%f a:%f b:%f c:%f\n", x0, x1, a, b, c);
			printf("collision thread:%d, tetVertIdx:%d, p:[%f,%f,%f], ball:[%f,%f,%f], x:%f dDir:[%f,%f,%f], triForce:[%f,%f,%f]\n",
				threadid, tetVertIdx, p[0], p[1], p[2], ballPos[0], ballPos[1], ballPos[2], x, dDir[0], dDir[1], dDir[2],
				force[indexX], force[indexY], force[indexZ]);
		}

	}
	else
	{
		insertionDepth[tetVertIdx] = 0;
	}
	return;
}

__global__ void calculateCollisionSphereFollowDDir(float* ballPos, float radius,
	float* positions, unsigned char* isCollide,
	float* directDirection,
	float* force, float* collisionDiag, float* insertionDepth,
	float collisionStiffness, int vertexNum)
{
	// ����������嶥��֮������λ�ù�ϵ�����嶥��ָ����������ײ��⣬�Զ���ʩ��������ײԼ��
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadid:%d, vertexNum:%d\n", threadid, vertexNum);
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;
	float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float d[3] = { p[0] - ballPos[0], p[1] - ballPos[1], p[2] - ballPos[2] };
	float d_square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	float distance = sqrt(d_square);
	//if(threadid<10)
	//printf("threadid %d ball[%f %f %f], p[%f %f %f] distance %f\n", threadid, 
	//	ballPos[0], ballPos[1], ballPos[2],
	//	p[0], p[1], p[2],distance);
	if (distance < radius)
	{
		insertionDepth[threadid] = radius - distance;
		// ����Χ�ڣ�����ָ����������������ײ���
		float dDir[3] = { directDirection[indexX], directDirection[indexY], directDirection[indexZ] };
		float collisionNormal[3] = { directDirection[indexX],directDirection[indexY] ,directDirection[indexZ] };
		// (d+x*directDir)^2==r^2 ��x

		float a = dDir[0] * dDir[0] + dDir[1] * dDir[1] + dDir[2] * dDir[2];
		float b = 2 * (d[0] * dDir[0] + d[1] * dDir[1] + d[2] * dDir[2]);
		float c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radius * radius;
		float x0 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		float x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float x = x1;
		//printf("dDir[%f,%f,%f],a:%f b:%f c:%f x0:%f x1:%f\n", dDir[0], dDir[1], dDir[2], a, b, c, x0, x1);
		if ((x1 > 0) && (x0 < 0))
		{
			x = x1;
		}
		else
		{
			printf("Error: x0=%f, x1=%f a:%f b:%f c:%f\n", x0, x1, a, b, c);
		}
		//// collisionPos = p+x*dDir
		//float collisionPos[3] = { p[0] + dDir[0] * x, p[1] + dDir[1] * x, p[2] + dDir[2] * x };
		//// calibrated collision normal(ʵ�����ǽ������������ĺ�����1.�������������� 2.����ָ������)
		//collisionNormal[0] = collisionPos[0] - ballPos[0];
		//collisionNormal[1] = collisionPos[1] - ballPos[1];
		//collisionNormal[2] = collisionPos[2] - ballPos[2];
		//float col_len = sqrt(collisionNormal[0] * collisionNormal[0] + collisionNormal[1] * collisionNormal[1] + collisionNormal[2] * collisionNormal[2]);

		//collisionNormal[0] /= col_len;
		//collisionNormal[1] /= col_len;
		//collisionNormal[2] /= col_len;

		//float deltaPos[3] = { x * dDir[0], x * dDir[1], x * dDir[2] };

		//triForce[indexX] += collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
		//triForce[indexY] += collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
		//triForce[indexZ] += collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
		//triCollisionDiag[indexX] += collisionStiffness * collisionNormal[0] * collisionNormal[0];
		//triCollisionDiag[indexY] += collisionStiffness * collisionNormal[1] * collisionNormal[1];
		//triCollisionDiag[indexZ] += collisionStiffness * collisionNormal[2] * collisionNormal[2];

		force[indexX] += collisionStiffness * dDir[0] * x;
		force[indexY] += collisionStiffness * dDir[1] * x;
		force[indexZ] += collisionStiffness * dDir[2] * x;
		collisionDiag[indexX] += collisionStiffness * dDir[0];
		collisionDiag[indexY] += collisionStiffness * dDir[1];
		collisionDiag[indexZ] += collisionStiffness * dDir[2];
		isCollide[threadid] = 1;

		float forceLen = sqrt(force[indexX] * force[indexX] + force[indexY] * force[indexY] + force[indexZ] * force[indexZ]);
		//printf("collision thread:%d, p:[%f,%f,%f], ball:[%f,%f,%f], x:%f dDir:[%f,%f,%f], triForce:[%f,%f,%f]\n",
		//	threadid, p[0], p[1], p[2], ballPos[0], ballPos[1], ballPos[2], x, dDir[0], dDir[1], dDir[2],
		//	triForce[indexX], triForce[indexY], triForce[indexZ]);
	}
	else
	{
		insertionDepth[threadid] = 0;
	}
	return;
}
int runcalculateCollisionSphereContinue(float* ball_pos, float* ball_pos_prev, float radius, float collisionStiffness, float adsorbStiffness, float frictionStiffness, bool useClusterCollision)
{
	float d[3] = { ball_pos[0] - ball_pos_prev[0], ball_pos[1] - ball_pos_prev[1], ball_pos[2] - ball_pos_prev[2] };
	float d_len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	if (d_len < 0.5)// λ�ƺ�С��������ɢ��ײ��⡣
	{
		//printf("collision stiffness:%f\n", collisionStiffness);
		runcalculateCollisionSphere(radius, collisionStiffness, 0, useClusterCollision);
	}
	else
	{
		printf("continue collision\n");
		// ��ǰʵ�ֲ�ʹ��toolShift����ʵ��¼��ײ��
		//runcalculateToolShift(d_len, radius, 0);
		runcalculateCollisionCylinder(d_len, radius, collisionStiffness, adsorbStiffness, frictionStiffness, 0);
	}
	return 0;
}


int runcalculateCollisionSphereContinueMU(float* ball_pos, float* ball_pos_prev, float radius, float collisionStiffness, float adsorbStiffness, float frictionStiffness)
{
	float d[3] = { ball_pos[0] - ball_pos_prev[0], ball_pos[1] - ball_pos_prev[1], ball_pos[2] - ball_pos_prev[2] };
	float d_len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	if (d_len < 0.5)// λ�ƺ�С��������ɢ��ײ��⡣
	{
		runcalculateCollisionSphereMU(radius, collisionStiffness, 0);
	}
	else
	{
		runcalculateToolShiftMU(d_len, radius, 0);
		runcalculateCollisionCylinderMU(d_len, radius, collisionStiffness, adsorbStiffness, frictionStiffness, 0);
	}
	return 0;
}