#pragma once
#include "cuda.h"
#include "cuda_runtime.h"  
#include "gpuvar.h"


#pragma region  cuda_pd.cu����



/**************************************************PD����**************************************************/
//�����ʼ״̬
int runcalculateST(float damping, float dt);

int runcalculateRestPos();
int runCalculateTetEdgeSpringConstraint();
int runcalculateIF();//����������������
//�������λ��
int runcalculatePOS(float omega, float dt);

//�����ٶ�
int runcalculateV(float dt);
//�������ײ��ǣ�����ײ��ĶԽ�Ԫ��
int runClearCollision();
int runClearForce();

//����Բ������ײ
int runcalculateToolShift(float halfLength, float radius, int cylinderIdx);
int runcalculateCollisionCylinder(float halfLength, float radius, 
	float collisionStiffness, float adsorbStiffness, float frictionStiffness, 
	int idx);
int runcalculateCollisionCylinderMU(float halfLength, float radius,
	float collisionStiffness, float adsorbStiffness, float frictionStiffness,
	int idx);

int runcalculateCollisionSphere(float ball_radius, float collisionStiffness, int toolIdx, bool useClusterCollision);

int runcalculateRestPosForceWithMeshPos(float toolRadius);

int runUpdateInnerTetVertDDir();
int runUpdateSurfaceTetVertDDir();
int runNormalizeDDir();
int runUpdateTetVertDirectDirection();

void printCudaError();
void printCudaError(const char* funcName);

#pragma endregion


#pragma region  cuda_pd_MU.cu����
//ֱ����������λ�ø���Meshλ��
extern "C" int runUpdateMeshPosMU();

extern "C" int runcalculateSTMU(float damping, float dt);

//���㶥��ĳ��ٶ�
extern "C" int runcalculateSTMU_part(float damping, float dt);

//�����ײ��Ϣ
extern "C" int runClearCollisionMU();

//���㶥�������
extern "C" int runcalculateIFMU();

//ʹ����ͶӰ��������ɢ��ײ���

//��ײ���
extern "C" int runcalculateCollisionCylinderMU(
	float length, float radius,
	float collisionStiffness, float adsorbStiffness,
	int flag);


int runcalculateCollisionSphereMU(float ball_radius, float collisionStiffness, int toolIdx);
int runcalculateCollisionSphereContinueMU(
	float* ball_pos, float* ball_pos_prev, float radius, 
	float collisionStiffness, float adsorbStiffness, float frictionStiffness);

//����Rest-pos��
extern "C" int runcalculateIFRestMU(float halfLength, float radius);
extern "C" int runcalculateRestPosForceWithTetPos(float toolRadius);

//�����ٶ�
extern "C" int runcalculateVMU(float dt);

//����ָ������
extern "C" int setDDirwithNormal();


//����mesh���㷨��
extern "C" int runUpdateMeshNormalMU();

#pragma endregion



#pragma region  cuda_pd.cu����

__global__ void applyToolForce(float* tetVertCollisionForce_d, float* hapticForceTotal_D, float* hapticForceLast_D,
	float collisionStiffness, float* tetVertCollisionDiag_d,
	int* continuousFrameCounter_D, int maxContinuousFrameCount,
	float* lastNonZeroForce_D,
	int hapticCounter,
	int vertexNum
);

/**************************************************PD����**************************************************/

//���㲼��Լ���ĸն�ϵ��
//__global__ void calculateMeshStiffness(float* cylinderPosLeft, float* cylinderDirLeft,
//	float* cylinderPosRight, float* cylinderDirRight, float length, float radius, float* positions,
//	unsigned int* isCollide, unsigned int* CollideFlagLeft, unsigned int* CollideFlagRight,
//	float* meshStiffness, int vertexNum);

//���㲼��ģ�͵�Լ����
__global__ void calculateMeshForce(float* positions, int* skeletonMesh, float* force, float* collisionDiag,
	float* meshPositions, float* meshNormals, float* meshStiffness, int vertexNum);
__global__ void calculateRestPosStiffness(float* ballPos, unsigned char* toolCollideFlag, float* positions, unsigned char* isCollide, float* meshStiffness, int toolNum, int vertexNum);
__global__ void calculateRestPosStiffnessWithMesh(
	float* ballPos, unsigned char* toolCollideFlag,
	float* positions,
	unsigned char* isCollide, float* meshStiffness,
	int toolNum, int vertexNum);
__global__ void calculateRestPosStiffnessWithMesh_part(
	float* ballPos, float  ballRadius,
	unsigned char* toolCollideFlag, float* positions,
	unsigned char* isCollide, float* meshStiffness,
	int toolNum, int* sortedTetVertIndices, int startIdx, int activeElementNum);
__global__ void calculateRestPosForceWithMeshPos(
	float* positions, int* skeletonMesh,
	float* force, float* collisionDiag,
	float* meshPositions, unsigned char* isCollide,
	float* meshStiffness, int vertexNum);


int runCheckCollisionForceAngle(float collisionStiffness);

__global__ void CheckCollisionForceAngle(float* collisionForceLast, float* collisionForce,
	float* collisionDiag, float collisionStiffness, int vertexNum);

//����st
//__global__ void calculateST(float* positions, float* velocity, float* externForce,
//	float* old_positions, float* prev_positions, float* last_Positions, float* fixed,
//	int vertexNum, float damping, float dt,
//	int roughGridStart, int roughtGridEnd);
__global__ void calculateST(float* positions, float* velocity, float* externForce,
	float* old_positions, float* prev_positions, float* last_Positions, float* fixed,
	int vertexNum, float gravityX, float gravityY, float gravityZ, float damping, float dt);

//����F,R
__global__ void calculateIF(float* positions, int* tetIndex, float* tetInvD3x3, float* tetInvD3x4,
	float* force, float* tetVolumn, bool* active, int tetNum, float* volumnStiffness);
__global__ void calculateIF_part(float* positions, int* tetIndex,
	float* tetInvD3x3, float* tetInvD3x4,
	float* force, float* tetVolumn, float* volumnStiffness, 
	int * sortedTetIdx, int offset, int activeElementNum);

__global__ void calculateTetEdgeSpringConstraint(
	float* positions, 
	float* force, 
	float* springStiffness, float* springOrigin, int * springIndex, 
	int springNum);

//������ײ��������ײ��
__global__ void calculateCPlane(float* planeNormal, float* planePos, float* positions,
	float* force, float* collisionDiag, int vertexNum, float collisionStiffness);

//�����ײ��Ϣ
__global__ void clearCollision(unsigned int* isCollide, float* collisionDiag,
	float* adsorbForceLeft, float* adsorbForceRight, float* force, float* collisionForce, int vertexNum);


//����ץȡ��
__global__ void calculateAdsorbForce(float* cylinderPos, float* cylinderDirX, float* cylinderDirY, float* cylinderDirZ,
	float* positions, unsigned int* isCollide, float* force, float* collisionDiag, float* relativePosition, int vertexNum, float adsorbStiffness);

__global__ void calculateAdsorbForceForHaptic(float* spherePos, int* sphereConnectStart, int* sphereConnectCount, unsigned int* sphereConnects,
	float* sphereConnectLength, int* sphereGrabFlag, float* adsorbForce, int sphereNum);
//������ײ��(��Բ����)
__global__ void calculateCollisionCylinder(float* cylinderPos, float* cylinderDir, float* cylinderV, float halfLength, float radius, float* positions,
	float* velocity, float* force, unsigned int* isCollide, float* collisionDiag, float* volumnDiag, int vertexNum, float collisionStiffness, float frictionStiffness);
//������ײ��(��Բ����)
__global__ void calculateCollisionCylinder(float* cylinderPos, float* cylinderDir, float* cylinderV, float halfLength, float radius, float forceDirX, float forceDirY, float forceDirZ,
	float* positions, float* velocity, float* force, unsigned int* isCollide, float* collisionDiag, float* volumnDiag, int vertexNum, float collisionStiffness, float frictionStiffness);
__global__ void calculateCollisionCylinderGraphical(float* cylinderPos, float* cylinderDir, float* cylinderV, float halfLength, float radius, float* positions, unsigned int* isCollide, int vertexNum);
__global__ void calculateToolShift(
	float* cylinderPos, float* cylinderDir,
	float* directDir,
	float halfLength, float radius,
	float* positions,
	float* cylinderShift,
	int vertexNum);
__global__ void calculateCollisionCylinderSDF(float* cylinderLastPos, float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* positions, float* force, unsigned char* isCollide, unsigned char* collideFlag,
	float* collisionDiag,
	int vertexNum,
	float collisionStiffness, float* collisionForce,
	float* directDirection, float* cylinderShift);
__global__ void calculateCollisionCylinderSDF(float* cylinderLastPos, float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* positions, float* force, unsigned char* isCollide, unsigned char* collideFlag,
	float* collisionDiag,
	int* sortedIndices, int offset, int activeElementNum,
	float collisionStiffness, float* collisionForce,
	float* directDirection, float* cylinderShift);

//ʹ��������ײ���ı�������ײ����㷨
__global__ void calculateCollisionCylinderAdvance_without_directDir(
	float* cylinderLastPos, float* cylinderPos,
	float* cylinderDir, float* cylinderV,
	float halfLength, float radius,
	float* positions, float* velocity, float* force,
	unsigned int* isCollide,
	float* collisionDiag,
	float* volumnDiag,
	int vertexNum, float collisionStiffness, float frictionStiffness,
	float* collisionForce, float* cylinderShift);

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
	float* collisionForce, float* directDir, float* cylinderShift);

__global__ void calculateCollisionSphere(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness, int vertexNum);

int runClearFc();
int runHapticCollisionSphereForTri(float toolR, float collisionStiffness, float k_c, int toolIdx);
int runHapticCollisionSphereForTet(float toolR, float p_collisionStiffness, float kc, int toolIdx);
__global__ void hapticCollisionSphere(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness,
	float* toolDeltaPos, float* F_c, float* partialFc, float k_c, int* collisionNumPtr, int vertexNum);
__global__ void hapticCollisionSphere_Merge(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection,
	float* triForce, float* triCollisionForce, float* triCollisionDiag, float* triInsertionDepth,
	float* tetVertForce, float* tetVertCollisionForce, float* tetVertCollisionDiag, float* tetVertInsertionDepth,
	int* mapping,
	float collisionStiffness,
	float* toolDeltaPos, float* F_c, float* partialFc, float k_c, int* collisionNumPtr, int vertexNum);
int runHapticCollisionSphere_Merged(float toolR, float p_collisionStiffness, float kc, int toolIdx);

int runHapticCollisionCylinder_Merged_With_Sphere(float toolR, float param_toolLength, float p_collisionStiffness, float kc, int toolIdx, float sphere_R);
int runHapticCollisionCylinder_Merged(float toolR, float toolLength, float p_collisionStiffness, float kc, int toolIdx);
__global__ void hapticCollisionCylinder_Merge(
	float* cylinderLastPos, float * cylinderPose,
	float halfLength, float radius, float sphere_r, float* triPositions,
	float* velocity, int* mapping, float* triForce,
	float* triCollisionForce, float* triCollisionDiag, float* triInsertionDepth, float* triVertProjectedPos, float* tetVertForce,
	float* tetVertCollisionForce, float* tetVertCollisionDiag, float* tetInsertionDepth, unsigned char* isCollide,
	int vertexNum,
	float collisionStiffness, float frictionStiffness, float* directDir,
	float* cylinderShift, int* collisionNumPtr);
int runDeviceCalculateContact(float k_c);
__global__ void CalculateContact(float* nonPenetrationDirection, float* triVertPosition, float* projectedPosition, float* insertionDepth,
	float* toolPose, float* toolDeltaPos,
	unsigned char* isCollide, float* total_FC, float* totalPartial_FC_X,
	float* totalPartial_FC_Omega, float* total_TC, float* totalPartial_TC_X,
	float* totalPartial_TC_Omega, float k_c);

__global__ void calculateCollisionSphere(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness, 
	int* sortedTetVertIndices, int offset, int activeElementNum);
__global__ void calculateCollisionSphereCluster(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness,
	int* tetIndex, int* tetVertRelatedTetInfo, int* tetVertRelatedTetIdx,
	int* sortedTetVertIndices, int offset, int activeElementNum);

__global__ void calculateVanillaCollisionSphere(float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection, float* force, float* collisionForce,
	float* collisionDiag, float* insertionDepth, float collisionStiffness,
	int* sortedTetVertIndices, int offset, int activeElementNum);

__global__ void calculateCollisionSphereFollowDDir(float* ballPos, float radius,
	float* positions, unsigned char* isCollide,
	float* directDirection,
	float* force, float* collisionDiag, float* insertionDepth,
	float collisionStiffness, int vertexNum);

int runcalculateCollisionSphereContinue(float* ball_pos, float* ball_pos_prev, float radius, float collisionStiffness, float adsorbStiffness, float frictionStiffness, bool useClusterCollision);

__global__ void calculateCollisionSphere_without_calibration(float* ballPos, float radius,
	float* positions, unsigned int* isCollide,
	float* directDirection, int* directIndex,
	float* force, float* collisionDiag, float collisionStiffness,
	int vertexNum);
int runCalculateFc(float collisionStiffness, float kc);
__global__ void calculateTotalFc(
	float* collisionForce, unsigned int* isCollide,
	float collisionStiffness, float kc,
	float* totalFc, float* totalPartialFc, int* collisionNum,
	int vertexNum);
__global__ void calculateTotalFc_with_dDir(
	float* directDir, float* insertionDepth, float kc,
	unsigned int* isCollide,
	float* totalFc, float* totalPartialFc, int* collisionNum,
	int vertexNum
);
__global__ void hapticCollisionSphere_Merge_with_Torque(
	float* ballPos, float radius,
	float* positions, unsigned char* isCollide, int toolIdx,
	unsigned char* toolCollideFlag, float* directDirection,
	float* triForce, float* triCollisionForce, float* triCollisionDiag, float* triInsertionDepth,
	float* tetVertForce, float* tetVertCollisionForce, float* tetVertCollisionDiag, float* tetVertInsertionDepth,
	int* mapping,
	float collisionStiffness,
	float* toolDeltaPos,
	float* total_FC, float* totalPartial_FC_X, float* totalPartial_FC_Omega,
	float* total_TC, float* totalPartial_TC_X, float* totalPartial_TC_Omega,
	float k_c, int* collisionNumPtr, int vertexNum);

__device__ void DeviceVec3toSkewSymmetricMatrix(
	float* v, float* m
);
__device__ void DeviceMatrixDotVec(
	float* m, float* v, float* result
);
__device__ void DeviceMat3MulMat3(
	float* m0, float* m1, float* result
);
__device__ void DeviceVec3MulVec3T(
	float* v0, float* v1, float* result
);
__device__ void DeviceScaleMulMat3(float s, float* m, float* result);
__device__ void DeviceMat3AddMat3(float* m0, float* m1, float* result);
__device__ void DeviceMat3AtomicAddMat3(float* m, float* result);

__device__ void DeviceCalculateContact(
	float* dDir, float k_c, float depth, float* p, float * Xg_grasp,
	float* toolDeltaPos,
	float* total_FC, float* totalPartial_FC_X, float* totalPartial_FC_Omega,
	float* total_TC, float* totalPartial_TC_X, float* totalPartial_TC_Omega, bool printInfo
);//��������Ӵ����Լ����صĸ���device������
__device__ void PrintVec3(float* vec);
__device__ void PrintMat3(float* mat);
__device__ void DeviceCalculateFC(
	float k_c, float* deltaPos,
	float* F_c,
	float* toolDeltaPos, float * total_F_c);
__device__ void DeviceCalculateFC(
	float k_c, float d, float* dDir,
	float* F_c,
	float* toolDeltaPos, float* total_F_c);
__device__ void DeviceCalculateTC(
	float* F_c, float* r,
	float* point_TC,
	float* totalTC
);
__device__ void DeviceCalculatePartial_FC_Omega(
	float* normal, float* r, float k_c, float depth,
	float* partial,
	float* totalPartial_FC_Omega
);
__device__ void DeviceCalculatePartial_FC_X(
	float* dDir, float k_c, float* partialFCX, float * totalPartialFc);

__device__ void DeviceCalculatePartial_TC_X(
	float k_c, float* r, float* directDir, float* F_c,
	float* partialTCX,
	float* totalPartialTCX
);

__device__ void DeviceCalculatePartial_TC_Omega(
	float k_c, float* F_c, float* r, float depth, float* dDir,
	float* partialTCOmega,
	float* totalPartialTCOmega
);


//������Ҫ����ȡ�����������
__global__ void calculateGrabCylinder(float* cylinderPos, float* cylinderDirZ,
	float* cylinderDirY, float* cylinderDirX, float grappleX, float grappleY, float grappleZ,
	float* positions, unsigned int* isCollide, unsigned int* isCollideHalf, int vertexNum,
	float* relativePosition, int* directIndex, int* sphereGrabFlag);

__global__ void calculateGrabOBB(float* grapperUpPos, float* grapperUpDirZ, float* grapperUpDirY,
	float* grapperUpDirX, float* grapperDownPos, float* grapperDownDirZ, float* grapperDownDirY,
	float* grapperDownDirX, float grappleX, float grappleY, float grappleZ,
	float* positions, int vertexNum, unsigned int* collideFlag);

//�����ȡ��2.0
__global__ void calculateGrabForce(float* grapperPos, float* grapperDirZ, float* grapperDirY, float* grapperDirX,
	float grappleX, float grappleY, float grappleZ, float* positions, unsigned int* isCollide,
	int vertexNum, float adsorbStiffness, float* force, float* collisionDiag, unsigned int grabFlag);

//���㶥�㵽������ľ���
__device__ float calculateCylinderDis(float posx, float posy, float posz, float dirx, float diry, float dirz,
	float vertx, float verty, float vertz, float length);

//���ץȡ��ײ���
__global__ void clearGrabCollide(unsigned int* isCollide, int vertexNum);

//������ײ��������
__global__ void calculateCollisionGrab(float* cylinderLastPos, float* cylinderPos, float* grabDir, float radius,
	float* positions, unsigned int* isCollide, float* collisionDiag, int vertexNum,
	float collisionStiffness, float* force, float* directDir);
//����ƫ������
__global__ void calculateToolShift(
	float* cylinderPos, float* cylinderDir,
	float* directDir,
	float halfLength, float radius,
	float* positions,
	float* cylinderShift,
	int vertexNum);

//���㱻�и��������
__global__ void calculateCut(
	float* cylinderPos, float* cylinderDirY, float* cylinderDirX, float* cylinderDirZ, float knifeX, float knifeZ,
	float* positions, float* velocity, float* force, float* old_positions, float* last_positions, float* prev_positions, float* fixed, float* volumnDiag,
	int* tetIndex, unsigned int* tetDrawIndex, bool* active, bool* tetCut, int tetNum, int* vertexNum,
	unsigned int* vertexTetConnectCount, unsigned int* vertexTetConnectStart, int* vertexTetConnect);

__global__ void calculateCut2(float* cylinderPos, float* cylinderDirY, float* cylinderDirX, float* cylinderDirZ, float knifeX, float knifeZ,
	float* positions, float* velocity, float* force, float* old_positions, float* last_positions, float* prev_positions, float* fixed, float* volumnDiag,
	int* tetIndex, unsigned int* tetDrawIndex, int* vertexNum, int originVertexNum, bool* vertexSplit, unsigned int* vertexTetConnectCount, unsigned int* vertexTetConnectStart, int* vertexTetConnect);

__device__ void spliteTet(
	float* positions, float* velocity, float* force, float* old_positions, float* last_positions, float* prev_positions, float* fixed, float* volumnDiag,
	unsigned int append, unsigned int origin);

__device__ void updateTetIndex(int* tetIndex, unsigned int* tetDrawIndex, unsigned int tet, unsigned int append, unsigned int origin, int* vertexTetConnect, int connetIndex);

//�жϾ���ƽ����߶ε��ཻ
__device__ bool edgeCut(float* pos0, float* pos1, float* toolPos, float* dirX, float* dirY, float* dirZ, float x, float z);

__device__ bool cylinderCollision(float* pos, float* dir, float* vert, float length, float radius, float* t, float* collisionNormal, float* collisionPos);
__device__ bool cylinderCollision_withDepth(float * pose, float* vert, float length, float radius, float sphere_r, float* t, float* depth, float* distance, float* collisionNormal, float* collisionPos);

//���Զ����obb��Χ�н�����ײ��⣨ģ��ץǯץȡ�ķ�Χ��
__device__ bool obbCollision(float posx, float posy, float posz, float dirXx, float dirXy, float dirXz, float dirYx, float dirYy, float dirYz, float dirZx, float dirZy, float dirZz,
	float vertx, float verty, float vertz, float width, float length, float height);

//ʹ�����ߺ�������ཻ���
__device__ bool sphereRayCollision(float* grabPos, float* meshPos, float radius, float* t, float* collisionNormal, float* collisionPos, float* meshNormal);

//ʹ�����ߺ�����˶��켣�����ཻ���
__device__ bool sphereRayCollisionContinue(float* currentPos, float* meshPos, float radius, float* moveDir, float moveDistance, float* collisionPos, float* collisionNormal);
//ʹ��������ײ������������ײ���ж�
__device__ bool cylinderCollisionContinue_without_directDir(
	float length, float moveDistance, float radius,
	float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir,
	float* moveDir, float* position,
	float* t, float* collisionNormal,
	float* collisionPos);


//����position
__global__ void calculatePOS(float* positions, float* force, float* fixed, float* mass,
	float* next_positions, float* prev_positions, float* old_positions,
	float* volumnDiag, float* collisionDiag, float* collisionForce,
	int vertexNum, float dt, float omega);
__global__ void calculatePOS(float* positions, float* force, float* fixed, float* mass,
	float* next_positions, float* prev_positions, float* old_positions,
	float* volumnDiag, float* collisionDiag, float* collisionForce,
	int* sortedIndices, int offset, int activeElementNum, float dt, float omega);

//�����ٶȸ���
__global__ void calculateV(float* positions, float* velocity, float* last_positions, int vertexNum, float dt);
__global__ void calculateV(float* positions, float* velocity, float* last_positions, int* sortedIndices, int offset, int activeElementNum, float dt);

__global__ void calculateMultiGridConstriant(float* positions, float* force, float* collisionDiag,
	float* multiGridConnectInfo, int vertexNum, float stiffness);

//��������������
__global__ void calculateRoughGridExternForce(float* positions, float* force, float* diag,
	float* multiGridConnectInfo, int vertexNum);

//���������������Լ��
__global__ void calculateConnectForce(float* positions, int* connectIndex, float* connectWeight, float* force, int vertexNum, float connectStiffness);


//������㷨����Ϣ����Ҫ���¼���
__global__ void clearNormalTet(float* vertexNormal, int vertexNum);
//������������淨��
__global__ void updateTetNormal_old(float* tetPositions, unsigned int* tetSurfaceIndex,
	float* tetSurfaceNormal, int tetSurfaceNum);

__global__ void updateTetNormal(float* tetPositions, unsigned int* tetSurfaceIndex, float* tetNormals,
	float* tetSurfaceNormal, int tetSurfaceNum);


//���ݶ���λ�ú�������Ƭ������Ƭ����
__global__ void updateMeshNormal(float* meshPosition, float* meshNormal,
	unsigned int* meshTriangle, int meshTriangleNum);
//���߹�һ��
__global__ void normalizeMeshNormal(float* meshNormal, int meshVertexNum);


__global__ void  updateDrawMesh(float* meshPosition, float* meshNormal, float* drawMeshPosition,
	float* drawMeshNormal, unsigned int* map, int vertexNum);

__global__ void updatetetMeshPosition(float* meshPosition, float* meshNormal, unsigned int* skeletonIndex,
	float* skeletonCoord, float* tetPositions, int meshVertexNum);
int runUpdateSpherePosition();


__global__ void updateSpherePosition(float* spherePos, unsigned int* skeletonIndex,
	float* skeletonCoord, float* tetPositions, int sphereNum);

__global__ void updateSphereDirectDirection(float* spherePos,
	float* tetPositions, float* directDir, int* directIndex, int sphereNum);

__global__ void check(float* directDir, int vertexNum);

__global__ void updateInnerTetVertDirectDirection(
	float* tetVertPositions,
	int* bindingTetVertIndices, float* bindingWeight,
	float* directDir,
	int vertexNum);
__global__ void updateSurfaceTetVertDirectDirection(
	int* onSurfaceTetVertIndices,
	int* TetVertNearestTriVertIndices, float* triVertNorm,
	float* tetVertDDir, float* tetVertPos, float* triVertPos, int surfaceTetVertNum);
__global__ void normalizeDDir(float* dDir, int pointNum);
__global__ void setNonPenetrationDirWithTriVertNormal(float* nonPenetrationDir, float* normal, int vertexNum);


__global__ void updateShellDirectDirection(
	float* shellPos,
	float* tetPositions,
	float* directDir,
	int* directIndex,
	int vertexNum);

__global__ void updateDirectDirection_using_OuterShell(float* outerShellPos, float* tetPositions,
	float* directDir, int* directIndex, int vertexNum);

__global__ void updateSphereNormal(float* tetSurfaceNormals, float* directDir, int* directIndex, int vertexNum);
int runClearHashTable();
__global__ void clearHash(HashEntry_D* hashTable, int hashNum);
int runUpdateBoundBox();
__global__ void buildAABBBox(float* aabb, float* positions, int* tetIndex, int aabbBoxNum);
int runUpdateSphereHash(float lowX, float lowY, float lowZ, float  averageEdge);
__global__ void updateVertexHash(float* positions, float* externForce, HashEntry_D* hashTable,
	float lowX, float lowY, float lowZ, float averageEdge, int hashNum, unsigned long currentTimeStep,
	int sphereNum);
int runSelfCollisionDetection(float lowX, float lowY, float lowZ, float averageEdge, float externForceStiffness);
__global__ void selfCollisionDetection(float* positions, float* externForce, HashEntry_D* hashTable,
	unsigned int* isCollide, float lowX, float lowY, float lowZ, float averageEdge, int hashNum,
	int sphereNum, float externForceStiffness, unsigned long currentTimeStep);
int runSetExternForce();

__global__ void setExternForce(float* tetExternForce, float* sphereExternForce, unsigned int* sphereTetIndex,
	float* sphereTetCoord, int sphereNum);


int runUpdateSurfaceBoundBox();
__global__ void buildSurfaceAABBBox(float* aabb, float* positions, unsigned int* tetSurfaceIndex, int aabbBoxNum);

int runProjectPointOnSurface(float collisionStiffness);
__global__ void projectPointOnSurface(float* aabb, float* positions, unsigned int* isCollide,
	unsigned int* tetSurfaceIndex, float* tetSufaceNormal, float* force, float* collisionDiag,
	float* externForce, int vertexNum, int tetSurfaceNum, float collisionStiffness);
int runSetHapticExternForce(float* hapticExternGPU);

__global__ void setHapticExternForce(float* hapticExtern, float* externForce, unsigned int* collide, int vertexNum);

__device__ float VectorNormal_D(float* vec0);

__device__ float VectorDot_D(float* vec0, float* vec1);

__device__ float MatrixDet_3_D(float* A);

__device__ float Matrix_Inverse_3_D(float* A, float* R);
__device__ void MatrixProduct_3_D(const float* A, const float* B, float* R);

__device__ void MatrixProduct_D(float* A, float* B, float* R, int nx, int ny, int nz);

__device__ void MatrixSubstract_3_D(float* A, float* B, float* R);

__device__ void GetRotation_D(float F[3][3], float R[3][3]);

__device__ float invSqrt(float x);

__device__ unsigned int  XORHash(int x, int y, int z, int hashSize);

__device__ float minGPU(float pos0, float pos1, float pos2, float pos3);
__device__ float maxGPU(float pos0, float pos1, float pos2, float pos3);

__device__ float tetDot_D(float* a, float* b);

__device__ void tetCross_D(float* a, float* b, float* c);

__device__ float tetNormal_D(float* vec0);

__device__ float vecLen_D(float* vec0);

__global__ void calculateVec3Len(float* vec, float* len, int vecNum);

__device__ float tetSolveFormula_D(float* xAxis, float* yAxis, float* zAxis, float* target, float* x, float* y, float* z);

__device__ float tetPointLineDistance_D(float* lineStart, float* lineDir, float* point);

__device__ float tetPointPointDistance_D(float* start, float* end);

__device__ void tetSolveInsect_D(float* lineDir, float* toolDir, float* VSubO, float radius, float* solve0, float* solve1);

__device__ void tetSolveInsectSphere_D(float* lineDir, float* VSubO, float radius, float* solve0, float* solve1);

#pragma endregion


#pragma region  cuda_pd_MU.cu����

__global__ void UpdateMeshPosMU(float* positions, int* skeletonIndex, float* Tet_positions, int vertexNum);
//���㶥��ĳ��ٶ�


__global__ void calculateSTMU(float* positions, float* old_positions, float* prev_positions, float* velocity, float* externForce, float* fixed, int vertexNum, float gravityX, float gravityY, float gravityZ, float damping, float dt);



__global__ void clearCollisionMU(unsigned int* isCollide, unsigned int* collideFlagLeft, unsigned int* collideFlagRight, float* collisionDiag, float* force, float* collisionForce, float* insertionDepth, int vertexNum);

//���㶥�������
extern "C" int runcalculateIFMU();

__global__ void calculateIFMU(float* positions, float* force, 
	float* springStiffness, float* springOrigin, unsigned int* springIndex, 
	float* triVertFixed,
	int springNum);
__global__ void calculateIFMU(float* positions, float* force,
	float* springStiffness, float* springOrigin, unsigned int* springIndex,
	int* sortedSpringIndices, int offset, int activeElementNum);
//���㹤�ߵ�ƫ��
extern "C" int runcalculateToolShiftMU(float halfLength, float radius, int flag);

__global__ void calculateToolShiftMU(float* cylinderPos, float* cylinderDir, float* directDir, float halfLength, float radius, float* positions, float* cylinderShift, int vertexNum);

//ʹ����ͶӰ��������ɢ��ײ���
__device__ bool cylinderRayCollisionMU(float* cylinderPos, float* cylinderDir, float vertx, float verty, float vertz, float* moveDir, float length, float radius, float* t, float* sln, float* collisionNormal, float* collisionPos);

__global__ void calculateCollisionCylinderGraphicalMU(float* cylinderPos, float* cylinderDir, float* cylinderV, float halfLength, float radius, float* positions, unsigned int* isCollide, int vertexNum);

//������Ҫ����ȡ�����������
__global__ void calculateGrabCylinderMU(float* cylinderPos, float* cylinderDirZ, float* cylinderDirY, float* cylinderDirX, float grappleX, float grappleY, float grappleZ, float* positions,
	unsigned int* isCollide, unsigned int* isCollideHalf, int vertexNum, float* relativePosition);

//���Զ����obb��Χ�н�����ײ���(ģ��ץǯץȡ�ķ�Χ)
__device__ bool obbCollisionMU(float posx, float posy, float posz, float dirXx, float dirXy, float dirXz, float dirYx, float dirYy,
	float dirYz, float dirZx, float dirZy, float dirZz, float vertx, float verty, float vertz, float width, float length, float height);

//���㶥��������--��ȡ��
__global__ void calculateGrabForceMU(float* grapperPos, float* grapperDirZ, float* grapperDirY, float* grapperDirX, float grappleX, float grappleY, float grappleZ,
	float* positions, unsigned int* isCollide, int vertexNum, float adsorbStiffness, float* force, float* collisionDiag, unsigned int grabFlag);

//����ץȡ��--��ȡ���
__global__ void calculateAdsorbForceMU(float* cylinderPos, float* cylinderDirX, float* cylinderDirY, float* cylinderDirZ, float* positions, unsigned int* isCollide,
	float* force, float* collisionDiag, float* relativePosition, int vertexNum, float adsorbStiffness);
//�ϲ���ȡ����ײ���
__global__ void mergeCollideMU(unsigned char* isCollide, unsigned int* CollideFlag, unsigned int* isGrap, int vertexNum);
//ʹ��������ײ���ı�������ײ����㷨
__global__ void calculateCollisionCylinderAdvanceMU(
	float* cylinderLastPos, float* cylinderPos, float* cylinderDir,
	float halfLength, float radius,
	float* positions, float* force,
	unsigned char* isCollide, unsigned int* collideFlag, float* collisionDiag,
	int vertexNum,
	float collisionStiffness, float* collisionForce,
	float* directDir, float* cylinderShift);

//ʹ��������ײ������������ײ���ж�
__device__ bool cylinderCollisionContinueMU(float length, float moveDistance, float radius, float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir, float* moveDir, float* position, float* t, float* collisionNormal, float* collisionPos, float* directDir);


__global__ void calculateCollisionSphereMU(float* ballPos, float radius,
	float* positions, unsigned int* isCollide,
	float* directDirection,
	float* force, float* collisionDiag, float* insertionDepth,
	float collisionStiffness, int vertexNum);


//����Rest-pos��

__global__ void calculateIFRestStiffnessMU(float* cylinderPosLeft, float* cylinderDirLeft, float* cylinderPosRight, float* cylinderDirRight,
	float length, float radius, float* positions, unsigned int* isCollide, unsigned int* CollideFlagLeft, unsigned int* CollideFlagRight, float* restStiffness, int vertexNum);
//���㶥�㵽������ľ���
__device__ float calculateCylinderDisMU(float posx, float posy, float posz, float dirx, float diry, float dirz, float vertx, float verty, float vertz, float length);

__global__ void calculateIFRestMU(float* positions, int* skeletonIndex, float* force, float* collisionDiag, float* rest_positions, float* restStiffness, int vertexNum);

__global__ void calculateRestStiffnessWithTet(float* ballPos, unsigned char* toolCollideFlag, float* positions, unsigned char* isCollide, float* restStiffness, int toolNum, int vertexNum);
__global__ void calculateRestStiffnessWithTet(float* ballPos, float toolRadius, 
	unsigned char* toolCollideFlag, float* positions, 
	unsigned char* isCollide, float* restStiffness, 
	int toolNum, int* sortedIndices, int offset, int activeElementNum);
__global__ void calculateRestPosWithTetPosMU(float* positions, int* skeletonIndex, float* force, float* collisionDiag, float* rest_positions, float* restStiffness, int vertexNum);
__global__ void calculateRestPosWithTetPosMU(float* positions, int* skeletonIndex, float* force, float* collisionDiag,
	float* rest_positions, float* restStiffness,
	int* sortedIndices, int offset, int activeElementNum);

//�б�ѩ�����λ��
extern "C" int runcalculatePosMU(float omega, float dt);

__global__ void calculatePOSMU(float* positions, float* force, float* fixed, float* mass, float* next_positions, float* prev_positions,
	float* old_positions, float* springDiag, float* collisionDiag, float* collisionForce, int vertexNum, float dt, float omega);
__global__ void calculatePOSMU(float* positions, float* force, float* fixed, float* mass, float* next_positions, float* prev_positions,
	float* old_positions, float* springDiag, float* collisionDiag, float* collisionForce, 
	int* sortedIndices, int offset, int activeElementNum, 
	float dt, float omega);


//�����ٶ�

__global__ void calculateVMU(float* positions, float* velocity, float* old_positions, int vertexNum, float dt);
__global__ void calculateVMU(float* positions, float* velocity, float* old_positions,
	int* sortedIndices, int offset, int activeElementNum,
	float dt);

//����ָ������

__global__ void updateDirectDirectionMU(float* spherePos, float* tetPositions, float* directDir, int* directIndex, int vertexNum);

__global__ void updateSphereNormalMU(float* tetSurfaceNormals, float* directDir, int* directIndex, int vertexNum);

__global__ void updateShellDirectDirectionMU(
	float* shellPos, //���ڼ���ָ�����������
	float* meshPositions, //�������񶥵�λ��
	float* directDir, // �������ǰ�����ָ����������һ�����ģ�
	int* directIndex, // �洢 �������񶥵�-��Ƕ��� ��Ӧ
	int vertexNum);

//����mesh���㷨��

//������㷨����Ϣ����Ҫ���¼���
__global__ void clearNormalMU(float* meshNormal, float* totAngle, int vertexNum);

//���ݶ���λ�ú�������Ƭ������Ƭ����
__global__ void updateMeshNormalMU(float* meshPosition, float* meshNormal, float* totAngle, unsigned int* meshTriangle, int meshTriangleNum);
__global__ void updateMeshNormalMU(float* meshPosition, float* meshNormal,
	float* totAngle, unsigned int* meshTriangle,
	int* sortedTriIndices, int offset, int activeElementNum);
//���߹�һ��
__global__ void normalizeMeshNormalMU(float* meshNormal, float* totAngle, int meshVertexNum);

__global__ void normalizeMeshtriVertNorm_debug(float* meshNormal, float* meshPosition, float* totAngle, int meshVertexNum);
__global__ void normalizeMeshtriVertNorm_debug(float* meshNormal, float* meshPosition, float* totAngle,
	int* sortedTriVertIndices, int offset, int activeElementNum);
//���»�������

__global__ void  updateDrawMeshMU(float* meshPosition, float* meshNormal, float* drawMeshPosition, float* drawMeshNormal, unsigned int* map, int vertexNum);

/**************************************************��������*************************************************/

__device__ float tettriVertNorm_d(float* vec0);

__device__ void tetCrossMU_D(float* a, float* b, float* c);

__device__ float tetDotMU_D(float* a, float* b);

__device__ float tetSolveFormulaMU_D(float* xAxis, float* yAxis, float* zAxis, float* target, float* x, float* y, float* z);

__device__ float tetPointLineDistanceMU_D(float* lineStart, float* lineDir, float* point);

__device__ float tetPointPointDistanceMU_D(float* start, float* end);

//���ߺ�Բ����
__device__ void tetSolveInsectMU_D(float* lineDir, float* toolDir, float* VSubO, float radius, float* solve0, float* solve1);

//���ߺ�����
__device__ void tetSolveInsectSphereMU_D(float* lineDir, float* VSubO, float radius, float* solve0, float* solve1);
#pragma endregion

__device__ bool cylinderRayCollisionDetection(
	float* cylinderPos, float* cylinderDir,
	float vertx, float verty, float vertz,
	float* moveDir, // ָ����������ײ���*���嶥��*���˶�����
	float length, float radius,
	float* t, float* sln,
	float* collisionNormal, float* collisionPos);
//ʹ��������ײ������������ײ���ж�
__device__ bool cylinderCollisionContinue(
	float length, float moveDistance, float radius,
	float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir,
	float* moveDir, float* position,
	float* t, float* collisionNormal,
	float* collisionPos,
	float* directDir);


//����SDF��������ײ���
__device__ bool cylinderCollisionContinueSDF(float length, float moveDistance, float radius,
	float* cylinderPos, float* cylinderLastPos, float* cylinderDir, float* moveDir, float* position,
	float* directDir, float* collisionNormal, float* collisionPos);


//��Բ������ɢ��ײ��⣬��ָ������
__device__ bool cylinderCollisionSDF(float* pos, float* dir, float* vert, float* directDir, float length,
	float radius, float* t, float* collisionNormal, float* collisionPos);

__global__ void calculateRestPos(float* positions, float* rest_positions, float* force, float* collisionDiag, float* restStiffness, int vertexNum);
__global__ void calculateRestPos_part(float* positions, float* rest_positions, float* force, float* collisionDiag, float* restStiffness, int* sortedTetVertIndices, int offset, int activeElement);