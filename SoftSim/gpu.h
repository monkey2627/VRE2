#pragma once
#include <chrono> // ���ڸ߾��ȼ��㺯������ʱ��
#include "cuda.h"
#include "cuda_runtime.h"  
//#include "cuda_gl_interop.h"
//#include "device_functions.h"

#define HASH_BUCKET_NUM 10
struct HashEntry_D {
	unsigned long timeStamp;		//ʱ��������Բ���ÿ�ζ����³�ʼ��hash��
	int count;
	int buckets[HASH_BUCKET_NUM];
};

enum HAPTIC_BUTTON {
	normal = 0,
	grab = 1,
	cut = 2
};

#define GRAVITY				-0.0f

#pragma region  cuda_pd.cu

//���������ֹ���
extern  float* cylinderShiftLeft_D;
extern	float* cylinderLastPosLeft_D;
extern	float* cylinderPosLeft_D;//Բ������
extern  float* cylinderGraphicalPosLeft_D;
extern	float* cylinderDirZLeft_D;
extern	float* cylinderDirYLeft_D;
extern	float* cylinderDirXLeft_D;
extern	float* cylinderVLeft_D;		//�������ٶ�
extern	int	  cylinderButtonLeft_D;//Բ������Ϊ���Ƿ��Ǽ�ȡ��0Ϊ������1Ϊ��ȡ
extern	float* relativePositionLeft_D;
extern	bool	firstGrabLeft_D;
extern  unsigned int* isGrapLeft_D;				//ץǯ�պϺ����ײ��ϵ
extern	unsigned int* isGrapHalfLeft_D;			//ץǯ�պϹ����е���ײ��ϵ
extern  float* adsorbForceLeft_D;

extern	float* tetgrapperUpPosLeft_D;
extern	float* tetgrapperDownPosLeft_D;
extern	float* tetgrapperUpDirZLeft_D;
extern	float* tetgrapperUpDirXLeft_D;
extern	float* tetgrapperUpDirYLeft_D;
extern	float* tetgrapperDownDirZLeft_D;
extern	float* tetgrapperDownDirXLeft_D;
extern	float* tetgrapperDownDirYLeft_D;
extern	int* grabFlagLeft_D;
extern	unsigned int* collideFlagLeft_D;		//���ֹ��ߵ���ײ���

extern  float* cylinderShiftRight_D;
extern	float* cylinderLastPosRight_D;
extern	float* cylinderPosRight_D;//Բ������
extern	float* cylinderDirZRight_D;
extern	float* cylinderDirYRight_D;
extern	float* cylinderDirXRight_D;
extern	float* cylinderVRight_D;		//�������ٶ�
extern	int		cylinderButtonRight_D;//Բ������Ϊ���Ƿ��Ǽ�ȡ��0Ϊ������1Ϊ��ȡ


extern	float* relativePositionRight_D;
extern	bool	firstGrabRight_D;
extern  unsigned int* isGrapRight_D;		//ץǯ�պϺ����ײ��ϵ
extern	unsigned int* isGrapHalfRigth_D;	//ץǯ�պϹ����е���ײ��ϵ
extern  float* adsorbForceRight_D;
extern	float* tetgrapperUpPosRight_D;
extern	float* tetgrapperDownPosRight_D;
extern	float* tetgrapperUpDirZRight_D;
extern	float* tetgrapperUpDirXRight_D;
extern	float* tetgrapperUpDirYRight_D;
extern	float* tetgrapperDownDirZRight_D;
extern	float* tetgrapperDownDirXRight_D;
extern	float* tetgrapperDownDirYRight_D;
extern	int* grabFlagRight_D;
extern	unsigned int* collideFlagRight_D;

//����������������
extern	int		tetVertNum_d;
extern	int		tetNum_d;
//extern  int		originVetexNum_D;
extern	int* verNumPtr_D;
extern  float* tetVertPos_d;
extern	float* normals_D;
extern  float* tetVertPos_last_d;
extern  float* tetVertPos_old_d;
extern  float* tetVertPos_prev_d;
extern  float* tetVertPos_next_d;
extern  float* tetVertVelocity_d;
extern  float* tetVertExternForce_d;
extern  float* tetVertMass_d;
extern  int* tetIndex_d;
//extern  unsigned int* tetDrawIndex_D;
//extern  float* tetVolumeDiag_d;
extern  float* tetVertCollisionDiag_d;
extern  float* tetVertFixed_d;
extern  bool* tetActive_d;
extern  float* tetInvD3x3_d;//Ԥ����
extern  float* tetInvD3x4_d;//Ԥ����
extern  float* tetVolume_d;//Ԥ����
extern  float* tetVertForce_d;
extern	float* tetStiffness_d;

//extern float* multiGridConnectInfo_D;

extern float* outerShell_D;

//��ײԼ����
extern float* collisionForce_D;
extern float* collisionForceLast_D;
extern float* insertionDepth_D;

//Ϊ��ײ���㿪�ٿռ�
extern float* planeNormal_D;//ƽ�����
extern float* planePos_D;

extern float* ballPos_D;
extern float* radius_d;

extern unsigned int* isCollide_D;			//�Ƿ�������ײ����ʾ�Ƿ���������ײ��
extern unsigned int* isCollideGraphical_D;   //���⹤���Ƿ�����涥�㷢����ײ��
extern unsigned int* CollideFlag_D;			//��־λ�������Ƿ�������ײ

extern unsigned int* isSelfCollide_D;		//�Ƿ�������ײ����ʾ�Ƿ���������ײ��
extern float* sphereExternForce_D;	//����ײ�ܵ�������


//mesh������mesh���������ڲ���Ƥ��
//extern float* tetMeshPosion_D;
//extern float* tetMeshNormal_D;	//�𶥵㷨��
extern unsigned int* tetMeshTriangle_D;
extern int* tetSkeletonIndex_D;
extern float* tetSkeletonCoord_D;
extern int				tetMeshVertexNumber_D;
extern int				tetMeshTriangleNumber_D;

//ץȡ����
extern float timer;
extern float timeTop;
extern float timerLeft;
extern float timeTopLeft;
extern float timerRight;
extern float timeTopRight;

//���������mesh����
extern unsigned int* tetSurfaceIndex_D;
//��������������η���
extern float* tetSurfaceNormal_D;
extern int				tetSurfaceNum_D;


//��ģ�͵�λ��
extern int				sphereNum_D;
extern float* spherePositions_D;
extern unsigned int* sphereTetIndex_D;
extern float* sphereTetCoord_D;
extern unsigned int* sphereConnect_D;
extern float* sphereConnectLength_D;
extern int* sphereConnectCount_D;
extern int* sphereConnectStart_D;
//��ģ�͵�ָ������
extern int* sphereDirectIndex_D;
extern float* sphereDirectDirection_D;

//������֮���������Ϣ
extern int* connectIndex_D;
extern float* connectWeight_D;

//���ڴ洢ָ������
extern float* directDirection_D;
extern int* directIndex_D;

// �������˵Ĺ�����Ϣ
extern float* left_qg_from_HapticTool_D;
extern float* left_last_qg_from_HapticTool_D;
extern float* right_qg_from_HapticTool_D;
extern float* right_last_qg_from_HapticTool_D;



extern float* hapticDeformationExternForce_D;
extern float* hapticDeformationExternForceTotal_D;
extern int hapticCounter_D;
// ��¼�ö��㱻����ʩ��ѹ��������֡����
extern int* hapticContinuousFrameNumOfCollision_D;
extern float* hapticLastNonZeroToolForce_D;
extern int MAX_CONTINUOUS_FRAME_COUNT_D;


//opengl ͨ��

//�����Ļ�������
extern int				drawMeshVertexNumber_D;
extern int				drawMeshTriangleNumber_D;
extern float* drawMeshPosition_D;
extern float* drawMeshNormal_D;
extern float* drawMeshPosition_H;
extern float* drawMeshNormal_H;

extern unsigned int* drawMeshskeletonCoord_D;			//��ƽ��mesh�Ķ�Ӧ��ϵ


//������ײ��Ҫ�İ�Χ��
extern float* tetSurfaceAABB_D;


//����ײ�������										
extern int				aabbBoxNum_D;
extern float* aabbBoxs_D;
extern int				hashNum_D;
extern HashEntry_D* hashTable_D;
extern float* vertexLineAABB_D;

/**************************************************����ģ������**************************************************/
extern	int	triVertNum_d;
extern	float* triVertPos_d;
extern	float* triVertNorm_d;
extern  float* triVertRestPos_d;

//�벼��ģ�͵�ӳ���ϵ
extern int* skeletonMesh_D;
extern float* meshStiffness_D;  //Լ���ն�ϵ��
extern float* meshRestPosStiffness_D;
//��ײ��Ϣ
extern float* collisionPos_Tool;
extern float* collisionNormal_Tool;
extern unsigned int* collisionFlag_Tool;

// �ۼӶ���Թ���ʩ�ӵ����������ݶ�
extern float* totalFc_D;
extern float* totalPartialFc_D;
extern int* collisionNum_D;
#pragma endregion


#pragma region  cuda_pdMU.cu

extern	int		triVertNum_d;
extern	int		triEdgeNum_d;
extern	float* triVertPos_d;
extern  float* triVertRestPos_d;
extern	float* triVertPos_old_d;
extern	float* triVertPos_prev_d;
extern	float* triVertPos_next_d;
extern	float* triVertVelocity_d;
extern	float* triVertExternForce_d;
extern	float* triVertMass_d;
extern	float* triVertNorm_d;
extern	float* triVertNormAccu_d;

//spring
extern	unsigned int* triEdgeIndex_d;
extern	float* triEdgeOrgLength_d;
extern	float* triEdgeDiag_d;
extern	float* triVertCollisionDiag_d;
extern	float* triVertRestStiffness_d;

extern	float* triVertFixed_d;
extern	float*  triVertForce_d;
extern	float* triEdgeStiffness_d;

//ָ������
extern	float* directDirectionMU_D;
extern	int* directIndexMU_D;

//ץǯ����ײ��Ϣ
extern float	grapperRadiusMU_D;
extern float	grapperLengthMU_D;

extern	unsigned int* triVertisCollide_d;  //�Ƿ�����ײ
extern	unsigned int* CollideFlagMU_D;	//�����Ƿ�����ײ

//ץȡ
extern	bool	firstGrabLeftMU_D;
extern	bool	firstGrabRightMU_D;
extern	unsigned int* isGrabLeftMU_D;  //ץǯ�պϺ����ײ��ϵ
extern	unsigned int* isGrabRigthMU_D;
extern	unsigned int* isGrabHalfLeftMU_D;		//ץǯ�պϹ����еıպϹ�ϵ
extern	unsigned int* isGrabHalfRightMU_D;
extern	float* relativePositionLeftMU_D;
extern	float* relativePositionRightMU_D;
extern	unsigned int* CollideFlagLeftMU_D;		//��־λ�������Ƿ�������֯������ײ
extern	unsigned int* CollideFlagRightMU_D;

//��ײԼ����
extern	float* triVertCollisionForce_d;
extern float* insertionDepthMU_D;

#pragma endregion


#pragma region hapticCuda.cu 
extern float* hapticDeformationInterpolatePositions_D;
extern float* hapticDeformationPrePositions_D;
extern float* hapticDeformationPositions_D;
extern float* hapticDeformationNormals_D;
extern float* hapticDeformationExternForce_D;
extern float* hapticDeformationExternForceTotal_D;
extern int				hapticCounter_D;
extern float* hapticCollisionZone_D;		//�����ײ�����߶ε��ĸ�����
extern int				hapticDeformationNum_D;
extern int				hapticDeformationNumMem_D;
extern int* hapticContinuousFrameNumOfCollision_D;     // ��¼�ö��㱻����ʩ��ѹ��������֡����


extern unsigned int* hapticIsCollide_D;		//�����Ƿ�����ײ
extern float* hapticConstraintPoints_D;	//Լ������
extern float* hapticConstraintNormals_D;  // �����巨����
extern float* hapticConstraintZone_D;
extern int* haptic_collisionIndex_to_vertIndex_array_D; //��ײ�����±��Ӧ�Ķ����±�

extern float* hapticCylinderPos_D;
extern float* hapticCylinderPhysicalPos_D;
extern float* hapticCylinderDir_D;
extern int* hapticIndex_D;

extern unsigned int* hapticQueueIndex_D;
extern unsigned int* hapticAuxSumArray_D;

extern int				hapticAABBBoxNum_D;
extern float* hapticAABBBoxs_D;
extern float* hapticTriangleNormal_D;
//����Ҫ����
extern int* hapticSurfaceIndex_D;

extern int				hapticSphereNum_D;
extern float* hapticSphereInfo_D;
extern float* hapticSphereDirectDirection_D;	//���ָ������
extern float* hapticSphereForce_D;	//���յ�����ײ��
extern unsigned int* hapticSphereIsCollide_D;
extern float* hapticSphereCollisionZone_D;
extern int* hapticSphereindex_D;
extern unsigned int* hapticSphereQueueIndex_D;
extern unsigned int* hapticSphereAuxSumArray_D;
extern float* hapticSphereConstraintPoints_D;	//Լ������
extern float* hapticSphereConstraintZone_D;
extern float* hapticSphereConstraintDirection_D;  //Լ��ָ������

//������������Ȩ��
extern unsigned int* hapticSphereTetIndex_D;
extern float* hapticSphereTetCoord_D;

//�����ֵ�������ײԼ����Ϣ
extern int				hapticSphereConstraintNumLeft;
extern float* hapticSphereConstraintPosLeft;
extern float* hapticSphereConstraintZoneLeft;
extern float* hapticSphereConstraintDirectionLeft;
extern int				hapticSphereConstraintNumRight;
extern float* hapticSphereConstraintPosRight;
extern float* hapticSphereConstraintZoneRight;
extern float* hapticSphereConstraintDirectionRight;

// ���ֵ����ײԼ����Ϣ
extern unsigned int* hapticPointQueueIndex_D;
extern unsigned int* hapticPointAuxSumArray_D;

extern int		hapticPointConstraintNumLeft;
extern float* hapticPointConstraintPosLeft;
extern float* hapticPointConstraintNormalLeft;
extern float* hapticPointConstraintZoneLeft;

extern float* hapticVertexForceOrthogonalToTool_D;

extern float hapticCollisionStiffness_D;
extern int MAX_CONTINUOUS_FRAME_COUNT_D;
#pragma endregion


#pragma region  cuda_pd.cu����

int runApplyToolForce(float collisionStiffness);

__global__ void applyToolForce(float* collisionForce_D, float* hapticForceTotal_D, float* hapticForceLast_D,
	float collisionStiffness, float* tetVertCollisionDiag_d,
	int* continuousFrameCounter_D, int maxContinuousFrameCount,
	float* lastNonZeroForce_D,
	int hapticCounter,
	int vertexNum
);

/**************************************************PD����**************************************************/
//�����ʼ״̬
int runcalculateST(float damping, float dt, int roughGridStart, int roughGridEnd);

int runcalculateIF();
//�������λ��
int runcalculatePOS(float omega, float dt, int start, int end);

//�����ٶ�
int runcalculateV(float dt);
//�������ײ��ǣ�����ײ��ĶԽ�Ԫ��
int runClearCollision();
//����ƽ����ײ
int runcalculateCPlane(float collisionStiffness);

//ֻ���㼷ѹʱ��ָ������������ʱ������
int runcalculateToolShift(float halfLength, float radius, int flag);

//����Բ������ײ
int runcalculateCollisionCylinder(float halfLength, float radius, float collisionStiffness, float adsorbStiffness, float frictionStiffness, float forceDirX, float forceDirY, float forceDirZ, int flag);

int runcalculateCollisionBall(float ball_radius, float collisionStiffness);

//����������
int runcalculateConnectForce(float connectStiffness);

int runcalculateMeshForce(float halfLength, float radius);

int runcalculateMeshRestPos(float halfLength, float radius);

//���㲼��Լ���ĸն�ϵ��
__global__ void calculateMeshStiffness(float* cylinderPosLeft, float* cylinderDirLeft,
	float* cylinderPosRight, float* cylinderDirRight, float length, float radius, float* positions,
	unsigned int* isCollide, unsigned int* CollideFlagLeft, unsigned int* CollideFlagRight,
	float* meshStiffness, int vertexNum);

//���㲼��ģ�͵�Լ����
__global__ void calculateMeshForce(float* positions, int* skeletonMesh, float* force, float* collisionDiag,
	float* meshPositions, float* meshNormals, float* meshStiffness, int vertexNum);

__global__ void calculateRestPosForceWithMeshPos(
	float* positions, int* skeletonMesh,
	float* force, float* collisionDiag,
	float* meshPositions, unsigned int* isCollide,
	float meshStiffness, int vertexNum);


int runMultiGridConstriant(float multiGridStiffness);

int runRoughGridUpdate();

int runCheckCollisionForceAngle(float collisionStiffness);

__global__ void CheckCollisionForceAngle(float* collisionForceLast, float* collisionForce,
	float* collisionDiag, float collisionStiffness, int vertexNum);

//����st
__global__ void calculateST(float* positions, float* velocity, float* externForce,
	float* old_positions, float* prev_positions, float* last_Positions, float* fixed,
	int vertexNum, float damping, float dt,
	int roughGridStart, int roughtGridEnd);

//����F,R
__global__ void calculateIF(float* positions, int* tetIndex, float* tetInvD3x3, float* tetInvD3x4,
	float* force, float* tetVolumn, bool* active, int tetNum, float* volumnStiffness);

//������ײ��������ײ��
__global__ void calculateCPlane(float* planeNormal, float* planePos, float* positions,
	float* force, float* collisionDiag, int vertexNum, float collisionStiffness);

//�����ײ��Ϣ
__global__ void clearCollision(unsigned int* isCollide, unsigned int* isCollideGraphical, float* collisionDiag, 
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

__global__ void calculateCollisionBall(float* ballPos, float radius,
	float* positions, unsigned int* isCollide,
	float* directDirection,
	float* force, float* collisionDiag, float* insertionDepth,
	float collisionStiffness, int vertexNum);

__global__ void calculateCollisionBall_without_calibration(float* ballPos, float radius,
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

//ʹ��������ײ������������ײ���ж�
__device__ bool cylinderCollisionContinue(
	float length, float moveDistance, float radius,
	float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir,
	float* moveDir, float* position,
	float* t, float* collisionNormal,
	float* collisionPos,
	float* directDir);


//����position
__global__ void calculatePOS(float* positions, float* force, float* fixed, float* mass,
	float* next_positions, float* prev_positions, float* old_positions,
	float* volumnDiag, float* collisionDiag, float* collisionForce,
	int vertexNum, float dt, float omega);

//�����ٶȸ���
__global__ void calculateV(float* positions, float* velocity, float* last_positions, int vertexNum, float dt);

__global__ void calculateMultiGridConstriant(float* positions, float* force, float* collisionDiag,
	float* multiGridConnectInfo, int vertexNum, float stiffness);

//��������������
__global__ void calculateRoughGridExternForce(float* positions, float* force, float* diag, 
	float* multiGridConnectInfo, int vertexNum);

//���������������Լ��
__global__ void calculateConnectForce(float* positions, int* connectIndex, float* connectWeight, float* force, int vertexNum, float connectStiffness);


int runUpdateTetNormal();

//������㷨����Ϣ����Ҫ���¼���
__global__ void clearNormalTet(float* vertexNormal, int vertexNum);
//������������淨��
__global__ void updateTetNormal_old(float* tetPositions, unsigned int* tetSurfaceIndex,
	float* tetSurfaceNormal, int tetSurfaceNum);

__global__ void updateTetNormal(float* tetPositions, unsigned int* tetSurfaceIndex, float* tetNormals,
	float* tetSurfaceNormal, int tetSurfaceNum);

//����mesh�������������±��淨��
int runUpdateMeshNormal();

//���ݶ���λ�ú�������Ƭ������Ƭ����
__global__ void updateMeshNormal(float* meshPosition, float* meshNormal,
	unsigned int* meshTriangle, int meshTriangleNum);
//���߹�һ��
__global__ void normalizeMeshNormal(float* meshNormal, int meshVertexNum);

int runUpdateDrawMesh();

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


int runCheckDirectDirection();


int runUpdateDirectDirection();

int runUpdateShellDirectDirection();

__global__ void updateDirectDirection(float* spherePos, float* tetPositions, float* directDir, int* directIndex, int vertexNum);

__global__ void updateShellDirectDirection(
	float* shellPos,
	float* tetPositions,
	float* directDir,
	int* directIndex,
	int vertexNum);

int runUpdateDirectDirection_using_OuterShell();

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

__device__ float tetSolveFormula_D(float* xAxis, float* yAxis, float* zAxis, float* target, float* x, float* y, float* z);

__device__ float tetPointLineDistance_D(float* lineStart, float* lineDir, float* point);

__device__ float tetPointPointDistance_D(float* start, float* end);

__device__ void tetSolveInsect_D(float* lineDir, float* toolDir, float* VSubO, float radius, float* solve0, float* solve1);

__device__ void tetSolveInsectSphere_D(float* lineDir, float* VSubO, float radius, float* solve0, float* solve1);

void printCudaError();
#pragma endregion


#pragma region  cuda_pd_MU.cu����
//ֱ����������λ�ø���Meshλ��
extern "C" int runUpdateMeshPosMU();

__global__ void UpdateMeshPosMU(float* positions, int* skeletonIndex, float* Tet_positions, int vertexNum);
//���㶥��ĳ��ٶ�
extern "C" int runcalculateSTMU(float damping, float dt);
extern "C" int runMapResources();

//���㶥��ĳ��ٶ�
extern "C" int runcalculateSTMU_part(float damping, float dt);

__global__ void calculateSTMU(float* positions, float* old_positions, float* prev_positions, float* velocity, float* externForce, float* fixed, int vertexNum, float damping, float dt);

//�����ײ��Ϣ
extern "C" int runClearCollisionMU();

__global__ void clearCollisionMU(unsigned int* isCollide, unsigned int* collideFlagLeft, unsigned int* collideFlagRight, float* collisionDiag, float* force, float* collisionForce, float* insertionDepth, int vertexNum);

//���㶥�������
extern "C" int runcalculateIFMU();

__global__ void calculateIFMU(float* positions, float* force, float* springStiffness, float* springOrigin, unsigned int* springIndex, int springNum);

//���㹤�ߵ�ƫ��
extern "C" int runcalculateToolShiftMU(float halfLength, float radius, int flag);

__global__ void calculateToolShiftMU(float* cylinderPos, float* cylinderDir, float* directDir, float halfLength, float radius, float* positions, float* cylinderShift, int vertexNum);

//ʹ����ͶӰ��������ɢ��ײ���
__device__ bool cylinderRayCollisionMU(float* cylinderPos, float* cylinderDir, float vertx, float verty, float vertz, float* moveDir, float length, float radius, float* t, float* sln, float* collisionNormal, float* collisionPos);


//��ײ���
extern "C" int runcalculateCollisionCylinderMU(
	float length, float radius,
	float collisionStiffness, float adsorbStiffness,
	int flag);
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
__global__ void mergeCollideMU(unsigned int* isCollide, unsigned int* CollideFlag, unsigned int* isGrap, int vertexNum);
//ʹ��������ײ���ı�������ײ����㷨
__global__ void calculateCollisionCylinderAdvanceMU(
	float* cylinderLastPos, float* cylinderPos, float* cylinderDir,
	float halfLength, float radius,
	float* positions, float* force,
	unsigned int* isCollide, unsigned int* collideFlag, float* collisionDiag,
	int vertexNum,
	float collisionStiffness, float* collisionForce,
	float* directDir, float* cylinderShift);

//ʹ��������ײ������������ײ���ж�
__device__ bool cylinderCollisionContinueMU(float length, float moveDistance, float radius, float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir, float* moveDir, float* position, float* t, float* collisionNormal, float* collisionPos, float* directDir);

int runCalculateCollisionBallMU(float ball_radius, float collisionStiffness);

__global__ void calculateCollisionBallMU(float* ballPos, float radius,
	float* positions, unsigned int* isCollide,
	float* directDirection,
	float* force, float* collisionDiag, float* insertionDepth,
	float collisionStiffness, int vertexNum);


//����Rest-pos��
extern "C" int runcalculateIFRestMU(float halfLength, float radius);

__global__ void calculateIFRestStiffnessMU(float* cylinderPosLeft, float* cylinderDirLeft, float* cylinderPosRight, float* cylinderDirRight,
	float length, float radius, float* positions, unsigned int* isCollide, unsigned int* CollideFlagLeft, unsigned int* CollideFlagRight, float* restStiffness, int vertexNum);
//���㶥�㵽������ľ���
__device__ float calculateCylinderDisMU(float posx, float posy, float posz, float dirx, float diry, float dirz, float vertx, float verty, float vertz, float length);

__global__ void calculateIFRestMU(float* positions, int* skeletonIndex, float* force, float* collisionDiag, float* rest_positions, float* restStiffness, int vertexNum);
//�б�ѩ�����λ��
extern "C" int runcalculatePosMU(float omega, float dt);

__global__ void calculatePOSMU(float* positions, float* force, float* fixed, float* mass, float* next_positions, float* prev_positions,
	float* old_positions, float* springDiag, float* collisionDiag, float* collisionForce, int vertexNum, float dt, float omega);


//�����ٶ�
extern "C" int runcalculateVMU(float dt);

__global__ void calculateVMU(float* positions, float* velocity, float* old_positions, int vertexNum, float dt);


//����ָ������
extern "C" int runUpdateDirectDirectionMU();

__global__ void updateDirectDirectionMU(float* spherePos, float* tetPositions, float* directDir, int* directIndex, int vertexNum);

__global__ void updateSphereNormalMU(float* tetSurfaceNormals, float* directDir, int* directIndex, int vertexNum);

extern "C" int runUpdateShellDirectDirectionMU();
__global__ void updateShellDirectDirectionMU(
	float* shellPos, //���ڼ���ָ�����������
	float* meshPositions, //�������񶥵�λ��
	float* directDir, // �������ǰ�����ָ����������һ�����ģ�
	int* directIndex, // �洢 �������񶥵�-��Ƕ��� ��Ӧ
	int vertexNum);

//����mesh���㷨��
extern "C" int runUpdateMeshNormalMU();

//������㷨����Ϣ����Ҫ���¼���
__global__ void clearNormalMU(float* meshNormal, float* totAngle, int vertexNum);

//���ݶ���λ�ú�������Ƭ������Ƭ����
__global__ void updateMeshNormalMU(float* meshPosition, float* meshNormal, float* totAngle, unsigned int* meshTriangle, int meshTriangleNum);

//���߹�һ��
__global__ void normalizeMeshNormalMU(float* meshNormal, float* totAngle, int meshVertexNum);

__global__ void normalizeMeshtriVertNorm_debug(float* meshNormal, float* meshPosition, float* totAngle, int meshVertexNum);
//���»�������
extern "C" int runUpdateDrawMeshMU();

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


#pragma region hapticCuda.cu����
//�������˵�Բ����Ͷ������ײ���
extern "C" int runHapticCollision(float halfLength, float radius);


//�������˵�������ײ���
extern "C" int runHapticContinueCollision(float* start, float* end, float halfLength, float radius);

//���ߺ�����
extern "C" int runHapticCollisionSphere(float halfLength, float radius);

//���ߺ�����--Mesh
extern "C" int runHapticCollisionSphere_Tri(float halfLength, float radius);

// �ۼ���������֮֡�乤�߶�����ʩ�ӵ�������
extern "C" int runAccumulateExternForce(int point_num);

// ������ͬ�����嶥��λ�õ�ʱ�����У��Ʋⶥ��λ�õı仯��
extern "C" int runUpdateSurfacePointPosition(float dt, int point_num);
//�����������Ϣ���ݵ������嶥��
extern "C" int runDispatchSphereToTet();
extern "C" int runDispatchForceToTetVertex();

// û��ʩ����
__global__ void hapticCalculateCCylinder(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* tetPositions, unsigned int* isCollide, float* zone, int vertexNum, int* index);

// �Զ����з�����
__global__ void hapticCalculateMeshCylinder(
	float* cylinderPos, float* cylinderPhysicalPos,
	float* cylinderDir, float halfLength, float radius,
	float* tetPositions,
	float* vertexNormals,
	unsigned int* isCollide,
	float* vertexForce,
	float* zone,
	int* continuousFrameCounter, int max_continuous_frame,
	int vertexNum,
	int* index);

__global__ void hapticCalculateMeshCapsule(float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* tetPositions,
	float* vertexNormals,
	unsigned int* isCollide,
	float* vertexForce, // �ӹ���ָ����ײ��ķ�����
	float* zone,
	int vertexNum,
	int* index);

// �����صı��涥���뽺�������ײ��⡣û������Ӱ�������û�з�֧������Զ���ʩ��ѹ������Ӱ��collisionDiag��
__global__ void hapticCollision_MeshCapsule(float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* vertexPositions,
	float* vertexNormals,
	unsigned int* isCollide,
	float* vertexForce, // �ӹ���ָ����ײ��ķ�����
	float* collisionDiag, float collisionStiffness,
	float* zone,
	int vertexNum,
	int* index);

__global__ void hapticCalculateSurfaceCylinder(
	float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* vertexPositions, // ȫ�������嶥������
	int* surfaceIndices,// ��ģ�ͱ���Ķ����±�
	unsigned int* isCollide, // ������Ƿ���ײ��������surfaceTriangleNumһ��
	float* zone,// �������ײ���ڹ����ϵ����λ�ã�������surfaceTriangleNumһ��
	int surfaceTriangleNum,// ���������ε�����
	int* index//???
);
//����ǰ׺��
__global__ void hapticCalculatePrefixSum(unsigned int* isCollide, unsigned int* queueIndex, unsigned int* auxArray, int vertexNum);
__global__ void hapticAddCollisionToQueue(unsigned int* isCollide, float* tetPositions, float* tetNormals, float* zone, float* constraintPoints, float* constraintNormals, float* constraintZone, unsigned int* queueIndex, unsigned int* auxArray, int vertexNum);
__global__ void hapticAddCollisionToQueue_SaveMap(unsigned int* isCollide, \
	float* tetPositions, float* tetNormals, float* zone, \
	float* constraintPoints, float* constraintNormals, float* constraintZone, \
	unsigned int* queueIndex, unsigned int* auxArray, int vertexNum, \
	int* collisionIndex_to_vertIndex_array);

//��ʼ������ײ�ļ��
__global__ void hapticCalculateContinueCylinder(float startx, float starty, float startz, float endx, float endy, float endz, int* index, float* positions, float* boxs, float* triangleNormal, int boxNum);

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
	int* index);

__device__ bool hapticCylinderCollisionContinue(
	float length, float radius,
	float* HPos, float* SPos,
	float* cylinderDir,
	float* position,
	float* collisionNormal, float* collisionPos);

//Բ���������ײ
__global__ void hapticCalculateCylinderSphere(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* sphereInfo, float* sphereForce, unsigned int* isCollide, float* zone, int* index, int sphereNum);

//Բ���������ײ--Mesh
__global__ void hapticCalculateCylinderSphere_Tri(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* sphereInfos, float* sphereForce, unsigned int* isCollide, float* zone, int* index, int sphereNum);

//�����ݷ��õ�������
__global__ void hapticAddSphereCollisionToQueue(unsigned int* isCollide, float* sphereInfos, float* zone, float* directDirection, float* constraintPoints, float* constraintZone, float* constraintDirection, unsigned int* queueIndex, unsigned int* auxArray, int sphereNum);

//�����ݷ��õ�������--Mesh
__global__ void hapticAddSphereCollisionToQueue_Tri(unsigned int* isCollide, float* sphereInfos, float* zone, float* constraintPoints, float* constraintZone, unsigned int* queueIndex, unsigned int* auxArray, int sphereNum);

//�����߶κͰ�Χ�е��ཻ
__device__ bool hapticLineSegAABBInsect(float* start, float* end, float* p0, float* p1, float* boxs);

//�����߶κ������ε��ཻ
__device__ bool hapticLineSegTriangleInsect(float* start, float* end, float* pos0, float* pos1, float* pos2, float* triangleNormal, float* ans);

//
__global__ void dispatchToTet(unsigned int* skeletonIndex, float* skeletonCoord, float* externForce, float* sphereForce, unsigned int* isCollide, int sphereNum);
__global__ void dispatchForceToTetVertex(
	float* externForce,
	float* vertexForce,
	unsigned int* isCollide,
	int vertexNum);

//���¶���λ�ã�ÿ���̼߳���һ�������λ�ø��¡�
__global__ void hapticUpdatePointPosition(
	float* mass,
	float* position,
	float* velocity,
	float* forceFromTool,
	float dt,
	int point_num);

//�ۼ���������֮֡�乤�߶�����ʩ�ӵ�������
__global__ void AccumulateExternForce(
	float* externForceTotal,
	float* externForce,
	int point_num
);

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
	float* collisionPos // ���㱻�ų���ɨ�������λ��
);


//��������
__device__ void hapticSwap(float* a, float* b);
__device__ float hapticClamp(float a, float min, float max);
__device__ void hapticCross(float* a, float* b, float* c);
__device__ float hapticDot(float* a, float* b);
__device__ float ContactForceDecay(float distance, float original_radius, float max_radius);
#pragma endregion