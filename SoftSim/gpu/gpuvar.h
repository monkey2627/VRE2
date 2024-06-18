#pragma once
#include <chrono> // ���ڸ߾��ȼ��㺯������ʱ��
#include "cuda.h"
#include "cuda_runtime.h"  

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


//#define OUTPUT_INFO
#define LOOK_THREAD 808

#pragma region  cuda_pd.cu

extern	float             gravityX_d;//����
extern	float				gravityY_d;//����
extern	float				gravityZ_d;//����

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
extern int tetSpringNum_d;
extern int tetActiveNum_d;
extern int tetVertActiveNum_d; 
extern int tetActiveOffset_d;
extern int tetVertActiveOffset_d;

extern  float* tetVertPos_d;
extern  float* tetVertRestPos_d;
extern	float* normals_D;
extern  float* tetVertPos_last_d;
extern  float* tetVertPos_old_d;
extern  float* tetVertPos_prev_d;
extern  float* tetVertPos_next_d;
extern  float* tetVertVelocity_d;
extern  float* tetVertExternForce_d;
extern  float* tetVertMass_d;
extern  int* tetIndex_d;

extern  float* tetVertCollisionDiag_d;
extern  float* tetVertFixed_d;
extern  bool* tetActive_d;
extern  float* tetInvD3x3_d;//Ԥ����
extern  float* tetInvD3x4_d;//Ԥ����
extern  float* tetVolume_d;//Ԥ����
extern float* tetVolumeDiag_d;
extern  float* tetVertForce_d;
extern float* tetVertForceLen_d;
extern	float* tetStiffness_d;
extern float* tetVertRestStiffness_d;
extern int* tetVert2TriVertMapping_d;
extern float* tetVertfromTriStiffness_d;

extern int* tetSpringIndex_d;
extern float* tetSpringOrgLen_d;
extern float* tetSpringStiffness_d;

extern int* tetVertRelatedTetIdx_d; // �����嶥����ص��������ţ�����ӦΪtetNum*4
extern int* tetVertRelatedTetInfo_d; // ��¼�����嶥���Ӧ�ġ���������塱��ʼ������������ ����ΪtetVertNum*2

extern int onSurfaceTetVertNum_d;
extern int* onSurfaceTetVertIndices_d; // ���������嶥���±�
extern int* tetVertBindingTetVertIndices_d; // �����嶥��󶨵ı������Ƕ����±꣬�����ڲ�ʹ������������¼���ָ������������ΪtetVertNum_d*3
extern float* tetVertBindingTetVertWeight_d; // �󶨵ı����������񶥵�Ա��淨�����Ĺ���Ȩ�أ�����ΪtetVertNum*3;

//��ײԼ����
extern float* tetVertCollisionForce_d;
extern float* tetVertCollisionForceLen_d;
extern float* tetVertCollisionForceLast_d;
extern float* tetVertInsertionDepth_d;
extern float* triVertInsertionDepth_d;

//Ϊ��ײ���㿪�ٿռ�
extern float* planeNormal_D;//ƽ�����
extern float* planePos_D;

extern float* toolPositionAndDirection_d;
extern float* toolPosePrev_d;
extern float* radius_d;

extern int* hapticCollisionNum_d;
extern float* toolContactDeltaPos_triVert_d;
extern float* totalFC_d;
extern float* totalPartial_FC_X_d;
extern float* totalPartial_FC_Omega_d;
extern float* totalTC_d;
extern float* totalPartial_TC_X_d;
extern float* totalPartial_TC_Omega_d;

extern unsigned int* tetIsCollide_d;			//�Ƿ�������ײ
extern unsigned int* isCollideGraphical_D;   //���⹤���Ƿ�����涥�㷢����ײ��
extern unsigned int* CollideFlag_D;			//��־λ�������Ƿ�������ײ

extern unsigned int* isSelfCollide_D;		//�Ƿ�������ײ����ʾ�Ƿ���������ײ��
extern float* sphereExternForce_D;	//����ײ�ܵ�������


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
extern float* tetVertNonPenetrationDir_d;
extern int* tetShellIdx_d;

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


#define GRABED_TETIDX 64

//������ײ��Ҫ�İ�Χ��
extern float* tetSurfaceAABB_D;


//����ײ�������										
extern int				aabbBoxNum_D;
extern float* aabbBoxs_D;
extern int				hashNum_D;
extern HashEntry_D* hashTable_D;
extern float* vertexLineAABB_D;

/**************************************************����ģ������**************************************************/
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
extern  int		triVertOrgNum_d;
extern	float* triVertPos_d;
extern  float* triVertRestPos_d;
extern	float* triVertPos_old_d;
extern	float* triVertPos_prev_d;
extern	float* triVertPos_next_d;
extern	float* triVertVelocity_d;
extern	float* triVertExternForce_d;
extern	float* triVertMass_d;
extern	float* triVertFixed_d;
extern	float* triVertForce_d;
extern	float* triVertNorm_d;
extern	float* triVertNormAccu_d;
extern float* triVertNonPenetrationDir_d;
extern float* triVertProjectedPos_d;
extern int* triShellIdx_d;

//spring
extern	int		triEdgeNum_d;
extern	unsigned int* triEdgeIndex_d;
extern	float* triEdgeOrgLength_d;
extern	float* triEdgeDiag_d;
extern	float* triVertCollisionDiag_d;
extern	float* triVertRestStiffness_d;
extern  int* triVert2TetVertMapping_d;
extern  float* triVertfromTetStiffness_d;
extern	float* triEdgeStiffness_d;


extern int triNum_d;
extern  unsigned int* triIndex_d;


extern	float* directDirection_D;
//ָ������
extern	float* directDirectionMU_D;
extern	int* directIndexMU_D;

//ץǯ����ײ��Ϣ
extern float	grapperRadiusMU_D;
extern float	grapperLengthMU_D;

extern	unsigned char* triVertisCollide_d;  //�Ƿ�����ײ
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



#pragma region Բ������
///����Բ����������˳������
extern int cylinderNum_d;
extern float* cylinderShift_d;
extern float* cylinderLastPos_d;
extern float* cylinderPos_d;//Բ������
extern float* cylinderDirZ_d;// Բ������
extern float* cylinderV_d;		//�������ٶ�
extern unsigned char* toolCollideFlag_d;
#pragma endregion 

#pragma region ��ײ���
extern unsigned char* tetVertisCollide_d;

#pragma endregion 