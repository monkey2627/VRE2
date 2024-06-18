#include "gpuvar.h"
#include "gpufun.h"

#pragma region ģ��

#pragma endregion 


#pragma region ������������������໥���õ�
int* tetVert2TriVertMapping_d;			// ��������ı������Ƕ����±꣬ tetVertNum_d
float*  tetVertfromTriStiffness_d;				//����������������restpos�ն�ϵ��

int* triVert2TetVertMapping_d;  // ��������������嶥���±꣬ triVertNum_d
float* triVertfromTetStiffness_d;				//����������������restpos�ն�ϵ��
#pragma endregion 

#pragma region �������α�
float             gravityX_d;//����
float				gravityY_d;//����
float				gravityZ_d;//����
int				tetVertNum_d;				//���������񶥵�����
int				tetNum_d;					//����������
int				tetSpringNum_d;				//���������񵯻�����
int tetActiveNum_d; // �ֶη�������Ҫ���µ�����������
int tetVertActiveNum_d; // �ֶη�������Ҫ���µ������嶥������
int tetActiveOffset_d; // �ֶη����е���ʼλ�������sortedTetIndices��ƫ����
int tetVertActiveOffset_d;// �ֶη����е���ʼλ�������sortedTetVertIndices��ƫ����
float* tetVertPos_d;				//��ǰλ�ã�tetVertNum_d*3
float* tetVertRestPos_d;				//��ʼλ�ã�tetVertNum_d*3
float* tetVertPos_last_d;			//��һʱ��λ�ã�tetVertNum_d*3
float* tetVertPos_old_d;			//st��tetVertNum_d*3
float* tetVertPos_prev_d;			//��һ�ε�����tetVertNum_d*3
float* tetVertPos_next_d;			//��һ��λ�ã�tetVertNum_d*3
float* tetVertVelocity_d;			//�ٶȣ�tetVertNum_d*3
float* tetVertExternForce_d;		//������tetVertNum_d*3
float* tetVertMass_d;				//������tetVertNum_d*3
int* tetIndex_d;					//����������
float* tetVertFixed_d;				//�����嶥���Ƿ�̶���0.0f��ʾû�й̶���tetVertNum_d
bool* tetActive_d;				//�������Ƿ��Ǽ���ģ��Ƿ������Σ�tetNum_d
float* tetInvD3x3_d;				//�����, tetNum_d*9
float* tetInvD3x4_d;				//Ac�� tetNum_d*12
float* tetVolume_d;				//�����������tetNum_d
float* tetVolumeDiag_d;			//�����嶥���α��ݶȣ�tetVertNum_d
int* tetSpringIndex_d;
float* tetSpringOrgLen_d;
float* tetSpringStiffness_d;
float* tetVertCollisionForce_d;
float* tetVertCollisionForceLen_d;
float* tetVertCollisionDiag_d;		//�����嶥���ϵ���ײ���ݶȣ�tetVertNum_d*3
float* tetVertForce_d;					//��������, tetVertNum_d*3
float* tetVertForceLen_d;
float* tetStiffness_d;				//�����嵯��ϵ����tetNum_d
float* tetVertRestStiffness_d;     // �����嶥���ԭʼ�����reststiffness tetVertNum_d
float* tetVertNonPenetrationDir_d; //�����嶥��Ĳ�Ƕ�뷽��

int* tetVertRelatedTetIdx_d; // �����嶥����ص��������ţ�����ӦΪtetNum*4
int* tetVertRelatedTetInfo_d; // ��¼�����嶥���Ӧ�ġ���������塱��ʼ������������ ����ΪtetVertNum*2

int onSurfaceTetVertNum_d;
int* onSurfaceTetVertIndices_d; // ���������嶥���±�
int* tetVertBindingTetVertIndices_d; // �����嶥��󶨵ı������Ƕ����±꣬�����ڲ�ʹ������������¼���ָ������������ΪtetVertNum_d*3
float* tetVertBindingTetVertWeight_d; // �󶨵ı����������񶥵�Ա��淨�����Ĺ���Ȩ�أ�����ΪtetVertNum*3;
#pragma endregion 

#pragma region �����������α�
// vertex
int		triVertNum_d;			  //�������񶥵�����
int		triVertOrgNum_d;		  //δϸ�ֵı����������񶥵�����
int		triEdgeNum_d;			  //��������ߵ�����
int		triNum_d; 				  //����������������

float* triVertPos_d;			  //�������񶥵㣬3*triVertNum_d
float* triVertRestPos_d;		  //�������񶥵�RestPos�� 3*triVertNum_d
float* triVertPos_old_d;		  // 3*triVertNum_d
float* triVertPos_prev_d;		  // 3*triVertNum_d
float* triVertPos_next_d;		  // 3*triVertNum_d
float* triVertVelocity_d;		  // 3*triVertNum_d
float* triVertExternForce_d;	  // 3*triVertNum_d
float* triVertMass_d;			  // �������񶥵�������triVertNum_d
float* triVertNorm_d;			  // �������񶥵㷨������3*triVertNum_d
float* triVertNormAccu_d;		  // �����ö���������μн�֮�ͣ�triVertNum_d

float* triVertNonPenetrationDir_d;// �������񶥵�ָ������, triVertNum_d*3
float* triVertProjectedPos_d; // �������񶥵���ײ֮��ͶӰ���Ĺ��߱���λ�ã����δ������ײ��ά�ֶ���ԭλ�� triVertNum_d*3
int* triShellIdx_d;			  // �������񶥵��Ӧ����Ƕ����±꣬ triVertNum_d*3
//spring
unsigned int* triEdgeIndex_d;	  // ��������߶�Ӧ�Ķ����±꣬ 2*triEdgeNum_d
float* triEdgeOrgLength_d;		  // ���������ԭ���� triEdgeNum_d
float* triEdgeDiag_d;			  // �������񶥵����򵯻����������ݶȣ�3*triVertNum_d
float* triVertCollisionDiag_d;	  // �������񶥵�������ײ���������ݶȣ�3*triVertNum_d
float* triVertRestStiffness_d;	  // �������񶥵��ϵ�restposԼ���նȣ� triVertNum_d

float* triVertFixed_d;			  // �������񶥵��Ƿ��ǹ̶��ģ�0.0Ϊ�ǹ̶����㣬triVertNum_d
float* triVertForce_d;			  // �������񶥵��ϵ�����3*triVertNum_d
float* triEdgeStiffness_d;		  // ���Ǳߵĵ��ɸնȣ� triEdgeNum_d

#pragma endregion 

#pragma region Բ������
///����Բ����������˳������
int cylinderNum_d;
char* cylinderActive_d;
float* cylinderShift_d;
float* cylinderRaidus_d;
float* cylinderLastPos_d;
float* cylinderPos_d;//Բ������
float* cylinderDirZ_d;// Բ������
float* cylinderV_d; //�������ٶ�
unsigned char* toolCollideFlag_d; // �����Ƿ������嶥���غ�
#pragma endregion 

#pragma region �����
///�������������˳������
int sphereNum_d;
char* sphereActive_d;
float* sphereShift_d;
float* sphereRaidus_d;
float* sphereLastPos_d;
float* spherePos_d;//Բ������
float* sphereV_d; //�������ٶ�
#pragma endregion 


#pragma region ��ײ���
unsigned char* tetVertisCollide_d;
unsigned char* triVertisCollide_d;// �������񶥵��Ƿ���ײ�� triVertNum_d

#pragma endregion 


#pragma region ��ײԼ����
float* tetVertInsertionDepth_d;	// �洢�����嶥���ڹ����е�Ƕ�����
float* triVertInsertionDepth_d; // �洢���涥���ڹ����е�Ƕ�����
#pragma endregion 

//��ײ��Ϣ
float* collisionPos_Tool;
float* collisionNormal_Tool;
unsigned int* collisionFlag_Tool;

//���ڼ���Լ��
unsigned int* tetIsCollide_d;
unsigned int* isCollideGraphical_D;
unsigned int* CollideFlag_D;			//��־λ�������Ƿ�������ײ
unsigned int* isGrap_D;
unsigned int* isSelfCollide_D;
float* sphereExternForce_D;




//mesh[���ڼ����������������Ƥ]
float* tetMeshPosion_D;
float* tetMeshNormal_D;	//�𶥵㷨��
unsigned int* tetMeshTriangle_D;
int* tetSkeletonIndex_D;
float* tetSkeletonCoord_D;
int				tetMeshVertexNumber_D;
int				tetMeshTriangleNumber_D;

//ץȡ����
float timer;
float timeTop;
float timerLeft;
float timeTopLeft;
float timerRight;
float timeTopRight;

//���������mesh��Ϣ������������ײ�ļ�⣩
unsigned int* tetSurfaceIndex_D;
float* tetSurfaceNormal_D;
int				tetSurfaceNum_D;

//��ģ�͵�λ��
int				sphereNum_D;
float* spherePositions_D;
unsigned int* sphereTetIndex_D;
float* sphereTetCoord_D;
unsigned int* sphereConnect_D;
float* sphereConnectLength_D;
int* sphereConnectCount_D;
int* sphereConnectStart_D;
//��ģ�͵�ָ������
int* sphereDirectIndex_D;
float* sphereDirectDirection_D;



//������ײ�����Ҫ�İ�Χ��
float* tetSurfaceAABB_D;

//����ײ�������
int				aabbBoxNum_D;
float* aabbBoxs_D;
int				hashNum_D;
HashEntry_D* hashTable_D;
float* vertexLineAABB_D;

// �ۼӶ���Թ���ʩ�ӵ����������ݶ�
float* totalFc_D;
float* totalPartialFc_D;
int* collisionNum_D;


//�����ֹ���
float* cylinderShiftLeft_D;
float* cylinderLastPosLeft_D;
float* cylinderPosLeft_D;//Բ������
float* cylinderGraphicalPosLeft_D;
float* cylinderDirZLeft_D;// Բ�����ᣨ�ߣ�����Ҳ���ǹ��ߵķ���
float* cylinderDirYLeft_D;
float* cylinderDirXLeft_D;
float* cylinderVLeft_D;		//�������ٶ�
int		cylinderButtonLeft_D;//Բ������Ϊ���Ƿ��Ǽ�ȡ��0Ϊ������1Ϊ��ȡ
float* relativePositionLeft_D;
bool	firstGrabLeft_D;
unsigned int* isGrapLeft_D;				//ץǯ�պϺ����ײ��ϵ
unsigned int* isGrapHalfLeft_D;			//ץǯ�պϹ����е���ײ��ϵ
float* adsorbForceLeft_D;
float* tetgrapperUpPosLeft_D;
float* tetgrapperDownPosLeft_D;
float* tetgrapperUpDirZLeft_D;
float* tetgrapperUpDirXLeft_D;
float* tetgrapperUpDirYLeft_D;
float* tetgrapperDownDirZLeft_D;
float* tetgrapperDownDirXLeft_D;
float* tetgrapperDownDirYLeft_D;
int* grabFlagLeft_D;
float* left_qg_from_HapticTool_D;
float* left_last_qg_from_HapticTool_D;
unsigned int* collideFlagLeft_D;		//���ֹ��ߵ���ײ���


float* cylinderShiftRight_D;
float* cylinderLastPosRight_D;
float* cylinderPosRight_D;//Բ������
float* cylinderDirZRight_D;
float* cylinderDirYRight_D;
float* cylinderDirXRight_D;
float* cylinderVRight_D;		//�������ٶ�
int		cylinderButtonRight_D;//Բ������Ϊ���Ƿ��Ǽ�ȡ��0Ϊ������1Ϊ��ȡ
float* relativePositionRight_D;
bool	firstGrabRight_D;
unsigned int* isGrapRight_D;		//ץǯ�պϺ����ײ��ϵ
unsigned int* isGrapHalfRigth_D;	//ץǯ�պϹ����е���ײ��ϵ
float* adsorbForceRight_D;
float* tetgrapperUpPosRight_D;
float* tetgrapperDownPosRight_D;
float* tetgrapperUpDirZRight_D;
float* tetgrapperUpDirXRight_D;
float* tetgrapperUpDirYRight_D;
float* tetgrapperDownDirZRight_D;
float* tetgrapperDownDirXRight_D;
float* tetgrapperDownDirYRight_D;
int* grabFlagRight_D;
float* right_qg_from_HapticTool_D;
float* right_last_qg_from_HapticTool_D;
unsigned int* collideFlagRight_D;// ���ֹ���

float* hapticLastNonZeroToolForce_D;// ��¼������ײ��Զ���ʩ�ӵġ����һ�η����ԭʼ����

float* directDirection_D;
int* directIndex_D;

