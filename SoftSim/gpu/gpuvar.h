#pragma once
#include <chrono> // 用于高精度计算函数运行时间
#include "cuda.h"
#include "cuda_runtime.h"  

#define HASH_BUCKET_NUM 10
struct HashEntry_D {
	unsigned long timeStamp;		//时间戳，可以不用每次都重新初始化hash表
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

extern	float             gravityX_d;//重力
extern	float				gravityY_d;//重力
extern	float				gravityZ_d;//重力

//定义左右手工具
extern  float* cylinderShiftLeft_D;
extern	float* cylinderLastPosLeft_D;
extern	float* cylinderPosLeft_D;//圆柱参数
extern  float* cylinderGraphicalPosLeft_D;
extern	float* cylinderDirZLeft_D;
extern	float* cylinderDirYLeft_D;
extern	float* cylinderDirXLeft_D;
extern	float* cylinderVLeft_D;		//工具线速度
extern	int	  cylinderButtonLeft_D;//圆柱的行为，是否是夹取，0为正常，1为夹取
extern	float* relativePositionLeft_D;
extern	bool	firstGrabLeft_D;
extern  unsigned int* isGrapLeft_D;				//抓钳闭合后的碰撞关系
extern	unsigned int* isGrapHalfLeft_D;			//抓钳闭合过程中的碰撞关系
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
extern	unsigned int* collideFlagLeft_D;		//左手工具的碰撞标记

extern  float* cylinderShiftRight_D;
extern	float* cylinderLastPosRight_D;
extern	float* cylinderPosRight_D;//圆柱参数
extern	float* cylinderDirZRight_D;
extern	float* cylinderDirYRight_D;
extern	float* cylinderDirXRight_D;
extern	float* cylinderVRight_D;		//工具线速度
extern	int		cylinderButtonRight_D;//圆柱的行为，是否是夹取，0为正常，1为夹取


extern	float* relativePositionRight_D;
extern	bool	firstGrabRight_D;
extern  unsigned int* isGrapRight_D;		//抓钳闭合后的碰撞关系
extern	unsigned int* isGrapHalfRigth_D;	//抓钳闭合过程中的碰撞关系
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

//顶点数和四面体数
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
extern  float* tetInvD3x3_d;//预处理
extern  float* tetInvD3x4_d;//预处理
extern  float* tetVolume_d;//预处理
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

extern int* tetVertRelatedTetIdx_d; // 四面体顶点相关的四面体编号，长度应为tetNum*4
extern int* tetVertRelatedTetInfo_d; // 记录四面体顶点对应的“相关四面体”起始点和相关数量， 长度为tetVertNum*2

extern int onSurfaceTetVertNum_d;
extern int* onSurfaceTetVertIndices_d; // 表面四面体顶点下标
extern int* tetVertBindingTetVertIndices_d; // 四面体顶点绑定的表面三角顶点下标，用于在不使用球树的情况下计算指导向量。长度为tetVertNum_d*3
extern float* tetVertBindingTetVertWeight_d; // 绑定的表面三角网格顶点对表面法向量的贡献权重，长度为tetVertNum*3;

//碰撞约束力
extern float* tetVertCollisionForce_d;
extern float* tetVertCollisionForceLen_d;
extern float* tetVertCollisionForceLast_d;
extern float* tetVertInsertionDepth_d;
extern float* triVertInsertionDepth_d;

//为碰撞计算开辟空间
extern float* planeNormal_D;//平面参数
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

extern unsigned int* tetIsCollide_d;			//是否发生了碰撞
extern unsigned int* isCollideGraphical_D;   //虚拟工具是否与表面顶点发生碰撞。
extern unsigned int* CollideFlag_D;			//标志位，整体是否发生了碰撞

extern unsigned int* isSelfCollide_D;		//是否发生了碰撞【表示是否发生了自碰撞】
extern float* sphereExternForce_D;	//球碰撞受到的外力


extern unsigned int* tetMeshTriangle_D;
extern int* tetSkeletonIndex_D;
extern float* tetSkeletonCoord_D;
extern int				tetMeshVertexNumber_D;
extern int				tetMeshTriangleNumber_D;

//抓取变量
extern float timer;
extern float timeTop;
extern float timerLeft;
extern float timeTopLeft;
extern float timerRight;
extern float timeTopRight;

//四面体表面mesh索引
extern unsigned int* tetSurfaceIndex_D;
//四面体表面三角形法线
extern float* tetSurfaceNormal_D;
extern int				tetSurfaceNum_D;


//球模型的位置
extern int				sphereNum_D;
extern float* spherePositions_D;
extern unsigned int* sphereTetIndex_D;
extern float* sphereTetCoord_D;
extern unsigned int* sphereConnect_D;
extern float* sphereConnectLength_D;
extern int* sphereConnectCount_D;
extern int* sphereConnectStart_D;
//球模型的指导向量
extern int* sphereDirectIndex_D;
extern float* sphereDirectDirection_D;

//变形体之间的连接信息
extern int* connectIndex_D;
extern float* connectWeight_D;

//用于存储指导向量
extern float* tetVertNonPenetrationDir_d;
extern int* tetShellIdx_d;

// 力反馈端的工具信息
extern float* left_qg_from_HapticTool_D;
extern float* left_last_qg_from_HapticTool_D;
extern float* right_qg_from_HapticTool_D;
extern float* right_last_qg_from_HapticTool_D;



extern float* hapticDeformationExternForce_D;
extern float* hapticDeformationExternForceTotal_D;
extern int hapticCounter_D;
// 记录该顶点被工具施加压力的连续帧数。
extern int* hapticContinuousFrameNumOfCollision_D;
extern float* hapticLastNonZeroToolForce_D;
extern int MAX_CONTINUOUS_FRAME_COUNT_D;


#define GRABED_TETIDX 64

//连续碰撞需要的包围盒
extern float* tetSurfaceAABB_D;


//自碰撞所需变量										
extern int				aabbBoxNum_D;
extern float* aabbBoxs_D;
extern int				hashNum_D;
extern HashEntry_D* hashTable_D;
extern float* vertexLineAABB_D;

/**************************************************布料模型数据**************************************************/
//与布料模型的映射关系
extern int* skeletonMesh_D;
extern float* meshStiffness_D;  //约束刚度系数
extern float* meshRestPosStiffness_D;
//碰撞信息
extern float* collisionPos_Tool;
extern float* collisionNormal_Tool;
extern unsigned int* collisionFlag_Tool;

// 累加顶点对工具施加的力和力的梯度
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
//指导向量
extern	float* directDirectionMU_D;
extern	int* directIndexMU_D;

//抓钳的碰撞信息
extern float	grapperRadiusMU_D;
extern float	grapperLengthMU_D;

extern	unsigned char* triVertisCollide_d;  //是否发生碰撞
extern	unsigned int* CollideFlagMU_D;	//整体是否发生碰撞

//抓取
extern	bool	firstGrabLeftMU_D;
extern	bool	firstGrabRightMU_D;
extern	unsigned int* isGrabLeftMU_D;  //抓钳闭合后的碰撞关系
extern	unsigned int* isGrabRigthMU_D;
extern	unsigned int* isGrabHalfLeftMU_D;		//抓钳闭合过程中的闭合关系
extern	unsigned int* isGrabHalfRightMU_D;
extern	float* relativePositionLeftMU_D;
extern	float* relativePositionRightMU_D;
extern	unsigned int* CollideFlagLeftMU_D;		//标志位，工具是否与软组织发生碰撞
extern	unsigned int* CollideFlagRightMU_D;

//碰撞约束力
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
extern float* hapticCollisionZone_D;		//标记碰撞点在线段的哪个区域
extern int				hapticDeformationNum_D;
extern int				hapticDeformationNumMem_D;
extern int* hapticContinuousFrameNumOfCollision_D;     // 记录该顶点被工具施加压力的连续帧数。


extern unsigned int* hapticIsCollide_D;		//顶点是否发生碰撞
extern float* hapticConstraintPoints_D;	//约束顶点
extern float* hapticConstraintNormals_D;  // 四面体法向量
extern float* hapticConstraintZone_D;
extern int* haptic_collisionIndex_to_vertIndex_array_D; //碰撞队列下标对应的顶点下标

extern float* hapticCylinderPos_D;
extern float* hapticCylinderPhysicalPos_D;
extern float* hapticCylinderDir_D;
extern int* hapticIndex_D;

extern unsigned int* hapticQueueIndex_D;
extern unsigned int* hapticAuxSumArray_D;

extern int				hapticAABBBoxNum_D;
extern float* hapticAABBBoxs_D;
extern float* hapticTriangleNormal_D;
//不需要更新
extern int* hapticSurfaceIndex_D;

extern int				hapticSphereNum_D;
extern float* hapticSphereInfo_D;
extern float* hapticSphereDirectDirection_D;	//球的指导向量
extern float* hapticSphereForce_D;	//球收到的碰撞力
extern unsigned int* hapticSphereIsCollide_D;
extern float* hapticSphereCollisionZone_D;
extern int* hapticSphereindex_D;
extern unsigned int* hapticSphereQueueIndex_D;
extern unsigned int* hapticSphereAuxSumArray_D;
extern float* hapticSphereConstraintPoints_D;	//约束顶点
extern float* hapticSphereConstraintZone_D;
extern float* hapticSphereConstraintDirection_D;  //约束指导向量

//球树的索引的权重
extern unsigned int* hapticSphereTetIndex_D;
extern float* hapticSphereTetCoord_D;

//左右手的球树碰撞约束信息
extern int				hapticSphereConstraintNumLeft;
extern float* hapticSphereConstraintPosLeft;
extern float* hapticSphereConstraintZoneLeft;
extern float* hapticSphereConstraintDirectionLeft;
extern int				hapticSphereConstraintNumRight;
extern float* hapticSphereConstraintPosRight;
extern float* hapticSphereConstraintZoneRight;
extern float* hapticSphereConstraintDirectionRight;

// 左手点壳碰撞约束信息
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



#pragma region 圆柱参数
///所有圆柱参数，按顺序排列
extern int cylinderNum_d;
extern float* cylinderShift_d;
extern float* cylinderLastPos_d;
extern float* cylinderPos_d;//圆柱参数
extern float* cylinderDirZ_d;// 圆柱长轴
extern float* cylinderV_d;		//工具线速度
extern unsigned char* toolCollideFlag_d;
#pragma endregion 

#pragma region 碰撞相关
extern unsigned char* tetVertisCollide_d;

#pragma endregion 