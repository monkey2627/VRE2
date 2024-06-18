#pragma once
#include <chrono> // 用于高精度计算函数运行时间
#include "cuda.h"
#include "cuda_runtime.h"  
//#include "cuda_gl_interop.h"
//#include "device_functions.h"

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

#define GRAVITY				-0.0f

#pragma region  cuda_pd.cu

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
extern  float* tetInvD3x3_d;//预处理
extern  float* tetInvD3x4_d;//预处理
extern  float* tetVolume_d;//预处理
extern  float* tetVertForce_d;
extern	float* tetStiffness_d;

//extern float* multiGridConnectInfo_D;

extern float* outerShell_D;

//碰撞约束力
extern float* collisionForce_D;
extern float* collisionForceLast_D;
extern float* insertionDepth_D;

//为碰撞计算开辟空间
extern float* planeNormal_D;//平面参数
extern float* planePos_D;

extern float* ballPos_D;
extern float* radius_d;

extern unsigned int* isCollide_D;			//是否发生了碰撞【表示是否发生了自碰撞】
extern unsigned int* isCollideGraphical_D;   //虚拟工具是否与表面顶点发生碰撞。
extern unsigned int* CollideFlag_D;			//标志位，整体是否发生了碰撞

extern unsigned int* isSelfCollide_D;		//是否发生了碰撞【表示是否发生了自碰撞】
extern float* sphereExternForce_D;	//球碰撞受到的外力


//mesh操作【mesh的四面体内部蒙皮】
//extern float* tetMeshPosion_D;
//extern float* tetMeshNormal_D;	//逐顶点法线
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
extern float* directDirection_D;
extern int* directIndex_D;

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


//opengl 通信

//修正的绘制网格
extern int				drawMeshVertexNumber_D;
extern int				drawMeshTriangleNumber_D;
extern float* drawMeshPosition_D;
extern float* drawMeshNormal_D;
extern float* drawMeshPosition_H;
extern float* drawMeshNormal_H;

extern unsigned int* drawMeshskeletonCoord_D;			//与平滑mesh的对应关系


//连续碰撞需要的包围盒
extern float* tetSurfaceAABB_D;


//自碰撞所需变量										
extern int				aabbBoxNum_D;
extern float* aabbBoxs_D;
extern int				hashNum_D;
extern HashEntry_D* hashTable_D;
extern float* vertexLineAABB_D;

/**************************************************布料模型数据**************************************************/
extern	int	triVertNum_d;
extern	float* triVertPos_d;
extern	float* triVertNorm_d;
extern  float* triVertRestPos_d;

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

//指导向量
extern	float* directDirectionMU_D;
extern	int* directIndexMU_D;

//抓钳的碰撞信息
extern float	grapperRadiusMU_D;
extern float	grapperLengthMU_D;

extern	unsigned int* triVertisCollide_d;  //是否发生碰撞
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


#pragma region  cuda_pd.cu函数

int runApplyToolForce(float collisionStiffness);

__global__ void applyToolForce(float* collisionForce_D, float* hapticForceTotal_D, float* hapticForceLast_D,
	float collisionStiffness, float* tetVertCollisionDiag_d,
	int* continuousFrameCounter_D, int maxContinuousFrameCount,
	float* lastNonZeroForce_D,
	int hapticCounter,
	int vertexNum
);

/**************************************************PD解算**************************************************/
//计算初始状态
int runcalculateST(float damping, float dt, int roughGridStart, int roughGridEnd);

int runcalculateIF();
//计算更新位置
int runcalculatePOS(float omega, float dt, int start, int end);

//计算速度
int runcalculateV(float dt);
//清空自碰撞标记，和碰撞项的对角元素
int runClearCollision();
//计算平面碰撞
int runcalculateCPlane(float collisionStiffness);

//只计算挤压时的指导向量，其他时候不运行
int runcalculateToolShift(float halfLength, float radius, int flag);

//计算圆柱体碰撞
int runcalculateCollisionCylinder(float halfLength, float radius, float collisionStiffness, float adsorbStiffness, float frictionStiffness, float forceDirX, float forceDirY, float forceDirZ, int flag);

int runcalculateCollisionBall(float ball_radius, float collisionStiffness);

//计算连接力
int runcalculateConnectForce(float connectStiffness);

int runcalculateMeshForce(float halfLength, float radius);

int runcalculateMeshRestPos(float halfLength, float radius);

//计算布料约束的刚度系数
__global__ void calculateMeshStiffness(float* cylinderPosLeft, float* cylinderDirLeft,
	float* cylinderPosRight, float* cylinderDirRight, float length, float radius, float* positions,
	unsigned int* isCollide, unsigned int* CollideFlagLeft, unsigned int* CollideFlagRight,
	float* meshStiffness, int vertexNum);

//计算布料模型的约束力
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

//计算st
__global__ void calculateST(float* positions, float* velocity, float* externForce,
	float* old_positions, float* prev_positions, float* last_Positions, float* fixed,
	int vertexNum, float damping, float dt,
	int roughGridStart, int roughtGridEnd);

//计算F,R
__global__ void calculateIF(float* positions, int* tetIndex, float* tetInvD3x3, float* tetInvD3x4,
	float* force, float* tetVolumn, bool* active, int tetNum, float* volumnStiffness);

//计算碰撞力（软碰撞）
__global__ void calculateCPlane(float* planeNormal, float* planePos, float* positions,
	float* force, float* collisionDiag, int vertexNum, float collisionStiffness);

//清空碰撞信息
__global__ void clearCollision(unsigned int* isCollide, unsigned int* isCollideGraphical, float* collisionDiag, 
	float* adsorbForceLeft, float* adsorbForceRight, float* force, float* collisionForce, int vertexNum);


//计算抓取力
__global__ void calculateAdsorbForce(float* cylinderPos, float* cylinderDirX, float* cylinderDirY, float* cylinderDirZ,
	float* positions, unsigned int* isCollide, float* force, float* collisionDiag, float* relativePosition, int vertexNum, float adsorbStiffness);

__global__ void calculateAdsorbForceForHaptic(float* spherePos, int* sphereConnectStart, int* sphereConnectCount, unsigned int* sphereConnects,
	float* sphereConnectLength, int* sphereGrabFlag, float* adsorbForce, int sphereNum);
//计算碰撞力(和圆柱体)
__global__ void calculateCollisionCylinder(float* cylinderPos, float* cylinderDir, float* cylinderV, float halfLength, float radius, float* positions,
	float* velocity, float* force, unsigned int* isCollide, float* collisionDiag, float* volumnDiag, int vertexNum, float collisionStiffness, float frictionStiffness);
//计算碰撞力(和圆柱体)
__global__ void calculateCollisionCylinder(float* cylinderPos, float* cylinderDir, float* cylinderV, float halfLength, float radius, float forceDirX, float forceDirY, float forceDirZ,
	float* positions, float* velocity, float* force, unsigned int* isCollide, float* collisionDiag, float* volumnDiag, int vertexNum, float collisionStiffness, float frictionStiffness);
__global__ void calculateCollisionCylinderGraphical(float* cylinderPos, float* cylinderDir, float* cylinderV, float halfLength, float radius, float* positions, unsigned int* isCollide, int vertexNum);

//使用连续碰撞检测的变形体碰撞检测算法
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

//使用连续碰撞检测的变形体碰撞检测算法
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
//计算需要被夹取的区域的粒子
__global__ void calculateGrabCylinder(float* cylinderPos, float* cylinderDirZ,
	float* cylinderDirY, float* cylinderDirX, float grappleX, float grappleY, float grappleZ,
	float* positions, unsigned int* isCollide, unsigned int* isCollideHalf, int vertexNum,
	float* relativePosition, int* directIndex, int* sphereGrabFlag);

__global__ void calculateGrabOBB(float* grapperUpPos, float* grapperUpDirZ, float* grapperUpDirY,
	float* grapperUpDirX, float* grapperDownPos, float* grapperDownDirZ, float* grapperDownDirY,
	float* grapperDownDirX, float grappleX, float grappleY, float grappleZ,
	float* positions, int vertexNum, unsigned int* collideFlag);

//计算夹取力2.0
__global__ void calculateGrabForce(float* grapperPos, float* grapperDirZ, float* grapperDirY, float* grapperDirX,
	float grappleX, float grappleY, float grappleZ, float* positions, unsigned int* isCollide,
	int vertexNum, float adsorbStiffness, float* force, float* collisionDiag, unsigned int grabFlag);

//计算顶点到胶囊体的距离
__device__ float calculateCylinderDis(float posx, float posy, float posz, float dirx, float diry, float dirz,
	float vertx, float verty, float vertz, float length);

//清除抓取碰撞标记
__global__ void clearGrabCollide(unsigned int* isCollide, int vertexNum);

//计算碰撞力（和球）
__global__ void calculateCollisionGrab(float* cylinderLastPos, float* cylinderPos, float* grabDir, float radius,
	float* positions, unsigned int* isCollide, float* collisionDiag, int vertexNum,
	float collisionStiffness, float* force, float* directDir);
//计算偏移向量
__global__ void calculateToolShift(
	float* cylinderPos, float* cylinderDir,
	float* directDir,
	float halfLength, float radius,
	float* positions,
	float* cylinderShift,
	int vertexNum);

//计算被切割的四面体
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

//判断矩形平面和线段的相交
__device__ bool edgeCut(float* pos0, float* pos1, float* toolPos, float* dirX, float* dirY, float* dirZ, float x, float z);

__device__ bool cylinderCollision(float* pos, float* dir, float* vert, float length, float radius, float* t, float* collisionNormal, float* collisionPos);


//与自定义的obb包围盒进行碰撞检测（模拟抓钳抓取的范围）
__device__ bool obbCollision(float posx, float posy, float posz, float dirXx, float dirXy, float dirXz, float dirYx, float dirYy, float dirYz, float dirZx, float dirZy, float dirZz,
	float vertx, float verty, float vertz, float width, float length, float height);

//使用射线和球进行相交检测
__device__ bool sphereRayCollision(float* grabPos, float* meshPos, float radius, float* t, float* collisionNormal, float* collisionPos, float* meshNormal);

//使用射线和球的运动轨迹进行相交检测
__device__ bool sphereRayCollisionContinue(float* currentPos, float* meshPos, float radius, float* moveDir, float moveDistance, float* collisionPos, float* collisionNormal);
//使用连续碰撞检测进行物理碰撞的判断
__device__ bool cylinderCollisionContinue_without_directDir(
	float length, float moveDistance, float radius,
	float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir,
	float* moveDir, float* position,
	float* t, float* collisionNormal,
	float* collisionPos);

//使用连续碰撞检测进行物理碰撞的判断
__device__ bool cylinderCollisionContinue(
	float length, float moveDistance, float radius,
	float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir,
	float* moveDir, float* position,
	float* t, float* collisionNormal,
	float* collisionPos,
	float* directDir);


//计算position
__global__ void calculatePOS(float* positions, float* force, float* fixed, float* mass,
	float* next_positions, float* prev_positions, float* old_positions,
	float* volumnDiag, float* collisionDiag, float* collisionForce,
	int vertexNum, float dt, float omega);

//计算速度更新
__global__ void calculateV(float* positions, float* velocity, float* last_positions, int vertexNum, float dt);

__global__ void calculateMultiGridConstriant(float* positions, float* force, float* collisionDiag,
	float* multiGridConnectInfo, int vertexNum, float stiffness);

//计算粗网格的受力
__global__ void calculateRoughGridExternForce(float* positions, float* force, float* diag, 
	float* multiGridConnectInfo, int vertexNum);

//计算物理顶点的连接约束
__global__ void calculateConnectForce(float* positions, int* connectIndex, float* connectWeight, float* force, int vertexNum, float connectStiffness);


int runUpdateTetNormal();

//清除顶点法线信息，需要重新计算
__global__ void clearNormalTet(float* vertexNormal, int vertexNum);
//更新四面体表面法线
__global__ void updateTetNormal_old(float* tetPositions, unsigned int* tetSurfaceIndex,
	float* tetSurfaceNormal, int tetSurfaceNum);

__global__ void updateTetNormal(float* tetPositions, unsigned int* tetSurfaceIndex, float* tetNormals,
	float* tetSurfaceNormal, int tetSurfaceNum);

//根据mesh三角形索引更新表面法线
int runUpdateMeshNormal();

//根据顶点位置和三角面片计算面片索引
__global__ void updateMeshNormal(float* meshPosition, float* meshNormal,
	unsigned int* meshTriangle, int meshTriangleNum);
//法线归一化
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


#pragma region  cuda_pd_MU.cu函数
//直接用四面体位置更新Mesh位置
extern "C" int runUpdateMeshPosMU();

__global__ void UpdateMeshPosMU(float* positions, int* skeletonIndex, float* Tet_positions, int vertexNum);
//计算顶点的初速度
extern "C" int runcalculateSTMU(float damping, float dt);
extern "C" int runMapResources();

//计算顶点的初速度
extern "C" int runcalculateSTMU_part(float damping, float dt);

__global__ void calculateSTMU(float* positions, float* old_positions, float* prev_positions, float* velocity, float* externForce, float* fixed, int vertexNum, float damping, float dt);

//清除碰撞信息
extern "C" int runClearCollisionMU();

__global__ void clearCollisionMU(unsigned int* isCollide, unsigned int* collideFlagLeft, unsigned int* collideFlagRight, float* collisionDiag, float* force, float* collisionForce, float* insertionDepth, int vertexNum);

//计算顶点的受力
extern "C" int runcalculateIFMU();

__global__ void calculateIFMU(float* positions, float* force, float* springStiffness, float* springOrigin, unsigned int* springIndex, int springNum);

//计算工具的偏移
extern "C" int runcalculateToolShiftMU(float halfLength, float radius, int flag);

__global__ void calculateToolShiftMU(float* cylinderPos, float* cylinderDir, float* directDir, float halfLength, float radius, float* positions, float* cylinderShift, int vertexNum);

//使用了投影矫正的离散碰撞检测
__device__ bool cylinderRayCollisionMU(float* cylinderPos, float* cylinderDir, float vertx, float verty, float vertz, float* moveDir, float length, float radius, float* t, float* sln, float* collisionNormal, float* collisionPos);


//碰撞检测
extern "C" int runcalculateCollisionCylinderMU(
	float length, float radius,
	float collisionStiffness, float adsorbStiffness,
	int flag);
__global__ void calculateCollisionCylinderGraphicalMU(float* cylinderPos, float* cylinderDir, float* cylinderV, float halfLength, float radius, float* positions, unsigned int* isCollide, int vertexNum);

//计算需要被夹取的区域的粒子
__global__ void calculateGrabCylinderMU(float* cylinderPos, float* cylinderDirZ, float* cylinderDirY, float* cylinderDirX, float grappleX, float grappleY, float grappleZ, float* positions,
	unsigned int* isCollide, unsigned int* isCollideHalf, int vertexNum, float* relativePosition);

//与自定义的obb包围盒进行碰撞检测(模拟抓钳抓取的范围)
__device__ bool obbCollisionMU(float posx, float posy, float posz, float dirXx, float dirXy, float dirXz, float dirYx, float dirYy,
	float dirYz, float dirZx, float dirZy, float dirZz, float vertx, float verty, float vertz, float width, float length, float height);

//计算顶点吸附力--夹取中
__global__ void calculateGrabForceMU(float* grapperPos, float* grapperDirZ, float* grapperDirY, float* grapperDirX, float grappleX, float grappleY, float grappleZ,
	float* positions, unsigned int* isCollide, int vertexNum, float adsorbStiffness, float* force, float* collisionDiag, unsigned int grabFlag);

//计算抓取力--夹取完成
__global__ void calculateAdsorbForceMU(float* cylinderPos, float* cylinderDirX, float* cylinderDirY, float* cylinderDirZ, float* positions, unsigned int* isCollide,
	float* force, float* collisionDiag, float* relativePosition, int vertexNum, float adsorbStiffness);
//合并夹取的碰撞结果
__global__ void mergeCollideMU(unsigned int* isCollide, unsigned int* CollideFlag, unsigned int* isGrap, int vertexNum);
//使用连续碰撞检测的变形体碰撞检测算法
__global__ void calculateCollisionCylinderAdvanceMU(
	float* cylinderLastPos, float* cylinderPos, float* cylinderDir,
	float halfLength, float radius,
	float* positions, float* force,
	unsigned int* isCollide, unsigned int* collideFlag, float* collisionDiag,
	int vertexNum,
	float collisionStiffness, float* collisionForce,
	float* directDir, float* cylinderShift);

//使用连续碰撞检测进行物理碰撞的判断
__device__ bool cylinderCollisionContinueMU(float length, float moveDistance, float radius, float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir, float* moveDir, float* position, float* t, float* collisionNormal, float* collisionPos, float* directDir);

int runCalculateCollisionBallMU(float ball_radius, float collisionStiffness);

__global__ void calculateCollisionBallMU(float* ballPos, float radius,
	float* positions, unsigned int* isCollide,
	float* directDirection,
	float* force, float* collisionDiag, float* insertionDepth,
	float collisionStiffness, int vertexNum);


//计算Rest-pos力
extern "C" int runcalculateIFRestMU(float halfLength, float radius);

__global__ void calculateIFRestStiffnessMU(float* cylinderPosLeft, float* cylinderDirLeft, float* cylinderPosRight, float* cylinderDirRight,
	float length, float radius, float* positions, unsigned int* isCollide, unsigned int* CollideFlagLeft, unsigned int* CollideFlagRight, float* restStiffness, int vertexNum);
//计算顶点到胶囊体的距离
__device__ float calculateCylinderDisMU(float posx, float posy, float posz, float dirx, float diry, float dirz, float vertx, float verty, float vertz, float length);

__global__ void calculateIFRestMU(float* positions, int* skeletonIndex, float* force, float* collisionDiag, float* rest_positions, float* restStiffness, int vertexNum);
//切比雪夫更新位置
extern "C" int runcalculatePosMU(float omega, float dt);

__global__ void calculatePOSMU(float* positions, float* force, float* fixed, float* mass, float* next_positions, float* prev_positions,
	float* old_positions, float* springDiag, float* collisionDiag, float* collisionForce, int vertexNum, float dt, float omega);


//更新速度
extern "C" int runcalculateVMU(float dt);

__global__ void calculateVMU(float* positions, float* velocity, float* old_positions, int vertexNum, float dt);


//更新指导向量
extern "C" int runUpdateDirectDirectionMU();

__global__ void updateDirectDirectionMU(float* spherePos, float* tetPositions, float* directDir, int* directIndex, int vertexNum);

__global__ void updateSphereNormalMU(float* tetSurfaceNormals, float* directDir, int* directIndex, int vertexNum);

extern "C" int runUpdateShellDirectDirectionMU();
__global__ void updateShellDirectDirectionMU(
	float* shellPos, //用于计算指导向量的外壳
	float* meshPositions, //表面网格顶点位置
	float* directDir, // 输出：当前顶点的指导向量（归一化过的）
	int* directIndex, // 存储 表面网格顶点-外壳顶点 对应
	int vertexNum);

//更新mesh顶点法线
extern "C" int runUpdateMeshNormalMU();

//清除顶点法线信息，需要重新计算
__global__ void clearNormalMU(float* meshNormal, float* totAngle, int vertexNum);

//根据顶点位置和三角面片计算面片索引
__global__ void updateMeshNormalMU(float* meshPosition, float* meshNormal, float* totAngle, unsigned int* meshTriangle, int meshTriangleNum);

//法线归一化
__global__ void normalizeMeshNormalMU(float* meshNormal, float* totAngle, int meshVertexNum);

__global__ void normalizeMeshtriVertNorm_debug(float* meshNormal, float* meshPosition, float* totAngle, int meshVertexNum);
//更新绘制网格
extern "C" int runUpdateDrawMeshMU();

__global__ void  updateDrawMeshMU(float* meshPosition, float* meshNormal, float* drawMeshPosition, float* drawMeshNormal, unsigned int* map, int vertexNum);

/**************************************************辅助函数*************************************************/

__device__ float tettriVertNorm_d(float* vec0);

__device__ void tetCrossMU_D(float* a, float* b, float* c);

__device__ float tetDotMU_D(float* a, float* b);

__device__ float tetSolveFormulaMU_D(float* xAxis, float* yAxis, float* zAxis, float* target, float* x, float* y, float* z);

__device__ float tetPointLineDistanceMU_D(float* lineStart, float* lineDir, float* point);

__device__ float tetPointPointDistanceMU_D(float* start, float* end);

//射线和圆柱求交
__device__ void tetSolveInsectMU_D(float* lineDir, float* toolDir, float* VSubO, float radius, float* solve0, float* solve1);

//射线和球求交
__device__ void tetSolveInsectSphereMU_D(float* lineDir, float* VSubO, float radius, float* solve0, float* solve1);
#pragma endregion


#pragma region hapticCuda.cu函数
//力反馈端的圆柱体和顶点的碰撞检测
extern "C" int runHapticCollision(float halfLength, float radius);


//力反馈端的连续碰撞检测
extern "C" int runHapticContinueCollision(float* start, float* end, float halfLength, float radius);

//工具和球树
extern "C" int runHapticCollisionSphere(float halfLength, float radius);

//工具和球树--Mesh
extern "C" int runHapticCollisionSphere_Tri(float halfLength, float radius);

// 累加两个变形帧之间工具对软体施加的外力。
extern "C" int runAccumulateExternForce(int point_num);

// 在两次同步软体顶点位置的时间间隔中，推测顶点位置的变化。
extern "C" int runUpdateSurfacePointPosition(float dt, int point_num);
//将球的受力信息传递到四面体顶点
extern "C" int runDispatchSphereToTet();
extern "C" int runDispatchForceToTetVertex();

// 没有施加力
__global__ void hapticCalculateCCylinder(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* tetPositions, unsigned int* isCollide, float* zone, int vertexNum, int* index);

// 对顶点有反馈力
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
	float* vertexForce, // 从工具指向碰撞点的反馈力
	float* zone,
	int vertexNum,
	int* index);

// 最朴素的表面顶点与胶囊体的碰撞检测。没有扩大影响面积，没有分支。但会对顶点施加压力，会影响collisionDiag。
__global__ void hapticCollision_MeshCapsule(float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* vertexPositions,
	float* vertexNormals,
	unsigned int* isCollide,
	float* vertexForce, // 从工具指向碰撞点的反馈力
	float* collisionDiag, float collisionStiffness,
	float* zone,
	int vertexNum,
	int* index);

__global__ void hapticCalculateSurfaceCylinder(
	float* cylinderPos, float* cylinderDir, float halfLength, float radius,
	float* vertexPositions, // 全体四面体顶点坐标
	int* surfaceIndices,// 在模型表面的顶点下标
	unsigned int* isCollide, // 输出：是否碰撞，长度与surfaceTriangleNum一致
	float* zone,// 输出：碰撞点在工具上的相对位置，长度与surfaceTriangleNum一致
	int surfaceTriangleNum,// 表面三角形的数量
	int* index//???
);
//计算前缀和
__global__ void hapticCalculatePrefixSum(unsigned int* isCollide, unsigned int* queueIndex, unsigned int* auxArray, int vertexNum);
__global__ void hapticAddCollisionToQueue(unsigned int* isCollide, float* tetPositions, float* tetNormals, float* zone, float* constraintPoints, float* constraintNormals, float* constraintZone, unsigned int* queueIndex, unsigned int* auxArray, int vertexNum);
__global__ void hapticAddCollisionToQueue_SaveMap(unsigned int* isCollide, \
	float* tetPositions, float* tetNormals, float* zone, \
	float* constraintPoints, float* constraintNormals, float* constraintZone, \
	unsigned int* queueIndex, unsigned int* auxArray, int vertexNum, \
	int* collisionIndex_to_vertIndex_array);

//开始连续碰撞的检测
__global__ void hapticCalculateContinueCylinder(float startx, float starty, float startz, float endx, float endy, float endz, int* index, float* positions, float* boxs, float* triangleNormal, int boxNum);

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
	int* index);

__device__ bool hapticCylinderCollisionContinue(
	float length, float radius,
	float* HPos, float* SPos,
	float* cylinderDir,
	float* position,
	float* collisionNormal, float* collisionPos);

//圆柱和球的碰撞
__global__ void hapticCalculateCylinderSphere(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* sphereInfo, float* sphereForce, unsigned int* isCollide, float* zone, int* index, int sphereNum);

//圆柱和球的碰撞--Mesh
__global__ void hapticCalculateCylinderSphere_Tri(float* cylinderPos, float* cylinderDir, float halfLength, float radius, float* sphereInfos, float* sphereForce, unsigned int* isCollide, float* zone, int* index, int sphereNum);

//将数据放置到队列中
__global__ void hapticAddSphereCollisionToQueue(unsigned int* isCollide, float* sphereInfos, float* zone, float* directDirection, float* constraintPoints, float* constraintZone, float* constraintDirection, unsigned int* queueIndex, unsigned int* auxArray, int sphereNum);

//将数据放置到队列中--Mesh
__global__ void hapticAddSphereCollisionToQueue_Tri(unsigned int* isCollide, float* sphereInfos, float* zone, float* constraintPoints, float* constraintZone, unsigned int* queueIndex, unsigned int* auxArray, int sphereNum);

//计算线段和包围盒的相交
__device__ bool hapticLineSegAABBInsect(float* start, float* end, float* p0, float* p1, float* boxs);

//计算线段和三角形的相交
__device__ bool hapticLineSegTriangleInsect(float* start, float* end, float* pos0, float* pos1, float* pos2, float* triangleNormal, float* ans);

//
__global__ void dispatchToTet(unsigned int* skeletonIndex, float* skeletonCoord, float* externForce, float* sphereForce, unsigned int* isCollide, int sphereNum);
__global__ void dispatchForceToTetVertex(
	float* externForce,
	float* vertexForce,
	unsigned int* isCollide,
	int vertexNum);

//更新顶点位置，每个线程计算一个顶点的位置更新。
__global__ void hapticUpdatePointPosition(
	float* mass,
	float* position,
	float* velocity,
	float* forceFromTool,
	float dt,
	int point_num);

//累加两个变形帧之间工具对软体施加的外力。
__global__ void AccumulateExternForce(
	float* externForceTotal,
	float* externForce,
	int point_num
);

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
	float* collisionPos // 顶点被排出到扫描体外的位置
);


//辅助函数
__device__ void hapticSwap(float* a, float* b);
__device__ float hapticClamp(float a, float min, float max);
__device__ void hapticCross(float* a, float* b, float* c);
__device__ float hapticDot(float* a, float* b);
__device__ float ContactForceDecay(float distance, float original_radius, float max_radius);
#pragma endregion