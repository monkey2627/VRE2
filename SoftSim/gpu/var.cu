#include "gpuvar.h"
#include "gpufun.h"

#pragma region 模板

#pragma endregion 


#pragma region 四面体与表面三角形相互作用的
int* tetVert2TriVertMapping_d;			// 距离最近的表面三角顶点下标， tetVertNum_d
float*  tetVertfromTriStiffness_d;				//表面网格对四面体的restpos刚度系数

int* triVert2TetVertMapping_d;  // 距离最近的四面体顶点下标， triVertNum_d
float* triVertfromTetStiffness_d;				//表面网格对四面体的restpos刚度系数
#pragma endregion 

#pragma region 四面体形变
float             gravityX_d;//重力
float				gravityY_d;//重力
float				gravityZ_d;//重力
int				tetVertNum_d;				//四面体网格顶点数量
int				tetNum_d;					//四面体数量
int				tetSpringNum_d;				//四面体网格弹簧数量
int tetActiveNum_d; // 分段仿真中需要更新的四面体数量
int tetVertActiveNum_d; // 分段仿真中需要更新的四面体顶点数量
int tetActiveOffset_d; // 分段仿真中的起始位置相对于sortedTetIndices的偏移量
int tetVertActiveOffset_d;// 分段仿真中的起始位置相对于sortedTetVertIndices的偏移量
float* tetVertPos_d;				//当前位置，tetVertNum_d*3
float* tetVertRestPos_d;				//初始位置，tetVertNum_d*3
float* tetVertPos_last_d;			//上一时刻位置，tetVertNum_d*3
float* tetVertPos_old_d;			//st，tetVertNum_d*3
float* tetVertPos_prev_d;			//上一次迭代，tetVertNum_d*3
float* tetVertPos_next_d;			//下一步位置，tetVertNum_d*3
float* tetVertVelocity_d;			//速度，tetVertNum_d*3
float* tetVertExternForce_d;		//外力，tetVertNum_d*3
float* tetVertMass_d;				//质量，tetVertNum_d*3
int* tetIndex_d;					//四面体索引
float* tetVertFixed_d;				//四面体顶点是否固定，0.0f表示没有固定，tetVertNum_d
bool* tetActive_d;				//四面体是否是激活的，是否参与变形，tetNum_d
float* tetInvD3x3_d;				//逆矩阵, tetNum_d*9
float* tetInvD3x4_d;				//Ac阵， tetNum_d*12
float* tetVolume_d;				//四面体体积，tetNum_d
float* tetVolumeDiag_d;			//四面体顶点形变梯度，tetVertNum_d
int* tetSpringIndex_d;
float* tetSpringOrgLen_d;
float* tetSpringStiffness_d;
float* tetVertCollisionForce_d;
float* tetVertCollisionForceLen_d;
float* tetVertCollisionDiag_d;		//四面体顶点上的碰撞力梯度，tetVertNum_d*3
float* tetVertForce_d;					//顶点受力, tetVertNum_d*3
float* tetVertForceLen_d;
float* tetStiffness_d;				//四面体弹性系数，tetNum_d
float* tetVertRestStiffness_d;     // 四面体顶点对原始顶点的reststiffness tetVertNum_d
float* tetVertNonPenetrationDir_d; //四面体顶点的不嵌入方向

int* tetVertRelatedTetIdx_d; // 四面体顶点相关的四面体编号，长度应为tetNum*4
int* tetVertRelatedTetInfo_d; // 记录四面体顶点对应的“相关四面体”起始点和相关数量， 长度为tetVertNum*2

int onSurfaceTetVertNum_d;
int* onSurfaceTetVertIndices_d; // 表面四面体顶点下标
int* tetVertBindingTetVertIndices_d; // 四面体顶点绑定的表面三角顶点下标，用于在不使用球树的情况下计算指导向量。长度为tetVertNum_d*3
float* tetVertBindingTetVertWeight_d; // 绑定的表面三角网格顶点对表面法向量的贡献权重，长度为tetVertNum*3;
#pragma endregion 

#pragma region 表面三角形形变
// vertex
int		triVertNum_d;			  //三角网格顶点数量
int		triVertOrgNum_d;		  //未细分的表面三角网格顶点数量
int		triEdgeNum_d;			  //三角网格边的数量
int		triNum_d; 				  //三角网格三角数量

float* triVertPos_d;			  //三角网格顶点，3*triVertNum_d
float* triVertRestPos_d;		  //三角网格顶点RestPos， 3*triVertNum_d
float* triVertPos_old_d;		  // 3*triVertNum_d
float* triVertPos_prev_d;		  // 3*triVertNum_d
float* triVertPos_next_d;		  // 3*triVertNum_d
float* triVertVelocity_d;		  // 3*triVertNum_d
float* triVertExternForce_d;	  // 3*triVertNum_d
float* triVertMass_d;			  // 三角网格顶点质量，triVertNum_d
float* triVertNorm_d;			  // 三角网格顶点法向量，3*triVertNum_d
float* triVertNormAccu_d;		  // 包含该顶点的三角形夹角之和，triVertNum_d

float* triVertNonPenetrationDir_d;// 三角网格顶点指导向量, triVertNum_d*3
float* triVertProjectedPos_d; // 三角网格顶点碰撞之后被投影到的工具表面位置，如果未发生碰撞，维持顶点原位， triVertNum_d*3
int* triShellIdx_d;			  // 三角网格顶点对应的外壳顶点下标， triVertNum_d*3
//spring
unsigned int* triEdgeIndex_d;	  // 三角网格边对应的顶点下标， 2*triEdgeNum_d
float* triEdgeOrgLength_d;		  // 三角网格边原长， triEdgeNum_d
float* triEdgeDiag_d;			  // 三角网格顶点上因弹簧力产生的梯度，3*triVertNum_d
float* triVertCollisionDiag_d;	  // 三角网格顶点上因碰撞力产生的梯度，3*triVertNum_d
float* triVertRestStiffness_d;	  // 三角网格顶点上的restpos约束刚度， triVertNum_d

float* triVertFixed_d;			  // 三角网格顶点是否是固定的，0.0为非固定顶点，triVertNum_d
float* triVertForce_d;			  // 三角网格顶点上的力，3*triVertNum_d
float* triEdgeStiffness_d;		  // 三角边的弹簧刚度， triEdgeNum_d

#pragma endregion 

#pragma region 圆柱参数
///所有圆柱参数，按顺序排列
int cylinderNum_d;
char* cylinderActive_d;
float* cylinderShift_d;
float* cylinderRaidus_d;
float* cylinderLastPos_d;
float* cylinderPos_d;//圆柱参数
float* cylinderDirZ_d;// 圆柱长轴
float* cylinderV_d; //工具线速度
unsigned char* toolCollideFlag_d; // 工具是否与软体顶点重合
#pragma endregion 

#pragma region 球参数
///所有球参数，按顺序排列
int sphereNum_d;
char* sphereActive_d;
float* sphereShift_d;
float* sphereRaidus_d;
float* sphereLastPos_d;
float* spherePos_d;//圆柱参数
float* sphereV_d; //工具线速度
#pragma endregion 


#pragma region 碰撞相关
unsigned char* tetVertisCollide_d;
unsigned char* triVertisCollide_d;// 三角网格顶点是否碰撞， triVertNum_d

#pragma endregion 


#pragma region 碰撞约束力
float* tetVertInsertionDepth_d;	// 存储四面体顶点在工具中的嵌入深度
float* triVertInsertionDepth_d; // 存储表面顶点在工具中的嵌入深度
#pragma endregion 

//碰撞信息
float* collisionPos_Tool;
float* collisionNormal_Tool;
unsigned int* collisionFlag_Tool;

//用于计算约束
unsigned int* tetIsCollide_d;
unsigned int* isCollideGraphical_D;
unsigned int* CollideFlag_D;			//标志位，整体是否发生了碰撞
unsigned int* isGrap_D;
unsigned int* isSelfCollide_D;
float* sphereExternForce_D;




//mesh[用于计算四面体外骨骼蒙皮]
float* tetMeshPosion_D;
float* tetMeshNormal_D;	//逐顶点法线
unsigned int* tetMeshTriangle_D;
int* tetSkeletonIndex_D;
float* tetSkeletonCoord_D;
int				tetMeshVertexNumber_D;
int				tetMeshTriangleNumber_D;

//抓取变量
float timer;
float timeTop;
float timerLeft;
float timeTopLeft;
float timerRight;
float timeTopRight;

//四面体表面mesh信息（将用于自碰撞的检测）
unsigned int* tetSurfaceIndex_D;
float* tetSurfaceNormal_D;
int				tetSurfaceNum_D;

//球模型的位置
int				sphereNum_D;
float* spherePositions_D;
unsigned int* sphereTetIndex_D;
float* sphereTetCoord_D;
unsigned int* sphereConnect_D;
float* sphereConnectLength_D;
int* sphereConnectCount_D;
int* sphereConnectStart_D;
//球模型的指导向量
int* sphereDirectIndex_D;
float* sphereDirectDirection_D;



//连续碰撞检测需要的包围盒
float* tetSurfaceAABB_D;

//自碰撞所需变量
int				aabbBoxNum_D;
float* aabbBoxs_D;
int				hashNum_D;
HashEntry_D* hashTable_D;
float* vertexLineAABB_D;

// 累加顶点对工具施加的力和力的梯度
float* totalFc_D;
float* totalPartialFc_D;
int* collisionNum_D;


//左右手工具
float* cylinderShiftLeft_D;
float* cylinderLastPosLeft_D;
float* cylinderPosLeft_D;//圆柱参数
float* cylinderGraphicalPosLeft_D;
float* cylinderDirZLeft_D;// 圆柱长轴（高）方向？也就是工具的方向。
float* cylinderDirYLeft_D;
float* cylinderDirXLeft_D;
float* cylinderVLeft_D;		//工具线速度
int		cylinderButtonLeft_D;//圆柱的行为，是否是夹取，0为正常，1为夹取
float* relativePositionLeft_D;
bool	firstGrabLeft_D;
unsigned int* isGrapLeft_D;				//抓钳闭合后的碰撞关系
unsigned int* isGrapHalfLeft_D;			//抓钳闭合过程中的碰撞关系
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
unsigned int* collideFlagLeft_D;		//左手工具的碰撞标记


float* cylinderShiftRight_D;
float* cylinderLastPosRight_D;
float* cylinderPosRight_D;//圆柱参数
float* cylinderDirZRight_D;
float* cylinderDirYRight_D;
float* cylinderDirXRight_D;
float* cylinderVRight_D;		//工具线速度
int		cylinderButtonRight_D;//圆柱的行为，是否是夹取，0为正常，1为夹取
float* relativePositionRight_D;
bool	firstGrabRight_D;
unsigned int* isGrapRight_D;		//抓钳闭合后的碰撞关系
unsigned int* isGrapHalfRigth_D;	//抓钳闭合过程中的碰撞关系
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
unsigned int* collideFlagRight_D;// 右手工具

float* hapticLastNonZeroToolForce_D;// 记录工具碰撞体对顶点施加的、最近一次非零的原始力。

float* directDirection_D;
int* directIndex_D;

