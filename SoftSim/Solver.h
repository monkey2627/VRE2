#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <utility>
#include <algorithm>
//cuda
#include "gpu/gpufun.h"
//logger
#include "logger.h" 
#include "../3rd/glm/glm.hpp"

typedef std::pair<float, int> dist_idx;
class Solver
{
	const bool m_useFixedStiffness = true;//在运行时指定stiffness
public:
	//迭代次数,等参数
	int m_iterateNum=12;
	float volumnStiffness=1000.0f;

	bool m_useTetEdgeSpring = true;
	float tetSpringStiffnessDefault = 10.0f;
	bool m_useClusterCollision = false;
	float collisionStiffness=1500.0f;
	float connectStiffness=1000.0f;
	float distanceStiffness=200.0f;
	float selfCollisionStiffness=5000.0f;
	float adsorbStiffness=10000.0f;
	float frictionStiffness=5.0f;
	float externForceStiffness=500.0f;

	float tetVertOrgRestStiffness = 5.0f;
	float edgeStiffnessDefault = 1500.0f;
	float rho=0.9992f;
	float m_scale=1.0f;
	float dampingForTetVert = 0.95f;
	float dampingForTriVert = 0.2f;

	double m_opTime;

	double m_hapticTriOpTime;
	double m_hapticQGOpTime;
	double m_haptic6dofTime;

	float m_gravityX = 0;
	float m_gravityY = 0;
	float m_gravityZ = 0;

#pragma region 外界提供给求解器的数据
	
	///四面体
	std::vector<int> m_tetIndex;//四面体索引	
	std::vector<float> m_tetStiffness;//四面体的软硬
	std::vector<float> m_tetVertPos;//顶点位置	
	std::vector<float> m_tetVertFixed;//是否固定	
	std::vector<float> m_tetVertMass;//顶点质量
	std::vector<float> m_tetVertRestStiffness;//每个点restpos的约束
	std::vector<float> m_tetVertfromTriStiffness;//三角形顶点对四面体的约束刚度
	std::vector<int> m_onSurfaceTetVertIndices; // 在表面的四面体顶点下标
	std::vector<float> m_tetVertDirectDirection;// 四面体顶点的指导向量

	///原始表面三角形
	std::vector<int> m_triIndexOrg;// 原始三角形索引
	std::vector<float> m_triVertPosOrg; // 原始三角网格表面顶点坐标
	std::vector<int> m_triUVIndexOrg;// 细分三角形索引
	std::vector<float> m_triUVOrg; // 细分三角网格表面顶点坐标
	std::vector<float> m_triVertfromTetStiffness;

	// 区分上下半的弹簧索引...
	std::vector<int> m_tetUpperIndex;
	std::vector<int> m_tetLowerIndex;

	///圆柱体
	int m_cylinderNum = 1;
	float m_radius; // 会被SetToolTipRadius更改

#pragma endregion


#pragma region IO
	void SavePointCloud(std::vector<float> points, std::vector<int> indices, std::string filename);
	void SavePointCloud(std::vector<float> points, std::string filename);
#pragma endregion


#pragma region 可视化数据
	std::vector<unsigned int> m_triIndex;// 细分三角形索引
	std::vector<float> m_triVertPos; // 细分三角网格表面顶点坐标
	std::vector<float> m_triVertColor; // 细分三角网格表面顶点颜色//可视化用
	std::vector<float> m_triVertNorm;// 细分三角网格表面顶点法向量
	std::vector<unsigned int> m_triUVIndex;// 细分三角形索引
	std::vector<float> m_triUV; // 细分三角网格表面顶点坐标

	std::vector<float> m_tetVertCollisionForce;
	std::vector<float> m_tetVertCollisionForceLen;
	std::vector<float> m_tetVertVolumnForce;
	std::vector<float> m_tetVertVolumnForceLen;
#pragma endregion

#pragma region 四面体约束	//是否处于有效状态
	std::vector<char> m_tetActive;
	//四面体矩阵的逆//用于计算变形梯度
	std::vector<float> m_tetInvD3x3;
	//用于计算对角阵
	std::vector<float> m_tetInvD3x4;
	//四面体体积，长度：tetNum
	std::vector<float> m_tetVolume;
	//线性系统的对角矩阵的ACT*AC部分, 长度：tetNum
	std::vector<float> m_tetVolumeDiag;

	std::vector<int> m_tetSpringIndex; // 四面体网格弹簧数组，
	std::vector<float> m_tetSpringOrgLength; // 四面体弹簧原长
	std::vector<float> m_tetSpringStiffness; 
	float m_volumnSum;
	std::vector<int> m_mapTetVertIndexToTriVertIndex;//四面体对应表面顶点，一对一
	std::vector<int> m_mapTriVertIndexToTetVertSetIndex;//表面顶点对四面体顶点，一对二
	std::vector<int> m_tetVertBindingTetVertIdx; // 四面体顶点最近的三个表面网格顶点下标。
	std::vector<float> m_TetVertIndexBindingWeight;

#pragma endregion

#pragma region 表面弹簧约束
	std::vector<unsigned int> m_edgeIndex;// 弹簧索引
	std::vector<float> m_edgeStiffness;// 弹簧刚度
	std::vector<float> m_edgeVertPos;//弹簧顶点
	std::vector<float> m_edgeVertFixed;
	std::vector<float> m_edgeVertMass;
	std::vector<float> m_edgeOrgLength; // 弹簧原长
	std::vector<float> m_springDiag;

	std::vector<int> m_triShellIdx;// 存储表面顶点对应的外壳顶点下标
	std::vector<float> m_triDirectDir; // 初始化表面三角顶点指导向量

#pragma endregion
	


	bool m_useRestPos = false;
	bool m_useTetTriInteraction = true;

	void CopyToGPU();
	void SolverInit();
	void PreMalloc();
	void Step(float dt);
	void ApplyGravity();
	//void UpdateCollision();//更新圆柱体的碰撞
	void UpdateDirectDirectionTet();
	void UpdateDirectDirectionTri();

	int GetTetVertNum(void);
	int GetTetNum(void);
	int GetTetSpringNum(void);
	int GetSurfaceTriNum(void);
	int GetSurfaceVertNum(void);
	int GetOrgTriVertNum(void);
	int GetSpringNum(void);

	void AddExtraSpring();
	void InitSpringConstraint();
	void InitVolumeConstraint();
	void SurfaceSubdivision();
	void GenerateTetSpring();
	void GenerateTetVertDirectDir();
	void GenerateTetVertDDirWithTri();
	std::vector<int> m_tetVertRelatedTetIdx;
	std::vector<int> m_tetVertRelatedTetInfo;

	enum FILEMODE { W, R };
	std::string m_binFile;
	std::string m_objFile;
	std::string m_tetFile;

	FILEMODE m_fileMode;
	bool m_readFromBin = false;
	void InitFromFile();
	void ReadObjFile(const char* name);
	void ReadObjPoints(const char* name, std::vector<float>& points);
	void ReadTetFile(const char* name);
	void SaveMesh(const char* filename);
	void FileStream(FILE* fid);
	void FileStream(FILE* fid,std::vector<int>& vec, std::string msg);
	void FileStream(FILE* fid, std::vector<char>& vec, std::string msg);
	void FileStream(FILE* fid, std::vector<float>& vec, std::string msg);
	void FileStream(FILE* fid, std::vector<unsigned int>& vec, std::string msg);
	void ReadFromBin();
	void AddFixedPoint(float x, float y, float z, float r);
	void WriteToBin();

#pragma region 调试使用
	unsigned int renderStepNumPassed = 0;
	unsigned int hapticStepNum = 0;
	float m_toolTrans[16];
	void GetToolDir(float* dir);
	void GetToolPos(float* pos);
	void GetToolTip(float* pos);
	void GetTool6DOFPose(float* pose_6dof);
	float toolLength = 1;
	void OutputTrans();
	bool OUTPUT_TIME_TO_CSV = false;
	std::ofstream timeInfo;
	std::ofstream timeInfo_6dof;
#pragma endregion
};

// 数学计算
float Matrix_Inverse_3(float* A, float* R);