#include <chrono>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <glm.hpp>
#include "Solver.h"
#include "virtual_coupling/haptic.h"
#include "HapticDevice.h"


struct SoftObject {
	///存储物体信息
	int m_unityId = -1;//在unity场景中的id
	std::string m_Type;//创建物体用的名称
	std::string m_Name;//创建物体用的名称
	std::string m_binFile;//二进制文件名称
	std::string m_objFile;
	std::string m_tetFile;

	int m_vertNum;///该物体实际的顶点
	int m_renderVertNum;///该物体渲染模型的顶点
	int m_triNum;///该物体三角网格的顶点
	float m_RT[16];//表示物体旋转平移的矩阵
	std::vector<int> m_renderVertGPUMapping;//每个渲染顶点与GPU中顶点的对应关系
	std::vector<int> m_triIdx;///原始三角形
	std::vector<int> m_renderTriIdx;///原始三角形
	std::vector<float> m_uv;//纹理坐标
	std::vector<float> m_renderTriVerts;//渲染三角形顶点
	std::vector<float> m_renderTriNorm;//渲染三角形法向量
	void ReadFromFile();
	void FileStream(FILE* fid);
	void ReadFromBin();
};

class Manager {
public:
	SoftObject m_rectumObj;
	Solver m_softHapticSolver;
	Haptic m_vcHaptic;
	HapticDevice m_hapticDevice;
	void Init();
};