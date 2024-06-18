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
	///�洢������Ϣ
	int m_unityId = -1;//��unity�����е�id
	std::string m_Type;//���������õ�����
	std::string m_Name;//���������õ�����
	std::string m_binFile;//�������ļ�����
	std::string m_objFile;
	std::string m_tetFile;

	int m_vertNum;///������ʵ�ʵĶ���
	int m_renderVertNum;///��������Ⱦģ�͵Ķ���
	int m_triNum;///��������������Ķ���
	float m_RT[16];//��ʾ������תƽ�Ƶľ���
	std::vector<int> m_renderVertGPUMapping;//ÿ����Ⱦ������GPU�ж���Ķ�Ӧ��ϵ
	std::vector<int> m_triIdx;///ԭʼ������
	std::vector<int> m_renderTriIdx;///ԭʼ������
	std::vector<float> m_uv;//��������
	std::vector<float> m_renderTriVerts;//��Ⱦ�����ζ���
	std::vector<float> m_renderTriNorm;//��Ⱦ�����η�����
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