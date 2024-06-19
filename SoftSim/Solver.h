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
	const bool m_useFixedStiffness = true;//������ʱָ��stiffness
public:
	//��������,�Ȳ���
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

#pragma region ����ṩ�������������
	
	///������
	std::vector<int> m_tetIndex;//����������	
	std::vector<float> m_tetStiffness;//���������Ӳ
	std::vector<float> m_tetVertPos;//����λ��	
	std::vector<float> m_tetVertFixed;//�Ƿ�̶�	
	std::vector<float> m_tetVertMass;//��������
	std::vector<float> m_tetVertRestStiffness;//ÿ����restpos��Լ��
	std::vector<float> m_tetVertfromTriStiffness;//�����ζ�����������Լ���ն�
	std::vector<int> m_onSurfaceTetVertIndices; // �ڱ���������嶥���±�
	std::vector<float> m_tetVertDirectDirection;// �����嶥���ָ������

	///ԭʼ����������
	std::vector<int> m_triIndexOrg;// ԭʼ����������
	std::vector<float> m_triVertPosOrg; // ԭʼ����������涥������
	std::vector<int> m_triUVIndexOrg;// ϸ������������
	std::vector<float> m_triUVOrg; // ϸ������������涥������
	std::vector<float> m_triVertfromTetStiffness;

	// �������°�ĵ�������...
	std::vector<int> m_tetUpperIndex;
	std::vector<int> m_tetLowerIndex;

	///Բ����
	int m_cylinderNum = 1;
	float m_radius; // �ᱻSetToolTipRadius����

#pragma endregion


#pragma region IO
	void SavePointCloud(std::vector<float> points, std::vector<int> indices, std::string filename);
	void SavePointCloud(std::vector<float> points, std::string filename);
#pragma endregion


#pragma region ���ӻ�����
	std::vector<unsigned int> m_triIndex;// ϸ������������
	std::vector<float> m_triVertPos; // ϸ������������涥������
	std::vector<float> m_triVertColor; // ϸ������������涥����ɫ//���ӻ���
	std::vector<float> m_triVertNorm;// ϸ������������涥�㷨����
	std::vector<unsigned int> m_triUVIndex;// ϸ������������
	std::vector<float> m_triUV; // ϸ������������涥������

	std::vector<float> m_tetVertCollisionForce;
	std::vector<float> m_tetVertCollisionForceLen;
	std::vector<float> m_tetVertVolumnForce;
	std::vector<float> m_tetVertVolumnForceLen;
#pragma endregion

#pragma region ������Լ��	//�Ƿ�����Ч״̬
	std::vector<char> m_tetActive;
	//������������//���ڼ�������ݶ�
	std::vector<float> m_tetInvD3x3;
	//���ڼ���Խ���
	std::vector<float> m_tetInvD3x4;
	//��������������ȣ�tetNum
	std::vector<float> m_tetVolume;
	//����ϵͳ�ĶԽǾ����ACT*AC����, ���ȣ�tetNum
	std::vector<float> m_tetVolumeDiag;

	std::vector<int> m_tetSpringIndex; // ���������񵯻����飬
	std::vector<float> m_tetSpringOrgLength; // �����嵯��ԭ��
	std::vector<float> m_tetSpringStiffness; 
	float m_volumnSum;
	std::vector<int> m_mapTetVertIndexToTriVertIndex;//�������Ӧ���涥�㣬һ��һ
	std::vector<int> m_mapTriVertIndexToTetVertSetIndex;//���涥��������嶥�㣬һ�Զ�
	std::vector<int> m_tetVertBindingTetVertIdx; // �����嶥������������������񶥵��±ꡣ
	std::vector<float> m_TetVertIndexBindingWeight;

#pragma endregion

#pragma region ���浯��Լ��
	std::vector<unsigned int> m_edgeIndex;// ��������
	std::vector<float> m_edgeStiffness;// ���ɸն�
	std::vector<float> m_edgeVertPos;//���ɶ���
	std::vector<float> m_edgeVertFixed;
	std::vector<float> m_edgeVertMass;
	std::vector<float> m_edgeOrgLength; // ����ԭ��
	std::vector<float> m_springDiag;

	std::vector<int> m_triShellIdx;// �洢���涥���Ӧ����Ƕ����±�
	std::vector<float> m_triDirectDir; // ��ʼ���������Ƕ���ָ������

#pragma endregion
	


	bool m_useRestPos = false;
	bool m_useTetTriInteraction = true;

	void CopyToGPU();
	void SolverInit();
	void PreMalloc();
	void Step(float dt);
	void ApplyGravity();
	//void UpdateCollision();//����Բ�������ײ
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

#pragma region ����ʹ��
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

// ��ѧ����
float Matrix_Inverse_3(float* A, float* R);