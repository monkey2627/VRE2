//////////所有和初始化相关的加到这个函数中
//////////包括文件读取，系统矩阵初始化

#include <io.h> 
#include <stdio.h>
#include <Windows.h>
#include <map>
#include "Solver.h"
#include "gpu/gpuvar.h"
#include "bridge.h"
using namespace std;

bool VALIDATE = true;
//矩阵求逆
float Matrix_Inverse_3(float* A, float* R)				//R=inv(A)
{
	R[0] = A[4] * A[8] - A[7] * A[5];
	R[1] = A[7] * A[2] - A[1] * A[8];
	R[2] = A[1] * A[5] - A[4] * A[2];
	R[3] = A[5] * A[6] - A[3] * A[8];
	R[4] = A[0] * A[8] - A[2] * A[6];
	R[5] = A[2] * A[3] - A[0] * A[5];
	R[6] = A[3] * A[7] - A[4] * A[6];
	R[7] = A[1] * A[6] - A[0] * A[7];
	R[8] = A[0] * A[4] - A[1] * A[3];
	float det = A[0] * R[0] + A[3] * R[1] + A[6] * R[2];
	float inv_det = 1 / det;
	for (int i = 0; i < 9; i++)	R[i] *= inv_det;
	return det;
}

float l2_dis(float* p0, float* p1, unsigned int dim = 3)
{
	float result = 0;
	for (int i = 0; i < dim; i++)
	{
		result += (p1[i] - p0[i]) * (p1[i] - p0[i]);
	}
	return sqrt(result);
}

int g_unityID;

void Solver::AddFixedPoint(float x, float y, float z, float r)
{
	float q[3] = { x, y, z };
	float p[3];
	for (int i = 0; i < GetTetVertNum(); i++)
	{
		p[0] = m_tetVertPos[i * 3 + 0];
		p[1] = m_tetVertPos[i * 3 + 1];
		p[2] = m_tetVertPos[i * 3 + 2];
		if (l2_dis(p, q) < r)
		{
			m_tetVertFixed[i] = 1e9;
		}
	}
	printf("\t\tSurfaceVertNum:%d\n", GetSurfaceVertNum());
	for (int i = 0; i < GetSurfaceVertNum(); i++)
	{
		p[0] = m_triVertPos[i * 3 + 0];
		p[1] = m_triVertPos[i * 3 + 1];
		p[2] = m_triVertPos[i * 3 + 2];
		auto dis = l2_dis(p, q);
		printf("id:%d distance: %f\n", i, dis);
		if (l2_dis(p, q) < r)
		{
			m_edgeVertFixed[i] = 1e9;
			printf("threadid:%d, fixed:%f\n", i, m_edgeVertFixed[i]);
		}
	}
}
void Solver::SolverInit() {
	if (access(m_binFile.c_str(), 0) != 0) {
		InitFromFile();

		GenerateTetSpring();
		GenerateTetVertDDirWithTri(); // 必须在表面网格细分之前
		SurfaceSubdivision();		
		InitSpringConstraint();
		InitVolumeConstraint();		
		if (strcmp(m_binFile.c_str(), "./data/softdeformation__sf.bin") == 0)
		{
			// Add fixed point
			printf("AddFixedPoint\n");
			AddFixedPoint(3.0, 5.0, 0.0, 2.5);
			AddFixedPoint(-3.0, 5.0, 0.0, 1);
		}
		if (strcmp(m_binFile.c_str(), "./data/liver.bin") == 0)
		{
			// Add fixed point
			printf("AddFixedPoint\n");
			AddFixedPoint(-0.9, 0.5, -1.5, 0.7);
		}
		if (strcmp(m_binFile.c_str(), "./data/two-objects.bin") == 0)
		{
			printf("AddFixedPoint two-objects\n");
			AddFixedPoint(0, 3.75, 0, 4.4);
		}
		WriteToBin();
	}
	else {
		printf("read bin file: %s\n", m_binFile.c_str());
		ReadFromBin();
	}
	if (m_useFixedStiffness) {
		fill(m_tetStiffness.begin(), m_tetStiffness.end(), volumnStiffness);
		fill(m_edgeStiffness.begin(), m_edgeStiffness.end(), edgeStiffnessDefault);
		fill(m_tetSpringStiffness.begin(), m_tetSpringStiffness.end(), tetSpringStiffnessDefault);
	}
	UDLog("解算器初始化结束");

	PreMalloc();
	CopyToGPU();

}


void Solver::InitSpringConstraint() {
	for (int i = 0; i < GetSurfaceVertNum(); i++) {
		/// <summary>
		/// 后期会挪地方
		/// </summary>
		m_edgeVertMass.push_back(0.005f);
		m_edgeVertFixed.push_back(0.0f);
	}
	//初始化求解过程中使用的对角元素
	m_springDiag.resize(GetSurfaceVertNum());
	for (int i = 0; i < GetSpringNum(); i++) {
		//获取弹簧顶点索引
		unsigned int index0 = m_edgeIndex[2 * i + 0];
		unsigned int index1 = m_edgeIndex[2 * i + 1];

		m_springDiag[index0] += m_edgeStiffness[i];
		m_springDiag[index1] += m_edgeStiffness[i];
	}
}

int Solver::GetTetVertNum(void)
{
	return m_tetVertPos.size() / 3;
}


int Solver::GetTetNum(void)
{
	return m_tetIndex.size() / 4;
}

int Solver::GetTetSpringNum(void)
{
	return m_tetSpringIndex.size() / 2;
}

int Solver::GetSurfaceTriNum(void)
{
	return m_triIndex.size() / 3;
}

int Solver::GetSurfaceVertNum(void)
{
	return m_edgeVertPos.size() / 3;
}

int Solver::GetOrgTriVertNum(void)
{
	return m_triVertPosOrg.size() / 3;
}
int Solver::GetSpringNum(void)
{
	return m_edgeIndex.size() / 2;
}

void Solver::InitVolumeConstraint()
{
	m_volumnSum = 0.0;
	// 预分配
	m_tetVertRestStiffness.resize(GetTetVertNum());
	m_tetVertfromTriStiffness.resize(GetTetVertNum());
	m_tetVertMass.resize(GetTetVertNum());
	m_tetVolumeDiag.resize(GetTetVertNum());
	m_tetInvD3x3.resize(GetTetNum() * 9);
	m_tetInvD3x4.resize(GetTetNum() * 12);
	printf("tetInvD3x4 size: %d\n", GetTetNum() * 12);
	for (int i = 0; i < GetTetNum(); i++) 
	{
		m_tetStiffness.push_back(volumnStiffness);
		//计算每个四面体初始化的shape矩阵的逆
		int vIndex0 = m_tetIndex[i * 4 + 0];
		int vIndex1 = m_tetIndex[i * 4 + 1];
		int vIndex2 = m_tetIndex[i * 4 + 2];
		int vIndex3 = m_tetIndex[i * 4 + 3];
		//先计算shape矩阵
		float D[9];
		D[0] = m_tetVertPos[vIndex1 * 3 + 0] - m_tetVertPos[vIndex0 * 3 + 0];
		D[1] = m_tetVertPos[vIndex2 * 3 + 0] - m_tetVertPos[vIndex0 * 3 + 0];
		D[2] = m_tetVertPos[vIndex3 * 3 + 0] - m_tetVertPos[vIndex0 * 3 + 0];
		D[3] = m_tetVertPos[vIndex1 * 3 + 1] - m_tetVertPos[vIndex0 * 3 + 1];
		D[4] = m_tetVertPos[vIndex2 * 3 + 1] - m_tetVertPos[vIndex0 * 3 + 1];
		D[5] = m_tetVertPos[vIndex3 * 3 + 1] - m_tetVertPos[vIndex0 * 3 + 1];
		D[6] = m_tetVertPos[vIndex1 * 3 + 2] - m_tetVertPos[vIndex0 * 3 + 2];
		D[7] = m_tetVertPos[vIndex2 * 3 + 2] - m_tetVertPos[vIndex0 * 3 + 2];
		D[8] = m_tetVertPos[vIndex3 * 3 + 2] - m_tetVertPos[vIndex0 * 3 + 2];
		//计算D的逆,顺便记录体积
		m_tetVolume.push_back(fabs(Matrix_Inverse_3(D, &m_tetInvD3x3[i * 9])) / 6.0);
		m_volumnSum += m_tetVolume[i];
		//cout << m_tetVolume[i] << endl;

		//计算质量
		m_tetVertMass[vIndex0] += m_tetVolume[i] / 4;
		m_tetVertMass[vIndex1] += m_tetVolume[i] / 4;
		m_tetVertMass[vIndex2] += m_tetVolume[i] / 4;
		m_tetVertMass[vIndex3] += m_tetVolume[i] / 4;

		float* inv_D = &m_tetInvD3x3[i * 9];
		//论文中的AC矩阵
		m_tetInvD3x4[i * 12 + 0] = -inv_D[0] - inv_D[3] - inv_D[6];
		m_tetInvD3x4[i * 12 + 1] = inv_D[0];
		m_tetInvD3x4[i * 12 + 2] = inv_D[3];
		m_tetInvD3x4[i * 12 + 3] = inv_D[6];
		

		m_tetInvD3x4[i * 12 + 4] = -inv_D[1] - inv_D[4] - inv_D[7];
		m_tetInvD3x4[i * 12 + 5] = inv_D[1];
		m_tetInvD3x4[i * 12 + 6] = inv_D[4];
		m_tetInvD3x4[i * 12 + 7] = inv_D[7];

		m_tetInvD3x4[i * 12 + 8] = -inv_D[2] - inv_D[5] - inv_D[8];
		m_tetInvD3x4[i * 12 + 9] = inv_D[2];
		m_tetInvD3x4[i * 12 + 10] = inv_D[5];
		m_tetInvD3x4[i * 12 + 11] = inv_D[8];

		if (vIndex0 == 0)
		{
			printf("3x4[%f %f %f %f]\n", m_tetInvD3x4[i * 12 + 0], m_tetInvD3x4[i * 12 + 1], m_tetInvD3x4[i * 12 + 2], m_tetInvD3x4[i * 12 + 3]);
			printf("3x4[%f %f %f %f]\n", m_tetInvD3x4[i * 12 + 4], m_tetInvD3x4[i * 12 + 5], m_tetInvD3x4[i * 12 + 6], m_tetInvD3x4[i * 12 + 7]);
			printf("3x4[%f %f %f %f]\n", m_tetInvD3x4[i * 12 + 8], m_tetInvD3x4[i * 12 + 9], m_tetInvD3x4[i * 12 + 10], m_tetInvD3x4[i * 12 + 11]);
		}
		//记录该点的对应的对角矩阵分量（用于雅各比迭代，因为只需要对角阵就可以实现）
		m_tetVolumeDiag[vIndex0] += m_tetInvD3x4[i * 12 + 0] * m_tetInvD3x4[i * 12 + 0] * m_tetVolume[i] * m_tetStiffness[i];//第i个四面体中的第一个点
		m_tetVolumeDiag[vIndex0] += m_tetInvD3x4[i * 12 + 4] * m_tetInvD3x4[i * 12 + 4] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex0] += m_tetInvD3x4[i * 12 + 8] * m_tetInvD3x4[i * 12 + 8] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex1] += m_tetInvD3x4[i * 12 + 1] * m_tetInvD3x4[i * 12 + 1] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex1] += m_tetInvD3x4[i * 12 + 5] * m_tetInvD3x4[i * 12 + 5] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex1] += m_tetInvD3x4[i * 12 + 9] * m_tetInvD3x4[i * 12 + 9] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex2] += m_tetInvD3x4[i * 12 + 2] * m_tetInvD3x4[i * 12 + 2] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex2] += m_tetInvD3x4[i * 12 + 6] * m_tetInvD3x4[i * 12 + 6] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex2] += m_tetInvD3x4[i * 12 + 10] * m_tetInvD3x4[i * 12 + 10] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex3] += m_tetInvD3x4[i * 12 + 3] * m_tetInvD3x4[i * 12 + 3] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex3] += m_tetInvD3x4[i * 12 + 7] * m_tetInvD3x4[i * 12 + 7] * m_tetVolume[i] * m_tetStiffness[i];
		m_tetVolumeDiag[vIndex3] += m_tetInvD3x4[i * 12 + 11] * m_tetInvD3x4[i * 12 + 11] * m_tetVolume[i] * m_tetStiffness[i];
	}
}

void Solver::SurfaceSubdivision()
{
	UDLog("对表面网格开始细分");
	////对现有的三角网格进行细分
	std::set<std::pair<unsigned int, unsigned int>>	edgeSet;
	std::set<std::pair<unsigned int, unsigned int>>	uvEdgeSet;
	int triNum = m_triIndexOrg.size() / 3;

	//////找出所有的边
	for (unsigned int i = 0; i < triNum; i++) {
		int tri0 = m_triIndexOrg[3 * i + 0];
		int tri1 = m_triIndexOrg[3 * i + 1];
		int tri2 = m_triIndexOrg[3 * i + 2];
		edgeSet.insert(std::make_pair(std::min(tri0, tri1), std::max(tri0, tri1)));
		edgeSet.insert(std::make_pair(std::min(tri0, tri2), std::max(tri0, tri2)));
		edgeSet.insert(std::make_pair(std::min(tri1, tri2), std::max(tri1, tri2)));
	}
	for (unsigned int i = 0; i < triNum; i++) {
		int tri0 = m_triUVIndexOrg[3 * i + 0];
		int tri1 = m_triUVIndexOrg[3 * i + 1];
		int tri2 = m_triUVIndexOrg[3 * i + 2];
		uvEdgeSet.insert(std::make_pair(std::min(tri0, tri1), std::max(tri0, tri1)));
		uvEdgeSet.insert(std::make_pair(std::min(tri0, tri2), std::max(tri0, tri2)));
		uvEdgeSet.insert(std::make_pair(std::min(tri1, tri2), std::max(tri1, tri2)));
	}
	///
	for (int i = 0; i < m_triVertPosOrg.size(); i++)
		m_edgeVertPos.push_back(m_triVertPosOrg[i]);
	for (int i = 0; i < m_triUVOrg.size(); i++)
		m_triUV.push_back(m_triUVOrg[i]);
	int vertNum = m_triVertPosOrg.size() / 3;
	int uvNum = m_triUVOrg.size() / 2;
	//////对边进行细分，一个边会变成2个
	std::map<std::pair<int, int>, int> newVertMap; // <边起点triVertIndex, 边终点triVertIndex> -> 生成的三角形顶点下标。 如果set中两个数值相同，表明是原始网格中存在的顶点
	std::map<int, std::pair<int, int>> newVertMapInv;
	for (auto iter = edgeSet.begin(); iter != edgeSet.end(); iter++) {
		int tri0 = iter->first*3;
		int tri1 = iter->second*3;
		float px_new = (m_triVertPosOrg[tri0] + m_triVertPosOrg[tri1]) * 0.5f;
		float py_new = (m_triVertPosOrg[tri0 + 1] + m_triVertPosOrg[tri1 + 1]) * 0.5f;
		float pz_new = (m_triVertPosOrg[tri0 + 2] + m_triVertPosOrg[tri1 + 2]) * 0.5f;
		m_edgeVertPos.push_back(px_new);
		m_edgeVertPos.push_back(py_new);
		m_edgeVertPos.push_back(pz_new);
		newVertMap[*iter] = vertNum;
		newVertMapInv[vertNum] = *iter;

		float lx = (m_triVertPosOrg[tri0 + 0] - m_triVertPosOrg[tri1 + 0]);
		float ly = (m_triVertPosOrg[tri0 + 1] - m_triVertPosOrg[tri1 + 1]);
		float lz = (m_triVertPosOrg[tri0 + 2] - m_triVertPosOrg[tri1 + 2]);
		float dist = sqrtf(lx * lx + ly * ly + lz * lz)*0.5f;

		m_edgeIndex.push_back(iter->first);
		m_edgeIndex.push_back(vertNum);
		m_edgeStiffness.push_back(edgeStiffnessDefault);
		m_edgeOrgLength.push_back(dist);

		m_edgeIndex.push_back(iter->second);
		m_edgeIndex.push_back(vertNum);
		m_edgeStiffness.push_back(edgeStiffnessDefault);
		m_edgeOrgLength.push_back(dist);
		vertNum++;
	}
	m_triVertNorm.resize(m_edgeVertPos.size());

	std::map<std::pair<int, int>, int>	 newUVMap;
	for (auto iter = uvEdgeSet.begin(); iter != uvEdgeSet.end(); iter++) {
		int tri0 = iter->first * 2;
		int tri1 = iter->second * 2;
		float uvx_new = (m_triUVOrg[tri0] + m_triUVOrg[tri1]) * 0.5f;
		float uvy_new = (m_triUVOrg[tri0 + 1] + m_triUVOrg[tri1 + 1]) * 0.5f;
		m_triUV.push_back(uvx_new);
		m_triUV.push_back(uvy_new);
		newUVMap[*iter] = uvNum;
		uvNum++;
	}
	///重新组织三角形，将一个三角形变成4个
	for (unsigned int i = 0; i < triNum; i++) {
		unsigned int tri0 = m_triIndexOrg[3 * i + 0];
		unsigned int tri1 = m_triIndexOrg[3 * i + 1];
		unsigned int tri2 = m_triIndexOrg[3 * i + 2];
		auto tri3 = newVertMap.find(std::make_pair(std::min(tri0, tri1), std::max(tri0, tri1)));
		auto tri4 = newVertMap.find(std::make_pair(std::min(tri1, tri2), std::max(tri1, tri2)));
		auto tri5 = newVertMap.find(std::make_pair(std::min(tri0, tri2), std::max(tri0, tri2)));

		unsigned int uv0 = m_triUVIndexOrg[3 * i + 0];
		unsigned int uv1 = m_triUVIndexOrg[3 * i + 1];
		unsigned int uv2 = m_triUVIndexOrg[3 * i + 2];
		auto uv3 = newUVMap.find(std::make_pair(std::min(uv0, uv1), std::max(uv0, uv1)));
		auto uv4 = newUVMap.find(std::make_pair(std::min(uv1, uv2), std::max(uv1, uv2)));
		auto uv5 = newUVMap.find(std::make_pair(std::min(uv0, uv2), std::max(uv0, uv2)));
		if (tri3 == newVertMap.end() || tri4 == newVertMap.end() || tri5 == newVertMap.end() ||
			uv3 == newUVMap.end() || uv4 == newUVMap.end() || uv5 == newUVMap.end()) {
			UDError("三角形索引错误");
			continue;
		}
		m_triIndex.push_back(tri0);
		m_triIndex.push_back(tri3->second);
		m_triIndex.push_back(tri5->second);
		m_triUVIndex.push_back(uv0);
		m_triUVIndex.push_back(uv3->second);
		m_triUVIndex.push_back(uv5->second);

		m_triIndex.push_back(tri1);
		m_triIndex.push_back(tri4->second);
		m_triIndex.push_back(tri3->second);
		m_triUVIndex.push_back(uv1);
		m_triUVIndex.push_back(uv4->second);
		m_triUVIndex.push_back(uv3->second);

		m_triIndex.push_back(tri2);
		m_triIndex.push_back(tri5->second);
		m_triIndex.push_back(tri4->second);
		m_triUVIndex.push_back(uv2);
		m_triUVIndex.push_back(uv5->second);
		m_triUVIndex.push_back(uv4->second);

		m_triIndex.push_back(tri3->second);
		m_triIndex.push_back(tri4->second);
		m_triIndex.push_back(tri5->second);
		m_triUVIndex.push_back(uv3->second);
		m_triUVIndex.push_back(uv4->second);
		m_triUVIndex.push_back(uv5->second);
	}


	//////找到表面三角形顶点与四面体顶点的对应关系
	int tetVertNum = GetTetVertNum();
	m_mapTriVertIndexToTetVertSetIndex.resize(GetSurfaceVertNum() * 2);
	m_mapTetVertIndexToTriVertIndex.resize(GetTetVertNum(),-1);	
	m_triVertfromTetStiffness.resize(GetSurfaceVertNum());

	int triVertNum = m_triVertPosOrg.size() / 3;
	std::vector<int> triVertOrg2tetVert(triVertNum);
	for (int i = 0; i < triVertNum; i++) {
		float trix = m_triVertPosOrg[i * 3];
		float triy = m_triVertPosOrg[i * 3+1];
		float triz = m_triVertPosOrg[i * 3+2];
		float mindist = FLT_MAX;
		int idx = -1;
		for (int j = 0; j < tetVertNum; j++) {
			float tetx = m_tetVertPos[j * 3];
			float tety = m_tetVertPos[j * 3 + 1];
			float tetz = m_tetVertPos[j * 3 + 2];
			float dist = sqrt((trix - tetx) * (trix - tetx) + (triy - tety) * (triy - tety) + (triz - tetz) * (triz - tetz));
			if (dist > mindist)
				continue;
			idx = j;
			mindist = dist;
		}
		if (mindist > 0.00015) {///3d max保存obj默认的精度是小数点后四位
			UDError("三角形顶点没有找到对应的四面体顶点");
			continue;
		}
		if (m_mapTetVertIndexToTriVertIndex[idx] > 0)
			UDError("两个三角形顶点对应到一个三角形上了");
		m_mapTetVertIndexToTriVertIndex[idx] = i;
		triVertOrg2tetVert[i] = idx;
		m_mapTriVertIndexToTetVertSetIndex[i * 2] = idx;
		m_mapTriVertIndexToTetVertSetIndex[i * 2+1] = idx;
	}
	triVertNum = GetSurfaceVertNum();
	for (int i = m_triVertPosOrg.size() / 3; i < triVertNum; i++) {
		auto twoEnd = newVertMapInv.find(i);
		if (twoEnd == newVertMapInv.end()) {
			UDError("twoEnd == newVertMapInv.end()");
			continue;
		}
		m_mapTriVertIndexToTetVertSetIndex[i * 2] = triVertOrg2tetVert[twoEnd->second.first];
		m_mapTriVertIndexToTetVertSetIndex[i * 2 + 1] = triVertOrg2tetVert[twoEnd->second.second];
	}

	m_triVertNorm.resize(m_edgeVertPos.size());
	m_triVertPos.resize(m_edgeVertPos.size());
	m_triVertPos.assign(m_edgeVertPos.begin(), m_edgeVertPos.end());
	m_triVertColor.resize(m_edgeVertPos.size() / 3 * 4, 0.9);
	UDLog("对表面网格细分完毕");

	/////////进行验证////////
	for (int i = 0; i < m_mapTetVertIndexToTriVertIndex.size(); i++) {
		int triIdx = m_mapTetVertIndexToTriVertIndex[i] * 3;
		if (triIdx < 0)
			continue;
		int tetIdx = i * 3;
		float tetx = m_tetVertPos[tetIdx];
		float tety = m_tetVertPos[tetIdx + 1];
		float tetz = m_tetVertPos[tetIdx + 2];
		float trix = m_edgeVertPos[triIdx];
		float triy = m_edgeVertPos[triIdx + 1];
		float triz = m_edgeVertPos[triIdx + 2];
		if (fabs(tetx - trix) > 0.0001 ||
			fabs(tety - triy) > 0.0001 ||
			fabs(tetz - triz) > 0.0001) {
			cout << "m_mapTetVertIndexToTriVertIndex error " << i << ":" << m_mapTetVertIndexToTriVertIndex[i] << endl;
		}
	}

	for (int i = 0; i < m_mapTriVertIndexToTetVertSetIndex.size()/2; i++) {
		int triIdx = i * 3;
		float trix = m_edgeVertPos[triIdx];
		float triy = m_edgeVertPos[triIdx + 1];
		float triz = m_edgeVertPos[triIdx + 2];

		int tetIdx0 = m_mapTriVertIndexToTetVertSetIndex[i*2] * 3;
		int tetIdx1 = m_mapTriVertIndexToTetVertSetIndex[i*2+1] * 3;

		float tetx = (m_tetVertPos[tetIdx0]+ m_tetVertPos[tetIdx1])*0.5f;
		float tety = (m_tetVertPos[tetIdx0+1] + m_tetVertPos[tetIdx1+1]) * 0.5f;
		float tetz = (m_tetVertPos[tetIdx0+2] + m_tetVertPos[tetIdx1+2]) * 0.5f;

		if (fabs(tetx - trix) > 0.0001 ||
			fabs(tety - triy) > 0.0001 ||
			fabs(tetz - triz) > 0.0001) {
			cout << "m_mapTriVertIndexToTetVertSetIndex " << i << ":" << m_mapTriVertIndexToTetVertSetIndex[i] << endl;
		}
	}
}
void Solver::SavePointCloud(vector<float> points, vector<int> indices, string filename)
{
	ofstream file(filename);
	for (int i = 0; i < indices.size(); i++)
	{
		int index = indices[i];
		file << "v " << points[index * 3 + 0] << " " << points[index * 3 + 1] << " " << points[index * 3 + 2] << endl;
	}
	file.close();
}
void Solver::SavePointCloud(vector<float> points, string filename)
{
	ofstream file(filename);
	for (int i = 0; i < points.size()/3; i++)
	{
		file << "v " << points[i * 3 + 0] << " " << points[i * 3 + 1] << " " << points[i * 3 + 2] << endl;
	}
	file.close();
}

void Solver::GenerateTetSpring()
{
	int tetNum = GetTetNum();
	int tetVertNum = GetTetVertNum();
	std::vector<int> sharedTetNum;
	sharedTetNum.resize(tetVertNum);
	fill(sharedTetNum.begin(), sharedTetNum.end(), 0);

	std::vector<int> relatedTetIdx;
	std::vector<int> relatedTetInfo;
	std::vector<std::vector<int>> relatedTetBuffer;
	for (int i = 0; i < tetVertNum; i++)
	{
		std::vector<int> t;
		relatedTetBuffer.push_back(t);
	}

	std::set<std::pair<unsigned int, unsigned int>>	edgeSet;
	for (int i = 0; i < GetTetNum(); i++)
	{
		// 一个四面体6个弹簧
		int tetVertIdx0 = m_tetIndex[i * 4 + 0];
		int tetVertIdx1 = m_tetIndex[i * 4 + 1];
		int tetVertIdx2 = m_tetIndex[i * 4 + 2];
		int tetVertIdx3 = m_tetIndex[i * 4 + 3];

		relatedTetBuffer[tetVertIdx0].push_back(i);
		relatedTetBuffer[tetVertIdx1].push_back(i);
		relatedTetBuffer[tetVertIdx2].push_back(i);
		relatedTetBuffer[tetVertIdx3].push_back(i);

		sharedTetNum[tetVertIdx0]++;
		sharedTetNum[tetVertIdx1]++;
		sharedTetNum[tetVertIdx2]++;
		sharedTetNum[tetVertIdx3]++;
		edgeSet.insert(std::make_pair(std::min(tetVertIdx0, tetVertIdx1), std::max(tetVertIdx0, tetVertIdx1)));
		edgeSet.insert(std::make_pair(std::min(tetVertIdx0, tetVertIdx2), std::max(tetVertIdx0, tetVertIdx2)));
		edgeSet.insert(std::make_pair(std::min(tetVertIdx0, tetVertIdx3), std::max(tetVertIdx0, tetVertIdx3)));
		edgeSet.insert(std::make_pair(std::min(tetVertIdx1, tetVertIdx2), std::max(tetVertIdx1, tetVertIdx2)));
		edgeSet.insert(std::make_pair(std::min(tetVertIdx1, tetVertIdx3), std::max(tetVertIdx1, tetVertIdx3)));
		edgeSet.insert(std::make_pair(std::min(tetVertIdx2, tetVertIdx3), std::max(tetVertIdx2, tetVertIdx3)));
	}

	int startIdx = 0;
	for (int i = 0; i < tetVertNum; i++)
	{
		auto tetVertRelatedTetIdx = relatedTetBuffer[i];
		relatedTetInfo.push_back(startIdx);
		relatedTetInfo.push_back(tetVertRelatedTetIdx.size());
		for (int idx : tetVertRelatedTetIdx)
		{
			relatedTetIdx.push_back(idx);
		}
		startIdx += tetVertRelatedTetIdx.size();
	}
	m_tetVertRelatedTetIdx = relatedTetIdx;
	m_tetVertRelatedTetInfo = relatedTetInfo;
	
	ofstream sharedTetNumFile("validate/sharedTetNum.txt");
	for (int i = 0; i < sharedTetNum.size(); i++)
	{
		sharedTetNumFile << sharedTetNum[i] << endl;
	}
	sharedTetNumFile.close();
	ofstream ofile("validate/tetSpringIdx.txt");
	for (auto iter = edgeSet.begin(); iter != edgeSet.end(); ++iter)
	{
		int idx0 = iter->first;
		int idx1 = iter->second;
		m_tetSpringIndex.push_back(idx0);
		m_tetSpringIndex.push_back(idx1);
		ofile << idx0 << " " << idx1 << endl;
		float pos0x = m_tetVertPos[idx0 * 3 + 0];
		float pos0y = m_tetVertPos[idx0 * 3 + 1];
		float pos0z = m_tetVertPos[idx0 * 3 + 2];
		float pos1x = m_tetVertPos[idx1 * 3 + 0];
		float pos1y = m_tetVertPos[idx1 * 3 + 1];
		float pos1z = m_tetVertPos[idx1 * 3 + 2];
		float dx = pos0x - pos1x;
		float dy = pos0y - pos1y;
		float dz = pos0z - pos1z;
		float orgLen = sqrt(dx * dx + dy * dy + dz * dz);
		//printf("idx0=%d [%f %f %f] idx1=%d [%f %f %f] d:%f\n",
		//	idx0, pos0x, pos0y, pos0z,
		//	idx1, pos1x, pos1y, pos1z,
		//	orgLen);
		m_tetSpringOrgLength.push_back(orgLen);
		m_tetSpringStiffness.push_back(tetSpringStiffnessDefault);
	}
	ofile.close();
}
void Solver::GenerateTetVertDDirWithTri()
{
	auto comp = [&](dist_idx a, dist_idx b) -> bool {return a.first < b.first; };
	float tetVert[3];
	float triVert[3];
	float surfaceTetVert[3];

	int tetVertNum = GetTetVertNum();
	int triVertNum = m_triVertPosOrg.size() / 3;

	m_tetVertBindingTetVertIdx.resize(tetVertNum * 3);
	m_TetVertIndexBindingWeight.resize(tetVertNum * 3);
	// 计算在表面的四面体顶点下标
	vector<int> tet2tri;//存在未对应的，设置初值为-1
	vector<int> tri2tet;//满的
	tet2tri.resize(tetVertNum);
	fill(tet2tri.begin(), tet2tri.end(), -1);
	tri2tet.resize(triVertNum);
	for (int i = 0; i < tetVertNum; i++)
	{
		tetVert[0] = m_tetVertPos[i * 3 + 0];
		tetVert[1] = m_tetVertPos[i * 3 + 1];
		tetVert[2] = m_tetVertPos[i * 3 + 2];
		for (int j = 0; j < triVertNum; j++)
		{
			triVert[0] = m_triVertPosOrg[j * 3 + 0];
			triVert[1] = m_triVertPosOrg[j * 3 + 1];
			triVert[2] = m_triVertPosOrg[j * 3 + 2];
			float d = l2_dis(triVert, tetVert);
			if (d < 1e-7)
			{
				// tet point on surface
				m_onSurfaceTetVertIndices.push_back(i);
				tet2tri[i] = j;
				tri2tet[j] = i;
				printf("tetPoint:%f %f %f, triPoint %f %f %f\n",
					tetVert[0], tetVert[1], tetVert[2],
					triVert[0], triVert[1], triVert[2]);
				break;
			}
		}
	}
	printf("onsurface point num:%d\n", m_onSurfaceTetVertIndices.size());
	// 表面三角网格到表面四面体顶点之间的对应关系
	int triNum = m_triIndexOrg.size() / 3;
	float t0[3], t1[3], t2[3];
	for (int tetVertIdx = 0; tetVertIdx < tetVertNum; tetVertIdx++)
	{
		tetVert[0] = m_tetVertPos[tetVertIdx * 3 + 0];
		tetVert[1] = m_tetVertPos[tetVertIdx * 3 + 1];
		tetVert[2] = m_tetVertPos[tetVertIdx * 3 + 2];
		if (tet2tri[tetVertIdx] == -1) // 四面体顶点在内部
		{
			// 遍历表面三角形，找到最近的表面三角形
			float min_dis = FLT_MAX;
			int nearestTriIdx = 0;
			for (int i = 0; i < triNum; i++)
			{
				int triVertIdx0 = m_triIndexOrg[i * 3 + 0];
				int triVertIdx1 = m_triIndexOrg[i * 3 + 1];
				int triVertIdx2 = m_triIndexOrg[i * 3 + 2];
				int ti0 = tri2tet[triVertIdx0];
				int ti1 = tri2tet[triVertIdx1];
				int ti2 = tri2tet[triVertIdx2];
				t0[0] = m_tetVertPos[ti0 * 3 + 0];
				t0[1] = m_tetVertPos[ti0 * 3 + 1];
				t0[2] = m_tetVertPos[ti0 * 3 + 2];
				t1[0] = m_tetVertPos[ti1 * 3 + 0];
				t1[1] = m_tetVertPos[ti1 * 3 + 1];
				t1[2] = m_tetVertPos[ti1 * 3 + 2];
				t2[0] = m_tetVertPos[ti2 * 3 + 0];
				t2[1] = m_tetVertPos[ti2 * 3 + 1];
				t2[2] = m_tetVertPos[ti2 * 3 + 2];
				float d0 = l2_dis(t0, tetVert);
				float d1 = l2_dis(t1, tetVert);
				float d2 = l2_dis(t2, tetVert);
				float d = d0 + d1 + d2;
				if (d < min_dis)
				{
					min_dis = d;
					nearestTriIdx = i;
				}
			}
			int triVertIdx0 = m_triIndexOrg[nearestTriIdx * 3 + 0];
			int triVertIdx1 = m_triIndexOrg[nearestTriIdx * 3 + 1];
			int triVertIdx2 = m_triIndexOrg[nearestTriIdx * 3 + 2];
			m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 0] = tri2tet[triVertIdx0];
			m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 1] = tri2tet[triVertIdx1];
			m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 2] = tri2tet[triVertIdx2];
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 0] = 1.0f / 3.0f;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 1] = 1.0f / 3.0f;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 2] = 1.0f / 3.0f;
		}
		else
		{
			m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 0] = tetVertIdx;
			m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 1] = tetVertIdx;
			m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 2] = tetVertIdx;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 0] = 0;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 1] = 0;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 2] = 0;
		}
	}
}
void Solver::GenerateTetVertDirectDir()
{// 计算四面体顶点对应的
	auto comp = [&](dist_idx a, dist_idx b) -> bool {return a.first < b.first; };
	float tetVert[3];
	float triVert[3];
	float surfaceTetVert[3];
	
	int tetVertNum = GetTetVertNum();
	int triVertNum = m_triVertPosOrg.size() / 3;

	m_tetVertBindingTetVertIdx.resize(tetVertNum * 3);
	m_TetVertIndexBindingWeight.resize(tetVertNum * 3);
	ofstream output("ddirFile.txt");
	// 计算在表面的四面体顶点下标
	for (int i = 0; i < tetVertNum; i++)
	{
		tetVert[0] = m_tetVertPos[i * 3 + 0];
		tetVert[1] = m_tetVertPos[i * 3 + 1];
		tetVert[2] = m_tetVertPos[i * 3 + 2];
		for (int j = 0; j < triVertNum; j++)
		{
			triVert[0] = m_triVertPosOrg[j * 3 + 0];
			triVert[1] = m_triVertPosOrg[j * 3 + 1];
			triVert[2] = m_triVertPosOrg[j * 3 + 2];
			float d = l2_dis(triVert, tetVert);
			if (d < 1e-7)
			{
				// tet point on surface
				m_onSurfaceTetVertIndices.push_back(i);
				printf("pre computation: tetPoint:%f %f %f, triPoint %f %f %f\n",
					tetVert[0], tetVert[1], tetVert[2],
					triVert[0], triVert[1], triVert[2]);
				break;
			}
		}
	}
	printf("onsurface point num:%d\n", m_onSurfaceTetVertIndices.size());
	for (int tetVertIdx = 0; tetVertIdx < tetVertNum; tetVertIdx++)
	{
		tetVert[0] = m_tetVertPos[tetVertIdx * 3 + 0];
		tetVert[1] = m_tetVertPos[tetVertIdx * 3 + 1];
		tetVert[2] = m_tetVertPos[tetVertIdx * 3 + 2];
		
		vector<dist_idx> buf;

		for (int surfaceIdx = 0; surfaceIdx < m_onSurfaceTetVertIndices.size(); surfaceIdx++)
		{
			int idx = m_onSurfaceTetVertIndices[surfaceIdx];
			surfaceTetVert[0] = m_tetVertPos[idx * 3 + 0];
			surfaceTetVert[1] = m_tetVertPos[idx * 3 + 1];
			surfaceTetVert[2] = m_tetVertPos[idx * 3 + 2];
			float dist = l2_dis(tetVert, surfaceTetVert);
			dist_idx t(dist, idx);
			buf.push_back(t);
		}
		sort(buf.begin(), buf.end(), comp);

		// 最近的三个表面四面体顶点的下标 TV->TetVertex
		int bindingTVIdx0 = buf[0].second;
		int bindingTVIdx1 = buf[1].second;
		int bindingTVIdx2 = buf[2].second;
		m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 0] = bindingTVIdx0;
		m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 1] = bindingTVIdx1;
		m_tetVertBindingTetVertIdx[tetVertIdx * 3 + 2] = bindingTVIdx2;

		if (buf[0].first < 1e-7)// 当前四面体顶点在表面
		{
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 0] = 0;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 1] = 0;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 2] = 0;
			output << "on surface " << tetVert[0] << " " << tetVert[1] << " " << tetVert[2] << endl;
		}
		else // 当前四面体顶点在内部
		{
			// TODO: 添加计算重心坐标的代码
			float coord0 = 1.0f / 3.0f;
			float coord1 = 1.0f / 3.0f;
			float coord2 = 1.0f / 3.0f;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 0] = coord0;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 1] = coord1;
			m_TetVertIndexBindingWeight[tetVertIdx * 3 + 2] = coord2;
			output << "inner point " << tetVert[0] << " " << tetVert[1] << " " << tetVert[2] << endl;
			auto found0 = find(m_onSurfaceTetVertIndices.begin(), m_onSurfaceTetVertIndices.end(), bindingTVIdx0);
			if (found0 == m_onSurfaceTetVertIndices.end())
			{
				output << "binding point 0 not on surface" << endl;
			}
			auto found1 = find(m_onSurfaceTetVertIndices.begin(), m_onSurfaceTetVertIndices.end(), bindingTVIdx1);
			if (found1 == m_onSurfaceTetVertIndices.end())
			{
				output << "binding point 1 not on surface" << endl;
			}
			auto found2 = find(m_onSurfaceTetVertIndices.begin(), m_onSurfaceTetVertIndices.end(), bindingTVIdx2);
			if (found2 == m_onSurfaceTetVertIndices.end())
			{
				output << "binding point 2 not on surface" << endl;
			}
		}
		
		int milestone = tetVertNum / 10;
		if (tetVertIdx % milestone == 0)
		{
			printf("tet vert direct dir processed %f %%\n", 100*(float)tetVertIdx / (float)tetVertNum);
		}
	}

	
}
void Solver::PreMalloc()
{
	int tetVertNum = GetTetVertNum();
	int tetNum = GetTetNum();
	int tetSpringNum = GetTetSpringNum();
	printf("PreMalloc info: tetVertNum:%d tetNum:%d tetSpringNum:%d\n", tetVertNum, tetNum, tetSpringNum);
	// 四面体部分

	m_tetVertCollisionForce.resize(tetVertNum * 3);
	m_tetVertCollisionForceLen.resize(tetVertNum);
	m_tetVertVolumnForce.resize(tetVertNum * 3); 
	m_tetVertVolumnForceLen.resize(tetVertNum);
	m_tetVertDirectDirection.resize(tetVertNum * 3);

	cudaMalloc((void**)&tetIndex_d, tetNum * 4 * sizeof(int));
	cudaMalloc((void**)&tetStiffness_d, tetNum * sizeof(float));
	cudaMalloc((void**)&tetVertPos_d, tetVertNum*3*sizeof(float));
	cudaMalloc((void**)&tetVertRestPos_d, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetActive_d, tetNum * sizeof(bool));
	cudaMemset(tetActive_d, true, tetNum * sizeof(bool));
	cudaMalloc((void**)&tetVertMass_d, tetVertNum*sizeof(float));

	cudaMalloc((void**)&tetInvD3x3_d, tetNum*9*sizeof(float));
	cudaMalloc((void**)&tetInvD3x4_d, tetNum*12*sizeof(float));
	cudaMalloc((void**)&tetVolume_d, tetNum*sizeof(float));
	cudaMalloc((void**)&tetVolumeDiag_d, tetVertNum * sizeof(float));
	cudaMalloc((void**)&tetVertRestStiffness_d, tetVertNum * sizeof(float));
	cudaMalloc((void**)&tetVert2TriVertMapping_d, tetVertNum * sizeof(int));
	cudaMalloc((void**)&tetVertfromTriStiffness_d, tetVertNum * sizeof(float));
	

	cudaMalloc((void**)&tetVertVelocity_d, tetVertNum*3*sizeof(float));
	cudaMemset(tetVertVelocity_d, 0.0f, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetVertExternForce_d, tetVertNum * 3 * sizeof(float));
	cudaMemset(tetVertExternForce_d, 0.0f, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetVertForce_d, tetVertNum * 3 * sizeof(float));
	cudaMemset(tetVertForce_d, 0.0f, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetVertForceLen_d, tetVertNum * sizeof(float));
	cudaMemset(tetVertForceLen_d, 0.0f, tetVertNum * sizeof(float));
	cudaMalloc((void**)&tetVertFixed_d, tetVertNum * sizeof(float));
	cudaMemset(tetVertFixed_d, 0.0f, tetVertNum * sizeof(float));
	cudaMalloc((void**)&tetVertPos_old_d, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetVertPos_prev_d, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetVertPos_last_d, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetVertPos_next_d, tetVertNum * 3 * sizeof(float));

	cudaMalloc((void**)&tetSpringIndex_d, tetSpringNum * 2 * sizeof(int));
	cudaMalloc((void**)&tetSpringOrgLen_d, tetSpringNum * sizeof(float));
	cudaMalloc((void**)&tetSpringStiffness_d, tetSpringNum * sizeof(float));

	cudaMalloc((void**)&tetVertRelatedTetIdx_d, tetNum * 4 * sizeof(int));
	cudaMalloc((void**)&tetVertRelatedTetInfo_d, tetVertNum * 2 * sizeof(int));

	// 四面体碰撞
	
	cudaMalloc((void**)&tetVertisCollide_d, tetVertNum * sizeof(char));
	cudaMalloc((void**)&tetIsCollide_d, tetVertNum * sizeof(unsigned int));
	cudaMemset(tetIsCollide_d, 0, tetVertNum * sizeof(unsigned int));
	cudaMalloc((void**)&tetVertCollisionForce_d, tetVertNum * 3 * sizeof(float));
	cudaMemset(tetVertCollisionForce_d, 0.0f, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetVertCollisionForceLen_d, tetVertNum * sizeof(float));
	cudaMemset(tetVertCollisionForceLen_d, 0.0f, tetVertNum * sizeof(float));
	cudaMalloc((void**)&tetVertCollisionDiag_d, tetVertNum * 3 * sizeof(float));
	cudaMemset(tetVertCollisionDiag_d, 0.0f, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetVertInsertionDepth_d, tetVertNum * sizeof(float));

	// 四面体顶点指导向量
	cudaMalloc((void**)&tetVertNonPenetrationDir_d, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&tetShellIdx_d, tetVertNum * sizeof(int));
	cudaMalloc((void**)&tetVertBindingTetVertIndices_d, tetVertNum * 3 * sizeof(int));
	cudaMalloc((void**)&tetVertBindingTetVertWeight_d, tetVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&onSurfaceTetVertIndices_d, m_onSurfaceTetVertIndices.size() * sizeof(int));

	// 表面布料网格部分
	int meshVertNum = GetSurfaceVertNum();
	int springNum = GetSpringNum();
	int triNum = GetSurfaceTriNum();
	
	// 表面三角网格顶点信息
	cudaMalloc((void**)&triVertPos_d, meshVertNum*3*sizeof(float));
	cudaMalloc((void**)&triVertRestPos_d,     meshVertNum*3*sizeof(float));
	cudaMalloc((void**)&triVertPos_old_d,     meshVertNum*3*sizeof(float));
	cudaMalloc((void**)&triVertPos_prev_d,    meshVertNum*3*sizeof(float));
	cudaMalloc((void**)&triVertPos_next_d,    meshVertNum*3*sizeof(float));
	cudaMalloc((void**)&triVertVelocity_d,    meshVertNum*3*sizeof(float));
	cudaMalloc((void**)&triVertExternForce_d, meshVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&triVertMass_d,     meshVertNum*sizeof(float));
	cudaMalloc((void**)&triVertFixed_d,    meshVertNum*sizeof(float));
	cudaMemset(triVertFixed_d, 0.0f, meshVertNum * sizeof(float));
	cudaMalloc((void**)&triVertNorm_d,     meshVertNum*3*sizeof(float));
	cudaMalloc((void**)&triVertNormAccu_d, meshVertNum*sizeof(float));
	cudaMalloc((void**)&triVertForce_d, meshVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&triVertRestStiffness_d, meshVertNum * sizeof(float));
	cudaMalloc((void**)&triVertfromTetStiffness_d, meshVertNum * sizeof(float));
	cudaMalloc((void**)&triVert2TetVertMapping_d,  meshVertNum *2* sizeof(int));
	cudaMalloc((void**)&triVertNonPenetrationDir_d, meshVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&triVertProjectedPos_d, meshVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&triShellIdx_d, meshVertNum * sizeof(int));
	// 表面三角网格弹簧数据
	triEdgeNum_d = springNum;
	cudaMalloc((void**)&triEdgeIndex_d, springNum * 2 * sizeof(unsigned int));
	cudaMalloc((void**)&triEdgeOrgLength_d, springNum * sizeof(float));
	cudaMalloc((void**)&triEdgeStiffness_d, springNum * sizeof(float));
	cudaMalloc((void**)&triEdgeDiag_d, meshVertNum * 3 * sizeof(float));

	// 表面三角网格碰撞信息
	cudaMalloc((void**)&triVertisCollide_d, meshVertNum * sizeof(unsigned char));
	cudaMemset(triVertisCollide_d, 0, meshVertNum * sizeof(unsigned char));
	cudaMalloc((void**)&triVertCollisionForce_d, meshVertNum * 3 * sizeof(float));
	cudaMemset(triVertCollisionForce_d, 0.0f, meshVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&triVertCollisionDiag_d, meshVertNum * 3 * sizeof(float));
	cudaMemset(triVertCollisionDiag_d, 0.0f, meshVertNum * 3 * sizeof(float));
	cudaMalloc((void**)&triVertInsertionDepth_d, meshVertNum * sizeof(float));

	///圆柱体信息
	cylinderNum_d = m_cylinderNum;
	cudaMalloc((void**)&cylinderShift_d, m_cylinderNum*3 * sizeof(float));
	cudaMalloc((void**)&cylinderLastPos_d, m_cylinderNum * 3 * sizeof(float));
	cudaMalloc((void**)&cylinderPos_d, m_cylinderNum * 3 * sizeof(float));
	cudaMalloc((void**)&cylinderDirZ_d, m_cylinderNum * 3 * sizeof(float));
	cudaMalloc((void**)&cylinderV_d, m_cylinderNum * 3 * sizeof(float));
	cudaMalloc((void**)&toolCollideFlag_d, m_cylinderNum * sizeof(unsigned int));


	// 表面三角网格 三角形对应的顶点下标
	cudaMalloc((void**)&triIndex_d, triNum * 3 * sizeof(unsigned int));

	// 小球位置信息
	cudaMalloc((void**)&toolPositionAndDirection_d, 6 * sizeof(float));
	cudaMalloc((void**)&toolPosePrev_d, 6 * sizeof(float));
	cudaMalloc((void**)&radius_d, sizeof(float));
	printCudaError("PreMalloc");
	cudaMalloc((void**)&hapticCollisionNum_d, sizeof(int));
	cudaMalloc((void**)&toolContactDeltaPos_triVert_d, 3 * sizeof(float) * meshVertNum);
	cudaMalloc((void**)&totalFC_d, 3 * sizeof(float));
	cudaMalloc((void**)&totalPartial_FC_X_d, 9 * sizeof(float));
	cudaMalloc((void**)&totalPartial_FC_Omega_d, 9 * sizeof(float));
	cudaMalloc((void**)&totalTC_d, 3 * sizeof(float));
	cudaMalloc((void**)&totalPartial_TC_X_d, 9 * sizeof(float));
	cudaMalloc((void**)&totalPartial_TC_Omega_d, 9 * sizeof(float));


}

void Solver::GetToolDir(float* dir)
{
	dir[0] = m_toolTrans[8];
	dir[1] = m_toolTrans[9];
	dir[2] = m_toolTrans[10];
	//dir[0] = m_toolTrans[2];
	//dir[1] = m_toolTrans[6];
	//dir[2] = m_toolTrans[10];
}

void Solver::GetToolPos(float* pos)
{
	pos[0] = m_toolTrans[12];
	pos[1] = m_toolTrans[13];
	pos[2] = m_toolTrans[14];
}

void Solver::GetToolTip(float* pos)
{
	float ball_pos[3], tip_dir[3];
	GetToolPos(ball_pos);
	GetToolDir(tip_dir);
	pos[0] = ball_pos[0] + tip_dir[0] * toolLength;
	pos[1] = ball_pos[1] + tip_dir[1] * toolLength;
	pos[2] = ball_pos[2] + tip_dir[2] * toolLength;
}

void Solver::GetTool6DOFPose(float* pose_6dof)
{
	GetToolPos(pose_6dof);
	GetToolDir(pose_6dof + 3);
}

void Solver::OutputTrans()
{
	printf("m_toolTrans:");
	for (int i = 0; i < 16; i++)
	{
		printf("%f ", m_toolTrans[i]);
	}
	printf("\n");
}