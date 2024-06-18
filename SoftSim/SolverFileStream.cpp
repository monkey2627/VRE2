//////////所有和初始化相关的加到这个函数中
//////////包括文件读取，系统矩阵初始化
#include <Windows.h>
#include "Solver.h"
#include "bridge.h"
using namespace std;

bool WriteStream(FILE* fid, const char* buffer, uint32_t length, string msg) {

	int ret = fwrite(buffer, sizeof(char), length, fid);
	if (ret != length) {
		UDError("Write buffer error: " + to_string(ret- length) + " " + msg);
		return false;
	}
	return true;
}

bool ReadStream(FILE* fid, char* buffer, uint32_t length, string msg) {

	int ret = fread(buffer, sizeof(char), length, fid);
	if (ret != length) {
		UDError("Read buffer error:  " + to_string(ret- length) + " " + msg);
		return false;
	}
	return true;

}

bool WriteNum(FILE* fid, uint32_t num, string msg) {
	int ret = fwrite(&num, sizeof(uint32_t), 1, fid);
	if (num == 0)
		UDError("Write num zero: " + msg);

	if (ret != 1) {
		UDError("Write num error: " + to_string(ret) + " " + msg);
		return false;
	}
	return true;
}

uint32_t ReadNum(FILE* fid, string msg) {
	uint32_t num;
	int ret = fread(&num, sizeof(uint32_t), 1, fid);
	if (ret != 1 || num <= 0) {
		UDError("read num error: " + to_string(num) + " " + msg);
		return -100;
	}
	return num;
}

void Solver::FileStream(FILE* fid) {
	int ret, num;


#pragma region
	int bufferi[1024];

	if (FILEMODE::R == m_fileMode) {
		ret = fread(&bufferi[0], sizeof(int), 1024, fid);
		if (ret != 1024) {
			fclose(fid);
			UDError("Read  Num error!" + to_string(ret));
			return;
		}
		memcpy(&m_volumnSum, &bufferi[0], sizeof(int));

	}
	else {
		memcpy(&bufferi[0], &m_volumnSum, sizeof(int));
		ret = fwrite(&bufferi[0], sizeof(int), 1024, fid);
		if (ret != 1024) {
			fclose(fid);
			UDError("Write Num error!" + to_string(ret));
			return;
		}

	}
#pragma endregion  各种变量


	//std::vector<int> m_tetIndex;//四面体索引	
	FileStream(fid, m_tetIndex, "m_tetIndex");
	//std::vector<float> m_tetStiffness;//四面体的软硬
	FileStream(fid, m_tetStiffness, "m_tetStiffness");
	//std::vector<float> m_tetVertPos;//顶点位置	
	FileStream(fid, m_tetVertPos, "m_tetVertPos");
	//std::vector<float> m_tetVertFixed;//是否固定	
	FileStream(fid, m_tetVertFixed, "m_tetVertFixed");
	//std::vector<float> m_tetVertMass;//顶点质量
	FileStream(fid, m_tetVertMass, "m_tetVertMass");
	//std::vector<float> m_tetVertRestStiffness;//每个点restpos的约束
	FileStream(fid, m_tetVertRestStiffness, "m_tetVertRestStiffness");
	//std::vector<float> m_tetVertfromTriStiffness;////三角形顶点对四面体的约束刚度
	FileStream(fid, m_tetVertfromTriStiffness, "m_tetVertfromTriStiffness");

	FileStream(fid, m_onSurfaceTetVertIndices, "m_onSurfaceTetVertIndices");
	

	///原始表面三角形
	//std::vector<int> m_triIndexOrg;// 原始三角形索引
	FileStream(fid, m_triIndexOrg, "m_triIndexOrg");
	//std::vector<float> m_triVertPosOrg; // 原始三角网格表面顶点坐标
	FileStream(fid, m_triVertPosOrg, "m_triVertPosOrg");
	//std::vector<int> m_triUVIndexOrg;// 细分三角形索引
	FileStream(fid, m_triUVIndexOrg, "m_triUVIndexOrg");
	//std::vector<float> m_triUVOrg; // 细分三角网格表面顶点坐标
	FileStream(fid, m_triUVOrg, "m_triUVOrg");


	//std::vector<unsigned int> m_triIndex;// 细分三角形索引
	FileStream(fid, m_triIndex, "m_triIndex");
	//std::vector<float> m_triVertPos; // 细分三角网格表面顶点坐标
	FileStream(fid, m_triVertPos, "m_triVertPos");
	//std::vector<float> m_triVertColor; // 细分三角网格表面顶点颜色//可视化用
	FileStream(fid, m_triVertColor, "m_triVertColor");
	//std::vector<float> m_triVertNorm;// 细分三角网格表面顶点法向量
	FileStream(fid, m_triVertNorm, "m_triVertNorm");
	//std::vector<unsigned int> m_triUVIndex;// 细分三角形索引
	FileStream(fid, m_triUVIndex, "m_triUVIndex");
	//std::vector<float> m_triUV; // 细分三角网格表面顶点坐标
	FileStream(fid, m_triUV, "m_triUV");

	
	//std::vector<char> m_tetActive;//是否处于有效状态
	FileStream(fid, m_tetActive, "m_tetActive");
	//std::vector<float> m_tetInvD3x3;//四面体矩阵的逆//用于计算变形梯度
	FileStream(fid, m_tetInvD3x3, "m_tetInvD3x3");
	//std::vector<float> m_tetInvD3x4;//用于计算对角阵
	FileStream(fid, m_tetInvD3x4, "m_tetInvD3x4");
	//std::vector<float> m_tetVolume;//四面体体积，长度：tetNum
	FileStream(fid, m_tetVolume, "m_tetVolume");
	//std::vector<float> m_tetVolumeDiag;//线性系统的对角矩阵的ACT*AC部分, 长度：tetNum
	FileStream(fid, m_tetVolumeDiag, "m_tetVolumeDiag");
	//std::vector<int> m_mapTetVertIndexToTriVertIndex;//四面体对应表面顶点，一对一
	FileStream(fid, m_mapTetVertIndexToTriVertIndex, "m_mapTetVertIndexToTriVertIndex");
	//std::vector<int> m_mapTriVertIndexToTetVertSetIndex;//表面顶点对四面体顶点，一对二
	FileStream(fid, m_mapTriVertIndexToTetVertSetIndex, "m_mapTriVertIndexToTetVertSetIndex");

	//std::vector<int> m_tetVertBindingTetVertIdx; // 四面体顶点最近的三个表面网格顶点下标。
	FileStream(fid, m_tetVertBindingTetVertIdx, "m_tetVertBindingTetVertIdx");
	//std::vector<float> m_TetVertIndexBindingWeight;
	FileStream(fid, m_TetVertIndexBindingWeight, "m_TetVertIndexBindingWeight");

	//std::vector<int> m_tetSpringIndex; // 四面体网格弹簧数组，
	//std::vector<float> m_tetSpringOrgLength; // 四面体弹簧原长
	//std::vector<float> m_tetSpringStiffness;
	FileStream(fid, m_tetSpringIndex, "m_tetSpringIndex");
	FileStream(fid, m_tetSpringOrgLength, "m_tetSpringOrgLength");
	FileStream(fid, m_tetSpringStiffness, "m_tetSpringStiffness");

	FileStream(fid, m_tetVertRelatedTetIdx, "m_tetVertRelatedTetIdx");
	FileStream(fid, m_tetVertRelatedTetInfo, "m_tetVertRelatedTetInfo");

	////std::vector<float> m_tetDirectDir; // 初始化四面体顶点指导向量
	//FileStream(fid, m_tetDirectDir, "m_tetDirectDir");

	//std::vector<unsigned int> m_edgeIndex;// 弹簧索引
	FileStream(fid, m_edgeIndex, "m_edgeIndex");
	//std::vector<float> m_edgeStiffness;// 弹簧刚度
	FileStream(fid, m_edgeStiffness, "m_edgeStiffness");
	//std::vector<float> m_edgeVertPos;//弹簧顶点
	FileStream(fid, m_edgeVertPos, "m_edgeVertPos");
	//std::vector<float> m_edgeVertFixed;
	FileStream(fid, m_edgeVertFixed, "m_edgeVertFixed");
	//std::vector<float> m_edgeVertMass;
	FileStream(fid, m_edgeVertMass, "m_edgeVertMass");
	//std::vector<float> m_edgeOrgLength; // 弹簧原长
	FileStream(fid, m_edgeOrgLength, "m_edgeOrgLength");
	//std::vector<float> m_springDiag;
	FileStream(fid, m_springDiag, "m_springDiag");
	//std::vector<float> m_triVertfromTetStiffness;
	FileStream(fid, m_triVertfromTetStiffness, "m_triVertfromTetStiffness");

	//FileStream(fid, m_partCenter, "m_partCenter");
	//FileStream(fid, m_mapTriVertIndexToPartIndex, "m_mapTriVertIndexToPartIndex");
	//FileStream(fid, m_mapTetVertIndexToPartIndex, "m_mapTetVertIndexToPartIndex");
	//FileStream(fid, m_mapTetIndexToPartIndex, "m_mapTetIndexToPartIndex");
	//FileStream(fid, m_mapTriIndexToPartIndex, "m_mapTriIndexToPartIndex");
	//FileStream(fid, m_mapSpringIndexToPartIndex, "m_mapSpringIndexToPartIndex");

	//FileStream(fid, m_triVertPartInfo, "m_triVertPartInfo");
	//FileStream(fid, m_sortedTriVertIndices, "m_sortedTriVertIndices");
	//FileStream(fid, m_triPartInfo, "m_triPartInfo");
	//FileStream(fid, m_sortedTriIndices, "m_sortedTriIndices");
	//FileStream(fid, m_tetVertPartInfo, "m_tetVertPartInfo");
	//FileStream(fid, m_sortedTetVertIndices, "m_sortedTetVertIndices");
	//FileStream(fid, m_tetPartInfo, "m_tetPartInfo");
	//FileStream(fid, m_sortedTetIndices, "m_sortedTetIndices");
	//FileStream(fid, m_springPartInfo, "m_springPartInfo");
	//FileStream(fid, m_sortedSpringIndices, "m_sortedSpringIndices");
}

void Solver::FileStream(FILE* fid, std::vector<char>& vec, std::string msg) {
	int ret, num;
	UDLog("处理变量：" + msg);
	if (FILEMODE::R == m_fileMode) {
		num = ReadNum(fid, msg);
		if (num < 0)
			return;
		vec.resize(num);
		if (!ReadStream(fid, (char*)&vec[0], num * sizeof(char), msg))
			return;
		UDLog("读取变量成功：" + msg);
	}
	else {
		num = vec.size();
		if (!WriteNum(fid, num, msg))
			return;
		if (!WriteStream(fid, (char*)&vec[0], num * sizeof(char), msg))
			return;
		UDLog("写入变量成功：" + msg);
	}
}

void Solver::FileStream(FILE* fid, std::vector<int>& vec, std::string msg) {
	int ret, num;
	UDLog("处理变量：" + msg);
	if (FILEMODE::R == m_fileMode) {
		num = ReadNum(fid, msg);
		if (num < 0)
			return;
		vec.resize(num);
		if (!ReadStream(fid, (char*)&vec[0], num * sizeof(int), msg))
			return;
		UDLog("读取变量成功：" + msg);
	}
	else {
		num = vec.size();
		if (!WriteNum(fid, num, msg))
			return;
		if (!WriteStream(fid, (char*)&vec[0], num * sizeof(int), msg))
			return;
		UDLog("写入变量成功：" + msg);
	}
}

void Solver::FileStream(FILE* fid, std::vector<float>& vec, std::string msg) {
	int ret;
	uint32_t num;
	UDLog("处理变量：" + msg);
	if (FILEMODE::R == m_fileMode) {
		num = ReadNum(fid, msg);
		if (num < 0)
			return;
		vec.resize(num);
		if (!ReadStream(fid, (char*)&vec[0], num * sizeof(float), msg))
			return;
		UDLog("读取变量成功：" + msg);
	}
	else {
		num = vec.size();
		if (!WriteNum(fid, num, msg))
			return;
		if (!WriteStream(fid, (char*)&vec[0], num * sizeof(float), msg))
			return;
		UDLog("写入变量成功：" + msg);
	}
}

void Solver::FileStream(FILE* fid, std::vector<unsigned int>& vec, std::string msg) {
	int ret, num;
	UDLog("处理变量：" + msg);
	if (FILEMODE::R == m_fileMode) {
		num = ReadNum(fid, msg);
		if (num < 0)
			return;
		vec.resize(num);
		if (!ReadStream(fid, (char*)&vec[0], num * sizeof(unsigned int), msg))
			return;
		UDLog("读取变量成功：" + msg);
	}
	else {
		num = vec.size();
		if (!WriteNum(fid, num, msg))
			return;
		if (!WriteStream(fid, (char*)&vec[0], num * sizeof(unsigned int), msg))
			return;
		UDLog("写入变量成功：" + msg);
	}
}

void Solver::ReadFromBin() {
	UDLog("从二进制文件中读取数据: " + m_binFile);
	m_fileMode = R;
	FILE* fid = fopen(m_binFile.c_str(), "rb");
	if (!fid) {
		UDError("Read File Error: " + m_binFile);
		fclose(fid);
		return;
	}
	FileStream(fid);
	fclose(fid);
}
void Solver::WriteToBin() {
	UDLog("将仿真数据写到二进制文件中: " + m_binFile);
	m_fileMode = W;
	FILE* fid = fopen(m_binFile.c_str(), "wb");
	if (!fid) {
		UDError("Write File Error: " + m_binFile);
		fclose(fid);
		return;
	}
	FileStream(fid);
	fclose(fid);
}