//////////���кͳ�ʼ����صļӵ����������
//////////�����ļ���ȡ��ϵͳ�����ʼ��
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
#pragma endregion  ���ֱ���


	//std::vector<int> m_tetIndex;//����������	
	FileStream(fid, m_tetIndex, "m_tetIndex");
	//std::vector<float> m_tetStiffness;//���������Ӳ
	FileStream(fid, m_tetStiffness, "m_tetStiffness");
	//std::vector<float> m_tetVertPos;//����λ��	
	FileStream(fid, m_tetVertPos, "m_tetVertPos");
	//std::vector<float> m_tetVertFixed;//�Ƿ�̶�	
	FileStream(fid, m_tetVertFixed, "m_tetVertFixed");
	//std::vector<float> m_tetVertMass;//��������
	FileStream(fid, m_tetVertMass, "m_tetVertMass");
	//std::vector<float> m_tetVertRestStiffness;//ÿ����restpos��Լ��
	FileStream(fid, m_tetVertRestStiffness, "m_tetVertRestStiffness");
	//std::vector<float> m_tetVertfromTriStiffness;////�����ζ�����������Լ���ն�
	FileStream(fid, m_tetVertfromTriStiffness, "m_tetVertfromTriStiffness");

	FileStream(fid, m_onSurfaceTetVertIndices, "m_onSurfaceTetVertIndices");
	

	///ԭʼ����������
	//std::vector<int> m_triIndexOrg;// ԭʼ����������
	FileStream(fid, m_triIndexOrg, "m_triIndexOrg");
	//std::vector<float> m_triVertPosOrg; // ԭʼ����������涥������
	FileStream(fid, m_triVertPosOrg, "m_triVertPosOrg");
	//std::vector<int> m_triUVIndexOrg;// ϸ������������
	FileStream(fid, m_triUVIndexOrg, "m_triUVIndexOrg");
	//std::vector<float> m_triUVOrg; // ϸ������������涥������
	FileStream(fid, m_triUVOrg, "m_triUVOrg");


	//std::vector<unsigned int> m_triIndex;// ϸ������������
	FileStream(fid, m_triIndex, "m_triIndex");
	//std::vector<float> m_triVertPos; // ϸ������������涥������
	FileStream(fid, m_triVertPos, "m_triVertPos");
	//std::vector<float> m_triVertColor; // ϸ������������涥����ɫ//���ӻ���
	FileStream(fid, m_triVertColor, "m_triVertColor");
	//std::vector<float> m_triVertNorm;// ϸ������������涥�㷨����
	FileStream(fid, m_triVertNorm, "m_triVertNorm");
	//std::vector<unsigned int> m_triUVIndex;// ϸ������������
	FileStream(fid, m_triUVIndex, "m_triUVIndex");
	//std::vector<float> m_triUV; // ϸ������������涥������
	FileStream(fid, m_triUV, "m_triUV");

	
	//std::vector<char> m_tetActive;//�Ƿ�����Ч״̬
	FileStream(fid, m_tetActive, "m_tetActive");
	//std::vector<float> m_tetInvD3x3;//������������//���ڼ�������ݶ�
	FileStream(fid, m_tetInvD3x3, "m_tetInvD3x3");
	//std::vector<float> m_tetInvD3x4;//���ڼ���Խ���
	FileStream(fid, m_tetInvD3x4, "m_tetInvD3x4");
	//std::vector<float> m_tetVolume;//��������������ȣ�tetNum
	FileStream(fid, m_tetVolume, "m_tetVolume");
	//std::vector<float> m_tetVolumeDiag;//����ϵͳ�ĶԽǾ����ACT*AC����, ���ȣ�tetNum
	FileStream(fid, m_tetVolumeDiag, "m_tetVolumeDiag");
	//std::vector<int> m_mapTetVertIndexToTriVertIndex;//�������Ӧ���涥�㣬һ��һ
	FileStream(fid, m_mapTetVertIndexToTriVertIndex, "m_mapTetVertIndexToTriVertIndex");
	//std::vector<int> m_mapTriVertIndexToTetVertSetIndex;//���涥��������嶥�㣬һ�Զ�
	FileStream(fid, m_mapTriVertIndexToTetVertSetIndex, "m_mapTriVertIndexToTetVertSetIndex");

	//std::vector<int> m_tetVertBindingTetVertIdx; // �����嶥������������������񶥵��±ꡣ
	FileStream(fid, m_tetVertBindingTetVertIdx, "m_tetVertBindingTetVertIdx");
	//std::vector<float> m_TetVertIndexBindingWeight;
	FileStream(fid, m_TetVertIndexBindingWeight, "m_TetVertIndexBindingWeight");

	//std::vector<int> m_tetSpringIndex; // ���������񵯻����飬
	//std::vector<float> m_tetSpringOrgLength; // �����嵯��ԭ��
	//std::vector<float> m_tetSpringStiffness;
	FileStream(fid, m_tetSpringIndex, "m_tetSpringIndex");
	FileStream(fid, m_tetSpringOrgLength, "m_tetSpringOrgLength");
	FileStream(fid, m_tetSpringStiffness, "m_tetSpringStiffness");

	FileStream(fid, m_tetVertRelatedTetIdx, "m_tetVertRelatedTetIdx");
	FileStream(fid, m_tetVertRelatedTetInfo, "m_tetVertRelatedTetInfo");

	////std::vector<float> m_tetDirectDir; // ��ʼ�������嶥��ָ������
	//FileStream(fid, m_tetDirectDir, "m_tetDirectDir");

	//std::vector<unsigned int> m_edgeIndex;// ��������
	FileStream(fid, m_edgeIndex, "m_edgeIndex");
	//std::vector<float> m_edgeStiffness;// ���ɸն�
	FileStream(fid, m_edgeStiffness, "m_edgeStiffness");
	//std::vector<float> m_edgeVertPos;//���ɶ���
	FileStream(fid, m_edgeVertPos, "m_edgeVertPos");
	//std::vector<float> m_edgeVertFixed;
	FileStream(fid, m_edgeVertFixed, "m_edgeVertFixed");
	//std::vector<float> m_edgeVertMass;
	FileStream(fid, m_edgeVertMass, "m_edgeVertMass");
	//std::vector<float> m_edgeOrgLength; // ����ԭ��
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
	UDLog("���������" + msg);
	if (FILEMODE::R == m_fileMode) {
		num = ReadNum(fid, msg);
		if (num < 0)
			return;
		vec.resize(num);
		if (!ReadStream(fid, (char*)&vec[0], num * sizeof(char), msg))
			return;
		UDLog("��ȡ�����ɹ���" + msg);
	}
	else {
		num = vec.size();
		if (!WriteNum(fid, num, msg))
			return;
		if (!WriteStream(fid, (char*)&vec[0], num * sizeof(char), msg))
			return;
		UDLog("д������ɹ���" + msg);
	}
}

void Solver::FileStream(FILE* fid, std::vector<int>& vec, std::string msg) {
	int ret, num;
	UDLog("���������" + msg);
	if (FILEMODE::R == m_fileMode) {
		num = ReadNum(fid, msg);
		if (num < 0)
			return;
		vec.resize(num);
		if (!ReadStream(fid, (char*)&vec[0], num * sizeof(int), msg))
			return;
		UDLog("��ȡ�����ɹ���" + msg);
	}
	else {
		num = vec.size();
		if (!WriteNum(fid, num, msg))
			return;
		if (!WriteStream(fid, (char*)&vec[0], num * sizeof(int), msg))
			return;
		UDLog("д������ɹ���" + msg);
	}
}

void Solver::FileStream(FILE* fid, std::vector<float>& vec, std::string msg) {
	int ret;
	uint32_t num;
	UDLog("���������" + msg);
	if (FILEMODE::R == m_fileMode) {
		num = ReadNum(fid, msg);
		if (num < 0)
			return;
		vec.resize(num);
		if (!ReadStream(fid, (char*)&vec[0], num * sizeof(float), msg))
			return;
		UDLog("��ȡ�����ɹ���" + msg);
	}
	else {
		num = vec.size();
		if (!WriteNum(fid, num, msg))
			return;
		if (!WriteStream(fid, (char*)&vec[0], num * sizeof(float), msg))
			return;
		UDLog("д������ɹ���" + msg);
	}
}

void Solver::FileStream(FILE* fid, std::vector<unsigned int>& vec, std::string msg) {
	int ret, num;
	UDLog("���������" + msg);
	if (FILEMODE::R == m_fileMode) {
		num = ReadNum(fid, msg);
		if (num < 0)
			return;
		vec.resize(num);
		if (!ReadStream(fid, (char*)&vec[0], num * sizeof(unsigned int), msg))
			return;
		UDLog("��ȡ�����ɹ���" + msg);
	}
	else {
		num = vec.size();
		if (!WriteNum(fid, num, msg))
			return;
		if (!WriteStream(fid, (char*)&vec[0], num * sizeof(unsigned int), msg))
			return;
		UDLog("д������ɹ���" + msg);
	}
}

void Solver::ReadFromBin() {
	UDLog("�Ӷ������ļ��ж�ȡ����: " + m_binFile);
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
	UDLog("����������д���������ļ���: " + m_binFile);
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