//////////所有和初始化相关的加到这个函数中
//////////包括文件读取，系统矩阵初始化
#include <Windows.h>
#include "Solver.h"
#include "bridge.h"
using namespace std;



void Solver::ReadObjFile(const char* name) {
	UDLog("开始读取：" + string(name));
	int verNum = 0, triNum = 0, uvNum = 0;

	std::ifstream file(name);
	if (!file) {
		UDError(string("打开文件失败：") + name);
		return;
	}

	//some scratch memory
	char buffer[1024];
	while (file) {
		file >> buffer;
		if (strcmp(buffer, "vn") == 0) {
			//normals
			// ignore
			float x, y, z;
			file >> x >> y >> z;
		}
		else if (strcmp(buffer, "vt") == 0) {
			//texture coords
			// ignore
			float u, v;
			file >> u >> v;
			m_triUVOrg.push_back(u);
			m_triUVOrg.push_back(v);
			uvNum++;
		}
		else if (buffer[0] == 'v') {
			//positions;
			float x, y, z;
			file >> x >> y >> z;
			m_triVertPosOrg.push_back(x);
			m_triVertPosOrg.push_back(y);
			m_triVertPosOrg.push_back(z);
			verNum++;
		}
		else if (buffer[0] == 's' || buffer[0] == 'g' || buffer[0] == 'o') {
			// ignore smoothing groups, groups and objects
			char linebuf[256];
			file.getline(linebuf, 256);
		}
		else if (strcmp(buffer, "mtllib") == 0)
		{
			// ignored
			std::string MaterialFile;
			file >> MaterialFile;
		}
		else if (strcmp(buffer, "usemtl") == 0)
		{
			// read Material name
			std::string materialName;
			file >> materialName;
		}
		else if (buffer[0] == 'f') {
			// faces
			for (int i = 0; i < 3; ++i) {
				int v = -1;
				int vt = 1;
				int vn = -1;

				file >> v;
				if (!file.eof())
				{
					// failed to read another index continue on
					if (file.fail())
					{
						file.clear();
						break;
					}

					if (file.peek() == '/')
					{
						file.ignore();

						if (file.peek() != '/')
						{
							file >> vt;
						}

						if (file.peek() == '/')
						{
							file.ignore();
							file >> vn;
						}
					}
				}
				m_triIndexOrg.push_back(v - 1);
				m_triUVIndexOrg.push_back(vt - 1);
			}
			//printf("\n");
			triNum++;
		}
		else if (buffer[0] == '#')
		{
			// comment
			char linebuf[256];
			file.getline(linebuf, 256);
		}
	}

	if (uvNum == 0) {
		m_triUVIndexOrg = m_triIndexOrg;
		UDError(string("obj文件没有纹理坐标：") + name);
		for (int i = 0; i < verNum; i++) {
			m_triUVOrg.push_back(0.5);
			m_triUVOrg.push_back(0.5);
		}
	}
	UDLog(string("顶点数量：") + to_string(verNum));
	UDLog(string("三角数量：") + to_string(m_triIndexOrg.size()/3));
	UDLog("读取完毕：" + string(name));
}

void Solver::ReadObjPoints(const char* name, std::vector<float>& points)
{
	UDLog("开始读取：" + string(name));
	int verNum = 0, triNum = 0, uvNum = 0;

	std::ifstream file(name);
	if (!file) {
		UDError(string("打开文件失败：") + name);
		return;
	}

	//some scratch memory
	std::string buffer;
	while (file) {
		getline(file, buffer);
		if (buffer[0] == '#')
		{
			continue;
		}
		if (buffer[0] == 'v') {
			//positions;
			char c;
			float x, y, z;
			sscanf(buffer.c_str(), "%c %f %f %f", &c, &x, &y, &z);
			//printf("p[%f %f %f]\n", x, y, z);
			points.push_back(x);
			points.push_back(y);
			points.push_back(z);
			verNum++;
		}
	}
	UDLog(string("外壳顶点数量：") + to_string(verNum));
}

void Solver::ReadTetFile(const char* inputFile)
{
	UDLog("开始读取：" + string(inputFile));
	char filename[1024];
	char buffer[128];

	int eleNum;
	int number = 0;

	ifstream file(inputFile);
	if (!file) {
		UDError(std::string("打开文件错误：")+ inputFile);
		return;
	}
	while (file) {
		file >> buffer;
		if (strcmp(buffer, "$Nodes") == 0) {
			file >> number;
			unsigned int idx;
			float x, y, z;
			UDLog(string("四面体顶点数量：") + to_string(number));
			for (int i = 0; i < number; i++) {
				file >> idx >> x >> y >> z;
				m_tetVertPos.push_back(x);
				m_tetVertPos.push_back(y);
				m_tetVertPos.push_back(z);
			}
		}
		else if (strcmp(buffer, "$Elements") == 0) {
			file >> eleNum;
			int idx, type, tag, phy, elem;
			unsigned int i0, i1, i2, i3;
			UDLog(string("四面体数量：") + to_string(eleNum));
			for (int i = 0; i < eleNum; i++) {
				file >> idx >> type >> tag >> phy >> elem;
				if (type == 2) { //surface indices
					file >> i0 >> i1 >> i2;
				}
				else if (type == 4) { //tet indices
					file >> i0 >> i1 >> i2 >> i3;
					m_tetIndex.push_back(i0 - 1);
					m_tetIndex.push_back(i1 - 1);
					m_tetIndex.push_back(i2 - 1);
					m_tetIndex.push_back(i3 - 1);
					m_tetActive.push_back(1);
				}
			}
		}
	}
	file.close();
	m_tetVertFixed.resize(m_tetVertPos.size() / 3, 0);
	UDLog("读取完毕：" + string(inputFile));
}

void Solver::InitFromFile() {
	ReadTetFile(m_tetFile.c_str());
	ReadObjFile(m_objFile.c_str());
}

void Solver::SaveMesh(const char* filename) {


	ofstream file(filename);

	if (!file)
		return;

	file << "# Beihang University" << endl;
	file << "# vertices" << endl;
	for (uint32_t i = 0; i < m_triVertPos.size()/3; ++i)
	{
		
		file << "v " << m_triVertPos[i*3] << " " << m_triVertPos[i * 3+1] << " " << m_triVertPos[i * 3+2] << endl;
	}
	file << "# " << m_triVertPos.size() / 3 << " vertices" << endl;

	file << "# normals" << endl;
	for (uint32_t i = 0; i < m_triVertPos.size() / 3; ++i)
	{

		file << "vn " << m_triVertNorm[i * 3] << " " << m_triVertNorm[i * 3 + 1] << " " << m_triVertNorm[i * 3 + 2] << endl;
	}
	file << "# " << m_triVertPos.size() / 3 << " vertices" << endl;

	file << "# faces" << endl;;
	for (uint32_t i = 0; i < m_triIndex.size() / 3; ++i)
	{
		uint32_t j = 3 * i;
		file << "f " <<
			m_triIndex[j] + 1 << "/" << m_triIndex[j] + 1 << " " <<
			m_triIndex[j + 1] + 1 << "/" << m_triIndex[j + 1] + 1 << " " <<
			m_triIndex[j + 2] + 1 << "/" << m_triIndex[j + 2] + 1 << endl;
	}
	file << "# " << m_triIndex.size() / 3 << " faces" << endl;


	file.close();

}

