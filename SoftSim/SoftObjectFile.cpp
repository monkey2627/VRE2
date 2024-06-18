#include <fstream>
#include "manager.h"
using namespace std;
void SoftObject::ReadFromFile()
{
//	std::vector<BHPoint3> vertices;
//#pragma region
//	fstream file(m_objFile.c_str());
//	if (!file)		return;
//	const uint32_t kMaxLineLength = 1024;
//	char buffer[kMaxLineLength];
//	float x, y, z, u, v;
//	while (file) {
//		file >> buffer;
//
//		if (strcmp(buffer, "vn") == 0) {// normals
//			file >> x >> y >> z;
//		}
//		else if (strcmp(buffer, "vt") == 0) {
//			// texture coords
//			file >> u >> v;
//			m_surfUV.push_back(UV(u, v));
//		}
//		else if (buffer[0] == 'v') {// positions
//			file >> x >> y >> z;
//			vertices.push_back(BHPoint3(x, y, z));
//		}
//		else if (buffer[0] == 'f') {
//			// faces
//			Tri tttt;
//			for (int i = 0; i < 3; ++i) {
//				int v = -1;
//				int vt = 1;
//				int vn = -1;
//
//				file >> v;
//				if (!file.eof()) {
//					// failed to read another index continue on
//					if (file.fail()) {
//						file.clear();
//						break;
//					}
//
//					if (file.peek() == '/') {
//						file.ignore();
//
//						if (file.peek() != '/') {
//							file >> vt;
//						}
//
//						if (file.peek() == '/') {
//							file.ignore();
//							file >> vn;
//						}
//					}
//				}
//
//				tttt.pi[i] = v - 1;
//				tttt.uvi[i] = vt - 1;
//			}//for (int i = 0; i < 3; ++i)
//			m_surfTris.push_back(tttt);
//		}//else if (buffer[0] == 'f')
//		else {
//			char linebuf[1024];
//			file.getline(linebuf, 1024);
//		}
//	}
//	file.close();
//
//	m_surfVertNum = vertices.size();
//	m_surfTris.resize(m_surfTris.size());
//	m_surfNorm.resize(m_surfVertNum);
//	m_surfNormCount.resize(m_surfVertNum);
//	if (0 == m_surfUV.size())
//		m_surfUV.resize(m_surfVertNum);
//#pragma endregion ¶ÁÈ¡OBJÎÄ¼þ
}
void SoftObject::FileStream(FILE* fid)
{

}
void SoftObject::ReadFromBin()
{

}