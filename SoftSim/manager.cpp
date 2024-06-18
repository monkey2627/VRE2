#include "bridge.h"
#include "manager.h"
#include "gpu/gpuvar.h"
void Manager::Init()
{

	int count = 0;
	int i = 0;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		UDError("û�з���CUDA�豸");
		return;
	}

	cudaDeviceProp prop;
	for (i = 0; i < count; i++) {
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
			if (prop.major >= 1) 
				break;

	}
	if (i == count) {
		UDError("û���豸֧��CUDA");
		return;
	}
	cudaSetDevice(i);
	UDLog("CUDA�豸���ƣ�"+ std::string(prop.name));

	//m_softHapticSolver.m_binFile = "./data/softdeformation__sf.bin";
	//m_softHapticSolver.m_objFile = "./data/softdeformation__sf.obj";
	//m_softHapticSolver.m_tetFile = "./data/softdeformation__sf.msh";

	//m_softHapticSolver.m_binFile = "./data/liver.bin";
	//m_softHapticSolver.m_objFile = "./data/liver.obj";
	//m_softHapticSolver.m_tetFile = "./data/liver.msh";

	m_softHapticSolver.m_binFile = "./data/two-objects.bin";
	m_softHapticSolver.m_objFile = "./data/two-objects.obj";
	m_softHapticSolver.m_tetFile = "./data/two-objects.msh";

	m_softHapticSolver.SolverInit();
	/// �������������������������ռ�����
	UDLog("��ʼ��ʼ������");
	m_rectumObj.m_Type = "soft";
	m_rectumObj.m_Name = "rectum";

	m_hapticDevice.InitHapticDevice();
	m_vcHaptic.Init();
}

