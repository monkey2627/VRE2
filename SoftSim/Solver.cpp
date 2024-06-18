#include <Windows.h>
#include "Solver.h"
#include "gpu/gpuvar.h"
#include "bridge.h"

const int OUTPUT_DDIR_ITER_NUM = -1;

LARGE_INTEGER nFreq;
LARGE_INTEGER t1;
LARGE_INTEGER t2;


void Solver::UpdateDirectDirectionTet() {
	runUpdateTetVertDirectDirection();
}

void Solver::UpdateDirectDirectionTri() {
	setDDirwithNormal();
}

//void Solver::UpdateCollision() {
//	float _pos[3];
//	float _dir[3];
//	GetToolPos(_pos);
//	GetToolDir(_dir);
//
//	cudaMemcpy(cylinderLastPos_d, cylinderPos_d, 3 * sizeof(float) * cylinderNum_d, cudaMemcpyDeviceToDevice);
//	cudaMemcpy(cylinderPos_d, _pos, 3 * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(cylinderDirZ_d, _dir, 3 * sizeof(float), cudaMemcpyHostToDevice);
//	printCudaError("UpdateCollision");
//}

void Solver::CopyToGPU() {
	// 四面体顶点信息
	tetVertNum_d = GetTetVertNum();
	tetNum_d = GetTetNum();
	tetSpringNum_d = GetTetSpringNum();
	// 四面体顶点信息

	printf("tetVertNum:%d tetNum:%d\n", tetVertNum_d, tetNum_d);
	cudaMemcpy(tetVertPos_d, m_tetVertPos.data(), m_tetVertPos.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVertRestPos_d, m_tetVertPos.data(), m_tetVertPos.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVertMass_d, m_tetVertMass.data(), m_tetVertMass.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetInvD3x3_d, m_tetInvD3x3.data(), m_tetInvD3x3.size()* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetInvD3x4_d, m_tetInvD3x4.data(), m_tetInvD3x4.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVolume_d, m_tetVolume.data(), m_tetVolume.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVolumeDiag_d, m_tetVolumeDiag.data(), m_tetVolumeDiag.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetIndex_d, m_tetIndex.data(), m_tetIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVertFixed_d, m_tetVertFixed.data(), m_tetVertFixed.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetActive_d, m_tetActive.data(), m_tetActive.size() * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(tetStiffness_d, m_tetStiffness.data(), m_tetStiffness.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVert2TriVertMapping_d, m_mapTetVertIndexToTriVertIndex.data(), m_mapTetVertIndexToTriVertIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVertRestStiffness_d, m_tetVertRestStiffness.data(), m_tetVertRestStiffness.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVertfromTriStiffness_d, m_tetVertfromTriStiffness.data(), m_tetVertfromTriStiffness.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetSpringIndex_d, m_tetSpringIndex.data(), m_tetSpringIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(tetSpringOrgLen_d, m_tetSpringOrgLength.data(), m_tetSpringOrgLength.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetSpringStiffness_d, m_tetSpringStiffness.data(), m_tetSpringStiffness.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVertRelatedTetIdx_d, m_tetVertRelatedTetIdx.data(), m_tetVertRelatedTetIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVertRelatedTetInfo_d, m_tetVertRelatedTetInfo.data(), m_tetVertRelatedTetInfo.size() * sizeof(int), cudaMemcpyHostToDevice);
	printCudaError("CopyToGPU other error");

	// 四面体顶点指导向量
	//cudaMemcpy(tetVertNonPenetrationDir_d, m_tetDirectDir.data(), m_tetDirectDir.size() * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(tetShellIdx_d, m_tetShellIdx.data(), m_tetShellIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
	printf("bindingTetVertIdx size:%d %d\n", m_tetVertBindingTetVertIdx.size(), m_TetVertIndexBindingWeight.size());
	cudaMemcpy(tetVertBindingTetVertIndices_d, m_tetVertBindingTetVertIdx.data(), m_tetVertBindingTetVertIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(tetVertBindingTetVertWeight_d, m_TetVertIndexBindingWeight.data(), m_TetVertIndexBindingWeight.size()*sizeof(float), cudaMemcpyHostToDevice);
	onSurfaceTetVertNum_d = m_onSurfaceTetVertIndices.size();
	cudaMemcpy(onSurfaceTetVertIndices_d, m_onSurfaceTetVertIndices.data(), m_onSurfaceTetVertIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

	// 三角网格信息
	triVertNum_d = GetSurfaceVertNum();
	triVertOrgNum_d = GetOrgTriVertNum();
	int springNum = GetSpringNum();
	triEdgeNum_d = springNum;
	triNum_d = GetSurfaceTriNum();
	printf("triVertNum:%d springNum:%d triNum:%d\n", triVertNum_d, springNum, triNum_d);
	cudaMemcpy(triVertPos_d, m_edgeVertPos.data(), m_edgeVertPos.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triVertRestPos_d, m_edgeVertPos.data(), m_edgeVertPos.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triVertPos_old_d, m_edgeVertPos.data(), m_edgeVertPos.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triVertPos_prev_d, m_edgeVertPos.data(), m_edgeVertPos.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triVertPos_next_d, m_edgeVertPos.data(), m_edgeVertPos.size() * sizeof(float), cudaMemcpyHostToDevice);
	//m_edgeVertMass.assign(m_edgeVertMass.size(), 0.005);
	cudaMemcpy(triVertMass_d, m_edgeVertMass.data(), m_edgeVertMass.size()*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMemcpy(triVertFixed_d, m_edgeVertFixed.data(), m_edgeVertFixed.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triVert2TetVertMapping_d, m_mapTriVertIndexToTetVertSetIndex.data(), m_mapTriVertIndexToTetVertSetIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
	// 三角网格弹簧信息
	cudaMemcpy(triEdgeIndex_d, m_edgeIndex.data(), m_edgeIndex.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(triEdgeOrgLength_d, m_edgeOrgLength.data(), m_edgeOrgLength.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triEdgeDiag_d, m_springDiag.data(), m_springDiag.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triEdgeStiffness_d, m_edgeStiffness.data(), m_edgeStiffness.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triVertfromTetStiffness_d, m_triVertfromTetStiffness.data(), m_triVertfromTetStiffness.size() * sizeof(float), cudaMemcpyHostToDevice);
	// 三角形顶点下标
	cudaMemcpy(triIndex_d, m_triIndex.data(), m_triIndex.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

	printCudaError("CopyToGPU triVertPart");
}

void Solver::Step(float dt) // 软体变形端 ~60Hz
{
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&t1);
	ApplyGravity();
	float omega = 1.0;
	runcalculateST(dampingForTetVert, dt);
	runcalculateSTMU(dampingForTriVert, dt); 
	for (int i = 0; i < m_iterateNum; i++) {
		
		//计算体积力
		runcalculateIF();
		if (m_useTetEdgeSpring)
		{
			runCalculateTetEdgeSpringConstraint();
		}
		if (m_useRestPos)
		{
			runcalculateRestPos();
		}

		if(m_useTetTriInteraction)
			runcalculateRestPosForceWithMeshPos(m_radius);

		omega = 4 / (4 - rho * rho * omega);
		runcalculatePOS(omega, dt);

		// 清除顶点上的力和对角元素
		runClearForce();
	}
	runClearCollision();

	runcalculateV(dt);//更新速度
	UpdateDirectDirectionTet();	////更新指导向量

	//更新表面法向量，用来可视化
	runUpdateMeshNormalMU();
	UpdateDirectDirectionTri();

	cudaMemcpy(m_triVertNorm.data(), triVertNorm_d, GetSurfaceVertNum() * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_triVertPos.data(), triVertPos_d, GetSurfaceVertNum() * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_tetVertPos.data(), tetVertPos_d, GetTetVertNum() * 3 * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(m_tetVertCollisionForce.data(), tetVertCollisionForce_d, GetTetVertNum() * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_tetVertVolumnForce.data(), tetVertForce_d, GetTetVertNum() * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_tetVertCollisionForceLen.data(), tetVertCollisionForceLen_d, GetTetVertNum() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_tetVertVolumnForceLen.data(), tetVertForceLen_d, GetTetVertNum() * sizeof(float), cudaMemcpyDeviceToHost);

	renderStepNumPassed++;
	//printf("Step()->renderStepNumPassed:%d\n", renderStepNumPassed);
	if (renderStepNumPassed == 30)
	{
		std::ofstream collisionForceFile("validate/collisionForce.txt");
		std::ofstream volumnForceFile("validate/volumnForce.txt");
		for (int i = 0; i < GetTetVertNum(); i++)
		{
			collisionForceFile << m_tetVertCollisionForce[i * 3] << " " << m_tetVertCollisionForce[i * 3 + 1] << " " << m_tetVertCollisionForce[i * 3 + 2] << " " << m_tetVertCollisionForceLen[i] << std::endl;
			volumnForceFile << m_tetVertVolumnForce[i * 3] << " " << m_tetVertVolumnForce[i * 3 + 1] << " " << m_tetVertVolumnForce[i * 3 + 2] << " " << m_tetVertVolumnForceLen[i] << std::endl;
		}
		collisionForceFile.close();
		volumnForceFile.close();
	}

	printCudaError("Solver::Step end");
	QueryPerformanceCounter(&t2);

	m_opTime = (t2.QuadPart - t1.QuadPart) / (double)nFreq.QuadPart;

	if (renderStepNumPassed == OUTPUT_DDIR_ITER_NUM)
	{
		cudaMemcpy(m_tetVertDirectDirection.data(), tetVertNonPenetrationDir_d, GetTetVertNum() * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	}
}

void Solver::	ApplyGravity()
{
	gravityX_d = m_gravityX;
	gravityY_d = m_gravityY;
	gravityZ_d = m_gravityZ;
}

