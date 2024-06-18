#include "haptic.h"
#include <Windows.h>
#include "../gpu/gpuvar.h"
#include "../gpu/gpufun.h"

LARGE_INTEGER hapticNFreq;
LARGE_INTEGER hapticT1;
LARGE_INTEGER hapticT2;

LARGE_INTEGER tri_NFreq;
LARGE_INTEGER tri_T1;
LARGE_INTEGER tri_T2;

LARGE_INTEGER qg_NFreq;
LARGE_INTEGER qg_T1;
LARGE_INTEGER qg_T2;

LARGE_INTEGER solve_NFreq;
LARGE_INTEGER solve_6dof_T1;
LARGE_INTEGER solve_6dof_T2;
LARGE_INTEGER solve_quant_T1;
LARGE_INTEGER solve_quant_T2;

void Haptic::Init() {
	m_virtualCoupling.set_k(m_kVc, m_kc, m_kVct);
	m_F_vc.resize(6);
	m_qg.resize(6);
	m_qh.resize(6);
	m_dir_g.resize(3);
	m_dir_h.resize(3);
	m_hapticToolTrans.resize(16);
	m_virtualToolTrans.resize(16);
	fill(m_qg.begin(), m_qg.end(), 0.0f);
	fill(m_qh.begin(), m_qh.end(), 0.0f);
	fill(m_dir_g.begin(), m_dir_g.end(), 0.0f);
	fill(m_dir_h.begin(), m_dir_h.end(), 0.0f);
}

void Haptic::SetQH(const float* toolTrans)
{
	memcpy(m_qh.data(), &toolTrans[12], 3 * sizeof(float));
	memcpy(toolDir, &toolTrans[8], 3 * sizeof(float));
	// Omega = \theta * K
	// \theta Ϊ��ת�Ƕȣ���λΪ���ȣ� KΪ��ת�ᡣ
	Eigen::Quaterniond q;
	mat3 rot;
	rot << toolTrans[0], toolTrans[4], toolTrans[8],
		toolTrans[1], toolTrans[5], toolTrans[9],
		toolTrans[2], toolTrans[6], toolTrans[10];
	q = rot; // ���Ը���Ԫ��ֱ�Ӹ�ֵ3x3����ת���󣬻��Զ������Ӧ����Ԫ����
	double halfTheta = acos(q.w());
	vec3 rotAxis(q.x(), q.y(), q.z());
	rotAxis.normalize();

	rotAxis *= (2 * halfTheta);
	m_qh[3] = rotAxis.x();
	m_qh[4] = rotAxis.y();
	m_qh[5] = rotAxis.z();

	auto dir_h = m_virtualCoupling.calculateToolDir(rotAxis);

	m_dir_h[0] = dir_h.x();
	m_dir_h[1] = dir_h.y();
	m_dir_h[2] = dir_h.z();

	memcpy(m_hapticToolTrans.data(), toolTrans, 16 * sizeof(float));
}

void Haptic::HapticCollision(CollisionMode mode)
{
	printCudaError("hapticCollision start");
	if (m_hapticStepNum == 0)
	{
		cudaMemcpy(toolPosePrev_d, toolPositionAndDirection_d, 6 * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(toolPositionAndDirection_d, m_qh.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(toolPositionAndDirection_d + 3, m_dir_h.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
	}
	else
	{
		cudaMemcpy(toolPosePrev_d, toolPositionAndDirection_d, 6 * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(toolPositionAndDirection_d, m_qg.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(toolPositionAndDirection_d + 3, m_dir_g.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
	}


	runClearFc();
	if (mode == TETVERT_MODE)
	{
		runHapticCollisionSphereForTet(m_radius, m_collisionStiffness, m_kc, 0);
	}
	else if (mode == TRIVERT_MODE)
	{
		//runClearCollisionMU();
		runHapticCollisionSphereForTri(m_radius, m_collisionStiffness, m_kc, 0);
	}
	else if (mode == MERGE_MODE)
	{
		runHapticCollisionSphere_Merged(m_radius, m_collisionStiffness, m_kc, 0);
	}
	else if (mode == CYLINDER_MERGE_MODE)
	{
		// ��ײ���
		runHapticCollisionCylinder_Merged(m_radius, m_cylinderLength, m_collisionStiffness, m_kc, 0);
		// ������ײ������㹤���ϵĽӴ���
		runDeviceCalculateContact(m_kc);
	}
}

void Haptic::HapticStep() // ��1000Hz����
{
	//QueryPerformanceFrequency(&hapticNFreq);
	//QueryPerformanceCounter(&hapticT1);

	float omega = 1.0;
	float dt = 0.001;

	//�����ײ��Ϣ
	runClearCollisionMU();

	QueryPerformanceFrequency(&qg_NFreq);
	QueryPerformanceCounter(&qg_T1);
	LARGE_INTEGER T0, T1, T2;
	LONGLONG cd_time = 0;
	LONGLONG solve_6dof_time = 0;
	///ţ�ٷ�ֻ����һ��
	for (int i = 0; i < 1; i++) {
		QueryPerformanceCounter(&T0);
		//��ײ���
		HapticCollision(CYLINDER_MERGE_MODE);
		QueryPerformanceCounter(&T1);
		cd_time += T1.QuadPart - T0.QuadPart;
		// ��⹤��λ��
		UpdateQG();
		QueryPerformanceCounter(&T2);
		solve_6dof_time += T2.QuadPart - T1.QuadPart;
	}
	QueryPerformanceCounter(&qg_T2);
	m_hapticQGOpTime = (qg_T2.QuadPart - qg_T1.QuadPart) / (double)qg_NFreq.QuadPart;//��λΪ��
	double cd_time_in_ms = cd_time / (double)qg_NFreq.QuadPart * 1000;
	double solve_6dof_time_in_ms = solve_6dof_time / (double)qg_NFreq.QuadPart * 1000;
	QueryPerformanceFrequency(&tri_NFreq);
	QueryPerformanceCounter(&tri_T1);
	// ���±�������������Ϣ
	{
		omega = 1.0;
		runcalculateSTMU(m_dampingForTriVert, dt);
		//������⣬�����������߻�ը����û�п��ܶ��ѭ�����һ�ε�������
		for (int i = 0; i < 1; i++) {
			//��������
			runcalculateIFMU();
			if (m_useTetTriInteraction)
			{
				runcalculateRestPosForceWithTetPos(m_radius);
			}
			//����λ��
			omega = 4 / (4 - m_rho * m_rho * omega);
			runcalculatePosMU(omega, dt);
		}
		runcalculateVMU(dt);

		// WARNING: ���������·���������ζˡ���Ϊ��ʾ��ʱ����Ҫ�ȶ��ı��淨��������ָ�����������ᷢ��ͻ�䣬��Ծɵķ�����Ҳ���á�
	}
	QueryPerformanceCounter(&tri_T2);
	m_hapticTriOpTime = (tri_T2.QuadPart - tri_T1.QuadPart) / (double)tri_NFreq.QuadPart;

	m_hapticStepNum++;
	//QueryPerformanceCounter(&hapticT2);

	//m_hapticStepOpTime = (hapticT2.QuadPart - hapticT1.QuadPart) / (double)hapticNFreq.QuadPart;
}

void Haptic::UpdateQG()
{
	// ţ�ٷ����
	//////////////////////////////////////////////////////	
	// ţ�ٷ�
	using namespace Eigen;
	// calculate partial F_vc
	float t[9];
	cudaMemcpy(t, totalPartial_FC_X_d, 9 * sizeof(float), cudaMemcpyDeviceToHost);
	mat3 partial_Fc;
	partial_Fc << t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8];
	float fc[3];
	cudaMemcpy(fc, totalFC_d, 3 * sizeof(float), cudaMemcpyDeviceToHost);
	printCudaError("Fc copy");

	int collisionNum;
	cudaMemcpy(&collisionNum, hapticCollisionNum_d, sizeof(int), cudaMemcpyDeviceToHost);
	//printf("collisionNum: %d\n", collisionNum);
	printCudaError("collisionNum copy error");

	vec3 Fc;
	Fc << fc[0], fc[1], fc[2];
	// calculate ����eigen��ⷽ��

	vec3 Xh, Omega_h;
	Xh << m_qh[0], m_qh[1], m_qh[2];
	Omega_h << m_qh[3], m_qh[4], m_qh[5];
	vec3 Xg, Omega_g;
	Xg << m_qg[0], m_qg[1], m_qg[2];
	Omega_g << m_qh[3], m_qh[4], m_qh[5];// ���⹤����̬����������ͬ

	vec3 tool_dir = m_virtualCoupling.calculateToolDir(Omega_g);
	m_dir_g[0] = tool_dir.x();
	m_dir_g[1] = tool_dir.y();
	m_dir_g[2] = tool_dir.z();

	auto delta_x = m_virtualCoupling.solve(Xh, Xg, Fc, partial_Fc, collisionNum);

	m_qg[0] += delta_x[0];
	m_qg[1] += delta_x[1];
	m_qg[2] += delta_x[2];

	vec3 Xg_new;
	Xg_new << m_qg[0], m_qg[1], m_qg[2];
	memcpy(m_virtualToolTrans.data(), m_hapticToolTrans.data(), 16 * sizeof(float));
	m_virtualToolTrans[12] = m_qg[0];
	m_virtualToolTrans[13] = m_qg[1];
	m_virtualToolTrans[14] = m_qg[2];

	vec3 F_vc_saturated = m_virtualCoupling.calculateFVC(Xg_new, Xh);
	F_vc[0] = F_vc_saturated[0];
	F_vc[1] = F_vc_saturated[1];
	F_vc[2] = F_vc_saturated[2];
}

void Haptic::UpdateQGwithChebyshev(float omega)
{
	float EPS = 1E-7;
	float t[9];
	cudaMemcpy(t, totalPartial_FC_X_d, 9 * sizeof(float), cudaMemcpyDeviceToHost);
	float diag[3];
	diag[0] = t[0];
	diag[1] = t[4];
	diag[2] = t[8];
	printf("diag:%f %f %f\n", diag[0], diag[1], diag[2]);

	float fc[3];
	cudaMemcpy(fc, totalFC_d, 3 * sizeof(float), cudaMemcpyDeviceToHost);

	int collisionNum;
	cudaMemcpy(&collisionNum, hapticCollisionNum_d, sizeof(int), cudaMemcpyDeviceToHost);

	vec3 Xh;
	Xh << m_qh[0], m_qh[1], m_qh[2];
	// �Խ�Ԫ��Ϊ0��ô�죿֮ǰ�б�ѩ����·�ĸ��constantDiag����
	// ����Խ�Ԫ��Ϊ0����ô�����ά���������Ͳ���䡣��Ϊ����û��������Ҳ��û�й��ԡ�
	// ��ô���ά���Ͻ�������λ�ñ仯��
	float deltaPos[3];
	bool deltaFlag = false;
	for (int i = 0; i < 3; i++)
	{
		if (abs(diag[i]) < EPS)
			deltaPos[i] = 0;
		else
		{
			deltaPos[i] = fc[i] / diag[i];
			deltaFlag = true;
		}
	}

	if (deltaFlag)
	{
		printf("deltaPos:%f %f %f\n", deltaPos[0], deltaPos[1], deltaPos[2]);
		deltaPos[0] *= 0.1 * omega;
		deltaPos[1] *= 0.1 * omega;
		deltaPos[2] *= 0.1 * omega;
		m_qg[0] = deltaPos[0] + m_qg[0];
		m_qg[1] = deltaPos[1] + m_qg[1];
		m_qg[2] = deltaPos[2] + m_qg[2];
	}
	else
	{
		printf("deltaFlag==false\n");
		m_qg[0] = m_qh[0];
		m_qg[1] = m_qh[1];
		m_qg[2] = m_qh[2];
	}

	vec3 Xg_new;
	Xg_new << m_qg[0], m_qg[1], m_qg[2];

	vec3 F_vc_saturated = m_virtualCoupling.calculateFVC(Xg_new, Xh);
	F_vc[0] = F_vc_saturated[0];
	F_vc[1] = F_vc_saturated[1];
	F_vc[2] = F_vc_saturated[2];
}

void Haptic::ComputeLeftToolForce(const float* trans, double* force)
{
	QueryPerformanceFrequency(&hapticNFreq);
	QueryPerformanceCounter(&hapticT1);
	//printf("ComputeLeftToolForce:m_renderStepNumPassed:%d\n", m_renderStepNumPassed);
	if (m_renderStepNumPassed == 0)//�����ǰû�м����ͼ��֡�������㵱ǰ�������
		return;
	printCudaError("ComputeLeftToolForce start");
	SetQH(trans);
	if (m_hapticStepNum == 0)
	{
		// ��ʼ�����⹤�ߵĳ����λ��
		memcpy(m_qg.data(), m_qh.data(), 6 * sizeof(float));
		cout << m_qg[0] << m_qg[1] << m_qg[2] << endl;
	}
	HapticStep();
	force[0] = -F_vc[0];
	force[1] = -F_vc[1];
	force[2] = -F_vc[2];
	//float forceLen = sqrt(F_vc[0] * F_vc[0] + F_vc[1] * F_vc[1] + F_vc[2] * F_vc[2]);
	//printf("force len:%f\n", forceLen);

	QueryPerformanceCounter(&hapticT2);

	m_hapticStepOpTime = (hapticT2.QuadPart - hapticT1.QuadPart) / (double)hapticNFreq.QuadPart;
}