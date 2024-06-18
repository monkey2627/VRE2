#include "calculate.h"
#include <Windows.h>
bool OUTPUT_FVC = true;
bool PRINT_COLLISION = false;

int L_MAX = 10;
extern bool USE_HAPTIC_COLLISION;

double damping = 0.5;
double DELTA_X_MAX = 0.02;
double DELTA_ANGLE_MAX = 0.2 * 3.1415926 / 180;
// double DELTA_X_MAX = 0.002;
double FVC_MAX = 3.2;
double EPS = 1e-8;

using namespace Eigen;
using namespace std;
#define FULL6DOF

calculator::calculator()
{
	m_k_vc = 5;
	m_k_c = 5;
	m_k_vct = 1;

	_r_lin = FVC_MAX / m_k_vc / 2;
	PRINT_INFO = false;
}

calculator::~calculator()
{
}

void calculator::set_k(ValType k_vc, ValType k_c, ValType k_vct)
{
	m_k_vc = k_vc;
	m_k_c = k_c;

	_r_lin = FVC_MAX / k_vc / 2;

	m_k_vct = k_vct;
}

mat3 calculator::calculatePartialFVC_X(vec3 qg, vec3 qh)
{
	double r = l2dis(qg, qh);
	mat3 result;
	if (r < _r_lin)
	{
		mat3 identity;
		identity.setIdentity();
		result = -m_k_vc * identity;
	}
	else
	{
		vec3 e_r = (qg - qh) / r;
		double f_r = saturationMapping(r);
		double f_r_derivative = saturationMappingDerivative(r);
		
		mat3 identity;
		identity.setIdentity();
		mat3 e_r_matrix = e_r*e_r.transpose();

		auto partial = -f_r / r * identity + (f_r/r - f_r_derivative) * e_r_matrix;
		result = partial;
	}
	m_partial_FVC_X = result;
	return result;
}

vec3 calculator::calculateFC(Point& p, ValType d, ValType k)
{
	if (d >= 0)
	{
		cerr << "depth:" << d << " is greater than 0, collision have not occured.CHECK\n";
		return vec3(0, 0, 0);
	}
		
	return k * d * p.normal;
}

mat3 calculator::calculatePartialFC_X(Point& p, ValType k)
{
	return -k * p.normal * p.normal.transpose();
}

mat3 calculator::calculatePartialFVC_Omega()
{
	mat3 result;
	result.setZero();
	return result;
}

ValType calculator::saturationMapping(ValType r)
{
	ValType result;
	if (r < _r_lin)
	{
		result = m_k_vc * r;
	}
	else
	{
		double weight;
		double upper_index = 2 * m_k_vc * (_r_lin - r) / FVC_MAX;
		weight = 1 - 0.5 * exp(upper_index);
		result = FVC_MAX * weight;
	}
	return ValType(result);
}

ValType calculator::saturationMappingDerivative(ValType r)
{
	ValType result = m_k_vc * exp(2 * m_k_vc * (_r_lin - r) / FVC_MAX);
	return result;
}

vec3 calculator::solve_delta_x(vec3 F_C, vec3 F_VC, mat3 partial_F_C, mat3 partial_F_VC)
{
	mat3 A = partial_F_C + partial_F_VC;
	vec3 b = -(F_C + F_VC);

	vec3 delta_x = A.ldlt().solve(b);

	return delta_x;
}

Quaterniond calculator::calculate_qt(vec3 Omega_h, vec3 Omega_g)
{
	Quaterniond result;
	Quaterniond quat_g = Omega2Quaternion(Omega_g);
	Quaterniond quat_h = Omega2Quaternion(Omega_h);
	result = quat_h * quat_g.inverse();

	if (result.w() < 0)
	{
		quat_g = Quaterniond(-quat_g.w(), -quat_g.x(), -quat_g.y(), -quat_g.z());
		result = quat_h * quat_g.inverse();
	}
	Quaterniond t(m_k_vct * result.w(), m_k_vct * result.x(), m_k_vct * result.y(), m_k_vct * result.z());
	return t;
}

vec3 calculator::calculateTVC(Quaterniond qt)
{
	vec3 result(qt.x(), qt.y(), qt.z());
	return result;
}

mat3 calculator::calculatePartialTVC_Omega(Quaterniond qt)
{
	mat3 result; 
	vec3 TVC = calculateTVC(qt);
	double scalar_qT = qt.w();
	mat3 part0;
	part0.setIdentity();
	mat3 part2 = genTildeMatrix(TVC);
	result = 0.5 * (scalar_qT * part0 - part2);
	return result;
}

mat3 calculator::calculatePartialTVC_X()
{
	mat3 result;
	result.setZero();
	return result;
}

mat3 calculator::genTildeMatrix(vec3 r)
{
	mat3 result;
	result << 0, -r(2), r(1),
		r(2), 0, -r(0),
		-r(1), r(0), 0;
	return result;
}

vec3 calculator::calculateToolDir(vec3 Omega)
{
	Quaterniond q = Omega2Quaternion(Omega);
	mat3 rot = q.toRotationMatrix();
	vec3 dir(0, 0, 1);
	return rot * dir;
}

Eigen::Quaterniond calculator::Omega2Quaternion(vec3 Omega)
{
	double theta = Omega.norm();
	vec3 axis = Omega.normalized();
	double halfTheta = theta / 2;
	double w = cos(halfTheta);
	double x = sin(halfTheta) * axis.x();
	double y = sin(halfTheta) * axis.y();
	double z = sin(halfTheta) * axis.z();
	Quaterniond q;
	if(w>0)
	{
		q = Quaterniond(w, x, y, z);
	}
	else
	{
		q = Quaterniond(-w, -x, -y, -z);
	}
	return q;
}

vec3 calculator::Quaternion2Omega(Eigen::Quaterniond q)
{
	double l = q.norm();
	if (abs(l - 1) > 1e-6)
		cout << "quaternion have not been normalized" << endl;
	double halfTheta = acos(q.w());
	vec3 rotAxis(q.x(), q.y(), q.z());
	vec3 omega = rotAxis.normalized() * (halfTheta * 2);
	
	return omega;
}

vec3 calculator::solve(vec3 Xh, vec3 Xg, vec3 F_c, mat3 partial_F_C, int collisionNum)
{
	vec3 F_VC = calculateFVC(Xg, Xh);
	mat3 partial_F_VC = calculatePartialFVC_X(Xg, Xh);

	if (collisionNum > L_MAX)
	{
		F_c = F_c / collisionNum * L_MAX;
		partial_F_C = partial_F_C / collisionNum * L_MAX;
	}
	vec3 delta_x = solve_delta_x(F_c, F_VC, partial_F_C, partial_F_VC);

	if(l2dis(delta_x)>10)
	{
		delta_x = solve_delta_x_with_degeneration(Xh, Xg, F_c, F_VC, partial_F_C, partial_F_VC);
	}

	// damping and limit max move speed
	delta_x = delta_x * damping;
	if (l2dis(delta_x) > DELTA_X_MAX)
	{
		delta_x = delta_x / l2dis(delta_x) * DELTA_X_MAX;
	}
	return delta_x;
}

Eigen::Matrix<double,6,1> calculator::solve_6DOF(vec3 Xh, vec3 Xg, vec3 Omega_h, vec3 Omega_g,
	vec3 F_c, vec3 T_c,
	mat3 partial_FC_X, mat3 partial_FC_Omega,
	mat3 partial_TC_X, mat3 partial_TC_Omega,
	vec3& updatedX, vec3& updatedOmega,
	int collisionNum, double* times)
{
	// times: 
	float l = 1;
	vec3 Dir_g = calculateToolDir(Omega_g);
	vec3 Dir_h = calculateToolDir(Omega_h);
	vec3 Xg_grasp = Xg + Dir_g * l;
	vec3 Xh_grasp = Xh + Dir_h * l;

	LARGE_INTEGER Freq;
	QueryPerformanceFrequency(&Freq);
	LARGE_INTEGER T1;
	LARGE_INTEGER T2;
	QueryPerformanceCounter(&T1);
	vec3 F_VC = calculateFVC(Xg_grasp, Xh_grasp);
	auto qT = calculate_qt(Omega_h, Omega_g);
	
	vec3 T_VC = -calculateTVC(qT);
	QueryPerformanceCounter(&T2);
	double time_cal_F_and_T = (T2.QuadPart - T1.QuadPart) / (double)Freq.QuadPart*1000;
	times[0] = time_cal_F_and_T;

	if (collisionNum > L_MAX)
	{
		F_c = F_c / collisionNum * L_MAX;
		partial_FC_X = partial_FC_X / collisionNum * L_MAX;
	}
	QueryPerformanceCounter(&T1);
	mat3 partial_FVC_X = calculatePartialFVC_X(Xg_grasp, Xh_grasp);
	mat3 partial_TVC_Omega = calculatePartialTVC_Omega(qT);
	mat3 partial_FVC_Omega = calculatePartialFVC_Omega();// zero matrix
	mat3 partial_TVC_X = calculatePartialTVC_X();// zero matrix
	QueryPerformanceCounter(&T2);
	double time_cal_partial = (T2.QuadPart - T1.QuadPart) / (double)Freq.QuadPart*1000;
	times[1] = time_cal_partial;

	Matrix<double, 6, 6> mat;
	mat.block(0, 0, 3, 3) = partial_FC_X + partial_FVC_X;
	mat.block(0, 3, 3, 3) = partial_FC_Omega + partial_FVC_Omega;
	mat.block(3, 0, 3, 3) = partial_TC_X + partial_TVC_X;
	mat.block(3, 3, 3, 3) = partial_TC_Omega + partial_TVC_Omega;

	vec6 vec;
	vec.block(0, 0, 3, 1) = F_c + F_VC;
	vec.block(3, 0, 3, 1) = T_c + T_VC;
	vec = -vec;

	QueryPerformanceCounter(&T1);
	//vec6 delta = mat.ldlt().solve(vec);
	vec6 delta = mat.lu().solve(vec);
	QueryPerformanceCounter(&T2);
	double time_simple_solve = (T2.QuadPart - T1.QuadPart) / (double)Freq.QuadPart * 1000;
	times[2] = time_simple_solve;

	QueryPerformanceCounter(&T1);
	if (delta.norm() > 10)
	{
		//printf("solve qg with degeneration\n");
		vec6 poseH, poseG;
		poseH.block(0, 0, 3, 1) = Xh_grasp; poseH.block(3, 0, 3, 1) = Omega_h;
		poseG.block(0, 0, 3, 1) = Xg_grasp; poseG.block(3, 0, 3, 1) = Omega_g;
		delta = solve_with_degeneration(poseH, poseG, mat, vec);
	}
	QueryPerformanceCounter(&T2);
	double time_solve_degen = (T2.QuadPart - T1.QuadPart) / (double)Freq.QuadPart * 1000;
	times[3] = time_solve_degen;

	QueryPerformanceCounter(&T1);
	// damping and limit max move speed
	vec3 delta_3dof = delta.block(0, 0, 3, 1);
	delta_3dof = delta_3dof * damping;
	if (l2dis(delta_3dof) > DELTA_X_MAX)
	{
		delta_3dof = delta_3dof / l2dis(delta_3dof) * DELTA_X_MAX;
	}
	// limit max Omega 限制最大角速度
	vec3 dir = Dir_g;
	vec3 deltaOmega = delta.block(3, 0, 3, 1);

	vec3 new_g_omega = Omega_g;
	vec3 limitedDeltaOmega;
	double theta = deltaOmega.norm();
	//limitedDeltaOmega = deltaOmega.normalized() * 0.02;

	if (theta > 1e-6)
	{
		vec3 rotAxis = deltaOmega / theta;
		if (theta > DELTA_ANGLE_MAX)
		{
			theta = DELTA_ANGLE_MAX;
		}
		limitedDeltaOmega = theta * rotAxis;
		new_g_omega = new_g_omega + limitedDeltaOmega;
	}

	delta.block(0, 0, 3, 1) = delta_3dof;
	QueryPerformanceCounter(&T2);
	double time_solve_rotation = (T2.QuadPart - T1.QuadPart) / (double)Freq.QuadPart * 1000;
	times[4] = time_solve_rotation;

	updatedX = Xg + delta_3dof;
	updatedOmega = new_g_omega;
	return delta;
}

mat3 calculator::SkewSymmetricMatrix(vec3 v)
{
	mat3 m;
	m << 0, -v(2), v(1),
		v(2), 0, -v(0),
		-v(1), v(0), 0;
	return m;
}

vec3 calculator::calculateFVC(vec3 Xg, vec3 Xh)
{
	double r = l2dis(Xg, Xh);
	vec3 e_r;
	if (r > EPS)// 避免除零错误
		e_r = (Xg - Xh) / r;
	else
		e_r = vec3(0, 0, 0);
	double f_r = saturationMapping(r);
	vec3 F_VC = -f_r * e_r;
	return F_VC;
}

vec6 calculator::solve_with_degeneration(vec6 poseH, vec6 poseG, mat6 A, vec6 b)
{
	float sigma_threashold = 0.001;

	Eigen::JacobiSVD<mat6> svd(A, ComputeFullU | ComputeFullV);
	mat6 U = svd.matrixU();
	mat6 V = svd.matrixV();
	auto sigma = svd.singularValues();
	mat6 sigmaMatrix;
	sigmaMatrix.setZero();
	sigmaMatrix.diagonal() << sigma;

	int zero_dim_num = 0;
	for (int i = 5; i >= 0; i--)
	{
		if (sigma[i] < sigma_threashold)
			continue;
		else
		{
			zero_dim_num = 5 - i;
			break;
		}
	}
	int non_zero_dim_num = 6 - zero_dim_num;

	vec6 delta_x;
	if (zero_dim_num == 0)
	{
		// degeneration haven't occured, solve the system directly
		printf("degeneration haven't occured, solve the system directly\n");
		delta_x = V * sigmaMatrix.inverse() * U.transpose() * b;
	}
	else
	{
		// degeneration occured. Solve delta_x as two parts: x_d and x_nd.
		// For more information, see in "Real-time Reduced Large-Deformation Models and Distributed Contact for Computer Graphicsand Haptics"
		// p168 sec 4.10
		printf("non zero dim num:%d\n", non_zero_dim_num);
		auto U_nd = U.block(0, 0, 6, non_zero_dim_num);
		auto U_d = U.block(0, non_zero_dim_num, 6, zero_dim_num);
		auto V_nd = V.block(0, 0, 6, non_zero_dim_num);
		auto V_d = V.block(0, non_zero_dim_num, 6, zero_dim_num);
		auto sigmaMatrix_nd = sigmaMatrix.block(0, 0, non_zero_dim_num, non_zero_dim_num);
		auto sigmaMatrix_d = sigmaMatrix.block(non_zero_dim_num, non_zero_dim_num, zero_dim_num, zero_dim_num);
		auto rebuilt_nd = U_nd * sigmaMatrix_nd * V_nd.transpose();
		auto rebuilt_d = U_d * sigmaMatrix_d * V_d.transpose();

		auto x_nd = V_nd * sigmaMatrix_nd.inverse() * U_nd.transpose() * b;
		auto x_d = V_d * V_d.transpose() * (poseH - poseG);

		delta_x = x_nd + x_d;
	}

	return delta_x;
}

vec3 calculator::solve_delta_x_with_degeneration(vec3 Xh, vec3 Xg, vec3 F_C, vec3 F_VC, mat3 partial_F_C, mat3 partial_F_VC)
{
	float sigma_threashold = 0.001;
	mat3 A = partial_F_C + partial_F_VC;
	vec3 b = -(F_C + F_VC);

	Eigen::JacobiSVD<mat3> svd(A, ComputeFullU | ComputeFullV);
	mat3 U = svd.matrixU();
	mat3 V = svd.matrixV();
	auto sigma = svd.singularValues();
	mat3 sigmaMatrix;
	sigmaMatrix.setZero();
	sigmaMatrix.diagonal() << sigma;

	int zero_dim_num = 0;
	if (sigma[2] < sigma_threashold)
	{
		zero_dim_num = 1;
		if (sigma[1] < sigma_threashold)
		{
			zero_dim_num = 2;
			if (sigma[0] < sigma_threashold)
				zero_dim_num = 3;
		}
	}
	int non_zero_dim_num = 3 - zero_dim_num;

	vec3 delta_x;
	if (zero_dim_num == 0)
	{
		// degeneration haven't occured, solve the system directly
		delta_x = V * sigmaMatrix.inverse() * U.transpose() * b;
	}
	else
	{
		// degeneration occured. Solve delta_x as two parts: x_d and x_nd.
		// For more information, see in "Real-time Reduced Large-Deformation Models and Distributed Contact for Computer Graphicsand Haptics"
		// p168 sec 4.10
		auto U_nd = U.block(0, 0, 3, non_zero_dim_num);
		auto U_d = U.block(0, non_zero_dim_num, 3, zero_dim_num);
		auto V_nd = V.block(0, 0, 3, non_zero_dim_num);
		auto V_d = V.block(0, non_zero_dim_num, 3, zero_dim_num);
		auto sigmaMatrix_nd = sigmaMatrix.block(0, 0, non_zero_dim_num, non_zero_dim_num);
		auto sigmaMatrix_d = sigmaMatrix.block(non_zero_dim_num, non_zero_dim_num, zero_dim_num, zero_dim_num);
		auto rebuilt_nd = U_nd * sigmaMatrix_nd * V_nd.transpose();
		auto rebuilt_d = U_d * sigmaMatrix_d * V_d.transpose();

		auto x_nd = V_nd * sigmaMatrix_nd.inverse() * U_nd.transpose() * b;
		auto x_d = V_d * V_d.transpose() * (Xh - Xg);

		delta_x = x_nd + x_d;

	}

	return delta_x;
}

vec3 calculator::calculateR(vec3 p, vec3 Xg_grasp)
{
	vec3 r = p - Xg_grasp;
	return r;
}

vec3 calculator::calculateFC(vec3 N0, double d)
{
	if (d > 0)
	{
		cout << "depth should less than 0!" << endl;
		return vec3(0, 0, 0);
	}
	
	vec3 Fc = m_k_c * d * N0;
	return Fc;
}

vec3 calculator::calculateTC(vec3 r, vec3 Fc)
{
	mat3 r_tilde = SkewSymmetricMatrix(r);
	vec3 Tc = r_tilde * Fc;
	return Tc;
}

mat3 calculator::calculatePartialFC_X(vec3 N0)
{
	mat3 partialFC_X = -m_k_c * N0 * N0.transpose();
	return partialFC_X;
}

mat3 calculator::calculatePartialFC_Omega(vec3 r, vec3 N0, double d)
{
	mat3 r_tilde = SkewSymmetricMatrix(r);
	mat3 part1 = m_k_c * N0 * N0.transpose() * r_tilde;
	mat3 part2 = m_k_c * d * SkewSymmetricMatrix(N0);
	return part1 + part2;
}

mat3 calculator::calculatePartialTC_X(vec3 Fc, vec3 r, vec3 N0)
{
	mat3 part1 = SkewSymmetricMatrix(Fc);
	mat3 part2 = -m_k_c * SkewSymmetricMatrix(r) * (N0 * N0.transpose());
	return part1 + part2;
}

mat3 calculator::calculatePartialTC_Omega(vec3 Fc, vec3 r, vec3 N0, double d)
{
	mat3 r_tilde = SkewSymmetricMatrix(r);
	mat3 part1 = -SkewSymmetricMatrix(Fc) * r_tilde;
	mat3 part2 = m_k_c * r_tilde * (N0 * N0.transpose()) * r_tilde;
	mat3 part3 = m_k_c * d * r_tilde * SkewSymmetricMatrix(N0);
	return part1 + part2 + part3;
}

Point::Point()
{
	pos << 0, 0, 0;
	normal << 0, 0, 0;
}

Point::Point(vec3 p, vec3 n)
	:pos(p), normal(n)
{

}

bool Point::normalize()
{
	normal.normalize();
	return true;
}
