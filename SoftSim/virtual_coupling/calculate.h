#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>

//#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "typedefs.h"
#include "utils.h"
#include <Eigen/src/Geometry/Quaternion.h>

constexpr auto TOOL_R = 1;

class calculator
{
public:
	calculator();
	~calculator();
	
	void set_k(ValType k_vc, ValType k_c, ValType k_vct=1);

	vec3 calculateFVC(vec3 Xg, vec3 Xh);
	mat3 calculatePartialFVC_X(vec3 qg, vec3 qh);
	mat3 calculatePartialFVC_Omega();
	
	vec3 calculateFC(Point& p, ValType d, ValType k);
	mat3 calculatePartialFC_X(Point& p, ValType k);
	// mat3 calculateParitalFC_Omega(Point& p, vec3 qg);
	
	ValType saturationMapping(ValType r);
	ValType saturationMappingDerivative(ValType r);

	vec3 solve(vec3 Xh, vec3 Xg, vec3 F_c, mat3 partial_F_C, int collisionNum);
	Eigen::Matrix<double,6,1> solve_6DOF(vec3 Xh, vec3 Xg, vec3 Omega_h, vec3 Omega_g,
		vec3 F_c, vec3 T_c, 
		mat3 partial_FC_X, mat3 partial_FC_Omega,
		mat3 partial_TC_X, mat3 partial_TC_Omega, 
		vec3& updatedX, vec3& updatedDir,
		int collisionNum, double* times);

	mat3 SkewSymmetricMatrix(vec3 v);

	Eigen::Quaterniond calculate_qt(vec3 Omega_h, vec3 Omega_g);
	vec3 calculateTVC(Eigen::Quaterniond qt);
	mat3 calculatePartialTVC_Omega(Eigen::Quaterniond qt);
	mat3 calculatePartialTVC_X();

	mat3 genTildeMatrix(vec3 r);

	vec3 calculateToolDir(vec3 Omega);
	Eigen::Quaterniond Omega2Quaternion(vec3 Omega);
	vec3 Quaternion2Omega(Eigen::Quaterniond q);

	vec3 calculateR(vec3 p, vec3 Xg_grasp);
	vec3 calculateFC(vec3 N0, double d);
	vec3 calculateTC(vec3 r, vec3 Fc);
	mat3 calculatePartialFC_X(vec3 N0);
	mat3 calculatePartialFC_Omega(vec3 r, vec3 N0, double d);
	mat3 calculatePartialTC_X(vec3 Fc, vec3 r, vec3 N0);
	mat3 calculatePartialTC_Omega(vec3 Fc, vec3 r, vec3 N0, double d);
private:
	bool PRINT_INFO;
	vec3 solve_delta_x(vec3 F_C, vec3 F_VC, mat3 partial_F_C, mat3 partial_F_VC);
	vec3 solve_delta_x_with_degeneration(vec3 Xh, vec3 Xg, vec3 F_C, vec3 F_VC, mat3 partial_F_C, mat3 partial_F_VC);
	vec6 solve_with_degeneration(vec6 poseH, vec6 poseG, mat6 A, vec6 b);
	double _r_lin;
	ValType m_k_vc;
	ValType m_k_c;
	ValType m_k_vct;

	vec3 m_F_VC;
	mat3 m_partial_FVC_X;

	vec3 m_T_VC;
	mat3 m_partial_FVC_Omega;
};