#pragma once
#include "calculate.h"
class Haptic {
public:
	float m_radius = 0.2;
	float m_cylinderLength = 30;
	float m_collisionStiffness = 2500;
	float m_dampingForTriVert = 0.2f;
	float m_rho = 0.9992f;
	calculator m_virtualCoupling;
	enum CollisionMode { TETVERT_MODE, TRIVERT_MODE, MERGE_MODE, CYLINDER_MERGE_MODE };
	std::vector<float> m_qg; // 6-dim [x, y, z, OmegaX, OmegaY, OmegaZ]
	std::vector<float> m_qh; // 6-dim [x, y, z, OmegaX, OmegaY, OmegaZ]
	std::vector<float> m_dir_g;
	std::vector<float> m_dir_h;

	std::vector<float> m_hapticToolTrans;
	std::vector<float> m_virtualToolTrans;

	float m_kVc = 5;
	float m_kc = 10;
	float m_kVct = 10;
	int m_hapticIterNum = 6;
	float toolDir[3];
	double F_vc[6]; // 虚拟匹配力，将虚拟工具对齐到物理工具上的力。从虚拟工具指向物理工具，与输出到力反馈工具上的力方向相反

	bool inPDFashion = false;
	double m_hapticStepOpTime;
	double m_hapticTriOpTime;
	double m_hapticQGOpTime;
	double m_haptic6dofTime;
	unsigned int m_hapticStepNum = 0;
	unsigned int m_renderStepNumPassed = 0;
	bool m_useTetTriInteraction = true;

	void Init();
	void SetQH(const float* toolTrans);
	void HapticStep();
	void HapticCollision(CollisionMode mode);
	void UpdateQG();
	void UpdateQGwithChebyshev(float omega);

	void ComputeLeftToolForce(const float* trans, double* force);
	// Output
	std::vector<float> m_F_vc; // 6_dim [Fx, Fy, Fz, Tx, Ty, Tz]
};