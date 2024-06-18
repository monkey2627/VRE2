#include "manager.h"
#include "bridge.h"

Manager g_manager;

float* bridge_qg;
float* bridge_qh;
float* hapticToolTrans;
float* virtualToolTrans;

bool g_useHapticDevice;

void UDLog(std::string mess) {}
void UDError(std::string mess) {}
void UDWarning(std::string mess) {}

void PDInit()
{
	g_manager.Init();
	bridge_qg = new float[6];
	bridge_qh = new float[6];
	hapticToolTrans = new float[16];
	virtualToolTrans = new float[16];
	g_useHapticDevice = true;
}

void PDUpdate()
{
	float dt = 1.8f / 60.0f;
	g_manager.m_softHapticSolver.Step(dt);
	g_manager.m_vcHaptic.m_renderStepNumPassed = g_manager.m_softHapticSolver.renderStepNumPassed;

	if (g_useHapticDevice)
	{
		bridge_qg[0] = g_manager.m_vcHaptic.m_qg[0];
		bridge_qg[1] = g_manager.m_vcHaptic.m_qg[1];
		bridge_qg[2] = g_manager.m_vcHaptic.m_qg[2];
		bridge_qg[3] = g_manager.m_vcHaptic.m_qg[3];
		bridge_qg[4] = g_manager.m_vcHaptic.m_qg[4];
		bridge_qg[5] = g_manager.m_vcHaptic.m_qg[5];

		bridge_qh[0] = g_manager.m_vcHaptic.m_qh[0];
		bridge_qh[1] = g_manager.m_vcHaptic.m_qh[1];
		bridge_qh[2] = g_manager.m_vcHaptic.m_qh[2];
		bridge_qh[3] = g_manager.m_vcHaptic.m_qh[3];
		bridge_qh[4] = g_manager.m_vcHaptic.m_qh[4];
		bridge_qh[5] = g_manager.m_vcHaptic.m_qh[5];

		memcpy(hapticToolTrans, g_manager.m_vcHaptic.m_hapticToolTrans.data(), 16 * sizeof(float));
		memcpy(virtualToolTrans, g_manager.m_vcHaptic.m_virtualToolTrans.data(), 16 * sizeof(float));
	}
	else
	{
		// 因为帧率不一样，在这种操作方式中，虚拟工具在感觉上会更慢的和物理工具位置对齐
		g_manager.m_vcHaptic.m_qh[0] = bridge_qh[0];
		g_manager.m_vcHaptic.m_qh[1] = bridge_qh[1];
		g_manager.m_vcHaptic.m_qh[2] = bridge_qh[2];

		g_manager.m_vcHaptic.HapticStep();

		bridge_qg[0] = g_manager.m_vcHaptic.m_qg[0];
		bridge_qg[1] = g_manager.m_vcHaptic.m_qg[1];
		bridge_qg[2] = g_manager.m_vcHaptic.m_qg[2];
	}
	SetEndoscopePos(g_manager.m_softHapticSolver.m_toolTrans);

}

void StopHaptic()
{
	g_manager.m_hapticDevice.StopHapticDevice();
}

float GetSimTime() {
	return g_manager.m_softHapticSolver.m_opTime;
}

float GetHapticTime()
{
	return g_manager.m_vcHaptic.m_hapticStepOpTime;
}

float GetQGTime()
{
	return g_manager.m_vcHaptic.m_hapticQGOpTime;
}
float GetTriTime()
{
	return g_manager.m_vcHaptic.m_hapticTriOpTime;
}

void SetToolTipRadius(float r) {
	g_manager.m_softHapticSolver.m_radius = r;
}

void SetCameraLength(float l) {
	g_manager.m_softHapticSolver.toolLength = l;
}

void SetCollisionMode(bool clusterCollision)
{
	g_manager.m_softHapticSolver.m_useClusterCollision = clusterCollision;
}


void GetRegion(float* region) {
	int tvnum = g_manager.m_softHapticSolver.m_tetVertPos.size() / 3;
	region[0] = FLT_MAX;
	region[1] = FLT_MAX;
	region[2] = FLT_MAX;
	region[3] = -FLT_MAX;
	region[4] = -FLT_MAX;
	region[5] = -FLT_MAX;
	for (int i = 0; i < tvnum; i++) {
		int j = i * 3;
		float x = g_manager.m_softHapticSolver.m_tetVertPos[j];
		float y = g_manager.m_softHapticSolver.m_tetVertPos[j+1];
		float z = g_manager.m_softHapticSolver.m_tetVertPos[j+2];
		region[0] = std::min(x, region[0]);
		region[1] = std::min(y, region[1]);
		region[2] = std::min(z, region[2]);

		region[3] = std::max(x, region[3]);
		region[4] = std::max(y, region[4]);
		region[5] = std::max(z, region[5]);
	}
}

void Draw3DScene() {
	Draw3DMesh(g_manager.m_softHapticSolver.m_triVertPos.data(),
		g_manager.m_softHapticSolver.m_triVertNorm.data(),
		g_manager.m_softHapticSolver.m_triVertColor.data(),
		g_manager.m_softHapticSolver.m_triIndex.data(),
		g_manager.m_softHapticSolver.m_triIndex.size());

}

float* GetPointsPtr() 
{
	return g_manager.m_softHapticSolver.m_tetVertPos.data();
}

float* GetForceIntensityPtr(int mode)
{
	if (mode == 0)
		return g_manager.m_softHapticSolver.m_tetVertVolumnForceLen.data();
	else if (mode == 1)
		return g_manager.m_softHapticSolver.m_tetVertCollisionForceLen.data();
}

int GetPointsNum() 
{
	return g_manager.m_softHapticSolver.m_tetVertPos.size()/3;
}

int ResetXg(float x, float y, float z)
{
	g_manager.m_vcHaptic.m_qg[0] = x;
	g_manager.m_vcHaptic.m_qg[1] = y;
	g_manager.m_vcHaptic.m_qg[2] = z;
	return 0;
}

void SetGravityX(float x) {
	g_manager.m_softHapticSolver.m_gravityX = x;
}

void SetGravityY(float y) {
	g_manager.m_softHapticSolver.m_gravityY = y;
}

void SetGravityZ(float z) {
	g_manager.m_softHapticSolver.m_gravityZ = z;
}

float GetGravityX() {
	return g_manager.m_softHapticSolver.m_gravityX;
}

float GetGravityY() {
	return g_manager.m_softHapticSolver.m_gravityY;
}

float GetGravityZ() {
	return g_manager.m_softHapticSolver.m_gravityZ;
}

void ComputeLeftToolForce(float* rt, double* f) {
	return g_manager.m_vcHaptic.ComputeLeftToolForce(rt, f);
}
//
int main(int argc, char* argv[]) {
	g_manager.Init();
	bridge_qg = new float[6];
	bridge_qh = new float[6];
	hapticToolTrans = new float[16];
	virtualToolTrans = new float[16];
	g_useHapticDevice = true;
	GLInit();
	do {
		PDUpdate();
		AddFrameCount();
	} while (GLUpdate());
	StopHaptic();
	GLDestroy();
	return 0;
}