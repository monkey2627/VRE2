#include <string>

extern float* bridge_qg;
extern float* bridge_qh;
extern float* hapticToolTrans;
extern float* virtualToolTrans;
extern bool g_useHapticDevice;
extern float g_cylinderLength;

void PDInit();
void PDUpdate();
void UDLog(std::string mess);
void UDError(std::string mess);
void UDWarning(std::string mess);

void StopHaptic();

void GLInit();
bool GLUpdate();
void GLDestroy();
void ComputeLeftToolForce(float* rt, double* f);
void Draw3DMesh(float* vert, float* norm, float * color, unsigned int* tri,  int triNum);
void Draw3DScene();
void GetRegion(float* region);
void AddFrameCount();
void SetEndoscopePos(float* trans);
void SetToolTipRadius(float r);
void SetCameraLength(float l);
void SetCollisionMode(bool clusterCollision);
float GetSimTime();
float GetHapticTime();
float GetQGTime();
float GetTriTime();
float* GetPointsPtr();
float* GetForceIntensityPtr(int mode);
int GetPointsNum();
int ResetXg(float x, float y, float z);
void SetGravityX(float x);
void SetGravityY(float y);
void SetGravityZ(float z);
float GetGravityX();
float GetGravityY();
float GetGravityZ();
