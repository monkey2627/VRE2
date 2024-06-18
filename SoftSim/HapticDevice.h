#include <Windows.h>
#include "mat44.h"
#define TransSize 64
#define  MotionScale 0.1f
static const int AngleDelta = 1;
static const int MaxAngleI = 60;
static const unsigned char BIT1 = 0x1;
static const unsigned char BIT2 = 0x2;
static const unsigned char BIT3 = 0x4;
static const unsigned char BIT4 = 0x8;
static const unsigned char BIT5 = 0x10;
static const unsigned char BIT6 = 0x20;
static const unsigned char BIT7 = 0x40;
static const unsigned char BIT8 = 0x80;

static const unsigned short BIT9 = 0x100;
static const unsigned short BIT10 = 0x200;
static const unsigned short BIT11 = 0x400;
static const unsigned short BIT12 = 0x800;
static const unsigned short BIT13 = 0x1000;
static const unsigned short BIT14 = 0x2000;
static const unsigned short BIT15 = 0x4000;
static const unsigned short BIT16 = 0x8000;

struct HapticTranslator {
	////对力反馈设备进行移动，从而能够更加贴近器官
	std::string configFile;
	Matrix44 trans2Organ;
	Matrix44 trans2World;
	Matrix44 deviceTrans;
	bool ready;
	void LoadTransFormFile();
	void Trans2Organ(float* r);
	void TransForce2World(float* f);
	void TransForce2World(double* f);
	Matrix44 GenMatrixform3dxMax(float rx, float ry, float rz, float tx, float ty, float tz);
	void GenMatrixfromUnity(float rx, float ry, float rz, float tx, float ty, float tz);
	HapticTranslator();
};

class HapticDevice {

public:
	std::string m_leftDeviceName = "Default Device";
	std::string m_rightDeviceName = "RightFeeli";
	HMODULE  m_deviceDriverDLL;
	struct HapticInfor {
		HapticTranslator translator;
		float M[16];
		unsigned char State[5];
		unsigned short Encoders[16];
		unsigned short Encoders2[16];
		int deviceHandle = -1;
		int bt0, bt1;
		int anglei = 0;
	};

	HapticInfor m_leftDevice;
	HapticInfor m_rightDevice;


	float m_leftBuffer[TransSize];
	float m_rightBuffer[TransSize];

	bool m_stopFlag = false;
	bool m_hapticThreadRunFlag = false;
	///////通用
	int(*createServoLoop)();
	int(*stopServoLoop)();
	int(*destroyServoLoop)();
	int(*init_phantom)(const char* configname);
	int(*disable_phantom)(unsigned int index);
	int(*startServoLoop)(int(_stdcall* fntServo)(void*), void* lpParam);
	int(*get_stylus_matrix)(unsigned int index, float(*matrix)[16]);
	int(*update_phantom)(unsigned int index);
	int(*enable_phantom_forces)(unsigned int index);
	int(*disable_phantom_forces)(unsigned int index);
	int(*send_phantom_force)(unsigned int index, const float forces[3]);
	////不用的
	int(*is_phantom_forces_enabled)(unsigned int index);
	int(*get_phantom_pos)(unsigned int index, float pos[3]);
	int(*update_calibration)(unsigned int index);
	int(*get_phantom_joint_angles)(unsigned int index, float angles[6]);
	int(*phantom_status)(unsigned int index);
	int(*command_motor_dac_values)(unsigned int index, long MotorDACValues[6]);
	int(*get_phantom_info)(unsigned int index, void* pPhantomInfo, unsigned int size);
	int(*get_encoder_values)(unsigned int index, long* a2);

	///////Phantom
	int(*get_stylus_switch)(unsigned int index, int no);

	void InitHapticDevice();
	void StopHapticDevice();
};