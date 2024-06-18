#include "bridge.h"
#include "HapticDevice.h"


void Mul44(const Matrix44& left, const Matrix44& right, Matrix44& res) {
	res[0] = left[0] * right[0] + left[4] * right[1] + left[8] * right[2] + left[12] * right[3];
	res[1] = left[1] * right[0] + left[5] * right[1] + left[9] * right[2] + left[13] * right[3];
	res[2] = left[2] * right[0] + left[6] * right[1] + left[10] * right[2] + left[14] * right[3];
	res[3] = left[3] * right[0] + left[7] * right[1] + left[11] * right[2] + left[15] * right[3];

	res[4] = left[0] * right[4] + left[4] * right[5] + left[8] * right[6] + left[12] * right[7];
	res[5] = left[1] * right[4] + left[5] * right[5] + left[9] * right[6] + left[13] * right[7];
	res[6] = left[2] * right[4] + left[6] * right[5] + left[10] * right[6] + left[14] * right[7];
	res[7] = left[3] * right[4] + left[7] * right[5] + left[11] * right[6] + left[15] * right[7];

	res[8] = left[0] * right[8] + left[4] * right[9] + left[8] * right[10] + left[12] * right[11];
	res[9] = left[1] * right[8] + left[5] * right[9] + left[9] * right[10] + left[13] * right[11];
	res[10] = left[2] * right[8] + left[6] * right[9] + left[10] * right[10] + left[14] * right[11];
	res[11] = left[3] * right[8] + left[7] * right[9] + left[11] * right[10] + left[15] * right[11];

	res[12] = left[0] * right[12] + left[4] * right[13] + left[8] * right[14] + left[12] * right[15];
	res[13] = left[1] * right[12] + left[5] * right[13] + left[9] * right[14] + left[13] * right[15];
	res[14] = left[2] * right[12] + left[6] * right[13] + left[10] * right[14] + left[14] * right[15];
	res[15] = left[3] * right[12] + left[7] * right[13] + left[11] * right[14] + left[15] * right[15];
}

void HapticTranslator::LoadTransFormFile() {
	///////////////�� 3ds Max ����������
	FILE* fid = fopen(configFile.c_str(), "r");
	if (!fid)
		return;
	float toolRx, toolRy, toolRz;
	float toolTx, toolTy, toolTz;
	float organRx, organRy, organRz;
	float organTx, organTy, organTz;

	fscanf(fid, "%f", &toolRx);
	fscanf(fid, "%f", &toolRy);
	fscanf(fid, "%f", &toolRz);

	fscanf(fid, "%f", &toolTx);
	fscanf(fid, "%f", &toolTy);
	fscanf(fid, "%f", &toolTz);

	fscanf(fid, "%f", &organRx);
	fscanf(fid, "%f", &organRy);
	fscanf(fid, "%f", &organRz);

	fscanf(fid, "%f", &organTx);
	fscanf(fid, "%f", &organTy);
	fscanf(fid, "%f", &organTz);
	fclose(fid);
	//////�ӽǶ�ת������
	organRx *= 0.017453292519943f;
	organRy *= 0.017453292519943f;
	organRz *= 0.017453292519943f;
	toolRx *= 0.017453292519943f;
	toolRy *= 0.017453292519943f;
	toolRz *= 0.017453292519943f;

	/////////���������ƶ��ľ���
	deviceTrans = GenMatrixform3dxMax(toolRx, toolRy, toolRz, toolTx, toolTy, toolTz);
	Matrix44 organrt_inv = GenMatrixform3dxMax(organRx, organRy, organRz, organTx, organTy, organTz);
	Matrix44 organrt = AffineInverse(organrt_inv);
	Mul44(organrt, deviceTrans, trans2Organ);
	trans2World = AffineInverse(trans2Organ);
	ready = true;
}

void HapticTranslator::Trans2Organ(float* res) {

	res[12] *= MotionScale;
	res[13] *= MotionScale;
	res[14] *= MotionScale;

	if (!ready)
		return;
	float right[32];
	memcpy(right, res, 16 * sizeof(float));
	Matrix44& left = trans2Organ;
	res[0] = left[0] * right[0] + left[4] * right[1] + left[8] * right[2] + left[12] * right[3];
	res[1] = left[1] * right[0] + left[5] * right[1] + left[9] * right[2] + left[13] * right[3];
	res[2] = left[2] * right[0] + left[6] * right[1] + left[10] * right[2] + left[14] * right[3];
	res[3] = left[3] * right[0] + left[7] * right[1] + left[11] * right[2] + left[15] * right[3];

	res[4] = left[0] * right[4] + left[4] * right[5] + left[8] * right[6] + left[12] * right[7];
	res[5] = left[1] * right[4] + left[5] * right[5] + left[9] * right[6] + left[13] * right[7];
	res[6] = left[2] * right[4] + left[6] * right[5] + left[10] * right[6] + left[14] * right[7];
	res[7] = left[3] * right[4] + left[7] * right[5] + left[11] * right[6] + left[15] * right[7];

	res[8] = left[0] * right[8] + left[4] * right[9] + left[8] * right[10] + left[12] * right[11];
	res[9] = left[1] * right[8] + left[5] * right[9] + left[9] * right[10] + left[13] * right[11];
	res[10] = left[2] * right[8] + left[6] * right[9] + left[10] * right[10] + left[14] * right[11];
	res[11] = left[3] * right[8] + left[7] * right[9] + left[11] * right[10] + left[15] * right[11];

	res[12] = left[0] * right[12] + left[4] * right[13] + left[8] * right[14] + left[12] * right[15];
	res[13] = left[1] * right[12] + left[5] * right[13] + left[9] * right[14] + left[13] * right[15];
	res[14] = left[2] * right[12] + left[6] * right[13] + left[10] * right[14] + left[14] * right[15];
	res[15] = left[3] * right[12] + left[7] * right[13] + left[11] * right[14] + left[15] * right[15];
}

void HapticTranslator::TransForce2World(float* f) {
	if (!ready) return;
	float xin = f[0]; float yin = f[1]; float zin = f[2];
	f[0] = xin * trans2World[0] + yin * trans2World[4] + zin * trans2World[8];
	f[1] = xin * trans2World[1] + yin * trans2World[5] + zin * trans2World[9];
	f[2] = xin * trans2World[2] + yin * trans2World[6] + zin * trans2World[10];
}

void HapticTranslator::TransForce2World(double* f) {
	if (!ready) return;
	float xin = f[0]; float yin = f[1]; float zin = f[2];
	f[0] = xin * trans2World[0] + yin * trans2World[4] + zin * trans2World[8];
	f[1] = xin * trans2World[1] + yin * trans2World[5] + zin * trans2World[9];
	f[2] = xin * trans2World[2] + yin * trans2World[6] + zin * trans2World[10];
}

Matrix44 HapticTranslator::GenMatrixform3dxMax(float rx, float ry, float rz, float tx, float ty, float tz) {
	//����x��ʱ�룬����y��ʱ�룬����z��ʱ��
	float rotate[3][3];
	double sin_x = sin(rx);
	double cos_x = cos(rx);
	double sin_y = sin(ry);
	double cos_y = cos(ry);
	double sin_z = sin(rz);
	double cos_z = cos(rz);

	rotate[0][0] = cos_z * cos_y;
	rotate[0][1] = -sin_z * cos_x + cos_z * sin_y * sin_x;
	rotate[0][2] = sin_z * sin_x + cos_z * sin_y * cos_x;

	rotate[1][0] = sin_z * cos_y;
	rotate[1][1] = cos_z * cos_x + sin_z * sin_y * sin_x;
	rotate[1][2] = -sin_x * cos_z + cos_x * sin_y * sin_z;

	rotate[2][0] = -sin_y;
	rotate[2][1] = sin_x * cos_y;
	rotate[2][2] = cos_x * cos_y;

	Matrix44 gltrans;
	//gltrans[0] = rotate[0][0];
	//gltrans[1] = rotate[0][1];
	//gltrans[2] = rotate[0][2];
	//gltrans[3] = 0.0f;
	//gltrans[4] = rotate[1][0];
	//gltrans[5] = rotate[1][1];
	//gltrans[6] = rotate[1][2];
	//gltrans[7] = 0.0f;
	//gltrans[8] = rotate[2][0];
	//gltrans[9] = rotate[2][1];
	//gltrans[10] = rotate[2][2];
	//gltrans[11] = 0.0f;
	//gltrans[12] = tx;
	//gltrans[13] = ty;
	//gltrans[14] = tz;
	//gltrans[15] = 1.0f;

	gltrans[0] = rotate[0][0];
	gltrans[1] = rotate[1][0];
	gltrans[2] = rotate[2][0];
	gltrans[3] = 0.0f;
	gltrans[4] = rotate[0][1];
	gltrans[5] = rotate[1][1];
	gltrans[6] = rotate[2][1];
	gltrans[7] = 0.0f;
	gltrans[8] = rotate[0][2];
	gltrans[9] = rotate[1][2];
	gltrans[10] = rotate[2][2];
	gltrans[11] = 0.0f;
	gltrans[12] = tx;
	gltrans[13] = ty;
	gltrans[14] = tz;
	gltrans[15] = 1.0f;

	return gltrans;
}

void HapticTranslator::GenMatrixfromUnity(float rx, float ry, float rz, float tx, float ty, float tz) {
	float rotate[3][3];
	double sin_x = sin(rx);
	double cos_x = cos(rx);
	double sin_y = sin(ry);
	double cos_y = cos(ry);
	double sin_z = sin(rz);
	double cos_z = cos(rz);

	rotate[0][0] = cos_z * cos_y;
	rotate[0][1] = -sin_z * cos_x + cos_z * sin_y * sin_x;
	rotate[0][2] = sin_z * sin_x + cos_z * sin_y * cos_x;

	rotate[1][0] = sin_z * cos_y;
	rotate[1][1] = cos_z * cos_x + sin_z * sin_y * sin_x;
	rotate[1][2] = -sin_x * cos_z + cos_x * sin_y * sin_z;

	rotate[2][0] = -sin_y;
	rotate[2][1] = sin_x * cos_y;
	rotate[2][2] = cos_x * cos_y;

	trans2Organ[0] = rotate[0][0];
	trans2Organ[1] = rotate[1][0];
	trans2Organ[2] = rotate[2][0];
	trans2Organ[3] = 0.0f;
	trans2Organ[4] = rotate[0][1];
	trans2Organ[5] = rotate[1][1];
	trans2Organ[6] = rotate[2][1];
	trans2Organ[7] = 0.0f;
	trans2Organ[8] = rotate[0][2];
	trans2Organ[9] = rotate[1][2];
	trans2Organ[10] = rotate[2][2];
	trans2Organ[11] = 0.0f;
	trans2Organ[12] = tx * 1000.0f;
	trans2Organ[13] = ty * 1000.0f;
	trans2Organ[14] = tz * 1000.0f;
	trans2Organ[15] = 1.0f;
	trans2World = AffineInverse(trans2Organ);
	ready = true;
}

HapticTranslator::HapticTranslator() {
	trans2Organ = Matrix44::kIdentity;
	trans2World = Matrix44::kIdentity;
	ready = false;
}

int _stdcall SetHapticState(void* pParam) {
	HapticDevice* h = (HapticDevice*)pParam;
	if (!h->m_hapticThreadRunFlag)
		UDLog("�������߳̿�ʼ����");

	h->m_hapticThreadRunFlag = true;

	if (h->m_stopFlag) {
		UDLog("�������߳�׼���˳�");
		Sleep(50);
		return -1;
	}
	int lh = h->m_leftDevice.deviceHandle;
	int rh = h->m_rightDevice.deviceHandle;

	if (lh >= 0) {
		h->update_phantom(lh);
		h->get_stylus_matrix(lh, &h->m_leftDevice.M);
		memcpy(h->m_leftBuffer, h->m_leftDevice.M, 16 * sizeof(float));
		h->m_leftDevice.translator.Trans2Organ(h->m_leftBuffer);
		h->m_leftDevice.bt0 = h->get_stylus_switch(lh, 0);
		h->m_leftDevice.bt1 = h->get_stylus_switch(lh, 1);
		if (h->m_leftDevice.bt0) {
			h->m_leftDevice.anglei += AngleDelta;
			h->m_leftDevice.anglei = min(h->m_leftDevice.anglei, MaxAngleI);
		}
		else {
			h->m_leftDevice.anglei -= AngleDelta;
			h->m_leftDevice.anglei = max(h->m_leftDevice.anglei, 0);
		}
		h->m_leftBuffer[16] = h->m_leftDevice.anglei;
		h->m_leftBuffer[17] = 0;
		h->m_leftBuffer[18] = h->m_leftDevice.bt1;
	}///////����������

	if (g_useHapticDevice)
	{
		double f[3];
		///�˴�������
		ComputeLeftToolForce(h->m_leftBuffer, f);
		h->m_leftDevice.translator.TransForce2World(f);
		float ff[3];
		ff[0] = f[0]; ff[1] = f[1]; ff[2] = f[2];
		h->send_phantom_force(lh, ff);
	}

	if (rh >= 0) {
		h->update_phantom(rh);
		h->get_stylus_matrix(rh, &h->m_rightDevice.M);
		memcpy(h->m_rightBuffer, h->m_rightDevice.M, 16 * sizeof(float));
		h->m_rightDevice.translator.Trans2Organ(h->m_rightBuffer);
		h->m_rightDevice.bt0 = h->get_stylus_switch(rh, 0);
		h->m_rightDevice.bt1 = h->get_stylus_switch(rh, 1);
		if (h->m_rightDevice.bt0) {
			h->m_rightDevice.anglei += AngleDelta;
			h->m_rightDevice.anglei = min(h->m_rightDevice.anglei, MaxAngleI);
		}
		else {
			h->m_rightDevice.anglei -= AngleDelta;
			h->m_rightDevice.anglei = max(h->m_rightDevice.anglei, 0);
		}
		h->m_rightBuffer[16] = h->m_rightDevice.anglei;
		h->m_rightBuffer[17] = 0;
		h->m_rightBuffer[18] = h->m_rightDevice.bt1;

		double f[3];
		//�˴�������h->m_lpFlexiblescene->m_collisionManager.ComputeRightToolForce(h->m_rightBuffer, f);
		h->m_rightDevice.translator.TransForce2World(f);
		float ff[3];
		ff[0] = f[0]; ff[1] = f[1]; ff[2] = f[2];
		h->send_phantom_force(rh, ff);

	}///////����������
	return 0;
}

void HapticDevice::InitHapticDevice() {

	m_deviceDriverDLL = LoadLibrary(L"PhantomIoLib42.dll");
	if (m_deviceDriverDLL == NULL) {
		UDError("�޷���PhantomIoLib42.dll");
	}
	else {
		UDLog("��PhantomIoLib42.dll�ɹ�");
	}

	if (m_deviceDriverDLL != NULL) {
		//��������ѭ��
		createServoLoop = (int(*)()) GetProcAddress(m_deviceDriverDLL, "createServoLoop");
		//ֹͣ����ѭ��
		stopServoLoop = (int(*)())GetProcAddress(m_deviceDriverDLL, "stopServoLoop");
		//���ٷ���ѭ��
		destroyServoLoop = (int(*)())GetProcAddress(m_deviceDriverDLL, "destroyServoLoop");
		//��ʼ���豸
		init_phantom = (int(*)(const char* configname))GetProcAddress(m_deviceDriverDLL, "init_phantom");
		//�Ƴ��豸
		disable_phantom = (int(*)(unsigned int index))GetProcAddress(m_deviceDriverDLL, "disable_phantom");
		//��ʼ����ѭ��
		startServoLoop = (int(*)(int(_stdcall * fntServo)(void*), void* lpParam))GetProcAddress(m_deviceDriverDLL, "startServoLoop");
		//��������
		update_phantom = (int(*)(unsigned int index))GetProcAddress(m_deviceDriverDLL, "update_phantom");
		//����
		update_calibration = (int(*)(unsigned int index))GetProcAddress(m_deviceDriverDLL, "update_calibration");
		//ʹ����
		enable_phantom_forces = (int(*)(unsigned int index))GetProcAddress(m_deviceDriverDLL, "enable_phantom_forces");
		//������
		disable_phantom_forces = (int(*)(unsigned int index))GetProcAddress(m_deviceDriverDLL, "disable_phantom_forces");
		//ʩ����
		send_phantom_force = (int(*)(unsigned int index, const float forces[3]))GetProcAddress(m_deviceDriverDLL, "send_phantom_force");
		//�仯����
		get_stylus_matrix = (int(*)(unsigned int index, float(*matrix)[16]))GetProcAddress(m_deviceDriverDLL, "get_stylus_matrix");


		if (createServoLoop == nullptr)
			UDError("�޷���ȡcreateServoLoop���");
		else
			UDLog("createServoLoop��ڻ�ȡ�ɹ�");

		if (stopServoLoop == nullptr)
			UDError("�޷���ȡstopServoLoop���");
		else
			UDLog("stopServoLoop��ڻ�ȡ�ɹ�");

		if (destroyServoLoop == nullptr)
			UDError("�޷���ȡdestroyServoLoop���");
		else
			UDLog("destroyServoLoop��ڻ�ȡ�ɹ�");

		if (init_phantom == nullptr)
			UDError("�޷���ȡinit_phantom���");
		else
			UDLog("init_phantom��ڻ�ȡ�ɹ�");

		if (disable_phantom == nullptr)
			UDError("�޷���ȡdisable_phantom���");
		else
			UDLog("disable_phantom��ڻ�ȡ�ɹ�");

		if (startServoLoop == nullptr)
			UDError("�޷���ȡstartServoLoop���");
		else
			UDLog("startServoLoop��ڻ�ȡ�ɹ�");

		if (update_phantom == nullptr)
			UDError("�޷���ȡupdate_phantom���");
		else
			UDLog("update_phantom��ڻ�ȡ�ɹ�");

		if (enable_phantom_forces == nullptr)
			UDError("�޷���ȡenable_phantom_forces���");
		else
			UDLog("enable_phantom_forces��ڻ�ȡ�ɹ�");

		if (disable_phantom_forces == nullptr)
			UDError("�޷���ȡdisable_phantom_forces���");
		else
			UDLog("createServoLoop��ڻ�ȡ�ɹ�");

		if (send_phantom_force == nullptr)
			UDError("�޷���ȡsend_phantom_force���");
		else
			UDLog("send_phantom_force��ڻ�ȡ�ɹ�");

		if (get_stylus_matrix == nullptr)
			UDError("�޷���ȡget_stylus_matrix���");
		else
			UDLog("get_stylus_matrix��ڻ�ȡ�ɹ�");
	}

	get_stylus_switch = (int(*)(unsigned int index, int no))GetProcAddress(m_deviceDriverDLL, "get_stylus_switch");
	if (get_stylus_switch == nullptr)
		UDError("�޷���ȡget_stylus_switch���");
	else
		UDLog("get_stylus_switch��ڻ�ȡ�ɹ�");

	if (m_deviceDriverDLL && createServoLoop) {
		createServoLoop();
	}

	if (m_deviceDriverDLL && init_phantom) {
		m_leftDevice.deviceHandle = init_phantom(m_leftDeviceName.c_str());
		m_rightDevice.deviceHandle = init_phantom(m_rightDeviceName.c_str());
	}

	if (m_leftDevice.deviceHandle < 0) {
		UDError("�޷���"+ m_leftDeviceName);
	}
	else {
		UDLog("��"+ m_leftDeviceName+"�ɹ�");
		enable_phantom_forces(m_leftDevice.deviceHandle);
	}


	if (m_rightDevice.deviceHandle < 0) {
		UDError("�޷���" + m_rightDeviceName);
	}
	else {
		UDLog("��" + m_rightDeviceName + "�ɹ�");
		enable_phantom_forces(m_rightDevice.deviceHandle);
	}

	if (m_deviceDriverDLL && startServoLoop) {
		startServoLoop(SetHapticState, this);
	}

	if (m_leftDevice.deviceHandle == -1 && m_rightDevice.deviceHandle == -1) {
		UDLog("�����˶��������");
	}


}

void HapticDevice::StopHapticDevice() {
	if (m_deviceDriverDLL != NULL && !m_stopFlag) {
		m_stopFlag = true;
		Sleep(50);
		if (stopServoLoop && m_hapticThreadRunFlag) {
			stopServoLoop();
			UDLog("stopServoLoop����");
		}
		if (destroyServoLoop) {
			destroyServoLoop();
			UDLog("destroyServoLoop����");
		}
		if (disable_phantom_forces) {
			disable_phantom_forces(m_leftDevice.deviceHandle);
			disable_phantom_forces(m_rightDevice.deviceHandle);
			UDLog("disable_phantom_forces����");
		}
		if (disable_phantom) {
			disable_phantom(m_leftDevice.deviceHandle);
			disable_phantom(m_rightDevice.deviceHandle);
			UDLog("disable_phantom����");
		}
		UDLog("�������豸ֹͣ");
	}
}