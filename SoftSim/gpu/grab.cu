#include "gpuvar.h"
#include "gpufun.h"

//��ת
__device__ float* cylinderShiftMU;
__device__ float* cylinderLastPosMU;
__device__ float* cylinderPosMU;
__device__ float* cylinderGraphicalPosMU;
__device__ float* cylinderDirZMU;
__device__ float* cylinderDirYMU;
__device__ float* cylinderDirXMU;
__device__ float* cylinderVMU;
__device__ float* relativePositionMU;
__device__ unsigned int* isGrapMU;
__device__ unsigned int* isGrapHalfMU;
__device__ float* adsorbForceMU;
__device__ float* grapperUpDirXMU;
__device__ float* grapperUpDirYMU;
__device__ float* grapperUpDirZMU;
__device__ float* grapperDownDirXMU;
__device__ float* grapperDownDirYMU;
__device__ float* grapperDownDirZMU;
__device__ unsigned int* collideFlag;

extern "C" int runcalculateCollisionCylinderMU(
	float length, float radius,
	float collisionStiffness, float adsorbStiffness,
	int flag)
{

	//ѡȡ�����ֹ���
	int		cylinderButton;
	bool* firstGrab;

	if (flag == 1) {  //Left
		cylinderShiftMU = cylinderShiftLeft_D;
		cylinderLastPosMU = cylinderLastPosLeft_D;
		cylinderPosMU = cylinderPosLeft_D;
		cylinderGraphicalPosMU = cylinderGraphicalPosLeft_D;
		cylinderDirZMU = cylinderDirZLeft_D;
		cylinderDirYMU = cylinderDirYLeft_D;
		cylinderDirXMU = cylinderDirXLeft_D;
		cylinderVMU = cylinderVLeft_D;
		relativePositionMU = relativePositionLeftMU_D;
		isGrapMU = isGrabLeftMU_D;
		isGrapHalfMU = isGrabHalfLeftMU_D;
		cylinderButton = cylinderButtonLeft_D;
		firstGrab = &firstGrabLeftMU_D;
		adsorbForceMU = adsorbForceLeft_D;
		grapperUpDirXMU = tetgrapperUpDirXLeft_D;
		grapperUpDirYMU = tetgrapperUpDirYLeft_D;
		grapperUpDirZMU = tetgrapperUpDirZLeft_D;
		grapperDownDirXMU = tetgrapperDownDirXLeft_D;
		grapperDownDirYMU = tetgrapperDownDirYLeft_D;
		grapperDownDirZMU = tetgrapperDownDirZLeft_D;
		collideFlag = CollideFlagLeftMU_D;
		timer = timerLeft;
		timeTop = timeTopLeft;
	}
	else {  //Right
		cylinderShiftMU = cylinderShiftRight_D;
		cylinderLastPosMU = cylinderLastPosRight_D;
		cylinderPosMU = cylinderPosRight_D;
		cylinderDirZMU = cylinderDirZRight_D;
		cylinderDirYMU = cylinderDirYRight_D;
		cylinderDirXMU = cylinderDirXRight_D;
		cylinderVMU = cylinderVRight_D;
		relativePositionMU = relativePositionRightMU_D;
		isGrapMU = isGrabRigthMU_D;
		isGrapHalfMU = isGrabHalfRightMU_D;
		cylinderButton = cylinderButtonRight_D;
		firstGrab = &firstGrabRightMU_D;
		adsorbForceMU = adsorbForceRight_D;
		grapperUpDirXMU = tetgrapperUpDirXRight_D;
		grapperUpDirYMU = tetgrapperUpDirYRight_D;
		grapperUpDirZMU = tetgrapperUpDirZRight_D;
		grapperDownDirXMU = tetgrapperDownDirXRight_D;
		grapperDownDirYMU = tetgrapperDownDirYRight_D;
		grapperDownDirZMU = tetgrapperDownDirZRight_D;
		collideFlag = CollideFlagRightMU_D;
		timer = timerRight;
		timeTop = timeTopRight;
	}

	//�жϹ���״̬
	switch (cylinderButton)
	{
	case grab: {
		//break;
		int threadNum = 512;
		int blockNum = (triVertNum_d + threadNum - 1) / threadNum;
		//��һ�μ�ȡʱ�жϼ�ȡ����
		if (*firstGrab) {
			//printf("mesh grab\n");
			//���㱻��ȡ������Ķ���																							//���Ƽ�ȡ����Ĵ�С
			calculateGrabCylinderMU << <blockNum, threadNum >> > (cylinderPosMU, cylinderDirZMU, cylinderDirYMU, cylinderDirXMU, 0.5, 1.0, 2.2, triVertPos_d, isGrapMU, isGrapHalfMU, triVertNum_d, relativePositionMU);
			//cudaDeviceSynchronize();
			*firstGrab = false;
		}

		if (timer < timeTop) { //��ȡ���м���̣���û�м�ס
			calculateGrabForceMU << <blockNum, threadNum >> > (cylinderPosMU, grapperUpDirZMU, grapperUpDirYMU, grapperUpDirXMU, 0.5, 0.5, 2.2, triVertPos_d, isGrapHalfMU, triVertNum_d, adsorbStiffness, triVertForce_d, triVertCollisionDiag_d, 1);
			//cudaDeviceSynchronize();
			calculateGrabForceMU << <blockNum, threadNum >> > (cylinderPosMU, grapperDownDirZMU, grapperDownDirYMU, grapperDownDirXMU, 0.5, 0.5, 2.2, triVertPos_d, isGrapHalfMU, triVertNum_d, adsorbStiffness, triVertForce_d, triVertCollisionDiag_d, 2);
		}
		else { //�����ȫ��ס
			//��������ײ��⣬���Ǳ���֮ǰ����ײ��Ϣ��Լ����λ�ø��Ź����˶�
			calculateAdsorbForceMU << <blockNum, threadNum >> > (cylinderPosMU, cylinderDirXMU, cylinderDirYMU, cylinderDirZMU, triVertPos_d, isGrapMU, triVertForce_d, triVertCollisionDiag_d, relativePositionMU, triVertNum_d, adsorbStiffness);
			//�������ֵ�collide���кϲ�
			mergeCollideMU << <blockNum, threadNum >> > (triVertisCollide_d, collideFlag, isGrapMU, triVertNum_d);
		}

		cudaDeviceSynchronize();
		break;
	}
	case normal: {
		int threadNum = 512;
		int blockNum = (triVertNum_d + threadNum - 1) / threadNum;
		*firstGrab = true;
		//������ײ���
		calculateCollisionCylinderAdvanceMU << <blockNum, threadNum >> > (cylinderLastPosMU, cylinderPosMU, cylinderDirZMU,      length, radius, triVertPos_d, triVertForce_d, 	triVertisCollide_d, collideFlag, triVertCollisionDiag_d,  triVertNum_d, collisionStiffness, triVertCollisionForce_d,  directDirectionMU_D, cylinderShiftMU);
		calculateCollisionCylinderAdvanceMU << <blockNum, threadNum >> > (cylinderLastPosMU, cylinderPosMU, grapperUpDirZMU, 2.0, radius * 0.7, triVertPos_d,  triVertForce_d, triVertisCollide_d, collideFlag, triVertCollisionDiag_d, triVertNum_d, collisionStiffness * 4, triVertCollisionForce_d, directDirectionMU_D, cylinderShiftMU);
		calculateCollisionCylinderAdvanceMU << <blockNum, threadNum >> > (cylinderLastPosMU, cylinderPosMU, grapperDownDirZMU, 2.0, radius * 0.7, triVertPos_d,  triVertForce_d, triVertisCollide_d, collideFlag, triVertCollisionDiag_d, triVertNum_d, collisionStiffness * 4, triVertCollisionForce_d, directDirectionMU_D, cylinderShiftMU);

		cudaDeviceSynchronize();
	}
	default:
		break;
	}
	return 0;
}


//������Ҫ����ȡ�����������
__global__ void calculateGrabCylinderMU(float* cylinderPos, float* cylinderDirZ, float* cylinderDirY, float* cylinderDirX, float grappleX, float grappleY, float grappleZ, float* positions, unsigned int* isCollide, unsigned int* isCollideHalf, int vertexNum, float* relativePosition) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	isCollide[threadid] = 0;
	isCollideHalf[threadid] = 0;
	float collisionNormal[3];
	float collisionPos[3];
	float t = 0.0;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	bool collisionUp = obbCollisionMU(cylinderPos[0], cylinderPos[1], cylinderPos[2], cylinderDirX[0], cylinderDirX[1], cylinderDirX[2], cylinderDirY[0], cylinderDirY[1], cylinderDirY[2], -cylinderDirZ[0], -cylinderDirZ[1], -cylinderDirZ[2], positions[indexX], positions[indexY], positions[indexZ], grappleX * 1.5, grappleY, grappleZ);
	if (collisionUp) {
		//���ñ�־λ
		isCollide[threadid] = 1;
		//���㶥���ƫ��ֵ
		float vertexPosShift = (positions[indexX] - cylinderPos[0]) * cylinderDirY[0] + (positions[indexY] - cylinderPos[1]) * cylinderDirY[1] + (positions[indexZ] - cylinderPos[2]) * cylinderDirY[2];
		vertexPosShift = abs(vertexPosShift);
		//��¼��ײ��͹��ߵ����λ��
		relativePosition[indexX] = positions[indexX] - cylinderDirY[0] * (vertexPosShift - 0.05) - cylinderPos[0];
		relativePosition[indexY] = positions[indexY] - cylinderDirY[1] * (vertexPosShift - 0.05) - cylinderPos[1];
		relativePosition[indexZ] = positions[indexZ] - cylinderDirY[2] * (vertexPosShift - 0.05) - cylinderPos[2];
	}

	bool collisionDown = obbCollisionMU(cylinderPos[0], cylinderPos[1], cylinderPos[2], cylinderDirX[0], cylinderDirX[1], cylinderDirX[2], -cylinderDirY[0], -cylinderDirY[1], -cylinderDirY[2], -cylinderDirZ[0], -cylinderDirZ[1], -cylinderDirZ[2], positions[indexX], positions[indexY], positions[indexZ], grappleX * 1.5, grappleY, grappleZ);
	if (collisionDown) {
		isCollide[threadid] = 1;
		float vertexPosShift = (positions[indexX] - cylinderPos[0]) * cylinderDirY[0] + (positions[indexY] - cylinderPos[1]) * cylinderDirY[1] + (positions[indexZ] - cylinderPos[2]) * cylinderDirY[2];
		vertexPosShift = abs(vertexPosShift);
		relativePosition[indexX] = positions[indexX] + cylinderDirY[0] * (vertexPosShift - 0.05) - cylinderPos[0];
		relativePosition[indexY] = positions[indexY] + cylinderDirY[1] * (vertexPosShift - 0.05) - cylinderPos[1];
		relativePosition[indexZ] = positions[indexZ] + cylinderDirY[2] * (vertexPosShift - 0.05) - cylinderPos[2];
	}
	//δ��ײ��ֱ���˳�
	if (isCollide[threadid] != 1) return;

	//����ֲ�����
	float x = relativePosition[indexX] * cylinderDirX[0] + relativePosition[indexY] * cylinderDirX[1] + relativePosition[indexZ] * cylinderDirX[2];
	float y = relativePosition[indexX] * cylinderDirY[0] + relativePosition[indexY] * cylinderDirY[1] + relativePosition[indexZ] * cylinderDirY[2];
	float z = relativePosition[indexX] * cylinderDirZ[0] + relativePosition[indexY] * cylinderDirZ[1] + relativePosition[indexZ] * cylinderDirZ[2];
	//��¼�ֲ�����
	relativePosition[indexX] = x;
	relativePosition[indexY] = y;
	relativePosition[indexZ] = z;
}

//���㶥��������--��ȡ��
__global__ void calculateGrabForceMU(float* grapperPos, float* grapperDirZ, float* grapperDirY, float* grapperDirX, float grappleX, float grappleY, float grappleZ, float* positions, unsigned int* isCollide, int vertexNum, float adsorbStiffness, float* force, float* collisionDiag, unsigned int grabFlag) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float relativePos[3];
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	if (isCollide[threadid] == 0) {
		bool collisionFlag = obbCollisionMU(grapperPos[0], grapperPos[1], grapperPos[2], grapperDirX[0], grapperDirX[1], grapperDirX[2], grapperDirY[0], grapperDirY[1], grapperDirY[2], grapperDirZ[0], grapperDirZ[1], grapperDirZ[2], positions[indexX], positions[indexY], positions[indexZ], grappleX * 1.5, grappleY, grappleZ);
		if (!collisionFlag) return;
		//���ñ�־λ--���ĸ�ץǯ��ײ
		isCollide[threadid] = grabFlag;
	}

	if (isCollide[threadid] != grabFlag) return;
	//���㶥���ƫ��ֵ
	float vertexPosShift = (positions[indexX] - grapperPos[0]) * grapperDirY[0] + (positions[indexY] - grapperPos[1]) * grapperDirY[1] + (positions[indexZ] - grapperPos[2]) * grapperDirY[2];
	//vertexPosShift = abs(vertexPosShift);
	if (vertexPosShift < 0) vertexPosShift = 0;
	//��¼��ײ��͹��ߵ����λ��
	relativePos[0] = positions[indexX] - grapperDirY[0] * (vertexPosShift - 0.05) - grapperPos[0];
	relativePos[1] = positions[indexY] - grapperDirY[1] * (vertexPosShift - 0.05) - grapperPos[1];
	relativePos[2] = positions[indexZ] - grapperDirY[2] * (vertexPosShift - 0.05) - grapperPos[2];

	//����ֲ�����
	float x = relativePos[0] * grapperDirX[0] + relativePos[1] * grapperDirX[1] + relativePos[2] * grapperDirX[2];
	float y = relativePos[0] * grapperDirY[0] + relativePos[1] * grapperDirY[1] + relativePos[2] * grapperDirY[2];
	float z = relativePos[0] * grapperDirZ[0] + relativePos[1] * grapperDirZ[1] + relativePos[2] * grapperDirZ[2];

	float deltaPos[3];

	//����ƫ������
	float deltax = x * grapperDirX[0] + y * grapperDirY[0] + z * grapperDirZ[0];
	float deltay = x * grapperDirX[1] + y * grapperDirY[1] + z * grapperDirZ[1];
	float deltaz = x * grapperDirX[2] + y * grapperDirY[2] + z * grapperDirZ[2];

	float targetPosx = deltax + grapperPos[0];
	float targetPosy = deltay + grapperPos[1];
	float targetPosz = deltaz + grapperPos[2];

	float distance = calculateCylinderDisMU(grapperPos[0], grapperPos[1], grapperPos[2], grapperDirZ[0], grapperDirZ[1], grapperDirZ[2], targetPosx, targetPosy, targetPosz, 1.5);
	float k;
	//k = 1.0;
	k = 1 / (1 + exp(12 * distance - 5));
	adsorbStiffness = k * adsorbStiffness;

	deltaPos[0] = targetPosx - positions[indexX];
	deltaPos[1] = targetPosy - positions[indexY];
	deltaPos[2] = targetPosz - positions[indexZ];

	//ÿ�ζ������㣬�����ۼ�
	force[indexX] += adsorbStiffness * deltaPos[0];
	force[indexY] += adsorbStiffness * deltaPos[1];
	force[indexZ] += adsorbStiffness * deltaPos[2];

	collisionDiag[indexX] += adsorbStiffness;
	collisionDiag[indexY] += adsorbStiffness;
	collisionDiag[indexZ] += adsorbStiffness;
}

//����ץȡ��--��ȡ���
__global__ void calculateAdsorbForceMU(float* cylinderPos, float* cylinderDirX, float* cylinderDirY, float* cylinderDirZ, float* positions, unsigned int* isCollide, float* force, float* collisionDiag, float* relativePosition, int vertexNum, float adsorbStiffness) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//���������ײ���ֱ������
	if (isCollide[threadid] != 1) return;

	//����ײ��ͼ�����Ҫ���µ�λ�ã��ټ���attachԼ��
	float posx = positions[indexX];
	float posy = positions[indexY];
	float posz = positions[indexZ];
	float deltaPos[3];


	//����ƫ������
	float deltax = relativePosition[indexX] * cylinderDirX[0] + relativePosition[indexY] * cylinderDirY[0] + relativePosition[indexZ] * cylinderDirZ[0];
	float deltay = relativePosition[indexX] * cylinderDirX[1] + relativePosition[indexY] * cylinderDirY[1] + relativePosition[indexZ] * cylinderDirZ[1];
	float deltaz = relativePosition[indexX] * cylinderDirX[2] + relativePosition[indexY] * cylinderDirY[2] + relativePosition[indexZ] * cylinderDirZ[2];



	float targetPosx = deltax + cylinderPos[0];
	float targetPosy = deltay + cylinderPos[1];
	float targetPosz = deltaz + cylinderPos[2];

	float distance = calculateCylinderDisMU(cylinderPos[0], cylinderPos[1], cylinderPos[2], -cylinderDirZ[0], -cylinderDirZ[1], -cylinderDirZ[2], targetPosx, targetPosy, targetPosz, 1.5);
	float k;
	//k = 1.0;
	k = 1 / (1 + exp(12 * distance - 5));
	adsorbStiffness = k * adsorbStiffness;

	deltaPos[0] = targetPosx - posx;
	deltaPos[1] = targetPosy - posy;
	deltaPos[2] = targetPosz - posz;



	//ÿ�ζ������㣬�ۼӿ���
	force[indexX] += adsorbStiffness * deltaPos[0];
	force[indexY] += adsorbStiffness * deltaPos[1];
	force[indexZ] += adsorbStiffness * deltaPos[2];

	//�ᱻ���㣬�����ۼ�
	collisionDiag[indexX] += adsorbStiffness;
	collisionDiag[indexY] += adsorbStiffness;
	collisionDiag[indexZ] += adsorbStiffness;
}


//�ϲ���ȡ����ײ���
__global__ void mergeCollideMU(unsigned char* isCollide, unsigned int* CollideFlag, unsigned int* isGrap, int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	if (isGrap[threadid] != 0) {
		isCollide[threadid] = 2;
		CollideFlag[0] = 2;
	}
}

//���㶥�㵽������ľ���
__device__ float calculateCylinderDisMU(float posx, float posy, float posz, float dirx, float diry, float dirz, float vertx, float verty, float vertz, float length) {
	float pos1x = posx + dirx * length;
	float pos1y = posy + diry * length;
	float pos1z = posz + dirz * length;
	float posdx = pos1x - posx;
	float posdy = pos1y - posy;
	float posdz = pos1z - posz;

	float dx = vertx - posx;
	float dy = verty - posy;
	float dz = vertz - posz;

	float t = dirx * dx + diry * dy + dirz * dz;
	t /= length;
	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	dx = vertx - posx - t * posdx;
	dy = verty - posy - t * posdy;
	dz = vertz - posz - t * posdz;
	float distance = sqrt(dx * dx + dy * dy + dz * dz);
	return distance;
}


//���Զ����obb��Χ�н�����ײ���(ģ��ץǯץȡ�ķ�Χ)
__device__ bool obbCollisionMU(float posx, float posy, float posz, float dirXx, float dirXy, float dirXz, float dirYx, float dirYy, float dirYz, float dirZx, float dirZy, float dirZz, float vertx, float verty, float vertz, float width, float length, float height) {
	float x = (vertx - posx) * dirXx + (verty - posy) * dirXy + (vertz - posz) * dirXz;
	float y = (vertx - posx) * dirYx + (verty - posy) * dirYy + (vertz - posz) * dirYz;
	float z = (vertx - posx) * dirZx + (verty - posy) * dirZy + (vertz - posz) * dirZz;

	if (z < 0 || z > height) return false;
	if (y < 0 || y > length) return false;
	if (x < -width || x > width) return false;
	return true;
}

//ʹ����ͶӰ��������ɢ��ײ���
__device__ bool cylinderRayCollisionMU(float* cylinderPos, float* cylinderDir, float vertx, float verty, float vertz, float* moveDir, 
	float length, float radius, float* t, float* sln, float* collisionNormal, float* collisionPos) {

	float cylinder0x, cylinder0y, cylinder0z;
	cylinder0x = cylinderPos[0];
	cylinder0y = cylinderPos[1];
	cylinder0z = cylinderPos[2];
	float cylinder1x, cylinder1y, cylinder1z;
	cylinder1x = cylinderPos[0] + cylinderDir[0] * length;
	cylinder1y = cylinderPos[1] + cylinderDir[1] * length;
	cylinder1z = cylinderPos[2] + cylinderDir[2] * length;

	float cylinderdx = cylinder1x - cylinder0x;
	float cylinderdy = cylinder1y - cylinder0y;
	float cylinderdz = cylinder1z - cylinder0z;
	float dx = vertx - cylinder0x;
	float dy = verty - cylinder0y;
	float dz = vertz - cylinder0z;

	*t = cylinderDir[0] * dx + cylinderDir[1] * dy + cylinderDir[2] * dz;

	*t /= length;

	if (*t < 0) {
		*t = 0;
	}
	else if (*t > 1) {
		*t = 1;
	}

	dx = vertx - cylinder0x - (*t) * cylinderdx;
	dy = verty - cylinder0y - (*t) * cylinderdy;
	dz = vertz - cylinder0z - (*t) * cylinderdz;

	float distance = sqrt(dx * dx + dy * dy + dz * dz);
	if (distance > radius) return false;

	//������ײ����ͶӰ�Ľ���
	float moveLength = sqrt(moveDir[0] * moveDir[0] + moveDir[1] * moveDir[1] + moveDir[2] * moveDir[2]);
	moveDir[0] /= moveLength;
	moveDir[1] /= moveLength;
	moveDir[2] /= moveLength;

	//��������
	collisionNormal[0] = moveDir[0];
	collisionNormal[1] = moveDir[1];
	collisionNormal[2] = moveDir[2];

	float projectx = cylinder0x + (*t) * cylinderdx;
	float projecty = cylinder0y + (*t) * cylinderdy;
	float projectz = cylinder0z + (*t) * cylinderdz;

	//����local��,���һ��һԪ���η���
	float solution;
	float SN = (vertx - projectx) * (collisionNormal[0]) + (verty - projecty) * (collisionNormal[1]) + (vertz - projectz) * (collisionNormal[2]);
	float SS = (vertx - projectx) * (vertx - projectx) + (verty - projecty) * (verty - projecty) + (vertz - projectz) * (vertz - projectz);
	solution = -SN + sqrt(SN * SN - SS + radius * radius);//ֻȡ����

	if (solution != solution) return false;

	//���⴫�ݳ�ȥ
	*sln = solution;

	collisionPos[0] = vertx + collisionNormal[0] * solution;
	collisionPos[1] = verty + collisionNormal[1] * solution;
	collisionPos[2] = vertz + collisionNormal[2] * solution;

	//�ٴ���������
	dx = collisionPos[0] - projectx;
	dy = collisionPos[1] - projecty;
	dz = collisionPos[2] - projectz;
	distance = sqrt(dx * dx + dy * dy + dz * dz);
	collisionNormal[0] = dx / distance;
	collisionNormal[1] = dy / distance;
	collisionNormal[2] = dz / distance;

	return true;

}


//ʹ��������ײ���ı�������ײ����㷨
__global__ void calculateCollisionCylinderAdvanceMU(
	float* cylinderLastPos, float* cylinderPos, float* cylinderDir,
	float halfLength, float radius,
	float* positions, float* force,
	unsigned char* isCollide, unsigned int* collideFlag, float* collisionDiag,
	int vertexNum,
	float collisionStiffness, float* collisionForce,
	float* directDir, float* cylinderShift)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float t = 0.0;
	float solution = 0.0;

	float collisionNormal[3];
	float collisionPos[3];
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//��ƫ��������Ϊ��λ����
	float shiftLength = sqrt(cylinderShift[0] * cylinderShift[0] + cylinderShift[1] * cylinderShift[1] + cylinderShift[2] * cylinderShift[2]);
	if (shiftLength > 0.01f) {
		cylinderShift[0] /= shiftLength;
		cylinderShift[1] /= shiftLength;
		cylinderShift[2] /= shiftLength;
	}

	//ָ��������ײ���ķ���
	float moveDir[3];
	moveDir[0] = directDir[indexX];
	moveDir[1] = directDir[indexY];
	moveDir[2] = directDir[indexZ];

	//ƫ��һ���뾶���ȣ���������뾶Ϊԭ����������ʵ��ƫ�ĵ�Բ��
	float newPos[3];
	newPos[0] = cylinderPos[0] + cylinderShift[0] * 0.5 * radius;
	newPos[1] = cylinderPos[1] + cylinderShift[1] * 0.5 * radius;
	newPos[2] = cylinderPos[2] + cylinderShift[2] * 0.5 * radius;
	radius *= 1.5;
	//radius *= 2.0;

	float tetPosition[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	float toolMoveDir[3] = { cylinderLastPos[0] - cylinderPos[0], cylinderLastPos[1] - cylinderPos[1], cylinderLastPos[2] - cylinderPos[2] };
	float moveDistance = tettriVertNorm_d(toolMoveDir);

	//if(threadid==5758)
	//	printf("threadid:%d p[%f %f %f]\n", threadid, tetPosition[0], tetPosition[1], tetPosition[2]);

	if (moveDistance > 0.5) {
		//ʹ��������ײ����������������ײ
		bool collisionContinues = cylinderCollisionContinueMU(halfLength, moveDistance, radius, cylinderPos, cylinderLastPos, cylinderDir, 
			toolMoveDir, tetPosition, &t, collisionNormal, collisionPos, moveDir);
		if (!collisionContinues) return;
	}
	else {
		//ʹ��ָ�������������ײ���
		bool collision = cylinderRayCollisionMU(newPos, cylinderDir, positions[indexX], positions[indexY], positions[indexZ], 
			moveDir, halfLength, radius, &t, &solution, collisionNormal, collisionPos);
		if (!collision) return;
	}

	float deltaPos[3];
	deltaPos[0] = collisionPos[0] - positions[indexX];
	deltaPos[1] = collisionPos[1] - positions[indexY];
	deltaPos[2] = collisionPos[2] - positions[indexZ];


	//���ñ�־λ
	isCollide[threadid] = 1;
	collideFlag[0] = 1;
}


// ʹ��������ײ������������ײ���ж�
__device__ bool cylinderCollisionContinueMU(float length, float moveDistance, float radius, float* cylinderPos, float* cylinderLastPos, float* cylinderDir, float* moveDir, float* position, float* t, float* collisionNormal, float* collisionPos, float* directDir) {
	//���ȼ�����˶�ƽ��ķ�������
	float normal[3];
	tetCrossMU_D(cylinderDir, moveDir, normal);
	tettriVertNorm_d(normal);

	//���������Ҫ�ı���
	float VSubO[3] = { position[0] - cylinderPos[0] ,position[1] - cylinderPos[1] ,position[2] - cylinderPos[2] };
	float lineStart0[3] = { cylinderPos[0] ,cylinderPos[1] ,cylinderPos[2] };
	float lineStart1[3] = { cylinderLastPos[0] ,cylinderLastPos[1] ,cylinderLastPos[2] };
	float lineStart2[3] = { cylinderPos[0] + cylinderDir[0] * length ,cylinderPos[1] + cylinderDir[1] * length,cylinderPos[2] + cylinderDir[2] * length };


	//����Ҫ�Ƚ���һ����ײ��⣬�������Ƿ�����ײ��������������Ҫ������ײ���


	//1.�����ھֲ�����ϵ�е����꣬��������ֲ����겻�������ģ����Բ��ܺ�����е����ʹ�ø�˹��Ԫ
	float x, y, z;
	float det = tetSolveFormulaMU_D(cylinderDir, moveDir, normal, VSubO, &x, &y, &z);

	if (x != x || y != y || z != z) return false;

	float distance = 0.0;
	//2.���������ҵ��������ڵ�����
	if (x > length && y > moveDistance) {
		//����㵽��ľ���
		float basePoint[3] = { cylinderLastPos[0] + length * cylinderDir[0],cylinderLastPos[1] + length * cylinderDir[1] , cylinderLastPos[2] + length * cylinderDir[2] };
		distance = tetPointPointDistanceMU_D(position, basePoint);
	}
	else if (x > length && y < moveDistance && y>0.0) {
		//����㵽�ߵľ���
		distance = tetPointLineDistanceMU_D(lineStart2, moveDir, position);
	}
	else if (x > length && y < 0.0) {
		distance = tetPointPointDistanceMU_D(position, lineStart2);
	}
	else if (x > 0.0 && x < length && y > moveDistance) {
		distance = tetPointLineDistanceMU_D(lineStart1, cylinderDir, position);
	}
	else if (x > 0.0 && x < length && y < moveDistance && y>0.0) {
		//����㵽��ľ���
		distance = abs(z);
	}
	else if (x > 0.0 && x < length && y < 0.0) {
		distance = tetPointLineDistanceMU_D(lineStart0, cylinderDir, position);
	}
	else if (x<0.0 && y > moveDistance) {
		distance = tetPointPointDistanceMU_D(position, cylinderLastPos);
	}
	else if (x < 0.0 && y < moveDistance && y>0.0) {
		distance = tetPointLineDistanceMU_D(lineStart0, moveDir, position);
	}
	else if (x < 0.0 && y < 0.0) {
		distance = tetPointPointDistanceMU_D(position, cylinderPos);
	}


	//3.�жϾ���
	if (distance > radius) return false;


	//printf("x:%f,y:%f,z:%f\n", x, y, z);

	//4. �����������ײ�ų�λ��
	//����Ԫһ�η���,�����������Բ�����м���
	float lineDir[3] = { moveDir[0],moveDir[1], moveDir[2] };


	float v0[3] = { position[0] - lineStart0[0] ,position[1] - lineStart0[1] ,position[2] - lineStart0[2] };
	float v1[3] = { position[0] - lineStart1[0] ,position[1] - lineStart1[1] ,position[2] - lineStart1[2] };
	float v2[3] = { position[0] - lineStart2[0] ,position[1] - lineStart2[1] ,position[2] - lineStart2[2] };


	//��Բ���ཻ
	float solve00, solve01;
	float solve10, solve11;
	tetSolveInsectMU_D(lineDir, cylinderDir, v0, radius, &solve00, &solve01);
	tetSolveInsectMU_D(lineDir, moveDir, v0, radius, &solve10, &solve11);
	float solve = min(solve11, solve01);


	//�����ཻ
	float solve20, solve21;
	tetSolveInsectSphereMU_D(lineDir, v0, radius, &solve20, &solve21);
	solve = min(solve, solve21);


	if (solve != solve) return false;
	//printf("x:%f,y:%f,z:%f, solve: %f\n",x,y,z, solve);


	//����λ�õõ������ų���λ��
	collisionPos[0] = position[0] - lineDir[0] * solve;
	collisionPos[1] = position[1] - lineDir[1] * solve;
	collisionPos[2] = position[2] - lineDir[2] * solve;

	//���¶������ײ���ߣ��򹤾������Ͻ���ͶӰ
	float projPos[3] = { collisionPos[0] - cylinderPos[0],collisionPos[1] - cylinderPos[1],collisionPos[2] - cylinderPos[2] };
	float proj = tetDotMU_D(projPos, cylinderDir);
	projPos[0] = collisionPos[0] - cylinderPos[0] - cylinderDir[0] * proj;
	projPos[1] = collisionPos[1] - cylinderPos[1] - cylinderDir[1] * proj;
	projPos[2] = collisionPos[2] - cylinderPos[2] - cylinderDir[2] * proj;

	tettriVertNorm_d(projPos);
	collisionNormal[0] = projPos[0];
	collisionNormal[1] = projPos[1];
	collisionNormal[2] = projPos[2];

	//printf("continue: x:%f,y:%f,z:%f,solve:%f\n", collisionPos[0], collisionPos[1], collisionPos[2],solve);
	//printf("continue: nx:%f,ny:%f,nz:%f\n", collisionNormal[0], collisionNormal[1], collisionNormal[2]);
	return true;
}



__device__ float tettriVertNorm_d(float* vec0) {
	float length = vec0[0] * vec0[0] + vec0[1] * vec0[1] + vec0[2] * vec0[2];
	length = sqrt(length);
	vec0[0] /= length;
	vec0[1] /= length;
	vec0[2] /= length;
	return length;
}


__device__ void tetCrossMU_D(float* a, float* b, float* c) {
	//��˼��������η���
	c[0] = a[1] * b[2] - b[1] * a[2];
	c[1] = a[2] * b[0] - b[2] * a[0];
	c[2] = a[0] * b[1] - b[0] * a[1];
}

__device__ float tetDotMU_D(float* a, float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ float tetSolveFormulaMU_D(float* xAxis, float* yAxis, float* zAxis, float* target, float* x, float* y, float* z) {
	//ʹ�ø�˹��Ԫ
	float a[3][3] = {
		{ xAxis[0], yAxis[0], zAxis[0] },
		{ xAxis[1], yAxis[1], zAxis[1] },
		{ xAxis[2], yAxis[2], zAxis[2] },
	};

	float b[3] = { target[0],target[1],target[2] };

	//��һ����Ԫ
	float k = -a[1][0] / a[0][0];
	a[1][0] += a[0][0] * k;
	a[1][1] += a[0][1] * k;
	a[1][2] += a[0][2] * k;
	b[1] += b[0] * k;

	k = -a[2][0] / a[0][0];
	a[2][0] += a[0][0] * k;
	a[2][1] += a[0][1] * k;
	a[2][2] += a[0][2] * k;
	b[2] += b[0] * k;

	//�ڶ�����Ԫ
	k = -a[2][1] / a[1][1];
	a[2][1] += a[1][1] * k;
	a[2][2] += a[1][2] * k;
	b[2] += b[1] * k;

	//������
	*z = b[2] / a[2][2];
	*y = (b[1] - (*z) * a[1][2]) / a[1][1];
	*x = (b[0] - (*y) * a[0][1] - (*z) * a[0][2]) / a[0][0];

	return 0;
}

__device__ float tetPointLineDistanceMU_D(float* lineStart, float* lineDir, float* point) {
	float v[3] = { point[0] - lineStart[0],point[1] - lineStart[1], point[2] - lineStart[2] };

	float d = tetDotMU_D(lineDir, v);

	float projPos[3] = { lineStart[0] + d * lineDir[0],lineStart[1] + d * lineDir[1] ,lineStart[2] + d * lineDir[2] };

	projPos[0] = point[0] - projPos[0];
	projPos[1] = point[1] - projPos[1];
	projPos[2] = point[2] - projPos[2];

	return tettriVertNorm_d(projPos);
}

__device__ float tetPointPointDistanceMU_D(float* start, float* end) {
	float x = start[0] - end[0];
	float y = start[1] - end[1];
	float z = start[2] - end[2];
	return sqrt(x * x + y * y + z * z);
}

//���ߺ�Բ����
__device__ void tetSolveInsectMU_D(float* lineDir, float* toolDir, float* VSubO, float radius, float* solve0, float* solve1) {

	float temp0 = tetDotMU_D(VSubO, toolDir);
	float temp1 = -tetDotMU_D(lineDir, toolDir);

	float Bvector[3] = { VSubO[0] - temp0 * toolDir[0], VSubO[1] - temp0 * toolDir[1],  VSubO[2] - temp0 * toolDir[2] };
	float Avector[3] = { -lineDir[0] - temp1 * toolDir[0],-lineDir[1] - temp1 * toolDir[1],-lineDir[2] - temp1 * toolDir[2] };


	float A = tetDotMU_D(Avector, Avector);
	float B = 2.0 * tetDotMU_D(Avector, Bvector);
	float C = tetDotMU_D(Bvector, Bvector) - radius * radius;
	float delta = B * B - 4 * A * C;
	*solve0 = (-sqrt(delta) - B) / (2.0 * A);
	*solve1 = (sqrt(delta) - B) / (2.0 * A);
	if (*solve1 < 0.0) *solve1 = 0.0;
}

//���ߺ�����
__device__ void tetSolveInsectSphereMU_D(float* lineDir, float* VSubO, float radius, float* solve0, float* solve1) {
	float dir[3] = { -lineDir[0],-lineDir[1] ,-lineDir[2] };

	float distance = tetDotMU_D(VSubO, VSubO);

	float A = 1.0;
	float B = tetDotMU_D(dir, VSubO) * 2.0;
	float C = distance - radius * radius;

	float delta = B * B - 4 * A * C;

	*solve0 = (-sqrt(delta) - B) / (2.0 * A);
	*solve1 = (sqrt(delta) - B) / (2.0 * A);
	if (*solve1 < 0.0) *solve1 = 0.0;
}

__global__ void calculateToolShiftMU(float* cylinderPos, float* cylinderDir, float* directDir, 
	float halfLength, float radius, float* positions, float* cylinderShift, int vertexNum) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float t = 0.0;
	float solution = 0.0;

	float collisionNormal[3];
	float collisionPos[3];
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//����һ�����ߵ���ײ���
	//ָ��������ײ���ķ���
	float moveDir[3];
	moveDir[0] = directDir[indexX];
	moveDir[1] = directDir[indexY];
	moveDir[2] = directDir[indexZ];

	//ʹ��ָ�������������ײ���
	bool collision = cylinderRayCollisionMU(cylinderPos, cylinderDir, positions[indexX], positions[indexY], positions[indexZ], 
		moveDir, halfLength, radius, &t, &solution, collisionNormal, collisionPos);
	if (!collision) return;

	//�ۼӵõ�ƫ������
	atomicAdd(cylinderShift + 0, -directDir[indexX]);
	atomicAdd(cylinderShift + 1, -directDir[indexY]);
	atomicAdd(cylinderShift + 2, -directDir[indexZ]);
}



////����Բ������ײ
//int runcalculateCollisionCylinder(float halfLength, float radius, float collisionStiffness, float adsorbStiffness, float frictionStiffness, float forceDirX, float forceDirY, float forceDirZ, int flag)
//{
//	/*cudaGraphicsMapResources(1, &tetIndex_OPENGL, 0);
//	size_t size = tetNum_d * 12 * sizeof(unsigned int);
//	cudaGraphicsResourceGetMappedPointer((void **)&tetDrawIndex_D, &size, tetIndex_OPENGL);*/
//
//	//ѡȡ�����ֵĹ���
//	int		cylinderButton;
//	bool* firstGrab;
//
//	if (flag == 1) {
//		cylinderShift = cylinderShiftLeft_D;
//		cylinderLastPos = cylinderLastPosLeft_D;
//		cylinderPos = cylinderPosLeft_D;// ����λ�˻�ͼ��λ�ˣ�����ʱ���ƽ�������ֵ��ʲô��
//		cylinderGraphicalPos = cylinderGraphicalPosLeft_D;// ͼ��λ��
//		cylinderDirZ = cylinderDirZLeft_D;
//		cylinderDirY = cylinderDirYLeft_D;
//		cylinderDirX = cylinderDirXLeft_D;
//		cylinderV = cylinderVLeft_D;
//		relativePosition = relativePositionLeft_D;
//		isGrap = isGrapLeft_D;
//		isGrapHalf = isGrapHalfLeft_D;
//		cylinderButton = cylinderButtonLeft_D;
//		firstGrab = &firstGrabLeft_D;
//		adsorbForce = adsorbForceLeft_D;
//		//grapperUpPos = tetgrapperUpPosLeft_D;
//		//grapperDownPos = tetgrapperDownPosLeft_D;
//		grapperUpDirX = tetgrapperUpDirXLeft_D;
//		grapperUpDirY = tetgrapperUpDirYLeft_D;
//		grapperUpDirZ = tetgrapperUpDirZLeft_D;
//		grapperDownDirX = tetgrapperDownDirXLeft_D;
//		grapperDownDirY = tetgrapperDownDirYLeft_D;
//		grapperDownDirZ = tetgrapperDownDirZLeft_D;
//		sphereGrabFlag = grabFlagLeft_D;
//		collideFlag = collideFlagLeft_D;
//		timer = timerLeft;
//		timeTop = timeTopLeft;
//	}
//	else {
//		cylinderShift = cylinderShiftRight_D;
//		cylinderLastPos = cylinderLastPosRight_D;
//		cylinderPos = cylinderPosRight_D;
//		cylinderDirZ = cylinderDirZRight_D;
//		cylinderDirY = cylinderDirYRight_D;
//		cylinderDirX = cylinderDirXRight_D;
//		cylinderV = cylinderVRight_D;
//		relativePosition = relativePositionRight_D;
//		isGrap = isGrapRight_D;
//		isGrapHalf = isGrapHalfRigth_D;
//		cylinderButton = cylinderButtonRight_D;
//		firstGrab = &firstGrabRight_D;
//		adsorbForce = adsorbForceRight_D;
//		//grapperUpPos = tetgrapperUpPosRight_D;
//		//grapperDownPos = tetgrapperDownPosRight_D;
//		grapperUpDirX = tetgrapperUpDirXRight_D;
//		grapperUpDirY = tetgrapperUpDirYRight_D;
//		grapperUpDirZ = tetgrapperUpDirZRight_D;
//		grapperDownDirX = tetgrapperDownDirXRight_D;
//		grapperDownDirY = tetgrapperDownDirYRight_D;
//		grapperDownDirZ = tetgrapperDownDirZRight_D;
//		sphereGrabFlag = grabFlagRight_D;
//		timer = timerRight;
//		timeTop = timeTopRight;
//	}
//
//	//������һ���жϣ��Ƿ��ڼ�ȡ״̬
//	switch (cylinderButton)
//	{
//
//	case grab: {
//		int  threadNum = 512;
//		int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
//		//��ÿ�μ�ȡ�ĵ�һ�ν��м�ȡ������ж�
//		if (*firstGrab) {
//			//printf("grab\n");
//			//���㱻��ȡ������Ķ���																					//���Ƽ�ȡ�����С
//			calculateGrabCylinder << <blockNum, threadNum >> > (cylinderPos, cylinderDirZ, cylinderDirY, cylinderDirX, 0.5, 1.0, 2.2, tetVertPos_d, isGrap, isGrapHalf, tetVertNum_d, relativePosition, directIndex_D, sphereGrabFlag);
//			//cudaDeviceSynchronize();
//			*firstGrab = false;
//		}
//		//calculateGrabOBB << <blockNum, threadNum >> > (grapperUpPos, grapperUpDirZ, grapperUpDirY, grapperUpDirX, grapperDownPos, grapperDownDirZ, grapperDownDirY, grapperDownDirX, 0.48, 0.43, 2.0, tetVertPos_d, tetVertNum_d, CollideFlag_D);
//		//��������ײ��⣬���Ǳ���֮ǰ����ײ��Ϣ��Լ����λ�ø��Ź����˶�
//		//cudaDeviceSynchronize();
//		//calculateAdsorbForce << <blockNum, threadNum >> > (cylinderPos, cylinderDirX, cylinderDirY, cylinderDirZ, tetVertPos_d, isGrap, tetVertForce_d, tetCollisionDiag_d, relativePosition, tetVertNum_d, adsorbStiffness, CollideFlag_D);
//
//		if (timer < timeTop) { //��ȡ���м���̣���û����ȫ��ס
//			calculateGrabForce << <blockNum, threadNum >> > (cylinderPos, grapperUpDirZ, grapperUpDirY, grapperUpDirX, 0.5, 0.5, 2.2, tetVertPos_d, isGrapHalf, tetVertNum_d, adsorbStiffness, tetVertForce_d, tetVertCollisionDiag_d, 1);
//			//cudaDeviceSynchronize();
//			calculateGrabForce << <blockNum, threadNum >> > (cylinderPos, grapperDownDirZ, grapperDownDirY, grapperDownDirX, 0.5, 0.5, 2.2, tetVertPos_d, isGrapHalf, tetVertNum_d, adsorbStiffness, tetVertForce_d, tetVertCollisionDiag_d, 2);
//		}
//		else { //�����ȫ��ס
//			calculateAdsorbForce << <blockNum, threadNum >> > (cylinderPos, cylinderDirX, cylinderDirY, cylinderDirZ, tetVertPos_d, isGrap, tetVertForce_d, tetVertCollisionDiag_d, relativePosition, tetVertNum_d, adsorbStiffness);
//			//�������ֵ�collide���кϲ� from Dou �õ���ʱ���ٺϲ���ش��롣
//			//mergeCollideMU << <blockNum, threadNum >> > (triVertIsCollide_d, collideFlag, isGrapMU, triVertNum_d);
//		}
//
//		//�����������˵����������ݸ���������
//		blockNum = (sphereNum_D + threadNum - 1) / threadNum;
//		//calculateAdsorbForceForHaptic << <blockNum, threadNum >> > (spherePositions_D, sphereConnectStart_D, sphereConnectCount_D, sphereConnect_D, sphereConnectLength_D, sphereGrabFlag, adsorbForce, sphereNum_D);
//		cudaDeviceSynchronize();
//		break;
//	}
//	case normal: {
//		int  threadNum = 512;
//		int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
//		*firstGrab = true;
//		//���м�����ײ
//		//calculateCollisionCylinderGraphical << <blockNum, threadNum >> > (
//		//	cylinderGraphicalPos, cylinderDirZ, cylinderV, halfLength, radius, 
//		//	tetVertPos_d, isCollideGraphical_D, 
//		//	tetVertNum_d);
//		//calculateCollisionCylinder << <blockNum, threadNum >> > (cylinderPos, cylinderDirZ, cylinderV, halfLength, radius,
//		//	tetVertPos_d, tetVertVelocity_d, tetVertForce_d, tetIsCollide_d,
//		//	tetCollisionDiag_d, tetVolumeDiag_d, tetVertNum_d, collisionStiffness, frictionStiffness);
//		// -----------------------------------------------------------------------
//#ifdef WITH_DIRECTDIR
//		calculateCollisionCylinderAdvance << <blockNum, threadNum >> > (
//			cylinderLastPos, cylinderPos, cylinderDirZ, cylinderV, halfLength, radius,
//			tetVertPos_d, tetVertVelocity_d, tetVertForce_d, tetIsCollide_d,
//			tetCollisionDiag_d, tetVolumeDiag_d,
//			tetVertNum_d,
//			collisionStiffness,
//			frictionStiffness,
//			tetCollisionForce_d,
//			directDirection_D,
//			cylinderShift);
//#else
//		calculateCollisionCylinderAdvance_without_directDir << <blockNum, threadNum >> > (
//			cylinderLastPos, cylinderPos, cylinderDirZ, cylinderV, halfLength, radius,
//			tetVertPos_d, tetVertVelocity_d, tetVertForce_d, tetIsCollide_d,
//			tetVertCollisionDiag_d, tetVolumeDiag_d,
//			tetVertNum_d,
//			collisionStiffness,
//			frictionStiffness,
//			tetVertCollisionForce_d,
//			cylinderShift);
//#endif // WITH_DIRECTDIR
//		// -----------------------------------------------------------------------
//
//
//		cudaDeviceSynchronize();
//
//		break;
//	}
//	default:
//		break;
//	}
//
//	//��ץȡ������
//	if (*firstGrab) {
//		cudaMemset(sphereGrabFlag, 0, sizeof(int) * sphereNum_D);
//		//printf("clear\n");
//	}
//
//	return 0;
//}


//����ץȡ��
__global__ void calculateAdsorbForce(float* cylinderPos, float* cylinderDirX, float* cylinderDirY, float* cylinderDirZ, float* positions, unsigned int* isCollide, float* force, float* collisionDiag, float* relativePosition, int vertexNum, float adsorbStiffness) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//���������ײ���ֱ������
	if (isCollide[threadid] == 0) return;

	//����ײ��ͼ�����Ҫ���µ�λ�ã��ټ���attachԼ��
	float posx = positions[indexX];
	float posy = positions[indexY];
	float posz = positions[indexZ];
	float deltaPos[3];


	//����ƫ������
	float deltax = relativePosition[indexX] * cylinderDirX[0] + relativePosition[indexY] * cylinderDirY[0] + relativePosition[indexZ] * cylinderDirZ[0];
	float deltay = relativePosition[indexX] * cylinderDirX[1] + relativePosition[indexY] * cylinderDirY[1] + relativePosition[indexZ] * cylinderDirZ[1];
	float deltaz = relativePosition[indexX] * cylinderDirX[2] + relativePosition[indexY] * cylinderDirY[2] + relativePosition[indexZ] * cylinderDirZ[2];

	float targetPosx = deltax + cylinderPos[0];
	float targetPosy = deltay + cylinderPos[1];
	float targetPosz = deltaz + cylinderPos[2];

	float distance = calculateCylinderDis(cylinderPos[0], cylinderPos[1], cylinderPos[2], -cylinderDirZ[0], -cylinderDirZ[1], -cylinderDirZ[2], targetPosx, targetPosy, targetPosz, 1.5);
	float k;
	//k = 1.0;
	k = 1 / (1 + exp(12 * distance - 5));
	adsorbStiffness = k * adsorbStiffness;

	deltaPos[0] = targetPosx - posx;
	deltaPos[1] = targetPosy - posy;
	deltaPos[2] = targetPosz - posz;

	//ÿ�ζ������㣬�ۼӿ���
	force[indexX] += adsorbStiffness * deltaPos[0];
	force[indexY] += adsorbStiffness * deltaPos[1];
	force[indexZ] += adsorbStiffness * deltaPos[2];

	//�ᱻ���㣬�����ۼ�
	collisionDiag[indexX] += adsorbStiffness;
	collisionDiag[indexY] += adsorbStiffness;
	collisionDiag[indexZ] += adsorbStiffness;


}

//���㶥�㵽������ľ���
__device__ float calculateCylinderDis(float posx, float posy, float posz, float dirx, float diry, float dirz, float vertx, float verty, float vertz, float length) {
	float pos1x = posx + dirx * length;
	float pos1y = posy + diry * length;
	float pos1z = posz + dirz * length;
	float posdx = pos1x - posx;
	float posdy = pos1y - posy;
	float posdz = pos1z - posz;

	float dx = vertx - posx;
	float dy = verty - posy;
	float dz = vertz - posz;

	float t = dirx * dx + diry * dy + dirz * dz;
	t /= length;
	if (t < 0) {
		t = 0;
	}
	else if (t > 1) {
		t = 1;
	}

	dx = vertx - posx - t * posdx;
	dy = verty - posy - t * posdy;
	dz = vertz - posz - t * posdz;
	float distance = sqrt(dx * dx + dy * dy + dz * dz);
	return distance;
}

//ʹ��������ײ���ı�������ײ����㷨
__global__ void calculateCollisionCylinderAdvance_without_directDir(
	float* cylinderLastPos, float* cylinderPos,
	float* cylinderDir, float* cylinderV,
	float halfLength, float radius,
	float* positions, float* velocity, float* force,
	unsigned int* isCollide,
	float* collisionDiag,
	float* volumnDiag,
	int vertexNum, float collisionStiffness, float frictionStiffness,
	float* collisionForce, float* cylinderShift)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float t = 0.0;
	float solution = 0.0;

	float collisionNormal[3];
	float collisionPos[3];
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	//��ƫ��������Ϊ��λ����
	float shiftLength = sqrt(cylinderShift[0] * cylinderShift[0] + cylinderShift[1] * cylinderShift[1] + cylinderShift[2] * cylinderShift[2]);
	if (shiftLength > 0.01f) {
		cylinderShift[0] /= shiftLength;
		cylinderShift[1] /= shiftLength;
		cylinderShift[2] /= shiftLength;
	}

	float tetPosition[3] = { positions[indexX] ,positions[indexY] ,positions[indexZ] };
	float toolMoveDir[3] = { cylinderLastPos[0] - cylinderPos[0],cylinderLastPos[1] - cylinderPos[1], cylinderLastPos[2] - cylinderPos[2] };
	float moveDistance = tetNormal_D(toolMoveDir);

	float ratio = 0.0f;
	float newPos[3];
	newPos[0] = cylinderPos[0] + cylinderShift[0] * ratio * radius;
	newPos[1] = cylinderPos[1] + cylinderShift[1] * ratio * radius;
	newPos[2] = cylinderPos[2] + cylinderShift[2] * ratio * radius;
	float w = moveDistance / radius;
	float enlarged_radius = radius * (1.5 - 0.5 / w);

	if (moveDistance > 0.05) {
		//�޸�ʹ��������ײ����������������ײ
		// �޶���ָ��������������ײ���
		bool collisionContinus = cylinderCollisionContinue_without_directDir(halfLength, moveDistance, enlarged_radius, cylinderPos, cylinderLastPos, cylinderDir, toolMoveDir, tetPosition, &t, collisionNormal, collisionPos);
		if (!collisionContinus) return;
		//printf("lianxu\n");
	}
	else {
		//ʹ��ָ�������������ײ���
		////bool collision = cylinderRayCollisionDetection(newPos, cylinderDir, positions[indexX], positions[indexY], positions[indexZ], moveDir, halfLength, radius, &t, &solution, collisionNormal, collisionPos);
		//bool collision = cylinderRayCollisionDetection(cylinderPos, cylinderDir, positions[indexX], positions[indexY], positions[indexZ], moveDir, halfLength, radius, &t, &solution, collisionNormal, collisionPos);
		float vert[3] = { positions[indexX], positions[indexY], positions[indexZ] };
		bool collision = cylinderCollision(cylinderPos, cylinderDir, vert, halfLength, radius, &t, collisionNormal, collisionPos);
		if (!collision) return;
		//printf("---lisan\n");
	}

	float deltaPos[3];
	deltaPos[0] = collisionPos[0] - positions[indexX];
	deltaPos[1] = collisionPos[1] - positions[indexY];
	deltaPos[2] = collisionPos[2] - positions[indexZ];

	// ������ײ����Ӵ�����
	float temp[3];
	temp[0] = collisionStiffness * (collisionNormal[0] * collisionNormal[0] * deltaPos[0] + collisionNormal[0] * collisionNormal[1] * deltaPos[1] + collisionNormal[0] * collisionNormal[2] * deltaPos[2]);
	temp[1] = collisionStiffness * (collisionNormal[1] * collisionNormal[0] * deltaPos[0] + collisionNormal[1] * collisionNormal[1] * deltaPos[1] + collisionNormal[1] * collisionNormal[2] * deltaPos[2]);
	temp[2] = collisionStiffness * (collisionNormal[2] * collisionNormal[0] * deltaPos[0] + collisionNormal[2] * collisionNormal[1] * deltaPos[1] + collisionNormal[2] * collisionNormal[2] * deltaPos[2]);
	collisionForce[indexX] += temp[0];
	collisionForce[indexY] += temp[1];
	collisionForce[indexZ] += temp[2];


	//����Ħ����
	float friction[3];
	friction[0] = 0.0;
	friction[1] = 0.0;
	friction[2] = 0.0;
	float frictionT[3];	//����Ħ����
	float frictionN[3];	//����Ħ����
						//��������˶����ٶȣ���������ٶ�������
	float v[3];
	v[0] = velocity[indexX] - cylinderV[0];
	v[1] = velocity[indexY] - cylinderV[1];
	v[2] = velocity[indexZ] - cylinderV[2];
	float c[3];
	c[0] = collisionStiffness * collisionNormal[0] * collisionNormal[0];
	c[1] = collisionStiffness * collisionNormal[1] * collisionNormal[1];
	c[2] = collisionStiffness * collisionNormal[2] * collisionNormal[2];
	c[0] += volumnDiag[threadid];
	c[1] += volumnDiag[threadid];
	c[2] += volumnDiag[threadid];
	//b-c(x-st)������໥������
	float relatedForce[3];
	relatedForce[0] = force[indexX] + c[0] * v[0] * 0.0009;
	relatedForce[1] = force[indexY] + c[1] * v[1] * 0.0009;
	relatedForce[2] = force[indexZ] + c[2] * v[2] * 0.0009;


	//�������������Ħ����
	float product = relatedForce[0] * collisionNormal[0]\
		+ relatedForce[1] * collisionNormal[1] \
		+ relatedForce[2] * collisionNormal[2];
	//if (product < 0) {
	//	//���㴹ֱ����
	//	frictionN[0] = -product*collisionNormal[0];
	//	frictionN[1] = -product*collisionNormal[1];
	//	frictionN[2] = -product*collisionNormal[2];

	//	//�����������
	//	relatedForce[0] -= product*collisionNormal[0];
	//	relatedForce[1] -= product*collisionNormal[1];
	//	relatedForce[2] -= product*collisionNormal[2];

	//	//�Ƿ񳬹���Ħ����ֵ
	//	float relatedlength = sqrt(relatedForce[0]* relatedForce[0]+ relatedForce[1]* relatedForce[1]+ relatedForce[2]* relatedForce[2]);
	//	
	//	float frictionNlength = sqrt(frictionN[0]* frictionN[0]+ frictionN[1]* frictionN[1]+ frictionN[2]* frictionN[2]);
	//	if (relatedlength <= frictionNlength*frictionStiffness) {
	//		
	//		frictionT[0] = -relatedForce[0];
	//		frictionT[1] = -relatedForce[1];
	//		frictionT[2] = -relatedForce[2];
	//
	//	}
	//	else {
	//		frictionT[0] = -frictionStiffness*frictionNlength*(relatedForce[0]/relatedlength);
	//		frictionT[1] = -frictionStiffness*frictionNlength*(relatedForce[1]/relatedlength);
	//		frictionT[2] = -frictionStiffness*frictionNlength*(relatedForce[2]/relatedlength);
	//	}

	//	friction[0] = frictionT[0];
	//	friction[1] = frictionT[1];
	//	friction[2] = frictionT[2];
	//}

	//������
	force[indexX] += temp[0] + friction[0];
	force[indexY] += temp[1] + friction[1];
	force[indexZ] += temp[2] + friction[2];
	//force[indexX] += friction[0];
	//force[indexY] += friction[1];
	//force[indexZ] += friction[2];


	//����Խ�Ԫ�ض�Ӧ��ֵ
	collisionDiag[indexX] += collisionStiffness * collisionNormal[0] * collisionNormal[0];
	collisionDiag[indexY] += collisionStiffness * collisionNormal[1] * collisionNormal[1];
	collisionDiag[indexZ] += collisionStiffness * collisionNormal[2] * collisionNormal[2];

	//���ñ�־λ
	isCollide[threadid] = 1;
}

//ר����Բ������ײ��������һ����װ
__device__ bool cylinderCollision(float* pos, float* dir, float* vert, float length, float radius, float* t, float* collisionNormal, float* collisionPos) {
	float cylinder0x, cylinder0y, cylinder0z;
	cylinder0x = pos[0];
	cylinder0y = pos[1];
	cylinder0z = pos[2];
	float cylinder1x, cylinder1y, cylinder1z;
	cylinder1x = pos[0] + dir[0] * length;
	cylinder1y = pos[1] + dir[1] * length;
	cylinder1z = pos[2] + dir[2] * length;

	float cylinderdx = cylinder1x - cylinder0x;
	float cylinderdy = cylinder1y - cylinder0y;
	float cylinderdz = cylinder1z - cylinder0z;
	float dx = vert[0] - cylinder0x;
	float dy = vert[1] - cylinder0y;
	float dz = vert[2] - cylinder0z;
	*t = dir[0] * dx + dir[1] * dy + dir[2] * dz;

	*t /= length;

	if (*t < 0) {
		*t = 0;
	}
	else if (*t > 1) {
		*t = 1;
	}

	dx = vert[0] - cylinder0x - (*t) * cylinderdx;
	dy = vert[1] - cylinder0y - (*t) * cylinderdy;
	dz = vert[2] - cylinder0z - (*t) * cylinderdz;

	float distance = sqrt(dx * dx + dy * dy + dz * dz);
	if (distance > radius) return false;
	if (distance < 1e-5)// ���������������������ײ��Ӧ������ά��ԭ״��
	{
		collisionNormal[0] = 0;
		collisionNormal[1] = 0;
		collisionNormal[2] = 0;
		collisionPos[0] = vert[0];
		collisionPos[1] = vert[1];
		collisionPos[2] = vert[2];
	}
	else
	{
		collisionNormal[0] = dx / distance;
		collisionNormal[1] = dy / distance;
		collisionNormal[2] = dz / distance;
		collisionPos[0] = vert[0] + collisionNormal[0] * (radius - distance);
		collisionPos[1] = vert[1] + collisionNormal[1] * (radius - distance);
		collisionPos[2] = vert[2] + collisionNormal[2] * (radius - distance);
	}

	return true;
}

__device__ bool cylinderCollision_withDepth(float * pose, float* vert, float length, float radius, float* t, float* depth, float* dist, float* collisionNormal, float* collisionPos)
{
	float cylinder0x, cylinder0y, cylinder0z;
	cylinder0x = pose[0];
	cylinder0y = pose[1];
	cylinder0z = pose[2];
	float dir[3] = { pose[3], pose[4], pose[5] };
	float cylinder1x, cylinder1y, cylinder1z;
	cylinder1x = pose[0] + dir[0] * length;
	cylinder1y = pose[1] + dir[1] * length;
	cylinder1z = pose[2] + dir[2] * length;

	float cylinderdx = cylinder1x - cylinder0x;
	float cylinderdy = cylinder1y - cylinder0y;
	float cylinderdz = cylinder1z - cylinder0z;
	float dx = vert[0] - cylinder0x;
	float dy = vert[1] - cylinder0y;
	float dz = vert[2] - cylinder0z;
	*t = dir[0] * dx + dir[1] * dy + dir[2] * dz;

	*t /= length;
	/*printf("tool len: %f\n", length);*/
	if (*t < 0) {
		*t = 0;
		*depth = 0;
		// ע�������return, ��Բ�����������Բ�򡣽�����
		//return false;
	}
	else if (*t > 1) {
		*t = 1;
		*depth = 0;
		// ע�������return, ��Բ�����������Բ�򡣽�����
		//return false;
	}

	dx = vert[0] - cylinder0x - (*t) * cylinderdx;
	dy = vert[1] - cylinder0y - (*t) * cylinderdy;
	dz = vert[2] - cylinder0z - (*t) * cylinderdz;

	float distance = sqrt(dx * dx + dy * dy + dz * dz);
	*dist = distance;

	if (distance > radius) return false;
	if (distance < 1e-5)// ���������������������ײ��Ӧ������ά��ԭ״��
	{
		collisionNormal[0] = 0;
		collisionNormal[1] = 0;
		collisionNormal[2] = 0;
		collisionPos[0] = vert[0];
		collisionPos[1] = vert[1];
		collisionPos[2] = vert[2];
		*depth = 0;
	}
	else
	{
		collisionNormal[0] = dx / distance;
		collisionNormal[1] = dy / distance;
		collisionNormal[2] = dz / distance;
		float d = radius - distance;
		collisionPos[0] = vert[0] + collisionNormal[0] * d;
		collisionPos[1] = vert[1] + collisionNormal[1] * d;
		collisionPos[2] = vert[2] + collisionNormal[2] * d;
		//printf("collided, vert: %f %f %f, toolpos: %f %f %f dir: %f %f %f\nd: %f t%%: %f\n", vert[0], vert[1], vert[2], 
		//	pose[0], pose[1], pose[2], dir[0], dir[1], dir[2], d, *t);
		*depth = d;
	}

	return true;
}


//������Ҫ����ȡ�����������
__global__ void calculateGrabCylinder(float* cylinderPos, float* cylinderDirZ, float* cylinderDirY, float* cylinderDirX, float grappleX, float grappleY, float grappleZ, float* positions, unsigned int* isCollide, unsigned int* isCollideHalf, int vertexNum, float* relativePosition, int* directIndex, int* sphereGrabFlag) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	isCollide[threadid] = 0;
	isCollideHalf[threadid] = 0;
	float collisionNormal[3];
	float collisionPos[3];
	float t = 0.0;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	bool collisionUp = obbCollision(cylinderPos[0], cylinderPos[1], cylinderPos[2], cylinderDirX[0], cylinderDirX[1], cylinderDirX[2], cylinderDirY[0], cylinderDirY[1], cylinderDirY[2], -cylinderDirZ[0], -cylinderDirZ[1], -cylinderDirZ[2], positions[indexX], positions[indexY], positions[indexZ], 1.5 * grappleX, grappleY, grappleZ);
	if (collisionUp) {
		//���ñ�־λ
		isCollide[threadid] = 1;
		//���㶥���ƫ��ֵ
		float vertexPosShift = (positions[indexX] - cylinderPos[0]) * cylinderDirY[0] + (positions[indexY] - cylinderPos[1]) * cylinderDirY[1] + (positions[indexZ] - cylinderPos[2]) * cylinderDirY[2];
		vertexPosShift = abs(vertexPosShift);
		//��¼��ײ��͹��ߵ����λ��
		relativePosition[indexX] = positions[indexX] - cylinderDirY[0] * (vertexPosShift - 0.05) - cylinderPos[0];
		relativePosition[indexY] = positions[indexY] - cylinderDirY[1] * (vertexPosShift - 0.05) - cylinderPos[1];
		relativePosition[indexZ] = positions[indexZ] - cylinderDirY[2] * (vertexPosShift - 0.05) - cylinderPos[2];
	}

	bool collisionDown = obbCollision(cylinderPos[0], cylinderPos[1], cylinderPos[2], cylinderDirX[0], cylinderDirX[1], cylinderDirX[2], -cylinderDirY[0], -cylinderDirY[1], -cylinderDirY[2], -cylinderDirZ[0], -cylinderDirZ[1], -cylinderDirZ[2], positions[indexX], positions[indexY], positions[indexZ], 1.5 * grappleX, grappleY, grappleZ);
	if (collisionDown) {
		isCollide[threadid] = 1;
		float vertexPosShift = (positions[indexX] - cylinderPos[0]) * cylinderDirY[0] + (positions[indexY] - cylinderPos[1]) * cylinderDirY[1] + (positions[indexZ] - cylinderPos[2]) * cylinderDirY[2];
		vertexPosShift = abs(vertexPosShift);
		relativePosition[indexX] = positions[indexX] + cylinderDirY[0] * (vertexPosShift - 0.05) - cylinderPos[0];
		relativePosition[indexY] = positions[indexY] + cylinderDirY[1] * (vertexPosShift - 0.05) - cylinderPos[1];
		relativePosition[indexZ] = positions[indexZ] + cylinderDirY[2] * (vertexPosShift - 0.05) - cylinderPos[2];
	}

	//δ��ײֱ���˳�
	if (isCollide[threadid] != 1) return;


	//����ֲ�����
	float x = relativePosition[indexX] * cylinderDirX[0] + relativePosition[indexY] * cylinderDirX[1] + relativePosition[indexZ] * cylinderDirX[2];
	float y = relativePosition[indexX] * cylinderDirY[0] + relativePosition[indexY] * cylinderDirY[1] + relativePosition[indexZ] * cylinderDirY[2];
	float z = relativePosition[indexX] * cylinderDirZ[0] + relativePosition[indexY] * cylinderDirZ[1] + relativePosition[indexZ] * cylinderDirZ[2];
	//��¼�ֲ�����
	relativePosition[indexX] = x;
	relativePosition[indexY] = y;
	relativePosition[indexZ] = z;

	//�����������˵�ץȡ��
	//1. ��ȡ�ͱ�ץȡ���ӵ������ڵ�
	int sphereIndex = directIndex[threadid];

	//2. ����ץȡ��־��Ϊ1
	sphereGrabFlag[sphereIndex] = 1;

}

//�����ȡ��2.0
__global__ void calculateGrabForce(float* grapperPos, float* grapperDirZ, float* grapperDirY, float* grapperDirX, float grappleX, float grappleY, float grappleZ, float* positions, unsigned int* isCollide, int vertexNum, float adsorbStiffness, float* force, float* collisionDiag, unsigned int grabFlag) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float relativePos[3];
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	if (isCollide[threadid] == 0) {
		bool collisionFlag = obbCollision(grapperPos[0], grapperPos[1], grapperPos[2], grapperDirX[0], grapperDirX[1], grapperDirX[2], grapperDirY[0], grapperDirY[1], grapperDirY[2], grapperDirZ[0], grapperDirZ[1], grapperDirZ[2], positions[indexX], positions[indexY], positions[indexZ], grappleX * 1.5, grappleY, grappleZ);
		if (!collisionFlag) return;
		//���ñ�־λ--���ĸ�ץǯ��ײ
		isCollide[threadid] = grabFlag;
	}

	if (isCollide[threadid] != grabFlag) return;
	//���㶥���ƫ��ֵ
	float vertexPosShift = (positions[indexX] - grapperPos[0]) * grapperDirY[0] + (positions[indexY] - grapperPos[1]) * grapperDirY[1] + (positions[indexZ] - grapperPos[2]) * grapperDirY[2];
	//vertexPosShift = abs(vertexPosShift);
	if (vertexPosShift < 0) vertexPosShift = 0;
	//��¼��ײ��͹��ߵ����λ��
	relativePos[0] = positions[indexX] - grapperDirY[0] * (vertexPosShift - 0.05) - grapperPos[0];
	relativePos[1] = positions[indexY] - grapperDirY[1] * (vertexPosShift - 0.05) - grapperPos[1];
	relativePos[2] = positions[indexZ] - grapperDirY[2] * (vertexPosShift - 0.05) - grapperPos[2];

	//����ֲ�����
	float x = relativePos[0] * grapperDirX[0] + relativePos[1] * grapperDirX[1] + relativePos[2] * grapperDirX[2];
	float y = relativePos[0] * grapperDirY[0] + relativePos[1] * grapperDirY[1] + relativePos[2] * grapperDirY[2];
	float z = relativePos[0] * grapperDirZ[0] + relativePos[1] * grapperDirZ[1] + relativePos[2] * grapperDirZ[2];

	float deltaPos[3];

	//����ƫ������
	float deltax = x * grapperDirX[0] + y * grapperDirY[0] + z * grapperDirZ[0];
	float deltay = x * grapperDirX[1] + y * grapperDirY[1] + z * grapperDirZ[1];
	float deltaz = x * grapperDirX[2] + y * grapperDirY[2] + z * grapperDirZ[2];

	float targetPosx = deltax + grapperPos[0];
	float targetPosy = deltay + grapperPos[1];
	float targetPosz = deltaz + grapperPos[2];

	float distance = calculateCylinderDis(grapperPos[0], grapperPos[1], grapperPos[2], grapperDirZ[0], grapperDirZ[1], grapperDirZ[2], targetPosx, targetPosy, targetPosz, 1.5);
	float k;
	//k = 1.0;
	k = 1 / (1 + exp(12 * distance - 5));
	adsorbStiffness = k * adsorbStiffness;

	deltaPos[0] = targetPosx - positions[indexX];
	deltaPos[1] = targetPosy - positions[indexY];
	deltaPos[2] = targetPosz - positions[indexZ];

	//ÿ�ζ������㣬�����ۼ�
	force[indexX] += adsorbStiffness * deltaPos[0];
	force[indexY] += adsorbStiffness * deltaPos[1];
	force[indexZ] += adsorbStiffness * deltaPos[2];

	collisionDiag[indexX] += adsorbStiffness;
	collisionDiag[indexY] += adsorbStiffness;
	collisionDiag[indexZ] += adsorbStiffness;

}


//���Զ����obb��Χ�н�����ײ��⣨ģ��ץǯץȡ�ķ�Χ��
__device__ bool obbCollision(float posx, float posy, float posz, float dirXx, float dirXy, float dirXz, float dirYx, float dirYy, float dirYz, float dirZx, float dirZy, float dirZz, float vertx, float verty, float vertz, float width, float length, float height) {
	float x = (vertx - posx) * dirXx + (verty - posy) * dirXy + (vertz - posz) * dirXz;
	float y = (vertx - posx) * dirYx + (verty - posy) * dirYy + (vertz - posz) * dirYz;
	float z = (vertx - posx) * dirZx + (verty - posy) * dirZy + (vertz - posz) * dirZz;

	if (z<0 || z>height) return false;
	if (y<0 || y>length) return false;
	if (x<-width || x>width) return false;

	return true;
}

//ʹ��������ײ������������ײ���ж�
__device__ bool cylinderCollisionContinue(
	float length, float moveDistance, float radius,
	float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir,
	float* moveDir, float* position,
	float* t, float* collisionNormal,
	float* collisionPos,
	float* directDir)
{
	float tt, sol;
	float collisionN[3], collisionP[3];
	// ��ͼ��λ�˵�Բ������ײ��⡣
	//bool collision = cylinderCollision(cylinderPos, cylinderDir, position, length, radius, &tt, collisionN, collisionP);
	bool collision = cylinderRayCollisionDetection(cylinderLastPos, cylinderDir,
		position[0], position[1], position[2],
		directDir, // �����ָ���������Ͱ󶨵������ڵ�λ����ء�
		length, radius,
		&tt, &sol,
		collisionN, collisionP);
	float toolMoveVec[3] = { cylinderPos[0] - cylinderLastPos[0],cylinderPos[1] - cylinderLastPos[1], cylinderPos[2] - cylinderLastPos[2] };
	tetNormal_D(toolMoveVec);
	tetNormal_D(directDir);
	float cos_val = tetDot_D(toolMoveVec, directDir);
	if (collision)
	{
		collisionNormal[0] = collisionN[0];
		collisionNormal[1] = collisionN[1];
		collisionNormal[2] = collisionN[2];

		collisionPos[0] = collisionP[0];
		collisionPos[1] = collisionP[1];
		collisionPos[2] = collisionP[2];
		return true;
	}

	//���ȼ�����˶�ƽ��ķ�������
	float normal[3];
	tetCross_D(cylinderDir, moveDir, normal);
	tetNormal_D(normal);

	//���������Ҫ�ı���
	float VSubO[3] = { position[0] - cylinderPos[0] ,position[1] - cylinderPos[1] ,position[2] - cylinderPos[2] };//���߼��ָ�������ײ�������
	float lineStart0[3] = { cylinderPos[0] ,cylinderPos[1] ,cylinderPos[2] };// ��ǰ���߼�ˣ��������߼�ˣ�
	float lineStart1[3] = { cylinderLastPos[0] ,cylinderLastPos[1] ,cylinderLastPos[2] };// ��һ֡���߼��(��ͼ�ι��߼��)
	float lineStart2[3] = { cylinderPos[0] + cylinderDir[0] * length ,cylinderPos[1] + cylinderDir[1] * length,cylinderPos[2] + cylinderDir[2] * length };// ��ǰ֡����β������������β����


	//����Ҫ�Ƚ���һ����ײ��⣬�������Ƿ�����ײ��������������Ҫ������ײ���


	//1.�����ھֲ�����ϵ�е����꣬��������ֲ����겻�������ģ����Բ��ܺ�����е����ʹ�ø�˹��Ԫ
	// �����᣺ ���߷���cylinderDir���˶�����moveDir�����߷������˶������ųɵ�ƽ��ķ�����normal
	// ������ײ��������������ɵľֲ�����ϵ����[x, y, z] �ô������������������ϵ�£�x������ײ���ڹ�����ͶӰ��λ�ã�y�������˶������ϵ��˶�����
	// ��˹��Ԫ��[A|I] ֻʹ����֮��ļӼ���A���I��I����A�������
	float x, y, z;
	float det = tetSolveFormula_D(cylinderDir, moveDir, normal, VSubO, &x, &y, &z);

	if (x != x || y != y || z != z) return false;

	float distance = 0.0;
	bool flag = false;
	//2.���������ҵ��������ڵ�����
	if (x > length && y > moveDistance) {
		//����㵽��ľ���
		float basePoint[3] = { cylinderLastPos[0] + length * cylinderDir[0],cylinderLastPos[1] + length * cylinderDir[1] , cylinderLastPos[2] + length * cylinderDir[2] };
		distance = tetPointPointDistance_D(position, basePoint);
		flag = true;
	}
	else if (x > length && y < moveDistance && y>0.0) {
		//����㵽�ߵľ���
		distance = tetPointLineDistance_D(lineStart2, moveDir, position);
	}
	else if (x > length && y < 0.0) {
		distance = tetPointPointDistance_D(position, lineStart2);
	}
	else if (x > 0.0 && x < length && y > moveDistance) {
		distance = tetPointLineDistance_D(lineStart1, cylinderDir, position);
		flag = true;
	}
	else if (x > 0.0 && x < length && y < moveDistance && y>0.0) {
		//����㵽��ľ���
		distance = abs(z);
	}
	else if (x > 0.0 && x < length && y < 0.0) {
		distance = tetPointLineDistance_D(lineStart0, cylinderDir, position);
	}
	else if (x<0.0 && y > moveDistance) {
		distance = tetPointPointDistance_D(position, cylinderLastPos);
		flag = true;
	}
	else if (x < 0.0 && y < moveDistance && y>0.0) {
		distance = tetPointLineDistance_D(lineStart0, moveDir, position);
	}
	else if (x < 0.0 && y < 0.0) {
		distance = tetPointPointDistance_D(position, cylinderPos);
	}


	//3.�жϾ���
	if (distance > radius) return false;
	//if (flag) return false;

	//printf("x:%f,y:%f,z:%f\n", x, y, z);

	//4. �����������ײ�ų�λ��
	//����Ԫһ�η���,�����������Բ�����м���
	float lineDir[3] = { moveDir[0],moveDir[1], moveDir[2] };


	float v0[3] = { position[0] - lineStart0[0] ,position[1] - lineStart0[1] ,position[2] - lineStart0[2] };
	float v1[3] = { position[0] - lineStart1[0] ,position[1] - lineStart1[1] ,position[2] - lineStart1[2] };
	float v2[3] = { position[0] - lineStart2[0] ,position[1] - lineStart2[1] ,position[2] - lineStart2[2] };


	//��Բ���ཻ
	float solve00, solve01;
	float solve10, solve11;
	tetSolveInsect_D(lineDir, cylinderDir, v0, radius, &solve00, &solve01);
	tetSolveInsect_D(lineDir, moveDir, v0, radius, &solve10, &solve11);
	float solve = min(solve11, solve01);
	//tetSolveInsect_D(lineDir, cylinderDir, v1, radius, &solve00, &solve01);
	//solve = min(solve, solve01);
	//tetSolveInsect_D(lineDir, moveDir, v2, radius, &solve00, &solve01);
	//solve = min(solve, solve01);


	//�����ཻ
	float solve20, solve21;
	tetSolveInsectSphere_D(lineDir, v0, radius, &solve20, &solve21);
	solve = min(solve, solve21);
	//printf("%f\n", solve);
	//tetSolveInsectSphere_D(lineDir, v1, radius, &solve10, &solve11);
	//solve = min(solve, solve11);
	//tetSolveInsectSphere_D(lineDir, v2, radius, &solve10, &solve11);
	//solve = min(solve, solve11);
	//tetSolveInsectSphere_D(lineDir, VSubO, radius, &solve10, &solve11);
	//solve = min(solve, solve11);

	if (solve != solve) return false;
	//printf("x:%f,y:%f,z:%f, solve: %f\n",x,y,z, solve);

	//����λ�õõ������ų���λ��
	collisionPos[0] = position[0] - lineDir[0] * solve;
	collisionPos[1] = position[1] - lineDir[1] * solve;
	collisionPos[2] = position[2] - lineDir[2] * solve;

	//���¶������ײ���ߣ��򹤾������Ͻ���ͶӰ
	float projPos[3] = { collisionPos[0] - cylinderPos[0],collisionPos[1] - cylinderPos[1],collisionPos[2] - cylinderPos[2] };
	float proj = tetDot_D(projPos, cylinderDir);
	projPos[0] = collisionPos[0] - cylinderPos[0] - cylinderDir[0] * proj;
	projPos[1] = collisionPos[1] - cylinderPos[1] - cylinderDir[1] * proj;
	projPos[2] = collisionPos[2] - cylinderPos[2] - cylinderDir[2] * proj;

	tetNormal_D(projPos);
	collisionNormal[0] = projPos[0];
	collisionNormal[1] = projPos[1];
	collisionNormal[2] = projPos[2];

	return true;
}

__device__ bool cylinderRayCollisionDetection(
	float* cylinderPos, float* cylinderDir,
	float vertx, float verty, float vertz,
	float* moveDir, // ָ����������ײ���*���嶥��*���˶�����
	float length, float radius,
	float* t, float* sln,
	float* collisionNormal, float* collisionPos) {

	float cylinder0x, cylinder0y, cylinder0z; // ���߼��
	cylinder0x = cylinderPos[0];
	cylinder0y = cylinderPos[1];
	cylinder0z = cylinderPos[2];
	float cylinder1x, cylinder1y, cylinder1z; // ����β��
	cylinder1x = cylinderPos[0] + cylinderDir[0] * length;
	cylinder1y = cylinderPos[1] + cylinderDir[1] * length;
	cylinder1z = cylinderPos[2] + cylinderDir[2] * length;

	float cylinderdx = cylinder1x - cylinder0x;
	float cylinderdy = cylinder1y - cylinder0y;
	float cylinderdz = cylinder1z - cylinder0z;
	float dx = vertx - cylinder0x;
	float dy = verty - cylinder0y;
	float dz = vertz - cylinder0z;

	*t = cylinderDir[0] * dx + cylinderDir[1] * dy + cylinderDir[2] * dz;

	*t /= length;

	if (*t < 0) {
		*t = 0;
	}
	else if (*t > 1) {
		*t = 1;
	}

	//�ӹ���ָ��Ӵ���ġ���ֱ�ڹ��ߵ�����
	dx = vertx - cylinder0x - (*t) * cylinderdx;
	dy = verty - cylinder0y - (*t) * cylinderdy;
	dz = vertz - cylinder0z - (*t) * cylinderdz;

	float distance = sqrt(dx * dx + dy * dy + dz * dz);
	if (distance > radius) return false;// �빤��������ľ���

	//������ײ����ͶӰ�Ľ���
	float moveLength = sqrt(moveDir[0] * moveDir[0] + moveDir[1] * moveDir[1] + moveDir[2] * moveDir[2]);
	moveDir[0] /= moveLength;
	moveDir[1] /= moveLength;
	moveDir[2] /= moveLength;

	//���������������moveDir�������ָ�������ķ���directDirection_D���ӵ�ǰ����ָ����󶨵�������ģ�
	// �༴�䱻ѹ�ȵ��˶����򡣶��㳯�˷����˶����Խ��Ͷ���Ƕ�빤�ߵ���ȣ�
	collisionNormal[0] = moveDir[0];
	collisionNormal[1] = moveDir[1];
	collisionNormal[2] = moveDir[2];

	//�����������ϵ���ײ��ͶӰ
	float projectx = cylinder0x + (*t) * cylinderdx;
	float projecty = cylinder0y + (*t) * cylinderdy;
	float projectz = cylinder0z + (*t) * cylinderdz;

	//����local��,���һ��һԪ���η���
	float solution;
	float SN = (vertx - projectx) * (collisionNormal[0]) + (verty - projecty) * (collisionNormal[1]) + (vertz - projectz) * (collisionNormal[2]);
	float SS = (vertx - projectx) * (vertx - projectx) + (verty - projecty) * (verty - projecty) + (vertz - projectz) * (vertz - projectz);
	solution = -SN + sqrt(SN * SN - SS + radius * radius);//ֻȡ����

	if (isnan(solution)) return false;

	//���⴫�ݳ�ȥ �����������������ǣ���������Ϊ�������߷���ͶӰ�����㵽���߱���ľ��롣��sqrt�Ĳ���Ϊ����ʱ��û�н��㣬����nan�����
	*sln = solution;

	collisionPos[0] = vertx + collisionNormal[0] * solution;
	collisionPos[1] = verty + collisionNormal[1] * solution;
	collisionPos[2] = vertz + collisionNormal[2] * solution;

	//�ٴ��������򣬸�Ϊ�ӹ�������ָ��Ӵ����ѹ����
	dx = collisionPos[0] - projectx;
	dy = collisionPos[1] - projecty;
	dz = collisionPos[2] - projectz;
	distance = sqrt(dx * dx + dy * dy + dz * dz);
	collisionNormal[0] = dx / distance;
	collisionNormal[1] = dy / distance;
	collisionNormal[2] = dz / distance;

	return true;
}


__device__ void tetCross_D(float* a, float* b, float* c) {
	//��˼��������η���
	c[0] = a[1] * b[2] - b[1] * a[2];
	c[1] = a[2] * b[0] - b[2] * a[0];
	c[2] = a[0] * b[1] - b[0] * a[1];
}

__device__ float tetDot_D(float* a, float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ float tetNormal_D(float* vec0) {
	float length = vec0[0] * vec0[0] + vec0[1] * vec0[1] + vec0[2] * vec0[2];
	length = sqrt(length);
	vec0[0] /= length;
	vec0[1] /= length;
	vec0[2] /= length;
	return length;
}

__device__ float tetSolveFormula_D(float* xAxis, float* yAxis, float* zAxis, float* target, float* x, float* y, float* z) {
	//ʹ�ø�˹��Ԫ
	float a[3][3] = {
		{ xAxis[0], yAxis[0], zAxis[0] },
		{ xAxis[1], yAxis[1], zAxis[1] },
		{ xAxis[2], yAxis[2], zAxis[2] },
	};

	float b[3] = { target[0],target[1],target[2] };

	//��һ����Ԫ
	float k = -a[1][0] / a[0][0];
	a[1][0] += a[0][0] * k;
	a[1][1] += a[0][1] * k;
	a[1][2] += a[0][2] * k;
	b[1] += b[0] * k;

	k = -a[2][0] / a[0][0];
	a[2][0] += a[0][0] * k;
	a[2][1] += a[0][1] * k;
	a[2][2] += a[0][2] * k;
	b[2] += b[0] * k;

	//�ڶ�����Ԫ
	k = -a[2][1] / a[1][1];
	a[2][1] += a[1][1] * k;
	a[2][2] += a[1][2] * k;
	b[2] += b[1] * k;

	//������
	*z = b[2] / a[2][2];
	*y = (b[1] - (*z) * a[1][2]) / a[1][1];
	*x = (b[0] - (*y) * a[0][1] - (*z) * a[0][2]) / a[0][0];

	return 0;
}

__device__ float tetPointLineDistance_D(float* lineStart, float* lineDir, float* point) {
	float v[3] = { point[0] - lineStart[0],point[1] - lineStart[1], point[2] - lineStart[2] };

	float d = tetDot_D(lineDir, v);

	float projPos[3] = { lineStart[0] + d * lineDir[0],lineStart[1] + d * lineDir[1] ,lineStart[2] + d * lineDir[2] };

	projPos[0] = point[0] - projPos[0];
	projPos[1] = point[1] - projPos[1];
	projPos[2] = point[2] - projPos[2];

	return tetNormal_D(projPos);
}

__device__ float tetPointPointDistance_D(float* start, float* end) {
	float x = start[0] - end[0];
	float y = start[1] - end[1];
	float z = start[2] - end[2];
	return sqrt(x * x + y * y + z * z);
}

//���ߺ�Բ����
__device__ void tetSolveInsect_D(float* lineDir, float* toolDir, float* VSubO, float radius, float* solve0, float* solve1) {

	float temp0 = tetDot_D(VSubO, toolDir);
	float temp1 = -tetDot_D(lineDir, toolDir);

	float Bvector[3] = { VSubO[0] - temp0 * toolDir[0], VSubO[1] - temp0 * toolDir[1],  VSubO[2] - temp0 * toolDir[2] };
	float Avector[3] = { -lineDir[0] - temp1 * toolDir[0],-lineDir[1] - temp1 * toolDir[1],-lineDir[2] - temp1 * toolDir[2] };


	float A = tetDot_D(Avector, Avector);
	float B = 2.0 * tetDot_D(Avector, Bvector);
	float C = tetDot_D(Bvector, Bvector) - radius * radius;
	float delta = B * B - 4 * A * C;
	*solve0 = (-sqrt(delta) - B) / (2.0 * A);
	*solve1 = (sqrt(delta) - B) / (2.0 * A);
	if (*solve1 < 0.0) *solve1 = 0.0;
}

//���ߺ�����
__device__ void tetSolveInsectSphere_D(float* lineDir, float* VSubO, float radius, float* solve0, float* solve1) {
	float dir[3] = { -lineDir[0],-lineDir[1] ,-lineDir[2] };

	float distance = tetDot_D(VSubO, VSubO);

	float A = 1.0;
	float B = tetDot_D(dir, VSubO) * 2.0;
	float C = distance - radius * radius;

	float delta = B * B - 4 * A * C;

	*solve0 = (-sqrt(delta) - B) / (2.0 * A);
	*solve1 = (sqrt(delta) - B) / (2.0 * A);
	if (*solve1 < 0.0) *solve1 = 0.0;
}

//ʹ��������ײ������������ײ���ж�
__device__ bool cylinderCollisionContinue_without_directDir(
	float length, float moveDistance, float radius,
	float* cylinderPos, float* cylinderLastPos,
	float* cylinderDir,
	float* moveDir, float* position,
	float* t, float* collisionNormal,
	float* collisionPos)
{
	float tt, sol;
	float collisionN[3], collisionP[3];
	// ��ͼ��λ�˵�Բ������ײ��⡣
	bool collision = cylinderCollision(cylinderPos, cylinderDir, position, length, radius, &tt, collisionN, collisionP);
	//bool collision = cylinderRayCollisionDetection(cylinderLastPos, cylinderDir,
	//	position[0], position[1], position[2],
	//	directDir, // �����ָ���������Ͱ󶨵�����ڵ�λ����ء�
	//	length, radius,
	//	&tt, &sol,
	//	collisionN, collisionP);
	float toolMoveVec[3] = { cylinderPos[0] - cylinderLastPos[0],cylinderPos[1] - cylinderLastPos[1], cylinderPos[2] - cylinderLastPos[2] };
	tetNormal_D(toolMoveVec);
	//tetNormal_D(directDir);
	//float cos_val = tetDot_D(toolMoveVec, directDir);
	if (collision)
	{
		collisionNormal[0] = collisionN[0];
		collisionNormal[1] = collisionN[1];
		collisionNormal[2] = collisionN[2];

		collisionPos[0] = collisionP[0];
		collisionPos[1] = collisionP[1];
		collisionPos[2] = collisionP[2];
		return true;
	}

	//���ȼ�����˶�ƽ��ķ�������
	float normal[3];
	tetCross_D(cylinderDir, moveDir, normal);
	tetNormal_D(normal);

	//���������Ҫ�ı���
	float VSubO[3] = { position[0] - cylinderPos[0] ,position[1] - cylinderPos[1] ,position[2] - cylinderPos[2] };//���߼��ָ�������ײ�������
	float lineStart0[3] = { cylinderPos[0] ,cylinderPos[1] ,cylinderPos[2] };// ��ǰ���߼�ˣ��������߼�ˣ�
	float lineStart1[3] = { cylinderLastPos[0] ,cylinderLastPos[1] ,cylinderLastPos[2] };// ��һ֡���߼��(��ͼ�ι��߼��)
	float lineStart2[3] = { cylinderPos[0] + cylinderDir[0] * length ,cylinderPos[1] + cylinderDir[1] * length,cylinderPos[2] + cylinderDir[2] * length };// ��ǰ֡����β������������β����


	//����Ҫ�Ƚ���һ����ײ��⣬�������Ƿ�����ײ��������������Ҫ������ײ���


	//1.�����ھֲ�����ϵ�е����꣬��������ֲ����겻�������ģ����Բ��ܺ�����е����ʹ�ø�˹��Ԫ
	// �����᣺ ���߷���cylinderDir���˶�����moveDir�����߷������˶������ųɵ�ƽ��ķ�����normal
	// ������ײ��������������ɵľֲ�����ϵ����[x, y, z] �ô������������������ϵ�£�x������ײ���ڹ�����ͶӰ��λ�ã�y�������˶������ϵ��˶�����
	// ��˹��Ԫ��[A|I] ֻʹ����֮��ļӼ���A���I��I����A�������
	float x, y, z;
	float det = tetSolveFormula_D(cylinderDir, moveDir, normal, VSubO, &x, &y, &z);

	if (x != x || y != y || z != z) return false;

	float distance = 0.0;
	bool flag = false;
	//2.���������ҵ��������ڵ�����
	if (x > length && y > moveDistance) {
		//����㵽��ľ���
		float basePoint[3] = { cylinderLastPos[0] + length * cylinderDir[0],cylinderLastPos[1] + length * cylinderDir[1] , cylinderLastPos[2] + length * cylinderDir[2] };
		distance = tetPointPointDistance_D(position, basePoint);
		flag = true;
	}
	else if (x > length && y < moveDistance && y>0.0) {
		//����㵽�ߵľ���
		distance = tetPointLineDistance_D(lineStart2, moveDir, position);
	}
	else if (x > length && y < 0.0) {
		distance = tetPointPointDistance_D(position, lineStart2);
	}
	else if (x > 0.0 && x < length && y > moveDistance) {
		distance = tetPointLineDistance_D(lineStart1, cylinderDir, position);
		flag = true;
	}
	else if (x > 0.0 && x < length && y < moveDistance && y>0.0) {
		//����㵽��ľ���
		distance = abs(z);
	}
	else if (x > 0.0 && x < length && y < 0.0) {
		distance = tetPointLineDistance_D(lineStart0, cylinderDir, position);
	}
	else if (x<0.0 && y > moveDistance) {
		distance = tetPointPointDistance_D(position, cylinderLastPos);
		flag = true;
	}
	else if (x < 0.0 && y < moveDistance && y>0.0) {
		distance = tetPointLineDistance_D(lineStart0, moveDir, position);
	}
	else if (x < 0.0 && y < 0.0) {
		distance = tetPointPointDistance_D(position, cylinderPos);
	}


	//3.�жϾ���
	if (distance > radius) return false;
	//if (flag) return false;

	//printf("x:%f,y:%f,z:%f\n", x, y, z);

	//4. �����������ײ�ų�λ��
	//����Ԫһ�η���,�����������Բ�����м���
	float lineDir[3] = { moveDir[0],moveDir[1], moveDir[2] };


	float v0[3] = { position[0] - lineStart0[0] ,position[1] - lineStart0[1] ,position[2] - lineStart0[2] };
	float v1[3] = { position[0] - lineStart1[0] ,position[1] - lineStart1[1] ,position[2] - lineStart1[2] };
	float v2[3] = { position[0] - lineStart2[0] ,position[1] - lineStart2[1] ,position[2] - lineStart2[2] };


	//��Բ���ཻ
	float solve00, solve01;
	float solve10, solve11;
	tetSolveInsect_D(lineDir, cylinderDir, v0, radius, &solve00, &solve01);
	tetSolveInsect_D(lineDir, moveDir, v0, radius, &solve10, &solve11);
	float solve = min(solve11, solve01);
	//tetSolveInsect_D(lineDir, cylinderDir, v1, radius, &solve00, &solve01);
	//solve = min(solve, solve01);
	//tetSolveInsect_D(lineDir, moveDir, v2, radius, &solve00, &solve01);
	//solve = min(solve, solve01);


	//�����ཻ
	float solve20, solve21;
	tetSolveInsectSphere_D(lineDir, v0, radius, &solve20, &solve21);
	solve = min(solve, solve21);
	//printf("%f\n", solve);
	//tetSolveInsectSphere_D(lineDir, v1, radius, &solve10, &solve11);
	//solve = min(solve, solve11);
	//tetSolveInsectSphere_D(lineDir, v2, radius, &solve10, &solve11);
	//solve = min(solve, solve11);
	//tetSolveInsectSphere_D(lineDir, VSubO, radius, &solve10, &solve11);
	//solve = min(solve, solve11);

	if (solve != solve) return false;
	//printf("x:%f,y:%f,z:%f, solve: %f\n",x,y,z, solve);

	//����λ�õõ������ų���λ��
	collisionPos[0] = position[0] - lineDir[0] * solve;
	collisionPos[1] = position[1] - lineDir[1] * solve;
	collisionPos[2] = position[2] - lineDir[2] * solve;

	//���¶������ײ���ߣ��򹤾������Ͻ���ͶӰ
	float projPos[3] = { collisionPos[0] - cylinderPos[0],collisionPos[1] - cylinderPos[1],collisionPos[2] - cylinderPos[2] };
	float proj = tetDot_D(projPos, cylinderDir);
	projPos[0] = collisionPos[0] - cylinderPos[0] - cylinderDir[0] * proj;
	projPos[1] = collisionPos[1] - cylinderPos[1] - cylinderDir[1] * proj;
	projPos[2] = collisionPos[2] - cylinderPos[2] - cylinderDir[2] * proj;

	tetNormal_D(projPos);
	collisionNormal[0] = projPos[0];
	collisionNormal[1] = projPos[1];
	collisionNormal[2] = projPos[2];

	return true;
}

