#pragma once
#include <Eigen/Core>

typedef double ValType;
typedef Eigen::Matrix<ValType, 3, 3> mat3;
typedef Eigen::Matrix<ValType, 3, 1> vec3;
typedef Eigen::Matrix<ValType, 6, 1> vec6;
typedef Eigen::Matrix<ValType, 6, 6> mat6;

struct collisionInfoForQuad
{
	//���ߵİ뾶�ͳ���
	float	toolr, tooll;
	//��İ뾶��λ������
	float	objx, objy, objz, objr;
	//�����幤�ߵķ���
	float	dirx, diry, dirz;
	//��ײ����
	float	zone;
	//ָ������
	float	dirAx, dirAy, dirAz;
};

struct CollisionInfo
{
	// ������Ϣ�����߰뾶�����߳��ȣ����߳���
	float toolr, tooll;
	float tool_dir_x, tool_dir_y, tool_dir_z;

	// ������ײ�ı��涥����Ϣ��λ�ã���������
	float point_x, point_y, point_z;
	float normal_x, normal_y, normal_z;
};

class Point
{
public:
	Point();
	Point(vec3 p, vec3 n);
	bool normalize();
	vec3 pos;
	vec3 normal;
};