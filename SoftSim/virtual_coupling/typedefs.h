#pragma once
#include <Eigen/Core>

typedef double ValType;
typedef Eigen::Matrix<ValType, 3, 3> mat3;
typedef Eigen::Matrix<ValType, 3, 1> vec3;
typedef Eigen::Matrix<ValType, 6, 1> vec6;
typedef Eigen::Matrix<ValType, 6, 6> mat6;

struct collisionInfoForQuad
{
	//工具的半径和长度
	float	toolr, tooll;
	//球的半径和位置坐标
	float	objx, objy, objz, objr;
	//胶囊体工具的方向
	float	dirx, diry, dirz;
	//碰撞区域
	float	zone;
	//指导向量
	float	dirAx, dirAy, dirAz;
};

struct CollisionInfo
{
	// 工具信息（工具半径，工具长度，工具朝向）
	float toolr, tooll;
	float tool_dir_x, tool_dir_y, tool_dir_z;

	// 发生碰撞的表面顶点信息（位置，法向量）
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