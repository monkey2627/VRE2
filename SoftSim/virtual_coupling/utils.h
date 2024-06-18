#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "typedefs.h"
#include <cmath>

using namespace std;

double l2dis(vec3 p1, vec3 p2=vec3(0,0,0));
void print_line();
void print_vec3(vec3 v);
void print_vec6(Eigen::Matrix<double, 6, 1> v);
void print_vec6(vector<float> v);
void print_mat3(mat3 m);

inline string RemoveComments(string line)
{
	int comment_index = line.find("#");
	return line.substr(0, comment_index);
}

template <typename T>
inline T string_to_T(std::string const& val) {
	std::istringstream istr(val);
	T returnVal;
	if (!(istr >> returnVal))
		std::cout << "CFG: Not a valid " + (std::string) typeid(T).name() +
		" received!\n";
	return returnVal;
}

