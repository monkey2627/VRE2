#include "utils.h"

double l2dis(vec3 p1, vec3 p2)// ≈∑ œæ‡¿Î
{
    vec3 v = p1 - p2;
    return sqrt(v.dot(v));
}

void print_line()
{
    cout << "-----------------------" << endl;
}

void print_vec3(vec3 v)
{
    printf_s("[%lf %lf %lf]\n", v(0), v(1), v(2));
}

void print_vec6(Eigen::Matrix<double, 6, 1> v)
{
    printf_s("[%lf %lf %lf %lf %lf %lf]\n", v(0), v(1), v(2), v(3), v(4), v(5));
}

void print_vec6(vector<float> v)
{
    printf_s("[%lf %lf %lf %lf %lf %lf]\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

void print_mat3(mat3 m)
{
    cout << m << endl;
}