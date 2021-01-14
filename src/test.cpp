#include <iostream>
#include "../include/Eigen/Eigen"
// #include <Eigen/Eigen>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace Eigen;

const int p = 1000;
const int n = 1000;

int main()
{
    std::chrono::time_point<std::chrono::system_clock> start, end;

    Eigen::MatrixXd a;
    Eigen::setNbThreads(8);
    cout << Eigen::nbThreads() << endl;

    MatrixXd m1(n, p);
    VectorXd m2(n);

    VectorXd m_res(p);

    m1.setRandom();
    m2.setRandom();

    start = std::chrono::system_clock::now();
    m_res.noalias() = m1.transpose() * m2;

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    // start = std::chrono::system_clock::now();
    // end = std::chrono::system_clock::now();
    // elapsed_seconds = end - start;
    start = std::chrono::system_clock::now();
#pragma omp parallel for
    for (int i = 0; i < p; i++)
    {
        m_res(i) = m1.col(i).dot(m2);
    }
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
#pragma omp parallel for
    for (int i = 0; i < p; i++)
    {
        m_res(i) = m1.col(i).dot(m2);
    }
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    system("pause");
    return 0;
}