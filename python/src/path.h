//
// Created by jtwok on 2020/3/8.
//
#ifndef SRC_PATH_H
#define SRC_PATH_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;
#else

#include <Eigen\Eigen>
#include "List.h"

#endif

#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"

List sequential_path(Data &data, Algorithm *algorithm, Metric *metric, Eigen::VectorXi sequence, Eigen::VectorXd lambda_seq);

List gs_path(Data &data, Algorithm *algorithm, Metric *metric,
             int s_min, int s_max, int K_max, double epsilon);

double det(double a[], double b[]);

// calculate the intersection of two lines
// if parallal, need_flag = false.
void line_intersection(double line1[2][2], double line2[2][2], double intersection[], bool &need_flag);

// boundary: s=smin, s=max, lambda=lambda_min, lambda_max
// line: crosses p and is parallal to u
// calculate the intersections between boundary and line
void cal_intersections(double p[], double u[], int s_min, int s_max, double lambda_min, double lambda_max, int a[], int b[]);

void golden_section_search(Data &data, Algorithm *algorithm, Metric *metric, double p[], double u[], int s_min, int s_max, double log_lambda_min, double log_lambda_max, double best_arg[],
                           Eigen::VectorXd &beta1, double &coef01, double &train_loss1, double &ic1, Eigen::MatrixXd &ic_sequence);

void seq_search(Data &data, Algorithm *algorithm, Metric *metric, double p[], double u[], int s_min, int s_max, double log_lambda_min, double log_lambda_max, double best_arg[],
                Eigen::VectorXd &beta1, double &coef01, double &train_loss1, double &ic1, int nlambda, Eigen::MatrixXd &ic_sequence);

List pgs_path(Data &data, Algorithm *algorithm, Metric *metric, int s_min, int s_max, double log_lambda_min, double log_lambda_max, int powell_path, int nlambda);

#endif //SRC_PATH_H
