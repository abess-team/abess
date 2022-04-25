#include<iostream>
#include<cstring>
#include<nlopt.h>

string foo(){
    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, 10);
    // std::cout << nlopt_algorithm_to_string(nlopt_get_algorithm(opt)) << std::endl;
    string s = nlopt_algorithm_to_string(nlopt_get_algorithm(opt));
    nlopt_destroy(opt);
    return s;
}
