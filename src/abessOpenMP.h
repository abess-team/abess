#ifndef SRC_ABESSOPENMP_H
#define SRC_ABESSOPENMP_H

#ifdef _OPENMP
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#else
#ifndef DISABLE_OPENMP
#ifndef R_BUILD
// use pragma message instead of warning
#pragma message(                                         \
    "Warning: OpenMP is not available, "                 \
    "project will be compiled into single-thread code. " \
    "Use OpenMP-enabled compiler to get benefit of multi-threading.")
#endif
#endif
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_num_procs() { return 1; }
inline void omp_set_num_threads(int nthread) {}
inline void omp_set_dynamic(int flag) {}
#endif

#endif  // SRC_ABESSOPENMP_H
