#ifndef KASTORS_MAIN_H
#define KASTORS_MAIN_H

struct user_parameters {
    int check;
    int succeed;
    char* string2display;
    int niter;
    int titer;
    int matrix_size;
    int submatrix_size;
    int blocksize;
    int iblocksize;
    int cutoff_depth;
    int cutoff_size;
};

extern double run(struct user_parameters* params);

#endif
