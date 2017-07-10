#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "getoptions.h"

/* Used by example programs to evaluate command line options */

void get_options(int argc, char *argv[], const char *specs[], int *types, ...) {
    va_list ap;
    int type, i;
    int *intval;
    double *doubleval;
    long *longval;
    char *stringval;

    va_start(ap, types);

    while (((type = *types++) != 0) && (specs != 0)) {
        switch (type) {
            case INTARG:
                intval = (int *)va_arg(ap, int *);
                for (i = 1; i < (argc - 1); i++)
                    if (!(strcmp(argv[i], specs[0]))) {
                        *intval = atoi(argv[i + 1]);
                        argv[i][0] = 0;
                        argv[i + 1][0] = 0;
                    }
                break;
            case DOUBLEARG:
                doubleval = (double *)va_arg(ap, double *);
                for (i = 1; i < (argc - 1); i++)
                    if (!(strcmp(argv[i], specs[0]))) {
                        *doubleval = atof(argv[i + 1]);
                        argv[i][0] = 0;
                        argv[i + 1][0] = 0;
                    }
                break;
            case LONGARG:
                longval = (long *)va_arg(ap, long *);
                for (i = 1; i < (argc - 1); i++)
                    if (!(strcmp(argv[i], specs[0]))) {
                        *longval = atol(argv[i + 1]);
                        argv[i][0] = 0;
                        argv[i + 1][0] = 0;
                    }
                break;
            case BOOLARG:
                intval = (int *)va_arg(ap, int *);
                *intval = 0;
                for (i = 1; i < argc; i++)
                    if (!(strcmp(argv[i], specs[0]))) {
                        *intval = 1;
                        argv[i][0] = 0;
                    }
                break;
            case STRINGARG:
                stringval = (char *)va_arg(ap, char *);
                for (i = 1; i < (argc - 1); i++)
                    if (!(strcmp(argv[i], specs[0]))) {
                        strcpy(stringval, argv[i + 1]);
                        argv[i][0] = 0;
                        argv[i + 1][0] = 0;
                    }
                break;
            case BENCHMARK:
                intval = (int *)va_arg(ap, int *);
                *intval = 0;
                for (i = 1; i < argc; i++) {
                    if (!(strcmp(argv[i], specs[0]))) {
                        *intval = 2;
                        if ((i + 1) < argc) {
                            if (!(strcmp(argv[i + 1], "short"))) *intval = 1;
                            if (!(strcmp(argv[i + 1], "medium"))) *intval = 2;
                            if (!(strcmp(argv[i + 1], "long"))) *intval = 3;
                            argv[i + 1][0] = 0;
                        }
                        argv[i][0] = 0;
                    }
                }
                break;
        }
        specs++;
    }
    va_end(ap);

    for (i = 1; i < argc; i++)
        if (argv[i][0] != 0) printf("\nInvalid option: %s\n", argv[i]);
}

#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

static unsigned long rand_nxt = 0;

int hc_rand(void) {
    int result;
    rand_nxt = rand_nxt * 1103515245 + 12345;
    result = (rand_nxt >> 16) % ((unsigned int)RAND_MAX + 1);
    return result;
}
