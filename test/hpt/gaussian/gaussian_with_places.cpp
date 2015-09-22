#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "hcpp.h"

#define DOUBLEPAD 1
#define INTPAD 1

hcpp::place_t** cache_pls;

int *maxindex;
double *maxval;

struct timeval tvBegin, tvEnd, tvDiff;


int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;
	return (diff<0);
}

void seed_rand(int thread_n, struct drand48_data *buffer)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	srand48_r(tv.tv_sec * thread_n + tv.tv_usec, buffer);
}


void print_matrix(double** A, int n)
{
	int i,j;
	printf("Matrix:\n");
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			printf("%f ", A[i][j]);
		}
		printf("\n");
	}
}

void print_rhs(double **A, int n)
{
	int i;
	printf("Coefficients:\n");
	for (i = 0; i < n; i++)
		printf("%f\n", A[i][n]);
	printf("\n");
}

void print_vector(double *x, int n)
{
	int i;
	printf("Vector:\n");
	for (i = 0; i < n; i++)
		printf("%f ", x[i]);
	printf("\n");
}

	
double floatabs(double x)
{
        if (x < 0.0)
                return -x;
        return x;
}

void swap(double *a, double *b)
{
	double t = *a;
	*a = *b;
	*b = t;
}

void swap_rows(double **A, int n, int i, int j)
{
        int k;
        for (k = 0; k < n+1; k++)
                swap(&A[i][k], &A[j][k]);
}

void find_max_thread(double **A, int n , int j, int numthreads, int threadid)
{
        int i;
        double t, max = -1;
        int index = -1;

        for (i = n - 1 - threadid; i >= j; i -= numthreads) {
                if ((t = floatabs(A[i][j])) > max) {
                        max = t;
                        index = i;
                }
        }

        maxval[DOUBLEPAD*threadid] = max;
        maxindex[INTPAD*threadid] = index;
}

int maxloc(int numthreads)
{
        int i;
        int globalMaxIndex = maxindex[0];
        double globalMax = maxval[0];

        for (i = 1; i < numthreads; i++) {
                if (maxval[DOUBLEPAD*i] > globalMax) {
                        globalMaxIndex = maxindex[INTPAD*i];
                        globalMax = maxval[DOUBLEPAD*i];
                }
        }
        return globalMaxIndex;
}

double** gaussian(double** orig, int n, int numthreads)
{
	double **A = (double**)malloc(n * sizeof(double*));
        maxindex = (int*)malloc(INTPAD*numthreads * sizeof(int));
        maxval = (double*)malloc(DOUBLEPAD*numthreads * sizeof(double));

	hcpp::finish([=]() {
		for (int p = 0; p < numthreads; p++) {
			hcpp::asyncAt (cache_pls[p+2], [=]() {
				for (int i = n-1-p; i >= 0; i -= numthreads)
					A[i] = (double*)malloc((n+1) * sizeof(double));
			});
		}
	});

	hcpp::finish([=]() {
		for (int p = 0; p < numthreads; p++) {
			struct drand48_data drand_buffer;
			seed_rand(p, &drand_buffer);
			hcpp::asyncAt (cache_pls[p+2], [=,&drand_buffer]() {
				for (int i = n-1-p; i >= 0; i -= numthreads)
					for (int j = 0; j < n+1; j++) {
						drand48_r(&drand_buffer, &A[i][j]);
						orig[i][j] = A[i][j];
					}
			});
		}
	});


	gettimeofday(&tvBegin, NULL);
	//print_matrix(A, n);
	//print_rhs(A, n);
	
        for (int j = 0; j < n-1; j++) {
		for (int p = 0; p < numthreads; p++) {
			hcpp::finish([=]() {
                		hcpp::asyncAt (cache_pls[p+2], [=]() { 
					find_max_thread(A, n , j, numthreads, p);
				});
			});
		}
                int ksave = maxloc(numthreads);
                swap_rows(A, n, ksave, j);

		hcpp::finish([=]() {
			for (int p = 0; p < numthreads; p++) {
				hcpp::asyncAt (cache_pls[p+2], [=]() {
                			for (int k = n - 1 - p; k >= j+1; k -= numthreads) {
                        			double m = A[k][j] / A[j][j];
                        			for (int i = j; i <= n; i++)
                                			A[k][i] = A[k][i] - m * A[j][i];
					}
				});
                	}

        	});
	}
	return A;

}


double* back_substitute(double **A, int n)
{
	int i, j;
	double *x = (double*)malloc(n * sizeof(double));

	for (i = n-1; i >= 0; i--) {
		double sum = A[i][n];
		for (j = n-1; j > i; j--)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}
	return x;
}
		
void verify(double **A, double* x, int n)
{
	int i, j;
	double l2norm = 0.0;
	double* residual = (double*)malloc(n*sizeof(double));
	for (i = 0; i < n; i++) {
		residual[i] = 0.0;
		for (j = 0; j < n; j++)
			residual[i] += A[i][j] * x[j];
		residual[i] -= A[i][n];
		l2norm += residual[i] * residual[i];
	}

	l2norm = sqrt(l2norm);


	printf("L2 Norm = %f\n", l2norm);
	
}

int main(int argc, char** argv)
{
	hcpp::init(&argc, argv);
	double **orig, **A, *x;
	int n, i;


        int numcaches = hcpp::hc_get_num_places(hcpp::CACHE_PLACE);
        //printf("numplaces: %d\n", numcaches);
        cache_pls = (hcpp::place_t**)malloc(sizeof(hcpp::place_t*)*numcaches);
        hcpp::hc_get_places(cache_pls, hcpp::CACHE_PLACE);


	printf("Enter the size of the matrix: \n");
	scanf("%d", &n);

	
	orig = (double**)malloc(n * sizeof(double*));
	for (i = 0; i < n; i++)
		orig[i] = (double*)malloc((n+1) * sizeof(double));
	A = gaussian(orig, n, numcaches-2);
	x = back_substitute(A, n);
	gettimeofday(&tvEnd, NULL);
	timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
	verify(orig, x, n);
	printf("\nGaussian Elimination with places\n");
	printf("\nTime : %ld.%06ld\n\n", tvDiff.tv_sec, tvDiff.tv_usec);

	//print_matrix(A, n);
	//print_rhs(A, n);
	//print_vector(x, n);

	hcpp::finalize();
	return 0;
}
