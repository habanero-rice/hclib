#include <stdio.h>
__device__ inline int hclib_get_current_worker() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template<class functor_type>
__global__ void wrapper_kernel(unsigned iter_offset, unsigned niters, functor_type functor) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < niters) {
        functor(iter_offset + tid);
    }
}
template<class functor_type>
static void kernel_launcher(const char *kernel_lbl, unsigned iter_offset, unsigned niters, functor_type functor) {
    const int threads_per_block = 256;
    const int nblocks = (niters + threads_per_block - 1) / threads_per_block;
    functor.transfer_to_device();
    const unsigned long long start = capp_current_time_ns();
    wrapper_kernel<<<nblocks, threads_per_block>>>(iter_offset, niters, functor);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error while synchronizing kernel - %s\n", cudaGetErrorString(err));
        exit(2);
    }
    const unsigned long long end = capp_current_time_ns();
    fprintf(stderr, "%s %llu ns\n", kernel_lbl, end - start);
    functor.transfer_from_device();
}
#ifdef __cplusplus
#ifdef __CUDACC__
#endif
#endif
/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"
#include <omp.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);
extern int num_omp_threads;

int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float *pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index, i;
    float min_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts + (i * nfeatures), nfeatures);  /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims)
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}


/*----< kmeans_clustering() >---------------------------------------------*/
class pragma173_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
        __device__ int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float *pts,         /* [npts][nfeatures] */
                       int     npts) {
            {
    int index, i;
    float min_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts + (i * nfeatures), nfeatures);  /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}
        }
        __device__ __inline
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims) {
            {
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}
        }
    int index;
    float* volatile feature;
    float* volatile h_feature;
    int i;
    int nfeatures;
    float* volatile clusters;
    float* volatile h_clusters;
    int nclusters;
    int* volatile membership;
    int* volatile h_membership;
    float delta;
    int* volatile partial_new_centers_len;
    int* volatile h_partial_new_centers_len;
    int j;
    float* volatile partial_new_centers;
    float* volatile h_partial_new_centers;

    public:
        pragma173_omp_parallel_hclib_async(int set_index,
                float* set_feature,
                int set_i,
                int set_nfeatures,
                float* set_clusters,
                int set_nclusters,
                int* set_membership,
                float set_delta,
                int* set_partial_new_centers_len,
                int set_j,
                float* set_partial_new_centers) {
            index = set_index;
            h_feature = set_feature;
            i = set_i;
            nfeatures = set_nfeatures;
            h_clusters = set_clusters;
            nclusters = set_nclusters;
            h_membership = set_membership;
            delta = set_delta;
            h_partial_new_centers_len = set_partial_new_centers_len;
            j = set_j;
            h_partial_new_centers = set_partial_new_centers;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        feature = NULL;
        clusters = NULL;
        membership = NULL;
        partial_new_centers_len = NULL;
        partial_new_centers = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 5, h_feature, h_clusters, h_membership, h_partial_new_centers_len, h_partial_new_centers);
        device_allocations = (void **)malloc(nallocations * sizeof(void *));
        for (i = 0; i < nallocations; i++) {
            err = cudaMalloc((void **)&device_allocations[i], host_allocation_sizes[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            err = cudaMemcpy((void *)device_allocations[i], (void *)host_allocations[i], host_allocation_sizes[i], cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            if (feature == NULL && (char *)h_feature >= (char *)host_allocations[i] && ((char *)h_feature - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_feature - (char *)host_allocations[i]);
                memcpy((void *)(&feature), (void *)(&tmp), sizeof(void *));
            }
            if (clusters == NULL && (char *)h_clusters >= (char *)host_allocations[i] && ((char *)h_clusters - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_clusters - (char *)host_allocations[i]);
                memcpy((void *)(&clusters), (void *)(&tmp), sizeof(void *));
            }
            if (membership == NULL && (char *)h_membership >= (char *)host_allocations[i] && ((char *)h_membership - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_membership - (char *)host_allocations[i]);
                memcpy((void *)(&membership), (void *)(&tmp), sizeof(void *));
            }
            if (partial_new_centers_len == NULL && (char *)h_partial_new_centers_len >= (char *)host_allocations[i] && ((char *)h_partial_new_centers_len - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_partial_new_centers_len - (char *)host_allocations[i]);
                memcpy((void *)(&partial_new_centers_len), (void *)(&tmp), sizeof(void *));
            }
            if (partial_new_centers == NULL && (char *)h_partial_new_centers >= (char *)host_allocations[i] && ((char *)h_partial_new_centers - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_partial_new_centers - (char *)host_allocations[i]);
                memcpy((void *)(&partial_new_centers), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(feature || h_feature == NULL);
        assert(clusters || h_clusters == NULL);
        assert(membership || h_membership == NULL);
        assert(partial_new_centers_len || h_partial_new_centers_len == NULL);
        assert(partial_new_centers || h_partial_new_centers == NULL);

    }

    void transfer_from_device() {
        cudaError_t err;
        int i;
        for (i = 0; i < nallocations; i++) {
            err = cudaMemcpy((void *)host_allocations[i], (void *)device_allocations[i], host_allocation_sizes[i], cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            err = cudaFree(device_allocations[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        }
    }

        __device__ void operator()(int i) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
	        /* find the index of nestest cluster centers */					
            int tid = hclib_get_current_worker();				
	        index = find_nearest_point(feature + (i * nfeatures),
		             nfeatures,
		             clusters,
		             nclusters);				
	        /* if membership changes, increase delta by 1 */
	        if (membership[i] != index) delta += 1.0;

	        /* assign the membership to object i */
	        membership[i] = index;
				
	        /* update new cluster centers : sum of all objects located
		       within */
	        partial_new_centers_len[tid * nclusters + index]++;				
	        for (j=0; j<nfeatures; j++)
		       partial_new_centers[tid * nclusters * nfeatures + index * nfeatures + j] += feature[i * nfeatures + j];
            }
            }
        }
};

float* kmeans_clustering(float *feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{

    int      i, j, k, n=0, index, loop=0;
    int     *new_centers_len;			/* [nclusters]: no. of points in each cluster */
	float  **new_centers;				/* [nclusters][nfeatures] */
	float  *clusters;					/* out: [nclusters][nfeatures] */
    float    delta;
        
    double   timing;

	int      nthreads;
    int    *partial_new_centers_len;
    float *partial_new_centers;

    nthreads = num_omp_threads; 

    /* allocate space for returning variable clusters[] */
    clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));

    /* randomly pick cluster centers */
    for (i=0; i<nclusters; i++) {
        //n = (int)rand() % npoints;
        for (j=0; j<nfeatures; j++)
            clusters[i * nfeatures + j] = feature[n * nfeatures + j];
		n++;
    }

    for (i=0; i<npoints; i++)
		membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;


    partial_new_centers_len = (int *)calloc(nthreads * nclusters, sizeof(int));

    partial_new_centers = (float *)calloc(nthreads * nclusters * nfeatures, sizeof(float));

    printf("num of threads = %d\n", num_omp_threads);
    do {
        delta = 0.0;
        {
 { const int niters = (npoints) - (0);
const int iters_offset = (0);
kernel_launcher("pragma173_omp_parallel", iters_offset, niters, pragma173_omp_parallel_hclib_async(index, feature, i, nfeatures, clusters, nclusters, membership, delta, partial_new_centers_len, j, partial_new_centers));
 } 
        } /* end of #pragma omp parallel */

        /* let the main thread perform the array reduction */
        for (i=0; i<nclusters; i++) {
            for (j=0; j<nthreads; j++) {
                new_centers_len[i] += partial_new_centers_len[j * nclusters + i];
                partial_new_centers_len[j * nclusters + i] = 0.0;
                for (k=0; k<nfeatures; k++) {
                    new_centers[i][k] += partial_new_centers[j * nclusters * nfeatures + i * nfeatures + k];
                    partial_new_centers[j * nclusters * nfeatures + i * nfeatures + k] = 0.0;
                }
            }
        }    

		/* replace old cluster centers with new_centers */
		for (i=0; i<nclusters; i++) {
            for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0)
					clusters[i * nfeatures + j] = new_centers[i][j] / new_centers_len[i];
				new_centers[i][j] = 0.0;   /* set back to 0 */
			}
			new_centers_len[i] = 0;   /* set back to 0 */
		}
        
    } while (delta > threshold && loop++ < 500);

    
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
} 

