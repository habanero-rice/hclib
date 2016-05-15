#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
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
                       float **pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index, i;
    float min_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
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
typedef struct _pragma190_omp_parallel {
    int i;
    int j;
    int (*k_ptr);
    int (*n_ptr);
    int index;
    int (*loop_ptr);
    int (*(*new_centers_len_ptr));
    float (*(*(*new_centers_ptr)));
    float (*(*(*clusters_ptr)));
    float delta;
    double (*timing_ptr);
    int (*nthreads_ptr);
    int (*(*(*partial_new_centers_len_ptr)));
    float (*(*(*(*partial_new_centers_ptr))));
    float (*(*(*feature_ptr)));
    int nfeatures;
    int npoints;
    int nclusters;
    float (*threshold_ptr);
    int (*(*membership_ptr));
    pthread_mutex_t reduction_mutex;
 } pragma190_omp_parallel;

static void pragma190_omp_parallel_hclib_async(void *____arg, const int ___iter0);
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{

    int      i, j, k, n=0, index, loop=0;
    int     *new_centers_len;			/* [nclusters]: no. of points in each cluster */
	float  **new_centers;				/* [nclusters][nfeatures] */
	float  **clusters;					/* out: [nclusters][nfeatures] */
    float    delta;
        
    double   timing;

	int      nthreads;
    int    **partial_new_centers_len;
    float ***partial_new_centers;

    nthreads = num_omp_threads; 

    /* allocate space for returning variable clusters[] */
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

    /* randomly pick cluster centers */
    for (i=0; i<nclusters; i++) {
        //n = (int)rand() % npoints;
        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[n][j];
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


    partial_new_centers_len    = (int**) malloc(nthreads * sizeof(int*));
    partial_new_centers_len[0] = (int*)  calloc(nthreads*nclusters, sizeof(int));
    for (i=1; i<nthreads; i++)
		partial_new_centers_len[i] = partial_new_centers_len[i-1]+nclusters;

	partial_new_centers    =(float***)malloc(nthreads * sizeof(float**));
    partial_new_centers[0] =(float**) malloc(nthreads*nclusters * sizeof(float*));
    for (i=1; i<nthreads; i++)
        partial_new_centers[i] = partial_new_centers[i-1] + nclusters;

	for (i=0; i<nthreads; i++)
	{
        for (j=0; j<nclusters; j++)
            partial_new_centers[i][j] = (float*)calloc(nfeatures, sizeof(float));
	}
	printf("num of threads = %d\n", num_omp_threads);
    do {
        delta = 0.0;
        {
 { 
pragma190_omp_parallel *new_ctx = (pragma190_omp_parallel *)malloc(sizeof(pragma190_omp_parallel));
new_ctx->i = i;
new_ctx->j = j;
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->index = index;
new_ctx->loop_ptr = &(loop);
new_ctx->new_centers_len_ptr = &(new_centers_len);
new_ctx->new_centers_ptr = &(new_centers);
new_ctx->clusters_ptr = &(clusters);
new_ctx->delta = delta;
new_ctx->timing_ptr = &(timing);
new_ctx->nthreads_ptr = &(nthreads);
new_ctx->partial_new_centers_len_ptr = &(partial_new_centers_len);
new_ctx->partial_new_centers_ptr = &(partial_new_centers);
new_ctx->feature_ptr = &(feature);
new_ctx->nfeatures = nfeatures;
new_ctx->npoints = npoints;
new_ctx->nclusters = nclusters;
new_ctx->threshold_ptr = &(threshold);
new_ctx->membership_ptr = &(membership);
new_ctx->delta = 0;
const int init_err = pthread_mutex_init(&new_ctx->reduction_mutex, NULL);
assert(init_err == 0);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = npoints;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma190_omp_parallel_hclib_async, new_ctx, 1, domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(new_ctx);
delta = new_ctx->delta;
 } 
        } /* end of #pragma omp parallel */

        /* let the main thread perform the array reduction */
        for (i=0; i<nclusters; i++) {
            for (j=0; j<nthreads; j++) {
                new_centers_len[i] += partial_new_centers_len[j][i];
                partial_new_centers_len[j][i] = 0.0;
                for (k=0; k<nfeatures; k++) {
                    new_centers[i][k] += partial_new_centers[j][i][k];
                    partial_new_centers[j][i][k] = 0.0;
                }
            }
        }    

		/* replace old cluster centers with new_centers */
		for (i=0; i<nclusters; i++) {
            for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];
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
static void pragma190_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma190_omp_parallel *ctx = (pragma190_omp_parallel *)____arg;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int index; index = ctx->index;
    float delta; delta = ctx->delta;
    int nfeatures; nfeatures = ctx->nfeatures;
    int npoints; npoints = ctx->npoints;
    int nclusters; nclusters = ctx->nclusters;
    hclib_start_finish();
    do {
    i = ___iter0;
{
	        /* find the index of nestest cluster centers */					
            int tid = hclib_get_current_worker();
	        index = find_nearest_point((*(ctx->feature_ptr))[i],
		             nfeatures,
		             (*(ctx->clusters_ptr)),
		             nclusters);				
	        /* if membership changes, increase delta by 1 */
	        if ((*(ctx->membership_ptr))[i] != index) delta += 1.0;

	        /* assign the membership to object i */
	        (*(ctx->membership_ptr))[i] = index;
				
	        /* update new cluster centers : sum of all objects located
		       within */
	        (*(ctx->partial_new_centers_len_ptr))[tid][index]++;				
	        for (j=0; j<nfeatures; j++)
		       (*(ctx->partial_new_centers_ptr))[tid][index][j] += (*(ctx->feature_ptr))[i][j];
            } ;     } while (0);
    const int lock_err = pthread_mutex_lock(&ctx->reduction_mutex);
    assert(lock_err == 0);
    ctx->delta += delta;
    const int unlock_err = pthread_mutex_unlock(&ctx->reduction_mutex);
    assert(unlock_err == 0);
    ; hclib_end_finish_nonblocking();

}



