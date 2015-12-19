///////////////////////////////////////////////////////////////////////////////
// This example implements a simple all-pairs n-body gravitational force
// calculation using a 4D vector class called Vec4f. Vec4f uses the HEMI 
// Portable CUDA C/C++ Macros to enable all of the code for the class to be 
// shared between host code compiled by the host compiler and device or host 
// code compiled with the NVIDIA CUDA C/C++ compiler, NVCC. The example
// also shares most of the all-pairs gravitationl force calculation code 
// between device and host, while demonstrating how optimized device 
// implementations can be substituted as needed.
//
// This sample also uses hemi::Array to simplify host/device memory allocation
// and host-device data transfers.
///////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"

#include "vec4f.h"
#include "nbody.h"

#include <time.h>
#include <sys/time.h>

#include "hclib_cpp.h"

extern Vec4f centerOfMass(const Vec4f *bodies, int N);
extern void allPairsForcesCuda(Vec4f *forceVectors, const Vec4f *bodies, int N, bool useShared);

// Nanoseconds
unsigned long long get_clock_gettime() {
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (unsigned long long)t.tv_nsec + s;
}

void allPairsForcesSequential(Vec4f *forceVectors, const Vec4f *bodies, int N) 
{
  for (int i = 0; i < N; i++) {
    forceVectors[i] = accumulateForce(bodies[i], bodies, N);
  }
}

inline float randFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

void randomizeBodies(Vec4f *bodies, int N)
{
  srand(437893);
  for (int i = 0; i < N; i++) {
    Vec4f &b = bodies[i];
    b.x = randFloat(-1000.f, 1000.f);
    b.y = randFloat(-1000.f, 1000.f);
    b.z = randFloat(-1000.f, 1000.f);
    b.w = randFloat(0.1f, 1000.f);
  }
}

int main(int argc, char **argv)
{
  hclib::launch(&argc, argv, []() {
    int N = 16384;
    Vec4f targetBody(0.5f, 0.5f, 0.5f, 10.0f);
    Vec4f *bodies = hclib::allocate_at<Vec4f>(hclib::get_current_place(), N, 0);
    Vec4f *forceVectors = hclib::allocate_at<Vec4f>(hclib::get_current_place(), N,
            0);
    
    randomizeBodies(bodies, N);

    // Call a host function defined in a .cu compilation unit
    // that uses host/device shared class member functions
    Vec4f com = centerOfMass(bodies, N);
    printf("Center of mass is (%f, %f, %f)\n", com.x, com.y, com.z);

    unsigned long long start_time, ns;

    // Call host function defined in a .cpp compilation unit
    // that uses host/device shared functions and class member functions
    printf("CPU: Computing all-pairs gravitational forces on %d bodies\n", N);

    start_time = get_clock_gettime();
    allPairsForcesSequential(forceVectors, bodies, N);
    
    printf("CPU: Force vector 0: (%0.3f, %0.3f, %0.3f), %d: (%0.3f, %0.3f, %0.3f)\n", 
           forceVectors[0].x, 
           forceVectors[0].y, 
           forceVectors[0].z,
           N - 1, forceVectors[N - 1].x, 
           forceVectors[N - 1].y, 
           forceVectors[N - 1].z);

    ns = get_clock_gettime() - start_time;

    printf("CPU: %llu ns\n", ns);

    start_time = get_clock_gettime();

    // Call device function defined in a .cu compilation unit
    // that uses host/device shared functions and class member functions
    printf("GPU: Computing all-pairs gravitational forces on %d bodies\n", N);

    int num_gpus;
    place_t **gpu_places = hclib::get_nvgpu_places(&num_gpus);
    place_t *gpu_pl = gpu_places[0];
    printf("GPU: Using %s\n", hclib::get_place_name(gpu_pl));

    randomizeBodies(bodies, N);

    // Allocation
    Vec4f *d_bodies = hclib::allocate_at<Vec4f>(gpu_pl, N, 0);
    Vec4f *d_forceVectors = hclib::allocate_at<Vec4f>(gpu_pl, N, 0);

    // Transfer in
    hclib::ddf_t *bodies_copy_event = hclib::async_copy(gpu_pl, d_bodies,
            hclib::get_current_place(), bodies, N, NULL, NULL);

    // Kernel
    loop_domain_t loop = {0, N, 1, 128};
    hclib::ddf_t **compute_deps = (hclib::ddf_t **)malloc(
            2 * sizeof(hclib::ddf_t *));
    compute_deps[0] = bodies_copy_event; compute_deps[1] = NULL;
    accumulate_force_functor forces_functor(d_forceVectors, d_bodies, N);
    hclib::ddf_t *compute_event = hclib::forasync1D_future(
            (loop_domain_t *)&loop, forces_functor, FORASYNC_MODE_FLAT, gpu_pl,
            compute_deps);
    // allPairsForcesCuda(d_forceVectors, d_bodies, N, false);

    // Transfer out
    hclib::ddf_t **out_deps = (hclib::ddf_t **)malloc(2 * sizeof(hclib::ddf_t *));
    out_deps[0] = compute_event, out_deps[1] = NULL;
    hclib::ddf_t *forces_copy_event = hclib::async_copy(
            hclib::get_current_place(), forceVectors, gpu_pl, d_forceVectors, N,
            out_deps, NULL);

    // Wait
    hclib::ddf_wait(forces_copy_event);
      
    printf("GPU: Force vector 0: (%0.3f, %0.3f, %0.3f), %d: (%0.3f, %0.3f, %0.3f)\n", 
           forceVectors[0].x, 
           forceVectors[0].y, 
           forceVectors[0].z,
           N - 1, forceVectors[N - 1].x, 
           forceVectors[N - 1].y, 
           forceVectors[N - 1].z);
   
    ns = get_clock_gettime() - start_time;

    printf("GPU: %llu ns\n", ns);

    hclib::free_at<Vec4f>(hclib::get_current_place(), bodies);
    hclib::free_at<Vec4f>(hclib::get_current_place(), forceVectors);
    hclib::free_at<Vec4f>(gpu_pl, d_bodies);
    hclib::free_at<Vec4f>(gpu_pl, d_forceVectors);
/* 
    StartTimer();
    
    // Call a different device function defined in a .cu compilation unit
    // that uses the same host/device shared functions and class member functions 
    // as above
    printf("GPU: Computing optimized all-pairs gravitational forces on %d bodies\n", N);
      
    allPairsForcesCuda(forceVectors.writeOnlyDevicePtr(), bodies.readOnlyDevicePtr(), N, true);
      
    printf("GPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
           forceVectors.readOnlyHostPtr()[0].x, 
           forceVectors.readOnlyHostPtr()[0].y, 
           forceVectors.readOnlyHostPtr()[0].z);
    
    ms = GetTimer();
    
    printf("GPU+shared: %f ms\n", ms);
*/
  });
  
  return 0;
}
