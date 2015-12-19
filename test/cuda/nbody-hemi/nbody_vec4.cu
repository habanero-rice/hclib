#define HEMI_DEBUG
#include "vec4f.h"
#include "nbody.h"
#include <stdio.h>

// Softening constant prevents division by zero
#define softeningSquared 0.01f

// Compute gravitational force between two bodies.
// Body mass is stored in w component of the Vec4f.
__host__ __device__ Vec4f gravitation(const Vec4f &i, const Vec4f &j)
{
  Vec4f ij = j - i;
  ij.w = 0;

  float invDist = ij.inverseLength(softeningSquared);
  
  return ij * (j.w * invDist * invDist * invDist);
}

// Compute the gravitational force induced on "target" body by all 
// masses in the bodies array.
__host__ __device__ Vec4f accumulateForce(const Vec4f &target,
        const Vec4f *bodies, int N) {
  Vec4f force(0, 0, 0, 0);

  for (int j = 0; j < N; j++) {
    force += gravitation(target, bodies[j]);
  }

  return force * target.w;
}

accumulate_force_functor::accumulate_force_functor(Vec4f *set_forceVectors, Vec4f *set_bodies,
        int set_N) : forceVectors(set_forceVectors), bodies(set_bodies),
        N(set_N) { }

__host__ __device__ void accumulate_force_functor::operator()(int idx) {
    forceVectors[idx] = accumulateForce(bodies[idx], bodies, N);
}

// Example of using a host/device class from host code in a .cu file
// Sum the masses of the bodies
Vec4f centerOfMass(const Vec4f *bodies, int N) 
{
  float totalMass = 0.0f;
  Vec4f com = Vec4f(0, 0, 0, 0);
  for (int i = 0; i < N; i++) {
    totalMass += bodies[i].w;
    com += Vec4f(bodies[i].x, bodies[i].y, bodies[i].z, 0) * bodies[i].w;
  }
  com /= totalMass;
  return com;
}
