#ifndef NBODY_H
#define NBODY_H

// Compute gravitational force between two bodies.
// Body mass is stored in w component of the Vec4f.
__host__ __device__ Vec4f gravitation(const Vec4f &i, const Vec4f &j);

// Compute the gravitational force induced on "target" body by all 
// masses in the bodies array.
__host__ __device__ Vec4f accumulateForce(const Vec4f &target,
        const Vec4f *bodies, int N);

class accumulate_force_functor {
    private:
        Vec4f *forceVectors;
        Vec4f *bodies;
        int N;

    public:
        accumulate_force_functor(Vec4f *set_forceVectors, Vec4f *set_bodies,
                int set_N);

        __host__ __device__ void operator()(int idx);
};

#endif
