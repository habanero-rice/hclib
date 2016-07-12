#include <stdio.h>
__device__ inline int hclib_get_current_worker() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template<class functor_type>
__global__ void wrapper_kernel(unsigned niters, functor_type functor) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < niters) {
        functor(tid);
    }
}
template<class functor_type>
static void kernel_launcher(const char *kernel_lbl, unsigned niters, functor_type functor) {
    const int threads_per_block = 256;
    const int nblocks = (niters + threads_per_block - 1) / threads_per_block;
    functor.transfer_to_device();
    const unsigned long long start = capp_current_time_ns();
    wrapper_kernel<<<nblocks, threads_per_block>>>(niters, functor);
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
// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>

struct cfd_double3 { double x, y, z; };

#define block_length 8

/*
 * Options
 *
 */
#define GAMMA 1.4
#define iterations 5

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


/*
 * Generic functions
 */
template <typename T>
T* alloc(int N)
{
	return new T[N];
}

template <typename T>
void dealloc(T* array)
{
	delete[] array;
}

class pragma63_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double* volatile dst;
    double* volatile h_dst;
    double* volatile src;
    double* volatile h_src;

    public:
        pragma63_omp_parallel_hclib_async(double* set_dst,
                double* set_src) {
            h_dst = set_dst;
            h_src = set_src;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        dst = NULL;
        src = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 2, h_dst, h_src);
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
            if (dst == NULL && (char *)h_dst >= (char *)host_allocations[i] && ((char *)h_dst - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_dst - (char *)host_allocations[i]);
                memcpy((void *)(&dst), (void *)(&tmp), sizeof(void *));
            }
            if (src == NULL && (char *)h_src >= (char *)host_allocations[i] && ((char *)h_src - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_src - (char *)host_allocations[i]);
                memcpy((void *)(&src), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(dst || h_dst == NULL);
        assert(src || h_src == NULL);

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
		dst[i] = src[i];
	}
            }
        }
};

void copy(double *dst, double *src, int N)
{
 { const int niters = (N) - (0);
kernel_launcher("pragma63_omp_parallel", niters, pragma63_omp_parallel_hclib_async(dst, src));
 } 
} 


void dump(double* variables, int nel, int nelr)
{


	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << variables[i*NVAR + VAR_DENSITY] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++) file << variables[i*NVAR + (VAR_MOMENTUM+j)] << " ";
			file << std::endl;
		}
	}

	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << variables[i*NVAR + VAR_DENSITY_ENERGY] << std::endl;
	}

}

/*
 * Element-based Cell-centered FVM solver functions
 */
double ff_variable[NVAR];
cfd_double3 ff_flux_contribution_momentum_x;
cfd_double3 ff_flux_contribution_momentum_y;
cfd_double3 ff_flux_contribution_momentum_z;
cfd_double3 ff_flux_contribution_density_energy;


class pragma112_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double* volatile variables;
    double* volatile h_variables;
    volatile double ff_variable[5];

    public:
        pragma112_omp_parallel_hclib_async(double* set_variables,
                double set_ff_variable[5]) {
            h_variables = set_variables;
            memcpy((void *)ff_variable, (void *)set_ff_variable, sizeof(ff_variable));

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        variables = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 1, h_variables);
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
            if (variables == NULL && (char *)h_variables >= (char *)host_allocations[i] && ((char *)h_variables - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_variables - (char *)host_allocations[i]);
                memcpy((void *)(&variables), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(variables || h_variables == NULL);

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
		for(int j = 0; j < NVAR; j++) variables[i*NVAR + j] = ff_variable[j];
	}
            }
        }
};

void initialize_variables(int nelr, double* variables)
{
 { const int niters = (nelr) - (0);
kernel_launcher("pragma112_omp_parallel", niters, pragma112_omp_parallel_hclib_async(variables, ff_variable));
 } 
} 

inline void compute_flux_contribution(double& density, cfd_double3& momentum, double& density_energy, double& pressure, cfd_double3& velocity, cfd_double3& fc_momentum_x, cfd_double3& fc_momentum_y, cfd_double3& fc_momentum_z, cfd_double3& fc_density_energy)
{
	fc_momentum_x.x = velocity.x*momentum.x + pressure;
	fc_momentum_x.y = velocity.x*momentum.y;
	fc_momentum_x.z = velocity.x*momentum.z;

	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y*momentum.y + pressure;
	fc_momentum_y.z = velocity.y*momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z*momentum.z + pressure;

	double de_p = density_energy+pressure;
	fc_density_energy.x = velocity.x*de_p;
	fc_density_energy.y = velocity.y*de_p;
	fc_density_energy.z = velocity.z*de_p;
}

inline void compute_velocity(double& density, cfd_double3& momentum, cfd_double3& velocity)
{
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}

inline double compute_speed_sqd(cfd_double3& velocity)
{
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline double compute_pressure(double& density, double& density_energy, double& speed_sqd)
{
	return (double(GAMMA)-double(1.0))*(density_energy - double(0.5)*density*speed_sqd);
}

inline double compute_speed_of_sound(double& density, double& pressure)
{
	return std::sqrt(double(GAMMA)*pressure/density);
}



class pragma165_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
        __device__ inline void compute_velocity(double& density, cfd_double3& momentum, cfd_double3& velocity) {
            {
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}
        }
        __device__ inline double compute_speed_sqd(cfd_double3& velocity) {
            {
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}
        }
        __device__ inline double compute_pressure(double& density, double& density_energy, double& speed_sqd) {
            {
	return (double(GAMMA)-double(1.0))*(density_energy - double(0.5)*density*speed_sqd);
}
        }
        __device__ inline double compute_speed_of_sound(double& density, double& pressure) {
            {
	return std::sqrt(double(GAMMA)*pressure/density);
}
        }
    double* volatile variables;
    double* volatile h_variables;
    double* volatile step_factors;
    double* volatile h_step_factors;
    double* volatile areas;
    double* volatile h_areas;

    public:
        pragma165_omp_parallel_hclib_async(double* set_variables,
                double* set_step_factors,
                double* set_areas) {
            h_variables = set_variables;
            h_step_factors = set_step_factors;
            h_areas = set_areas;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        variables = NULL;
        step_factors = NULL;
        areas = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 3, h_variables, h_step_factors, h_areas);
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
            if (variables == NULL && (char *)h_variables >= (char *)host_allocations[i] && ((char *)h_variables - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_variables - (char *)host_allocations[i]);
                memcpy((void *)(&variables), (void *)(&tmp), sizeof(void *));
            }
            if (step_factors == NULL && (char *)h_step_factors >= (char *)host_allocations[i] && ((char *)h_step_factors - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_step_factors - (char *)host_allocations[i]);
                memcpy((void *)(&step_factors), (void *)(&tmp), sizeof(void *));
            }
            if (areas == NULL && (char *)h_areas >= (char *)host_allocations[i] && ((char *)h_areas - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_areas - (char *)host_allocations[i]);
                memcpy((void *)(&areas), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(variables || h_variables == NULL);
        assert(step_factors || h_step_factors == NULL);
        assert(areas || h_areas == NULL);

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
		double density = variables[NVAR*i + VAR_DENSITY];

		cfd_double3 momentum;
		momentum.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
		momentum.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
		momentum.z = variables[NVAR*i + (VAR_MOMENTUM+2)];

		double density_energy = variables[NVAR*i + VAR_DENSITY_ENERGY];
		cfd_double3 velocity;	   compute_velocity(density, momentum, velocity);
		double speed_sqd      = compute_speed_sqd(velocity);
		double pressure       = compute_pressure(density, density_energy, speed_sqd);
		double speed_of_sound = compute_speed_of_sound(density, pressure);

		// dt = double(0.5) * std::sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
		step_factors[i] = double(0.5) / (std::sqrt(areas[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
	}
            }
        }
};

void compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
{
 { const int niters = (nelr) - (0);
kernel_launcher("pragma165_omp_parallel", niters, pragma165_omp_parallel_hclib_async(variables, step_factors, areas));
 } 
} 


/*
 *
 *
*/

class pragma196_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
        __device__ inline void compute_velocity(double& density, cfd_double3& momentum, cfd_double3& velocity) {
            {
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}
        }
        __device__ inline double compute_speed_sqd(cfd_double3& velocity) {
            {
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}
        }
        __device__ inline double compute_pressure(double& density, double& density_energy, double& speed_sqd) {
            {
	return (double(GAMMA)-double(1.0))*(density_energy - double(0.5)*density*speed_sqd);
}
        }
        __device__ inline double compute_speed_of_sound(double& density, double& pressure) {
            {
	return std::sqrt(double(GAMMA)*pressure/density);
}
        }
        __device__ inline void compute_flux_contribution(double& density, cfd_double3& momentum, double& density_energy, double& pressure, cfd_double3& velocity, cfd_double3& fc_momentum_x, cfd_double3& fc_momentum_y, cfd_double3& fc_momentum_z, cfd_double3& fc_density_energy) {
            {
	fc_momentum_x.x = velocity.x*momentum.x + pressure;
	fc_momentum_x.y = velocity.x*momentum.y;
	fc_momentum_x.z = velocity.x*momentum.z;

	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y*momentum.y + pressure;
	fc_momentum_y.z = velocity.y*momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z*momentum.z + pressure;

	double de_p = density_energy+pressure;
	fc_density_energy.x = velocity.x*de_p;
	fc_density_energy.y = velocity.y*de_p;
	fc_density_energy.z = velocity.z*de_p;
}
        }
    double* volatile variables;
    double* volatile h_variables;
    int* volatile elements_surrounding_elements;
    int* volatile h_elements_surrounding_elements;
    double* volatile normals;
    double* volatile h_normals;
    volatile double smoothing_coefficient;
    volatile double ff_variable[5];
    struct cfd_double3 ff_flux_contribution_density_energy;
    struct cfd_double3 ff_flux_contribution_momentum_x;
    struct cfd_double3 ff_flux_contribution_momentum_y;
    struct cfd_double3 ff_flux_contribution_momentum_z;
    double* volatile fluxes;
    double* volatile h_fluxes;

    public:
        pragma196_omp_parallel_hclib_async(double* set_variables,
                int* set_elements_surrounding_elements,
                double* set_normals,
                double set_smoothing_coefficient,
                double set_ff_variable[5],
                struct cfd_double3 *set_ff_flux_contribution_density_energy,
                struct cfd_double3 *set_ff_flux_contribution_momentum_x,
                struct cfd_double3 *set_ff_flux_contribution_momentum_y,
                struct cfd_double3 *set_ff_flux_contribution_momentum_z,
                double* set_fluxes) {
            h_variables = set_variables;
            h_elements_surrounding_elements = set_elements_surrounding_elements;
            h_normals = set_normals;
            smoothing_coefficient = set_smoothing_coefficient;
            memcpy((void *)ff_variable, (void *)set_ff_variable, sizeof(ff_variable));
            memcpy((void *)&ff_flux_contribution_density_energy, set_ff_flux_contribution_density_energy, sizeof(struct cfd_double3));
            memcpy((void *)&ff_flux_contribution_momentum_x, set_ff_flux_contribution_momentum_x, sizeof(struct cfd_double3));
            memcpy((void *)&ff_flux_contribution_momentum_y, set_ff_flux_contribution_momentum_y, sizeof(struct cfd_double3));
            memcpy((void *)&ff_flux_contribution_momentum_z, set_ff_flux_contribution_momentum_z, sizeof(struct cfd_double3));
            h_fluxes = set_fluxes;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        variables = NULL;
        elements_surrounding_elements = NULL;
        normals = NULL;
        fluxes = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 4, h_variables, h_elements_surrounding_elements, h_normals, h_fluxes);
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
            if (variables == NULL && (char *)h_variables >= (char *)host_allocations[i] && ((char *)h_variables - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_variables - (char *)host_allocations[i]);
                memcpy((void *)(&variables), (void *)(&tmp), sizeof(void *));
            }
            if (elements_surrounding_elements == NULL && (char *)h_elements_surrounding_elements >= (char *)host_allocations[i] && ((char *)h_elements_surrounding_elements - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_elements_surrounding_elements - (char *)host_allocations[i]);
                memcpy((void *)(&elements_surrounding_elements), (void *)(&tmp), sizeof(void *));
            }
            if (normals == NULL && (char *)h_normals >= (char *)host_allocations[i] && ((char *)h_normals - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_normals - (char *)host_allocations[i]);
                memcpy((void *)(&normals), (void *)(&tmp), sizeof(void *));
            }
            if (fluxes == NULL && (char *)h_fluxes >= (char *)host_allocations[i] && ((char *)h_fluxes - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_fluxes - (char *)host_allocations[i]);
                memcpy((void *)(&fluxes), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(variables || h_variables == NULL);
        assert(elements_surrounding_elements || h_elements_surrounding_elements == NULL);
        assert(normals || h_normals == NULL);
        assert(fluxes || h_fluxes == NULL);

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
		int j, nb;
		cfd_double3 normal; double normal_len;
		double factor;

		double density_i = variables[NVAR*i + VAR_DENSITY];
		cfd_double3 momentum_i;
		momentum_i.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
		momentum_i.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
		momentum_i.z = variables[NVAR*i + (VAR_MOMENTUM+2)];

		double density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];

		cfd_double3 velocity_i;             				 compute_velocity(density_i, momentum_i, velocity_i);
		double speed_sqd_i                          = compute_speed_sqd(velocity_i);
		double speed_i                              = std::sqrt(speed_sqd_i);
		double pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
		double speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
		cfd_double3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
		cfd_double3 flux_contribution_i_density_energy;
		compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

		double flux_i_density = double(0.0);
		cfd_double3 flux_i_momentum;
		flux_i_momentum.x = double(0.0);
		flux_i_momentum.y = double(0.0);
		flux_i_momentum.z = double(0.0);
		double flux_i_density_energy = double(0.0);

		cfd_double3 velocity_nb;
		double density_nb, density_energy_nb;
		cfd_double3 momentum_nb;
		cfd_double3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
		cfd_double3 flux_contribution_nb_density_energy;
		double speed_sqd_nb, speed_of_sound_nb, pressure_nb;

		for(j = 0; j < NNB; j++)
		{
			nb = elements_surrounding_elements[i*NNB + j];
			normal.x = normals[(i*NNB + j)*NDIM + 0];
			normal.y = normals[(i*NNB + j)*NDIM + 1];
			normal.z = normals[(i*NNB + j)*NDIM + 2];
			normal_len = std::sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);

			if(nb >= 0) 	// a legitimate neighbor
			{
				density_nb =        variables[nb*NVAR + VAR_DENSITY];
				momentum_nb.x =     variables[nb*NVAR + (VAR_MOMENTUM+0)];
				momentum_nb.y =     variables[nb*NVAR + (VAR_MOMENTUM+1)];
				momentum_nb.z =     variables[nb*NVAR + (VAR_MOMENTUM+2)];
				density_energy_nb = variables[nb*NVAR + VAR_DENSITY_ENERGY];
													compute_velocity(density_nb, momentum_nb, velocity_nb);
				speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
				pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
				speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
													compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);

				// artificial viscosity
				factor = -normal_len*smoothing_coefficient*double(0.5)*(speed_i + std::sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
				flux_i_density += factor*(density_i-density_nb);
				flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
				flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
				flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
				flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

				// accumulate cell-centered fluxes
				factor = double(0.5)*normal.x;
				flux_i_density += factor*(momentum_nb.x+momentum_i.x);
				flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
				flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
				flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
				flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);

				factor = double(0.5)*normal.y;
				flux_i_density += factor*(momentum_nb.y+momentum_i.y);
				flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
				flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
				flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
				flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);

				factor = double(0.5)*normal.z;
				flux_i_density += factor*(momentum_nb.z+momentum_i.z);
				flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
				flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
				flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
				flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
			}
			else if(nb == -1)	// a wing boundary
			{
				flux_i_momentum.x += normal.x*pressure_i;
				flux_i_momentum.y += normal.y*pressure_i;
				flux_i_momentum.z += normal.z*pressure_i;
			}
			else if(nb == -2) // a far field boundary
			{
				factor = double(0.5)*normal.x;
				flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
				flux_i_density_energy += factor*(ff_flux_contribution_density_energy.x+flux_contribution_i_density_energy.x);
				flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.x + flux_contribution_i_momentum_x.x);
				flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.x + flux_contribution_i_momentum_y.x);
				flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.x + flux_contribution_i_momentum_z.x);

				factor = double(0.5)*normal.y;
				flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
				flux_i_density_energy += factor*(ff_flux_contribution_density_energy.y+flux_contribution_i_density_energy.y);
				flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.y + flux_contribution_i_momentum_x.y);
				flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.y + flux_contribution_i_momentum_y.y);
				flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.y + flux_contribution_i_momentum_z.y);

				factor = double(0.5)*normal.z;
				flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
				flux_i_density_energy += factor*(ff_flux_contribution_density_energy.z+flux_contribution_i_density_energy.z);
				flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.z + flux_contribution_i_momentum_x.z);
				flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.z + flux_contribution_i_momentum_y.z);
				flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.z + flux_contribution_i_momentum_z.z);

			}
		}

		fluxes[i*NVAR + VAR_DENSITY] = flux_i_density;
		fluxes[i*NVAR + (VAR_MOMENTUM+0)] = flux_i_momentum.x;
		fluxes[i*NVAR + (VAR_MOMENTUM+1)] = flux_i_momentum.y;
		fluxes[i*NVAR + (VAR_MOMENTUM+2)] = flux_i_momentum.z;
		fluxes[i*NVAR + VAR_DENSITY_ENERGY] = flux_i_density_energy;
	}
            }
        }
};

void compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes)
{
	double smoothing_coefficient = double(0.2f);

 { const int niters = (nelr) - (0);
kernel_launcher("pragma196_omp_parallel", niters, pragma196_omp_parallel_hclib_async(variables, elements_surrounding_elements, normals, smoothing_coefficient, ff_variable, &ff_flux_contribution_density_energy, &ff_flux_contribution_momentum_x, &ff_flux_contribution_momentum_y, &ff_flux_contribution_momentum_z, fluxes));
 } 
} 

class pragma327_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double* volatile step_factors;
    double* volatile h_step_factors;
    volatile int j;
    double* volatile variables;
    double* volatile h_variables;
    double* volatile old_variables;
    double* volatile h_old_variables;
    double* volatile fluxes;
    double* volatile h_fluxes;

    public:
        pragma327_omp_parallel_hclib_async(double* set_step_factors,
                int set_j,
                double* set_variables,
                double* set_old_variables,
                double* set_fluxes) {
            h_step_factors = set_step_factors;
            j = set_j;
            h_variables = set_variables;
            h_old_variables = set_old_variables;
            h_fluxes = set_fluxes;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        step_factors = NULL;
        variables = NULL;
        old_variables = NULL;
        fluxes = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 4, h_step_factors, h_variables, h_old_variables, h_fluxes);
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
            if (step_factors == NULL && (char *)h_step_factors >= (char *)host_allocations[i] && ((char *)h_step_factors - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_step_factors - (char *)host_allocations[i]);
                memcpy((void *)(&step_factors), (void *)(&tmp), sizeof(void *));
            }
            if (variables == NULL && (char *)h_variables >= (char *)host_allocations[i] && ((char *)h_variables - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_variables - (char *)host_allocations[i]);
                memcpy((void *)(&variables), (void *)(&tmp), sizeof(void *));
            }
            if (old_variables == NULL && (char *)h_old_variables >= (char *)host_allocations[i] && ((char *)h_old_variables - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_old_variables - (char *)host_allocations[i]);
                memcpy((void *)(&old_variables), (void *)(&tmp), sizeof(void *));
            }
            if (fluxes == NULL && (char *)h_fluxes >= (char *)host_allocations[i] && ((char *)h_fluxes - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_fluxes - (char *)host_allocations[i]);
                memcpy((void *)(&fluxes), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(step_factors || h_step_factors == NULL);
        assert(variables || h_variables == NULL);
        assert(old_variables || h_old_variables == NULL);
        assert(fluxes || h_fluxes == NULL);

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
		double factor = step_factors[i]/double(RK+1-j);

		variables[NVAR*i + VAR_DENSITY] = old_variables[NVAR*i + VAR_DENSITY] + factor*fluxes[NVAR*i + VAR_DENSITY];
		variables[NVAR*i + VAR_DENSITY_ENERGY] = old_variables[NVAR*i + VAR_DENSITY_ENERGY] + factor*fluxes[NVAR*i + VAR_DENSITY_ENERGY];
		variables[NVAR*i + (VAR_MOMENTUM+0)] = old_variables[NVAR*i + (VAR_MOMENTUM+0)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+0)];
		variables[NVAR*i + (VAR_MOMENTUM+1)] = old_variables[NVAR*i + (VAR_MOMENTUM+1)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+1)];
		variables[NVAR*i + (VAR_MOMENTUM+2)] = old_variables[NVAR*i + (VAR_MOMENTUM+2)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+2)];
	}
            }
        }
};

void time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
{
 { const int niters = (nelr) - (0);
kernel_launcher("pragma327_omp_parallel", niters, pragma327_omp_parallel_hclib_async(step_factors, j, variables, old_variables, fluxes));
 } 
} 
/*
 * Main function
 */
int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];

{
	// set far field conditions
	{
		const double angle_of_attack = double(3.1415926535897931 / 180.0) * double(deg_angle_of_attack);

		ff_variable[VAR_DENSITY] = double(1.4);

		double ff_pressure = double(1.0);
		double ff_speed_of_sound = sqrt(GAMMA*ff_pressure / ff_variable[VAR_DENSITY]);
		double ff_speed = double(ff_mach)*ff_speed_of_sound;

		cfd_double3 ff_velocity;
		ff_velocity.x = ff_speed*double(cos((double)angle_of_attack));
		ff_velocity.y = ff_speed*double(sin((double)angle_of_attack));
		ff_velocity.z = 0.0;

		ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
		ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
		ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocity.z;

		ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*(double(0.5)*(ff_speed*ff_speed)) + (ff_pressure / double(GAMMA-1.0));

		cfd_double3 ff_momentum;
		ff_momentum.x = *(ff_variable+VAR_MOMENTUM+0);
		ff_momentum.y = *(ff_variable+VAR_MOMENTUM+1);
		ff_momentum.z = *(ff_variable+VAR_MOMENTUM+2);
		compute_flux_contribution(ff_variable[VAR_DENSITY], ff_momentum, ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy);
	}
	int nel;
	int nelr;

	// read in domain geometry
	double* areas;
	int* elements_surrounding_elements;
	double* normals;
	{
		std::ifstream file(data_file_name);

		file >> nel;
		nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));

		areas = new double[nelr];
		elements_surrounding_elements = new int[nelr*NNB];
		normals = new double[NDIM*NNB*nelr];

		// read in data
		for(int i = 0; i < nel; i++)
		{
			file >> areas[i];
			for(int j = 0; j < NNB; j++)
			{
				file >> elements_surrounding_elements[i*NNB + j];
				if(elements_surrounding_elements[i*NNB+j] < 0) elements_surrounding_elements[i*NNB+j] = -1;
				elements_surrounding_elements[i*NNB + j]--; //it's coming in with Fortran numbering

				for(int k = 0; k < NDIM; k++)
				{
					file >>  normals[(i*NNB + j)*NDIM + k];
					normals[(i*NNB + j)*NDIM + k] = -normals[(i*NNB + j)*NDIM + k];
				}
			}
		}

		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			areas[i] = areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				elements_surrounding_elements[i*NNB + j] = elements_surrounding_elements[last*NNB + j];
				for(int k = 0; k < NDIM; k++) normals[(i*NNB + j)*NDIM + k] = normals[(last*NNB + j)*NDIM + k];
			}
		}
	}

	// Create arrays and set initial conditions
	double* variables = alloc<double>(nelr*NVAR);
	initialize_variables(nelr, variables);

	double* old_variables = alloc<double>(nelr*NVAR);
	double* fluxes = alloc<double>(nelr*NVAR);
	double* step_factors = alloc<double>(nelr);

	// these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;

	// Begin iterations
	for(int i = 0; i < iterations; i++)
	{
		copy(old_variables, variables, nelr*NVAR);

		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors);

		for(int j = 0; j < RK; j++)
		{
			compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
			time_step(j, nelr, old_variables, variables, step_factors, fluxes);
		}
	}

	std::cout << "Saving solution..." << std::endl;
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;


	std::cout << "Cleaning up..." << std::endl;
	dealloc<double>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<double>(normals);

	dealloc<double>(variables);
	dealloc<double>(old_variables);
	dealloc<double>(fluxes);
	dealloc<double>(step_factors);
    }

	std::cout << "Done..." << std::endl;

	return 0;
}
