#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
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

typedef struct _pragma63_omp_parallel {
    double (*(*dst_ptr));
    double (*(*src_ptr));
    int (*N_ptr);
 } pragma63_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma63_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int i) {
        }
};

#else
static void pragma63_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
void copy(double *dst, double *src, int N)
{
 { 
pragma63_omp_parallel *new_ctx = (pragma63_omp_parallel *)malloc(sizeof(pragma63_omp_parallel));
new_ctx->dst_ptr = &(dst);
new_ctx->src_ptr = &(src);
new_ctx->N_ptr = &(N);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = N;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((N) - (0), pragma63_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma63_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 
} 

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma63_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma63_omp_parallel *ctx = (pragma63_omp_parallel *)____arg;
    do {
    int i;     i = ___iter0;
{
		(*(ctx->dst_ptr))[i] = (*(ctx->src_ptr))[i];
	} ;     } while (0);
}

#endif



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


typedef struct _pragma112_omp_parallel {
    int (*nelr_ptr);
    double (*(*variables_ptr));
 } pragma112_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma112_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int i) {
        }
};

#else
static void pragma112_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
void initialize_variables(int nelr, double* variables)
{
 { 
pragma112_omp_parallel *new_ctx = (pragma112_omp_parallel *)malloc(sizeof(pragma112_omp_parallel));
new_ctx->nelr_ptr = &(nelr);
new_ctx->variables_ptr = &(variables);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = nelr;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((nelr) - (0), pragma112_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma112_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 
} 

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma112_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma112_omp_parallel *ctx = (pragma112_omp_parallel *)____arg;
    do {
    int i;     i = ___iter0;
{
		for(int j = 0; j < NVAR; j++) (*(ctx->variables_ptr))[i*NVAR + j] = ff_variable[j];
	} ;     } while (0);
}

#endif


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



typedef struct _pragma165_omp_parallel {
    int (*nelr_ptr);
    double (*(*variables_ptr));
    double (*(*areas_ptr));
    double (*(*step_factors_ptr));
 } pragma165_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma165_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int i) {
        }
};

#else
static void pragma165_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
void compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
{
 { 
pragma165_omp_parallel *new_ctx = (pragma165_omp_parallel *)malloc(sizeof(pragma165_omp_parallel));
new_ctx->nelr_ptr = &(nelr);
new_ctx->variables_ptr = &(variables);
new_ctx->areas_ptr = &(areas);
new_ctx->step_factors_ptr = &(step_factors);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = nelr;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((nelr) - (0), pragma165_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma165_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 
} 

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma165_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma165_omp_parallel *ctx = (pragma165_omp_parallel *)____arg;
    hclib_start_finish();
    do {
    int i;     i = ___iter0;
{
		double density = (*(ctx->variables_ptr))[NVAR*i + VAR_DENSITY];

		cfd_double3 momentum;
		momentum.x = (*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+0)];
		momentum.y = (*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+1)];
		momentum.z = (*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+2)];

		double density_energy = (*(ctx->variables_ptr))[NVAR*i + VAR_DENSITY_ENERGY];
		cfd_double3 velocity;	   compute_velocity(density, momentum, velocity);
		double speed_sqd      = compute_speed_sqd(velocity);
		double pressure       = compute_pressure(density, density_energy, speed_sqd);
		double speed_of_sound = compute_speed_of_sound(density, pressure);

		// dt = double(0.5) * std::sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
		(*(ctx->step_factors_ptr))[i] = double(0.5) / (std::sqrt((*(ctx->areas_ptr))[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
	} ;     } while (0);
    ; hclib_end_finish_nonblocking();

}

#endif



/*
 *
 *
*/

typedef struct _pragma196_omp_parallel {
    double (*smoothing_coefficient_ptr);
    int (*nelr_ptr);
    int (*(*elements_surrounding_elements_ptr));
    double (*(*normals_ptr));
    double (*(*variables_ptr));
    double (*(*fluxes_ptr));
 } pragma196_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma196_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int i) {
        }
};

#else
static void pragma196_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
void compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes)
{
	double smoothing_coefficient = double(0.2f);

 { 
pragma196_omp_parallel *new_ctx = (pragma196_omp_parallel *)malloc(sizeof(pragma196_omp_parallel));
new_ctx->smoothing_coefficient_ptr = &(smoothing_coefficient);
new_ctx->nelr_ptr = &(nelr);
new_ctx->elements_surrounding_elements_ptr = &(elements_surrounding_elements);
new_ctx->normals_ptr = &(normals);
new_ctx->variables_ptr = &(variables);
new_ctx->fluxes_ptr = &(fluxes);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = nelr;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((nelr) - (0), pragma196_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma196_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 
} 

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma196_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma196_omp_parallel *ctx = (pragma196_omp_parallel *)____arg;
    hclib_start_finish();
    do {
    int i;     i = ___iter0;
{
		int j, nb;
		cfd_double3 normal; double normal_len;
		double factor;

		double density_i = (*(ctx->variables_ptr))[NVAR*i + VAR_DENSITY];
		cfd_double3 momentum_i;
		momentum_i.x = (*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+0)];
		momentum_i.y = (*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+1)];
		momentum_i.z = (*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+2)];

		double density_energy_i = (*(ctx->variables_ptr))[NVAR*i + VAR_DENSITY_ENERGY];

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
			nb = (*(ctx->elements_surrounding_elements_ptr))[i*NNB + j];
			normal.x = (*(ctx->normals_ptr))[(i*NNB + j)*NDIM + 0];
			normal.y = (*(ctx->normals_ptr))[(i*NNB + j)*NDIM + 1];
			normal.z = (*(ctx->normals_ptr))[(i*NNB + j)*NDIM + 2];
			normal_len = std::sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);

			if(nb >= 0) 	// a legitimate neighbor
			{
				density_nb =        (*(ctx->variables_ptr))[nb*NVAR + VAR_DENSITY];
				momentum_nb.x =     (*(ctx->variables_ptr))[nb*NVAR + (VAR_MOMENTUM+0)];
				momentum_nb.y =     (*(ctx->variables_ptr))[nb*NVAR + (VAR_MOMENTUM+1)];
				momentum_nb.z =     (*(ctx->variables_ptr))[nb*NVAR + (VAR_MOMENTUM+2)];
				density_energy_nb = (*(ctx->variables_ptr))[nb*NVAR + VAR_DENSITY_ENERGY];
													compute_velocity(density_nb, momentum_nb, velocity_nb);
				speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
				pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
				speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
													compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);

				// artificial viscosity
				factor = -normal_len*(*(ctx->smoothing_coefficient_ptr))*double(0.5)*(speed_i + std::sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
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

		(*(ctx->fluxes_ptr))[i*NVAR + VAR_DENSITY] = flux_i_density;
		(*(ctx->fluxes_ptr))[i*NVAR + (VAR_MOMENTUM+0)] = flux_i_momentum.x;
		(*(ctx->fluxes_ptr))[i*NVAR + (VAR_MOMENTUM+1)] = flux_i_momentum.y;
		(*(ctx->fluxes_ptr))[i*NVAR + (VAR_MOMENTUM+2)] = flux_i_momentum.z;
		(*(ctx->fluxes_ptr))[i*NVAR + VAR_DENSITY_ENERGY] = flux_i_density_energy;
	} ;     } while (0);
    ; hclib_end_finish_nonblocking();

}

#endif


typedef struct _pragma327_omp_parallel {
    int (*j_ptr);
    int (*nelr_ptr);
    double (*(*old_variables_ptr));
    double (*(*variables_ptr));
    double (*(*step_factors_ptr));
    double (*(*fluxes_ptr));
 } pragma327_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma327_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int i) {
        }
};

#else
static void pragma327_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
void time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
{
 { 
pragma327_omp_parallel *new_ctx = (pragma327_omp_parallel *)malloc(sizeof(pragma327_omp_parallel));
new_ctx->j_ptr = &(j);
new_ctx->nelr_ptr = &(nelr);
new_ctx->old_variables_ptr = &(old_variables);
new_ctx->variables_ptr = &(variables);
new_ctx->step_factors_ptr = &(step_factors);
new_ctx->fluxes_ptr = &(fluxes);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = nelr;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((nelr) - (0), pragma327_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma327_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 
} 

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma327_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma327_omp_parallel *ctx = (pragma327_omp_parallel *)____arg;
    do {
    int i;     i = ___iter0;
{
		double factor = (*(ctx->step_factors_ptr))[i]/double(RK+1-(*(ctx->j_ptr)));

		(*(ctx->variables_ptr))[NVAR*i + VAR_DENSITY] = (*(ctx->old_variables_ptr))[NVAR*i + VAR_DENSITY] + factor*(*(ctx->fluxes_ptr))[NVAR*i + VAR_DENSITY];
		(*(ctx->variables_ptr))[NVAR*i + VAR_DENSITY_ENERGY] = (*(ctx->old_variables_ptr))[NVAR*i + VAR_DENSITY_ENERGY] + factor*(*(ctx->fluxes_ptr))[NVAR*i + VAR_DENSITY_ENERGY];
		(*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+0)] = (*(ctx->old_variables_ptr))[NVAR*i + (VAR_MOMENTUM+0)] + factor*(*(ctx->fluxes_ptr))[NVAR*i + (VAR_MOMENTUM+0)];
		(*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+1)] = (*(ctx->old_variables_ptr))[NVAR*i + (VAR_MOMENTUM+1)] + factor*(*(ctx->fluxes_ptr))[NVAR*i + (VAR_MOMENTUM+1)];
		(*(ctx->variables_ptr))[NVAR*i + (VAR_MOMENTUM+2)] = (*(ctx->old_variables_ptr))[NVAR*i + (VAR_MOMENTUM+2)] + factor*(*(ctx->fluxes_ptr))[NVAR*i + (VAR_MOMENTUM+2)];
	} ;     } while (0);
}

#endif

/*
 * Main function
 */
typedef struct _main_entrypoint_ctx {
    const char (*data_file_name);
    int argc;
    char (*(*argv));
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    const char (*data_file_name); data_file_name = ctx->data_file_name;
    int argc; argc = ctx->argc;
    char (*(*argv)); argv = ctx->argv;
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
    } ;     free(____arg);
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];

main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->data_file_name = data_file_name;
new_ctx->argc = argc;
new_ctx->argv = argv;
const char *deps[] = { "system" };
hclib_launch(main_entrypoint, new_ctx, deps, 1);


	std::cout << "Done..." << std::endl;

	return 0;
} 
