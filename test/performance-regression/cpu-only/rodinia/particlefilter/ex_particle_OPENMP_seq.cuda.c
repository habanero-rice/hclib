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
static void kernel_launcher(unsigned niters, functor_type functor) {
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
    fprintf(stderr, "CAPP %llu ns\n", end - start);
    functor.transfer_from_device();
}
#ifdef __cplusplus
#ifdef __CUDACC__
#endif
#endif
/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP 
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#define PI 3.1415926535897932
/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;
/*****************************
*GET_TIME
*returns a long int representing the time
*****************************/
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
        return (float) (end_time - start_time) / (1000 * 1000);
}
/** 
* Takes in a double and returns an integer that approximates to that double
* @return if the mantissa < .5 => return value < input value; else return value > input value
*/
double roundDouble(double value){
	int newValue = (int)(value);
	if(value - newValue < .5)
	return newValue;
	else
	return newValue++;
}
/**
* Set values of the 3D array to a newValue if that value is equal to the testValue
* @param testValue The value to be replaced
* @param newValue The value to replace testValue with
* @param array3D The image vector
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
*/
void setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ){
	int x, y, z;
	for(x = 0; x < *dimX; x++){
		for(y = 0; y < *dimY; y++){
			for(z = 0; z < *dimZ; z++){
				if(array3D[x * *dimY * *dimZ+y * *dimZ + z] == testValue)
				array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
			}
		}
	}
}
/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double randu(int * seed, int index)
{
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index]/((double) M));
}
/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double randn(int * seed, int index){
	/*Box-Muller algorithm*/
	double u = randu(seed, index);
	double v = randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/**
* Sets values of 3D matrix using randomly generated numbers from a normal distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed){
	int x, y, z;
	for(x = 0; x < *dimX; x++){
		for(y = 0; y < *dimY; y++){
			for(z = 0; z < *dimZ; z++){
				array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (int)(5*randn(seed, 0));
			}
		}
	}
}
/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
void strelDisk(int * disk, int radius)
{
	int diameter = radius*2 - 1;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			double distance = sqrt(pow((double)(x-radius+1),2) + pow((double)(y-radius+1),2));
			if(distance < radius)
			disk[x*diameter + y] = 1;
		}
	}
}
/**
* Dilates the provided video
* @param matrix The video to be dilated
* @param posX The x location of the pixel to be dilated
* @param posY The y location of the pixel to be dilated
* @param poxZ The z location of the pixel to be dilated
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param error The error radius
*/
void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
{
	int startX = posX - error;
	while(startX < 0)
	startX++;
	int startY = posY - error;
	while(startY < 0)
	startY++;
	int endX = posX + error;
	while(endX > dimX)
	endX--;
	int endY = posY + error;
	while(endY > dimY)
	endY--;
	int x,y;
	for(x = startX; x < endX; x++){
		for(y = startY; y < endY; y++){
			double distance = sqrt( pow((double)(x-posX),2) + pow((double)(y-posY),2) );
			if(distance < error)
			matrix[x*dimY*dimZ + y*dimZ + posZ] = 1;
		}
	}
}

/**
* Dilates the target matrix using the radius as a guide
* @param matrix The reference matrix
* @param dimX The x dimension of the video
* @param dimY The y dimension of the video
* @param dimZ The z dimension of the video
* @param error The error radius to be dilated
* @param newMatrix The target matrix
*/
void imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix)
{
	int x, y, z;
	for(z = 0; z < dimZ; z++){
		for(x = 0; x < dimX; x++){
			for(y = 0; y < dimY; y++){
				if(matrix[x*dimY*dimZ + y*dimZ + z] == 1){
					dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
				}
			}
		}
	}
}
/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
void getneighbors(int * se, int numOnes, double * neighbors, int radius){
	int x, y;
	int neighY = 0;
	int center = radius - 1;
	int diameter = radius*2 -1;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(se[x*diameter + y]){
				neighbors[neighY*2] = (int)(y - center);
				neighbors[neighY*2 + 1] = (int)(x - center);
				neighY++;
			}
		}
	}
}
/**
* The synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
* @param I The video itself
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames of the video
* @param seed The seed array used for number generation
*/
void videoSequence(int * I, int IszX, int IszY, int Nfr, int * seed){
	int k;
	int max_size = IszX*IszY*Nfr;
	/*get object centers*/
	int x0 = (int)roundDouble(IszY/2.0);
	int y0 = (int)roundDouble(IszX/2.0);
	I[x0 *IszY *Nfr + y0 * Nfr  + 0] = 1;
	
	/*move point*/
	int xk, yk, pos;
	for(k = 1; k < Nfr; k++){
		xk = abs(x0 + (k-1));
		yk = abs(y0 - 2*(k-1));
		pos = yk * IszY * Nfr + xk *Nfr + k;
		if(pos >= max_size)
		pos = 0;
		I[pos] = 1;
	}
	
	/*dilate matrix*/
	int * newMatrix = (int *)malloc(sizeof(int)*IszX*IszY*Nfr);
	imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
	int x, y;
	for(x = 0; x < IszX; x++){
		for(y = 0; y < IszY; y++){
			for(k = 0; k < Nfr; k++){
				I[x*IszY*Nfr + y*Nfr + k] = newMatrix[x*IszY*Nfr + y*Nfr + k];
			}
		}
	}
	free(newMatrix);
	
	/*define background, add noise*/
	setIf(0, 100, I, &IszX, &IszY, &Nfr);
	setIf(1, 228, I, &IszX, &IszY, &Nfr);
	/*add noise*/
	addNoise(I, &IszX, &IszY, &Nfr, seed);
}
/**
* Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
double calcLikelihoodSum(int * I, int * ind, int numOnes){
	double likelihoodSum = 0.0;
	int y;
	for(y = 0; y < numOnes; y++)
	likelihoodSum += (pow((I[ind[y]] - 100),2) - pow((I[ind[y]]-228),2))/50.0;
	return likelihoodSum;
}
/**
* Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
* @note This function uses sequential search
* @param CDF The CDF
* @param lengthCDF The length of CDF
* @param value The value to be found
* @return The index of value in the CDF; if value is never found, returns the last index
*/
int findIndex(double * CDF, int lengthCDF, double value){
	int index = -1;
	int x;
	for(x = 0; x < lengthCDF; x++){
		if(CDF[x] >= value){
			index = x;
			break;
		}
	}
	if(index == -1){
		return lengthCDF-1;
	}
	return index;
}
/**
* Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
* @note This function uses binary search before switching to sequential search
* @param CDF The CDF
* @param beginIndex The index to start searching from
* @param endIndex The index to stop searching
* @param value The value to find
* @return The index of value in the CDF; if value is never found, returns the last index
* @warning Use at your own risk; not fully tested
*/
int findIndexBin(double * CDF, int beginIndex, int endIndex, double value){
	if(endIndex < beginIndex)
	return -1;
	int middleIndex = beginIndex + ((endIndex - beginIndex)/2);
	/*check the value*/
	if(CDF[middleIndex] >= value)
	{
		/*check that it's good*/
		if(middleIndex == 0)
		return middleIndex;
		else if(CDF[middleIndex-1] < value)
		return middleIndex;
		else if(CDF[middleIndex-1] == value)
		{
			while(middleIndex > 0 && CDF[middleIndex-1] == value)
			middleIndex--;
			return middleIndex;
		}
	}
	if(CDF[middleIndex] > value)
	return findIndexBin(CDF, beginIndex, middleIndex+1, value);
	return findIndexBin(CDF, middleIndex-1, endIndex, value);
}
/**
* The implementation of the particle filter using OpenMP for many frames
* @see http://openmp.org/wp/
* @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
* @param I The video to be run
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames
* @param seed The seed array used for random number generation
* @param Nparticles The number of particles to be used
*/
class pragma383_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double* volatile weights;
    double* volatile h_weights;
    int x;
    volatile int Nparticles;

    public:
        pragma383_omp_parallel_hclib_async(double* set_weights,
                int set_x,
                int set_Nparticles) {
            h_weights = set_weights;
            x = set_x;
            Nparticles = set_Nparticles;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        weights = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 1, h_weights);
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
            if (weights == NULL && (char *)h_weights >= (char *)host_allocations[i] && ((char *)h_weights - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_weights - (char *)host_allocations[i]);
                memcpy((void *)(&weights), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(weights || h_weights == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
		weights[x] = 1/((double)(Nparticles));
	}
            }
        }
};

class pragma398_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double* volatile arrayX;
    double* volatile h_arrayX;
    int x;
    volatile double xe;
    double* volatile arrayY;
    double* volatile h_arrayY;
    volatile double ye;

    public:
        pragma398_omp_parallel_hclib_async(double* set_arrayX,
                int set_x,
                double set_xe,
                double* set_arrayY,
                double set_ye) {
            h_arrayX = set_arrayX;
            x = set_x;
            xe = set_xe;
            h_arrayY = set_arrayY;
            ye = set_ye;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        arrayX = NULL;
        arrayY = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 2, h_arrayX, h_arrayY);
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
            if (arrayX == NULL && (char *)h_arrayX >= (char *)host_allocations[i] && ((char *)h_arrayX - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayX - (char *)host_allocations[i]);
                memcpy((void *)(&arrayX), (void *)(&tmp), sizeof(void *));
            }
            if (arrayY == NULL && (char *)h_arrayY >= (char *)host_allocations[i] && ((char *)h_arrayY - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayY - (char *)host_allocations[i]);
                memcpy((void *)(&arrayY), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(arrayX || h_arrayX == NULL);
        assert(arrayY || h_arrayY == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
            }
        }
};

class pragma412_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
        __device__ double randn(int * seed, int index) {
            {
	/*Box-Muller algorithm*/
	double u = randu(seed, index);
	double v = randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
        }
        __device__ double randu(int * seed, int index) {
            {
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index]/((double) M));
}
        }
    double* volatile arrayX;
    double* volatile h_arrayX;
    int x;
    volatile int A;
    volatile int C;
    volatile long M;
    int* volatile seed;
    int* volatile h_seed;
    double* volatile arrayY;
    double* volatile h_arrayY;

    public:
        pragma412_omp_parallel_hclib_async(double* set_arrayX,
                int set_x,
                int set_A,
                int set_C,
                long set_M,
                int* set_seed,
                double* set_arrayY) {
            h_arrayX = set_arrayX;
            x = set_x;
            A = set_A;
            C = set_C;
            M = set_M;
            h_seed = set_seed;
            h_arrayY = set_arrayY;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        arrayX = NULL;
        seed = NULL;
        arrayY = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 3, h_arrayX, h_seed, h_arrayY);
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
            if (arrayX == NULL && (char *)h_arrayX >= (char *)host_allocations[i] && ((char *)h_arrayX - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayX - (char *)host_allocations[i]);
                memcpy((void *)(&arrayX), (void *)(&tmp), sizeof(void *));
            }
            if (seed == NULL && (char *)h_seed >= (char *)host_allocations[i] && ((char *)h_seed - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_seed - (char *)host_allocations[i]);
                memcpy((void *)(&seed), (void *)(&tmp), sizeof(void *));
            }
            if (arrayY == NULL && (char *)h_arrayY >= (char *)host_allocations[i] && ((char *)h_arrayY - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayY - (char *)host_allocations[i]);
                memcpy((void *)(&arrayY), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(arrayX || h_arrayX == NULL);
        assert(seed || h_seed == NULL);
        assert(arrayY || h_arrayY == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
			arrayX[x] += 1 + 5*randn(seed, x);
			arrayY[x] += -2 + 2*randn(seed, x);
		}
            }
        }
};

class pragma420_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
        __device__ double roundDouble(double value) {
            {
	int newValue = (int)(value);
	if(value - newValue < .5)
	return newValue;
	else
	return newValue++;
}
        }
    int y;
    volatile int countOnes;
    int indX;
    double* volatile arrayX;
    double* volatile h_arrayX;
    int x;
    double* volatile objxy;
    double* volatile h_objxy;
    int indY;
    double* volatile arrayY;
    double* volatile h_arrayY;
    int* volatile ind;
    int* volatile h_ind;
    volatile int IszY;
    volatile int Nfr;
    volatile int k;
    volatile int max_size;
    double* volatile likelihood;
    double* volatile h_likelihood;
    int* volatile I;
    int* volatile h_I;

    public:
        pragma420_omp_parallel_hclib_async(int set_y,
                int set_countOnes,
                int set_indX,
                double* set_arrayX,
                int set_x,
                double* set_objxy,
                int set_indY,
                double* set_arrayY,
                int* set_ind,
                int set_IszY,
                int set_Nfr,
                int set_k,
                int set_max_size,
                double* set_likelihood,
                int* set_I) {
            y = set_y;
            countOnes = set_countOnes;
            indX = set_indX;
            h_arrayX = set_arrayX;
            x = set_x;
            h_objxy = set_objxy;
            indY = set_indY;
            h_arrayY = set_arrayY;
            h_ind = set_ind;
            IszY = set_IszY;
            Nfr = set_Nfr;
            k = set_k;
            max_size = set_max_size;
            h_likelihood = set_likelihood;
            h_I = set_I;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        arrayX = NULL;
        objxy = NULL;
        arrayY = NULL;
        ind = NULL;
        likelihood = NULL;
        I = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 6, h_arrayX, h_objxy, h_arrayY, h_ind, h_likelihood, h_I);
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
            if (arrayX == NULL && (char *)h_arrayX >= (char *)host_allocations[i] && ((char *)h_arrayX - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayX - (char *)host_allocations[i]);
                memcpy((void *)(&arrayX), (void *)(&tmp), sizeof(void *));
            }
            if (objxy == NULL && (char *)h_objxy >= (char *)host_allocations[i] && ((char *)h_objxy - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_objxy - (char *)host_allocations[i]);
                memcpy((void *)(&objxy), (void *)(&tmp), sizeof(void *));
            }
            if (arrayY == NULL && (char *)h_arrayY >= (char *)host_allocations[i] && ((char *)h_arrayY - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayY - (char *)host_allocations[i]);
                memcpy((void *)(&arrayY), (void *)(&tmp), sizeof(void *));
            }
            if (ind == NULL && (char *)h_ind >= (char *)host_allocations[i] && ((char *)h_ind - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_ind - (char *)host_allocations[i]);
                memcpy((void *)(&ind), (void *)(&tmp), sizeof(void *));
            }
            if (likelihood == NULL && (char *)h_likelihood >= (char *)host_allocations[i] && ((char *)h_likelihood - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_likelihood - (char *)host_allocations[i]);
                memcpy((void *)(&likelihood), (void *)(&tmp), sizeof(void *));
            }
            if (I == NULL && (char *)h_I >= (char *)host_allocations[i] && ((char *)h_I - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_I - (char *)host_allocations[i]);
                memcpy((void *)(&I), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(arrayX || h_arrayX == NULL);
        assert(objxy || h_objxy == NULL);
        assert(arrayY || h_arrayY == NULL);
        assert(ind || h_ind == NULL);
        assert(likelihood || h_likelihood == NULL);
        assert(I || h_I == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
			//compute the likelihood: remember our assumption is that you know
			// foreground and the background image intensity distribution.
			// Notice that we consider here a likelihood ratio, instead of
			// p(z|x). It is possible in this case. why? a hometask for you.		
			//calc ind
			for(y = 0; y < countOnes; y++){
				indX = roundDouble(arrayX[x]) + objxy[y*2 + 1];
				indY = roundDouble(arrayY[x]) + objxy[y*2];
				ind[x*countOnes + y] = fabs((double)(indX*IszY*Nfr + indY*Nfr + k));
				if(ind[x*countOnes + y] >= max_size)
					ind[x*countOnes + y] = 0;
			}
			likelihood[x] = 0;
			for(y = 0; y < countOnes; y++)
				likelihood[x] += (pow((I[ind[x*countOnes + y]] - 100),2) - pow((I[ind[x*countOnes + y]]-228),2))/50.0;
			likelihood[x] = likelihood[x]/((double) countOnes);
		}
            }
        }
};

class pragma443_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double* volatile weights;
    double* volatile h_weights;
    int x;
    double* volatile likelihood;
    double* volatile h_likelihood;

    public:
        pragma443_omp_parallel_hclib_async(double* set_weights,
                int set_x,
                double* set_likelihood) {
            h_weights = set_weights;
            x = set_x;
            h_likelihood = set_likelihood;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        weights = NULL;
        likelihood = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 2, h_weights, h_likelihood);
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
            if (weights == NULL && (char *)h_weights >= (char *)host_allocations[i] && ((char *)h_weights - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_weights - (char *)host_allocations[i]);
                memcpy((void *)(&weights), (void *)(&tmp), sizeof(void *));
            }
            if (likelihood == NULL && (char *)h_likelihood >= (char *)host_allocations[i] && ((char *)h_likelihood - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_likelihood - (char *)host_allocations[i]);
                memcpy((void *)(&likelihood), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(weights || h_weights == NULL);
        assert(likelihood || h_likelihood == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
			weights[x] = weights[x] * exp(likelihood[x]);
		}
            }
        }
};

class pragma450_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double sumWeights;
    double* volatile weights;
    double* volatile h_weights;
    int x;

    public:
        pragma450_omp_parallel_hclib_async(double set_sumWeights,
                double* set_weights,
                int set_x) {
            sumWeights = set_sumWeights;
            h_weights = set_weights;
            x = set_x;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        weights = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 1, h_weights);
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
            if (weights == NULL && (char *)h_weights >= (char *)host_allocations[i] && ((char *)h_weights - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_weights - (char *)host_allocations[i]);
                memcpy((void *)(&weights), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(weights || h_weights == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
			sumWeights += weights[x];
		}
            }
        }
};

class pragma456_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double* volatile weights;
    double* volatile h_weights;
    int x;
    volatile double sumWeights;

    public:
        pragma456_omp_parallel_hclib_async(double* set_weights,
                int set_x,
                double set_sumWeights) {
            h_weights = set_weights;
            x = set_x;
            sumWeights = set_sumWeights;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        weights = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 1, h_weights);
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
            if (weights == NULL && (char *)h_weights >= (char *)host_allocations[i] && ((char *)h_weights - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_weights - (char *)host_allocations[i]);
                memcpy((void *)(&weights), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(weights || h_weights == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
			weights[x] = weights[x]/sumWeights;
		}
            }
        }
};

class pragma465_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double xe;
    double* volatile arrayX;
    double* volatile h_arrayX;
    int x;
    double* volatile weights;
    double* volatile h_weights;
    double ye;
    double* volatile arrayY;
    double* volatile h_arrayY;

    public:
        pragma465_omp_parallel_hclib_async(double set_xe,
                double* set_arrayX,
                int set_x,
                double* set_weights,
                double set_ye,
                double* set_arrayY) {
            xe = set_xe;
            h_arrayX = set_arrayX;
            x = set_x;
            h_weights = set_weights;
            ye = set_ye;
            h_arrayY = set_arrayY;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        arrayX = NULL;
        weights = NULL;
        arrayY = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 3, h_arrayX, h_weights, h_arrayY);
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
            if (arrayX == NULL && (char *)h_arrayX >= (char *)host_allocations[i] && ((char *)h_arrayX - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayX - (char *)host_allocations[i]);
                memcpy((void *)(&arrayX), (void *)(&tmp), sizeof(void *));
            }
            if (weights == NULL && (char *)h_weights >= (char *)host_allocations[i] && ((char *)h_weights - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_weights - (char *)host_allocations[i]);
                memcpy((void *)(&weights), (void *)(&tmp), sizeof(void *));
            }
            if (arrayY == NULL && (char *)h_arrayY >= (char *)host_allocations[i] && ((char *)h_arrayY - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayY - (char *)host_allocations[i]);
                memcpy((void *)(&arrayY), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(arrayX || h_arrayX == NULL);
        assert(weights || h_weights == NULL);
        assert(arrayY || h_arrayY == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
			xe += arrayX[x] * weights[x];
			ye += arrayY[x] * weights[x];
		}
            }
        }
};

class pragma490_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    double* volatile u;
    double* volatile h_u;
    int x;
    volatile double u1;
    volatile int Nparticles;

    public:
        pragma490_omp_parallel_hclib_async(double* set_u,
                int set_x,
                double set_u1,
                int set_Nparticles) {
            h_u = set_u;
            x = set_x;
            u1 = set_u1;
            Nparticles = set_Nparticles;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        u = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 1, h_u);
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
            if (u == NULL && (char *)h_u >= (char *)host_allocations[i] && ((char *)h_u - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_u - (char *)host_allocations[i]);
                memcpy((void *)(&u), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(u || h_u == NULL);

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

        __device__ void operator()(int x) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
			u[x] = u1 + x/((double)(Nparticles));
		}
            }
        }
};

class pragma498_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
        __device__ int findIndex(double * CDF, int lengthCDF, double value) {
            {
	int index = -1;
	int x;
	for(x = 0; x < lengthCDF; x++){
		if(CDF[x] >= value){
			index = x;
			break;
		}
	}
	if(index == -1){
		return lengthCDF-1;
	}
	return index;
}
        }
    int i;
    double* volatile CDF;
    double* volatile h_CDF;
    volatile int Nparticles;
    double* volatile u;
    double* volatile h_u;
    int j;
    double* volatile xj;
    double* volatile h_xj;
    double* volatile arrayX;
    double* volatile h_arrayX;
    double* volatile yj;
    double* volatile h_yj;
    double* volatile arrayY;
    double* volatile h_arrayY;

    public:
        pragma498_omp_parallel_hclib_async(int set_i,
                double* set_CDF,
                int set_Nparticles,
                double* set_u,
                int set_j,
                double* set_xj,
                double* set_arrayX,
                double* set_yj,
                double* set_arrayY) {
            i = set_i;
            h_CDF = set_CDF;
            Nparticles = set_Nparticles;
            h_u = set_u;
            j = set_j;
            h_xj = set_xj;
            h_arrayX = set_arrayX;
            h_yj = set_yj;
            h_arrayY = set_arrayY;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        CDF = NULL;
        u = NULL;
        xj = NULL;
        arrayX = NULL;
        yj = NULL;
        arrayY = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 6, h_CDF, h_u, h_xj, h_arrayX, h_yj, h_arrayY);
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
            if (CDF == NULL && (char *)h_CDF >= (char *)host_allocations[i] && ((char *)h_CDF - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_CDF - (char *)host_allocations[i]);
                memcpy((void *)(&CDF), (void *)(&tmp), sizeof(void *));
            }
            if (u == NULL && (char *)h_u >= (char *)host_allocations[i] && ((char *)h_u - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_u - (char *)host_allocations[i]);
                memcpy((void *)(&u), (void *)(&tmp), sizeof(void *));
            }
            if (xj == NULL && (char *)h_xj >= (char *)host_allocations[i] && ((char *)h_xj - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_xj - (char *)host_allocations[i]);
                memcpy((void *)(&xj), (void *)(&tmp), sizeof(void *));
            }
            if (arrayX == NULL && (char *)h_arrayX >= (char *)host_allocations[i] && ((char *)h_arrayX - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayX - (char *)host_allocations[i]);
                memcpy((void *)(&arrayX), (void *)(&tmp), sizeof(void *));
            }
            if (yj == NULL && (char *)h_yj >= (char *)host_allocations[i] && ((char *)h_yj - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_yj - (char *)host_allocations[i]);
                memcpy((void *)(&yj), (void *)(&tmp), sizeof(void *));
            }
            if (arrayY == NULL && (char *)h_arrayY >= (char *)host_allocations[i] && ((char *)h_arrayY - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_arrayY - (char *)host_allocations[i]);
                memcpy((void *)(&arrayY), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(CDF || h_CDF == NULL);
        assert(u || h_u == NULL);
        assert(xj || h_xj == NULL);
        assert(arrayX || h_arrayX == NULL);
        assert(yj || h_yj == NULL);
        assert(arrayY || h_arrayY == NULL);

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

        __device__ void operator()(int j) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
			i = findIndex(CDF, Nparticles, u[j]);
			if(i == -1)
				i = Nparticles-1;
			xj[j] = arrayX[i];
			yj[j] = arrayY[i];
			
		}
            }
        }
};

void particleFilter(int * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles){
	
	int max_size = IszX*IszY*Nfr;
	long long start = get_time();
	//original particle centroid
	double xe = roundDouble(IszY/2.0);
	double ye = roundDouble(IszX/2.0);
	
	//expected object locations, compared to center
	int radius = 5;
	int diameter = radius*2 - 1;
	int * disk = (int *)malloc(diameter*diameter*sizeof(int));
	strelDisk(disk, radius);
	int countOnes = 0;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(disk[x*diameter + y] == 1)
				countOnes++;
		}
	}
	double * objxy = (double *)malloc(countOnes*2*sizeof(double));
	getneighbors(disk, countOnes, objxy, radius);
	
	long long get_neighbors = get_time();
	printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start, get_neighbors));
	//initial weights are all equal (1/Nparticles)
	double * weights = (double *)malloc(sizeof(double)*Nparticles);
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma383_omp_parallel_hclib_async(weights, x, Nparticles));
 } 
	long long get_weights = get_time();
	printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors, get_weights));
	//initial likelihood to 0.0
	double * likelihood = (double *)malloc(sizeof(double)*Nparticles);
	double * arrayX = (double *)malloc(sizeof(double)*Nparticles);
	double * arrayY = (double *)malloc(sizeof(double)*Nparticles);
	double * xj = (double *)malloc(sizeof(double)*Nparticles);
	double * yj = (double *)malloc(sizeof(double)*Nparticles);
	double * CDF = (double *)malloc(sizeof(double)*Nparticles);
	double * u = (double *)malloc(sizeof(double)*Nparticles);
	int * ind = (int*)malloc(sizeof(int)*countOnes*Nparticles);
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma398_omp_parallel_hclib_async(arrayX, x, xe, arrayY, ye));
 } 
	int k;
	
	printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, get_time()));
	int indX, indY;
	for(k = 1; k < Nfr; k++){
		long long set_arrays = get_time();
		//apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma412_omp_parallel_hclib_async(arrayX, x, A, C, M, seed, arrayY));
 } 
		long long error = get_time();
		printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
		//particle filter likelihood
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma420_omp_parallel_hclib_async(y, countOnes, indX, arrayX, x, objxy, indY, arrayY, ind, IszY, Nfr, k, max_size, likelihood, I));
 } 
		long long likelihood_time = get_time();
		printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error, likelihood_time));
		// update & normalize weights
		// using equation (63) of Arulampalam Tutorial
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma443_omp_parallel_hclib_async(weights, x, likelihood));
 } 
		long long exponential = get_time();
		printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time, exponential));
		double sumWeights = 0;
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma450_omp_parallel_hclib_async(sumWeights, weights, x));
 } 
		long long sum_time = get_time();
		printf("TIME TO SUM WEIGHTS TOOK: %f\n", elapsed_time(exponential, sum_time));
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma456_omp_parallel_hclib_async(weights, x, sumWeights));
 } 
		long long normalize = get_time();
		printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n", elapsed_time(sum_time, normalize));
		xe = 0;
		ye = 0;
		// estimate the object location by expected values
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma465_omp_parallel_hclib_async(xe, arrayX, x, weights, ye, arrayY));
 } 
		long long move_time = get_time();
		printf("TIME TO MOVE OBJECT TOOK: %f\n", elapsed_time(normalize, move_time));
		printf("XE: %lf\n", xe);
		printf("YE: %lf\n", ye);
		double distance = sqrt( pow((double)(xe-(int)roundDouble(IszY/2.0)),2) + pow((double)(ye-(int)roundDouble(IszX/2.0)),2) );
		printf("%lf\n", distance);
		//display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling
		
		
		CDF[0] = weights[0];
		for(x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}
		long long cum_sum = get_time();
		printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time, cum_sum));
		double u1 = (1/((double)(Nparticles)))*randu(seed, 0);
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma490_omp_parallel_hclib_async(u, x, u1, Nparticles));
 } 
		long long u_time = get_time();
		printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
		int j, i;
		
 { const int niters = (Nparticles) - (0);
kernel_launcher(niters, pragma498_omp_parallel_hclib_async(i, CDF, Nparticles, u, j, xj, arrayX, yj, arrayY));
 } 
		long long xyj_time = get_time();
		printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n", elapsed_time(u_time, xyj_time));
		
		//#pragma omp parallel for shared(weights, Nparticles) private(x)
		for(x = 0; x < Nparticles; x++){
			//reassign arrayX and arrayY
			arrayX[x] = xj[x];
			arrayY[x] = yj[x];
			weights[x] = 1/((double)(Nparticles));
		}
		long long reset = get_time();
		printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
	}
	free(disk);
	free(objxy);
	free(weights);
	free(likelihood);
	free(xj);
	free(yj);
	free(arrayX);
	free(arrayY);
	free(CDF);
	free(u);
	free(ind);
} 
int main(int argc, char * argv[]){
	
	char* usage = "openmp.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
	//check number of arguments
	if(argc != 9)
	{
		printf("%s\n", usage);
		return 0;
	}
	//check args deliminators
	if( strcmp( argv[1], "-x" ) ||  strcmp( argv[3], "-y" ) || strcmp( argv[5], "-z" ) || strcmp( argv[7], "-np" ) ) {
		printf( "%s\n",usage );
		return 0;
	}
	
	int IszX, IszY, Nfr, Nparticles;
	
	//converting a string to a integer
	if( sscanf( argv[2], "%d", &IszX ) == EOF ) {
	   printf("ERROR: dimX input is incorrect");
	   return 0;
	}
	
	if( IszX <= 0 ) {
		printf("dimX must be > 0\n");
		return 0;
	}
	
	//converting a string to a integer
	if( sscanf( argv[4], "%d", &IszY ) == EOF ) {
	   printf("ERROR: dimY input is incorrect");
	   return 0;
	}
	
	if( IszY <= 0 ) {
		printf("dimY must be > 0\n");
		return 0;
	}
	
	//converting a string to a integer
	if( sscanf( argv[6], "%d", &Nfr ) == EOF ) {
	   printf("ERROR: Number of frames input is incorrect");
	   return 0;
	}
	
	if( Nfr <= 0 ) {
		printf("number of frames must be > 0\n");
		return 0;
	}
	
	//converting a string to a integer
	if( sscanf( argv[8], "%d", &Nparticles ) == EOF ) {
	   printf("ERROR: Number of particles input is incorrect");
	   return 0;
	}
	
	if( Nparticles <= 0 ) {
		printf("Number of particles must be > 0\n");
		return 0;
	}
	//establish seed
	int * seed = (int *)malloc(sizeof(int)*Nparticles);
	int i;
	for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
	//malloc matrix
	int * I = (int *)malloc(sizeof(int)*IszX*IszY*Nfr);
	long long start = get_time();
	//call video sequence
	videoSequence(I, IszX, IszY, Nfr, seed);
	long long endVideoSequence = get_time();
	printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
	//call particle filter
particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);

	long long endParticleFilter = get_time();
	printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
	printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));
	
	free(seed);
	free(I);
	return 0;
}
