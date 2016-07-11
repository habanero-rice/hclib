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
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
//#define NUM_THREAD 4
#define OPEN


FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <num_threads> <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	BFSGraph( argc, argv);
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
class pragma136_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    bool* h_graph_mask;
    bool* h_h_graph_mask;
    struct Node* h_graph_nodes;
    struct Node* h_h_graph_nodes;
    int* h_graph_edges;
    int* h_h_graph_edges;
    bool* h_graph_visited;
    bool* h_h_graph_visited;
    int* h_cost;
    int* h_h_cost;
    bool* h_updating_graph_mask;
    bool* h_h_updating_graph_mask;

    public:
        pragma136_omp_parallel_hclib_async(bool* set_h_graph_mask,
                struct Node* set_h_graph_nodes,
                int* set_h_graph_edges,
                bool* set_h_graph_visited,
                int* set_h_cost,
                bool* set_h_updating_graph_mask) {
            h_h_graph_mask = set_h_graph_mask;
            h_h_graph_nodes = set_h_graph_nodes;
            h_h_graph_edges = set_h_graph_edges;
            h_h_graph_visited = set_h_graph_visited;
            h_h_cost = set_h_cost;
            h_h_updating_graph_mask = set_h_updating_graph_mask;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        h_graph_mask = NULL;
        h_graph_nodes = NULL;
        h_graph_edges = NULL;
        h_graph_visited = NULL;
        h_cost = NULL;
        h_updating_graph_mask = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 6, h_h_graph_mask, h_h_graph_nodes, h_h_graph_edges, h_h_graph_visited, h_h_cost, h_h_updating_graph_mask);
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
            if (h_graph_mask == NULL && (char *)h_h_graph_mask >= (char *)host_allocations[i] && ((char *)h_h_graph_mask - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_graph_mask - (char *)host_allocations[i]);
                memcpy((void *)(&h_graph_mask), (void *)(&tmp), sizeof(void *));
            }
            if (h_graph_nodes == NULL && (char *)h_h_graph_nodes >= (char *)host_allocations[i] && ((char *)h_h_graph_nodes - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_graph_nodes - (char *)host_allocations[i]);
                memcpy((void *)(&h_graph_nodes), (void *)(&tmp), sizeof(void *));
            }
            if (h_graph_edges == NULL && (char *)h_h_graph_edges >= (char *)host_allocations[i] && ((char *)h_h_graph_edges - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_graph_edges - (char *)host_allocations[i]);
                memcpy((void *)(&h_graph_edges), (void *)(&tmp), sizeof(void *));
            }
            if (h_graph_visited == NULL && (char *)h_h_graph_visited >= (char *)host_allocations[i] && ((char *)h_h_graph_visited - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_graph_visited - (char *)host_allocations[i]);
                memcpy((void *)(&h_graph_visited), (void *)(&tmp), sizeof(void *));
            }
            if (h_cost == NULL && (char *)h_h_cost >= (char *)host_allocations[i] && ((char *)h_h_cost - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_cost - (char *)host_allocations[i]);
                memcpy((void *)(&h_cost), (void *)(&tmp), sizeof(void *));
            }
            if (h_updating_graph_mask == NULL && (char *)h_h_updating_graph_mask >= (char *)host_allocations[i] && ((char *)h_h_updating_graph_mask - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_updating_graph_mask - (char *)host_allocations[i]);
                memcpy((void *)(&h_updating_graph_mask), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(h_graph_mask);
        assert(h_graph_nodes);
        assert(h_graph_edges);
        assert(h_graph_visited);
        assert(h_cost);
        assert(h_updating_graph_mask);

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

        __device__ void operator()(int tid) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
                if (h_graph_mask[tid] == true){ 
                    h_graph_mask[tid]=false;
                    for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++)
                    {
                        int id = h_graph_edges[i];
                        if(!h_graph_visited[id])
                        {
                            h_cost[id]=h_cost[tid]+1;
                            h_updating_graph_mask[id]=true;
                        }
                    }
                }
            }
            }
        }
};

class pragma153_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    bool* h_updating_graph_mask;
    bool* h_h_updating_graph_mask;
    bool* h_graph_mask;
    bool* h_h_graph_mask;
    bool* h_graph_visited;
    bool* h_h_graph_visited;
    volatile bool stop;

    public:
        pragma153_omp_parallel_hclib_async(bool* set_h_updating_graph_mask,
                bool* set_h_graph_mask,
                bool* set_h_graph_visited,
                bool set_stop) {
            h_h_updating_graph_mask = set_h_updating_graph_mask;
            h_h_graph_mask = set_h_graph_mask;
            h_h_graph_visited = set_h_graph_visited;
            stop = set_stop;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        h_updating_graph_mask = NULL;
        h_graph_mask = NULL;
        h_graph_visited = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 3, h_h_updating_graph_mask, h_h_graph_mask, h_h_graph_visited);
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
            if (h_updating_graph_mask == NULL && (char *)h_h_updating_graph_mask >= (char *)host_allocations[i] && ((char *)h_h_updating_graph_mask - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_updating_graph_mask - (char *)host_allocations[i]);
                memcpy((void *)(&h_updating_graph_mask), (void *)(&tmp), sizeof(void *));
            }
            if (h_graph_mask == NULL && (char *)h_h_graph_mask >= (char *)host_allocations[i] && ((char *)h_h_graph_mask - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_graph_mask - (char *)host_allocations[i]);
                memcpy((void *)(&h_graph_mask), (void *)(&tmp), sizeof(void *));
            }
            if (h_graph_visited == NULL && (char *)h_h_graph_visited >= (char *)host_allocations[i] && ((char *)h_h_graph_visited - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_h_graph_visited - (char *)host_allocations[i]);
                memcpy((void *)(&h_graph_visited), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(h_updating_graph_mask);
        assert(h_graph_mask);
        assert(h_graph_visited);

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

        __device__ void operator()(int tid) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
                if (h_updating_graph_mask[tid] == true){
                    h_graph_mask[tid]=true;
                    h_graph_visited[tid]=true;
                    stop=true;
                    h_updating_graph_mask[tid]=false;
                }
            }
            }
        }
};

void BFSGraph( int argc, char** argv) 
{
        int no_of_nodes = 0;
        int edge_list_size = 0;
        char *input_f;
	int	 num_omp_threads;
	
	if(argc!=3){
	Usage(argc, argv);
	exit(0);
	}
    
	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
	
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);
   
	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	// source=0; //tesing code line

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    


	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	printf("Start traversing the tree\n");

{
	int k=0;
	bool stop;
	do
        {
            //if no thread changes this value then the loop stops
            stop=false;

            //omp_set_num_threads(num_omp_threads);
 { const int niters = (no_of_nodes) - (0);
kernel_launcher(niters, pragma136_omp_parallel_hclib_async(h_graph_mask, h_graph_nodes, h_graph_edges, h_graph_visited, h_cost, h_updating_graph_mask));
 } 

 { const int niters = (no_of_nodes) - (0);
kernel_launcher(niters, pragma153_omp_parallel_hclib_async(h_updating_graph_mask, h_graph_mask, h_graph_visited, stop));
 } 
            k++;
        }
	while(stop);
    }

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);

} 

