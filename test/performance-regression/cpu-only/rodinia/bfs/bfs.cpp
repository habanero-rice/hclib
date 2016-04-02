#include "hclib.h"
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
typedef struct _pragma127 {
    int argc;
    char **argv;
    int no_of_nodes;
    int edge_list_size;
    char *input_f;
    int num_omp_threads;
    int source;
    Node *h_graph_nodes;
    _Bool *h_graph_mask;
    _Bool *h_updating_graph_mask;
    _Bool *h_graph_visited;
    int start;
    int edgeno;
    int id;
    int cost;
    int *h_graph_edges;
    int *h_cost;
    int k;
    _Bool stop;
 } pragma127;

typedef struct _pragma144 {
    int argc;
    char **argv;
    int no_of_nodes;
    int edge_list_size;
    char *input_f;
    int num_omp_threads;
    int source;
    Node *h_graph_nodes;
    _Bool *h_graph_mask;
    _Bool *h_updating_graph_mask;
    _Bool *h_graph_visited;
    int start;
    int edgeno;
    int id;
    int cost;
    int *h_graph_edges;
    int *h_cost;
    int k;
    _Bool stop;
 } pragma144;

static void pragma127_hclib_async(void *____arg, const int ___iter);
static void pragma144_hclib_async(void *____arg, const int ___iter);
typedef struct _main_entrypoint_ctx {
    int argc;
    char **argv;
    int no_of_nodes;
    int edge_list_size;
    char *input_f;
    int num_omp_threads;
    int source;
    Node *h_graph_nodes;
    _Bool *h_graph_mask;
    _Bool *h_updating_graph_mask;
    _Bool *h_graph_visited;
    int start;
    int edgeno;
    int id;
    int cost;
    int *h_graph_edges;
    int *h_cost;
 } main_entrypoint_ctx;

static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int no_of_nodes; no_of_nodes = ctx->no_of_nodes;
    int edge_list_size; edge_list_size = ctx->edge_list_size;
    char *input_f; input_f = ctx->input_f;
    int num_omp_threads; num_omp_threads = ctx->num_omp_threads;
    int source; source = ctx->source;
    Node *h_graph_nodes; h_graph_nodes = ctx->h_graph_nodes;
    _Bool *h_graph_mask; h_graph_mask = ctx->h_graph_mask;
    _Bool *h_updating_graph_mask; h_updating_graph_mask = ctx->h_updating_graph_mask;
    _Bool *h_graph_visited; h_graph_visited = ctx->h_graph_visited;
    int start; start = ctx->start;
    int edgeno; edgeno = ctx->edgeno;
    int id; id = ctx->id;
    int cost; cost = ctx->cost;
    int *h_graph_edges; h_graph_edges = ctx->h_graph_edges;
    int *h_cost; h_cost = ctx->h_cost;
{
	int k=0;
	bool stop;
	do
        {
            //if no thread changes this value then the loop stops
            stop=false;

            //omp_set_num_threads(num_omp_threads);
 { 
pragma127 *ctx = (pragma127 *)malloc(sizeof(pragma127));
ctx->argc = argc;
ctx->argv = argv;
ctx->no_of_nodes = no_of_nodes;
ctx->edge_list_size = edge_list_size;
ctx->input_f = input_f;
ctx->num_omp_threads = num_omp_threads;
ctx->source = source;
ctx->h_graph_nodes = h_graph_nodes;
ctx->h_graph_mask = h_graph_mask;
ctx->h_updating_graph_mask = h_updating_graph_mask;
ctx->h_graph_visited = h_graph_visited;
ctx->start = start;
ctx->edgeno = edgeno;
ctx->id = id;
ctx->cost = cost;
ctx->h_graph_edges = h_graph_edges;
ctx->h_cost = h_cost;
ctx->k = k;
ctx->stop = stop;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = no_of_nodes;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma127_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 

 { 
pragma144 *ctx = (pragma144 *)malloc(sizeof(pragma144));
ctx->argc = argc;
ctx->argv = argv;
ctx->no_of_nodes = no_of_nodes;
ctx->edge_list_size = edge_list_size;
ctx->input_f = input_f;
ctx->num_omp_threads = num_omp_threads;
ctx->source = source;
ctx->h_graph_nodes = h_graph_nodes;
ctx->h_graph_mask = h_graph_mask;
ctx->h_updating_graph_mask = h_updating_graph_mask;
ctx->h_graph_visited = h_graph_visited;
ctx->start = start;
ctx->edgeno = edgeno;
ctx->id = id;
ctx->cost = cost;
ctx->h_graph_edges = h_graph_edges;
ctx->h_cost = h_cost;
ctx->k = k;
ctx->stop = stop;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = no_of_nodes;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma144_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 
            k++;
        }
	while(stop);
    } ; }

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

main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
ctx->argc = argc;
ctx->argv = argv;
ctx->no_of_nodes = no_of_nodes;
ctx->edge_list_size = edge_list_size;
ctx->input_f = input_f;
ctx->num_omp_threads = num_omp_threads;
ctx->source = source;
ctx->h_graph_nodes = h_graph_nodes;
ctx->h_graph_mask = h_graph_mask;
ctx->h_updating_graph_mask = h_updating_graph_mask;
ctx->h_graph_visited = h_graph_visited;
ctx->start = start;
ctx->edgeno = edgeno;
ctx->id = id;
ctx->cost = cost;
ctx->h_graph_edges = h_graph_edges;
ctx->h_cost = h_cost;
hclib_launch(main_entrypoint, ctx);
free(ctx);


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

}  static void pragma127_hclib_async(void *____arg, const int ___iter) {
    pragma127 *ctx = (pragma127 *)____arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int no_of_nodes; no_of_nodes = ctx->no_of_nodes;
    int edge_list_size; edge_list_size = ctx->edge_list_size;
    char *input_f; input_f = ctx->input_f;
    int num_omp_threads; num_omp_threads = ctx->num_omp_threads;
    int source; source = ctx->source;
    Node *h_graph_nodes; h_graph_nodes = ctx->h_graph_nodes;
    _Bool *h_graph_mask; h_graph_mask = ctx->h_graph_mask;
    _Bool *h_updating_graph_mask; h_updating_graph_mask = ctx->h_updating_graph_mask;
    _Bool *h_graph_visited; h_graph_visited = ctx->h_graph_visited;
    int start; start = ctx->start;
    int edgeno; edgeno = ctx->edgeno;
    int id; id = ctx->id;
    int cost; cost = ctx->cost;
    int *h_graph_edges; h_graph_edges = ctx->h_graph_edges;
    int *h_cost; h_cost = ctx->h_cost;
    int k; k = ctx->k;
    _Bool stop; stop = ctx->stop;
    hclib_start_finish();
    do {
    int tid;     tid = ___iter;
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
            } ;     } while (0);
    ; hclib_end_finish();
}

static void pragma144_hclib_async(void *____arg, const int ___iter) {
    pragma144 *ctx = (pragma144 *)____arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    int no_of_nodes; no_of_nodes = ctx->no_of_nodes;
    int edge_list_size; edge_list_size = ctx->edge_list_size;
    char *input_f; input_f = ctx->input_f;
    int num_omp_threads; num_omp_threads = ctx->num_omp_threads;
    int source; source = ctx->source;
    Node *h_graph_nodes; h_graph_nodes = ctx->h_graph_nodes;
    _Bool *h_graph_mask; h_graph_mask = ctx->h_graph_mask;
    _Bool *h_updating_graph_mask; h_updating_graph_mask = ctx->h_updating_graph_mask;
    _Bool *h_graph_visited; h_graph_visited = ctx->h_graph_visited;
    int start; start = ctx->start;
    int edgeno; edgeno = ctx->edgeno;
    int id; id = ctx->id;
    int cost; cost = ctx->cost;
    int *h_graph_edges; h_graph_edges = ctx->h_graph_edges;
    int *h_cost; h_cost = ctx->h_cost;
    int k; k = ctx->k;
    _Bool stop; stop = ctx->stop;
    hclib_start_finish();
    do {
    int tid;     tid = ___iter;
{
                if (h_updating_graph_mask[tid] == true){
                    h_graph_mask[tid]=true;
                    h_graph_visited[tid]=true;
                    stop=true;
                    h_updating_graph_mask[tid]=false;
                }
            } ;     } while (0);
    ; hclib_end_finish();
}



