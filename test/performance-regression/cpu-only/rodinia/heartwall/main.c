#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include <avilib.h>
#include <avimod.h>
#include <omp.h>

#include "define.h"


//===============================================================================================================================================================================================================200
//	WRITE DATA FUNCTION
//===============================================================================================================================================================================================================200

void write_data(	char* filename,
			int frameNo,
			int frames_processed,
			int endoPoints,
			int* input_a,
			int* input_b,
			int epiPoints,
			int* input_2a,
			int* input_2b){

	//================================================================================80
	//	VARIABLES
	//================================================================================80

	FILE* fid;
	int i,j;
	char c;

	//================================================================================80
	//	OPEN FILE FOR READING
	//================================================================================80

	fid = fopen(filename, "w+");
	if( fid == NULL ){
		printf( "The file was not opened for writing\n" );
		return;
	}


	//================================================================================80
	//	WRITE VALUES TO THE FILE
	//================================================================================80
      fprintf(fid, "Total AVI Frames: %d\n", frameNo);	
      fprintf(fid, "Frames Processed: %d\n", frames_processed);	
      fprintf(fid, "endoPoints: %d\n", endoPoints);
      fprintf(fid, "epiPoints: %d", epiPoints);
	for(j=0; j<frames_processed;j++)
	  {
	    fprintf(fid, "\n---Frame %d---",j);
	    fprintf(fid, "\n--endo--\n",j);
	    for(i=0; i<endoPoints; i++){
	      fprintf(fid, "%d\t", input_a[j+i*frameNo]);
	    }
	    fprintf(fid, "\n");
	    for(i=0; i<endoPoints; i++){
	      // if(input_b[j*size+i] > 2000) input_b[j*size+i]=0;
	      fprintf(fid, "%d\t", input_b[j+i*frameNo]);
	    }
	    fprintf(fid, "\n--epi--\n",j);
	    for(i=0; i<epiPoints; i++){
	      //if(input_2a[j*size_2+i] > 2000) input_2a[j*size_2+i]=0;
	      fprintf(fid, "%d\t", input_2a[j+i*frameNo]);
	    }
	    fprintf(fid, "\n");
	    for(i=0; i<epiPoints; i++){
	      //if(input_2b[j*size_2+i] > 2000) input_2b[j*size_2+i]=0;
	      fprintf(fid, "%d\t", input_2b[j+i*frameNo]);
	    }
	  }
	// 	================================================================================80
	//		CLOSE FILE
		  //	================================================================================80

	fclose(fid);

}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

typedef struct _pragma556_omp_parallel {
    int i;
    int (*frames_processed_ptr);
    public_struct (*public_s_ptr);
    private_struct (*private_s_ptr)[51];
    char (*(*video_file_name_ptr));
    avi_t (*(*d_frames_ptr));
    int (*omp_num_threads_ptr);
    int (*argc_ptr);
    char (*(*(*argv_ptr)));
 } pragma556_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma556_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int i) {
        }
};

#else
static void pragma556_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
int main(int argc, char *argv []){

	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	// counters
	int i;
	int frames_processed;

	// parameters
	public_struct public_s;
	private_struct private_s[ALL_POINTS];

	//======================================================================================================================================================
	// 	FRAMES
	//======================================================================================================================================================

 	
	
	if(argc!=4){
		printf("ERROR: usage: heartwall <inputfile> <num of frames> <num of threads>\n");
		exit(1);
	}
	
	char* video_file_name;
	video_file_name = argv[1];
	
	avi_t* d_frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
	if (d_frames == NULL)  {
		   AVI_print_error((char *) "Error with AVI_open_input_file");
		   return -1;
	}

	public_s.d_frames = d_frames;
	public_s.frames = AVI_video_frames(public_s.d_frames);
	public_s.frame_rows = AVI_video_height(public_s.d_frames);
	public_s.frame_cols = AVI_video_width(public_s.d_frames);
	public_s.frame_elem = public_s.frame_rows * public_s.frame_cols;
	public_s.frame_mem = sizeof(fp) * public_s.frame_elem;

	//======================================================================================================================================================
	// 	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================

	
	frames_processed = atoi(argv[2]);
	if(frames_processed<0 || frames_processed>public_s.frames){
		printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", frames_processed, public_s.frames);
		return 0;
	}
	
	int omp_num_threads;
	omp_num_threads = atoi(argv[3]);
	if (omp_num_threads <=0){
	   printf ("num of threads must be a positive integer");
	   return 0;
	}
	
	printf("num of threads: %d\n", omp_num_threads);
	
	//======================================================================================================================================================
	//	INPUTS
	//======================================================================================================================================================

	//====================================================================================================
	//	ENDO POINTS
	//====================================================================================================

	public_s.endoPoints = ENDO_POINTS;
	public_s.d_endo_mem = sizeof(int) * public_s.endoPoints;
	public_s.d_endoRow = (int *)malloc(public_s.d_endo_mem);
	public_s.d_endoRow[ 0] = 369;
	public_s.d_endoRow[ 1] = 400;
	public_s.d_endoRow[ 2] = 429;
	public_s.d_endoRow[ 3] = 452;
	public_s.d_endoRow[ 4] = 476;
	public_s.d_endoRow[ 5] = 486;
	public_s.d_endoRow[ 6] = 479;
	public_s.d_endoRow[ 7] = 458;
	public_s.d_endoRow[ 8] = 433;
	public_s.d_endoRow[ 9] = 404;
	public_s.d_endoRow[10] = 374;
	public_s.d_endoRow[11] = 346;
	public_s.d_endoRow[12] = 318;
	public_s.d_endoRow[13] = 294;
	public_s.d_endoRow[14] = 277;
	public_s.d_endoRow[15] = 269;
	public_s.d_endoRow[16] = 275;
	public_s.d_endoRow[17] = 287;
	public_s.d_endoRow[18] = 311;
	public_s.d_endoRow[19] = 339;
	public_s.d_endoCol = (int *)malloc(public_s.d_endo_mem);
	public_s.d_endoCol[ 0] = 408;
	public_s.d_endoCol[ 1] = 406;
	public_s.d_endoCol[ 2] = 397;
	public_s.d_endoCol[ 3] = 383;
	public_s.d_endoCol[ 4] = 354;
	public_s.d_endoCol[ 5] = 322;
	public_s.d_endoCol[ 6] = 294;
	public_s.d_endoCol[ 7] = 270;
	public_s.d_endoCol[ 8] = 250;
	public_s.d_endoCol[ 9] = 237;
	public_s.d_endoCol[10] = 235;
	public_s.d_endoCol[11] = 241;
	public_s.d_endoCol[12] = 254;
	public_s.d_endoCol[13] = 273;
	public_s.d_endoCol[14] = 300;
	public_s.d_endoCol[15] = 328;
	public_s.d_endoCol[16] = 356;
	public_s.d_endoCol[17] = 383;
	public_s.d_endoCol[18] = 401;
	public_s.d_endoCol[19] = 411;
	public_s.d_tEndoRowLoc = (int *)malloc(public_s.d_endo_mem * public_s.frames);
	public_s.d_tEndoColLoc = (int *)malloc(public_s.d_endo_mem * public_s.frames);

	//====================================================================================================
	//	EPI POINTS
	//====================================================================================================

	public_s.epiPoints = EPI_POINTS;
	public_s.d_epi_mem = sizeof(int) * public_s.epiPoints;
	public_s.d_epiRow = (int *)malloc(public_s.d_epi_mem);
	public_s.d_epiRow[ 0] = 390;
	public_s.d_epiRow[ 1] = 419;
	public_s.d_epiRow[ 2] = 448;
	public_s.d_epiRow[ 3] = 474;
	public_s.d_epiRow[ 4] = 501;
	public_s.d_epiRow[ 5] = 519;
	public_s.d_epiRow[ 6] = 535;
	public_s.d_epiRow[ 7] = 542;
	public_s.d_epiRow[ 8] = 543;
	public_s.d_epiRow[ 9] = 538;
	public_s.d_epiRow[10] = 528;
	public_s.d_epiRow[11] = 511;
	public_s.d_epiRow[12] = 491;
	public_s.d_epiRow[13] = 466;
	public_s.d_epiRow[14] = 438;
	public_s.d_epiRow[15] = 406;
	public_s.d_epiRow[16] = 376;
	public_s.d_epiRow[17] = 347;
	public_s.d_epiRow[18] = 318;
	public_s.d_epiRow[19] = 291;
	public_s.d_epiRow[20] = 275;
	public_s.d_epiRow[21] = 259;
	public_s.d_epiRow[22] = 256;
	public_s.d_epiRow[23] = 252;
	public_s.d_epiRow[24] = 252;
	public_s.d_epiRow[25] = 257;
	public_s.d_epiRow[26] = 266;
	public_s.d_epiRow[27] = 283;
	public_s.d_epiRow[28] = 305;
	public_s.d_epiRow[29] = 331;
	public_s.d_epiRow[30] = 360;
	public_s.d_epiCol = (int *)malloc(public_s.d_epi_mem);
	public_s.d_epiCol[ 0] = 457;
	public_s.d_epiCol[ 1] = 454;
	public_s.d_epiCol[ 2] = 446;
	public_s.d_epiCol[ 3] = 431;
	public_s.d_epiCol[ 4] = 411;
	public_s.d_epiCol[ 5] = 388;
	public_s.d_epiCol[ 6] = 361;
	public_s.d_epiCol[ 7] = 331;
	public_s.d_epiCol[ 8] = 301;
	public_s.d_epiCol[ 9] = 273;
	public_s.d_epiCol[10] = 243;
	public_s.d_epiCol[11] = 218;
	public_s.d_epiCol[12] = 196;
	public_s.d_epiCol[13] = 178;
	public_s.d_epiCol[14] = 166;
	public_s.d_epiCol[15] = 157;
	public_s.d_epiCol[16] = 155;
	public_s.d_epiCol[17] = 165;
	public_s.d_epiCol[18] = 177;
	public_s.d_epiCol[19] = 197;
	public_s.d_epiCol[20] = 218;
	public_s.d_epiCol[21] = 248;
	public_s.d_epiCol[22] = 276;
	public_s.d_epiCol[23] = 304;
	public_s.d_epiCol[24] = 333;
	public_s.d_epiCol[25] = 361;
	public_s.d_epiCol[26] = 391;
	public_s.d_epiCol[27] = 415;
	public_s.d_epiCol[28] = 434;
	public_s.d_epiCol[29] = 448;
	public_s.d_epiCol[30] = 455;
	public_s.d_tEpiRowLoc = (int *)malloc(public_s.d_epi_mem * public_s.frames);
	public_s.d_tEpiColLoc = (int *)malloc(public_s.d_epi_mem * public_s.frames);

	//====================================================================================================
	//	ALL POINTS
	//====================================================================================================

	public_s.allPoints = ALL_POINTS;

	//======================================================================================================================================================
	//	CONSTANTS
	//======================================================================================================================================================

	public_s.tSize = 25;
	public_s.sSize = 40;
	public_s.maxMove = 10;
	public_s.alpha = 0.87;

	//======================================================================================================================================================
	//	SUMS
	//======================================================================================================================================================

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].in_partial_sum = (fp *)malloc(sizeof(fp) * 2*public_s.tSize+1);
		private_s[i].in_sqr_partial_sum = (fp *)malloc(sizeof(fp) * 2*public_s.tSize+1);
		private_s[i].par_max_val = (fp *)malloc(sizeof(fp) * (2*public_s.tSize+2*public_s.sSize+1));
		private_s[i].par_max_coo = (int *)malloc(sizeof(int) * (2*public_s.tSize+2*public_s.sSize+1));
	}

	//======================================================================================================================================================
	// 	INPUT 2 (SAMPLE AROUND POINT)
	//======================================================================================================================================================

	public_s.in2_rows = 2 * public_s.sSize + 1;
	public_s.in2_cols = 2 * public_s.sSize + 1;
	public_s.in2_elem = public_s.in2_rows * public_s.in2_cols;
	public_s.in2_mem = sizeof(fp) * public_s.in2_elem;

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].d_in2 = (fp *)malloc(public_s.in2_mem);
		private_s[i].d_in2_sqr = (fp *)malloc(public_s.in2_mem);
	}

	//======================================================================================================================================================
	// 	INPUT (POINT TEMPLATE)
	//======================================================================================================================================================

	public_s.in_mod_rows = public_s.tSize+1+public_s.tSize;
	public_s.in_mod_cols = public_s.in_mod_rows;
	public_s.in_mod_elem = public_s.in_mod_rows * public_s.in_mod_cols;
	public_s.in_mod_mem = sizeof(fp) * public_s.in_mod_elem;

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].d_in_mod = (fp *)malloc(public_s.in_mod_mem);
		private_s[i].d_in_sqr = (fp *)malloc(public_s.in_mod_mem);
	}

	//======================================================================================================================================================
	// 	ARRAY OF TEMPLATES FOR ALL POINTS
	//======================================================================================================================================================

	public_s.d_endoT = (fp *)malloc(public_s.in_mod_mem * public_s.endoPoints);
	public_s.d_epiT = (fp *)malloc(public_s.in_mod_mem * public_s.epiPoints);

	//======================================================================================================================================================
	// 	SETUP private_s POINTERS TO ROWS, COLS  AND TEMPLATE
	//======================================================================================================================================================

	for(i=0; i<public_s.endoPoints; i++){
		private_s[i].point_no = i;
		private_s[i].in_pointer = private_s[i].point_no * public_s.in_mod_elem;
		private_s[i].d_Row = public_s.d_endoRow;												// original row coordinates
		private_s[i].d_Col = public_s.d_endoCol;													// original col coordinates
		private_s[i].d_tRowLoc = public_s.d_tEndoRowLoc;									// updated row coordinates
		private_s[i].d_tColLoc = public_s.d_tEndoColLoc;										// updated row coordinates
		private_s[i].d_T = public_s.d_endoT;														// templates
	}

	for(i=public_s.endoPoints; i<public_s.allPoints; i++){
		private_s[i].point_no = i-public_s.endoPoints;
		private_s[i].in_pointer = private_s[i].point_no * public_s.in_mod_elem;
		private_s[i].d_Row = public_s.d_epiRow;
		private_s[i].d_Col = public_s.d_epiCol;
		private_s[i].d_tRowLoc = public_s.d_tEpiRowLoc;
		private_s[i].d_tColLoc = public_s.d_tEpiColLoc;
		private_s[i].d_T = public_s.d_epiT;
	}

	//======================================================================================================================================================
	// 	CONVOLUTION
	//======================================================================================================================================================

	public_s.ioffset = 0;
	public_s.joffset = 0;
	public_s.conv_rows = public_s.in_mod_rows + public_s.in2_rows - 1;												// number of rows in I
	public_s.conv_cols = public_s.in_mod_cols + public_s.in2_cols - 1;												// number of columns in I
	public_s.conv_elem = public_s.conv_rows * public_s.conv_cols;												// number of elements
	public_s.conv_mem = sizeof(fp) * public_s.conv_elem;

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].d_conv = (fp *)malloc(public_s.conv_mem);
	}

	//======================================================================================================================================================
	// 	CUMULATIVE SUM
	//======================================================================================================================================================

	//====================================================================================================
	//	PAD ARRAY
	//====================================================================================================
	//====================================================================================================
	//	VERTICAL CUMULATIVE SUM
	//====================================================================================================

	public_s.in2_pad_add_rows = public_s.in_mod_rows;
	public_s.in2_pad_add_cols = public_s.in_mod_cols;
	public_s.in2_pad_rows = public_s.in2_rows + 2*public_s.in2_pad_add_rows;
	public_s.in2_pad_cols = public_s.in2_cols + 2*public_s.in2_pad_add_cols;
	public_s.in2_pad_elem = public_s.in2_pad_rows * public_s.in2_pad_cols;
	public_s.in2_pad_mem = sizeof(fp) * public_s.in2_pad_elem;

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].d_in2_pad = (fp *)malloc(public_s.in2_pad_mem);
	}

	//====================================================================================================
	//	SELECTION, SELECTION 2, SUBTRACTION
	//====================================================================================================
	//====================================================================================================
	//	HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	public_s.in2_pad_cumv_sel_rowlow = 1 + public_s.in_mod_rows;													// (1 to n+1)
	public_s.in2_pad_cumv_sel_rowhig = public_s.in2_pad_rows - 1;
	public_s.in2_pad_cumv_sel_collow = 1;
	public_s.in2_pad_cumv_sel_colhig = public_s.in2_pad_cols;
	public_s.in2_pad_cumv_sel2_rowlow = 1;
	public_s.in2_pad_cumv_sel2_rowhig = public_s.in2_pad_rows - public_s.in_mod_rows - 1;
	public_s.in2_pad_cumv_sel2_collow = 1;
	public_s.in2_pad_cumv_sel2_colhig = public_s.in2_pad_cols;
	public_s.in2_sub_rows = public_s.in2_pad_cumv_sel_rowhig - public_s.in2_pad_cumv_sel_rowlow + 1;
	public_s.in2_sub_cols = public_s.in2_pad_cumv_sel_colhig - public_s.in2_pad_cumv_sel_collow + 1;
	public_s.in2_sub_elem = public_s.in2_sub_rows * public_s.in2_sub_cols;
	public_s.in2_sub_mem = sizeof(fp) * public_s.in2_sub_elem;

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].d_in2_sub = (fp *)malloc(public_s.in2_sub_mem);
	}

	//====================================================================================================
	//	SELECTION, SELECTION 2, SUBTRACTION, SQUARE, NUMERATOR
	//====================================================================================================

	public_s.in2_sub_cumh_sel_rowlow = 1;
	public_s.in2_sub_cumh_sel_rowhig = public_s.in2_sub_rows;
	public_s.in2_sub_cumh_sel_collow = 1 + public_s.in_mod_cols;
	public_s.in2_sub_cumh_sel_colhig = public_s.in2_sub_cols - 1;
	public_s.in2_sub_cumh_sel2_rowlow = 1;
	public_s.in2_sub_cumh_sel2_rowhig = public_s.in2_sub_rows;
	public_s.in2_sub_cumh_sel2_collow = 1;
	public_s.in2_sub_cumh_sel2_colhig = public_s.in2_sub_cols - public_s.in_mod_cols - 1;
	public_s.in2_sub2_sqr_rows = public_s.in2_sub_cumh_sel_rowhig - public_s.in2_sub_cumh_sel_rowlow + 1;
	public_s.in2_sub2_sqr_cols = public_s.in2_sub_cumh_sel_colhig - public_s.in2_sub_cumh_sel_collow + 1;
	public_s.in2_sub2_sqr_elem = public_s.in2_sub2_sqr_rows * public_s.in2_sub2_sqr_cols;
	public_s.in2_sub2_sqr_mem = sizeof(fp) * public_s.in2_sub2_sqr_elem;

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].d_in2_sub2_sqr = (fp *)malloc(public_s.in2_sub2_sqr_mem);
	}

	//======================================================================================================================================================
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================

	//====================================================================================================
	//	PAD ARRAY
	//====================================================================================================
	//====================================================================================================
	//	VERTICAL CUMULATIVE SUM
	//====================================================================================================

	//====================================================================================================
	//	SELECTION, SELECTION 2, SUBTRACTION
	//====================================================================================================
	//====================================================================================================
	//	HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	//====================================================================================================
	//	SELECTION, SELECTION 2, SUBTRACTION, DIFFERENTIAL LOCAL SUM, DENOMINATOR A, DENOMINATOR, CORRELATION
	//====================================================================================================

	//======================================================================================================================================================
	//	TEMPLATE MASK CREATE
	//======================================================================================================================================================

	public_s.tMask_rows = public_s.in_mod_rows + (public_s.sSize+1+public_s.sSize) - 1;
	public_s.tMask_cols = public_s.tMask_rows;
	public_s.tMask_elem = public_s.tMask_rows * public_s.tMask_cols;
	public_s.tMask_mem = sizeof(fp) * public_s.tMask_elem;

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].d_tMask = (fp *)malloc(public_s.tMask_mem);
	}

	//======================================================================================================================================================
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================

	public_s.mask_rows = public_s.maxMove;
	public_s.mask_cols = public_s.mask_rows;
	public_s.mask_elem = public_s.mask_rows * public_s.mask_cols;
	public_s.mask_mem = sizeof(fp) * public_s.mask_elem;

	//======================================================================================================================================================
	//	MASK CONVOLUTION
	//======================================================================================================================================================

	public_s.mask_conv_rows = public_s.tMask_rows;												// number of rows in I
	public_s.mask_conv_cols = public_s.tMask_cols;												// number of columns in I
	public_s.mask_conv_elem = public_s.mask_conv_rows * public_s.mask_conv_cols;												// number of elements
	public_s.mask_conv_mem = sizeof(fp) * public_s.mask_conv_elem;
	public_s.mask_conv_ioffset = (public_s.mask_rows-1)/2;
	if((public_s.mask_rows-1) % 2 > 0.5){
		public_s.mask_conv_ioffset = public_s.mask_conv_ioffset + 1;
	}
	public_s.mask_conv_joffset = (public_s.mask_cols-1)/2;
	if((public_s.mask_cols-1) % 2 > 0.5){
		public_s.mask_conv_joffset = public_s.mask_conv_joffset + 1;
	}

	for(i=0; i<public_s.allPoints; i++){
		private_s[i].d_mask_conv = (fp *)malloc(public_s.mask_conv_mem);
	}

	//======================================================================================================================================================
	//	PRINT FRAME PROGRESS START
	//======================================================================================================================================================

	printf("frame progress: ");
	fflush(NULL);

	//======================================================================================================================================================
	//	KERNEL
	//======================================================================================================================================================

	for(public_s.frame_no=0; public_s.frame_no<frames_processed; public_s.frame_no++){

	//====================================================================================================
	//	GETTING FRAME
	//====================================================================================================

		// Extract a cropped version of the first frame from the video file
		public_s.d_frame = get_frame(public_s.d_frames,				// pointer to video file
													public_s.frame_no,				// number of frame that needs to be returned
													0,										// cropped?
													0,										// scaled?
													1);									// converted

	//====================================================================================================
	//	PROCESSING
	//====================================================================================================

 { 
pragma556_omp_parallel *new_ctx = (pragma556_omp_parallel *)malloc(sizeof(pragma556_omp_parallel));
new_ctx->i = i;
new_ctx->frames_processed_ptr = &(frames_processed);
new_ctx->public_s_ptr = &(public_s);
new_ctx->private_s_ptr = &(private_s);
new_ctx->video_file_name_ptr = &(video_file_name);
new_ctx->d_frames_ptr = &(d_frames);
new_ctx->omp_num_threads_ptr = &(omp_num_threads);
new_ctx->argc_ptr = &(argc);
new_ctx->argv_ptr = &(argv);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = public_s.allPoints;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((public_s.allPoints) - (0), pragma556_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma556_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 

	//====================================================================================================
	//	FREE MEMORY FOR FRAME
	//====================================================================================================

		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		// free(public_s.d_frame);

	//====================================================================================================
	//	PRINT FRAME PROGRESS
	//====================================================================================================

		printf("%d ", public_s.frame_no);
		fflush(NULL);

	} ; 

	//======================================================================================================================================================
	//	PRINT FRAME PROGRESS END
	//======================================================================================================================================================

	printf("\n");
	fflush(NULL);

	//======================================================================================================================================================
	//	DEALLOCATION
	//======================================================================================================================================================

	//==================================================50
	//	DUMP DATA TO FILE
	//==================================================50
#ifdef OUTPUT
	write_data(	"result.txt",
			public_s.frames,
			frames_processed,		
				public_s.endoPoints,
				public_s.d_tEndoRowLoc,
				public_s.d_tEndoColLoc,
				public_s.epiPoints,
				public_s.d_tEpiRowLoc,
				public_s.d_tEpiColLoc);

#endif



	//====================================================================================================
	//	COMMON
	//====================================================================================================

	free(public_s.d_endoRow);
	free(public_s.d_endoCol);
	free(public_s.d_tEndoRowLoc);
	free(public_s.d_tEndoColLoc);
	free(public_s.d_endoT);

	free(public_s.d_epiRow);
	free(public_s.d_epiCol);
	free(public_s.d_tEpiRowLoc);
	free(public_s.d_tEpiColLoc);
	free(public_s.d_epiT);

	//====================================================================================================
	//	POINTERS
	//====================================================================================================

	for(i=0; i<public_s.allPoints; i++){
		free(private_s[i].in_partial_sum);
		free(private_s[i].in_sqr_partial_sum);
		free(private_s[i].par_max_val);
		free(private_s[i].par_max_coo);

		free(private_s[i].d_in2);
		free(private_s[i].d_in2_sqr);

		free(private_s[i].d_in_mod);
		free(private_s[i].d_in_sqr);

		free(private_s[i].d_conv);

		free(private_s[i].d_in2_pad);

		free(private_s[i].d_in2_sub);

		free(private_s[i].d_in2_sub2_sqr);

		free(private_s[i].d_tMask);
		free(private_s[i].d_mask_conv);
	}

    return 0;
} 

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma556_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma556_omp_parallel *ctx = (pragma556_omp_parallel *)____arg;
    int i; i = ctx->i;
    hclib_start_finish();
    do {
    i = ___iter0;
{
			kernel(	(*(ctx->public_s_ptr)),
						(*(ctx->private_s_ptr))[i]);
		} ;     } while (0);
    ; hclib_end_finish_nonblocking();

}

#endif


//========================================================================================================================================================================================================
//========================================================================================================================================================================================================
//	END OF FILE
//========================================================================================================================================================================================================
//========================================================================================================================================================================================================
