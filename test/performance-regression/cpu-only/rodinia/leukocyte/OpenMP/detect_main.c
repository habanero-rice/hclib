#include "hclib.h"
#include "find_ellipse.h"
#include "track_ellipse.h"

int omp_num_threads = 1;

typedef struct _main_entrypoint_ctx {
    long long program_start_time;
    int num_frames;
    char (*video_file_name);
    avi_t (*cell_file);
    int i;
    int j;
    int (*crow);
    int (*ccol);
    int pair_counter;
    int x_result_len;
    int Iter;
    int ns;
    int k_count;
    int n;
    MAT (*cellx);
    MAT (*celly);
    MAT (*A);
    double (*GICOV_spots);
    double (*t);
    double (*G);
    double (*x_result);
    double (*y_result);
    double (*V);
    double (*QAX_CENTERS);
    double (*QAY_CENTERS);
    double threshold;
    double radius;
    double delta;
    double dt;
    double b;
    MAT (*image_chopped);
    MAT (*grad_x);
    MAT (*grad_y);
    int argc;
    char (*(*argv));
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    long long program_start_time; program_start_time = ctx->program_start_time;
    int num_frames; num_frames = ctx->num_frames;
    char (*video_file_name); video_file_name = ctx->video_file_name;
    avi_t (*cell_file); cell_file = ctx->cell_file;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int (*crow); crow = ctx->crow;
    int (*ccol); ccol = ctx->ccol;
    int pair_counter; pair_counter = ctx->pair_counter;
    int x_result_len; x_result_len = ctx->x_result_len;
    int Iter; Iter = ctx->Iter;
    int ns; ns = ctx->ns;
    int k_count; k_count = ctx->k_count;
    int n; n = ctx->n;
    MAT (*cellx); cellx = ctx->cellx;
    MAT (*celly); celly = ctx->celly;
    MAT (*A); A = ctx->A;
    double (*GICOV_spots); GICOV_spots = ctx->GICOV_spots;
    double (*t); t = ctx->t;
    double (*G); G = ctx->G;
    double (*x_result); x_result = ctx->x_result;
    double (*y_result); y_result = ctx->y_result;
    double (*V); V = ctx->V;
    double (*QAX_CENTERS); QAX_CENTERS = ctx->QAX_CENTERS;
    double (*QAY_CENTERS); QAY_CENTERS = ctx->QAY_CENTERS;
    double threshold; threshold = ctx->threshold;
    double radius; radius = ctx->radius;
    double delta; delta = ctx->delta;
    double dt; dt = ctx->dt;
    double b; b = ctx->b;
    MAT (*image_chopped); image_chopped = ctx->image_chopped;
    MAT (*grad_x); grad_x = ctx->grad_x;
    MAT (*grad_y); grad_y = ctx->grad_y;
    int argc; argc = ctx->argc;
    char (*(*argv)); argv = ctx->argv;
{
	// Get GICOV matrix corresponding to image gradients
	long long GICOV_start_time = get_time();
	MAT *gicov = ellipsematching(grad_x, grad_y);
	
	// Square GICOV values
	MAT *max_gicov = m_get(gicov->m, gicov->n);
	for (i = 0; i < gicov->m; i++) {
		for (j = 0; j < gicov->n; j++) {
			double val = m_get_val(gicov, i, j);
			m_set_val(max_gicov, i, j, val * val);
		}
	}
	
	long long GICOV_end_time = get_time();
	
	// Dilate the GICOV matrix
	long long dilate_start_time = get_time();
	MAT *strel = structuring_element(12);
	MAT *img_dilated = dilate_f(max_gicov, strel);
	long long dilate_end_time = get_time();
	
	// Find possible matches for cell centers based on GICOV and record the rows/columns in which they are found
	pair_counter = 0;
	crow = (int *) malloc(max_gicov->m * max_gicov->n * sizeof(int));
	ccol = (int *) malloc(max_gicov->m * max_gicov->n * sizeof(int));
	for (i = 0; i < max_gicov->m; i++) {
		for (j = 0; j < max_gicov->n; j++) {
			if (!(m_get_val(max_gicov,i,j) == 0.0) && (m_get_val(img_dilated,i,j) == m_get_val(max_gicov,i,j))) {
				crow[pair_counter] = i;
				ccol[pair_counter] = j;
				pair_counter++;
			}
		}
	}
	
	GICOV_spots = (double *) malloc(sizeof(double)*pair_counter);
	for (i = 0; i < pair_counter; i++) {
		GICOV_spots[i] = m_get_val(gicov, crow[i], ccol[i]);
    }
	
	G = (double *) calloc(pair_counter, sizeof(double));
	x_result = (double *) calloc(pair_counter, sizeof(double));
	y_result = (double *) calloc(pair_counter, sizeof(double));
	
	x_result_len = 0;
	for (i = 0; i < pair_counter; i++) {
		if ((crow[i] > 29) && (crow[i] < BOTTOM - TOP + 39)) {
			x_result[x_result_len] = ccol[i];
			y_result[x_result_len] = crow[i] - 40;
			G[x_result_len] = GICOV_spots[i];
			x_result_len++;
		}
	}
	
	// Make an array t which holds each "time step" for the possible cells
	t = (double *) malloc(sizeof(double) * 36);
	for (i = 0; i < 36; i++) {
		t[i] = (double)i * 2.0 * PI / 36.0;
	}
	
	// Store cell boundaries (as simple circles) for all cells
	cellx = m_get(x_result_len, 36);
	celly = m_get(x_result_len, 36);
	for(i = 0; i < x_result_len; i++) {
		for(j = 0; j < 36; j++) {
			m_set_val(cellx, i, j, x_result[i] + radius * cos(t[j]));
			m_set_val(celly, i, j, y_result[i] + radius * sin(t[j]));
		}
	}
	
	A = TMatrix(9,4);

	
	V = (double *) calloc(pair_counter, sizeof(double));
	QAX_CENTERS = (double * )calloc(pair_counter, sizeof(double));
	QAY_CENTERS = (double *) calloc(pair_counter, sizeof(double));

	// For all possible results, find the ones that are feasibly leukocytes and store their centers
	k_count = 0;
	for (n = 0; n < x_result_len; n++) {
		if ((G[n] < -1 * threshold) || G[n] > threshold) {
			MAT * x, *y;
			VEC * x_row, * y_row;
			x = m_get(1, 36);
			y = m_get(1, 36);

			x_row = v_get(36);
			y_row = v_get(36);

			// Get current values of possible cells from cellx/celly matrices
			x_row = get_row(cellx, n, x_row);
			y_row = get_row(celly, n, y_row);
			uniformseg(x_row, y_row, x, y);

			// Make sure that the possible leukocytes are not too close to the edge of the frame
			if ((m_min(x) > b) && (m_min(y) > b) && (m_max(x) < cell_file->width - b) && (m_max(y) < cell_file->height - b)) {
				MAT * Cx, * Cy, *Cy_temp, * Ix1, * Iy1;
				VEC  *Xs, *Ys, *W, *Nx, *Ny, *X, *Y;
				Cx = m_get(1, 36);
				Cy = m_get(1, 36);
				Cx = mmtr_mlt(A, x, Cx);
				Cy = mmtr_mlt(A, y, Cy);
				
				Cy_temp = m_get(Cy->m, Cy->n);
				
				for (i = 0; i < 9; i++) {
					m_set_val(Cy, i, 0, m_get_val(Cy, i, 0) + 40.0);
                }
					
				// Iteratively refine the snake/spline
				for (i = 0; i < Iter; i++) {
					int typeofcell;
					
					if(G[n] > 0.0) {
                        typeofcell = 0;
                    } else {
                        typeofcell = 1;
                    }
					
					splineenergyform01(Cx, Cy, grad_x, grad_y, ns, delta, 2.0 * dt, typeofcell);
				}
				
				X = getsampling(Cx, ns);
				for (i = 0; i < Cy->m; i++) {
					m_set_val(Cy_temp, i, 0, m_get_val(Cy, i, 0) - 40.0);
                }
				Y = getsampling(Cy_temp, ns);
				
				Ix1 = linear_interp2(grad_x, X, Y);
				Iy1 = linear_interp2(grad_x, X, Y);
				Xs = getfdriv(Cx, ns);
				Ys = getfdriv(Cy, ns);
				
				Nx = v_get(Ys->dim);
				for (i = 0; i < Ys->dim; i++) {
					v_set_val(Nx, i, v_get_val(Ys, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));
                }
					
				Ny = v_get(Xs->dim);
				for (i = 0; i < Xs->dim; i++) {
					v_set_val(Ny, i, -1.0 * v_get_val(Xs, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));
                }
					
				W = v_get(Nx->dim);
				for (i = 0; i < Nx->dim; i++) {
					v_set_val(W, i, m_get_val(Ix1, 0, i) * v_get_val(Nx, i) + m_get_val(Iy1, 0, i) * v_get_val(Ny, i));
                }
					
				V[n] = mean(W) / std_dev(W);
				
				//get means of X and Y values for all "snaxels" of the spline contour, thus finding the cell centers
				QAX_CENTERS[k_count] = mean(X);
				QAY_CENTERS[k_count] = mean(Y) + TOP;
				
				k_count++;
				
				// Free memory
				v_free(W);
				v_free(Ny);
				v_free(Nx);
				v_free(Ys);
				v_free(Xs);
				m_free(Iy1);
				m_free(Ix1);
				v_free(Y);
				v_free(X);
				m_free(Cy_temp);
				m_free(Cy);
				m_free(Cx);				
			}
			
			// Free memory
			v_free(y_row);
			v_free(x_row);
			m_free(y);
			m_free(x);
		}
	}

	// Free memory
	free(V);
	free(ccol);
	free(crow);
	free(GICOV_spots);
	free(t);
	free(G);
	free(x_result);
	free(y_result);
	m_free(A);
	m_free(celly);
	m_free(cellx);
	m_free(img_dilated);
	m_free(max_gicov);
	m_free(gicov);
	m_free(grad_y);
	m_free(grad_x);
	
	// Report the total number of cells detected
	printf("Cells detected: %d\n\n", k_count);
	
	// Report the breakdown of the detection runtime
	printf("Detection runtime\n");
	printf("-----------------\n");
	printf("GICOV computation: %.5f seconds\n", ((float) (GICOV_end_time - GICOV_start_time)) / (1000*1000));
	printf("   GICOV dilation: %.5f seconds\n", ((float) (dilate_end_time - dilate_start_time)) / (1000*1000));
	printf("            Total: %.5f seconds\n", ((float) (get_time() - program_start_time)) / (1000*1000));
	
	// Now that the cells have been detected in the first frame,
	//  track the ellipses through subsequent frames
	if (num_frames > 1) printf("\nTracking cells across %d frames\n", num_frames);
	else                printf("\nTracking cells across 1 frame\n");
	long long tracking_start_time = get_time();
	int num_snaxels = 20;
	ellipsetrack(cell_file, QAX_CENTERS, QAY_CENTERS, k_count, radius, num_snaxels, num_frames);
	printf("           Total: %.5f seconds\n", ((float) (get_time() - tracking_start_time)) / (float) (1000*1000*num_frames));
    } ; }

int main(int argc, char ** argv) {

	// Keep track of the start time of the program
  long long program_start_time = get_time();
	
	// Let the user specify the number of frames to process
	int num_frames = 1;
	
	if (argc !=4){
		fprintf(stderr, "usage: %s <num of frames> <num of threads><input file>", argv[0]);
		exit(1);
	}
	
	if (argc > 1){
		num_frames = atoi(argv[1]);
		omp_num_threads = atoi(argv[2]);
		}
	printf("Num of threads: %d\n", omp_num_threads);
	// Open video file
	char *video_file_name;
	video_file_name = argv[3];
	
	avi_t *cell_file = AVI_open_input_file(video_file_name, 1);
	if (cell_file == NULL)	{
		AVI_print_error("Error with AVI_open_input_file");
		return -1;
	}
	
	int i, j, *crow, *ccol, pair_counter = 0, x_result_len = 0, Iter = 20, ns = 4, k_count = 0, n;
	MAT *cellx, *celly, *A;
	double *GICOV_spots, *t, *G, *x_result, *y_result, *V, *QAX_CENTERS, *QAY_CENTERS;
	double threshold = 1.8, radius = 10.0, delta = 3.0, dt = 0.01, b = 5.0;
	
	// Extract a cropped version of the first frame from the video file
	MAT *image_chopped = get_frame(cell_file, 0, 1, 0);
	printf("Detecting cells in frame 0\n");
	
	// Get gradient matrices in x and y directions
	MAT *grad_x = gradient_x(image_chopped);
	MAT *grad_y = gradient_y(image_chopped);
	
	m_free(image_chopped);
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->program_start_time = program_start_time;
new_ctx->num_frames = num_frames;
new_ctx->video_file_name = video_file_name;
new_ctx->cell_file = cell_file;
new_ctx->i = i;
new_ctx->j = j;
new_ctx->crow = crow;
new_ctx->ccol = ccol;
new_ctx->pair_counter = pair_counter;
new_ctx->x_result_len = x_result_len;
new_ctx->Iter = Iter;
new_ctx->ns = ns;
new_ctx->k_count = k_count;
new_ctx->n = n;
new_ctx->cellx = cellx;
new_ctx->celly = celly;
new_ctx->A = A;
new_ctx->GICOV_spots = GICOV_spots;
new_ctx->t = t;
new_ctx->G = G;
new_ctx->x_result = x_result;
new_ctx->y_result = y_result;
new_ctx->V = V;
new_ctx->QAX_CENTERS = QAX_CENTERS;
new_ctx->QAY_CENTERS = QAY_CENTERS;
new_ctx->threshold = threshold;
new_ctx->radius = radius;
new_ctx->delta = delta;
new_ctx->dt = dt;
new_ctx->b = b;
new_ctx->image_chopped = image_chopped;
new_ctx->grad_x = grad_x;
new_ctx->grad_y = grad_y;
new_ctx->argc = argc;
new_ctx->argv = argv;
hclib_launch(main_entrypoint, new_ctx);
free(new_ctx);

	
	// Report total program execution time
    printf("\nTotal application run time: %.5f seconds\n", ((float) (get_time() - program_start_time)) / (1000*1000));

	return 0;
} 
