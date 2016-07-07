#include "hclib.h"
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	KERNEL FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

#include "define.h"
#include <math.h>

void kernel(public_struct public_s,
				private_struct private_s){

	//======================================================================================================================================================
	//	COMMON VARIABLES
	//======================================================================================================================================================

	int ei_new;
	fp* d_in;
	int rot_row;
	int rot_col;
	int in2_rowlow;
	int in2_collow;
	int ic;
	int jc;
	int jp1;
	int ja1, ja2;
	int ip1;
	int ia1, ia2;
	int ja, jb;
	int ia, ib;
	fp s;
	int i;
	int j;
	int row;
	int col;
	int ori_row;
	int ori_col;
	int position;
	fp sum;
	int pos_ori;
	fp temp;
	fp temp2;
	int location;
	int cent;
	int tMask_row; 
	int tMask_col;
	fp largest_value_current = 0;
	fp largest_value = 0;
	int largest_coordinate_current = 0;
	int largest_coordinate = 0;
	fp fin_max_val = 0;
	int fin_max_coo = 0;
	int largest_row;
	int largest_col;
	int offset_row;
	int offset_col;
	fp in_final_sum;
	fp in_sqr_final_sum;
	fp mean;
	fp mean_sqr;
	fp variance;
	fp deviation;
	fp denomT;
	int pointer;
	int ori_pointer;
	int loc_pointer;
	int ei_mod;

	//======================================================================================================================================================
	//	GENERATE TEMPLATE
	//======================================================================================================================================================

	// generate templates based on the first frame only
	if(public_s.frame_no == 0){

		// update temporary row/col coordinates
		pointer = private_s.point_no*public_s.frames+public_s.frame_no;
		private_s.d_tRowLoc[pointer] = private_s.d_Row[private_s.point_no];
		private_s.d_tColLoc[pointer] = private_s.d_Col[private_s.point_no];

		// pointers to: current frame, template for current point
		d_in = &private_s.d_T[private_s.in_pointer];

		// update template, limit the number of working threads to the size of template
		for(col=0; col<public_s.in_mod_cols; col++){
			for(row=0; row<public_s.in_mod_rows; row++){

				// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
				ori_row = private_s.d_Row[private_s.point_no] - 25 + row - 1;
				ori_col = private_s.d_Col[private_s.point_no] - 25 + col - 1;
				ori_pointer = ori_col*public_s.frame_rows+ori_row;

				// update template
				d_in[col*public_s.in_mod_rows+row] = public_s.d_frame[ori_pointer];

			}
		}

	}

	//======================================================================================================================================================
	//	PROCESS POINTS
	//======================================================================================================================================================

	// process points in all frames except for the first one
	if(public_s.frame_no != 0){

		//====================================================================================================
		//	INPUTS
		//====================================================================================================

		//==================================================
		//	1) SETUP POINTER TO POINT TO CURRENT FRAME FROM BATCH
		//	2) SELECT INPUT 2 (SAMPLE AROUND POINT) FROM FRAME			SAVE IN d_in2			(NOT LINEAR IN MEMORY, SO NEED TO SAVE OUTPUT FOR LATER EASY USE)
		//	3) SQUARE INPUT 2									SAVE IN d_in2_sqr
		//==================================================

		// pointers and variables
		in2_rowlow = private_s.d_Row[private_s.point_no] - public_s.sSize;							// (1 to n+1)
		in2_collow = private_s.d_Col[private_s.point_no] - public_s.sSize;

		// work
		for(col=0; col<public_s.in2_cols; col++){
			for(row=0; row<public_s.in2_rows; row++){

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + in2_rowlow - 1;
			ori_col = col + in2_collow - 1;
			temp = public_s.d_frame[ori_col*public_s.frame_rows+ori_row];
			private_s.d_in2[col*public_s.in2_rows+row] = temp;
			private_s.d_in2_sqr[col*public_s.in2_rows+row] = temp*temp;

			}
		}

		//==================================================
		//	1) GET POINTER TO INPUT 1 (TEMPLATE FOR THIS POINT) IN TEMPLATE ARRAY				(LINEAR IN MEMORY, SO DONT NEED TO SAVE, JUST GET POINTER)
		//	2) ROTATE INPUT 1									SAVE IN d_in_mod
		//	3) SQUARE INPUT 1									SAVE IN d_in_sqr
		//==================================================

		// variables
		d_in = &private_s.d_T[private_s.in_pointer];

		// work
		for(col=0; col<public_s.in_mod_cols; col++){
			for(row=0; row<public_s.in_mod_rows; row++){

			// rotated coordinates
			rot_row = (public_s.in_mod_rows-1) - row;
			rot_col = (public_s.in_mod_rows-1) - col;
			pointer = rot_col*public_s.in_mod_rows+rot_row;

			// execution
			temp = d_in[pointer];
			private_s.d_in_mod[col*public_s.in_mod_rows+row] = temp;
			private_s.d_in_sqr[pointer] = temp * temp;

			}
		}

		//==================================================
		//	1) GET SUM OF INPUT 1
		//	2) GET SUM OF INPUT 1 SQUARED
		//==================================================

		in_final_sum = 0;
		for(i = 0; i<public_s.in_mod_elem; i++){
			in_final_sum = in_final_sum + d_in[i];
		}

		in_sqr_final_sum = 0;
		for(i = 0; i<public_s.in_mod_elem; i++){
			in_sqr_final_sum = in_sqr_final_sum + private_s.d_in_sqr[i];
		}

		//==================================================
		//	3) DO STATISTICAL CALCULATIONS
		//	4) GET DENOMINATOR T
		//==================================================

		mean = in_final_sum / public_s.in_mod_elem;													// gets mean (average) value of element in ROI
		mean_sqr = mean * mean;
		variance  = (in_sqr_final_sum / public_s.in_mod_elem) - mean_sqr;							// gets variance of ROI
		deviation = sqrt(variance);																// gets standard deviation of ROI

		denomT = sqrt((fp)(public_s.in_mod_elem-1))*deviation;

		//====================================================================================================
		//	1) CONVOLVE INPUT 2 WITH ROTATED INPUT 1					SAVE IN d_conv
		//====================================================================================================

		// work
		for(col=1; col<=public_s.conv_cols; col++){

			// column setup
			j = col + public_s.joffset;
			jp1 = j + 1;
			if(public_s.in2_cols < jp1){
				ja1 = jp1 - public_s.in2_cols;
			}
			else{
				ja1 = 1;
			}
			if(public_s.in_mod_cols < j){
				ja2 = public_s.in_mod_cols;
			}
			else{
				ja2 = j;
			}

			for(row=1; row<=public_s.conv_rows; row++){

				// row range setup
				i = row + public_s.ioffset;
				ip1 = i + 1;
				
				if(public_s.in2_rows < ip1){
					ia1 = ip1 - public_s.in2_rows;
				}
				else{
					ia1 = 1;
				}
				if(public_s.in_mod_rows < i){
					ia2 = public_s.in_mod_rows;
				}
				else{
					ia2 = i;
				}

				s = 0;

				// getting data
				for(ja=ja1; ja<=ja2; ja++){
					jb = jp1 - ja;
					for(ia=ia1; ia<=ia2; ia++){
						ib = ip1 - ia;
						s = s + private_s.d_in_mod[public_s.in_mod_rows*(ja-1)+ia-1] * private_s.d_in2[public_s.in2_rows*(jb-1)+ib-1];
					}
				}

				private_s.d_conv[(col-1)*public_s.conv_rows+(row-1)] = s;

		}
	}
		//====================================================================================================
		//	LOCAL SUM 1
		//====================================================================================================

		//==================================================
		//	1) PADD ARRAY										SAVE IN d_in2_pad
		//==================================================

		// work
		for(col=0; col<public_s.in2_pad_cols; col++){
			for(row=0; row<public_s.in2_pad_rows; row++){

			// execution
			if(	row > (public_s.in2_pad_add_rows-1) &&														// do if has numbers in original array
				row < (public_s.in2_pad_add_rows+public_s.in2_rows) && 
				col > (public_s.in2_pad_add_cols-1) && 
				col < (public_s.in2_pad_add_cols+public_s.in2_cols)){
				ori_row = row - public_s.in2_pad_add_rows;
				ori_col = col - public_s.in2_pad_add_cols;
				private_s.d_in2_pad[col*public_s.in2_pad_rows+row] = private_s.d_in2[ori_col*public_s.in2_rows+ori_row];
			}
			else{																			// do if otherwise
				private_s.d_in2_pad[col*public_s.in2_pad_rows+row] = 0;
			}

			}
		}

		//==================================================
		//	1) GET VERTICAL CUMULATIVE SUM						SAVE IN d_in2_pad
		//==================================================

		for(ei_new = 0; ei_new < public_s.in2_pad_cols; ei_new++){

			// figure out column position
			pos_ori = ei_new*public_s.in2_pad_rows;

			// loop through all rows
			sum = 0;
			for(position = pos_ori; position < pos_ori+public_s.in2_pad_rows; position = position + 1){
				private_s.d_in2_pad[position] = private_s.d_in2_pad[position] + sum;
				sum = private_s.d_in2_pad[position];
			}

		}

		//==================================================
		//	1) MAKE 1st SELECTION FROM VERTICAL CUMULATIVE SUM
		//	2) MAKE 2nd SELECTION FROM VERTICAL CUMULATIVE SUM
		//	3) SUBTRACT THE TWO SELECTIONS						SAVE IN d_in2_sub
		//==================================================

		// work
		for(col=0; col<public_s.in2_sub_cols; col++){
			for(row=0; row<public_s.in2_sub_rows; row++){

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + public_s.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + public_s.in2_pad_cumv_sel_collow - 1;
			temp = private_s.d_in2_pad[ori_col*public_s.in2_pad_rows+ori_row];

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + public_s.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + public_s.in2_pad_cumv_sel2_collow - 1;
			temp2 = private_s.d_in2_pad[ori_col*public_s.in2_pad_rows+ori_row];

			// subtraction
			private_s.d_in2_sub[col*public_s.in2_sub_rows+row] = temp - temp2;

			}
		}

		//==================================================
		//	1) GET HORIZONTAL CUMULATIVE SUM						SAVE IN d_in2_sub
		//==================================================

		for(ei_new = 0; ei_new < public_s.in2_sub_rows; ei_new++){

			// figure out row position
			pos_ori = ei_new;

			// loop through all rows
			sum = 0;
			for(position = pos_ori; position < pos_ori+public_s.in2_sub_elem; position = position + public_s.in2_sub_rows){
				private_s.d_in2_sub[position] = private_s.d_in2_sub[position] + sum;
				sum = private_s.d_in2_sub[position];
			}

		}

		//==================================================
		//	1) MAKE 1st SELECTION FROM HORIZONTAL CUMULATIVE SUM
		//	2) MAKE 2nd SELECTION FROM HORIZONTAL CUMULATIVE SUM
		//	3) SUBTRACT THE TWO SELECTIONS TO GET LOCAL SUM 1
		//	4) GET CUMULATIVE SUM 1 SQUARED						SAVE IN d_in2_sub2_sqr
		//	5) GET NUMERATOR									SAVE IN d_conv
		//==================================================

		// work
		for(col=0; col<public_s.in2_sub2_sqr_cols; col++){
			for(row=0; row<public_s.in2_sub2_sqr_rows; row++){

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + public_s.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + public_s.in2_sub_cumh_sel_collow - 1;
			temp = private_s.d_in2_sub[ori_col*public_s.in2_sub_rows+ori_row];

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + public_s.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + public_s.in2_sub_cumh_sel2_collow - 1;
			temp2 = private_s.d_in2_sub[ori_col*public_s.in2_sub_rows+ori_row];
			
			// subtraction
			temp2 = temp - temp2;

			// squaring
			private_s.d_in2_sub2_sqr[col*public_s.in2_sub2_sqr_rows+row] = temp2 * temp2; 

			// numerator
			private_s.d_conv[col*public_s.in2_sub2_sqr_rows+row] = private_s.d_conv[col*public_s.in2_sub2_sqr_rows+row] - temp2 * in_final_sum / public_s.in_mod_elem;

			}
		}

		//====================================================================================================
		//	LOCAL SUM 2
		//====================================================================================================

		//==================================================
		//	1) PAD ARRAY										SAVE IN d_in2_pad
		//==================================================

		// work
		for(col=0; col<public_s.in2_pad_cols; col++){
			for(row=0; row<public_s.in2_pad_rows; row++){

			// execution
			if(	row > (public_s.in2_pad_add_rows-1) &&													// do if has numbers in original array
				row < (public_s.in2_pad_add_rows+public_s.in2_rows) && 
				col > (public_s.in2_pad_add_cols-1) && 
				col < (public_s.in2_pad_add_cols+public_s.in2_cols)){
				ori_row = row - public_s.in2_pad_add_rows;
				ori_col = col - public_s.in2_pad_add_cols;
				private_s.d_in2_pad[col*public_s.in2_pad_rows+row] = private_s.d_in2_sqr[ori_col*public_s.in2_rows+ori_row];
			}
			else{																							// do if otherwise
				private_s.d_in2_pad[col*public_s.in2_pad_rows+row] = 0;
			}

			}
		}

		//==================================================
		//	2) GET VERTICAL CUMULATIVE SUM						SAVE IN d_in2_pad
		//==================================================

		//work
		for(ei_new = 0; ei_new < public_s.in2_pad_cols; ei_new++){

			// figure out column position
			pos_ori = ei_new*public_s.in2_pad_rows;

			// loop through all rows
			sum = 0;
			for(position = pos_ori; position < pos_ori+public_s.in2_pad_rows; position = position + 1){
				private_s.d_in2_pad[position] = private_s.d_in2_pad[position] + sum;
				sum = private_s.d_in2_pad[position];
			}

		}

		//==================================================
		//	1) MAKE 1st SELECTION FROM VERTICAL CUMULATIVE SUM
		//	2) MAKE 2nd SELECTION FROM VERTICAL CUMULATIVE SUM
		//	3) SUBTRACT THE TWO SELECTIONS						SAVE IN d_in2_sub
		//==================================================

		// work
		for(col=0; col<public_s.in2_sub_cols; col++){
			for(row=0; row<public_s.in2_sub_rows; row++){

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + public_s.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + public_s.in2_pad_cumv_sel_collow - 1;
			temp = private_s.d_in2_pad[ori_col*public_s.in2_pad_rows+ori_row];

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + public_s.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + public_s.in2_pad_cumv_sel2_collow - 1;
			temp2 = private_s.d_in2_pad[ori_col*public_s.in2_pad_rows+ori_row];

			// subtract
			private_s.d_in2_sub[col*public_s.in2_sub_rows+row] = temp - temp2;

			}
		}

		//==================================================
		//	1) GET HORIZONTAL CUMULATIVE SUM						SAVE IN d_in2_sub
		//==================================================

		for(ei_new = 0; ei_new < public_s.in2_sub_rows; ei_new++){

			// figure out row position
			pos_ori = ei_new;

			// loop through all rows
			sum = 0;
			for(position = pos_ori; position < pos_ori+public_s.in2_sub_elem; position = position + public_s.in2_sub_rows){
				private_s.d_in2_sub[position] = private_s.d_in2_sub[position] + sum;
				sum = private_s.d_in2_sub[position];
			}

		}

		//==================================================
		//	1) MAKE 1st SELECTION FROM HORIZONTAL CUMULATIVE SUM
		//	2) MAKE 2nd SELECTION FROM HORIZONTAL CUMULATIVE SUM
		//	3) SUBTRACT THE TWO SELECTIONS TO GET LOCAL SUM 2
		//	4) GET DIFFERENTIAL LOCAL SUM
		//	5) GET DENOMINATOR A
		//	6) GET DENOMINATOR
		//	7) DIVIDE NUMBERATOR BY DENOMINATOR TO GET CORRELATION	SAVE IN d_conv
		//==================================================

		// work
		for(col=0; col<public_s.conv_cols; col++){
			for(row=0; row<public_s.conv_rows; row++){

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + public_s.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + public_s.in2_sub_cumh_sel_collow - 1;
			temp = private_s.d_in2_sub[ori_col*public_s.in2_sub_rows+ori_row];

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + public_s.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + public_s.in2_sub_cumh_sel2_collow - 1;
			temp2 = private_s.d_in2_sub[ori_col*public_s.in2_sub_rows+ori_row];

			// subtract
			temp2 = temp - temp2;

			// diff_local_sums
			temp2 = temp2 - (private_s.d_in2_sub2_sqr[col*public_s.conv_rows+row] / public_s.in_mod_elem);

			// denominator A
			if(temp2 < 0){
				temp2 = 0;
			}
			temp2 = sqrt(temp2);

			// denominator
			temp2 = denomT * temp2;
			
			// correlation
			private_s.d_conv[col*public_s.conv_rows+row] = private_s.d_conv[col*public_s.conv_rows+row] / temp2;

			}
		}

		//====================================================================================================
		//	TEMPLATE MASK CREATE
		//====================================================================================================

		// parameters
		cent = public_s.sSize + public_s.tSize + 1;
		pointer = public_s.frame_no-1+private_s.point_no*public_s.frames;
		tMask_row = cent + private_s.d_tRowLoc[pointer] - private_s.d_Row[private_s.point_no] - 1;
		tMask_col = cent + private_s.d_tColLoc[pointer] - private_s.d_Col[private_s.point_no] - 1;

		//work
		for(ei_new = 0; ei_new < public_s.tMask_elem; ei_new++){
			private_s.d_tMask[ei_new] = 0;
		}
		private_s.d_tMask[tMask_col*public_s.tMask_rows + tMask_row] = 1;


		//====================================================================================================
		//	1) MASK CONVOLUTION
		//	2) MULTIPLICATION
		//====================================================================================================

		// work
		// for(col=1; col<=public_s.conv_cols; col++){
		for(col=1; col<=public_s.mask_conv_cols; col++){

			// col setup
			j = col + public_s.mask_conv_joffset;
			jp1 = j + 1;
			if(public_s.mask_cols < jp1){
				ja1 = jp1 - public_s.mask_cols;
			}
			else{
				ja1 = 1;
			}
			if(public_s.tMask_cols < j){
				ja2 = public_s.tMask_cols;
			}
			else{
				ja2 = j;
			}

			// for(row=1; row<=public_s.conv_rows; row++){
			for(row=1; row<=public_s.mask_conv_rows; row++){

				// row setup
				i = row + public_s.mask_conv_ioffset;
				ip1 = i + 1;
				
				if(public_s.mask_rows < ip1){
					ia1 = ip1 - public_s.mask_rows;
				}
				else{
					ia1 = 1;
				}
				if(public_s.tMask_rows < i){
					ia2 = public_s.tMask_rows;
				}
				else{
					ia2 = i;
				}

				s = 0;

				// get data
				for(ja=ja1; ja<=ja2; ja++){
					jb = jp1 - ja;
					for(ia=ia1; ia<=ia2; ia++){
						ib = ip1 - ia;
						s = s + private_s.d_tMask[public_s.tMask_rows*(ja-1)+ia-1] * 1;
					}
				}

				private_s.d_mask_conv[(col-1)*public_s.conv_rows+(row-1)] = private_s.d_conv[(col-1)*public_s.conv_rows+(row-1)] * s;

			}

		}

		//====================================================================================================
		//	MAXIMUM VALUE
		//====================================================================================================

		//==================================================
		//	SEARCH
		//==================================================

		fin_max_val = 0;
		fin_max_coo = 0;
		for(i=0; i<public_s.mask_conv_elem; i++){
			if(private_s.d_mask_conv[i]>fin_max_val){
				fin_max_val = private_s.d_mask_conv[i];
				fin_max_coo = i;
			}
		}

		//==================================================
		//	OFFSET
		//==================================================

		// convert coordinate to row/col form
		largest_row = (fin_max_coo+1) % public_s.mask_conv_rows - 1;											// (0-n) row
		largest_col = (fin_max_coo+1) / public_s.mask_conv_rows;												// (0-n) column
		if((fin_max_coo+1) % public_s.mask_conv_rows == 0){
			largest_row = public_s.mask_conv_rows - 1;
			largest_col = largest_col - 1;
		}

		// calculate offset
		largest_row = largest_row + 1;																	// compensate to match MATLAB format (1-n)
		largest_col = largest_col + 1;																	// compensate to match MATLAB format (1-n)
		offset_row = largest_row - public_s.in_mod_rows - (public_s.sSize - public_s.tSize);
		offset_col = largest_col - public_s.in_mod_cols - (public_s.sSize - public_s.tSize);
		pointer = private_s.point_no*public_s.frames+public_s.frame_no;
		private_s.d_tRowLoc[pointer] = private_s.d_Row[private_s.point_no] + offset_row;
		private_s.d_tColLoc[pointer] = private_s.d_Col[private_s.point_no] + offset_col;

	}

	//======================================================================================================================================================
	//	COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================

	// if the last frame in the bath, update template
	if(public_s.frame_no != 0 && (public_s.frame_no)%10 == 0){

		// update coordinate
		loc_pointer = private_s.point_no*public_s.frames+public_s.frame_no;
		private_s.d_Row[private_s.point_no] = private_s.d_tRowLoc[loc_pointer];
		private_s.d_Col[private_s.point_no] = private_s.d_tColLoc[loc_pointer];

		// update template, limit the number of working threads to the size of template
		for(col=0; col<public_s.in_mod_cols; col++){
			for(row=0; row<public_s.in_mod_rows; row++){

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = private_s.d_Row[private_s.point_no] - 25 + row - 1;
			ori_col = private_s.d_Col[private_s.point_no] - 25 + col - 1;
			ori_pointer = ori_col*public_s.frame_rows+ori_row;

			// update template
			d_in[col*public_s.in_mod_rows+row] = public_s.alpha*d_in[col*public_s.in_mod_rows+row] + (1.00-public_s.alpha)*public_s.d_frame[ori_pointer];

			}
		}

	}

}

	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	END OF FUNCTION
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
