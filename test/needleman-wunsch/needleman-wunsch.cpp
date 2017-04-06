/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include "hclib_cpp.h"

#define GAP_PENALTY -1
#define TRANSITION_PENALTY -2
#define TRANSVERSION_PENALTY -4
#define MATCH 2

enum Nucleotide {GAP=0, ADENINE, CYTOSINE, GUANINE, THYMINE};

signed char char_mapping ( char c ) {
    signed char to_be_returned = -1;
    switch(c) {
        case '_': to_be_returned = GAP; break;
        case 'A': to_be_returned = ADENINE; break;
        case 'C': to_be_returned = CYTOSINE; break;
        case 'G': to_be_returned = GUANINE; break;
        case 'T': to_be_returned = THYMINE; break;
    }
    return to_be_returned;
}

void print_matrix ( int** matrix, int n_rows, int n_columns ) {
    int i, j;
    for ( i = 0; i < n_rows; ++i ) {
        for ( j = 0; j < n_columns; ++j ) {
            fprintf(stdout, "%d ", matrix[i][j]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout,"--------------------------------\n");
}

static char alignment_score_matrix[5][5] =
{
    {GAP_PENALTY,GAP_PENALTY,GAP_PENALTY,GAP_PENALTY,GAP_PENALTY},
    {GAP_PENALTY,MATCH,TRANSVERSION_PENALTY,TRANSITION_PENALTY,TRANSVERSION_PENALTY},
    {GAP_PENALTY,TRANSVERSION_PENALTY, MATCH,TRANSVERSION_PENALTY,TRANSITION_PENALTY},
    {GAP_PENALTY,TRANSITION_PENALTY,TRANSVERSION_PENALTY, MATCH,TRANSVERSION_PENALTY},
    {GAP_PENALTY,TRANSVERSION_PENALTY,TRANSITION_PENALTY,TRANSVERSION_PENALTY, MATCH}
};

size_t clear_whitespaces_do_mapping ( signed char* buffer, long lsize ) {
    size_t non_ws_index = 0, traverse_index = 0;

    while ( traverse_index < lsize ) {
        char curr_char = buffer[traverse_index];
        switch ( curr_char ) {
            case 'A': case 'C': case 'G': case 'T':
                /*this used to be a copy not also does mapping*/
                buffer[non_ws_index++] = char_mapping(curr_char);
                break;
        }
        ++traverse_index;
    }
    return non_ws_index;
}

signed char* read_file( FILE* file, size_t* n_chars ) {
    fseek (file, 0L, SEEK_END);
    long file_size = ftell (file);
    fseek (file, 0L, SEEK_SET);

    signed char *file_buffer = (signed char *)malloc((1+file_size)*sizeof(signed char));

    size_t n_read_from_file = fread(file_buffer, sizeof(signed char), file_size, file);
    file_buffer[file_size] = '\n';

    /* shams' sample inputs have newlines in them */
    *n_chars = clear_whitespaces_do_mapping(file_buffer, file_size);
    return file_buffer;
}

typedef struct {
    hclib::promise_t<int*>* bottom_row;
    hclib::promise_t<int*>* right_column;
    hclib::promise_t<int*>* bottom_right;
} Tile_t;


int main ( int argc, char* argv[] ) {
    hclib::launch([&]() {
        int i, j;

        int tile_width = (int) atoi (argv[3]);
        int tile_height = (int) atoi (argv[4]);

        int n_tiles_width;
        int n_tiles_height;

        if ( argc < 5 ) {
            fprintf(stderr, "Usage: %s fileName1 fileName2 tileWidth tileHeight\n", argv[0]);
            exit(1);
        }

        signed char* string_1;
        signed char* string_2;

        char* file_name_1 = argv[1];
        char* file_name_2 = argv[2];

        FILE* file_1 = fopen(file_name_1, "r");
        if (!file_1) { fprintf(stderr, "could not open file %s\n",file_name_1); exit(1); }
        size_t n_char_in_file_1 = 0;
        string_1 = read_file(file_1, &n_char_in_file_1);
        fprintf(stdout, "Size of input string 1 is %lu\n", n_char_in_file_1 );

        FILE* file_2 = fopen(file_name_2, "r");
        if (!file_2) { fprintf(stderr, "could not open file %s\n",file_name_2); exit(1); }
        size_t n_char_in_file_2 = 0;
        string_2 = read_file(file_2, &n_char_in_file_2);
        fprintf(stdout, "Size of input string 2 is %lu\n", n_char_in_file_2 );

        fprintf(stdout, "Tile width is %d\n", tile_width);
        fprintf(stdout, "Tile height is %d\n", tile_height);

        n_tiles_width = n_char_in_file_1/tile_width;
        n_tiles_height = n_char_in_file_2/tile_height;

        fprintf(stdout, "Imported %d x %d tiles.\n", n_tiles_width, n_tiles_height);

        fprintf(stdout, "Allocating tile matrix\n");

        // sagnak: all workers allocate their own copy of tile matrix
        Tile_t** tile_matrix = (Tile_t **) malloc(sizeof(Tile_t*)*(n_tiles_height+1)); 
        for ( i = 0; i < n_tiles_height+1; ++i ) {
            tile_matrix[i] = (Tile_t *) malloc(sizeof(Tile_t)*(n_tiles_width+1));
            for ( j = 0; j < n_tiles_width+1; ++j ) {
                tile_matrix[i][j].bottom_row = new hclib::promise_t<int*>();
                tile_matrix[i][j].right_column = new hclib::promise_t<int*>();
                tile_matrix[i][j].bottom_right = new hclib::promise_t<int*>();
            }
        }

        fprintf(stdout, "Allocated tile matrix\n");

        int* allocated = (int*)malloc(sizeof(int));
        allocated[0] = 0;
        tile_matrix[0][0].bottom_right->put(allocated);

        for ( j = 1; j < n_tiles_width + 1; ++j ) {
            allocated = (int*)malloc(sizeof(int)*tile_width);
            for( i = 0; i < tile_width ; ++i ) {
                allocated[i] = GAP_PENALTY*((j-1)*tile_width+i+1);
            }
            tile_matrix[0][j].bottom_row->put(allocated);

            allocated = (int*)malloc(sizeof(int));
            allocated[0] = GAP_PENALTY*(j*tile_width); //sagnak: needed to handle tilesize 2
            tile_matrix[0][j].bottom_right->put(allocated);
        }

        for ( i = 1; i < n_tiles_height + 1; ++i ) {
            allocated = (int*)malloc(sizeof(int)*tile_height);
            for ( j = 0; j < tile_height ; ++j ) {
                allocated[j] = GAP_PENALTY*((i-1)*tile_height+j+1);
            }
            tile_matrix[i][0].right_column->put(allocated);

            allocated = (int*)malloc(sizeof(int));
            allocated[0] = GAP_PENALTY*(i*tile_height); //sagnak: needed to handle tilesize 2
            tile_matrix[i][0].bottom_right->put(allocated);
        }


        struct timeval begin,end;
        gettimeofday(&begin,0);

        HCLIB_FINISH {
            for (int i = 1; i < n_tiles_height+1; ++i ) {
                for (int j = 1; j < n_tiles_width+1; ++j ) {
                hclib::async_await([=] {
                        int index, ii, jj;
                        int* above_tile_bottom_row = (int *)tile_matrix[i-1][j  ].bottom_row->get_future()->get();
                        int* left_tile_right_column = (int *)tile_matrix[  i][j-1].right_column->get_future()->get(); 
                        int* diagonal_tile_bottom_right = (int *)tile_matrix[i-1][j-1].bottom_right->get_future()->get();

                        int  * curr_tile_tmp = (int*) malloc(sizeof(int)*(1+tile_width)*(1+tile_height));
                        int ** curr_tile = (int**) malloc(sizeof(int*)*(1+tile_height));
                        for (index = 0; index < tile_height+1; ++index) {
                            curr_tile[index] = &curr_tile_tmp[index*(1+tile_width)];
                        }

                        curr_tile[0][0] = diagonal_tile_bottom_right[0];
                        for ( index = 1; index < tile_height+1; ++index ) {
                            curr_tile[index][0] = left_tile_right_column[index-1];
                        }

                        for ( index = 1; index < tile_width+1; ++index ) {
                            curr_tile[0][index] = above_tile_bottom_row[index-1];
                        }

                        for ( ii = 1; ii < tile_height+1; ++ii ) {
                            for ( jj = 1; jj < tile_width+1; ++jj ) {
                                signed char char_from_1 = string_1[(j-1)*tile_width+(jj-1)];
                                signed char char_from_2 = string_2[(i-1)*tile_height+(ii-1)];

                                int diag_score = curr_tile[ii-1][jj-1] + alignment_score_matrix[char_from_2][char_from_1];
                                int left_score = curr_tile[ii  ][jj-1] + alignment_score_matrix[char_from_1][GAP];
                                int  top_score = curr_tile[ii-1][jj  ] + alignment_score_matrix[GAP][char_from_2];

                                int bigger_of_left_top = (left_score > top_score) ? left_score : top_score;
                                curr_tile[ii][jj] = (bigger_of_left_top > diag_score) ? bigger_of_left_top : diag_score;
                            }
                        }

                        int* curr_bottom_right = (int*)malloc(sizeof(int));
                        curr_bottom_right[0] = curr_tile[tile_height][tile_width];
                        tile_matrix[i][j].bottom_right->put(curr_bottom_right);

                        int* curr_right_column = (int*)malloc(sizeof(int)*tile_height);
                        for ( index = 0; index < tile_height; ++index ) {
                            curr_right_column[index] = curr_tile[index+1][tile_width];
                        }
                        tile_matrix[i][j].right_column->put(curr_right_column);

                        int* curr_bottom_row = (int*)malloc(sizeof(int)*tile_width);
                        for ( index = 0; index < tile_width; ++index ) {
                            curr_bottom_row[index] = curr_tile[tile_height][index+1];
                        }
                        tile_matrix[i][j].bottom_row->put(curr_bottom_row);

                        free(curr_tile);
                        free(curr_tile_tmp);
                    }, tile_matrix[i][j-1].right_column->get_future(),
                    tile_matrix[i-1][j].bottom_row->get_future(),
                    tile_matrix[i-1][j-1].bottom_right->get_future());
                }
            }
        }

        gettimeofday(&end,0);
        fprintf(stdout, "The computation took %f seconds\n",((end.tv_sec - begin.tv_sec)*1000000+(end.tv_usec - begin.tv_usec))*1.0/1000000);

        int score = ((int *)(tile_matrix[n_tiles_height][n_tiles_width].bottom_row->get_future()->get()))[tile_width-1];
        fprintf(stdout, "score: %d\n", score);
    });

    return 0;
}
