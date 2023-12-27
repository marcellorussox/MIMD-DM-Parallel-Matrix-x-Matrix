#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>
#include "../headers/utilities.h"
#include "../headers/data_structures.h"


double* initialize_matrix(int matrix_order) {
	
	double* matrix = (double*) calloc((matrix_order * matrix_order), sizeof(double));
	return matrix;
	
}

double* generate_random_matrix(int matrix_order) {
	srand(time(NULL));
	double* matrix = initialize_matrix(matrix_order);
	if(matrix) {
		for(int i = 0; i < matrix_order; i++)
			for(int j = 0; j < matrix_order; j++)
				matrix[i*matrix_order+j] = (double)rand()/(double)RAND_MAX;
	}
    return matrix;
	
}


void print_matrix(double* matrix, int matrix_order) {
	
	for(int i = 0; i < matrix_order; i++) {
    	for(int j = 0; j < matrix_order; j++)
    		printf(" [%lf]", matrix[i*matrix_order+j]);
    	printf("\n");
	}
	
}


void create_grid(
	MPI_Comm* grid, MPI_Comm* sub_rgrid, MPI_Comm* sub_cgrid,
	int mpi_rank, int mpi_size, int grid_order, int* periods, int reorder, int* coords) {
	
	int dimensions[2] = {grid_order, grid_order};
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, reorder, grid);
	MPI_Cart_coords(*grid, mpi_rank, 2, coords);
	int remains[2] = {0, 1};
	MPI_Cart_sub(*grid, remains, sub_rgrid);
	remains[0] = 1;
	remains[1] = 0;
	MPI_Cart_sub(*grid, remains, sub_cgrid);

}


void send_matrix_from_processor_0(
	double* matrix, int matrix_order, int sub_matrix_order, MPI_Comm grid, int mpi_size) {
	
	int coords[2];
	int start_row;
	int start_column;
	
	for(int processor = 1; processor < mpi_size; processor++) {
		MPI_Cart_coords(grid, processor, 2, coords);
		start_row = coords[0] * sub_matrix_order;
		start_column = coords[1] * sub_matrix_order;
		for(int row_offset = 0; row_offset < sub_matrix_order; row_offset++)
			MPI_Send(
				&matrix[(start_row+row_offset)*matrix_order+start_column], 
				sub_matrix_order, MPI_DOUBLE, processor, D_TAG + processor, MPI_COMM_WORLD
			);
	}

}


void receive_matrix_from_processor_0(
	double* sub_matrix, int sub_matrix_order, int mpi_rank) {

	MPI_Status status;
	for(int row = 0; row < sub_matrix_order; row++){
		MPI_Recv(
			&sub_matrix[row*sub_matrix_order], sub_matrix_order, MPI_DOUBLE, 
			0, D_TAG + mpi_rank, MPI_COMM_WORLD, &status
		);
	}

}


void bmr(
	double* sub_matrix_a, double* sub_matrix_b, double* sub_matrix_c, int sub_matrix_order,
	MPI_Comm grid, MPI_Comm sub_rgrid, MPI_Comm sub_cgrid, int grid_order, int coords[2]) {	

	MPI_Request request;
	MPI_Status status;

	int sub_matrix_a_broadcaster_coords[2];
	int sub_matrix_a_broadcaster_rank;
	
	/* Matrice di appoggio per il broadcasting */
	
	double* tmp_matrix = initialize_matrix(sub_matrix_order);
			
	/* Calcolo rank sotto griglia colonna del receiver della sotto matrice B */	
	
	int sub_matrix_b_receiver_rank;
	int sub_matrix_b_receiver_coords[2] = {
		(coords[0] + grid_order - 1) % grid_order,
		coords[1]
	};
	MPI_Cart_rank(sub_cgrid, sub_matrix_b_receiver_coords, &sub_matrix_b_receiver_rank);
		
	/* Calcolo rank sotto griglia colonna del sender della sotto matrice B */	
	
	int sub_matrix_b_sender_rank;
	int sub_matrix_b_sender_coords[2] = {
		(coords[0] + 1) % grid_order,
		sub_matrix_b_sender_coords[1] = coords[1]
	};
    MPI_Cart_rank(sub_cgrid, sub_matrix_b_sender_coords, &sub_matrix_b_sender_rank);
    
    /* Calcolo rank nella sotto griglia colonna del processore chiamante */
    
    int rank_cgrid;
    MPI_Cart_rank(sub_cgrid, coords, &rank_cgrid);
	
	/* Inizio BMR */
	
	for(int step = 0; step < grid_order; step++) {
		
		/* Coordinate del processore che deve inviare la sotto matrice A */
		
		sub_matrix_a_broadcaster_coords[0] = coords[0];
        sub_matrix_a_broadcaster_coords[1] = (coords[0] + step) % grid_order;
		
		
		if(!step) {		// Primo passo
			
			/* Broadcasting */
			
			if(coords[0] == coords[1]){
                sub_matrix_a_broadcaster_coords[1] = coords[1];
                memcpy(tmp_matrix, sub_matrix_a, sub_matrix_order*sub_matrix_order*sizeof(double));
            }
			
			MPI_Cart_rank(sub_rgrid, sub_matrix_a_broadcaster_coords, &sub_matrix_a_broadcaster_rank);
			
            MPI_Bcast(tmp_matrix, sub_matrix_order*sub_matrix_order, 
				MPI_DOUBLE, sub_matrix_a_broadcaster_rank, sub_rgrid
			);
			
			/* Multiply */

            multiply(tmp_matrix, sub_matrix_b, sub_matrix_c, sub_matrix_order);
			
		} else {	// Passi successivi
			
			/* Broadcasting sulle diagonale superiori alla principale (k + step) */
			
            if(coords[1] == sub_matrix_a_broadcaster_coords[1]){
            	memcpy(tmp_matrix, sub_matrix_a, sub_matrix_order*sub_matrix_order*sizeof(double));
            }
            sub_matrix_a_broadcaster_rank = (sub_matrix_a_broadcaster_rank+1)%grid_order;

            MPI_Bcast(tmp_matrix, sub_matrix_order*sub_matrix_order, 
				MPI_DOUBLE, sub_matrix_a_broadcaster_rank, sub_rgrid
			);

            /* Rolling */
            
            MPI_Isend(sub_matrix_b, sub_matrix_order*sub_matrix_order, 
				MPI_DOUBLE, sub_matrix_b_receiver_rank, 
				D_TAG + sub_matrix_b_receiver_rank, sub_cgrid, &request
			);
			
            MPI_Recv(sub_matrix_b, sub_matrix_order*sub_matrix_order, 
				MPI_DOUBLE, sub_matrix_b_sender_rank, D_TAG + rank_cgrid, sub_cgrid, &status
			);

            /* Multiply */
            
            multiply(tmp_matrix, sub_matrix_b, sub_matrix_c, sub_matrix_order);
			
		}
		
	}
	
}


void merge(
	double* matrix_c, int matrix_order, double* sub_matrix_c, 
	int sub_matrix_order, MPI_Comm grid, int mpi_size, int mpi_rank) {
	
	MPI_Status status;
	int start_row;
	int start_columns;
	int coords[2];

	if(!mpi_rank) {  // Se il chiamante è il processore 0 allora si riceve
		
		/* Si scorre su tutti i processori */
		
		for(int processor = 0; processor < mpi_size; processor++) {
			
			MPI_Cart_coords(grid, processor, 2, coords);
			start_row = coords[0] * sub_matrix_order;
			start_columns = coords[1] * sub_matrix_order;
			
			if(processor) {  // Se non è il processore 0 allora si riceve
				for(int row_offset = 0; row_offset < sub_matrix_order; row_offset++) {
					MPI_Recv(
						&sub_matrix_c[row_offset*sub_matrix_order], sub_matrix_order,
						MPI_DOUBLE, processor, D_TAG + processor, MPI_COMM_WORLD, &status
					);		
					memcpy(
						&matrix_c[(start_row+row_offset)*matrix_order+start_columns], 
						&sub_matrix_c[row_offset*sub_matrix_order],
						sub_matrix_order*sizeof(double)
					);
				}
			} else {  // Altrimenti si copia la propria sotto matrice C nella matrice C finale
				for(int row_offset = 0; row_offset < sub_matrix_order; row_offset++)
					memcpy(
						&matrix_c[row_offset*matrix_order], 
						&sub_matrix_c[row_offset*sub_matrix_order],
						sub_matrix_order*sizeof(double)
					);
			}
			
		}
		
	} else {   // Altrimenti si invia al processore 0
		
		for(int row_offset = 0; row_offset < sub_matrix_order; row_offset++)
			MPI_Send(
				&sub_matrix_c[row_offset*sub_matrix_order], sub_matrix_order,
				MPI_DOUBLE, 0, D_TAG + mpi_rank, MPI_COMM_WORLD
			);
			
	}

}


void multiply(double* matrix_a, double* matrix_b, double* matrix_c, int matrix_order) {
	
	for(int i = 0; i < matrix_order; i++)
		for(int j = 0; j < matrix_order; j++)
			for(int k = 0; k < matrix_order; k++)
				matrix_c[i*matrix_order+j] += (matrix_a[i*matrix_order+k] * matrix_b[k*matrix_order+j]);

}
