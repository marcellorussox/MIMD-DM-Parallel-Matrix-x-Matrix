#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>
#include "headers/utilities.h"
#include "headers/data_structures.h"

int main(int argc, char** argv) {

	int mpi_rank;                 // ID MPI del processore corrente;
	int mpi_size;                 // Numero totale di processori nel communicator;

	double start_time;            // Tempo di inizio del calcolo;
	double end_time;              // Tempo di fine del calcolo;
	double delta_time;            // Differenza temporale tra inizio e fine del calcolo;
	double max_time;              // Tempo massimo impiegato per il calcolo;

	int matrix_order;             // Dimensione delle matrici quadrate;
	double* matrix_a;             // Matrice A (sinistra) utilizzata nella moltiplicazione;
	double* matrix_b;             // Matrice B (destra) utilizzata nella moltiplicazione;
	double* matrix_c;             // Matrice risultante dalla moltiplicazione;

	int sub_matrix_order;         // Dimensione delle sotto-matrici quadrate (sotto-problema);
	double* sub_matrix_a;         // Sotto-matrice A (sinistra) utilizzata nella moltiplicazione (sotto-problema);
	double* sub_matrix_b;         // Sotto-matrice B (destra) utilizzata nella moltiplicazione (sotto-problema);
	double* sub_matrix_c;         // Sotto-matrice risultante dalla moltiplicazione (sotto-problema);

	int periods[2] = {0};         // Array delle periodicità della griglia di processori;
	int coords[2];                // Coordinate del processore nella griglia;

	int grid_order;               // Dimensione della griglia quadrata di processori;
	MPI_Comm grid;                // Griglia di processori;
	MPI_Comm sub_rgrid;           // Sotto-griglia di riga;
	MPI_Comm sub_cgrid;           // Sotto-griglia di colonna;

	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	
	/* Controllo degli argomenti */
	
	if(!mpi_rank) {	
		check_params(argc, argv, mpi_size);	
	}
	
	
	/* Lettura e distribuzione degli argomenti passati in ingresso */

	if(!mpi_rank)
		matrix_order = atoi(argv[2]);

	MPI_Bcast(&matrix_order, 1, MPI_INT, 0, MPI_COMM_WORLD);
	grid_order = (int)sqrt(mpi_size);
	sub_matrix_order = matrix_order/grid_order;
	
	
	/* Generazione pseudo-randomica delle matrici da moltiplicare */
	
	if(!mpi_rank) {
		matrix_a = generate_random_matrix(matrix_order);
		matrix_b = generate_random_matrix(matrix_order);
		if(!matrix_a || !matrix_b) {
			printf("\n <!> ERROR: Unable to allocate memory.\n");
			MPI_Abort(MPI_COMM_WORLD, ERR_MEMORY);
		}
	}
	
	/* Stampa della matrici generate (solo se con ordine inferiore a 10) */
		
	if(!mpi_rank && matrix_order <= 10) {
		printf("\n > Generated Matrix A \n\n");
		print_matrix(matrix_a, matrix_order);
		printf("\n\n > Generated Matrix B \n\n");
		print_matrix(matrix_b, matrix_order);
	}

	
	
	/* Se singolo processore allora effettua il prodotto sequenziale, altrimenti parallelo */
	
	if (mpi_size != 1) {
		/* Creazione della griglia e delle sotto-griglie riga e colonna */

		create_grid(
			&grid, &sub_rgrid, &sub_cgrid, mpi_rank, mpi_size, grid_order, periods, 0, coords
		);
		
		
		/* 	Allocazione memoria */
		
		sub_matrix_a = initialize_matrix(sub_matrix_order);
		sub_matrix_b = initialize_matrix(sub_matrix_order);
		sub_matrix_c = initialize_matrix(sub_matrix_order);


		if(!mpi_rank && (!sub_matrix_a || !sub_matrix_b || !sub_matrix_c)) {
			printf("\n <!> ERROR: Unable to allocate memory.\n");
			MPI_Abort(MPI_COMM_WORLD, ERR_MEMORY);
		}
		if(!mpi_rank) {
			matrix_c = initialize_matrix(matrix_order);
			if(!matrix_c) {
				printf("\n <!> ERROR: Unable to allocate memory.\n");
				MPI_Abort(MPI_COMM_WORLD, ERR_MEMORY);
			}
		}
		
		
		/* Distribuzione delle matrici */
		
		if(!mpi_rank) {
			send_matrix_from_processor_0(matrix_a, matrix_order, sub_matrix_order, grid, mpi_size);
			send_matrix_from_processor_0(matrix_b, matrix_order, sub_matrix_order, grid, mpi_size);
			for(int i = 0; i < sub_matrix_order; i++)
				for(int j = 0; j < sub_matrix_order; j++) {
					sub_matrix_a[i*sub_matrix_order+j] = matrix_a[i*matrix_order+j];
					sub_matrix_b[i*sub_matrix_order+j] = matrix_b[i*matrix_order+j];
				}
		} else {
			receive_matrix_from_processor_0(sub_matrix_a, sub_matrix_order, mpi_rank);
			receive_matrix_from_processor_0(sub_matrix_b, sub_matrix_order, mpi_rank);
			fileLog("Sono il proc %d e sono arrivato qua\n",mpi_rank);
		}
		
		/* Sincronizzazione dei processori e salvataggio timestamp di inizio */
		
		MPI_Barrier(MPI_COMM_WORLD);
		start_time = MPI_Wtime();
		
		
		/* Applicazione della strategia di comunicazione BMR */
		
		bmr(
			sub_matrix_a, sub_matrix_b, sub_matrix_c, sub_matrix_order, 
			grid, sub_rgrid, sub_cgrid, grid_order, coords
		); 
		
 
		/* Composizione del risultato finale */
		
		merge(matrix_c, matrix_order, sub_matrix_c, sub_matrix_order, grid, mpi_size, mpi_rank);
		
		
		/* Calcolo del tempo impiegato */
		
		end_time = MPI_Wtime();
		delta_time = end_time - start_time;
    	MPI_Reduce(&delta_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		
	} else {
		
		/* Salvataggio timestamp di inizio */
		
		start_time = MPI_Wtime();
		
		
		/* Calcolo del prodotto */
		
		matrix_c = initialize_matrix(matrix_order);
		if(!matrix_c) {
			printf("\n <!> ERROR: Unable to allocate memory.\n");
			MPI_Abort(MPI_COMM_WORLD, ERR_MEMORY);
		}
		multiply(matrix_a, matrix_b, matrix_c, matrix_order);

		
		/* Calcolo del tempo impiegato */
		
		end_time = MPI_Wtime();
		max_time = end_time - start_time;
		
	}
	
	
	/* Stampa della matrice calcolata (solo se con ordine inferiore a 10) e del tempo impiegato */
	
	if(!mpi_rank) {
		if(matrix_order <= 10) {
			printf("\n\n > Product Matrix C \n\n");
			print_matrix(matrix_c, matrix_order);
		}
		printf(" Overall time: %lf\n", max_time);
	}
	
	MPI_Finalize();
	return 0;
	
}