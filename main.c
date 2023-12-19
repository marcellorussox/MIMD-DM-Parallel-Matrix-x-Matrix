#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>


#define SD_ARG_ORDER      "-o"

#define DD_ARG_ORDER      "--order"
#define DD_ARG_HELP       "--help"

#define SCC               200
#define SCC_ARGS          201
#define SCC_HELP          202

#define ERR_ARGC          400
#define ERR_NO_ORDER      401
#define ERR_ORDER         402
#define ERR_PROC          403
#define ERR_MEMORY        404

#define D_TAG              22


void help(char* program_name);
void print_matrix(double*, int);
void create_grid(MPI_Comm*, MPI_Comm*, MPI_Comm*,int, int, int, int*, int, int*);
void send_matrix_from_processor_0(double*, int, int, MPI_Comm, int);
void receive_matrix_from_processor_0(double*, int, int);
void merge(double*, int, double*, int, MPI_Comm, int, int);
void multiply(double*, double*, double*, int);
void bmr(double*, double*, double*, int, MPI_Comm, MPI_Comm, MPI_Comm, int, int[2]);

int check_args(int, char**, int);

double* initialize_matrix(int);
double* generate_random_matrix(int);


int main(int argc, char** argv) {
	
	/*		
		args_error: Codice errore ottenuto dalla verifica degli argomenti;
		
		mpi_rank: Identificativo MPI del processore;
		mpi_size: Numero di processori del communicator;
		
		start_time: Tempo inizio somma;
		end_time: Tempo fine somma;
		delta_time: Differenza temporale tra inizio e fine somma;
		max_time: Tempo di somma massimo;
		
		matrix_order: Ordine delle matrici quadrate;
		matrix_a: Matrice A (sinistra) da impiegare nel prodotto;
		matrix_b: Matrice B (destra) da impiegare nel prodotto;
		matrix_c: Matrice risultate dal prodotto;
		
		sub_matrix_order: Ordine delle sotto matrici quadrate (sotto-problema);
		sub_matrix_a: Sotto matrice A (sinistra) da impiegare nel prodotto (sotto-problema);
		sub_matrix_b: Sotto matrice B (destra) da impiegare nel prodotto (sotto-problema);
		sub_matrix_c: Sotto matrice risultate dal prodotto (sotto-problema);
		
		periods: Array delle perioditcit� della griglia di processori
		coords: Coordinate del processore nella griglia
		
		grid_order: Ordine della griglia quadrata di processori;
		grid: Griglia di processori;
		sub_rgrid: Sotto griglia di riga;
		sub_cgrid: Sotto griglia di colonna;
	*/
	
	int args_error;
	
	int mpi_rank;
	int mpi_size;
	
	double start_time;
	double end_time;
	double delta_time;
	double max_time;

	int matrix_order;
	double* matrix_a;
	double* matrix_b;
	double* matrix_c;
	
	int sub_matrix_order;
	double* sub_matrix_a;
	double* sub_matrix_b;
	double* sub_matrix_c;
	
	int periods[2] = {0};
	int coords[2];
	
	int grid_order;
	MPI_Comm grid;
	MPI_Comm sub_rgrid;
	MPI_Comm sub_cgrid;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	
	/* Controllo degli argomenti */
	
	if(!mpi_rank) {
		
		args_error = check_args(argc, argv, mpi_size);
		
		switch(args_error) {
			
			case SCC_HELP:
				help(argv[0]);
				break;
			
			case ERR_ARGC:
				printf(
					"\n <!> ERROR: Invalid number of arguments! For additional info type %s.\n",
					DD_ARG_HELP
				);
				break;
				
			case ERR_NO_ORDER:
				printf(
					"\n <!> ERROR: Expected [%s %s] argument! For additional info type %s.\n",
					SD_ARG_ORDER, DD_ARG_ORDER, DD_ARG_HELP
				);
				break;
				
			case ERR_ORDER:
				printf(
					"\n <!> ERROR: Invalid value for argument [%s %s]! For additional info type %s.\n",
					SD_ARG_ORDER, DD_ARG_ORDER, DD_ARG_HELP
				);
				break;
				
			case ERR_PROC:
				printf(
					"\n <!> ERROR: Invalid number of processors used. For additional info type %s.\n",
					DD_ARG_HELP
				);
				break;
			
		}
		
		if(args_error != SCC_ARGS && args_error != SCC_HELP)
			MPI_Abort(MPI_COMM_WORLD, args_error); 
		
	}
	
	
	/* Propagazione del codice SCC_HELP */

	if(mpi_size != 1)
		MPI_Bcast(&args_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(args_error == SCC_HELP) {
		MPI_Finalize();
		return 0;
	}
	
	
	/* Lettura e distribuzione degli argomenti passati in ingresso */

	if(!mpi_rank)
		matrix_order = atoi(argv[2]);
	MPI_Bcast(&matrix_order, 1, MPI_INT, 0, MPI_COMM_WORLD);
	grid_order = (int)sqrt(mpi_size);
	sub_matrix_order = matrix_order/grid_order;
	
	
	/* Generazione pseudo-randomica delle matrici da moltiplicare */
	
	if(!mpi_rank) {
		srand(time(NULL));
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


/*

	Stampa a video l'help del programma
	
	@params:
		char* program_name: Nome del programma
	
	@return: 
		void
	
*/

void help(char* program_name) {
	
	printf(
		"\n > Usage: %s [%s %s] <value>",
		program_name,
		SD_ARG_ORDER, DD_ARG_ORDER
	);

	printf("\n\n\tMandatory arguments:");
	printf(
		"\n\t   %s  %-20s Order of the square matrix", 
		SD_ARG_ORDER, DD_ARG_ORDER
	);

	printf("\n\n\tError codes:");
	printf("\n\t   %d %-20s Invalid number of arguments", ERR_ARGC, "ERR_ARGC");
	printf(
		"\n\t   %d %-20s Mandatory argument [%s %s] not provided",
		ERR_NO_ORDER, "ERR_NO_ORDER",
		SD_ARG_ORDER, DD_ARG_ORDER
	);
	printf(
		"\n\t   %d %-20s Invalid order of the square matrix provided",
		ERR_ORDER, "ERR_ORDER"
	);
	printf(
		"\n\t   %d %-20s Invalid number of processors", 
		ERR_PROC, "ERR_PROC"
	);
	printf(
		"\n\t   %d %-20s Unable to allocate memory", 
		ERR_MEMORY, "ERR_MEMORY"
	);
	
	printf("\n\n\tAdditional Info:");
	printf(
		"\n\t   The argument [%s %s] must be a multiple of the number of processors used", 
		SD_ARG_ORDER, DD_ARG_ORDER
	);
	printf(
		"\n\t   The number of processors used must be a perfect square"
	);
	printf(
		"\n\t   Communication strategy used: BMR\n"
	);
}


/*

	Verifica l'integrita' degli argomenti passati in ingresso al programma
	
	@params:
		int argc: Numero di argomenti passati in ingresso al programma
		char* argv[]: Argomenti passati in ingresso al programma
		int mpi_size: Numero di processori impiegati
	
	@return: 
		int: Codice errore/successo
	
*/

int check_args(int argc, char** argv, int mpi_size) {
	
	if(argc == 2 && !strcmp(argv[1], DD_ARG_HELP))
		return SCC_HELP;
	
	if(argc == 3) {
		
		if(sqrt(mpi_size) != (int)sqrt(mpi_size))
			return ERR_PROC;
		
		if(strcmp(argv[1], SD_ARG_ORDER) && strcmp(argv[1], DD_ARG_ORDER))
			return ERR_NO_ORDER;
			
		int matrix_order = atoi(argv[2]);
		
		if(matrix_order <= 0 || (matrix_order % (int)sqrt(mpi_size)))
			return ERR_ORDER;
		
		return SCC_ARGS; 
		
	}

	return ERR_ARGC;
	
}


/*

	Inizializza una matrice quadrata dinamica di ordine  matrix_order
	
	@params:
		int matrix_order: Ordine della matrice quadrata
		
	@return: 
		double*: Matrice quadrata inizializzata
	
*/

double* initialize_matrix(int matrix_order) {
	
	double* matrix = (double*) calloc((matrix_order * matrix_order), sizeof(double));
	return matrix;
	
}


/*

	Genera una matrice quadrata pseudo-randomica di ordine matrix_order
	
	@params:
		int matrix_order: Ordine della matrice quadrata
		double lower: Limite inferiore del valore degli elementi
		double upper: Limite superiore del valore degli elementi
	
	@return: 
		double*: Matrice quadrata generata
	
*/

double* generate_random_matrix(int matrix_order) {
	
	double* matrix = initialize_matrix(matrix_order);
	if(matrix) {
		for(int i = 0; i < matrix_order; i++)
			for(int j = 0; j < matrix_order; j++)
				//matrix[i*matrix_order+j] = (double)rand()/(double)RAND_MAX;
				matrix[i*matrix_order+j] = 1;
	}
    return matrix;
	
}


/*

	Effettua la stampa della matrice passata in ingresso
	
	@params:
		double* matrix: Matrice da stampare
		int matrix_order: Ordine della matrice quadrata
	
	@return: 
		void
	
*/

void print_matrix(double* matrix, int matrix_order) {
	
	for(int i = 0; i < matrix_order; i++) {
    	for(int j = 0; j < matrix_order; j++)
    		printf(" [%lf]", matrix[i*matrix_order+j]);
    	printf("\n");
	}
	
}


/*

	Crea una griglia di processori di ordine grid_order e due 
	sotto griglie su quest'ultima, di cui una di riga e l'altra di colonna
	
	@params:
		MPI_Comm* grid: Griglia creata
		MPI_Comm* sub_rgrid: Sotto griglia riga creata
		MPI_Comm* sub_cgrid: Sotto griglia colonna creata
		int mpi_rank: Rank del processore chiamante
		int mpi_size: Numero di processori impiegati
		int periods: Periodicit� della griglia
		int reorder: Indica se riordinare il rank dei processori
		int* coords: Coordinate assegnate al processore
	
	@return: 
		void
	
*/

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


/*

	Suddivide e invia la matrice passata in ingresso, dal processore 0 ai restanti
	
	@params:
		double* matrix: Matrice da inviare
		int matrix_order: Ordine della matrice quadrata
		int sub_matrix_order: Ordine della sotto matrice da inviare
		MPI_Comm grid: Griglia di processori
		int mpi_size: Numero di processori impiegati
	
	@return: 
		void
	
*/

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


/*

	Riceve la sotto matrice inviata dal processore 0
	
	@params:
		double* sub_matrix: Sotto matrice
		int sub_matrix_order: Ordine della sotto matrice quadrata
		int mpi_rank: Rank del processore chiamante
	
	@return: 
		void
	
*/

void receive_matrix_from_processor_0(
	double* sub_matrix, int sub_matrix_order, int mpi_rank) {

	MPI_Status status;
	for(int row = 0; row < sub_matrix_order; row++)
		MPI_Recv(
			&sub_matrix[row*sub_matrix_order], sub_matrix_order, MPI_DOUBLE, 
			0, D_TAG + mpi_rank, MPI_COMM_WORLD, &status
		);

}


/*

	Effettua il prodotto righe per colonne parallelo mediante la stregia BMR
	
	@params:
		double* sub_matrix_a: Sotto matrice A (sinistra) da impiegare nel prodotto
		double* sub_matrix_b: Sotto matrice B (destra) da impiegare nel prodotto
		double* sub_matrix_c: Sotto matrice C risultante
		int sub_matrix_order: Ordine delle sotto matrice quadrata
		MPI_Comm grid: Griglia di processori
		MPI_Comm sub_rgrid: Sotto griglia di riga
		MPI_Comm sub_cgrid: Sotto griglia di colonna
		int grid_order: Ordine della griglia quadrata di processori
		int coords[2]: Coordinate nella griglia del processore chiamante
	
	@return: 
		void
	
*/

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


/*

	Unisce tutte le sotto matrici calcolate dai singoli processori in un'unica matrice C
	
	@params:
		double* matrix_c: Matrice risultate dall'operazione di unione
		int matrix_order: Ordine della matrice quadrata risultante
		double* sub_matrix_c: Sotto matrice da inserire in quella finale
		int sub_matrix_order: Ordine della sotto matrice quadrata
		MPI_Comm grid: Griglia di processori
		int mpi_size: Numero di processori impiegati
		int mpi_rank: Rank del processore chiamante
	
	@return: 
		void
	
*/

void merge(
	double* matrix_c, int matrix_order, double* sub_matrix_c, 
	int sub_matrix_order, MPI_Comm grid, int mpi_size, int mpi_rank) {
	
	MPI_Status status;
	int start_row;
	int start_columns;
	int coords[2];

	if(!mpi_rank) {  // Se il chiamante � il processore 0 allora si riceve
		
		/* Si scorre su tutti i processori */
		
		for(int processor = 0; processor < mpi_size; processor++) {
			
			MPI_Cart_coords(grid, processor, 2, coords);
			start_row = coords[0] * sub_matrix_order;
			start_columns = coords[1] * sub_matrix_order;
			
			if(processor) {  // Se non � il processore 0 allora si riceve
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


/*

	Effettua Il prodotto righe per colonne delle matrici quadrate passate in ingresso
	
	@params:
		double* matrix_a: Matrice A (sinistra)
		double* matrix_b: Matrice B (destra)
		double* matrix_c: Risultato del prodotto
		int matrix_order: Ordine delle matrici quadrate
	
	@return: 
		void
	
*/

void multiply(double* matrix_a, double* matrix_b, double* matrix_c, int matrix_order) {
	
	for(int i = 0; i < matrix_order; i++)
		for(int j = 0; j < matrix_order; j++)
			for(int k = 0; k < matrix_order; k++)
				matrix_c[i*matrix_order+j] += (matrix_a[i*matrix_order+k] * matrix_b[k*matrix_order+j]);

}
