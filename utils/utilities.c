#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>
#include "../headers/utilities.h"
#include <stdarg.h>

void print_how_to_use(char* program_name){
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

int get_params_code(int argc, char** argv, int mpi_size) {
	
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

int check_params(int argc, char** argv, int mpi_size, int mpi_rank){
    int errorCode = get_params_code(argc,argv,mpi_size);
	if(!mpi_rank) {	
    	switch(errorCode) {
			
			case SCC_HELP:
				print_how_to_use(argv[0]);
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

		if(errorCode != SCC_ARGS && errorCode != SCC_HELP)
			MPI_Abort(MPI_COMM_WORLD, errorCode); 
	}	

	if(mpi_size != 1)
		MPI_Bcast(&errorCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(errorCode == SCC_HELP) {
		MPI_Finalize();
		exit(0);
	}
    
	return 0;
}








