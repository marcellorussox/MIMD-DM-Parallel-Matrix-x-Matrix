#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#define D_TAG 22

void print_matrix(double*, int);
void create_grid(MPI_Comm*, MPI_Comm*, MPI_Comm*,int, int, int, int*, int, int*);
void send_matrix_from_processor_0(double*, int, int, MPI_Comm, int);
void receive_matrix_from_processor_0(double*, int, int);
void merge(double*, int, double*, int, MPI_Comm, int, int);
void multiply(double*, double*, double*, int);
void bmr(double*, double*, double*, int, MPI_Comm, MPI_Comm, MPI_Comm, int, int[2]);


double* initialize_matrix(int);
double* generate_random_matrix(int);

#endif