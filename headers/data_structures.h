#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#define D_TAG 22

void print_matrix(double* matrix, int matrix_order);
void create_grid(MPI_Comm* grid, MPI_Comm* sub_rgrid, MPI_Comm* sub_cgrid, int mpi_rank, int mpi_size, int grid_order, int* periods, int reorder, int* coords);
void send_matrix_from_processor_0(double* matrix, int matrix_order, int sub_matrix_order, MPI_Comm grid, int mpi_size);
void receive_matrix_from_processor_0(double* sub_matrix, int sub_matrix_order, int mpi_rank);
void merge(double* matrix_c, int matrix_order, double* sub_matrix_c, int sub_matrix_order, MPI_Comm grid, int mpi_size, int mpi_rank);
void multiply(double* matrix_a, double* matrix_b, double* matrix_c, int matrix_order);
void bmr(double* sub_matrix_a, double* sub_matrix_b, double* sub_matrix_c, int sub_matrix_order,
         MPI_Comm grid, MPI_Comm sub_rgrid, MPI_Comm sub_cgrid, int grid_order, int coords[2]);


double* initialize_matrix(int matrix_order);
double* generate_random_matrix(int matrix_order);

#endif
