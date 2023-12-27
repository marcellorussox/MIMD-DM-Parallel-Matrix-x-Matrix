#ifndef UTILITIES_H
#define UTILITIES_H

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


void print_how_to_use(char* program_name);
int get_params_code(int argc, char** argv, int mpi_size);
int check_params(int argc, char** argv, int mpi_size);

#endif
