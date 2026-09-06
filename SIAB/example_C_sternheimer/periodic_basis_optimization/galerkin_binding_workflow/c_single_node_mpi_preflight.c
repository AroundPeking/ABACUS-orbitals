#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int rank, size, sum;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (sum != 6) MPI_Abort(MPI_COMM_WORLD, 3);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) puts("MPI_PREFLIGHT_OK ranks=4");
    MPI_Finalize();
    return 0;
}
