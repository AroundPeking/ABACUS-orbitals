#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Exercise the existing MPI transport before starting the physical producer. */
int main(int argc, char **argv)
{
    int rank, size, length, failures = 0, total_failures = 0;
    char host[MPI_MAX_PROCESSOR_NAME], version[MPI_MAX_LIBRARY_VERSION_STRING];
    const int count = 131072;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(host, &length);
    printf("rank=%d host=%s\n", rank, host);
    fflush(stdout);
    if (size != 48) MPI_Abort(MPI_COMM_WORLD, 2);
    double *data = malloc((size_t)count * sizeof(double));
    int *send = malloc((size_t)size * sizeof(int));
    int *recv = malloc((size_t)size * sizeof(int));
    if (!data || !send || !recv) MPI_Abort(MPI_COMM_WORLD, 3);
    double start = MPI_Wtime();
    int round = 0, stop = 0;
    while (!stop) {
        int root = round % size;
        if (rank == root)
            for (int i = 0; i < count; ++i) data[i] = (double)(i + round);
        MPI_Bcast(data, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
        for (int i = 0; i < count; ++i)
            if (data[i] != (double)(i + round)) ++failures;
        int sum = 0;
        MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (sum != size * (size - 1) / 2) ++failures;
        for (int i = 0; i < size; ++i) send[i] = rank * size + i;
        MPI_Alltoall(send, 1, MPI_INT, recv, 1, MPI_INT, MPI_COMM_WORLD);
        for (int i = 0; i < size; ++i)
            if (recv[i] != i * size + rank) ++failures;
        MPI_Barrier(MPI_COMM_WORLD);
        ++round;
        if (rank == 0) stop = (round >= 48 && MPI_Wtime() - start >= 30.0);
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (total_failures) MPI_Abort(MPI_COMM_WORLD, 4);
    if (rank == 0) {
        MPI_Get_library_version(version, &length);
        printf("mpi_library=%.*s\n", length, version);
        printf("MPI_COMM_PREFLIGHT_OK ranks=%d rounds=%d seconds=%.3f\n",
               size, round, MPI_Wtime() - start);
    }
    free(data);
    free(send);
    free(recv);
    MPI_Finalize();
    return 0;
}
