#include <mpi.h> 
#include <stdio.h>

#include "render.h"

/*
    TODO:
        ·   Añadir soporte para custom arguments (shaders, dimensiones, fps, fotogramas)
        ·   Que no renderize vídeo si hay 1 o menos frames
        ·   Que el shader se envíe del maestro a los esclavos
*/

int main(int argc, char** argv)
{
    int id, num;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num);

    slave(id, num, 1920, 1080, 60, 300);

    if(id == 0) get_video(60, 300);

    MPI_Finalize();

    return 0;
}