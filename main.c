#include <mpi.h> 
#include <stdio.h>
#include <stdlib.h>

#include "render.h"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command-line arguments
    if(argc < 4) {
        if(rank == 0)
            printf("Usage: %s <shader name> <width> <height> <fps> <frames>\nThe shader filename must end in .glsl, the \".glsl\" must not be included in the first argument\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    const char* shader_name = argv[1];

    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    int fps = atoi(argv[4]);
    int frames = atoi(argv[5]);
    
    if(rank == 0) 
        printf("[MASTER]:\tRendering %dx%d at %d fps for %d frames\n", width, height, fps, frames);
    
    render(rank, size, shader_name, width, height, fps, frames);
    if(frames > 1 && rank == 0)
        get_video(shader_name, fps, frames);

    MPI_Finalize();
    return 0;
}