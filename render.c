#include <mpi.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "render.h"

int numPlaces (int n) {
    if (n == 0) return 1;
    return floor (log10 (abs (n))) + 1;
}

void sendImage(Image* img, int id, int num, int width, int height) {

    const int channels = 4;  // RGBA
    int sliceWidth = width / num;  // Width of this process's slice
    int bytesPerRow = sliceWidth * channels;  // Bytes per row for this slice
    
    unsigned char* data = (unsigned char*)img->data;
    
    // The image should be sliceWidth x height, not full width
    // Send each row of the slice
    for(int row = 0; row < height; row++) {
        int offset = row * bytesPerRow;  // Offset within the slice image
        MPI_Send(data + offset, bytesPerRow, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
}

void recvImage(Image* img, int num, int width, int height) {

    const int channels = 4;  // RGBA
    int sliceWidth = width / num;
    int bytesPerRow = sliceWidth * channels;
    int fullRowBytes = width * channels;
    
    unsigned char* data = (unsigned char*)img->data;
    MPI_Status status;
    
    // Receive slices from each worker process
    for(int row = 0; row < height; row++) {
        for(int id = 1; id < num; id++) {
            int sliceOffset = row * fullRowBytes + (id * sliceWidth * channels);
            MPI_Recv(data + sliceOffset, bytesPerRow, MPI_UNSIGNED_CHAR, id, 0, MPI_COMM_WORLD, &status);
        }
    }
}

void slave(int id, int num, int width, int height, int fps, int frames) {
    
    float time = .0;
    
    InitWindow(512, 512, "Renderizador");
    
    // Create a dummy white texture to draw with the shader
    Image whiteImg = GenImageColor(width, height, WHITE);
    Texture2D whiteTex = LoadTextureFromImage(whiteImg);
    UnloadImage(whiteImg);
    
    RenderTexture2D display = LoadRenderTexture(width/num * (id > 0) + width * (!id), height);
    Shader shader = LoadShader(0, "./sdf_0.glsl");
    
    int locResolution = GetShaderLocation(shader, "u_resolution");
    int locTime = GetShaderLocation(shader, "u_time");
    
    Vector2 resolution = { (float)width, (float)height };
    
    SetShaderValue(shader, locResolution, &resolution, SHADER_UNIFORM_VEC2);
    
    Image frame[frames];

    int digits = numPlaces(frames);

    for(int i = 0; i < frames; ++i) {

        SetShaderValue(shader, locTime, &time, SHADER_UNIFORM_FLOAT);

        BeginTextureMode(display);

            BeginShaderMode(shader);
                // Draw the white texture - this will have proper 0-1 texture coordinates
                DrawTexturePro(
                    whiteTex,
                    (Rectangle){width/num*id, 0, width/num, height},  // Source: right half
                    (Rectangle){0, 0, width/num, height},  // Dest: right half
                    (Vector2){0, 0},
                    0.0f,
                    WHITE
                );

            EndShaderMode();
        EndTextureMode();

        BeginDrawing();
            ClearBackground(BLACK);
        EndDrawing();

        frame[i] = LoadImageFromTexture(display.texture);

        time += (float)frames/(float)fps;

    }

    if(id == 0)
        for(int i = 0; i < frames; ++i) {
            recvImage(&frame[i], num, width, height);
            ImageFlipVertical(&frame[i]);

            char str[64];
            sprintf(str, "output_%0*d.png", digits, i);
            ExportImage(frame[i], str);
        }
    else 
        for(int i = 0; i < frames; ++i)
            sendImage(&frame[i], id, num, width, height); 

    for(int i = 0; i < frames; ++i)
        UnloadImage(frame[i]);

    UnloadTexture(whiteTex);
    UnloadRenderTexture(display);
    UnloadShader(shader);
    CloseWindow();
}

void get_video(int fps, int frames) {
    char command[64];
    sprintf(command, "ffmpeg -framerate %i -i output_%%0%dd.png -c:v libx264 -pix_fmt yuv420p output.mp4", fps, numPlaces(frames));
    printf("Converting to video with command:\n\t%s\n", command);
    system(command);
}