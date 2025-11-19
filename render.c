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

void recvImage(Image* img, unsigned char* tmp, int num, int width, int height) {

    const int channels = 4;  // RGBA
    int sliceWidth = width / num;

    int bytesPerRow = sliceWidth * channels;
    int fullRowBytes = width * channels;
    
    MPI_Status status;
    
    // Receive slices from each worker process
    for(int id = 1; id < num; id++) {
        MPI_Recv(tmp, (width/num) * height * 4, MPI_UNSIGNED_CHAR, id, 0, MPI_COMM_WORLD, &status);
        for(int row = 0; row < height; row++) {
            int sliceOffset = row * fullRowBytes + (id * bytesPerRow);
            memcpy(img->data + sliceOffset, tmp + row * bytesPerRow, bytesPerRow);
        }
    }

}

void render(int id, int num, const char* name, int width, int height, int fps, int frames) {
    
    InitWindow(512, 512, "Renderizador");

    Shader shader;
    char *shader_code = NULL;
    size_t shader_code_size = 0;

    if(id == 0) {
        char shader_file_name[64];
        snprintf(shader_file_name, sizeof(shader_file_name), "%s.glsl", name);
        printf("[MASTER]:\tShader filename: %s\n", shader_file_name);
        
        FILE* shader_file = fopen(shader_file_name, "rb");  // Use "rb" for binary mode
        if(!shader_file) {
            printf("[MASTER]:\tError: Could not open shader file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        fseek(shader_file, 0, SEEK_END);
        shader_code_size = ftell(shader_file);
        fseek(shader_file, 0, SEEK_SET);
        
        printf("[MASTER]:\tShader code size: %zu bytes\n", shader_code_size);
        
        shader_code = (char*)malloc(shader_code_size + 1);
        if(!shader_code) {
            printf("[MASTER]:\tError: Could not allocate memory!\n");
            fclose(shader_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        size_t read = fread(shader_code, 1, shader_code_size, shader_file);
        shader_code[shader_code_size] = '\0';
        
        fclose(shader_file);
    }

    // Broadcast size first
    MPI_Bcast(&shader_code_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Allocate on worker processes
    if(id > 0)
        shader_code = (char*)malloc(shader_code_size + 1);


    // Broadcast the actual shader code
    MPI_Bcast(shader_code, shader_code_size + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Now load the shader
    shader = LoadShaderFromMemory(NULL, shader_code);

    free(shader_code);

    float time = .0;
    
    // Create a dummy white texture to draw with the shader
    Image whiteImg = GenImageColor(width, height, WHITE);
    Texture2D whiteTex = LoadTextureFromImage(whiteImg);
    UnloadImage(whiteImg);
    
    RenderTexture2D display = LoadRenderTexture(width/num * (id > 0) + width * (!id), height);
    
    
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
                
                DrawTexturePro(
                    whiteTex,
                    (Rectangle){width/num*id, 0, width/num, height}, 
                    (Rectangle){0, 0, width/num, height},  
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

        time += 1./(float)fps;

    }

    if(id == 0) {

        unsigned char* tmp_buffer = malloc((width/num) * height * 4);

        for(int i = 0; i < frames; ++i) {
            recvImage(&frame[i], tmp_buffer, num, width, height);
            ImageFlipVertical(&frame[i]);

            char str[64];
            sprintf(str, "%s_%0*d.png", name, digits, i);
            ExportImage(frame[i], str);
        }

        free(tmp_buffer);

    } else 
        for(int i = 0; i < frames; ++i)
            MPI_Send(frame[i].data, (width/num) * height * 4, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);

    for(int i = 0; i < frames; ++i)
        UnloadImage(frame[i]);

    UnloadTexture(whiteTex);
    UnloadRenderTexture(display);
    UnloadShader(shader);
    CloseWindow();
}

void get_video(const char* name, int fps, int frames) {
    char command[256];
    sprintf(command, "ffmpeg -framerate %i -i %s_%%0%dd.png -c:v libx264 -pix_fmt yuv420p %s.mp4", fps, name, numPlaces(frames), name);
    printf("Converting to video with command:\n\t%s\n", command);
    system(command);
    system("rm *.png");
}