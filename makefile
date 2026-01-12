np?=4
hostfile=./hostfile.txt

SHADER?=sdf_0
ARGS?=$(SHADER) 1920 1080 30 300
OUTPUT=renderizador

CC=mpicc
CFLAGS=-lraylib -lm -ldl -lpthread -lGL -lX11  -g
LDFLAGS=-lraylib -lm -ldl -lpthread -lGL -lX11 -g

SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)

default: $(OUTPUT)

$(OUTPUT): $(OBJ)
	$(CC) $(OBJ) -o $(OUTPUT) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(OUTPUT)
	rm *.png
	rm *.mp4

run: $(OUTPUT)
	mpirun -np $(np) $(OUTPUT) $(ARGS)

run_cluster: $(OUTPUT)
	mpirun --hostfile $(hostfile) -np $(np) $(OUTPUT) $(ARGS)
