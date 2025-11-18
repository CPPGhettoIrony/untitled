np?=4
OUTPUT=renderizador

CC=mpicc
CFLAGS=-lraylib -lm -ldl -lpthread -lGL -lX11  -g
LDFLAGS=-lraylib -lm -ldl -lpthread -lGL -lX11 -g

SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)

all: $(OUTPUT)

$(OUTPUT): $(OBJ)
	$(CC) $(OBJ) -o $(OUTPUT) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(OUTPUT)

run: $(OUTPUT)
	mpirun -np $(np) $(OUTPUT)