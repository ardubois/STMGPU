all: teste1 teste2

teste2: STM.o
	gcc -o teste2 teste2.c STM.o

teste1: stm
	gcc -o teste1 teste1.c STM.o

stm: STM.c 
	gcc -fPIC -o STM.o -c STM.c

clean:
	rm STM.o teste1
