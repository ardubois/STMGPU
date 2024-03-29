all: teste1 teste2 teste3 teste4

debug: dstm dteste1 dteste2 dteste4

bank: STM.o
	gcc -g -o bank bank.c STM.o

teste4: STM.o
	gcc -o teste4 teste4.c STM.o

teste3: STM.o
	gcc -o teste3 teste3.c STM.o

teste2: STM.o
	gcc -o teste2 teste2.c STM.o

teste1: stm
	gcc -o teste1 teste1.c STM.o

stm: STM.c 
	gcc -fPIC -o STM.o -c STM.c

dteste4: STM.o
	gcc -g -o teste4 teste4.c STM.o
dteste2: STM.o
	gcc -g -o teste2 teste2.c STM.o

dteste1: dstm
	gcc -g -o teste1 teste1.c STM.o

dstm: STM.c 
	gcc -g -fPIC -o STM.o -c STM.c 
clean:
	rm STM.o teste1 teste2
