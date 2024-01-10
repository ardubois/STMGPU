all: main

main: STM.o
	nvcc -rdc=true --maxrregcount 63 STM.o teste.cu -o teste

api: STM.cu 
	nvcc -c -rdc=true --maxrregcount 63 STM.cu -o STM.o

clean:
	rm STM.o teste
