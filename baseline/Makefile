all: main

main: api
	nvcc -rdc=true --maxrregcount 63 API.o exe.cu -o bank

api: API.cu 
	nvcc -c -rdc=true --maxrregcount 63 API.cu -o API.o

clean:
	rm API.o bank
