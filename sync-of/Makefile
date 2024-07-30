all: bank

bank: api
	nvcc -rdc=true STM.o bank.cu -o bank
teste: api
	nvcc -rdc=true --maxrregcount 63 STM.o teste.cu -o teste

api: STM.cu 
	nvcc -c -rdc=true STM.cu -o STM.o

clean:
	rm STM.o bank
