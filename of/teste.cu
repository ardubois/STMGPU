#include "STM.cuh"
//#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

#define N_OBJECTS 1000

__device__ float rand_() {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
       curandState state;
        curand_init(clock64(), i, 0, &state);

       return curand_uniform(&state);

}

__global__
void foo(STMData* stm_data){
  
   

   //  printf("ale %d",(int)(rand_()*10)); 0 até 9
   
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   printf("id %d\n", id);
   TX_Data* tx_data = TX_Init(stm_data,id);
   printf("after init\n");
   int n_trans = 10;
   int trans = 0;
   while(trans<n_trans)
   {
        int o1 = (int)(rand_() * N_OBJECTS) ;
        int o2;
        do{
            o2 = (int) (rand_() * N_OBJECTS);
        }while(o1 == o2);
      printf("o1: %d, o2: %d\n",o1,o2);

   
        int aborted;
        do{
            TX_Start(stm_data,tx_data);
            printf("depois do start\n");
            aborted = 0;
            int* ptr1 = TX_Open_Write(stm_data,tx_data,o1);
            if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
            {
                int* ptr2 = TX_Open_Write(stm_data,tx_data,o2);
                if(ptr2 !=0 )
                {
                    if(*ptr1 > 10)
                    {
                    *ptr1 -= 20;
                    *ptr2 += 10;
                    }
                    TX_commit(stm_data,tx_data);
                    if(stm_data->tr_state[tx_data->tr_id] == COMMITTED)
                        {trans ++;
                          __syncthreads();
                          TX_garbage_collect(stm_data,tx_data);
                          __syncthreads();
                        }
                }
            }
            assert(stm_data->tr_state[tx_data->tr_id] != ACTIVE);
            if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
            {
                TX_abort_tr(stm_data,tx_data);
                aborted = 1;
            }
            if(!aborted)
                assert(stm_data->tr_state[tx_data->tr_id] != ACTIVE);
      
        }while(aborted);
        printf("commit %d!\n",trans);
   }
 //   return NULL;
    printf("saiu!\n");
    int addr_locator =  stm_data -> vboxes[1];
      Locator* locator = &stm_data -> locators[addr_locator];
     //print_locator(stm_data,locator);
   }
}

int main()
{
  int num_objects = N_OBJECTS;
  int num_locators = MAX_LOCATORS;
  int num_tx = 1000;

  int num_blocks = num_tx;
  int num_threads = 1;
 
  STMData* stm_data = STM_start(num_objects, num_tx, num_locators); 
  init_objects(stm_data,num_objects,100);
  init_locators(stm_data,num_tx,num_locators);
  STMData *d_stm_data = STM_copy_to_device(stm_data);

  foo<<<num_blocks,num_threads>>>(d_stm_data);
  cudaError_t kernelErr = cudaGetLastError();
  if(kernelErr != cudaSuccess) printf("Error kernel: %s\n", cudaGetErrorString(kernelErr));

  CUDA_CHECK_ERROR( cudaDeviceSynchronize() , " synchronize ");
  //kernelErr = cudaGetLastError();
  //if(kernelErr != cudaSuccess) printf("Error synchronize: %s\n", cudaGetErrorString(kernelErr));
  printf("ACABOU!\n");
  STM_copy_from_device(d_stm_data,stm_data);
  printf("FIM!\n");
  print_stats(stm_data);
  
  
}

