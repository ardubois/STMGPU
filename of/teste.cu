#include "STM.cuh"
//#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#define N_OBJECTS 100

__device__ float rand_() {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
       curandState state;
        curand_init(clock64(), i, 0, &state);

       return curand_uniform(&state);

}

__global__
void foo(void* p){
   
   STMData* stm_data = (STMData*) p;
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   TX_Data* tx_data = TX_Init(stm_data,id);
   int n_trans = 10;
   int trans = 0;
   while(trans<n_trans)
   {
        int o1 = (int)rand_() % (N_OBJECTS -1);
        int o2;
        do{
            o2 = (int) rand_() % (N_OBJECTS -1);
        }while(o1 == o2);
      //printf("o1: %d, o2: %d\n",o1,o2);
   
        int aborted;
        do{
            TX_Start(stm_data,tx_data);
            aborted = 0;
            int* ptr1 = TX_Open_Write(stm_data,tx_data,o1);
            if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
            {
                int* ptr2 = TX_Open_Write(stm_data,tx_data,o2);
                if(ptr2 !=0 )
                {
                    if(*ptr1 > 10)
                    {
                    *ptr1 -= 10;
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
   }
 //   return NULL;
    
}

int main()
{
  int num_objects = N_OBJECTS;
  int num_locators = MAX_LOCATORS;
  int num_tx = 100;

  int num_blocks = num_tx;
  int num_threads = 1;
 
  STMData* stm_data = STM_start(num_objects, num_tx, num_locators); 
  init_objects(stm_data,num_objects,100);
  init_locators(stm_data,num_tx,num_locators);
  
  STMData *d_stm_data = STM_copy(stm_data);

  foo<<<num_blocks,num_threads>>>(d_stm_data);
  cudaError_t kernelErr = cudaGetLastError();
  if(kernelErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(kernelErr));

  printf("FIM!\n");
//  print_stats(stm_data);
  
  
}
