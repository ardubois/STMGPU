#include "STM.h"
#include <stdlib.h>
#include <pthread.h>

#define N_OBJECTS 100

pthread_barrier_t barrier;

void* foo(void* p){
   
   STMData* stm_data = (STMData*) p;
   TX_Data* tx_data = TX_Init(stm_data);
   int n_trans = 10;
   int trans = 0;
   while(trans<n_trans)
   {
        int o1 = rand() % (N_OBJECTS -1);
        int o2;
        do{
            o2 = rand() % (N_OBJECTS -1);
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
                          pthread_barrier_wait(&barrier);
                          TX_garbage_collect(stm_data,tx_data);
                          pthread_barrier_wait(&barrier);
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
    return NULL;
    
}

int main()
{
  int num_objects = N_OBJECTS;
  int num_locators = MAX_LOCATORS;
  int num_tx = 100;
 
  STMData* stm_data = STM_start(num_objects, num_tx, num_locators); 
  init_objects(stm_data,num_objects,100);
  init_locators(stm_data,num_tx,num_locators);

  pthread_barrier_init(&barrier, NULL, num_tx);

  pthread_t threads[num_tx];

  for(int i=0; i< num_tx; i++)
   {
    pthread_create(&threads[i],NULL, foo, stm_data);
   }
  
  //pthread_create(&tid1, NULL, foo, stm_data); 
  
  for(int i=0; i< num_tx; i++)
   {
    pthread_join(threads[i],NULL);
   }

  printf("FIM!\n");
  print_stats(stm_data);
  
  
}

