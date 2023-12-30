#include "STM.h"
#include <pthread.h>


void* foo(void* p){
   
   STMData* stm_data = (STMData*) p;
   TX_Data* tx_data = TX_Init(stm_data);
   int aborted;
   do{
      aborted = 0;
      int value=TX_Open_Read(stm_data,tx_data,0);
      if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
      {
        int* ptr_value=TX_Open_Write(stm_data,tx_data,1);
        if(ptr_value !=0 )
        {
          *ptr_value = *ptr_value +1 ;
          TX_commit(stm_data,tx_data);
        }
      }
      //assert(stm_data->tr_state[tx_data->tr_id] != ACTIVE);
      if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
      {
        TX_abort_tr(stm_data,tx_data);
        aborted = 1;
       
      }
      if(!aborted)
          assert(stm_data->tr_state[tx_data->tr_id] != ACTIVE);
      
   }while(aborted);
    
    return NULL;
    
}

int main()
{
  int num_objects = 2;
  int num_locators = MAX_LOCATORS;
  int num_tx = 2000;
 
  STMData* stm_data = STM_start(num_objects, num_tx, num_locators); 
  init_objects(stm_data,num_objects);
  init_locators(stm_data,num_tx,num_locators);

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

  
  print_locator(stm_data,stm_data->vboxes[0]);
  print_locator(stm_data,stm_data->vboxes[1]);
  print_stats(stm_data);
  
  
}

