#include "STM.cuh"

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

int tr_id_gen=0;

STMData* STM_start(int numObjects, int numTransactions, int numLocators)
{
    STMData *meta_data = (STMData*) malloc(sizeof(STMData));
    //meta_data-> objects = malloc(numObjects * sizeof(Locator));
    meta_data -> n_objects = numObjects;
   // meta_data-> objects_data = malloc(2*numObjects * sizeof(int));
    meta_data-> vboxes = (int*) malloc(numObjects * sizeof(int));
    meta_data-> tr_state = (int*) malloc((numTransactions+2) * sizeof(int)); // 1 for the always committed Tr and 1 for the always aborted 
    meta_data-> locators = (Locator*) malloc((numObjects + (numLocators * numTransactions)) * sizeof(Locator));
   // printf("Total locators: %d", (numObjects + (numLocators * numTransactions)));
    meta_data-> locators_data = (int*) malloc(((2*numObjects)+(2*numLocators * numTransactions)) * sizeof(int));
    meta_data -> num_locators = numLocators;
    meta_data -> tx_data  = (TX_Data*) malloc(numTransactions * sizeof(TX_Data));
    meta_data -> num_tr = numTransactions;
    meta_data -> tr_state[numTransactions] = COMMITTED;
    meta_data -> tr_state[numTransactions+1] = ABORTED;
    return meta_data;
}

void STM_copy_from_device(STMData* d_stm_data, STMData* stm_data)
{
    int numObjects = stm_data -> n_objects;
    int numTransactions = stm_data -> num_tr;
    int numLocators = stm_data -> num_locators;
   
   STMData local_ddata;
   CUDA_CHECK_ERROR(cudaMemcpy(&local_ddata, d_stm_data,  sizeof(STMData), cudaMemcpyDeviceToHost), " copy to device stm data");

    CUDA_CHECK_ERROR(cudaMemcpy(stm_data->vboxes, local_ddata.vboxes, numObjects * sizeof(int), cudaMemcpyDeviceToHost), "copy to device vboxes ");
    CUDA_CHECK_ERROR(cudaMemcpy(stm_data->tr_state, local_ddata.tr_state, (numTransactions+2) * sizeof(int), cudaMemcpyDeviceToHost), "copy to device tr state");
    CUDA_CHECK_ERROR(cudaMemcpy( stm_data->locators, local_ddata.locators, (numObjects + (numLocators * numTransactions)) * sizeof(Locator), cudaMemcpyDeviceToHost), "copy to device locators ");
    CUDA_CHECK_ERROR(cudaMemcpy(stm_data->locators_data, local_ddata.locators_data,((2*numObjects)+(2*numLocators * numTransactions)) * sizeof(int), cudaMemcpyDeviceToHost), "copy to device locators data ");
    CUDA_CHECK_ERROR(cudaMemcpy(stm_data->tx_data, local_ddata.tx_data,numTransactions * sizeof(TX_Data), cudaMemcpyDeviceToHost), "copy to device tx data ");

    fix_pointers_locators(stm_data,stm_data->locators_data);

   }

STMData* STM_copy_to_device(STMData* stm_data)
{
    int numObjects = stm_data -> n_objects;
    int numTransactions = stm_data -> num_tr;
    int numLocators = stm_data -> num_locators;

    STMData *meta_data = (STMData*) malloc(sizeof(STMData));
    //meta_data-> objects = malloc(numObjects * sizeof(Locator));
    meta_data -> n_objects = numObjects;
   // meta_data-> objects_data = malloc(2*numObjects * sizeof(int));
    int* d_vboxes;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_vboxes, numObjects * sizeof(int)), " malloc vboxexs ");
    CUDA_CHECK_ERROR( cudaMemcpy(d_vboxes, stm_data->vboxes, numObjects * sizeof(int), cudaMemcpyHostToDevice), " mem copy vboxes");
    meta_data-> vboxes = d_vboxes;

    int* d_tr_state;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tr_state, (2+numTransactions) * sizeof(int)), " malloc tr state");
    CUDA_CHECK_ERROR( cudaMemcpy(d_tr_state, stm_data->tr_state, ((2+numTransactions) * sizeof(int)), cudaMemcpyHostToDevice), " copy tr state");
    meta_data-> tr_state = d_tr_state; // 1 for the always committed Tr and 1 for the always aborted 

    int* d_locators_data;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_locators_data, ((2*numObjects)+(2*numLocators * numTransactions)) * sizeof(int)), " malloc locators data");
    CUDA_CHECK_ERROR( cudaMemcpy(d_locators_data, stm_data->locators_data, ((2*numObjects)+(2*numLocators * numTransactions)) * sizeof(int), cudaMemcpyHostToDevice), " copy locators data");
    meta_data-> locators_data = d_locators_data; 


    Locator* d_locators;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_locators, (numObjects + (numLocators * numTransactions)) * sizeof(Locator)), " malloc locators ");
    fix_pointers_locators(stm_data,d_locators_data);
    CUDA_CHECK_ERROR( cudaMemcpy(d_locators, stm_data->locators, (numObjects + (numLocators * numTransactions)) * sizeof(Locator), cudaMemcpyHostToDevice), " copy locators");
    meta_data-> locators = d_locators; 


    TX_Data* d_tx_data;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tx_data, numTransactions * sizeof(TX_Data)), " malloc tx  data");
    CUDA_CHECK_ERROR( cudaMemcpy(d_tx_data, stm_data->tx_data, numTransactions * sizeof(TX_Data), cudaMemcpyHostToDevice), " copy tx data");
    meta_data-> tx_data = d_tx_data; 

    meta_data -> num_locators = numLocators;
    meta_data -> num_tr = numTransactions;

    STMData* d_stm_data;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_stm_data,  sizeof(STMData)), " malloc stm data ");
    CUDA_CHECK_ERROR( cudaMemcpy(d_stm_data, meta_data, sizeof(STMData), cudaMemcpyHostToDevice), " copy  stm data");

    
    
    free(meta_data);
    return d_stm_data;
}

__device__ TX_Data* TX_Init(STMData* stm_data, int tx_id, int* locator_queue)
{
  
    TX_Data *d = &stm_data -> tx_data[tx_id];

    int numLocators = stm_data -> num_locators;
    
    //int locator_queue[MAX_LOCATORS];

    d-> tr_id = tx_id;
    d-> next_locator = 0;
    d -> locator_queue = locator_queue;
    d -> read_set.size =0;
    d -> write_set.size = 0;
    d -> n_aborted = 0;
    d -> n_committed = 0;
    d -> cm_enemy = -1;
    d -> cm_aborts = 0;
    stm_data -> tr_state[d->tr_id] = ACTIVE;

    for(int i = 0; i<numLocators;i++)
    {
      d->locator_queue[i] = (tx_id * stm_data -> num_locators) + i;
     // printf("q[%d] = %d\n", i, (tx_id * stm_data -> num_locators) + i);
    }

    return d;
}

__device__ void TX_Start(STMData* stm_data, TX_Data* d)
{
   
    d -> read_set.size =0;
    d -> write_set.size = 0;
    //d -> n_aborted = 0;
   // d -> n_committed = 0;
    
    if(stm_data -> tr_state[d->tr_id]== COMMITTED)
    {
      d-> enemies_size = 0;
      d -> cm_enemy = -1;
      d -> cm_aborts = 0;
    }
    stm_data -> tr_state[d->tr_id] = ACTIVE;
    
}

__device__ void TX_garbage_collect(STMData* stm_data, TX_Data* tx_data)
{
  if(tx_data -> next_locator > 0)
  {
    int used_locators[MAX_LOCATORS];
    int used_pos = 0;
    tx_data -> next_locator--;
    int next = tx_data -> next_locator;

    do{
      int next_locator = tx_data -> locator_queue [next];
      Locator* locator = &stm_data -> locators[next_locator];

      if(stm_data -> vboxes[locator->object] == next_locator)
      {
            used_locators[used_pos] = next_locator;
            used_pos ++;
            next --;
      } else {
            tx_data -> locator_queue [tx_data -> next_locator] = next_locator;
            int id = stm_data -> locators[next_locator].id;
            assert(atomicCAS(&stm_data -> locators[next_locator].id,id ,id+1)==id);
            tx_data -> next_locator --;
            next --;
      }
    } while(next >= 0);
    
    int pos_queue = tx_data -> next_locator;
    tx_data -> next_locator ++;
    assert(tx_data -> next_locator == used_pos);
    used_pos--;
    assert(used_pos == pos_queue);
    
    while(pos_queue >= 0)
    { 
      tx_data -> locator_queue [pos_queue] = used_locators[used_pos];
      pos_queue --;
      used_pos --;
    }
  }
}


/*
void TX_garbage_collect(STMData* stm_data, TX_Data* tx_data)
{
  
  int used_locators[WriteSetSize];
  int used_pos = 0;
  tx_data -> next_locator--;
  int next = tx_data -> next_locator;
  do {
    int found = 0;
    
    int next_locator = tx_data -> locator_queue [next];
    Locator* locator = &stm_data -> locators[next_locator]; 
    for(int i=0;i<tx_data->write_set.size;i++)
    {
      if(tx_data->write_set.locators[i] == locator)
      {
        if(stm_data -> vboxes[tx_data -> write_set.objects[i]] == locator)
        {
          found = 1;
          used_locators[used_pos] = next_locator;
          used_pos ++;
          break;
        } else
          {break;}
      }
    }
    if(found)
    {
      next --;
    } else {
      tx_data -> locator_queue [tx_data -> next_locator] = next_locator;
      tx_data -> next_locator --;
      next --;
    }
    
  } while(next>=0);
    
    int pos_queue = tx_data -> next_locator;
    tx_data -> next_locator ++;
    //printf("next: %d  used: %d\n",tx_data -> next_locator,used_pos);
    assert(tx_data -> next_locator == used_pos);
    used_pos--;
    assert(used_pos == pos_queue);
    assert(used_pos <= tx_data->write_set.size);
    while(pos_queue >= 0)
    { 
      tx_data -> locator_queue [pos_queue] = used_locators[used_pos];
      pos_queue --;
      used_pos --;
    }
}
*/

__device__ int TX_new_locator(STMData* stm_data, TX_Data* tx_data)
{
  int next_locator = tx_data -> locator_queue [tx_data->next_locator];
 // Locator* locator = &stm_data -> locators[next_locator]; 
  tx_data -> next_locator++;
  if(tx_data -> next_locator == MAX_LOCATORS)
    {
      printf("Max locators reached!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
     assert(tx_data -> next_locator < MAX_LOCATORS);
    
     //print_stats(stm_data);
     //exit(0);
    }
  return next_locator;
  
  /*
  Locator* locator = stm_data -> locators;
  locator += (tx_data->tr_id * stm_data -> num_locators) + tx_data-> next_locator;
  tx_data -> next_locator++;
  if(tx_data -> next_locator == MAX_LOCATORS)
    {
     printf("Max locators reached!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
     print_stats(stm_data);
     exit(0);
    }
  return locator;
*/

}

__device__ int TX_validate_readset(STMData* stm_data, TX_Data* tx_data)
{
  if (tx_data-> write_set.size == 0)
  { return 1; }

  ReadSet* read_set = &tx_data-> read_set;
  int size = tx_data-> read_set.size;
  
  for (int i=0;i<size;i++)
  {
    int *current_value = 0;
   
    if(!( stm_data -> vboxes[read_set->object[i]] == read_set -> locator[i]))
    { return 0;}
    Locator *loc = &stm_data -> locators[read_set -> locator[i]];
    if (stm_data->tr_state[loc -> owner] == COMMITTED)
       { current_value = loc -> new_version;}
    if (stm_data->tr_state[loc -> owner] == ABORTED || stm_data->tr_state[loc -> owner] == ACTIVE)
       { current_value = loc -> old_version;}
    assert(current_value != 0);
    int id = loc -> id;
    if(read_set -> id[i] != id)
      {return 0;}
    if(read_set -> value[i] != current_value )
      {return 0;}
  }
  return 1;
}

    


/*
void TX_free_writeset(STMData* stm_data, TX_Data* tx_data, int state){
  if (state == COMMITTED)
  {
    for(int i=0;i<tx_data->write_set.size;i++)
    {
      printf("\nowner: %d, me: %d\n\n",tx_data->write_set.locators[i]->owner,tx_data->tr_id);
      assert(__sync_bool_compare_and_swap(&tx_data->write_set.locators[i]->owner,tx_data->tr_id,stm_data-> num_tr));
    }
  } else{
for(int i=0;i<tx_data->write_set.size;i++)
    {
      printf("\nowner: %d, me: %d\n\n",tx_data->write_set.locators[i]->owner,tx_data->tr_id);
      assert(__sync_bool_compare_and_swap(&tx_data->write_set.locators[i]->owner,tx_data->tr_id,stm_data-> num_tr+1));
    }
  }

}
*/
__device__ int TX_commit(STMData* stm_data, TX_Data* tx_data)
{ __threadfence();
  if(TX_validate_readset(stm_data,tx_data))
  {
     if(atomicCAS(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,COMMITTED)==ACTIVE)
     {
      //TX_free_writeset(stm_data,tx_data, COMMITTED);
      for(int i=0;i<tx_data->write_set.size;i++)
      {
      //printf("\nLocator: %d, owner: %d, me: %d\n\n",tx_data->write_set.locators[i],stm_data->locators[tx_data->write_set.locators[i]].owner,tx_data->tr_id);
        assert(atomicCAS(&stm_data-> locators[tx_data->write_set.locators[i]].owner,tx_data->tr_id,stm_data-> num_tr)==tx_data->tr_id);
      }
      tx_data -> n_committed ++;
      return 1;
     }
  }
    atomicCAS(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
    assert(stm_data->tr_state[tx_data->tr_id]==ABORTED);
     return 0;         
}

__device__ void open_locator(STMData* stm_data, TX_Data* tx_data, 
                              uint object, Locator *locator, int *addr)
{

 Locator* curr_locator;
 int id2;
  do{  
    do{
       *addr =  stm_data -> vboxes[object];
        curr_locator = &stm_data -> locators[*addr];
        locator -> id = curr_locator -> id;
    } while(*addr !=  stm_data -> vboxes[object]);
    
    locator -> owner =  curr_locator -> owner;
    locator -> object =  curr_locator -> object;
    locator -> new_version =  curr_locator -> new_version;
    locator -> old_version=  curr_locator -> old_version;
    id2 = curr_locator -> id;
  } while(locator -> id != id2);
}

__device__  int* TX_Open_Write(STMData* stm_data, TX_Data* tx_data, uint object)
{
  
  Locator locator_copy;
  int addr_locator;     
   while (stm_data->tr_state[tx_data->tr_id] != ABORTED)
   {
    
     open_locator(stm_data, tx_data, object, &locator_copy, &addr_locator);
     Locator *locator = &locator_copy;
     if (locator -> owner == tx_data->tr_id)
        return locator -> new_version;

     int addr_new_locator = TX_new_locator(stm_data,tx_data);
   //   int next_locator = tx_data -> locator_queue [tx_data->next_locator];
      Locator *new_locator = &stm_data -> locators[addr_new_locator];
      new_locator -> owner = tx_data->tr_id;
      new_locator -> object = object;
      //printf("OW: Object %d, Transaction: %d Locator: %d (Owner %d), new Locator %d, next locator %d, queue %d\n",object,tx_data->tr_id, addr_locator,locator -> owner,addr_new_locator, next_locator,tx_data->next_locator);
      assert(locator -> owner != new_locator -> owner);
      
      switch (stm_data->tr_state[locator -> owner]) {
            case COMMITTED:
              *new_locator-> old_version =  *locator->new_version;
              *new_locator-> new_version = *new_locator-> old_version;
              break;
            case ABORTED:
              *new_locator->old_version =  *locator->old_version;
              *new_locator-> new_version = *new_locator-> old_version;
              break;
            case ACTIVE: 
             // printf("here\n");
              if(TX_contention_manager(stm_data,tx_data, new_locator->owner,locator->owner))
              {
                if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
                { 
                  __threadfence();
                  if(atomicCAS(&stm_data->tr_state[locator -> owner],ACTIVE ,ABORTED)==ACTIVE)
                  {
                   *new_locator->old_version = *locator->old_version;
                   *new_locator-> new_version = *new_locator-> old_version;
                  } else {
                  atomicCAS(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
                  assert(stm_data->tr_state[tx_data->tr_id]==ABORTED);
                  tx_data -> next_locator--;
                  continue;
                 }
                } else {
                  tx_data -> next_locator--;
                  continue;
                }
              } else{
                  // __sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
                  // assert(stm_data->tr_state[tx_data->tr_id]==ABORTED);
                   atomicCAS(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
                  assert(stm_data->tr_state[tx_data->tr_id]==ABORTED);
                   tx_data -> next_locator--;
                   continue;
              }
               
              break;
            default:
                printf("TX_Write: invalid tr state! Locator %d, Owner %d, state %d\n",addr_locator,locator -> owner,stm_data->tr_state[locator -> owner]);
                 assert(0);
                //exit(0);
          }
   
      if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
      {  __threadfence();
         if(atomicCAS(&stm_data -> vboxes[object],addr_locator ,addr_new_locator)==addr_locator){
            //printf("CAS\n");
            //print_locator(stm_data,new_locator);
            WriteSet* write_set = &tx_data-> write_set;
            int size = tx_data-> write_set.size;
            write_set -> locators[size] = addr_new_locator;
            write_set -> objects[size] = object;
            write_set -> size ++;
            
          //  printf("Write set added: Locator: %d owner %d size%d\n", addr_new_locator, stm_data -> locators[addr_new_locator].owner, write_set -> size);
             
            if(TX_validate_readset(stm_data,tx_data))
              {//print_locator(stm_data,new_locator);
               return new_locator->new_version;
              }
            else
              {
                atomicCAS(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
                assert(stm_data->tr_state[tx_data->tr_id] == ABORTED) ;
              }
         } else { //
              //__sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
              tx_data -> next_locator--;
              continue;
         }
      }
    assert(stm_data->tr_state[tx_data->tr_id] != ACTIVE) ;    
    return 0; 
   }
   assert(stm_data->tr_state[tx_data->tr_id] == ABORTED) ;
   return 0;
}

__device__  int is_enemy(TX_Data* tx_data,unsigned int enemy)
{
  for(int i = 0; i< tx_data->enemies_size;i++)
  {
    if(tx_data->cm_enemies[i]== enemy)
      return 1;
  }
  return 0;
}

__device__  int TX_contention_manager9(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
  if(is_enemy(tx_data,enemy))
    return 1;
  else
  {
    tx_data->cm_enemies[tx_data->enemies_size] = enemy;
    tx_data->enemies_size++;
  }
  return 0;
}


__device__  int TX_contention_manager6(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
  if(tx_data->cm_enemy == enemy)
  { 
    tx_data->cm_aborts ++;
    if(tx_data->cm_aborts>=10)
      return 1;
  } else {
    tx_data->cm_enemy = enemy;
    tx_data->cm_aborts =0;
  }
  return 0;
}



__device__  int TX_contention_manager8(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
  if(tx_data->cm_enemy == enemy)
  { 
    tx_data->cm_aborts ++;
    if(tx_data->cm_aborts>=100)
    {
      TX_Data *data_enemy = &stm_data -> tx_data[enemy];
      if(data_enemy-> write_set.size < tx_data ->write_set.size)
      {
        return 1;
      }
     
    }
  } else {
    tx_data->cm_enemy = enemy;
    tx_data->cm_aborts =0;
  }
  return 0;
}

__device__  int TX_contention_manager7(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{

  if(tx_data->cm_aborts > 10)
  { tx_data->cm_aborts =0;
    return 1;
  } else {
    tx_data->cm_aborts ++;   
    return 0;
  }
}

__device__  int TX_contention_manager3(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
  if(tx_data->n_aborted > BACKOFF)
  { 
    TX_Data *data_enemy = &stm_data -> tx_data[enemy];
    if(data_enemy-> write_set.size < tx_data ->write_set.size)
    {
      if (data_enemy-> n_aborted < tx_data->n_aborted)
        return 1;
      else
        return 0;
    }
     if (data_enemy-> n_aborted < tx_data->n_aborted)
        return 1;
      else
        return 0;
  }
  return 0;
}

__device__  int TX_contention_manager2(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
  if(tx_data->n_aborted > BACKOFF)
  { 
    TX_Data *data_enemy = &stm_data -> tx_data[enemy];
    if(data_enemy-> write_set.size < tx_data ->write_set.size)
       return 1;
    else return 0;
  }
  return 0;
}

__device__  int TX_contention_manager5(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
  if(tx_data->n_aborted > 100)
  { 
    return 1;
    
  }
  return 1;
}

__device__  int TX_contention_manager1(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
    if(enemy < me)
       return 1;
    return 0;
}

__device__  int TX_contention_manager4(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
    TX_Data *data_enemy = &stm_data -> tx_data[enemy];
    if(data_enemy-> write_set.size < tx_data ->write_set.size)
    {
      return 1;
    }
    if(data_enemy-> write_set.size == tx_data ->write_set.size)
    {
      if (data_enemy-> n_aborted < tx_data->n_aborted)
        return 1;
      else
        return 0;
    }
      
  return 0;
}
// best 4
__device__  int TX_contention_manager(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
  return TX_contention_manager4(stm_data,tx_data, me, enemy);
}


__device__ int TX_Open_Read(STMData* stm_data, TX_Data* tx_data, uint object)
{
    int* version;
    int addr_locator;
    Locator *locator;
    int id;
    do{  
      addr_locator= stm_data -> vboxes[object];
      locator = &stm_data-> locators[addr_locator];
      id = locator -> id;
    }while(stm_data -> vboxes[object] != addr_locator);
     
    switch (stm_data->tr_state[locator -> owner]) {
            case COMMITTED:
              version =  locator->new_version;
              break;
            case ABORTED:
              version =  locator->old_version;
              break;
            case ACTIVE: 
              version = locator->old_version;
              break;
            default:
                printf("TX_Read: invalid tr state!\n");
               // exit(0);
          }

    int id2 =  locator -> id;
    if(id != id2)
          {return 0;}
              
    if(TX_validate_readset(stm_data,tx_data))
              {    
                  ReadSet* read_set = &tx_data-> read_set;
                  int size = tx_data-> read_set.size;
                  read_set -> locator[size] = addr_locator;
                  read_set -> value[size] = version;
                  read_set -> object[size] = object;
                  read_set -> id[size] = id;
                  read_set -> size ++;
                  return *version;
              }
    atomicCAS(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
    assert(stm_data->tr_state[tx_data->tr_id] == ABORTED) ;
    return 0; 
}



__device__ void TX_abort_tr(STMData* stm_data, TX_Data* tx_data){


  for (int i = 0; i < tx_data->write_set.size; i ++)
  { //printf("\nAbort Locator: %d, owner: %d, me: %d\n\n",tx_data->write_set.locators[i],stm_data->locators[tx_data->write_set.locators[i]].owner,tx_data->tr_id);
    assert(atomicCAS(&stm_data-> locators[tx_data -> write_set.locators[i]].owner, tx_data->tr_id , (stm_data -> num_tr+1))==tx_data->tr_id);
    //{
     // printf("nao deveria\n owner = %d, id = %d,locaotr: %d\n",stm_data-> locators[tx_data -> write_set.locators[i]].owner,  tx_data->tr_id,tx_data ->write_set.locators[i]);
     //// exit(0);
    //}
  }

  assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
  //TX_free_writeset(stm_data,tx_data, ABORTED);
  //assert(__sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ABORTED,ACTIVE));
  
  //tx_data-> read_set.size = 0;
  //tx_data -> write_set.size = 0;
  tx_data -> n_aborted ++;
}

void fix_pointers_locators(STMData* stm_data, int* locators_data)
{
  int initial_locators = stm_data -> num_locators * stm_data-> num_tr;
  int num_objects = stm_data -> n_objects;
  for(int i=0; i< (initial_locators+num_objects);i++)
  {

    stm_data-> locators[i].new_version = locators_data+2*i;
    stm_data-> locators[i].old_version = locators_data+2*i+1;

  }
}

void init_objects(STMData* stm_data,int num_objects, int value)
{
  stm_data -> tr_state[stm_data->num_tr] = COMMITTED;
  int initial_locators = stm_data -> num_locators * stm_data-> num_tr;
  int pos = 0;
  //printf("value: %d\n",value);
  for(int i=initial_locators;i<(initial_locators+num_objects);i++)
  { 
//    printf("init objects: %d\n",i);
    stm_data->locators_data[2*i] = value;
    stm_data->locators_data[2*i+1] = 0;
    stm_data-> locators[i].new_version = &stm_data->locators_data[2*i];
    stm_data-> locators[i].old_version = &stm_data->locators_data[2*i+1];
    stm_data-> locators[i].owner = stm_data->num_tr;
    stm_data->vboxes[pos] = i;
    
  
    pos++;
  
  }
}

void init_locators(STMData* stm_data,int num_tx, int num_locators)
{
int total_locators = num_tx*num_locators;
for(int i=0;i<total_locators;i++)
  {
    stm_data-> locators_data[2*i] = 0;
    stm_data-> locators_data[2*i+1] = 0;
    stm_data-> locators[i].new_version = &stm_data->locators_data[2*i];
    stm_data-> locators[i].old_version = &stm_data->locators_data[2*i+1];
    stm_data-> locators[i].id = 0;
  }
}

void print_vboxes(STMData* stm_data)
{
   for(int i=0; i < stm_data->n_objects;i++)
   {
    print_locator(stm_data,&stm_data->locators[stm_data->vboxes[i]]);
   }
}


__host__ __device__ void print_tr_state(int tr_state)
{
    switch ( tr_state )
  {
    case 1 :
    printf ("ACTIVE");
    break;
    
    case 2 :
    printf ("COMMITTED");
    break;
    
    case 3 :
    printf ("ABORTED");
    break;
    
    default :
    printf ("Unknown thread state!");
  }
}
__host__ __device__ void print_locator(STMData* stm_data,Locator *locator)
{
  printf("Locator %p:\n", locator);
  printf("- owner: %d\n",locator->owner);
  printf("- state: %d, ", stm_data-> tr_state[locator->owner]);
  print_tr_state(stm_data-> tr_state[locator->owner]);
  printf("\n");
  printf("- new_version %d (%p)\n", *locator->new_version,locator->new_version);
  printf("- old_version %d (%p)\n", *locator->old_version,locator->old_version);
}


void print_data(STMData* stm_data)
{
  int size = stm_data -> num_tr;
//  printf("size: %d\n",size);
  TX_Data* tx_data = stm_data -> tx_data;
  int aborted = 0 ;
  int committed = 0;
  int size_locators = 0; 
  for(int i=0; i< size; i++)
  {
    aborted += tx_data[i].n_aborted;
    committed += tx_data[i].n_committed;
    size_locators += tx_data[i].next_locator;
  }
  printf("OFG-STM\tcommits\t%d\taborts\t%d\tlocators\t%d\n",committed, aborted,size_locators);

}

void print_stats(STMData* stm_data)
{
  int size = stm_data -> num_tr;
//  printf("size: %d\n",size);
  TX_Data* tx_data = stm_data -> tx_data;
  int aborted = 0 ;
  int committed = 0;
  for(int i=0; i< size; i++)
  {
    aborted += tx_data[i].n_aborted;
    committed += tx_data[i].n_committed;
  }
  printf("\n\nTotal Aborts: %d Total Commits: %d\n\n\n", aborted, committed);

  int total = 0;
  //printf("nobjects: %d\n",stm_data -> n_objects);
  for (int i = 0; i < stm_data -> n_objects; i++)
  {
    Locator *loc = &stm_data-> locators[stm_data -> vboxes[i]];
    //print_locator(stm_data,loc);
    //printf("i: %d \n", stm_data -> vboxes[i] );
    //if (stm_data->tr_state[loc->owner] != COMMITTED)
    //{
    //  printf("print stats not commited\n");
   //  print_locator(stm_data,loc);
   // }
   // assert(stm_data->tr_state[loc->owner] == COMMITTED);
   if (stm_data->tr_state[loc->owner] == COMMITTED)
   {
    total += *loc -> new_version;
   } else{
    total += *loc -> old_version;
   }
  }
  printf("Total data: %d\n",total);
}

