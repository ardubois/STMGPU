#include "STM.h"
#include <stdlib.h>
#include <stdio.h>

STMData* STM_start(int numObjects, int numTransactions, int numLocators)
{
    STMData *meta_data = malloc(sizeof(STMData));
    meta_data-> objects = malloc(numObjects * sizeof(Locator));
    meta_data-> objects_data = malloc(2*numObjects * sizeof(int));
    meta_data-> vboxes = malloc(numObjects * sizeof(Locator*));
    meta_data-> tr_state = malloc(numTransactions * sizeof(ushort)+1);
    meta_data-> locators = malloc(numLocators * numTransactions * sizeof(Locator));
    meta_data-> locators_data = malloc(2*numLocators * numTransactions * sizeof(int));
    meta_data -> num_locators = numLocators;
    meta_data -> tx_data  = malloc(numTransactions * sizeof(TX_Data));
    meta_data -> num_tr = numTransactions;
    return meta_data;
}

TX_Data* TX_Init(STMData* stm_data){
    unsigned int tx_id = 0;

    TX_Data *d = &stm_data -> tx_data[tx_id];
    d-> tr_id = tx_id;
    d-> next_locator = 0;
    d -> read_set.size =0;
    d -> write_set.size = 0;
    return d;
}

Locator* TX_new_locator(STMData* stm_data, TX_Data* tx_data)
{
  Locator* locator = stm_data -> locators;
  locator += (tx_data->tr_id * stm_data -> num_locators) + tx_data-> next_locator;
  tx_data -> next_locator++;
  return locator;
}

int TX_validate_readset(STMData* stm_data, TX_Data* tx_data)
{
  ReadSet* read_set = &tx_data-> read_set;
  int size = tx_data-> read_set.size;
  //int* t = read_set -> locator[size]->new_version;
  //int* t = read_set -> value[size];
  for (int i=0;i<size;i++)
  {
       if(stm_data -> vboxes[read_set->object[size]] == read_set -> locator[size])
       {
          if(stm_data->tr_state[read_set -> locator[size]->owner] == COMMITTED)
          {
            if(!(read_set -> locator[size]->new_version == read_set -> value[size]))
            {
              return 0;
            }
          }
          if(stm_data->tr_state[read_set -> locator[size]->owner] == ABORTED ||stm_data->tr_state[read_set -> locator[size]->owner] == ACTIVE)
          {
            if(!(read_set -> locator[size]-> old_version== read_set -> value[size]))
            {
              return 0;
            }
          }

       }
  }
  return 1;
}

int TX_commit(STMData* stm_data, TX_Data* tx_data)
{
  if(TX_validate_readset(stm_data,tx_data))
  {
     return __sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,COMMITTED);
  }
     return 0;         
}
int* TX_Open_Write(STMData* stm_data, TX_Data* tx_data, uint object)
{

    Locator *locator = stm_data -> vboxes[object];
    Locator *new_locator = TX_new_locator(stm_data,tx_data);
    new_locator ->owner = tx_data->tr_id;
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
              if(__sync_bool_compare_and_swap(&stm_data->tr_state[locator -> owner],ACTIVE ,ABORTED))
              {
                 *new_locator->old_version = *locator->old_version;
                 *new_locator-> new_version = *new_locator-> old_version;
              } else
                 {
                  __sync_bool_compare_and_swap(&stm_data->tr_state[new_locator -> owner],ACTIVE ,ABORTED);
                 }
              break;
            default:
                printf("TX_Read: invalid tr state!\n");
                exit(0);
          }
    if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
         if(__sync_bool_compare_and_swap(&stm_data -> vboxes[object],locator ,new_locator))
            if(TX_validate_readset(stm_data,tx_data))
              {
               return new_locator->new_version;
              }
            else
              {
                __sync_bool_compare_and_swap(&stm_data->tr_state[new_locator -> owner],ACTIVE ,ABORTED);
              }
    return 0; 
}

int TX_Open_Read(STMData* stm_data, TX_Data* tx_data, uint object)
{
    int* version;
    Locator *locator = stm_data -> vboxes[object];
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
                exit(0);
          }
    
    ReadSet* read_set = &tx_data-> read_set;
    int size = tx_data-> read_set.size;
    read_set -> locator[size] = locator;
    read_set -> value[size] = version;
    read_set -> object[size] = object;
    read_set -> size ++;
    if(TX_validate_readset(stm_data,tx_data))
              {
               return *version;
              }
    return 0; 
}

void init_objects(STMData* stm_data,int num_objects)
{
  stm_data -> tr_state[stm_data->num_tr] = COMMITTED;
  for(int i=0;i<num_objects;i++)
  {
    stm_data->objects_data[2*i] = 10;
    stm_data->objects_data[2*i+1] = 0;
    stm_data-> objects[i].new_version = &stm_data->objects_data[2*i];
    stm_data-> objects[i].old_version = &stm_data->objects_data[2*i+1];
    stm_data-> objects[i].owner = stm_data->num_tr;
    stm_data->vboxes[i] = &stm_data-> objects[i];
  
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
    
  }
}

int main()
{
  int num_objects = 2;
  int num_locators = 10;
  int num_tx = 1;
  STMData* stm_data = STM_start(num_objects, num_tx, num_locators); 
  init_objects(stm_data,num_objects);
  Locator* loc = stm_data->vboxes[0];
  
  init_locators(stm_data,num_tx,num_locators);

  TX_Data* tx_data = TX_Init(stm_data);
  int value=TX_Open_Read(stm_data,tx_data,0);
  printf("value %d\n",value);
  int* ptr_value=TX_Open_Write(stm_data,tx_data,1);
  printf("value %d\n",*ptr_value);
  TX_commit(stm_data,tx_data);
  loc = stm_data->vboxes[1];
  printf("Locator new_value: %d\n",*loc->new_version);
  
}