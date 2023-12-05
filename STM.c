#include "STM.h"
#include <stdlib.h>
#include <stdio.h>

STMData* STM_start(int numObjects, int numTransactions, int numLocators)
{
    STMData *meta_data = malloc(sizeof(STMData));
    meta_data-> objects = malloc(numObjects * sizeof(Locator));
    meta_data-> vboxes = malloc(numObjects * sizeof(Locator*));
    meta_data-> tr_state = malloc(numTransactions * sizeof(ushort));
    meta_data-> locators = malloc(numLocators * numTransactions * sizeof(Locator));
    meta_data -> num_locators = numLocators;
    meta_data -> tx_data  = malloc(numTransactions * sizeof(TX_Data));
    return meta_data;
}

TX_Data* TX_Init(STMData* stm_data){
    TX_Data *d = stm_data -> tx_data+1;
    d-> tr_id = 1;
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

int* TX_Open_Write(STMData* stm_data, TX_Data* tx_data, uint object)
{

    Locator *locator = stm_data -> vboxes[object];
    Locator *new_locator = TX_new_locator(stm_data,tx_data);
    switch (stm_data->tr_state[locator -> owner]) {
            case COMMITTED:
              new_locator-> old_version =  locator->new_version;
              break;
            case ABORTED:
              new_locator->old_version =  locator->old_version;
              break;
            case ACTIVE: 
              new_locator->old_version = locator->old_version;
              break;
            default:
                printf("TX_Read: invalid tr state!\n");
                exit(0);
          }
    __sync_val_compare_and_swap(&stm_data -> vboxes[object],locator ,new_locator);
    
    return new_locator->new_version; 
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
    return *version; 
}


