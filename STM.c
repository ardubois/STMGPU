#include "STM.h"



int tr_id_gen=0;

STMData* STM_start(int numObjects, int numTransactions, int numLocators)
{
    STMData *meta_data = malloc(sizeof(STMData));
    meta_data-> objects = malloc(numObjects * sizeof(Locator));
    meta_data -> n_objects = numObjects;
    meta_data-> objects_data = malloc(2*numObjects * sizeof(int));
    meta_data-> vboxes = malloc(numObjects * sizeof(Locator*));
    meta_data-> tr_state = malloc(numTransactions * sizeof(ushort)+2); // 1 for the always committed Tr and 1 for the always aborted 
    meta_data-> locators = malloc(numLocators * numTransactions * sizeof(Locator));
    meta_data-> locators_data = malloc(2*numLocators * numTransactions * sizeof(int));
    meta_data -> num_locators = numLocators;
    meta_data -> tx_data  = malloc(numTransactions * sizeof(TX_Data));
    meta_data -> num_tr = numTransactions;
    meta_data -> tr_state[numTransactions] = COMMITTED;
    meta_data -> tr_state[numTransactions+1] = ABORTED;
    return meta_data;
}

TX_Data* TX_Init(STMData* stm_data)
{
   unsigned int tx_id;

   do{

      tx_id = tr_id_gen;
   }while(! __sync_bool_compare_and_swap(&tr_id_gen, tx_id ,tx_id+1));
   

    TX_Data *d = &stm_data -> tx_data[tx_id];
    d-> tr_id = tx_id;
    d-> next_locator = 0;
    d -> read_set.size =0;
    d -> write_set.size = 0;
    d -> n_aborted = 0;
    d -> n_committed = 0;
    d -> cm_enemy = -1;
    d -> cm_aborts = 0;
    stm_data -> tr_state[d->tr_id] = ACTIVE;

    return d;
}

Locator* TX_new_locator(STMData* stm_data, TX_Data* tx_data)
{
  Locator* locator = stm_data -> locators;
  locator += (tx_data->tr_id * stm_data -> num_locators) + tx_data-> next_locator;
  tx_data -> next_locator++;
  if(tx_data -> next_locator == MAX_LOCATORS)
    {
     printf("Max locators reached");
     exit(0);
    }
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
       if(stm_data -> vboxes[read_set->object[i]] == read_set -> locator[i])
       {
          if(stm_data->tr_state[read_set -> locator[i]->owner] == COMMITTED)
          {
            if(!(read_set -> locator[i]->new_version == read_set -> value[i]))
            {
              return 0;
            }
            continue;
          }
          if(stm_data->tr_state[read_set -> locator[i]->owner] == ABORTED ||stm_data->tr_state[read_set -> locator[i]->owner] == ACTIVE)
          {
            if(!(read_set -> locator[i]-> old_version== read_set -> value[i]))
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
     if( __sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,COMMITTED))
     {
      tx_data -> n_committed ++;
      return 1;
     }
  }
    __sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
    assert(stm_data->tr_state[tx_data->tr_id]==ABORTED);
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
              if(TX_contention_manager(stm_data,tx_data, new_locator->owner,locator->owner))
              {
                if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
                {
                  if(__sync_bool_compare_and_swap(&stm_data->tr_state[locator -> owner],ACTIVE ,ABORTED))
                  {
                   *new_locator->old_version = *locator->old_version;
                   *new_locator-> new_version = *new_locator-> old_version;
                  } else {
                  __sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
                  assert(stm_data->tr_state[tx_data->tr_id]==ABORTED);
                  tx_data -> next_locator--;
                 }
                } else {
                  tx_data -> next_locator--;
                }
              } else{
                   __sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
                   assert(stm_data->tr_state[tx_data->tr_id]==ABORTED);
                   tx_data -> next_locator--;
              }
                
              break;
            default:
                printf("TX_Read: invalid tr state!\n");
                exit(0);
          }
   
    if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
         if(__sync_bool_compare_and_swap(&stm_data -> vboxes[object],locator ,new_locator)){
            WriteSet* write_set = &tx_data-> write_set;
            int size = tx_data-> write_set.size;
            write_set -> locators[size] = new_locator;
            write_set -> size ++;
           print_locator(stm_data,new_locator);
            if(TX_validate_readset(stm_data,tx_data))
              {
               return new_locator->new_version;
              }
            else
              {
                __sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);
              }
         } else { __sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ACTIVE ,ABORTED);}
  
    assert(stm_data->tr_state[tx_data->tr_id] != ACTIVE) ;    
    return 0; 
}

int TX_contention_manager3(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
    if(enemy < me)
       return 1;
    return 0;
}

int TX_contention_manager2(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
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

int TX_contention_manager(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy)
{
  if(tx_data->n_aborted > BACKOFF)
  { 
    return 1;
    
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



void TX_abort_tr(STMData* stm_data, TX_Data* tx_data){


  for (int i = 0; i < tx_data->write_set.size; i ++)
  {
    if(!__sync_bool_compare_and_swap(&tx_data -> write_set.locators[i]->owner, tx_data->tr_id , (stm_data -> num_tr)+1))
    {
      printf("nao deveria\n owner = %d, id = %d,locaotr: %p\n",tx_data -> write_set.locators[i]->owner,  tx_data->tr_id,tx_data ->write_set.locators[i]);
      exit(0);
    }
  }

  assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
  assert(__sync_bool_compare_and_swap(&stm_data->tr_state[tx_data->tr_id],ABORTED,ACTIVE));

  tx_data-> read_set.size = 0;
  tx_data -> write_set.size = 0;
  tx_data -> n_aborted ++;
}

void init_objects(STMData* stm_data,int num_objects)
{
  stm_data -> tr_state[stm_data->num_tr] = COMMITTED;
  for(int i=0;i<num_objects;i++)
  {
    stm_data->objects_data[2*i] = 100;
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

void print_vboxes(STMData* stm_data, Locator **vboxes)
{
   printf("d");
}


void print_tr_state(int tr_state)
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
void print_locator(STMData* stm_data,Locator *locator)
{
  printf("Locator %p:\n", locator);
  printf("- owner: %d\n",locator->owner);
  printf("- state: %d, ", stm_data-> tr_state[locator->owner]);
  print_tr_state(stm_data-> tr_state[locator->owner]);
  printf("\n");
  printf("- new_version %d\n", *locator->new_version);
  printf("- old_version %d\n", *locator->old_version);
}




void print_stats(STMData* stm_data)
{
  int size = stm_data -> num_tr;
  TX_Data* tx_data = stm_data -> tx_data;
  int aborted = 0 ;
  int committed = 0;
  for(int i=0; i< size; i++)
  {
    aborted += tx_data[i].n_aborted;
    committed += tx_data[i].n_committed;
  }
  printf("Total Aborts: %d Total Commits: %d\n", aborted, committed);

  int total = 0;
  for (int i = 0; i < stm_data -> n_objects; i++)
  {
    Locator *loc = stm_data -> vboxes[i];
    assert(stm_data->tr_state[loc->owner] == COMMITTED);
    total += *loc -> new_version;
  }
  printf("Total data: %d\n",total);
}

