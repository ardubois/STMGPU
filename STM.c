STMData* STM_start(int numObjects, int numTransactions, int numLocators)
{
    STMData *meta_data = malloc(sizeof(STMData));
    meta_data-> objects = malloc(numObjects * sizeof(Locator));
    meta_data-> vboxes = malloc(numObjects * sizeof(*Locator));
    meta_data-> tr_state = malloc(numTransactions * sizeof(ushort));
    meta_data-> committed_tr_state = malloc(numTransactions * sizeof(ushort));
    meta_data-> locators = malloc(numLocators * numTransactions * size(Locator))
    meta_data -> num_locators = numLocators;
    return meta_data;
}

TX_Data* TX_Init(STMData* stm_data){
    TX_Data d;
    d-> tr_id = 1;
    d-> next_locator = 0;
    return &d;
}

Locator* TX_new_locator(STMData stm_data, TX_Data tx_data)
{
  Locator* locator = meta_data -> locators;
  locator += (d->tr_id * stm_data -> num_locators) + tx_data-> next_locator;
  stm_data -> next_locator++;
  return locator;
}

int TX_Open_Read(STMData stm_data, TX_Data tx_data, uint object)
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
    
    ReadSet* read_set = tx_data-> read_set;
    int size = tx_data-> size;
    read_set -> locator[size] = locator;
    read_set -> value[size] = version;
    read_set -> object[size] = object;
    read_set -> size ++;
    return *version; 
}


int* TX_Open_Write(STMData stm_data, TX_Data tx_data, uint object)
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
    
    ReadSet* read_set = tx_data-> read_set;
    int size = tx_data-> size;
    read_set -> locator[size] = locator;
    read_set -> value[size] = version;
    read_set -> object[size] = object;
    read_set -> size ++;
    return *version; 
}