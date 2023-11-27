STMData* STM_start(int numObjects, int numTransactions, int numLocators)
{
    STMData *meta_data = malloc(sizeof(STMData));
    meta_data-> objects = malloc(numObjects * sizeof(Locator));
    meta_data-> vboxes = malloc(numObjects * sizeof(*Locator));
    meta_data-> tr_state = malloc(numTransactions * sizeof(ushort));
    meta_data-> committed_tr_state = malloc(numTransactions * sizeof(ushort));

    return meta_data;
}

TX_Data* TX_Init(STMData* stm_data){
    TX_Data d;
    return &d;
}

int TX_Read(STMData stm_data, TX_Data tx_data, uint object)
{
    
}