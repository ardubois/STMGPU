#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define BACKOFF 200
#define WriteSetSize	10
#define ReadSetSize		10

#define MAX_LOCATORS 60

#define ACTIVE      1
#define COMMITTED   2
#define ABORTED     3

typedef struct Locator_
{
	unsigned int owner;
	int* new_version;
	int* old_version;
} Locator;

typedef struct readSet_
{
	unsigned short size;
    Locator* locator[ReadSetSize];
    unsigned int object[ReadSetSize];
    int* value[ReadSetSize];
} ReadSet;

typedef struct writeSet_
{
    unsigned short size;
    Locator* locators[WriteSetSize];
} WriteSet;

typedef struct TX_Data_
{
    unsigned int tr_id;
    unsigned short next_locator;
    int* locator_queue;
    ReadSet read_set;
    WriteSet write_set;
    unsigned short n_aborted;
    unsigned short n_committed; // maximum 1
    unsigned int cm_enemy;
    unsigned int cm_aborts;
} TX_Data;

typedef struct STMData_
{
	Locator* objects;
    int n_objects;
    int* objects_data;
    Locator** vboxes;
	unsigned short* tr_state;
    Locator* locators;
    int* locators_data;
    unsigned short num_locators;
    unsigned short num_tr;
    TX_Data* tx_data;
} STMData;

STMData* STM_start(int numObjects, int numTransactions, int numLocators);
TX_Data* TX_Init(STMData* stm_data);
void TX_Start(STMData* stm_data, TX_Data* d);
Locator* TX_new_locator(STMData* stm_data, TX_Data* tx_data);
int TX_validate_readset(STMData* stm_data, TX_Data* tx_data);
int TX_commit(STMData* stm_data, TX_Data* tx_data);
int* TX_Open_Write(STMData* stm_data, TX_Data* tx_data, uint object);
int TX_Open_Read(STMData* stm_data, TX_Data* tx_data, uint object);
void TX_abort_tr(STMData* stm_data, TX_Data* tx_data);
int TX_contention_manager(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy);
void TX_garbage_collect(STMData* stm_data, TX_Data* tx_data);

void init_locators(STMData* stm_data,int num_tx, int num_locators);
void init_objects(STMData* stm_data,int num_objects, int value);

void print_vboxes(STMData* stm_data, Locator **vboxes);
void print_tr_state(int tr_state);
void print_locator(STMData* stm_data,Locator *locator);
void print_stats(STMData* stm_data);