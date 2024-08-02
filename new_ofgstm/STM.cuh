//#ifndef STM_API_H
//#define STM_API_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <malloc.h>

#define BACKOFF 10
#define WriteSetSize	200
#define ReadSetSize		200

#define MAX_LOCATORS 10000

#define ACTIVE      1
#define COMMITTED   2
#define ABORTED     3

typedef struct Locator_
{
    int owner;
    int object;
	int* new_version;
	int* old_version;
    int id;
} Locator;

typedef struct readSet_
{
	unsigned short size;
    int locator[ReadSetSize];
    unsigned int object[ReadSetSize];
    int* value[ReadSetSize];
    int id[ReadSetSize];
} ReadSet;

typedef struct writeSet_
{
    unsigned short size;
    int objects[WriteSetSize];
    int locators[WriteSetSize];
} WriteSet;

typedef struct TX_Data_
{
    unsigned int tr_id;
    int next_locator;
    int* locator_queue;
    ReadSet read_set;
    WriteSet write_set;
    unsigned short n_aborted;
    unsigned short n_committed; // maximum 1
    int cm_enemy;
    unsigned int cm_aborts;
    int cm_enemies[WriteSetSize];
    int enemies_size;
} TX_Data;

typedef struct STMData_
{
	//Locator* objects;
    int n_objects;
   // int* objects_data;
    int* vboxes;
	int* tr_state;
    Locator* locators;
    int* locators_data;
    unsigned short num_locators;
    unsigned short num_tr;
    TX_Data* tx_data;
} STMData;

STMData* STM_start(int numObjects, int numTransactions, int numLocators);
STMData* STM_copy_to_device(STMData* stm_data);
void STM_copy_from_device(STMData* d_stm_data, STMData* stm_data);

__device__ TX_Data* TX_Init(STMData* stm_data, int tx_id, int* locator_queue);
__device__ void TX_Start(STMData* stm_data, TX_Data* d);
__device__ int TX_new_locator(STMData* stm_data, TX_Data* tx_data);
__device__ int TX_validate_readset(STMData* stm_data, TX_Data* tx_data);
__device__ int TX_commit(STMData* stm_data, TX_Data* tx_data);
__device__ int* TX_Open_Write(STMData* stm_data, TX_Data* tx_data, uint object);
__device__ int TX_Open_Read(STMData* stm_data, TX_Data* tx_data, uint object);
__device__ void TX_abort_tr(STMData* stm_data, TX_Data* tx_data);
__device__  int TX_contention_manager(STMData* stm_data, TX_Data* tx_data,unsigned int me, unsigned int enemy);
__device__ void TX_garbage_collect(STMData* stm_data, TX_Data* tx_data);

void init_locators(STMData* stm_data,int num_tx, int num_locators);
void init_objects(STMData* stm_data,int num_objects, int value);
void fix_pointers_locators(STMData* stm_data, int* locators_data);

void print_vboxes(STMData* stm_data);
__host__ __device__ void print_tr_state(int tr_state);
__host__ __device__ void print_locator(STMData* stm_data,Locator *locator);
void print_stats(STMData* stm_data);
void print_data(STMData* stm_data);

//#endif