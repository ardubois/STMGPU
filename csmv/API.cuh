#ifndef STM_API_H
#define STM_API_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#define MaxWriteSetSize		200
#define MaxReadSetSize		200

#define TXRecordSize		65536
#define MaxVersions 		10

//#define threads_per_block	256

#define bufDec(x)	(x-1+MaxVersions) % MaxVersions
#define bufInc(x)	(x+1) % MaxVersions
#define advance_pointer(x) (x+1) % TXRecordSize 	
#define decrease_pointer(x) (x-1+TXRecordSize) % TXRecordSize

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

#define RAND_R_FNC(seed) ({ \
    uint64_t next = seed; \
    uint64_t result; \
    next *= 1103515245; \
    next += 12345; \
    result = (uint64_t) (next / 65536) % 2048; \
    next *= 1103515245; \
    next += 12345; \
    result <<= 10; \
    result ^= (uint64_t) (next / 65536) % 1024; \
    next *= 1103515245; \
    next += 12345; \
    result <<= 10; \
    result ^= (uint64_t) (next / 65536) % 1024; \
    seed = next; \
    result; \
})


typedef struct globaldata_
{
	volatile ushort head_ptr;
	volatile ushort tail_ptr;
	volatile int value[MaxVersions];
	volatile uint version[MaxVersions];
} VertionedDataItem;

typedef struct metadata_
{
	ushort tp;
	ushort r_hp;
	int w_hp;
} TMmetadata;

typedef struct TxRecord_
{
	uint transactionNumber;
	ushort n_writes;
	uint writeSet[MaxWriteSetSize];
	//bool recordCommitted;
} TXRecord;

typedef struct readSet_
{
	ushort size;
	uint addrs[MaxReadSetSize];
} readSet;

typedef struct writeSet_
{
	ushort size;
	uint addrs[MaxWriteSetSize];
	int value[MaxWriteSetSize];
} writeSet;

typedef struct scoreboard_
{
	uint volatile valid_entry;
	uint volatile lane_result[32];
} warpResult;

typedef struct Statistics_
{
	int nbCommits;
	int nbAbortsRecordAge;
	int nbAbortsReadWrite;
	int nbAbortsWriteWrite;
	int nbAbortsDataAge;
} Statistics;

typedef struct times_
{
	long long int runtime;
	long long int commit;
	long long int dataWrite;
	long long int validation;
	long long int recordWrite;
	long long int wait;
	long int comparisons;
	long int nbReadOnly;
	long int nbUpdates;
} time_rate;

cudaError_t TXInit(int* dataArray, uint dataSize, uint threadNum, VertionedDataItem** host_data, VertionedDataItem** d_data, readSet** d_rs, writeSet** d_ws, TXRecord** d_records, warpResult** d_wRes);

void TXEnd(int dataSize, VertionedDataItem* host_data, VertionedDataItem** d_data, readSet** d_rs, writeSet** d_ws, warpResult** d_wRes);

__device__ uint TXBegin(uint tid, writeSet* ws, readSet* rs);

__device__ bool TXWrite(VertionedDataItem* data, int value, int addr, writeSet* ws, uint tid);

__device__ int TXRead(VertionedDataItem* data, int addr, uint timestamp, readSet* rs, writeSet* ws, uint tid, bool* isAborted);

__device__ int TXReadOnly(VertionedDataItem* data, int addr, uint timestamp, readSet* rs, writeSet* ws, uint tid, bool* isAborted);

__device__ int TXAddToRecord(TMmetadata* metadata, TXRecord* TxRecords, readSet* read_log, writeSet* write_log, Statistics* stats, time_rate* times, int timestamp, int tid);

__device__ void TXWriteBack(int newtimestamp, VertionedDataItem* data, writeSet write_log);


#endif
