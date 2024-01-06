////////////////////
////	BS		////
////////////////////

#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include "STM.h"
#include <stdlib.h>
#include <pthread.h>

pthread_barrier_t barrier;

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
	long long int total;
	long long int runtime;
	long long int commit;
	long long int dataWrite;
	long long int val1;
	long long int val2;
	long long int recordWrite;
	long long int wastedTime;
	long int comparisons;
	long int nbReadOnly;
	long int nbUpdates;	
} time_rate;

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

int *flag;

#define KERNEL_DURATION 5
#define DISJOINT 0



int waitMem;

struct args {
    int *flag;
	unsigned int seed;
	int prRead;
	unsigned int roSize; 
	unsigned int txSize; 
	unsigned int dataSize; 
	unsigned int threadNum; 
	STMData* stm_data; 
	Statistics* stats; 
	time_rate* times;
};


void* bank_kernel(void *p)
{
	//bool result;

	struct args* args = (struct args*) p;

	int *flag = args -> flag;
	unsigned int seed = args -> seed; 
	int prRead =  args -> prRead;
	unsigned int roSize = args -> roSize;
	unsigned int txSize = args -> txSize;
	unsigned int dataSize = args -> dataSize;
    unsigned int threadNum = args -> threadNum;
	STMData* stm_data = args -> stm_data;
	Statistics* stats = args -> stats;
	time_rate* times = args -> times;

	
	long mod = 0xFFFF;
	long rnd;
	long probRead=prRead;// = prRead * 0xFFFF;

    TX_Data* tx_data = TX_Init(stm_data);
    int id =  tx_data -> tr_id; 

	uint64_t state = seed+id;
	
	int value=0;
	int read;
	int  addr1, addr2;
	//profile metrics
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx;
	long long int stop_aborted_tx=0, wastedTime=0;
	long long int start_time_total;

	long int updates=0, reads=0;
	//dijoint accesses variables


	while((*flag & 1)==0)
	{
		waitMem = *flag;
		wastedTime=0;
		///////
			rnd = (RAND_R_FNC(seed) %10) +1;
//        printf("rand %d  -  %d\n",rnd,txSize);
        //printf("rnd %d probread %d\n", rnd, probRead);
 
		start_time_total = clock();
		do
		{	
			start_time_tx = clock();
			TX_Start(stm_data,tx_data);
			
			//Read-Only TX
			if(rnd <= probRead)
			{
				value=0;
				for(int i=0; i<roSize && stm_data->tr_state[tx_data->tr_id] != ABORTED; i++)//for(int i=0; i<roSize && txData.isAborted==false; i++)//
				{
			
					addr1 = RAND_R_FNC(state)%dataSize;
					read=TX_Open_Read(stm_data,tx_data,addr1);
					if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
					{
						value += read;
					}
				}	
				if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
				{
					TX_abort_tr(stm_data,tx_data);
					__sync_add_and_fetch(&(stats->nbAbortsDataAge), 1);
					continue;
				}
				//if(value != 10*dataSize)
				//	printf("T%d found an invariance fail: %d\n", id, value);
			}
			//Update TX
			else
			{
                 // printf("write!\n");
				for(int i=0; i<txSize && stm_data->tr_state[tx_data->tr_id] != ABORTED; i++)
				{
					addr1 = RAND_R_FNC(state)%dataSize;
					addr2 = RAND_R_FNC(state)%dataSize;
					//printf("Address %d %d txsize %d!!!!!!!!!!!\n",addr1,addr2,txSize);
					int* ptr1 = TX_Open_Write(stm_data,tx_data,addr1);
            		if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
            		{
                		int* ptr2 = TX_Open_Write(stm_data,tx_data,addr2);
                		if(ptr2 !=0 )
                		{
                    		*ptr1 -= 1;
                    		*ptr2 += 1;
                        }
				    }
				}
				if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
				{
					TX_abort_tr(stm_data,tx_data);
					__sync_add_and_fetch(&(stats->nbAbortsDataAge), 1);
					continue;
				}
			}
			start_time_commit = clock(); 
  			TX_commit(stm_data,tx_data);
			if(stm_data->tr_state[tx_data->tr_id] == COMMITTED)
                        {//trans ++;
                          pthread_barrier_wait(&barrier);
                          TX_garbage_collect(stm_data,tx_data);
                          pthread_barrier_wait(&barrier);
                        }
  			stop_time_commit = clock();
  			if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
			{
				stop_aborted_tx = clock();
				wastedTime += stop_aborted_tx - start_time_tx;
			}
			stop_time_tx = clock();
		}
		while(stm_data->tr_state[tx_data->tr_id] != COMMITTED);
		//printf("commited\n!!!!!!!!!!!!!!!!!!!");
		__sync_add_and_fetch(&(stats->nbCommits), 1);
		if(tx_data -> write_set.size==0)
			reads++;
		else
			updates++;		

		times[id].total   += stop_time_tx - start_time_total;
		times[id].runtime += stop_time_tx - start_time_tx;
		times[id].commit  += stop_time_commit - start_time_commit;
		times[id].wastedTime	 += wastedTime;

	}
	times[id].nbReadOnly = reads;
	times[id].nbUpdates  = updates;
}

void getKernelOutput(Statistics *h_stats, time_rate *h_times, uint threadNum, int peak_clk, float totT_ms, uint verbose)
{
  	double avg_total=0, avg_runtime=0, avg_commit=0, avg_wb=0, avg_val1=0, avg_val2=0, avg_rwb=0, avg_comp=0, avg_waste=0;
	long int totUpdates=0, totReads=0;
	for(int i=0; i<threadNum; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_total   += h_times[i].total;
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_wb 		+= h_times[i].dataWrite;
		avg_val1	+= h_times[i].val1;
		avg_val2	+= h_times[i].val2;
		avg_rwb		+= h_times[i].recordWrite;
		avg_comp 	+= h_times[i].comparisons;
		avg_waste	+= h_times[i].wastedTime;
	
		totUpdates 	+= h_times[i].nbUpdates;
		totReads	+= h_times[i].nbReadOnly;
	}
	
	long int denom = (long)h_stats->nbCommits*peak_clk;
	avg_total	/= denom;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_wb 		/= denom;
	avg_val1 	/= denom;
	avg_val2 	/= denom;
	avg_rwb 	/= denom;
	avg_comp	/= h_stats->nbCommits;
	avg_waste	/= denom;

	float rt_commit=0.0, rt_wb=0.0, rt_val1=0.0, rt_val2=0.0, rt_rwb=0.0, dummy=0.0;
	rt_commit	=	avg_commit / avg_runtime;
	rt_wb	 	=	avg_wb / avg_runtime;
	rt_val1	 	=	avg_val1 / avg_runtime;
	rt_val2	 	=	avg_val2 / avg_runtime;
	rt_rwb	 	=	avg_rwb / avg_runtime;

	int nbAborts = h_stats->nbAbortsDataAge + h_stats->nbAbortsRecordAge + h_stats->nbAbortsReadWrite;

	
	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nAbortDataAge\t%f %%\nAbortRecAge\t%f %%\nAbortReadWrite\t%f %%\nAbortPreVal\t%f %%\n\nTotal\t\t%f\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\nWaitTime\t%f\t%.2f%%\nPreValidation\t%f\t%.2f%%\n1stValidation\t%f\t%.2f%%\nRecInsertVals\t%f\t%.2f%%\nRecInsert\t%f\t%.2f%%\nWriteBack\t%f\t%.2f%%\nWaste\t\t%f\n\nComparisons\t%f\nTotalUpdates\t%d\nTotalReads\t%d\n", 
			(float)nbAborts/(nbAborts+h_stats->nbCommits)*100.0,
			h_stats->nbCommits/totT_ms*1000.0,
			(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsWriteWrite/(nbAborts+h_stats->nbCommits)*100.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			dummy,
			dummy,
			dummy,
			dummy,
			avg_val1,
			rt_val1*100.0,
			avg_val2,
			rt_val2*100.0,
			avg_rwb,
			rt_rwb*100.0,
			avg_wb,
			rt_wb*100.0,
			avg_waste,
			avg_comp,
			totUpdates,
			totReads
			);
	else
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", 
			(float)nbAborts/(nbAborts+h_stats->nbCommits)*100.0,
			h_stats->nbCommits/totT_ms*1000.0,
			(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsWriteWrite/(nbAborts+h_stats->nbCommits)*100.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			dummy,
			dummy,
			dummy,
			dummy,
			avg_val1,
			rt_val1*100.0,
			avg_val2,
			rt_val2*100.0,
			avg_rwb,
			rt_rwb*100.0,
			avg_wb,
			rt_wb*100.0,
			avg_waste
			);
}

int main(int argc, char *argv[])
{
	unsigned int blockNum, threads_per_block, roSize, threadSize, dataSize, seed, verbose;
	float prRead;

	STMData* metadata;  
	
	Statistics *h_stats;
	time_rate  *h_times;

  	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) nb bank accounts               \n"
	  "  2) client config - nb threads     \n"
	  "  3) client config - nb blocks      \n"
	  "  4) prob read TX                   \n"
	  "  5) read TX Size                   \n"
	  "  6) update TX Size                 \n"
	  "  7) verbose		                   \n"
	"";
	const int NB_ARGS = 8;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	seed 				= 1;
	dataSize			= atoi(argv[argCnt++]);
	threads_per_block	= atoi(argv[argCnt++]);
	blockNum		 	= atoi(argv[argCnt++]);
	prRead 				= atoi(argv[argCnt++]);
	roSize 				= atoi(argv[argCnt++]);
	threadSize			= atoi(argv[argCnt++]);
	verbose				= atoi(argv[argCnt++]);

//#if DISJOINT
//	dataSize=100*blockNum*threads_per_block;
//#endif
	
	int total_threads = blockNum*threads_per_block;
	h_times = (time_rate*) calloc(total_threads,sizeof(time_rate));
	h_stats = (Statistics*)calloc(1,sizeof(Statistics));
	
    STMData* stm_data = STM_start(dataSize, total_threads, MAX_LOCATORS); 
    init_objects(stm_data,dataSize,10);
    init_locators(stm_data,total_threads,MAX_LOCATORS);

	float tKernel_ms = 0.0, totT_ms = 0.0;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	//printf("aqui.\n");
	flag = calloc(1,sizeof(int));
  	*flag = 0;

	//cudaEventRecord(start); 
//	printf("aqui.\n");

    struct args* args = calloc(1, sizeof(struct args));

	args -> flag = flag;
	args -> seed = seed;
    args -> prRead =  prRead;
	args -> roSize = roSize;
	args ->  txSize = threadSize;
    args -> dataSize = dataSize;
	args ->  threadNum =  total_threads;
	args -> stm_data = stm_data;
	args -> stats = h_stats;
	args -> times = h_times;

	pthread_barrier_init(&barrier, NULL, total_threads);

    pthread_t threads[total_threads];

  for(int i=0; i< total_threads; i++)
   {
    pthread_create(&threads[i],NULL, bank_kernel, args);
   }
  
  //pthread_create(&tid1, NULL, foo, stm_data); 
  
  //for(int i=0; i< total_threads; i++)
   //{
   // pthread_join(threads[i],NULL);
  // }

	//bank_kernel(args);
  	//cudaEventRecord(stop);
		
	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

  	//cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;
    int peak_clk=1;
  	
  	getKernelOutput(h_stats, h_times, blockNum*threads_per_block, peak_clk, totT_ms, verbose);
	//TXEnd(dataSize, h_data, &d_data, &records, &metadata);

	free(h_stats);
	free(h_times);
	print_stats(stm_data);
	return 0;
}
