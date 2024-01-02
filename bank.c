////////////////////
////	BS		////
////////////////////

#include <time.h>
#include <unistd.h>

#include "STM.h"
#include <stdlib.h>
#include <pthread.h>

int *flag;

#define KERNEL_DURATION 5
#define DISJOINT 0

__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}


__device__ int waitMem;

__global__ void bank_kernel(int *flag, unsigned int seed, float prRead, unsigned int roSize, unsigned int txSize, unsigned int dataSize, 
								unsigned int threadNum, VertionedDataItem* data, TXRecord* record, TMmetadata* metadata, Statistics* stats, time_rate* times)
{
	local_metadata txData;
	bool result;

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	long mod = 0xFFFF;
	long rnd;
	long probRead;// = prRead * 0xFFFF;

	uint64_t state = seed+id;
	
	int value=0;
	int addr;
	//profile metrics
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx;
	long long int stop_aborted_tx=0, wastedTime=0;
	long long int start_time_total;

	long int updates=0, reads=0;
	//dijoint accesses variables
#if DISJOINT
	int min, max;
	min = dataSize/threadNum*id;
	max = dataSize/threadNum*(id+1)-1;
#endif

	while((*flag & 1)==0)
	{
		waitMem = *flag;
		wastedTime=0;
		///////
		//decide whether the thread will do update or read-only tx
		if(get_lane_id()==0)
		{
			rnd = RAND_R_FNC(state) & mod;
		}
		rnd = __shfl_sync(0xffffffff, rnd, 0);
		probRead = prRead * 0xFFFF;
		///////
		start_time_total = clock64();
		do
		{	
			start_time_tx = clock64();
			TXBegin(*metadata, &txData);
			
			//Read-Only TX
			if(rnd < probRead)
			{
				value=0;
				for(int i=0; i<dataSize && txData.isAborted==false; i++)//for(int i=0; i<roSize && txData.isAborted==false; i++)//
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					value+=TXReadOnly(data, i, &txData);
				}
				if(txData.isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
				//if(value != 10*dataSize)
				//	printf("T%d found an invariance fail: %d\n", id, value);
			}
			//Update TX
			else
			{
/*				for(int i=0; i<max(txSize,roSize) && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					if(i<roSize)
						value = TXRead(data, addr, &txData);
					if(i<txSize)
						TXWrite(data, value+(1), addr, &txData);
*/
				for(int i=0; i<txSize && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					value = TXRead(data, addr, &txData); 
					TXWrite(data, value-(1), addr, &txData);	

			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					value = TXRead(data, addr, &txData); 
					TXWrite(data, value+(1), addr, &txData);
				}
				if(txData.isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
			}
			start_time_commit = clock64(); 
  			result=TXCommit(id,record,data,metadata,txData,stats,times);
  			stop_time_commit = clock64();
  			if(!result)
			{
				stop_aborted_tx = clock64();
				wastedTime += stop_aborted_tx - start_time_tx;
			}
			stop_time_tx = clock64();
		}
		while(!result);
		atomicAdd(&(stats->nbCommits), 1);
		if(txData.ws.size==0)
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

	VertionedDataItem *h_data;
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
	prRead 				= (atoi(argv[argCnt++])/100.0);
	roSize 				= atoi(argv[argCnt++]);
	threadSize			= atoi(argv[argCnt++]);
	verbose				= atoi(argv[argCnt++]);

#if DISJOINT
	dataSize=100*blockNum*threads_per_block;
#endif
	
	int total_threads = blockNum*threads_per_block;
	h_times = (time_rate*) calloc(total_threads,sizeof(time_rate));
	h_stats = (Statistics*)calloc(1,sizeof(Statistics));
	
    STMData* stm_data = STM_start(dataSize, total_threads, MAX_LOCATORS); 
    init_objects(stm_data,dataSize,10);
    init_locators(stm_data,total_threads,MAX_LOCATORS);

	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
  	CUDA_CHECK_ERROR(cudaMallocManaged(&flag, sizeof(int)), "Could not alloc");
  	*flag = 0;

	cudaEventRecord(start); 
	bank_kernel<<<gridDist, blockDist>>>(flag, seed, prRead, roSize, threadSize, dataSize, blockNum*threads_per_block, d_data, records, metadata, d_stats, d_times);
  	cudaEventRecord(stop);
		
	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

	CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");
  	
  	cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;

  	
  	getKernelOutput(h_stats, h_times, blockNum*threads_per_block, peak_clk, totT_ms, verbose);
	TXEnd(dataSize, h_data, &d_data, &records, &metadata);

	free(h_stats);
	free(h_times);
	cudaFree(d_stats);
	cudaFree(d_times);
	
	return 0;
}
