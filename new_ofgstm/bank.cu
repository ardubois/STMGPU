////////////////////
////	BS		////
////////////////////

#include <time.h>
#include "STM.cuh"
#include "util.cuh"
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>

#define KERNEL_DURATION 5
#define DISJOINT 0

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


#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}


__device__ int waitMem;


__device__ float rand_() {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
       curandState state;
        curand_init(clock64(), i, 0, &state);

       return curand_uniform(&state);

}

__global__ void bank_kernel(int *flag, unsigned int seed, int prRead, unsigned int roSize, unsigned int txSize, unsigned int dataSize, 
								unsigned int threadNum, STMData* stm_data, Statistics* stats, time_rate* times)
{
	
	//bool result=false;

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	//long mod = 0xFFFF;
	int rnd;
	int probRead= prRead;

	
    int locator_queue[MAX_LOCATORS];
	TX_Data* tx_data = TX_Init(stm_data,id,locator_queue);
	
	int value=0;
	int addr,addr1,addr2;
	//profile metrics
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx;
	long long int stop_aborted_tx=0, wastedTime=0;
	long long int start_time_total;

	long int updates=0, reads=0;
	//dijoint accesses variables
	int read;
#if DISJOINT
	int min, max;
	int read;
	min = dataSize/threadNum*id;
	max = dataSize/threadNum*(id+1)-1;
#endif
  //int x=0;
	while((*flag & 1)==0)
	{ //  x++;
		waitMem = *flag;
		wastedTime=0;
		///////
		//decide whether the thread will do update or read-only tx
		

		///////
			rnd = ((int)(rand_()*10)) + 1;     //  (RAND_R_FNC(seed) %10) +1;
//        printf("rand %d  -  %d\n",rnd,txSize);
		///////
		start_time_total = clock64();
		do
		{	
			start_time_tx = clock64();
			//TXBegin(*metadata, &txData);
			TX_Start(stm_data,tx_data);
            //printf("1TXISABORTED %d\n", txData.isAborted);			
			//Read-Only TX
			//printf("rnd %d probread %d\n", rnd, probRead);
			int read_only = (rnd <= probRead); 
			if(rnd <= probRead)
			{
				//printf("rnd %d probread %d\n", rnd, probRead);
				value=0;
				for(int i=0; i<roSize && stm_data->tr_state[tx_data->tr_id] != ABORTED; i++)//for(int i=0; i<roSize && txData.isAborted==false; i++)//
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
				//	addr = RAND_R_FNC(state)%dataSize;
					addr = (int)(rand_()*dataSize);
			#endif
					read=TX_Open_Read(stm_data,tx_data,addr);
					if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
					{
						value += read;
					}
				}
				if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
				{
					TX_abort_tr(stm_data,tx_data);
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
//printf("txsize: %d\n",txSize);
//printf("TXISABORTED %d\n", txData.isAborted);
//printf("true %d\n", true);
//printf("false %d\n", false);
//result =0;
//printf("2TXISABORTED %d\n", txData.isAborted);
				for(int i=0; i<txSize && stm_data->tr_state[tx_data->tr_id] != ABORTED; i++)
				{
					addr1 = (int)(rand_()*dataSize);
                    addr2 = (int)(rand_()*dataSize);
				
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
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
			}
			start_time_commit = clock64(); 
			TX_commit(stm_data,tx_data);
			stop_time_commit = clock64();

			if(stm_data->tr_state[tx_data->tr_id] == COMMITTED)
                        {//trans ++;
                   //       __syncthreads();
				     if(!read_only)
                          TX_garbage_collect(stm_data,tx_data);
						 // printf("COMMITED: %d -- ABORTED %d\n",stm_data->tr_state[stm_data-> num_tr],stm_data->tr_state[stm_data-> num_tr+1]);
                     //     __syncthreads();
                        }
  			if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
			{
				TX_abort_tr(stm_data,tx_data);
					atomicAdd(&(stats->nbAbortsDataAge), 1);
				stop_aborted_tx = clock64();
				wastedTime += stop_aborted_tx - start_time_tx;
			}
			stop_time_tx = clock64();
		}
		while(stm_data->tr_state[tx_data->tr_id] != COMMITTED);
		atomicAdd(&(stats->nbCommits), 1);
		if(tx_data -> write_set.size==0)
			reads++;
		else
			updates++;		

		times[id].total   += stop_time_tx - start_time_total;
		times[id].runtime += stop_time_tx - start_time_tx;
		times[id].commit  += stop_time_commit - start_time_commit;
		times[id].wastedTime	 += wastedTime;

	}
	//printf("reads %d updates %d", reads,updates);
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

	//printf("--- nbaborts:  %d nbcommits: %d\n\n----------",h_stats->nbAbortsDataAge,h_stats->nbCommits);
	if(verbose)
	    printf("OFG-STM\nCommits: %d\n", h_stats->nbCommits);
	   /*printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nAbortDataAge\t%f %%\nAbortRecAge\t%f %%\nAbortReadWrite\t%f %%\nAbortPreVal\t%f %%\n\nTotal\t\t%f\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\nWaitTime\t%f\t%.2f%%\nPreValidation\t%f\t%.2f%%\n1stValidation\t%f\t%.2f%%\nRecInsertVals\t%f\t%.2f%%\nRecInsert\t%f\t%.2f%%\nWriteBack\t%f\t%.2f%%\nWaste\t\t%f\n\nComparisons\t%f\nTotalUpdates\t%d\nTotalReads\t%d\n", 
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
			);*/
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
	int prRead;
   
 //  printf("start progran\n");
	
	Statistics *h_stats, *d_stats;
	time_rate *d_times, *h_times;

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
  // printf("read para\n");
	seed 				= 1;
	dataSize			= atoi(argv[argCnt++]);
	threads_per_block	= atoi(argv[argCnt++]);
	blockNum		 	= atoi(argv[argCnt++]);
	prRead 				= atoi(argv[argCnt++]);
	roSize 				= atoi(argv[argCnt++]);
	threadSize			= atoi(argv[argCnt++]);
	verbose				= atoi(argv[argCnt++]);

#if DISJOINT
	dataSize=100*blockNum*threads_per_block;
#endif
	

	h_times = (time_rate*) calloc(blockNum*threads_per_block,sizeof(time_rate));
	h_stats = (Statistics*)calloc(1,sizeof(Statistics));
	

	//Select the GPU Device
	cudaError_t result;
	result = cudaSetDevice(0);
	if(result != cudaSuccess) fprintf(stderr, "Failed to set Device: %s\n", cudaGetErrorString(result));

	int peak_clk=1;
	cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
  	if (err != cudaSuccess) {printf("cuda err: %d at line %d\n", (int)err, __LINE__); return 1;}

	result = cudaMalloc((void **)&d_stats, sizeof(Statistics));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_times, blockNum*threads_per_block*sizeof(time_rate));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ratio: %s\n", cudaGetErrorString(result));
	
	

	dim3 blockDist(threads_per_block,1,1);
	dim3 gridDist(blockNum, 1, 1);

	
	cudaMemcpy(d_times, h_times, blockNum*threads_per_block*sizeof(time_rate), cudaMemcpyHostToDevice);
	cudaMemcpy(d_stats, h_stats, sizeof(Statistics), cudaMemcpyHostToDevice);

    int num_objects = dataSize;
	int num_locators = MAX_LOCATORS;
	int num_tx = threads_per_block * blockNum;
	STMData* stm_data = STM_start(num_objects, num_tx, num_locators); 
	printf("Data size: %d",dataSize);
    init_objects(stm_data,num_objects,100);
    init_locators(stm_data,num_tx,num_locators);

	STMData *d_stm_data = STM_copy_to_device(stm_data);

	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *flag;
  	CUDA_CHECK_ERROR(cudaMallocManaged(&flag, sizeof(int)), "Could not alloc");
  	*flag = 0;

	cudaEventRecord(start); 
	bank_kernel<<<gridDist, blockDist>>>(flag, seed, prRead, roSize, threadSize, dataSize, blockNum*threads_per_block, d_stm_data, d_stats, d_times);
  	cudaEventRecord(stop);
		
	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

	CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");
  	
  	cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;

	//get the output performance metrics
	cudaMemcpy(h_stats, d_stats, sizeof(Statistics), cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_times, d_times, blockNum*threads_per_block*sizeof(time_rate), cudaMemcpyDeviceToHost);
  	
  	getKernelOutput(h_stats, h_times, blockNum*threads_per_block, peak_clk, totT_ms, verbose);
	
	
  //kernelErr = cudaGetLastError();
  //if(kernelErr != cudaSuccess) printf("Error synchronize: %s\n", cudaGetErrorString(kernelErr));
  
  STM_copy_from_device(d_stm_data,stm_data);
  printf("FIM!\n");
  print_stats(stm_data);
    print_data(stm_data);

	free(h_stats);
	free(h_times);
	cudaFree(d_stats);
	cudaFree(d_times);
	
	return 0;
}
