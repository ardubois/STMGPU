////////////////////
////	GB		////
////////////////////

#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include <string.h>
#include <unistd.h>
#include <cassert>
#include <inttypes.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <inttypes.h>
#include <cuda_runtime.h>

#define SENDER_VERSION BASE_SEND

#include "../sync_lib/common.h"
#include "API.cuh"

#include <curand.h>
#include <curand_kernel.h>

__device__ float rand_() {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
       curandState state;
        curand_init(clock64(), i, 0, &state);

       return curand_uniform(&state);

}


/*
 * app specific config
 */
////////////////////
#define NUM_RECEIVER 1
#define MSG_SIZE_OFFLOAD 2
#define MSG_SIZE_MAX MSG_SIZE_OFFLOAD
///////////////////

#define ALG_ARG_DEF uint* token
#define ALG_ARG  token
#define PRINT_DEBUG_ 1

#define SERV_ARG_DEF TMmetadata* metadata, TXRecord* records, readSet* rs, writeSet* ws, warpResult* wRes, Statistics* stats, time_rate* times
#define SERV_ARG metadata, records, rs, ws, wRes, stats, times

#define PERF_METRICS 1
#define DISJOINT 0
#define KERNEL_DURATION 4

__device__ __forceinline__ void critcal_section(SERV_ARG_DEF, uint val0, int val1)
{

	uint tid = val0;
	uint timestamp = val1;
	int result;

	//validation
	//////////////////
#if PRINT_DEBUG_ == 0
	if(get_lane_id()==0) printf("\t\tS%d: recv %d %d\n", thread_id_x()/32, tid/32, timestamp);
#endif

	result=TXAddToRecord(metadata, records, rs, ws, stats, times, timestamp, tid);

#if PRINT_DEBUG_ == 0
	if(get_lane_id()==0) printf("\t\t\tS%d: sent %d %d\n", thread_id_x()/32, tid/32, result);
#endif
	
	wRes[tid/32].lane_result[tid%32]=result;
	__threadfence();
	if(get_lane_id() == 0)
		wRes[tid/32].valid_entry=1;

#if PRINT_DEBUG_ == 0
	if(get_lane_id()==0) printf("\t\t\t\tS%d: finished sending to %d\n", thread_id_x()/32, tid/32);
#endif

}

#include "../sync_lib/one_phase_def.h"
#include "../sync_lib/msg_config.h"
#include "../sync_lib/msg_aux.h"
#include "../sync_lib/msg_passing.h"
#include "../sync_lib/one_phase_server.h"

__device__ int waitMem;

__global__ void client_kernel(gbc_t gbc, int *flag, uint64_t seed, uint threadNum, uint dataSize, VertionedDataItem* data, readSet* rs, writeSet* ws, warpResult* wRes, int prRead, int roSize, int upSize, 
								Statistics* stats, time_rate* times) {

	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint wid = tid/32;

	long mod = 0xFFFF;
	long rnd;
	int probRead = prRead; 

	init_base(_offload_tail_ptr_cpy);

	__syncthreads();

	uint64_t state = seed+tid;
	int value, timestamp, addr, result;
	bool isAborted;

	uint dst = 0;
	uint val0 = tid;
	int val1;

	uint valid_msg, retry=1;
	uint saved_write_ptr = NOT_FOUND;

	long long int start_writeback, stop_writeback;
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, stop_time_tx;
	long long int start_wait=0, stop_wait=0;

	long int updates=0, reads=0;
#if DISJOINT
	//disjoint accesses variables
	int min, max;
	min = dataSize/threadNum*tid;
	max = dataSize/threadNum*(tid+1)-1;
#endif

	while((*flag & 1)==0)
	{ //printf("aqui\n");
		waitMem = *flag;
		retry=1;

		///////
		
		rnd = ((int)(rand_()*10)) + 1;  
        //printf("inicio\n");
		do
		{
			if(retry==1)start_time_tx = clock64();
			timestamp=TXBegin(tid, ws, rs);
			isAborted=false;
			valid_msg = 1;

			//Read-Only TX
			if(rnd <= probRead)
			{
				for(int i=0, value=0; i<roSize && isAborted==false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = (int)(rand_()*dataSize);
			#endif
					value+=TXReadOnly(data, addr, timestamp, rs, ws, tid, &isAborted);
				}
				if(isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
				//printf("t%d: ro %d\n", id, value);
			}
			//Update TX
			else
			{
				for(int i=0; i<upSize && isAborted==false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = (int)(rand_()*dataSize);
			#endif
					value = TXRead(data, addr, timestamp, rs, ws, tid, &isAborted); 
					TXWrite(data, value-(tid*10+100), addr, ws, tid);

			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = (int)(rand_()*dataSize);
			#endif
					value = TXRead(data, addr, timestamp, rs, ws, tid, &isAborted); 
					TXWrite(data, value+(tid*10+100), addr, ws, tid);
				}
				if(isAborted==true)
				{
					atomicAdd(&(stats->nbAbortsDataAge), 1);
					continue;
				}
			}
			
			/////////////////////////////
			//Commit process
			/////////////////////////////
			if(retry==1)start_time_commit = clock64();
			if(rnd < probRead)
			{
				start_wait = stop_wait = clock64();
				start_writeback = clock64();
				atomicAdd(&(stats->nbCommits), 1);
				stop_writeback = clock64();
				stop_time_commit = clock64();
				stop_time_tx = clock64();
				retry=0;
				reads++;
			}
			else
			{
				if(retry==0)
					val1=-1;
				else
					val1=timestamp;
				
				if(retry==1) start_wait = clock64();
				if(get_lane_id()==0)
					saved_write_ptr = atomicAdd(&(gbc.write_ptr[dst]), 32);
				saved_write_ptr = shuffle_idx(saved_write_ptr, 0) + get_lane_id();
				do
				{
					if(base_send(gbc, dst, valid_msg, saved_write_ptr, _offload_tail_ptr_cpy, SEND_ARG))
						valid_msg = 0;
				}
				while(vote_ballot(valid_msg) != 0);
				
				if(get_lane_id()==0)
					while(wRes[wid].valid_entry==0);
				result = wRes[wid].lane_result[get_lane_id()];
				if(retry==1) stop_wait = clock64();

				if(result != 0)
				{
					start_writeback = clock64();
					TXWriteBack(result, data, ws[tid]);
					stop_writeback = clock64();
					stop_time_commit = clock64();
					stop_time_tx = clock64();
					atomicAdd(&(stats->nbCommits), 1);
					retry=0;
					updates++;
				}
				//reset scoreboard
				if(get_lane_id()==0)
					wRes[wid].valid_entry=0;
			}
		}while(vote_ballot(retry) != 0);
		
		times[tid].runtime 	+= stop_time_tx - start_time_tx;
		times[tid].commit 	+= stop_time_commit - start_time_commit;
		times[tid].dataWrite+= stop_writeback - start_writeback;
		times[tid].wait 	+= stop_wait - start_wait;
	}
	printf("SAIU %d\n",tid);
	times[tid].nbReadOnly = reads;
	times[tid].nbUpdates  = updates;

	//exit process
	base_exit(gbc);
	printf("encerrou %d\n",tid);
}

__device__ void worker_thread(gbc_pack_t gbc_pack, SERV_ARG_DEF)
{  //printf("worker thread\n");
	uint m_warp_id = thread_id_x() / 32;

	VAR_BUF_DEF0
	uint stage_buf0 = 0;
	uint lock_id0;

	//for(stage_buf0=0; stage_buf0 < WORK_BUFF_SIZE_MAX; stage_buf0++)
	{		
		process_buffer_main_worker(gbc_pack, m_warp_id, 0, VAR_BUF0,
				stage_buf0,lock_id0, SERV_ARG);
	}
}


__global__ void server_kernel(int *flag,gbc_pack_t gbc_pack, readSet* rs, writeSet* ws, TXRecord* records, warpResult* wRes, Statistics* stats, time_rate* times)
{
	__shared__ TMmetadata metadata;
	//__shared__ uint txNumber[TXRecordSize];
   // printf("server kernel\n");
	init_recv(gbc_pack);
	// printf("init recv\n");
	gc_receiver_leader(gbc_pack);
	// printf("leader\n");
	while(1)
	{
		worker_thread(gbc_pack, &metadata, records, rs, ws, wRes, stats, times);
	}
}


__global__ void parent_kernel(int *flag, uint total_sender_bk, uint sender_block_size,	uint total_recevier_bk, uint recv_block_size, gbc_pack_t gbc_pack,
								uint64_t seed, uint dataSize, VertionedDataItem* data, readSet* rs, writeSet* ws, TXRecord* records, warpResult* wRes,
								float prRead, int roSize, int upSize, Statistics* stats, time_rate* times) {

//	for(int i=0; i<SCOREBOARD_SIZE/32; i++)
//		validSB[i]=0;
    printf("parent kernel\n");
	cudaStream_t s2;
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

	server_kernel<<<total_recevier_bk, recv_block_size, 0, s2>>>(flag,gbc_pack, rs, ws, records, wRes, stats, times);

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

	client_kernel<<<total_sender_bk, sender_block_size, 0, s1>>>(
			gbc_pack.gbc[CHANNEL_OFFLOAD], flag, seed, total_sender_bk*sender_block_size, dataSize, data, rs, ws, wRes,
			prRead, roSize, upSize, stats, times);

}

void test_fine_grain_offloading(int seed, int dataSize, int client_block_size, int total_client_bk, int server_block_size, float prRead, int roSize, int upSize, int verbose)
{

	int total_server_bk=1;
//	void (*server_kernel)(gbc_pack_t,
//	ALG_ARG_DEF);
//	server_kernel = server_one_phase;
	gbc_pack_t gbc_pack;
	create_gbc(gbc_pack, total_client_bk, client_block_size, server_block_size);
///////////////
	
	int* bankArray;
	VertionedDataItem *h_data, *d_data;
	TXRecord* records;
	warpResult* wRes;
	readSet* rs;
	writeSet* ws;

	time_rate *h_times, *d_times;
	Statistics *h_stats, *d_stats;

  	//time measurement variables
  	int peak_clk=1;
	cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	h_times = (time_rate*) calloc(total_client_bk*client_block_size,sizeof(time_rate));
	h_stats = (Statistics*)calloc(1,sizeof(Statistics));


	bankArray = (int*)malloc(dataSize*sizeof(int));
	for(int i=0; i<dataSize; i++)
	{
		bankArray[i]=1000;
	}
	//Allocate memory in the device
	cudaError_t result;
	result = TXInit(bankArray, dataSize, client_block_size*total_client_bk, &h_data, &d_data, &rs, &ws, &records, &wRes);
	if(result != cudaSuccess) fprintf(stderr, "Failed TM Initialization: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_stats, sizeof(Statistics));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_times, total_client_bk*client_block_size*sizeof(time_rate));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ratio: %s\n", cudaGetErrorString(result));

	//transfer the necessary data from the host to the device
	cudaMemcpy(d_times, h_times, total_client_bk*client_block_size*sizeof(time_rate), cudaMemcpyHostToDevice);
	cudaMemcpy(d_stats, h_stats, sizeof(Statistics), cudaMemcpyHostToDevice);

	int *flag;
  	CUDA_CHECK_ERROR(cudaMallocManaged(&flag, sizeof(int)), "Could not alloc");
  	*flag = 0;
     
	///////////////
	//kernel stuff
	cudaEventRecord(start);
	parent_kernel<<<1, 1>>>(flag, total_client_bk, client_block_size,
								total_server_bk, server_block_size, gbc_pack,
								1, dataSize, d_data, rs, ws, records, wRes,
								prRead, roSize, upSize, d_stats, d_times);
	cudaEventRecord(stop);

	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//assert(0);
	printf("viva satan.\n");
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

	CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");
	//////////////
	
	//Take the time the kernel took to complete
	cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;
    printf("free\n");
	free_gbc(gbc_pack);
	TXEnd(dataSize, h_data, &d_data, &rs, &ws, &wRes);

	//Copy metric data back to the host
	cudaMemcpy(h_stats, d_stats, sizeof(Statistics), cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_times, d_times, total_client_bk*client_block_size*sizeof(time_rate), cudaMemcpyDeviceToHost);

  	//Treat the data to generate output
	double avg_runtime=0, avg_commit=0, avg_wb=0, avg_val=0, avg_rwb=0, avg_wait=0, avg_comp=0, avg_send=0;
	long int totUpdates=0, totReads=0;
	for(int i=0; i<total_client_bk*client_block_size; i++)
	{
		if(h_times[i].runtime < 0) printf("T%d: %li\n", i, h_times[i].runtime);
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_wb 		+= h_times[i].dataWrite;
		avg_val		+= h_times[i].validation;
		avg_rwb		+= h_times[i].recordWrite;
		avg_wait	+= h_times[i].wait;
		avg_comp 	+= h_times[i].comparisons;
	
		totUpdates 	+= h_times[i].nbUpdates;
		totReads	+= h_times[i].nbReadOnly;
	}
	avg_wait = avg_wait - avg_rwb - avg_val;
	long int denom = (long)h_stats->nbCommits*peak_clk;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_wb 		/= denom;
	avg_val 	/= denom;
	avg_rwb 	/= denom;
	avg_wait	/= denom;
	avg_comp	/= h_stats->nbCommits;

	float rt_commit=0.0, rt_wb=0.0, rt_val=0.0, rt_rwb=0.0, rt_wait=0.0, rt_send=0.0;
	rt_commit	=	avg_commit / avg_runtime;
	rt_wb	 	=	avg_wb / avg_runtime;
	rt_val	 	=	avg_val / avg_runtime;
	rt_rwb	 	=	avg_rwb / avg_runtime;
	rt_wait		=	avg_wait / avg_runtime;

	int nbAborts = h_stats->nbAbortsDataAge + h_stats->nbAbortsRecordAge + h_stats->nbAbortsReadWrite + h_stats->nbAbortsWriteWrite;
    printf("nbCommits: %d\n", h_stats->nbCommits);
	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nAbortDataAge\t%f %%\nAbortRecAge\t%f %%\nAbortReadWrite\t%f %%\nAbortPreVal\t%f %%\n\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\nWaitTime\t%f\t%.2f%%\nPreValidation\t%f\t%.2f%%\nValidation\t%f\t%.2f%%\nRecInsert\t%f\t%.2f%%\nWriteBack\t%f\t%.2f%%\n\nComparisons\t%f\nTotalUpdates\t%d\nTotalReads\t%d\n", 
			(float)nbAborts/(nbAborts+h_stats->nbCommits)*100.0,
			h_stats->nbCommits/totT_ms*1000.0,
			(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsWriteWrite/(nbAborts+h_stats->nbCommits)*100.0,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			avg_wait,
			rt_wait*100.0,
			avg_send,
			rt_send*100.0,
			avg_val,
			rt_val*100.0,
			avg_rwb,
			rt_rwb*100.0,
			avg_wb,
			rt_wb*100.0,
			avg_comp,
			totUpdates,
			totReads
			);
	else
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%d\n", 
			(float)nbAborts/(nbAborts+h_stats->nbCommits)*100.0,
			h_stats->nbCommits/totT_ms*1000.0,
			(float)h_stats->nbAbortsDataAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsRecordAge/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsReadWrite/(nbAborts+h_stats->nbCommits)*100.0,
			(float)h_stats->nbAbortsWriteWrite/(nbAborts+h_stats->nbCommits)*100.0,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			avg_wait,
			rt_wait*100.0,
			avg_send,
			rt_send*100.0,
			avg_val,
			rt_val*100.0,
			avg_rwb,
			rt_rwb*100.0,
			avg_wb,
			rt_wb*100.0,
			avg_comp,
			totUpdates,
			totReads
			);

	free(h_stats);
	free(h_times);
	cudaFree(d_stats);
	cudaFree(d_times);
}

int main(int argc, char *argv[]) {

	int client_block_size, server_block_size, verbose;
	int total_client_bk;
	int dataSize, roSize, upSize;
	int prRead;

	  	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) nb bank accounts               \n"
	  "  2) client config - nb threads     \n"
	  "  3) client config - nb blocks      \n"
	  "  4) server config - nb threads     \n"
	  "  5) prob read TX                   \n"
	  "  6) read TX Size                   \n"
	  "  7) update TX Size                 \n"
	  "  8) verbose		                   \n"
	"";
	const int NB_ARGS = 9;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	dataSize			= atoi(argv[argCnt++]);
	client_block_size	= atoi(argv[argCnt++]);
	total_client_bk	 	= atoi(argv[argCnt++]);
	server_block_size	= atoi(argv[argCnt++]);
	prRead 				= atoi(argv[argCnt++]);
	roSize 				= atoi(argv[argCnt++]);
	upSize				= atoi(argv[argCnt++]);
	verbose				= atoi(argv[argCnt++]);

#if DISJOINT
	dataSize=10*total_client_bk*client_block_size;
#endif

	cudaSetDevice(0);
	for (int i = 0; i < 1; i++) {
		test_fine_grain_offloading(i, dataSize, client_block_size, total_client_bk, server_block_size, prRead, roSize, upSize, verbose);
	}
	return 0;
}
