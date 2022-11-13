__global__ void sum(cuComplex* result, cuComplex* a)
{
  //shared mem size = 8*8*8*sizeof(cuComplex)
  extern __shared__ cuComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<8*8*8) {
    result_s[tid].x = 0;
    result_s[tid].y = 0;
    
    result_s[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
        result_s[tid].x += result_s[tid+s].x;	
		result_s[tid].y += result_s[tid+s].y;
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x].x = result_s[0].x;
      result[blockIdx.x].y = result_s[0].y;
    }   
  }
}

__global__ void sum(float* result, float* a)
{
  //shared mem size = 8*8*8*sizeof(cuComplex)
  extern __shared__ float result_s_real[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<8*8*8) {
    result_s_real[tid] = 0;
    
    result_s_real[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
        result_s_real[tid] += result_s_real[tid+s];	
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x] = result_s_real[0];
    }   
  }
}

__global__ void maximum(cuComplex* result, cuComplex* a)
{
  //shared mem size = 8*8*8*sizeof(cuComplex)
  extern __shared__ cuComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<8*8*8) {
    result_s[tid].x = 0;
    result_s[tid].y = 0;
    
    result_s[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
        if(result_s[tid+s].x*result_s[tid+s].x+result_s[tid+s].y*result_s[tid+s].y >
	        result_s[tid].x*result_s[tid].x+result_s[tid].y*result_s[tid].y) {
				
	  result_s[tid].x = result_s[tid+s].x;
	  result_s[tid].y = result_s[tid+s].y;	
	
	}  
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x].x = result_s[0].x; 
      result[blockIdx.x].y = result_s[0].y;  
    }   
  }
}

