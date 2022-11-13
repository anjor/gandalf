/*
__device__ unsigned int get_idx(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_idy(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_idz(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}

__device__ int iget_idx(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ int iget_idy(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ int iget_idz(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}
*/

__device__ unsigned int get_idx(void) {return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ unsigned int get_idy(void) {return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ unsigned int get_idz(void) {return blockIdx.z*blockDim.z+threadIdx.z;}

__device__ int iget_idx(void) {return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ int iget_idy(void) {return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ int iget_idz(void) {return blockIdx.z*blockDim.z+threadIdx.z;}


__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g) 
{
  return cuCaddf(f,g);
} 

__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g)
{
  return cuCsubf(f,g);
}  

__host__ __device__ cuComplex operator*(float scaler, cuComplex f) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, float scaler) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g)
{
  return cuCmulf(f,g);
}

__host__ __device__ cuComplex operator/(cuComplex f, float scaler)
{
  cuComplex result;
  result.x = f.x / scaler;
  result.y = f.y / scaler;
  return result;
}

__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) 
{
  return cuCdivf(f,g);
}


__host__ __device__ cuComplex exp(cuComplex arg)
{
  cuComplex res;
  float s, c;
  float e = expf(arg.x);
  sincosf(arg.y, &s, &c);
  res.x = c * e;
  res.y = s * e;
  return res;
}

__host__ __device__ cuComplex pow(cuComplex arg, int power)
{
  cuComplex res;
  float r = sqrt(pow(arg.x,2) + pow(arg.y,2));
  float theta = M_PI/2.0;
  if(arg.x != 0.0) theta = atan(arg.y/arg.x);
  res.x = pow(r, power) * cos(power * theta);
  res.y = pow(r, power) * sin(power * theta);
  return res;
}

__host__ __device__ cuComplex conjg(cuComplex arg)
{
  cuComplex conjugate;
  conjugate.x = arg.x;
  conjugate.y = -arg.y;
  return conjugate;

}

__host__ __device__ int sgn(float k) {
	
	if(k>0) return 1;
	if(k<0) return -1;
	return 0;

}
