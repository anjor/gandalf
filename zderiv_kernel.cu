__global__ void zderiv(cuComplex* result, cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  cuComplex res, cj;
  cj.x = 0.0f;
  cj.y = 1.0f;
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;

	  res = cj*kz(idz)*f[index];
	  result[index] = res;
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
		int IDZ = idz + zThreads*i;
		 res = cj*kz(IDZ)*f[index];
		 result[index] = res;
      }
    }
  }    	
} 

__global__ void absk_closure(cuComplex* f, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
      //result(ky,kx,kz)= i*kz*f(ky,kx,kz)
      f[index].x = exp(abs(kz(idz))*scaler) * f[index].x;
      f[index].y = exp(abs(kz(idz))*scaler) * f[index].y;    
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
		int IDZ = idz + zThreads*i;
	
		f[index].x = exp(abs(kz(IDZ))*scaler) * f[index].x;
        f[index].y = exp(abs(kz(IDZ))*scaler) * f[index].y;  
      }
    }
  }    	
} 
