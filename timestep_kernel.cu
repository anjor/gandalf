__global__ void linstep(cuComplex* fNew, cuComplex* fOld, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  cuComplex cj;
  cj.x = 0.0f;
  cj.y = 1.0f;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;

		fNew[index] = fOld[index] * exp(cj*kz(idz)*dt);

    }
  }
  
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
		int IDZ = idz + zThreads*i;

		fNew[index] = fOld[index] * exp(cj*kz(IDZ)*dt);
	
      }
    }
  }
}


__global__ void fwdeuler(cuComplex* fNew, cuComplex* nl, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;

		fNew[index] = fNew[index] + nl[index]*dt;

    }
  }
  
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;

		fNew[index] = fNew[index] + nl[index]*dt;
       }
     }
    }     
}

__global__ void fwdeuler(float* fNew, float* nl, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;

		fNew[index] = fNew[index] + nl[index]*dt;

    }
  }
  
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;

		fNew[index] = fNew[index] + nl[index]*dt;
       }
     }
    }     
}


