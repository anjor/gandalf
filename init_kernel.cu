__global__ void kInit(float* kx, float* ky, float* kz) 
{
  int idx = iget_idx();
  int idz = iget_idz();
  unsigned int idy = get_idy();

  if(idy<Ny/2+1 && idx<Nx) {
      
    ky[idy] = (float) idy/Y0;
      
    if(idx<Nx/2+1) {					
      kx[idx] = (float) idx/X0;					
    } else {						
      kx[idx] = (float) (idx - Nx)/X0;				
    }
  }
  
  if(Nz<=zThreads) { 
    if(idz<Nz) {
      if(idz<(Nz/2+1))
        kz[idz] = (float) idz/Z0;
      else
        kz[idz] = (float) (idz - Nz)/Z0;
    }	
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idz<zThreads) {
	int IDZ = idz + zThreads*i;
	if(IDZ<Nz){
	if(IDZ<(Nz/2+1))
	  kz[IDZ] = (float) IDZ/Z0;
	else
	  kz[IDZ] = (float) (IDZ - Nz)/Z0;
      }
	  }
    }
  } 
}     

__global__ void kPerpInit(float* kPerp2, float* kx, float* ky)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
   
  if(idy<(Ny/2+1) && idx<Nx) {
    unsigned int index = idy + (Ny/2+1)*idx;
      
    kPerp2[index] = -kx[idx]*kx[idx] -ky[idy]*ky[idy]; 	     
  }
}   

__global__ void kPerpInvInit(float* kPerp2Inv, float* kPerp2)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  
  if(idy<(Ny/2+1) && idx<Nx) {
    unsigned int index = idy + (Ny/2+1)*idx;
      
    if(index !=0) kPerp2Inv[index] = (float) 1.0f / (kPerp2[index]);  
    
  }
  kPerp2Inv[0] = 0.0;
}   
