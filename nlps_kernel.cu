__global__ void deriv(cuComplex* f, cuComplex* fdx, cuComplex* fdy)                        
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  cuComplex cj; cj.x = 0.0f; cj.y = 1.0f;
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
     //df/dy
     fdy[index] = cj * ky(idy) * f[index];
    
     //df/dx
     fdx[index] = cj * kx(idx) * f[index];
   }
  } 
  else {
   for(int i=0; i<Nz/zThreads; i++) { 
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    //df/dx
    fdy[index] = cj * ky(idy) * f[index];
    
    //df/dy
    fdx[index] = cj * kx(idx) * f[index];
    }
   }
  } 
}  

__global__ void mask(cuComplex* mult) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
    if( (idy>(Ny-1)/3  || (idx>(Nx-1)/3 && idx<2*Nx/3+1) || (idz>(Nz-1)/3 && idz<2*Nz/3+1) ) ) {
      mult[index].x = 0.0f;
      mult[index].y = 0.0f;
    }  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	 int IDZ = idz + zThreads*i;
    
     if( (idy>(Ny-1)/3  || (idx>(Nx-1)/3 && idx<2*Nx/3+1) || (IDZ>(Nz-1)/3 && IDZ<2*Nz/3+1) ) ) {
       mult[index].x = 0.0f;
       mult[index].y = 0.0f;
     }  
    }
   }
  }
}      
  
  

__global__ void bracket(float* mult, float* fdx, float* fdy, 
                      float* gdx, float* gdy, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
   if(idy<(Ny) && idx<Nx && idz<Nz ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz;

    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny) && idx<Nx && idz<zThreads ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*zThreads*i;
    
    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
    }
   }
  } 
 
}  
