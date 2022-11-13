// Multiply array by kperp**2
__global__ void multKPerp(cuComplex* fK, cuComplex* f, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  // kPerp2 is defined with a minus sign, kperp2 = -( kx**2 + ky**2)
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
        fK[index] = f[index] * kPerp2(idx, idy) * scaler;
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
          fK[index] = f[index] * kPerp2(idx,idy) * scaler;
        
      }
    }
  }
}       
__global__ void multKPerpInv(cuComplex* fK, cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  // kPerp2 is defined with a minus sign, kperp2 = -( kx**2 + ky**2)
  // But that's alright, because you want to divide through by nabla^2 = -kperp^2
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
        fK[index] = .5f * f[index] * kPerp2Inv(idx,idy);
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
          fK[index] = .5f * f[index] * kPerp2Inv(idx,idy);
        
      }
    }
  }
}       

// Multiply array by kx
__global__ void multKx(cuComplex* fK, cuComplex* f) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      fK[index] = f[index]*kx(idx);
      		 
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        fK[index] = f[index]*kx(idx);
        
      }
    }
  }
}   

// Multiply array by ky
__global__ void multKy(cuComplex* fK, cuComplex* f) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      fK[index].x = f[index].x * ky(idy);
      fK[index].y = f[index].y * ky(idy);
      		 
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        fK[index].x = f[index].x * ky(idy);
        fK[index].y = f[index].y * ky(idy);
        
      }
    }
  }
}           

// Add, subtract arrays
__global__ void addsubt(cuComplex* result, cuComplex* f, cuComplex* g, float a)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
        result[index] = f[index] + a*g[index];
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        result[index] = f[index] + a*g[index];
      }
    }
  }
}           
// Multiply arrays C = A*B
__global__ void mult(float* C, float* A, float* B)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads){
  	if(idy<Ny && idx<Nx && idz<Nz){
		int index = idy + Ny*idx + Nx*Ny*idz;
		C[index] = A[index]*B[index];
		}
	}
  
  else{
  	for(int i=0; i<Nz/zThreads; i++){
		if(idy<Ny && idx<Nx && idz<zThreads){
			int index = idy + Ny*idx + Nx*Ny*idz + Nx*Ny*zThreads*i;
			C[index] = A[index]*B[index];
			}
		}
	}
}
// Square root
__global__ void squareroot(float* A)
{
  unsigned int idx = get_idx();
  A[idx] = sqrt(A[idx]);
}
// Divide arrays : C[index] = A[index]/B[index] if B[index]!=0
__global__ void divide(float* C, float* A, float* B)
{
  unsigned int index = get_idx();
  if(abs(B[index]) > 1.e-8) C[index] = A[index]/B[index];
  else C[index] = 0.0f;
}
// Square a real array and save it in f[].x
__global__ void square(float* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<Ny && idx<Nx && idz<Nz) {
      unsigned int index = idy + Ny*idx + Nx*Ny*idz;
      
      f[index] = f[index]*f[index];
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<Ny && idx<Nx && idz<zThreads) {
        unsigned int index = idy + Ny*idx + Nx*Ny*idz + Nx*Ny*zThreads*i;
	
        f[index] = f[index]*f[index];
      }
    }
  }
}    
// Square a complex array and save it in f[].x
__global__ void squareComplex(cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      f[index].x = f[index].x*f[index].x + f[index].y * f[index].y;
      f[index].y = 0;
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        f[index].x = f[index].x*f[index].x + f[index].y * f[index].y;
		f[index].y = 0;
      }
    }
  }
}    
// Fix fft
__global__ void fixFFT(cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
	if(idy!=0) f[index] = 2.0f*f[index];
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
	if(idy!=0) f[index] = 2.0f*f[index];
      }
    }

  }
}        	

////////////////////////////////////////
// Scale operations
////////////////////////////////////////
// Scale a complex array by a real number
__global__ void scale(cuComplex* b, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*idz;
    
      b[index].x = scaler*b[index].x;
      b[index].y = scaler*b[index].y;
    }
  }
    
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
		b[index].x = scaler*b[index].x;
        b[index].y = scaler*b[index].y; 
      }
    }
  }    	
} 
// Scale a real array by a real number
__global__ void scale(float* b, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*idz;
    
      b[index] = scaler*b[index];
    }
  }
    
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
		b[index] = scaler*b[index];
      }
    }
  }    	
} 
// Scale a complex array by a real number and save it in result
__global__ void scale(cuComplex* result, cuComplex* b, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*idz;
    
      result[index].x = scaler*b[index].x;
      result[index].y = scaler*b[index].y;
    }
  }
    
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
		result[index].x = scaler*b[index].x;
        result[index].y = scaler*b[index].y; 
      }
    }
  }    	
} 

//copies f(ky[i]) into fky
__global__ void kycopy(cuComplex* fky, cuComplex* f, int i) {
    
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();

  if(idy<Nz && idx<Nx) {
    unsigned int index = idx + (Nx)*idy;
    fky[index].x = f[i + index*(Ny/2+1)].x;
    fky[index].y = f[i + index*(Ny/2+1)].y;
  } 
}      

/////////////////////////////////////////
// Zeroing out arrays
/////////////////////////////////////////
__global__ void zero(cuComplex* f, int nx, int ny, int nz) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(nz<=zThreads) {
   if(idy<ny && idx<nx && idz<nz) {
    unsigned int index = idy + ny*idx + nx*ny*idz;
    
    f[index].x = 0;
    f[index].y = 0;
   }
  }
  else {
   for(int i=0; i<nz/zThreads; i++) {
    if(idy<ny && idx<nx && idz<zThreads) {
    unsigned int index = idy + ny*idx + nx*ny*idz + nx*ny*zThreads*i;
    
    f[index].x = 0;
    f[index].y = 0;
    }
   }
  }    
}    

__global__ void zero(float* f, int nx, int ny, int nz) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(nz<=zThreads) {
   if(idy<ny && idx<nx && idz<nz) {
    unsigned int index = idy + ny*idx + nx*ny*idz;
    
    f[index] = 0;
   }
  }
  else {
   for(int i=0; i<nz/zThreads; i++) {
    if(idy<ny && idx<nx && idz<zThreads) {
    unsigned int index = idy + ny*idx + nx*ny*idz + nx*ny*zThreads*i;
    
    f[index] = 0;
    }
   }
  }    
}    

