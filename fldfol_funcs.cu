//////////////////////////////////////////////////////////////////////
// Basic diagnostic Kernels
//////////////////////////////////////////////////////////////////////
// kz spec
__global__ void kzspec(cuComplex* k2field2, float* energy_kp)
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  // X0 and Y0 needs to be 1 for this to make sense.

  if(Nz<=zThreads) {
  	if(idx<Nx && idy<Ny/2+1 && idz<Nz){
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
        atomicAdd(energy_kp + idz, k2field2[index].x);
      }
	}
  else {
  	for(int i=0; i<Nz/zThreads; i++) {
		if(idx<Nx && idy<Ny/2+1 && idz<zThreads) {
				unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
				int IDZ = idz + zThreads*i;
				atomicAdd(energy_kp + IDZ, k2field2[index].x);
			}
		}
	}
}


//////////////////////////////////////////////////////////////////////
// fldfol kernels
//////////////////////////////////////////////////////////////////////

// Initialize x, y
__global__ void initxy(float* xs, float* ys)
{
  unsigned int idx = get_idx();

  if (idx < Nx*Ny )
    {
        xs[idx] = xx(blockIdx.x);
        ys[idx] = yy(threadIdx.x);
    }

}

// Take a step along the fieldline :
// dby is actually -dby. This is taken care of in the current routine
__global__ void zstep_fld (float* x0, float* y0, float* dbx, float* dby, float dz, float* x1,float*  y1)
{

  unsigned int idx = get_idx();

  if (idx<Nx*Ny)
    {
      x1[idx] = x0[idx] + dbx[idx]*dz;
      x1[idx] = fmodf(x1[idx], 2.*M_PI);
      if(x1[idx]<0) x1[idx] = fmodf(x1[idx]+2.*M_PI, 2.*M_PI);
        

      y1[idx] = y0[idx] - dby[idx]*dz;
      y1[idx] = fmodf(y1[idx], 2.*M_PI);
      if(y1[idx]<0) y1[idx] = fmodf(y1[idx]+2.*M_PI, 2.*M_PI);
    }
    
}

// Interpolate field at x,y 
__global__ void interp (float* x, float* y, float* fld, float* interpfld)
{
  unsigned int idx = get_idx();


  if (idx<Nx*Ny)
    {
        
       int idx0  = (int) (floor((x[idx]/2.0/M_PI)*Nx)) %Nx;
       int idy0  = (int) (floor((y[idx]/2.0/M_PI)*Ny)) %Ny;

       float dx = (x[idx]/2.0/M_PI)*Nx - (float) idx0;
       float dy = (y[idx]/2.0/M_PI)*Ny - (float) idy0;

       int idx00 = idy0 + Ny*idx0;
       int idx01 = ((idy0+1)%Ny) + Ny*idx0;
       int idx10 = idy0 + Ny*((idx0+1)%Nx);
       int idx11 = ((idy0+1)%Ny) + Ny*((idx0+1)%Nx);

       interpfld[idx] = fld[idx00] *dx*dy
                   +fld[idx01] *dx*(1-dy)
                   +fld[idx10] *(1-dx)*dy
                   +fld[idx11] *(1-dx)*(1-dy);
    }


}

__global__ void z_interp (float* result, float* current, float* next)
{
    unsigned int idx = get_idx();
    if(idx < Nx*Ny) result[idx] = (current[idx] + next[idx])/2.0;
}

//////////////////////////////////////////////////////////////////////
// fldfol functions
//////////////////////////////////////////////////////////////////////
void calculate_uxy_dbxy (cuComplex* zp, cuComplex* zm, float* ux, float* uy, float* dbx, float* dby)
{
      cuComplex *kPhi, *kA;
      cuComplex *dx, *dy;

      // Allocate Phi, Apar, dx, dy
      cudaMalloc((void**) &kPhi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      cudaMalloc((void**) &kA, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

      cudaMalloc((void**) &dx, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      cudaMalloc((void**) &dy, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

      // Calculate Phi, Apar
	  addsubt<<<dimGrid,dimBlock>>> (kPhi, zp, zm, 1);
	  //kPhi = zp+zm
	  	
	  scale<<<dimGrid,dimBlock>>> (kPhi, .5);
	  //kPhi = .5*(zp+zm) = phi
	  
	  addsubt<<<dimGrid,dimBlock>>> (kA, zp, zm, -1);
	  //kA = zp-zm
	  
	  scale<<<dimGrid,dimBlock>>> (kA, .5);
	  //kA = .5*(zp-zm) = A

      deriv <<<dimGrid, dimBlock>>>(kPhi, dx, dy);
      // uy has a negative sign
      if(cufftExecC2R(plan_C2R, dx, uy) != CUFFT_SUCCESS) printf("ux calculation failed \n");
      if(cufftExecC2R(plan_C2R, dy, ux) != CUFFT_SUCCESS) printf("uy calculation failed \n");
      deriv <<<dimGrid, dimBlock>>>(kA, dx, dy);
      // uy has a negative sign
      if(cufftExecC2R(plan_C2R, dx, dby) != CUFFT_SUCCESS) printf("dbx calculation failed \n");
      if(cufftExecC2R(plan_C2R, dy, dbx) != CUFFT_SUCCESS) printf("dby calculation failed \n");


      cudaFree(kPhi); cudaFree(kA);
      cudaFree(dx); cudaFree(dy);

}

//////////////////////////////////////////////////////////////////////
// Actual fieldline following function
void fldfol (float* ux, float* uy, float* dbx, float* dby, float* GmR, float* uxfld, float* uyfld, float* dbxfld, float* dbyfld, float* Gmfld)
{
      float *xs, *ys, *xstar, *ystar;
      float *tempdbx, *tempdby, *tempdbx2, *tempdby2;

      cudaMalloc((void**) &xs, sizeof(float)*Nx*Ny);
      cudaMalloc((void**) &ys, sizeof(float)*Nx*Ny);
      cudaMalloc((void**) &xstar, sizeof(float)*Nx*Ny);
      cudaMalloc((void**) &ystar, sizeof(float)*Nx*Ny);

      cudaMalloc((void**) &tempdbx, sizeof(float)*Nx*Ny);
      cudaMalloc((void**) &tempdby, sizeof(float)*Nx*Ny);
      cudaMalloc((void**) &tempdbx2, sizeof(float)*Nx*Ny);
      cudaMalloc((void**) &tempdby2, sizeof(float)*Nx*Ny);
      // Initialize x,y for fieldline following
      initxy <<<Nx, Ny>>>(xs, ys);
      float dz = Z0/((float) Nz);
      for (int idz=0; idz<Nz; idz++)
        {

            for (int m=0; m<Nm; m++) 
                {
                int offset = m*Nx*Ny*Nz + idz*Nx*Ny;
                interp <<<Nx, Ny>>>(xs, ys, GmR+offset, Gmfld+offset);

                }

              int offset = Nx*Ny*idz;
              interp <<<Nx, Ny>>>(xs, ys, ux+offset, uxfld+offset);
              interp <<<Nx, Ny>>>(xs, ys, uy+offset, uyfld+offset);
              interp <<<Nx, Ny>>>(xs, ys, dbx+offset, dbxfld+offset);
              interp <<<Nx, Ny>>>(xs, ys, dby+offset, dbyfld+offset);
  
              zstep_fld <<<Nx, Ny>>>(xs, ys, dbxfld+offset, dbyfld+offset, dz/2.0, xstar, ystar);
  
              interp <<<Nx, Ny>>>(xstar, ystar, dbx+offset, tempdbx);
              interp <<<Nx, Ny>>>(xstar, ystar, dby+offset, tempdby);
  
              offset  = Nx*Ny*((idz+1)%Nz);
              interp <<<Nx, Ny>>>(xstar, ystar, dbx+offset, tempdbx2);
              interp <<<Nx, Ny>>>(xstar, ystar, dby+offset, tempdby2);
  
              z_interp<<<Nx, Ny>>>(tempdbx, tempdbx, tempdbx2);
              z_interp<<<Nx, Ny>>>(tempdby, tempdby, tempdby2);
  
              zstep_fld <<<Nx, Ny>>>(xs, ys, tempdbx, tempdby, dz, xs, ys);
              //printf("After idz=%d: %s\n", idz, cudaGetErrorString(cudaGetLastError()));

        }


        cudaFree(xs); cudaFree(ys);
        cudaFree(xstar); cudaFree(ystar);
        cudaFree(tempdbx); cudaFree(tempdby);
        cudaFree(tempdbx2); cudaFree(tempdby2);
}
