#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>

int devid;
__constant__ int Nx, Ny, Nz, Nm;
__constant__ int zThreads;
int totalThreads;
char *runname;

cuComplex *f_d, *g_d, *Gm;
float *ux, *uy, *dbx, *dby, *GmR;
float *xs, *ys;

float *uxfld, *uyfld, *dbxfld, *dbyfld;
float *Gmfld; 


__constant__ float X0, Y0, Z0;

cufftHandle plan_C2R, plan_R2C;
dim3 dimGrid, dimBlock;

FILE *alf_kzfile, *alf_kzkpfile;
FILE *slmkzfile, *slmkzkpfile;



#include "c_fortran_namelist3.c"
#include "device_funcs.cu"
#include "k_funcs.cu"
#include "work_kernel.cu"
#include "nlps_kernel.cu"
#include "diag_kernel.cu"
#include "fldfol_funcs.cu"

void diagnostics(float* ux, float*uy, float* dbx, float*dby, float* Gm);
void setup_device(), setup_grid_block(), fft_plan_create(), fft_plan_destroy(), read_namelist(char* filename);
void restartRead(cuComplex* zp, cuComplex*zm, cuComplex* Gm, float* tim);
void init_diag(), close_diag(), allocate_arrays();


int main (int argc, char* argv[])
{
    
    if (argc<1) printf( "Usage: ./fldfol runname");
    else
    {
	  // Assuming argv[1] is the runname
	  runname = argv[1];
	  char str[255];
	  strcpy(str, runname);
	  strcat(str, ".in");
	  printf("Reading from %s \n", str);

      // Read namelist
	  read_namelist(str);
      printf("After read_namelist: %s\n", cudaGetErrorString(cudaGetLastError()));

      // Allocate arrays
      allocate_arrays();
      printf("After allocate_arrays: %s\n", cudaGetErrorString(cudaGetLastError()));

      // Initialize diagnostics
      init_diag();
      printf("After init_diag: %s\n", cudaGetErrorString(cudaGetLastError()));

	  // Setup dimgrid and dimblock
      setup_grid_block();
      printf("After setup_grid_block: %s\n", cudaGetErrorString(cudaGetLastError()));

      // Create fft plans
      fft_plan_create();

      float tim=0;
      // Read in data
      restartRead(f_d, g_d, Gm, &tim);

      printf("After restartread: %s\n", cudaGetErrorString(cudaGetLastError()));

      // Calculate ux, uy, dbx, dby
      calculate_uxy_dbxy (f_d, g_d, ux, uy, dbx, dby);

      // Convert Gm to real space
      for (int m=0; m<Nm; m++) if(cufftExecC2R(plan_C2R, Gm+m*Nx*(Ny/2+1)*Nz, GmR+m*Nx*Ny*Nz) != CUFFT_SUCCESS) printf("Gm conversion to real space failed for m=%d\n", m);

      // Free f_d, g_d and Gm
      cudaFree(f_d); cudaFree(g_d); cudaFree(Gm);

      fldfol(ux, uy, dbx, dby, GmR, uxfld, uyfld, dbxfld, dbyfld, Gmfld);

      cudaFree(ux); cudaFree(uy);
      cudaFree(dbx); cudaFree(dby);
      cudaFree(GmR);
      cudaFree(xs); cudaFree(ys);

      diagnostics(uxfld, uyfld, dbxfld, dbyfld, Gmfld);

      close_diag();



    }
}

//////////////////////////////////////////////////////////////////////
// Diagnostic subroutines
//////////////////////////////////////////////////////////////////////
void aw_kzkp(cuComplex* up2, cuComplex* dbp2)
{
    // AVK: This could be improved upon
    int ikpmax = (int) ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );
    float kpmax = ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );
    float *kinEnergy_kp, *magEnergy_kp;

    cudaMalloc((void**) &kinEnergy_kp, sizeof(float)*ikpmax*Nz);
    cudaMalloc((void**) &magEnergy_kp, sizeof(float)*ikpmax*Nz);

    zero<<<Nz,ikpmax>>>(kinEnergy_kp, Nz*ikpmax,1,1);
    zero<<<Nz,ikpmax>>>(magEnergy_kp, Nz*ikpmax,1,1);

    float *kinEnergy_kp_h, *magEnergy_kp_h;

    kinEnergy_kp_h = (float*) malloc(sizeof(float)*ikpmax*Nz);
    magEnergy_kp_h = (float*) malloc(sizeof(float)*ikpmax*Nz);
    
        //loop through the ky's
        for(int ikp=1; ikp<ikpmax; ikp++) {
            kz_kpshellsum<<<dimGrid, dimBlock>>>(up2, ikp, kinEnergy_kp);
            kz_kpshellsum<<<dimGrid, dimBlock>>>(dbp2, ikp, magEnergy_kp);
        }


        cudaMemcpy(kinEnergy_kp_h, kinEnergy_kp, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);
        cudaMemcpy(magEnergy_kp_h, magEnergy_kp, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);


    for(int ikp=1; ikp<ikpmax; ikp++) {
        int ikz_kp = 0 + Nz*ikp;
        fprintf(alf_kzkpfile, "%g \t %g \t %g \t %g\n", kz(0), ((float) ikp/ikpmax)*kpmax, kinEnergy_kp_h[ikz_kp], magEnergy_kp_h[ikz_kp]);
        }
    fprintf(alf_kzkpfile, "\n");
    for(int ikz=1; ikz<=(Nz-1)/3; ikz++){
        for(int ikp =1; ikp<ikpmax; ikp++){
            int ikz_kp = ikz + Nz*ikp;
            int mikz_kp = (Nz-ikz) + Nz*ikp;
            fprintf(alf_kzkpfile, "%g \t %g \t %g \t %g\n", kz(ikz), ((float) ikp/ikpmax)*kpmax, kinEnergy_kp_h[ikz_kp] + kinEnergy_kp_h[mikz_kp], magEnergy_kp_h[ikz_kp] + magEnergy_kp_h[mikz_kp]);
        }

    fprintf(alf_kzkpfile, "\n");
        }
    fprintf(alf_kzkpfile, "\n");
    
    cudaFree(kinEnergy_kp); cudaFree(magEnergy_kp);
    free(kinEnergy_kp_h); free(magEnergy_kp_h);
    
    
}    
void aw_kz(cuComplex* uperp2, cuComplex* dbperp2)
{
    float *up2_kz, *dbp2_kz;

    cudaMalloc((void**) &up2_kz, sizeof(float)*Nz);
    cudaMalloc((void**) &dbp2_kz, sizeof(float)*Nz);

    zero<<<1,Nz>>>(up2_kz, Nz,1,1);
    zero<<<1,Nz>>>(dbp2_kz, Nz,1,1);

    float *up2_kz_h, *dbp2_kz_h;

    up2_kz_h = (float*) malloc(sizeof(float)*Nz);
    dbp2_kz_h = (float*) malloc(sizeof(float)*Nz);
    
    kzspec <<<dimGrid, dimBlock>>>(uperp2, up2_kz);
    kzspec <<<dimGrid, dimBlock>>>(dbperp2, dbp2_kz);

    cudaMemcpy(up2_kz_h, up2_kz, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
    cudaMemcpy(dbp2_kz_h, dbp2_kz, sizeof(float)*Nz, cudaMemcpyDeviceToHost);

    for(int ikz=0; ikz<=(Nz-1)/3; ikz++){
        int mikz = (Nz-ikz);

        fprintf(alf_kzfile, "%g \t %g\t %g\n",
        kz(ikz), up2_kz_h[ikz]+up2_kz_h[mikz], dbp2_kz_h[ikz]+dbp2_kz_h[mikz]);
        }

    fprintf(alf_kzfile, "\n");

    cudaFree(up2_kz); cudaFree(dbp2_kz);
    free(up2_kz_h); free(dbp2_kz_h);
    
    
}    

//Slow mode kz-kp spectra
void slow_kzkp(cuComplex* Gm2,int m)
{
  float kpmax = ( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );
  int ikpmax = (int) ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );

  float* Gmenkzkp;
  cudaMalloc((void**) &Gmenkzkp, sizeof(float)*ikpmax*Nz);

  zero<<<ikpmax, Nz>>>(Gmenkzkp, Nz*ikpmax, 1, 1);

  for (int ikp=1; ikp<ikpmax; ikp++){
  	kz_kpshellsum<<<dimGrid, dimBlock>>>(Gm2, ikp, Gmenkzkp);
	}

  float *Gmenkzkp_h;
  Gmenkzkp_h = (float*) malloc(sizeof(float)*Nz*ikpmax);
  
  cudaMemcpy(Gmenkzkp_h, Gmenkzkp, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);

  for(int ikp=1; ikp<ikpmax; ikp++){
  	int ikz_kp = 0+Nz*ikp;
  	fprintf(slmkzkpfile, "%d\t %g\t %g\t %g\n", m, kz(0), ((float) ikp), Gmenkzkp_h[ikz_kp]);
  
    }
    fprintf(slmkzkpfile, "\n");
  for(int ikz=1; ikz<=(Nz-1)/3; ikz++){
    for(int ikp=1; ikp<ikpmax; ikp++){
        int ikz_kp = ikz+Nz*ikp;
        int mikz_kp = (Nz-ikz) + Nz*ikp;
        fprintf(slmkzkpfile, "%d\t %g\t %g\t %g\n", m, kz(ikz), ((float) ikp), Gmenkzkp_h[ikz_kp] + Gmenkzkp_h[mikz_kp]);

      }
      fprintf(slmkzkpfile, "\n");
  }

  cudaFree(Gmenkzkp);
  free(Gmenkzkp_h);


}
// Slow mode kz spectra
void slow_kz(cuComplex* Gm2,int m)
{
  float* Gmenkz;
  cudaMalloc((void**) &Gmenkz, sizeof(float)*Nz);

  zero<<<1, Nz>>>(Gmenkz, Nz, 1, 1);

  kzspec <<<dimGrid, dimBlock>>>(Gm2, Gmenkz);

  float *Gmenkz_h;
  Gmenkz_h = (float*) malloc(sizeof(float)*Nz);
  
  cudaMemcpy(Gmenkz_h, Gmenkz, sizeof(float)*Nz, cudaMemcpyDeviceToHost);

  for(int ikz=0; ikz<=(Nz-1)/3; ikz++) fprintf(slmkzfile, "%d\t %g\t %g\n", m, kz(ikz), Gmenkz_h[ikz]+Gmenkz_h[Nz-ikz]);

  fprintf(slmkzfile, "\n");

  cudaFree(Gmenkz);
  free(Gmenkz_h);


}

void diagnostics(float* ux, float* uy, float* dbx, float* dby, float*Gm)
  {
    
    cuComplex *ux2, *uy2, *dbx2, *dby2;

    cudaMalloc((void**) &ux2, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &uy2, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &dbx2, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &dby2, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    if(cufftExecR2C(plan_R2C, ux, ux2) != CUFFT_SUCCESS) printf ("ux FFT failed \n");
    if(cufftExecR2C(plan_R2C, uy, uy2) != CUFFT_SUCCESS) printf ("uy FFT failed \n");

    if(cufftExecR2C(plan_R2C, dbx, dbx2) != CUFFT_SUCCESS) printf ("ux FFT failed \n");
    if(cufftExecR2C(plan_R2C, dby, dby2) != CUFFT_SUCCESS) printf ("uy FFT failed \n");

    cudaFree(ux); cudaFree(uy); cudaFree(dbx); cudaFree(dby);

    scale <<<dimGrid, dimBlock>>>(ux2, 1.0f/((float) Nx*Ny*Nz));
    scale <<<dimGrid, dimBlock>>>(uy2, 1.0f/((float) Nx*Ny*Nz));
    scale <<<dimGrid, dimBlock>>>(dbx2, 1.0f/((float) Nx*Ny*Nz));
    scale <<<dimGrid, dimBlock>>>(dby2, 1.0f/((float) Nx*Ny*Nz));

    mask<<<dimGrid,dimBlock>>>(ux2);
    mask<<<dimGrid,dimBlock>>>(uy2);
    mask<<<dimGrid,dimBlock>>>(dbx2);
    mask<<<dimGrid,dimBlock>>>(dby2);

    squareComplex <<<dimGrid, dimBlock>>> (ux2);
    squareComplex <<<dimGrid, dimBlock>>> (uy2);
    squareComplex <<<dimGrid, dimBlock>>> (dbx2);
    squareComplex <<<dimGrid, dimBlock>>> (dby2);

    fixFFT <<<dimGrid, dimBlock>>>(ux2);
    fixFFT <<<dimGrid, dimBlock>>>(uy2);
    fixFFT <<<dimGrid, dimBlock>>>(dbx2);
    fixFFT <<<dimGrid, dimBlock>>>(dby2);

    addsubt<<<dimGrid, dimBlock>>>(ux2, ux2, uy2, 1);
    addsubt<<<dimGrid, dimBlock>>>(dbx2, dbx2, dby2, 1);

    aw_kz(ux2, dbx2);
    fflush(alf_kzfile);

    aw_kzkp(ux2, dbx2);
    fflush(alf_kzkpfile);

    cudaFree(uy2); cudaFree(dbx2); cudaFree(dby2);

    for (int m=0; m<Nm; m++)
     {

        if (cufftExecR2C(plan_R2C, Gm + m*Nx*Ny*Nz, ux2) != CUFFT_SUCCESS) printf ("Gm FFT failed m=%d\n", m);

        scale <<<dimGrid, dimBlock>>> (ux2, 1.0f/((float) Nx*Ny*Nz));
        mask <<<dimGrid, dimBlock>>> (ux2);

        squareComplex<<<dimGrid, dimBlock>>>(ux2);
        fixFFT<<<dimGrid, dimBlock>>>(ux2);

        slow_kz(ux2, m);

        slow_kzkp(ux2, m);


     }

     cudaFree(GmR); cudaFree(ux2);

     fprintf(slmkzfile, "\n");
     fflush(slmkzfile);

     fprintf(slmkzkpfile, "\n");
     fflush(slmkzkpfile);


  }

//////////////////////////////////////////////////////////////////////
// Set up functions
//////////////////////////////////////////////////////////////////////
// Setup device
void setup_device(){

  // Device information
  int ct, dev;
  struct cudaDeviceProp prop;

  cudaGetDeviceCount(&ct);
  printf("Device Count: %d\n",ct);

  cudaSetDevice(devid);
  cudaGetDevice(&dev);
  printf("Device ID: %d\n",dev);

  cudaGetDeviceProperties(&prop,dev);
  printf("Device Name: %s\n", prop.name);

  printf("Major mode: %d\n", prop.major);
  printf("Global Memory (bytes): %lu\n", (unsigned long) prop.totalGlobalMem);
  printf("Shared Memory per Block (bytes): %lu\n", (unsigned long) prop.sharedMemPerBlock);
  printf("Registers per Block: %d\n", prop.regsPerBlock);
  printf("Warp Size (threads): %d\n", prop.warpSize); 
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Max Size of Block Dimension (threads): %d * %d * %d\n", prop.maxThreadsDim[0], 
	 prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max Size of Grid Dimension (blocks): %d * %d * %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

  }

void setup_grid_block(){
  int dev;
  struct cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop,dev);

  //////////////////////////////////////////////////////////
	//set up normal dimGrid/dimBlock config  
	int zBlockThreads = prop.maxThreadsDim[2];
	*&zThreads = zBlockThreads*prop.maxGridSize[2];
	totalThreads = prop.maxThreadsPerBlock;

	if(Nz > zBlockThreads) dimBlock.z = zBlockThreads;
	else dimBlock.z = Nz;
	int xy = (int) totalThreads/dimBlock.z;
	int blockxy = (int) sqrt((float) xy);
	//dimBlock = threadsPerBlock, dimGrid = numBlocks
	dimBlock.x = blockxy;
	dimBlock.y = blockxy;

	if(Nz>zThreads) {
	  dimBlock.x = (unsigned int) sqrt((float) totalThreads/zBlockThreads);
	  dimBlock.y = (unsigned int) sqrt((float) totalThreads/zBlockThreads);
	  dimBlock.z = zBlockThreads;
	}  

	dimGrid.x = (unsigned int) ceil((float) Nx/dimBlock.x + 0);
	dimGrid.y = (unsigned int) ceil((float) Ny/dimBlock.y + 0);
	if(prop.maxGridSize[2]==1) dimGrid.z = 1;    
	else dimGrid.z = (unsigned int) ceil((float) Nz/dimBlock.z) ;
	cudaMemcpyToSymbol(zThreads, &zThreads, sizeof(int));
	printf("zthreads = %d, zblockthreads = %d \n", zThreads, zBlockThreads);
	
	printf("dimGrid = %d, %d, %d \t dimBlock = %d, %d, %d \n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

//    if (Nx*Ny < totalThreads)
//      {
//        dimBlock2D.x = (int) sqrt((float) Nx*Ny);
//        dimBlock2D.y = (int) sqrt((float) Nx*Ny);
//      }
//   else
//     {
//       dimGrid2D.x = Nx*Ny / totalThreads;
//       dimGrid2D.y = 1; dimGrid2D.z = 1;
//
//       dimBlock2D.x = (int) sqrt((float) totalThreads);
//       dimBlock2D.y = (int) sqrt((float) totalThreads);
//       dimBlock2D.z = 1;
//     }
//	printf("dimGrid2D = %d, dimBlock2D = %d, %d, %d \n", dimGrid2D.x, dimBlock2D.x, dimBlock2D.y, dimBlock2D.z);
//

	}

// FFT plans
void fft_plan_create()
{
	if(cufftPlan3d(&plan_C2R, Nz, Nx, Ny, CUFFT_C2R) != CUFFT_SUCCESS) {
		printf("plan_C2R creation failed. Don't trust results. \n");
		};
	if(cufftPlan3d(&plan_R2C, Nz, Nx, Ny, CUFFT_R2C) != CUFFT_SUCCESS) {
		printf("plan_R2C creation failed. Don't trust results. \n");
		}
}
void fft_plan_destroy()
{
	if(cufftDestroy(plan_C2R) != CUFFT_SUCCESS){
		printf("plan_C2R destruction failed. \n");
	  }
	if(cufftDestroy(plan_R2C) != CUFFT_SUCCESS){
		printf("plan_R2C destruction failed. \n");
	  }

}


// Read in required parameters
void read_namelist(char* filename)
{
  struct fnr_struct namelist_struct = fnr_read_namelist_file(filename);

  //Device
  if(fnr_get_int(&namelist_struct, "dev", "devid", &devid)) devid = 0;
  setup_device();

  //Grid
  if(fnr_get_int(&namelist_struct, "grid", "Nx", &Nx)) *&Nx = 16;
  cudaMemcpyToSymbol(Nx, &Nx, sizeof(int));
  if(fnr_get_int(&namelist_struct, "grid", "Ny", &Ny)) *&Ny = 16;
  cudaMemcpyToSymbol(Ny, &Ny, sizeof(int));
  if(fnr_get_int(&namelist_struct, "grid", "Nz", &Nz)) *&Nz = 16;
  cudaMemcpyToSymbol(Nz, &Nz, sizeof(int));
  if(fnr_get_float(&namelist_struct, "grid", "X0", &X0)) *&X0 = 1.0f;
  cudaMemcpyToSymbol(X0, &X0, sizeof(float));
  if(fnr_get_float(&namelist_struct, "grid", "Y0", &Y0)) *&Y0 = 1.0f;
  cudaMemcpyToSymbol(Y0, &Y0, sizeof(float));
  if(fnr_get_float(&namelist_struct, "grid", "Z0", &Z0)) *&Z0 = 1.0f;
  cudaMemcpyToSymbol(Z0, &Z0, sizeof(float));

  printf("Grid read \n");

  // Slow
  	fnr_get_int(&namelist_struct, "grid", "Nm", &Nm);
    cudaMemcpyToSymbol(Nm, &Nm, sizeof(int),0,cudaMemcpyHostToDevice);
  printf("Slow read \n");

}

// Read data from .res file
void restartRead(cuComplex* zp, cuComplex* zm, cuComplex* Gm, float* tim)
{
  char str[255];
  FILE *restart;
  strcpy(str, runname);
  strcat(str,".res");
  restart = fopen(str, "rb");

  cuComplex *zp_h;
  cuComplex *zm_h;
  cuComplex *Gm_h;
  
  zp_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  zm_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  Gm_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm);

  fread(tim,sizeof(float),1,restart);

  fread(zp_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
  cudaMemcpy(zp, zp_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
  fread(zm_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
  cudaMemcpy(zm, zm_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);

  fread(Gm_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm, 1, restart);
  cudaMemcpy(Gm, Gm_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm, cudaMemcpyHostToDevice);
    
  fclose(restart);

  free(zp_h);
  free(zm_h);
  free(Gm_h);
  
  
}


// Initialize diagnostics
void init_diag(){

	  char str[255];

      /////////////////////////////////////////////////////////////////
      //Alfven diagnostics
      /////////////////////////////////////////////////////////////////
	  strcpy(str, runname);
	  strcat(str, ".awkpar");
      alf_kzfile = fopen( str, "w+");

	  strcpy(str, runname);
	  strcat(str, ".awkparkperp");
      alf_kzkpfile = fopen( str, "w+");
      /////////////////////////////////////////////////////////////////
      //Slow mode diagnostics
      /////////////////////////////////////////////////////////////////

      // m-kz-kp spectrum
	  strcpy(str, runname);
	  strcat(str, ".slowkpar");
      slmkzfile = fopen(str, "w+");

	  strcpy(str, runname);
	  strcat(str, ".slowkparkperp");
      slmkzkpfile = fopen(str, "w+");
}

void close_diag()
{
    fclose(alf_kzfile);
    fclose(slmkzfile);

    fclose(alf_kzkpfile);
    fclose(slmkzkpfile);

}
void allocate_arrays()
  {

    cudaMalloc((void**) &f_d, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &g_d, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Gm, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm);

    cudaMalloc((void**) &ux, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &uy, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &dbx, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &dby, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &GmR, sizeof(float)*Nx*Ny*Nz*Nm);

    cudaMalloc((void**) &uxfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &uyfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &dbxfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &dbyfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Gmfld, sizeof(float)*Nx*Ny*Nz*Nm);


    cudaMalloc((void**) &xs, sizeof(float)*Nx*Ny);
    cudaMalloc((void**) &ys, sizeof(float)*Nx*Ny);



  }


