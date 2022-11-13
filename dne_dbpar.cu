#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>

int devid;
__constant__ int Nx, Ny, Nz, Nm;
__constant__ int zThreads;
int totalThreads;
char *Prunname, *Mrunname;
FILE *dne_dbpar_file;

__constant__ float X0, Y0, Z0;

cuComplex *Pf_d, *Pg_d, *PGm;
float *Pux, *Puy, *Pdbx, *Pdby, *PGmR;
float *xs, *ys;

float *Puxfld, *Puyfld, *Pdbxfld, *Pdbyfld;
float *PGmfld; 

cuComplex *Mf_d, *Mg_d, *MGm;
float *Mux, *Muy, *Mdbx, *Mdby, *MGmR;

float *Muxfld, *Muyfld, *Mdbxfld, *Mdbyfld;
float *MGmfld; 

cuComplex* dne, *dbpar;

cufftHandle plan_C2R, plan_R2C;
dim3 dimGrid, dimBlock;

float tau, Zcharge, beta, sigma;
float Lambda;
int lambsign;

#include "c_fortran_namelist3.c"
#include "device_funcs.cu"
#include "k_funcs.cu"
#include "work_kernel.cu"
#include "nlps_kernel.cu"
#include "diag_kernel.cu"
#include "fldfol_funcs.cu"

void fft_plan_create();
void fft_plan_destroy();

void read_namelist(char* filename);
void allocate_arrays();
void init_diag();
void restartRead(char* runname, cuComplex* zp, cuComplex*zm, cuComplex* Gm, float* tim);
void close_diag();
void energy_kz_kperp(cuComplex* dne, cuComplex* dbpar, FILE* file);
void PM_to_dne_dbpar(float* PGmfld, float* MGmfld, cuComplex* dne, cuComplex* dbpar);

void setup_device();
void setup_grid_block();

int main (int argc, char* argv[])
{
    if(argc<2) printf( "Usage: ./dne_dbpar Prun Mrun");
    else
    {
    Prunname = argv[1];
    char str[255];
    strcpy(str, Prunname);
    strcat(str, ".in");
	printf("Prun: %s \n", str);

    Mrunname = argv[2];
    strcpy(str, Mrunname);
    strcat(str, ".in");
	printf("Mrun: %s \n", str);

	read_namelist(str);
    setup_grid_block();
    allocate_arrays();

    init_diag();

    fft_plan_create();

    float Ptim=0, Mtim=0;
    restartRead(Prunname, Pf_d, Pg_d, PGm, &Ptim);
    restartRead(Mrunname, Mf_d, Mg_d, MGm, &Mtim);
    if (Ptim != Mtim) printf("!!!Times don't match!!!\n");
    // Calculate ux, uy, dbx, dby
    calculate_uxy_dbxy (Pf_d, Pg_d, Pux, Puy, Pdbx, Pdby);
    calculate_uxy_dbxy (Mf_d, Mg_d, Mux, Muy, Mdbx, Mdby);

    // Convert Gm to real space
    for (int m=0; m<Nm; m++) 
        {
        if(cufftExecC2R(plan_C2R, PGm+m*Nx*(Ny/2+1)*Nz, PGmR+m*Nx*Ny*Nz) != CUFFT_SUCCESS)
        printf("PGm conversion to real space failed for m=%d, %d\n", m, cufftExecC2R(plan_C2R, PGm+m*Nx*(Ny/2+1)*Nz, PGmR+m*Nx*Ny*Nz));
        if(cufftExecC2R(plan_C2R, MGm+m*Nx*(Ny/2+1)*Nz, MGmR+m*Nx*Ny*Nz) != CUFFT_SUCCESS) printf("MGm conversion to real space failed for m=%d\n", m);
        }

    cudaFree(Pf_d); cudaFree(Pg_d); cudaFree(PGm);
    cudaFree(Mf_d); cudaFree(Mg_d); cudaFree(MGm);

    fldfol(Pux, Puy, Pdbx, Pdby, PGmR, Puxfld, Puyfld, Pdbxfld, Pdbyfld, PGmfld);
    printf("After first fldfol: %s\n",cudaGetErrorString(cudaGetLastError()));
    fldfol(Mux, Muy, Mdbx, Mdby, MGmR, Muxfld, Muyfld, Mdbxfld, Mdbyfld, MGmfld);
    printf("After second fldfol: %s\n",cudaGetErrorString(cudaGetLastError()));

    cudaFree(Pux); cudaFree(Puy);
    cudaFree(Pdbx); cudaFree(Pdby);
    cudaFree(PGmR);
    cudaFree(Mux); cudaFree(Muy);
    cudaFree(Mdbx); cudaFree(Mdby);
    cudaFree(MGmR);
    cudaFree(xs); cudaFree(ys);

    PM_to_dne_dbpar(PGmfld, MGmfld, dne, dbpar);
    printf("After Pm_to_dne_dbpar: %s\n",cudaGetErrorString(cudaGetLastError()));

    cudaFree(PGmfld); cudaFree(MGmfld);

    squareComplex<<<dimGrid, dimBlock>>>(dne);
    squareComplex<<<dimGrid, dimBlock>>>(dbpar);

    fixFFT<<<dimGrid,dimBlock>>>(dne);
    fixFFT<<<dimGrid,dimBlock>>>(dbpar);

    energy_kz_kperp(dne, dbpar, dne_dbpar_file);
    printf("After energy_kz_kperp: %s\n",cudaGetErrorString(cudaGetLastError()));

    close_diag();
    cudaFree(dne); cudaFree(dbpar);
    fft_plan_destroy();


    }

}

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
	if(fnr_get_float(&namelist_struct, "slow" ,"beta", &beta)) beta = 1.0f;
	if(fnr_get_float(&namelist_struct, "slow" ,"Zcharge", &Zcharge)) Zcharge = 1.0f;
	if(fnr_get_float(&namelist_struct, "slow" ,"tau", &tau)) tau = 1.0f;
	if(fnr_get_int(&namelist_struct, "slow", "lambsign", &lambsign)) lambsign = 1;
    Lambda = -tau/Zcharge + 1.0/beta + lambsign*sqrt((1.0 + tau/Zcharge)*(1.0 + tau/Zcharge) + 1.0/(beta*beta)); // Lambda depends on beta.
    sigma = 1.0 + tau/Zcharge + sqrt((1.0+tau/Zcharge)*(1.0+tau/Zcharge) + 1./beta/beta);

}

void allocate_arrays()
{
    cudaMalloc((void**) &Pf_d, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Pg_d, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &PGm, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm);
    cudaMalloc((void**) &Mf_d, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Mg_d, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &MGm, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm);

    cudaMalloc((void**) &Pux, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Puy, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Pdbx, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Pdby, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &PGmR, sizeof(float)*Nx*Ny*Nz*Nm);

    cudaMalloc((void**) &Puxfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Puyfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Pdbxfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Pdbyfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &PGmfld, sizeof(float)*Nx*Ny*Nz*Nm);

    cudaMalloc((void**) &Mux, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Muy, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Mdbx, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Mdby, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &MGmR, sizeof(float)*Nx*Ny*Nz*Nm);

    cudaMalloc((void**) &Muxfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Muyfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Mdbxfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &Mdbyfld, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &MGmfld, sizeof(float)*Nx*Ny*Nz*Nm);

    cudaMalloc((void**) &xs, sizeof(float)*Nx*Ny);
    cudaMalloc((void**) &ys, sizeof(float)*Nx*Ny);

    cudaMalloc((void**) &dne, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &dbpar, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);


}

void restartRead(char* runname, cuComplex* zp, cuComplex* zm, cuComplex* Gm, float* tim)
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

void PM_to_dne_dbpar(float* PGmfld, float* MGmfld, cuComplex* dne, cuComplex* dbpar)
{

    cuComplex *PGmfld_k, *MGmfld_k;
    cudaMalloc((void**) &PGmfld_k, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &MGmfld_k, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

    if(cufftExecR2C(plan_R2C, PGmfld, PGmfld_k) != CUFFT_SUCCESS) printf("PGm FFT failed\n");
    if(cufftExecR2C(plan_R2C, MGmfld, MGmfld_k) != CUFFT_SUCCESS) printf("MGm FFT failed\n");

    float scalar = -2.0*tau/sigma/Zcharge/beta;
    addsubt<<<dimGrid, dimBlock>>>(dne, MGmfld_k, PGmfld_k, scalar);
    scalar = 1.0/(1.0 - 2.0*(1. + tau/Zcharge)/beta/sigma/sigma);
    scale<<<dimGrid, dimBlock>>>(dne, scalar);

    scalar = -(1.0+Zcharge/tau)/sigma;
    addsubt<<<dimGrid, dimBlock>>>(dbpar, PGmfld_k, dne, scalar);

    scale<<<dimGrid, dimBlock>>>(dne, 1.0f/((float) Nx*Ny*Nz));
    scale<<<dimGrid, dimBlock>>>(dbpar, 1.0f/((float) Nx*Ny*Nz));
    mask<<<dimGrid, dimBlock>>>(dne);
    mask<<<dimGrid, dimBlock>>>(dbpar);

    cudaFree(PGmfld_k); cudaFree(MGmfld_k);


}

void energy_kz_kperp(cuComplex* kPhi, cuComplex* kA, FILE* alf_kzkpfile)
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
            kz_kpshellsum<<<dimGrid, dimBlock>>>(kPhi, ikp, kinEnergy_kp);
            kz_kpshellsum<<<dimGrid, dimBlock>>>(kA, ikp, magEnergy_kp);
        }

        cudaMemcpy(kinEnergy_kp_h, kinEnergy_kp, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);
        cudaMemcpy(magEnergy_kp_h, magEnergy_kp, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);


    for(int ikz=0; ikz<=(Nz-1)/3; ikz++){
        for(int ikp =1; ikp<ikpmax; ikp++){
            int ikz_kp = ikz + Nz*ikp;
            int mikz_kp = (Nz-ikz) + Nz*ikp;
            fprintf(alf_kzkpfile, "%g \t %g \t %1.12e \t %1.12e\n", kz(ikz), ((float) ikp/ikpmax)*kpmax, kinEnergy_kp_h[ikz_kp] + kinEnergy_kp_h[mikz_kp], magEnergy_kp_h[ikz_kp] + magEnergy_kp_h[mikz_kp]);
        }

    fprintf(alf_kzkpfile, "\n");
        }
    fprintf(alf_kzkpfile, "\n");
    
    cudaFree(kinEnergy_kp); cudaFree(magEnergy_kp);
    free(kinEnergy_kp_h); free(magEnergy_kp_h);
    
    
}    

void init_diag()
{
	  char str[255];
      strcpy(str, Prunname);
      strcat(str, ".dnedbpar");
      dne_dbpar_file = fopen(str, "w+");

}

void close_diag()
{
    fclose(dne_dbpar_file);
}
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


