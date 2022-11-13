/* 
*** AVK 14/07/2014 notes:***
  
  2) Minimize temporary variables
  3) Change 3d FFT to 3 1d FFT
  6) Check how beta appears in normalization -- only through Lambda

  timestep.cu:
  1) Try and see if force_d can be removed. 
***AVK 25/07/2013 notes:***

 1) Write destroy_arrays functions

 3) Start fixing diagnostics/flow:
 	a) Define diagnostic calls depending on solve_alfven, solve_slow etc.
	b) Write a separate allocate/free file
	c) Rewrite Alfven diagnostics: think of minimizing arrays
	d) Rewrite slow diagnostics: think of minimizing arrays

 4) Do time advances in this order:
 	a) time_adv_phase_mix
	b) time_adv_alf : Done
	c) time_adv_full

 5) Build a test suite : Physics tests

***************
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>
#include <ctime>

/////////////////////////////
// Input parameters
/////////////////////////////
//Device
int devid;

// algorithm choices
bool debug, restart;
int nwrite, nforce; // Set nforce very large if driven is false
float maxdt, cfl;

// Initial conditions
bool decaying, driven, orszag_tang, noise;

//Computation Grid
__constant__ int Nx, Ny, Nz, zThreads;
__constant__ int Nm;
__constant__ float X0, Y0, Z0;
int nsteps;
float endtime;

// Grid, block setup for GPU
dim3 dimGrid, dimBlock;
int totalThreads;

// forcing
int nkstir;
int *kstir_x, *kstir_y, *kstir_z;
int gm_nkstir;
int *gm_kstir_x, *gm_kstir_y, *gm_kstir_z;
float fampl, gm_fampl;

// dissipation
int alpha_z, alpha_hyper;
float nu_kz, nu_hyper;

// slow modes
float beta, lambda_user;
int alpha_m, alpha_kp_g, alpha_kz_g;
float nu_coll, nu_kp_g, nu_kz_g;
float tau, Zcharge;
int lambsign;
float Lambda;
int force_m;

// Internal variables

char *runname;
char *restartname;
char stopfile[255];
bool file_exists(char *filename);
void read_namelist(char *filename);
void restartRead(cuComplex* zp, cuComplex* zm, cuComplex* Gm, float* tim);
void restartWrite(cuComplex* zp, cuComplex* zm, cuComplex* Gm, float tim);
FILE *energyfile, *alf_kzkpfile;
FILE *energyfile2, *alf_kparkperpfile;
FILE *slowfile, *slmkzkpfile, *slfluxfile;
FILE *slowfile2, *slmkparkperpfile, *slfluxfile2;
FILE *pmfile;
FILE *pmfile2;

//Dummy arrays
cuComplex *temp1, *temp2, *temp3, *temp4;
cuComplex *padded;
cuComplex *dx, *dy;
float *fdxR, *fdyR, *gdxR, *gdyR;
float *GmR;
float *uxfld, *uyfld, *dbxfld, *dbyfld;
float *Gmfld;

cufftHandle plan_C2R, plan_R2C, plan2d_C2R;


//////////////////////////////////////////////////////////////////////
// Include files
//////////////////////////////////////////////////////////////////////
#include "c_fortran_namelist3.c"
#include "device_funcs.cu"
#include "k_funcs.cu"
#include "zderiv_kernel.cu"
#include "reduc_kernel.cu"
#include "work_kernel.cu"
#include "maxReduc.cu"
#include "sumReduc_nopad.cu"
#include "init_kernel.cu"
#include "init_func.cu"
#include "nlps_kernel.cu"
#include "timestep_kernel.cu"
#include "diag_kernel.cu"
#include "damping_kernel.cu"
#include "nlps.cu"
#include "fldfol_funcs.cu"
#include "diagnostics.cu"
#include "nonlin.cu"
#include "courant.cu"
#include "forcing.cu"
#include "slowmodes.cu"
#include "timestep.cu"

// Declare all required arrays
// Declare host variables
cuComplex *f, *g;

// Declare device variables
cuComplex *f_d, *g_d;
cuComplex *f_d_tmp, *g_d_tmp;
cuComplex *Gm, *Gm_tmp;


// Declare functions
void allocate_arrays();
void destroy_arrays();
void setup_device();
void setup_grid_block();

//////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
  if ( argc < 1 ) 
    {
      printf( "Usage: ./gandalf runname");
    }
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

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start, 0);
  

      // Allocate arrays
      allocate_arrays();

	  // Setup dimgrid and dimblock
      setup_grid_block();

	  // Stopfile
      strcpy(stopfile, runname);
      strcat(stopfile, ".stop");

	  // Initialize diagnostics
	  init_diag();

	  // Create FFT plans
	  fft_plan_create();

	//////////////////////////////////////////////////////////
    float dt = 1.e-2;
    float tim=0;
    int istep=0;
    srand( (unsigned) time(NULL));
	//////////////////////////////////////////////////////////
	//If not restarting, initialize
	if(!restart){
		if(debug) printf("Not restarting. \n");

		//////////////////////////////////////////////////////////
		// Initialize Phi and Psi
		finit(f,g);
		
		// Initialize Slow modes.
        //slow_init(Gm); 
        for(int m=0; m<Nm; m++) {
            zero<<<dimGrid, dimBlock>>>(Gm +m*Nx*(Ny/2+1)*Nz, Nx, Ny/2+1, Nz);
            zero<<<dimGrid, dimBlock>>>(Gm_tmp +m*Nx*(Ny/2+1)*Nz, Nx, Ny/2+1, Nz);
        }
			   
        // Transfer the fields to Device
        cudaMemcpy(f_d, f, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
        if(debug) printf("f Initialization on device: %s\n",cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(g_d, g, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
        if(debug) printf("g Initialization on device: %s\n",cudaGetErrorString(cudaGetLastError()));
     }
	// If restarting:
	else{
		if(debug) printf("Restarting. \n");
		restartRead(f_d, g_d, Gm, &tim);

		printf("Time after restart: %f \n", tim);
	}

	//////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////

    // Zeroth step
    courant(&dt, f_d, g_d);
    printf("dt = %f\n", dt);
    advance(f_d_tmp, f_d, g_d_tmp, g_d, Gm_tmp, Gm, dt, istep);
    diagnostics(f_d, g_d, Gm, Gm_tmp, tim);
    istep++;

    while(istep < nsteps) {

		if(istep % nwrite ==0) {
			printf("%f      %d\n",tim,istep);
			printf("dt = %f\n",dt);
			diagnostics(f_d, g_d, Gm, Gm_tmp, tim);
			printf("%s\n",cudaGetErrorString(cudaGetLastError()));
			}

		// Check CFL condition
		courant(&dt, f_d, g_d);

		// Time advance RMHD & slow mode equations
		advance(f_d_tmp, f_d, g_d_tmp, g_d, Gm_tmp, Gm, dt, istep);

		tim+=dt;
		istep++;

		// Check if stopfile exists.
		if(file_exists(stopfile)) {
			printf("Simulation stopped by user with stopfile \n");
			break;
		}
	
  } 

    diagnostics(f_d, g_d, Gm, Gm_tmp, tim);

	if(debug) printf("Before restart write \n");
	restartWrite(f_d, g_d, Gm, tim);
    if(debug) printf("Restart write done \n");
    printf("Done.\n");
	
	
	float elapsed_time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Total time(ms) = %f \n", elapsed_time);
	printf("Average time (ms) = %f \n", elapsed_time/istep);

	////////////////////
	// Clean up Area
	////////////////////

	// Destroy fft plan and close diagnostics
	fft_plan_destroy(); 
    close_diag();

	// Free all arrays
	destroy_arrays();
	if(driven) {free(kstir_x); free(kstir_y); free(kstir_z);}

	}  	
	return 0;
}


bool file_exists(char *filename){
	if(FILE *file = fopen(filename, "r"))
	{
		fclose(file);
		return true;
	 }
	return false;
}

void read_namelist(char* filename)
{
  struct fnr_struct namelist_struct = fnr_read_namelist_file(filename);

  int debug_i, restart_i;
  int decaying_i, driven_i, orszag_tang_i, noise_i;
  
  //Device
  if(fnr_get_int(&namelist_struct, "dev", "devid", &devid)) devid = 0;
  setup_device();
  // algo 
  if(fnr_get_int(&namelist_struct, "algo", "debug", &debug_i)) debug_i=0;
   if(debug_i == 0) { debug = false; }
   else {debug = true;}
  if(fnr_get_int(&namelist_struct, "algo", "restart", &restart_i)) restart_i = 0;
   if(restart_i == 0) { restart = false;}
   else {restart = true;}
  if(restart) fnr_get_string(&namelist_struct, "algo", "rest", &restartname);
  if(fnr_get_int(&namelist_struct, "algo", "nwrite", &nwrite)) nwrite = 1;
  if(fnr_get_int(&namelist_struct, "algo", "nforce", &nforce)) nforce = 1;
  if(fnr_get_float(&namelist_struct, "algo", "maxdt", &maxdt)) maxdt = .1;
  if(fnr_get_float(&namelist_struct, "algo", "cfl", &cfl)) cfl = .1;

  if(debug) printf("Algo read \n");

  // Initial conditions
  fnr_get_int(&namelist_struct, "init", "decaying", &decaying_i);
   if(decaying_i == 0) { decaying = false; }
   else {decaying = true;}
  fnr_get_int(&namelist_struct, "init", "driven", &driven_i);
   if(driven_i == 0) { driven = false; }
   else {driven = true;}
   fnr_get_int(&namelist_struct, "init", "orszag_tang", &orszag_tang_i);
   if(orszag_tang_i == 0) { orszag_tang = false; }
   else {orszag_tang = true;}
   if(orszag_tang) printf("orszag_tang \n");
   fnr_get_int(&namelist_struct, "init", "noise", &noise_i);
   if(noise_i == 0) { noise = false; }
   else {noise = true;}

  if(debug) printf("Initial conditions read \n");

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
  if(fnr_get_int(&namelist_struct, "grid", "nsteps", &nsteps)) {
    printf("nsteps not specified. Exiting ... \n");
    exit(1);
    }

  if(debug) printf("Grid read \n");

  // Dissipation
  if(fnr_get_int(&namelist_struct, "dissipation", "alpha_z", &alpha_z)) alpha_z = 2;
  if(fnr_get_float(&namelist_struct, "dissipation", "nu_kz", &nu_kz)) nu_kz = 1.0f;
  if(fnr_get_int(&namelist_struct, "dissipation", "alpha_hyper", &alpha_hyper)) alpha_hyper = 2;
  if(fnr_get_float(&namelist_struct, "dissipation", "nu_hyper", &nu_hyper)) nu_hyper = 1.0f;

  if(debug) printf("Dissipation read \n");

  // Forcing
   fnr_get_int(&namelist_struct, "forcing", "nkstir", &nkstir);
   fnr_get_int(&namelist_struct, "forcing", "gm_nkstir", &gm_nkstir);
   fnr_get_float(&namelist_struct, "forcing", "fampl", &fampl);
   fnr_get_float(&namelist_struct, "forcing", "gm_fampl", &gm_fampl);

   if(driven){

		kstir_x = (int*) malloc(sizeof(int)*nkstir);
		kstir_y = (int*) malloc(sizeof(int)*nkstir);
		kstir_z = (int*) malloc(sizeof(int)*nkstir);

		gm_kstir_x = (int*) malloc(sizeof(int)*gm_nkstir);
		gm_kstir_y = (int*) malloc(sizeof(int)*gm_nkstir);
		gm_kstir_z = (int*) malloc(sizeof(int)*gm_nkstir);
	

 // Initialize Forcing modes
   char tmp_str[255];
   char buffer[20];
   int f_k;
   for(int ikstir=0; ikstir<nkstir; ikstir++) {
     
     strcpy(tmp_str,"stir_");
     sprintf(buffer,"%d", ikstir);
	 strcat(tmp_str, buffer);
	 fnr_get_int(&namelist_struct, tmp_str, "kx", &f_k);
	 kstir_x[ikstir] = (f_k + Nx) % Nx;
	 fnr_get_int(&namelist_struct, tmp_str, "ky", &f_k);
	 kstir_y[ikstir] = (f_k + Ny) % Ny;
	 fnr_get_int(&namelist_struct, tmp_str, "kz", &f_k);
	 kstir_z[ikstir] = (f_k + Nz) % Nz;

    }
   for(int ikstir=0; ikstir<gm_nkstir; ikstir++) {
     
     strcpy(tmp_str,"gm_stir_");
     sprintf(buffer,"%d", ikstir);
	 strcat(tmp_str, buffer);
	 fnr_get_int(&namelist_struct, tmp_str, "kx", &f_k);
	 gm_kstir_x[ikstir] = (f_k + Nx) % Nx;
	 fnr_get_int(&namelist_struct, tmp_str, "ky", &f_k);
	 gm_kstir_y[ikstir] = (f_k + Ny) % Ny;
	 fnr_get_int(&namelist_struct, tmp_str, "kz", &f_k);
	 gm_kstir_z[ikstir] = (f_k + Nz) % Nz;

    }
   }

  if(debug) printf("Forcing read \n");

  // Slow
  	fnr_get_int(&namelist_struct, "grid", "Nm", &Nm);
    cudaMemcpyToSymbol(Nm, &Nm, sizeof(int),0,cudaMemcpyHostToDevice);
    if(debug) printf("Nm read \n");
	if(fnr_get_float(&namelist_struct, "slow" ,"beta", &beta)) beta = 1.0f;
    if(debug) printf("beta read \n");
	if(fnr_get_float(&namelist_struct, "slow" ,"Zcharge", &Zcharge)) Zcharge = 1.0f;
	if(fnr_get_float(&namelist_struct, "slow" ,"tau", &tau)) tau = 1.0f;
	if(fnr_get_float(&namelist_struct, "slow" ,"nu_coll", &nu_coll)) nu_coll = 1.0f;
	if(fnr_get_int(&namelist_struct, "slow" ,"alpha_m", &alpha_m)) alpha_m = 2;
	if(fnr_get_int(&namelist_struct, "slow" ,"alpha_kp_g", &alpha_kp_g)) alpha_kp_g = 2;
	if(fnr_get_int(&namelist_struct, "slow" ,"alpha_kz_g", &alpha_kz_g)) alpha_kz_g = 2;
	if(fnr_get_float(&namelist_struct, "slow" ,"nu_kp_g", &nu_kp_g)) nu_kp_g = 1.0f;
	if(fnr_get_float(&namelist_struct, "slow" ,"nu_kz_g", &nu_kz_g)) nu_kz_g = 1.0f;
    if(debug) printf("slow dissipation read \n");
	if(fnr_get_int(&namelist_struct, "slow", "lambsign", &lambsign)) lambsign = 1;
	if(debug) printf("lambsign = %d \n", lambsign);
    Lambda = -tau/Zcharge + 1.0/beta + lambsign*sqrt((1.0 + tau/Zcharge)*(1.0 + tau/Zcharge) + 1.0/(beta*beta)); // Lambda depends on beta.
	if(fnr_get_float(&namelist_struct, "slow", "lambda_user", &lambda_user)) lambda_user = 0.0f; 
	if(lambda_user !=0.0) Lambda = lambda_user;
	printf("Lambda = %g\n", Lambda);
	if(fnr_get_int(&namelist_struct, "slow" ,"force_m", &force_m)) force_m = 1;


  if(debug) printf("Slow read \n");

}


//////////////////////////////////////////////////////////////////////
// Restart routines
//////////////////////////////////////////////////////////////////////

void restartWrite(cuComplex* zp, cuComplex* zm, cuComplex* Gm, float tim)
{
  if(debug) printf("Entering restart write\n");
  char str[255];
  FILE *restart;
  strcpy(str, runname);
  strcat(str,".res");
  restart = fopen(str, "wb");
  if(debug) printf("Opened restart file to write\n");

  cuComplex *zp_h;
  cuComplex *zm_h;
  cuComplex *Gm_h;
  if(debug) printf("Declared arrays \n");
  
  if(debug) printf("Nx = %d, Ny = %d, Nz = %d, Nm = %d \n", Nx, Ny, Nz, Nm);

  zp_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMemcpy(zp_h, zp, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
  if(debug) printf("Copying over zp %s\n",cudaGetErrorString(cudaGetLastError()));
  zm_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMemcpy(zm_h, zm, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
  if(debug) printf("Copying over zm %s\n",cudaGetErrorString(cudaGetLastError()));
  Gm_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm);
  cudaMemcpy(Gm_h, Gm, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm, cudaMemcpyDeviceToHost);
  if(debug) printf("Copying over Gm %s\n",cudaGetErrorString(cudaGetLastError()));
  if(debug) printf("Allocated arrays \n");
  
  fwrite(&tim,sizeof(float),1,restart);
  if(debug) printf("Wrote istep, tim \n");
  
  fwrite(zp_h,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,1,restart);
  if(debug) printf("Wrote zp \n");
  fwrite(zm_h,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,1,restart);
  if(debug) printf("Wrote zm \n");
  fwrite(Gm_h,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm,1,restart);
  if(debug) printf("Wrote Gm \n");
  
  
  fclose(restart);

  free(zp_h); free(zm_h); 
  free(Gm_h); 
  
  
}


void restartRead(cuComplex* zp, cuComplex* zm, cuComplex* Gm, float* tim)
{
  char str[255];
  FILE *restart;
  strcpy(str, restartname);
  strcat(str,".res");
  if(debug) printf("Restartfile = %s \n", str);
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

	}

////////////////////////////////////////
// Array allocation/destruction functions
////////////////////////////////////////
// Allocate arrays
void allocate_arrays()
 {

      printf("Allocating arrays...\n");
      // Allocate host arrays
      f = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      g = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

      // Allocate device arrays
      cudaMalloc((void**) &f_d, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      cudaMalloc((void**) &g_d, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      cudaMalloc((void**) &f_d_tmp, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      cudaMalloc((void**) &g_d_tmp, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

      cudaMalloc((void**) &Gm, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm);
      cudaMalloc((void**) &Gm_tmp, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm);

      cudaMalloc((void**) &temp1, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      cudaMalloc((void**) &temp2, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      cudaMalloc((void**) &temp3, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      cudaMalloc((void**) &temp4, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

      cudaMalloc((void**) &fdxR, sizeof(float)*Nx*Ny*Nz);
      cudaMalloc((void**) &fdyR, sizeof(float)*Nx*Ny*Nz);
      cudaMalloc((void**) &gdxR, sizeof(float)*Nx*Ny*Nz);
      cudaMalloc((void**) &gdyR, sizeof(float)*Nx*Ny*Nz);
      cudaMalloc((void**) &GmR, sizeof(float)*Nx*Ny*Nz*Nm);

      cudaMalloc((void**) &uxfld, sizeof(float)*Nx*Ny*Nz);
      cudaMalloc((void**) &uyfld, sizeof(float)*Nx*Ny*Nz);
      cudaMalloc((void**) &dbxfld, sizeof(float)*Nx*Ny*Nz);
      cudaMalloc((void**) &dbyfld, sizeof(float)*Nx*Ny*Nz);
      cudaMalloc((void**) &Gmfld, sizeof(float)*Nx*Ny*Nz*Nm);

	  cudaMalloc((void**) &dx, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
	  cudaMalloc((void**) &dy, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

      cudaMalloc((void**) &padded, sizeof(cuComplex)*Nx*Ny*Nz);

 }
void destroy_arrays(){

	// Destroy host arrays
	free(f); free(g);

	// Destroy device arrays containing fields
	cudaFree(f_d); cudaFree(g_d);
    cudaFree(Gm); cudaFree(Gm_tmp);


	// Destroy dummy arrays
		// Fields
	cudaFree(f_d_tmp); cudaFree(g_d_tmp);

		// nonlin
	cudaFree(temp1); cudaFree(temp2); cudaFree(temp3); cudaFree(temp4);

	cudaFree(fdxR); cudaFree(fdyR); cudaFree(gdxR); cudaFree(gdyR);
    cudaFree(GmR);
	cudaFree(uxfld); cudaFree(uyfld); cudaFree(dbxfld); cudaFree(dbyfld);
    cudaFree(Gmfld);

	cudaFree(dx); cudaFree(dy);

		// courant
	cudaFree(padded);
}


