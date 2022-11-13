//////////////////////////////////////////////////////////////////////
	// Slow mode diagnostics 
//////////////////////////////////////////////////////////////////////
////////////////////////////////////////
// Slow mode energy diagnostic
	void slenergy(cuComplex* Gm2, float tim, int m, FILE* slowfile){
		
	  cuComplex *Gmen_h;
	  Gmen_h = (cuComplex*) malloc(sizeof(cuComplex));

	  sumReduc(Gmen_h, Gm2, padded);
	  //Gmen_h[0].x = Gmen_h[0].x/((float) Nx*Ny*Nz*1.);
	  printf("m = %d \t energy = %g\t", m, Gmen_h[0].x);
	  fprintf(slowfile, "%g\t", Gmen_h[0].x);

	  free(Gmen_h); 

	}
// Slow mode kzkp spectra
void slow_kzkp(cuComplex* Gm2,float tim, int m, FILE* slmkzkpfile) 
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
  	fprintf(slmkzkpfile, "%g\t %d\t %g\t %g\t %g\n", tim, m, kz(0), ((float) ikp), Gmenkzkp_h[ikz_kp]);
  
    }
    fprintf(slmkzkpfile, "\n");
  for(int ikz=0; ikz<=(Nz-1)/3; ikz++){
    for(int ikp=1; ikp<ikpmax; ikp++){
        int ikz_kp = ikz+Nz*ikp;
        fprintf(slmkzkpfile, "%g\t %d\t %g\t %g\t %g\n", tim, m, kz(ikz), ((float) ikp), Gmenkzkp_h[ikz_kp]);

      }
      fprintf(slmkzkpfile, "\n");
  }

  for(int ikz=2*Nz/3+1; ikz<Nz; ikz++){
    for(int ikp=1; ikp<ikpmax; ikp++){
        int ikz_kp = ikz+Nz*ikp;
        fprintf(slmkzkpfile, "%g\t %d\t %g\t %g\t %g\n", tim, m, kz(ikz), ((float) ikp), Gmenkzkp_h[ikz_kp]);

      }
      fprintf(slmkzkpfile, "\n");
  }

  cudaFree(Gmenkzkp);
  free(Gmenkzkp_h);


}
void phmix(cuComplex* Gm2, float tim, int m, FILE *pmfile) {

  float kpmax = ( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );
  int ikpmax = (int) ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );

  float *Gpen, *Gmen;
  cudaMalloc((void**) &Gpen, sizeof(float)*ikpmax*Nz);
  cudaMalloc((void**) &Gmen, sizeof(float)*ikpmax*Nz);

  zero<<<ikpmax, Nz>>>(Gpen, Nz*ikpmax, 1, 1);
  zero<<<ikpmax, Nz>>>(Gmen, Nz*ikpmax, 1, 1);

  float *Gpen_h, *Gmen_h;
  Gpen_h = (float*) malloc(sizeof(float)*ikpmax*Nz);
  Gmen_h = (float*) malloc(sizeof(float)*ikpmax*Nz);


  for (int ikp=1; ikp<ikpmax; ikp++){
    kz_kpshellsum<<<dimGrid, dimBlock>>>(Gm2, ikp, Gpen);
    kz_kpshellsum<<<dimGrid, dimBlock>>>(Gm2 + Nx*(Ny/2+1)*Nz, ikp, Gmen);
  }


  cudaMemcpy(Gpen_h, Gpen, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(Gmen_h, Gmen, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);

  for(int ikp=1; ikp<ikpmax; ikp++){
  	int ikz_kp = 0+Nz*ikp;
  	fprintf(pmfile, "%g\t %d\t %g\t %g\t %g\t %g\n", tim, m, kz(0), ((float) ikp), Gpen_h[ikz_kp], Gmen_h[ikz_kp]);
  
    }
    fprintf(pmfile, "\n");

  for(int ikz=0; ikz<=(Nz-1)/3; ikz++){
    for(int ikp=1; ikp<ikpmax; ikp++){
        int ikz_kp = ikz+Nz*ikp;
        fprintf(pmfile, "%g\t %d\t %g\t %g\t %g\t %g\n", tim, m, kz(ikz), ((float) ikp), Gpen_h[ikz_kp], Gmen_h[ikz_kp]);

      }
      fprintf(pmfile, "\n");
  }

  for(int ikz=2*Nz/3+1; ikz<Nz; ikz++){
    for(int ikp=1; ikp<ikpmax; ikp++){
        int ikz_kp = ikz+Nz*ikp;
        fprintf(pmfile, "%g\t %d\t %g\t %g\t %g\t %g\n", tim, m, kz(ikz), ((float) ikp), Gpen_h[ikz_kp], Gmen_h[ikz_kp]);
      }
      fprintf(pmfile, "\n");
  }

  cudaFree(Gpen); cudaFree(Gmen);
  free(Gpen_h); free(Gmen_h);
}

void sl_m_flux(cuComplex* Gm, cuComplex* flux, int m, float tim, FILE* slfluxfile) {

	// Initialize flux array to zero
	zero<<<dimGrid, dimBlock>>>(flux, Nx, Ny/2+1, Nz);

	//flux = i*kpar*sqrt(2*(m+1))*g_{m+1} * conjugate(g_m)
	// Real part of this array is Gamma_mk
	m_flux<<<dimGrid, dimBlock>>>(Gm+m*Nx*(Ny/2+1)*Nz, flux, m);

    float kpmax = ( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );
    int ikpmax = (int) ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );

    float *Gamma_mk;
    cudaMalloc((void**) &Gamma_mk, sizeof(float)*ikpmax*Nz);
	zero<<<ikpmax, Nz>>>(Gamma_mk, Nz*ikpmax, 1, 1);

	for(int ikp=0; ikp<ikpmax; ikp++)
		kz_kpshellsum<<<dimGrid, dimBlock>>>(flux, ikp, Gamma_mk);

	float *Gamma_mk_h;
	Gamma_mk_h = (float*) malloc(sizeof(float)*ikpmax*Nz);

	cudaMemcpy(Gamma_mk_h, Gamma_mk, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);

	for(int ikz=0; ikz<=(Nz-1)/3; ikz++){
  	  for(int ikp=1; ikp<ikpmax; ikp++){
		int ikz_kp = ikz + Nz*ikp;
		fprintf(slfluxfile, "%g\t %d\t %g\t %g\t %g\n", tim, m, kz(ikz), ((float) ikp), Gamma_mk_h[ikz_kp]);
	  }

	  fprintf(slfluxfile, "\n");
	}

	for(int ikz=2*Nz/3+1; ikz<Nz; ikz++){
  	  for(int ikp=1; ikp<ikpmax; ikp++){
		int ikz_kp = ikz + Nz*ikp;
		fprintf(slfluxfile, "%g\t %d\t %g\t %g\t %g\n", tim, m, kz(ikz), ((float) ikp), Gamma_mk_h[ikz_kp]);
	  }

	  fprintf(slfluxfile, "\n");
	}

	cudaFree(Gamma_mk);
	free(Gamma_mk_h);


}
//////////////////////////////////////////////////////////////////////
	// Alfven diagnostics
//////////////////////////////////////////////////////////////////////

// kz- Kperp spectra of Alfven waves

	void energy_kz_kperp(cuComplex* kPhi, cuComplex* kA, float time, FILE* alf_kzkpfile)
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

			if(debug) {printf("kz_kpshellsum: %s\n",cudaGetErrorString(cudaGetLastError())); }

			cudaMemcpy(kinEnergy_kp_h, kinEnergy_kp, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);
			cudaMemcpy(magEnergy_kp_h, magEnergy_kp, sizeof(float)*ikpmax*Nz, cudaMemcpyDeviceToHost);

			if(debug) { printf("Copying shell sums ME: %s\n",cudaGetErrorString(cudaGetLastError())); }

		for(int ikz=0; ikz<=(Nz-1)/3; ikz++){
			for(int ikp =1; ikp<ikpmax; ikp++){
				int ikz_kp = ikz + Nz*ikp;
				int mikz_kp = (Nz-ikz) + Nz*ikp;
				fprintf(alf_kzkpfile, "%g \t %g \t %g \t %1.12e \t %1.12e\n", time, kz(ikz), ((float) ikp/ikpmax)*kpmax, kinEnergy_kp_h[ikz_kp] + kinEnergy_kp_h[mikz_kp], magEnergy_kp_h[ikz_kp] + magEnergy_kp_h[mikz_kp]);
			}

		fprintf(alf_kzkpfile, "\n");
			}
		fprintf(alf_kzkpfile, "\n");
		
		cudaFree(kinEnergy_kp); cudaFree(magEnergy_kp);
		free(kinEnergy_kp_h); free(magEnergy_kp_h);
		
		
	}    

////////////////////////////////////////
// Total energy
	void energy(cuComplex* kPhi, cuComplex* kA, float time, FILE* energyfile)
	{
	  if(debug) {printf("Entering energy\n");}

	  cuComplex *padded;
	  cudaMalloc((void**) &padded, sizeof(cuComplex)*Nx*Ny*Nz);
		
		
	  cuComplex *totEnergy_h, *kinEnergy_h, *magEnergy_h;

	  totEnergy_h = (cuComplex*) malloc(sizeof(cuComplex));
	  kinEnergy_h = (cuComplex*) malloc(sizeof(cuComplex));
	  magEnergy_h = (cuComplex*) malloc(sizeof(cuComplex));

	  // integrate kPhi to find kinetic energy
	  //sumReduc(kinEnergy_h, kPhi, padded);
	  sumReduc_gen(kinEnergy_h, kPhi, padded, Nx, Ny, Nz);
	  if(debug){ printf("sumreduc kPhi: %s\n",cudaGetErrorString(cudaGetLastError())); }
		
	  // integrate kA to find magnetic energy
	  //sumReduc(magEnergy_h, kA, padded);
	  sumReduc_gen(magEnergy_h, kA, padded, Nx, Ny, Nz);
	  if(debug){ printf("sumreduc kA: %s\n",cudaGetErrorString(cudaGetLastError())); }
		
	  //calculate total energy
	  totEnergy_h[0].x = kinEnergy_h[0].x + magEnergy_h[0].x;
		
	  cudaFree(padded);

	  fprintf(energyfile, "\t%g\t%g\t%g\t%g\n", time, totEnergy_h[0].x, kinEnergy_h[0].x, magEnergy_h[0].x);
	  printf("Total Energy = %g\t Kin Energy = %g\t Magnetic Energy = %g\n", totEnergy_h[0].x, kinEnergy_h[0].x, magEnergy_h[0].x);

	  free(totEnergy_h); free(kinEnergy_h); free(magEnergy_h); 
		
	  if(debug) {printf("Exiting energy\n");}
		
	}    

//////////////////////////////////////////////////////////////////////
	// Main diagnostic calling functions
//////////////////////////////////////////////////////////////////////

void alf_diagnostics(cuComplex* kPhi, cuComplex* kA, float time, FILE* alf_kzkpfile,
FILE *energyfile){

		energy_kz_kperp(kPhi, kA, time, alf_kzkpfile);
		fflush(alf_kzkpfile);

		energy(kPhi, kA, time, energyfile);
		fflush(energyfile);


}

// Slow mode diagnostics
void slow_diagnostics(cuComplex* Gm, float time, FILE* slowfile, FILE*
slmkzkpfile, FILE* slfluxfile, FILE* pmfile){

    cuComplex* Gm_tmp;
    cudaMalloc((void**) &Gm_tmp, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm);
    cudaMemcpy(Gm_tmp, Gm, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm, cudaMemcpyDeviceToDevice);
    cuComplex *Gtilde_tmp;
    cudaMalloc((void**) &Gtilde_tmp, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*2);

	fprintf(slowfile, "%g\t",time);

	for(int m=0; m<Nm-1; m++) {


        // Calculate kz-kp spectra
	    squareComplex<<<dimGrid, dimBlock>>>(Gm_tmp + m*Nx*(Ny/2+1)*Nz);
	    fixFFT<<<dimGrid, dimBlock>>>(Gm_tmp + m*Nx*(Ny/2+1)*Nz);

		slenergy(Gm_tmp + m*Nx*(Ny/2+1)*Nz, time, m, slowfile);
		slow_kzkp(Gm_tmp + m*Nx*(Ny/2+1)*Nz, time, m, slmkzkpfile);

        zero<<<dimGrid, dimBlock>>>(Gtilde_tmp,Nx,Ny/2+1,Nz);
        zero<<<dimGrid, dimBlock>>>(Gtilde_tmp+Nx*(Ny/2+1)*Nz,Nx,Ny/2+1,Nz);

        // Calculate energy in Gm+, Gm- as defined by Alex
		Gm_to_Gtilde_p<<<dimGrid, dimBlock>>>(Gm +m*Nx*(Ny/2+1)*Nz, Gtilde_tmp, m);
		squareComplex<<<dimGrid, dimBlock>>>(Gtilde_tmp);
		fixFFT<<<dimGrid, dimBlock>>>(Gtilde_tmp);

		Gm_to_Gtilde_m<<<dimGrid, dimBlock>>>(Gm +m*Nx*(Ny/2+1)*Nz, Gtilde_tmp +Nx*(Ny/2+1)*Nz, m);
		squareComplex<<<dimGrid, dimBlock>>>(Gtilde_tmp + Nx*(Ny/2+1)*Nz);
		fixFFT<<<dimGrid, dimBlock>>>(Gtilde_tmp + Nx*(Ny/2+1)*Nz);

        phmix(Gtilde_tmp, time, m, pmfile);
        fprintf(pmfile,"\n");

		sl_m_flux(Gm, Gtilde_tmp, m, time, slfluxfile);
		fprintf(slfluxfile,"\n");


	}

    // m = Nm-1 non phmix diagnostics
    int m = Nm-1;
    squareComplex<<<dimGrid, dimBlock>>>(Gm + m*Nx*(Ny/2+1)*Nz);
    fixFFT<<<dimGrid, dimBlock>>>(Gm + m*Nx*(Ny/2+1)*Nz);

    slenergy(Gm + m*Nx*(Ny/2+1)*Nz, time, m, slowfile);
    slow_kzkp(Gm + m*Nx*(Ny/2+1)*Nz, time, m, slmkzkpfile);

    cudaFree(Gtilde_tmp);
    cudaFree(Gm_tmp);
	
    fprintf(slowfile, "\n");
	fprintf(slmkzkpfile, "\n");
	fprintf(pmfile, "\n");

	fflush(slowfile);
	fflush(slmkzkpfile);
	fflush(slfluxfile);
	fflush(pmfile);

}
//////////////////////////////////////////////////////////////////////
// Initialize diagnostics
//////////////////////////////////////////////////////////////////////
void init_diag(){

	  char str[255];

      /////////////////////////////////////////////////////////////////
      //Alfven diagnostics
      /////////////////////////////////////////////////////////////////
	  strcpy(str, runname);
	  strcat(str, ".energy");
      energyfile = fopen( str, "w+");
	  strcpy(str, runname);
	  strcat(str, ".speckzkp");
      alf_kzkpfile = fopen( str, "w+");

	  strcpy(str, runname);
	  strcat(str, ".energy2");
      energyfile2 = fopen( str, "w+");
	  strcpy(str, runname);
	  strcat(str, ".alfkparkperp");
      alf_kparkperpfile = fopen( str, "w+");
      /////////////////////////////////////////////////////////////////
      //Slow mode diagnostics
      /////////////////////////////////////////////////////////////////

      // Total slow mode energy
	  strcpy(str, runname);
	  strcat(str, ".slenergy");
      slowfile = fopen(str, "w+");
      // m-kz-kp spectrum
	  strcpy(str, runname);
	  strcat(str, ".mkzkp");
      slmkzkpfile = fopen(str, "w+");

      // Flux diagnostics
	  strcpy(str, runname);
	  strcat(str, ".flux");
      slfluxfile = fopen(str, "w+");

      // C+,C- vs m,kperp,kz
	  strcpy(str, runname);
	  strcat(str, ".pm"); 
	  pmfile = fopen(str, "w+");

      // Total slow mode energy
	  strcpy(str, runname);
	  strcat(str, ".slenergy2");
      slowfile2 = fopen(str, "w+");
      // m-kz-kp spectrum
	  strcpy(str, runname);
	  strcat(str, ".mkparkperp");
      slmkparkperpfile = fopen(str, "w+");

      // Flux diagnostics
	  strcpy(str, runname);
	  strcat(str, ".flux2");
      slfluxfile2 = fopen(str, "w+");

      // C+,C- vs m,kperp,kz
	  strcpy(str, runname);
	  strcat(str, ".pm2"); 
	  pmfile2 = fopen(str, "w+");


}
//////////////////////////////////////////////////////////////////////
void close_diag(){

	fclose(energyfile);
	fclose(alf_kzkpfile);
	fclose(energyfile2);
	fclose(alf_kparkperpfile);

	fclose(slowfile);
    fclose(slmkzkpfile);
    fclose(slfluxfile);
    fclose(pmfile);
	fclose(slowfile2);
    fclose(slmkparkperpfile);
    fclose(slfluxfile2);
    fclose(pmfile2);

}

void diagnostics(cuComplex* zp, cuComplex* zm, cuComplex* Gm, cuComplex* Gm_tmp, float time){

    ////////////////////////////////////////
    // Alfven kz-kp spectra
    ////////////////////////////////////////
    // Calculate kperp**2 * phi and kperp**2 A for all alfven diagnostics
    addsubt<<<dimGrid,dimBlock>>> (temp1, zp, zm, 1);
    //kPhi = zp+zm
        
    scale<<<dimGrid,dimBlock>>> (temp1, .5);
    //temp1 = .5*(zp+zm) = phi
    
    addsubt<<<dimGrid,dimBlock>>> (temp2, zp, zm, -1);
    //temp2 = zp-zm
    
    scale<<<dimGrid,dimBlock>>> (temp2, .5);
    //temp2 = .5*(zp-zm) = A

    squareComplex<<<dimGrid,dimBlock>>> (temp1);
    //temp1 = phi**2
    
    squareComplex<<<dimGrid,dimBlock>>> (temp2);
    //temp2 = A**2

    fixFFT<<<dimGrid,dimBlock>>>(temp1);
    fixFFT<<<dimGrid,dimBlock>>>(temp2);
    
    multKPerp<<<dimGrid,dimBlock>>> (temp1, temp1, -1);
    //temp1 = (kperp**2) * (phi**2)

    multKPerp<<<dimGrid,dimBlock>>> (temp2, temp2, -1);
    //temp2 = (kperp**2) * (A**2)

    alf_diagnostics(temp1, temp2, time, alf_kzkpfile, energyfile);

    ////////////////////////////////////////
    // kpar-kperp spectra
    ////////////////////////////////////////

    // Calculate ux, uy, dbx, dby
    calculate_uxy_dbxy(zp, zm, fdxR, fdyR, gdxR, gdyR);

    // Convert Gm to real space
    for (int m=0; m<Nm; m++) if(cufftExecC2R(plan_C2R, Gm+m*Nx*(Ny/2+1)*Nz, GmR+m*Nx*Ny*Nz) != CUFFT_SUCCESS) printf("Gm conversion to real space failed for m=%d\n", m);

    // Follow fieldlines to obtain the fields along the field line
    fldfol(fdxR, fdyR, gdxR, gdyR, GmR, uxfld, uyfld, dbxfld, dbyfld, Gmfld);

    if(cufftExecR2C(plan_R2C, uxfld, temp1) != CUFFT_SUCCESS) printf ("ux FFT failed \n");
    if(cufftExecR2C(plan_R2C, uyfld, temp2) != CUFFT_SUCCESS) printf ("uy FFT failed \n");

    if(cufftExecR2C(plan_R2C, dbxfld, temp3) != CUFFT_SUCCESS) printf ("ux FFT failed \n");
    if(cufftExecR2C(plan_R2C, dbyfld, temp4) != CUFFT_SUCCESS) printf ("uy FFT failed \n");
    scale <<<dimGrid, dimBlock>>>(temp1, 1.0f/((float) Nx*Ny*Nz));
    scale <<<dimGrid, dimBlock>>>(temp2, 1.0f/((float) Nx*Ny*Nz));
    scale <<<dimGrid, dimBlock>>>(temp3, 1.0f/((float) Nx*Ny*Nz));
    scale <<<dimGrid, dimBlock>>>(temp4, 1.0f/((float) Nx*Ny*Nz));

    mask<<<dimGrid,dimBlock>>>(temp1);
    mask<<<dimGrid,dimBlock>>>(temp2);
    mask<<<dimGrid,dimBlock>>>(temp3);
    mask<<<dimGrid,dimBlock>>>(temp4);

    squareComplex <<<dimGrid, dimBlock>>> (temp1);
    squareComplex <<<dimGrid, dimBlock>>> (temp2);
    squareComplex <<<dimGrid, dimBlock>>> (temp3);
    squareComplex <<<dimGrid, dimBlock>>> (temp4);

    fixFFT <<<dimGrid, dimBlock>>>(temp1);
    fixFFT <<<dimGrid, dimBlock>>>(temp2);
    fixFFT <<<dimGrid, dimBlock>>>(temp3);
    fixFFT <<<dimGrid, dimBlock>>>(temp4);

    addsubt<<<dimGrid, dimBlock>>>(temp1, temp1, temp2, 1);
    addsubt<<<dimGrid, dimBlock>>>(temp3, temp3, temp4, 1);

    //Alfven spectra
    alf_diagnostics(temp1, temp3, time, alf_kparkperpfile, energyfile2);

    //Slow mode spectra
    for (int m=0; m<Nm; m++)
     {

        if (cufftExecR2C(plan_R2C, Gmfld + m*Nx*Ny*Nz, Gm_tmp + m*Nx*(Ny/2+1)*Nz) != CUFFT_SUCCESS) printf ("Gm FFT failed m=%d\n", m);

        scale <<<dimGrid, dimBlock>>> (Gm_tmp + m*Nx*(Ny/2+1)*Nz, 1.0f/((float) Nx*Ny*Nz));
        mask <<<dimGrid, dimBlock>>> (Gm_tmp + m*Nx*(Ny/2+1)*Nz);

     }
    slow_diagnostics(Gm_tmp, time, slowfile2, slmkparkperpfile, slfluxfile2, pmfile2);

    slow_diagnostics(Gm, time, slowfile, slmkzkpfile, slfluxfile, pmfile);
}
