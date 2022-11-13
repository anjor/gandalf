// Nonlinear timestepping routine using Noah's trick
void alf_adv(cuComplex *zpNew, cuComplex *zpOld, cuComplex *zpstar, cuComplex *zmNew, cuComplex *zmOld, cuComplex *zmstar, float dt)
{
	nonlin(zpstar, zmstar, temp1, temp2, temp3);
	// temp1 = {zp, -kperp**2 *zm} + {zm, -kperp**2 zp}, temp2 = -kperp**2{zp,zm}

	//ZP
		addsubt<<<dimGrid, dimBlock>>>(temp3, temp1, temp2, -1);
		// temp3 = temp1 - temp2 for zp
		// Coeff of 0.5 in front of the nonlinear term is included in multkperpinv
		multKPerpInv<<<dimGrid, dimBlock>>>(temp3, temp3);
		// multiply by kperp2**(-1)
		linstep<<<dimGrid,dimBlock>>>(temp3, temp3, dt);
		// multiply nonlinear term by integrating factor
		linstep<<<dimGrid,dimBlock>>>(zpNew, zpOld, dt);
		// zpNew = zpOld*exp(i*kz*dt)
		fwdeuler<<<dimGrid,dimBlock>>>(zpNew, temp3, dt);
		// Add in the nonlinear term

	//ZM
		addsubt<<<dimGrid, dimBlock>>>(temp3, temp1, temp2, 1);
		// temp3 = bracket1 - bracket2 for zp
		// Coeff of .5 in front of the nonlinear term is included in multkperpinv
		multKPerpInv<<<dimGrid, dimBlock>>>(temp3, temp3);
		// multiply by kperp2**(-1)
		linstep<<<dimGrid, dimBlock>>>(temp3, temp3, -dt);
		// multiply nonlinear term by integrating factor
		linstep<<<dimGrid,dimBlock>>>(zmNew, zmOld, -dt);
		// zmNew = zmOld*exp(-i*kz*dt)
		fwdeuler<<<dimGrid,dimBlock>>>(zmNew, temp3, dt);
		// Add in the nonlinear term
}

// Timestepping routine for alfven,slow,nonlinear,nondebug
void advance(cuComplex *zpNew, cuComplex *zpOld, cuComplex *zmNew, cuComplex *zmOld,
cuComplex *Gmnew, cuComplex *Gmold, float dt, int istep)
{
  ////////////
  // Forcing//
  ////////////
  // Currently this calculates the forcing term on the CPU, and transfers it over to the
  // GPU. There are only a few modes that are driven, so parallelization doesn't help
  // much? Are there any benefits of moving the whole calculation over to the GPU?

  if(driven && istep%nforce==0){

    // Use temp1 to hold forcing
	zero<<<dimGrid,dimBlock>>>(temp1, Nx, Ny/2+1, Nz);

    // Alfven wave forcing
	forcing(temp1, dt, kstir_x, kstir_y, kstir_z, fampl);

	fwdeuler<<<dimGrid, dimBlock>>>(zpOld, temp1, dt);
	fwdeuler<<<dimGrid, dimBlock>>>(zmOld, temp1, dt);
	

    // Slow mode forcing
	zero<<<dimGrid,dimBlock>>>(temp1, Nx, Ny/2+1, Nz);
	forcing(temp1, dt, gm_kstir_x, gm_kstir_y, gm_kstir_z, gm_fampl);

	fwdeuler<<<dimGrid, dimBlock>>>(Gmold + force_m*Nx*(Ny/2+1)*Nz, temp1, dt);

    //temp1 is free now


  }
  
  /////////////////
  // Timestepping//
  /////////////////
  // Half slow step
  /////////////////
  // temp1 is Phi, temp2 is Psi and temp3 is Gmcomb
  slow_adv(zpOld, zmOld, Gmnew, Gmold, Gmold, dt/2.0, temp1, temp2, temp3);

  /////////////////
  // Half Alfven step
  /////////////////
  alf_adv(zpNew, zpOld, zpOld, zmNew, zmOld, zmOld, dt/2.0);

  /////////////////
  // Full Slow step
  /////////////////
  slow_adv(zpNew, zmNew, Gmnew, Gmold, Gmnew, dt, temp1, temp2, temp3);

  /////////////////
  // Full Alfven step
  /////////////////
  alf_adv(zpNew, zpOld, zpNew, zmNew, zmOld, zmNew, dt);
  
  damp_hyper<<<dimGrid,dimBlock>>>(zpNew, nu_hyper, alpha_hyper, dt);
  damp_hyper<<<dimGrid,dimBlock>>>(zmNew, nu_hyper, alpha_hyper, dt);
  
  dampz<<<dimGrid,dimBlock>>>(zpNew, nu_kz, alpha_z, dt);
  dampz<<<dimGrid,dimBlock>>>(zmNew, nu_kz, alpha_z, dt);

  //////////////////////
  // Slow modes damping
  //////////////////////
  for(int m=0; m<3; m++){
	// Viscous damping of slow modes
		damp_hyper<<<dimGrid, dimBlock>>>(Gmnew + m*Nx*(Ny/2+1)*Nz, nu_kp_g, alpha_kp_g, dt);
		dampz<<<dimGrid, dimBlock>>>(Gmnew + m*Nx*(Ny/2+1)*Nz, nu_kz_g, alpha_kz_g, dt);
  }

  for(int m=0; m<Nm; m++){
		damp_hyper<<<dimGrid, dimBlock>>>(Gmnew + m*Nx*(Ny/2+1)*Nz, nu_kp_g, alpha_kp_g, dt);
		dampz<<<dimGrid, dimBlock>>>(Gmnew + m*Nx*(Ny/2+1)*Nz, nu_kz_g, alpha_kz_g, dt);

	// Collisional damping of slow modes:
	  float scaler = exp(-(nu_coll * dt * pow(((float)m/(Nm)),alpha_m)));
	  scale<<<dimGrid, dimBlock>>>(Gmnew + m*Nx*(Ny/2+1)*Nz, scaler);
  }



  //now we copy the results, the zNew's, to the zOld's 
  cudaMemcpy(zpOld, zpNew, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
  cudaMemcpy(zmOld, zmNew, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);			   
  cudaMemcpy(Gmold, Gmnew, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz*Nm, cudaMemcpyDeviceToDevice);
   
}

