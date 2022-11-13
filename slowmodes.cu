///////////////////////////////////////////////////////////
void s_nonlin(cuComplex* src, cuComplex* Gm, cuComplex* Gcomb, cuComplex* Phi, cuComplex* Psi, float dt)
{

	// {Phi, Gm} term
	NLPS(src, Phi, Gm);	
	scale<<<dimGrid, dimBlock>>>(src, -dt);

	// {Psi, Gcomb} term
	NLPS(temp4, Psi, Gcomb);	
	fwdeuler<<<dimGrid, dimBlock>>>(src, temp4, dt*sqrt(beta));
    if(debug) printf("fwdeuler in sl_nonlin, : %s\n",cudaGetErrorString(cudaGetLastError()));

}
///////////////////////////////////////////////////////////
void s_lin(cuComplex* result, cuComplex* Gcomb, float dt)

{
	zderiv<<<dimGrid, dimBlock>>>(temp4, Gcomb);
	fwdeuler<<<dimGrid, dimBlock>>>(result, temp4, -dt*sqrt(beta));
}
///////////////////////////////////////////////////////////

void slow_adv(cuComplex* zp, cuComplex*  zm, cuComplex* Gmnew, cuComplex* Gmold, cuComplex* Gmstar, float dt, cuComplex* Phi, cuComplex* Psi, cuComplex* Gmcomb)
{
	//// Calculate Phi=Phi
	addsubt<<<dimGrid, dimBlock>>>(Phi, zp, zm, 1);
	scale<<<dimGrid, dimBlock>>>(Phi, .5);

	////  Calculate Psi=Psi
	addsubt<<<dimGrid, dimBlock>>>(Psi, zp, zm, -1);
	scale<<<dimGrid, dimBlock>>>(Psi, .5);

    //Gmcomb holds Gmcomb = sqrt(m+1)g_{m+1} + sqrt{m} g_{m-1}
	// m = 0
	zero<<<dimGrid, dimBlock>>>(Gmcomb, Nx, Ny/2+1, Nz);
	scale<<<dimGrid, dimBlock>>>(Gmcomb, Gmstar + Nx*(Ny/2+1)*Nz, sqrt(0.5));
	s_nonlin(Gmnew, Gmstar, Gmcomb, Phi, Psi, dt);
	s_lin(Gmnew, Gmcomb, dt);
	addsubt<<<dimGrid, dimBlock>>>(Gmnew, Gmnew, Gmold, 1);

	// m = 1
	zero<<<dimGrid, dimBlock>>>(Gmcomb, Nx, Ny/2+1, Nz);
    scale<<<dimGrid, dimBlock>>>(Gmcomb, Gmstar, sqrt(0.5)*(1.0 - 1.0/Lambda));
    fwdeuler<<<dimGrid, dimBlock>>>(Gmcomb, Gmstar + 2*Nx*(Ny/2+1)*Nz , 1);
	s_nonlin(Gmnew +Nx*(Ny/2+1)*Nz, Gmstar +Nx*(Ny/2+1)*Nz, Gmcomb, Phi, Psi, dt);
	s_lin(Gmnew +Nx*(Ny/2+1)*Nz, Gmcomb, dt);
	addsubt<<<dimGrid, dimBlock>>>(Gmnew + Nx*(Ny/2+1)*Nz, Gmnew + Nx*(Ny/2+1)*Nz, Gmold + Nx*(Ny/2+1)*Nz, 1);


	// m = 2 to Nm-2
	for(int m = 2; m < Nm-1; m++){
		// Calculate Gmcomb
		zero<<<dimGrid, dimBlock>>>(Gmcomb, Nx, Ny/2+1, Nz);
		scale<<<dimGrid, dimBlock>>>(Gmcomb, Gmstar + (m-1)*Nx*(Ny/2+1)*Nz, sqrt((float) m/2.0));
		fwdeuler<<<dimGrid, dimBlock>>>(Gmcomb, Gmstar + (m+1)*Nx*(Ny/2+1)*Nz, sqrt(((float) m + 1.0)/2.0));

		// Calculate nonlinear term
		s_nonlin(Gmnew + m*Nx*(Ny/2+1)*Nz, Gmstar + m*Nx*(Ny/2+1)*Nz, Gmcomb, Phi, Psi, dt);
		// Calculate linear term
		s_lin(Gmnew + m*Nx*(Ny/2+1)*Nz, Gmcomb, dt);

		// Add everything together
		addsubt<<<dimGrid, dimBlock>>>(Gmnew + m*Nx*(Ny/2+1)*Nz, Gmnew + m*Nx*(Ny/2+1)*Nz, Gmold + m*Nx*(Ny/2+1)*Nz, 1);
	}

	// m = Nm-1
	zero<<<dimGrid, dimBlock>>>(Gmcomb, Nx, Ny/2+1, Nz);
	scale<<<dimGrid, dimBlock>>>(Gmcomb, Gmstar + (Nm-2)*Nx*(Ny/2+1)*Nz, sqrt((float) (Nm-1)/2.0));
	
	// Closure g_{M+1} = g_{M-1}>
	//fwdeuler<<<dimGrid, dimBlock>>>(Gmcomb, Gmstar + (Nm-2)*Nx*(Ny/2+1)*Nz, sqrt((float) Nm/2.0));

	s_nonlin(Gmnew + (Nm-1)*Nx*(Ny/2+1)*Nz, Gmstar + (Nm-1)*Nx*(Ny/2+1)*Nz, Gmcomb, Phi, Psi, dt);

	s_lin(Gmnew + (Nm-1)*Nx*(Ny/2+1)*Nz, Gmcomb, dt);

	addsubt<<<dimGrid, dimBlock>>>(Gmnew + (Nm-1)*Nx*(Ny/2+1)*Nz, Gmnew + (Nm-1)*Nx*(Ny/2+1)*Nz, Gmold + (Nm-1)*Nx*(Ny/2+1)*Nz, 1);




}
