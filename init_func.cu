//////////////////////////////////////////////////////////////////////
	// Alfven initialization
//////////////////////////////////////////////////////////////////////
void finit(cuComplex *f, cuComplex *g)
{
	int iky, ikx, ikz, index;
	float ran;

	// Zero out f and g
	for(ikz = 0; ikz<Nz; ikz ++){
		for(ikx = 0; ikx < Nx; ikx++){
			for(iky = 0; iky < Ny/2+1; iky++){
			index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;

			f[index].x = 0.0;
			f[index].y = 0.0;

			g[index].x = 0.0;
			g[index].y = 0.0; 

			}
		}
	}

////////////////////////////////////////////////////////////////////////
//	// Random initialization
////////////////////////////////////////////////////////////////////////
if(noise){
	float k2;
    float ampl = 1.e+0/(Nx*Ny*Nz);
	for(iky=1;iky<=(Ny-1)/3+1; iky++){ 
		for(ikz=1;ikz<=(Nz-1)/3+1; ikz++){
			for(ikx=1;ikx<=(Nx-1)/3+1; ikx++){

			index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
			//k2 = pow((float) iky/Y0,2) + pow((float) ikx/X0,2) + pow((float) ikz/Z0,2);
			k2 = pow((float) iky/Y0,2)  + pow((float) ikz/Z0,2);


			// AVK: working Density of states effect?
			ran = ((float) rand()) / ((float) RAND_MAX + 1);
			f[index].x = (sqrt(ampl/k2)/k2) * cos(ran*2.0*M_PI);
			f[index].y = (sqrt(ampl/k2)/k2) * sin(ran*2.0*M_PI);
			ran = ((float) rand()) / ((float) RAND_MAX + 1);
			g[index].x = (sqrt(ampl/k2)/k2) * cos(ran*2.0*M_PI);
			g[index].y = (sqrt(ampl/k2)/k2) * sin(ran*2.0*M_PI);

			/*index = + (Ny/2+1)*(Nx-ikx) + (Ny/2+1)*Nx*(Nz-ikz);

			// AVK: working Density of states effect?
			ran = ((float) rand()) / ((float) RAND_MAX + 1);
			f[index].x = sqrt(ampl) * cos(ran*2.0*M_PI);
			f[index].y = sqrt(ampl) * sin(ran*2.0*M_PI);
			ran = ((float) rand()) / ((float) RAND_MAX + 1);
			g[index].x = sqrt(ampl) * cos(ran*2.0*M_PI);
			g[index].y = sqrt(ampl) * sin(ran*2.0*M_PI);*/


			}
		}
	}
}

////////////////////////////////////////////////////////////////////////
//	// Decaying Alfven cascade run
////////////////////////////////////////////////////////////////////////
if(decaying){
	//float xi0 = 1.e+2/((Nx/3)*(Ny/3)*(Nz/3));
	float xi0 = 1.e-2;

	for(ikz=1; ikz<Nz/4; ikz++){
		for(ikx=1; ikx<(Nx-1)/3; ikx++){
			for(iky=1; iky<(Ny-1)/3 ; iky++){

				index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;

				ran = ((float) rand()) / ((float) RAND_MAX + 1);

				f[index].x = cos(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
				f[index].y = sin(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));

				//ran = ((float) rand()) / ((float) RAND_MAX + 1);
				g[index].x = cos(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));       
				g[index].y = sin(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
			}
		}
	}

	for(ikz = 3*Nz/4; ikz<Nz; ikz++){
		for(ikx=2*Nx/3 + 1; ikx<Nx; ikx++){
			for(iky=1; iky<Ny/3; iky++){

				index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;

				ran = ((float) rand()) / ((float) RAND_MAX + 1);

				f[index].x = cos(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
				f[index].y = sin(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));

				//ran = ((float) rand()) / ((float) RAND_MAX + 1);
				g[index].x = cos(ran * 2.0* M_PI)* sqrt(xi0 * pow(iky,-10.0f/1.0f));       
				g[index].y = sin(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
			}
		}
	} 
}

////////////////////////////////////////////////////////////////////////
//	// Orszag-Tang initial conditions
////////////////////////////////////////////////////////////////////////
if(orszag_tang){
	//phi = -2(cosx + cosy)
	//A = 2cosy + cos2x
	//f = z+ = phi + A
	//g = z- = phi - A

	ikx = 1; iky = 0; ikz = 0;
	index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

	f[index].x = -1.0;
	g[index].x = -1.0;

	ikx = Nx-1; iky = 0; ikz = 0;
	index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

	f[index].x = -1.0;
	g[index].x = -1.0;

	ikx = 2; iky = 0; ikz = 0;
	index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

	f[index].x = 0.50;
	g[index].x = -0.50;

	ikx = Nx-2; iky = 0; ikz = 0;
	index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

	f[index].x = 0.50;
	g[index].x = -0.50;
	

	ikx = 0; iky = 1; ikz = 0;
	index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

	g[index].x = -2.0; 

 }  



}

//////////////////////////////////////////////////////////////////////
	// Slow mode initialization
//////////////////////////////////////////////////////////////////////


