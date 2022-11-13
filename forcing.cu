// Make this more like Viriato: Input kp1 and kp2 from input file. Then calculate all the
// kperps in between, and force them.
void forcing( cuComplex *force, float dt, int *kstir_x, int *kstir_y, int *kstir_z, float
fampl)
{
	float phase,amp,kp;
	cuComplex *temp;
	unsigned int index;
	temp = (cuComplex*) malloc(sizeof(cuComplex));

	int id = rand() % nkstir;

		kp = sqrt(-kPerp2(kstir_x[id], kstir_y[id]));

		float ran_amp = ( (float) rand()) / ((float) RAND_MAX + 1.0 );
		amp = 1.0*(1.0/abs(kp)) * sqrt(abs(1.0*(fampl/dt)*log(ran_amp)));
		phase = M_PI*(2.0*( (float) rand()) / ((float) RAND_MAX + 1.0 ) -1.0);

		temp[0].x = amp*cos(phase);
		temp[0].y = amp*sin(phase);

		index = kstir_y[id] + (Ny/2+1)*kstir_x[id] +Nx*(Ny/2+1)*kstir_z[id];
		cudaMemcpy(force + index, temp, sizeof(cuComplex), cudaMemcpyHostToDevice);
		if(debug) {printf("Copying over forcing term : %s, id = %d, index = %d\n",cudaGetErrorString(cudaGetLastError()), id, index);}

		// Reality condition
		if(kstir_y[id] == 0){

		temp[0].y = -temp[0].y;
		index = kstir_y[id] + (Ny/2+1)*((Nx-kstir_x[id])%Nx) + Nx*(Ny/2+1)*((Nz-kstir_z[id])%Nz);
		cudaMemcpy(force + index, temp, sizeof(cuComplex), cudaMemcpyHostToDevice);
		if(debug) {printf("Copying over complex conjugate of forcing term : %s\n",cudaGetErrorString(cudaGetLastError()));}

		}

	free(temp);

	if(debug) {printf("Exiting forcing : %s\n",cudaGetErrorString(cudaGetLastError()));}

}
