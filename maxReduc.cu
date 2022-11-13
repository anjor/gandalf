void maxReduc(cuComplex* max, cuComplex* f, cuComplex* padded) 
{
    
	zero<<<dimGrid, dimBlock>>>(padded,Nx,Ny,Nz);
	if(debug) printf("zero padded in maxreduc: %s\n",cudaGetErrorString(cudaGetLastError()));
	if(debug) printf("dimGrid = %d, %d, %d \t dimBlock = %d, %d, %d \n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
	if(debug) printf("Nx = %d, Ny = %d, Nz = %d \n", Nx, Ny, Nz);
    cudaMemcpy(padded,f,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);
    
    dim3 dimBlockReduc(8,8,8);
    //dimBlock.x=dimBlock.y=dimBlock.z=8;
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz <= 512) {
      dimBlockReduc.x = Nx;
      dimBlockReduc.y = Ny;
      dimBlockReduc.z = Nz;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    //dimGrid.x=gridx;
    //dimGrid.y=dimGrid.z=1;
    
    maximum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      // dimGrid.x = 8
      maximum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded, padded);
      // result = 8 elements
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    maximum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded,padded);  
    
    cudaMemcpy(max, padded, sizeof(cuComplex), cudaMemcpyDeviceToHost);

}    

void sumReduc(cuComplex* result, cuComplex* f, cuComplex* padded) 
{
	zero<<<dimGrid, dimBlock>>>(padded, Nx, Ny, Nz);
	if(debug) printf("dimGrid = %d, %d, %d \t dimBlock = %d, %d, %d \n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
	if(debug) printf("zero padded in sumreduc: %s\n",cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(padded,f,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);
    
    dim3 dimBlockReduc(8,8,8);
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz <= 512) {
      dimBlockReduc.x = Nx;
      dimBlockReduc.y = Ny;
      dimBlockReduc.z = Nz;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded, padded);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded,padded);  
    
    cudaMemcpy(result, padded, sizeof(cuComplex), cudaMemcpyDeviceToHost);

}    

void sumReduc(float* result, float* f, float* padded) 
{
	zero<<<dimGrid, dimBlock>>>(padded, Nx, Ny, Nz);
    cudaMemcpy(padded,f,sizeof(float)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);
    
    dim3 dimBlockReduc(8,8,8);
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz <= 512) {
      dimBlockReduc.x = Nx;
      dimBlockReduc.y = Ny;
      dimBlockReduc.z = Nz;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(padded, padded);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(padded,padded);  
    
    cudaMemcpy(result, padded, sizeof(float), cudaMemcpyDeviceToHost);

}    


////////////////////////////////////////
void sumReduc_kz(cuComplex* result, cuComplex* f, cuComplex* padded) 
{
	zero<<<dimGrid, dimBlock>>>(padded,Nx,Ny,1);
    cudaMemcpy(padded,f,sizeof(cuComplex)*Nx*(Ny/2+1),cudaMemcpyDeviceToDevice);
    
    dim3 dimBlockReduc(8,8,8);
    int gridx = (Nx*Ny)/512;
    
    if (Nx*Ny <= 512) {
      dimBlockReduc.x = Nx;
      dimBlockReduc.y = Ny;
      dimBlockReduc.z = 1;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded, padded);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded,padded);  
    
    cudaMemcpy(result, padded, sizeof(cuComplex), cudaMemcpyDeviceToHost);

}    

////////////////////////////////////////
void sumReduc_gen(cuComplex* result, cuComplex* f, cuComplex* padded, int nx, int ny, int
nz) 
{
	zero<<<dimGrid, dimBlock>>>(padded, nx, ny, nz);
	if(debug) printf("nx = %d, ny = %d, nz = %d \n", nx, ny, nz);
	if(debug) printf("dimGrid = %d, %d, %d \t dimBlock = %d, %d, %d \n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
	if(debug) printf("zero padded in sumreduc: %s\n",cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(padded,f,sizeof(cuComplex)*nx*(ny/2+1)*nz,cudaMemcpyDeviceToDevice);
    
    dim3 dimBlockReduc(8,8,8);
    int gridx = (nx*ny*nz)/512;
    
    if (nx*ny*nz <= 512) {
      dimBlockReduc.x = nx;
      dimBlockReduc.y = ny;
      dimBlockReduc.z = nz;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded, padded);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(padded,padded);  
    
    cudaMemcpy(result, padded, sizeof(cuComplex), cudaMemcpyDeviceToHost);

}    

void sumReduc_gen(float* result, float* f, float* padded, int nx, int ny, int nz) 
{
	zero<<<dimGrid, dimBlock>>>(padded, nx, ny, nz);
    cudaMemcpy(padded,f,sizeof(float)*nx*(ny/2+1)*nz,cudaMemcpyDeviceToDevice);
    
    dim3 dimBlockReduc(8,8,8);
    int gridx = (nx*ny*nz)/512;
    
    if (nx*ny*nz <= 512) {
      dimBlockReduc.x = nx;
      dimBlockReduc.y = ny;
      dimBlockReduc.z = nz;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(padded, padded);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(padded,padded);  
    
    cudaMemcpy(result, padded, sizeof(float), cudaMemcpyDeviceToHost);

}    

void sumReduc_gen(float* result, float* f, int nx, int ny, int nz) 
{
	//zero<<<dimGrid, dimBlock>>>(padded, nx, ny, nz);
    //cudaMemcpy(padded,f,sizeof(float)*nx*(ny/2+1)*nz,cudaMemcpyDeviceToDevice);
    
    dim3 dimBlockReduc(8,8,8);
    int gridx = (nx*ny*nz)/512;
    
    if (nx*ny*nz <= 512) {
      dimBlockReduc.x = nx;
      dimBlockReduc.y = ny;
      dimBlockReduc.z = nz;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(f, f);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(f, f);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*8*8*8>>>(f,f);  
    
    cudaMemcpy(result, f, sizeof(float), cudaMemcpyDeviceToHost);

}    
