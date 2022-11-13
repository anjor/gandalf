void sumReduc_nopad(cuComplex* result, cuComplex* f) 
{
    //block size is 8*8*8=512, so that all of each block fits in shared memory
    /*dim3 dimBlock(8,8,8);
    //gridx is the number of blocks configured
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz <= 512) {
      dimBlock.x = Nx;
      dimBlock.y = Ny;
      dimBlock.z = Nz;
      gridx = 1;
    }   */

	dim3 dimBlockReduc(32,1,16);
	//gridx is the number of blocks configured
	int gridx = (Nx*Nz)/512;
    
			if (Nx*Nz <= 512) {
			  dimBlockReduc.x = Nx;
			  dimBlockReduc.y = 1;
			  dimBlockReduc.z = Nz;
		    gridx = 1;
			}  
    
    dim3 dimGridReduc(gridx,1,1);
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(f, f);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(f, f);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cuComplex)*8*8*8>>>(f,f);  
    
    cudaMemcpy(result, f, sizeof(cuComplex), cudaMemcpyDeviceToHost);

}    
