void courant(float* dt,  cuComplex* zp, cuComplex* zm)
{
  // temp1 holds velocity
  zero<<<dimGrid, dimBlock>>>(padded, Nx, Ny, Nz);

  cuComplex *max;
  max = (cuComplex*) malloc(sizeof(cuComplex));
    
  float vxmax, vymax, omega_zmax;
    
  ///////////////////////////////////////////////////////
    
  //calculate max(ky*zp)
    
  multKy<<<dimGrid,dimBlock>>>(temp1, zp);
  if(debug) printf("multKy zp: %s\n",cudaGetErrorString(cudaGetLastError()));

  //temp1 = ky*zp
  maxReduc(max,temp1,padded); 

  vxmax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);		        

  /////////////////////////////////////////////////////////
    
  //calculate max(ky*zm)
    
  multKy<<<dimGrid,dimBlock>>>(temp1,zm);
  if(debug) printf("multKy zm: %s\n",cudaGetErrorString(cudaGetLastError()));

  //temp1 = ky*zm
    
  maxReduc(max,temp1,padded);
    
  if(sqrt(max[0].x*max[0].x+max[0].y*max[0].y) > vxmax) {
    vxmax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
  }  
   
  //////////////////////////////////////////////////////////
    
  //calculate max(kx*zp)
    
  multKx<<<dimGrid,dimBlock>>>(temp1, zp);
  if(debug) printf("multKx zp: %s\n",cudaGetErrorString(cudaGetLastError()));
  
  //temp1 = kx*zp
    
  maxReduc(max,temp1,padded);
    
  vymax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
    		     
  ///////////////////////////////////////////////////////
    
  //calculate max(kx*zm)
    
  multKx<<<dimGrid,dimBlock>>>(temp1,zm);
  if(debug) printf("multKx zm: %s\n",cudaGetErrorString(cudaGetLastError()));
  
  //temp1 = kx*zm
    
  maxReduc(max,temp1,padded);
    
  if( sqrt(max[0].x*max[0].x+max[0].y*max[0].y) > vymax) {
    vymax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
  }  
    
  /////////////////////////////////////////////////////////
  // omega_zmax
  omega_zmax = ((float) sqrt(Nm)) * ((float) (Nz-1)/3) * sqrt(beta)/(Z0*cfl);

  /////////////////////////////////////////////////////////
    
  //find dt

  if(vxmax>=vymax) *dt = (float) cfl *M_PI*X0/(vxmax*Nx);
  else *dt = (float) cfl*M_PI*Y0/(vymax*Ny);
  
  if(1.0/(*dt) <  omega_zmax) *dt = 1.0f/(omega_zmax);
  if(*dt >  maxdt) *dt = maxdt;

  free(max);
  if (debug) {printf("Exiting courant.cu dt = %f\n", *dt);}

  // temp1 is free now
    
}

