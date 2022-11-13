//////////////////////
// Damp in z 
//////////////////////
__global__ void dampz(cuComplex* znew, float nu_kz, int alpha_z, float dt)
{
  if(Nz>1){
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
	  int idmax = (Nz-1)/3;

		znew[index] = znew[index] *exp(- nu_kz*dt*pow(abs(kz(idz)/kz(idmax)),2*alpha_z));
		}
		}
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;

		int IDZ = idz + zThreads*i;
		int idmax = (Nz-1)/3;

		znew[index] = znew[index] *exp(- nu_kz*dt*pow(abs(kz(IDZ)/kz(idmax)),2*alpha_z));
	      }
	    }
	  }
	}

}
	
//////////////////////
// Kperp Damp 
//////////////////////
__global__ void damp_hyper(cuComplex* znew, float nu_hyper, int alpha_hyper, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
	  int idxmax = (Nx-1)/3;
	  int idymax = (Ny-1)/3;

		znew[index] = znew[index] *exp(-nu_hyper*dt*pow(abs(kPerp2(idx,idy)/kPerp2(idxmax,idymax)),alpha_hyper));
		}
	}
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	    int idxmax = (Nx-1)/3;
	    int idymax = (Ny-1)/3;

		znew[index] = znew[index] *exp(-nu_hyper*dt*pow(abs(kPerp2(idx,idy)/kPerp2(idxmax,idymax)),alpha_hyper) );
     }
    }
   }
}


