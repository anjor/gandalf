// kz-kp shell
__global__ void kz_kpshellsum(cuComplex* k2field2, int ikp, float* energy_kp)
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  // X0 and Y0 needs to be 1 for this to make sense.
  int ikz_kp;

  if(Nz<=zThreads) {
  	if(idx<Nx && idy<Ny/2+1 && idz<Nz){
		if(sqrt(abs(kPerp2(idx,idy))) >= ikp-0.5 && sqrt(abs(kPerp2(idx,idy))) < ikp + 0.5)  {
			unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
			ikz_kp = idz + Nz*ikp;
			atomicAdd(energy_kp + ikz_kp, k2field2[index].x);
			}
		}
	}
  else {
  	for(int i=0; i<Nz/zThreads; i++) {
		if(idx<Nx && idy<Ny/2+1 && idz<zThreads) {
			if(sqrt(abs(kPerp2(idx,idy))) >= ikp-0.5 && sqrt(abs(kPerp2(idx,idy))) < ikp+0.5)  {
				unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
				int IDZ = idz + zThreads*i;
				ikz_kp = IDZ + Nz*ikp;
				atomicAdd(energy_kp + ikz_kp, k2field2[index].x);
				}
			}
		}
	}
}
__global__ void kz_kpshellsum(float* k2field2, int ikp, float* energy_kp)
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  // X0 and Y0 needs to be 1 for this to make sense.
  int ikz_kp;

  if(Nz<=zThreads) {
  	if(idx<Nx/2+1 && idz<Nz){
		if(sqrt(abs(kPerp2(idx,idy))) >= ikp-0.5 && sqrt(abs(kPerp2(idx,idy))) < ikp+0.5)  {
			unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
			ikz_kp = idz + Nz*ikp;
			atomicAdd(energy_kp + ikz_kp, k2field2[index]);
			}
		}
	}
  else {
  	for(int i=0; i<Nz/zThreads; i++) {
		if(idx<Nx/2+1 && idz<zThreads) {
			if(sqrt(abs(kPerp2(idx,idy))) >= ikp-0.5 && sqrt(abs(kPerp2(idx,idy))) < ikp+0.5)  {
				unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
				int IDZ = idz + zThreads*i;
				ikz_kp = IDZ + Nz*ikp;
				atomicAdd(energy_kp + ikz_kp, k2field2[index]);
				}
			}
		}
	}
}



// Converts Gm to Gm+ or Gm- as defined by Alex
__global__ void Gm_to_Gtilde_p(cuComplex* Gm, cuComplex* gmtilde, int m){

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  cuComplex cj;
  cj.x = 0.0f; cj.y = 1.0f;
  if(Nz<=zThreads) {
  	if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
		unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
		unsigned int index_next = index + Nx*(Ny/2+1)*Nz;

		gmtilde[index] = pow(cj*sgn(kz(idz)),m)* \
					(Gm[index] + cj*sgn(kz(idz))*Gm[index_next])/2.0;

		}
	}
  else {
  	for(int i=0; i<Nz/zThreads; i++) {
		if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
			int IDZ = idz + zThreads*i;
			unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*IDZ;
			unsigned int index_next = index + Nx*(Ny/2+1)*Nz;

				gmtilde[index] = pow(cj*sgn(kz(IDZ)),m)* \
							(Gm[index] + cj*sgn(kz(IDZ))*Gm[index_next])/2.0;
		}
	}
	}
}

// Converts Gm to Gm+ or Gm- as defined by Alex
__global__ void Gm_to_Gtilde_m(cuComplex* Gm, cuComplex* gmtilde, int m){

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  cuComplex cj;
  cj.x = 0.0f; cj.y = 1.0f;
  if(Nz<=zThreads) {
  	if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
		unsigned int index = idy + (Ny/2+1)*idx+Nx*(Ny/2+1)*idz;
		unsigned int index_next = index + Nx*(Ny/2+1)*Nz;

		gmtilde[index] = pow(-1.,m)* pow(cj*sgn(kz(idz)),m)* \
					(Gm[index] - cj*sgn(kz(idz))*Gm[index_next])/2.0;

		}
	}
  else {
  	for(int i=0; i<Nz/zThreads; i++) {
		if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
			int IDZ = idz + zThreads*i;
			unsigned int index = idy + (Ny/2+1)*idx+Nx*(Ny/2+1)*IDZ;
			unsigned int index_next = index + Nx*(Ny/2+1)*Nz;

				gmtilde[index] = pow(-1.,m)* pow(cj*sgn(kz(IDZ)),m)* \
							(Gm[index] - cj*sgn(kz(IDZ))*Gm[index_next])/2.0;
		}
	}
	}
}
// Calculates exact flux in m
__global__ void m_flux(cuComplex* Gm, cuComplex* flux, int m){

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  cuComplex cj; cj.x = 0.0f; cj.y = 1.0f;

  if(Nz<=zThreads) {
  	if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
		unsigned int index = idy + (Ny/2+1)*idx+Nx*(Ny/2+1)*idz;
		unsigned int index_next = index + Nx*(Ny/2+1)*Nz;

		flux[index] = cj*kz(idz) *sqrt(2.*(m+1)) *(Gm[index_next]*conjg(Gm[index]));

		}
	}
  else {
  	for(int i=0; i<Nz/zThreads; i++) {
		if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
			int IDZ = idz + zThreads*i;
			unsigned int index = idy + (Ny/2+1)*idx+Nx*(Ny/2+1)*IDZ;
			unsigned int index_next = index + Nx*(Ny/2+1)*Nz;

			flux[index] = cj*kz(idz) *sqrt(2.*(m+1)) *(Gm[index_next]*conjg(Gm[index]));
		}
	}
	}
}
