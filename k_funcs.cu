// Wavenumber functions
__host__ __device__ float kx(int ikx)
{
	if(ikx<Nx/2 +1) return ikx/X0;
	else return (ikx-Nx)/X0;

}
__host__ __device__ float ky(int iky)
{
	if(iky<Ny/2+1) return (float) iky/Y0;
	else return 0;
}
__host__ __device__ float kz(int ikz)
{
	if(ikz<Nz/2 +1) return ikz/Z0;
	else return (ikz-Nz)/Z0;

}
// Real space functions
__host__ __device__ float xx(int ix)
{
	if(ix<Nx) return ((float) ix/Nx) *2.0f*M_PI*X0;
	else return -1.0f;
}

__host__ __device__ float yy(int iy)
{
	if(iy<Ny) return ((float) iy/Ny) *2.0f*M_PI*Y0;
	else return -1.0f;
}

__host__ __device__ float zz(int iz)
{
	if(iz<Nz) return ((float) iz/Nz) *2.0f*M_PI*Z0;
	else return -1.0f;
}

// kPerp2 functions
__host__ __device__ float kPerp2(int ikx, int iky)
{
		float kp2 = -kx(ikx)*kx(ikx) -ky(iky)*ky(iky);
		return kp2;
}

__host__ __device__ float kPerp2Inv(int ikx, int iky)
{
		float kp2 = -kx(ikx)*kx(ikx) -ky(iky)*ky(iky);
		if(ikx !=0 || iky !=0) return 1.0f/kp2;
		else return 0.0f;
}
