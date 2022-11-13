import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

vmini = 1.e-6
vmaxi = 1.e-0

sum = lambda x,y: x+y
logand = np.logical_and

# Number of kx, ky in a kp shell
def kperp(kx, ky):
    return np.sqrt(kx**2 + ky**2)

def Nkp(Nx, Ny):
    ikpmax = np.floor(kperp((Nx-1)/3, (Ny-1)/3))
    N = np.zeros(ikpmax)
    for ikp in np.arange(ikpmax):
        for i in np.arange((Nx-1)/3):
            for j in np.arange((Ny-1)/3):
                if (kperp(i,j)>=ikp-0.5) and (kperp(i,j)<ikp+0.5):
                    N[ikp] = N[ikp] + 1

    return 4*N

def Nkp_ikpmax(ikpmax):
    N = np.zeros(ikpmax)
    Nx = 3*np.ceil(ikpmax/np.sqrt(2.)) + 1
    Ny = Nx
    return Nkp(Nx, Ny)

def fixed_kp(F):
    ikpmax = len(np.unique(F[:,1]))
    N = Nkp_ikpmax(ikpmax)
    F_fixed = np.copy(F)
    for i in np.arange(ikpmax):
        F_fixed[F_fixed[:,1]==i,2] = F[F[:,1]==i,2] * (2.*np.pi*i/N[i])
        F_fixed[F_fixed[:,1]==i,3] = F[F[:,1]==i,3] * (2.*np.pi*i/N[i])

    return F_fixed
        
# Data input functions
def load_all(run):
    file = run + ".alfkparkperp.ave"
    F = np.loadtxt(file)
    G = fixed_kp(F)
    return G

# Integrated one-d spectra functions
def intkz_vskp(F, col=2, run=''):
    print "kperp spectrum integrated over kz"
    kps = np.unique(F[:,1])
    dat = np.array(map(lambda kp: reduce(sum, F[F[:,1]==kp,col]), kps))

    plt.plot(kps,dat,label = run+' ')
    plt.xlabel(r'$k_\perp$',fontsize=20)

def intkp_vskz(F, col=2, run=''):
    print "kz spectrum integrated over kp"
    kzs = np.unique(F[:,0])
    dat = np.array(map(lambda kz: reduce(sum, F[F[:,0]==kz,col]), kzs))

    plt.plot(kzs,dat,label = run+' ')
    plt.xlabel(r'$k_\parallel$',fontsize=20)

# 1-d spectra at fixed points
def vskp(F, kz, col=2, run=''):
    print "Plotting F vs kp at fixed kz=", kz, " for run=", run
    dat = F[F[0]==kz,col]
    kps = np.unique(F[:,1])

    labelstring = run + " m = " + str(m) + r", $k_\parallel$ = " + str(kz)
    plt.plot(kps, dat,label=labelstring)

    plt.xlabel(r'$k_\perp$', fontsize=20) 
    
def vskz(F, kp,col=2, run=''):
    print "Plotting F vs kz at fixed kp=", kp, " for run=", run
    dat = F[F[:,1]==kp,col]
    kzs = np.unique(F[:,0])

    labelstring = run + " m = " + str(m) + r", $k_\perp$ = " + str(kp)
    plt.plot(kzs,dat, label=labelstring)

    plt.xlabel(r'$k_\parallel$', fontsize=20) 

# 2-d spectra at fixed points
def vskpkz(F, col=2, vmin=vmini, vmax=vmaxi, run=''):
    print "Plotting F vs kp,kz for run = ", run
    kz = np.unique(F[:,0])
    kp = np.unique(F[:,1])
    kpsgrid, kzsgrid = np.meshgrid(kp, kz)
    data = F[:,col].reshape(len(kz),len(kp))

    plt.pcolormesh(kpsgrid, kzsgrid, data,vmin=vmin,vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=20)
    plt.ylabel(r'$k_\parallel$', fontsize=20)

# Misc functions
def perp_visc_damp(F, eta, n):
    dat = np.copy(F)
    dat[:,2] = dat[:,2]*eta*(dat[:,1]/dat[:,1].max())**(2.*n)
    return dat

def par_visc_damp(F, eta, n):
    dat = np.copy(F)
    dat[:,2] = dat[:,2]*eta*(dat[:,0]/dat[:,0].max())**(2.*n)
    return dat


if __name__ == "__main__":
    import sys
    # Load data from simulation

    #Fsat_vss(F), Fsat_vss(Fpm), Fsat_vss(Fm)
    #Fsat_vskp(F), Fsat_vskp(Fpm), Fsat_vskp(Fm), Fsat_vskp(Fgwh)
    #Fsat_vskz(F), Fsat_vskz(Fpm), Fsat_vskz(Fm), Fsat_vskz(Fgwh)
    


