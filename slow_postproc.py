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
    ikpmax = len(np.unique(F[:,2]))
    N = Nkp_ikpmax(ikpmax)
    F_fixed = np.copy(F)
    for i in np.arange(ikpmax):
        F_fixed[F_fixed[:,2]==i,3] = F[F[:,2]==i,3] * (2.*np.pi*i/N[i])

    return F_fixed
        
# Data input functions
def load_mkzkp(run):
    file = run + ".mkparkperp.ave"
    F = np.loadtxt(file)
    G = fixed_kp(F)
    #G[:,3] = G[:,3]*np.sqrt(G[:,0]+1.)
    return G

def load_cpm(run):
    file = run + ".pm2.ave"
    F = np.loadtxt(file)
    F_m = np.delete(F, 3, 1)
    G = fixed_kp(F)
    G_m = fixed_kp(F_m)
    G[:,4] = G_m[:,3]
    #G[:,3] = G[:,3]*np.sqrt(G[:,0]+1.)
    #G[:,4] = G[:,4]*np.sqrt(G[:,0]+1.)
    return G

def load_flux(run):
    file = run + ".flux2.ave"
    F = np.loadtxt(file)
    G = fixed_kp(F)
    return G

def load_gm(run):
    file = run + ".slenergy2"
    gm = np.loadtxt(file)
    return gm

def load_all(run):
    F = load_mkzkp(run)
    F_pm = load_cpm(run)
    F_flux = load_flux(run)
    return F,F_flux,F_pm

def alex_sol(F,A,gperp,gpar):
    F_alex = np.empty_like(F)
    F_alex[:,0:3] = F[:,0:3]
    for el in F_alex:
        el[3] = A*el[2]/((gperp*el[1]**2 + gpar*el[2]**2)**1.5)

    return F_alex

def timetr(gm, m, run=''):
    plt.semilogy(gm[:,0], gm[:, m+1], label= run+ ' m=' + str(m))
    plt.xlabel(r'Time $(L/v_{th})$')
    plt.ylabel(r'$\langle g_m^2 \rangle$')
    leg = plt.legend(loc = 0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

# Integrated one-d spectra functions
def energy_per_volume(F, alpha):
    dat = 0
    for el in F:
        if(el[0]!=0):
            dat += el[3]
        else:
            dat += (1 + alpha) * el[3]

    vol = np.sqrt(F[:,0].max()) * F[:,1].max() * F[:,2].max()
    print dat/vol
    
def Fsat_vss(F, scale=1.0, run=''):
    print "Plotting Saturated F vs s for run = " + run
    ms = np.unique(F[:,0])
    dat = np.array(map(lambda m: reduce(sum, F[F[:,0]==m,3]), ms))

    ss = np.array(map(lambda m: scale*np.sqrt(m), ms))
    plt.plot(ss,dat,label = run+' ')
    plt.xlabel('s',fontsize=16)
    plt.ylabel(r'$F$',fontsize=16)
    leg = plt.legend(loc=0, fancybox = True)
    leg.get_frame().set_alpha(0.5)

def Fsat_vskp(F, run=''):
    print "Plotting Saturated F vs kp for run = " + run
    kps = np.unique(F[:,2])
    dat = np.array(map(lambda kp: reduce(sum, F[logand(F[:,2]==kp,F[:,0]!=0),3]), kps))

    plt.plot(kps,dat,label = run+' ')
    plt.xlabel(r'$k_\perp$',fontsize=16)
    plt.ylabel(r'$F$',fontsize=16)
    leg = plt.legend(loc=0, fancybox = True)
    leg.get_frame().set_alpha(0.5)

def comp_Fsat_vskp(F, run=''):
    print "Plotting Saturated F vs kp for run = " + run
    kps = np.unique(F[:,2])
    dat = np.array(map(lambda kp: kp*reduce(sum, F[F[:,2]==kp,3]), kps))

    plt.plot(kps,dat,label = run)
    plt.xlabel(r'$k_\perp$',fontsize=16)
    plt.ylabel(r'$F$',fontsize=16)
    leg = plt.legend(loc=0, fancybox = True)
    leg.get_frame().set_alpha(0.5)

def Fsat_vskz(F, run=''):
    print "Plotting Saturated F vs s for run = " + run
    kzs = np.unique(F[:,1])
    dat = np.array(map(lambda kz: reduce(sum, F[logand(F[:,1]==kz, F[:,0]!=0),3]), kzs))

    plt.plot(kzs,dat,label = run+' ')
    plt.xlabel(r'$k_\parallel$',fontsize=16)
    plt.ylabel(r'$F$',fontsize=16)
    plt.ylim(ymin=1.e-4)
    leg = plt.legend(loc=0, fancybox = True)
    leg.get_frame().set_alpha(0.5)

def comp_Fsat_vskz(F, run=''):
    print "Plotting Saturated F vs s for run = " + run
    kzs = np.unique(F[:,1])
    dat = np.array(map(lambda kz: kz*reduce(sum, F[F[:,1]==kz,3]), kzs))

    plt.plot(kzs,dat,label = run)
    plt.xlabel(r'$k_\parallel$',fontsize=16)
    plt.ylabel(r'$F k_\parallel$',fontsize=16)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(ymin=1.e-4)
    leg = plt.legend(loc=0, fancybox = True)
    leg.get_frame().set_alpha(0.5)

def intkp_vskz(F, m, run=''):
    print "Plotting F vs kz at fixed m=", m, " integrated over kperp for run= ", run
    kz = np.unique(F[:,1])
    dat = np.zeros(len(kz))
    dat = np.array(map(lambda kpar: reduce(sum, F[logand(F[:,0]==m, F[:,1]==kpar),3]), kz))
    
    labelstring= run + " m = " + str(m) 
    plt.plot(kz,dat, label=labelstring)

    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def comp_intkp_vskz(F, m, run=''):
    print "Plotting F vs kz at fixed m=", m, " integrated over kperp for run= ", run
    kz = np.unique(F[:,1])
    dat = np.zeros(len(kz))
    dat = np.array(map(lambda kpar: kpar*reduce(sum, F[logand(F[:,0]==m, F[:,1]==kpar),3]), kz))
    
    labelstring= run+" m = " + str(m) 
    plt.plot(kz,dat, label=labelstring)

    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$F k_\parallel$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def intkp_vss(F, kz, run=''):
    print "Plotting F vs s at fixed kz=", kz, " integrated over kperp for run= ", run
    ms = np.unique(F[:,0])
    dat = np.zeros(len(ms))
    dat = np.array(map(lambda m: reduce(sum, F[logand(F[:,0]==m, F[:,1]==kz),3]), ms))
    
    labelstring= run + r'$k_\parallel$ = '+ str(kz) 
    plt.plot(np.sqrt(ms),dat, label=labelstring)

    plt.xlabel(r'$s$', fontsize=16)
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def comp_intkz_vskp(F, m, run=''):
    print "Plotting F vs kperp at fixed m=", m, " integrated over kz for run= ", run
    kps = np.unique(F[:,2])
    dat = np.zeros(len(kps))
    dat = np.array(map(lambda kp: kp*reduce(sum, F[logand(F[:,0]==m, F[:,2]==kp),3]), kps))

    labelstring= run + " m = " + str(m) 
    plt.plot(kps,dat, label=labelstring)

    plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$F k_\perp$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def intkz_vskp(F, m, run=''):
    print "Plotting F vs kperp at fixed m=", m, " integrated over kz for run= ", run
    kps = np.unique(F[:,2])
    dat = np.zeros(len(kps))
    dat = np.array(map(lambda kp: reduce(sum, F[logand(F[:,0]==m, F[:,2]==kp),3]), kps))

    labelstring= run + " m = " + str(m) 
    plt.plot(kps,dat, label=labelstring)

    plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def intkz_vsm(F, kp, run=''):
    print "Plotting F vs m at fixed kp=", kp, " integrated over kz for run= ", run
    ms = np.unique(F[:,0])
    dat = np.zeros(len(ms))
    dat = np.array(map(lambda m: reduce(sum, F[logand(F[:,2]==kp, F[:,0]==m),3]), ms))
    
    labelstring= run + " kp = " + str(kp) 
    plt.plot(ms,dat, label=labelstring)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ymin=1.e-8)

    plt.xlabel(r'$m$', fontsize=16)
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def intkz_vss(F, kp, run=''):
    print "Plotting F vs s at fixed kp=", kp, " integrated over kz"
    ms = np.unique(F[:,0])
    dat = np.zeros(len(ms))
    dat = np.array(map(lambda m: reduce(sum, F[logand(F[:,2]==kp, F[:,0]==m),3]), ms))
    
    labelstring= run+r" $k_\perp$ = " + str(kp) 
    plt.plot(np.sqrt(ms),dat, label=labelstring)
    plt.yscale('log')
    plt.ylim(ymin=1.e-8)

    plt.xlabel(r'$s$', fontsize=16)
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

# 1-d spectra at fixed points
def vskp(F, m, kz, run=''):
    print "Plotting F vs kp at fixed m=", m, " kz=", kz, " for run=", run
    dat = np.array(filter(lambda x:x[0]==m and x[1]==kz, F))[:,2:]

    labelstring = run + " m = " + str(m) + r", $k_\parallel$ = " + str(kz)
    plt.plot(dat[:,0], dat[:,1],label=labelstring)

    plt.xlabel(r'$k_\perp$', fontsize=16) 
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    
def vskz(F, m, kp,run=''):
    print "Plotting F vs kz at fixed m=", m, " kp=", kp, " for run=", run
    dat = np.array(filter(lambda x:x[0]==m and x[2]==kp,F))
    dat = np.delete(dat, 0, 1)
    dat = np.delete(dat, 1, 1)

    labelstring = run + " m = " + str(m) + r", $k_\perp$ = " + str(kp)
    plt.plot(dat[:,0],dat[:,1], label=labelstring)

    plt.xlabel(r'$k_\parallel$', fontsize=16) 
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def neg_vskz(F, m, kp,run=''):
    print "Plotting F vs kz at fixed m=", m, " kp=", kp, " for run=", run
    dat = np.array(filter(lambda x:x[0]==m and x[2]==kp,F))
    dat = np.delete(dat, 0, 1)
    dat = np.delete(dat, 1, 1)

    labelstring = run + " m = " + str(m) + r", $k_\perp$ = " + str(kp)
    plt.plot(-dat[:,0],dat[:,1], label=labelstring)
    plt.ylim(ymin=1.e-8)

    plt.xlabel(r'$k_\parallel$', fontsize=16) 
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def gwh_F_vskz(F, m, kp, run=''):
    F_m = np.delete(F, 3, 1)
    vskz(F, m, kp, run)
    neg_vskz(F_m, m, kp, run)

def vsm(F, kp, kz,run=''):
    print "Plotting F vs m at fixed kp=", kp, " kz=", kz, " for run=", run
    dat = np.array(filter(lambda x:x[1]==kz and x[2]==kp,F))
    dat = np.delete(dat, 1, 1)
    dat = np.delete(dat, 1, 1)

    labelstring = run + r"$k_\perp$ = " + str(kp) + r", $k_\parallel$ = " + str(kz)
    plt.plot(dat[:,0],dat[:,1], label=labelstring)
    plt.xscale('log')
    plt.yscale('log')

    plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$m$', fontsize=16) 
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def vss(F, kp, kz,run=''):
    print "Plotting F vs s at fixed kp=", kp, " kz=", kz, " for run=", run
    dat = np.array(filter(lambda x:x[1]==kz and x[2]==kp,F))
    dat = np.delete(dat, 1, 1)
    dat = np.delete(dat, 1, 1)

    labelstring = run + r"$k_\perp$ = " + str(kp) + r", $k_\parallel$ = " + str(kz)
    plt.plot(np.sqrt(dat[:,0]),dat[:,1], label=labelstring)
    plt.yscale('log')
    plt.ylim(ymin=1.e-8)

    plt.xlabel(r'$s$', fontsize=16) 
    plt.ylabel(r'$F$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def supp_pm_vskp(Fpm, m, kz, run=''):
    print "Plotting Suppression factor vs kperp at fixed m=", m, " kz=", kz, " for run=", run
    dat = np.array(filter(lambda x:x[0]==m and x[1]==kz, Fpm))
    dat = np.delete(dat, 0, 1)
    dat = np.delete(dat, 0, 1)
    
    labelstring = run + " m = " + str(m) + r", $k_\parallel$ = " + str(kz)
    plt.plot(dat[:,0],(dat[:,1]-dat[:,2])/(dat[:,1]+dat[:,2]), label=labelstring)
    plt.xscale('log')
    plt.ylim(ymin=-1.0,ymax=1.0)

    plt.xlabel(r'$k_\perp$', fontsize=16) 
    plt.ylabel(r'$\frac{F^+ - F^-}{F^+ + F^-}$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def supp_vskp(F, Flux, m, kz, run=''):
    print "Plotting Flux/Energy vs kperp at fixed m=", m, " kz=", kz, " for run=", run
    Cm = np.array(filter(lambda x:x[0]==m and x[1]==kz, F))
    Cm = np.delete(Cm, 0, 1)
    Cm = np.delete(Cm, 0, 1)

    fl = np.array(filter(lambda x:x[0]==m and x[1]==kz, Flux))
    fl = np.delete(fl, 0, 1)
    fl = np.delete(fl, 0, 1)

    labelstring = run + " m = " + str(m) + r", $k_\parallel$ = " + str(kz)
    plt.plot(Cm[:,0],fl[:,1]/Cm[:,1], label=labelstring)
    plt.xscale('log')

    plt.xlabel(r'$k_\perp$', fontsize=16) 
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def supp_pm_vsm(Fpm, kp, kz,run=''):
    print "Plotting F vs m at fixed kp=", kp, " kz=", kz, " for run=", run
    dat = np.array(filter(lambda x:x[1]==kz and x[2]==kp, Fpm))
    dat = np.delete(dat, 1, 1)
    dat = np.delete(dat, 1, 1)

    labelstring = run + r"$k_\perp$ = " + str(kp) + r", $k_\parallel$ = " + str(kz)
    plt.plot(dat[:,0],(dat[:,1]-dat[:,2])/(dat[:,1]+dat[:,2]), label=labelstring)
    plt.xscale('log')
    plt.ylim(ymin=-1.0, ymax = 1.0)

    plt.xlabel(r'$m$', fontsize=16) 
    plt.ylabel(r'$\frac{F^+ - F^-}{F^+ + F^-}$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def supp_vsm(F, Flux, kp, kz,run=''):
    print "Plotting F vs m at fixed kp=", kp, " kz=", kz, " for run=", run
    Cm = np.array(filter(lambda x:x[1]==kz and x[2]==kp, F))
    Cm = np.delete(Cm, 1, 1)
    Cm = np.delete(Cm, 1, 1)

    fl = np.array(filter(lambda x:x[1]==kz and x[2]==kp, Flux))
    fl = np.delete(fl, 1, 1)
    fl = np.delete(fl, 1, 1)

    labelstring = run + r"$k_\perp$ = " + str(kp) + r", $k_\parallel$ = " + str(kz)
    plt.plot(Cm[:,0],fl[:,1]/Cm[:,1], label=labelstring)
    plt.xscale('log')
    plt.ylim(ymin=-1.0, ymax = 1.0)

    plt.xlabel(r'$m$', fontsize=16) 
    plt.ylabel(r'$\frac{\Gamma_m}{F_m}$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def supp_pm_vss(Fpm, kp, kz,run=''):
    print "Plotting F vs s at fixed kp=", kp, " kz=", kz, " for run=", run
    dat = np.array(filter(lambda x:x[1]==kz and x[2]==kp, Fpm))
    dat = np.delete(dat, 1, 1)
    dat = np.delete(dat, 1, 1)

    labelstring = run + r"$k_\perp$ = " + str(kp) + r", $k_\parallel$ = " + str(kz)
    plt.plot(np.sqrt(dat[:,0]),(dat[:,1]-dat[:,2])/(dat[:,1]+dat[:,2]), label=labelstring)
    plt.ylim(ymin=-1.0,ymax=1.0)

    plt.xlabel(r'$s$', fontsize=16) 
    plt.ylabel(r'$\frac{F^+ - F^-}{F^+ + F^-}$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def supp_vss(F, Flux, kp, kz,run=''):
    print "Plotting F vs s at fixed kp=", kp, " kz=", kz, " for run=", run
    Cm = np.array(filter(lambda x:x[1]==kz and x[2]==kp, F))
    Cm = np.delete(Cm, 1, 1)
    Cm = np.delete(Cm, 1, 1)

    fl = np.array(filter(lambda x:x[1]==kz and x[2]==kp, Flux))
    fl = np.delete(fl, 1, 1)
    fl = np.delete(fl, 1, 1)

    labelstring = run + r"$k_\perp$ = " + str(kp) + r", $k_\parallel$ = " + str(kz)
    plt.plot(np.sqrt(Cm[:,0]),fl[:,1]/Cm[:,1], label=labelstring)
    plt.ylim(ymin=-1.0,ymax=1.0)

    plt.xlabel(r'$s$', fontsize=16) 
    plt.ylabel(r'$\frac{\Gamma_m}{F_m}$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def supp_pm_vskz(Fpm, m, kp,run=''):
    print "Plotting F vs kz at fixed m=", m, " kp=", kp, " for run=", run
    dat = np.array(filter(lambda x:x[0]==m and x[2]==kp, Fpm))
    dat = np.delete(dat, 0, 1)
    dat = np.delete(dat, 1, 1)

    labelstring = run + " m = " + str(m) + r", $k_\perp$ = " + str(kp)
    plt.plot(dat[:,0],(dat[:,1]-dat[:,2])/(dat[:,1]+dat[:,2]), label=labelstring)
    plt.xscale('log')
    plt.ylim(ymin=-1.0, ymax=1.0)

    plt.xlabel(r'$k_\parallel$', fontsize=16) 
    plt.ylabel(r'$\frac{F^+ - F^-}{F^+ + F^-}$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def supp_vskz(F, Flux, m, kp,run=''):
    print "Plotting F vs kz at fixed m=", m, " kp=", kp, " for run=", run
    Cm = np.array(filter(lambda x:x[0]==m and x[2]==kp, F))
    Cm = np.delete(Cm, 0, 1)
    Cm = np.delete(Cm, 1, 1)


    fl = np.array(filter(lambda x:x[0]==m and x[2]==kp, Flux))
    fl = np.delete(fl, 0, 1)
    fl = np.delete(fl, 1, 1)

    labelstring = run + " m = " + str(m) + r", $k_\perp$ = " + str(kp)
    plt.plot(Cm[:,0],fl[:,1]/Cm[:,1], label=labelstring)
    plt.xscale('log')
    #plt.ylim(ymin=0.0, ymax=0.5)

    plt.xlabel(r'$k_\parallel$', fontsize=16) 
    plt.ylabel(r'$\frac{Re[\tilde{g}_{m+1}\tilde{g}^\star_m}{C_m}$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def der0_intkp_vss(Fpm, run=''):
    print "Plotting C+(1) - C-(1) integrated over kp"
    F1 = np.array(filter(lambda x:x[1]==1, Fpm))
    F1 = np.delete(F1, 1, 1)
    ms = np.unique(F1[:,0])

    F = np.array(map(lambda m: reduce(sum, (F1[F1[:,0]==m,2] - F1[F1[:,0]==m,3])/2.), ms))


    plt.plot(np.sqrt(ms), F, label='test')

    plt.xlabel(r'$k_\perp$',fontsize=16)
    plt.ylabel(r'$\left|F_{m,q}^+ - F_{m,-q}^- \right|/2$',fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def der0_intkp_vss(Fpm, run=''):
    print "Plotting C+(1) - C-(1) "
    F1 = np.array(filter(lambda x:x[1]==1, Fpm))
    F1 = np.delete(F1, 1, 1)
    ms = np.unique(F1[:,0])
    der = np.array(map(lambda m: reduce(sum, (F1[F1[:,0]==m,2] - F1[F1[:,0]==m,3])/2.), ms))

    plt.plot(np.sqrt(ms), der, label=run + ' ' )

    plt.xlabel(r'$s$',fontsize=16)
    plt.ylabel(r'$\left|F_{m,q}^+ - F_{m,-q}^- \right|/2.$',fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def der0_vskp(Fpm, m, run=''):
    print "Plotting C+(1) - C-(1) for m=",m, "for run =", run
    F1 = np.array(filter(lambda x:x[0]==m and x[1]==1, Fpm))
    F1 = np.delete(F1, 0, 1)
    F1 = np.delete(F1, 0, 1)

    plt.plot(F0[:,0], np.abs(F1[:,1] - F1[:,2])/2., label=run + ' m = ' + str(m) )
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ymin=1.e-8)

    plt.xlabel(r'$k_\perp$',fontsize=16)
    plt.ylabel(r'$\left|F_{m,q}^+ - F_{m,-q}^- \right|/2.$',fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def der0_vsskp(Fpm, vmin=vmini, vmax=vmaxi, run=''):
    print "Plotting F+(1) - F-(1) for run =", run
    F0 = np.delete(Fpm[Fpm[:,1]==0],1,1)
    F1 = np.delete(Fpm[Fpm[:,1]==1],1,1)

    s = np.sqrt(np.unique(F0[:,0]))
    kp = np.unique(F0[:,1])
    kpsgrid, ssgrid = np.meshgrid(kps, s)
    data = (np.abs(F1[:,2] - F1[:,3] )/2.).reshape(len(s),len(kp))

    plt.pcolormesh(kpgrid,ssgrid,data,vmin=vmin,vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\perp$',fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def Fpm_ratio_vss(Fpm, kp, kz,run=''):
    print "Plotting F-/F+ vs s at fixed kp=", kp, " kz=", kz, " for run=", run
    dat = np.delete(Fpm[Fpm[:,2]==kp],2,1)
    dat = np.delete(dat[dat[:,1]==kz],1,1)

    labelstring = run + r"$k_\perp$ = " + str(kp) + r", $k_\parallel$ = " + str(kz)
    plt.plot(np.sqrt(dat[:,0]),dat[:,2]/dat[:,1], label=labelstring)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$s$', fontsize=16) 
    plt.ylabel(r'$F^-/F^+$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

# 2-d spectra at fixed points
def vskpkz(F, m, vmin=vmini, vmax=vmaxi, run=''):
    print "Plotting F vs kp,kz at fixed m = ", m, " for run = ", run
    kz = np.unique(F[:,1])
    kp = np.unique(F[:,2])
    kpsgrid, kzsgrid = np.meshgrid(kp, kz)
    data = F[F[:,0]==m,3].reshape(len(kz),len(kp))

    plt.pcolormesh(kpsgrid, kzsgrid, data,vmin=vmin,vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$k_\parallel$', fontsize=16)

def vsmkz(F, kp, vmin=vmini, vmax=vmaxi, run = ''):
    print "Plotting F vs m,kz at fixed kp = ", kp, " for run = ", run
    ms = np.unique(F[:,0])
    kzs = np.unique(F[:,1])
    kzsgrid, msgrid = np.meshgrid(kzs, ms)
    data = F[F[:,2]==kp,3].reshape(len(ms),len(kzs))

    plt.pcolormesh(kzsgrid,msgrid,data,vmin=vmin,vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$m$', fontsize=16)

def vsskz(F, kp, vmin=vmini, vmax=vmaxi, run = ''):
    print "Plotting F vs s,kz at fixed kp = ", kp, " for run = ", run
    ms = np.unique(F[:,0])
    kzs = np.sort(np.unique(F[:,1]))
    kzsgrid, msgrid = np.meshgrid(kzs, ms)
    data = F[F[:,2]==kp,3].reshape(len(ms),len(kzs))
    if kzs[1]<0: data = data[:,::-1]

    plt.pcolormesh(kzsgrid,np.sqrt(msgrid),data,vmin=vmin, vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def gwh_intkp_vsskz(F, vmin=vmini, vmax=vmaxi):
    print "Plotting F vs s,kz integrated over kp"
    ms = np.unique(F[:,0])
    pkzs = np.unique(F[:,1])
    nkzs = np.sort(-pkzs[1:])
    kzs = np.append(nkzs,pkzs)
    kzsgrid, msgrid = np.meshgrid(kzs, ms)

    datan = np.zeros(len(ms)*len(nkzs))
    datap = np.zeros(len(ms)*len(pkzs))
    datan = np.array(map(lambda m: map(lambda nkz: reduce(sum, F[logand(F[:,0]==m, F[:,1]==-nkz),4]),nkzs),ms))
    datap = np.array(map(lambda m: map(lambda pkz: reduce(sum, F[logand(F[:,0]==m, F[:,1]==pkz),3]),pkzs),ms))
    datan = datan.reshape(len(ms),len(nkzs))
    datap = datap.reshape(len(ms),len(pkzs))
    data = np.zeros((len(ms),len(kzs)))
    data[:,:len(nkzs)] = datan
    data[:,len(nkzs):] = datap

    plt.pcolormesh(kzsgrid,np.sqrt(msgrid),data,vmin=vmin,vmax=vmax, norm=LogNorm())
    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def intkp_vsskz(F, vmin=vmini, vmax=vmaxi):
    print "Plotting F vs s,kz integrated over kp"
    ms = np.unique(F[:,0])
    kzs = np.unique(F[:,1])
    kzsgrid, msgrid = np.meshgrid(kzs, ms)
    data = np.zeros(len(ms)*len(kzs))
    data = np.array(map(lambda m: map(lambda kz: reduce(sum, F[logand(F[:,0]==m, F[:,1]==kz),3]),kzs),ms)).reshape(len(ms),len(kzs))

    plt.pcolormesh(kzsgrid,np.sqrt(msgrid),data,vmin=vmin, vmax=vmax, norm=LogNorm())
    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def vsmkp(F, kz, vmin=vmini, vmax=vmaxi, run = ''):
    print "Plotting F vs m,kp at fixed kz = ", kz, " for run = ", run
    ms = np.unique(F[:,0])
    kps = np.unique(F[:,1])
    kpsgrid, msgrid = np.meshgrid(kps, ms)
    data = F[F[:,1]==kz,3].reshape(len(ms),len(kps))

    plt.pcolormesh(kpsgrid,msgrid,data,vmin=vmin,vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$m$', fontsize=16)

def vsskp(F, kz, vmin=vmini, vmax=vmaxi):
    print "Plotting F vs s,kp at fixed kz = ", kz
    ms = np.unique(F[:,0])
    kps = (np.unique(F[:,2]))
    kpsgrid, msgrid = np.meshgrid(kps, ms)
    data = F[F[:,1]==kz,3].reshape(len(ms),len(kps))
    if kps[0]<0: data = data[:,::-1]

    plt.pcolormesh(kps,np.sqrt(ms),data,vmin=vmin,vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def intkz_vsmkp(F, vmin=vmini, vmax=vmaxi, run=''):
    print "Plotting F vs m,kp integrated over kz for run = ", run
    ms = np.unique(F[:,0])
    kps = np.unique(F[:,2])
    kpsgrid, msgrid = np.meshgrid(kps, ms)
    data = np.zeros(len(ms)*len(kps))
    data = np.array(map(lambda m: map(lambda kp: reduce(sum, F[logand(F[:,0]==m, F[:,2]==kp),3]),kps),ms)).reshape(len(ms), len(kps))

    plt.pcolormesh(kpsgrid,msgrid,data,vmin=vmin,vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$m$', fontsize=16)

def comp_intkz_vsskp(F, vmin=vmini, vmax=vmaxi):
    print "Plotting F vs s,kp integrated over kz "
    ms = np.unique(F[:,0])
    kps = np.unique(F[:,2])
    data = np.zeros(len(ms)*len(kps))
    data = np.array(map(lambda m: map(lambda kp: kp*reduce(sum, F[logand(F[:,0]==m, F[:,2]==kp),3]),kps),ms)).reshape(len(ms), len(kps))

    plt.pcolormesh(kps,np.sqrt(ms),data,vmin=vmin,vmax=vmax, norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def intkz_vsskp(F, vmin=vmini, vmax=vmaxi):
    print "Plotting F vs s,kp integrated over kz "
    ms = np.unique(F[:,0])
    kps = np.unique(F[:,2])
    data = np.zeros(len(ms)*len(kps))
    data = np.array(map(lambda m: map(lambda kp: reduce(sum, F[logand(F[:,0]==m, F[:,2]==kp),3]),kps),ms)).reshape(len(ms), len(kps))

    plt.pcolormesh(kps,np.sqrt(ms),data,vmin=vmin,vmax=vmax, norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def s2F_intkz_vsskp(F, vmin=vmini, vmax=vmaxi):
    print "Plotting F vs s,kp integrated over kz "
    ms = np.unique(F[:,0])
    kps = np.unique(F[:,2])
    kpsgrid, msgrid = np.meshgrid(kps, ms)
    data = np.zeros(len(ms)*len(kps))
    data = np.array(map(lambda m: map(lambda kp: reduce(sum, (m+1)*F[logand(F[:,0]==m, F[:,2]==kp),3]),kps),ms)).reshape(len(ms), len(kps))

    plt.pcolormesh(kpsgrid,np.sqrt(msgrid),data,vmin=vmin,vmax=vmax, norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def selfsim_intkz_vsskp(F, vmin=vmini, vmax=vmaxi):
    print "Plotting F vs s,kp integrated over kz "
    ms = np.unique(F[:,0])
    kps = np.unique(F[:,2])
    data = np.zeros(len(ms)*len(kps))
    data = np.array(map(lambda m: map(lambda kp: reduce(sum, (m+1.)*F[logand(F[:,0]==m, F[:,2]==kp),3]),kps),ms)).reshape(len(ms), len(kps))

    kpsgrid, msgrid = np.meshgrid(kps, ms)
    kpsgrid = np.array(map(lambda m: kpsgrid[m,:]/np.sqrt(m+1.), ms))
    plt.pcolormesh(kpsgrid,np.sqrt(msgrid),data,vmin=vmin,vmax=vmax, norm=LogNorm())

    plt.xlabel(r'$k_\perp/s$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def Fpm_ratio_vskpkz(Fpm, m, vmin=vmini, vmax=vmaxi):
    print "Plotting F-/F+ vs kp,kz at fixed m = ", m, " for run = ", run
    kzs = np.unique(Fpm[:,1])
    kps = np.unique(Fpm[:,2])
    kpsgrid, kzsgrid = np.meshgrid(kps, kzs)
    Fp = Fpm[Fpm[:,0]==m,3].reshape(len(kzs),len(kps))
    Fm = Fpm[Fpm[:,0]==m,4].reshape(len(kzs),len(kps))

    plt.pcolormesh(kpsgrid, kzsgrid, Fm/Fp,vmin=vmin,vmax=vmax,norm=LogNorm())

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$k_\parallel$', fontsize=16)

def supp_pm_vskpkz(F, m, vmin=0.0, vmax=1.0,run=''):
    print "Plotting suppression vs kp,kz at fixed m = ", m, " for run = ", run
    kzs = np.unique(F[:,1])
    kps = np.unique(F[:,2])
    kpsgrid, kzsgrid = np.meshgrid(kps, kzs)
    data = (F[F[:,0]==m,3] - F[F[:,0]==m,4])/(F[F[:,0]==m,3] + F[F[:,0]==m,4])
    data = data.reshape(len(kzs),len(kps))

    plt.pcolormesh(kpsgrid,kzsgrid,data,vmin=vmin,vmax=vmax)

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$k_\parallel$', fontsize=16)
    
def supp_pm_vsmkz(F, kp, vmin=0.0, vmax=1.0, run = ''):
    print "Plotting suppression vs m,kz at fixed kp = ", kp, " for run = ", run
    ms = np.unique(F[:,0])
    kzs = np.unique(F[:,1])
    kzsgris, msgrid = np.meshgrid(kzs, ms)
    data = (F[F[:,2]==kp,3] - F[F[:,2]==kp,4])/(F[F[:,2]==kp,3] + F[F[:,2]==kp,4]).reshape(len(ms),len(kzs))

    plt.pcolormesh(kzsgrid,msgrid,data,vmin=vmin, vmax=vmax)

    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$m$', fontsize=16)

def supp_pm_vsskz(F, kp, vmin=0.0, vmax=1.0, run = ''):
    print "Plotting suppression vs s,kz at fixed kp = ", kp, " for run = ", run
    ms = np.unique(F[:,0])
    kzs = np.unique(F[:,1])
    kzsgrid, msgrid = np.meshgrid(kzs, ms)
    data = (F[F[:,2]==kp,3] - F[F[:,2]==kp,4])/(F[F[:,2]==kp,3] + F[F[:,2]==kp,4]).reshape(len(ms),len(kzs))

    plt.pcolormesh(kzsgrid,np.sqrt(msgrid),data,vmin=vmin, vmax=vmax)

    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def supp_pm_vsmkp(F, kz, vmin=0.0, vmax=1.0, run = ''):
    print "Plotting suppression vs m,kp at fixed kz = ", kz, " for run = ", run
    ms = np.unique(F[:,0])
    kps = np.unique(F[:,2])
    kpsgrid, msgrid = np.meshgrid(kps, ms)
    data = (F[F[:,1]==kz,3] - F[F[:,1]==kz,4])/(F[F[:,1]==kz,3] + F[F[:,1]==kz,4]).reshape(len(ms),len(kps))

    plt.pcolormesh(kpsgrid,msgrid,data,vmin=vmin, vmax=vmax)

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$m$', fontsize=16)

def supp_pm_vsskp(F, kz, vmin=0.0, vmax=1.0, run = ''):
    print "Plotting suppression vs m,kp at fixed kz = ", kz, " for run = ", run
    ms = np.unique(F[:,0])
    kps = np.unique(F[:,2])
    kpsgrid, msgrid = np.meshgrid(kps, ms)
    data = (F[F[:,1]==kz,3] - F[F[:,1]==kz,4])/(F[F[:,1]==kz,3] + F[F[:,1]==kz,4]).reshape(len(ms),len(kps))

    plt.pcolormesh(kpsgrid,np.sqrt(msgrid),data,vmin=vmin, vmax=vmax)

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def geff_vskpkz(F, Flux, m, vmin=0.0, vmax=1.0,run=''):
    print "Plotting suppression vs kp,kz at fixed m = ", m, " for run = ", run
    kzs = np.unique(F[:,1])
    kps = np.unique(F[:,2])
    kpsgrid, kzsgrid = np.meshgrid(kps, kzs)
    Cm = F[F[:,0]==m,3].reshape(len(kzs),len(kps))
    fl = (Flux[Flux[:,0]==m,3]/Flux[Flux[:,0]==m,1]).reshape(len(kzs),len(kps))

    plt.pcolormesh(kpsgrid,kzsgrid,fl/Cm,vmin=vmin, vmax=vmax)

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$k_\parallel$', fontsize=16)
    
def supp_vsskz(F, Flux, kp, vmin=0.0, vmax=1.0, run = ''):
    print "Plotting suppression vs s,kz at fixed kp = ", kp, " for run = ", run
    ms = np.unique(F[:,0])[:-1]
    kzs = np.unique(F[:,1])[1:]
    kzsgrid, msgrid = np.meshgrid(kzs, ms)
    Cm = np.array(filter(lambda x:x[2]==kp and x[1]!=0 and x[0]!=ms.max()+1, F))[:,3].reshape(len(ms),len(kzs))
    fl = np.array(filter(lambda x:x[2]==kp and x[1]!=0 and x[0]!=ms.max()+1, Flux))[:,3]/np.array(filter(lambda x:x[2]==kp and x[1]!=0 and x[0]!=ms.max()+1, Flux))[:,1]
    fl = fl.reshape(len(ms), len(kzs))

    plt.pcolormesh(kzsgrid,np.sqrt(msgrid),fl/Cm/np.sqrt(2.),vmin=vmin, vmax=vmax)
    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def supp_intkp_vsskz(F, Flux, vmin=0.0, vmax=1.0, run = ''):
    print "Plotting suppression vs s,kz  for run = ", run
    ms = np.unique(F[:,0])[:-1]
    kzs = np.unique(F[:,1])[1:]
    kzsgrid, msgrid = np.meshgrid(kzs, ms)
    Cm = np.array(map(lambda m: map(lambda kz: reduce(sum, F[logand(F[:,0]==m, F[:,1]==kz),3]), kzs), ms)).reshape(len(ms), len(kzs))
    fl = np.array(map(lambda m: map(lambda kz: reduce(sum, Flux[logand(Flux[:,0]==m, Flux[:,1]==kz),3])/kz, kzs), ms)).reshape(len(ms), len(kzs))

    plt.pcolormesh(kzsgrid,np.sqrt(msgrid),fl/Cm/np.sqrt(2.),vmin=vmin, vmax=vmax)
    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def supp_vsskp(F, Flux, kz, vmin=0.0, vmax=1.0, run = ''):
    print "Plotting suppression vs m,kp at fixed kz = ", kz, " for run = ", run
    ms = np.unique(F[:,0])[:-1]
    kps = np.unique(F[:,2])
    kpsgrid, msgrid = np.meshgrid(kps, ms)
    Cm = np.array(filter(lambda x: x[0]!=ms.max()+1 and x[1]==kz, F))[:,3].reshape(len(ms),len(kps))
    fl = (np.array(filter(lambda x: x[0]!=ms.max()+1 and x[1]==kz,Flux))[:,3]/kz).reshape(len(ms), len(kps))

    plt.pcolormesh(kpsgrid,np.sqrt(msgrid),fl/Cm/np.sqrt(2.),vmin=0.0, vmax=1.0)
    plt.xscale('log')

    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)

def supp_intkz_vsskp(F, Flux, vmin=0.0, vmax=1.0, run = ''):
    print "Plotting suppression vs s,kp  for run = ", run
    ms = np.unique(F[:,0])[:-1]
    kps = np.unique(F[:,2])
    kpsgrid, msgrid = np.meshgrid(kps, ms)
    def sel_nzero_mkp(F, m, kp): 
        return filter(lambda x:x[0]==m and x[1]!=0 and x[2]==kp, F)

    Cm = np.array(map(lambda m: map(lambda kp: reduce(sum, sel_nzero_mkp(F,m,kp)[3]), kps), ms)).reshape(len(ms), len(kps))
    fl = np.array(map(lambda m: map(lambda kp: reduce(sum, (sel_nzero_mkp(Flux,m,kp)[3])/(sel_nzero_mkp(Flux,m,kp)[1])), kps), ms)).reshape(len(ms), len(kps)) 
    plt.pcolormesh(kpsgrid,np.sqrt(msgrid),fl/Cm/np.sqrt(2.),vmin=vmin, vmax=vmax)
    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$s$', fontsize=16)
    


    

# looping functions, in order to pass lists
def loop_vskp(F, ms, kzs, run):
    for m in ms:
        for kz in kzs:
            vskp(F, m, kz, run)
def loop_vskz(F, ms, kps, run):
    for m in ms:
        for kp in kps:
            vskz(F, m, kp, run)
def loop_vsm(F, kps, kzs, run):
    for kp in kps:
        for kz in kzs:
            vsm(F, kp, kz, run)

# Misc functions
def coll_damp(F, nu, n):
    dat = np.copy(F)
    dat[:,3] = dat[:,3]*nu*(dat[:,0]/dat[:,0].max())**n
    dat[0,3]=0.0
    dat[1,3]=0.0
    dat[2,3]=0.0
    return dat

def perp_visc_damp(F, eta, n):
    dat = np.copy(F)
    dat[:,3] = dat[:,3]*eta*(dat[:,2]/dat[:,2].max())**(2.*n)
    return dat

def par_visc_damp(F, eta, n):
    dat = np.copy(F)
    dat[:,3] = dat[:,3]*eta*(dat[:,1]/dat[:,1].max())**(2.*n)
    return dat


def calc_s_der(F, kp, kz, m1, m2):
    dat1 = F[F[:,0]==m1,1:]
    dat1 = dat1[dat1[:,0]==kz,1:]
    dat1 = dat1[dat1[:,0]==kp,1]

    dat2 = F[F[:,0]==m2,1:]
    dat2 = dat2[dat2[:,0]==kz,1:]
    dat2 = dat2[dat2[:,0]==kp,1]

    return (np.log(dat2) - np.log(dat1))/(np.sqrt(m2) - np.sqrt(m1))

def gwh_F(Fpm):
    F_gwh = np.zeros(np.shape(Fpm))
    F_gwh[:,0] = np.copy(Fpm[:,0])
    F_gwh[:,1] = np.copy(-Fpm[:,1])
    F_gwh[:,2] = np.copy(-Fpm[:,2])
    F_gwh[:,3] = Fpm[:,4]
    return F_gwh

def ave_s_vskpkz(Fpm,Sc=1, run='', vmin=0.0, vmax=1.0):
    print "Plotting width in s vs kperp as defined by GWH for run = ", run
    kzs = np.unique(Fpm[:,1])
    kperp = np.unique(Fpm[:,2])
    kpsgrid, kzsgrid = np.meshgrid(kperp, kzs)
    dat = np.zeros(len(kperp)*len(kzs))
    norm = np.zeros(len(kperp)*len(kzs))

    dat = np.array(map(lambda kz: map(lambda kp: reduce(sum, (Fpm[logand(Fpm[:,2]==kp, Fpm[:,1]==kz),3] + Fpm[logand(Fpm[:,2]==kp, Fpm[:,1]==kz),4])*np.sqrt(Fpm[logand(Fpm[:,2]==kp, Fpm[:,1]==kz),0])), kperp), kzs))
    norm = np.array(map(lambda kz: map(lambda kp:reduce(sum, (Fpm[logand(Fpm[:,2]==kp, Fpm[:,1]==kz, Fpm[:,0]!=0),3] + Fpm[logand(Fpm[:,2]==kp, Fpm[:,1]==kz, Fpm[:,0]!=0),4])), kperp), kzs))
    
    
    print reduce(sum, (Fpm[:,3]+Fpm[:,4])*np.sqrt(Fpm[:,0]))/reduce(sum, (Fpm[:,3] + Fpm[:,4]))
    dat = dat.reshape(len(kzs), len(kperp))
    norm = norm.reshape(len(kzs), len(kperp))
    dat=dat/norm

    plt.pcolormesh(kpsgrid, kzsgrid, dat, vmin=vmin, vmax=vmax)
    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$k_\parallel$', fontsize=16)
    plt.colorbar(ticks = ticks)

def ave_s_vskz(Fpm,Sc=1, run=''):
    print "Plotting width in s vs kz as defined by GWH for run = ", run
    kzs = np.unique(Fpm[:,1])
    dat = np.zeros((len(kzs),2))
    norm = np.zeros(len(kzs))
    dat[:,0] = kzs
    dat[:,1] = np.array(map(lambda kz:reduce(sum, (Fpm[Fpm[:,1]==kz,3] + Fpm[Fpm[:,1]==kz,4])*np.sqrt(Fpm[Fpm[:,1]==kz,0])), kzs))

    norm[:] = np.array(map(lambda kz:reduce(sum, (Fpm[logand(Fpm[:,1]==kz, Fpm[:,0]!=0),3] + Fpm[logand(Fpm[:,1]==kz, Fpm[:,0]!=0),4])), kzs))
    
    
    print reduce(sum, dat[:,1])/reduce(sum, norm[:])

    plt.plot(dat[:,0], dat[:,1]/Sc/norm, label=run)
    #plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$\langle s F \rangle/\langle F \rangle$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def ave_s_vskp(Fpm,Sc=1, run=''):
    print "Plotting width in s vs kz as defined by GWH for run = ", run
    kperp = np.unique(Fpm[:,2])
    dat = np.zeros((len(kperp),2))
    norm = np.zeros(len(kperp))
    dat[:,0] = kperp
    dat[:,1] = np.array(map(lambda kp:reduce(sum, (Fpm[Fpm[:,2]==kp,3] + Fpm[Fpm[:,2]==kp,4])*np.sqrt(Fpm[Fpm[:,2]==kp,0])), kperp))

    norm[:] = np.array(map(lambda kp:reduce(sum, (Fpm[logand(Fpm[:,2]==kp, Fpm[:,0]!=0),3] + Fpm[logand(Fpm[:,2]==kp, Fpm[:,0]!=0),4])), kperp))
    
    
    print reduce(sum, dat[:,1])/reduce(sum, norm[:])

    plt.plot(dat[:,0], dat[:,1]/Sc/norm, label=run)
    #plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$\langle s F \rangle/\langle F \rangle$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

        
def ave_s_vskz_plus(Fpm,Sc=1, run=''):
    print "Plotting width in s vs kz as defined by GWH for run = ", run
    kzs = np.unique(Fpm[:,1])
    dat = np.zeros((len(kzs),2))
    norm = np.zeros(len(kzs))
    dat[:,0] = kzs
    dat[:,1] = np.array(map(lambda kz:reduce(sum, (Fpm[Fpm[:,1]==kz,3] )*np.sqrt(Fpm[Fpm[:,1]==kz,0])), kzs))

    norm[:] = np.array(map(lambda kz:reduce(sum, (Fpm[logand(Fpm[:,1]==kz, Fpm[:,0]!=0),3] )), kzs))
    
    
    print reduce(sum, dat[:,1])/reduce(sum, norm[:])

    plt.plot(dat[:,0], dat[:,1]/Sc/norm, label=run)
    #plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$\langle s F^+ \rangle/\langle F^+ \rangle$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def ave_s_vskp_plus(Fpm,Sc=1, run=''):
    print "Plotting width in s vs kz as defined by GWH for run = ", run
    kperp = np.unique(Fpm[:,2])
    dat = np.zeros((len(kperp),2))
    norm = np.zeros(len(kperp))
    dat[:,0] = kperp
    dat[:,1] = np.array(map(lambda kp:reduce(sum, (Fpm[Fpm[:,2]==kp,3] )*np.sqrt(Fpm[Fpm[:,2]==kp,0])), kperp))

    norm[:] = np.array(map(lambda kp:reduce(sum, (Fpm[logand(Fpm[:,2]==kp, Fpm[:,0]!=0),3] )), kperp))
    
    
    print reduce(sum, dat[:,1])/reduce(sum, norm[:])

    plt.plot(dat[:,0], dat[:,1]/Sc/norm, label=run)
    #plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$\langle s F^+ \rangle/\langle F^+ \rangle$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def ave_s_vskz_minus(Fpm,Sc=1, run=''):
    print "Plotting width in s vs kz as defined by GWH for run = ", run
    kzs = np.unique(Fpm[:,1])
    dat = np.zeros((len(kzs),2))
    norm = np.zeros(len(kzs))
    dat[:,0] = kzs
    dat[:,1] = np.array(map(lambda kz:reduce(sum, (Fpm[Fpm[:,1]==kz,4] )*np.sqrt(Fpm[Fpm[:,1]==kz,0])), kzs))

    norm[:] = np.array(map(lambda kz:reduce(sum, (Fpm[logand(Fpm[:,1]==kz, Fpm[:,0]!=0),4] )), kzs))
    
    
    print reduce(sum, dat[:,1])/reduce(sum, norm[:])

    plt.plot(dat[:,0], dat[:,1]/Sc/norm, label=run)
    #plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\parallel$', fontsize=16)
    plt.ylabel(r'$\langle s F^- \rangle/\langle F^- \rangle$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def ave_s_vskp_minus(Fpm,Sc=1, run=''):
    print "Plotting width in s vs kz as defined by GWH for run = ", run
    kperp = np.unique(Fpm[:,2])
    dat = np.zeros((len(kperp),2))
    norm = np.zeros(len(kperp))
    dat[:,0] = kperp
    dat[:,1] = np.array(map(lambda kp:reduce(sum, (Fpm[Fpm[:,2]==kp,4] )*np.sqrt(Fpm[Fpm[:,2]==kp,0])), kperp))

    norm[:] = np.array(map(lambda kp:reduce(sum, (Fpm[logand(Fpm[:,2]==kp, Fpm[:,0]!=0),4] )), kperp))
    
    
    print reduce(sum, dat[:,1])/reduce(sum, norm[:])

    plt.plot(dat[:,0], dat[:,1]/Sc/norm, label=run)
    #plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$\langle s F^- \rangle/\langle F^- \rangle$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def ave_kz_vskp(Fpm,run=''):
    print "Plotting width in kpar vs kperp as defined by GWH for run = ", run
    kperp = np.unique(Fpm[:,2])
    dat = np.zeros((len(kperp),2))
    norm = np.zeros(len(kperp))
    Fkzn0 = Fpm[Fpm[:,1]!=0]
    Fkz0 = Fpm[Fpm[:,1]==0]
    dat[:,0] = kperp
    dat[:,1] = map(lambda kp:reduce(lambda x,y:x+y, (Fpm[Fpm[:,2]==kp,3] - Fpm[Fpm[:,2]==kp,4])*Fpm[Fpm[:,2]==kp,1]), kperp)

    norm[:] = np.array(map(lambda kp:reduce(lambda x,y:x+y, (Fkzn0[Fkzn0[:,2]==kp,3] + Fkzn0[Fkzn0[:,2]==kp,4])), kperp))
    norm[:] += np.array(map(lambda kp:reduce(lambda x,y:x+y, Fkz0[Fkz0[:,2]==kp,3]), kperp))

    plt.plot(dat[:,0], dat[:,1]/norm, label=run)
    #plt.ylim(ymin=1.e-8)
    plt.xlabel(r'$k_\perp$', fontsize=16)
    plt.ylabel(r'$\langle k_\parallel F \rangle$', fontsize=16)
    leg = plt.legend(loc=0, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    



if __name__ == "__main__":
    import sys
    # Load data from simulation

    #Fsat_vss(F), Fsat_vss(Fpm), Fsat_vss(Fm)
    #Fsat_vskp(F), Fsat_vskp(Fpm), Fsat_vskp(Fm), Fsat_vskp(Fgwh)
    #Fsat_vskz(F), Fsat_vskz(Fpm), Fsat_vskz(Fm), Fsat_vskz(Fgwh)
    


