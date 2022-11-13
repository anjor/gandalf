void nonlin( cuComplex *zpOld, cuComplex *zmOld, cuComplex *bracket1, cuComplex *bracket2, cuComplex* zK)
{
    multKPerp<<<dimGrid,dimBlock>>> (zK,zmOld, 1);
    //1) zK = -kPerp2*zmOld
    
    NLPS(bracket1,zpOld,zK);
    //2) bracket1 = {zp,-kperp2*zm}
    
    multKPerp<<<dimGrid,dimBlock>>> (zK,zpOld,1);
    //3) zK = -kPerp2*zpOld
    
    NLPS(bracket2,zmOld,zK);
    //4) bracket2 = {zm,-kPerp2*zp}
    
    addsubt<<<dimGrid,dimBlock>>> (bracket1, bracket1, bracket2, 1);  //result put in bracket1
    //5) bracket1 = {zp,-kPerp2*zm}+{zm,-kPerp2*zp}
    
    NLPS(bracket2,zpOld,zmOld);
    //6) bracket2 = {zp,zm}
    
    multKPerp<<<dimGrid,dimBlock>>> (bracket2, bracket2,1);
    //7) bracket2 = -kPerp2*[{zp,zm}]

    //bracket1 and bracket2 are same for zeta+ and zeta-, only difference is whether they are added
    //or subtracted
}

