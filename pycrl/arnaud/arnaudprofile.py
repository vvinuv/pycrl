import numpy as np
import CosmologyFunctions as cf
from numba import jit


# Converting fortran code by E.Komatsu to Python 

# A sample code for computing a GAS pressure profile, Pgas(x) [x=r/r500], and
# the corresponding SZ profile, Tsz(x) [x=theta/theta500], from the generalized 
# NFW model of Arnaud et al., A&A, 517, 92 (2010)

# Output file "x_pgas_tsz.txt" contains
# 1st: x
# 2nd: Pgas(x) in units of eV/cm^3, where x=r/r500
# 3rd: Tsz(x) in units of micro Kelvin [Rayleigh-Jeans], where x=theta/theta500
@jit(nopython=True)
def pgnfw(x): # x=r/r500
    c = 1.177
    g = 0.3081
    a = 1.0510
    b = 5.4905
    P0 = 8.403
    h0 = 0.702
    pgnfw = P0*(0.7/h0)**1.5/(c*x)**g/(1.0+(c*x)**a)**((b-g)/a)
    return pgnfw

@jit(nopython=True)
def integrate_pressure(los_array, x):
    # x=r/r500
    #los_array integrating the pressure profile along the line of sight radius
    c = 1.177
    g = 0.3081
    a = 1.0510
    b = 5.4905
    P0 = 8.403
    h0 = 0.702
    
    x = np.sqrt(los_array**2.0 + x**2.0)
    dx = (x[1] - x[0])
    return (P0*(0.7/h0)**1.5/(c*x)**g/(1.0+(c*x)**a)**((b-g)/a)).sum() * dx

def ArnaudProfile(z, m500, xin=8e-3, xout=6.0, om0=0.277, h0=0.702, 
                  xspace=1000, losspace=1000, filename='py_x_pgas_tsz.txt',
                  doprint=False):
    '''
    z : Redshift of cluster
    m500 : Mass at 500 times critical density of cluster
    xin, xout : r/r500 
    om0, h0 : Cosmology parameters
    '''
    cosmo = cf.CosmologyFunctions(z)
    rhoc = cosmo.rho_crit() # critical density in units of h^2 Msun Mpc^-3
    da = cosmo.angular_diameter_distance() # proper distance, h^-1 Mpc
    r500 = (3.0*m500/4.0/3.14159/500./rhoc)**(1.0/3.0) # h^-1 Mpc
    theta500 = r500/da # radians
    Ez = np.sqrt(om0*(1.0+z)**3.0+1.0-om0)

    if doprint:
        print 'Redshift > {:.2f}'.format(z)
        print 'Da(physical) > {:.2f}  h^-1 Mpc'.format(da)
        print 'M500 > {:.2e} h^-1 Msun'.format(m500)
        print 'R500 > {:.2f} h^-1 Mpc'.format(r500)
        print 'theta500 > {:.2f} arcmin'.format(theta500*10800.0/3.14159)
        print 'Pressure is cut at r/R500 > {d}'.format(xout)

    # calculate profiles
    y = np.linspace(xin, xout, xspace) # y=theta/theta500 or r/r500
    los_array = np.linspace(-xout, xout, losspace)
    # See, e.g., Appendix D1 of Komatsu et al., arXiv:1001.4538
    pgas3d = 1.65*(h0/0.7)**2.0*Ez**(8.0/3.0)*(m500/3e14/0.7)**(2.0/3.0+0.12)* pgnfw(y)/0.5176

    pgas2d = np.array([1.65*(h0/0.7)**2.0*Ez**(8.0/3.0)*(m500/3e14/0.7)**(2.0/3.0+0.12) * integrate_pressure(los_array, x) * (r500/h0)/0.5176 for x in y])

    Tsz=-2.0*283.0*(pgas2d/50.0) # uK
    np.savetxt(filename, np.transpose([y,pgas3d,Tsz]), fmt='%.5f %5e %.5e')

if __name__=='__main__':
    # pressure profile is cut at r=xout*r500
    xin = 8e-3
    xout = 6.0 
    om0 = 0.277
    h0 = 0.702
    # Sample parameters: Coma cluster
    z = 0.0231
    #h^-1 Msun, computed from M500-Tx relation of Vikhlinin et al with Tx=8.4keV
    m500 = 6.61e14 
    ArnaudProfile(z, m500)


