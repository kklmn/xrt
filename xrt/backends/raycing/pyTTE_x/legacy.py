from __future__ import division, print_function
import sys

import numpy as np

from scipy.integrate import ode
from scipy.constants.codata import physical_constants

#import xraylib

def takagitaupin(scantype,scan,constant,polarization,crystal_str,hkl,asymmetry,thickness,displacement_jacobian = None,debyeWaller=1.0,min_int_step=1e-10):
    '''
    1D TT-solver function. Deprecated
    
    Input:
    scantype = 'energy' or 'angle'
    scan =  relative to the Bragg's energy in meV (energy scan) OR relative to the Bragg's angle in arcsec (angle scan). 
            scan is an numpy array containing these values OR an integer N generating automatic scan with N points. 
    constant = incidence angle respect to the diffracting planes in degrees (energy scan) OR photon energy in keV (angle scan) 
    polarization = 'sigma' or 'pi'
    crystal_str = supports crystals included in xraylib, e.g. 'Si', 'Ge', 'LiF'
    hkl = [h,k,l] (Miller indices)
    asymmetry = asymmetry angle (in deg, positive to clockwise direction)
    thickness = crystal thickness in microns
    displacement_jacobian = a function giving the jacobian of the displacement field as function of position (x,y). 
                            Note: y points upwards the crystal 
    debyeWaller = Debye-Waller factor
    min_int_step = minumum integration step
    '''

    if scantype == 'energy':
        is_escan = True
        scantype = 'energy'
    elif scantype == 'angle':
        is_escan = False
        scantype = 'angle'


    if scantype == 'energy':
        print('Computing elastic line for ' + str(hkl) + ' reflection of ' \
              + crystal_str + ' crystal in energy domain.' )
        is_escan = True
    elif scantype == 'angle':
        print('Computing elastic line for ' + str(hkl) + ' reflection of ' \
              + crystal_str + ' crystal in angle domain.' )
        is_escan = False

    #type conversions
    if type(scan) is int:
        print('AUTOMATIC LIMITS NOT IMPLEMENTED YET!')
        print('Function terminated.')
        return None

    scan=np.array(scan)
    
    #Unit conversions
    thickness = thickness*1e-6 #wafer thickness in meters

    #constants
    crystal = None # xraylib.Crystal_GetCrystal(crystal_str)

    hc = physical_constants['Planck constant in eV s'][0]*physical_constants['speed of light in vacuum'][0]*1e3 #in meV*m
    d = 0 # xraylib.Crystal_dSpacing(crystal,*hkl)*1e-10 #in m
    V = crystal['volume']*1e-30 # volume of unit cell in m^3
    r_e = physical_constants['classical electron radius'][0]
    h = 2*np.pi/d

    print('')
    print('Crystal     : ', crystal_str)
    print('Reflection  : ', hkl)
    print('d_hkl       : ', d, ' m')
    print('Cell volume : ', V, ' m^3')
    print('')


    #asymmetry angle
    phi=np.radians(asymmetry)

    #Setup scan variables and constants
    if is_escan:
        escan=scan

        th0=np.radians(constant)
        th=th0

        #Conversion of incident photon energy to wavelength
        E0 = hc/(2*d*np.sin(th)) #in meV

        wavelength = hc/(E0+escan) #in m
        k = 2*np.pi/wavelength #in 1/m

    else:
        E0 = constant*1e6 #in meV

        wavelength = hc/E0 #in m
        k = 2*np.pi/wavelength #in 1/m

        if not hc/(2*d*E0) > 1:
            th0 = np.arcsin(hc/(2*d*E0))
        else:
            print('Given energy below the backscattering energy!')
            print('Setting theta to 90 deg.')
            th0 = np.pi/2

        ascan = scan*np.pi/648000 #from arcsec to rad
        th = th0+ascan

    #Incidence and exit angles
    alpha0 = th+phi
    alphah = th-phi
    
    #Direction parameters
    gamma0 = np.ones(scan.shape)/np.sin(alpha0)
    gammah = np.ones(scan.shape)/np.sin(alphah)

    if np.mean(gammah) < 0:
        print('The direction of diffraction in to the crystal -> Laue case')
        geometry = 'laue'
    else:
        print('The direction of diffraction out of the crystal -> Bragg case')
        geometry = 'bragg'

    #Polarization
    if polarization == 'sigma':
        C = 1;
        print('Solving for sigma-polarization')
    else:
        C = np.cos(2*th);
        print('Solving for pi-polarization')

    print('Asymmetry angle : ', phi,' rad, ', np.degrees(phi), ' deg')
    print('Wavelength      : ', hc/E0*1e10, ' Angstrom ')
    print('Energy          : ', E0*1e-6, ' keV ')
    
    print('Bragg angle     : ', th0,' rad, ', np.degrees(th0), ' deg')
    print('Incidence angle : ', th0+phi,' rad ', np.degrees(th0+phi), ' deg')
    print('Exit angle      : ', th0-phi,' rad ', np.degrees(th0-phi), ' deg')
    print('')

    #Compute susceptibilities
    if is_escan:
        F0 = np.zeros(escan.shape,dtype=np.complex128)
        Fh = np.zeros(escan.shape,dtype=np.complex128)
        Fb = np.zeros(escan.shape,dtype=np.complex128)

        for ii in range(escan.size):    
            F0[ii] = 0 # xraylib.Crystal_F_H_StructureFactor(crystal, (E0+escan[ii])*1e-6, 0, 0, 0, 1.0, 1.0)
            Fh[ii] = 0 # xraylib.Crystal_F_H_StructureFactor(crystal, (E0+escan[ii])*1e-6, hkl[0], hkl[1], hkl[2], debyeWaller, 1.0)
            Fb[ii] = 0 # xraylib.Crystal_F_H_StructureFactor(crystal, (E0+escan[ii])*1e-6, -hkl[0], -hkl[1], -hkl[2], debyeWaller, 1.0)
    else:
        F0 = 0 # xraylib.Crystal_F_H_StructureFactor(crystal, E0*1e-6, 0, 0, 0, 1.0, 1.0)
        Fh = 0 # xraylib.Crystal_F_H_StructureFactor(crystal, E0*1e-6, hkl[0], hkl[1], hkl[2], debyeWaller, 1.0)
        Fb = 0 # xraylib.Crystal_F_H_StructureFactor(crystal, E0*1e-6, -hkl[0], -hkl[1], -hkl[2], debyeWaller, 1.0)

    cte = - r_e * wavelength*wavelength/(np.pi * V)
    chi0 = np.conj(cte*F0)
    chih = np.conj(cte*Fh)
    chib = np.conj(cte*Fb)

    print('F0   : ',np.mean(F0))
    print('Fh   : ',np.mean(Fh))
    print('Fb   : ',np.mean(Fb))
    print('')
    print('chi0 : ',np.mean(chi0))
    print('chih : ',np.mean(chih))
    print('chib : ',np.mean(chib))
    print('')
    print('(Mean F and chi values for energy scan)')
    print('')

    ######################
    #COEFFICIENTS FOR TTE#
    ######################

    #For solving ksi = Dh/D0 
    c0 = k*chi0/2*(gamma0+gammah)*np.ones(scan.shape)
    ch = k*C*chih*gammah/2*np.ones(scan.shape)
    cb = k*C*chib*gamma0/2*np.ones(scan.shape)

    #For solving Y = D0 
    g0 = k*chi0/2*gamma0*np.ones(scan.shape)
    gb = k*C*chib/2*gamma0*np.ones(scan.shape)

    #deviation from the kinematical Bragg condition for unstrained crystal
    beta = h*gammah*(np.sin(th)-wavelength/(2*d))

    #For deformation, the strain term function defined later stepwise 
    if displacement_jacobian == None:
        def strain_term(z): 
            return 0

    #INTEGRATION

    #Define ODEs and their Jacobians
    if geometry == 'bragg':
        print('Transmission in the Bragg case not implemented!')
        reflectivity = np.zeros(scan.shape)
        transmission = -np.ones(scan.shape)
    else:
        forward_diffraction = np.zeros(scan.shape)
        diffraction = np.zeros(scan.shape)

    #Solve the equation
    sys.stdout.write('Solving...0%')
    sys.stdout.flush()
    
    for step in range(len(scan)):
        #local variables for speedup
        c0_step   = c0[step]
        cb_step   = cb[step]
        ch_step   = ch[step]
        beta_step = beta[step]
        g0_step   = g0[step]
        gb_step   = gb[step]
        gammah_step = gammah[step]

        #Define deformation term for bent crystal
        if not displacement_jacobian == None:
            #Precomputed sines and cosines
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            if is_escan:
                cot_alpha0 = np.cos(alpha0)/np.sin(alpha0)
                sin_alphah = np.sin(alphah)
                cos_alphah = np.cos(alphah)

                def strain_term(z):
                    x = -z*cot_alpha0
                    u_jac = displacement_jacobian(x,z)
                    duh_dsh = h*(sin_phi*cos_alphah*u_jac[0][0] 
                                +sin_phi*sin_alphah*u_jac[0][1]
                                +cos_phi*cos_alphah*u_jac[1][0]
                                +cos_phi*sin_alphah*u_jac[1][1]
                                )
                    return gammah_step*duh_dsh
            else:
                cot_alpha0 = np.cos(alpha0[step])/np.sin(alpha0[step])
                sin_alphah = np.sin(alphah[step])
                cos_alphah = np.cos(alphah[step])

                def strain_term(z):
                    x = -z*cot_alpha0
                    u_jac = displacement_jacobian(x,z)
                    duh_dsh = h*(sin_phi*cos_alphah*u_jac[0][0]
                                +sin_phi*sin_alphah*u_jac[0][1]
                                +cos_phi*cos_alphah*u_jac[1][0] 
                                +cos_phi*sin_alphah*u_jac[1][1]
                                )
                    return gammah_step*duh_dsh
        
        if geometry == 'bragg':
            def ksiprime(z,ksi):
                return 1j*(cb_step*ksi*ksi+(c0_step+beta_step+strain_term(z))*ksi+ch_step)

            def ksiprime_jac(z,ksi):
                return 2j*cb_step*ksi+1j*(c0_step+beta_step+strain_term(z))

            r=ode(ksiprime,ksiprime_jac)
        else:
            def TTE(z,Y):
                return [1j*(cb_step*Y[0]*Y[0]+(c0_step+beta_step+strain_term(z))*Y[0]+ch_step),\
                        -1j*(g0_step+gb_step*Y[0])*Y[1]]

            def TTE_jac(z,Y):
                return [[2j*cb_step*Y[0]+1j*(c0_step+beta_step+strain_term(z)), 0],\
                        [-1j*gb_step*Y[1],-1j*(g0_step+gb_step*Y[0])]]

            r=ode(TTE,TTE_jac)

        r.set_integrator('zvode',method='bdf',with_jacobian=True, min_step=min_int_step,max_step=1e-4,nsteps=50000)

        if geometry == 'bragg':
            r.set_initial_value(0,-thickness)
            res=r.integrate(0)     
            reflectivity[step]=np.abs(res[0])**2*gamma0[step]/gammah[step] #gamma-part takes into account beam footprints 
        else:
            r.set_initial_value([0,1],0)
            res=r.integrate(-thickness)
            diffraction[step] = np.abs(res[0]*res[1])**2
            forward_diffraction[step] = np.abs(res[1])**2

        sys.stdout.write('\rSolving...%0.1f%%' % (100*(step+1)/len(scan),))  
        sys.stdout.flush()

    sys.stdout.write('\r\nDone.\n')
    sys.stdout.flush()

    if geometry == 'bragg':    
        return reflectivity, transmission
    else:    
        return diffraction, forward_diffraction

