# -*- coding: utf-8 -*-
from __future__ import division, print_function
from scipy.integrate import ode
from .ttcrystal import TTcrystal
from .quantity import Quantity
from .ttscan import TTscan
import matplotlib.pyplot as plt
import multiprocessing as multiprocess
import numpy as np
import time
#import xraylib
import sys
sys.path.append(r"../../../")
import xrt.backends.raycing as raycing
import xrt.backends.raycing.myopencl as mcl


class TakagiTaupin:

    '''
    The main class of the pyTTE package to calculate the 1D Takagi-Taupin curves
    of bent crystals.

    Parameters
    ----------
        TTcrystal_object : TTcrystal or str
            An existing TTcrystal instance for crystal parameters or a path to 
            a file where they are defined. For any other types the crystal 
            parameters are not set.

        TTscan_object : TTscan or str 
            An existing TTscan instance for scan parameters or a path to a file 
            where they are defined. For any other types the scan parameters are 
            not set.

    Attributes
    ----------
        crystal_object : TTcrystal
            Contains the crystal and deformation parameters

        scan_object : TTscan
            Contains the scan parameters and the solver settings

        solution : dict
            Stores the result of the TakagiTaupin.run(). Has the following keys:

                scan : Quantity of type energy or angle  
                    The scanned energy or angle points.

                beta : Numpy array
                    The scanned points given in terms of deviation from the
                    kinematical Bragg condition i.e. 
                        beta = h*(sin theta - lambda/2*d_h)

                bragg_energy : Quantity of type energy
                    Energy of incident photons at beta = 0
                
                bragg_angle : Quantity of type angle
                    Angle between the incident beam and the diffraction planes
                    at beta = 0
                
                geometry : str
                    Either 'bragg' for reflection geometry or 'laue' for 
                    transmission geometry
                
                reflectivity : Numpy array (present if geometry == 'bragg')
                    The reflectivity of the crystal for each scan point.

                transmission : Numpy array (present if geometry == 'bragg')
                    The transmission though the crystal for each scan point

                diffraction : Numpy array (present if geometry == 'laue')
                    The diffracted beam for each scan point

                forward_diffraction : Numpy array (present if geometry == 'laue')
                    The forward diffracted (transmitted beam) for each scan point.

                output_type : str
                    'intensity' if output reflectivity/diffractivity/transmission 
                    are given in terms of wave intensities or 'photon_flux' if 
                    in terms of photon fluxes. 

                crystal_parameters : str
                    String representation of self.crystal_object

                scan_parameters : str
                    String representation of self.scan_object
                    
                solver_output_log : str
                    Logged stdout output of self.run().

    '''    


    ##############################
    # Methods for initialization #
    ##############################    

    def __init__(self, TTcrystal_object = None, TTscan_object = None):

        self.crystal_object = None
        self.scan_object    = None
        self.solution       = None

        self.set_crystal(TTcrystal_object)
        self.set_scan(TTscan_object)


    def set_crystal(self, TTcrystal_object):
        '''
        Sets the crystal parameters for the scan instance. Any existing solutions
        are cleared.

        Parameters
        ----------
        
        TTcrystal_object : TTcrystal or str
            An existing TTcrystal instance for crystal parameters or a path to 
            a file where they are defined. For any other types the crystal 
            parameters are not set and the solution is not cleared.
        '''

        if isinstance(TTcrystal_object, TTcrystal): 
            self.crystal_object = TTcrystal_object
            self.solution       = None
        elif type(TTcrystal_object) == str:
            try:
                self.crystal_object = TTcrystal(TTcrystal_object)
                self.solution       = None
            except Exception as e:
                print(e)
                print('Error initializing TTcrystal from file! Crystal not set.')
        else:
            print('ERROR! Not an instance of TTcrystal or None! Crystal not set.')



    def set_scan(self, TTscan_object):
        '''
        Sets the crystal parameters for the scan instance. Any existing solutions
        are cleared.

        Parameters
        ----------
        
        TTscan_object : TTscan or str
            An existing TTscan instance for scan parameters or a path to a file 
            where they are defined. For any other types the scan parameters are 
            not set and the solution is not cleared.
        '''

        if isinstance(TTscan_object, TTscan):
            self.scan_object = TTscan_object
            self.solution    = None
        elif type(TTscan_object) == str:
            try:
                self.scan_object = TTscan(TTscan_object)
                self.solution    = None
            except Exception as e:
                print(e)
                print('Error initializing TTscan from file! Scan not set.')
        else:
            print('ERROR! Not an instance of TTscan or None! Scan not set.')


    ####################################
    # Auxiliary methods for TT solving #
    ####################################    

    @staticmethod
    def calculate_structure_factors(crystal, hkl, energy, debye_waller):
        '''
        Calculates the structure factors F_0 F_h and F_bar{h}.
        
        Parameters
        ----------
        
        crystal : dict
            Dictionary returned by xraylib's Crystal_GetCrystal()

        hkl : list of ints
            3 element list containing the Miller indeces of the reflection
            
        energy : Quantity instance of type energy.
            Energy of photons. energy.values be a single number or an array

        debye_waller : float
            The Debye-Waller factor
        '''

        energy_in_keV = energy.in_units('keV')
        
        #preserve the original shape and reshape energy to 1d array
        orig_shape = energy_in_keV.shape
        energy_in_keV = energy_in_keV.reshape(-1)

        F0 = np.zeros(energy_in_keV.shape, dtype=np.complex)
        Fh = np.zeros(energy_in_keV.shape, dtype=np.complex)
        Fb = np.zeros(energy_in_keV.shape, dtype=np.complex)        

#        for i in range(energy_in_keV.size):    
#            F0[i] = xraylib.Crystal_F_H_StructureFactor(crystal, energy_in_keV[i], 0, 0, 0, 1.0, 1.0)
#            Fh[i] = xraylib.Crystal_F_H_StructureFactor(crystal, energy_in_keV[i],  hkl[0],  hkl[1],  hkl[2], debye_waller, 1.0)
#            Fb[i] = xraylib.Crystal_F_H_StructureFactor(crystal, energy_in_keV[i], -hkl[0], -hkl[1], -hkl[2], debye_waller, 1.0)
        d = crystal.d
        F0, Fh, Fb  = crystal.get_F_chi(energy_in_keV*1000,
                                                    0.5/d)[:3]

        return F0.reshape(orig_shape), Fh.reshape(orig_shape), Fb.reshape(orig_shape)


    def run(self):
        '''
        Calculates the 1D Takagi-Taupin curve for given TTcrystal and TTscan
        given to the instance. Stores the result and metadata in self.solution.

        Returns
        -------

        #In the Bragg (reflection) geometry#

        scan.value : 1D Numpy array
                    
        reflectivity : 1D Numpy array
            
        transmission : 1D Numpy array
        
        
        #In the Laue (transmission) geometry#

        scan.value : 1D Numpy array
            
        diffraction : 1D Numpy array
        
        forward_diffraction : 1D Numpy array

        '''
        

        #Check that the required scan parameters are in place
        if self.crystal_object is None:
            print('ERROR! No crystal data found, TTcrystal object needed.')
            return
        if self.scan_object is None:
            print('ERROR! No scan data found, TTscan object needed.')
            return

        #define a function for logging the input
        def print_and_log(printable, log_string):
            '''
            Prints the input in stdout and appends it to log_string.
    
            Parameters
            ----------
            printable
                Object to be printed
            log_string
                String to append the stdout to.                
    
            Returns
            -------
            new_log_string
                String that was printed on the screen appended to log_string
    
            '''
            print_str = str(printable)
            print(print_str)
            return log_string + print_str + '\n'            

        output_log = ''
        verbose_output = True
                
        startmsg = '\nSolving the 1D Takagi-Taupin equation for '\
                 + self.crystal_object.crystal_data['name']\
                 + '(' + str(self.crystal_object.hkl[0]) + ','\
                 + str(self.crystal_object.hkl[1]) + ','\
                 + str(self.crystal_object.hkl[2]) + ')'\
                 + ' reflection'

        output_log = print_and_log(startmsg,output_log)
        output_log = print_and_log('-' * len(startmsg) + '\n',output_log)
        
        ################################################
        #Calculate the constants needed by TT-equations#
        ################################################

        #Introduction of shorthands
#        crystal   = self.crystal_object.crystal_data
        hkl       = self.crystal_object.hkl
        phi       = self.crystal_object.asymmetry
        thickness = self.crystal_object.thickness

        debye_waller          = self.crystal_object.debye_waller
        displacement_jacobian = self.crystal_object.displacement_jacobian
        djparams = None if self.crystal_object.Rx.value == float('inf') else self.crystal_object.djparams

        scan_type     = self.scan_object.scantype
        scan_constant = self.scan_object.constant
        scan_points   = self.scan_object.scan
        polarization  = self.scan_object.polarization

        hc  = Quantity(1.23984193,'eV um')                         #Planck's constant * speed of light
#        d   = Quantity(xraylib.Crystal_dSpacing(crystal,*hkl),'A') #spacing of Bragg planes
        d = Quantity(self.crystal_object.xrt_crystal.d,'A')
#        V   = Quantity(crystal['volume'],'A^3')                    #volume of unit cell
        V = Quantity(self.crystal_object.xrt_crystal.get_a()**3,'A^3')
#        print("d", d, "V", V)
        r_e = Quantity(2.81794033e-15,'m')                         #classical electron radius
        h   = 2*np.pi/d                                            #length of reciprocal lattice vector

        #Energies and angles corresponding to the constant parameter and its counterpart given by Bragg's law
        if scan_type == 'energy':
            theta_bragg = scan_constant
            energy_bragg = hc/(2*d*np.sin(theta_bragg.in_units('rad')))
        else:
            energy_bragg = scan_constant
            if not (hc/(2*d*energy_bragg)).in_units('1') > 1:
                theta_bragg = Quantity(np.arcsin((hc/(2*d*energy_bragg)).in_units('1')), 'rad')
            else:
                output_log = print_and_log('Given energy below the backscattering energy!',output_log)
                output_log = print_and_log('Setting theta to 90 deg.',output_log)
                
                theta_bragg = Quantity(90, 'deg')


        if scan_points[0] == 'automatic':
            
            ################################################
            #Estimate the optimal scan limits automatically#
            ################################################
            
            F0, Fh, Fb = TakagiTaupin.calculate_structure_factors(self.crystal_object.xrt_crystal, 
                                                                  hkl, 
                                                                  energy_bragg, 
                                                                  debye_waller)

            #conversion factor from crystal structure factor to susceptibility
            cte = -(r_e * (hc/energy_bragg)**2/(np.pi * V)).in_units('1') 

            chi0 = np.conj(cte*F0)
            chih = np.conj(cte*Fh)
            chib = np.conj(cte*Fb)

            gamma0 = 1/np.sin((theta_bragg+phi).in_units('rad'))
            gammah = 1/np.sin((theta_bragg-phi).in_units('rad'))

            #Find the (rough) maximum and minimum of the deformation term
            if displacement_jacobian is not None:
                z = np.linspace(0,-thickness.in_units('um'),1000)
                x = -z*np.cos((theta_bragg+phi).in_units('rad'))/np.sin((theta_bragg+phi).in_units('rad'))

                sin_phi = np.sin(phi.in_units('rad'))
                cos_phi = np.cos(phi.in_units('rad'))
                sin_alphah = np.sin((theta_bragg-phi).in_units('rad'))
                cos_alphah = np.cos((theta_bragg-phi).in_units('rad'))

                u_jac = displacement_jacobian(x[0],z[0])
                h_um = h.in_units('um^-1')

                def_min = h_um*(  sin_phi*cos_alphah*u_jac[0][0]
                                + sin_phi*sin_alphah*u_jac[0][1]
                                + cos_phi*cos_alphah*u_jac[1][0] 
                                + cos_phi*sin_alphah*u_jac[1][1])            
                def_max = def_min

                for i in range(1,z.size):
                    u_jac = displacement_jacobian(x[i],z[i])

                    deform = h_um*(  sin_phi*cos_alphah*u_jac[0][0]
                                   + sin_phi*sin_alphah*u_jac[0][1]
                                   + cos_phi*cos_alphah*u_jac[1][0] 
                                   + cos_phi*sin_alphah*u_jac[1][1])            
                    if deform < def_min:
                        def_min = deform
                    if deform > def_max:
                        def_max = deform
            else:
                def_min = 0.0
                def_max = 0.0

            #expand the range by the backscatter Darwin width
            if np.sin(2*theta_bragg.in_units('rad') > np.sqrt(2*np.sqrt(np.abs(chih*chib)))):
                darwin_width_term = 2*np.sqrt(np.abs(chih*chib))/np.sin(2*theta_bragg.in_units('rad'))*h*np.cos(theta_bragg.in_units('rad'))
            else:
                darwin_width_term = np.sqrt(2*np.sqrt(np.abs(chih*chib)))*h*np.cos(theta_bragg.in_units('rad'))

            k_bragg = 2*np.pi*energy_bragg/hc 
            b_const_term = -0.5*k_bragg*(1 + gamma0/gammah)*np.real(chi0)

            beta_min = b_const_term - Quantity(def_max,'um^-1') - 2*darwin_width_term
            beta_max = b_const_term - Quantity(def_min,'um^-1') + 2*darwin_width_term

            #convert beta limits to scan vectors
            if scan_type == 'energy':
                energy_min = beta_min*energy_bragg/(h*np.sin(theta_bragg.in_units('rad')))
                energy_max = beta_max*energy_bragg/(h*np.sin(theta_bragg.in_units('rad')))

                output_log = print_and_log('Using automatically determined scan limits:',output_log)
                output_log = print_and_log('E min: ' + str(energy_min.in_units('meV')) + ' meV',output_log)                
                output_log = print_and_log('E max: ' + str(energy_max.in_units('meV')) + ' meV\n',output_log)

                scan = Quantity(np.linspace(energy_min.in_units('meV'),energy_max.in_units('meV'),scan_points[1]),'meV')
                scan_steps = scan.value.size
                scan_shape = scan.value.shape
            else:
                sin_th_min = np.sin(theta_bragg.in_units('rad'))+(beta_min/h).in_units('1')
                sin_th_max = np.sin(theta_bragg.in_units('rad'))+(beta_max/h).in_units('1')

                theta_min = Quantity(np.arcsin(sin_th_min),'rad')-theta_bragg

                #This avoids taking arcsin of values over 1
                if sin_th_max > 1:
                    #Mirror the theta range w.t.r. 90 deg
                    theta_max = Quantity(np.pi-np.arcsin(sin_th_min),'rad')-theta_bragg
                else:
                    theta_max = Quantity(np.arcsin(sin_th_max),'rad')-theta_bragg

                output_log = print_and_log('Using automatically determined scan limits:',output_log)
                output_log = print_and_log('Theta min: ' + str(theta_min.in_units('urad')) + ' urad',output_log)                
                output_log = print_and_log('Theta max: ' + str(theta_max.in_units('urad')) + ' urad\n',output_log)

                scan = Quantity(np.linspace(theta_min.in_units('urad'),theta_max.in_units('urad'),scan_points[1]),'urad')
                scan_steps = scan.value.size
                scan_shape = scan.value.shape
        else:            
            #Use the scan limits given by the user            
            scan = scan_points[1]
            scan_steps = scan.value.size
            scan_shape = scan.value.shape

        #Convert relative scans to absolute energies or incidence angles
        if scan_type == 'energy':
            theta  = theta_bragg
            energy  = energy_bragg + scan
        else:
            energy = energy_bragg
            theta = theta_bragg + scan

        #Wavelength and wavenumber
        wavelength = hc/energy
        k = 2*np.pi/wavelength

        #Incidence and exit angles
        alpha0 = theta+phi
        alphah = theta-phi
        
        #Direction parameters
        gamma0 = np.ones(scan_shape)/np.sin(alpha0.in_units('rad'))
        gammah = np.ones(scan_shape)/np.sin(alphah.in_units('rad'))

        if np.mean(gammah) < 0:
            output_log = print_and_log('The direction of diffraction in to the crystal -> Laue case',output_log)
            geometry = 'laue'
        else:
            output_log = print_and_log('The direction of diffraction out of the crystal -> Bragg case',output_log)
            geometry = 'bragg'

        #Polarization factor
        if polarization == 'sigma':
            C = 1
            output_log = print_and_log('Solving for sigma-polarization',output_log)
        else:
            C = np.cos(2*theta.in_units('rad'))
            output_log = print_and_log('Solving for pi-polarization',output_log)
        if verbose_output:
            output_log = print_and_log('Asymmetry angle : ' + str(phi.in_units('deg')) + ' deg',output_log)
            output_log = print_and_log('Wavelength      : ' + str((hc/energy_bragg).in_units('nm')) + ' nm',output_log)
            output_log = print_and_log('Energy          : ' + str(energy_bragg.in_units('keV')) + ' keV',output_log)
            output_log = print_and_log('Bragg angle     : ' + str(theta_bragg.in_units('deg')) + ' deg',output_log)
            output_log = print_and_log('Incidence angle : ' + str((theta_bragg+phi).in_units('deg')) + ' deg',output_log)
            output_log = print_and_log('Exit angle      : ' + str((theta_bragg-phi).in_units('deg')) + ' deg\n',output_log)    

        #Compute susceptibilities       
        F0, Fh, Fb = TakagiTaupin.calculate_structure_factors(self.crystal_object.xrt_crystal, 
                                                              hkl, 
                                                              energy, 
                                                              debye_waller)

        #conversion factor from crystal structure factor to susceptibility
        cte = -(r_e * (hc/energy_bragg)**2/(np.pi * V)).in_units('1') 

        chi0 = np.conj(cte*F0)
        chih = np.conj(cte*Fh)
        chib = np.conj(cte*Fb)
        if verbose_output:       
            output_log = print_and_log('Structure factors',output_log) 
            output_log = print_and_log('F0   : '+ str(np.mean(F0)),output_log)         
            output_log = print_and_log('Fh   : '+ str(np.mean(Fh)),output_log)         
            output_log = print_and_log('Fb   : '+ str(np.mean(Fb))+'\n',output_log)         
            output_log = print_and_log('Susceptibilities',output_log) 
            output_log = print_and_log('chi0 : '+ str(np.mean(chi0)),output_log)         
            output_log = print_and_log('chih : '+ str(np.mean(chih)),output_log)         
            output_log = print_and_log('chib : '+ str(np.mean(chib))+'\n',output_log)         
            output_log = print_and_log('(Mean F and chi values for energy scan)\n',output_log)                 

        ######################
        #COEFFICIENTS FOR TTE#
        ######################

        #For solving ksi = Dh/D0 
        c0 = 0.5*k*chi0*(gamma0+gammah)*np.ones(scan_shape)
        ch = 0.5*k*C*chih*gammah*np.ones(scan_shape)
        cb = 0.5*k*C*chib*gamma0*np.ones(scan_shape)

        #For solving Y = D0 
        g0 = 0.5*k*chi0*gamma0*np.ones(scan_shape)
        gb = 0.5*k*C*chib*gamma0*np.ones(scan_shape)

        #Deviation from the kinematical Bragg condition for unstrained crystal
        #To avoid catastrophic cancellation, the terms in the subtraction are
        #explicitly casted to 64 bit floats.
        beta_term1 = np.sin(theta.in_units('rad')).astype(np.float64)
        beta_term2 = wavelength.in_units('nm').astype(np.float64)
        beta_term3 = (2*d.in_units('nm')).astype(np.float64)
        
        beta = h*gammah*(beta_term1 - beta_term2/beta_term3).astype(np.float)

        #############
        #INTEGRATION#
        #############

        #Define ODEs and their Jacobians
        if geometry == 'bragg':
            output_log = print_and_log('Transmission in the Bragg case not implemented!',output_log) 
            reflectivity = np.zeros(scan_shape)
            transmission = -np.ones(scan_shape)
        else:
            forward_diffraction = np.zeros(scan_shape)
            diffraction = np.zeros(scan_shape)

        #Fix the length scale to microns for solving
        c0   =   c0.in_units('um^-1'); ch = ch.in_units('um^-1'); cb = cb.in_units('um^-1')
        g0   =   g0.in_units('um^-1'); gb = gb.in_units('um^-1')
        beta = beta.in_units('um^-1'); h  =  h.in_units('um^-1')
        thickness = thickness.in_units('um')

        def integrate_single_scan_step(step):
            #local variables for speedup
            c0_step   = c0[step]
            cb_step   = cb[step]
            ch_step   = ch[step]
            beta_step = beta[step]
            g0_step   = g0[step]
            gb_step   = gb[step]
            gammah_step = gammah[step]

            if djparams is not None:
                coef1 = djparams[0]
                coef2 = djparams[1]
                invR1 = djparams[2]

            #Define deformation term for bent crystal
            if displacement_jacobian is not None:
                #Precomputed sines and cosines
                sin_phi = np.sin(phi.in_units('rad'))
                cos_phi = np.cos(phi.in_units('rad'))

                if scan_type == 'energy':
                    cot_alpha0 = np.cos(alpha0.in_units('rad'))/np.sin(alpha0.in_units('rad'))
                    sin_alphah = np.sin(alphah.in_units('rad'))
                    cos_alphah = np.cos(alphah.in_units('rad'))
                else:
                    cot_alpha0 = np.cos(alpha0.in_units('rad')[step])/np.sin(alpha0.in_units('rad')[step])
                    sin_alphah = np.sin(alphah.in_units('rad')[step])
                    cos_alphah = np.cos(alphah.in_units('rad')[step])

                def strain_term(z):
                    x = -z*cot_alpha0
                    duh_dsh = h*(sin_phi*cos_alphah*(-invR1)*(z+0.5*thickness)
                                +sin_phi*sin_alphah*(-invR1*x + coef2*(z+0.5*thickness))
                                +cos_phi*cos_alphah*invR1*x 
                                +cos_phi*sin_alphah*coef1*(z+0.5*thickness)
                                )
                    return gammah_step*duh_dsh

                def strain_term_remote(z):
                    x = -z*cot_alpha0
                    u_jac = displacement_jacobian(x,z)
                    duh_dsh = h*(sin_phi*cos_alphah*u_jac[0][0]
                                +sin_phi*sin_alphah*u_jac[0][1]
                                +cos_phi*cos_alphah*u_jac[1][0] 
                                +cos_phi*sin_alphah*u_jac[1][1]
                                )
                    return gammah_step*duh_dsh
            else:
                #Non-bent crystal 
                def strain_term(z): 
                    return 0
            
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

            r.set_integrator('zvode',method='bdf',with_jacobian=True, 
                             min_step=self.scan_object.integration_step.in_units('um'),
                             max_step=thickness,nsteps=2500000)
        
            #Update the solving process
            lock.acquire()
            steps_calculated.value = steps_calculated.value + 1
            if verbose_output:
                sys.stdout.write('\rSolving...%0.2f%%' % (100*(steps_calculated.value)/scan_steps,))  
            sys.stdout.flush()
            lock.release()            

            if geometry == 'bragg':
                if self.scan_object.start_depth is not None:
                    start_depth = self.scan_object.start_depth.in_units('um')
                    if start_depth > 0 or start_depth < -thickness:
                        output_log = print_and_log('Warning! The given starting depth ' + str(start_depth)\
                                                   + 'um is outside the crystal!', output_log) 
                    r.set_initial_value(0,start_depth)
                else:
                    r.set_initial_value(0,-thickness)

                res=r.integrate(0)
                
                if self.scan_object.output_type == 'photon_flux':
                    reflectivity = np.abs(res[0])**2*gamma0[step]/gammah[step]
                else:
                    reflectivity = np.abs(res[0])**2

                transmission = -1 #Not implemented yet
                    
                return reflectivity, transmission, np.complex(res)*np.sqrt(np.abs(gamma0[step]/np.abs(gammah[step])))
            else:
                if self.scan_object.start_depth is not None:
                    output_log = print_and_log('Warning! The alternative start depth is negleted in the Laue case.', output_log) 
                r.set_initial_value([0,1],0)
                res=r.integrate(-thickness)
                if self.scan_object.output_type == 'photon_flux':
                    diffraction = np.abs(res[0]*res[1])**2*gamma0[step]/np.abs(gammah[step])
                else:                    
                    diffraction = np.abs(res[0]*res[1])**2
                    
                forward_diffraction = np.abs(res[1])**2

                return diffraction, forward_diffraction, np.complex(res[0]*res[1]*np.sqrt(np.abs(gamma0[step]/np.abs(gammah[step]))))

        n_cores = multiprocess.cpu_count()

        output_log = print_and_log('\nCalculating the TT-curve using ' + str(n_cores) + ' cores.', output_log)     

        #Solve the equation
#        sys.stdout.write('Solving...0%')
        sys.stdout.flush()
        t0 = time.time()
        def mp_init(l,v):
            global lock
            global steps_calculated
            lock = l
            steps_calculated = v

        pool = multiprocess.Pool(n_cores,initializer=mp_init, initargs=(multiprocess.Lock(), multiprocess.Value('i', 0)))
        output = np.array(pool.map(integrate_single_scan_step,range(scan_steps)))
        pool.close()
        pool.join()

        sys.stdout.write('\r\nDone.\n')
        sys.stdout.write('\r\nDone. Calculation time is {}s\n'.format(time.time()-t0))
        sys.stdout.flush()

        if geometry == 'bragg':    
            reflectivity = output[:,0]
            transmission = output[:,1]
            complex_amps = np.squeeze(np.array(output[:, 2]))
            self.solution = {'scan' : scan,
                             'beta' : beta,
                             'bragg_energy' : energy_bragg,
                             'bragg_angle'  : theta_bragg,                             
                             'geometry' : 'bragg', 
                             'reflectivity' : reflectivity, 
                             'transmission': transmission,
                             'output_type' : self.scan_object.output_type,
                             'crystal_parameters' : str(self.crystal_object),
                             'scan_parameters' : str(self.scan_object),
                             'solver_output_log' : output_log
                            }

            return scan.value, reflectivity, transmission, complex_amps
        else:    
            diffraction = output[:,0]
            forward_diffraction = output[:,1]      
            complex_amps = np.squeeze(np.array(output[:, 2]))
            self.solution = {'scan' : scan, 
                             'beta' : beta,
                             'bragg_energy' : energy_bragg,
                             'bragg_angle'  : theta_bragg,
                             'geometry' : 'laue', 
                             'diffraction' : diffraction, 
                             'forward_diffraction': forward_diffraction,
                             'output_type' : self.scan_object.output_type,
                             'crystal_parameters' : str(self.crystal_object),
                             'scan_parameters' : str(self.scan_object),
                             'solver_output_log' : output_log
                            }

            return scan.value, diffraction, forward_diffraction, complex_amps

    def plot(self):
        '''
        Plots the calculated solution
        '''

        if self.solution is None:
            print('No calculated Takagi-Taupin curves found! Call run() first!')
            return

        if self.solution['geometry'] == 'bragg':
            plt.plot(self.solution['scan'].value, self.solution['reflectivity'])
            if Quantity._type2str(self.solution['scan'].unit) == 'energy':
                plt.xlabel('Energy (' + self.solution['scan'].units() + ')')
            else:
                plt.xlabel('Angle (' + self.solution['scan'].units() + ')')

            if self.solution['output_type'] == 'photon_flux':
                plt.ylabel('Reflectivity in terms of photon flux')
            else:
                plt.ylabel('Reflectivity in terms of wave intensity')                

            plt.title('Takagi-Taupin solution in the Bragg case')
        else:
            plt.plot(self.solution['scan'].value, self.solution['forward_diffraction'],label = 'Forward-diffraction')
            plt.plot(self.solution['scan'].value, self.solution['diffraction'],label = 'Diffraction')

            if Quantity._type2str(self.solution['scan'].unit) == 'energy':
                plt.xlabel('Energy (' + self.solution['scan'].units() + ')')
            else:
                plt.xlabel('Angle (' + self.solution['scan'].units() + ')')

            if self.solution['output_type'] == 'photon_flux':
                plt.ylabel('Photon flux w.r.t incident')
            else:
                plt.ylabel('Intensity w.r.t incident')         

            plt.legend()
            plt.title('Takagi-Taupin solution in the Laue case')
            
        plt.show()

    def save(self, path, include_header = True):
        '''
        Saves the solution to a file.
        
        Parameters
        ----------
        
        path : str
            Path of the save file
            
        include_header : bool, optional
            Determines if the metadata is included in the header. The default 
            is True.

        Returns
        -------
        
        None.

        '''

        if self.solution is None:
            print('No calculated Takagi-Taupin curves found! Call run() first!')
            return


        #Build the data matrix
        data = []
        data.append(self.solution['scan'].value.reshape(-1))
        
        if self.solution['geometry'] == 'bragg':
            data.append(self.solution['reflectivity'].reshape(-1))
        else:
            data.append(self.solution['diffraction'].reshape(-1))
            data.append(self.solution['forward_diffraction'].reshape(-1))

        data = np.array(data).T

        if include_header:
            header = str(self) + '\n' + 'SOLVER OUTPUT LOG\n'\
                     + '-----------------\n' + self.solution['solver_output_log'] + '\n'

            header = header + 'SOLUTION\n' + '--------\n' 
            header = header + 'Scan (' + self.solution['scan'].units() + ') '

            if self.solution['geometry'] == 'bragg':
                header =  header + 'Reflectivity'                
            else:
                header =  header + 'Diffraction Forward-diffraction'                
        else:
            header = ''

        np.savetxt(path, data, header=header)

    def __str__(self):
        return   'CRYSTAL PARAMETERS\n'\
               + '------------------\n\n'\
               + str(self.crystal_object) + '\n\n'\
               + 'SCAN PARAMETERS\n'\
               + '---------------\n\n'\
               + str(self.scan_object)
