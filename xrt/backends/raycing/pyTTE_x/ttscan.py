# -*- coding: utf-8 -*-

from __future__ import division, print_function
from .quantity import Quantity
from numpy import linspace

class TTscan:
    '''
    Class containing all the parameters for an energy or angle scan together 
    with the solver settings. An instance can be initialized either by giving 
    a path to a file defining the scan parameters, or passing them to the 
    function as keyword arguments. Keyword parameters are omitted if filepath 
    given.

    Parameters
    ----------
    
    filepath : str
        Path to the file with scan and solver parameters

    *OR*

    constant : Quantity of type energy or angle
        Determines value of the incident photon energy or the incidence angle 
        w.r.t to the Bragg plane fixed during the scan
    
    scan : Quantity of type energy or angle OR int
        Either a list of scan points wrapped in a Quantity e.g. 
        Quantity(np.linspace(-100,100,250),'meV') OR a non-negative integer 
        number of scan points for automatic scan range determination. The unit 
        of Quantity has to be angle if the unit of constant is energy and vice 
        versa.
        
    polarization : str
        'sigma' or 's' for sigma-polarized beam OR 'pi' or 'p' for pi-polarized
        beam
        
    (optional keywords)
        
    solver : str
        The solver used to integrate the 1D TT-equation. Currently only 
        'zvode_bdf' is supported.
        
    integration_step : Quantity of type length 
        Step size of the integrator. For 'zvode_bdf' this is the minimum step. 
        Default is Quantity(1e-10,'um').

    start_depth : Quantity of type length 
        An alternative starting point for the integration. Useful for thick 
        crystals in the Bragg case (not used in the Laue case). To make sense, 
        this should be between 0 and -thickness.

    output_type : str
        'intensity' if output reflectivity/diffractivity/transmission are given 
        in terms of wave intensities or 'photon_flux' in terms of photon fluxes. 
        Matters only for asymmetric cuts. Default is 'photon_flux'.

    Attributes
    ----------
    
    polarization : str
    
    constant : Quantity of type energy or angle

    scan : Quantity of type energy or angle (opposite of constant), or int
    
    solver : str
    
    integration_step : Quantity of type length
    
    start_depth : Quantity of type length
    
    output_type : Quantity of type length    
    '''
    
    def __init__(self, filepath = None, **kwargs):
        params = {}

        if filepath is not None:
            
            #####################################
            #Read crystal parameters from a file#
            #####################################

            #Overwrite possible kwargs 
            kwargs = {}
            
            with open(filepath,'r') as f:
               lines = f.readlines()

            #check and parse parameters
            for line in lines:
                line = line.strip()
                if len(line) > 0 and not line[0] == '#':  #skip empty and comment lines
                    ls = line.split() 
                    if ls[0] == 'constant' and len(ls) == 3:
                        kwargs['constant'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'scan':
                        if len(ls) == 5:
                            kwargs['scan'] = Quantity(linspace(float(ls[1]),float(ls[2]),int(ls[3])),ls[4])
                        elif len(ls) == 2:
                            kwargs['scan'] = int(ls[1])
                        else:
                            print('Skipped an invalid line in the file: ' + line)
                    elif ls[0] in ['polarization','solver','output_type'] and len(ls) == 2:
                        kwargs[ls[0]] = ls[1]
                    elif ls[0] in ['integration_step','start_depth'] and len(ls) == 3:
                        kwargs[ls[0]] = Quantity(float(ls[1]),ls[2])
                    else:
                        print('Skipped an invalid line in the file: ' + line)
 
        ###################################################
        #Check the presence of the required crystal inputs#
        ###################################################
        
        try:
            params['constant']     = kwargs['constant']
            params['scan']         = kwargs['scan']
            params['polarization'] = kwargs['polarization']
        except:
            raise KeyError('At least one of the required keywords constant, scan, or polarization is missing!')                

        #Optional parameters
        params['solver']           = kwargs.get('solver','zvode_bdf')
        params['output_type']           = kwargs.get('output_type','photon_flux')
        params['integration_step'] = kwargs.get('integration_step', Quantity(1e-10,'um'))
        params['start_depth']      = kwargs.get('start_depth', None)


        self.set_polarization(params['polarization'])
        self.set_scan(params['scan'], params['constant'])
        self.set_solver(params['solver'])
        self.set_output_type(params['output_type'])        
        self.set_integration_step(params['integration_step'])
        self.set_start_depth(params['start_depth'])

    def set_polarization(self,polarization):
        if type(polarization) == type('') and polarization.lower() in ['sigma','s']:
            self.polarization = 'sigma'
        elif type(polarization) == type('') and polarization.lower() in ['pi','p']:
            self.polarization = 'pi'
        else:
            raise ValueError("Invalid polarization! Choose either 'sigma' or 'pi'.")       

    def set_scan(self, scan, constant):        
        if isinstance(constant, Quantity) and constant.type() in ['angle', 'energy']:
            if constant.type() == 'angle':
                self.scantype = 'energy'
            else:
                self.scantype = 'angle'
            self.constant = constant.copy()
        else:
            raise ValueError('constant has to be an instance of Quantity of type energy or angle!')

        if isinstance(scan, Quantity) and scan.type() == self.scantype:
            self.scan = ('manual', scan.copy())
        elif type(scan) == type(1) and scan > 0:
            self.scan = ('automatic',scan)
        else:
            raise ValueError('scan has to be either a Quantity of type energy (for angle constant) or angle (for energy constant) or a non-negative integer!')

    def set_solver(self,solver):
        if type(solver) == type('') and solver.lower() in ['zvode_bdf']:
            self.solver = solver.lower()
        else:
            raise ValueError("Invalid solver! Currently only 'zvode_bdf' is supported.")    

    def set_output_type(self,output_type):
        if type(output_type) == type('') and output_type.lower() in ['intensity','photon_flux']:
            self.output_type = output_type.lower()
        else:
            raise ValueError("Invalid output_type! Has to be either 'intensity' or 'photon_flux'.")

    def set_integration_step(self, integration_step):
        if isinstance(integration_step, Quantity) and integration_step.type() == 'length':
            if not integration_step.value.size == 1:
                raise ValueError("Invalid integration step! Only single value is allowed.")
            self.integration_step = integration_step.copy()
        else:
            raise ValueError("Invalid integration step! Has to be an instance of Quantity of type length.")    

    def set_start_depth(self, start_depth):
        if isinstance(start_depth, Quantity) and start_depth.type() == 'length':
            if not start_depth.value.size == 1:
                raise ValueError("Invalid starting step! Only single value is allowed.")
            self.start_depth = start_depth.copy()
        elif start_depth == None:
            self.start_depth = None
        else:
            raise ValueError("Invalid starting depth! Has to be an instance of Quantity of type length or None.")    


    def __str__(self):
        
        if self.scan[0] == 'manual':
            N_points = self.scan[1].value.size
            limit_str = 'manual from ' + str(self.scan[1].value.min()) \
                        + ' to ' + str(self.scan[1].value.max()) + ' ' \
                        + self.scan[1].units()
        else:
            N_points = self.scan[1]
            limit_str = 'automatic'

        if self.output_type == 'intensity':
            output_type = 'wave intensity'
        else:
            output_type = 'photon flux'
            
        return 'Scan type     : ' + self.scantype + '\n' +\
               'Scan constant : ' + str(self.constant) +'\n' +\
               'Polarization  : ' + self.polarization  +'\n' +\
               'Scan points   : ' + str(N_points)  +'\n' +\
               'Scan range    : ' + limit_str   +'\n\n'+\
               'Output type                : ' + output_type +'\n' +\
               'Integrator                 : ' + self.solver + '\n'+\
               '(Minimum) integration step : ' + str(self.integration_step)+'\n'\
               'Alternative starting depth : ' + str(self.start_depth)+'\n'    


