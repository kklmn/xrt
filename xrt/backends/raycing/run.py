# -*- coding: utf-8 -*-
def run_process(beamline, shineOnly1stSource=False):
    """Must be redefined by user. Must return *outDict* - a dictionary of
    {'Beam names': Beam instances}, where 'Beam names' are then used for
    instantiating the plots."""
    raise NotImplementedError  # abstract
