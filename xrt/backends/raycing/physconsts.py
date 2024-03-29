# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "07 Jan 2016"

PI = 3.1415926535897932384626433832795
PI2 = 6.283185307179586476925286766559
SQRT2PI = PI2**0.5  # =2.5066282746310002
SQ3 = 1.7320508075688772935274463415059
SQ2 = 2**0.5  # =1.4142135623730951
SQPI = PI**0.5  # =1.7724538509055159

SIE0 = 1.602176565e-19
#E0 = 4.803e-10  # [esu]
C = 2.99792458e10  # [cm/sec]
E0 = SIE0 * C / 10
M0 = 9.109383701528e-28  # [g]
SIM0 = 9.109383701528e-31
M0C2 = 0.510998928  # MeV
HPLANCK = 6.626069573e-27  # [erg*sec]
EV2ERG = 1.602176565e-12  # Energy conversion from [eV] to [erg]
K2B = 2 * PI * M0 * C**2 * 0.001 / E0  # =10.710201593926415

# EMC = SIE0 / SIM0 / C[mm]
EMC = 0.5866791802416487 
SIHPLANCK = 6.626069573e-34
#SIM0 = M0 * 1e-3
SIC = C * 1e-2
FINE_STR = 1 / 137.03599976
#E2W = PI2 * SIE0 / SIH  # w = E2W * E[eV]
E2W = 1519267514747457.9195337718065469
E2WC = 5067.7309392068091
R0 = 2.817940285e-5  # A
AVOGADRO = 6.02214199e23  # atoms/mol
CHeVcm = HPLANCK * C / EV2ERG  # {c*h[eV*cm]}  = 0.00012398419297617678
CH = CHeVcm * 1e8  # {c*h[eV*A]}  = 12398.419297617678
CHBAR = CH / PI2  # {c*h/(2pi)[eV*A]}  = 1973.2697177417986
