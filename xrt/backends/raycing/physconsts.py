# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "07 Jan 2016"

PI = 3.1415926535897932384626433832795
PI2 = 6.283185307179586476925286766559
SQ3 = 1.7320508075688772935274463415059
SQ2 = 2**0.5
SQPI = PI**0.5

E0 = 4.803e-10  # [esu]
C = 2.99792458e10  # [cm/sec]
M0 = 9.10938291e-28  # [g]
M0C2 = 0.510998928  # MeV
HPLANCK = 6.626069573e-27  # [erg*sec]
EV2ERG = 1.602176565e-12  # Energy conversion from [eV] to [erg]
K2B = 2 * PI * M0 * C**2 * 0.001 / E0
SIE0 = 1.602176565e-19
SIHPLANCK = 6.626069573e-34
SIM0 = M0 * 1e-3
SIC = C * 1e-2
FINE_STR = 1 / 137.03599976
#E2W = PI2 * SIE0 / SIH  # w = E2W * E[eV]
E2W = 1519267514747457.9195337718065469
R0 = 2.817940285e-5  # A
AVOGADRO = 6.02214199e23  # atoms/mol
CHeVcm = HPLANCK * C / EV2ERG  # {c*h[eV*cm]}
CH = CHeVcm * 1e8  # {c*h[eV*A]}
CHBAR = CH / PI2  # {c*h/(2pi)[eV*A]}
