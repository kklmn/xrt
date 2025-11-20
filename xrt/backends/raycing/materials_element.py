# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

from .. import raycing

tablesCaching = True

elementsList = (
    'none', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
    'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U')


def read_f0_all():
    f0data = {}
    dataDir = os.path.dirname(__file__)
    with open(os.path.join(dataDir, 'data', 'f0_xop.dat')) as f:
        field_symbol = field_data = None
        for li in f:
            if li.startswith("#S"):
                fields = li.split()
#                field_z = int(fields[1])
                field_symbol = str(fields[-1]).strip()

            if li.startswith("#UP"):
                if sys.version_info < (3, 1):
                    li = f.next()
                else:
                    li = next(f)
                field_data = [float(x) for x in li.split()]
            if field_data and field_symbol:
                f0data[field_symbol] = field_data
                field_symbol = field_data = None
    return f0data


def read_f1f2_all():
    f1f2tables = {}
    dataDir = os.path.dirname(__file__)
    for tableFName in ['Henke', 'Chantler', 'BrCo']:
        pname = os.path.join(dataDir, 'data', tableFName+'.npz')
        with open(pname, 'rb') as f:
            res = np.load(f)
            f1f2tables[tableFName] = {k: np.array(v) for k, v in res.items()}
    return f1f2tables


def read_atomic_data_all():
    atomicDataDict = {}
    dataDir = os.path.dirname(__file__)
    with open(os.path.join(dataDir, 'data', 'AtomicData.dat')) as f:
        for li in f:
            fields = li.split()
            Z = int(fields[0])
            if Z > 0:
                atomicData = [float(x) for x in fields]
                atomicDataDict[Z] = atomicData[3]
    return atomicDataDict


if tablesCaching:
    table_f0 = read_f0_all()
    tables_f1f2 = read_f1f2_all()
    table_atomicData = read_atomic_data_all()


def read_atomic_data(elem):
    u"""
    Reads atomic data from ``AtomicData.dat`` file adopted from XOP [XOP]_.
    It has the following data:
    0  AtomicRadius[Å]  CovalentRadius[Å]  AtomicMass  BoilingPoint[K]
    MeltingPoint[K]  Density[g/ccm]  AtomicVolume
    CoherentScatteringLength[1E-12cm]  IncoherentX-section[barn]
    Absorption@1.8Å[barn]  DebyeTemperature[K]  ThermalConductivity[W/cmK]

    In :meth:`read_atomic_data` only the mass is inquired. The user may
    extend the method to get the other values by simply adding the
    corresponding array elements to the returned value."""
    if isinstance(elem, raycing.basestring):
        Z = elementsList.index(elem)
    elif isinstance(elem, int):
        Z = elem
    else:
        raise NameError('Wrong element')

    if tablesCaching:
        atomicData = table_atomicData.get(Z)
        if atomicData is not None:
            return atomicData

    dataDir = os.path.dirname(__file__)
    with open(os.path.join(dataDir, 'data', 'AtomicData.dat')) as f:
        for li in f:
            fields = li.split()
            if int(fields[0]) == Z:
                atomicData = [float(x) for x in fields]
                break
    return atomicData[3]


class Element(object):
    """This class serves for accessing the scattering factors f0, f1 and f2 of
    a chemical element. It can also report other atomic data listed in
    ``AtomicData.dat`` file adopted from XOP [XOP]_.
    """

    def __init__(self, elem=None, table='Chantler'):
        u"""
        *elem*: str or int
            The element can be specified by its name (case sensitive) or its
            ordinal number.

        *table*: str
            This parameter is explained in the description of
            :class:`Material`.


           """

        self.table = table
        self.elem = elem  # For compatibility and dynamic update

    @property
    def elem(self):
        return self._elem

    @elem.setter
    def elem(self, elem):
        if isinstance(elem, raycing.basestring):
            self.name = elem
            self.Z = elementsList.index(elem)
        elif isinstance(elem, int):
            self.name = elementsList[elem]
            self.Z = elem
        else:
            raise NameError('Wrong chemical element')
        self._elem = elem
        self.f0coeffs = self.read_f0_Kissel()
        self.mass = read_atomic_data(self.Z)
        if hasattr(self, '_table'):
            self.E, self.f1, self.f2 = self.read_f1f2_vs_E(table=self.table)

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, table):
        self._table = table
        if hasattr(self, 'name'):
            self.E, self.f1, self.f2 = self.read_f1f2_vs_E(table=table)

    def read_f0_Kissel(self):
        r"""
        Reads f0 scattering factors from the tabulation of XOP [XOP]_. These
        were calculated by [Kissel]_ and then parameterized as [Waasmaier]_:

        .. math::

            f_0\left(\frac{q}{4\pi}\right) = c + \sum_{i=1}^5{a_i\exp\left(-b_i
            \left(q/(4\pi)\right)^2\right)}

        where :math:`q/(4\pi) = \sin{\theta} / \lambda` and :math:`a_i`,
        :math:`b_i` and :math:`c` are the coefficients tabulated in the file
        ``f0_xop.dat``.

        .. [Kissel] L. Kissel, Radiation physics and chemistry **59** (2000)
           185-200, http://www-phys.llnl.gov/Research/scattering/RTAB.html

        .. [Waasmaier] D. Waasmaier & A. Kirfel, Acta Cryst. **A51** (1995)
           416-413
        """
        if tablesCaching and self.Z < len(elementsList):
            f0data = table_f0.get(elementsList[self.Z])
            if f0data is not None:
                return f0data

        dataDir = os.path.dirname(__file__)
        with open(os.path.join(dataDir, 'data', 'f0_xop.dat')) as f:
            for li in f:
                if li.startswith("#S"):
                    fields = li.split()
                    if int(fields[1]) == self.Z:
                        break
            else:
                raise ValueError('cannot find the element {0}'.format(self.Z))
            for li in f:
                if li.startswith("#UP"):
                    if sys.version_info < (3, 1):
                        li = f.next()
                    else:
                        li = next(f)
                    break
            else:
                raise ValueError('wrong file format!')
        return [float(x) for x in li.split()]
#              = [a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5 ]

    def get_f0(self, qOver4pi=0):  # qOver4pi = sin(theta) / lambda
        """Calculates f0 for the given *qOver4pi*."""
        return self.f0coeffs[5] + sum(
            a * np.exp(-b * qOver4pi**2)
            for a, b in zip(self.f0coeffs[:5], self.f0coeffs[6:]))

    def read_f1f2_vs_E(self, table):
        """Reads f1 and f2 scattering factors from the given *table* at the
        instantiation time."""
        table_fn = table.split()[0]
        f2key = '_f2tot' if 'total' in table else '_f2'

        if tablesCaching:
            f1f2data = tables_f1f2.get(table_fn)
            if f1f2data is not None:
                ef1f2 = (np.array(f1f2data[self.name+'_E']),
                         np.array(f1f2data[self.name+'_f1']),
                         np.array(f1f2data[self.name+f2key]))
                return ef1f2

        dataDir = os.path.dirname(__file__)
        pname = os.path.join(dataDir, 'data', table_fn+'.npz')

        with open(pname, 'rb') as f:
            res = np.load(f)
            ef1f2 = (np.array(res[self.name+'_E']),
                     np.array(res[self.name+'_f1']),
                     np.array(res[self.name+f2key]))
        return ef1f2

#        pname = os.path.join(dataDir, 'data', table+'.Ef')
#        E, f1, f2 = [], [], []
#        startFound = False
#        with open(pname, "rb") as f:
#            while True:
#                structEf1f2 = f.read(12)
#                if not structEf1f2:
#                    break
#                ELoc, f1Loc, f2Loc = struct.unpack_from("<3f", structEf1f2)
#                if startFound and ELoc == -1:
#                    break
#                if ELoc == -1 and f2Loc == self.Z:
#                    startFound = True
#                    continue
#                if startFound:
#                    E.append(ELoc)
#                    f1.append(f1Loc - self.Z)
#                    f2.append(f2Loc)
#        return np.array(E), np.array(f1), np.array(f2)

    def get_f1f2(self, E):
        """Calculates (interpolates) f1 and f2 for the given array *E*."""
        if np.any(E < self.E[0]) or np.any(E > self.E[-1]):
            raise ValueError(
                ('E={0} is out of the data table range ' +
                 '[{1}, {2}]!!! Use another table.').format(
                    E if isinstance(E, (int, float)) else
                    E[np.where((E < self.E[0]) | (E > self.E[-1]))], self.E[0],
                    self.E[-1]))
        f1 = np.interp(E, self.E, self.f1)
        f2 = np.interp(E, self.E, self.f2)
        return f1 + 1j*f2
