# -*- coding: utf-8 -*-

import unittest
import os
import sys

import matplotlib
matplotlib.use('Agg')

_XRT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
if _XRT_ROOT not in sys.path:
    sys.path.insert(0, _XRT_ROOT)  # analysis:ignore

from xrt import plotter as xrtp
from xrt.backends import raycing

from tests.raycing.auto._property_test_helpers import (
    assert_equivalent, assert_metadata_contract)


METADATA_CLASSES = (
    xrtp.XYCAxis,
    xrtp.XYCPlot,
)

AXIS_VALUES = {
    'label': 'energy',
    'unit': 'keV',
    'factor': None,
    'data': 'auto',
    'limits': [8., 9.],
    'offset': 8.5,
    'bins': 16,
    'ppb': 3,
    'density': 'kde',
    'invertAxis': True,
    'outline': 0.75,
    'fwhmFormatStr': '%.3f',
}

AXIS_EXPECTED = {
    'label': 'energy',
    'unit': 'keV',
    'factor': 1e-3,
    'factorInit': None,
    'data': raycing.get_energy,
    'dataInit': 'auto',
    'limits': [8., 9.],
    'limitsInit': [8., 9.],
    'offset': 8.5,
    'bins': 16,
    'ppb': 3,
    'pixels': 48,
    'density': 'kde',
    'invertAxis': True,
    'outline': 0.75,
    'fwhmFormatStr': '%.3f',
    'displayLabel': 'energy (keV)',
    'useCategory': False,
}

MICRO_SIGN = '\N{MICRO SIGN}'
GREEK_MU = '\N{GREEK SMALL LETTER MU}'

LENGTH_UNIT_FACTOR_CASES = (
    ('m', 1e-3),
    ('mm', 1.),
    ('um', 1e3),
    ('{0}m'.format(MICRO_SIGN), 1e3),
    ('{0}m'.format(GREEK_MU), 1e3),
    (r'$\mu$m', 1e3),
    ('nm', 1e6),
)

ANGLE_UNIT_FACTOR_CASES = (
    ('mrad', 1e3),
    ('urad', 1e6),
    ('{0}rad'.format(MICRO_SIGN), 1e6),
    ('{0}rad'.format(GREEK_MU), 1e6),
    (r'$\mu$rad', 1e6),
    ('deg', 57.29577951308232),
)

DATA_SETTER_CASES = (
    ('auto', raycing.get_energy, 'auto'),
    ('energy', raycing.get_energy, raycing.get_energy),
    ('x', raycing.get_x, raycing.get_x),
    ('not_a_beam_field', raycing.get_energy, 'auto'),
)

LIMITS_SETTER_CASES = (
    None,
    'symmetric',
    [-1., 1.],
)

OUTLINE_SETTER_CASES = (
    (-1., 0.),
    (0.25, 0.25),
    (2., 1.),
)

PLOT_VALUES = {
    'beam': 'beamScreen',
    'rayFlag': (1, 2),
    'aspect': 'auto',
    'xPos': 0,
    'yPos': 1,
    'ePos': 2,
    'title': 'plot title',
    'invertColorMap': True,
    'negative': True,
    'fluxKind': 'total',
    'fluxUnit': None,
    'fluxFormatStr': '%.2p',
    'contourLevels': [0.1, 0.5],
    'contourColors': ['red', 'blue'],
    'contourFmt': '%.2f',
    'contourFactor': 1000.,
    'saveName': ['plot.png', 'plot.pdf'],
    'persistentName': 'plot.pickle',
    'raycingParam': 2,
    'beamState': 'stateBeam',
    'beamC': 'colorBeam',
    'beamAbsorb': 'absorbedBeam',
    'showAbsorbed': True,
}

PLOT_STATE_ATTRS = tuple(PLOT_VALUES) + ('backend',)


def axis_state(axis):
    return {
        'label': axis.label,
        'unit': axis.unit,
        'factor': axis.factor,
        'factorInit': axis._factorInit,
        'data': axis.data,
        'dataInit': axis._dataInit,
        'limits': axis.limits,
        'limitsInit': axis._limitsInit,
        'offset': axis.offset,
        'bins': axis.bins,
        'ppb': axis.ppb,
        'pixels': axis.pixels,
        'density': axis.density,
        'invertAxis': axis.invertAxis,
        'outline': axis.outline,
        'fwhmFormatStr': axis.fwhmFormatStr,
        'displayLabel': axis.displayLabel,
        'useCategory': axis.useCategory,
    }


def expected_axis_state(label, unit, data, factor, limits, bins,
                        factor_init=None, fwhm_format='%.1f'):
    return {
        'label': label,
        'unit': unit,
        'factor': factor,
        'factorInit': factor_init,
        'data': data,
        'dataInit': 'auto',
        'limits': limits,
        'limitsInit': limits,
        'offset': 0,
        'bins': bins,
        'ppb': 2,
        'pixels': bins * 2,
        'density': 'histogram',
        'invertAxis': False,
        'outline': 0.5,
        'fwhmFormatStr': fwhm_format,
        'displayLabel': '{0} ({1})'.format(label, unit),
        'useCategory': False,
    }


def make_axis():
    return xrtp.XYCAxis(**AXIS_VALUES)


def apply_axis_values(axis, values):
    for attr in (
            'label', 'unit', 'factor', 'data', 'limits', 'offset', 'bins',
            'ppb', 'density', 'invertAxis', 'outline', 'fwhmFormatStr'):
        setattr(axis, attr, values[attr])


def make_x_axis():
    return xrtp.XYCAxis(
        label='x', unit='mm', factor=None, data='auto',
        limits=[-1., 1.], bins=4, ppb=2)


def make_y_axis():
    return xrtp.XYCAxis(
        label='z', unit='mm', factor=None, data='auto',
        limits=[-2., 2.], bins=5, ppb=2)


def make_c_axis():
    return xrtp.XYCAxis(
        label='energy', unit='keV', factor=None, data='auto',
        limits=[8., 9.], bins=6, ppb=2, fwhmFormatStr='%.4f')


def make_plot():
    return xrtp.XYCPlot(
        xaxis=make_x_axis(), yaxis=make_y_axis(), caxis=make_c_axis(),
        **PLOT_VALUES)


def plot_state(plot):
    return {attr: getattr(plot, attr) for attr in PLOT_STATE_ATTRS}


def expected_plot_state():
    expected = dict(PLOT_VALUES)
    expected['backend'] = 'raycing'
    return expected


def close_plot(plot):
    fig = getattr(plot, 'fig', None)
    if fig is not None:
        xrtp.plt.close(fig)


class PlotPropertyMetadataTest(unittest.TestCase):
    def test_get_params_contract(self):
        for cls in METADATA_CLASSES:
            with self.subTest(cls=cls.__name__):
                assert_metadata_contract(
                    self, cls, require_all_arguments=False)


class AxisPropertyRoundTripTest(unittest.TestCase):
    def test_axis_init_roundtrip(self):
        axis = make_axis()

        assert_equivalent(self, axis_state(axis), AXIS_EXPECTED)

    def test_axis_post_init_roundtrip(self):
        axis = xrtp.XYCAxis(label='x', unit='mm', data='auto')

        apply_axis_values(axis, AXIS_VALUES)

        assert_equivalent(self, axis_state(axis), AXIS_EXPECTED)

    def test_axis_explicit_data_name_uses_getter(self):
        axis = xrtp.XYCAxis(
            label='energy', unit='keV', factor=None, data='energy')

        self.assertIs(axis.data, raycing.get_energy)
        self.assertIs(axis._dataInit, raycing.get_energy)

    def test_axis_label_setter_updates_unit_factor_and_data(self):
        axis = xrtp.XYCAxis(label='x', unit='mm', data='auto')

        axis.label = 'energy'

        assert_equivalent(self, axis.label, 'energy')
        assert_equivalent(self, axis.unit, 'eV')
        assert_equivalent(self, axis.factor, 1.)
        self.assertIs(axis.data, raycing.get_energy)
        assert_equivalent(self, axis.displayLabel, 'energy (eV)')

    def test_axis_unit_setter_length_factor_catalog(self):
        for unit, expected_factor in LENGTH_UNIT_FACTOR_CASES:
            with self.subTest(unit=unit):
                axis = xrtp.XYCAxis(
                    label='x', unit='mm', factor=None, data='auto')

                axis.unit = unit

                assert_equivalent(self, axis.unit, unit)
                assert_equivalent(self, axis.factor, expected_factor)
                assert_equivalent(self, axis._factorInit, None)
                assert_equivalent(
                    self, axis.displayLabel, 'x ({0})'.format(unit))

    def test_axis_unit_setter_angle_factor_catalog(self):
        for unit, expected_factor in ANGLE_UNIT_FACTOR_CASES:
            with self.subTest(unit=unit):
                axis = xrtp.XYCAxis(
                    label="x'", unit='mrad', factor=None, data='auto')

                axis.unit = unit

                assert_equivalent(self, axis.unit, unit)
                assert_equivalent(self, axis.factor, expected_factor)
                assert_equivalent(self, axis._factorInit, None)
                assert_equivalent(
                    self, axis.displayLabel, "x' ({0})".format(unit))

    def test_axis_factor_setter_explicit_and_auto_paths(self):
        axis = xrtp.XYCAxis(
            label='energy', unit='eV', factor=2., data='auto')

        axis.unit = 'keV'
        assert_equivalent(self, axis.factor, 2.)
        assert_equivalent(self, axis._factorInit, 2.)

        axis.factor = None
        assert_equivalent(self, axis.factor, 1e-3)
        assert_equivalent(self, axis._factorInit, None)

    def test_axis_data_setter_catalog(self):
        for value, expected_data, expected_init in DATA_SETTER_CASES:
            with self.subTest(value=value):
                axis = xrtp.XYCAxis(
                    label='energy', unit='eV', factor=None, data='auto')

                axis.data = value

                self.assertIs(axis.data, expected_data)
                assert_equivalent(self, axis._dataInit, expected_init)

    def test_axis_limits_setter_catalog(self):
        axis = xrtp.XYCAxis(label='x', unit='mm', data='auto')

        for value in LIMITS_SETTER_CASES:
            with self.subTest(value=value):
                axis.limits = value

                assert_equivalent(self, axis.limits, value)
                assert_equivalent(self, axis._limitsInit, value)

    def test_axis_bins_and_ppb_setters_update_pixels(self):
        axis = xrtp.XYCAxis(
            label='x', unit='mm', data='auto', bins=5, ppb=2)

        for attr, value, expected_pixels in (
                ('bins', 7, 14),
                ('ppb', 4, 28),
                ('bins', 3, 12)):
            with self.subTest(attr=attr, value=value):
                setattr(axis, attr, value)

                assert_equivalent(self, getattr(axis, attr), value)
                assert_equivalent(self, axis.pixels, expected_pixels)

    def test_axis_outline_setter_catalog(self):
        axis = xrtp.XYCAxis(label='x', unit='mm', data='auto')

        for value, expected in OUTLINE_SETTER_CASES:
            with self.subTest(value=value):
                axis.outline = value

                assert_equivalent(self, axis.outline, expected)


class PlotPropertyRoundTripTest(unittest.TestCase):
    def test_plot_init_roundtrip(self):
        plot = make_plot()
        self.addCleanup(close_plot, plot)

        assert_equivalent(self, plot_state(plot), expected_plot_state())
        assert_equivalent(
            self, axis_state(plot.xaxis),
            expected_axis_state(
                'x', 'mm', raycing.get_x, 1., [-1., 1.], 4))
        assert_equivalent(
            self, axis_state(plot.yaxis),
            expected_axis_state(
                'z', 'mm', raycing.get_z, 1., [-2., 2.], 5))
        assert_equivalent(
            self, axis_state(plot.caxis),
            expected_axis_state(
                'energy', 'keV', raycing.get_energy, 1e-3, [8., 9.], 6,
                fwhm_format='%.4f'))
        assert_equivalent(self, plot.total2D.shape, (5, 4))
        assert_equivalent(self, plot.total2D_RGB.shape, (5, 4, 3))

    def test_plot_post_init_assignment_roundtrip(self):
        plot = xrtp.XYCPlot(
            beam='baseBeam', xaxis=make_x_axis(), yaxis=make_y_axis(),
            caxis=make_c_axis(), title='base plot')
        self.addCleanup(close_plot, plot)

        plot.xaxis = make_x_axis()
        plot.yaxis = make_y_axis()
        plot.caxis = make_c_axis()
        for attr, value in PLOT_VALUES.items():
            setattr(plot, attr, value)

        assert_equivalent(self, plot_state(plot), expected_plot_state())
        assert_equivalent(
            self, axis_state(plot.xaxis),
            expected_axis_state(
                'x', 'mm', raycing.get_x, 1., [-1., 1.], 4))
        assert_equivalent(
            self, axis_state(plot.yaxis),
            expected_axis_state(
                'z', 'mm', raycing.get_z, 1., [-2., 2.], 5))
        assert_equivalent(
            self, axis_state(plot.caxis),
            expected_axis_state(
                'energy', 'keV', raycing.get_energy, 1e-3, [8., 9.], 6,
                fwhm_format='%.4f'))

    def test_plot_category_caxis_init(self):
        plot = xrtp.XYCPlot(beam='beamScreen', caxis='category')
        self.addCleanup(close_plot, plot)

        self.assertEqual(plot.ePos, 0)
        self.assertTrue(plot.caxis.useCategory)
        assert_equivalent(self, plot.caxis.label, 'energy')
        assert_equivalent(self, plot.caxis.unit, 'eV')
        assert_equivalent(self, plot.caxis.factor, 1.)
        assert_equivalent(self, plot.caxis.fwhmFormatStr, '%.2f')

    def test_plot_serialization_roundtrip(self):
        plot = make_plot()
        self.addCleanup(close_plot, plot)

        serialized = xrtp.serialize_plots([plot])
        deserialized = xrtp.deserialize_plots({'Project': {'plots': serialized}})
        for deserialized_plot in deserialized:
            self.addCleanup(close_plot, deserialized_plot)

        self.assertEqual(len(deserialized), 1)
        roundtrip = deserialized[0]
        assert_equivalent(self, plot_state(roundtrip), expected_plot_state())
        assert_equivalent(
            self, axis_state(roundtrip.xaxis),
            expected_axis_state(
                'x', 'mm', raycing.get_x, 1., [-1., 1.], 4,
                factor_init=1.))
        assert_equivalent(
            self, axis_state(roundtrip.yaxis),
            expected_axis_state(
                'z', 'mm', raycing.get_z, 1., [-2., 2.], 5,
                factor_init=1.))
        assert_equivalent(
            self, axis_state(roundtrip.caxis),
            expected_axis_state(
                'energy', 'keV', raycing.get_energy, 1e-3, [8., 9.], 6,
                factor_init=1e-3, fwhm_format='%.4f'))


if __name__ == '__main__':
    unittest.main()
