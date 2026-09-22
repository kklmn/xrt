# -*- coding: utf-8 -*-
"""Optical elements composed from independently defined surfaces."""

import inspect
import numpy as np

from ... import raycing
from .. import sources as rs
from .base import OE
from .dcm import DCM


def _member_property(name, member, original=None):
    """Expose a mirror attribute while retaining DCM initialization."""
    def getter(self):
        if not getattr(self, '_montel_ready', False):
            return original.fget(self) if original else self.__dict__.get(name)
        return getattr(getattr(self, member), name.rstrip('2'))

    def setter(self, value):
        if not getattr(self, '_montel_ready', False):
            if original:
                original.fset(self, value)
            else:
                self.__dict__[name] = value
            return
        setattr(getattr(self, member), name.rstrip('2'), value)

    return property(getter, setter)


class MontelMirror(DCM):
    """Two orthogonal mirrors that may be hit in either order.

    ``mirrorH`` and ``mirrorV`` supply the surfaces, limits and materials. Their
    positions and orientations are controlled by this compound element. The
    vertical mirror uses the orthogonal Montel orientation derived from the
    horizontal mirror's pitch, yaw and position roll.

    ``beamLocal1`` contains the first hit, expressed in ``mirrorH``'s frame;
    ``beamLocal2`` contains the last hit, expressed in ``mirrorV``'s frame.
    These frame names do not identify which surface a ray actually hit.
    """

    hiddenMethods = ['reflect', 'multiple_reflect', 'propagate_wave']
    hiddenParams = ['order', 'bragg', 'cryst1roll', 'cryst2roll',
                    'cryst2pitch', 'cryst2finePitch', 'cryst2perpTransl',
                    'cryst2longTransl', 'surface', 'fixedOffset',
                    'braggOffset']
    _is_montel_compound = True

    # These are the same attributes a DCM exposes to xrtGlow. The limits,
    # material and shape continue to be owned by the individual mirrors.
    material = _member_property('material', 'mirrorH', OE.material)
    material2 = _member_property('material2', 'mirrorV', DCM.material2)
    shape = _member_property('shape', 'mirrorH')
    isParametric = _member_property('isParametric', 'mirrorH')
    limPhysX = _member_property('limPhysX', 'mirrorH', OE.limPhysX)
    limPhysY = _member_property('limPhysY', 'mirrorH', OE.limPhysY)
    limOptX = _member_property('limOptX', 'mirrorH', OE.limOptX)
    limOptY = _member_property('limOptY', 'mirrorH', OE.limOptY)
    limPhysX2 = _member_property('limPhysX2', 'mirrorV', DCM.limPhysX2)
    limPhysY2 = _member_property('limPhysY2', 'mirrorV', DCM.limPhysY2)
    limOptX2 = _member_property('limOptX2', 'mirrorV', DCM.limOptX2)
    limOptY2 = _member_property('limOptY2', 'mirrorV', DCM.limOptY2)
    surfPhysX = _member_property('surfPhysX', 'mirrorH')
    surfPhysY = _member_property('surfPhysY', 'mirrorH')
    surfOptX = _member_property('surfOptX', 'mirrorH')
    surfOptY = _member_property('surfOptY', 'mirrorH')
    surfPhysX2 = _member_property('surfPhysX2', 'mirrorV')
    surfPhysY2 = _member_property('surfPhysY2', 'mirrorV')
    surfOptX2 = _member_property('surfOptX2', 'mirrorV')
    surfOptY2 = _member_property('surfOptY2', 'mirrorV')

    def __init__(self, bl=None, name='', center=(0, 0, 0),
                 mirrorH=None, mirrorV=None, **kwargs):
        self._montel_ready = False
        super(MontelMirror, self).__init__(bl=bl, name=name, center=center,
                                           **kwargs)
        # Detached defaults allow a newly added Qook element to work before
        # its two mirror references have been selected.
        self._defaultMirrorH = OE(name='{0} horizontal'.format(self.name))
        self._defaultMirrorV = OE(name='{0} vertical'.format(self.name))
        for mirror in (self._defaultMirrorH, self._defaultMirrorV):
            mirror.bl = bl
            mirror.lostNum = self.lostNum if bl is not None else -1
        self._mirrorH = None
        self._mirrorV = None
        self.mirrorH = mirrorH
        self.mirrorV = mirrorV
        # References can resolve after construction; geometry prepares poses.
        self._montel_ready = True

    def _resolve_mirror(self, value, default):
        if value is None or value == 'None':
            return default
        mirror = raycing.normalize_ref(value, self.bl, 'oe', target='object')
        if not isinstance(mirror, OE):
            return default  # A reference may be loaded before its mirror.
        if mirror is self or mirror.bl is not self.bl:
            raise ValueError('a Montel mirror must be a different OE on its beamline')
        return mirror

    @property
    def mirrorH(self):
        return self._resolve_mirror(self._mirrorH, self._defaultMirrorH)

    @mirrorH.setter
    def mirrorH(self, value):
        self._mirrorH = None if value is None or value == 'None' else value

    @property
    def mirrorV(self):
        return self._resolve_mirror(self._mirrorV, self._defaultMirrorV)

    @mirrorV.setter
    def mirrorV(self, value):
        self._mirrorV = None if value is None or value == 'None' else value

    def _assign_mirror_poses(self):
        """Keep each mirror's optical definition but replace its pose."""
        if self.mirrorH is self.mirrorV:
            raise ValueError('mirrorH and mirrorV must be different elements')
        pose = ('center', 'roll', 'yaw', 'positionRoll',
                'rotationSequence', 'extraPitch', 'extraRoll', 'extraYaw',
                'extraRotationSequence', 'pitch')
        for mirror in (self.mirrorH, self.mirrorV):
            for attr in pose:
                value = getattr(self, attr)
                setattr(mirror, attr, list(value) if attr == 'center' else value)
        self.mirrorV.positionRoll = self.mirrorH.positionRoll + np.pi / 2.
        self.mirrorV.rotationSequence = (
            self.mirrorH.rotationSequence[2:] +
            self.mirrorH.rotationSequence[:2])
        self.mirrorV.yaw = self.mirrorH.pitch
        # Set pitch last: parametric mirrors recalculate their surface here.
        self.mirrorV.pitch = -self.mirrorH.yaw

    def get_surface_limits(self):
        if not getattr(self, '_montel_ready', False):
            return DCM.get_surface_limits(self)
        self._assign_mirror_poses()
        for mirror, suffix in ((self.mirrorH, ''), (self.mirrorV, '2')):
            mirror.get_surface_limits()
            for axis in ('X', 'Y'):
                for kind in ('Phys', 'Opt'):
                    attr = 'surf{0}{1}'.format(kind, axis)
                    setattr(self, attr + suffix, getattr(mirror, attr))

    def local_to_global(self, lb, returnBeam=False, **kwargs):
        """Use the DCM surface selector for the corresponding child frame."""
        self._assign_mirror_poses()
        mirror = self.mirrorV if kwargs.get('is2ndXtal', False) else self.mirrorH
        return mirror.local_to_global(lb, returnBeam=returnBeam, **kwargs)

    def local_z1(self, x, y):
        return self.local_z(x, y)

    def local_n1(self, x, y):
        return self.local_n(x, y)

    def _surface_mirror(self, is2ndXtal=False):
        self._assign_mirror_poses()
        return self.mirrorV if is2ndXtal else self.mirrorH

    def local_z(self, x, y):
        return self._surface_mirror().local_z(x, y)

    def local_n(self, x, y):
        return self._surface_mirror().local_n(x, y)

    def xyz_to_param(self, x, y, z):
        return self._surface_mirror().xyz_to_param(x, y, z)

    def param_to_xyz(self, s, phi, r):
        return self._surface_mirror().param_to_xyz(s, phi, r)

    def local_r(self, s, phi):
        return self._surface_mirror().local_r(s, phi)

    def local_r1(self, s, phi):
        return self.local_r(s, phi)

    def local_r2(self, s, phi):
        return self._surface_mirror(is2ndXtal=True).local_r(s, phi)

    def local_z2(self, x, y):
        return self._surface_mirror(is2ndXtal=True).local_z(x, y)

    def local_n2(self, x, y):
        return self._surface_mirror(is2ndXtal=True).local_n(x, y)

    def _in_reference_frame(self, global_beam, is2ndXtal=False):
        """Represent global hit positions and directions in a mirror's frame."""
        reference = self.mirrorV if is2ndXtal else self.mirrorH
        local = rs.Beam(copyFrom=global_beam)
        raycing.global_to_virgin_local(
            self.bl, global_beam, local, reference.center)
        raycing.rotate_beam(
            local, rotationSequence=reference.rotationSequence,
            pitch=-reference.pitch,
            roll=-(reference.roll + reference.positionRoll),
            yaw=-reference.yaw)
        raycing.rotate_beam(
            local, rotationSequence=reference.extraRotationSequence,
            pitch=-reference.extraPitch, roll=-reference.extraRoll,
            yaw=-reference.extraYaw)
        dx = getattr(reference, 'dx', 0)
        if dx:
            local.x -= dx
        return local

    @raycing.append_to_flow_decorator
    def double_reflect(self, beam=None):
        """Trace both orders and merge the reflected rays.

        .. Returned values: beamGlobal, beamLocal1, beamLocal2
        """
        self._assign_mirror_poses()

        # The child reflections are internal operations, not separate flow
        # steps. Calling the undecorated method also avoids replacing their
        # standalone GUI output entries when the compound is propagated.
        previous = getattr(self.bl, '_suspend_flow_recording', False)
        self.bl._suspend_flow_recording = True
        try:
            v_global1, v_local1 = self.mirrorV.reflect.__wrapped__(
                self.mirrorV, beam)
            h_global2, h_local2 = self.mirrorH.reflect.__wrapped__(
                self.mirrorH, v_global1)
            h_global1, h_local1 = self.mirrorH.reflect.__wrapped__(
                self.mirrorH, beam)
            v_global2, v_local2 = self.mirrorV.reflect.__wrapped__(
                self.mirrorV, h_global1)
        finally:
            self.bl._suspend_flow_recording = previous

        v_first = v_local1.state == 1
        h_after_v = v_first & (h_local2.state == 1)
        h_first = h_local1.state == 1
        v_after_h = h_first & (v_local2.state == 1)
        count_vh = v_first.astype(int) + h_after_v.astype(int)
        count_hv = h_first.astype(int) + v_after_h.astype(int)

        # Match the Montel example: a successful H-first path takes precedence
        # where both trial orders found a reflection.
        h_path = count_hv > 0
        v_path = (count_vh > 0) & ~h_path
        counts = np.where(h_path, count_hv, count_vh)

        beam_global = rs.Beam(copyFrom=v_global2)
        beam_global.replace_by_index(v_path, h_global2)
        beam_global.nRefl = counts
        beam_global.state[counts > 0] = 1

        beam_local1 = rs.Beam(copyFrom=beam)
        beam_local2 = rs.Beam(copyFrom=beam)
        beam_local1.state[:] = 0
        beam_local2.state[:] = 0
        beam_local1.nRefl = np.zeros_like(counts)
        beam_local2.nRefl = np.zeros_like(counts)

        first_v = self._in_reference_frame(v_global1)
        first_h = self._in_reference_frame(h_global1)
        last_vh = self._in_reference_frame(h_global2, is2ndXtal=True)
        last_hv = self._in_reference_frame(v_global2, is2ndXtal=True)
        single_v = self._in_reference_frame(v_global1, is2ndXtal=True)
        single_h = self._in_reference_frame(h_global1, is2ndXtal=True)

        beam_local1.replace_by_index(v_path, first_v)
        beam_local1.replace_by_index(h_path, first_h)
        beam_local2.replace_by_index(v_path & h_after_v, last_vh)
        beam_local2.replace_by_index(v_path & ~h_after_v, single_v)
        beam_local2.replace_by_index(h_path & v_after_h, last_hv)
        beam_local2.replace_by_index(h_path & ~v_after_h, single_h)
        beam_local1.state[counts > 0] = 1
        beam_local2.state[counts > 0] = 1
        beam_local1.nRefl[counts > 0] = 1
        beam_local2.nRefl[:] = counts

        raycing.append_to_flow(
            self.double_reflect,
            [beam_global, beam_local1, beam_local2], inspect.currentframe())
        return beam_global, beam_local1, beam_local2
