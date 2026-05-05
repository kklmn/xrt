# -*- coding: utf-8 -*-
"""Describe the EPICS records exposed by raycing beamline elements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .device import to_valid_var_name
from .._named_arrays import Center, Limits
from .._sets_units import orientationArgSet, shapeArgSet
from ..physconsts import CH


@dataclass(frozen=True)
class PvSpec:
    """One Phoebus/EPICS-visible process variable."""

    record: str
    label: str
    property_path: str
    kind: str
    access: str = "rw"
    initial_value: Any = None
    group: str = "Properties"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ElementSpec:
    """A beamline element and the records generated for it."""

    uuid: str
    name: str
    valid_name: str
    class_name: str
    category: str
    pvs: List[PvSpec]


def prefixed_pv_name(record: str, prefix_macro: str = "$(P)") -> str:
    """Return a Phoebus PV expression for an EpicsDevice record suffix."""

    return f"{prefix_macro}{record}"


def workflow_records(
        epics_map: Optional[Dict[str, Optional[str]]] = None) -> List[PvSpec]:
    """Return the beamline workflow records always created by EpicsDevice."""

    _ = epics_map  # EpicsDevice does not apply epicsMap to workflow records.
    return [
        PvSpec("Acquire", "Acquire", "Acquire", "bool_command", "rw",
               0, "Propagation"),
        PvSpec("AcquireStatus", "Acquire status", "AcquireStatus",
               "bool_status", "ro", 0, "Propagation"),
        PvSpec("AutoUpdate", "Auto update", "AutoUpdate", "bool_toggle",
               "rw", 1, "Propagation"),
    ]


def element_records(oe_obj: Any,
                    epics_map: Optional[Dict[str, Optional[str]]] = None
                    ) -> List[PvSpec]:
    """Return the records EpicsDevice would expose for *oe_obj*.

    The logic intentionally mirrors ``EpicsDevice.__init__`` without importing
    ``softioc`` or creating an IOC.
    """

    oename = to_valid_var_name(getattr(oe_obj, "name", "unnamed"))
    pvs: List[PvSpec] = []

    def add(default_record: str, label: str, property_path: str, kind: str,
            access: str = "rw", initial_value: Any = None,
            group: str = "Properties",
            metadata: Optional[Dict[str, Any]] = None) -> None:
        if not _include_record(default_record, epics_map):
            return
        mapped = _map_record(default_record, epics_map)
        pvs.append(PvSpec(mapped, label, property_path, kind, access,
                          initial_value, group, metadata or {}))

    _add_energy_record(oe_obj, oename, add)
    _add_image_records(oe_obj, oename, add)

    pv_fields = {"name"} | orientationArgSet | shapeArgSet
    if hasattr(oe_obj, "shine") and hasattr(oe_obj, "nrays"):
        pv_fields.add("nrays")
    for arg_name in _ordered_fields(pv_fields):
        if arg_name in ["shape", "renderStyle"]:
            continue
        if not hasattr(oe_obj, arg_name):
            continue

        if arg_name in ["name", "rotationSequence"]:
            add(f"{oename}:{arg_name}", _label(arg_name), arg_name, "string",
                "rw", str(getattr(oe_obj, arg_name)), _group(arg_name))
        elif arg_name == "center":
            center_obj = getattr(oe_obj, arg_name)
            if isinstance(center_obj, (list, tuple)) and len(center_obj) == 3:
                center_obj = Center(center_obj)
            for field_name in ["x", "y", "z"]:
                value = getattr(center_obj, field_name, None)
                add(f"{oename}:{arg_name}:{field_name}",
                    f"{_label(arg_name)} {field_name}",
                    f"{arg_name}.{field_name}", "number", "rw", value,
                    _group(arg_name))
        elif arg_name in ["limPhysX", "limPhysY", "limPhysX2", "limPhysY2"]:
            lim_obj = getattr(oe_obj, arg_name)
            if isinstance(lim_obj, Limits):
                for index, field_name in enumerate(["lmin", "lmax"]):
                    add(f"{oename}:{arg_name}:{field_name}",
                        f"{_label(arg_name)} {field_name}",
                        f"{arg_name}.{field_name}", "number", "rw",
                        lim_obj[index], _group(arg_name))
        elif arg_name == "blades":
            blades_obj = getattr(oe_obj, "blades")
            if isinstance(blades_obj, dict):
                for field_name, value in blades_obj.items():
                    add(f"{oename}:blades:{field_name}",
                        f"Blade {field_name}", f"blades.{field_name}",
                        "number", "rw", value, _group(arg_name))
        else:
            initial_value = getattr(oe_obj, arg_name)
            if isinstance(initial_value, (int, float, np.number)):
                add(f"{oename}:{arg_name}", _label(arg_name), arg_name,
                    "number", "rw", initial_value, _group(arg_name))

    return pvs


def iter_element_specs(bl: Any,
                       epics_map: Optional[Dict[str, Optional[str]]] = None
                       ) -> Iterable[ElementSpec]:
    """Yield EPICS element specs for a raycing BeamLine."""

    elements = bl.iter_oes_ordered() if hasattr(bl, "iter_oes_ordered") else\
        getattr(bl, "oesDict", {}).items()
    for oeid, oeline in elements:
        oe_obj = oeline[0]
        name = getattr(oe_obj, "name", oeid)
        yield ElementSpec(
            uuid=oeid,
            name=name,
            valid_name=to_valid_var_name(name),
            class_name=f"{oe_obj.__class__.__module__}."
                       f"{oe_obj.__class__.__name__}",
            category=categorize_element(bl, oe_obj, oeline),
            pvs=element_records(oe_obj, epics_map))


def categorize_element(bl: Any, oe_obj: Any, oeline: Any = None) -> str:
    """Return the BOB output folder for *oe_obj*."""

    if oe_obj in getattr(bl, "sources", []):
        return "sources"
    if oe_obj in getattr(bl, "slits", []):
        return "apertures"
    if oe_obj in getattr(bl, "screens", []):
        return "screens"
    if oe_obj in getattr(bl, "oes", []):
        return "oes"
    if oeline and len(oeline) > 1 and oeline[1] == 0:
        return "sources"

    module = getattr(oe_obj.__class__, "__module__", "")
    if ".apertures" in module:
        return "apertures"
    if ".screens" in module:
        return "screens"
    if ".sources" in module:
        return "sources"
    return "oes"


def template_records(category: str) -> List[PvSpec]:
    """Return generic record templates for an element category."""

    common = [
        PvSpec("$(E):name", "Name", "name", "string", "rw",
               "$(E)", "Identity"),
        PvSpec("$(E):center:x", "Center x", "center.x", "number", "rw",
               0, "Position"),
        PvSpec("$(E):center:y", "Center y", "center.y", "number", "rw",
               0, "Position"),
        PvSpec("$(E):center:z", "Center z", "center.z", "number", "rw",
               0, "Position"),
    ]
    orient = [
        PvSpec("$(E):pitch", "Pitch", "pitch", "number", "rw",
               0, "Orientation"),
        PvSpec("$(E):roll", "Roll", "roll", "number", "rw",
               0, "Orientation"),
        PvSpec("$(E):yaw", "Yaw", "yaw", "number", "rw",
               0, "Orientation"),
    ]
    limits = [
        PvSpec("$(E):limPhysX:lmin", "limPhysX lmin", "limPhysX.lmin",
               "number", "rw", -10, "Limits"),
        PvSpec("$(E):limPhysX:lmax", "limPhysX lmax", "limPhysX.lmax",
               "number", "rw", 10, "Limits"),
        PvSpec("$(E):limPhysY:lmin", "limPhysY lmin", "limPhysY.lmin",
               "number", "rw", -10, "Limits"),
        PvSpec("$(E):limPhysY:lmax", "limPhysY lmax", "limPhysY.lmax",
               "number", "rw", 10, "Limits"),
    ]

    if category == "sources":
        return common + [
            PvSpec("$(E):nrays", "Number of rays", "nrays", "number",
                   "rw", 100000, "Source"),
        ] + orient
    if category == "screens":
        return common + [
            PvSpec("$(E):image", "Image", "image", "image", "ro",
                   None, "Image", {"width": 256, "height": 256}),
            PvSpec("$(E):histShape:width", "Image width",
                   "histShape.width", "number", "rw", 256, "Image"),
            PvSpec("$(E):histShape:height", "Image height",
                   "histShape.height", "number", "rw", 256, "Image"),
        ] + limits
    if category == "apertures":
        return common + [
            PvSpec("$(E):blades:left", "Blade left", "blades.left",
                   "number", "rw", -10, "Blades"),
            PvSpec("$(E):blades:right", "Blade right", "blades.right",
                   "number", "rw", 10, "Blades"),
            PvSpec("$(E):blades:bottom", "Blade bottom", "blades.bottom",
                   "number", "rw", -10, "Blades"),
            PvSpec("$(E):blades:top", "Blade top", "blades.top",
                   "number", "rw", 10, "Blades"),
            PvSpec("$(E):r", "Radius", "r", "number", "rw",
                   1, "Shape"),
        ] + limits

    return common + [
        PvSpec("$(E):ENERGY", "Energy", "bragg.energy", "number",
               "rw", 9000, "Material"),
        PvSpec("$(E):bragg", "Bragg", "bragg", "number", "rw",
               0, "Orientation"),
        PvSpec("$(E):braggOffset", "Bragg offset", "braggOffset",
               "number", "rw", 0, "Orientation"),
        PvSpec("$(E):positionRoll", "Position roll", "positionRoll",
               "number", "rw", 0, "Orientation"),
        PvSpec("$(E):rotationSequence", "Rotation sequence",
               "rotationSequence", "string", "rw", "RzRyRx",
               "Orientation"),
    ] + orient + limits + [
        PvSpec("$(E):limPhysX2:lmin", "limPhysX2 lmin", "limPhysX2.lmin",
               "number", "rw", -10, "Limits"),
        PvSpec("$(E):limPhysX2:lmax", "limPhysX2 lmax", "limPhysX2.lmax",
               "number", "rw", 10, "Limits"),
        PvSpec("$(E):limPhysY2:lmin", "limPhysY2 lmin", "limPhysY2.lmin",
               "number", "rw", -10, "Limits"),
        PvSpec("$(E):limPhysY2:lmax", "limPhysY2 lmax", "limPhysY2.lmax",
               "number", "rw", 10, "Limits"),
        PvSpec("$(E):R", "R", "R", "number", "rw", 0, "Shape"),
        PvSpec("$(E):r", "r", "r", "number", "rw", 0, "Shape"),
        PvSpec("$(E):Rm", "Rm", "Rm", "number", "rw", 0, "Shape"),
        PvSpec("$(E):Rs", "Rs", "Rs", "number", "rw", 0, "Shape"),
        PvSpec("$(E):p", "p", "p", "number", "rw", 0, "Shape"),
        PvSpec("$(E):q", "q", "q", "number", "rw", 0, "Shape"),
        PvSpec("$(E):f1", "f1", "f1", "number", "rw", 0, "Shape"),
        PvSpec("$(E):f2", "f2", "f2", "number", "rw", 0, "Shape"),
        PvSpec("$(E):n", "n", "n", "number", "rw", 0, "Shape"),
        PvSpec("$(E):period", "Period", "period", "number", "rw",
               0, "Shape"),
    ]


def _include_record(record: str,
                    epics_map: Optional[Dict[str, Optional[str]]]) -> bool:
    return not epics_map or record in epics_map


def _map_record(record: str,
                epics_map: Optional[Dict[str, Optional[str]]]) -> str:
    if not epics_map:
        return record
    mapped = epics_map.get(record)
    return record if mapped is None else mapped


def _add_energy_record(oe_obj: Any, oename: str, add) -> None:
    material = getattr(oe_obj, "material", None)
    if material is None or not hasattr(material, "get_Bragg_angle"):
        return

    try:
        if hasattr(oe_obj, "bragg"):
            e_field = "bragg.energy"
            initial_e = np.abs(CH / (2 * material.d * np.sin(
                    oe_obj.bragg - oe_obj.braggOffset)))
        else:
            e_field = "pitch.energy"
            initial_e = np.abs(CH / (2 * material.d * np.sin(oe_obj.pitch)))
    except Exception:
        e_field = "bragg.energy" if hasattr(oe_obj, "bragg") else\
            "pitch.energy"
        initial_e = None

    add(f"{oename}:ENERGY", "Energy", e_field, "number", "rw",
        initial_e, "Material")


def _add_image_records(oe_obj: Any, oename: str, add) -> None:
    if not hasattr(oe_obj, "expose") or getattr(oe_obj, "limPhysX", None) is None:
        return

    hist_shape = getattr(oe_obj, "histShape", [256, 256])
    try:
        width, height = int(hist_shape[0]), int(hist_shape[1])
        image_length = width * height
    except Exception:
        width, height, image_length = 256, 256, 65536

    add(f"{oename}:image", "Image", "image", "image", "ro", None,
        "Image", {"width": width, "height": height, "length": image_length})

    for index, field_name in enumerate(["width", "height"]):
        value = width if index == 0 else height
        add(f"{oename}:histShape:{field_name}", f"Image {field_name}",
            f"histShape.{field_name}", "number", "rw", value, "Image")


def _ordered_fields(fields: Iterable[str]) -> Iterable[str]:
    preferred = [
        "name", "nrays", "center", "pitch", "roll", "yaw", "bragg",
        "braggOffset",
        "rotationSequence", "positionRoll", "x", "z", "limPhysX",
        "limPhysY", "limPhysX2", "limPhysY2", "blades", "opening", "R",
        "r", "Rm", "Rs", "p", "q", "f1", "f2", "pAxis", "parabolaAxis",
        "n", "period", "fileName", "orientation"]
    seen = set()
    for field_name in preferred:
        if field_name in fields:
            seen.add(field_name)
            yield field_name
    for field_name in sorted(fields):
        if field_name not in seen:
            yield field_name


def _group(arg_name: str) -> str:
    if arg_name in {"name"}:
        return "Identity"
    if arg_name in {"nrays"}:
        return "Source"
    if arg_name in {"center"}:
        return "Position"
    if arg_name in orientationArgSet:
        return "Orientation"
    if arg_name.startswith("limPhys"):
        return "Limits"
    if arg_name == "blades":
        return "Blades"
    return "Shape"


def _label(arg_name: str) -> str:
    special = {
        "braggOffset": "Bragg offset",
        "rotationSequence": "Rotation sequence",
        "positionRoll": "Position roll",
        "limPhysX": "limPhysX",
        "limPhysY": "limPhysY",
        "limPhysX2": "limPhysX2",
        "limPhysY2": "limPhysY2",
        "fileName": "File name",
        "pAxis": "p axis",
        "parabolaAxis": "Parabola axis",
    }
    return special.get(arg_name, arg_name[:1].upper() + arg_name[1:])
