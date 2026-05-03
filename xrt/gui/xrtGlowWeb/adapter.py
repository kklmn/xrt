# -*- coding: utf-8 -*-
"""
Adapters between xrt beamline objects and browser-friendly payloads.
"""

import importlib
import ast
import copy
import json
import os
import queue
import runpy
import time
from collections import OrderedDict
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np

from ...backends import raycing
from ...backends.raycing import msg_start, msg_stop, msg_exit
from ...backends.raycing import propagationProcess
from ...backends.raycing.run import run_process
from ...backends.raycing.sources import Beam
from ...backends.raycing import screens as rscreens


DEFAULT_MAX_RAYS = 10000


class XrtGlowWebError(RuntimeError):
    pass


def clean_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (dict, OrderedDict)):
        return {str(key): clean_json(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def loads_json_file(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=OrderedDict)


def ensure_project_layout(data):
    if data is None:
        return make_demo_layout()
    if isinstance(data, dict) and "Project" in data:
        return data
    if isinstance(data, dict):
        return {"Project": data}
    raise XrtGlowWebError("Expected a beamline layout dictionary.")


def load_object_spec(spec):
    if ":" not in spec:
        raise XrtGlowWebError(
            "Use module_or_path:object_name for --beamline, for example "
            "examples/demo.py:build_beamline.")

    mod_or_path, obj_name = spec.split(":", 1)
    if os.path.exists(mod_or_path):
        namespace = runpy.run_path(mod_or_path)
        try:
            return namespace[obj_name]
        except KeyError as exc:
            raise XrtGlowWebError(
                "Cannot find {!r} in {!r}.".format(obj_name, mod_or_path)
            ) from exc

    module = importlib.import_module(mod_or_path)
    try:
        return getattr(module, obj_name)
    except AttributeError as exc:
        raise XrtGlowWebError(
            "Cannot find {!r} in module {!r}.".format(obj_name, mod_or_path)
        ) from exc


def beamline_to_layout(beamline):
    try:
        run_process(beamline)
    except Exception:
        pass
    try:
        beamline.index_materials()
    except Exception:
        pass
    project = beamline.export_to_json()
    return getattr(beamline, "layoutStr", None) or {"Project": project}


def load_layout(layout_path=None, beamline_spec=None):
    if layout_path:
        if str(layout_path).lower().endswith(".xml"):
            beamline = raycing.BeamLine()
            beamline.load_from_xml(layout_path)
            return beamline_to_layout(beamline)
        return ensure_project_layout(loads_json_file(layout_path))
    if beamline_spec:
        obj = load_object_spec(beamline_spec)
        obj = obj() if callable(obj) else obj
        if isinstance(obj, dict):
            return ensure_project_layout(obj)
        if hasattr(obj, "export_to_json"):
            return beamline_to_layout(obj)
        raise XrtGlowWebError(
            "--beamline object must be a layout dict, BeamLine, or a factory "
            "returning one of these.")
    return make_demo_layout()


def make_demo_layout():
    project = OrderedDict()
    project["Beams"] = OrderedDict([
        ("None", None),
        ("demoSourceBeam", None),
        ("demoSampleBeam", None),
    ])
    project["Materials"] = OrderedDict()
    project["beamLine"] = OrderedDict([
        ("properties", OrderedDict([
            ("height", "0.0"),
            ("azimuth", "0.0"),
            ("alignE", "9000.0"),
        ])),
        ("_object", "xrt.backends.raycing.BeamLine"),
        ("demoSource", OrderedDict([
            ("properties", OrderedDict([
                ("bl", "beamLine"),
                ("name", "DemoSource"),
                ("center", "(0, 0, 0)"),
                ("nrays", "4000"),
                ("distx", "normal"),
                ("dx", "0.4"),
                ("distz", "normal"),
                ("dz", "0.4"),
                ("distxprime", "normal"),
                ("dxprime", "0.0004"),
                ("distzprime", "normal"),
                ("dzprime", "0.0004"),
                ("distE", "lines"),
                ("energies", "(9000.0,)"),
                ("polarization", "horizontal"),
            ])),
            ("_object", "xrt.backends.raycing.sources.GeometricSource"),
            ("shine", OrderedDict([
                ("parameters", OrderedDict([
                    ("toGlobal", "True"),
                    ("withAmplitudes", "False"),
                    ("accuBeam", "None"),
                ])),
                ("_object",
                 "xrt.backends.raycing.sources.GeometricSource.shine"),
                ("output", OrderedDict([
                    ("beamGlobal", "demoSourceBeam"),
                ])),
            ])),
        ])),
        ("demoSample", OrderedDict([
            ("properties", OrderedDict([
                ("bl", "beamLine"),
                ("name", "Sample"),
                ("center", "[0, 1200, 0]"),
                ("x", "auto"),
                ("z", "auto"),
            ])),
            ("_object", "xrt.backends.raycing.screens.Screen"),
            ("expose", OrderedDict([
                ("parameters", OrderedDict([
                    ("beam", "demoSourceBeam"),
                ])),
                ("_object", "xrt.backends.raycing.screens.Screen.expose"),
                ("output", OrderedDict([
                    ("beamLocal", "demoSampleBeam"),
                ])),
            ])),
        ])),
    ])
    project["FigureErrors"] = OrderedDict()
    project["plots"] = OrderedDict()
    project["run_ray_tracing"] = None
    project["description"] = "Built-in xrtGlowWeb demo"
    return {"Project": project}


def project_node(layout):
    return ensure_project_layout(layout)["Project"]


def beamline_record(layout):
    project = project_node(layout)
    for key, value in project.items():
        if isinstance(value, dict):
            object_name = str(value.get("_object", ""))
            if object_name.endswith("BeamLine") or "BeamLine" in object_name:
                return key, value
    for key, value in project.items():
        if isinstance(value, dict) and "properties" in value:
            return key, value
    raise XrtGlowWebError("Cannot find a beamline record in layout.")


def to_float(value, default=0.0):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_vec3(value):
    if isinstance(value, str):
        value = value.strip("()[] ")
        value = [part.strip() for part in value.split(",") if part.strip()]
    if value is None:
        return [0.0, 0.0, 0.0]
    if not isinstance(value, (list, tuple, np.ndarray)):
        return [0.0, 0.0, 0.0]
    vals = list(value)[:3]
    while len(vals) < 3:
        vals.append(0.0)
    return [to_float(vals[0]), to_float(vals[1]), to_float(vals[2])]


def parse_value(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def numeric_sequence(value, fallback=None):
    value = parse_value(value)
    if fallback is None:
        fallback = []
    if not isinstance(value, (list, tuple, np.ndarray)):
        return fallback
    return [to_float(item) for item in value]


def numeric_limits(value, fallback):
    vals = numeric_sequence(value)
    if len(vals) < 2:
        return list(fallback)
    lo, hi = vals[:2]
    if lo == hi:
        return list(fallback)
    return [min(lo, hi), max(lo, hi)]


def xrt_to_three(vec):
    return to_vec3(vec)


def element_kind(object_name):
    lowered = object_name.lower()
    if "source" in lowered:
        return "source"
    if "aperture" in lowered or "slit" in lowered:
        return "aperture"
    if "screen" in lowered:
        return "screen"
    if ".oes" in lowered or "mirror" in lowered or "crystal" in lowered:
        return "oe"
    return "element"


def scene_from_layout(layout):
    project = project_node(layout)
    beamline_name, record = beamline_record(layout)
    elements = []

    for uuid, entry in record.items():
        if not isinstance(entry, dict) or "properties" not in entry:
            continue
        props = entry.get("properties") or {}
        object_name = str(entry.get("_object", ""))
        center = to_vec3(props.get("center"))
        elements.append({
            "uuid": str(uuid),
            "name": str(props.get("name", uuid)),
            "kind": element_kind(object_name),
            "object": object_name,
            "center": center,
            "position": xrt_to_three(center),
            "props": clean_json(props),
            "geometry": element_geometry_payload(props, object_name),
        })

    render_instances = render_instances_from_layout(layout)
    for element in elements:
        instances = render_instances.get(element["uuid"]) or \
            render_instances.get(element["name"])
        if instances is None:
            instances = [{
                "nsIndex": 0,
                "is2ndXtal": False,
                "position": element["position"],
            }]
        element["renderInstances"] = instances

    flow = project.get("flow") or extract_flow_from_record(record)
    ordered_ids = [str(key) for key in flow.keys()]
    if not ordered_ids:
        ordered_ids = [item["uuid"] for item in sorted(
            elements, key=lambda item: item["center"][1])]

    centers = {item["uuid"]: item["position"] for item in elements}
    links = []
    previous = None
    for uuid in ordered_ids:
        if uuid not in centers:
            continue
        if previous is not None:
            links.append({"start": previous, "end": uuid,
                          "points": [centers[previous], centers[uuid]]})
        previous = uuid

    return {
        "beamline": beamline_name,
        "elements": elements,
        "links": links,
        "flow": clean_json(flow),
    }


def render_instances_from_layout(layout):
    try:
        bl = raycing.BeamLine()
        bl.deserialize(copy.deepcopy(ensure_project_layout(layout)))
    except Exception:
        return {}

    return render_instances_from_beamline(bl)


def render_instances_from_beamline(bl):
    instances = {}
    for oeid, line in bl.oesDict.items():
        oe = line[0]
        if oe is None:
            continue
        item = render_instances_for_oe(oe)
        instances[str(oeid)] = item
        instances[str(getattr(oe, "name", oeid))] = item
    return instances


def render_instances_for_oe(oe):
    item = [{
        "nsIndex": 0,
        "is2ndXtal": False,
        "position": xrt_to_three(surface_center(oe, is2ndXtal=False)),
    }]
    if hasattr(oe, "cryst2pitch"):
        item.append({
            "nsIndex": 1,
            "is2ndXtal": True,
            "position": xrt_to_three(surface_center(oe, is2ndXtal=True)),
        })
    return item


def surface_center(oe, is2ndXtal=False):
    if isinstance(oe, rscreens.Screen):
        try:
            return list(oe.local_to_global(0, 0, 0))
        except Exception:
            return to_vec3(getattr(oe, "center", [0, 0, 0]))
    if hasattr(oe, "local_to_global"):
        try:
            lb = Beam(nrays=1)
            gb = oe.local_to_global(
                lb, returnBeam=True, is2ndXtal=bool(is2ndXtal))
            return [float(gb.x[0]), float(gb.y[0]), float(gb.z[0])]
        except Exception:
            return to_vec3(getattr(oe, "center", [0, 0, 0]))
    return to_vec3(getattr(oe, "center", [0, 0, 0]))


def apply_scene_updates(scene, updates):
    scene = copy.deepcopy(scene)
    by_key = {}
    for element in scene.get("elements", []):
        by_key[str(element.get("uuid"))] = element
        by_key[str(element.get("name"))] = element

    for update in updates:
        element = by_key.get(str(update.get("uuid"))) or \
            by_key.get(str(update.get("name")))
        if element is None:
            continue
        for key in ("center", "position", "renderInstances"):
            if key in update:
                element[key] = update[key]

    centers = {
        str(element.get("uuid")): element.get("position")
        for element in scene.get("elements", [])
    }
    for link in scene.get("links", []):
        start = centers.get(str(link.get("start")))
        end = centers.get(str(link.get("end")))
        if start is not None and end is not None:
            link["points"] = [start, end]
    return scene


def element_geometry_payload(props, object_name):
    kind = element_kind(object_name)
    if kind == "source":
        dx = abs(to_float(props.get("dx"), 0.4)) or 0.4
        dz = abs(to_float(props.get("dz"), 0.4)) or 0.4
        return {"shape": "source", "width": dx, "height": dz, "depth": max(dx, dz)}
    if kind == "screen":
        lx = numeric_limits(props.get("limPhysX"), [-1.0, 1.0])
        ly = numeric_limits(props.get("limPhysY"), [-1.0, 1.0])
        width = max(abs(lx[1] - lx[0]), 1.0)
        height = max(abs(ly[1] - ly[0]), 1.0)
        return {"shape": "plane", "width": width, "height": height,
                "depth": max(min(width, height) * 0.02, 0.05)}
    if kind == "aperture":
        opening = numeric_sequence(props.get("opening"))
        if len(opening) >= 4:
            width = max(abs(opening[1] - opening[0]), 1.0)
            height = max(abs(opening[3] - opening[2]), 1.0)
        else:
            lx = numeric_limits(props.get("limPhysX"), [-1.0, 1.0])
            ly = numeric_limits(props.get("limPhysY"), [-1.0, 1.0])
            width = max(abs(lx[1] - lx[0]), 1.0)
            height = max(abs(ly[1] - ly[0]), 1.0)
        return {"shape": "aperture", "width": width, "height": height,
                "depth": max(min(width, height) * 0.08, 0.1)}

    lx = numeric_limits(props.get("limPhysX"), [-1.0, 1.0])
    ly = numeric_limits(props.get("limPhysY"), [-1.0, 1.0])
    width = max(abs(lx[1] - lx[0]), abs(to_float(props.get("zmax"), 0)) * 2, 1.0)
    height = max(abs(ly[1] - ly[0]), abs(to_float(props.get("zmax"), 0)) * 2, 1.0)
    depth = max(abs(to_float(props.get("t"), 0.1)), min(width, height) * 0.08, 0.1)
    lowered = object_name.lower()
    shape = "lens" if "lens" in lowered else "box"
    return {"shape": shape, "width": width, "height": height, "depth": depth}


def extract_flow_from_record(record):
    flow = OrderedDict()
    for uuid, entry in record.items():
        if not isinstance(entry, dict):
            continue
        methods = OrderedDict()
        for key, value in entry.items():
            if key in ("properties", "_object"):
                continue
            if isinstance(value, dict) and "parameters" in value:
                methods[key] = clean_json(value.get("parameters") or {})
        if methods:
            flow[str(uuid)] = methods
    return flow


def sample_indices(length, max_items):
    if length <= max_items:
        return np.arange(length)
    return np.linspace(0, length - 1, max_items).astype(int)


def beam_to_payload(beam, max_rays=DEFAULT_MAX_RAYS, color_axis_data=None):
    if beam is None or not hasattr(beam, "x"):
        return None

    x = np.asarray(getattr(beam, "x", []), dtype=float)
    y = np.asarray(getattr(beam, "y", np.zeros_like(x)), dtype=float)
    z = np.asarray(getattr(beam, "z", np.zeros_like(x)), dtype=float)
    if x.size == 0:
        return None

    state = np.asarray(getattr(beam, "state", np.ones_like(x)), dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    finite &= state > 0
    available = int(np.count_nonzero(finite))
    if available == 0:
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        available = int(np.count_nonzero(finite))

    indices = np.where(finite)[0]
    indices = indices[sample_indices(indices.size, int(max_rays))]
    positions = np.column_stack((x[indices], y[indices], z[indices]))

    jss = np.asarray(getattr(beam, "Jss", np.ones_like(x)), dtype=float)
    jpp = np.asarray(getattr(beam, "Jpp", np.zeros_like(x)), dtype=float)
    intensity = jss[indices] + jpp[indices]
    energy = np.asarray(getattr(beam, "E", np.zeros_like(x)), dtype=float)
    if color_axis_data is None:
        color_axis_data = energy if energy.size == x.size else intensity
    color_axis_data = np.asarray(color_axis_data, dtype=float)

    return {
        "count": int(indices.size),
        "available": available,
        "total": int(x.size),
        "indices": np.asarray(indices, dtype=int).tolist(),
        "positions": np.round(positions, 6).tolist(),
        "colorAxis": np.round(color_axis_data[indices], 9).tolist()
        if color_axis_data.size == x.size else [],
        "intensity": np.round(intensity, 9).tolist(),
        "energy": np.round(energy[indices], 6).tolist()
        if energy.size == x.size else [],
        "state": np.asarray(state[indices], dtype=int).tolist()
        if state.size == x.size else [],
    }


class BeamRenderContext:
    def __init__(self, layout, color_axis="energy", global_colors=True):
        self.layout = ensure_project_layout(layout)
        self.color_axis = color_axis
        self.global_colors = bool(global_colors)
        self.beamline = raycing.BeamLine()
        self.beamline.deserialize(self.layout)
        self.beamline.flowSource = "Qook"
        self.elements_by_id = {}
        self.elements_by_name = {}
        for oeid, line in self.beamline.oesDict.items():
            oe = line[0]
            self.elements_by_id[str(oeid)] = oe
            self.elements_by_name[str(getattr(oe, "name", oeid))] = oe
        self.get_color = getattr(
            raycing, "get_{}".format(color_axis), raycing.get_energy)

    def update_from_message(self, message):
        sender_id = str(message.get("sender_id", ""))
        sender_name = str(message.get("sender_name", ""))
        oe = self.get_element(sender_id, sender_name)
        if oe is None:
            return
        if "pos_attr" in message:
            attr = message.get("pos_attr")
            value = message.get("pos_value")
            if attr in ["footprint"]:
                setattr(oe, attr, value)
            else:
                setattr(oe, "_{}Val".format(attr), value)
                if hasattr(oe, attr):
                    try:
                        setattr(oe, attr, value)
                    except Exception:
                        pass
            if hasattr(oe, "get_orientation"):
                try:
                    oe.get_orientation()
                except Exception:
                    pass
        elif "diag_attr" in message:
            try:
                setattr(oe, message.get("diag_attr"), message.get("diag_value"))
            except Exception:
                pass

    def get_element(self, sender_id, sender_name):
        return self.elements_by_id.get(sender_id) or \
            self.elements_by_name.get(sender_name)

    def beam_for_render(self, beam, beam_key, sender_id, sender_name):
        if beam is None:
            return None
        beam_key_l = str(beam_key).lower()
        if "global" in beam_key_l:
            return beam

        oe = self.get_element(sender_id, sender_name)
        if oe is None:
            return beam

        beam_glo = Beam(copyFrom=beam)
        try:
            if isinstance(oe, rscreens.Screen):
                xg, yg, zg = oe.local_to_global(
                    x=beam_glo.x, y=beam_glo.y, z=beam_glo.z)
                beam_glo.x, beam_glo.y, beam_glo.z = xg, yg, zg
            elif hasattr(oe, "local_to_global"):
                oe.local_to_global(
                    beam_glo, is2ndXtal=("local2" in beam_key_l))
            else:
                raycing.virgin_local_to_global(
                    self.beamline, beam_glo, getattr(oe, "center", [0, 0, 0]))
        except Exception:
            try:
                raycing.virgin_local_to_global(
                    self.beamline, beam_glo, getattr(oe, "center", [0, 0, 0]))
            except Exception:
                return beam
        return beam_glo

    def color_data_for_render(self, beam, beam_key, sender_id, sender_name):
        if beam is None:
            return None
        beam_for_color = beam
        if self.global_colors and "global" not in str(beam_key).lower():
            beam_for_color = self.beam_for_render(
                beam, beam_key, sender_id, sender_name)
        try:
            return self.get_color(beam_for_color)
        except Exception:
            try:
                return raycing.get_energy(beam_for_color)
            except Exception:
                return None

    def scene_updates(self):
        updates = []
        instances = render_instances_from_beamline(self.beamline)
        for oeid, line in self.beamline.oesDict.items():
            oe = line[0]
            if oe is None:
                continue
            center = surface_center(oe, is2ndXtal=False)
            name = str(getattr(oe, "name", oeid))
            updates.append({
                "uuid": str(oeid),
                "name": name,
                "center": to_vec3(center),
                "position": xrt_to_three(center),
                "renderInstances": instances.get(str(oeid)) or
                instances.get(name) or render_instances_for_oe(oe),
            })
        return updates


def serialize_worker_message(message, max_rays=DEFAULT_MAX_RAYS,
                             render_context=None):
    if "beam" not in message:
        return clean_json(message)

    beams = OrderedDict()
    sender_id = str(message.get("sender_id", ""))
    sender_name = str(message.get("sender_name", ""))
    for beam_key, beam in message.get("beam", {}).items():
        if render_context is not None:
            beam_render = render_context.beam_for_render(
                beam, beam_key, sender_id, sender_name)
            color_data = render_context.color_data_for_render(
                beam, beam_key, sender_id, sender_name)
        else:
            beam_render = beam
            color_data = None
        payload = beam_to_payload(
            beam_render, max_rays=max_rays, color_axis_data=color_data)
        if payload is not None:
            beams[str(beam_key)] = payload

    return {
        "sender_id": sender_id,
        "sender_name": sender_name,
        "status": int(message.get("status", 0)),
        "beam": beams,
    }


class PropagationSession:
    def __init__(self, layout, max_rays=DEFAULT_MAX_RAYS):
        self.layout = ensure_project_layout(layout)
        self.max_rays = int(max_rays)
        self.element_centers = self._element_centers()
        self.render_context = BeamRenderContext(self.layout)
        self.source_ids = self._source_ids()
        self.input_queue = None
        self.output_queue = None
        self.process = None
        self.last_run = []
        self.last_error = None

    def start(self):
        if self.process is not None and self.process.is_alive():
            return
        self.stop()
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.process = Process(
            target=propagationProcess,
            args=(self.input_queue, self.output_queue))
        self.process.start()
        self.input_queue.put({
            "command": "create",
            "object_type": "beamline",
            "kwargs": self.layout,
        })

    def stop(self):
        if self.input_queue is not None:
            try:
                self.input_queue.put(msg_exit)
            except Exception:
                pass
        if self.process is not None:
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        self.process = None
        self.input_queue = None
        self.output_queue = None

    def modify(self, uuid, object_type, kwargs):
        self.start()
        self.input_queue.put({
            "command": "modify",
            "object_type": object_type,
            "uuid": uuid,
            "kwargs": kwargs,
        })

    def run_once(self, timeout=20.0, max_rays=None):
        self.start()
        max_rays = max(1, int(max_rays or self.max_rays))
        messages = []
        deadline = time.time() + float(timeout)
        self._drain_output()
        self._apply_ray_budget(max_rays)
        self.input_queue.put(msg_start)

        while time.time() < deadline:
            try:
                msg = self.output_queue.get(timeout=0.1)
            except queue.Empty:
                if self.process is not None and not self.process.is_alive():
                    self.last_error = "Propagation worker exited."
                    break
                continue

            self.render_context.update_from_message(msg)
            payload = serialize_worker_message(
                msg, max_rays=max_rays,
                render_context=self.render_context)
            messages.append(payload)
            if "repeat" in msg:
                break

        try:
            self.input_queue.put(msg_stop)
        except Exception:
            pass

        self.last_run = messages
        return messages

    def scene_updates(self):
        return self.render_context.scene_updates()

    def _element_centers(self):
        centers = {}
        for element in scene_from_layout(self.layout)["elements"]:
            centers[element["uuid"]] = element["center"]
            centers[element["name"]] = element["center"]
        return centers

    def _source_ids(self):
        source_ids = []
        for oeid, line in self.render_context.beamline.oesDict.items():
            oe = line[0]
            if oe is None:
                continue
            object_name = "{}.{}".format(
                oe.__class__.__module__, oe.__class__.__name__).lower()
            if "source" in object_name and hasattr(oe, "nrays"):
                source_ids.append(str(oeid))
        return source_ids

    def _apply_ray_budget(self, max_rays):
        if self.input_queue is None:
            return
        for source_id in self.source_ids:
            oe = self.render_context.elements_by_id.get(source_id)
            if oe is not None and hasattr(oe, "nrays"):
                try:
                    setattr(oe, "nrays", int(max_rays))
                except Exception:
                    pass
            self.input_queue.put({
                "command": "modify",
                "object_type": "oe",
                "uuid": source_id,
                "kwargs": {"nrays": int(max_rays)},
            })

    def _drain_output(self):
        if self.output_queue is None:
            return
        while True:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    def status(self):
        return {
            "worker": bool(
                self.process is not None and self.process.is_alive()),
            "lastMessageCount": len(self.last_run),
            "lastError": self.last_error,
        }
