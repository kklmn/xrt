# -*- coding: utf-8 -*-
"""Generate Phoebus Display Builder BOB files for raycing EPICS records."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from xml.dom import minidom

try:
    from .records import (
        PvSpec, iter_element_specs, prefixed_pv_name, template_records,
        workflow_records)
except ImportError:  # Allow running this file directly from a checkout.
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from xrt.backends.raycing.epics.records import (  # type: ignore
        PvSpec, iter_element_specs, prefixed_pv_name, template_records,
        workflow_records)


DEFAULT_OUTPUT = Path(__file__).resolve().parent
ELEMENT_CATEGORIES = ("sources", "oes", "apertures", "screens")


class BobWriter:
    """Small writer that uses phoebusgen when available, XML otherwise."""

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self._phoebusgen = None
        if backend in {"auto", "phoebusgen"}:
            try:
                import phoebusgen.screen as screen
                import phoebusgen.widget as widget
            except ImportError:
                if backend == "phoebusgen":
                    raise
            else:
                self._phoebusgen = (screen, widget)

    @property
    def backend_name(self) -> str:
        return "phoebusgen" if self._phoebusgen else "xml"

    def write(self, file_name: Path, title: str, pvs: List[PvSpec],
              macros: Optional[Dict[str, str]] = None) -> None:
        file_name.parent.mkdir(parents=True, exist_ok=True)
        if self._phoebusgen:
            self._write_with_phoebusgen(file_name, title, pvs, macros or {})
        else:
            self._write_with_xml(file_name, title, pvs, macros or {})

    def _write_with_phoebusgen(self, file_name: Path, title: str,
                               pvs: List[PvSpec],
                               macros: Dict[str, str]) -> None:
        screen_module, widget = self._phoebusgen
        rows_height, image_height = _layout_height(pvs)
        display = screen_module.Screen(title)
        _safe_call(display, "width", 760)
        _safe_call(display, "height", rows_height + image_height + 76)
        for key, value in macros.items():
            _safe_call(display, "macro", key, value)

        widgets = []
        widgets.append(widget.Label("title", title, 16, 12, 720, 28))
        y = 52
        for group, group_pvs in _group_pvs(pvs):
            widgets.append(widget.Label(f"{group} group", group, 16, y,
                                        720, 20))
            y += 26
            for pv in group_pvs:
                if pv.kind == "image":
                    image = widget.Image(pv.label, prefixed_pv_name(pv.record),
                                         180, y, 320, 240)
                    _safe_call(image, "data_width",
                               int(pv.metadata.get("width", 256)))
                    _safe_call(image, "data_height",
                               int(pv.metadata.get("height", 256)))
                    _safe_call(image, "auto_scale", True)
                    _safe_call(image, "color_mode_MONO")
                    widgets.extend([
                        widget.Label(f"{pv.label} label", pv.label, 32, y,
                                     132, 24),
                        image,
                        widget.TextUpdate(f"{pv.label} pv",
                                          prefixed_pv_name(pv.record),
                                          520, y, 200, 24),
                    ])
                    y += 252
                    continue

                widgets.append(widget.Label(f"{pv.label} label", pv.label,
                                            32, y, 132, 24))
                pv_name = prefixed_pv_name(pv.record)
                if pv.kind == "bool_command":
                    control = widget.BooleanButton(pv.label, pv_name, 180, y,
                                                   130, 26)
                    _safe_call(control, "mode_push")
                    _safe_call(control, "on_label", "Acquire")
                    _safe_call(control, "off_label", "Acquire")
                elif pv.kind == "bool_toggle":
                    control = widget.BooleanButton(pv.label, pv_name, 180, y,
                                                   130, 26)
                    _safe_call(control, "mode_toggle")
                    _safe_call(control, "on_label", "Auto")
                    _safe_call(control, "off_label", "Manual")
                elif pv.kind == "bool_status":
                    control = widget.LED(pv.label, pv_name, 180, y, 28, 24)
                elif pv.access == "ro":
                    control = widget.TextUpdate(pv.label, pv_name, 180, y,
                                                210, 24)
                else:
                    control = widget.TextEntry(pv.label, pv_name, 180, y,
                                               210, 24)
                widgets.append(control)
                widgets.append(widget.TextUpdate(f"{pv.label} pv", pv_name,
                                                 420, y, 300, 24))
                y += 30

        display.add_widget(widgets)
        display.write_screen(str(file_name))

    def _write_with_xml(self, file_name: Path, title: str,
                        pvs: List[PvSpec], macros: Dict[str, str]) -> None:
        rows_height, image_height = _layout_height(pvs)
        root = ET.Element("display", {"version": "2.0.0"})
        _text(root, "name", title)
        _text(root, "width", "760")
        _text(root, "height", str(rows_height + image_height + 76))
        if macros:
            macro_node = ET.SubElement(root, "macros")
            for key, value in macros.items():
                _text(macro_node, key, value)

        _widget(root, "label", "title", 16, 12, 720, 28,
                {"text": title})
        y = 52
        for group, group_pvs in _group_pvs(pvs):
            _widget(root, "label", f"{group} group", 16, y, 720, 20,
                    {"text": group, "font": ("Liberation Sans", 14, "BOLD")})
            y += 26
            for pv in group_pvs:
                if pv.kind == "image":
                    pv_name = prefixed_pv_name(pv.record)
                    _widget(root, "label", f"{pv.label} label", 32, y, 132,
                            24, {"text": pv.label})
                    _widget(root, "image", pv.label, 180, y, 320, 240, {
                        "pv_name": pv_name,
                        "data_width": str(int(pv.metadata.get("width", 256))),
                        "data_height": str(int(pv.metadata.get("height", 256))),
                        "auto_scale": "true",
                        "color_mode": "MONO",
                    })
                    _widget(root, "textupdate", f"{pv.label} pv", 520, y,
                            200, 24, {"pv_name": pv_name})
                    y += 252
                    continue

                pv_name = prefixed_pv_name(pv.record)
                _widget(root, "label", f"{pv.label} label", 32, y, 132, 24,
                        {"text": pv.label})
                if pv.kind == "bool_command":
                    _widget(root, "bool_button", pv.label, 180, y, 130, 26, {
                        "pv_name": pv_name,
                        "mode": "PUSH",
                        "on_label": "Acquire",
                        "off_label": "Acquire",
                        "show_led": "false",
                    })
                elif pv.kind == "bool_toggle":
                    _widget(root, "bool_button", pv.label, 180, y, 130, 26, {
                        "pv_name": pv_name,
                        "mode": "TOGGLE",
                        "on_label": "Auto",
                        "off_label": "Manual",
                        "show_led": "true",
                    })
                elif pv.kind == "bool_status":
                    _widget(root, "led", pv.label, 180, y, 28, 24, {
                        "pv_name": pv_name,
                    })
                elif pv.access == "ro":
                    _widget(root, "textupdate", pv.label, 180, y, 210, 24, {
                        "pv_name": pv_name,
                    })
                else:
                    _widget(root, "textentry", pv.label, 180, y, 210, 24, {
                        "pv_name": pv_name,
                    })
                _widget(root, "textupdate", f"{pv.label} pv", 420, y,
                        300, 24, {"pv_name": pv_name})
                y += 30

        file_name.write_text(_pretty_xml(root), encoding="utf-8")


def load_beamline(beamline_spec: Optional[str] = None,
                  layout_file: Optional[str] = None,
                  builder_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """Load a BeamLine either from a Python builder or Qook XML/JSON."""

    if bool(beamline_spec) == bool(layout_file):
        raise ValueError("Specify exactly one of --beamline or --layout")

    if layout_file:
        from ..beamline import BeamLine
        return BeamLine(fileName=layout_file)

    module_spec, callable_name = _split_callable_spec(beamline_spec)
    module = _load_module(module_spec)
    builder = getattr(module, callable_name)
    return builder(**(builder_kwargs or {}))


def generate_templates(output: Path, writer: BobWriter,
                       prefix: str = "") -> List[Path]:
    """Generate generic template BOBs."""

    paths: List[Path] = []
    for category in ELEMENT_CATEGORIES:
        path = output / category / f"{category[:-1]}_template.bob"
        writer.write(path, f"xrt {category[:-1]} EPICS",
                     template_records(category),
                     {"P": _prefix_macro(prefix), "E": "Element"})
        paths.append(path)

    path = output / "propagation" / "propagation_control.bob"
    writer.write(path, "xrt propagation control", workflow_records(),
                 {"P": _prefix_macro(prefix)})
    paths.append(path)
    return paths


def generate_for_beamline(bl: Any, output: Path, writer: BobWriter,
                          prefix: str = "",
                          epics_map: Optional[Dict[str, Optional[str]]] = None
                          ) -> List[Path]:
    """Generate one BOB file per EPICS-enabled element plus propagation."""

    paths: List[Path] = []
    workflow_path = output / "propagation" / "propagation_control.bob"
    writer.write(workflow_path, "xrt propagation control",
                 workflow_records(epics_map), {"P": _prefix_macro(prefix)})
    paths.append(workflow_path)

    for spec in iter_element_specs(bl, epics_map):
        if not spec.pvs:
            continue
        file_name = f"{spec.valid_name}.bob"
        path = output / spec.category / file_name
        title = f"{spec.name} EPICS"
        writer.write(path, title, spec.pvs,
                     {"P": _prefix_macro(prefix), "E": spec.valid_name})
        paths.append(path)
    return paths


def list_specs(bl: Any,
               epics_map: Optional[Dict[str, Optional[str]]] = None) -> str:
    """Return a human-readable EPICS element inventory."""

    lines = []
    for spec in iter_element_specs(bl, epics_map):
        lines.append(f"{spec.category}/{spec.valid_name} "
                     f"({spec.class_name})")
        for pv in spec.pvs:
            lines.append(f"  {pv.record:36s} {pv.access:2s} "
                         f"{pv.kind:12s} {pv.property_path}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beamline",
                        help="Python module or file plus builder, e.g. "
                             "path/to/bl.py:build_beamline")
    parser.add_argument("--layout",
                        help="xrtQook XML or JSON layout file")
    parser.add_argument("--builder-kw", action="append", default=[],
                        metavar="NAME=JSON",
                        help="Keyword argument for the beamline builder")
    parser.add_argument("--epics-map",
                        help="JSON file with EpicsDevice record remapping")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output folder for generated BOB files")
    parser.add_argument("--prefix", default="",
                        help="Default EPICS prefix macro value. A trailing "
                             "colon is added when needed.")
    parser.add_argument("--templates", action="store_true",
                        help="Generate generic category templates")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="List elements and records without writing BOBs")
    parser.add_argument("--backend", choices=["auto", "phoebusgen", "xml"],
                        default="auto",
                        help="Use phoebusgen when available, or write XML "
                             "directly.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.beamline = _clean_path_arg(args.beamline)
    args.layout = _clean_path_arg(args.layout)
    args.epics_map = _clean_path_arg(args.epics_map)
    args.output = _clean_path_arg(args.output)

    output = Path(args.output)
    epics_map = _load_epics_map(args.epics_map)
    builder_kwargs = _parse_builder_kwargs(args.builder_kw)
    writer = BobWriter(args.backend)

    if args.templates and (args.beamline or args.layout or args.list_only):
        parser.error("--templates cannot be combined with --beamline, "
                     "--layout, or --list")

    if args.list_only and not (args.beamline or args.layout):
        parser.error("--list requires --beamline or --layout")

    if args.templates:
        paths = generate_templates(output, writer, args.prefix)
    elif args.beamline or args.layout:
        bl = load_beamline(args.beamline, args.layout, builder_kwargs)
        if args.list_only:
            print(list_specs(bl, epics_map))
            return 0
        paths = generate_for_beamline(bl, output, writer, args.prefix,
                                      epics_map)
    else:
        paths = generate_templates(output, writer, args.prefix)

    for path in paths:
        print(path)
    print(f"Generated {len(paths)} BOB file(s) using {writer.backend_name}.")
    return 0


def _group_pvs(pvs: List[PvSpec]):
    order = [
        "Propagation", "Identity", "Source", "Material", "Image", "Position",
        "Orientation", "Limits", "Blades", "Shape", "Properties"]
    seen = []
    for pv in pvs:
        if pv.group not in seen:
            seen.append(pv.group)
    for group in order + sorted(set(seen) - set(order)):
        group_pvs = [pv for pv in pvs if pv.group == group]
        if group_pvs:
            yield group, group_pvs


def _layout_height(pvs: List[PvSpec]):
    row_count = len([pv for pv in pvs if pv.kind != "image"])
    group_count = len({pv.group for pv in pvs})
    image_count = len([pv for pv in pvs if pv.kind == "image"])
    return 52 + group_count * 26 + row_count * 30, image_count * 252


def _widget(parent, widget_type: str, name: str, x: int, y: int,
            width: int, height: int, props: Dict[str, Any]):
    widget = ET.SubElement(parent, "widget", {
        "type": widget_type,
        "version": "2.0.0",
    })
    _text(widget, "name", name)
    _text(widget, "x", str(x))
    _text(widget, "y", str(y))
    _text(widget, "width", str(width))
    _text(widget, "height", str(height))
    for key, value in props.items():
        if key == "font":
            family, size, style = value
            font = ET.SubElement(widget, "font")
            ET.SubElement(font, "font", {
                "family": str(family),
                "size": str(size),
                "style": str(style),
            })
        else:
            _text(widget, key, str(value))
    return widget


def _text(parent, tag: str, value: str):
    node = ET.SubElement(parent, tag)
    node.text = value
    return node


def _pretty_xml(root) -> str:
    rough = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def _safe_call(obj, method_name: str, *args):
    method = getattr(obj, method_name, None)
    if method is not None:
        method(*args)


def _split_callable_spec(spec: str):
    module_spec, sep, callable_name = spec.partition(":")
    return module_spec, callable_name if sep else "build_beamline"


def _load_module(module_spec: str):
    path = Path(module_spec)
    if path.exists():
        module_name = f"_xrt_epics_bob_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {module_spec}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_spec)


def _parse_builder_kwargs(items: Iterable[str]) -> Dict[str, Any]:
    result = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected NAME=JSON for --builder-kw, got {item}")
        result[key] = json.loads(value)
    return result


def _load_epics_map(file_name: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    if not file_name:
        return None
    with open(file_name, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _clean_path_arg(value: Optional[str]) -> Optional[str]:
    """Tolerate a Windows path whose trailing backslash escaped the quote."""

    if value is None or not sys.platform.startswith("win"):
        return value

    if value.endswith(("'", '"')) and not value.startswith(value[-1]):
        return value[:-1]
    if value.startswith(("'", '"')) and not value.endswith(value[0]):
        return value[1:]
    return value


def _prefix_macro(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith(":") else f"{prefix}:"


if __name__ == "__main__":
    raise SystemExit(main())
