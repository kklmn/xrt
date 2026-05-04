# xrtGlowWeb prototype

This is an experimental browser front end for xrtGlow-style beamline scenes.
It is deliberately isolated from `xrt.gui.xrtGlow`: the prototype reuses
raycing layout deserialization and propagation, but renders in a browser with a
self-contained WebGL viewer. This keeps the first embed prototype usable
without external JavaScript CDNs.

## Run the built-in demo

From the repository root:

```powershell
pixi run python -m xrt.gui.xrtGlowWeb --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

Press **Run** in the browser to trigger propagation and draw sampled beam rays.
Use **XML** to upload an xrtQook/xrtGlow XML layout into the current model. Use
the mouse wheel to scale the model, Ctrl-wheel to zoom the camera, drag to orbit,
and Shift-drag to pan. The X/Y/Z sliders use xrtGlow's logarithmic `0..7` scale
range.

The scene uses the same axis convention as xrtGlow: X is left-right, Y is
back-front along the beamline, and Z is up-down. The default scale is X=1000,
Y=1, Z=1000 for long beamlines with narrow beams. The X/Y/Z sliders still use
xrtGlow's logarithmic range, and the adjacent numeric boxes show and edit the
actual scale factors.

The **Opt**, **Slits**, and **Screens** checkboxes independently hide optical
surfaces, slit/aperture surfaces, and screen surfaces while keeping links,
labels, beams, and footprints visible.

Beam visibility follows the xrtGlow navigation split: **Footprints** are point
clouds from the source `beamGlobal` and from local beams on each downstream
element, including both local surfaces of multi-surface optics. **Beams** are
the ray-line connections between those footprints and are listed by their
starting and ending objects.

Surfaces are generated in Python with xrtGlow's `OEMesh3D` surface mesh logic
when the element transform is resolved, including realistic optics, screens,
and slit/aperture blades. Unresolved auto-aligned elements fall back to the
lightweight schematic mesh until propagation updates them.

Sources follow xrtGlow's source styling: `GeometricSource` instances render as
the geometric source abstraction, while physical magnetic sources render as the
alternating red/blue magnet array used by `prepare_magnets`/`render_magnets`.

## Run an existing Qook/xrtGlow XML layout

```powershell
pixi run python -m xrt.gui.xrtGlowWeb --layout examples\withRaycing\_QookBeamlines\lens1.xml --host 127.0.0.1 --port 8765
```

For the BioXAS example:

```powershell
pixi run python -m xrt.gui.xrtGlowWeb --layout examples\withRaycing\_QookBeamlines\BioXAS_Main.xml --host 127.0.0.1 --port 8765
```

## Run a Python beamline factory

The object must be importable as `module:object` or loaded from a file path as
`path\to\file.py:object`. It may return either a raycing `BeamLine` or a layout
dictionary compatible with xrtQook/xrtGlow.

```powershell
pixi run python -m xrt.gui.xrtGlowWeb --beamline examples\my_bl.py:build_beamline
```

## Development notes

- `/api/scene` returns a browser-friendly scene graph.
- `/api/layout` returns the original serialized layout.
- `POST /api/layout/upload` accepts an XML filename and content, reloads the
  current model, and returns the new scene.
- `POST /api/run?maxRays=10000` sets source `nrays`, runs propagation, returns
  sampled beam data, and refreshes auto-aligned scene transforms.
- `POST /api/modify` forwards a small `modify` message to the propagation
  session.

This first prototype sends JSON payloads for simplicity and renders a minimal
3D WebGL scene. Beam and footprint render payloads are prepared in Python:
local beams are transformed into global coordinates with the raycing element
objects, and color-axis values are computed with the same `raycing.get_*`
helpers used by xrtGlow. Large beam buffers should eventually move to a binary
WebSocket protocol, and the renderer can then grow toward xrtGlow's full
mesh/shader feature set.
