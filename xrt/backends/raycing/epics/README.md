# raycing EPICS Phoebus BOB files

This folder contains generated Phoebus Display Builder screens for the EPICS
records exposed by `xrt.backends.raycing.epics.EpicsDevice`.

Use `from xrt.backends.raycing.epics import EpicsDevice` for new code. The
root `xrt.backends.raycing.EpicsDevice` import is still re-exported for
compatibility.

The generated BOB screens use `$(P)` as the EPICS prefix macro. The template
screens also use `$(E)` for the sanitized element name.

## Output layout

Unless `--output` is given, files are written into this folder:

```text
xrt/backends/raycing/epics/
```

The generator creates these subfolders:

```text
sources/
oes/
apertures/
screens/
propagation/
```

Generic template files are named `source_template.bob`, `oe_template.bob`,
`aperture_template.bob`, `screen_template.bob` and
`propagation_control.bob`. Per-element files use the sanitized element name,
for example `screens/screen01.bob`.

Source screens include `nrays` in addition to name, center and orientation
records when the source object exposes an `nrays` attribute.

Screen image EPICS records are fixed-length waveforms. `EpicsDevice` reserves
space for at least `1024 * 1024` pixels by default, or the initial screen
`histShape` if it is larger. Pass `imageMaxLength` to `EpicsDevice` if larger
runtime histograms are expected.

## EPICS propagation behavior

When xrtGlow is started with `epicsPrefix`, the dynamic propagation worker
enables histogram generation for recorded propagation calls that already have a
`withHistogram` argument in the beamline flow. In practice, this changes
recorded `Screen.expose(..., withHistogram=False)` calls to
`withHistogram=True` during EPICS-driven dynamic propagation.

This is intentional: screen image PVs are only updated when screen histograms
are built. The change is limited to xrtGlow runs with EPICS enabled. It does
not change the Python default of `Screen.expose()`, non-EPICS scripts, or
propagation methods whose recorded flow kwargs do not include `withHistogram`.

## Command options

Run the generator with:

```bash
python -m xrt.backends.raycing.epics.generate_bob [options]
```

| Option | Meaning |
| --- | --- |
| `-h`, `--help` | Show the command help and exit. |
| `--beamline MODULE_OR_FILE[:CALLABLE]` | Load a Python beamline builder and generate one BOB file per EPICS-enabled element. The callable defaults to `build_beamline` when omitted. Examples: `my_beamline.py:build_beamline`, `package.module:create_bl`. |
| `--layout PATH` | Load an xrtQook XML or JSON layout and generate one BOB file per EPICS-enabled element. |
| `--builder-kw NAME=JSON` | Pass a JSON-decoded keyword argument to the Python beamline builder. Can be repeated. Example: `--builder-kw nrays=1000 --builder-kw hkl='[1, 1, 1]'`. |
| `--epics-map PATH` | Load an `EpicsDevice` record-remapping JSON file. Keys are default record suffixes, values are replacement suffixes, matching the `epicsMap` argument of `EpicsDevice`. |
| `--output PATH` | Choose the output folder. Default: `xrt/backends/raycing/epics`. |
| `--prefix PREFIX` | Set the default value of the `$(P)` macro. A trailing colon is added when `PREFIX` is non-empty and does not already end with `:`. Example: `--prefix BL` produces `BL:`. |
| `--templates` | Generate only the generic category templates. This is also the default when neither `--beamline` nor `--layout` is supplied. It cannot be combined with `--beamline`, `--layout` or `--list`. |
| `--list` | Print the element categories and EPICS record suffixes without writing BOB files. Requires `--beamline` or `--layout`. |
| `--backend {auto,phoebusgen,xml}` | Select the writer backend. `auto` uses `phoebusgen` when installed and otherwise writes Display Builder XML directly. `phoebusgen` requires the package. `xml` always uses the built-in XML writer. |

`--beamline` and `--layout` are alternatives. Use exactly one of them when
generating or listing records for a concrete beamline.

On Windows, avoid ending a double-quoted path with a backslash. Native command
line parsing can treat the final backslash as escaping the closing quote, so
the quote becomes part of the path. Prefer one of these forms:

```powershell
--output "c:\xrt_beamlines\beamline1"
--output "c:\xrt_beamlines\beamline1\."
--output "c:/xrt_beamlines/beamline1/"
```

Avoid:

```powershell
--output "c:\xrt_beamlines\beamline1\"
```

## Examples

Generate generic templates into the package folder:

```bash
python -m xrt.backends.raycing.epics.generate_bob
```

Generate generic templates explicitly into another folder:

```bash
python -m xrt.backends.raycing.epics.generate_bob \
    --templates \
    --output /tmp/xrt-epics-bobs
```

List the records exposed for a Qook layout:

```bash
python -m xrt.backends.raycing.epics.generate_bob \
    --layout examples/withRaycing/_QookBeamlines/lens1.xml \
    --list
```

Generate per-element screens for a Qook layout:

```bash
python -m xrt.backends.raycing.epics.generate_bob \
    --layout examples/withRaycing/_QookBeamlines/lens1.xml \
    --prefix BL
```

Generate per-element screens from a Python builder with arguments:

```bash
python -m xrt.backends.raycing.epics.generate_bob \
    --beamline path/to/beamline.py:build_beamline \
    --builder-kw nrays=10000 \
    --prefix BL \
    --output build/phoebus
```

Use an EPICS map:

```bash
python -m xrt.backends.raycing.epics.generate_bob \
    --layout path/to/layout.xml \
    --epics-map path/to/epics_map.json
```
