"""Surface appearance presets for Glow's Phong shaders."""

from typing import NamedTuple


class PhongMaterial(NamedTuple):
    ambient: tuple
    diffuse: tuple
    specular: tuple
    shininess: float


# Preserve the original appearance when an object uses renderColor='auto'.
AUTO_MATERIALS = {
    'Cu': PhongMaterial((0.8, 0.4, 0.15), (0.50, 0.25, 0.12),
                        (1., 0.9, 0.8), 100.),
    'Si': PhongMaterial((2 * 0.29225,) * 3, (0.50754,) * 3,
                        (1., 0.9, 0.8), 100.),
    'Quartz': PhongMaterial((0.84, 0.85, 0.87), (0.56, 0.62, 0.70),
                            (0.92, 0.96, 1.), 180.),
    'Screen': PhongMaterial((2 * 0.29225, 2.2 * 0.29225, 1.8 * 0.29225),
                            (0.50754, 1.1 * 0.50754, 0.9 * 0.50754),
                            (0.9, 1., 0.8), 80.),
    'Steel': PhongMaterial((0.42 * 2, 0.44 * 1.75, 0.46 * 1.75),
                           (0.38 * 2, 0.40 * 1.75, 0.42 * 1.75),
                           (0.72, 0.76, 0.80), 135.),
}

SELECTED_AMBIENT = (0.89225, 0.89225, 0.49225)

# Linear sRGB colors from Anton Palmqvist's Physically Based API v2 (CC0).
# https://github.com/AntonPalmqvist/physically-based-api/blob/main/deploy/v2/materials.json
# Roughness values for named finishes are chosen for Glow's lighting.
# None for IOR denotes a conductor, whose color supplies the specular term.
_SURFACES = {
    # name: (linear RGB, roughness, dielectric IOR)
    'Aluminum': ((0.916, 0.923, 0.924), 0., None),
    'Aluminum (machined)': ((0.916, 0.923, 0.924), 0.42, None),
    'Aluminum (black anodized)': ((0.05, 0.05, 0.05), 0.3, 1.5),
    'Aluminum (red anodized)': ((0.6, 0., 0.), 0.3, 1.5),
    'Brass': ((0.91, 0.778, 0.423), 0., None),
    'Cast Iron': ((0.027, 0.026, 0.026), 0.3, None),
    'Chromium': ((0.654, 0.685, 0.701), 0., None),
    'Concrete': ((0.51, 0.51, 0.51), 0.5, 1.5),
    'Concrete (sealed)': ((0.51, 0.51, 0.51), 0.25, 1.5),
    'Copper': ((0.932, 0.623, 0.522), 0., None),
    'Germanium': ((0.5, 0.517, 0.465), 0., None),
    'Gold': ((1.059, 0.773, 0.307), 0., None),
    'Iridium': ((0.745, 0.734, 0.704), 0., None),
    'Marble': ((0.83, 0.791, 0.753), 0., 1.5),
    'Nickel': ((0.697, 0.641, 0.563), 0., None),
    'Palladium': ((0.734, 0.7036, 0.662), 0., None),
    'Platinum': ((0.765, 0.73, 0.676), 0., None),
    'Porcelain': ((0.745, 0.745, 0.723), 0., 1.5),
    'Quartz': ((1., 1., 1.), 0., 1.54),
    'Silicon': ((0.345, 0.369, 0.426), 0., None),
    'Silver': ((0.991, 0.985, 0.974), 0., None),
    'Stainless Steel (polished)': ((0.669, 0.639, 0.598), 0., None),
    'Stainless Steel (satin)': ((0.669, 0.639, 0.598), 0.45, None),
    'Titanium': ((0.441, 0.4, 0.361), 0., None),
    'Tungsten': ((0.537, 0.536, 0.519), 0., None),

    # Approximate neutral colors for additional engineering materials.
    'Granite (honed)': ((0.18, 0.17, 0.16), 0.65, 1.5),
    'Granite (polished)': ((0.18, 0.17, 0.16), 0.2, 1.5),
    'Graphite': ((0.035, 0.035, 0.035), 0.65, 1.5),
    # Normal-incidence reflectance at 590 nm from Weaver et al.
    # https://doi.org/10.1103/PhysRevB.15.4115
    'Rhodium': ((0.78, 0.78, 0.78), 0., None),
}


def _to_phong(color, roughness, ior):
    color = tuple(min(channel, 1.) for channel in color)
    if ior is None:
        diffuse = (0., 0., 0.)
        specular = color
    else:
        diffuse = color
        specular = (((ior - 1.) / (ior + 1.)) ** 2,) * 3
    # The ambient term keeps conductors visible under Glow's fixed lighting.
    shininess = max(1., 128. * (1. - roughness) ** 2)
    return PhongMaterial(color, diffuse, specular, shininess)


RENDER_MATERIALS = {name: _to_phong(*surface)
                    for name, surface in _SURFACES.items()}
RENDER_MATERIALS['Steel'] = AUTO_MATERIALS['Steel']
RENDER_COLOR_CHOICES = ('auto', *sorted(RENDER_MATERIALS))


def get_render_material(name, default):
    """Resolve an object's preset, falling back to its automatic appearance."""
    if isinstance(name, str) and name in RENDER_MATERIALS:
        return RENDER_MATERIALS[name]
    return AUTO_MATERIALS[default]
