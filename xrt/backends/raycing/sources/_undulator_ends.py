"""Orbit and radiation quadrature for a periodic body with integrated-Hann ends."""

import numpy as np


def _end_velocity(u, periods, carrier):
    """Return W and G = U + i C, where G' = i W exp(i u), G(0) = 1."""
    length = 2*np.pi*periods
    slow = np.exp(1j*u/periods)
    window = 1 - u/length + slow.imag/(2*np.pi)
    # Integrating by parts leaves -int W' exp(i u). W' contains only
    # a constant and cos(u/m), so three Fourier primitives suffice.
    j0 = (carrier - 1)/1j
    jp = (carrier*slow - 1)/(1j*(1 + 1./periods))
    jm = (carrier*slow.conj() - 1)/(1j*(1 - 1./periods))
    return window, window*carrier + (j0 - (jp + jm)/2)/length


def _end_moments(u, periods):
    """Primitives from the join of U, C, U², C² and UC for the phase."""
    # Use the analytic velocity at Gauss nodes in short, fixed panels.
    # The highest frequency in a squared velocity is 2*(1+1/m).
    # Prefix integration is independent of the radiation grid, including
    # its coincident Clenshaw-Curtis endpoints and its order of traversal.
    edges = np.linspace(0., 2*np.pi*periods, 8*periods + 1)
    panel = np.clip(np.searchsorted(edges, u, side='right') - 1,
                    0, len(edges) - 2)
    starts = np.concatenate((edges[:-1], edges[panel]))
    widths = np.concatenate((np.diff(edges), u - edges[panel]))
    moments = np.zeros((5, len(starts)))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    for node, weight in zip(nodes, weights):
        s = starts + (node + 1)*widths/2
        _, g = _end_velocity(s, periods, np.exp(1j*s))
        x, y = g.real, g.imag
        moments += weight*np.array((x, y, x*x, y*y, x*y))
    moments *= widths/2
    n = len(edges) - 1
    prefix = np.column_stack((np.zeros(5), np.cumsum(moments[:, :n], axis=1)))
    return prefix[:, panel] + moments[:, n:]


def orbit_grids(source):
    """Rows shared by NumPy and OpenCL: r, weight, p, dp/dr, int p, int p².

    p = gamma*beta_perp. The longitudinal coordinate is a body phase in
    radians. Each end uses its own join as origin, increasing downstream.
    """
    m = int(source.terminationPeriods)
    kx, ky = source.Kx, source.Ky
    t, weights = source.tg, source.ag
    st, ct = source.sintg, source.costg
    sp, cp = source.sintgph, source.costgph
    body = np.column_stack((t, weights, ky*ct, -kx*cp, -ky*st, kx*sp,
                            ky*st, -kx*sp, (kx*kx + ky*ky)*t/2 +
                            (ky*ky*st*ct + kx*kx*sp*cp)/2))
    if source._taperVal:
        # At center c, p_x = A*f + a*b, A = 1-a*c, f = Ky*cos(t),
        # b = -t*f. Append b, b', int b, int f², int fb and int b².
        # These primitives are local to the region, avoiding large phases.
        sc = st*ct
        cos2 = ct*ct - st*st
        ff = ky**2*(t + sc)/2
        fb = -ky**2*(t*t/4 + t*sc/2 + cos2/8)
        bb = ky**2*(t**3/6 + t*t*sc/2 + t*cos2/4 - sc/4)
        body = np.column_stack((body, -ky*t*ct, ky*(t*st - ct),
                                ky*(1 - ct - t*st), ff, fb, bb))
    if source.terminatorGridScale:
        u = ((t + np.pi) + 2*np.pi*np.arange(m)[:, None]).ravel()
        carrier = -np.tile(ct + 1j*st, m)
        weights = np.tile(weights, m)
    else:
        u = m*(t + np.pi)
        carrier = np.exp(1j*u)
        weights = m*weights
    window, g = _end_velocity(u, m, carrier)
    iu, ic, iuu, icc, iuc = _end_moments(u, m)
    c, s = np.cos(source.phase), np.sin(source.phase)

    def end(side):
        sn = side*carrier.imag
        px = -ky*g.real
        py = kx*(c*g.real - side*s*g.imag)
        dpx = ky*window*sn
        dpy = -kx*window*(c*sn + s*carrier.real)
        # Carry the periodic phase primitives at t = +/-pi into both
        # ends. The linear body phase at those joins is applied separately.
        ix = -side*ky*iu
        iy = kx*(side*c*iu - s*ic + s)
        ip2 = (side*((ky*ky + (kx*c)**2)*iuu + (kx*s)**2*icc)
               - 2*kx*kx*c*s*iuc + kx*kx*c*s/2)
        rows = np.column_stack((side*u, weights, px, py, dpx, dpy, ix, iy, ip2))
        if source._taperVal:
            # p'_x = Ky*W*(A_join*sin(r) + a*cos(r)). The constants in
            # int b and int fb carry the body trajectory through the join.
            rows = np.column_stack((rows, ky*side*g.imag, ky*window*carrier.real,
                                    ky*(ic + 2), ky**2*side*iuu,
                                    -ky**2*(iuc + 1./8), ky**2*side*icc))
        return rows

    return body, end(-1), end(1)


def interference(q, periods):
    """Centered body sum and exit phase exp(i*pi*n*q), including at resonance."""
    order = np.rint(q)
    detuning = q - order
    parity = 1 - 2*np.remainder(order, 2)
    body = periods*np.sinc(periods*detuning)/np.sinc(detuning)
    rotation = np.exp(1j*np.pi*periods*detuning)
    if periods % 2:
        rotation *= parity
    else:
        body *= parity
    return body, rotation


def _sum_grid(grid, integrand, sum_over_nodes, *args):
    if not sum_over_nodes:
        fx, fy = integrand(grid.T, *args)
        return fx.sum(axis=-1), fy.sum(axis=-1)
    # Keep only one node's worth of ray arrays in memory.
    sx, sy = 0j, 0j
    for row in grid:
        fx, fy = integrand(row, *args)
        sx += fx
        sy += fy
    return sx, sy


def _regions(grids, periods):
    body, entrance, exit_grid = grids
    n = int(periods)
    yield entrance, -np.pi*n
    for i in range(n):
        yield body, np.pi*(2*i + 1 - n)
    yield exit_grid, np.pi*n


def _secular_q(offset, mean_p2, ky2, slope):
    """Secular integral of p² at the region origin, including quadratic taper."""
    change = slope*offset
    return offset*(mean_p2 + ky2/2*change*(change/3 - 1))


def _taper_node(row, slope, amplitude, ky2):
    r, weight, f, py, df, dpy, ix, iy, ip2, b, db, ib, ff, fb, bb = row
    # Expand (A*f+a*b)^2. The constant fixes the same phase origin as the
    # existing first-order tapered body. The a² term follows from beta_z.
    ip2 = (ip2 + (amplitude - 1)*(amplitude + 1)*ff +
           2*amplitude*slope*fb + slope*slope*bb - ky2*slope/4)
    return (r, weight, amplitude*f + slope*b, py, amplitude*df + slope*db,
            dpy, amplitude*ix + slope*ib, iy, ip2)


def fields(grids, periods, q, w, wu, gamma, theta, psi, sum_over_nodes,
           taper=0., ky2=0., mean_p2=0.):
    """Integrate both polarizations, vectorizing over nodes or over rays.

    With delta = gamma*(theta, psi) - p and D = 1 + |delta|², the
    acceleration field is exp(i*phase) d(2*gamma*delta/D)/dr. Its phase
    derivative is a*D. Expanding the latter gives the three tabulated
    orbit primitives and avoids subtracting two nearly equal velocities.
    """
    q, w, wu, gamma, theta, psi = np.broadcast_arrays(q, w, wu, gamma, theta, psi)
    h, v = gamma*theta, gamma*psi
    a = w/(2*wu*gamma**2)
    linear = 1 + h*h + v*v
    slope = taper/wu
    if not sum_over_nodes:
        h, v, a, linear, slope = (x[..., None] for x in (h, v, a, linear, slope))

    def integrand(row, amplitude=None):
        if taper:
            row = _taper_node(row, slope, amplitude, ky2)
        r, weight, px, py, dpx, dpy, ix, iy, ip2 = row
        dx, dy = h - px, v - py
        den = 1 + dx*dx + dy*dy  # 2*gamma²*(1 - n.beta) to paraxial order
        dot = 2*(dx*dpx + dy*dpy)
        phase = a*(linear*r - 2*(h*ix + v*iy) + ip2)
        wave = weight*np.exp(1j*phase)/(den*den)
        return wave*(dx*dot - den*dpx), wave*(dy*dot - den*dpy)

    es, ep = 0j, 0j
    if taper:
        for grid, offset in _regions(grids, periods):
            center_q = _secular_q(offset, mean_p2, ky2, slope)
            rotation = np.exp(1j*a*(linear*offset + center_q))
            if not sum_over_nodes:
                rotation = rotation[..., 0]
            sx, sy = _sum_grid(grid, integrand, sum_over_nodes, 1 - slope*offset)
            es += rotation*sx
            ep += rotation*sy
        return 2*gamma*wu*es, 2*gamma*wu*ep
    body, exit_phase = interference(q, periods)
    for grid, factor in zip(grids, (body, exit_phase.conj(), exit_phase)):
        sx, sy = _sum_grid(grid, integrand, sum_over_nodes)
        es += factor*sx
        ep += factor*sy
    return 2*gamma*wu*es, 2*gamma*wu*ep


def near_screen(distance, theta, psi, w_over_wu):
    """Center-to-pixel unit direction, 1-nz, range and propagation phasor."""
    tx, ty = np.tan(theta), np.tan(psi)
    transverse2 = tx*tx + ty*ty
    norm = np.sqrt(1 + transverse2)
    excess = transverse2/(1 + norm)  # norm - 1
    sign = np.sign(distance)
    nz_loss = excess/norm if distance >= 0 else 1 + 1/norm
    radius = abs(distance)*norm
    # Keep the transverse screen phase even when adding it to the much
    # larger axial carrier would round it away.
    axial_phase = w_over_wu*abs(distance)
    carrier = np.exp(1j*axial_phase)*np.exp(1j*(axial_phase*excess))
    return (sign*tx/norm, sign*ty/norm, nz_loss, radius), carrier


def near_fields(grids, periods, mean_p2, distance, w, wu, gamma, theta, psi,
                sum_over_nodes, taper=0., ky2=0.):
    """Finite-distance acceleration field with xrt's angular-flux normalization."""
    w, wu, gamma, theta, psi = np.broadcast_arrays(w, wu, gamma, theta, psi)
    k = w/wu
    screen, carrier = near_screen(distance, theta, psi, k)
    scale = 2*gamma*wu*carrier
    nx0, ny0, nz0_loss, radius = screen
    slope = taper/wu
    if not sum_over_nodes:
        gamma, k, nx0, ny0, nz0_loss, radius, slope = (
            x[..., None] for x in (gamma, k, nx0, ny0, nz0_loss, radius, slope))
    screen_x, screen_y = nx0*radius, ny0*radius
    gamma2 = gamma*gamma

    def path_difference(r, slip, x, y, length, sightline):
        # Rationalize L - (R - r) using L^2 - (R - r)^2.
        nx_ref, ny_ref, loss_ref, range_ref = sightline
        z = r - slip
        plane = slip + loss_ref*z - nx_ref*x - ny_ref*y
        gap = range_ref - r
        denom = length + np.maximum(gap, 0.)
        path = (2*range_ref*plane + x*x + y*y - (r + z)*slip) / np.where(
            denom == 0, 1., denom)
        return np.where(gap >= 0, path, length - gap)

    def integrand(row, screen_z, sightline, amplitude):
        if taper:
            row = _taper_node(row, slope, amplitude, ky2)
        r, weight, px, py, dpx, dpy, ix, iy, ip2 = row
        slip = (r + ip2)/(2*gamma2)
        x, y = ix/gamma, iy/gamma
        rx, ry, rz = screen_x - x, screen_y - y, (screen_z - r) + slip
        length = np.sqrt(rx*rx + ry*ry + rz*rz)
        nx, ny, nz = rx/length, ry/length, rz/length
        phase = k*path_difference(r, slip, x, y, length, sightline)

        # Evaluate the transverse triple product with the local sightline.
        # den = 2*gamma^2*(1 - n.beta), with beta_z expanded as in undulator_nf.
        loss = (nx*nx + ny*ny)/(1 + abs(nz)) + 2*np.maximum(-nz, 0.)
        dx, dy = gamma*nx - px, gamma*ny - py
        den = 1 + dx*dx + dy*dy + loss*(gamma2*loss - 1 - px*px - py*py)
        proj = 2*((dx + loss*px)*dpx + (dy + loss*py)*dpy)
        wave = weight*np.exp(1j*phase)/(den*den)
        return wave*(dx*proj - den*dpx), wave*(dy*proj - den*dpy)

    def region(grid, offset):
        # Translate the phase origin to the axis at this period center or
        # end join. Include the secular part of int p² in its axial delay.
        slip = (offset + _secular_q(offset, mean_p2, ky2, slope))/(2*gamma2)
        screen_z = (distance - offset) + slip
        length = np.sqrt(screen_x**2 + screen_y**2 + screen_z**2)
        rotation = np.exp(1j*k*path_difference(
            offset, slip, 0., 0., length, (nx0, ny0, nz0_loss, radius)))
        safe_length = np.where(length == 0, 1., length)
        nx, ny, nz = screen_x/safe_length, screen_y/safe_length, screen_z/safe_length
        loss = (nx*nx + ny*ny)/(1 + abs(nz)) + 2*np.maximum(-nz, 0.)
        sx, sy = _sum_grid(grid, integrand, sum_over_nodes, screen_z,
                           (nx, ny, loss, length), 1 - slope*offset)
        if not sum_over_nodes:
            rotation = rotation[..., 0]
        return rotation*sx, rotation*sy

    es, ep = 0j, 0j
    for grid, offset in _regions(grids, periods):
        sx, sy = region(grid, offset)
        es += sx
        ep += sy
    return scale*es, scale*ep
