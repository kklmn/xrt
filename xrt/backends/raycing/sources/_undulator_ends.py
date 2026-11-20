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
    m = source.terminationPeriods
    kx, ky = source.Kx, source.Ky
    t, weights = source.tg, source.ag
    st, ct = source.sintg, source.costg
    sp, cp = source.sintgph, source.costgph
    body = np.column_stack((t, weights, ky*ct, -kx*cp, -ky*st, kx*sp,
                            ky*st, -kx*sp, (kx*kx + ky*ky)*t/2 +
                            (ky*ky*st*ct + kx*kx*sp*cp)/2))
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
        return np.column_stack((side*u, weights, px, py, dpx, dpy, ix, iy, ip2))

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


def fields(grids, periods, q, w, wu, gamma, theta, psi, sum_over_nodes):
    """Integrate both polarizations, vectorizing over nodes or over rays.

    With delta = gamma*(theta, psi) - p and D = 1 + |delta|², the
    acceleration field is exp(i*phase) d(2*gamma*delta/D)/dr. Its phase
    derivative is a*D. Expanding the latter gives the three tabulated
    orbit primitives and avoids subtracting two nearly equal velocities.
    """
    q, w, wu, gamma, theta, psi = np.broadcast_arrays(q, w, wu, gamma, theta, psi)
    body, exit_phase = interference(q, periods)
    h, v = gamma*theta, gamma*psi
    a = w/(2*wu*gamma**2)
    linear = 1 + h*h + v*v
    if not sum_over_nodes:
        h, v, a, linear = (x[..., None] for x in (h, v, a, linear))

    def integrand(row):
        r, weight, px, py, dpx, dpy, ix, iy, ip2 = row
        dx, dy = h - px, v - py
        den = 1 + dx*dx + dy*dy  # 2*gamma²*(1 - n.beta) to paraxial order
        dot = 2*(dx*dpx + dy*dpy)
        phase = a*(linear*r - 2*(h*ix + v*iy) + ip2)
        wave = weight*np.exp(1j*phase)/(den*den)
        return wave*(dx*dot - den*dpx), wave*(dy*dot - den*dpy)

    es, ep = 0j, 0j
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
                sum_over_nodes):
    """Finite-distance acceleration field with xrt's angular-flux normalization."""
    w, wu, gamma, theta, psi = np.broadcast_arrays(w, wu, gamma, theta, psi)
    k = w/wu
    screen, carrier = near_screen(distance, theta, psi, k)
    scale = 2*gamma*wu*carrier
    nx0, ny0, nz0_loss, radius = screen
    if not sum_over_nodes:
        gamma, k, nx0, ny0, nz0_loss, radius = (
            x[..., None] for x in (gamma, k, nx0, ny0, nz0_loss, radius))
    screen_x, screen_y = nx0*radius, ny0*radius
    gamma2 = gamma*gamma

    def integrand(row, offset):
        r, weight, px, py, dpx, dpy, ix, iy, ip2 = row
        s = offset + r
        # int p^2 advances by 2*pi*mean_p2 per body period. The other
        # primitives already contain their periodic values at the joins.
        slip = (s + ip2 + mean_p2*offset)/(2*gamma2)  # s - z
        x, y, z = ix/gamma, iy/gamma, s - slip
        rx, ry, rz = screen_x - x, screen_y - y, (distance - s) + slip
        length = np.sqrt(rx*rx + ry*ry + rz*rz)
        nx, ny, nz = rx/length, ry/length, rz/length

        # Rationalize L - (R - s) using L^2 - (R - s)^2. This keeps the
        # node-dependent phase accurate even when the screen is very far away.
        plane = slip + nz0_loss*z - nx0*x - ny0*y  # s - n0.rho
        gap = radius - s
        path = (2*radius*plane + x*x + y*y - (s + z)*slip) / (
            length + np.maximum(gap, 0.))
        phase = k*np.where(gap >= 0, path, length - gap)

        # Evaluate the transverse triple product with the local sightline.
        # den = 2*gamma^2*(1 - n.beta), with beta_z expanded as in undulator_nf.
        loss = (nx*nx + ny*ny)/(1 + abs(nz)) + 2*np.maximum(-nz, 0.)
        dx, dy = gamma*nx - px, gamma*ny - py
        den = 1 + dx*dx + dy*dy + loss*(gamma2*loss - 1 - px*px - py*py)
        proj = 2*((dx + loss*px)*dpx + (dy + loss*py)*dpy)
        wave = weight*np.exp(1j*phase)/(den*den)
        return wave*(dx*proj - den*dpx), wave*(dy*proj - den*dpy)

    body, entrance, exit_grid = grids
    es, ep = _sum_grid(entrance, integrand, sum_over_nodes, -np.pi*periods)
    for i in range(int(periods)):
        center = np.pi*(2*i + 1 - periods)
        sx, sy = _sum_grid(body, integrand, sum_over_nodes, center)
        es += sx
        ep += sy
    sx, sy = _sum_grid(exit_grid, integrand, sum_over_nodes, np.pi*periods)
    return scale*(es + sx), scale*(ep + sy)
