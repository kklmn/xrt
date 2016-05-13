//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "10 Apr 2015"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

__kernel void undulator(const float alpha,
                        const float Kx,
                        const float Ky,
                        const float phase,
                        const int jend,
                        __global float* gamma,
                        __global float* wu,
                        __global float* w,
                        __global float* ww1,
                        __global float* ddphi,
                        __global float* ddpsi,
                        __global float* tg,
                        __global float* ag,
                        __global float2* Is_gl,
                        __global float2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    float2 beta;
    float2 eucos;
    float ucos, sinucos, cosucos, sintg, costg, sintgph, costgph;
    float2 zero2 = (float2)(0,0);
    float2 Is = zero2;
    float2 Ip = zero2;
    float wgwu = w[ii] / gamma[ii] / wu[ii];
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);
        ucos = (float)(ww1[ii]) * tg[j] + wgwu *
           (-Ky * ddphi[ii] * sintg + Kx * ddpsi[ii] * sintgph +
            0.25 / gamma[ii] * (Ky2 * sintg*costg + Kx2 * sintgph*costgph));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        beta.x = Ky / gamma[ii] * costg;
        beta.y = -Kx / gamma[ii] * costgph;

        Is += (ag[j] * (ddphi[ii] - beta.x)) * eucos;
        Ip += (ag[j] * (ddpsi[ii] - beta.y)) * eucos; }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
    }

__kernel void undulator_taper(const float alpha,
                              const float Kx,
                              const float Ky,
                              const float phase,
                              const int jend,
                              __global float* gamma,
                              __global float* wu,
                              __global float* w,
                              __global float* ww1,
                              __global float* ddtheta,
                              __global float* ddpsi,
                              __global float* tg,
                              __global float* ag,
                              __global float2* Is_gl,
                              __global float2* Ip_gl)
{
    const float E2W = 1.51926751475e15;
    const float C = 2.99792458e11;
    unsigned int ii = get_global_id(0);
    int j;

    float2 eucos, beta;
    float ucos, sintg, sin2tg, costg;
    float2 zero2 = (float2)(0,0);
    float2 Is = zero2;
    float2 Ip = zero2;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float alphaS = alpha * C / wu[ii] / E2W;
    float wgwu = w[ii] / gamma[ii] / wu[ii];

    for (j=0; j<jend; j++) {
        sintg = sin(tg[j]);
        costg = cos(tg[j]);
        sin2tg = 2. * sintg * costg; //sin(2 * tg[j]);
        ucos = ww1[ii] * tg[j] + wgwu *
            (-Ky * ddtheta[ii] *
                  (sintg + alphaS * (1. - costg - tg[j] * sintg)) +
             Kx * ddpsi[ii] * sin(tg[j] + phase) +
             0.125 / gamma[ii] *
                  (Ky2 * (sin2tg - 2. * alphaS *
                    (tg[j] * tg[j] + costg * costg + tg[j] * sin2tg)) +
                   Kx2 * sin(2. * (tg[j] + phase))));

        eucos.x = cos(ucos);
        eucos.y = sin(ucos);

        beta.x = Ky / gamma[ii] * costg;
        beta.y = -Kx / gamma[ii] * cos(tg[j] + phase);

        Is += ag[j] * (ddtheta[ii] - beta.x *
            (1 - alphaS * tg[j])) * eucos;
        Ip += ag[j] * (ddpsi[ii] - beta.y) * eucos; }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void undulator_nf(const float R0,
                            const float L0,
                            const float Kx,
                            const float Ky,
                            const float phase,
                            const int jend,
                            __global float* gamma,
                            __global float* wu,
                            __global float* w,
                            __global float* ww1,
                            __global float* ddphi,
                            __global float* ddpsi,
                            __global float* tg,
                            __global float* ag,
                            __global float2* Is_gl,
                            __global float2* Ip_gl)
{
    //const float E2W = 1.51926751475e15;
    //const float C = 2.99792458e11;
    const float PI2 = 6.283185307179586476925286766559;
    unsigned int ii = get_global_id(0);
    int j;

    float2 eucos;
    float ucos, sintg, costg, sintgph, costgph;
    float2 zero2 = (float2)(0,0);
    float2 Is = zero2;
    float2 Ip = zero2;
    float3 r, r0;
    float2 beta;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float gamma2 = gamma[ii] * gamma[ii];
    float betam = 1 - (1 + 0.5 * Kx2 + 0.5 * Ky2) / 2. / gamma2;
    float wR0 = R0 * PI2 / L0;

    r0.x = wR0 * tan(ddphi[ii]);
    r0.y = wR0 * tan(ddpsi[ii]);
    r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii]));

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);

        r.x = Ky / gamma[ii] * sintg;
        r.y = -Kx / gamma[ii] * sintgph;
        r.z = betam * tg[j] - 0.25 / gamma2 *
        (Ky2 * sintg * costg + Kx2 * sintgph * costgph);

      ucos = w[ii] / wu[ii] * (tg[j] + length(r0 - r));

      eucos.x = cos(ucos);
      eucos.y = sin(ucos);

      beta.x = Ky / gamma[ii] * costg;
      beta.y = -Kx / gamma[ii] * costgph;

      Is += ag[j] * (ddphi[ii] - beta.x) * eucos;
      Ip += ag[j] * (ddpsi[ii] - beta.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}


double2 f_beta(double Bx, double By, double Bz,
               double emcg,
               double2 beta)
{
    return emcg*(double2)(beta.y*Bz - By,
                          Bx - beta.x*Bz);
}

double3 f_traj(double revgamma, double2 beta)
{
    return (double3)(beta.x,
                     beta.y,
                     sqrt(revgamma-beta.x*beta.x-beta.y*beta.y));
}

double2 next_beta_rk(double2 beta, int iZeroStep, int iHalfStep,
                     int iFullStep, double rkStep, double emcg,
                     __global double* Bx,
                     __global double* By,
                     __global double* Bz)
{
    double2 k1Beta, k2Beta, k3Beta, k4Beta;

    k1Beta = rkStep * f_beta(Bx[iZeroStep],
                             By[iZeroStep],
                             Bz[iZeroStep],
                             emcg, beta);
    k2Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + 0.5*k1Beta);
    k3Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + 0.5*k2Beta);
    k4Beta = rkStep * f_beta(Bx[iFullStep],
                             By[iFullStep],
                             Bz[iFullStep],
                             emcg, beta + k3Beta);
    return beta + (k1Beta + 2.*k2Beta + 2.*k3Beta + k4Beta) / 6.;
}

double8 next_traj_rk(double2 beta, double3 traj, int iZeroStep, int iHalfStep,
                     int iFullStep, double rkStep, double emcg,
                     double revgamma,
                     __global double* Bx,
                     __global double* By,
                     __global double* Bz)
{
    double2 k1Beta, k2Beta, k3Beta, k4Beta;
    double3 k1Traj, k2Traj, k3Traj, k4Traj;

    k1Beta = rkStep * f_beta(Bx[iZeroStep],
                             By[iZeroStep],
                             Bz[iZeroStep],
                             emcg, beta);
    k1Traj = rkStep * f_traj(revgamma, beta);

    k2Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + 0.5*k1Beta);
    k2Traj = rkStep * f_traj(revgamma, beta + 0.5*k1Beta);

    k3Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + 0.5*k2Beta);
    k3Traj = rkStep * f_traj(revgamma, beta + 0.5*k2Beta);

    k4Beta = rkStep * f_beta(Bx[iFullStep],
                             By[iFullStep],
                             Bz[iFullStep],
                             emcg, beta + k3Beta);
    k4Traj = rkStep * f_traj(revgamma, beta + k3Beta);

    return (double8)(beta + (k1Beta + 2.*k2Beta + 2.*k3Beta + k4Beta)/6.,
                     traj + (k1Traj + 2.*k2Traj + 2.*k3Traj + k4Traj)/6.,
                     0., 0., 0.);
}


__kernel void undulator_custom(const int jend,
                                const int nwt,
                                const double lUnd,
                                __global double* gamma,
                                __global double* w,
                                __global double* ddphi,
                                __global double* ddpsi,
                                __global double* tg,
                                __global double* ag,
                                __global double* Bx,
                                __global double* By,
                                __global double* Bz,
                                __global double2* Is_gl,
                                __global double2* Ip_gl)
{
    const double E2W = 1.51926751475e15;
    const double C = 2.99792458e11;
    const double SIE0 = 1.602176565e-19;
    const double SIM0 = 9.10938291e-31;
    //const double PI = 3.1415926535897932384626433832795;
    const double PI2 = 6.283185307179586476925286766559;

    unsigned int ii = get_global_id(0);
    int j, k;
    int iBase, iZeroStep, iHalfStep, iFullStep;

    double ucos, rkStep, wu_int, betam_int;
    double emcg = lUnd * SIE0 / SIM0 / C / gamma[ii] / PI2;
    double revgamma = 1. - 1./gamma[ii]/gamma[ii];
    double8 betaTraj;
    double2 eucos;
    double2 zero2 = (double2)(0,0);
    double2 Is = zero2;
    double2 Ip = zero2;
    double2 beta, beta0;
    double3 traj, n, traj0;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    n.z = 1. - 0.5*(n.x*n.x + n.y*n.y);

    beta = zero2;
    beta0 = zero2;
    betam_int = 0;

    for (j=1; j<jend; j++) {
        iBase = 2*(j-1)*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            beta = next_beta_rk(beta, iZeroStep, iHalfStep, iFullStep,
                                rkStep, emcg, Bx, By, Bz);
            beta0 += beta * rkStep; } }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    beta0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = (double3)(0., 0., 0.);
    traj0 = (double3)(0., 0., 0.);

    for (j=1; j<jend; j++) {
        iBase = (j-1)*2*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            betaTraj = next_traj_rk(beta, traj,
                                    iZeroStep, iHalfStep, iFullStep,
                                    rkStep, emcg, revgamma, Bx, By, Bz);
            beta = betaTraj.s01;
            traj = betaTraj.s234;
            traj0 += traj * rkStep;
            betam_int += rkStep * sqrt(revgamma - beta.x*beta.x -
                                       beta.y*beta.y); } }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    traj0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = traj0;
    betam_int /= (tg[jend-1] - tg[0]);
    wu_int = PI2 * C * betam_int / lUnd / E2W;

    for (j=1; j<jend; j++) {
        iBase = 2*(j-1)*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            betaTraj = next_traj_rk(beta, traj,
            iZeroStep, iHalfStep, iFullStep,
            rkStep, emcg, revgamma, Bx, By, Bz);
            beta = betaTraj.s01;
            traj = betaTraj.s234; }

        mem_fence(CLK_LOCAL_MEM_FENCE);
        ucos = w[ii] / wu_int * (tg[j]  - dot(n, traj));
        eucos.x = cos(ucos);
        eucos.y = sin(ucos);
        Is += ag[j] * (ddphi[ii] - beta.x) * eucos;
        Ip += ag[j] * (ddpsi[ii] - beta.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void get_trajectory(const int jend,
                             const int nwt,
                             const double emcg,
                             const double gamma,
                             __global double* tg,
                             __global double* Bx,
                             __global double* By,
                             __global double* Bz,
                             __global double* betax,
                             __global double* betay,
                             __global double* betazav,
                             __global double* trajx,
                             __global double* trajy,
                             __global double* trajz)
{
    int j, k;
    int iBase, iZeroStep, iHalfStep, iFullStep;

    double rkStep;
    double revgamma = 1. - 1./gamma/gamma;
    double betam_int = 0;
    double2 zero2 = (double2)(0,0);
    double2 beta, beta0;
    double3 traj, traj0;
    double8 betaTraj;

    beta = zero2;
    beta0 = zero2;

    for (j=1; j<jend; j++) {
        iBase = (j-1)*2*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            beta = next_beta_rk(beta, iZeroStep, iHalfStep, iFullStep,
                                rkStep, emcg, Bx, By, Bz);
            beta0 += beta * rkStep; } }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    beta0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = (double3)(0., 0., 0.);
    traj0 = (double3)(0., 0., 0.);

    for (j=1; j<jend; j++) {
        iBase = 2*(j-1)*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            betaTraj = next_traj_rk(beta, traj,
                                    iZeroStep, iHalfStep, iFullStep,
                                    rkStep, emcg, revgamma, Bx, By, Bz);
            beta = betaTraj.s01;
            traj = betaTraj.s234;
            traj0 += traj * rkStep;
            betam_int += rkStep*sqrt(revgamma - beta.x*beta.x -
                                     beta.y*beta.y); } }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    traj0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = traj0;
    betam_int /= tg[jend-1] - tg[0];

    for (j=1; j<jend; j++) {
        iBase = 2*(j-1)*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            betaTraj = next_traj_rk(beta, traj,
                                    iZeroStep, iHalfStep, iFullStep,
                                    rkStep, emcg, revgamma, Bx, By, Bz);
            beta = betaTraj.s01;
            traj = betaTraj.s234; }

        betax[j] = beta.x;
        betay[j] = beta.y;
        betazav[j] = betam_int;
        trajx[j] = traj.x;
        trajy[j] = traj.y;
        trajz[j] = traj.z; }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void undulator_custom_filament(const int jend,
                                        const double wu,
                                        const double L0,
                                        const double R0,
                                        __global double* w,
                                        __global double* ddphi,
                                        __global double* ddpsi,
                                        __global double* tg,
                                        __global double* ag,
                                        __global double* betax,
                                        __global double* betay,
                                        __global double* trajx,
                                        __global double* trajy,
                                        __global double* trajz,
                                        __global double2* Is_gl,
                                        __global double2* Ip_gl)
{
    const double PI2 = 6.283185307179586476925286766559;

    unsigned int ii = get_global_id(0);
    int j;

    double ucos, wR0;

    double2 eucos;
    double2 Is = (double2)(0., 0.);
    double2 Ip = (double2)(0., 0.);

    double3 traj, n, r0;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    n.z = 1. - 0.5*(n.x*n.x + n.y*n.y);

    if (R0>0) {
        wR0 = R0 * PI2 / L0;
        r0.x = wR0 * tan(ddphi[ii]);
        r0.y = wR0 * tan(ddpsi[ii]);
        r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii])); }

    for (j=1; j<jend; j++) {
        traj.x = trajx[j];
        traj.y = trajy[j];
        traj.z = trajz[j];
        if (R0 > 0) {
            ucos = w[ii] / wu * (tg[j] + length(r0 - traj)); }
        else {
            ucos = w[ii] / wu * (tg[j] - dot(n, traj)); }
        eucos.x = cos(ucos);
        eucos.y = sin(ucos);
        Is += ag[j] * (n.x - betax[j]) * eucos;
        Ip += ag[j] * (n.y - betay[j]) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}