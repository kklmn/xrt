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
    const float PI = 3.1415926535897932384626433832795;
    unsigned int ii = get_global_id(0);
    int j;

    float3 beta, betaP, n, nnb;
    float2 eucos;
    float ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, gamma2, krel;
    float2 zero2 = (float2)(0,0);
    float2 Is = zero2;
    float2 Ip = zero2;
    float wgwu = w[ii] / gamma[ii] / wu[ii];
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    gamma2 = gamma[ii]*gamma[ii];

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - 0.5*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);
        ucos = ww1[ii] * tg[j] + wgwu *
           (-Ky * ddphi[ii] * sintg + Kx * ddpsi[ii] * sintgph +
            0.25 / gamma[ii] * (Ky2 * sintg*costg + Kx2 * sintgph*costgph));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        beta.x = Ky / gamma[ii] * costg;
        beta.y = -Kx / gamma[ii] * costgph;
        //beta.z = sqrt(1. - 1./gamma2 - beta.x*beta.x - beta.y*beta.y);
        beta.z = 1 - 0.5*(1./gamma2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wu[ii] / gamma[ii] * sintg;
        betaP.y = Kx * wu[ii] / gamma[ii] * sintgph;
        betaP.z = wu[ii] /gamma2*(Ky2*sintg*costg + Kx2*sintgph*costgph);
        //betaP.z = wu[ii]/beta.z/gamma2*(Ky2*sintg*costg + Kx2*sintgph*costgph);

        krel = 1. - dot(n, beta);
        nnb = cross(n, cross((n - beta), betaP))/krel/krel;
        //nnb = cross(n, cross(n, beta));

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }
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
                              __global float* ddphi,
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

    float2 eucos;
    float3 n, nnb, beta, betaP;
    float ucos, sintg, sin2tg, costg, sintgph, costgph, krel;
    float2 zero2 = (float2)(0,0);
    float2 Is = zero2;
    float2 Ip = zero2;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float alphaS = alpha * C / wu[ii] / E2W;
    float wgwu = w[ii] / gamma[ii] / wu[ii];
    float gamma2 = gamma[ii] * gamma[ii];
    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - 0.5*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);
        sin2tg = 2. * sintg * costg; //sin(2 * tg[j]);
        ucos = ww1[ii] * tg[j] + wgwu *
            (-Ky * n.x *
                  (sintg + alphaS * (1. - costg - tg[j] * sintg)) +
             Kx * n.y * sin(tg[j] + phase) +
             0.125 / gamma[ii] *
                  (Ky2 * (sin2tg - 2. * alphaS *
                    (tg[j] * tg[j] + costg * costg + tg[j] * sin2tg)) +
                   Kx2 * sin(2. * (tg[j] + phase))));

        eucos.x = cos(ucos);
        eucos.y = sin(ucos);

        beta.x = Ky / gamma[ii] * costg * (1 - alphaS * tg[j]);
        beta.y = -Kx / gamma[ii] * cos(tg[j] + phase);
        beta.z = 1 - 0.5*(1./gamma2 + beta.x*beta.x + beta.y*beta.y); 

        betaP.x = -Ky * wu[ii] / gamma[ii] * 
            (alphaS*costg + (1 - alphaS*tg[j])*sintg);
        betaP.y = Kx * wu[ii] / gamma[ii] * sintgph;
        betaP.z = wu[ii] /gamma2 * 
            (Kx2*sintgph*costgph + Ky2*(1 - alphaS*tg[j])*
             (alphaS * costg * costg + (1 - alphaS*tg[j])*sintg*costg));

        krel = 1. - dot(n, beta);
        nnb = cross(n, cross((n - beta), betaP))/krel/krel;
        //nnb = cross(n, cross(n, beta));

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

        //Is += ag[j] * (ddtheta[ii] - beta.x *
        //    (1 - alphaS * tg[j])) * eucos;
        //Ip += ag[j] * (ddpsi[ii] - beta.y) * eucos; }
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
    float ucos, sintg, costg, sintgph, costgph, krel;
    float2 zero2 = (float2)(0,0);
    float2 Is = zero2;
    float2 Ip = zero2;
    float3 r, r0, n, nnb, beta, betaP;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float gamma2 = gamma[ii] * gamma[ii];
    float betam = 1 - (1 + 0.5 * Kx2 + 0.5 * Ky2) / 2. / gamma2;
    float wR0 = R0 * PI2 / L0;

    r0.x = wR0 * tan(ddphi[ii]);
    r0.y = wR0 * tan(ddpsi[ii]);
    r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii]));

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - 0.5*(n.x*n.x + n.y*n.y);

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
        //beta.z = sqrt(1. - 1./gamma2 - beta.x*beta.x - beta.y*beta.y);
        beta.z = 1 - 0.5*(1./gamma2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wu[ii] / gamma[ii] * sintg;
        betaP.y = Kx * wu[ii] / gamma[ii] * sintgph;
        betaP.z = wu[ii] / gamma2 * (Ky2*sintg*costg + Kx2*sintgph*costgph);
        //betaP.z = wu[ii]/beta.z/gamma2*(Ky2*sintg*costg + Kx2*sintgph*costgph);

        krel = 1. - dot(n, beta);
        nnb = cross(n, cross((n - beta), betaP))/krel/krel;
        //nnb = cross(n, cross(n, beta));

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}
/*
__kernel void undulator_nf_byparts(const float alpha,
                        const float Kx,
                        const float Ky,
                        const float phase,
                        const int jend,
                        const float alim,
                        const float blim,
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

    float2 eg;
    float f, fP, varx;
    float g, gP, gPP, sing, cosg, sintgph, costgph;
    float sintg, costg;
    float sinph, cosph;
    float2 zero2 = (float2)(0,0);
    float2 Is = zero2;
    float2 Ip = zero2;
    float gam = gamma[ii];
    float wwu = w[ii] / wu[ii];
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float Ky2Kx2;
    float gamma2 = gam * gam;
    float phi = ddphi[ii];
    float psi = ddpsi[ii];
    float3 r, rP, rPP, n;
    float betam = 1 - (0.5 + 0.25*Kx2 + 0.25*Ky2) / gamma2;


    //n.x = phi;
    //n.y = psi;
    //n.z = sqrt(1 - phi*phi - psi*psi);

    r0.x = wR0 * tan(ddphi[ii]);
    r0.y = wR0 * tan(ddpsi[ii]);
    r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii]));

    sinph = sincos(phase, &cosph);
    for (j=0; j<jend; j++) {
        varx = tg[j];
        sintg = sincos(varx, &costg);
        sintgph = sintg*cosph + costg*sinph;
        costgph = costg*cosph - sintg*sinph;
        Ky2Kx2 = (Ky2 * sintg*costg + Kx2 * sintgph*costgph) / gamma2;

        r.x = Ky / gam * sintg;
        r.y = -Kx / gam * sintgph;
        r.z = betam * varx - 0.25 * Ky2Kx2;

        g = wwu * (varx + length(r0 - r));

        rP.x = Ky / gam * costg;
        rP.y = -Kx / gam * costgph;
        rP.z = betam - 0.25 / gamma2 *
            (Ky2 * (costg*costg - sintg*sintg) +
             Kx2 * (costgph*costgph - sintgph*sintgph));
        gP = wwu * (1 + dot(n, rP));

        rPP.x = -r.x;
        rPP.y = -r.y;
        rPP.z = Ky2Kx2;
        gPP = -wwu * dot(n, rPP);

        sing = sincos(g, &cosg);
        eg.x = -sing;
        eg.y = cosg;

        f = phi - Ky / gam * costg;
        fP = Ky / gam * sintg;
        Is += ag[j] * (fP - f*gPP/gP)/gP * eg;

        f = psi + Kx / gam * costgph;
        fP = -Kx / gam * sintgph;
        Ip += ag[j] * (fP - f*gPP/gP)/gP * eg;}


    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
    }
*/

float2 f_beta(float Bx, float By, float Bz,
               float emcg,
               float2 beta)
{
    return emcg*(float2)(beta.y*Bz - By,
                          Bx - beta.x*Bz);
}

float3 f_traj(float revgamma, float2 beta)
{
    return (float3)(beta.x,
                     beta.y,
                     sqrt(revgamma-beta.x*beta.x-beta.y*beta.y));
}

float2 next_beta_rk(float2 beta, int iZeroStep, int iHalfStep,
                     int iFullStep, float rkStep, float emcg,
                     __global float* Bx,
                     __global float* By,
                     __global float* Bz)
{
    float2 k1Beta, k2Beta, k3Beta, k4Beta;

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

float8 next_traj_rk(float2 beta, float3 traj, int iZeroStep, int iHalfStep,
                     int iFullStep, float rkStep, float emcg,
                     float revgamma,
                     __global float* Bx,
                     __global float* By,
                     __global float* Bz)
{
    float2 k1Beta, k2Beta, k3Beta, k4Beta;
    float3 k1Traj, k2Traj, k3Traj, k4Traj;

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

    return (float8)(beta + (k1Beta + 2.*k2Beta + 2.*k3Beta + k4Beta)/6.,
                     traj + (k1Traj + 2.*k2Traj + 2.*k3Traj + k4Traj)/6.,
                     0., 0., 0.);
}


__kernel void undulator_custom(const int jend,
                                const int nwt,
                                const float lUnd,
                                __global float* gamma,
                                __global float* w,
                                __global float* ddphi,
                                __global float* ddpsi,
                                __global float* tg,
                                __global float* ag,
                                __global float* Bx,
                                __global float* By,
                                __global float* Bz,
                                __global float2* Is_gl,
                                __global float2* Ip_gl)
{
    const float E2W = 1.51926751475e15;
    const float C = 2.99792458e11;
    const float SIE0 = 1.602176565e-19;
    const float SIM0 = 9.10938291e-31;
    //const float PI = 3.1415926535897932384626433832795;
    const float PI2 = 6.283185307179586476925286766559;

    unsigned int ii = get_global_id(0);
    int j, k, jb;
    int iBase, iZeroStep, iHalfStep, iFullStep;

    float ucos, rkStep, wu_int, betam_int, krel;
    float emcg = lUnd * SIE0 / SIM0 / C / gamma[ii] / PI2;
    float gamma2 = gamma[ii]*gamma[ii];
    float revgamma = 1. - 1./gamma2;
    float8 betaTraj;
    float2 eucos;
    float2 zero2 = (float2)(0,0);
    float2 Is = zero2;
    float2 Ip = zero2;
    float2 beta, beta0;
    float3 traj, n, traj0, betaC, betaP, nnb;

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
    traj = (float3)(0., 0., 0.);
    traj0 = (float3)(0., 0., 0.);

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

        jb = 2*j*nwt;

        betaC.x = beta.x;
        betaC.y = beta.y;
        betaC.z = 1 - 0.5*(1./gamma2 + betaC.x*betaC.x + betaC.y*betaC.y);

        betaP.x = wu_int * emcg * (betaC.y*Bz[jb] - By[jb]);
        betaP.y = wu_int * emcg * (-betaC.x*Bz[jb] + Bx[jb]);
        betaP.z = wu_int * emcg * (betaC.x*By[jb] - betaC.y*Bx[jb]); 

        krel = 1. - dot(n, betaC);
        nnb = cross(n, cross((n - betaC), betaP))/krel/krel;
        //nnb = cross(n, cross(n, beta));

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

        //Is += ag[j] * (ddphi[ii] - beta.x) * eucos;
        //Ip += ag[j] * (ddpsi[ii] - beta.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void get_trajectory(const int jend,
                             const int nwt,
                             const float emcg,
                             const float gamma,
                             __global float* tg,
                             __global float* Bx,
                             __global float* By,
                             __global float* Bz,
                             __global float* betax,
                             __global float* betay,
                             __global float* betazav,
                             __global float* trajx,
                             __global float* trajy,
                             __global float* trajz)
{
    int j, k;
    int iBase, iZeroStep, iHalfStep, iFullStep;

    float rkStep;
    float revgamma = 1. - 1./gamma/gamma;
    float betam_int = 0;
    float2 zero2 = (float2)(0,0);
    float2 beta, beta0;
    float3 traj, traj0;
    float8 betaTraj;

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
    traj = (float3)(0., 0., 0.);
    traj0 = (float3)(0., 0., 0.);

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
                                        const int nwt,
                                        const float emcg,
                                        const float gamma2,
                                        const float wu,
                                        const float L0,
                                        const float R0,
                                        __global float* w,
                                        __global float* ddphi,
                                        __global float* ddpsi,
                                        __global float* tg,
                                        __global float* ag,
                                        __global float* Bx,
                                        __global float* By,
                                        __global float* Bz,
                                        __global float* betax,
                                        __global float* betay,
                                        __global float* trajx,
                                        __global float* trajy,
                                        __global float* trajz,
                                        __global float2* Is_gl,
                                        __global float2* Ip_gl)
{
    const float PI2 = 6.283185307179586476925286766559;

    unsigned int ii = get_global_id(0);
    int j, jb;

    float ucos, wR0, krel;

    float2 eucos;
    float2 Is = (float2)(0., 0.);
    float2 Ip = (float2)(0., 0.);

    float3 traj, n, r0, betaC, betaP, nnb;

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

        jb = 2*j*nwt;

        betaC.x = betax[j];
        betaC.y = betay[j];
        betaC.z = 1 - 0.5*(1./gamma2 + betaC.x*betaC.x + betaC.y*betaC.y);

        betaP.x = wu * emcg * (betaC.y*Bz[jb] - By[jb]);
        betaP.y = wu * emcg * (-betaC.x*Bz[jb] + Bx[jb]);
        betaP.z = wu * emcg * (betaC.x*By[jb] - betaC.y*Bx[jb]); 

        krel = 1. - dot(n, betaC);
        nnb = cross(n, cross((n - betaC), betaP))/krel/krel;
        //nnb = cross(n, cross(n, beta));

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

        //Is += ag[j] * (n.x - betax[j]) * eucos;
        //Ip += ag[j] * (n.y - betay[j]) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}