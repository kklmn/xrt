//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "12 Aug 2021"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif
// "float" will be automatically replaced by "double" if precision is float64
__constant float QUAR = 0.25;
__constant float HALF = 0.5;
__constant float TWO = 2.;
__constant float SIX = 6.;
__constant float REVSIX = 1./6.;
__constant bool isHiPrecision = sizeof(TWO) == 8;
//__constant bool isHiPrecision = false;
__constant float2 zero2 = (float2)(0, 0);
__constant float3 zero3 = (float3)(0, 0, 0);
__constant float PI2 = (float)6.283185307179586476925286766559;
__constant float PI =  (float)3.1415926535897932384626433832795;

//__constant float E2W = 1.51926751475e15;
//__constant float E2W = 1519267514747457.9195337718065469;
__constant float C = 2.99792458e11;
__constant float E2WC = 5067.7309392068091;
__constant float SIE0 = 1.602176565e-19;
__constant float SIM0 = 9.10938291e-31;
__constant float EMC = 0.5866791802416487;

__constant float NEAR_CENTER = 5.0;

//__constant float RK4_STEPS[3] = {0.5, 0.5, 1.};
//__constant float RK4_B[4] = {1./6., 1./3., 1./3., 1./6.};
//__constant float RK4_A[9] = {0.5, 0.0, 0.0,
//                             0.0, 0.5, 0.0,
//                             0.0, 0.0, 1.0};
//
//__constant float RK8_B[10] =
//{41./840., 0, 0, 27./840., 272./840.,
//27./840., 216./840., 0, 216./840., 41./840.} ;
//
//__constant float RK8_A[81] = {
//4./27., 0, 0, 0, 0, 0, 0, 0, 0,
//1./18, 3./18., 0, 0, 0, 0, 0, 0, 0,
//1./12., 0, 3./12., 0, 0, 0, 0, 0, 0,
//1./8., 0, 0, 3./8., 0, 0, 0, 0, 0,
//13./54., 0, -27./54., 42./54., 8./54., 0, 0, 0, 0,
//389./4320., 0, -54./4320., 966./4320., -824./4320., 243./4320., 0, 0, 0,
//-234./20., 0, 81./20., -1164./20., 656./20., -122./20., 800./20., 0, 0,
//-127./288., 0, 18./288., -678./288., 456./288., -9./288., 576./288., 4./288., 0,
//1481./820., 0, -81./820., 7104./820., -3376./820., 72./820., -5040./820., -60./820., 720./820.};


__kernel void undulator(const float alpha,
                        const float Kx,
                        const float Ky,
//                        const float phase,
                        const int jend,
                        __global float* gamma,
                        __global float* wu,
                        __global float* w,
                        __global float* ww1,
                        __global float* ddphi,
                        __global float* ddpsi,
                        __global float* tg,
                        __global float* ag,
                        __global float* sintg,
                        __global float* costg,
                        __global float* sintgph,
                        __global float* costgph,
                        __global float2* Is_gl,
                        __global float2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    float3 beta, betaP, n, nnb;
    float2 eucos;
    float ucos, sinucos, cosucos, krel;
    float2 Is = zero2;
    float2 Ip = zero2;
    float wwu2 = w[ii] / (wu[ii] * wu[ii]);
    float revg = 1. / gamma[ii];
    float revg2 = revg * revg;
    float wug = wu[ii] * revg;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
//    n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1. - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        beta.x = Ky * revg * costg[j];
        beta.y = -Kx * revg * costgph[j];
//        beta.z = sqrt(1. - revg2 - beta.x*beta.x - beta.y*beta.y);
        beta.z = 1. - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * sintg[j];
        betaP.y = Kx * wug * sintgph[j];
        betaP.z = 0.;

        ucos = ww1[ii] * tg[j] + wwu2 * dot((n - QUAR*beta), betaP);

        betaP.z = -(betaP.x*beta.x + betaP.y*beta.y) / beta.z;
        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        if (isHiPrecision) {
            krel = 1. - dot(n, beta);
            nnb = cross(n, cross((n - beta), betaP))/(krel*krel); }
        else
            nnb = (n - beta) * w[ii];
//        nnb = (n-beta)*dot(n, betaP)/(krel*krel) - betaP/krel;

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
    }

__kernel void undulator_taper(const float alpha,
                              const float Kx,
                              const float Ky,
//                              const float phase,
                              const int jend,
                              const int nmax,
                              __global float* gamma,
                              __global float* wu,
                              __global float* w,
                              __global float* ww1,
                              __global float* ddphi,
                              __global float* ddpsi,
                              __global float* tg,
                              __global float* ag,
                              __global float* sintg,
                              __global float* costg,
                              __global float* sintgph,
                              __global float* costgph,
                              __global float2* Is_gl,
                              __global float2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j, np;

    float2 eucos;
    float3 n, nnb, beta, betaP;
    float ucos, sinucos, cosucos, sin2tg, sin2tgph, krel, zloc;
    float2 Is = zero2;
    float2 Ip = zero2;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float alphaS = alpha / wu[ii] / E2WC;
    float revg = 1. / gamma[ii];
    float revg2 = revg * revg;
    float wug = wu[ii] * revg;
    float wgwu = w[ii] * revg / wu[ii];
    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (np=0; np<nmax; np++) {
        for (j=0; j<jend; j++) {
            zloc = -(nmax-1)*PI + np*PI2 + tg[j];
            sin2tg = TWO * sintg[j] * costg[j];
            sin2tgph = TWO * sintgph[j] * costgph[j];
            ucos = ww1[ii] * zloc + wgwu *
                (-Ky * n.x * (sintg[j] + alphaS * (1. - costg[j] - zloc * sintg[j])) +
                 Kx * n.y * sintgph[j] + 0.125 * revg *
                       (Ky2 * (sin2tg - TWO * alphaS *
                        (zloc * zloc + costg[j] * costg[j] + zloc * sin2tg)) +
                        Kx2 * sin2tgph));

            sinucos = sincos(ucos, &cosucos);
            eucos.x = cosucos;
            eucos.y = sinucos;

            beta.x = Ky * revg * costg[j] * (1 - alphaS * zloc);
            beta.y = -Kx * revg * costgph[j];
            beta.z = 1 - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

            betaP.x = -Ky * wug * (alphaS*costg[j] + (1 - alphaS*zloc)*sintg[j]);
            betaP.y = Kx * wug * sintgph[j];
            betaP.z = wu[ii] * revg2 *
                (Kx2*sintgph[j]*costgph[j] + Ky2*(1 - alphaS*zloc)*
                 (alphaS * costg[j] * costg[j] + (1 - alphaS*zloc)*sintg[j]*costg[j]));

            if (isHiPrecision) {
                krel = 1. - dot(n, beta);
                nnb = cross(n, cross((n - beta), betaP))/(krel*krel); }
            else
                nnb = (n - beta) * w[ii];

            Is += (ag[j] * nnb.x) * eucos;
            Ip += (ag[j] * nnb.y) * eucos; }}

    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void undulator_nf(const float R0,
                           const float Kx,
                           const float Ky,
                           const int jend,
                           const int nmax,
                           __global float* gamma,
                           __global float* wu,
                           __global float* w,
                           __global float* ww1,
                           __global float* ddphi,
                           __global float* ddpsi,
                           __global float* tg,
                           __global float* ag,
                           __global float* sintg,
                           __global float* costg,
                           __global float* sintgph,
                           __global float* costgph,
                           __global float2* Is_gl,
                           __global float2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j, np;

    float2 eucos;
    float krel, LR, zloc, zterm, Kys, Kxsph, drs;
    float sinr0z, cosr0z, sinzloc, coszloc, sindrs, cosdrs;
    float2 Is = zero2;
    float2 Ip = zero2;
    float3 r, r0, n, nloc, nnb, beta, betaP, dr;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float revg = 1. / gamma[ii];
    float revg2 = revg * revg;
    float wwu = w[ii] / wu[ii];
    float betam = 1. - (1. + HALF * Kx2 + HALF * Ky2) * HALF * revg2;

    nloc.x = tan(ddphi[ii]);
    nloc.y = tan(ddpsi[ii]);
    nloc.z = 1.;

//    nloc /= length(nloc); // Only for the spherical screen

    r0 = R0 * nloc;
    sinr0z = sincos(wwu * r0.z, &cosr0z);

    for (np=0; np<nmax; np++) {
        for (j=0; j<jend; j++) {
            zterm = revg * (Ky2*sintg[j]*costg[j] + Kx2*sintgph[j]*costgph[j]);
            Kys = Ky * sintg[j];
            Kxsph = Kx * sintgph[j];
            zloc = -(nmax-1)*PI + np*PI2 + tg[j];

            r.x = Kys * revg;
            r.y = -Kxsph * revg;
            r.z = betam * zloc - QUAR * zterm * revg;

            dr = r0 - r;
            drs = 0.5*(dr.x*dr.x+dr.y*dr.y)/dr.z;

            sinzloc = sincos(wwu *zloc*(1-betam), &coszloc);
            sindrs = sincos(wwu *(drs + QUAR * zterm * revg), &cosdrs);

            LR = dr.z + drs; // - 0.5*drs*drs/dr.z;

            eucos.x = -sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -
                       cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs;
            eucos.y = -sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +
                       cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs;

            beta.x = Ky * revg * costg[j];
            beta.y = -Kx * revg * costgph[j];
//            beta.z = sqrt(1. - revg2 - beta.x*beta.x - beta.y*beta.y);
            beta.z = 1. - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

            betaP.x = -Kys; //* revg;
            betaP.y = Kxsph; //* revg;
            betaP.z = zterm; //* revg;
            n = dr/LR;

            krel = 1. - dot(n, beta);
            nnb = cross(n, cross((n - beta), betaP))/(krel*krel);
    //velocity field
    //        nnb = nnb + (n - beta)*(1 - beta*beta)*C / (krel*krel*R0);
    //            }
    //        else
//                nnb = (n - beta) * w[ii];

            Is += (ag[j] * nnb.x) * eucos;
            Ip += (ag[j] * nnb.y) * eucos;}}

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is * wu[ii] * revg;
    Ip_gl[ii] = Ip * wu[ii] * revg;
}

__kernel void undulator_full(const float alpha,
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
                             __global float* beta0x,
                             __global float* beta0y,
                             __global float* tg,
                             __global float* ag,
                             __global float2* Is_gl,
                             __global float2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    float2 eucos;
    float2 beta0 = (float2)(beta0x[ii], beta0y[ii]);
    float ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, krel;
    float2 Is = zero2;
    float2 Ip = zero2;
    float3 r, n, nnb, beta, betaP;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float revg = 1. / gamma[ii];
    float revg2 = revg * revg;
    float wug = wu[ii] * revg;
    float wwu = w[ii] / wu[ii];
    float betam = 1 - (1 + HALF * Kx2 + HALF * Ky2) * HALF * revg2;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);

        beta.x = Ky*costg*revg + beta0.x;
        beta.y= -Kx*costgph*revg + beta0.y;
        beta.z = 1. - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

        r.x = Ky*sintg*revg + beta0.x*tg[j];
        r.y = -Kx*sintgph*revg + beta0.y*tg[j];
        r.z = -QUAR*revg2*(Kx2*sintgph*costgph + Ky2*sintg*costg) + tg[j]*betam +
            Kx*beta0.y*sintgph*revg - Ky*beta0.x*sintg*revg -
            tg[j]*HALF*(beta0.x*beta0.x + beta0.y*beta0.y);


        //r.x = Ky / gamma[ii] * sintg;
        //r.y = -Kx / gamma[ii] * sintgph;
        //r.z = betam * tg[j] - 0.25 / gamma2 *
        //(Ky2 * sintg * costg + Kx2 * sintgph * costgph);

        ucos = wwu * (tg[j] - dot(n, r));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        //beta.x = Ky / gamma[ii] * costg;
        //beta.y = -Kx / gamma[ii] * costgph;
        //beta.z = sqrt(1. - 1./gamma2 - beta.x*beta.x - beta.y*beta.y);
        //beta.z = 1 - 0.5*(1./gamma2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * sintg;
        betaP.y = Kx * wug * sintgph;
        betaP.z = wu[ii] * revg2 * (Ky2*sintg*costg + Kx2*sintgph*costgph);
        //betaP.z = wu[ii]/beta.z/gamma2*(Ky2*sintg*costg + Kx2*sintgph*costgph);

        if (isHiPrecision) {
            krel = 1. - dot(n, beta);
            nnb = cross(n, cross((n - beta), betaP))/(krel*krel); }
        else
            nnb = (n - beta) * w[ii];

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void undulator_nf_full(const float R0,
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
                            __global float* beta0x,
                            __global float* beta0y,
                            __global float* tg,
                            __global float* ag,
                            __global float2* Is_gl,
                            __global float2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    float2 eucos;
    float ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, krel;
    float2 beta0 = (float2)(beta0x[ii], beta0y[ii]);
    float2 Is = zero2;
    float2 Ip = zero2;
    float3 r, r0, n, nnb, beta, betaP;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float revg = 1. / gamma[ii];
    float revg2 = revg * revg;
    float wug = wu[ii] * revg;
    float wwu = w[ii] / wu[ii];
    float betam = 1 - (1 + HALF * Kx2 + HALF * Ky2) * HALF * revg2;
    float wR0 = R0 * PI2 / L0;

    r0.x = wR0 * tan(ddphi[ii]);
    r0.y = wR0 * tan(ddpsi[ii]);
    r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii]));

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);

        //r.x = Ky / gamma[ii] * sintg;
        //r.y = -Kx / gamma[ii] * sintgph;
        //r.z = betam * tg[j] - 0.25 / gamma2 *
        //(Ky2 * sintg * costg + Kx2 * sintgph * costgph);
        beta.x = Ky*costg*revg + beta0.x;
        beta.y= -Kx*costgph*revg + beta0.y;
        beta.z = 1. - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

        r.x = Ky*sintg*revg + beta0.x*tg[j];
        r.y = -Kx*sintgph*revg + beta0.y*tg[j];
        r.z = -QUAR*revg2*(Kx2*sintgph*costgph + Ky2*sintg*costg) + tg[j]*betam +
            Kx*beta0.y*sintgph*revg - Ky*beta0.x*sintg*revg -
            tg[j]*HALF*(beta0.x*beta0.x + beta0.y*beta0.y);

        ucos = wwu * (tg[j] + length(r0 - r));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        //beta.x = Ky / gamma[ii] * costg;
        //beta.y = -Kx / gamma[ii] * costgph;
        //beta.z = sqrt(1. - 1./gamma2 - beta.x*beta.x - beta.y*beta.y);
        //beta.z = 1 - 0.5*(1./gamma2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * sintg;
        betaP.y = Kx * wug * sintgph;
        betaP.z = wu[ii] * revg2 * (Ky2*sintg*costg + Kx2*sintgph*costgph);
        //betaP.z = wu[ii]/beta.z/gamma2*(Ky2*sintg*costg + Kx2*sintgph*costgph);

        if (isHiPrecision) {
            krel = 1. - dot(n, beta);
            nnb = cross(n, cross((n - beta), betaP))/(krel*krel);
//velocity field
//            nnb = nnb + (n - beta)*(1 - beta*beta)*C / (krel*krel*R0);
            }
        else
            nnb = (n - beta) * w[ii];

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
float2 next_beta_rkn(float2 beta, int nmax,
                     float rkStep, float emcg,
                     float* Bx, float* By, float* Bz)
{
    float2 kBetaFinal = zero2;
    float2 fArg;
    float2 kf_beta[10];
    int i, j;
    for (i=0;i<nmax;i++) {
        fArg = zero2;
        for (j=0;j<i;j++) {
            fArg += RK8_A[(i-1)*(nmax-1)+j]*kf_beta[j];
        }
        kf_beta[i] = f_beta(
                Bx[i], By[i], Bz[i], emcg,
                rkStep*fArg+beta);
        kBetaFinal += kf_beta[i]*RK8_B[i];
    }
    return rkStep*kBetaFinal + beta;
}
float8 next_traj_rkn(float2 beta, float3 traj,
                     int nmax, float rkStep, float emcg, float revgamma,
                     float* Bx, float* By, float* Bz)
{
    float2 kBetaFinal=zero2;
    float3 kTrajFinal=zero3;
    float2 fArg;
    float2 kf_beta[10];
    float3 kf_traj[10];
    int i, j;
    for (i=0;i<nmax;i++) {
        fArg = zero2;
        for (j=0;j<i;j++) {
            fArg += RK8_A[(i-1)*(nmax-1)+j]*kf_beta[j];
        }
        kf_beta[i] = f_beta(
                Bx[i], By[i], Bz[i], emcg,
                rkStep*fArg+beta);
        kf_traj[i] = f_traj(revgamma, rkStep*fArg+beta);
        kBetaFinal += kf_beta[i]*RK8_B[i];
        kTrajFinal += kf_traj[i]*RK8_B[i];
    }
    return (float8)(rkStep*kBetaFinal+beta,
                    rkStep*kTrajFinal+traj, zero3);
}
*/
float2 f_beta_filament(float Bx, float By, float Bz, float emcg, float2 beta)
{
    return emcg*(float2)(beta.y*Bz - By,
                         Bx - beta.x*Bz);
}

float2 f_beta(float Bx, float By, float Bz, float2 beta)
{
    return (float2)(beta.y*Bz - By,
                    Bx - beta.x*Bz);
}

float3 f_traj_filament(float revgamma2, float2 beta)
{
    float smTerm = revgamma2+beta.x*beta.x+beta.y*beta.y;
    return (float3)(beta.x,
                    beta.y,
//                    sqrt(1. - smTerm));
                    1. - 0.5*smTerm - 0.125*smTerm*smTerm - 0.0625*smTerm*smTerm*smTerm);
}

float3 f_traj(float2 beta)
{
    return (float3)(beta.x,
                    beta.y,
                    -0.5*(beta.x*beta.x + beta.y*beta.y));
}
float2 next_beta_rk_filament(float2 beta, int iBase, float rkStep, float emcg,
                             __global float* Bx,
                             __global float* By,
                             __global float* Bz)
{
    float2 k1Beta, k2Beta, k3Beta, k4Beta;

    k1Beta = rkStep * f_beta_filament(Bx[iBase], By[iBase], Bz[iBase],
                                      emcg, beta);
    k2Beta = rkStep * f_beta_filament(Bx[iBase+1], By[iBase+1], Bz[iBase+1],
                                      emcg, beta + HALF*k1Beta);
    k3Beta = rkStep * f_beta_filament(Bx[iBase+1], By[iBase+1], Bz[iBase+1],
                                      emcg, beta + HALF*k2Beta);
    k4Beta = rkStep * f_beta_filament(Bx[iBase+2], By[iBase+2], Bz[iBase+2],
                                      emcg, beta + k3Beta);
    return beta + (k1Beta + TWO*k2Beta + TWO*k3Beta + k4Beta)*REVSIX;
}

float2 next_beta_rk(float2 beta, int iBase, float rkStep,
                     __global float* Bx,
                     __global float* By,
                     __global float* Bz)
{
    float2 k1Beta, k2Beta, k3Beta, k4Beta;

    k1Beta = rkStep * f_beta(Bx[iBase], By[iBase], Bz[iBase], beta);
    k2Beta = rkStep * f_beta(Bx[iBase+1], By[iBase+1], Bz[iBase+1],
                             beta + HALF*k1Beta);
    k3Beta = rkStep * f_beta(Bx[iBase+1], By[iBase+1], Bz[iBase+1],
                             beta + HALF*k2Beta);
    k4Beta = rkStep * f_beta(Bx[iBase+2], By[iBase+2], Bz[iBase+2],
                             beta + k3Beta);
    return beta + (k1Beta + TWO*k2Beta + TWO*k3Beta + k4Beta)*REVSIX;
}

float8 next_traj_rk_filament(float2 beta, float3 traj, int iBase, float rkStep,
                             float emcg, float revgamma,
                             __global float* Bx,
                             __global float* By,
                             __global float* Bz)
{
    float2 k1Beta, k2Beta, k3Beta, k4Beta;
    float3 k1Traj, k2Traj, k3Traj, k4Traj;

    k1Beta = rkStep * f_beta_filament(Bx[iBase], By[iBase], Bz[iBase],
                                      emcg, beta);
    k1Traj = rkStep * f_traj_filament(revgamma, beta);

    k2Beta = rkStep * f_beta_filament(Bx[iBase+1], By[iBase+1], Bz[iBase+1],
                                      emcg, beta + HALF*k1Beta);
    k2Traj = rkStep * f_traj_filament(revgamma, beta + HALF*k1Beta);

    k3Beta = rkStep * f_beta_filament(Bx[iBase+1], By[iBase+1], Bz[iBase+1],
                                      emcg, beta + HALF*k2Beta);
    k3Traj = rkStep * f_traj_filament(revgamma, beta + HALF*k2Beta);

    k4Beta = rkStep * f_beta_filament(Bx[iBase+2], By[iBase+2], Bz[iBase+2],
                                      emcg, beta + k3Beta);
    k4Traj = rkStep * f_traj_filament(revgamma, beta + k3Beta);

    return (float8)(beta + (k1Beta + TWO*k2Beta + TWO*k3Beta + k4Beta)*REVSIX,
                    traj + (k1Traj + TWO*k2Traj + TWO*k3Traj + k4Traj)*REVSIX,
                    0., 0., 0.);
}

float8 next_traj_rk(float2 beta, float3 traj, int iBase, float rkStep,
                     __global float* Bx,
                     __global float* By,
                     __global float* Bz)
{
    float2 k1Beta, k2Beta, k3Beta, k4Beta;
    float3 k1Traj, k2Traj, k3Traj, k4Traj;

    k1Beta = rkStep * f_beta(Bx[iBase], By[iBase], Bz[iBase], beta);
    k1Traj = rkStep * f_traj(beta);

    k2Beta = rkStep * f_beta(Bx[iBase+1], By[iBase+1], Bz[iBase+1],
                             beta + HALF*k1Beta);
    k2Traj = rkStep * f_traj(beta + HALF*k1Beta);

    k3Beta = rkStep * f_beta(Bx[iBase+1], By[iBase+1], Bz[iBase+1],
                             beta + HALF*k2Beta);
    k3Traj = rkStep * f_traj(beta + HALF*k2Beta);

    k4Beta = rkStep * f_beta(Bx[iBase+2], By[iBase+2], Bz[iBase+2],
                             beta + k3Beta);
    k4Traj = rkStep * f_traj(beta + k3Beta);

    return (float8)(beta + (k1Beta + TWO*k2Beta + TWO*k3Beta + k4Beta)*REVSIX,
                    traj + (k1Traj + TWO*k2Traj + TWO*k3Traj + k4Traj)*REVSIX,
                    0., 0., 0.);
}

__kernel void get_trajectory_filament(const int jend,
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
    int j;
    float rkStep;
    float gamma2 = gamma*gamma;
    float revg2 = 1./gamma2;
    float betam_int = 0;
    float emcg = EMC/gamma;
    float2 beta, beta0;
    float3 traj, traj0, trajC;
    int trajCN = 0;
    float8 betaTraj;
    beta = zero2;
    beta0 = zero2;

    for (j=0; j<jend-1; j++) {
        rkStep = tg[j+1] - tg[j];
        beta = next_beta_rk_filament(beta, j*2, rkStep, emcg, Bx, By, Bz);
        beta0 += beta * rkStep;
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    beta0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = (float3)(0., 0., 0.);
    traj0 = (float3)(0., 0., 0.);
    trajC = (float3)(0., 0., 0.);

    for (j=0; j<jend-1; j++) {
        rkStep = tg[j+1] - tg[j];
        betaTraj = next_traj_rk_filament(beta, traj, j*2,
                                         rkStep, emcg, revg2, Bx, By, Bz);
        beta = betaTraj.s01;
        traj = betaTraj.s234;
        traj0 += traj * rkStep;
        betam_int += rkStep*sqrt(1. - revg2 - beta.x*beta.x -
                                 beta.y*beta.y);
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    traj0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = traj0;
    betam_int /= tg[jend-1] - tg[0];

    betax[0] = beta0.x;
    betay[0] = beta0.y;
    trajx[0] = traj0.x;
    trajy[0] = traj0.y;
    trajz[0] = traj0.z;

    for (j=0; j<jend-1; j++) {
        rkStep = tg[j+1] - tg[j];
        betaTraj = next_traj_rk_filament(beta, traj, j*2,
                                         rkStep, emcg, revg2, Bx, By, Bz);
        beta = betaTraj.s01;
        traj = betaTraj.s234;

        betax[j+1] = beta.x;
        betay[j+1] = beta.y;
        betazav[j+1] = betam_int;
        trajx[j+1] = traj.x;
        trajy[j+1] = traj.y;
        trajz[j+1] = traj.z;
        if (fabs(traj.z) < NEAR_CENTER) {
            trajC += traj;
            trajCN += 1;
        }
    }
    if (trajCN > 0) {
        trajC /= trajCN;
        for (j=0; j<jend; j++) {
            trajx[j] -= trajC.x;
            trajy[j] -= trajC.y;
            trajz[j] -= trajC.z;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void custom_field_filament(const int jend,
                                    const float emcg,
                                    const float revg2,
                                    const float R0,
                                    const float wc,
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
    unsigned int ii = get_global_id(0);
    int j;

    float ucos, sinucos, cosucos, krel, LR, LRS, drs, zq;
    float sinr0z, cosr0z, sinzloc, coszloc, sindrs, cosdrs;
    float smTerm, rdrz;

    float2 eucos;
    float2 Is = (float2)(0., 0.);
    float2 Ip = (float2)(0., 0.);

    float3 traj, n, r0, betaC, betaP, nnb, dr;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
//    n.z = 1. - HALF*(n.x*n.x + n.y*n.y);
    zq = n.x*n.x + n.y*n.y;
    n.z = 1. - HALF*zq - 0.125*zq*zq - 0.0625*zq*zq*zq;
//    n /= length(n);

    if (R0>0) {
        n.x = tan(ddphi[ii]);
        n.y = tan(ddpsi[ii]);
        n.z = 1.;
//        n /= length(n);  // Only for the spherical screen
        r0 = R0 * n;
        sinr0z = sincos(wc * r0.z, &cosr0z);}

    for (j=0; j<jend; j++) {
        traj.x = trajx[j];
        traj.y = trajy[j];
        traj.z = trajz[j];

        betaC.x = betax[j];
        betaC.y = betay[j];
        smTerm = revg2 + betaC.x*betaC.x + betaC.y*betaC.y;
        betaC.z = 1. - 0.5*smTerm - 0.125*smTerm*smTerm - 0.0625*smTerm*smTerm*smTerm;
//        betaC.z = sqrt(1 - revg2 - betaC.x*betaC.x - betaC.y*betaC.y);
//        betaC.z = sqrt(1 - smTerm);
        if (R0 > 0) {
            dr = r0 - traj;
            rdrz = 1./dr.z;
            drs = (dr.x*dr.x+dr.y*dr.y)*rdrz;

            LRS = 0.5*drs - 0.125*drs*drs*rdrz + 0.0625*drs*drs*drs*rdrz*rdrz;
            LR = length(dr);

            sinzloc = sincos(wc * (tg[j]-traj.z), &coszloc);
            sindrs = sincos(wc * LRS, &cosdrs);
            eucos.x = -sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -
                       cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs;
            eucos.y = -sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +
                       cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs;
            n = dr / LR; }
        else {
            ucos = wc * (tg[j] - dot(n, traj));

            sinucos = sincos(ucos, &cosucos);
            eucos.x = cosucos;
            eucos.y = sinucos;}

        betaP.x = betaC.y*Bz[j] - betaC.z*By[j];
        betaP.y = -betaC.x*Bz[j] + betaC.z*Bx[j];
        betaP.z = betaC.x*By[j] - betaC.y*Bx[j];

        krel = 1. - dot(n, betaC);
        nnb = cross(n, cross((n - betaC), betaP))/(krel*krel);

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is * emcg;
    Ip_gl[ii] = Ip * emcg;
}

__kernel void get_trajectory(const int jend,
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
    int j;
    float rkStep;
    float betam_int = 0;
    float2 beta, beta0;
    float3 traj, traj0, trajC;
    int trajCN = 0;
    float8 betaTraj;
    beta = zero2;
    beta0 = zero2;

    for (j=0; j<jend-1; j++) {
        rkStep = tg[j+1] - tg[j];
        beta = next_beta_rk(beta, j*2, rkStep, Bx, By, Bz);
        beta0 += beta * rkStep;
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    beta0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = (float3)(0., 0., 0.);
    traj0 = (float3)(0., 0., 0.);
    trajC = (float3)(0., 0., 0.);

    for (j=0; j<jend-1; j++) {
        rkStep = tg[j+1] - tg[j];
        betaTraj = next_traj_rk(beta, traj, j*2, rkStep, Bx, By, Bz);
        beta = betaTraj.s01;
        traj = betaTraj.s234;
        traj0 += traj * rkStep;
//        betam_int += rkStep*sqrt(1. - revg2 - beta.x*beta.x -
//                                 beta.y*beta.y);
        betam_int += beta.x*beta.x + beta.y*beta.y;
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    traj0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = traj0;
    betam_int *= -0.5/(float)(jend-1);
    betax[0] = beta0.x;
    betay[0] = beta0.y;
    trajx[0] = traj0.x;
    trajy[0] = traj0.y;
    trajz[0] = traj0.z;

    for (j=0; j<jend-1; j++) {
        rkStep = tg[j+1] - tg[j];
        betaTraj = next_traj_rk(beta, traj, j*2, rkStep, Bx, By, Bz);
        beta = betaTraj.s01;
        traj = betaTraj.s234;

        betax[j+1] = beta.x;
        betay[j+1] = beta.y;
        betazav[j+1] = betam_int;
        trajx[j+1] = traj.x;
        trajy[j+1] = traj.y;
        trajz[j+1] = traj.z;
        if (fabs(traj.z) < NEAR_CENTER) {
            trajC += traj;
            trajCN += 1;
        }
    }
    if (trajCN > 0) {
        trajC /= trajCN;
        for (j=0; j<jend; j++) {
            trajx[j] -= trajC.x;
            trajy[j] -= trajC.y;
            trajz[j] -= trajC.z;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void custom_field(const int jend,
                           const float betazav,
                           const float R0,
                           __global float* gamma,
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
    unsigned int ii = get_global_id(0);
    int j;

    float ucos, sinucos, cosucos, krel, LR, LRS, drs, zq;
    float sinr0z, cosr0z, sinzloc, coszloc, sindrs, cosdrs;
    float gamma2 = gamma[ii]*gamma[ii];
    float revg2 = 1./gamma2;

    float emcg = EMC / gamma[ii];
    float emc2 = EMC*EMC;

    float betam = 1. + (betazav*emc2 - 0.5)*revg2;

    float smTerm, rdrz;

    float2 eucos;
    float2 Is = (float2)(0., 0.);
    float2 Ip = (float2)(0., 0.);

    float3 traj, n, r0, betaC, betaP, nnb, dr;

    float wc = w[ii] * E2WC / betam;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    zq = n.x*n.x + n.y*n.y;
    n.z = 1. - HALF*zq - 0.125*zq*zq - 0.0625*zq*zq*zq;
//    n /= length(n);

    if (R0>0) {
        n.x = tan(ddphi[ii]);
        n.y = tan(ddpsi[ii]);
        n.z = 1.;
//        n /= length(n);  // Only for the spherical screen
        r0 = R0 * n;
        sinr0z = sincos(wc * r0.z, &cosr0z);}

    for (j=0; j<jend; j++) {
        traj.x = emcg*trajx[j];
        traj.y = emcg*trajy[j];
        traj.z = tg[j]*(1.-0.5*revg2) + emc2*revg2*trajz[j];

        betaC.x = emcg*betax[j];
        betaC.y = emcg*betay[j];
        smTerm = revg2 + betaC.x*betaC.x + betaC.y*betaC.y;
        betaC.z = 1. - 0.5*smTerm - 0.125*smTerm*smTerm - 0.0625*smTerm*smTerm*smTerm;
//        betaC.z = sqrt(1 - revg2 - betaC.x*betaC.x - betaC.y*betaC.y);
//        betaC.z = sqrt(1 - smTerm);
        if (R0 > 0) {
            dr = r0 - traj;
            rdrz = 1./dr.z;
            drs = (dr.x*dr.x+dr.y*dr.y)*rdrz;

            LRS = 0.5*drs - 0.125*drs*drs*rdrz + 0.0625*drs*drs*drs*rdrz*rdrz;
            LR = length(dr);

            sinzloc = sincos(wc * (tg[j]-traj.z), &coszloc);
            sindrs = sincos(wc * LRS, &cosdrs);
            eucos.x = -sinr0z*sinzloc*cosdrs - sinr0z*coszloc*sindrs -
                       cosr0z*sinzloc*sindrs + cosr0z*coszloc*cosdrs;
            eucos.y = -sinr0z*sinzloc*sindrs + sinr0z*coszloc*cosdrs +
                       cosr0z*sinzloc*cosdrs + cosr0z*coszloc*sindrs;
            n = dr / LR; }
        else {
            ucos = wc * (tg[j] - dot(n, traj));

            sinucos = sincos(ucos, &cosucos);
            eucos.x = cosucos;
            eucos.y = sinucos;}

        betaP.x = betaC.y*Bz[j] - betaC.z*By[j];
        betaP.y = -betaC.x*Bz[j] + betaC.z*Bx[j];
        betaP.z = betaC.x*By[j] - betaC.y*Bx[j];

        krel = 1. - dot(n, betaC);
        nnb = cross(n, cross((n - betaC), betaP))/(krel*krel);
        //nnb = nnb + (n - betaC)*(1 - betaC*betaC)*C / (krel*krel*R0);
        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is * emcg;
    Ip_gl[ii] = Ip * emcg;
}