//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "03 Jul 2016"

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
__constant bool isHiPrecision = sizeof(TWO) == 8;
//__constant bool isHiPrecision = false;
__constant float2 zero2 = (float2)(0, 0);

__constant float PI2 = (float)6.283185307179586476925286766559;
//__constant float PI = (float)3.1415926535897932384626433832795;

__constant float E2W = 1.51926751475e15;
__constant float C = 2.99792458e11;
__constant float SIE0 = 1.602176565e-19;
__constant float SIM0 = 9.10938291e-31;

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

    float3 beta, betaP, n, nnb;
    float2 eucos;
    float ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, krel;
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
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);

        beta.x = Ky * revg * costg;
        beta.y = -Kx * revg * costgph;
//        beta.z = sqrt(1. - revg2 - beta.x*beta.x - beta.y*beta.y);
        beta.z = 1. - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * sintg;
        betaP.y = Kx * wug * sintgph;
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

    float2 eucos;
    float3 n, nnb, beta, betaP;
    float ucos, sinucos, cosucos, sintg, sin2tg, sin2tgph, costg, sintgph, costgph, krel;
    float2 Is = zero2;
    float2 Ip = zero2;
    float Kx2 = Kx * Kx;
    float Ky2 = Ky * Ky;
    float alphaS = alpha * C / wu[ii] / E2W;
    float revg = 1. / gamma[ii];
    float revg2 = revg * revg;
    float wug = wu[ii] * revg;
    float wgwu = w[ii] * revg / wu[ii];
    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);
        sin2tg = TWO * sintg * costg; //sin(2*tg[j]);
        sin2tgph = TWO * sintgph * costgph; //sin(2*(tg[j] + phase));
        ucos = ww1[ii] * tg[j] + wgwu *
            (-Ky * n.x * (sintg + alphaS * (1. - costg - tg[j] * sintg)) +
             Kx * n.y * sintgph + 0.125 * revg *
                   (Ky2 * (sin2tg - TWO * alphaS *
                    (tg[j] * tg[j] + costg * costg + tg[j] * sin2tg)) +
                    Kx2 * sin2tgph));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        beta.x = Ky * revg * costg * (1 - alphaS * tg[j]);
        beta.y = -Kx * revg * costgph;
        beta.z = 1 - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * (alphaS*costg + (1 - alphaS*tg[j])*sintg);
        betaP.y = Kx * wug * sintgph;
        betaP.z = wu[ii] * revg2 *
            (Kx2*sintgph*costgph + Ky2*(1 - alphaS*tg[j])*
             (alphaS * costg * costg + (1 - alphaS*tg[j])*sintg*costg));

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
    unsigned int ii = get_global_id(0);
    int j;

    float2 eucos;
    float ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, krel;
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
//    n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);

        r.x = Ky * revg * sintg;
        r.y = -Kx * revg * sintgph;
        r.z = betam * tg[j] - QUAR * revg2 *
        (Ky2 * sintg * costg + Kx2 * sintgph * costgph);

        ucos = wwu * (tg[j] + length(r0 - r));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        beta.x = Ky * revg * costg;
        beta.y = -Kx * revg * costgph;
//        beta.z = sqrt(1. - revg2 - beta.x*beta.x - beta.y*beta.y);
        beta.z = 1 - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * sintg;
        betaP.y = Kx * wug * sintgph;
        betaP.z = wu[ii] * revg2 * (Ky2*sintg*costg + Kx2*sintgph*costgph);
        //betaP.z = wu[ii]/beta.z*revg2*(Ky2*sintg*costg + Kx2*sintgph*costgph);

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
    //float betax0 = 0; //1e-6;
    //float betay0 = 0;
    //float rx0 = 2e-2*PI2/L0/wu[ii];
    //float ry0 = 0;
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
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
                             emcg, beta + HALF*k1Beta);
    k3Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + HALF*k2Beta);
    k4Beta = rkStep * f_beta(Bx[iFullStep],
                             By[iFullStep],
                             Bz[iFullStep],
                             emcg, beta + k3Beta);
    return beta + (k1Beta + TWO*k2Beta + TWO*k3Beta + k4Beta) / SIX;
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
                             emcg, beta + HALF*k1Beta);
    k2Traj = rkStep * f_traj(revgamma, beta + HALF*k1Beta);

    k3Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + HALF*k2Beta);
    k3Traj = rkStep * f_traj(revgamma, beta + HALF*k2Beta);

    k4Beta = rkStep * f_beta(Bx[iFullStep],
                             By[iFullStep],
                             Bz[iFullStep],
                             emcg, beta + k3Beta);
    k4Traj = rkStep * f_traj(revgamma, beta + k3Beta);

    return (float8)(beta + (k1Beta + TWO*k2Beta + TWO*k3Beta + k4Beta)/SIX,
                     traj + (k1Traj + TWO*k2Traj + TWO*k3Traj + k4Traj)/SIX,
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
    unsigned int ii = get_global_id(0);
    int j, k, jb;
    int iBase, iZeroStep, iHalfStep, iFullStep;

    float ucos, sinucos, cosucos, rkStep, wu_int, betam_int, krel;
    float revg = 1. / gamma[ii];
    float revg2 = revg * revg;
    float emcg = lUnd * SIE0 / SIM0 / C * revg / PI2;
    float revgamma = 1. - revg2;
    float8 betaTraj;
    float2 eucos;
    float2 Is = zero2;
    float2 Ip = zero2;
    float2 beta, beta0;
    float3 traj, n, traj0, betaC, betaP, nnb;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    n.z = 1. - HALF*(n.x*n.x + n.y*n.y);

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
        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        jb = 2*j*nwt;

        betaC.x = beta.x;
        betaC.y = beta.y;
        betaC.z = 1 - HALF*(revg2 + betaC.x*betaC.x + betaC.y*betaC.y);

        betaP.x = wu_int * emcg * (betaC.y*Bz[jb] - By[jb]);
        betaP.y = wu_int * emcg * (-betaC.x*Bz[jb] + Bx[jb]);
        betaP.z = wu_int * emcg * (betaC.x*By[jb] - betaC.y*Bx[jb]);

        if (isHiPrecision) {
            krel = 1. - dot(n, betaC);
            nnb = cross(n, cross((n - betaC), betaP))/(krel*krel); }
        else
            nnb = (n - betaC) * w[ii];

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
    unsigned int ii = get_global_id(0);
    int j, jb;

    float ucos, sinucos, cosucos, wR0, krel;
    float revg2 = 1./gamma2;

    float2 eucos;
    float2 Is = (float2)(0., 0.);
    float2 Ip = (float2)(0., 0.);

    float3 traj, n, r0, betaC, betaP, nnb;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    n.z = 1. - HALF*(n.x*n.x + n.y*n.y);

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
        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        jb = 2*j*nwt;

        betaC.x = betax[j];
        betaC.y = betay[j];
        betaC.z = 1 - HALF*(revg2 + betaC.x*betaC.x + betaC.y*betaC.y);

        betaP.x = wu * emcg * (betaC.y*Bz[jb] - By[jb]);
        betaP.y = wu * emcg * (-betaC.x*Bz[jb] + Bx[jb]);
        betaP.z = wu * emcg * (betaC.x*By[jb] - betaC.y*Bx[jb]);

        if (isHiPrecision) {
            krel = 1. - dot(n, betaC);
            nnb = cross(n, cross((n - betaC), betaP))/(krel*krel); }
        else
            nnb = (n - betaC) * w[ii];

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

        //Is += ag[j] * (n.x - betax[j]) * eucos;
        //Ip += ag[j] * (n.y - betay[j]) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}
