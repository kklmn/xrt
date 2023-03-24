//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "19 Dec 2021"

#define PRIVATELAYERS 1000
#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

#include "xrt_complex_float.cl"

__constant float PI = 3.141592653589793238;
__constant float twoPI = 6.283185307179586476;
__constant float halfPI = 1.570796326794896619;
__constant float revPI = 0.31830988618379067;
__constant float ch = 12398.4186;  // {5}   {c*h[eV*A]}
__constant float chbar = 1973.269606712496640;  // {c*hbar[eV*A]}
__constant float r0 = 2.817940285e-5;  // A
__constant float avogadro = 6.02214199e23;  // atoms/mol
//__constant float HALF = 0.5;
__constant float TWO = (float)2.;
__constant float ONE = (float)1.;

//__constant float QUAR = 0.25;
//__constant float HALF = 0.5;
//__constant float TWO = 2.;
//__constant float SIX = 6.;
__constant float REVSIX = 1./6.;
__constant float REV24 = 1./24.;
__constant float REV720 = 1./720.;
__constant float REV1440 = 1./1440.;

__constant float cx_dp[6] = {0.2, 0.3, 0.8, 8./9., 1., 1.};
__constant float c_out_dp[7] = {35./384., 0, 500./1113., 125./192., -2187./6784., 11./84., 0};
__constant float c_out_dp4[7] = {5179./57600., 0.0, 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.};
__constant float cy_dp[6][6] = {
        {0.2, 0, 0, 0, 0, 0},
        {3./40.,	9./40., 0, 0, 0, 0},
        {44./45., -56./15., 32./9., 0, 0, 0},
        {19372./6561., -25360./2187., 64448./6561.,-212./729., 0, 0},
        {9017./3168., -355./33., 46732./5247., 49./176.,-5103./18656., 0},
        {35./384., 0, 500./1113., 125./192.,-2187./6784., 11./84.}};


float get_f0(float qOver4pi, int iN, __global float* f0cfs) {
    float res = f0cfs[iN*11+5];
    for (int i=0;i<5;i++)
    {
            res += f0cfs[iN*11+i] * exp(-f0cfs[iN*11+i+6] * qOver4pi * qOver4pi);
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return res;
  }


float2 get_f1f2(float E, int iN, int VectorMax,
                 __global float* E_vector,
                 __global float* f1_vector, __global float* f2_vector) {
    int pn = floor((VectorMax-1)*log(E-E_vector[iN*300])/log(E_vector[iN*300+VectorMax]-E_vector[iN*300]));
    if (E_vector[iN*300+pn]>E)
      {
        while (E_vector[iN*300+pn] > E) pn--;
      }
    else
      {
        while (E_vector[iN*300+pn] < E) pn++;
        pn--;
      }
    mem_fence(CLK_LOCAL_MEM_FENCE);

    float dE = (E - E_vector[iN*300+pn]) / (E_vector[iN*300+pn+1]-E_vector[iN*300+pn]);

    float f1 = f1_vector[iN*300+pn] + dE *(f1_vector[iN*300+pn+1] - f1_vector[iN*300+pn]);
    float f2 = f2_vector[iN*300+pn] + dE *(f2_vector[iN*300+pn+1] - f2_vector[iN*300+pn]);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return (float2)(f1, f2);
  }

float8 get_structure_factor_fcc(float E, float factDW,
                                  int4 hkl_i, float sinThetaOverLambda,
                                  const int maxEl,  __global float* elements,
                                  __global float* f0cfs, __global float* E_vector,
                                  __global float* f1_vector, __global float* f2_vector)  {
        float4 hkl;
        hkl.x = (float)hkl_i.x;
        hkl.y = (float)hkl_i.y;
        hkl.z = (float)hkl_i.z;
        hkl.w = 0;
        float2 anomalousPart = get_f1f2(E, 0, round(elements[7]), E_vector, f1_vector, f2_vector);
        float2 F0 = 4 * ((float2)(elements[5],0) + anomalousPart) * factDW;
        float residue = dot(fabs(remainder(hkl,TWO)), ONE);
        float Fcoef=0.;

        if (residue==0 || residue==3) Fcoef = ONE;
        float2 Fhkl = 4 * Fcoef * factDW *
                         (r2cmp(get_f0(sinThetaOverLambda, 0, f0cfs)) + anomalousPart);
        return (float8)(F0, Fhkl, Fhkl, cmp0);
  }

float Si_dl_l(float t)  {
        float res;
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        if (t >= 0.0 && t < 30.0)
          {
            res = -2.154537e-004;
          }
        else if (t >= 30.0 && t < 130.0)
          {
            res = -2.303956e-014 * t4 + 7.834799e-011 * t3 -
                1.724143e-008 * t2 + 8.396104e-007 * t - 2.276144e-004;
          }
        else if (t >= 130.0 && t < 293.0)
          {
            res = -1.223001e-011 * t3 + 1.532991e-008 * t2 -
                3.263667e-006 * t - 5.217231e-005;
          }
        else if (t >= 293.0 && t <= 1000.0)
          {
            res = -1.161022e-012 * t3 + 3.311476e-009 * t2 +
                1.124129e-006 * t - 5.844535e-004;
          }
        else res = 1.0e+100;
        mem_fence(CLK_LOCAL_MEM_FENCE);
        return res;
  }

float2 get_distance(float8 lattice, float4 hkl)    {
        float2 res;
        float4 basis = lattice.lo;
        float4 angles = lattice.hi;

        float4 cos_abg = cos(angles);
        float4 sin_abg = sin(angles);

        float V = basis.x * basis.y * basis.z *
            sqrt(ONE - pown(cos_abg.x,2) - pown(cos_abg.y,2) - pown(cos_abg.z,2) + TWO*cos_abg.x*cos_abg.y*cos_abg.z);
        float rbx = ONE / basis.x, rby = ONE / basis.y, rbz = ONE / basis.z;
        float d = V * rbx * rby * rbz /
            sqrt(pown(hkl.x*sin_abg.x*rbx,2) + pown(hkl.y*sin_abg.y*rby,2) + pown(hkl.z*sin_abg.z*rbz,2) +
                 2*hkl.x*hkl.y * (cos_abg.x*cos_abg.y - cos_abg.z) *rbx*rby +
                 2*hkl.x*hkl.z * (cos_abg.x*cos_abg.z - cos_abg.y) *rbx*rbz +
                 2*hkl.y*hkl.z * (cos_abg.y*cos_abg.z - cos_abg.x) *rby*rbz);
        float chiToF = -r0 * revPI / V;  // minus!
        mem_fence(CLK_LOCAL_MEM_FENCE);
        res = (float2)(d,chiToF);
        return res;
    }

float2 get_distance_Si(float temperature, float4 dhkl)    {
        float aSi = 5.419490 * (Si_dl_l(temperature) - Si_dl_l(273.15) + ONE);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        float d = aSi/sqrt(dot(dhkl,dhkl));
        float chiToF = -r0 * revPI / pown(aSi,3);
        return (float2)(d,chiToF);
    }


float8 get_structure_factor_diamond(float E, float factDW,
                                  int4 hkl_i, float sinThetaOverLambda,
                                  const int maxEl, __global float* elements,
                                  __global float* f0cfs, __global float* E_vector,
                                  __global float* f1_vector, __global float* f2_vector)  {
        float8 res;
        float4 hkl;
        hkl.x = (float)hkl_i.x;
        hkl.y = (float)hkl_i.y;
        hkl.z = (float)hkl_i.z;
        hkl.w = 0;
        float2 diamondToFcc = cmp1 + exp_c(cmpi1 * halfPI * dot(hkl, ONE));
        float8 Fd = get_structure_factor_fcc(E, factDW,
                                  hkl_i, sinThetaOverLambda,
                                  maxEl, elements,
                                  f0cfs, E_vector,
                                  f1_vector, f2_vector);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        float2 F0 = (Fd.lo).lo;
        float2 Fhkl = (Fd.lo).hi;
        float2 Fhkl_ = (Fd.hi).lo;
        res =(float8)(F0 * 2,
                         prod_c(Fhkl,diamondToFcc),
                         prod_c(Fhkl_,conj_c(diamondToFcc)),
                         cmp0);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        return res;
  }

float8 get_structure_factor_general(float E, float factDW,
                                    int4 hkl_i, float sinThetaOverLambda,
                                    const int maxEl, __global float* elements,
                                    __global float* f0cfs, __global float* E_vector,
                                    __global float* f1_vector, __global float* f2_vector)    {
        int i;
        float4 hkl,xyz;
        float2 F0,Fhkl,Fhkl_,anomalousPart,fact,expiHr;
        hkl.x = (float)hkl_i.x;
        hkl.y = (float)hkl_i.y;
        hkl.z = (float)hkl_i.z;
        hkl.w = 0;
        F0 = cmp0; Fhkl = cmp0; Fhkl_ = cmp0;
        for (i=0;i<maxEl;i++)
            {
                anomalousPart = get_f1f2(E, i, round(elements[i*8+7]), E_vector, f1_vector, f2_vector);
                mem_fence(CLK_LOCAL_MEM_FENCE);
                F0 += elements[i*8+4] * (r2cmp(elements[i*8+5]) + anomalousPart) * factDW;
                fact = elements[i*8+4] * (r2cmp(get_f0(sinThetaOverLambda, i, f0cfs)) + anomalousPart) * factDW;
                mem_fence(CLK_LOCAL_MEM_FENCE);
                xyz = (float4)(elements[i*8],elements[i*8+1],elements[i*8+2],0);
                expiHr = exp_c((float2)(0, twoPI * dot(xyz, hkl)));
                Fhkl += prod_c(fact, expiHr);
                Fhkl_ += div_c(fact, expiHr);
                mem_fence(CLK_LOCAL_MEM_FENCE);
            }

        return (float8)(F0, Fhkl, Fhkl_, cmp0);
  }

float8 get_structure_factor_general_E(float E, float factDW,
                                    int4 hkl_i, float sinThetaOverLambda,
                                    const int maxEl, __global float* elements,
                                    __global float* f0cfs, __global float2* f1f2)    {
        int i;
        float4 hkl,xyz;
        float2 F0,Fhkl,Fhkl_,anomalousPart,fact,expiHr;
        hkl.x = (float)hkl_i.x;
        hkl.y = (float)hkl_i.y;
        hkl.z = (float)hkl_i.z;
        hkl.w = 0;
        F0 = cmp0; Fhkl = cmp0; Fhkl_ = cmp0;
        for (i=0;i<maxEl;i++)
            {
                anomalousPart = f1f2[i];
                mem_fence(CLK_LOCAL_MEM_FENCE);
                F0 += elements[i*8+4] * (r2cmp(elements[i*8+5]) + anomalousPart) * factDW;
                fact = elements[i*8+4] * (r2cmp(get_f0(sinThetaOverLambda, i, f0cfs)) + anomalousPart) * factDW;
                mem_fence(CLK_LOCAL_MEM_FENCE);
                xyz = (float4)(elements[i*8],elements[i*8+1],elements[i*8+2],0);
                expiHr = exp_c((float2)(0, twoPI * dot(xyz, hkl)));
                Fhkl += prod_c(fact, expiHr);
                Fhkl_ += div_c(fact, expiHr);
                mem_fence(CLK_LOCAL_MEM_FENCE);
            }

        return (float8)(F0, Fhkl, Fhkl_, cmp0);
  }



float2 for_one_polarization(float polFactor,
                             float2 chih, float2 chih_, float2 chi0,
                             float k02, float k0s, float kHs,
                             float2 alpha, float b, float thickness, int2 geom)  {
            float2 im1 = (float2)(0, 1);
            float2 ra, rb;
            float2 delta = sqrt_c(sqr_c(alpha) + polFactor * polFactor / b *
                                    prod_c(chih, chih_));

            float t = thickness * (float)1e7;
            float2 L = t * delta * k02 * HALF / kHs;

            if (geom.lo == 1) //Bragg
              {
                if (geom.hi == 1) //transmitted
                  {
                    ra = prod_c(rec_c(cos_c(L) - prod_c(prod_c(im1, alpha), div_c(sin_c(L),delta))),
                        exp_c(prod_c(im1,chi0 - alpha * b) * k02 * t * HALF / k0s));
                  }
                else // reflected
                  {
                    if (thickness == 0 || thickness > HALF) // is None:  # thick Bragg
                      {
                        ra = div_c(chih * polFactor, alpha + delta);
                        float2 ad = alpha - delta;
                        if (abs_c(ad) == 0) ad = (float2)(1e-100, 0);
                        rb = div_c(chih * polFactor, ad);

                        if (isnan(abs_c(ra)) || (abs_c(rb) < abs_c(ra))) ra = rb;
                      }
                    else
                    {
                    ra = polFactor * div_c(chih,(alpha + div_c(prod_c(im1,delta),tan_c(L))));
                    }
                  }
              }
            else  // Laue
              {
                if (geom.hi == 1) //transmitted
                  {
                    ra = prod_c(cos_c(L) + prod_c(prod_c(im1, alpha), div_c(sin_c(L),delta)),
                          exp_c(prod_c(im1,chi0 - alpha * b) * k02 * t * HALF / k0s));
                  }
                else
                  {
                    ra = prod_c(div_c(prod_c(chih * polFactor,sin_c(L)), delta),
                        exp_c(prod_c(im1, chi0 - alpha * b) * k02 * t * HALF / k0s));
                  }
              }
            if (geom.hi == 0) ra /= sqrt(fabs(b));
            return ra;
  }


float get_Bragg_angle(float E, float d)  {
        return asin(ch / (2 * d * E));
  }

float get_dtheta_symmetric_Bragg(float E, float d, int4 hkl,
                               float chiToF, float factDW,
                               const int maxEl, __global float* elements,
                               __global float* f0cfs, __global float* E_vector,
                               __global float* f1_vector, __global float* f2_vector)  {
        float8 F = get_structure_factor_general(E, factDW, hkl, HALF/d,
                                             maxEl, elements, f0cfs, E_vector, f1_vector, f2_vector);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        float2 chi0 = (F.lo).lo * chiToF * pown(ch/E, 2);
        return (chi0 / sin(2 * get_Bragg_angle(E, d))).lo;
  }

float get_dtheta_symmetric_Bragg_E(float E, float d, int4 hkl,
                               float chiToF, float factDW,
                               const int maxEl, __global float* elements,
                               __global float* f0cfs, __global float2* f1f2)  {
        float8 F = get_structure_factor_general_E(E, factDW, hkl, HALF / d,
                                             maxEl, elements, f0cfs, f1f2);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        float2 chi0 = (F.lo).lo * chiToF * pown(ch/E, 2);
        return (chi0/sin(2*get_Bragg_angle(E, d))).lo;
  }

float get_dtheta(float E, float d, int4 hkl,
                   float chiToF, float factDW,
                   const int maxEl, __global float* elements,
                   __global float* f0cfs, __global float* E_vector,
                   __global float* f1_vector, __global float* f2_vector,
                  float alpha, int2 geom)  {
        float symm_dt = get_dtheta_symmetric_Bragg(E, d,
                                                    hkl, chiToF, factDW,
                                                    maxEl, elements,
                                                    f0cfs, E_vector,
                                                    f1_vector, f2_vector);
        float thetaB = get_Bragg_angle(E, d);
        float geom_factor = geom.lo == 1 ? -ONE : ONE;
        float gamma0 = sin(thetaB + alpha);
        float gammah = geom_factor * sin(thetaB - alpha);
        float osqg0 = sqrt(ONE - gamma0*gamma0);
        return -(geom_factor*gamma0 - geom_factor*sqrt(gamma0*gamma0 +
                 geom_factor*(gamma0 - gammah) * osqg0 * symm_dt)) / osqg0;
  }


float get_dtheta_E(float E, float d, int4 hkl,
                   float chiToF, float factDW,
                   const int maxEl, __global float* elements,
                   __global float* f0cfs, __global float2* f1f2,
                  float alpha, int2 geom)  {
        float symm_dt = get_dtheta_symmetric_Bragg_E(E, d,
                                                      hkl, chiToF, factDW,
                                                      maxEl, elements,
                                                      f0cfs, f1f2);
        float thetaB = get_Bragg_angle(E, d);
        float geom_factor = geom.lo == 1 ? -ONE : ONE;
        float gamma0 = sin(thetaB + alpha);
        float gammah = geom_factor * sin(thetaB - alpha);
        float osqg0 = sqrt(ONE - gamma0*gamma0);
        return -(geom_factor*gamma0 - geom_factor*sqrt(gamma0*gamma0 +
                 geom_factor*(gamma0 - gammah) * osqg0 * symm_dt)) / osqg0;
  }

float4 get_amplitude_material(float E, int kind, float2 refrac_n, float beamInDotNormal,
                                float thickness, int fromVacuum)  {
    float2 n1, n2, n12s, rs, rp, p2, tf;
    float cosAlpha, sinAlpha;
    float2 n1cosAlpha, n2cosAlpha, n1cosBeta, n2cosBeta, cosBeta;
    if ((kind == 2) || (kind == 5))
        return (float4)(cmp1, cmp1);
    //n = self.get_refractive_index(E)
    if (fromVacuum > 0)
    {
        n1 = cmp1;
        n2 = refrac_n;
    }
    else
    {
        n1 = refrac_n;
        n2 = cmp1;
    }
    cosAlpha = fabs(beamInDotNormal);
    sinAlpha = sqrt(ONE - beamInDotNormal * beamInDotNormal);
    if (isnan(sinAlpha)) sinAlpha = 0.;
    n1cosAlpha = n1 * cosAlpha;
    n2cosAlpha = n2 * cosAlpha;
    n12s = div_c(n1, n2) * sinAlpha;
    cosBeta = sqrt_c(cmp1 - sqr_c(n12s));
    n2cosBeta = prod_c(n2, cosBeta);
    n1cosBeta = prod_c(n1, cosBeta);
    if (kind < 3)  // reflectivity
    {
        rs = div_c((n1cosAlpha - n2cosBeta), (n1cosAlpha + n2cosBeta));
        rp = div_c((n2cosAlpha - n1cosBeta),
                   (n2cosAlpha + n1cosBeta));
        if (kind == 1)
        {
            p2 = exp_c(E * thickness * (float)1e7 / chbar *
                        prod_c(TWO * cmpi1, n2cosBeta));
            rs = prod_c(rs, div_c((cmp1 - p2),
                                  (cmp1 - prod_c(sqr_c(rs), p2))));
            rp = prod_c(rp, div_c((cmp1 - p2),
                                  (cmp1 - prod_c(sqr_c(rp), p2))));
        }
    }
    else if (kind > 2)  // transmittivity
    {
        tf = sqrt(
            (prod_c(n2cosBeta, conj_c(n1))).s0 / cosAlpha) / abs_c(n1);
        rs = div_c((TWO * tf * n1cosAlpha), (n1cosAlpha + n2cosBeta));
        rp = div_c((TWO * tf * n1cosAlpha), (n2cosAlpha + n1cosBeta));
    }
    return (float4)(rs, rp); //, abs(n.imag) * E / chbar * 2e8  # 1/cm
}

float4 get_amplitude_internal(float E, float d, int4 hkl,
                               float chiToF, float factDW,
                               float beamInDotNormal,
                               float beamOutDotNormal,
                               float beamInDotHNormal,
                               float thickness, int2 geom,
                               const int maxEl,
                               __global float* elements,
                               __global float* f0cfs,
                               __global float* E_vector,
                               __global float* f1_vector,
                               __global float* f2_vector)  {
    float waveLength = ch / E;
    float k = twoPI / waveLength;
    float k0s = -beamInDotNormal * k;
    float kHs = -beamOutDotNormal * k;
    float revd = ONE / d;
    float b = k0s / kHs;
    float k0H = fabs(beamInDotHNormal) * (twoPI * revd) * k;
    float k02 = k * k;
    float H2 = pown(twoPI * revd, 2);

    float8 F = get_structure_factor_general(E, factDW, hkl, HALF * revd,
                               maxEl, elements, f0cfs, E_vector, f1_vector, 
                               f2_vector);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    float2 F0 = (F.lo).lo;
    float2 Fhkl = (F.lo).hi;
    float2 Fhkl_ = (F.hi).lo;
    float lambdaSquare = pown(waveLength, 2);
    float chiToFlambdaSquare = chiToF * lambdaSquare;
    float2 chi0 = conj_c(F0) * chiToFlambdaSquare;
    float2 chih = conj_c(Fhkl) * chiToFlambdaSquare;
    float2 chih_ = conj_c(Fhkl_) * chiToFlambdaSquare;
    float2 alpha = r2cmp((HALF*H2 - k0H) / k02) + chi0 * HALF * (ONE / b - ONE);
    float2 curveS = for_one_polarization(ONE,
                                chih, chih_, chi0,
                                k02, k0s, kHs,
                                alpha, b,
                                thickness, geom);  //# s polarization
    mem_fence(CLK_LOCAL_MEM_FENCE);
    float2 curveP = for_one_polarization(cos(TWO * get_Bragg_angle(E, d)),
                                chih, chih_, chi0,
                                k02, k0s, kHs,
                                alpha, b,
                                thickness, geom);  //# p polarization
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return (float4)(curveS, curveP);//  # , phi.real
}

float4 get_amplitude_internal_E(float E, float d, int4 hkl,
                               float chiToF, float factDW,
                               float beamInDotNormal,
                               float beamOutDotNormal,
                               float beamInDotHNormal,
                               float thickness, int2 geom,
                               const int maxEl,
                               __global float* elements,
                               __global float* f0cfs,
                               __global float2* f1f2)  {
    float waveLength = ch / E;
    float k = twoPI / waveLength;
    float k0s = -beamInDotNormal * k;
    float kHs = -beamOutDotNormal * k;
    float revd = ONE / d;
    float b = k0s / kHs;
    float k0H = fabs(beamInDotHNormal) * (twoPI * revd) * k;
    float k02 = k * k;
    float H2 = pown(twoPI * revd, 2);

    float8 F = get_structure_factor_general_E(E, factDW, hkl, HALF * revd,
                               maxEl, elements, f0cfs, f1f2);

    mem_fence(CLK_LOCAL_MEM_FENCE);
    float2 F0 = (F.lo).lo;
    float2 Fhkl = (F.lo).hi;
    float2 Fhkl_ = (F.hi).lo;
    float lambdaSquare = waveLength * waveLength;
    float chiToFlambdaSquare = chiToF * lambdaSquare;
    float2 chi0 = conj_c(F0) * chiToFlambdaSquare;
    float2 chih = conj_c(Fhkl) * chiToFlambdaSquare;
    float2 chih_ = conj_c(Fhkl_) * chiToFlambdaSquare;
    float2 alpha = r2cmp((HALF*H2 - k0H)/k02) + chi0 * HALF * (1/b - 1);
    float2 curveS = for_one_polarization(ONE,
                                chih, chih_, chi0,
                                k02, k0s, kHs,
                                alpha, b,
                                thickness, geom);  //# s polarization
    mem_fence(CLK_LOCAL_MEM_FENCE);
    float2 curveP = for_one_polarization(cos(TWO * get_Bragg_angle(E, d)),
                                chih, chih_, chi0,
                                k02, k0s, kHs,
                                alpha, b,
                                thickness, geom);  //# p polarization
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return (float4)(curveS, curveP);//  # , phi.real
}


float4 get_amplitude_multilayer_internal(const int npairs,
                                            float2 rbs_si,
                                            float2 rbs_pi,
                                            float2 rtb_si,
                                            float2 rtb_pi,
                                            float2 rvt_si,
                                            float2 rvt_pi,
                                            float2 p2ti,
                                            float2 p2bi)  {
    int i;
    float2 rij_s, rij_p, p2i, rj_s, rj_p, rj2i, ri_si, ri_pi;
    rj_s = rbs_si;
    rj_p = rbs_pi;
    int lsw = -1;

    for (i=2*npairs-1;i>0;i--)
    {
        lsw = -lsw;
        rij_s = lsw * rtb_si;
        rij_p = lsw * rtb_pi;
        p2i = (lsw + ONE) * p2bi * HALF - (lsw - ONE) * p2ti * HALF;

        rj2i = prod_c(rj_s, p2i);

        ri_si = div_c((rij_s + rj2i),
                (cmp1 + prod_c(rij_s, rj2i)));
        rj2i = prod_c(rj_p, p2i);
        ri_pi = div_c((rij_p + rj2i),
                (cmp1 + prod_c(rij_p, rj2i)));
        rj_s = ri_si;
        rj_p = ri_pi;
    }

    rij_s = rvt_si;
    rij_p = rvt_pi;

    p2i = p2ti;
    rj2i = prod_c(rj_s,p2i);
    ri_si = div_c((rij_s + rj2i),
                    (cmp1 + prod_c(rij_s, rj2i)));
    rj2i = prod_c(rj_p,p2i);
    ri_pi = div_c((rij_p + rj2i),
                    (cmp1 + prod_c(rij_p, rj2i)));
    return (float4)(ri_si, ri_pi);
}

float4 get_amplitude_graded_multilayer_internal(const int npairs,
                                            float2 rbs_si,
                                            float2 rbs_pi,
                                            float2 rtb_si,
                                            float2 rtb_pi,
                                            float2 rvt_si,
                                            float2 rvt_pi,
                                            float2 qti,
                                            float2 qbi,
										__global float* dti,
										__global float* dbi)  {
    int i;
    float2 rij_s, rij_p, p2i, rj_s, rj_p, rj2i, ri_si, ri_pi;
	float2 p2bi, p2ti;
    rj_s = rbs_si;
    rj_p = rbs_pi;
    int lsw = -1;
    for (i=2*npairs-1;i>0;i--)
    {
        lsw = -lsw;
        rij_s = lsw * rtb_si;
        rij_p = lsw * rtb_pi;

		if (lsw>0) {
			//p2b
			p2bi = exp_c(prod_c(cmpi1, dbi[i/2] * qbi));
            p2i = p2bi;
		}
		else {//p2t
			p2ti = exp_c(prod_c(cmpi1, dti[i/2] * qti));
            p2i = p2ti;
		}
        rj2i = prod_c(rj_s, p2i);

        ri_si = div_c((rij_s + rj2i),
                (cmp1 + prod_c(rij_s, rj2i)));
        rj2i = prod_c(rj_p, p2i);
        ri_pi = div_c((rij_p + rj2i),
                (cmp1 + prod_c(rij_p, rj2i)));
        rj_s = ri_si;
        rj_p = ri_pi;
    }

    rij_s = rvt_si;
    rij_p = rvt_pi;
    p2i = exp_c(prod_c(cmpi1, dti[0] * qti));
    rj2i = prod_c(rj_s,p2i);
    ri_si = div_c((rij_s + rj2i),
                    (cmp1 + prod_c(rij_s, rj2i)));
    rj2i = prod_c(rj_p,p2i);
    ri_pi = div_c((rij_p + rj2i),
                    (cmp1 + prod_c(rij_p, rj2i)));
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return (float4)(ri_si, ri_pi);
}

float4 get_amplitude_graded_multilayer_internal_tran(
        const int npairs,
        const float dsi,
        float2 rvt_si,
        float2 rvt_pi,
        float2 tvt_si,
        float2 tvt_pi,
        float2 rbs_si,
        float2 rbs_pi,
        float2 tbs_si,
        float2 tbs_pi,
        float2 rsv_si,
        float2 rsv_pi,
        float2 tsv_si,
        float2 tsv_pi,
        float2 rbt_si,
        float2 rbt_pi,
        float2 tbt_si,
        float2 tbt_pi,
        float2 rtb_si,
        float2 rtb_pi,
        float2 ttb_si,
        float2 ttb_pi,
        float2 qti,
        float2 qbi,
        float2 qsi,
        __global float* dti,
		__global float* dbi)  {
    int i;
    float2 rij_s, rij_p, tij_s, tij_p;
    float2 rj_s, rj_p, tj_s, tj_p;
    float2 ri_si, ri_pi, ti_si, ti_pi;
	 float2 iQT, p1i, p2i, rj2i, tj1i;

    rj_s = rsv_si;
    rj_p = rsv_pi;
    tj_s = tsv_si;
    tj_p = tsv_pi;

    for (i=2*npairs;i>0;i--)
    {
        if (i%2 == 0) {
            if (i == 0) {  // topmost layer
                rij_s = rvt_si;
                rij_p = rvt_pi;
                tij_s = tvt_si;
                tij_p = tvt_pi;
                iQT = qti * dti[i/2];
            } else if (i == 2*npairs) {  // substrate
                rij_s = rbs_si;
                rij_p = rbs_pi;
                tij_s = tbs_si;
                tij_p = tbs_pi;
                iQT = qsi * dsi;
            } else {
                rij_s = rbt_si;
                rij_p = rbt_pi;
                tij_s = tbt_si;
                tij_p = tbt_pi;
                iQT = qti * dti[i/2];
            }
        } else {
            rij_s = rtb_si;
            rij_p = rtb_pi;
            tij_s = ttb_si;
            tij_p = ttb_pi;
            iQT = qbi * dbi[i/2];
        }
        p1i = exp_c(prod_c(HALF*cmpi1, iQT));
        p2i = prod_c(p1i, p1i);

        rj2i = prod_c(rj_s, p2i);
        tj1i = prod_c(tj_s, p1i);
        ri_si = div_c((rij_s + rj2i), (cmp1 + prod_c(rij_s, rj2i)));
        ti_si = div_c(prod_c(tij_s, tj1i), (cmp1 + prod_c(rij_s, rj2i)));
        
        rj2i = prod_c(rj_p, p2i);
        tj1i = prod_c(tj_p, p1i);
        ri_pi = div_c((rij_p + rj2i), (cmp1 + prod_c(rij_p, rj2i)));
        ti_pi = div_c(prod_c(tij_p, tj1i), (cmp1 + prod_c(rij_p, rj2i)));

        rj_s = ri_si;
        rj_p = ri_pi;
        tj_s = ti_si;
        tj_p = ti_pi;
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    return (float4)(ti_si, ti_pi);
}

__kernel void get_amplitude(const int4 hkl,
                            const float factDW,
                            const float thickness,
                            const int geom_b,
                            const int maxEl,
                            __global float* BeamInDotNormal_gl,
                            __global float* BeamOutDotNormal_gl,
                            __global float* BeamInDotHNormal_gl,
                            __global float* Energy_gl,
                            __global float* IPDistance_gl,
                            __global float* ChiToF_gl,
                            __global float* elements,
                            __global float* f0cfs,
                            __global float* E_vector,
                            __global float* f1_vector,
                            __global float* f2_vector,
                            __global float2* refl_s,
                            __global float2* refl_p)  {
    unsigned int ii = get_global_id(0);
    float4 amplitudes;
    int2 geom;
    geom.lo = geom_b > 1 ? 1 : 0;
    geom.hi = (int)(fabs(remainder(geom_b,TWO)));
    amplitudes = get_amplitude_internal(Energy_gl[ii], IPDistance_gl[ii], hkl,
                               ChiToF_gl[ii], factDW,
                               BeamInDotNormal_gl[ii], BeamOutDotNormal_gl[ii],
                               BeamInDotHNormal_gl[ii],
                               thickness, geom, maxEl, elements,
                               f0cfs, E_vector,
                               f1_vector, f2_vector);
    refl_s[ii] = amplitudes.lo;
    refl_p[ii] = amplitudes.hi;
    mem_fence(CLK_LOCAL_MEM_FENCE);
}

__kernel void get_amplitude_multilayer(const int npairs,
                            __global float2* rbs_s,
                            __global float2* rbs_p,
                            __global float2* rtb_s,
                            __global float2* rtb_p,
                            __global float2* rvt_s,
                            __global float2* rvt_p,
                            __global float2* p2t,
                            __global float2* p2b,
                            __global float2* ri_s,
                            __global float2* ri_p)  {
    unsigned int ii = get_global_id(0);
    float4 amplitudes;
    amplitudes = get_amplitude_multilayer_internal(npairs,
                    rbs_s[ii], rbs_p[ii], rtb_s[ii], rtb_p[ii],
                    rvt_s[ii], rvt_p[ii], p2t[ii], p2b[ii]);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    ri_s[ii] = amplitudes.s01;
    ri_p[ii] = amplitudes.s23;
    mem_fence(CLK_LOCAL_MEM_FENCE);
}

__kernel void get_amplitude_graded_multilayer(const int npairs,
                            __global float2* rbs_s,
                            __global float2* rbs_p,
                            __global float2* rtb_s,
                            __global float2* rtb_p,
                            __global float2* rvt_s,
                            __global float2* rvt_p,
                            __global float2* qt_glo,
                            __global float2* qb_glo,
                            __global float* dti,
                            __global float* dbi,
                            __global float2* ri_s,
                            __global float2* ri_p)  {
    unsigned int ii = get_global_id(0);
    float4 amplitudes;
    amplitudes = get_amplitude_graded_multilayer_internal(npairs,
                    rbs_s[ii], rbs_p[ii], rtb_s[ii], rtb_p[ii],
                    rvt_s[ii], rvt_p[ii], qt_glo[ii], qb_glo[ii], dti, dbi);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    ri_s[ii] = amplitudes.s01;
    ri_p[ii] = amplitudes.s23;
    mem_fence(CLK_LOCAL_MEM_FENCE);
}

__kernel void get_amplitude_graded_multilayer_tran(
        const int npairs,
        const float dsi,
        __global float2* rvt_s,
        __global float2* rvt_p,
        __global float2* tvt_s,
        __global float2* tvt_p,
        __global float2* rbs_s,
        __global float2* rbs_p,
        __global float2* tbs_s,
        __global float2* tbs_p,
        __global float2* rsv_s,
        __global float2* rsv_p,
        __global float2* tsv_s,
        __global float2* tsv_p,
        __global float2* rbt_s,
        __global float2* rbt_p,
        __global float2* tbt_s,
        __global float2* tbt_p,
        __global float2* rtb_s,
        __global float2* rtb_p,
        __global float2* ttb_s,
        __global float2* ttb_p,
        __global float2* qt_glo,
        __global float2* qb_glo,
        __global float2* qs_glo,
        __global float* dti,
        __global float* dbi,
        __global float2* ti_s,
        __global float2* ti_p)  {
    unsigned int ii = get_global_id(0);
    float4 amplitudes;
    amplitudes = get_amplitude_graded_multilayer_internal_tran(
        npairs, dsi,
        rvt_s[ii], rvt_p[ii], tvt_s[ii], tvt_p[ii],
        rbs_s[ii], rbs_p[ii], tbs_s[ii], tbs_p[ii],
        rsv_s[ii], rsv_p[ii], tsv_s[ii], tsv_p[ii],
        rbt_s[ii], rbt_p[ii], tbt_s[ii], tbt_p[ii],
        rtb_s[ii], rtb_p[ii], ttb_s[ii], ttb_p[ii],
        qt_glo[ii], qb_glo[ii], qs_glo[ii], dti, dbi);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    ti_s[ii] = amplitudes.s01;
    ti_p[ii] = amplitudes.s23;
    mem_fence(CLK_LOCAL_MEM_FENCE);
}

__kernel void tt_laue_spherical_single(
                  const int nLayers,
                  const float2 Wgrad,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0t,
                  __global float2 *Dht,
                  __global float2 *D0,
                  __global float2 *Dh
                  )
{
      float2 C1, C2, C3, C4;
      float8 D0_line, Dh_line, Mvec;
      unsigned int ii = get_global_id(0);
      unsigned int iloc = ii*(nLayers+3);
      int i, j;
      C1 = (float2)(1,0) - W[ii];
      C2 = (float2)(1,0) - V[ii];
      C3 = (float2)(1,0) + W[ii];
      C4 = (float2)(1,0) + V[ii];

      float2 AB = prod_c(A[ii], B[ii]);

      D0_line = (float8)(prod_c(C4, C1),
                         prod_c(A[ii], C1),
                         AB,
                         prod_c(A[ii], C3));
      Dh_line = (float8)(prod_c(B[ii], C4),
                         AB,
                         prod_c(B[ii], C2),
                         prod_c(C3, C2));

      D0t[iloc] = (float2)(0.0, 0.0);
      Dht[iloc] = (float2)(0.0, 0.0);
      D0t[iloc+1] = (float2)(1.0, 0.0);
      Dht[iloc+1] = (float2)(0.0, 0.0);

      float2 EEr = rec_c(prod_c(C1, C2) - AB);

      for (i=0; i<nLayers-1; i++)
        {
          // i -- ordinary number of the layer
          D0t[iloc+i+3] = (float2)(0.0, 0.0);
          Dht[iloc+i+3] = (float2)(0.0, 0.0);

          for (j=0; j<i+2; j++)
            {
              // j -- ordinary number of the element in the layer
              Mvec = (float8)(D0t[iloc+j+1],
                               Dht[iloc+j+1],
                               D0t[iloc+j],
                               Dht[iloc+j]);
              D0t[iloc+j+1] = prod_c(EEr, dot_c4(D0_line, Mvec));
              Dht[iloc+j+1] = prod_c(EEr, dot_c4(Dh_line, Mvec));
              //printf("Ray %i, Layer %i, Element %i\n", ii, i, j);
            }
//          D0[ii*(nLayers+1)+i] = D0t[iloc+i+1];
//          Dh[ii*(nLayers+1)+i] = Dht[iloc+i+1];
        }
      //if (ii==0) printf("Calculation completed\n");
      mem_fence(CLK_LOCAL_MEM_FENCE);
      for (i=0; i<(nLayers+1); i++)
        {
          D0[ii*(nLayers+1)+i] = D0t[iloc+i+1];
          Dh[ii*(nLayers+1)+i] = Dht[iloc+i+1];
        }
//    D0[ii] = D0t[iloc+nLayers-1];
//    Dh[ii] = Dht[iloc+nLayers-1];
}

__kernel void tt_laue_plain(
                  const int nLayers,
                  const float2 Wgrad,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0,
                  __global float2 *Dh
                  )
{
      float2 C1, C2, C3, C4, D0t, Dht;
      float8 D0_line, Dh_line, Mvec;
      unsigned int ii = get_global_id(0);

      int i;
      C1 = (float2)(1.0, 0.0) - W[ii];
      C2 = (float2)(1.0, 0.0) - V[ii];
      C3 = (float2)(1.0, 0.0) + W[ii];
      C4 = (float2)(1.0, 0.0) + V[ii];

      float2 AB = prod_c(A[ii], B[ii]);

      D0_line = (float8)(prod_c(C4, C1),
                          prod_c(A[ii], C1),
                          AB,
                          prod_c(A[ii], C3));
      Dh_line = (float8)(prod_c(B[ii], C4),
                          AB,
                          prod_c(B[ii], C2),
                          prod_c(C3, C2));

      D0t = (float2)(1.0, 0.0);
      Dht = (float2)(0.0, 0.0);

      float2 EEr = rec_c(prod_c(C1, C2) - AB);
      for (i=0; i<nLayers; i++) {
          Mvec = (float8)(D0t,
                           Dht,
                           D0t,
                           Dht);
          D0t = prod_c(EEr, dot_c4(D0_line, Mvec));
          Dht = prod_c(EEr, dot_c4(Dh_line, Mvec));
        }

      mem_fence(CLK_LOCAL_MEM_FENCE);
      D0[ii] = D0t;
      Dh[ii] = Dht;
}

__kernel void tt_laue_plain_bent(
                  const int nOfLayers,
//                  const float2 Wgrad,
                  __global float2 *Wgrad_global,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0,
                  __global float2 *Dh
                  )
{

    float2 C1, C2, C3, C4, D0t, Dht, Aloc, Bloc, Wloc, Vloc, Wi, Wgrad;
    float8 D0_line, Dh_line, Mvec;
    float2 cOne = (float2)(1.0, 0.0);
    unsigned int ii = get_global_id(0);
    int i;
    
    Aloc = A[ii]; Bloc = B[ii]; Wloc = W[ii];
    Vloc = V[ii];
    Wgrad = Wgrad_global[ii];
    float2 AB = prod_c(Aloc, Bloc);
    C2 = cOne - Vloc;
    C4 = cOne + Vloc;

    float2 BC2 = prod_c(Bloc, C2);
    float2 BC4 = prod_c(Bloc, C4);

    D0t = cOne;
    Dht = (float2)(0.0, 0.0);

      for (i=0; i<nOfLayers; i++) {
          Wi = (i == 0) ? Wloc : Wloc + Wgrad*((float)i - HALF);
          C1 = cOne - Wi;
          C3 = cOne + Wi;

          D0_line = (float8)(prod_c(C4, C1),
                              prod_c(Aloc, C1),
                              AB,
                              prod_c(Aloc, C3));
          Dh_line = (float8)(BC4,
                              AB,
                              BC2,
                              prod_c(C3, C2));

          float2 EEr = rec_c(prod_c(C1, C2) - AB);
          Mvec = (float8)(D0t,
                           Dht,
                           D0t,
                           Dht);
          D0t = prod_c(EEr, dot_c4(D0_line, Mvec));
          Dht = prod_c(EEr, dot_c4(Dh_line, Mvec));
        }

      mem_fence(CLK_LOCAL_MEM_FENCE);
      D0[ii] = D0t;
      Dh[ii] = Dht;
}

/*
__kernel void tt_bragg_plain_single(
                  const int nLayers,
                  const float2 Wgrad,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0,
                  __global float2 *Dh
                  )
{
      float2 C1, C2, C3, C4, D0t, Dht;
      float8 D0_line, Dh_line, Mvec;
      unsigned int ii = get_global_id(0);

      int i;
      C1 = (float2)(1.0, 0.0) - W[ii];
      C2 = (float2)(1.0, 0.0) - V[ii];
      C3 = (float2)(1.0, 0.0) + W[ii];
      C4 = (float2)(1.0, 0.0) + V[ii];

      float2 AB = prod_c(A[ii], B[ii]);

      D0_line = (float8)(prod_c(C4, C1),
                          prod_c(A[ii], C1),
                          AB,
                          prod_c(A[ii], C3));
      Dh_line = (float8)(prod_c(B[ii], C4),
                          AB,
                          prod_c(B[ii], C2),
                          prod_c(C3, C2));

      D0t = (float2)(1.0, 0.0);
      Dht = (float2)(0.0, 0.0);

      float2 EEr = rec_c(prod_c(C1, C2) - AB);
      for (i=0; i<nLayers; i++) {
          Mvec = (float8)(D0t,
                           Dht,
                           D0t,
                           Dht);
          D0t = prod_c(EEr, dot_c4(D0_line, Mvec));
          Dht = prod_c(EEr, dot_c4(Dh_line, Mvec));
        }

      mem_fence(CLK_LOCAL_MEM_FENCE);
      D0[ii] = D0t;
      Dh[ii] = Dht;
}
*/
__kernel void tt_bragg_plain(
                  const int nLayers,
                  const float2 Wgrad,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0t,
                  __global float2 *Dht,
                  __global float2 *D0,
                  __global float2 *Dh
//                  __global float2 *Dh_pic
                  )
{
    int arrSize = PRIVATELAYERS;
    __private float2 D0Arr[PRIVATELAYERS+1];
    __private float2 DhArr[PRIVATELAYERS+1];
    float2 C1,C2,C3,C4;
    float8 D0_line, Dh_line, Mvec;
    int diff, stripeWidth, stripeHeight;

    unsigned int ii = get_global_id(0);
    int i, j, n;
    unsigned int cnt_loc = ii * (nLayers+1);
    C1 = (float2)(1,0) - W[ii];
    C2 = (float2)(1,0) - V[ii];
    C3 = (float2)(1,0) + W[ii];
    C4 = (float2)(1,0) + V[ii];

    float2 AB = prod_c(A[ii], B[ii]);
    D0_line = (float8)(prod_c(C4, C1),
                        prod_c(A[ii], C1),
                        AB,
                        prod_c(A[ii], C3));
    Dh_line = (float8)(prod_c(B[ii], C4),
                        AB,
                        prod_c(B[ii], C2),
                        prod_c(C3, C2));
    float2 EEr = rec_c(prod_c(C1, C2) - AB);

    D0t[cnt_loc] = (float2)(0.0, 0.0);
    Dht[cnt_loc] = (float2)(0.0, 0.0);

    int nStripes = ceil((float)nLayers/(float)arrSize);

    for (n=0; n<nStripes; n++) {
        diff = nLayers - n*arrSize;
        stripeWidth = min(arrSize, diff);
        stripeHeight = n*arrSize;
        for (j=1; j<stripeWidth+1; j++) {
            D0Arr[j] = (float2)(0.0, 0.0);
            DhArr[j] = (float2)(0.0, 0.0);
            }
        //Parallel propagation
        for (i=0; i<stripeHeight; i++) {

            D0Arr[0] = D0t[cnt_loc+i];
            DhArr[0] = Dht[cnt_loc+i];
            for (j=0; j<stripeWidth; j++) {
                Mvec = (float8)(D0Arr[j],
                                 DhArr[j],
                                 D0Arr[j+1],
                                 DhArr[j+1]);
                D0Arr[j+1] = prod_c(EEr, dot_c4(D0_line, Mvec));
                DhArr[j+1] = prod_c(EEr, dot_c4(Dh_line, Mvec));
                }
            D0t[cnt_loc+i] = D0Arr[stripeWidth];
            Dht[cnt_loc+i] = DhArr[stripeWidth];
            }
        //Triangular propagation
        for (i=0; i<stripeWidth; i++) {
            D0Arr[i] = (float2)(1.0, 0.0);
            DhArr[i] = (float2)(0.0, 0.0);
            for (j=i; j<stripeWidth; j++) {
                Mvec = (float8)(D0Arr[j],
                                 DhArr[j],
                                 D0Arr[j+1],
                                 DhArr[j+1]);
                D0Arr[j+1] = prod_c(EEr, dot_c4(D0_line, Mvec));
                DhArr[j+1] = prod_c(EEr, dot_c4(Dh_line, Mvec));
                }
            D0t[cnt_loc+stripeHeight+i] = D0Arr[stripeWidth];
            Dht[cnt_loc+stripeHeight+i] = DhArr[stripeWidth];
            }
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
//    if (ii==0) printf("Finally writing Dh=%v2g from Dht[%i]\n", Dht[cnt_loc+nLayers-1], cnt_loc+nLayers-1);
    Dh[ii] = Dht[cnt_loc+nLayers-1];
    D0[ii] = D0t[cnt_loc+nLayers-1];
}

// Below is heavily refactored and optimized code inherited from pyTTE
// https://github.com/aripekka/pyTTE

float strain_term(float z,
                  float thickness,
                  float4 coeffs,  // coeff1, coeff2, invR1, cot_alpha0
                  float4 scap    // sinphi*cosa, sinphi*sina, cosphi*cosa, cosphi*sina
                  ) {
    float xR1 = -z*coeffs.s2*coeffs.s3;
    float zt2 = z+HALF*thickness;
    float duh_dsh = zt2*(scap.s1*coeffs.s1 - scap.s0*coeffs.s2 + scap.s3*coeffs.s0)
                + xR1*(scap.s2 - scap.s1);
//    float duh_dsh = -scap.s0*coeffs.s2*zt2 + scap.s1*(-xR1 + coeffs.s1*zt2)
//                + scap.s2*xR1 + scap.s3*coeffs.s0*zt2;
    return duh_dsh; // h*gammah moved to beta sum, level up
}

float4 ksi_prime(float Cpol,
                float4 ksi,
                float2 c0_strain,
                float2 cb_step,
                float2 ch_step) {

    float2 ksi2_sigma = prod_c(ksi.xy, ksi.xy);
    float2 ksi2_pi = prod_c(ksi.zw, ksi.zw);
    float2 ksi_sigma = prod_c(c0_strain, ksi.xy) + prod_c(cb_step, ksi2_sigma) + ch_step;
    float2 ksi_pi = prod_c(c0_strain, ksi.zw) + prod_c(cb_step*Cpol, ksi2_pi) + ch_step*Cpol;
    return (float4)(-ksi_sigma.y, ksi_sigma.x, -ksi_pi.y, ksi_pi.x);
}

float8 d0h_prime(float Cpol,
                float8 d0h,
                float2 c0_strain,
                float2 cb_step,
                float2 ch_step,
                float2 g0_step) {

    float2 dh2_sigma = prod_c(d0h.s01, d0h.s01);
    float2 dh_out_sigma = prod_c(cb_step, dh2_sigma) + prod_c(c0_strain, d0h.s01) + ch_step;
    float2 d0_out_sigma = prod_c(g0_step+prod_c(cb_step, d0h.s01), d0h.s23);

    float2 dh2_pi = prod_c(d0h.s45, d0h.s45);
    float2 dh_out_pi = prod_c(cb_step*Cpol, dh2_pi) + ch_step*Cpol + prod_c(c0_strain, d0h.s45);
    float2 d0_out_pi = prod_c(g0_step+prod_c(cb_step*Cpol, d0h.s45), d0h.s67);

    return (float8)(-dh_out_sigma.y, dh_out_sigma.x,
                    d0_out_sigma.y, -d0_out_sigma.x,
                    -dh_out_pi.y, dh_out_pi.x,
                    d0_out_pi.y, -d0_out_pi.x);
}

float4 ksi_next_rk4(
                float zstep,
                float Cpol,
                float4 ksi,
                float2 ksi_z,
                float2 ksi_h,
                float2 cb,
                float2 ch) {

    float4 k1, k2, k3, k4;

    k1 = ksi_prime(Cpol, ksi, ksi_z, cb, ch);
    k2 = ksi_prime(Cpol, ksi + zstep*k1*HALF, ksi_z+ksi_h, cb, ch);
    k3 = ksi_prime(Cpol, ksi + zstep*k2*HALF, ksi_z+ksi_h, cb, ch);
    k4 = ksi_prime(Cpol, ksi + zstep*k3, ksi_z + TWO*ksi_h, cb, ch);

    return (float4)(ksi + zstep*(k1 + TWO*k2 + TWO*k3 + k4)*REVSIX);
}

float4 ksi_next_rkdp(
                float z,
                float zstep,
                float cz0,
                float *rk_error,
                float Cpol,
                float4 ksi,
                float2 strain_z0,
                float2 cb,
                float2 ch) {

    float4 k1, k2, k3, k4, k5, k6, k7, ksi_new4, ksi_new5;
    float2 strain_z;
    float err_sigma, err_pi;

    strain_z = strain_z0 + (float2)(cz0*(z), 0);
    k1 = zstep * ksi_prime(Cpol, ksi, strain_z, cb, ch);
    strain_z = strain_z0 + (float2)(cz0*(z+zstep/5), 0);
    k2 = zstep * ksi_prime(Cpol, ksi + k1/5, strain_z, cb, ch);
    strain_z = strain_z0 + (float2)(cz0*(z+3*zstep/10), 0);
    k3 = zstep * ksi_prime(Cpol, ksi + 3*k1/40 + 9*k2/40, strain_z, cb, ch);
    strain_z = strain_z0 + (float2)(cz0*(z+4*zstep/5), 0);
    k4 = zstep * ksi_prime(Cpol, ksi + 44*k1/45 - 56*k2/15 + 32*k3/9, strain_z, cb, ch);
    strain_z = strain_z0 + (float2)(cz0*(z+8*zstep/9), 0);
    k5 = zstep * ksi_prime(Cpol, ksi + 19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729, strain_z, cb, ch);
    strain_z = strain_z0 + (float2)(cz0*(z+zstep), 0);
    k6 = zstep * ksi_prime(Cpol, ksi + 9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656, strain_z, cb, ch);
    strain_z = strain_z0 + (float2)(cz0*(z+zstep), 0);
    ksi_new5 = ksi + 35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84;
    k7 = zstep * ksi_prime(Cpol, ksi_new5, strain_z, cb, ch);
    ksi_new4 = ksi + 5179*k1/57600 + 7571*k3/16695 + 393*k4/640 - 92097*k5/339200 + 187*k6/2100 + k7/40;

    err_sigma = fabs(abs_c(ksi_new4.s01) - abs_c(ksi_new5.s01));
    err_pi = fabs(abs_c(ksi_new4.s23) - abs_c(ksi_new5.s23));
    *rk_error = max(err_sigma, err_pi);

    return ksi_new5;
}

float8 d0h_next_rk4(float zstep,
                float Cpol,
                float8 d0h,
                float2 strain_z,
                float2 strain_h,
                float2 cb,
                float2 ch,
                float2 g0) {

    float halfStep = HALF * zstep;
    float8 k1, k2, k3, k4;

    k1 = d0h_prime(Cpol, d0h, strain_z, cb, ch, g0);
    k2 = d0h_prime(Cpol, d0h + zstep*k1*HALF, strain_z+strain_h, cb, ch, g0);
    k3 = d0h_prime(Cpol, d0h + zstep*k2*HALF, strain_z+strain_h, cb, ch, g0);
    k4 = d0h_prime(Cpol, d0h + zstep*k3, strain_z + TWO*strain_h, cb, ch, g0);

    return (float8)(d0h + zstep*(k1 + TWO*k2 + TWO*k3 + k4)*REVSIX);
}

float8 d0h_next_rkdp(
                float z,
                float zstep,
                float cz0,
                float *rk_error,
                float Cpol,
                float8 d0h,
                float2 strain_z0,
                float2 cb,
                float2 ch,
                float2 g0) {

    float8 k1, k2, k3, k4, k5, k6, k7, d0h_new4, d0h_new5;
    float2 strain_z;
    float err_sigma, err_pi;

    strain_z = strain_z0 + (float2)(cz0*(z), 0);
    k1 = zstep * d0h_prime(Cpol, d0h, strain_z, cb, ch, g0);
    strain_z = strain_z0 + (float2)(cz0*(z+zstep/5), 0);
    k2 = zstep * d0h_prime(Cpol, d0h + k1/5, strain_z, cb, ch, g0);
    strain_z = strain_z0 + (float2)(cz0*(z+3*zstep/10), 0);
    k3 = zstep * d0h_prime(Cpol, d0h + 3*k1/40 + 9*k2/40, strain_z, cb, ch, g0);
    strain_z = strain_z0 + (float2)(cz0*(z+4*zstep/5), 0);
    k4 = zstep * d0h_prime(Cpol, d0h + 44*k1/45 - 56*k2/15 + 32*k3/9, strain_z, cb, ch, g0);
    strain_z = strain_z0 + (float2)(cz0*(z+8*zstep/9), 0);
    k5 = zstep * d0h_prime(Cpol, d0h + 19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729, strain_z, cb, ch, g0);
    strain_z = strain_z0 + (float2)(cz0*(z+zstep), 0);
    k6 = zstep * d0h_prime(Cpol, d0h + 9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656, strain_z, cb, ch, g0);
    strain_z = strain_z0 + (float2)(cz0*(z+zstep), 0);
    d0h_new5 = d0h + 35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84;
    k7 = zstep * d0h_prime(Cpol, d0h_new5, strain_z, cb, ch, g0);
    d0h_new4 = d0h + 5179*k1/57600 + 7571*k3/16695 + 393*k4/640 - 92097*k5/339200 + 187*k6/2100 + k7/40;

    err_sigma = fabs(abs_c(d0h_new4.s01) - abs_c(d0h_new5.s01));
    err_pi = fabs(abs_c(d0h_new4.s45) - abs_c(d0h_new5.s45));
    *rk_error = max(err_sigma, err_pi);

    return d0h_new5;
}

__kernel void estimate_bent_width(const float c1,
                                 const float c2,
                                 const float ir1,
                                 const float phi,
                                 const float t,
                                 const float h,
                                 __global float *beta_const,
                                 __global float *thetaB,
                                 __global float *chcb_mod_global,

                                 __global float *thmin,                                 
                                 __global float *thmax                                
                                 ){

    float dwt, sin_p, cos_p, sin_a0, cos_a0, sin_ah, cos_ah, cot_a0, zstep, deform;
    float sintb, costb;
    unsigned int ii = get_global_id(0);
    
    sin_p = sincos(phi, &cos_p);        
    sin_a0 = sincos(thetaB[ii]+phi, &cos_a0);
    sin_ah = sincos(thetaB[ii]-phi, &cos_ah);
    
    if (sin_a0 == 0) {
        cot_a0 = 0;} 
    else {
        cot_a0 = cos_a0/sin_a0;}     
    
    zstep = t / 1000.;
    
    float4 coeffs = (float4)(c1, c2, ir1, cot_a0);
    float4 scap = (float4)(sin_p*cos_ah, sin_p*sin_ah,
                           cos_p*cos_ah, cos_p*sin_ah);        

    float sin2tb = sin(2*thetaB[ii]);
    sintb = sincos(thetaB[ii], &costb);

    float def_min = 1e16;
    float def_max = -1e16;

    float z = -t;
    for (int j=0; j<10001; j++) {
        deform = h * strain_term(z, t, coeffs, scap);
        if (deform < def_min) {def_min = deform;}
        if (deform > def_max) {def_max = deform;}
        z += zstep;
    }

    float chcb_mod = chcb_mod_global[ii];

    if (sin2tb > sqrt(2*chcb_mod)) {dwt = 2*chcb_mod*h*costb/sin2tb;}
    else {dwt = sqrt(2*chcb_mod)*h*costb;}

    float beta_min = beta_const[ii] - def_max - 2*dwt;
    float beta_max = beta_const[ii] - def_min + 2*dwt;
    
    float sinthmin = sintb + beta_min/h;
    float sinthmax = sintb + beta_max/h;
    
    thmin[ii] = asin(sinthmin)-thetaB[ii];
    if (sinthmax > 1) {thmax[ii]=asin(PI-sinthmin)-thetaB[ii];}
    else {thmax[ii]=asin(sinthmax)-thetaB[ii];}
    mem_fence(CLK_LOCAL_MEM_FENCE);
}

__kernel void get_amplitudes_pytte(const float c1,
                                   const float c2,
                                   const float ir1,
                                   const float phi,
                                   const float t,
                                   const float tol,
                                   const int stmax,
                                   const int jmax,
                                   const int geometry,
                                   __global float *hgh_global,
                                   __global float *hgh_beta_global,
                                   __global float *thetaB,
                                   __global float *alpha0,
                                   __global float *alphah,
                                   __global float2 *c0_global,  
                                   __global float2 *ch_global,
                                   __global float2 *cb_global,
                                   __global float2 *g0_global,
                                   __global float2 *ampS_out,
                                   __global float2 *ampP_out,
                                   __global float *np_glo
                                   ) {

    unsigned int ii = get_global_id(0);
    unsigned int j;
    float sin_a0, cos_a0, cot_a0, sin_p, cos_p, sin_ah, cos_ah, zstep;
    float Cpol = cos(2*thetaB[ii]);

    sin_a0 = sincos(alpha0[ii], &cos_a0);
    sin_ah = sincos(alphah[ii], &cos_ah);
    sin_p = sincos(phi, &cos_p);

    if (sin_a0 == 0) {
        cot_a0 = 0;}
    else {
        cot_a0 = cos_a0/sin_a0;}
    
    if (geometry==1) {cot_a0=1/cot_a0;};

    float4 scap = (float4)(sin_p*cos_ah, sin_p*sin_ah,
                           cos_p*cos_ah, cos_p*sin_ah);

    float cz1 = scap.s1*c2 - scap.s0*ir1 + scap.s3*c1;
    float cz0 = hgh_global[ii]*(cz1 + ir1*cot_a0*(scap.s1 - scap.s2));
    float cz2 = HALF*t*cz1;

    float2 strain_z0 = c0_global[ii] + (float2)(hgh_beta_global[ii] + hgh_global[ii]*cz2, 0);
    float2 strain_z = strain_z0;

//  Fixed step RK4
//    float2 strain_h = (float2)(HALF*zstep*cz0, 0);
//
//    for (j=0; j<jmax; j++) {
//        strain_z = strain_z0 + (float2)(cz0*(double)z, 0);
//        d0h = d0h_next_rk4(zstep, Cpol, d0h, strain_z, strain_h, cb, ch, g0);
//        z = z + zstep;
//    }
//
//    for (j=0; j<jmax; j++) {
////        ksi_z = c0_step+(float2)(hgh_beta_step+hgh*strain_term_z, 0);
//        strain_z = strain_z_z0 + (float2)(cz0*(double)z, 0);
//        ksi = ksi_next_rk4(zstep, Cpol, ksi, strain_z, strain_h, cb, ch);
//        z = z + zstep;
//        }


    float rke = 0;
    unsigned int np = 0;
    unsigned int tsteps = 0;

    if (geometry==1) {  // Laue. Moving condition outside of the loop is faster
        zstep = -t / (float)jmax;
        float8 d0h_new;
        float8 d0h = (float8)(0, 0, 1, 0, 0, 0, 1, 0);  // Sigma dhd0, Pi dhd0
        float z = 0;

        while ((z > -t)&&(tsteps<stmax)) {
            tsteps += 1;
            d0h_new = d0h_next_rkdp(z, zstep, cz0, &rke, Cpol, d0h, strain_z0, 
                                    cb_global[ii], ch_global[ii], g0_global[ii]);
            if (rke < tol) {
                d0h = d0h_new;
                z = z + zstep;
                np += 1;
            };
            if (rke > 0) {
                zstep *= min(0.9*sqrt(sqrt(tol/rke)), 4.0);
            } else {
                zstep *= 4.0;
            }    
            if (z+zstep < -t) {
                zstep = -t-z;
            };
        }
        mem_fence(CLK_LOCAL_MEM_FENCE);
        ampS_out[ii] = prod_c(d0h.s01, d0h.s23);  // dh sigma
        ampP_out[ii] = prod_c(d0h.s45, d0h.s67);  // dh pi

    } else {  // Bragg
        zstep = t / (float)jmax;
        float4 ksi_new;
        float4 ksi = (float4)(0, 0, 0, 0);
        float z = -t;
    
        while ((z < 0)&&(tsteps<stmax)) {
            tsteps += 1;
            ksi_new = ksi_next_rkdp(z, zstep, cz0, &rke, Cpol, ksi, strain_z0,
                                    cb_global[ii], ch_global[ii]);
            if (rke < tol) {
                ksi = ksi_new;
                z = z + zstep;
                np += 1;
            };
            if (rke > 0) {
                zstep *= min(0.9*sqrt(sqrt(tol/rke)), 4.0);
            } else {
                zstep *= 4.0;
            }
    
            if (z+zstep > 0) {
                zstep = -z;
            };
    
        }
        mem_fence(CLK_LOCAL_MEM_FENCE);
        ampS_out[ii] = ksi.xy;  // sigma
        ampP_out[ii] = ksi.zw;  // pi
    }

    np_glo[ii] = (float)np;
}

__kernel void tt_bragg_plain_bent(
                  const int nLayers,
                  __global float2 *Wgrad_global,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0t,
                  __global float2 *Dht,
                  __global float2 *D0,
                  __global float2 *Dh
                  )
{
    int arrSize = PRIVATELAYERS;
    __private float2 D0Arr[PRIVATELAYERS+1];
    __private float2 DhArr[PRIVATELAYERS+1];
    float2 C1, C2, C3, C4, Aloc, Bloc, Wloc, Vloc, Wi, Wgrad;
    float8 D0_line, Dh_line, Mvec;
    float2 cOne = (float2)(1.0, 0.0);
    int diff, stripeWidth, stripeHeight;

    unsigned int ii = get_global_id(0);
    int i, j, n, iLayer;
    unsigned int cnt_loc = ii * (nLayers+1);

    Aloc = A[ii]; Bloc = B[ii]; Wloc = W[ii];
    Vloc = V[ii];
    Wgrad = Wgrad_global[ii];
    float2 AB = prod_c(Aloc, Bloc);
    C2 = cOne - Vloc;
    C4 = cOne + Vloc;

    float2 BC2 = prod_c(Bloc, C2);
    float2 BC4 = prod_c(Bloc, C4);

    D0t[cnt_loc] = (float2)(0.0, 0.0);
    Dht[cnt_loc] = (float2)(0.0, 0.0);

    int nStripes = ceil((float)nLayers/(float)arrSize);

    for (n=0; n<nStripes; n++) {
        diff = nLayers - n*arrSize;
        stripeWidth = min(arrSize, diff);
        stripeHeight = n*arrSize;
        for (j=1; j<stripeWidth+1; j++) {
            D0Arr[j] = (float2)(0.0, 0.0);
            DhArr[j] = (float2)(0.0, 0.0);
            }
        //Parallel propagation
        for (i=0; i<stripeHeight; i++) {

            D0Arr[0] = D0t[cnt_loc+i];
            DhArr[0] = Dht[cnt_loc+i];
            for (j=0; j<stripeWidth; j++) {

                iLayer = n*arrSize + i + j + 2;
//                if (ii==0) printf("Parallel: n, i, j, iLayer: %i, %i, %i, %i\n", n, i, j, iLayer);
                Wi = (iLayer == 0) ? Wloc : Wloc + Wgrad * fabs((float)(iLayer)  - HALF) * HALF;
                C1 = cOne - Wi;
                C3 = cOne + Wi;

                D0_line = (float8)(prod_c(C4, C1),
                                    prod_c(Aloc, C1),
                                    AB,
                                    prod_c(Aloc, C3));
                Dh_line = (float8)(BC4,
                                    AB,
                                    BC2,
                                    prod_c(C3, C2));

                float2 EEr = rec_c(prod_c(C1, C2) - AB);

                Mvec = (float8)(D0Arr[j],
                                 DhArr[j],
                                 D0Arr[j+1],
                                 DhArr[j+1]);
                D0Arr[j+1] = prod_c(EEr, dot_c4(D0_line, Mvec));
                DhArr[j+1] = prod_c(EEr, dot_c4(Dh_line, Mvec));
                }
            D0t[cnt_loc+i] = D0Arr[stripeWidth];
            Dht[cnt_loc+i] = DhArr[stripeWidth];
            }
        //Triangular propagation
        for (i=0; i<stripeWidth; i++) {

            D0Arr[i] = (float2)(1.0, 0.0);
            DhArr[i] = (float2)(0.0, 0.0);
            for (j=i; j<stripeWidth; j++) {

                iLayer = 2*n*(arrSize) + i + j + 2;
//                if (ii==0) printf("Triangular: n, i, j, iLayer: %i, %i, %i, %i\n", n, i, j, iLayer);
                Wi = (iLayer == 0) ? Wloc : Wloc + Wgrad * fabs((float)(iLayer)  - HALF) * HALF;
                C1 = cOne - Wi;
                C3 = cOne + Wi;

                D0_line = (float8)(prod_c(C4, C1),
                                    prod_c(Aloc, C1),
                                    AB,
                                    prod_c(Aloc, C3));
                Dh_line = (float8)(BC4,
                                    AB,
                                    BC2,
                                    prod_c(C3, C2));

                float2 EEr = rec_c(prod_c(C1, C2) - AB);

                Mvec = (float8)(D0Arr[j],
                                 DhArr[j],
                                 D0Arr[j+1],
                                 DhArr[j+1]);
                D0Arr[j+1] = prod_c(EEr, dot_c4(D0_line, Mvec));
                DhArr[j+1] = prod_c(EEr, dot_c4(Dh_line, Mvec));
                }
            D0t[cnt_loc+stripeHeight+i] = D0Arr[stripeWidth];
            Dht[cnt_loc+stripeHeight+i] = DhArr[stripeWidth];
            }
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    Dh[ii] = Dht[cnt_loc+nLayers-1];
    D0[ii] = D0t[cnt_loc+nLayers-1];
}


__kernel void tt_bragg_plain_thickness_bent(
                  const int nLayers,
                  __global float2 *Wgrad_global,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0t,
                  __global float2 *Dht,
                  __global float2 *D0,
                  __global float2 *Dh
                  )
{
    int arrSize = PRIVATELAYERS;
    __private float2 D0Arr[PRIVATELAYERS+1];
    __private float2 DhArr[PRIVATELAYERS+1];
    float2 C1, C2, C3, C4, Aloc, Bloc, Wloc, Vloc, Wi, Wgrad;
    float8 D0_line, Dh_line, Mvec;
    float2 cOne = (float2)(1.0, 0.0);
    int diff, stripeWidth, stripeHeight;

    unsigned int ii = get_global_id(0);
    int i, j, n, iLayer;
    unsigned int cnt_loc = ii * (nLayers+1);

    Aloc = A[ii]; Bloc = B[ii]; Wloc = W[ii];
    Vloc = V[ii];
    Wgrad = Wgrad_global[ii];
    float2 AB = prod_c(Aloc, Bloc);
    C2 = cOne - Vloc;
    C4 = cOne + Vloc;

    float2 BC2 = prod_c(Bloc, C2);
    float2 BC4 = prod_c(Bloc, C4);

    D0t[cnt_loc] = (float2)(0.0, 0.0);
    Dht[cnt_loc] = (float2)(0.0, 0.0);

    int nStripes = ceil((float)nLayers/(float)arrSize);

    for (n=0; n<nStripes; n++) {
        diff = nLayers - n*arrSize;
        stripeWidth = min(arrSize, diff);
        stripeHeight = n*arrSize;
        for (j=1; j<stripeWidth+1; j++) {
            D0Arr[j] = (float2)(0.0, 0.0);
            DhArr[j] = (float2)(0.0, 0.0);
            }
        //Parallel propagation
        for (i=0; i<stripeHeight; i++) {

            D0Arr[0] = D0t[cnt_loc+i];
            DhArr[0] = Dht[cnt_loc+i];
            for (j=0; j<stripeWidth; j++) {

                iLayer = n*arrSize + i + j + 2;
//                if (ii==0) printf("Parallel: n, i, j, iLayer: %i, %i, %i, %i\n", n, i, j, iLayer);
                Wi = (iLayer == 0) ? Wloc : Wloc + Wgrad * fabs((float)(iLayer)  - HALF) * HALF;
                C1 = cOne - Wi;
                C3 = cOne + Wi;

                D0_line = (float8)(prod_c(C4, C1),
                                    prod_c(Aloc, C1),
                                    AB,
                                    prod_c(Aloc, C3));
                Dh_line = (float8)(BC4,
                                    AB,
                                    BC2,
                                    prod_c(C3, C2));

                float2 EEr = rec_c(prod_c(C1, C2) - AB);

                Mvec = (float8)(D0Arr[j],
                                 DhArr[j],
                                 D0Arr[j+1],
                                 DhArr[j+1]);
                D0Arr[j+1] = prod_c(EEr, dot_c4(D0_line, Mvec));
                DhArr[j+1] = prod_c(EEr, dot_c4(Dh_line, Mvec));
                }
            D0t[cnt_loc+i] = D0Arr[stripeWidth];
            Dht[cnt_loc+i] = DhArr[stripeWidth];
            }
        //Triangular propagation
        for (i=0; i<stripeWidth; i++) {

            D0Arr[i] = (float2)(1.0, 0.0);
            DhArr[i] = (float2)(0.0, 0.0);
            for (j=i; j<stripeWidth; j++) {

                iLayer = 2*n*(arrSize) + i + j + 2;
//                if (ii==0) printf("Triangular: n, i, j, iLayer: %i, %i, %i, %i\n", n, i, j, iLayer);
                Wi = (iLayer == 0) ? Wloc : Wloc + Wgrad * fabs((float)(iLayer)  - HALF) * HALF;
                C1 = cOne - Wi;
                C3 = cOne + Wi;

                D0_line = (float8)(prod_c(C4, C1),
                                    prod_c(Aloc, C1),
                                    AB,
                                    prod_c(Aloc, C3));
                Dh_line = (float8)(BC4,
                                    AB,
                                    BC2,
                                    prod_c(C3, C2));

                float2 EEr = rec_c(prod_c(C1, C2) - AB);

                Mvec = (float8)(D0Arr[j],
                                 DhArr[j],
                                 D0Arr[j+1],
                                 DhArr[j+1]);
                D0Arr[j+1] = prod_c(EEr, dot_c4(D0_line, Mvec));
                DhArr[j+1] = prod_c(EEr, dot_c4(Dh_line, Mvec));
                }
            D0t[cnt_loc+stripeHeight+i] = D0Arr[stripeWidth];
            Dht[cnt_loc+stripeHeight+i] = DhArr[stripeWidth];
            }
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    Dh[ii] = Dht[cnt_loc+nLayers-1];
    D0[ii] = D0t[cnt_loc+nLayers-1];
}

__kernel void tt_laue_spherical(
                  const int nLayers,
                  const float2 Wgrad,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0t,
                  __global float2 *Dht,
                  __global float2 *D0,
                  __global float2 *Dh
                  )
{
    int arrSize = PRIVATELAYERS;
    __private float2 D0Arr[PRIVATELAYERS+1];
    __private float2 DhArr[PRIVATELAYERS+1];
    float2 C1,C2,C3,C4;
    float8 D0_line, Dh_line, Mvec;
    int diff, stripeWidth, stripeHeight;

    unsigned int ii = get_global_id(0);
    int i, j, n;
    unsigned int cnt_loc = ii * (nLayers+1);
    C1 = (float2)(1,0) - W[ii];
    C2 = (float2)(1,0) - V[ii];
    C3 = (float2)(1,0) + W[ii];
    C4 = (float2)(1,0) + V[ii];

    float2 AB = prod_c(A[ii], B[ii]);
    D0_line = (float8)(prod_c(C4, C1),
                        prod_c(A[ii], C1),
                        AB,
                        prod_c(A[ii], C3));
    Dh_line = (float8)(prod_c(B[ii], C4),
                        AB,
                        prod_c(B[ii], C2),
                        prod_c(C3, C2));
    float2 EEr = rec_c(prod_c(C1, C2) - AB);

    D0t[cnt_loc] = (float2)(0.0, 0.0);
    Dht[cnt_loc] = (float2)(0.0, 0.0);

    int nStripes = ceil((float)nLayers/(float)(arrSize-1));

    for (n=0; n<nStripes; n++) {
        diff = nLayers - n*(arrSize-1) + 2;
        stripeWidth = min(arrSize, diff);
        stripeHeight = max(nLayers + 1 - (n+1)*(arrSize-1), 0);

        for (j=0; j<stripeWidth; j++) {
            D0Arr[j] = (float2)(0.0, 0.0);
            DhArr[j] = (float2)(0.0, 0.0);
        }
        //Parallel propagation
        for (i=0; i<stripeHeight; i++) {
            D0Arr[0] = D0t[cnt_loc+i];
            DhArr[0] = Dht[cnt_loc+i];
            for (j=1; j<stripeWidth; j++) {
                if ((n==0)&&(i==0)) {
                    D0Arr[1] = (float2)(1.0, 0.0);
                    DhArr[1] = (float2)(0.0, 0.0);
                } else {
                    Mvec = (float8)(D0Arr[j],
                                     DhArr[j],
                                     D0Arr[j-1],
                                     DhArr[j-1]);
                    D0Arr[j] = prod_c(EEr, dot_c4(D0_line, Mvec));
                    DhArr[j] = prod_c(EEr, dot_c4(Dh_line, Mvec));
                }
            }
            D0t[cnt_loc+i] = D0Arr[stripeWidth-1];
            Dht[cnt_loc+i] = DhArr[stripeWidth-1];
        }
        //Triangular propagation
       for (i=0; i<stripeWidth-1; i++) {
            D0Arr[0] = D0t[cnt_loc+i+stripeHeight];
            DhArr[0] = Dht[cnt_loc+i+stripeHeight];
            for (j=1; j<stripeWidth-i; j++) {
                Mvec = (float8)(D0Arr[j],
                                 DhArr[j],
                                 D0Arr[j-1],
                                 DhArr[j-1]);
                D0Arr[j] = prod_c(EEr, dot_c4(D0_line, Mvec));
                DhArr[j] = prod_c(EEr, dot_c4(Dh_line, Mvec));
            }
            D0[ii*(nLayers+1) + (nLayers-stripeHeight) - i] = D0Arr[stripeWidth-i-1];
            Dh[ii*(nLayers+1) + (nLayers-stripeHeight) - i] = DhArr[stripeWidth-i-1];
        }
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
}

__kernel void tt_laue_spherical_bent(
                  const int nLayers,
//                  const float2 Wgrad,
                  __global float2 *Wgrad_global,
                  __global float2 *A,
                  __global float2 *B,
                  __global float2 *W,
                  __global float2 *V,
                  __global float2 *D0t,
                  __global float2 *Dht,
                  __global float2 *D0,
                  __global float2 *Dh
                  )
{
    int arrSize = PRIVATELAYERS;
    __private float2 D0Arr[PRIVATELAYERS+1];
    __private float2 DhArr[PRIVATELAYERS+1];
    float2 C1, C2, C3, C4, Aloc, Bloc, Wloc, Vloc, Wi, Wgrad;
    float8 D0_line, Dh_line, Mvec;
    float2 cOne = (float2)(1.0, 0.0);
    int diff, stripeWidth, stripeHeight;

    unsigned int ii = get_global_id(0);
    int i, j, n, iLayer;
    unsigned int cnt_loc = ii * (nLayers+1);

    Aloc = A[ii]; Bloc = B[ii]; Wloc = W[ii];
    Vloc = V[ii];
    Wgrad = Wgrad_global[ii];
    float2 AB = prod_c(Aloc, Bloc);
    C2 = cOne - Vloc;
    C4 = cOne + Vloc;

    float2 BC2 = prod_c(Bloc, C2);
    float2 BC4 = prod_c(Bloc, C4);

    D0t[cnt_loc] = (float2)(0.0, 0.0);
    Dht[cnt_loc] = (float2)(0.0, 0.0);

    int nStripes = ceil((float)nLayers/(float)(arrSize-1));

    for (n=0; n<nStripes; n++) {
        diff = nLayers - n*(arrSize-1) + 2;
        stripeWidth = min(arrSize, diff);
        stripeHeight = max(nLayers + 1 - (n+1)*(arrSize-1), 0);

        for (j=0; j<stripeWidth; j++) {
            D0Arr[j] = (float2)(0.0, 0.0);
            DhArr[j] = (float2)(0.0, 0.0);
        }
        //Parallel propagation
        for (i=0; i<stripeHeight; i++) {
            D0Arr[0] = D0t[cnt_loc+i];
            DhArr[0] = Dht[cnt_loc+i];
            for (j=1; j<stripeWidth; j++) {
                iLayer = n*(arrSize-1) + i + j - 1;
                Wi = (iLayer == 0) ? Wloc : Wloc + Wgrad*((float)iLayer - HALF);
                C1 = cOne - Wi;
                C3 = cOne + Wi;

                D0_line = (float8)(prod_c(C4, C1),
                                    prod_c(Aloc, C1),
                                    AB,
                                    prod_c(Aloc, C3));
                Dh_line = (float8)(BC4,
                                    AB,
                                    BC2,
                                    prod_c(C3, C2));

                float2 EEr = rec_c(prod_c(C1, C2) - AB);

                if ((n==0)&&(i==0)) {
                    D0Arr[1] = (float2)(1.0, 0.0);
                    DhArr[1] = (float2)(0.0, 0.0);
                } else {
                    Mvec = (float8)(D0Arr[j],
                                     DhArr[j],
                                     D0Arr[j-1],
                                     DhArr[j-1]);
                    D0Arr[j] = prod_c(EEr, dot_c4(D0_line, Mvec));
                    DhArr[j] = prod_c(EEr, dot_c4(Dh_line, Mvec));
                }
            }
            D0t[cnt_loc+i] = D0Arr[stripeWidth-1];
            Dht[cnt_loc+i] = DhArr[stripeWidth-1];
        }
        //Triangular propagation

       for (i=0; i<stripeWidth-1; i++) {
            D0Arr[0] = D0t[cnt_loc+i+stripeHeight];
            DhArr[0] = Dht[cnt_loc+i+stripeHeight];
            for (j=1; j<stripeWidth-i; j++) {
                iLayer = n*(arrSize-1) + i + j + stripeHeight - 1;
                Wi = (iLayer == 0) ? Wloc : Wloc + Wgrad*((float)iLayer - HALF);
                C1 = cOne - Wi;
                C3 = cOne + Wi;

                D0_line = (float8)(prod_c(C4, C1),
                                    prod_c(Aloc, C1),
                                    AB,
                                    prod_c(Aloc, C3));
                Dh_line = (float8)(BC4,
                                    AB,
                                    BC2,
                                    prod_c(C3, C2));

                float2 EEr = rec_c(prod_c(C1, C2) - AB);
                Mvec = (float8)(D0Arr[j],
                                 DhArr[j],
                                 D0Arr[j-1],
                                 DhArr[j-1]);
                D0Arr[j] = prod_c(EEr, dot_c4(D0_line, Mvec));
                DhArr[j] = prod_c(EEr, dot_c4(Dh_line, Mvec));
            }
            D0[ii*(nLayers+1) + (nLayers-stripeHeight) - i] = D0Arr[stripeWidth-i-1];
            Dh[ii*(nLayers+1) + (nLayers-stripeHeight) - i] = DhArr[stripeWidth-i-1];
        }
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
