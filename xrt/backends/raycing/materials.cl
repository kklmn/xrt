//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "22 Jan 2016"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

#include "xrt_complex.cl"

__constant double PI =    3.141592653589793238;
__constant double ch = 12398.4186;  // {5}   {c*h[eV*A]}
__constant double twoPi = 6.283185307179586476;
__constant double chbar = 1973.269606712496640;  // {c*hbar[eV*A]}
__constant double r0 = 2.817940285e-5;  // A
__constant double avogadro = 6.02214199e23;  // atoms/mol

double get_f0(double qOver4pi, int iN, __global double* f0cfs)
  {
    double res = f0cfs[iN*11+5];
    //printf("c=%g\n",f0cfs[Zi*11+5]);
    for (int i=0;i<5;i++)
    {
            //printf("a[%i]=%g\n",i,f0cfs[Zi*11+i]);
            //printf("b[%i]=%g\n",i,f0cfs[Zi*11+i+6]);
            res += f0cfs[iN*11+i] * exp(-f0cfs[iN*11+i+6] * qOver4pi * qOver4pi);
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return res;
  }


double2 get_f1f2(double E, int iN, int VectorMax, 
                 __global double* E_vector, 
                 __global double* f1_vector, __global double* f2_vector)
  {
    //printf("VectMax, E, iN  %i %f %i\n",VectorMax,E,iN);
    //printf("Estart, Eend: %f, %f\n",E_vector[iN*300], E_vector[iN*300+VectorMax]);
    int pn = floor((VectorMax-1)*log(E-E_vector[iN*300])/log(E_vector[iN*300+VectorMax]-E_vector[iN*300]));
    //printf("pn %i\n",pn);
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

    double dE = (E - E_vector[iN*300+pn]) / (E_vector[iN*300+pn+1]-E_vector[iN*300+pn]);
    
    double f1 = f1_vector[iN*300+pn] + dE *(f1_vector[iN*300+pn+1] - f1_vector[iN*300+pn]);
    double f2 = f2_vector[iN*300+pn] + dE *(f2_vector[iN*300+pn+1] - f2_vector[iN*300+pn]);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return (double2)(f1, f2);
  }

double8 get_structure_factor_fcc(double E, double factDW, 
                                  int4 hkl_i, double sinThetaOverLambda, 
                                  const int maxEl,  __global double* elements, 
                                  __global double* f0cfs, __global double* E_vector,
                                  __global double* f1_vector, __global double* f2_vector)
  {
        double4 hkl;
        hkl.x = (double)hkl_i.x;
        hkl.y = (double)hkl_i.y;
        hkl.z = (double)hkl_i.z;
        hkl.w = 0;
        double2 anomalousPart = get_f1f2(E, 0, round(elements[7]), E_vector, f1_vector, f2_vector);
        double2 F0 = 4 * ((double2)(elements[5],0) + anomalousPart) * factDW;
        //printf("F0_gsf: %g + %gj\n", F0.x, F0.y);
        //printf("anomalous part: %g + %gj\n", anomalousPart.x, anomalousPart.y);
        double residue = dot(fabs(remainder(hkl,2.)),1.); 
        //printf("residue: %e", residue);
        double Fcoef=0.;

        if (residue==0 || residue==3) Fcoef = 1.;    
        //printf("f0: %g\n", get_f0(sinThetaOverLambda, Z, f0cfs));
        double2 Fhkl = 4 * Fcoef * factDW * 
                         (r2cmp(get_f0(sinThetaOverLambda, 0, f0cfs)) + anomalousPart);         
        //printf("Fhkl_gsf: %g + %gj\n", Fhkl.x, Fhkl.y);
        //mem_fence(CLK_LOCAL_MEM_FENCE);
        return (double8)(F0, Fhkl, Fhkl, cmp0);
  }

double Si_dl_l(double t)
  {
        double res;
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
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
  
double2 get_distance(double8 lattice, double4 hkl)
    {
        double2 res;
        double4 basis = lattice.lo;
        double4 angles = lattice.hi;

        double4 cos_abg = cos(angles);
        double4 sin_abg = sin(angles);

        double V = basis.x * basis.y * basis.z *
            sqrt(1 - pown(cos_abg.x,2) - pown(cos_abg.y,2) - pown(cos_abg.z,2) + 2*cos_abg.x*cos_abg.y*cos_abg.z);
        //mass = 0.;
        //for (i=0;i<nmax;i++)
        //    mass += elements[i].s4 * elements[i].s6;

        //rho = mass / avogadro / V * 1e24;
        double d = V / (basis.x * basis.y * basis.z) /
            sqrt(pown(hkl.x*sin_abg.x/basis.x,2) + pown(hkl.y*sin_abg.y/basis.y,2) + pown(hkl.z*sin_abg.z/basis.z,2) +
                 2*hkl.x*hkl.y * (cos_abg.x*cos_abg.y - cos_abg.z) / (basis.x*basis.y) +
                 2*hkl.x*hkl.z * (cos_abg.x*cos_abg.z - cos_abg.y) / (basis.x*basis.z) +
                 2*hkl.y*hkl.z * (cos_abg.y*cos_abg.z - cos_abg.x) / (basis.y*basis.z));
        double chiToF = -r0 / PI / V;  // minus!
        //printf("V, d: %g %g\n", V, d);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        res = (double2)(d,chiToF);
        return res;
    }

double2 get_distance_Si(double temperature, double4 dhkl)
    {
        //double2 res;
        double aSi = 5.419490 * (Si_dl_l(temperature) - Si_dl_l(273.15) + 1);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        double d = aSi/sqrt(dot(dhkl,dhkl));
        double chiToF = -r0 / PI / pown(aSi,3);
        return (double2)(d,chiToF);
    }


double8 get_structure_factor_diamond(double E, double factDW, 
                                  int4 hkl_i, double sinThetaOverLambda, 
                                  const int maxEl, __global double* elements, 
                                  __global double* f0cfs, __global double* E_vector,
                                  __global double* f1_vector, __global double* f2_vector)
  {
        double8 res;
        double4 hkl;
        hkl.x = (double)hkl_i.x;
        hkl.y = (double)hkl_i.y;
        hkl.z = (double)hkl_i.z;
        hkl.w = 0;
        //double2 im1 = (double2)(0,1); 
        double2 diamondToFcc = cmp1 + exp_c(cmpi1 * 0.5 * PI * dot(hkl,1));
        //printf("hkl: %i, %i, %i\n", hkl_i.x,hkl_i.y,hkl_i.z);
        //printf("diamond2fcc %g + %gj\n",diamondToFcc.x, diamondToFcc.y);
        double8 Fd = get_structure_factor_fcc(E, factDW, 
                                  hkl_i, sinThetaOverLambda, 
                                  maxEl, elements,
                                  f0cfs, E_vector,
                                  f1_vector, f2_vector);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        double2 F0 = (Fd.lo).lo;
        double2 Fhkl = (Fd.lo).hi;
        double2 Fhkl_ = (Fd.hi).lo;
        res =(double8)(F0 * 2, 
                         prod_c(Fhkl,diamondToFcc), 
                         prod_c(Fhkl_,conj_c(diamondToFcc)), 
                         cmp0);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        return res;                                                   
  }

double8 get_structure_factor_general(double E, double factDW, 
                                    int4 hkl_i, double sinThetaOverLambda, 
                                    const int maxEl, __global double* elements, 
                                    __global double* f0cfs, __global double* E_vector,
                                    __global double* f1_vector, __global double* f2_vector)
    {
        int i;
        double4 hkl,xyz;
        double2 F0,Fhkl,Fhkl_,anomalousPart,fact,expiHr; 
        //double2 zero2 = (double2)(0,0);
        hkl.x = (double)hkl_i.x;
        hkl.y = (double)hkl_i.y;
        hkl.z = (double)hkl_i.z;
        hkl.w = 0;
        F0 = cmp0; Fhkl = cmp0; Fhkl_ = cmp0;
        for (i=0;i<maxEl;i++)
            {
                anomalousPart = get_f1f2(E, i, round(elements[i*8+7]), E_vector, f1_vector, f2_vector);
                mem_fence(CLK_LOCAL_MEM_FENCE);
                F0 += elements[i*8+4] * (r2cmp(elements[i*8+5]) + anomalousPart) * factDW;
                fact = elements[i*8+4] * (r2cmp(get_f0(sinThetaOverLambda, i, f0cfs)) + anomalousPart) * factDW;
                mem_fence(CLK_LOCAL_MEM_FENCE);
                xyz = (double4)(elements[i*8],elements[i*8+1],elements[i*8+2],0);
                expiHr = exp_c((double2)(0,2 * PI * dot(xyz, hkl)));
                Fhkl += prod_c(fact, expiHr);
                Fhkl_ += div_c(fact, expiHr);
                mem_fence(CLK_LOCAL_MEM_FENCE);
            }

        return (double8)(F0, Fhkl, Fhkl_, cmp0);
  }

double8 get_structure_factor_general_E(double E, double factDW, 
                                    int4 hkl_i, double sinThetaOverLambda, 
                                    const int maxEl, __global double* elements, 
                                    __global double* f0cfs, __global double2* f1f2)
    {
        int i;
        double4 hkl,xyz;
        double2 F0,Fhkl,Fhkl_,anomalousPart,fact,expiHr; 
        //double2 zero2 = (double2)(0,0);
        hkl.x = (double)hkl_i.x;
        hkl.y = (double)hkl_i.y;
        hkl.z = (double)hkl_i.z;
        hkl.w = 0;
        F0 = cmp0; Fhkl = cmp0; Fhkl_ = cmp0;
        for (i=0;i<maxEl;i++)
            {
                anomalousPart = f1f2[i];
                mem_fence(CLK_LOCAL_MEM_FENCE);
                F0 += elements[i*8+4] * (r2cmp(elements[i*8+5]) + anomalousPart) * factDW;
                fact = elements[i*8+4] * (r2cmp(get_f0(sinThetaOverLambda, i, f0cfs)) + anomalousPart) * factDW;
                mem_fence(CLK_LOCAL_MEM_FENCE);
                xyz = (double4)(elements[i*8],elements[i*8+1],elements[i*8+2],0);
                expiHr = exp_c((double2)(0,2 * PI * dot(xyz, hkl)));
                Fhkl += prod_c(fact, expiHr);
                Fhkl_ += div_c(fact, expiHr);
                mem_fence(CLK_LOCAL_MEM_FENCE);
            }

        return (double8)(F0, Fhkl, Fhkl_, cmp0);
  }



double2 for_one_polarization(double polFactor, 
                             double2 chih, double2 chih_, double2 chi0,
                             double k02, double k0s, double kHs,
                             double2 alpha, double b, double thickness, int2 geom)
  {
            double2 im1 = (double2)(0,1);
            double2 ra;
            double2 rb;
            double2 delta = sqrt_c(sqr_c(alpha) + pown(polFactor,2) / b * 
                                    prod_c(chih, chih_));

            double t = thickness * 1.e7;
            double2 L = t * delta * k02 / 2 / kHs;
            //printf("geom.lo, hi: %i, %i\n",geom.lo, geom.hi);
            if (geom.lo == 1) //Bragg
              {    
                if (geom.hi == 1) //transmitted
                  {
                    //printf("Bragg transmitted, %i, %i\n", geom.lo, geom.hi); 
                    ra = prod_c(rec_c(cos_c(L) - prod_c(prod_c(im1, alpha), div_c(sin_c(L),delta))),
                        exp_c(prod_c(im1,chi0 - alpha * b) * k02 * t / 2 / k0s));
                  }
                else // reflected
                  {
                    if (thickness == 0 || thickness > 0.51) // is None:  # thick Bragg
                      {    
                        //printf("am I here?, t=%d\n",thickness);
                        ra = div_c(chih * polFactor,alpha + delta);
                        double2 ad = alpha - delta;
                        if (abs_c(ad) == 0) ad = (double2)(1e-100,0);
                        rb = div_c(chih * polFactor, ad);
                        if (isnan(abs_c(ra)) || (abs_c(rb) < abs_c(ra))) ra = rb;       
                        //return ra / sqrt(fabs(b));
                      }
                    else
                    {                    
                    //printf("Bragg reflected, %i, %i\n",geom.lo, geom.hi);
                    ra = polFactor * div_c(chih,(alpha + div_c(prod_c(im1,delta),tan_c(L))));
                    }
                  }
              }
            else  // Laue
              {
                if (geom.hi == 1) //transmitted
                  {  
                    //printf("Laue transmitted, %i, %i\n",geom.lo, geom.hi);
                    ra = prod_c(cos_c(L) + prod_c(prod_c(im1, alpha), div_c(sin_c(L),delta)),
                          exp_c(prod_c(im1,chi0 - alpha * b) * k02 * t / 2 / k0s));
                  }
                else
                  {
                    //printf("Laue reflected, %i, %i\n",geom.lo, geom.hi);
                    //printf("delta, L, t, k0s, k02: %g+j%g, %g+j%g, %g, %g, %g\n",
                    //       delta.x, delta.y, L.x, L.y, t, k0s, k02);
                    ra = prod_c(div_c(prod_c(chih * polFactor,sin_c(L)), delta),
                        exp_c(prod_c(im1, chi0 - alpha * b) * k02 * t / 2 / k0s));
                  }
              }
            if (geom.hi == 0) ra /= sqrt(fabs(b));
            return ra;
  }            


double get_Bragg_angle(double E, double d)
  {
        return asin(ch / (2 * d * E));
  }

double get_dtheta_symmetric_Bragg(double E, double d, int4 hkl, 
                               double chiToF, double factDW, 
                               const int maxEl, __global double* elements, 
                               __global double* f0cfs, __global double* E_vector, 
                               __global double* f1_vector, __global double* f2_vector)
  {
        double8 F = get_structure_factor_general(E, factDW, hkl, 0.5 / d,
                                             maxEl, elements, f0cfs, E_vector, f1_vector, f2_vector);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        double2 chi0 = (F.lo).lo * chiToF * pown(ch / E,2);
        return (chi0 / sin(2 * get_Bragg_angle(E, d))).lo;
  }

double get_dtheta_symmetric_Bragg_E(double E, double d, int4 hkl, 
                               double chiToF, double factDW, 
                               const int maxEl, __global double* elements, 
                               __global double* f0cfs, __global double2* f1f2)
  {
        double8 F = get_structure_factor_general_E(E, factDW, hkl, 0.5 / d,
                                             maxEl, elements, f0cfs, f1f2);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        double2 chi0 = (F.lo).lo * chiToF * pown(ch / E,2);
        return (chi0 / sin(2 * get_Bragg_angle(E, d))).lo;
  }

double get_dtheta(double E, double d, int4 hkl, 
                   double chiToF, double factDW, 
                   const int maxEl, __global double* elements, 
                   __global double* f0cfs, __global double* E_vector, 
                   __global double* f1_vector, __global double* f2_vector,
                  double alpha, int2 geom)
  {
        double symm_dt = get_dtheta_symmetric_Bragg(E, d,
                                                    hkl, chiToF, factDW,
                                                    maxEl, elements,
                                                    f0cfs, E_vector,
                                                    f1_vector, f2_vector);
        double thetaB = get_Bragg_angle(E, d);
        double geom_factor = geom.lo == 1 ? -1. : 1.;
        double gamma0 = sin(thetaB + alpha);
        double gammah = geom_factor * sin(thetaB - alpha);
        double osqg0 = sqrt(1. - gamma0*gamma0);
        return -(geom_factor*gamma0 - geom_factor*sqrt(gamma0*gamma0 +
                 geom_factor*(gamma0 - gammah) * osqg0 * symm_dt)) / osqg0;
  }


double get_dtheta_E(double E, double d, int4 hkl, 
                   double chiToF, double factDW, 
                   const int maxEl, __global double* elements, 
                   __global double* f0cfs, __global double2* f1f2,
                  double alpha, int2 geom)
  {
        double symm_dt = get_dtheta_symmetric_Bragg_E(E, d,
                                                      hkl, chiToF, factDW,
                                                      maxEl, elements,
                                                      f0cfs, f1f2);
        double thetaB = get_Bragg_angle(E, d);
        double geom_factor = geom.lo == 1 ? -1. : 1.;
        double gamma0 = sin(thetaB + alpha);
        double gammah = geom_factor * sin(thetaB - alpha);
        double osqg0 = sqrt(1. - gamma0*gamma0);
        return -(geom_factor*gamma0 - geom_factor*sqrt(gamma0*gamma0 +
                 geom_factor*(gamma0 - gammah) * osqg0 * symm_dt)) / osqg0;
  }

double4 get_amplitude_material(double E, int kind, double2 refrac_n, double beamInDotNormal, 
                                double thickness, int fromVacuum)
{
    double2 n1, n2, n12s, rs, rp, p2, tf;
    double cosAlpha, sinAlpha;
    double2 n1cosAlpha, n2cosAlpha, n1cosBeta, n2cosBeta, cosBeta;
    if ((kind == 2) || (kind == 5))
        return (double4)(cmp1, cmp1);
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
    sinAlpha = sqrt(1. - beamInDotNormal * beamInDotNormal);
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
            p2 = exp_c(E * thickness * 1.e7 / chbar * 
                        prod_c(2. * cmpi1, n2cosBeta));
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
        rs = div_c((2. * tf * n1cosAlpha), (n1cosAlpha + n2cosBeta)); 
        rp = div_c((2. * tf * n1cosAlpha), (n2cosAlpha + n1cosBeta));
    } 
    return (double4)(rs, rp); //, abs(n.imag) * E / chbar * 2e8  # 1/cm
}
  
double4 get_amplitude_internal(double E, double d, int4 hkl, 
                               double chiToF, double factDW, 
                               double beamInDotNormal, 
                               double beamOutDotNormal,
                               double beamInDotHNormal,
                               double thickness, int2 geom,
                               const int maxEl, 
                               __global double* elements, 
                               __global double* f0cfs,
                               __global double* E_vector, 
                               __global double* f1_vector, 
                               __global double* f2_vector)
{
    double waveLength = ch / E;  
    double k = twoPi / waveLength;
    //printf("k %g\n", k);
    //printf("bIDN, bODN: %g, %g\n", beamInDotNormal, beamOutDotNormal);
    double k0s = -beamInDotNormal * k;
    double kHs = -beamOutDotNormal * k;
    
    double b = k0s / kHs;
    double k0H = fabs(beamInDotHNormal) * (twoPi / d) * k;
    double k02 = k * k;
    double H2 = pown(twoPi / d, 2);
    
    
    //double8 F = get_structure_factor_diamond(E, factDW, hkl, 0.5 / d,
    //                           maxEl, elements, f0cfs, E_vector, f1_vector, f2_vector);
    double8 F = get_structure_factor_general(E, factDW, hkl, 0.5 / d,
                               maxEl, elements, f0cfs, E_vector, f1_vector, f2_vector);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    double2 F0 = (F.lo).lo;
    double2 Fhkl = (F.lo).hi;
    double2 Fhkl_ = (F.hi).lo;
    double lambdaSquare = pown(waveLength,2);
    double chiToFlambdaSquare = chiToF * lambdaSquare;
    double2 chi0 = conj_c(F0) * chiToFlambdaSquare;
    double2 chih = conj_c(Fhkl) * chiToFlambdaSquare;
    double2 chih_ = conj_c(Fhkl_) * chiToFlambdaSquare;
    //printf("chih, chih_: %g+j%g, %g+j%g\n", chih.x, chih.y, chih_.x, chih_.y);
    double2 alpha = r2cmp((0.5*H2 - k0H) / k02) + chi0 * 0.5 * (1 / b - 1);
    double2 curveS = for_one_polarization(1., 
                                chih, chih_, chi0, 
                                k02, k0s, kHs,
                                alpha, b,
                                thickness, geom);  //# s polarization
    mem_fence(CLK_LOCAL_MEM_FENCE);
    double2 curveP = for_one_polarization(cos(2. * get_Bragg_angle(E, d)), 
                                chih, chih_, chi0,
                                k02, k0s, kHs,
                                alpha, b,
                                thickness, geom);  //# p polarization
    mem_fence(CLK_LOCAL_MEM_FENCE);
    //printf("AS, AP: %g+j%g, %g+j%g\n", curveS.x, curveS.y, curveP.x, curveP.y);
    return (double4)(curveS, curveP);//  # , phi.real
}

double4 get_amplitude_internal_E(double E, double d, int4 hkl, 
                               double chiToF, double factDW, 
                               double beamInDotNormal, 
                               double beamOutDotNormal,
                               double beamInDotHNormal,
                               double thickness, int2 geom,
                               const int maxEl, 
                               __global double* elements, 
                               __global double* f0cfs,
                               __global double2* f1f2)
{
    double waveLength = ch / E;  
    double k = twoPi / waveLength;
    //printf("k %g\n", k);
    //printf("bIDN, bODN: %g, %g\n", beamInDotNormal, beamOutDotNormal);
    double k0s = -beamInDotNormal * k;
    double kHs = -beamOutDotNormal * k;
    
    double b = k0s / kHs;
    double k0H = fabs(beamInDotHNormal) * (twoPi / d) * k;
    double k02 = k * k;
    double H2 = pown(twoPi / d, 2);
    
    
    //double8 F = get_structure_factor_diamond(E, factDW, hkl, 0.5 / d,
    //                           maxEl, elements, f0cfs, E_vector, f1_vector, f2_vector);
    //double8 F = get_structure_factor_general(E, factDW, hkl, 0.5 / d,
    //                           maxEl, elements, f0cfs, E_vector, f1_vector, f2_vector);
    double8 F = get_structure_factor_general_E(E, factDW, hkl, 0.5 / d,
                               maxEl, elements, f0cfs, f1f2);

    mem_fence(CLK_LOCAL_MEM_FENCE);
    double2 F0 = (F.lo).lo;
    double2 Fhkl = (F.lo).hi;
    double2 Fhkl_ = (F.hi).lo;
    double lambdaSquare = pown(waveLength,2);
    double chiToFlambdaSquare = chiToF * lambdaSquare;
    double2 chi0 = conj_c(F0) * chiToFlambdaSquare;
    double2 chih = conj_c(Fhkl) * chiToFlambdaSquare;
    double2 chih_ = conj_c(Fhkl_) * chiToFlambdaSquare;
    //printf("chih, chih_: %g+j%g, %g+j%g\n", chih.x, chih.y, chih_.x, chih_.y);
    double2 alpha = r2cmp((0.5*H2 - k0H) / k02) + chi0 * 0.5 * (1 / b - 1);
    double2 curveS = for_one_polarization(1., 
                                chih, chih_, chi0, 
                                k02, k0s, kHs,
                                alpha, b,
                                thickness, geom);  //# s polarization
    mem_fence(CLK_LOCAL_MEM_FENCE);
    double2 curveP = for_one_polarization(cos(2. * get_Bragg_angle(E, d)), 
                                chih, chih_, chi0,
                                k02, k0s, kHs,
                                alpha, b,
                                thickness, geom);  //# p polarization
    mem_fence(CLK_LOCAL_MEM_FENCE);
    //printf("AS, AP: %g+j%g, %g+j%g\n", curveS.x, curveS.y, curveP.x, curveP.y);
    return (double4)(curveS, curveP);//  # , phi.real
}


double4 get_amplitude_multilayer_internal(const int npairs,
                                            double2 rbs_si,
                                            double2 rbs_pi,
                                            double2 rtb_si,
                                            double2 rtb_pi,
                                            double2 rvt_si,
                                            double2 rvt_pi,
                                            double2 p2ti,
                                            double2 p2bi)
{        
    int i;
    double2 rij_s, rij_p, p2i, rj_s, rj_p, rj2i, ri_si, ri_pi;
    //double2 rbt_s = -rtb_si;
    //double2 rbt_p = -rtb_pi;
    rj_s = rbs_si;
    rj_p = rbs_pi;
    int lsw = -1;

    for (i=2*npairs-1;i>0;i--)
    {
        lsw = -lsw;   
        rij_s = lsw * rtb_si; 
        rij_p = lsw * rtb_pi;
        p2i = (lsw + 1) * p2bi / 2. - (lsw - 1) * p2ti / 2.;

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
    //mem_fence(CLK_LOCAL_MEM_FENCE);
    return (double4)(ri_si, ri_pi);
}

double4 get_amplitude_graded_multilayer_internal(const int npairs,
                                            double2 rbs_si,
                                            double2 rbs_pi,
                                            double2 rtb_si,
                                            double2 rtb_pi,
                                            double2 rvt_si,
                                            double2 rvt_pi,
                                            double2 qti,
                                            double2 qbi,
										__global double* dti,
										__global double* dbi)
{        
    int i;
    double2 rij_s, rij_p, p2i, rj_s, rj_p, rj2i, ri_si, ri_pi;
	double2 p2bi, p2ti;
    //double2 rbt_s = -rtb_si;
    //double2 rbt_p = -rtb_pi;
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
        //p2i = (lsw + 1) * p2bi / 2. - (lsw - 1) * p2ti / 2.;

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
    //printf("dti[0] %g, dbi[0] %g\n",di[0].x, di[0].y);
    p2i = exp_c(prod_c(cmpi1, dti[0] * qti));
    rj2i = prod_c(rj_s,p2i);
    ri_si = div_c((rij_s + rj2i), 
                    (cmp1 + prod_c(rij_s, rj2i)));
    rj2i = prod_c(rj_p,p2i);
    ri_pi = div_c((rij_p + rj2i),
                    (cmp1 + prod_c(rij_p, rj2i)));
    mem_fence(CLK_LOCAL_MEM_FENCE);
    return (double4)(ri_si, ri_pi);
}


__kernel void get_amplitude(const int4 hkl,
                            const double factDW,
                            const double thickness,
                            const int geom_b,
                            const int maxEl,
                            __global double* BeamInDotNormal_gl,
                            __global double* BeamOutDotNormal_gl,
                            __global double* BeamInDotHNormal_gl,
                            __global double* Energy_gl,
                            __global double* IPDistance_gl,
                            __global double* ChiToF_gl,
                            __global double* elements,
                            __global double* f0cfs, 
                            __global double* E_vector, 
                            __global double* f1_vector, 
                            __global double* f2_vector,
                            __global double2* refl_s,
                            __global double2* refl_p)
{
    unsigned int ii = get_global_id(0);
    double4 amplitudes;
    int2 geom;
    geom.lo = geom_b > 1 ? 1 : 0;
    geom.hi = (int)(fabs(remainder(geom_b,2.)));
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
                            __global double2* rbs_s,
                            __global double2* rbs_p,
                            __global double2* rtb_s,
                            __global double2* rtb_p,
                            __global double2* rvt_s,
                            __global double2* rvt_p,
                            __global double2* p2t,
                            __global double2* p2b,
                            __global double2* ri_s,
                            __global double2* ri_p)
{
    unsigned int ii = get_global_id(0);
    double4 amplitudes;
    amplitudes = get_amplitude_multilayer_internal(npairs, 
                    rbs_s[ii], rbs_p[ii], rtb_s[ii], rtb_p[ii], 
                    rvt_s[ii], rvt_p[ii], p2t[ii], p2b[ii]);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    ri_s[ii] = amplitudes.s01;
    ri_p[ii] = amplitudes.s23;
    mem_fence(CLK_LOCAL_MEM_FENCE);
}

__kernel void get_amplitude_graded_multilayer(const int npairs,
                            __global double2* rbs_s,
                            __global double2* rbs_p,
                            __global double2* rtb_s,
                            __global double2* rtb_p,
                            __global double2* rvt_s,
                            __global double2* rvt_p,
                            __global double2* qt_glo,
                            __global double2* qb_glo,
                            __global double* dti,							
                            __global double* dbi,
                            __global double2* ri_s,
                            __global double2* ri_p)
{
    unsigned int ii = get_global_id(0);
    double4 amplitudes;
    amplitudes = get_amplitude_graded_multilayer_internal(npairs, 
                    rbs_s[ii], rbs_p[ii], rtb_s[ii], rtb_p[ii], 
                    rvt_s[ii], rvt_p[ii], qt_glo[ii], qb_glo[ii], dti, dbi);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    ri_s[ii] = amplitudes.s01;
    ri_p[ii] = amplitudes.s23;
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
 