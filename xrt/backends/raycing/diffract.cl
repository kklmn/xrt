//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "10 Apr 2015"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

//#include "materials.cl"
//__constant float Pi = 3.141592653589793;
__constant const double twoPi = (double)6.283185307179586476;
__constant const double invTwoPi = (double)(1./6.283185307179586476);
__constant const float invFourPi = (float)0.07957747154594767;
__constant float2 cmp0 = (float2)(0, 0);
__constant float2 cmpi1 = (float2)(0, 1);
__constant double2 cmp0_d = (double2)(0, 0);
__constant double2 cmpi1_d = (double2)(0, 1);
__constant const double2 cfactor = (double2)(0, -0.07957747154594767);

float2 prod_c(float2 a, float2 b)
{
    return (float2)(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
}
double2 prod_c_d(double2 a, double2 b)
{
    return (double2)(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
}
double to2pi(double val)
{
	return val - trunc(val*invTwoPi) * twoPi;
}
//double dot3(double3 a, double3 b)
//{
//    return (double)(a.x*b.x + a.y*b.y + a.z*b.z);
//}


//float2 conj_c(float2 a)
//  {
//    return (float2)(a.x, -a.y);
//  }
//float abs_c(float2 a)
//  {
//    return sqrt(a.x*a.x + a.y*a.y);
//  }
//float abs_c2(float2 a)
//  {
//    return (a.x*a.x + a.y*a.y);
//  }

//float arg_c(float2 a)
//  {
//    if(a.x > 0)
//      {
//        return atan(a.y / a.x);
//      }
//    else if(a.x < 0 && a.y >= 0)
//      {
//        return atan(a.y / a.x) + Pi;
//      }
//    else if(a.x < 0 && a.y < 0)
//      {
//        return atan(a.y / a.x) - Pi;
//      }
//    else if(a.x == 0 && a.y > 0)
//      {
//        return Pi/2;
//      }
//    else if(a.x == 0 && a.y < 0)
//      {
//        return -Pi/2;
//      }
//    else
//      {
//        return 0;
//      }
//  }

__kernel void integrate_kirchhoff(
                    //const unsigned int imageLength,
                    const unsigned int fullnrays,
                    __global float* x_glo,
                    __global float* y_glo,
                    __global float* z_glo,
                    __global float* cosGamma,
                    __global float2* Es,
                    __global float2* Ep,
                    __global float* k,
                    __global float3* beamOEglo,
                    __global float3* oe_surface_normal,
//                    __global float* beam_OE_loc_path,
                    __global float2* KirchS_gl,
                    __global float2* KirchP_gl,
                    __global float2* KirchA_gl,
                    __global float2* KirchB_gl,
                    __global float2* KirchC_gl)
{
    unsigned int i;
//    unsigned int imageLength = get_global_size(0);
    float3 beam_coord_glo, beam_angle_glo;
    float2 gi, giEs, giEp, cfactor;
    float2 KirchS_loc, KirchP_loc;
    float2 KirchA_loc, KirchB_loc, KirchC_loc;
    float pathAfter, cosAlpha, cr, sinphase, cosphase;
    unsigned int ii_screen = get_global_id(0);
    float phase;
    float invPathAfter, kip;

    KirchS_loc = cmp0;
    KirchP_loc = cmp0;
    KirchA_loc = cmp0;
    KirchB_loc = cmp0;
    KirchC_loc = cmp0;

    beam_coord_glo.x = x_glo[ii_screen];
    beam_coord_glo.y = y_glo[ii_screen];
    beam_coord_glo.z = z_glo[ii_screen];
    for (i=0; i<fullnrays; i++)
      {
        beam_angle_glo = beam_coord_glo - beamOEglo[i];
        pathAfter = length(beam_angle_glo);
        invPathAfter = 1. / pathAfter;
//        invPathAfter = rsqrt(dot(beam_angle_glo, beam_angle_glo));
        cosAlpha = dot(beam_angle_glo, oe_surface_normal[i]) * invPathAfter;
        phase = k[i] * pathAfter;
//        phase = k[i] / invPathAfter;
        kip = k[i] * invPathAfter;
        cr = kip * (cosAlpha + cosGamma[i]);
        sinphase = sincos(phase, &cosphase);
        gi = (float2)(cr * cosphase, cr * sinphase);
        giEs = prod_c(gi, Es[i]);
        giEp = prod_c(gi, Ep[i]);
        KirchS_loc += giEs;
        KirchP_loc += giEp;
        gi = k[i] * kip * (giEs+giEp);
        KirchA_loc += prod_c(gi, beam_angle_glo.x);
        KirchB_loc += prod_c(gi, beam_angle_glo.y);
        KirchC_loc += prod_c(gi, beam_angle_glo.z);
      }
  mem_fence(CLK_LOCAL_MEM_FENCE);

  cfactor = -cmpi1 * invFourPi;
  KirchS_gl[ii_screen] = prod_c(cfactor, KirchS_loc);
  KirchP_gl[ii_screen] = prod_c(cfactor, KirchP_loc);
  KirchA_gl[ii_screen] = KirchA_loc * invFourPi;
  KirchB_gl[ii_screen] = KirchB_loc * invFourPi;
  KirchC_gl[ii_screen] = KirchC_loc * invFourPi;

  mem_fence(CLK_LOCAL_MEM_FENCE);
}

//__kernel void integrate_kirchhoff_d(
//                    //const unsigned int imageLength,
//                    const unsigned int fullnrays,
//                    __global double* x_glo,
//                    __global double* y_glo,
//                    __global double* z_glo,
//                    __global double* cosGamma,
//                    __global double2* Es,
//                    __global double2* Ep,
//                    __global double* k,
//                    __global double3* beamOEglo,
//                    __global double3* oe_surface_normal,
////                    __global double* beam_OE_loc_path,
//                    __global double2* KirchS_gl,
//                    __global double2* KirchP_gl,
//                    __global double2* KirchA_gl,
//                    __global double2* KirchB_gl,
//                    __global double2* KirchC_gl)
//{
//    unsigned int i;
////    unsigned int imageLength = get_global_size(0);
//    double3 beam_coord_glo, beam_angle_glo;
//    double2 gi, giEs, giEp;
//    double2 KirchS_loc, KirchP_loc;
//    double2 KirchA_loc, KirchB_loc, KirchC_loc;
////    double pathAfter, cosAlpha, cr;
//    double cosAlpha, cr;
//    float sinphase, cosphase;
//    double phase;
//    double invPathAfter, kp;
//    unsigned int ii_screen = get_global_id(0);
//
//    KirchS_loc = cmp0_d;
//    KirchP_loc = cmp0_d;
//    KirchA_loc = cmp0_d;
//    KirchB_loc = cmp0_d;
//    KirchC_loc = cmp0_d;
//
//    beam_coord_glo.x = x_glo[ii_screen];
//    beam_coord_glo.y = y_glo[ii_screen];
//    beam_coord_glo.z = z_glo[ii_screen];
//    for (i=0; i<fullnrays; i++)
//    {
//        beam_angle_glo = beam_coord_glo - beamOEglo[i];
////        pathAfter = length(beam_angle_glo);
////        invPathAfter = 1. / pathAfter;
//        invPathAfter = rsqrt(dot(beam_angle_glo, beam_angle_glo));
////        phase = to2pi(k[i] * pathAfter);
//        phase = to2pi(k[i] / invPathAfter);
//        kp = k[i] * invPathAfter;
//        cosAlpha = dot(beam_angle_glo, oe_surface_normal[i]) * invPathAfter;
//        cr = kp * (cosAlpha + cosGamma[i]);
//        sinphase = sincos((float)phase, &cosphase);
//        gi = (double2)((double)cosphase, (double)sinphase);
//        giEs = prod_c_d(gi, Es[i]);
//        giEp = prod_c_d(gi, Ep[i]);
//        KirchS_loc += giEs * cr;
//        KirchP_loc += giEp * cr;
//        gi = k[i] * kp * (giEs + giEp);
//        KirchA_loc += gi * beam_angle_glo.x;
//        KirchB_loc += gi * beam_angle_glo.y;
//        KirchC_loc += gi * beam_angle_glo.z;
//    }
//    mem_fence(CLK_LOCAL_MEM_FENCE);
//
//    KirchS_gl[ii_screen] = prod_c_d(KirchS_loc, cfactor);
//    KirchP_gl[ii_screen] = prod_c_d(KirchP_loc, cfactor);
//    KirchA_gl[ii_screen] = KirchA_loc * invFourPi;
//    KirchB_gl[ii_screen] = KirchB_loc * invFourPi;
//    KirchC_gl[ii_screen] = KirchC_loc * invFourPi;
//
//    mem_fence(CLK_LOCAL_MEM_FENCE);
//}

//__kernel void integrate_fraunhofer(
//                    const unsigned int imageLength,
//                    const unsigned int fullnrays,
//                    const float chbar,
//                    __global float* cosGamma,
//                    __global float* a_glo,
//                    __global float* b_glo,
//                    __global float* c_glo,
//                    __global float2* Es,
//                    __global float2* Ep,
//                    __global float* E_loc,
//                    __global float3* beamOEglo,
//                    __global float3* oe_surface_normal,
//                    __global float* beam_OE_loc_path,
//                    __global float2* KirchS_gl,
//                    __global float2* KirchP_gl)
//
//{
//    unsigned int i;
//
////    float3 beam_coord_glo;
//    float3 beam_angle_glo;
//    float2 gi, KirchS_loc, KirchP_loc;
//    float pathAfter, cosAlpha, cr;
//    unsigned int ii_screen = get_global_id(0);
////    float wavelength;
//    float k, phase;
//
//    KirchS_loc = (float2)(0, 0);
//    KirchP_loc = KirchS_loc;
//
//    beam_angle_glo.x = a_glo[ii_screen];
//    beam_angle_glo.y = b_glo[ii_screen];
//    beam_angle_glo.z = c_glo[ii_screen];
//    //if (ii_screen==128 || ii_screen==129) printf("Pix %i, beam_coord_glo %0.32v3f\n",ii_screen, beam_coord_glo);
//    for (i=0; i<fullnrays; i++)
//      {
//        //printf("point %i\n",i);
//        pathAfter = -dot(beam_angle_glo, beamOEglo[i]);
//        cosAlpha = dot(beam_angle_glo, oe_surface_normal[i]);
//        k = E_loc[i] / chbar * 1.e7;
////        wavelength = twoPi / k;
//        phase = k * (pathAfter + beam_OE_loc_path[i]);
//        cr = (cosAlpha + cosGamma[i]) / pathAfter;
//        gi = (float2)(cr * cos(phase), cr * sin(phase));
//        KirchS_loc += prod_c(gi, Es[i]);
//        KirchP_loc += prod_c(gi, Ep[i]);
//      }
//  mem_fence(CLK_LOCAL_MEM_FENCE);
//
//  KirchS_gl[ii_screen] = -prod_c(cmpi1*k*invFourPi, KirchS_loc);
//  KirchP_gl[ii_screen] = -prod_c(cmpi1*k*invFourPi, KirchP_loc);
//  mem_fence(CLK_LOCAL_MEM_FENCE);
//}
