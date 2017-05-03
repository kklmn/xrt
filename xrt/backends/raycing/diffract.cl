//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "10 Apr 2015"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

//#include "materials.cl"
//__constant float Pi = 3.141592653589793;
//__constant float twoPi = 6.283185307179586476;
__constant float fourPi = (float)12.566370614359172;
__constant float2 cmp0 = (float2)(0, 0);
__constant float2 cmpi1 = (float2)(0, 1);

float2 prod_c(float2 a, float2 b)
  {
    return (float2)(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
  }
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
	unsigned int imageLength = get_global_size(0);
    float3 beam_coord_glo, beam_angle_glo;
    float2 gi, giEs, giEp, cfactor;
    float2 KirchS_loc, KirchP_loc;
    float2 KirchA_loc, KirchB_loc, KirchC_loc;
    float pathAfter, cosAlpha, cr, sinphase, cosphase;
    unsigned int ii_screen = get_global_id(0);
//    float wavelength;
    float phase, k2;

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
        cosAlpha = dot(beam_angle_glo, oe_surface_normal[i]) / pathAfter;
//        wavelength = twoPi / k;
//        phase = k * (pathAfter + beam_OE_loc_path[i]);
        phase = k[i] * pathAfter;
        cr = k[i] * (cosAlpha + cosGamma[i]) / pathAfter;
        sinphase = sincos(phase, &cosphase);
        gi = (float2)(cr * cosphase, cr * sinphase);
        giEs = prod_c(gi, Es[i]);
        giEp = prod_c(gi, Ep[i]);
        KirchS_loc += giEs;
        KirchP_loc += giEp;
        gi = k[i] * k[i] * (giEs+giEp) / pathAfter;
        KirchA_loc += prod_c(gi, beam_angle_glo.x);
        KirchB_loc += prod_c(gi, beam_angle_glo.y);
        KirchC_loc += prod_c(gi, beam_angle_glo.z);
      }
  mem_fence(CLK_LOCAL_MEM_FENCE);

  cfactor = -cmpi1 / fourPi;
  KirchS_gl[ii_screen] = prod_c(cfactor, KirchS_loc);
  KirchP_gl[ii_screen] = prod_c(cfactor, KirchP_loc);
  KirchA_gl[ii_screen] = KirchA_loc / fourPi;
  KirchB_gl[ii_screen] = KirchB_loc / fourPi;
  KirchC_gl[ii_screen] = KirchC_loc / fourPi;

  mem_fence(CLK_LOCAL_MEM_FENCE);
}

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
//  KirchS_gl[ii_screen] = -prod_c(cmpi1*k/fourPi, KirchS_loc);
//  KirchP_gl[ii_screen] = -prod_c(cmpi1*k/fourPi, KirchP_loc);
//  mem_fence(CLK_LOCAL_MEM_FENCE);
//}
