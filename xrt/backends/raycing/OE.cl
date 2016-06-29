//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "10 Apr 2015"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

//#define N_LAYERS 501
//#define N_INPUT 5001
//#define RANLUXCL_SUPPORT_DOUBLE
//#include "materials.cl"
//#include "pyopencl-ranluxcl.cl"

__constant float zEps = 1.e-12;
__constant bool isParametric = false;
__constant int maxIteration = 100;
//__constant long cl_gl=0;
//__global float *cl_gl;
//__constant float PI = 3.141592653589793238;
//__constant float ch = 12398.4186;  // {5}   {c*h[eV*A]}

float2 rotate_x(float y, float z, float cosangle, float sinangle)
{
    return (float2)(cosangle*y - sinangle*z, sinangle*y + cosangle*z);
}

float2 rotate_y(float x, float z, float cosangle, float sinangle)
{
    return (float2)(cosangle*x + sinangle*z, -sinangle*x + cosangle*z);
}

float2 rotate_z(float x, float y, float cosangle, float sinangle)
{
    return (float2)(cosangle*x - sinangle*y, sinangle*x + cosangle*y);
}

MY_LOCAL_Z

MY_LOCAL_N

MY_LOCAL_G

MY_XYZPARAM

float4 find_dz(float8 cl_plist,
               int local_f, float t, float x0, float y0, float z0,
               float a, float b, float c, int invertNormal, int derivOrder)
{
        /*Returns the z or r difference (in the local system) between the ray
        and the surface. Used for finding the intersection point.*/

        float x = x0 + a * t;
        float y = y0 + b * t;
        float z = z0 + c * t;
        float surf, dz;
        float3 surf3;
        int diffSign = 1;

        if (derivOrder == 0)
          {
            if (isParametric)
              {
                diffSign = -1;
                surf3 = xyz_to_param(cl_plist, x, y, z);
                x = surf3.x; y = surf3.y; z = surf3.z;
                surf = local_z(cl_plist, local_f, x, y);
              }
            else
              {
                surf = local_z(cl_plist, local_f, x, y);
              }
            if (!isnormal(surf)) surf = 0;
            dz = (z - surf) * diffSign * invertNormal;
          }
        else
          {
            surf3 = local_n(cl_plist, local_f, x, y).s012;
            //surf3 = (float3)(0,0,0);
            if (!isnormal(surf))
              {
                surf3.s0 = 0;
                surf3.s1 = 0;
                surf3.s2 = 1;
              }
            dz = (a * surf3.s0 + b * surf3.s1 + c * surf3.s2) * invertNormal;
          }
        //mem_fence(CLK_GLOBAL_MEM_FENCE);
        return (float4)(dz, x, y, z);
}

float4 _use_my_method(float8 cl_plist,
                      int local_f, float tMin, float tMax, float t1, float t2,
                      float x, float y, float z,
                      float a, float b, float c,
                      int invertNormal, int derivOrder, float dz1, float dz2,
                      float x2, float y2, float z2)
{

        float4 tmp;
        unsigned int numit=2;
        float t, dz;
        while ((fabs(dz2) > zEps) & (numit < maxIteration))
          {
            t = t1;
            dz = dz1;
            t1 = t2;
            dz1 = dz2;
            t2 = t - (t1 - t) * dz / (dz1 - dz);

            if (t2 < tMin) t2 = tMin;
            if (t2 > tMax) t2 = tMax;

            tmp = find_dz(cl_plist, local_f, t2, x, y, z, a, b, c,
                          invertNormal, derivOrder);

            dz2 = tmp.s0;
            x2 = tmp.s1;
            y2 = tmp.s2;
            z2 = tmp.s3;

            if (sign(dz2) == sign(dz1))
              {
                t1 = t;
                dz1 = dz;
              }
            numit++;
          }
        //mem_fence(CLK_GLOBAL_MEM_FENCE);
        return (float4)(t2, x2, y2, z2);
}
float4 _use_Brent_method(float8 cl_plist,
                      int local_f, float tMin, float tMax, float t1, float t2,
                      float x, float y, float z,
                      float a, float b, float c,
                      int invertNormal, int derivOrder, float dz1, float dz2,
                      float x2, float y2, float z2)
{
        float4 tmp;
        float tmpt, t3, t4, dz3, xa, xb, xc, xd, xs, xai, xbi, xci;
        float fa, fb, fc, fai, fbi, fci, fs;
        bool mflag, mf, cond1, cond2, cond3, cond4, cond5, conds, fafsNeg;


        if (fabs(dz1) < fabs(dz2))
          {
            tmpt = t1; t1 = t2; t2 = tmpt;
            tmpt = dz1; dz1 = dz2; dz2 = tmpt;
          }

        t3 = t1;
        dz3 = dz1;
        t4 = 0;

        mflag = true;
        unsigned int numit = 2;

        while ((fabs(dz2) > zEps) & (numit < maxIteration))
          {

            xa = t1; xb = t2; xc = t3; xd = t4;
            fa = dz1; fb = dz2; fc = dz3;
            mf = mflag;
            xs = 0;

            if ((fa != fc) & (fb != fc))
              {
                xai = xa; xbi = xb; xci = xc;
                fai = fa; fbi = fb; fci = fc;
                xs = xai * fbi * fci / (fai - fbi) / (fai - fci) +
                    fai * xbi * fci / (fbi - fai) / (fbi - fci) +
                    fai * fbi * xci / (fci - fai) / (fci - fbi);
              }
            else
              {
                xai = xa; xbi = xb;
                fai = fa; fbi = fb;
                xs = xbi - fbi * (xbi - xai) / (fbi - fai);
              }

            cond1 = (((xs < (3*xa + xb) / 4.) & (xs < xb)) |
                     ((xs > (3*xa + xb) / 4.) & (xs > xb)));
            cond2 = (mf & (fabs(xs - xb) >= (fabs(xb - xc) / 2.)));
            cond3 = ((!mf) & (fabs(xs - xb) >= (fabs(xc - xd) / 2.)));
            cond4 = (mf & (fabs(xb - xc) < zEps));
            cond5 = ((!mf) & (fabs(xc - xd) < zEps));
            conds = (cond1 | cond2 | cond3 | cond4 | cond5);

            if (conds)
              {
                xs = (xa + xb) / 2;
              }

            mf = conds;
            tmp = find_dz(cl_plist, local_f, xs, x, y, z, a, b, c,
                          invertNormal, derivOrder);

            fs = tmp.s0;
            x2 = tmp.s1;
            y2 = tmp.s2;
            z2 = tmp.s3;

            xd = xc; xc = xb; fc = fb;

            fafsNeg = (((fa < 0) & (fs > 0)) | ((fa > 0) & (fs < 0)));

            if (fafsNeg)
              {
                xb = xs; fb = fs;
              }
            else
              {
                xa = xs; fa = fs;
              }

            if (fabs(fa) < fabs(fb))
              {
                tmpt = xa; xa = xb; xb = tmpt;
                tmpt = fa; fa = fb; fb = tmpt;
              }

            t1 = xa; t2 = xb; t3 = xc; t4 = xd;
            dz1 = fa; dz2 = fb; dz3 = fc;
            mflag = mf;

            numit++;
          }

        return (float4)(t2, x2, y2, z2);
}


float4 find_intersection_internal(float8 cl_plist,
                          int local_f, float tMin, float tMax,
                          float t1, float t2,
                          float x, float y, float z,
                          float a, float b, float c,
                          int invertNormal, int derivOrder)
{
        /*Finds the ray parameter *t* at the intersection point with the
        surface. Requires *t1* and *t2* as input bracketing. The ray is
        determined by its origin point (*x*, *y*, *z*) and its normalized
        direction (*a*, *b*, *c*). *t* is then the distance between the origin
        point and the intersection point. *derivOrder* tells if minimized is
        the z-difference (=0) or its derivative (=1).*/
        float4 tmp1, res;
        float dz1, dz2;
        int state = 1;

        res = find_dz(cl_plist, local_f,
            t1, x, y, z, a, b, c, invertNormal, derivOrder);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        tmp1 = find_dz(cl_plist, local_f,
            t2, x, y, z, a, b, c, invertNormal, derivOrder);
        mem_fence(CLK_LOCAL_MEM_FENCE);
        dz1 = res.s0;
        dz2 = tmp1.s0;

        if (dz1 <= 0) state = -1;
        if (dz2 >= 0) state = -2;

        tmp1.s0 = t2;

        if (state == -1) tmp1.s0 = t1;

        res = tmp1;

        if (state > 0)
          {
            if (fabs(dz2) > fabs(dz1) * 20.)
              {
                res = _use_Brent_method(cl_plist, local_f,
                  tMin, tMax, t1, t2, x, y, z, a, b, c,
                  invertNormal, derivOrder, dz1, dz2, res.s1, res.s2, res.s3);
              }
            else
              {
                res = _use_my_method(cl_plist, local_f,
                  tMin, tMax, t1, t2, x, y, z, a, b, c,
                  invertNormal, derivOrder, dz1, dz2, res.s1, res.s2, res.s3);
              }
          }
        mem_fence(CLK_LOCAL_MEM_FENCE);
        return res;
}


__kernel void find_intersection(
                  const float8 cl_plist_gl,
                  const int invertNormal,
                  const int derivOrder,
                  const int local_zN,
                  const float tMin,
                  const float tMax,
                  __global float *t1,
                  __global float *x,
                  __global float *y,
                  __global float *z,
                  __global float *a,
                  __global float *b,
                  __global float *c,
                  __global float *t2,
                  __global float *x2,
                  __global float *y2,
                  __global float *z2
                  )
{
      float4 res;
      float8 cl_plist = cl_plist_gl;
      //unsigned int i = get_local_id(0);
      //unsigned int group_id = get_group_id(0);
      //unsigned int nloc = get_local_size(0);
      //unsigned int ii = nloc * group_id + i;
      unsigned int ii = get_global_id(0);
      res = find_intersection_internal(cl_plist, local_zN,
                                       tMin, tMax, t1[ii], t2[ii],
                                       x[ii], y[ii], z[ii],
                                       a[ii], b[ii], c[ii],
                                       invertNormal, derivOrder);
      //mem_fence(CLK_GLOBAL_MEM_FENCE);
      t2[ii] = res.s0; x2[ii] = res.s1; y2[ii] = res.s2; z2[ii] = res.s3;
}


float4 _grating_deflection(float4 abc, float4 oeNormal, float4 gNormal,
                            float E, float beamInDotNormal, float order,
                            bool isTransmitting, unsigned int ray)
  {
        float4 abc_out;
        float beamInDotG, G2, locOrder, orderLambda, sig, dn;
        beamInDotG = dot(abc,gNormal);
        G2 = pown(length(gNormal),2);
        locOrder = order;
        orderLambda = locOrder * ch / E * 1.e-7;
        sig = (isTransmitting) ? 1. : -1.;
        dn = beamInDotNormal + sig * sqrt(fabs(beamInDotNormal * beamInDotNormal -
                                        2. * beamInDotG * orderLambda -
                                        G2 * orderLambda * orderLambda));
        abc_out = abc - oeNormal*dn + gNormal*orderLambda;
        //mem_fence(CLK_LOCAL_MEM_FENCE);
        return abc_out;
  }

float8 reflect_crystal_internal(const float factDW,
                              const float thickness,
                              int2 geom,
                              const float8 lattice,
                              const float temperature,
                              float4 abc,
                              float E,
                              float4 planeNormal,
                              float4 surfNormal,
                              int4  hkl,
                              const int maxEl,
                              __global float* elements,
                              __global float* f0cfs,
                              __global float* E_vector,
                              __global float* f1_vector,
                              __global float* f2_vector, unsigned int ray
                              )
{
      //printf("got to rsi, ray %i\n", ray);
      float4 abc_out, dhkl, amplitudes, gNormalCryst, gNormal;
      float beamInDotNormal, d, chiToF, normalDotSurfNormal;
      float beamInDotSurfaceNormal, beamOutDotSurfaceNormal, dt;
      dhkl = (float4)(hkl.x, hkl.y, hkl.z, 0);
      float2 plane_d;
      //float epsilonB = 0.01;
      beamInDotNormal = dot(abc,planeNormal);

      if (temperature>0)
        {
            plane_d = get_distance_Si(temperature, dhkl);

        }
        else
        {
            //plane_d = get_distance_Si(temperature, dhkl);
            plane_d = get_distance(lattice, dhkl);
        }
      //plane_d = get_distance(lattice, dhkl);

      mem_fence(CLK_LOCAL_MEM_FENCE);
      d = plane_d.lo;
      chiToF = plane_d.hi;
      //printf("d, ctf: %g %g\n", d, chiToF);
      //dt = get_dtheta_symmetric_Bragg(E, d,
      //                 hkl, chiToF, factDW,
      //                 maxEl, elements,
      //                 f0cfs, E_vector,
      //                 f1_vector, f2_vector);
      dt = get_dtheta(E, d,
                      hkl, chiToF, factDW,
                      maxEl, elements,
                      f0cfs, E_vector,
                      f1_vector, f2_vector,
                      0, geom);
      mem_fence(CLK_LOCAL_MEM_FENCE);
      if (isnan(dt)) dt = 0;
      //printf("dt %g\n", dt);
      normalDotSurfNormal = dot(planeNormal,surfNormal);
      gNormalCryst = (planeNormal - normalDotSurfNormal*surfNormal) *
                          dt / d / 1e-7 *
                          sqrt(fabs(1. - pown(normalDotSurfNormal,2)));
       /*          if matSur.geom.endswith('Fresnel'):
                  if isinstance(self.order, int):
                      locOrder = self.order
                  else:
                      locOrder = np.array(self.order)[np.random.randint(
                          len(self.order), size=goodN.sum())]
                  if gNormal is None:
                      gNormal = local_g(lb.x[goodN], lb.y[goodN])
                  gNormal = np.asarray(gNormal, order='F') * locOrder
                  gNormal[0] += gNormalCryst[0]
                  gNormal[1] += gNormalCryst[1]
                  gNormal[2] += gNormalCryst[2]
              else:
        */        gNormal = gNormalCryst;
      abc_out = _grating_deflection(abc, planeNormal, gNormal,
                      E, beamInDotNormal, 1.,
                      false, ray);
      mem_fence(CLK_LOCAL_MEM_FENCE);
      beamInDotSurfaceNormal = dot(abc,surfNormal);
      beamOutDotSurfaceNormal = dot(abc_out,surfNormal);
      //if the beam reflects through the crystal use Laue model
      //geom.lo = 1,0:Bragg,Laue
      //geom.hi = 1,0:transmitted,reflected
      geom.lo = 1; //bragg
      //geom.hi = 0; //reflect
      if (beamInDotSurfaceNormal*beamOutDotSurfaceNormal > 0) geom.lo = 0;
      amplitudes = get_amplitude_internal(E, d, hkl,
                         chiToF, factDW,
                         beamInDotSurfaceNormal, beamOutDotSurfaceNormal,
                         beamInDotNormal,
                         thickness, geom, maxEl, elements,
                         f0cfs, E_vector,
                         f1_vector, f2_vector);
      mem_fence(CLK_LOCAL_MEM_FENCE);
      //printf("amplitudes %g %g %g %g\n", amplitudes.x,amplitudes.y,amplitudes.z,amplitudes.w);
      if (any(isnan(amplitudes.lo)))
                {
                  amplitudes.lo = cmp0;
                }
      if (any(isnan(amplitudes.hi)))
                {
                  amplitudes.hi = cmp0;
                }
      //sampling refracted or reflected beam
      /*if (abs_c(amplitudes.lo) + abs_c(amplitudes.hi) > 2)
        {
          geom.hi = 1;
          abc_out = abc;
          amplitudes = get_amplitude_internal(E, d, hkl,
                         chiToF, factDW,
                         beamInDotSurfaceNormal, beamOutDotSurfaceNormal,
                         beamInDotNormal,
                         thickness, geom,maxEl, elements,
                         f0cfs, E_vector,
                         f1_vector, f2_vector);
        }
      mem_fence(CLK_LOCAL_MEM_FENCE);
      if (any(isnan(amplitudes.lo)))
                {
                  amplitudes.lo = cmp0;
                }
      if (any(isnan(amplitudes.hi)))
                {
                  amplitudes.hi = cmp0;
                }*/
      abc_out.w = geom.lo;
      return (float8)(amplitudes,abc_out);
}

float8 reflect_crystal_internal_E(const float factDW,
                              const float thickness,
                              int2 geom,
                              const float8 lattice,
                              const float temperature,
                              float4 abc,
                              float E,
                              float4 planeNormal,
                              float4 surfNormal,
                              int4  hkl,
                              const int maxEl,
                              __global float* elements,
                              __global float* f0cfs,
                              __global float2* f1f2, unsigned int ray
                              )
{
      //printf("got to rsi, ray %i\n", ray);
      float4 abc_out, dhkl, amplitudes, gNormalCryst, gNormal;
      float beamInDotNormal, d, chiToF, normalDotSurfNormal;
      float beamInDotSurfaceNormal, beamOutDotSurfaceNormal, dt;
      dhkl = (float4)(hkl.x, hkl.y, hkl.z, 0);
      float2 plane_d;
//      float alpha = 0;  
      //float epsilonB = 0.01;
      beamInDotNormal = dot(abc,planeNormal);

      if (temperature>0)
        {
            plane_d = get_distance_Si(temperature, dhkl);

        }
        else
        {
            //plane_d = get_distance_Si(temperature, dhkl);
            plane_d = get_distance(lattice, dhkl);
        }
      //plane_d = get_distance(lattice, dhkl);

      mem_fence(CLK_LOCAL_MEM_FENCE);
      d = plane_d.lo;
      chiToF = plane_d.hi;
      //printf("d, ctf: %g %g\n", d, chiToF);
      //dt = get_dtheta_symmetric_Bragg_E(E, d,
      //                 hkl, chiToF, factDW,
      //                 maxEl, elements,
      //                 f0cfs, f1f2, alpha, geom);
      dt = get_dtheta_E(E, d,
                        hkl, chiToF, factDW,
                        maxEl, elements,
                        f0cfs, f1f2, 0, geom);
      mem_fence(CLK_LOCAL_MEM_FENCE);
      if (isnan(dt)) dt = 0;
      //printf("dt %g\n", dt);
      normalDotSurfNormal = dot(planeNormal,surfNormal);
      gNormalCryst = (planeNormal - normalDotSurfNormal*surfNormal) *
                          dt / d / 1e-7 *
                          sqrt(fabs(1. - pown(normalDotSurfNormal,2)));
       /*          if matSur.geom.endswith('Fresnel'):
                  if isinstance(self.order, int):
                      locOrder = self.order
                  else:
                      locOrder = np.array(self.order)[np.random.randint(
                          len(self.order), size=goodN.sum())]
                  if gNormal is None:
                      gNormal = local_g(lb.x[goodN], lb.y[goodN])
                  gNormal = np.asarray(gNormal, order='F') * locOrder
                  gNormal[0] += gNormalCryst[0]
                  gNormal[1] += gNormalCryst[1]
                  gNormal[2] += gNormalCryst[2]
              else:
        */        gNormal = gNormalCryst;
      abc_out = _grating_deflection(abc, planeNormal, gNormal,
                      E, beamInDotNormal, 1.,
                      false, ray);
      mem_fence(CLK_LOCAL_MEM_FENCE);
      beamInDotSurfaceNormal = dot(abc,surfNormal);
      beamOutDotSurfaceNormal = dot(abc_out,surfNormal);
      //if the beam reflects through the crystal use Laue model
      //geom.lo = 1,0:Bragg,Laue
      //geom.hi = 1,0:transmitted,reflected
      geom.lo = 1; //bragg
      //geom.hi = 0; //reflect
      if (beamInDotSurfaceNormal*beamOutDotSurfaceNormal > 0) geom.lo = 0;
      amplitudes = get_amplitude_internal_E(E, d, hkl,
                         chiToF, factDW,
                         beamInDotSurfaceNormal, beamOutDotSurfaceNormal,
                         beamInDotNormal,
                         thickness, geom, maxEl, elements,
                         f0cfs, f1f2);
      mem_fence(CLK_LOCAL_MEM_FENCE);
      //printf("amplitudes %g %g %g %g\n", amplitudes.x,amplitudes.y,amplitudes.z,amplitudes.w);
      if (any(isnan(amplitudes.lo)))
                {
                  amplitudes.lo = cmp0;
                }
      if (any(isnan(amplitudes.hi)))
                {
                  amplitudes.hi = cmp0;
                }
      //sampling refracted or reflected beam
      /*if (abs_c(amplitudes.lo) + abs_c(amplitudes.hi) > 2)
        {
          geom.hi = 1;
          abc_out = abc;
          amplitudes = get_amplitude_internal(E, d, hkl,
                         chiToF, factDW,
                         beamInDotSurfaceNormal, beamOutDotSurfaceNormal,
                         beamInDotNormal,
                         thickness, geom,maxEl, elements,
                         f0cfs, E_vector,
                         f1_vector, f2_vector);
        }
      mem_fence(CLK_LOCAL_MEM_FENCE);
      if (any(isnan(amplitudes.lo)))
                {
                  amplitudes.lo = cmp0;
                }
      if (any(isnan(amplitudes.hi)))
                {
                  amplitudes.hi = cmp0;
                }*/
      abc_out.w = geom.lo;
      return (float8)(amplitudes,abc_out);
}

float8 reflect_single_crystal(const float factDW,
                              const float thickness,
                              int2 geom,
                              const float8 lattice,
                              const float temperature,
                              float4 abc,
                              float E,
                              float4 planeNormal,
                              float4 surfNormal,
                              int4 hkl,
                              int nmax,
                              const int maxEl,
                              __global float* elements,
                              __global float* f0cfs,
                              __global float* E_vector,
                              __global float* f1_vector,
                              __global float* f2_vector,
                              unsigned int ray
                              )
  {

      float4 abc_out, max_abc;
      //unsigned int Z = 14; //for debug only
      float  iPlaneMod = 0;
      //float2 zero2 = (float2)(0,0);
      //abc=normalize(abc);
      abc_out = abc;
      max_abc = abc_out;
      float2 maxS = cmp0;
      float2 maxP = cmp0;
      //float beamDotiPlane;
      int ih, ik, il;
      float4 cutNormal = (float4)(1,1,1,0);
      float cutNormalMod = length(cutNormal);
      //int4 maxhkl = (int4)(0,0,0,0);
      float2 curveS, curveP;
      float8 dirflux;
      float4 iPlaneinCut, iPlane, iPlaneOrt;
      float4 dmaxhkl = (float4)(0,0,0,0);
      float rAngle = -acos(dot(planeNormal / length(planeNormal),cutNormal / length(cutNormal)));
      float Qw = cos(0.5*rAngle);
      float4 Qvp = cross(planeNormal / length(planeNormal),cutNormal / length(cutNormal));
      float4 Qv = Qvp / length(Qvp) * sin(0.5*rAngle);
      //setting up quarternions
      float Qxx = Qv.x*Qv.x;
      float Qxy = Qv.x*Qv.y;
      float Qxz = Qv.x*Qv.z;
      float Qxw = Qv.x*Qw;
      float Qyy = Qv.y*Qv.y;
      float Qyz = Qv.y*Qv.z;
      float Qyw = Qv.y*Qw;
      float Qzz = Qv.z*Qv.z;
      float Qzw = Qv.z*Qw;
      float4 Mrx = (float4)(1 - 2 * (Qyy + Qzz), 2 * (Qxy - Qzw), 2 * (Qxz + Qyw), 0);
      float4 Mry = (float4)(2 * (Qxy + Qzw), 1 - 2 * (Qxx + Qzz), 2 * (Qyz - Qxw), 0);
      float4 Mrz = (float4)(2 * (Qxz - Qyw), 2 * (Qyz + Qxw), 1 - 2 * (Qxx + Qyy), 0);

      for (ih=-nmax; ih<nmax+1; ih++)
        {
        for (ik=-nmax; ik<nmax+1; ik++)
          {
          for (il=-nmax; il<nmax+1; il++)
            {

              if (abs(ih)+abs(ik)+abs(il) == 0) continue;
              iPlane = (float4)(ih,ik,il,0);
              iPlaneMod = length(iPlane);
              iPlaneOrt = iPlane / iPlaneMod;
              iPlaneinCut.x = dot(Mrx,iPlaneOrt);
              iPlaneinCut.y = dot(Mry,iPlaneOrt);
              iPlaneinCut.z = dot(Mrz,iPlaneOrt);
              //beamDotiPlane = dot(abc,iPlaneinCut);
              dirflux = reflect_crystal_internal(factDW,thickness,geom,lattice,
                              temperature, abc,E,iPlaneinCut,surfNormal,
                              (int4)(ih,ik,il,0),maxEl,elements,
                              f0cfs,E_vector,f1_vector, f2_vector, ray
                              );
              curveS = (dirflux.lo).lo;
              curveP = (dirflux.lo).hi;
              abc_out = dirflux.hi;
              if (abs_c(curveS)+abs_c(curveP) >= abs_c(maxS)+abs_c(maxP))
                {
                  maxS = curveS;
                  maxP = curveP;
                  max_abc = abc_out;
                  //maxhkl = (int4)(ih,ik,il,0);
                  //dmaxhkl = iPlaneinCut;
                }
          }
        }
      }
      mem_fence(CLK_LOCAL_MEM_FENCE);
      max_abc.w = 0;
      return (float8)(maxS,maxP,max_abc);
  }

float8 reflect_harmonics(const float factDW,
                              const float thickness,
                              int2 geom,
                              const float8 lattice,
                              const float temperature,
                              float4 abc,
                              float E,
                              float4 planeNormal,
                              float4 surfNormal,
                              int4 hkl,
                              int nmax,
                              const int maxEl,
                              __global float* elements,
                              __global float* f0cfs,
                              __global float* E_vector,
                              __global float* f1_vector,
                              __global float* f2_vector,
                              unsigned int ray
                              )
  {

      float4 abc_out, max_abc;
      //float2 zero2 = (float2)(0,0);
      abc_out = abc;
      max_abc = abc_out;
      float2 maxS = cmp0;
      float2 maxP = cmp0;
      int nh;
      float2 curveS, curveP;
      float8 dirflux;

      for (nh=1;nh<nmax+1;nh++)
        {

              dirflux = reflect_crystal_internal(factDW,thickness,geom,lattice,
                              temperature, abc,E,planeNormal,surfNormal,
                              nh*hkl, maxEl, elements,
                              f0cfs,E_vector,f1_vector, f2_vector, ray
                              );
              curveS = (dirflux.lo).lo;
              curveP = (dirflux.lo).hi;
              abc_out = dirflux.hi;

              if (abs_c(curveS)+abs_c(curveP) >= abs_c(maxS)+abs_c(maxP))
                {
                  maxS = curveS;
                  maxP = curveP;
                  max_abc = abc_out;
                }

      }
      mem_fence(CLK_LOCAL_MEM_FENCE);
      max_abc.w = 0;
      return (float8)(maxS,maxP,max_abc);
  }

float8 reflect_powder(const float factDW,
                              const float thickness,
                              int2 geom,
                              const float8 lattice,
                              const float temperature,
                              float4 abc,
                              float E,
                              float4 planeNormal,
                              float4 surfNormal,
                              int4 hklmax,
                              const int maxEl,
                              __global float* elements,
                              __global float* f0cfs,
                              __global float* E_vector,
                              __global float* f1_vector,
                              __global float* f2_vector,
                              unsigned int ray
                              )
  {

      float4 abc_out;
      float4 max_abc = (float4)(0,0,0,0);
      //float2 zero2 = (float2)(0,0);
      float2 maxS = cmp0;
      float2 maxP = cmp0;
      int ih, ik, il;
      //int4 maxhkl;
      float2 curveS, curveP;
      float8 dirflux;
      for (ih = 0; ih < hklmax.x + 1; ih++)
        {
        for (ik = 0; ik < hklmax.y + 1; ik++)
          {
          for (il = 0; il < hklmax.z + 1; il++)
            {
              if (abs(ih)+abs(ik)+abs(il) == 0) continue;
              dirflux = reflect_crystal_internal(factDW,thickness,geom,lattice,
                              temperature, abc,E,planeNormal,surfNormal,
                              (int4)(ih,ik,il,0), maxEl, elements,
                              f0cfs,E_vector,f1_vector, f2_vector, ray);
              mem_fence(CLK_LOCAL_MEM_FENCE);
              curveS = (dirflux.lo).lo;
              curveP = (dirflux.lo).hi;
              //printf("AS, AP: %g+j%g, %g+j%g\n", curveS.x, curveS.y, curveP.x, curveP.y);

              abc_out = dirflux.hi;
              if (any(isnan(curveS)))
                {
                  curveS = cmp0;
                }
              if (any(isnan(curveP)))
                {
                  curveP = cmp0;
                }
              if (abs_c(curveS)+abs_c(curveP) > abs_c(maxS)+abs_c(maxP))
                {
                  maxS = curveS;
                  maxP = curveP;
                  max_abc = abc_out;
                }
     mem_fence(CLK_LOCAL_MEM_FENCE);
          }
        }
      }
     //printf("AS, AP: %g+j%g, %g+j%g\n", maxS.x, maxS.y, maxP.x, maxP.y);

     mem_fence(CLK_LOCAL_MEM_FENCE);
      max_abc.w = 0;
      return (float8)(maxS,maxP,max_abc);
  }

__kernel void reflect_crystal(const int calctype,
                            const int4 hkl,
                            const float factDW,
                            const float thickness,
                            const float temperature,
                            const int geom_b,
                            const int maxEl,
                            const float8 lattice,
                            __global float* a_gl,
                            __global float* b_gl,
                            __global float* c_gl,
                            __global float* E_gl,
                            __global float* planeNormalX,
                            __global float* planeNormalY,
                            __global float* planeNormalZ,
                            __global float* surfNormalX,
                            __global float* surfNormalY,
                            __global float* surfNormalZ,
                            __global float* elements,
                            __global float* f0cfs,
                            __global float* E_vector,
                            __global float* f1_vector,
                            __global float* f2_vector,
                            //__global ranluxcl_state_t *ranluxcltab,
                            __global float2* refl_s,
                            __global float2* refl_p,
                            __global float* a_out,
                            __global float* b_out,
                            __global float* c_out
                            )
  {
      unsigned int ii = get_global_id(0);
      float Energy = E_gl[ii];
      float8 dirflux;
      float4 planeNormal, surfNormal;
      float4 abc = (float4)(a_gl[ii],b_gl[ii],c_gl[ii],0);
      planeNormal = (float4)(planeNormalX[ii],planeNormalY[ii],planeNormalZ[ii],0);
      surfNormal = (float4)(surfNormalX[ii],surfNormalY[ii],surfNormalZ[ii],0);
      //printf("planeNormal: %g %g %g\n", planeNormal.x, planeNormal.y, planeNormal.z);
      //printf("surfNormal: %g %g %g\n", surfNormal.x, surfNormal.y, surfNormal.z);
      //uint ins = floor(Energy*100);
      //ranluxcl_state_t ranluxclstate;
      //ranluxcl_initialization(ins, ranluxcltab);
      //ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
      //printf("cl_gl addr: %i\n",cl_gl);
      int2 geom;
      geom.lo = geom_b > 1 ? 1 : 0;
      geom.hi = (int)(fabs(remainder(geom_b,2.)));
      //printf("calctype = %i\n", calctype);
      if (calctype > 100)
        {
          dirflux = reflect_harmonics(factDW,thickness,geom,lattice,
                              temperature, abc,Energy,planeNormal,surfNormal,
                              hkl, calctype-100, maxEl, elements,
                              f0cfs,E_vector,f1_vector,f2_vector, ii);
            mem_fence(CLK_LOCAL_MEM_FENCE);
        }
      else if (calctype > 10 && calctype < 100)
        {
          dirflux = reflect_single_crystal(factDW,thickness,geom,lattice,
                              temperature, abc,Energy,planeNormal,surfNormal,
                              hkl, calctype-10, maxEl, elements,
                              f0cfs,E_vector,f1_vector,f2_vector, ii);
            mem_fence(CLK_LOCAL_MEM_FENCE);
        }
      else if (calctype == 5)
        {
          dirflux = reflect_powder(factDW,thickness,geom,lattice,
                              temperature, abc,Energy,planeNormal,surfNormal,
                              hkl, maxEl, elements,
                              f0cfs,E_vector,f1_vector,f2_vector, ii);
            mem_fence(CLK_LOCAL_MEM_FENCE);
        }
      else // (calctype == 0)
        {
          dirflux = reflect_crystal_internal(factDW,thickness,geom,lattice,
                              temperature, abc,Energy,planeNormal,surfNormal,
                              hkl, maxEl, elements,
                              f0cfs,E_vector,f1_vector,f2_vector,ii);
            mem_fence(CLK_LOCAL_MEM_FENCE);
        }
      //params.s01 = ranluxcl64(&ranluxclstate).s01;
      //float4 randomnr = ranluxcl64(&ranluxclstate);


      //printf("S, P: %g+j%g, %g+j%g, ray %i\n",
      //      dirflux.s0, dirflux.s1, dirflux.s2, dirflux.s3, ii);
      refl_s[ii] = dirflux.s01;
      refl_p[ii] = dirflux.s23;
      a_out[ii] = dirflux.s4;
      b_out[ii] = dirflux.s5;
      c_out[ii] = dirflux.s6;
      mem_fence(CLK_GLOBAL_MEM_FENCE);
      //barrier(CLK_LOCAL_MEM_FENCE);
      //ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
  }

/*
__kernel void propagate_wave_material(
                    const unsigned int nInputRays,
                    const unsigned int nOutputRays,
                    const float chbar,
                    const float2 refrac_n,
                    const float thickness,
                    const int kind,
                    const unsigned int fromVacuum,
                    const float E_loc,
                    //__global float* cosGamma,
                    __global float* x_glo,
                    __global float* y_glo,
                    __global float* z_glo,
                    __global float* ag,
                    __global float2* Es,
                    __global float2* Ep,
                    //__global float* E_loc,
                    __global float4* beamOEglo,

                    __global float* surfNormalX,
                    __global float* surfNormalY,
                    __global float* surfNormalZ,
                    __global float2* KirchS_gl,
                    __global float2* KirchP_gl)

{
    unsigned int i;
    float4 beam_coord_glo, beam_angle_glo;
    float2 gi, KirchS_loc, KirchP_loc, Esr, Epr;
    float pathAfter, cosAlpha, cr;
    unsigned int ii_screen = get_global_id(0);
    float k, wavelength;
    float phase, ag_loc;
    KirchS_loc = cmp0;
    KirchP_loc = cmp0;
    float4 refl_amp=(float4)(cmp1,cmp1);
    float4 oe_surface_normal = (float4)(surfNormalX[ii_screen], 
                                            surfNormalY[ii_screen], 
                                            surfNormalZ[ii_screen], 0);

    beam_coord_glo.x = x_glo[ii_screen];
    beam_coord_glo.y = y_glo[ii_screen];
    beam_coord_glo.z = z_glo[ii_screen];
    beam_coord_glo.w = 0.;

    
    for (i=0; i<nInputRays; i++)
    {
        refl_amp = (float4)(cmp1,cmp1);
        beam_angle_glo = beam_coord_glo - beamOEglo[i];
        pathAfter = length(beam_angle_glo);
        cosAlpha = dot(beam_angle_glo, oe_surface_normal) / pathAfter;
        if (kind==0) //mirror
        {
            refl_amp = get_amplitude_material(E_loc, kind, refrac_n, cosAlpha, 
                                                thickness, fromVacuum);
        }
        else if (kind==6) //crystal
        {
            refl_amp = get_amplitude_material(E_loc, kind, refrac_n, cosAlpha, 
                                                thickness, fromVacuum);
        }

        k = E_loc / chbar * 1.e7;
        wavelength = twoPi / k;
        phase = k * pathAfter;
        Esr = prod_c(Es[i], refl_amp.s01);
        Epr = prod_c(Ep[i], refl_amp.s23);
        //Esr = Es[i];
        //Epr = Ep[i];
        cr = cosAlpha / pathAfter;
        gi = (float2)(cr * cos(phase), cr * sin(phase));
        //ag_loc = pown(ag[i],gau);
        
        KirchS_loc += (ag[i] * prod_c(gi, Esr));
        KirchP_loc += (ag[i] * prod_c(gi, Epr));
    }
  mem_fence(CLK_LOCAL_MEM_FENCE);
 
  KirchS_gl[ii_screen] = prod_c((-cmpi1/wavelength), KirchS_loc);
  KirchP_gl[ii_screen] = prod_c((-cmpi1/wavelength), KirchP_loc);
  mem_fence(CLK_LOCAL_MEM_FENCE);
}


__kernel void propagate_wave_crystal(
                    const unsigned int nInputRays,
                    const unsigned int nOutputRays,
                    const float chbar,

                    const int kind,
                    const int4 hkl,
                    const float factDW,
                    const float thickness,
                    const float temperature,
                    const int geom_b,
                    const int maxEl,
                    const float8 lattice,

                    const float E_loc,

                    __global float* x_glo,
                    __global float* y_glo,
                    __global float* z_glo,
                    __global float* ag,
                    __global float2* Es,
                    __global float2* Ep,

                    __global float4* beamOEglo,

                    __global float* planeNormalX,
                    __global float* planeNormalY,
                    __global float* planeNormalZ,
                    __global float* surfNormalX,
                    __global float* surfNormalY,
                    __global float* surfNormalZ,

                    __global float* elements,
                    __global float* f0cfs,
                    __global float2* f1f2,

                    __global float2* KirchS_gl,
                    __global float2* KirchP_gl)

{
    unsigned int i;
    int2 geom;
    float4 beam_coord_glo, beam_angle_glo, abc;
    float2 gi, KirchS_loc, KirchP_loc, Esr, Epr;
    float pathAfter, cosAlpha, cr;
    unsigned int ii_screen = get_global_id(0);
    float k, wavelength;
    float phase, ag_loc;
    KirchS_loc = cmp0;
    KirchP_loc = cmp0;
    float4 refl_amp=(float4)(cmp1,cmp1);

    geom.lo = geom_b > 1 ? 1 : 0;
    geom.hi = (int)(fabs(remainder(geom_b,2.)));

    float4 planeNormal = (float4)(planeNormalX[ii_screen],
                            planeNormalY[ii_screen],
                            planeNormalZ[ii_screen],0);

    float4 surfNormal = (float4)(surfNormalX[ii_screen],
                            surfNormalY[ii_screen],
                            surfNormalZ[ii_screen],0);



    beam_coord_glo.x = x_glo[ii_screen];
    beam_coord_glo.y = y_glo[ii_screen];
    beam_coord_glo.z = z_glo[ii_screen];
    beam_coord_glo.w = 0.;

    
    for (i=0; i<nInputRays; i++)
    {
        refl_amp = (float4)(cmp1,cmp1);
        beam_angle_glo = beam_coord_glo - beamOEglo[i];
        pathAfter = length(beam_angle_glo);
        cosAlpha = dot(beam_angle_glo, surfNormal) / pathAfter;
        abc = beam_angle_glo/pathAfter;
        if (kind==6) //crystal
        {
            refl_amp = (reflect_crystal_internal_E(factDW,thickness,geom,lattice,
                temperature, abc, E_loc, planeNormal, surfNormal,
                hkl, maxEl, elements,
                f0cfs, f1f2, i
                )).s0123; 
        }

        k = E_loc / chbar * 1.e7;
        wavelength = twoPi / k;
        phase = k * pathAfter;
        Esr = prod_c(Es[i], refl_amp.s01);
        Epr = prod_c(Ep[i], refl_amp.s23);
        //Esr = Es[i];
        //Epr = Ep[i];
        cr = cosAlpha / pathAfter;
        gi = (float2)(cr * cos(phase), cr * sin(phase));
        //ag_loc = pown(ag[i],gau);
        
        KirchS_loc += (ag[i] * prod_c(gi, Esr));
        KirchP_loc += (ag[i] * prod_c(gi, Epr));
    }
  mem_fence(CLK_LOCAL_MEM_FENCE);
 
  KirchS_gl[ii_screen] = prod_c((-cmpi1/wavelength), KirchS_loc);
  KirchP_gl[ii_screen] = prod_c((-cmpi1/wavelength), KirchP_loc);
  mem_fence(CLK_LOCAL_MEM_FENCE);
}
*/