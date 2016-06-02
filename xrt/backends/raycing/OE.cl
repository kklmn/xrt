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
#include "materials.cl"
//#include "pyopencl-ranluxcl.cl"

__constant double zEps = 1.e-12;
__constant bool isParametric = false;
__constant int maxIteration = 100;
//__constant long cl_gl=0;
//__global double *cl_gl;
//__constant double PI = 3.141592653589793238;
//__constant double ch = 12398.4186;  // {5}   {c*h[eV*A]}

double2 rotate_x(double y, double z, double cosangle, double sinangle)
{
    return (double2)(cosangle*y - sinangle*z, sinangle*y + cosangle*z);
}

double2 rotate_y(double x, double z, double cosangle, double sinangle)
{
    return (double2)(cosangle*x + sinangle*z, -sinangle*x + cosangle*z);
}

double2 rotate_z(double x, double y, double cosangle, double sinangle)
{
    return (double2)(cosangle*x - sinangle*y, sinangle*x + cosangle*y);
}

MY_LOCAL_Z

MY_LOCAL_N

MY_LOCAL_G

MY_XYZPARAM

double4 find_dz(double8 cl_plist,
               int local_f, double t, double x0, double y0, double z0,
               double a, double b, double c, int invertNormal, int derivOrder)
{
        /*Returns the z or r difference (in the local system) between the ray
        and the surface. Used for finding the intersection point.*/

        double x = x0 + a * t;
        double y = y0 + b * t;
        double z = z0 + c * t;
        double surf, dz;
        double3 surf3;
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
            //surf3 = (double3)(0,0,0);
            if (!isnormal(surf))
              {
                surf3.s0 = 0;
                surf3.s1 = 0;
                surf3.s2 = 1;
              }
            dz = (a * surf3.s0 + b * surf3.s1 + c * surf3.s2) * invertNormal;
          }
        //mem_fence(CLK_GLOBAL_MEM_FENCE);
        return (double4)(dz, x, y, z);
}

double4 _use_my_method(double8 cl_plist,
                      int local_f, double tMin, double tMax, double t1, double t2,
                      double x, double y, double z,
                      double a, double b, double c,
                      int invertNormal, int derivOrder, double dz1, double dz2,
                      double x2, double y2, double z2)
{

        double4 tmp;
        unsigned int numit=2;
        double t, dz;
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
        return (double4)(t2, x2, y2, z2);
}
double4 _use_Brent_method(double8 cl_plist,
                      int local_f, double tMin, double tMax, double t1, double t2,
                      double x, double y, double z,
                      double a, double b, double c,
                      int invertNormal, int derivOrder, double dz1, double dz2,
                      double x2, double y2, double z2)
{
        double4 tmp;
        double tmpt, t3, t4, dz3, xa, xb, xc, xd, xs, xai, xbi, xci;
        double fa, fb, fc, fai, fbi, fci, fs;
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

        return (double4)(t2, x2, y2, z2);
}


double4 find_intersection_internal(double8 cl_plist,
                          int local_f, double tMin, double tMax,
                          double t1, double t2,
                          double x, double y, double z,
                          double a, double b, double c,
                          int invertNormal, int derivOrder)
{
        /*Finds the ray parameter *t* at the intersection point with the
        surface. Requires *t1* and *t2* as input bracketing. The ray is
        determined by its origin point (*x*, *y*, *z*) and its normalized
        direction (*a*, *b*, *c*). *t* is then the distance between the origin
        point and the intersection point. *derivOrder* tells if minimized is
        the z-difference (=0) or its derivative (=1).*/
        double4 tmp1, res;
        double dz1, dz2;
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
                  const double8 cl_plist_gl,
                  const int invertNormal,
                  const int derivOrder,
                  const int local_zN,
                  const double tMin,
                  const double tMax,
                  __global double *t1,
                  __global double *x,
                  __global double *y,
                  __global double *z,
                  __global double *a,
                  __global double *b,
                  __global double *c,
                  __global double *t2,
                  __global double *x2,
                  __global double *y2,
                  __global double *z2
                  )
{
      double4 res;
      double8 cl_plist = cl_plist_gl;
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


double4 _grating_deflection(double4 abc, double4 oeNormal, double4 gNormal,
                            double E, double beamInDotNormal, double order,
                            bool isTransmitting, unsigned int ray)
  {
        double4 abc_out;
        double beamInDotG, G2, locOrder, orderLambda, sig, dn;
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

double8 reflect_crystal_internal(const double factDW,
                              const double thickness,
                              int2 geom,
                              const double8 lattice,
                              const double temperature,
                              double4 abc,
                              double E,
                              double4 planeNormal,
                              double4 surfNormal,
                              int4  hkl,
                              const int maxEl,
                              __global double* elements,
                              __global double* f0cfs,
                              __global double* E_vector,
                              __global double* f1_vector,
                              __global double* f2_vector, unsigned int ray
                              )
{
      //printf("got to rsi, ray %i\n", ray);
      double4 abc_out, dhkl, amplitudes, gNormalCryst, gNormal;
      double beamInDotNormal, d, chiToF, normalDotSurfNormal;
      double beamInDotSurfaceNormal, beamOutDotSurfaceNormal, dt;
      dhkl = (double4)(hkl.x, hkl.y, hkl.z, 0);
      double2 plane_d;
      //double epsilonB = 0.01;
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
      return (double8)(amplitudes,abc_out);
}

double8 reflect_crystal_internal_E(const double factDW,
                              const double thickness,
                              int2 geom,
                              const double8 lattice,
                              const double temperature,
                              double4 abc,
                              double E,
                              double4 planeNormal,
                              double4 surfNormal,
                              int4  hkl,
                              const int maxEl,
                              __global double* elements,
                              __global double* f0cfs,
                              __global double2* f1f2, unsigned int ray
                              )
{
      //printf("got to rsi, ray %i\n", ray);
      double4 abc_out, dhkl, amplitudes, gNormalCryst, gNormal;
      double beamInDotNormal, d, chiToF, normalDotSurfNormal;
      double beamInDotSurfaceNormal, beamOutDotSurfaceNormal, dt;
      dhkl = (double4)(hkl.x, hkl.y, hkl.z, 0);
      double2 plane_d;
      double alpha = 0;  
      //double epsilonB = 0.01;
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
      return (double8)(amplitudes,abc_out);
}

double8 reflect_single_crystal(const double factDW,
                              const double thickness,
                              int2 geom,
                              const double8 lattice,
                              const double temperature,
                              double4 abc,
                              double E,
                              double4 planeNormal,
                              double4 surfNormal,
                              int4 hkl,
                              int nmax,
                              const int maxEl,
                              __global double* elements,
                              __global double* f0cfs,
                              __global double* E_vector,
                              __global double* f1_vector,
                              __global double* f2_vector,
                              unsigned int ray
                              )
  {

      double4 abc_out, max_abc;
      //unsigned int Z = 14; //for debug only
      double  iPlaneMod = 0;
      //double2 zero2 = (double2)(0,0);
      //abc=normalize(abc);
      abc_out = abc;
      max_abc = abc_out;
      double2 maxS = cmp0;
      double2 maxP = cmp0;
      //double beamDotiPlane;
      int ih, ik, il;
      double4 cutNormal = (double4)(1,1,1,0);
      double cutNormalMod = length(cutNormal);
      //int4 maxhkl = (int4)(0,0,0,0);
      double2 curveS, curveP;
      double8 dirflux;
      double4 iPlaneinCut, iPlane, iPlaneOrt;
      double4 dmaxhkl = (double4)(0,0,0,0);
      double rAngle = -acos(dot(planeNormal / length(planeNormal),cutNormal / length(cutNormal)));
      double Qw = cos(0.5*rAngle);
      double4 Qvp = cross(planeNormal / length(planeNormal),cutNormal / length(cutNormal));
      double4 Qv = Qvp / length(Qvp) * sin(0.5*rAngle);
      //setting up quarternions
      double Qxx = Qv.x*Qv.x;
      double Qxy = Qv.x*Qv.y;
      double Qxz = Qv.x*Qv.z;
      double Qxw = Qv.x*Qw;
      double Qyy = Qv.y*Qv.y;
      double Qyz = Qv.y*Qv.z;
      double Qyw = Qv.y*Qw;
      double Qzz = Qv.z*Qv.z;
      double Qzw = Qv.z*Qw;
      double4 Mrx = (double4)(1 - 2 * (Qyy + Qzz), 2 * (Qxy - Qzw), 2 * (Qxz + Qyw), 0);
      double4 Mry = (double4)(2 * (Qxy + Qzw), 1 - 2 * (Qxx + Qzz), 2 * (Qyz - Qxw), 0);
      double4 Mrz = (double4)(2 * (Qxz - Qyw), 2 * (Qyz + Qxw), 1 - 2 * (Qxx + Qyy), 0);

      for (ih=-nmax; ih<nmax+1; ih++)
        {
        for (ik=-nmax; ik<nmax+1; ik++)
          {
          for (il=-nmax; il<nmax+1; il++)
            {

              if (abs(ih)+abs(ik)+abs(il) == 0) continue;
              iPlane = (double4)(ih,ik,il,0);
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
      return (double8)(maxS,maxP,max_abc);
  }

double8 reflect_harmonics(const double factDW,
                              const double thickness,
                              int2 geom,
                              const double8 lattice,
                              const double temperature,
                              double4 abc,
                              double E,
                              double4 planeNormal,
                              double4 surfNormal,
                              int4 hkl,
                              int nmax,
                              const int maxEl,
                              __global double* elements,
                              __global double* f0cfs,
                              __global double* E_vector,
                              __global double* f1_vector,
                              __global double* f2_vector,
                              unsigned int ray
                              )
  {

      double4 abc_out, max_abc;
      //double2 zero2 = (double2)(0,0);
      abc_out = abc;
      max_abc = abc_out;
      double2 maxS = cmp0;
      double2 maxP = cmp0;
      int nh;
      double2 curveS, curveP;
      double8 dirflux;

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
      return (double8)(maxS,maxP,max_abc);
  }

double8 reflect_powder(const double factDW,
                              const double thickness,
                              int2 geom,
                              const double8 lattice,
                              const double temperature,
                              double4 abc,
                              double E,
                              double4 planeNormal,
                              double4 surfNormal,
                              int4 hklmax,
                              const int maxEl,
                              __global double* elements,
                              __global double* f0cfs,
                              __global double* E_vector,
                              __global double* f1_vector,
                              __global double* f2_vector,
                              unsigned int ray
                              )
  {

      double4 abc_out;
      double4 max_abc = (double4)(0,0,0,0);
      //double2 zero2 = (double2)(0,0);
      double2 maxS = cmp0;
      double2 maxP = cmp0;
      int ih, ik, il;
      //int4 maxhkl;
      double2 curveS, curveP;
      double8 dirflux;
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
      return (double8)(maxS,maxP,max_abc);
  }

__kernel void reflect_crystal(const int calctype,
                            const int4 hkl,
                            const double factDW,
                            const double thickness,
                            const double temperature,
                            const int geom_b,
                            const int maxEl,
                            const double8 lattice,
                            __global double* a_gl,
                            __global double* b_gl,
                            __global double* c_gl,
                            __global double* E_gl,
                            __global double* planeNormalX,
                            __global double* planeNormalY,
                            __global double* planeNormalZ,
                            __global double* surfNormalX,
                            __global double* surfNormalY,
                            __global double* surfNormalZ,
                            __global double* elements,
                            __global double* f0cfs,
                            __global double* E_vector,
                            __global double* f1_vector,
                            __global double* f2_vector,
                            //__global ranluxcl_state_t *ranluxcltab,
                            __global double2* refl_s,
                            __global double2* refl_p,
                            __global double* a_out,
                            __global double* b_out,
                            __global double* c_out
                            )
  {
      unsigned int ii = get_global_id(0);
      double Energy = E_gl[ii];
      double8 dirflux;
      double4 planeNormal, surfNormal;
      double4 abc = (double4)(a_gl[ii],b_gl[ii],c_gl[ii],0);
      planeNormal = (double4)(planeNormalX[ii],planeNormalY[ii],planeNormalZ[ii],0);
      surfNormal = (double4)(surfNormalX[ii],surfNormalY[ii],surfNormalZ[ii],0);
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
      //double4 randomnr = ranluxcl64(&ranluxclstate);


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
                    const double chbar,
                    const double2 refrac_n,
                    const double thickness,
                    const int kind,
                    const unsigned int fromVacuum,
                    const double E_loc,
                    //__global double* cosGamma,
                    __global double* x_glo,
                    __global double* y_glo,
                    __global double* z_glo,
                    __global double* ag,
                    __global double2* Es,
                    __global double2* Ep,
                    //__global double* E_loc,
                    __global double4* beamOEglo,

                    __global double* surfNormalX,
                    __global double* surfNormalY,
                    __global double* surfNormalZ,
                    __global double2* KirchS_gl,
                    __global double2* KirchP_gl)

{
    unsigned int i;
    double4 beam_coord_glo, beam_angle_glo;
    double2 gi, KirchS_loc, KirchP_loc, Esr, Epr;
    double pathAfter, cosAlpha, cr;
    unsigned int ii_screen = get_global_id(0);
    double k, wavelength;
    double phase, ag_loc;
    KirchS_loc = cmp0;
    KirchP_loc = cmp0;
    double4 refl_amp=(double4)(cmp1,cmp1);
    double4 oe_surface_normal = (double4)(surfNormalX[ii_screen], 
                                            surfNormalY[ii_screen], 
                                            surfNormalZ[ii_screen], 0);

    beam_coord_glo.x = x_glo[ii_screen];
    beam_coord_glo.y = y_glo[ii_screen];
    beam_coord_glo.z = z_glo[ii_screen];
    beam_coord_glo.w = 0.;

    
    for (i=0; i<nInputRays; i++)
    {
        refl_amp = (double4)(cmp1,cmp1);
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
        gi = (double2)(cr * cos(phase), cr * sin(phase));
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
                    const double chbar,

                    const int kind,
                    const int4 hkl,
                    const double factDW,
                    const double thickness,
                    const double temperature,
                    const int geom_b,
                    const int maxEl,
                    const double8 lattice,

                    const double E_loc,

                    __global double* x_glo,
                    __global double* y_glo,
                    __global double* z_glo,
                    __global double* ag,
                    __global double2* Es,
                    __global double2* Ep,

                    __global double4* beamOEglo,

                    __global double* planeNormalX,
                    __global double* planeNormalY,
                    __global double* planeNormalZ,
                    __global double* surfNormalX,
                    __global double* surfNormalY,
                    __global double* surfNormalZ,

                    __global double* elements,
                    __global double* f0cfs,
                    __global double2* f1f2,

                    __global double2* KirchS_gl,
                    __global double2* KirchP_gl)

{
    unsigned int i;
    int2 geom;
    double4 beam_coord_glo, beam_angle_glo, abc;
    double2 gi, KirchS_loc, KirchP_loc, Esr, Epr;
    double pathAfter, cosAlpha, cr;
    unsigned int ii_screen = get_global_id(0);
    double k, wavelength;
    double phase, ag_loc;
    KirchS_loc = cmp0;
    KirchP_loc = cmp0;
    double4 refl_amp=(double4)(cmp1,cmp1);

    geom.lo = geom_b > 1 ? 1 : 0;
    geom.hi = (int)(fabs(remainder(geom_b,2.)));

    double4 planeNormal = (double4)(planeNormalX[ii_screen],
                            planeNormalY[ii_screen],
                            planeNormalZ[ii_screen],0);

    double4 surfNormal = (double4)(surfNormalX[ii_screen],
                            surfNormalY[ii_screen],
                            surfNormalZ[ii_screen],0);



    beam_coord_glo.x = x_glo[ii_screen];
    beam_coord_glo.y = y_glo[ii_screen];
    beam_coord_glo.z = z_glo[ii_screen];
    beam_coord_glo.w = 0.;

    
    for (i=0; i<nInputRays; i++)
    {
        refl_amp = (double4)(cmp1,cmp1);
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
        gi = (double2)(cr * cos(phase), cr * sin(phase));
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