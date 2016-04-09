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
        for(j=0; j<jend; j++)
          {
            sintg = sincos(tg[j], &costg);
            sintgph = sincos(tg[j] + phase, &costgph);
            ucos = (float)(ww1[ii]) * tg[j] + wgwu *
               (-Ky * ddphi[ii] * sintg + Kx * ddpsi[ii] * sintgph +
                0.125/gamma[ii] * (Ky2*2*sintg*costg + Kx2*2*sintgph*costgph));

            sinucos = sincos(ucos, &cosucos);
            eucos.x = sinucos;
            eucos.y = cosucos;

            beta.x = -Ky / gamma[ii] * costg;
            beta.y = Kx / gamma[ii] * costgph;

            Is += (ag[j] * (ddphi[ii] + beta.x)) * eucos;
            Ip += (ag[j] * (ddpsi[ii] + beta.y)) * eucos;

          }
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
        for(j=0; j<jend; j++)
          {
            sintg = sin(tg[j]);
            sin2tg = sin(2 * tg[j]);
            costg = cos(tg[j]);
            ucos = ww1[ii] * tg[j] + wgwu *
                (-Ky * ddtheta[ii] *(sintg + alphaS *
                                      (1 - costg - tg[j] * sintg)) +
                  Kx * ddpsi[ii] * sin(tg[j] + phase) + 
                  0.125 / gamma[ii] * 
                  (Ky2 * (sin2tg - 2 * alphaS *
                    (tg[j] * tg[j] + costg * costg + tg[j] * sin2tg)) +
                   Kx2 * sin(2. * (tg[j] + phase))));

            eucos.x = cos(ucos);
            eucos.y = sin(ucos);

            beta.x = -Ky / gamma[ii] * costg;
            beta.y = Kx / gamma[ii] * cos(tg[j] + phase);

            Is += ag[j] * (ddtheta[ii] + beta.x *
                (1 - alphaS * tg[j])) * eucos;
            Ip += ag[j] * (ddpsi[ii] + beta.y) * eucos;
          }
        mem_fence(CLK_LOCAL_MEM_FENCE);
        Is_gl[ii] = Is;
        Ip_gl[ii] = Ip;
}

__kernel void undulator_nf(const float R0,
                    const float Kx,
                    const float Ky,
                    const float phase,
                    const int jend,
                    __global float* gamma,
                    __global float* wu,
                    __global float* w, 
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
        float ucos;
        float2 zero2 = (float2)(0,0);
        float2 Is = zero2;
        float2 Ip = zero2;
        float3 r, r0;
        float2 beta;
        float Kx2 = Kx * Kx;
        float Ky2 = Ky * Ky;        
        float gamma2 = gamma[ii] * gamma[ii];
        float betam = 1 - (1 + 0.5 * Kx2 + 0.5 * Ky2) / 2. / gamma2;

        float wR0 = R0 / C * E2W;
        r0.x = wR0 * tan(-ddphi[ii]);
        r0.y = wR0 * tan(ddpsi[ii]);
        r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii]));

        for(j=0; j<jend; j++)
          {

          r.x = Ky / wu[ii] / gamma[ii] * sin(tg[j]);
          r.y = -Kx / wu[ii] / gamma[ii] * sin(tg[j] + phase);
          r.z = betam * tg[j] / wu[ii] - 0.125 / wu[ii] / gamma2 *
                (Ky2 * sin(2 * tg[j]) + Kx2 * sin(2 * (tg[j] + phase)));

          ucos = w[ii] * (tg[j] / wu[ii] + length(r0 - r));

          eucos.x = cos(ucos);
          eucos.y = sin(ucos);

          beta.x = -Ky / gamma[ii] * cos(tg[j]);
          beta.y = Kx / gamma[ii] * cos(tg[j] + phase);

          Is += ag[j] * (-ddphi[ii] + beta.x) * eucos;
          Ip += ag[j] * (ddpsi[ii] + beta.y) * eucos;
          }
        mem_fence(CLK_LOCAL_MEM_FENCE);

        Is_gl[ii] = Is;
        Ip_gl[ii] = Ip;
}


double2 f_beta(double Bx, double By, double Bz,
               double emcg,
               double2 beta)
    {
    return emcg*(double2)(beta.y * Bz - By,
                          Bx - beta.x * Bz);
    }

double3 f_traj(double gamma, double2 beta)
    {
    return (double3)(beta.x,
                     beta.y,
                     sqrt(1. - 1./gamma/gamma - beta.x*beta.x - beta.y*beta.y));
    }


__kernel void undulator_custom(const double Kx,
                    const double Ky,
                    const double phase,
                    const int jend,
                    const int nwt,
                    const double lUnd,
                    __global double* gamma,
                    __global double* wu,
                    __global double* w,
                    __global float* ww1,
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
        double emcg = lUnd * SIE0 / SIM0 / C / gamma[ii] / PI2;

        double2 eucos;
        double ucos, dir;
        double  sintg, costg, sintgph, costgph, sinucos, cosucos, ucos_ref;
        double2 eucos_ref;
        double2 zero2 = (double2)(0,0);
        double2 Is = zero2;
        double2 Ip = zero2;
        double3 r, traj, traj2, n, k1Traj, k2Traj, k3Traj, k4Traj, traj_min, traj_max;
        double2 beta_ref, beta, k1Beta, k2Beta, k3Beta, k4Beta, beta_min, beta_max;

        int iBase, iZeroStep, iHalfStep, iFullStep;
        double rkStep;
	   //double tg2r = 1. / wu[ii]; // lUnd / PI2 * E2W / C;
        n.x = ddphi[ii];
        n.y = ddpsi[ii];
        n.z = 1. - 0.5*(n.x*n.x + n.y*n.y);
        double Kx2 = Kx * Kx;
        double Ky2 = Ky * Ky;
        double gamma2 = gamma[ii] * gamma[ii];
        double revgamma = 1. - 1./gamma2;
        double betam = 1. - (1. + 0.5 * Kx2 + 0.5 * Ky2) / 2. / gamma2;
        double wgwu = w[ii] / gamma[ii] / wu[ii];
        double wu_int;

        beta_max = (double2)(-1e20, -1e20);
        beta_min = (double2)(1e20, 1e20);
        beta = zero2;
        double betam_int=0;

        for (j=1; j<jend; j++)
            {

            iBase = (j-1)*2*nwt;
            rkStep = (tg[j] - tg[j-1]) / nwt;
            for (k=0; k<nwt; k++)
                {
                iZeroStep  = iBase + 2*k;
                iHalfStep = iBase + 2*k + 1;
                iFullStep = iBase + 2*(k + 1);

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
                beta += (k1Beta + 2.*k2Beta + 2.*k3Beta + k4Beta)/6.;

                if (beta.x > beta_max.x) beta_max.x = beta.x;
                if (beta.y > beta_max.y) beta_max.y = beta.y;
                if (beta.x < beta_min.x) beta_min.x = beta.x;
                if (beta.y < beta_min.y) beta_min.y = beta.y;
                }
			}
        mem_fence(CLK_LOCAL_MEM_FENCE);
	   beta = -0.5*(beta_max + beta_min);

        traj_max = (double3)(-1.0e20, -1.0e20, -1.0e20);
        traj_min = (double3)(1.0e20, 1.0e20, 1.0e20);
        traj = (double3)(0., 0., 0.);

        for (j=1; j<jend; j++)
            {
            iBase = (j-1)*2*nwt;
            rkStep = (tg[j] - tg[j-1]) / nwt;
            for (k=0; k<nwt; k++)
                {
                iZeroStep  = iBase + 2*k;
                iHalfStep = iBase + 2*k + 1;
                iFullStep = iBase + 2*(k + 1);

                k1Beta = rkStep * f_beta(Bx[iZeroStep],
                                         By[iZeroStep],
                                         Bz[iZeroStep],
                                         emcg, beta);
                k1Traj = rkStep * f_traj(gamma[ii], beta);

                k2Beta = rkStep * f_beta(Bx[iHalfStep],
                                         By[iHalfStep],
                                         Bz[iHalfStep],
                                         emcg, beta + 0.5*k1Beta);
                k2Traj = rkStep * f_traj(gamma[ii], beta + 0.5*k1Beta);

                k3Beta = rkStep * f_beta(Bx[iHalfStep],
                                         By[iHalfStep],
                                         Bz[iHalfStep],
                                         emcg, beta + 0.5*k2Beta);
                k3Traj = rkStep * f_traj(gamma[ii], beta + 0.5*k2Beta);

                k4Beta = rkStep * f_beta(Bx[iFullStep],
                                         By[iFullStep],
                                         Bz[iFullStep],
                                         emcg, beta + k3Beta);
                k4Traj = rkStep * f_traj(gamma[ii], beta + k3Beta);

                beta += (k1Beta + 2.*k2Beta + 2.*k3Beta + k4Beta)/6.;
                traj += (k1Traj + 2.*k2Traj + 2.*k3Traj + k4Traj)/6.;
                betam_int += rkStep*sqrt(revgamma - beta.x*beta.x - beta.y*beta.y);

				if (traj.x > traj_max.x) traj_max.x = traj.x;
				if (traj.y > traj_max.y) traj_max.y = traj.y;
				if (traj.z > traj_max.z) traj_max.z = traj.z;
				if (traj.x < traj_min.x) traj_min.x = traj.x;
				if (traj.y < traj_min.y) traj_min.y = traj.y;
				if (traj.z < traj_min.z) traj_min.z = traj.z;
                }
			}
          mem_fence(CLK_LOCAL_MEM_FENCE);
		beta = -0.5*(beta_max + beta_min);
		traj = -0.5*(traj_max + traj_min);
          betam_int /= tg[jend-1] - tg[0];
          wu_int = PI2 * C * betam_int / lUnd / E2W;    
          //printf("wu=%2.8e, wu_int=%2.8e\n",wu[ii], wu_int);
        //printf("1-betam: %2.8e, 1-betam_int=%2.8e\n", 1.-betam, 1.-betam_int/(tg[jend-1]-tg[0]));

        for (j=1; j<jend; j++)
            {
            iBase = (j-1)*2*nwt;
            rkStep = (tg[j] - tg[j-1]) / nwt;
            for (k=0; k<nwt; k++)
                {
                iZeroStep  = iBase + 2*k;
                iHalfStep = iBase + 2*k + 1;
                iFullStep = iBase + 2*(k + 1);

                k1Beta = rkStep * f_beta(Bx[iZeroStep],
                                         By[iZeroStep],
                                         Bz[iZeroStep],
                                         emcg, beta);
                k1Traj = rkStep * f_traj(gamma[ii], beta);

                k2Beta = rkStep * f_beta(Bx[iHalfStep],
                                         By[iHalfStep],
                                         Bz[iHalfStep],
                                         emcg, beta + 0.5*k1Beta);
                k2Traj = rkStep * f_traj(gamma[ii], beta + 0.5*k1Beta);

                k3Beta = rkStep * f_beta(Bx[iHalfStep],
                                         By[iHalfStep],
                                         Bz[iHalfStep],
                                         emcg, beta + 0.5*k2Beta);
                k3Traj = rkStep * f_traj(gamma[ii], beta + 0.5*k2Beta);

                k4Beta = rkStep * f_beta(Bx[iFullStep],
                                         By[iFullStep],
                                         Bz[iFullStep],
                                         emcg, beta + k3Beta);
                k4Traj = rkStep * f_traj(gamma[ii], beta + k3Beta);

                beta += (k1Beta + 2.*k2Beta + 2.*k3Beta + k4Beta)/6.;
                traj += (k1Traj + 2.*k2Traj + 2.*k3Traj + k4Traj)/6.;
                }
            mem_fence(CLK_LOCAL_MEM_FENCE);
    		  //traj2 =  tg2r * traj;
    		  dir = dot(n, traj);
            //ucos = w[ii] / wu_int * (tg[j]  - dir);
            ucos = w[ii] / wu[ii] * (tg[j]  - dir);
            //eucos.x = cos(ucos);
            //eucos.y = sin(ucos);
/*
            sintg = sincos(tg[j], &costg);
            sintgph = sincos(tg[j] + phase, &costgph);
            ucos_ref = (double)(ww1[ii]) * tg[j] + wgwu *
               (-Ky * ddphi[ii] * sintg + Kx * ddpsi[ii] * sintgph +
                0.125/gamma[ii] * (Ky2*2*sintg*costg + Kx2*2*sintgph*costgph));

            sinucos = sincos(ucos_ref, &cosucos);
            eucos_ref.x = sinucos;
            eucos_ref.y = cosucos;

		  r.x = Ky / wu[ii] / gamma[ii] * sin(tg[j]);
            r.y = -Kx / wu[ii] / gamma[ii] * sin(tg[j] + phase);
            r.z = betam * tg[j] / wu[ii] - 0.125 / wu[ii] / gamma2 *
                  (Ky2 * sin(2 * tg[j]) + Kx2 * sin(2 * (tg[j] + phase)));

            beta_ref.x = Ky / gamma[ii] * cos(tg[j]);
            beta_ref.y = -Kx / gamma[ii] * cos(tg[j] + phase);


    		  //dir = dot(n, r);

            ucos_ref = w[ii] * (tg[j] / wu[ii] - dot(n, r));
            

			if (ii==0)
                {
                //printf("n: %2.12v3e\n", n);
			 //printf("[%i] r(int): %2.8v3e\n", j, traj2);
			 printf("[%i] r(int) %2.8v3e\n", j, traj2);
                printf("[%i] r(ref) %2.8v3e\n", j, r);
                //printf("[%i] dr %2.8v3e\n", j, traj2-r);
                printf("dir: %2.8e, dir_ref: %2.8e\n", dir, dot(n, r));
                //printf("1 - betam = %2.16e\n", 1. - betam);
                printf("ucos: %2.8e, ucos_ref: %2.8e, delta_ucos: %2.8e\n", ucos, ucos_ref, ucos-ucos_ref);
                //printf("1st: %2.8e, 1st_ref: %2.8e\n", w[ii] / wu[ii] * (1. - n.z*betam), ww1[ii]);
                //printf("2nd: %2.8e, 2nd_ref: %2.8e\n", betam * tg[j] / wu[ii],  wgwu * Ky * ddphi[ii] * sintg);
                
                printf("[%i] beta_int: %2.8v2e\n", j, beta);
                printf("[%i] beta_ref: %2.8v2e\n", j, beta_ref);
                }

*/
            eucos.x = cos(ucos);
            eucos.y = sin(ucos);
            Is += ag[j] * (ddphi[ii] - beta.x) * eucos;
            Ip += ag[j] * (ddpsi[ii] - beta.y) * eucos;
        }

        mem_fence(CLK_LOCAL_MEM_FENCE);

        Is_gl[ii] = Is;
        Ip_gl[ii] = Ip;
}