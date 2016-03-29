//__author__ = "Konstantin Klementiev", "Roman Chernikov"
//__date__ = "16 Sep 2014"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

__constant double2 cmp1 = (double2)(1,0);
__constant double2 cmpi1 = (double2)(0,1);
__constant double2 cmp0 = (double2)(0,0);

double2 r2cmp(double a)
  {
    return (double2)(a, 0);  
  } 

double2 conj_c(double2 a)
  {
    return (double2)(a.x, -a.y);
  }

double2 prod_c(double2 a, double2 b)
  {
    return (double2)(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
  }

double2 rec_c(double2 a)
  {
    return (double2)(a.x, -a.y) / (a.x * a.x + a.y * a.y);
  }

double abs_c(double2 a)
  {
    return sqrt(a.x * a.x + a.y * a.y);
  }

double abs_c2(double2 a)
  {
    return (a.x * a.x + a.y * a.y);
  }

double2 div_c(double2 a, double2 b)
  {
    return prod_c(a, conj_c(b)) / (b.x * b.x + b.y * b.y);       
  }

double2 sqrt_c(double2 a)
  {
    double phi = atan2(a.y, a.x);
    return (double2)(cos(0.5 * phi), sin(0.5 * phi)) * sqrt(length(a));
  }

double2 sqr_c(double2 a)
  {
      return prod_c(a,a);
  }

double2 exp_c(double2 a)
  {
    return (double2)(cos(a.y), sin(a.y)) * exp(a.x);
  }

double2 sin_c(double2 a)
  {
    return (double2)(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y));
  }  

double2 cos_c(double2 a)
  {
    return (double2)(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y));
  }    

double2 fasttan_c(double2 a)
  {
    double2 a2 = 2*a;
    return (double2)(sin(a2.x), sinh(a2.y)) / (cos(a2.x) + cosh(a2.y));
  }

double2 tan_c(double2 a)
  {
    return div_c(sin_c(a), cos_c(a));
  }

double2 dot_c4(double8 a, double8 b)
  {
    double2 p1, p2, p3, p4;
    p1 = prod_c((double2)(a.s0, a.s1), (double2)(b.s0, b.s1));
    p2 = prod_c((double2)(a.s2, a.s3), (double2)(b.s2, b.s3));
    p3 = prod_c((double2)(a.s4, a.s5), (double2)(b.s4, b.s5));
    p4 = prod_c((double2)(a.s6, a.s7), (double2)(b.s6, b.s7));
    return p1 + p2 + p3 + p4;
  }
