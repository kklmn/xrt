//__author__ = "Konstantin Klementiev", "Roman Chernikov"
//__date__ = "16 Sep 2014"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

__constant float2 cmp1 = (float2)(1,0);
__constant float2 cmpi1 = (float2)(0,1);
__constant float2 cmp0 = (float2)(0,0);
__constant float HALF = (float)0.5;


float2 r2cmp(float a)
  {
    return (float2)(a, 0);  
  } 

float2 conj_c(float2 a)
  {
    return (float2)(a.x, -a.y);
  }

float2 prod_c(float2 a, float2 b)
  {
    return (float2)(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
  }

float2 rec_c(float2 a)
  {
    return (float2)(a.x, -a.y) / (a.x * a.x + a.y * a.y);
  }

float abs_c(float2 a)
  {
    return sqrt(a.x * a.x + a.y * a.y);
  }

float abs_c2(float2 a)
  {
    return (a.x * a.x + a.y * a.y);
  }

float2 div_c(float2 a, float2 b)
  {
    return prod_c(a, conj_c(b)) / (b.x * b.x + b.y * b.y);       
  }

float2 sqrt_c(float2 a)
  {
    float phi = atan2(a.y, a.x);
    return (float2)(cos(HALF * phi), sin(HALF * phi)) * sqrt(length(a));
  }

float2 sqr_c(float2 a)
  {
      return prod_c(a,a);
  }

float2 exp_c(float2 a)
  {
    return (float2)(cos(a.y), sin(a.y)) * exp(a.x);
  }

float2 sin_c(float2 a)
  {
    return (float2)(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y));
  }  

float2 cos_c(float2 a)
  {
    return (float2)(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y));
  }    

float2 fasttan_c(float2 a)
  {
    float2 a2 = 2*a;
    return (float2)(sin(a2.x), sinh(a2.y)) / (cos(a2.x) + cosh(a2.y));
  }

float2 tan_c(float2 a)
  {
    return div_c(sin_c(a), cos_c(a));
  }

float2 dot_c4(float8 a, float8 b)
  {
    float2 p1, p2, p3, p4;
    p1 = prod_c((float2)(a.s0, a.s1), (float2)(b.s0, b.s1));
    p2 = prod_c((float2)(a.s2, a.s3), (float2)(b.s2, b.s3));
    p3 = prod_c((float2)(a.s4, a.s5), (float2)(b.s4, b.s5));
    p4 = prod_c((float2)(a.s6, a.s7), (float2)(b.s6, b.s7));
    return p1 + p2 + p3 + p4;
  }

