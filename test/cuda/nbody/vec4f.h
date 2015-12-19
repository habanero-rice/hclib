// A simple 4D vector with a variety of operators and other member functions 
// for doing 4D vector arithmetic. This code uses HEMI to be portable between
// standard host code (compiled with any C++ compiler, including NVCC) and 
// CUDA device code (compiled with NVCC).
#ifndef __VEC4F_H__
#define __VEC4F_H__

struct Vec4f
{
  float x, y, z, w;

  inline Vec4f() {};

  inline __host__ __device__ Vec4f(float xx, float yy, float zz, float ww) : x(xx), y(yy), z(zz), w(ww) {}

  inline Vec4f(const Vec4f& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

  inline __host__ __device__ Vec4f& operator=(const Vec4f& v) { 
    x = v.x; y = v.y; z = v.z; w = v.w; 
    return *this;
  }

  inline Vec4f operator+(const Vec4f& v) const { 
    return Vec4f(x+v.x, y+v.y, z+v.z, w+v.w); 
  }

  inline __host__ __device__ Vec4f operator-(const Vec4f& v) const {
    return Vec4f(x-v.x, y-v.y, z-v.z, w-v.w); 
  }

  inline Vec4f operator-() const {
    return Vec4f(-x, -y, -z, -w); 
  }

  inline __host__ __device__ Vec4f operator*(const Vec4f& v) const {
    return Vec4f(x*v.x, y*v.y, z*v.z, w*v.w); 
  }

  inline __host__ __device__ Vec4f operator*(float s) const {
    return Vec4f(x*s, y*s, z*s, w*s);
  }

  inline __host__ __device__ Vec4f& operator+=(const Vec4f& v) {
    x += v.x; y += v.y; z += v.z; w += v.w;
    return *this;
  }

  inline Vec4f& operator-=(const Vec4f& v) {
    x -= v.x; y -= v.y; z -= v.z; w -= v.w;
    return *this;
  }

  inline Vec4f& operator*=(const Vec4f& v) {
    x *= v.x; y *= v.y; z *= v.z; w *= v.w;
    return *this;
  }

  inline Vec4f& operator/=(const Vec4f& v) {
    x /= v.x; y /= v.y; z /= v.z; w /= v.w;
    return *this;
  }

  inline Vec4f& operator*=(float s) {
    x *= s; y *= s; z *= s; w *= s;
    return *this;
  }

  inline Vec4f& operator/=(float s) {
    x /= s; y /= s; z /= s; w /= s;
    return *this;
  }

  inline __host__ __device__ float dot(const Vec4f& v) const {
    return x*v.x + y*v.y + z*v.z + w*v.w;
  }

  inline __host__ __device__ float lengthSqr() const {
    return this->dot(*this);
  }

  inline float length() const {
    return sqrt(lengthSqr());
  }

  inline __host__ __device__ float inverseLength(float softening = 0.0f) const {
// #ifdef HEMI_DEV_CODE
//     return rsqrtf(lengthSqr() + softening);
// #else
    return 1.0f / sqrtf(lengthSqr() + softening);
// #endif
  }
};

#endif // __VEC4F_H__
