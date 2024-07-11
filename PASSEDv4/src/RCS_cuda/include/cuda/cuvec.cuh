/*
 * cuvec.cuh
 *
 *  Created on: Oct 16, 2014
 *      Author: cychiang
 */

#ifndef CUVEC_CUH_
#define CUVEC_CUH_

#include <iostream>
#include <iomanip>
#include "cuda.h"
#include <basic/vec.h>


namespace cu{

	void ChkErr(const char* msg){
		cudaError_t __err = cudaGetLastError();
		if (__err != cudaSuccess) {
//			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",
//					msg, cudaGetErrorString(__err), __FILE__, __LINE__);
			fprintf(stderr, "Fatal error: %s (%s)\n", msg, cudaGetErrorString(__err));
			fprintf(stderr, "*** FAILED - ABORTING ***\n");
			exit(1);
		}
	}

	// float
	__device__ const float PI=3.141592653589793;		// PI
	__device__ const float PI2=6.283185307179586;		// 2*PI
	__device__ const float PI4=12.566370614359172;		// 4*PI
	__device__ const float SQRT2=1.41421356237;			// square root of 2
	__device__ const float SQRT2_INV=0.707106781;		// inverse of square root of 2
	__device__ const float DTR=0.0174532925199;			// 1/180*PI
	__device__ const float RTD=57.2957795131;			// 180/PI
	__device__ const float C = 2.99792458e8;			// [m/s] light speed
	// double
	__device__ const double PId=3.141592653589793;		// PI
	__device__ const double PI2d=6.283185307179586;		// 2*PI
	__device__ const double PI4d=12.566370614359172;	// 4*PI
	__device__ const double SQRT2d=1.41421356237;		// square root of 2
	__device__ const double SQRT2_INVd=0.707106781;		// inverse of square root of 2
	__device__ const double DTRd=0.0174532925199;		// 1/180*PI
	__device__ const double RTDd=57.2957795131;			// 180/PI
	__device__ const double Cd = 2.99792458e8;			// [m/s] light speed


	//+-----------------------------------+
	//|              cuVEC                |
	//+-----------------------------------+
	template<typename T>
	class cuVEC {
	public:
		__host__ __device__ cuVEC():x(0),y(0),z(0){};
		__host__ __device__ cuVEC(const T x, const T y, const T z):x(x),y(y),z(z){};
		__host__ __device__ cuVEC(const cuVEC<T>& in){
			x = in.x;
			y = in.y;
			z = in.z;
		}
		template<typename T2> __host__ __device__ cuVEC(const cuVEC<T2>& in){
			x = (T)in.x;
			y = (T)in.y;
			z = (T)in.z;
		}
		__host__ __device__ cuVEC<T> operator-()const{
			return cuVEC<T>(-x,-y,-z);
		}
		__host__ __device__ cuVEC<T>& operator+=(const cuVEC<T>& R){
			x += R.x; y += R.y; z += R.z;
			return *this;
		}
		__host__ __device__ cuVEC<T>& operator+=(const T& R){
			x += R; y += R; z += R;
			return *this;
		}
		__host__ __device__ friend const cuVEC<T> operator+(const cuVEC<T>& L,const cuVEC<T>& R){
			return cuVEC<T>( L.x+R.x,L.y+R.y,L.z+R.z );
		}
		__host__ __device__ friend const cuVEC<T> operator-(const cuVEC<T>& L,const cuVEC<T>& R){
			return cuVEC<T>( L.x-R.x,L.y-R.y,L.z-R.z );
		}
		template<typename T2>
		__host__ __device__ friend const cuVEC<T> operator-(const cuVEC<T>& L,const cuVEC<T2>& R){
			return cuVEC<T>( L.x-R.x,L.y-R.y,L.z-R.z );
		}
		__host__ __device__ friend const cuVEC<T> operator*(const cuVEC<T>& L,const cuVEC<T>& R){
			return cuVEC<T>( L.x*R.x,L.y*R.y,L.z*R.z );
		}
		__host__ __device__ friend const cuVEC<T> operator*(const T& L, const cuVEC<T>& R){
			return cuVEC<T>( L*R.x,L*R.y,L*R.z );
		}
		__host__ __device__ friend const cuVEC<T> operator*(const cuVEC<T>& L, const T& R){
			return cuVEC<T>( L.x*R,L.y*R,L.z*R );
		}
		__host__ __device__ friend const cuVEC<T> operator/(const cuVEC<T>& L, const T& R){
			return cuVEC<T>( L.x/R,L.y/R,L.z/R );
		}
		__host__ __device__ T abs(){
			return sqrt(x*x+y*y+z*z);
		}
		__host__ __device__ T abs() const{
			return sqrt(x*x+y*y+z*z);
		}
		__device__ void Print(){
			printf("[%.10f,%.10f,%.10f]\n",x,y,z);
		}
		__device__ void Print()const{
			printf("[%.10f,%.10f,%.10f]\n",x,y,z);
		}
		__device__ void fromVEC(const cuVEC<T>& in){
			x = in.x;
			y = in.y;
			z = in.z;
		}
		// host converter
		__host__ void fromVEC(const vec::VEC<T>& in){
			x = in.x();
			y = in.y();
			z = in.z();
		}
		// CUDA memory
		__host__ void Create(const VEC<T>& h_vec, cuVEC<T>*& d_vec){
			cuVEC<T> tmp(h_vec.x(), h_vec.y(), h_vec.z());

			cudaMalloc(&d_vec, sizeof(cuVEC<T>));
			cudaMemcpy(d_vec, &tmp, sizeof(cuVEC<T>), cudaMemcpyHostToDevice);
		}
		__host__ void Free(cuVEC<T>*& d_vec){
			cudaFree(d_vec);
			//
			// Check Error
			//
			ChkErr("cu::cuEF::Free");
		}
	public:
		T x,y,z;
	};

	//
	// Misc. functions
	//
	template<typename T>
	__host__ __device__
	T abs(const cuVEC<T>& in){
		return sqrt(in.x*in.x+in.y*in.y+in.z*in.z);
	}

	template<typename T>
	__host__ __device__
	T dot(const cuVEC<T>& a,const cuVEC<T>& b){
		return (a.x*b.x + a.y*b.y + a.z*b.z);
	}

	__host__ __device__
	float dot(const cuVEC<double>& a,const cuVEC<float>& b){
		return (a.x*b.x + a.y*b.y + a.z*b.z);
	}

	__host__ __device__
	double dot(const double3& a,const cuVEC<double>& b){
		return (a.x*b.x + a.y*b.y + a.z*b.z);
	}

	template<typename T>
	__host__ __device__
	void dot(const cuVEC<T>& a,const cuVEC<T>& b, T& out){
		out = (a.x*b.x + a.y*b.y + a.z*b.z);
	}

	template<typename T>
	__host__ __device__
	cuVEC<T> cross(const cuVEC<T>& a,const cuVEC<T>& b){
		return cuVEC<T>(a.y*b.z - a.z*b.y, \
						a.z*b.x - a.x*b.z, \
						a.x*b.y - a.y*b.x);
	}

	__host__ __device__
	cuVEC<float> cross(const cuVEC<double>& a,const cuVEC<float>& b){
		return cuVEC<float>(a.y*b.z - a.z*b.y, \
							a.z*b.x - a.x*b.z, \
							a.x*b.y - a.y*b.x);
	}

	__device__
	cuVEC<float> Unit(const cuVEC<float>& a){
//		float inv = 1.f/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
//		double x = a.x;
//		double y = a.y;
//		double z = a.z;
//		float inv = rsqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
//		float inv = (float)rsqrt(x*x + y*y + z*z);
//		float inv = rsqrt(a.x*a.x + a.y*a.y + a.z*a.z);
		float inv = 1/a.abs();
		return cuVEC<float>(a.x*inv, a.y*inv, a.z*inv);
	}

	__device__
	cuVEC<double> Unit(const cuVEC<double>& a){
		double inv = 1/a.abs();
		return cuVEC<double>(a.x*inv, a.y*inv, a.z*inv);
	}

	__device__
	cuVEC<float> Min(const cuVEC<float>& a, const cuVEC<float>& b) {
		return cuVEC<float>( fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z) );
	}

	__device__
	cuVEC<float> Max(const cuVEC<float>& a, const cuVEC<float>& b) {
		return cuVEC<float>( fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z) );
	}

	template<typename T>
	__host__ __device__
	T Max3(const T a1, const T a2, const T a3){
		T out = a1;
		if(a2 > out){ out = a2; }
		if(a3 > out){ out = a3; }
		return out;
	}

	template<typename T>
	__host__ __device__
	T Min3(const T a1, const T a2, const T a3){
		T out = a1;
		if(a2 < out){ out = a2; }
		if(a3 < out){ out = a3; }
		return out;
	}

	template<typename T>
	__host__ __device__
	void swap(T& a, T& b){
		T tmp = a;
		a = b;
		b = tmp;
	}

	template<typename T>
	__device__
	T angle(const cuVEC<T>& a,const cuVEC<T>& b){
		T a_dot_b = dot(a,b);
		T ab = a.abs()*b.abs();
//		return acos(a_dot_b/ab);
		double cos_val=a_dot_b/ab;

		if(fabs(a_dot_b - ab) < 1E-14){
			return (T)0;
		}

		if(fabs(cos_val-1) < 1E-14){
			return (T)0;
		}else{
			return acos(cos_val);
		}
	}

	template<typename T>
	__device__
	T Norm2(const cuVEC<T>& Vec){
		return std::sqrt(Vec.x*Vec.x + Vec.y*Vec.y + Vec.z*Vec.z);
	}

	template<typename T>
	__device__
	cuVEC<T> ProjectOnPlane(const cuVEC<T>& Vec, const cuVEC<T>& N){
		// Vec: Arbitrary vector
		// N  : Normal vector
		cuVEC<T> Vec_proj = Vec - dot(Vec, N)/Norm2(N) * N;
		return Unit(Vec_proj);
	}

	template<typename T>
	__device__
	cuVEC<T> Unit3DNonZeroVector(const cuVEC<T>& vectorMatrix){
		double eps = 1e-16;

		// unitLengthVec3D is what the function returns
		double EuclideanNorm3D =  Norm2(vectorMatrix);
		cuVEC<T> unitLengthVec3D;
		if(EuclideanNorm3D <= eps) {    // Ensure vector does not have zero length
			printf("Found a vector of length zero; cannot continue processing.\n");
		} else {
			unitLengthVec3D = vectorMatrix / EuclideanNorm3D;
		}

		return unitLengthVec3D;
	}

	template<typename T>
	__device__
	T SignedAngleTwo3DVectors(const cuVEC<T>& startVec, const cuVEC<T>& endVec, const cuVEC<T>& rotateVec){
		double eps = 1e-16;

		if( std::abs(dot(rotateVec, cross(startVec, endVec))) < (2.5 * eps) ){
//			printf("WARNING: rotateVec is nearly coplanar with startVec and endVec.\n");
			return 0;
		}

		cuVEC<T> temp = cross(startVec, endVec); // temp is normal to both startVec & endVec

		if( dot(rotateVec, temp) < 0){
			temp = -temp;
		}

		// Get normal vector
		cuVEC<T> rotateVec2;

		if( (std::abs(temp.x) + std::abs(temp.y) + std::abs(temp.z)) > eps ) {
			rotateVec2 = Unit3DNonZeroVector(temp);
		}else{
			rotateVec2 = rotateVec;
		}

		// Numerator of tan(beta) = sin(beta)
		double sineBeta = dot(rotateVec2, cross(startVec, endVec));

		// Denominator of tan(beta) = cos(beta)
		double cosineBeta = dot(startVec, endVec);

		// Two-argument (four-quadrant) arc-tangent takes sine & cosine as arguments
		double beta = atan2(sineBeta, cosineBeta);

		return beta;
	}

	template<typename T>
	__device__
	bool CheckEffectiveDiffractionIncident(const cuVEC<T>& e1_x, const cuVEC<T>& e1_z, const cuVEC<T>& e2_x, const cuVEC<T>& sp, const size_t k) {
		// e1_x: Edge x component (on the plate for facet-1)
		// e1_z: Edge z component (along the edge for facet-1)
		// e2_x: Edge x component (on the plate for facet-2)
		// sp: Incident unit vector
		// SHOW: Display results or not? (default = false)

		T ang1 = SignedAngleTwo3DVectors(e1_x, -sp, e1_z);
		T ang2 = SignedAngleTwo3DVectors(e1_x, e2_x, e1_z);
//		double ang10 = ang1;
//		double ang20 = ang2;

		if (ang1 < 0) {
			ang1 = 360./180.*cu::PId + ang1;
		}

		if (ang2 < 0) {
			ang2 = 360./180.*cu::PId + ang2;
		}

//		double WA = 360./180.*cu::PI - ang2;


//		// DEBUG (START) ========================================================================================================================================================================
//		if(k==60602) {
//			printf("\n\n\n\n>>>> GPU >>>>\ne1_x=(%f,%f,%f), e1_z=(%f,%f,%f), e2_x=(%f,%f,%f), sp=(%f,%f,%f), ang10=%.10f, ang1=%.10f, ang20=%.10f, ang2=%.10f\n>>>>>>>>>>>>>\n\n",
//				   e1_x.x, e1_x.y, e1_x.z, e1_z.x, e1_z.y, e1_z.z, e2_x.x, e2_x.y, e2_x.z, sp.x, sp.y, sp.z, ang10, ang1, ang20, ang2);
//		}
//		// DEBUG (END) ==========================================================================================================================================================================


		if (ang2 > ang1) {
			return true;
		} else {
			return false;
		}
	}
};


#endif /* CUVEC_CUH_ */
