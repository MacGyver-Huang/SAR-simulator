/*
 * cu.cuh
 *
 *  Created on: Oct 14, 2014
 *      Author: cychiang
 */

#ifndef CUCLASS_CUH_
#define CUCLASS_CUH_


//#define MAXLEVEL 5


#include <iostream>
#include <iomanip>
#include "cuda.h"
#include <sar/sar.h>
#include <basic/vec.h>
#include <bvh/triangle.h>
#include <bvh/bvh.h>
#include <mesh/mesh.h>
#include <cuda/cuvec.cuh>
#include <cuda/cuopt.cuh>



using namespace std;
using namespace vec;
using namespace mesh;



namespace cu {
	//+-----------------------------------+
	//|          cuBVHFlatNode            |
	//+-----------------------------------+
	class cuRay {
	public:
		__host__ __device__ cuRay(){};
		__host__ __device__ cuRay(const cuVEC<double>& O, const cuVEC<double>& D):o(O),d(D){
			inv_d.x = 1.0/d.x;
			inv_d.y = 1.0/d.y;
			inv_d.z = 1.0/d.z;
		}
		// Mics.
		__device__ cuRay Reflection(const cuVEC<float>& hitPoint, const cuVEC<float>& N)const{

//			cuVEC<double> d2((double)d.x, (double)d.y, (double)d.z);
			cuVEC<double> N2 = N;
			cuVEC<double> uv = d - 2.0*dot(d, N2)*N2;

//			printf("uv.abs() = %.20f\n", uv.abs());


//			cuVEC<float> uv = Unit(d - 2.f*cu::dot(d, Normal)*Normal);
			return cuRay(hitPoint, uv);
		}
		__host__ void Create(const Ray& h_ray, cuRay*& d_ray){
			cuVEC<double> h_o( h_ray.o.x(), h_ray.o.y(), h_ray.o.z() );
			cuVEC<double> h_d( h_ray.d.x(), h_ray.d.y(), h_ray.d.z() );

			cuRay tmp(h_o, h_d);

			cudaMalloc(&d_ray, sizeof(cuRay));
			cudaMemcpy(d_ray, &tmp, sizeof(cuRay), cudaMemcpyHostToDevice);

			//
			// Check Error
			//
			ChkErr("cu::cuEF::Create");
		}
		__host__ void Free(cuRay*& d_ray){
			cudaFree(d_ray);
			//
			// Check Error
			//
			ChkErr("cu::cuEF::Free");
		}
		__device__ void Print()const{
			printf("+----------------+\n");
			printf("|  cuRay Class   |\n");
			printf("+----------------+\n");
			printf(" o     = "); o.Print();
			printf(" d     = "); d.Print();
			printf(" inv_d = "); inv_d.Print();
		}
		// host converter
		__host__ void fromRay(const Ray& in){
			o.fromVEC(in.o);
			d.fromVEC(in.d);
			inv_d.fromVEC(in.inv_d);
		}
	public:
		cuVEC<double> o; // Ray Origin
		cuVEC<double> d; // Ray Direction
		cuVEC<double> inv_d; // Inverse of each Ray Direction component
	};
	//+-----------------------------------+
	//|          cuBVHFlatNode            |
	//+-----------------------------------+
	class cuBBox {
	public:
		__host__ __device__ cuBBox(){};
		__host__ __device__ cuBBox(const cuVEC<float>& min, const cuVEC<float>& max):min(min), max(max){
			extent = max - min;
		}
		__host__ __device__ cuBBox(const cuVEC<float>& p):min(p),max(p){
			extent = max - min;
		}
		// Mics.
		__device__ float Max3(const float a1, const float a2, const float a3)const{
//			float out = a1;
//			if(a2 > out){ out = a2; }
//			if(a3 > out){ out = a3; }
//			return out;
			return fmaxf(fmaxf(a1,a2),a3);
		}
		__device__ float Min3(const float a1, const float a2, const float a3)const{
//			float out = a1;
//			if(a2 < out){ out = a2; }
//			if(a3 < out){ out = a3; }
//			return out;
			return fminf(fminf(a1,a2),a3);
		}
		__device__ bool intersect(const cuRay& ray, float *tnear, float *tfar) const{
			float xb, yb, zb;
			float xt, yt, zt;
			xb = (min.x - ray.o.x) * ray.inv_d.x;
			yb = (min.y - ray.o.y) * ray.inv_d.y;
			zb = (min.z - ray.o.z) * ray.inv_d.z;

			xt = (max.x - ray.o.x) * ray.inv_d.x;
			yt = (max.y - ray.o.y) * ray.inv_d.y;
			zt = (max.z - ray.o.z) * ray.inv_d.z;

		//	VEC<float> inv_d(1.0/ray.d.x(), 1.0/ray.d.y(), 1.0/ray.d.z());
		//
		//	xb = (min.x() - ray.o.x()) * inv_d.x();
		//	yb = (min.y() - ray.o.y()) * inv_d.y();
		//	zb = (min.z() - ray.o.z()) * inv_d.z();
		//
		//	xt = (max.x() - ray.o.x()) * inv_d.x();
		//	yt = (max.y() - ray.o.y()) * inv_d.y();
		//	zt = (max.z() - ray.o.z()) * inv_d.z();



//			float xmin = (xb < xt)? xb : xt;
//			float ymin = (yb < yt)? yb : yt;
//			float zmin = (zb < zt)? zb : zt;
//
//			float xmax = (xb > xt)? xb : xt;
//			float ymax = (yb > yt)? yb : yt;
//			float zmax = (zb > zt)? zb : zt;
//
//			*tnear = Max3(xmin, ymin, zmin);
//			*tfar  = Min3(xmax, ymax, zmax);

			*tnear = Max3(fminf(xb,xt), fminf(yb,yt), fminf(zb,zt));
			*tfar  = Min3(fmaxf(xb,xt), fmaxf(yb,yt), fmaxf(zb,zt));


		//	VEC<float> tbot = ray.inv_d * (min - ray.o);
		//	VEC<float> ttop = ray.inv_d * (max - ray.o);
		//
		//	VEC<float> tmin = mat::min(ttop, tbot);
		//	VEC<float> tmax = mat::max(ttop, tbot);
		//
		//	*tnear = Max3(tmin.x(), tmin.y(), tmin.z());
		//	*tfar  = Min3(tmax.x(), tmax.y(), tmax.z());

			return !(*tnear > *tfar) && *tfar > 0;

//			float xb, yb, zb;
//			float xt, yt, zt;
//			xb = (min.x - ray.o.x) * ray.inv_d.x;
//			yb = (min.y - ray.o.y) * ray.inv_d.y;
//			zb = (min.z - ray.o.z) * ray.inv_d.z;
//
//			xt = (max.x - ray.o.x) * ray.inv_d.x;
//			yt = (max.y - ray.o.y) * ray.inv_d.y;
//			zt = (max.z - ray.o.z) * ray.inv_d.z;
//
//
//			float xmin = fminf(xb, xt);
//			float ymin = fminf(yb, yt);
//			float zmin = fminf(zb, zt);
//
//			float xmax = fmaxf(xb, xt);
//			float ymax = fmaxf(yb, yt);
//			float zmax = fmaxf(zb, zt);
//
//			*tnear = fmaxf(fmaxf(xmin, ymin), zmin);
//			*tfar  = fminf(fminf(xmax, ymax), zmax);
//
////			float xmin = (xb < xt)? xb : xt;
////			float ymin = (yb < yt)? yb : yt;
////			float zmin = (zb < zt)? zb : zt;
////
////			float xmax = (xb > xt)? xb : xt;
////			float ymax = (yb > yt)? yb : yt;
////			float zmax = (zb > zt)? zb : zt;
////
////			*tnear = Max3(xmin, ymin, zmin);
////			*tfar  = Min3(xmax, ymax, zmax);
//
//			return !(*tnear > *tfar) && *tfar > 0;
		}
		__device__ void expandToInclude(const cuVEC<float>& p){
			min = cu::Min(min, p);
			max = cu::Max(max, p);
			extent = max - min;
		}
		__device__ void expandToInclude(const cuBBox& b){
			min = cu::Min(min, b.min);
			max = cu::Max(max, b.max);
			extent = max - min;
		}
		__host__ __device__ size_t maxDimension() const{
			size_t result = 0;
			if(extent.y > extent.x) result = 1;
			if(extent.z > extent.y) result = 2;
			return result;
		}
		__host__ __device__ float surfaceArea() const{
			return 2.f*( extent.x*extent.z + extent.x*extent.y + extent.y*extent.z );
		}
		// host to host converter
		__host__ void fromBBox(const BBox& in){
			extent.fromVEC(in.extent);
			min.fromVEC(in.min);
			max.fromVEC(in.max);
		}
	public:
		cuVEC<float> min, max, extent;
	};
	//+-----------------------------------+
	//|       cuIntersectionInfo          |
	//+-----------------------------------+
	template<typename T> class cuTRI;
	class cuIntersectionInfo {
	public:
		double t; 					// Intersection distance along the ray
		const cuTRI<float>* object;	// "Object pointer" that was hit
		cuVEC<float> hit;			// Location of the intersection
	};
	//+-----------------------------------+
	//|          cuTRI                    | << double
	//+-----------------------------------+
	template<typename T>
	class cuTRI {
	public:
		__host__ __device__ cuTRI():Area(),ca(),sa(),cb(),sb(),MatIdx(){};
		__device__ cuTRI(const cuVEC<T> v0, const cuVEC<T> v1, const cuVEC<T> v2, const long MatIDX){
			// Calculate
			cuVEC<double> vv0(v0.x, v0.y, v0.z);
			cuVEC<double> vv1(v1.x, v1.y, v1.z);
			cuVEC<double> vv2(v2.x, v2.y, v2.z);
			// Normal vector
			cuVEC<double> N2  = cu::Unit(cu::cross(vv1 - vv0, vv2 - vv0));
			// Center
			cuVEC<double> CC2 = (vv0 + vv1 + vv2)/3.0;

			// Assign values
			V0 = v0;
			V1 = v1;
			V2 = v2;
//			// Edge vector
//			P0 = v0 - v2;
//			P1 = v1 - v2;
//			P2 = cuVEC<T>(0,0,0);
			// Normal vector
			N  = cuVEC<T>(N2.x, N2.y, N2.z);
//			N  = Unit(cross(V1 - V0, V2 - V0));
			// Center
			CC = cuVEC<T>(CC2.x, CC2.y, CC2.z);
//			CC = (V0 + V1 + V2)/3;
			// Angle
			float beta  = acos(N2.z);		// 0 < beta < 180
			float alpha = atan2(N2.y,N2.x);	// -180 < alpha < 180
			// Area
			cuVEC<double> vv1_vv0 = vv1-vv0;
			cuVEC<double> vv2_vv1 = vv2-vv1;
			cuVEC<double> vv0_vv2 = vv0-vv2;
			cuVEC<double> EdgeLength( vv1_vv0.abs(), vv2_vv1.abs(), vv0_vv2.abs() );
//			cuVEC<double> EdgeLength( (V1-V0).abs(), (V2-V1).abs(), (V0-V2).abs() );
			double s = (EdgeLength.x + EdgeLength.y + EdgeLength.z)/2;
			Area = sqrt( s*(s-EdgeLength.x)*(s-EdgeLength.y)*(s-EdgeLength.z) );
			// Edge vector
			P0 =  vv0_vv2;
			P1 = -vv2_vv1;
			// Misc.
//			ca = cos(alpha);
//			sa = sin(alpha);
//			cb = cos(beta);
//			sb = sin(beta);
			sincosf(alpha, &sa, &ca);
			sincosf(beta,  &sb, &cb);
			// Material index
			MatIdx = MatIDX;
		}
		/**
		 * Constructor by giving all memeber variables
		 * @param[in]   V0   [m,m,m] 1st vertice position
		 * @param[in]   V1   [m,m,m] 2nd vertice position
		 * @param[in]   V2   [m,m,m] 3rd vertice position
		 * @param[in]   ea   [rad] All edge wedge angle (follow edge : {v0->v1},{v1->v2},{v2->v0})
		 * @param[in]   idx  [x] Self triangle index w.r.t. cad._PL
		 * @param[in]   idx_near [x] All near triangle index w.r.t. cad._PL (follow edge : {v0->v1},{v1->v2},{v2->v0})
		 */
		__device__ cuTRI(const cuVEC<T>& v0, const cuVEC<T>& v1, const cuVEC<T>& v2,
						 const long MatIDX, const double Ea[3], const size_t& Idx, const long Idx_near[3]){
			// Calculate
			cuVEC<double> vv0(v0.x, v0.y, v0.z);
			cuVEC<double> vv1(v1.x, v1.y, v1.z);
			cuVEC<double> vv2(v2.x, v2.y, v2.z);

			ea[0] = Ea[0];
			ea[1] = Ea[1];
			ea[2] = Ea[2];
			V0    = v0;
			V1    = v1;
			V2    = v2;
			// Normal vector
			cuVEC<double> N2  = cu::Unit(cu::cross(vv1 - vv0, vv2 - vv0));
			N  = cuVEC<T>(N2.x, N2.y, N2.z);
			// Center
			cuVEC<double> CC2 = (vv0 + vv1 + vv2)/3.0;
			CC = cuVEC<T>(CC2.x, CC2.y, CC2.z);
			// Additional of PTD
			idx   = Idx;
			idx_near[0] = Idx_near[0];
			idx_near[1] = Idx_near[1];
			idx_near[2] = Idx_near[2];
			MatIdx = MatIDX;
		}
		// hierarchy member function
		__device__ bool getIntersection(const cuRay& ray, cuIntersectionInfo* I) const{
//		__device__ bool getIntersection(const cuRay& ray, cuIntersectionInfo& I, long k) const{
//#ifndef EPSILON
//#define EPSILON 0.000001
//#endif
			//Find vectors for two edges sharing V1
			cuVEC<double> e1((double)V1.x - (double)V0.x, (double)V1.y - (double)V0.y, (double)V1.z - (double)V0.z);	// Edge 1
			cuVEC<double> e2((double)V2.x - (double)V0.x, (double)V2.y - (double)V0.y, (double)V2.z - (double)V0.z);	// Edge 2
//			cuVEC<double> ray_d((double)ray.d.x, (double)ray.d.y, (double)ray.d.z);
//			cuVEC<double> ray_o((double)ray.o.x, (double)ray.o.y, (double)ray.o.z);

			//Begin calculating determinant - also used to calculate u parameter
			cuVEC<double> P = cu::cross(ray.d, e2);
			//if determinant is near zero, ray lies in plane of triangle
			double det = cu::dot(e1, P);
			//NOT CULLING
//			if(det > -EPSILON && det < EPSILON) return false;
			if(det > cu::opt::c_EPSILON[1] && det < cu::opt::c_EPSILON[0]) return false;
//			if(fabs(det) < EPSILON) return false;
			double inv_det = 1.0 / det;

			//calculate distance from V0 to ray origin
			cuVEC<double> TT = ray.o - cuVEC<double>(V0);

			//Calculate u parameter and test bound
			float u = cu::dot(TT, P) * inv_det;
			//The intersection lies outside of the triangle
			if(u < 0.0|| u > cu::opt::c_ONE[0]) return false;

			//Prepare to test v parameter
			cuVEC<double> Q = cu::cross(TT, e1);

			//Calculate V parameter and test bound
			float v = cu::dot(ray.d, Q) * inv_det;

			//The intersection lies outside of the triangle
			if(v < 0.0 || u + v  > cu::opt::c_ONE[0]) return false;

			double t = cu::dot(e2, Q) * inv_det;	// Ray distance

			if(t <= cu::opt::c_EPSILON[0]) return false; // No hit, no win

//			if(t > EPSILON) { //ray intersection
				I->object = this;
				I->t = t;
//				I.hit = ray.o + (ray.d * t);
				I->hit.x = float( fma(ray.d.x, t, ray.o.x) );
				I->hit.y = float( fma(ray.d.y, t, ray.o.y) );
				I->hit.z = float( fma(ray.d.z, t, ray.o.z) );
//				I->hit.x = float(ray.o.x + (ray.d.x * t));
//				I->hit.y = float(ray.o.y + (ray.d.y * t));
//				I->hit.z = float(ray.o.z + (ray.d.z * t));
//				I.hit.x = float((double)ray.o.x + ((double)ray.d.x * (double)t));
//				I.hit.y = float((double)ray.o.y + ((double)ray.d.y * (double)t));
//				I.hit.z = float((double)ray.o.z + ((double)ray.d.z * (double)t));


//				if(k == 24920){
//					printf("%\n");
//					printf("cuTRI::getIntersection\n");
//					printf("e1      = "); e1.Print();
//					printf("e2      = "); e2.Print();
//					printf("EPSILON = %.10f\n", EPSILON);
//					printf("P       = "); P.Print();
//					printf("det     = %.10f\n", det);
//					printf("inv_det = %.10f\n", inv_det);
//					printf("TT      = "); TT.Print();
//					printf("u       = %.10f\n", u);
//					printf("Q       = "); Q.Print();
//					printf("v       = %.10f\n", v);
//					printf("t       = %.10f\n", t);
//					printf("I.t     = %.10f\n", I.t);
//					printf("ray.o   = (%.10f,%.10f,%.10f)\n", ray.o.x, ray.o.y, ray.o.z);
//					printf("ray.d   = (%.10f,%.10f,%.10f)\n", ray.d.x, ray.d.y, ray.d.z);
//					printf("I.hit   = (%.10f,%.10f,%.10f)\n", I.hit.x, I.hit.y, I.hit.z);
//					printf("%\n");
//				}



				return true;
//			}

//			// No hit, no win
//			return false;
		}
		__host__ __device__ cuVEC<float> getNormal() const{
			return N;
		}
		__host__ __device__ cuVEC<double> getNormalDouble() const{
			return cuVEC<double>(N.x, N.y, N.z);
		}
		// getter / setter
		/**
		 * Get self polygon index (editable)
		 * @return Return polygon index
		 */
//		__host__ __device__ size_t& IDX(){ return idx; }
		/**
		 * Get self polygon index
		 * @return Return polygon index
		 */
		__host__ __device__ const size_t& IDX()const{ return idx; }
		/**
		 * Get polygon index by giving index, idx<=3. (editable)
		 * @return Return polygon index
		 */
//		__host__ __device__ long& IDX_Near(const long idx){ return idx_near[idx]; }
		/**
		 * Get polygon index by giving index, idx<=3.
		 * @param[in]   idx   Polygon index
		 * @return Return polygon index
		 */
		__host__ __device__ const long& IDX_Near(const long idx)const{ return idx_near[idx]; }
		__host__ __device__ cuBBox getBBox() const{
			T x_min = fminf(fminf(V0.x, V1.x), V2.x);
			T y_min = fminf(fminf(V0.y, V1.y), V2.y);
			T z_min = fminf(fminf(V0.z, V1.z), V2.z);

			T x_max = fmaxf(fmaxf(V0.x, V1.x), V2.x);
			T y_max = fmaxf(fmaxf(V0.y, V1.y), V2.y);
			T z_max = fmaxf(fmaxf(V0.z, V1.z), V2.z);

			return cuBBox(cuVEC<float>(x_min,y_min,z_min), cuVEC<float>(x_max,y_max,z_max));
		}
		cuVEC<T> getCentroid()const{
			return CC;
		}
		__host__ __device__ void Print(){
			printf("+---------------------+\n");
			printf("| Triangle Class (cu) |\n");
			printf("+---------------------+\n");
			printf(" V0    = "); V0.Print();
			printf(" V1    = "); V1.Print();
			printf(" V2    = "); V2.Print();
			printf(" CC    = "); CC.Print();
			printf(" N     = "); N.Print();
			printf(" Area  = %f\n",Area);
//			printf(" alpha = %f\n",alpha);
//			printf(" beta  = %f\n",beta);
			// Additional for PTD
			T deg0 = (ea[0] == -1)? -1:(ea[0]*RTD);
			T deg1 = (ea[1] == -1)? -1:(ea[1]*RTD);
			T deg2 = (ea[2] == -1)? -1:(ea[2]*RTD);
			printf(" EA0   = %f [deg] -> factor = %f\n", deg0, 2-ea[0]/PI);
			printf(" EA1   = %f [deg] -> factor = %f\n", deg1, 2-ea[1]/PI);
			printf(" EA2   = %f [deg] -> factor = %f\n", deg2, 2-ea[2]/PI);
			printf(" IDX           = %ld\n", idx);
			printf(" IDX0 (v0->v1) = %ld\n", idx_near[0]);
			printf(" IDX1 (v1->v2) = %ld\n", idx_near[1]);
			printf(" IDX2 (v2->v0) = %ld\n", idx_near[2]);
		}
		__device__ void Print()const{
			printf("+---------------------+\n");
			printf("| Triangle Class (cu) |\n");
			printf("+---------------------+\n");
			printf(" V0    = "); V0.Print();
			printf(" V1    = "); V1.Print();
			printf(" V2    = "); V2.Print();
			printf(" CC    = "); CC.Print();
			printf(" N     = "); N.Print();
			printf(" Area  = %.10f\n",Area);
//			printf(" alpha = %.10f\n",alpha);
//			printf(" beta  = %.10f\n",beta);
			// Additional for PTD
			T deg0 = (ea[0] == -1)? -1:(ea[0]*RTD);
			T deg1 = (ea[1] == -1)? -1:(ea[1]*RTD);
			T deg2 = (ea[2] == -1)? -1:(ea[2]*RTD);
			printf(" EA0   = %f [deg] -> factor = %f\n", deg0, 2-ea[0]/PI);
			printf(" EA1   = %f [deg] -> factor = %f\n", deg1, 2-ea[1]/PI);
			printf(" EA2   = %f [deg] -> factor = %f\n", deg2, 2-ea[2]/PI);
			printf(" IDX           = %ld\n", idx);
			printf(" IDX0 (v0->v1) = %ld\n", idx_near[0]);
			printf(" IDX1 (v1->v2) = %ld\n", idx_near[1]);
			printf(" IDX2 (v2->v0) = %ld\n", idx_near[2]);
		}
		// host to host converter
		__host__ void fromTRI(const TRI<T>& in){
			V0.fromVEC(in.V0());	// Vertex
			V1.fromVEC(in.V1());	//
			V2.fromVEC(in.V2());	//
			P0.fromVEC(in.V0()-in.V2());	// Edge vector P0=v0-v2, P1=v1-v2, P2=[0,0,0]
			P1.fromVEC(in.V1()-in.V2());	//
//			P2.fromVEC(in.p2());	//
			CC.fromVEC(in.getCentroid());	// Central position
			N.fromVEC(in.getNormal());		// Normal vector

//			Area = in.AREA();		// Area
////			alpha = in.ALPHA();		// alpha
////			beta = in.BETA();		// beta
//			ca = in.CA();			// cos(alpha)
//			sa = in.SA();			// sin(alpha)
//			cb = in.CB();			// cos(beta)
//			sb = in.SB();			// sin(beta)

			// Area
			VEC<double> EdgeLength( (in.V1()-in.V0()).abs(), (in.V2()-in.V1()).abs(), (in.V0()-in.V2()).abs() );
			double s = (EdgeLength.x() + EdgeLength.y() + EdgeLength.z())/2;
			Area = sqrt( s*(s-EdgeLength.x())*(s-EdgeLength.y())*(s-EdgeLength.z()) );
			// Normal vector
			VEC<double> N2  = Unit(cross(in.V1()-in.V0(), in.V2()-in.V0()));
			// Angle (double)
			double beta  = acos(N2.z());			// 0 < beta < 180
			double alpha = atan2(N2.y(),N2.x());	// -180 < alpha < 180
			// cos & sin of alpha & beta
			ca = cos(alpha);
			sa = sin(alpha);
			cb = cos(beta);
			sb = sin(beta);

			MatIdx = in.MatIDX();	// Material index

			// Additional for PTD
			idx = in.IDX();
			ea[0] = in.EA(0);
			ea[1] = in.EA(1);
			ea[2] = in.EA(2);
			idx_near[0] = in.IDX_Near(0);
			idx_near[1] = in.IDX_Near(1);
			idx_near[2] = in.IDX_Near(2);
		}
		template<typename T2>
		__host__ __device__ void fromTRI(const cuTRI<T2>& in){
			V0.fromVEC(in.V0);	// Vertex
			V1.fromVEC(in.V1);	//
			V2.fromVEC(in.V2);	//
			P0.fromVEC(in.V0-in.V2);	// Edge vector P0=v0-v2, P1=v1-v2, P2=[0,0,0]
			P1.fromVEC(in.V1-in.V2);	//
			CC.fromVEC(in.CC);	// Central position
			N.fromVEC(in.N);		// Normal vector

			// Area
			Area = in.Area;
			ca = in.ca;
			sa = in.sa;
			cb = in.cb;
			sb = in.sb;

			MatIdx = in.MatIdx;	// Material index

			// Additional for PTD
			idx = in.idx;
			ea[0] = in.ea[0];
			ea[1] = in.ea[1];
			ea[2] = in.ea[2];
			idx_near[0] = in.idx_near[0];
			idx_near[1] = in.idx_near[1];
			idx_near[2] = in.idx_near[2];
		}
		/**
		 * Check the input triangle is equal or not?
		 * @param[in]   tri_in   (TRI class) input triangle class
		 * @return Return true if they are equal.
		 */
		template<typename T2>
		__host__ __device__ bool Equal(const cuTRI<T2>& tri_in){
			T sum = abs(V0 - tri_in.V0) + abs(V1 - tri_in.V1) + abs(V2 - tri_in.V2);
			return (sum < 1E-15)? true:false;
		}
	public:
		cuVEC<T> V0, V1, V2;	// Vertex
		cuVEC<T> P0, P1;		// Edge vector P0=v0-v2, P1=v1-v2, P2=[0,0,0]
		cuVEC<T> CC;			// Central position
		cuVEC<T> N;				// Normal vector
		float Area;				// Area
//		double alpha;			// alpha
//		double beta;			// beta
		float ca, sa, cb, sb;	// cos(alpha), sin(alpha), cos(beta), sin(beta)
		long MatIdx;			// Material index
		// Additional for PTD
		double ea[3];			// Wedge angle, ea0 = {v0->v1}, ea1 = {v1->v2}, ea2 = {v2->v0}
		size_t idx;				// Self triangle index w.r.t. cad._PL
		long idx_near[3];		// Near 3 triangle index w.r.t. cad._PL
	};
	//+-----------------------------------+
	//|          cuBVHFlatNode            |
	//+-----------------------------------+
	class cuBVHFlatNode {
	public:
		// host to host converter
		__host__ void fromBVHFlatNode(const BVHFlatNode& in){
			nPrims = in.nPrims;
			rightOffset = in.rightOffset;
			start = in.start;
			// cuBBox
			bbox.fromBBox(in.bbox);
		}
	public:
		cuBBox bbox;
		size_t start, nPrims, rightOffset;
	};
	//+-----------------------------------+
	//|          cuBVHTraversal           |
	//+-----------------------------------+
	class cuBVHTraversal {
	public:
		__host__ __device__ cuBVHTraversal():
		i(),mint(){};
		__host__ __device__ cuBVHTraversal(int i, float mint):i(i),mint(mint){};
	public:
		size_t i; 	// Node
		float mint; // Minimum hit time for this node.
	};
	//+-----------------------------------+
	//|              cuBVH                |
	//+-----------------------------------+
	class cuBVH {
	public:
		__host__ __device__ cuBVH():
		nNodes(),nLeafs(),leafSize(),build_primes(NULL),flatTree(NULL){};
		__host__ __device__ ~cuBVH(){};
		__host__ void Create(const BVH& bvh, cuBVH*& d_bvh, cuTRI<float>*& d_primes, size_t*& d_idx_poly, cuBVHFlatNode*& d_flatTree){
			//
			// Host
			//
			// 1. Make h_primes (d_primes)
			size_t n_primes = bvh.build_prims->size();
			cuTRI<float>* h_primes = new cuTRI<float>[n_primes];
			for(size_t i=0;i<n_primes;++i){
				// Obj* to TRI<float>*
				TRI<float>* tmp = (TRI<float>*)((*(bvh.build_prims))[i]);
				// assign
				h_primes[i].fromTRI(*tmp);
			}
			// 2. Make h_idx_poly (d_idx_poly)
			size_t* h_idx_poly = new size_t[n_primes];
			vector<size_t> idx_poly_tp = bvh.GetIdxPoly();
			for(size_t i=0;i<n_primes;++i){
				h_idx_poly[i] = idx_poly_tp[i];
			}
			// 3. Make h_flatTree (d_flatTree)
			cuBVHFlatNode* h_flatTree = new cuBVHFlatNode[bvh.nNodes];
			for(size_t i=0;i<bvh.nNodes;++i){
				h_flatTree[i].fromBVHFlatNode(bvh.flatTree[i]);
			}
			//
			// Allocation
			//
			cudaMalloc(&d_bvh, 		sizeof(cuBVH));
			cudaMalloc(&d_primes, 	n_primes*sizeof(cuTRI<float>));
			cudaMalloc(&d_idx_poly, n_primes*sizeof(size_t));
			cudaMalloc(&d_flatTree, bvh.nNodes*sizeof(cuBVHFlatNode));
			//
			// Duplicate to device variables
			//
			cudaMemcpy(&(d_bvh->nNodes),   	&(bvh.nNodes),   	sizeof(size_t), 					cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_bvh->nLeafs),   	&(bvh.nLeafs),   	sizeof(size_t), 					cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_bvh->leafSize), 	&(bvh.leafSize), 	sizeof(size_t), 					cudaMemcpyHostToDevice);
			cudaMemcpy(d_primes, 			h_primes, 			n_primes*sizeof(cuTRI<float>),		cudaMemcpyHostToDevice);
			cudaMemcpy(d_idx_poly, 			h_idx_poly, 		n_primes*sizeof(size_t),			cudaMemcpyHostToDevice);
			cudaMemcpy(d_flatTree, 			h_flatTree, 		bvh.nNodes*sizeof(cuBVHFlatNode),	cudaMemcpyHostToDevice);
			// assign
			cudaMemcpy(&(d_bvh->build_primes),	&d_primes, 		sizeof(cuTRI<float>*),		cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_bvh->idx_poly),		&d_idx_poly, 	sizeof(size_t*),			cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_bvh->flatTree), 		&d_flatTree,	sizeof(cuBVHFlatNode*),		cudaMemcpyHostToDevice);
			//
			// free temp space in Host
			//
			delete [] h_primes;
			delete [] h_idx_poly;
			delete [] h_flatTree;
			//
			// Check Error
			//
			ChkErr("cu::cuBVH::Create");
		}
		__host__ void Free(cuBVH* d_bvh, cuTRI<float>* d_primes, size_t* d_idx_poly, cuBVHFlatNode* d_flatTree){
			cudaFree(d_primes);
			cudaFree(d_idx_poly);
			cudaFree(d_flatTree);
			cudaFree(d_bvh);
			//
			// Check Error
			//
			ChkErr("cu::cuBVH::Free");
		}

		__forceinline__ __device__ bool getIntersection(const cuRay& ray, bool occlusion) const{

			cuIntersectionInfo I;

			I.t = 999999999.f;
			I.object = NULL;
			float bbhits[4];
			int32_t closer, other;

			// Working set
			cuBVHTraversal todo[64];
			int32_t stackptr = 0;

			// "Push" on the root node to the working set
			todo[stackptr].i = 0;
			todo[stackptr].mint = -9999999.f;

			while(stackptr>=0){
				// Pop off the next node to work on.
				int ni = todo[stackptr].i;
				float near = todo[stackptr].mint;
				stackptr--;
				const cuBVHFlatNode& node = flatTree[ni];

				// If this node is further than the closest found intersection, continue
				if(near > I.t){
					continue;
				}
				// Is leaf -> Intersect
				if( node.rightOffset == 0 ){
					for(uint32_t o=0;o<node.nPrims;++o){
						const cuTRI<float>* obj = &(build_primes[node.start+o]);
						bool hit = obj->getIntersection(ray, &I);
						// If we're only looking for occlusion, then any hit is good enough
						if(occlusion && hit){
							return true;
						}
					}

				}else{ // Not a leaf

					bool hitc0 = flatTree[ni+1].bbox.intersect(ray, bbhits, bbhits+1);
					bool hitc1 = flatTree[ni+node.rightOffset].bbox.intersect(ray, bbhits+2, bbhits+3);

					// We assume that the left child is a closer hit...
					closer = ni+1;
					other  = ni+node.rightOffset;

					// ... If the right child was actually closer, swap the relavent values.
					if((bbhits[2] < bbhits[0]) && hitc0 && hitc1){
						cu::swap(bbhits[0], bbhits[2]);
						cu::swap(bbhits[1], bbhits[3]);
						cu::swap(closer,other);
					}

					// It's possible that the nearest object is still in the other side, but we'll
					// check the further-awar node later...

					// Push the farther first
					if(hitc1){
						todo[++stackptr] = cuBVHTraversal(other, bbhits[2]);
					}

					// And now the closer (with overlap test)
					if(hitc0){
						todo[++stackptr] = cuBVHTraversal(closer, bbhits[0]);
					}
				}
			}

			// If we hit something,
			if(I.object != NULL){
				I.hit.x = fma(ray.d.x, I.t, ray.o.x);
				I.hit.y = fma(ray.d.y, I.t, ray.o.y);
				I.hit.z = fma(ray.d.z, I.t, ray.o.z);
			}

			return I.object != NULL;
		}


		__device__ bool getIntersection(const cuRay& ray, cuIntersectionInfo* I, bool occlusion) const{
			I->t = 999999999.f;
			I->object = NULL;
			float bbhits[4];
			int32_t closer, other;

			// remember previous status
			cuIntersectionInfo old_I = *I;
			bool old_hit = false;

			// Working set
			cuBVHTraversal todo[64];
			int32_t stackptr = 0;

			// "Push" on the root node to the working set
			todo[stackptr].i = 0;
			todo[stackptr].mint = -9999999.f;

			while(stackptr>=0){
				// Pop off the next node to work on.
				int ni = todo[stackptr].i;
				float near = todo[stackptr].mint;
				stackptr--;
//				const cuBVHFlatNode &node(flatTree[ ni ]);
				const cuBVHFlatNode& node = flatTree[ni];

//				if(k == 31){
//					printf("k = %ld, stackptr = %d, near = %.20f, I.t = %.20f\n", k, stackptr, near, I->t);
//				}

				// If this node is further than the closest found intersection, continue
				if(near > I->t){
					continue;
				}
				// Is leaf -> Intersect
				if( node.rightOffset == 0 ){
					for(uint32_t o=0;o<node.nPrims;++o){
						const cuTRI<float>* obj = &(build_primes[node.start+o]);
						bool hit = obj->getIntersection(ray, I);

						// Detection triangle facing direction
						// face to source -> hit = true
						// Otherwise      -> hit = false & intersection->object = NULL
						//                -> If hit by previous event, resotre it
						bool IsFace = (ray.d.x*obj->N.x +
									   ray.d.y*obj->N.y +
									   ray.d.z*obj->N.z ) < 1E-10;

						if(hit){
							// Hit & face to the source
							if(IsFace){
								old_hit = true;
								old_I = *I;
								break;
							// Hit but dose not face to the source
							}else{
								hit = old_hit;
								I->t = old_I.t;
								I->object = old_I.object;
								I->hit = old_I.hit;
							}
						}

//						if(k == 31 && hit){
//							printf("k = %ld, stackptr = %d\n", k, stackptr);
//							printf("k = %ld, node.start+o = %d\n", k, node.start+o);
//							printf("k = %ld, o = %d, nPrims = %ld, hit = %d\n", k, o, node.nPrims, hit);
//							printf("k = %ld, node.rightOffset = %ld\n", k, node.rightOffset);
//							printf("k = %ld, N = [%.20f,%.20f,%.20f]\n", k, obj->N.x, obj->N.y, obj->N.z);
//							printf("k = %ld, near = %.20f, I.t = %.20f\n", k, near, I->t);
//							printf(" I.obj != NULL = %d\n", (I->object != NULL));
//						}
//						bool hit = obj.getIntersection(ray, I, k);

						// If we're only looking for occlusion, then any hit is good enough
						if(occlusion && hit){
							return true;
						}
					}

				}else{ // Not a leaf

					bool hitc0 = flatTree[ni+1].bbox.intersect(ray, bbhits, bbhits+1);
					bool hitc1 = flatTree[ni+node.rightOffset].bbox.intersect(ray, bbhits+2, bbhits+3);

					// We assume that the left child is a closer hit...
					closer = ni+1;
					other  = ni+node.rightOffset;

					// ... If the right child was actually closer, swap the relavent values.
					if((bbhits[2] < bbhits[0]) && hitc0 && hitc1){
						cu::swap(bbhits[0], bbhits[2]);
						cu::swap(bbhits[1], bbhits[3]);
						cu::swap(closer,other);
					}

					// It's possible that the nearest object is still in the other side, but we'll
					// check the further-awar node later...

					// Push the farther first
					if(hitc1){
						todo[++stackptr] = cuBVHTraversal(other, bbhits[2]);
					}

					// And now the closer (with overlap test)
					if(hitc0){
						todo[++stackptr] = cuBVHTraversal(closer, bbhits[0]);
					}
				}
			}

			// If we hit something,
			if(I->object != NULL){
				I->hit.x = fma(ray.d.x, I->t, ray.o.x);
				I->hit.y = fma(ray.d.y, I->t, ray.o.y);
				I->hit.z = fma(ray.d.z, I->t, ray.o.z);
//				I->hit = ray.o + ray.d * I->t;
			}

			return I->object != NULL;
		}
		__host__ __device__ cuBVHFlatNode* GetflatTree() const{ return flatTree; }
	public:
		size_t nNodes, nLeafs, leafSize;
//		std::vector<Obj*>* build_prims;
		cuTRI<float>* build_primes;
		size_t* idx_poly;
		// Fast Traversal System
		cuBVHFlatNode* flatTree;
	};
	//+-----------------------------------+
	//|             cuMeshInc             | << double [v]
	//+-----------------------------------+
	class cuMeshInc {
	public:
		__host__ void Create(const MeshInc& h_inc, cuMeshInc*& d_inc, cuVEC<double>*& d_dirH_disH, cuVEC<double>*& d_dirV_disV){
			//
			// Note: Reduce the precision for double to float
			//
			size_t nH  = h_inc.dirH_disH.GetNum();
			size_t nV  = h_inc.dirV_disV.GetNum();
			size_t szH = nH * sizeof(cuVEC<double>);
			size_t szV = nV * sizeof(cuVEC<double>);
			// copy and down to float
			cuVEC<double>* tmp_H = new cuVEC<double>[nH];
			cuVEC<double>* tmp_V = new cuVEC<double>[nV];
			for(size_t i=0;i<nH;++i){
				// H
				tmp_H[i].x = double(h_inc.dirH_disH[i].x());
				tmp_H[i].y = double(h_inc.dirH_disH[i].y());
				tmp_H[i].z = double(h_inc.dirH_disH[i].z());
			}
			for(size_t i=0;i<nV;++i){
				// V
				tmp_V[i].x = double(h_inc.dirV_disV[i].x());
				tmp_V[i].y = double(h_inc.dirV_disV[i].y());
				tmp_V[i].z = double(h_inc.dirV_disV[i].z());
			}
			size_t 	      tmp_LH   = (size_t)h_inc.LH;
			size_t 	      tmp_LV   = (size_t)h_inc.LV;
			double 		  tmp_area = (double)h_inc.Area;
			cuVEC<double> tmp_Ps(h_inc.Ps.x(), h_inc.Ps.y(), h_inc.Ps.z());
			// memory allocation
			cudaMalloc(&d_inc, sizeof(cuMeshInc));
			cudaMalloc(&d_dirH_disH, szH);
			cudaMalloc(&d_dirV_disV, szV);
			// copy to data pointer
			cudaMemcpy(d_dirH_disH,    	  	tmp_H, 		  szH, 					  cudaMemcpyHostToDevice);
			cudaMemcpy(d_dirV_disV,   	  	tmp_V, 		  szV, 					  cudaMemcpyHostToDevice);
			// copy
			cudaMemcpy(&(d_inc->LH),      	&tmp_LH, 	  sizeof(size_t),		  cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_inc->LV),      	&tmp_LV, 	  sizeof(size_t),		  cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_inc->Area),    	&tmp_area, 	  sizeof(double),		  cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_inc->Ps),      	&tmp_Ps, 	  sizeof(cuVEC<double>),  cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_inc->dirH_disH), &d_dirH_disH, sizeof(cuVEC<double>*), cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_inc->dirV_disV), &d_dirV_disV, sizeof(cuVEC<double>*), cudaMemcpyHostToDevice);
			//
			// delete temp pointer
			//
			delete [] tmp_H;
			delete [] tmp_V;
			//
			// Check Error
			//
			ChkErr("cu::cuMeshInc::Create");
		}
		__host__ void Free(cuMeshInc* d_inc, cuVEC<double>* d_dirH_disH, cuVEC<double>* d_dirV_disV){
			cudaFree(d_dirH_disH);
			cudaFree(d_dirV_disV);
			cudaFree(d_inc);
			//
			// Check Error
			//
			ChkErr("cu::cuMeshInc::Free");
		}
		__device__ void GetCell(const size_t k, cuRay& ray){
			size_t j = k/LH;		// V index
			size_t i = k - j*LH;	// H index (== k%LH)

			cuVEC<double> CC = dirH_disH[i] - dirV_disV[j];	// temp point
			cuVEC<double> N  = cu::Unit(CC - Ps);			// Normal vector

			ray  = cuRay(CC, N);
		}
		__device__ void GetCell(const size_t k, cuVEC<double>& ray_o, cuVEC<double>& ray_d){
			size_t j = k/LH;		// V index
			size_t i = k - j*LH;	// H index (== k%LH)

			ray_o = dirH_disH[i] - dirV_disV[j];	// temp point
			ray_d  = cu::Unit(ray_o - Ps);			// Normal vector
		}
		__device__ void GetCell(const size_t k, double& ray_o_x, double& ray_o_y, double& ray_o_z,
												float& ray_d_x, float& ray_d_y, float& ray_d_z){
			size_t j = k/LH;		// V index
			size_t i = k - j*LH;	// H index (== k%LH)

			// temp point
			ray_o_x = dirH_disH[i].x - dirV_disV[j].x;
			ray_o_y = dirH_disH[i].y - dirV_disV[j].y;
			ray_o_z = dirH_disH[i].z - dirV_disV[j].z;

			double3 a;
			a.x = ray_o_x - Ps.x;
			a.y = ray_o_y - Ps.y;
			a.z = ray_o_z - Ps.z;

			// Normal vector
			double inv = 1.0/sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
			ray_d_x = a.x*inv;
			ray_d_y = a.y*inv;
			ray_d_z = a.z*inv;
		}
		__device__ void GetCell(const size_t k, cuVEC<double>& ray_o, cuVEC<double>& ray_d, double& ray_area){
			size_t j = k/LH;		// V index
			size_t i = k - j*LH;	// H index (== k%LH)

			ray_o = dirH_disH[i] - dirV_disV[j];	// temp point
			ray_d  = cu::Unit(ray_o - Ps);			// Normal vector
			ray_area = Area;
		}
		__device__ void GetCell(const size_t k, cuRay& ray, double& area){
			size_t j = k/LH;		// V index
			size_t i = k - j*LH;	// H index (== k%LH)

			cuVEC<double> CC = dirH_disH[i] - dirV_disV[j];	// temp point
			cuVEC<double> N  = cu::Unit(CC - Ps);			// Normal vector

			ray  = cuRay(CC, N);
			area = Area;
		}
		__device__ void GetCell(const size_t k, double& CC_x, double& CC_y, double& CC_z,
								double& N_x, double& N_y, double& N_z){
			size_t j = k/LH;		// V index
			size_t i = k - j*LH;	// H index (== k%LH)

//			cuVEC<double> CC = dirH_disH[i] - dirV_disV[j];	// temp point
//			cuVEC<double> N  = cu::Unit(CC - Ps);			// Normal vector
//
//			ray  = cuRay(CC, N);

//			cuVEC<double> CC = dirH_disH[i] - dirV_disV[j];	// Centre point
//			cuVEC<double> N  = cu::Unit(CC - Ps);			// Normal vector

			CC_x = dirH_disH[i].x - dirV_disV[j].x;
			CC_y = dirH_disH[i].y - dirV_disV[j].y;
			CC_z = dirH_disH[i].z - dirV_disV[j].z;

			double x = CC_x - Ps.x;
			double y = CC_y - Ps.y;
			double z = CC_z - Ps.z;

			double inv = 1./sqrt(x*x+y*y+z*z);
			N_x = x*inv;
			N_y = y*inv;
			N_z = z*inv;
		}
		__device__ size_t nPy(){
			return LH*LV;
		}
		__device__ void Print(const size_t N=1){
			printf("+------------------------------------+\n");
			printf("|          MeshInc Summary           |\n");
			printf("+------------------------------------+\n");
			printf("Rad    [m] = %f\n", Ps.abs());
			printf("LH     [#] = %ld\n", LH);
			printf("Area  [m2] = %f\n", Area);
			printf("+             \n");
			printf("   dirH - disH\n");
			printf("+             \n");
			for(int i=0;i<N;++i){
				dirH_disH[i].Print();
			}
			printf("+             \n");
			printf("   dirV - disV\n");
			printf("+             \n");
			for(int i=0;i<N;++i){
				dirV_disV[i].Print();
			}
		}
	public:
		size_t LH;					// [#] H direction samples
		size_t LV;					// [#] V direction samples
		double Area;				// [m^2] Area of each cell
		cuVEC<double> Ps;			// [m,m,m] Sensor position
		cuVEC<double>* dirH_disH;	// dirH_disH[i] = dirH*disH[i] - dH[i] * NN + PLOS
		cuVEC<double>* dirV_disV;	// dirV_disV[j] = dirV*disV[j] + dV[j] * NN
	};
	//+-----------------------------------+
	//|               cuCPLX              |
	//+-----------------------------------+
	template<typename T>
	class cuCPLX {
	public:
		// Constructure
		__host__ __device__ cuCPLX():r(0),i(0){};
		__host__ __device__ cuCPLX(T a,T b):r(a),i(b){};
		__host__ __device__ cuCPLX(const cuCPLX<T>& in):r(in.r),i(in.i){};
		template<typename T2> __host__ __device__ cuCPLX(const cuCPLX<T2>& in){
			r = (T)in.r;
			i = (T)in.i;
		}
		__host__ __device__ cuCPLX(const T& in):r(in),i((T)0){};
		// Operator overloading
		__host__ __device__ cuCPLX<T>& operator=(const cuCPLX<T>& b){
			r = b.r;
			i = b.i;
			return *this;
		}
		__host__ __device__ cuCPLX<T> operator-(){
			return cuCPLX<T>(-r,-i);
		}
		__host__ __device__ friend const cuCPLX<T> operator+(const cuCPLX<T>& L,const cuCPLX<T>& R){
			return cuCPLX<T>( L.r+R.r,L.i+R.i );
		}
		__host__ __device__ friend const cuCPLX<T> operator-(const cuCPLX<T>& L,const cuCPLX<T>& R){
			return cuCPLX<T>( L.r-R.r,L.i-R.i );
		}
		__host__ __device__ friend const cuCPLX<T> operator*(const cuCPLX<T>& L,const cuCPLX<T>& R){
			return cuCPLX<T>( L.r*R.r - L.i*R.i, L.r*R.i + L.i*R.r );
		}
		__host__ __device__ friend const cuCPLX<T> operator/(const cuCPLX<T>& L,const cuCPLX<T>& R){
			T R_conj_R = R.r*R.r + R.i*R.i;
			return cuCPLX<T>( (L.r*R.r + L.i*R.i)/R_conj_R, (L.i*R.r - L.r*R.i)/R_conj_R );
		}
		__host__ __device__ friend const cuCPLX<T> operator+(const cuCPLX<T>& L,const T& R){
			return cuCPLX<T>( L.r+R,L.i );
		}
		__host__ __device__ friend const cuCPLX<T> operator-(const cuCPLX<T>& L,const T& R){
			return cuCPLX<T>( L.r-R,L.i );
		}
		__host__ __device__ friend const cuCPLX<T> operator*(const cuCPLX<T>& L,const T& R){
			return cuCPLX<T>( L.r*R,L.i*R );
		}
		__host__ __device__ friend const cuCPLX<T> operator/(const cuCPLX<T>& L,const T& R){
			return cuCPLX<T>( L.r/R,L.i/R );
		}
		__host__ __device__ friend const cuCPLX<T> operator+(const T& L,const cuCPLX<T>& R){
			return cuCPLX<T>( L+R.r,R.i );
		}
		__host__ __device__ friend const cuCPLX<T> operator-(const T& L,const cuCPLX<T>& R){
			return cuCPLX<T>( L-R.r,-R.i );
		}
		__host__ __device__ friend const cuCPLX<T> operator*(const T& L,const cuCPLX<T>& R){
			return cuCPLX<T>( L*R.r,L*R.i );
		}
		__host__ __device__ friend const cuCPLX<T> operator/(const T& L,const cuCPLX<T>& R){
			T R_conj_R = R.r*R.r + R.i*R.i;
			return cuCPLX<T>( (L*R.r)/R_conj_R, (-L*R.i)/R_conj_R );
		}
		__host__ __device__ friend cuCPLX<T>& operator+=(cuCPLX<T>& res,const cuCPLX<T>& R){
			res.r += R.r;
			res.i += R.i;
			return res;
		}
		__host__ __device__ friend cuCPLX<T>& operator-=(cuCPLX<T>& res,const cuCPLX<T>& R){
			res.r -= R.r;
			res.i -= R.i;
			return res;
		}
		__host__ __device__ friend cuCPLX<T>& operator*=(cuCPLX<T>& res,const T& R){
			res.r *= R;
			res.i *= R;
			return res;
		}
		__host__ __device__ friend cuCPLX<T>& operator/=(cuCPLX<T>& res,const T& R){
			res.r /= R;
			res.i /= R;
			return res;
		}
		__host__ __device__ friend cuCPLX<T>& operator*=(cuCPLX<T>& res,const cuCPLX<T>& R){
			cuCPLX<T> L = res;
			res.r = L.r*R.r - L.i*R.i;
			res.i = L.r*R.i + L.i*R.r;
			return res;
		}
		// Misc.
		__host__ __device__ T abs(){
			return std::sqrt(r*r + i*i);
		}
		__host__ __device__ const T abs()const{
			return std::sqrt(r*r + i*i);
		}
		__host__ __device__ const cuCPLX<T> sqrt()const{
			double p = def::SQRT2_INV * std::sqrt(std::sqrt(r*r + i*i) + r);
//			double sgn = (i > 0)? 1.0:-1.0;
			double q = def::SQRT2_INV * std::sqrt(std::sqrt(r*r + i*i) - r);
			q = copysign(q, i);
			return cuCPLX<T>(T(p),T(q));
		}
		__host__ __device__ cuCPLX<T>& conj(){
			i = -i;
			return *this;
		}
		__host__ __device__ T phase(){
			return T(atan2(double(i),double(r)));
		}
		__host__ __device__ cuCPLX<T> exp(const cuCPLX<T>& x){
			T imag = x.i();
					if(isinf(x.r())){
						if(x.r() < T(0)){
							if(!isfinite(imag)){
								imag = T(1);
							}
						}else{
							if(imag == 0 || !isfinite(imag)){
								if(isinf(imag)){
									imag = T(NAN);
								}
								return CPLX<T>(x.r(), imag);
							}
						}
					}else{
						if(isnan(x.r()) && x.i() == 0){
							return x;
						}
					}
					T e = exp(x.r());
					return CPLX<T>(e * cos(imag), e * sin(imag));
		}
		__device__ void Print(){
			printf("(%.10f,%.10f)\n", r, i);
		}
	public:
		T r,i;
	};
	//+-----------------------------------+
	//|              cuScatter            | << double
	//+-----------------------------------+
	class cuScatter {
	public:
		__host__ __device__ cuScatter(){
			Level = 0;
		};
		__host__ __device__ cuScatter(const size_t Level):Level(Level){};
		__host__ __device__ cuScatter(const cuVEC<cuCPLX<float> >& Cplx, const cuCPLX<float>& ETS, const cuCPLX<float>& EPS, const size_t LEVEL){
			cplx = Cplx;
			Ets = ETS;
			Eps = EPS;
			Level = LEVEL;
		}
		__device__ void Print(){
			printf("+-----------------------+\n");
			printf("|    Scatter Results    |\n");
			printf("+-----------------------+\n");
			printf("Level = %ld\n", Level);
			printf("cplx  = [(%.10f,%.10f),(%.10f,%.10f),(%.10f,%.10f)]\n", cplx.x.r, cplx.x.i, cplx.y.r, cplx.y.i, cplx.z.r, cplx.z.i);
			printf("Ets   = "); Ets.Print();
			printf("Eps   = "); Eps.Print();
		}
	public:
		cuVEC<cuCPLX<float> > cplx;
		cuCPLX<float> Ets, Eps;
		size_t Level;
	};
	//+-----------------------------------+
	//|            cuSBRElement           |
	//+-----------------------------------+
	template<typename T>
	class cuSBRElement {
	public:
		__host__ __device__ cuSBRElement(){};
		__host__ __device__ cuSBRElement(const cuCPLX<T>& sumt, const cuCPLX<T>& sump):sumt(sumt),sump(sump){};
	public:
		cuCPLX<T> sumt;		// sum of theta component (V)
		cuCPLX<T> sump;		// sum of phi   component (H)
	};
	//+-----------------------------------+
	//|           cuThetaPhiVec           | << double
	//+-----------------------------------+
	class cuThetaPhiVec {
	public:
		__host__ __device__ cuThetaPhiVec(){}
		__host__ __device__ cuThetaPhiVec(const cuVEC<float>& Theta_vec, const cuVEC<float>& Phi_vec){
			theta_vec = Theta_vec;
			phi_vec   = Phi_vec;
		}
		__device__ void Print(){
			printf("theta_vec = "); theta_vec.Print();
			printf("phi_vec   = "); phi_vec.Print();
		}
	public:
		cuVEC<float> theta_vec;
		cuVEC<float> phi_vec;
	};
	//+-----------------------------------+
	//|           cuElectricField         | << double
	//+-----------------------------------+
	class cuElectricField {
	public:
		__host__ __device__ cuElectricField(){};
		__host__ __device__ cuElectricField(const cuVEC<double>& K, const cuVEC<double>& O, const cuVEC<cuCPLX<float> >& Cplx, const bool EnableThetaPhiVector=false):
			k(K),o(O),cplx(Cplx){
			// Find Global unit vector
			if(EnableThetaPhiVector){
				ThetaPhiVector(K, g);
			}
		}
		__host__ __device__ cuElectricField(const cuVEC<double>& K, const cuVEC<double>& O, const cuCPLX<float>& Et, const cuCPLX<float>& Ep):
			k(K),o(O){

			// Find Global unit vector
			ThetaPhiVector(K, g);

//			cuCPLX<double> Ep2((double)Ep.r, (double)Ep.i);
//			cuCPLX<double> Et2((double)Et.r, (double)Et.i);
//			cuVEC<double>  phi_vec((double)g.phi_vec.x, (double)g.phi_vec.y, (double)g.phi_vec.z);
//			cuVEC<double>  theta_vec((double)g.theta_vec.x, (double)g.theta_vec.y, (double)g.theta_vec.z);
//
//			cplx.x.r = float( Ep2.r * phi_vec.x + Et2.r * theta_vec.x );  cplx.x.i = float( Ep2.i * phi_vec.x + Et2.i * theta_vec.x );
//			cplx.y.r = float( Ep2.r * phi_vec.y + Et2.r * theta_vec.y );  cplx.x.i = float( Ep2.i * phi_vec.y + Et2.i * theta_vec.y );
//			cplx.z.r = float( Ep2.r * phi_vec.z + Et2.r * theta_vec.z );  cplx.x.i = float( Ep2.i * phi_vec.z + Et2.i * theta_vec.z );

			cplx.x = Ep * g.phi_vec.x + Et * g.theta_vec.x;
			cplx.y = Ep * g.phi_vec.y + Et * g.theta_vec.y;
			cplx.z = Ep * g.phi_vec.z + Et * g.theta_vec.z;

//			cplx = cuVEC<cuCPLX<float> >(Ep * g.phi_vec.x,   Ep * g.phi_vec.y,   Ep * g.phi_vec.z)   +
//				   cuVEC<cuCPLX<float> >(Et * g.theta_vec.x, Et * g.theta_vec.y, Et * g.theta_vec.z);
		}
		__host__ __device__ void ThetaPhiVector(const cuVEC<double>& k, cuThetaPhiVec& out){
			cuVEC<float> kk = -k;
//			float xy = hypotf(kk.x, kk.y);
			float xy = sqrtf(kk.x*kk.x + kk.y*kk.y);
			float st = xy;
			float ct = kk.z;
			float sp = kk.y/xy;
			float cp = kk.x/xy;

			out.theta_vec = cuVEC<float>(ct*cp, ct*sp, -st);
			out.phi_vec   = cuVEC<float>(-sp, cp, 0);
		}
		__device__ void AddPhase(const double phs){
//			double PHS = fmod(phs, def::PI2);
//
//			double cp = cos(-PHS);
//			double sp = sin(-PHS);
			double sp, cp;
//			sincos(+phs, &sp, &cp);
			sincos(-phs, &sp, &cp);
//			double cp = cos(-phs);
//			double sp = sin(-phs);

			cuCPLX<float> phase(cp, sp);

			cplx.x = cplx.x * phase;
			cplx.y = cplx.y * phase;
			cplx.z = cplx.z * phase;



//			double PHS = -fmod(phs, def::PI2);
//
////			double PHS = -double(int(phs*1E4) % int(def::PI2*1E4))/1E4;
//
//			// -use_fast_math
////			double cp, sp;
////			sincos(PHS, &sp, &cp);
//			double cp = cos(PHS);
//			double sp = cos(PHS);
//
//
////			double cp, sp;
////			sincos(PHS, &sp, &cp);
//
//
//			double cxr = cplx.x.r; double cxi = cplx.x.i;
//			double cyr = cplx.y.r; double cyi = cplx.y.i;
//			double czr = cplx.z.r; double czi = cplx.z.i;
//
//
//			// x
//			cplx.x.r = cxr*cp - cxi*sp;
//			cplx.x.i = cxr*sp + cxi*cp;
//			// y
//			cplx.y.r = cyr*cp - cyi*sp;
//			cplx.y.i = cyr*sp + cyi*cp;
//			// z
//			cplx.z.r = czr*cp - czi*sp;
//			cplx.z.i = czr*sp + czi*cp;
		}
//		__device__ void AddPhase(const float phs){
//			double PHS = fmod((double)phs, def::PI2);
//			double cp = cos(-PHS);
//			double sp = sin(-PHS);
//
//			double cxr = cplx.x.r; double cxi = cplx.x.i;
//			double cyr = cplx.y.r; double cyi = cplx.y.i;
//			double czr = cplx.z.r; double czi = cplx.z.i;
//
//
//			// x
//			cplx.x = cuCPLX<float>( cxr*cp - cxi*sp, cxr*sp + cxi*cp );
//			// y
//			cplx.y = cuCPLX<float>( cyr*cp - cyi*sp, cyr*sp + cyi*cp );
//			// z
//			cplx.z = cuCPLX<float>( czr*cp - czi*sp, czr*sp + czi*cp );
//
////			double PI2 = 6.283185307179586;
////			double phs2= (double)phs;
////			double PHS = modf(phs2, &PI2);
////
////			double cp, sp;
////			sincos(-phs2, &sp, &cp);
//////			cuCPLX<double> phase(cp, sp);
//////			cuCPLX<double> cplx_x(cplx.x.r, cplx.x.i);
//////			cuCPLX<double> cplx_y(cplx.y.r, cplx.y.i);
//////			cuCPLX<double> cplx_z(cplx.z.r, cplx.z.i);
////
////			// x
////			cplx.x.r = float( (double)cplx.x.r*cp - (double)cplx.x.i*sp );
////			cplx.x.i = float( (double)cplx.x.r*sp + (double)cplx.x.i*cp );
////			// y
////			cplx.y.r = float( (double)cplx.y.r*cp - (double)cplx.y.i*sp );
////			cplx.y.i = float( (double)cplx.y.r*sp + (double)cplx.y.i*cp );
////			// z
////			cplx.z.r = float( (double)cplx.z.r*cp - (double)cplx.z.i*sp );
////			cplx.z.i = float( (double)cplx.z.r*sp + (double)cplx.z.i*cp );
//
//
//
////			float PI2 = cu::PI2;
////			float PHS = modff(phs, &PI2);
//////			CPLX<double> phase = mat::exp(-PHS);
////
////			float cp, sp;
////			sincosf(-phs, &sp, &cp);
////			cuCPLX<float> phase(cp, sp);
////
////			cplx.x = cplx.x * phase;
////			cplx.y = cplx.y * phase;
////			cplx.z = cplx.z * phase;
//		}
	public:
		cuVEC<double> k;				// EM wave direction
		cuVEC<double> o;				// EM wave location inc global XYZ
		cuVEC<cuCPLX<float> > cplx;	// complex type of electric field
		cuThetaPhiVec g;			// theta_vec & phi_vec infomation
	};
	//+-----------------------------------+
	//|               cuTAYLOR            | << template <- double
	//+-----------------------------------+
	template<typename T>
	class cuTAYLOR{
	public:
		__host__ __device__ cuTAYLOR():Rg(0.5),Nt(5){};
		__host__ __device__ cuTAYLOR(const T Rg, const int Nt):Rg(Rg),Nt(Nt){};
		// misc.
		__device__ void Print(){
			printf("+----------------------------------+\n");
			printf("| cuTAYLOR(PO Approximation) class |\n");
			printf("+----------------------------------+\n");
			printf("Rg = %f\n", Rg);
			printf("Nt = %d\n", Nt);
		}
	public:
		T Rg;
		int Nt;
	};
	//+-----------------------------------+
	//|               cuEF                | << double
	//+-----------------------------------+
	template<typename T>
	class cuEF{
	public:
		__host__ __device__ cuEF():TxPol(),RxPol(){};
		__host__ __device__ cuEF(const char TxPol, const char RxPol, const cuTAYLOR<T>& Taylor):
			TxPol(TxPol),RxPol(RxPol),Taylor(Taylor){
			// Assign Tx Polarization
			if(TxPol == 'V'){
				Et = cuCPLX<T>(1,0); // TM(V)
				Ep = cuCPLX<T>(0,0);
			}else{
				Et = cuCPLX<T>(0,0); // TE(H)
				Ep = cuCPLX<T>(1,0);
			}
		}
		__host__ void Create(const EF& h_Ef, cuEF<T>*& d_Ef){
			cuTAYLOR<T> tmp1( T(h_Ef.Taylor().Rg()), h_Ef.Taylor().Nt() );
			cuEF<T> tmp2((h_Ef.TxPol())[0], (h_Ef.RxPol())[0], tmp1);

			cudaMalloc(&d_Ef, sizeof(cuEF<T>));
			cudaMemcpy(d_Ef, &tmp2, sizeof(cuEF<T>), cudaMemcpyHostToDevice);
			//
			// Check Error
			//
			ChkErr("cu::cuEF::Create");
		}
		__host__ void Free(cuEF*& d_Ef){
			cudaFree(d_Ef);
			//
			// Check Error
			//
			ChkErr("cu::cuEF::Free");
		}
		// misc.
		__device__ void Print(){
			printf("+----------------------------------+\n");
			printf("|    cuEF(Electric Field) class    |\n");
			printf("+----------------------------------+\n");
			printf("TxPol          = %c\n", TxPol);
			printf("RxPol          = %c\n", RxPol);
			printf("Taylor (Rg,Nt) = (%f,%d)\n", Taylor.Rg, Taylor.Nt);
			printf("Et (V)         = (%f,%f)\n", Et.r, Et.i);
			printf("Ep (H)         = (%f,%f)\n", Ep.r, Ep.i);
		}
	public:
		char TxPol;			// Transmitted polarization
		char RxPol;			// Recivied polarization
		cuTAYLOR<T> Taylor;	// Taylor approximation
		cuCPLX<T> Et;		// theta component (V)
		cuCPLX<T> Ep;		// theta component (H)
	};
	//+-----------------------------------+
	//|               cuRF                | << double
	//+-----------------------------------+
	class cuRF{
	public:
		__host__ __device__ cuRF(){};
		__host__ __device__ cuRF(const cuCPLX<float>& TE, const cuCPLX<float>& TM):TE(TE),TM(TM){};
		// misc.
		__device__ void Print(){
			printf("+-------------------------------------+\n");
			printf("|  cu::cuRF(Reflection Factor) class  |\n");
			printf("+-------------------------------------+\n");
			printf("TE (Perpendicular) = "); TE.Print();
			printf("TM (Parallel)      = "); TM.Print();
		}
	public:
		cuCPLX<float> TE, TM;
	};
	//+-----------------------------------+
	//|            cuMaterialDB           | << double
	//+-----------------------------------+
	class cuMaterialDB {
	public:
		__host__ __device__ cuMaterialDB():sz(0),idx(),er_r(),tang(),mr(),mi(),d(),ER(),MR(),ERr_MRr_Sqrt(){};
		__host__ void Create(const MaterialDB& MatDB, cuMaterialDB*& d_MatDB,
							 size_t*& d_Idx, double*& d_Er_r, double*& d_Tang,
							 double*& d_Mr, double*& d_Mi, double*& d_D,
							 cuCPLX<double>* d_ER, cuCPLX<double>* d_MR, double* d_ERr_MRr_Sqrt){
			//
			// Note: Reduce the precision for double to float
			//
			size_t N = MatDB.Mat.size();
			size_t* tmp_Idx   = new size_t[N];
			double* tmp_Er_r  = new double[N];
			double* tmp_Tang  = new double[N];
			double* tmp_Mr    = new double[N];
			double* tmp_Mi    = new double[N];
			double* tmp_D     = new double[N];
			// Derive
			cuCPLX<double>* tmp_ER    = new cuCPLX<double>[N];
			cuCPLX<double>* tmp_MR    = new cuCPLX<double>[N];
			double* tmp_ERr_MRr_Sqrt = new double[N];

			for(long i=0;i<N;++i){
				tmp_Idx[i]  = size_t(MatDB.Mat[i].idx());
				tmp_Er_r[i] = MatDB.Mat[i].er_r();
				tmp_Tang[i] = MatDB.Mat[i].tang();
				tmp_Mr[i]   = MatDB.Mat[i].mr();
				tmp_Mi[i]   = MatDB.Mat[i].mi();
				tmp_D[i]    = MatDB.Mat[i].d() * 0.001;
				// Derive
				tmp_ER[i] = cuCPLX<double>(tmp_Er_r[i], -tmp_Tang[i]*tmp_Er_r[i]);
				tmp_MR[i] = cuCPLX<double>(tmp_Mr[i],   -tmp_Mi[i]);
//				tmp_ERr_MRr_Sqrt[i] = def::PI2_C*sqrt(tmp_Er_r[i]* tmp_Mr[i]);
				tmp_ERr_MRr_Sqrt[i] = sqrt(tmp_Er_r[i]* tmp_Mr[i]) / def::C * MatDB.Mat[i].d() * 0.001;
				tmp_ERr_MRr_Sqrt[i] = fmodf(tmp_ERr_MRr_Sqrt[i], def::PI2);

			}
			// memory allocation
			cudaMalloc(&d_MatDB, 		sizeof(cuMaterialDB));
			cudaMalloc(&d_Idx,   		N*sizeof(size_t));
			cudaMalloc(&d_Er_r,  		N*sizeof(double));
			cudaMalloc(&d_Tang,  		N*sizeof(double));
			cudaMalloc(&d_Mr,    		N*sizeof(double));
			cudaMalloc(&d_Mi,    		N*sizeof(double));
			cudaMalloc(&d_D,     		N*sizeof(double));
			// Derive
			cudaMalloc(&d_ER,    		N*sizeof(cuCPLX<double>));
			cudaMalloc(&d_MR,    		N*sizeof(cuCPLX<double>));
			cudaMalloc(&d_ERr_MRr_Sqrt, N*sizeof(double));
			// copy data
			cudaMemcpy(d_Idx,    		tmp_Idx, 	N*sizeof(size_t), 			cudaMemcpyHostToDevice);
			cudaMemcpy(d_Er_r,   		tmp_Er_r, 	N*sizeof(double), 			cudaMemcpyHostToDevice);
			cudaMemcpy(d_Tang,   		tmp_Tang, 	N*sizeof(double), 			cudaMemcpyHostToDevice);
			cudaMemcpy(d_Mr,   	 		tmp_Mr, 	N*sizeof(double), 			cudaMemcpyHostToDevice);
			cudaMemcpy(d_Mi,  	 		tmp_Mi, 	N*sizeof(double), 			cudaMemcpyHostToDevice);
			cudaMemcpy(d_D,  	 		tmp_D, 		N*sizeof(double), 			cudaMemcpyHostToDevice);
			// Derive
			cudaMemcpy(d_ER,  	 		tmp_ER, 	N*sizeof(cuCPLX<double>), 	cudaMemcpyHostToDevice);
			cudaMemcpy(d_MR,  	 		tmp_MR, 	N*sizeof(cuCPLX<double>), 	cudaMemcpyHostToDevice);
			cudaMemcpy(d_ERr_MRr_Sqrt,  tmp_ERr_MRr_Sqrt, N*sizeof(double), 		cudaMemcpyHostToDevice);
			// copy pointer
			cudaMemcpy(&(d_MatDB->sz),      &N,       sizeof(size_t),   cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_MatDB->idx),     &d_Idx,   sizeof(size_t*),	cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_MatDB->er_r),    &d_Er_r,  sizeof(double*),	cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_MatDB->tang),    &d_Tang,  sizeof(double*),	cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_MatDB->mr),    	&d_Mr, 	  sizeof(double*),	cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_MatDB->mi),    	&d_Mi, 	  sizeof(double*),	cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_MatDB->d),    	&d_D, 	  sizeof(double*),	cudaMemcpyHostToDevice);
			// Derive
			cudaMemcpy(&(d_MatDB->ER),    	&d_ER, 	  sizeof(cuCPLX<double>*),	cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_MatDB->MR),    	&d_MR, 	  sizeof(cuCPLX<double>*),	cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_MatDB->ERr_MRr_Sqrt),    	&d_ERr_MRr_Sqrt, 	  sizeof(double),	cudaMemcpyHostToDevice);
			//
			// delete temp pointer
			//
			delete [] tmp_Idx;
			delete [] tmp_Er_r;
			delete [] tmp_Tang;
			delete [] tmp_Mr;
			delete [] tmp_Mi;
			delete [] tmp_D;
			delete [] tmp_ER;
			delete [] tmp_MR;
			delete [] tmp_ERr_MRr_Sqrt;

			//
			// Check Error
			//
			ChkErr("cu::cuMaterialDB::Create");
		}
		__host__ void Free(cuMaterialDB*& d_MatDB, size_t*& d_Idx, double*& d_Er_r,
						   double*& d_Tang, double*& d_Mr, double*& d_Mi, double*& d_D,
						   cuCPLX<double>* d_ER, cuCPLX<double>* d_MR, double* d_ERr_MRr_Sqrt){
			cudaFree(d_Idx);
			cudaFree(d_Er_r);
			cudaFree(d_Tang);
			cudaFree(d_Mr);
			cudaFree(d_Mi);
			cudaFree(d_D);
			cudaFree(d_ER);
			cudaFree(d_MR);
			cudaFree(d_ERr_MRr_Sqrt);
			cudaFree(d_MatDB);
			//
			// Check Error
			//
			ChkErr("cu::cuMaterialDB::Free");
		}
//		__device__ void Print(){
//			printf("+--------------+\n");
//			printf("|   Material   |\n");
//			printf("+--------------+\n");
//			printf("Idx    = %d\n", idx);
//			printf("er_r   = %f\n", er_r);
//			printf("tang   = %f\n", tang);
//			printf("mr     = %f\n", mr);
//			printf("mi     = %f\n", mi);
//			printf("d      = %f\n", d);
//		}
		__device__ void Print(const size_t i){
			printf("+--------------+\n");
			printf("|   Material   |\n");
			printf("+--------------+\n");
			printf("Idx    = %ld\n", idx[i]);
			printf("er_r   = %f\n", er_r[i]);
			printf("tang   = %f\n", tang[i]);
			printf("mr     = %f\n", mr[i]);
			printf("mi     = %f\n", mi[i]);
			printf("d      = %f\n", d[i]);
		}
	public:
		size_t sz;		// size
		size_t* idx;	// 1. (integer) [x] Index
//		string name;	// 2. (string)  [x] Name
//		string freq;	// 3. (string)  [x] Frequency
		double* er_r;	// 4. (float)   [x] Real part of relative permittivity, relative to Air [F/m]
		double* tang;	// 5. (float)   [x] Loss tangent (x10^-4)
		double* mr;		// 6. (float)   [x] real part of relative permeability, relative to Air [H/m]
		double* mi;		// 7. (float)   [x] imag part of relative permeability, relative to Air [H/m]
		double* d;		// 8. (float)   [mm] Depth of this Layer
//		string remark;	// 9. (string)  [x] Remark
		// Dervie parameters
		cuCPLX<double>* ER;	// relative permittivity
		cuCPLX<double>* MR;	// relative permeability
		double* ERr_MRr_Sqrt;	// Sqrt(Er.r * Mr.r)
	};
    //+-----------------------------------+
	//|              cuSAR                | << double
	//+-----------------------------------+
	class cuSAR{
	public:
		__host__ __device__ cuSAR():
		theta_l_min(),theta_l_max(),theta_l_MB(),theta_sqc(),k0(),PRF(),Fr(),DC(),
		BWrg(),SWrg(),Laz(),Lev(),ant_eff(),Nr(),Na(),gain_rg(),Ls(),Tr(),
		theta_az(),theta_rg(),Kr(){};
		__host__ __device__ cuSAR(const float theta_l_MB, const float theta_sqc, const double f0,
								  const float PRF, const float Fr,const float DC, const float BWrg,
								  const float SWrg, const float Laz, const float Lev,
								  const float ant_eff, const size_t Nr, const size_t Na):
			theta_l_MB(theta_l_MB),
			theta_sqc(theta_sqc),
//			f0(f0),
			k0(f0*def::PI2_C),
			PRF(PRF),
			Fr(Fr),
			DC(DC),
			BWrg(BWrg),
			SWrg(SWrg),
			Laz(Laz),
			Lev(Lev),
			ant_eff(ant_eff),
			Nr(Nr),
			Na(Na){
			// Calculate
			Calculate();
		}
		__host__ void Create(const SAR& h_sar, cuSAR*& d_sar){
			cuSAR tmp(float(h_sar.theta_l_MB()), float(h_sar.theta_sqc()), h_sar.f0(),
					  float(h_sar.PRF()), float(h_sar.Fr()), float(h_sar.DC()), float(h_sar.BWrg()),
					  float(h_sar.SWrg()), float(h_sar.Laz()), float(h_sar.Lev()), float(h_sar.ant_eff()),
					  size_t(h_sar.Nr()), size_t(h_sar.Na()));
			cudaMalloc(&d_sar, sizeof(cuSAR));
			cudaMemcpy(d_sar, &tmp, sizeof(cuSAR), cudaMemcpyHostToDevice);
			//
			// Check Error
			//
			ChkErr("cu::cuSAR::Create");
		}
		__host__ void Free(cuSAR*& d_sar){
			cudaFree(d_sar);
			//
			// Check Error
			//
			ChkErr("cu::cuEF::Free");
		}
		__device__ void Print(){
			printf("+-------------------------------------+\n");
			printf("|            def::SAR class           |\n");
			printf("+-------------------------------------+\n");
			printf("theta_l_min [deg] = %f\n", theta_l_min*RTD);
			printf("theta_l_MB  [deg] = %f\n", theta_l_MB*RTD);
			printf("theta_l_max [deg] = %f\n", theta_l_max*RTD);
			printf("theta_sqc   [deg] = %f\n", theta_sqc*RTD);
			printf("f0          [GHz] = %f\n", k0/def::PI2_C/1E9);
			printf("PRF          [Hz] = %f\n", PRF);
			printf("Fr          [MHz] = %f\n", Fr/1E6);
			printf("DC            [%%] = %f\n", DC);
			printf("BWrg        [MHz] = %f\n", BWrg/1E6);
			printf("SWrg          [m] = %f\n", SWrg);
			printf("Laz           [m] = %f\n", Laz);
			printf("Lev           [m] = %f\n", Lev);
			printf("ant_eff       [x] = %f\n", ant_eff);
			printf("Nr      [samples] = %ld\n", Nr);
			printf("Na      [samples] = %ld\n", Na);
			printf("gain_rg       [x] = %f\n", gain_rg);
			printf("Ls            [m] = %f\n", Ls);
//			printf("Lambda        [m] = %f\n", lambda);
			printf("Tr          [sec] = %f\n", Tr);
			printf("theta_az    [deg] = %f\n", theta_az*RTD);
			printf("theta_rg    [deg] = %f\n", theta_rg*RTD);
			printf("Kr       [Hz/sec] = %f\n", Kr);
		}
		// Set
//		__host__ __device__ void Setf0(const double f0):f0(f0){
//			Calculate();
//		}
		__host__ __device__ void SetTheta_l_MB(const float Theta_l_MB){
			theta_l_MB = Theta_l_MB;
			float lambda = def::PI2/k0;
			float theta_rg = 0.886*lambda/Lev;
			theta_l_min = theta_l_MB - theta_rg/2.;
			theta_l_max = theta_l_MB + theta_rg/2.;
		}
		__host__ __device__ void SetLev(const float LEv){
			Lev = LEv;
			float lambda = def::PI2/k0;
			float theta_rg = 0.886*lambda/Lev;
			theta_l_min = theta_l_MB - theta_rg/2.;
			theta_l_max = theta_l_MB + theta_rg/2.;
		}
	private:
		__host__ __device__ void Calculate(){
//			double lambda=def::C / f0;						// [m] Transmitted wavelength
			float lambda = def::PI2/k0;
//			k0 = def::PI2 / lambda;
			float theta_rg = 0.886*lambda/Lev;
			theta_l_min = theta_l_MB - theta_rg/2.;
			theta_l_max = theta_l_MB + theta_rg/2.;
			Tr=DC/100.0/PRF;						// [sec] duration time
			theta_az=0.886*lambda/Laz/ant_eff;		// [rad] beamwidth at azimuth direction
			theta_rg=0.886*lambda/Lev/ant_eff;		// [rad] beamwidth at slant range direction
			Kr=BWrg/Tr;								// [Hz/sec] Chirp rate
		}
	public:
//		string Sen_name;	// Sensor name
		float theta_l_min;	// [rad] (min)look angle
		float theta_l_max;	// [rad] (max)look angle
		float theta_l_MB;	// [rad] (Main beam)look angle
		float theta_sqc;	// [rad] SAR squint angle @ beam center
//		double f0;			// [Hz] Transmitted frequency
//		float lambda;		// [m] wavelength
		double k0;			// [1/m] wavenumber
		float PRF;			// [Hz] Pulse repeat frequency
		float Fr;			// [Hz] ADC sampling rate
		float DC;			// [%] Transmit duty cycle
		float BWrg;			// [Hz] bandwidth @ range
		// Add
		float SWrg;			// [m] Slant Range Swath width
		float Laz;			// [m] antenna size @ azimuth
		float Lev;			// [m] antenna size @ elevation
		// Add
		float ant_eff;		// [x] Antenna effect coefficient
		size_t Nr;			// [samples] Slant range samples
		size_t Na;			// [samples] Azimuth samples
		// other
		float gain_rg;		// [x] Range gain
		float Ls;			// [m] Synthetic aperture radar antenna size
		float Tr;			// [sec] duration time
		float theta_az;		// [rad] beamwidth at azimuth direction
		float theta_rg;		// [rad] beamwidth at slant range direction
		float Kr;			// [Hz/sec] Chirp rate

	};
	//+-----------------------------------+
	//|              cuORB                | << double
	//+-----------------------------------+
	class cuORB{
	public:
		// Constructure
		__host__ __device__ cuORB(){
//			name = "WGS84"					;// Datum name
			E_a = 6378137.0				;// [m] Earth semi-major axis (WGS84)
			E_b = 6356752.31414			;// [m] Earth semi-mirror axis(WGS84)
			f  =0.0033528106875095227413	;//(E_a-E_b)/E_a;	//[x] flatness
			e2 =0.0066943800355127669813	;//1.-E_b*E_b/(E_a*E_a) // [x] Eccentricity;
		}
		__host__ __device__ cuORB(double Ea, double Eb){
			E_a = Ea;
			E_b = Eb;
			f   = (Ea-Eb)/Ea;
			e2  = 1.-Eb*Eb/(Ea*Ea);
		}
		__host__ __device__ cuORB(double Ea, double Eb, double F, double E2){
			E_a = Ea;
			E_b = Eb;
			f   = F;
			e2  = E2;
		}
		// Get
		__host__ __device__ void Print(){
			printf("+-------------------------------------+\n");
			printf("|           cu::cuORB class           |\n");
			printf("+-------------------------------------+\n");
//			printf("name     = %sf\n", name);
			printf("E_a      = %.4f\n", E_a);
			printf("E_b      = %.4f\n", E_b);
			printf("f        = %.4f\n", f);
			printf("e2       = %.4f\n", e2);
		}
	public:
//		string name;	// Datum name
		double E_a;		// [m] Earth semi-major axis (WGS84)
		double E_b;		// [m] Earth semi-mirror axis(WGS84)
		double f;		//(E_a-E_b)/E_a;	//[x] flatness
		double e2;		//1.-E_b*E_b/(E_a*E_a) // [x] Eccentricity;
		//double GM = 398600.4405 		;// [km^3 s^-2] G(gravitational constant, M Earth's mass
	};
	//+-----------------------------------+
	//|              cuGEO                | << double
	//+-----------------------------------+
	class cuGEO{
	public:
		// Constructure
		__host__ __device__ cuGEO():lon(0),lat(0),h(0){};
		__host__ __device__ cuGEO(double a, double b, double c):lon(a),lat(b),h(c){};
		__host__ __device__ cuGEO(const cuGEO& in):lon(in.lon),lat(in.lat),h(in.h){};
		// Operator overloading
		__host__ __device__ cuGEO& operator=(const cuGEO& b){
			lon = b.lon; lat = b.lat; h = b.h;
			return *this;
		}
		__host__ __device__ const cuGEO operator+(const cuGEO& R){
			return cuGEO( lon+R.lon,lat+R.lat,h+R.h );
		}
		__host__ __device__ const cuGEO operator-(const cuGEO& R){
			return cuGEO( lon-R.lon,lat-R.lat,h-R.h );
		}
		__host__ __device__ const cuGEO operator*(const cuGEO& R){
			return cuGEO( lon*R.lon,lat*R.lat,h*R.h );
		}
		__host__ __device__ const cuGEO operator/(const cuGEO& R){
			return cuGEO( lon/R.lon,lat/R.lat,h/R.h );
		}
		// Misc.
		__host__ __device__ void Print(){
			printf("[%.8f, %.8f, %.8f]\n", lon, lat, h);
		}
		__host__ __device__ void PrintDeg(){
			printf("[%.8f, %.8f, %.8f]\n", lon*RTD, lat*RTD, h);
		}
	public:
		double lon,lat,h;
	};
} // end nsmespace cu




#endif /* CUCLASS_CUH_ */
