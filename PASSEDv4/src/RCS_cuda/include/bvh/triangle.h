//
//  triangle.h
//  PASSEDv4
//
//  Created by Steve Chiang on 11/15/22.
//  Copyright (c) 2022 Steve Chiang. All rights reserved.
//

#ifndef triangle_h
#define triangle_h

#include <basic/vec.h>
#include <bvh/obj.h>
#include <rcs/material.h>
#include <basic/mat.h>
#include <cmath>
#include <algorithm>

using namespace vec;
using namespace material;
using namespace mat;

#define EPSILON 0.000001

template<typename T>
T Max3(const T a1, const T a2, const T a3){
	T out = a1;
	if(a2 > out){ out = a2; }
	if(a3 > out){ out = a3; }
	return out;
}
template<typename T>
T Min3(const T a1, const T a2, const T a3){
	T out = a1;
	if(a2 < out){ out = a2; }
	if(a3 < out){ out = a3; }
	return out;
}


//+---------------------------+
//|        Tria class         |
//+---------------------------+
template<typename T>
class TRI : public Obj {
public:
	/**
	 * Default constructor
	 */
	TRI(){}
	/**
	 * Constructor by giving all memeber variables
	 * @param[in]   V0   [m,m,m] 1st vertice position
	 * @param[in]   V1   [m,m,m] 2nd vertice position
	 * @param[in]   V2   [m,m,m] 3rd vertice position
	 * @param[in]   ea   [rad] All edge wedge angle (follow edge : {v0->v1},{v1->v2},{v2->v0})
	 * @param[in]   idx  [x] Self triangle index w.r.t. cad._PL
	 * @param[in]   idx_near [x] All near triangle index w.r.t. cad._PL (follow edge : {v0->v1},{v1->v2},{v2->v0})
	 */
	TRI(const VEC<T>& V0, const VEC<T>& V1, const VEC<T>& V2,
		const double ea[3], const size_t& idx, const long idx_near[3]){
		_ea[0] = ea[0];
		_ea[1] = ea[1];
		_ea[2] = ea[2];
		_v0    = V0;
		_v1    = V1;
		_v2    = V2;
		_idx   = idx;
		_idx_near[0] = idx_near[0];
		_idx_near[1] = idx_near[1];
		_idx_near[2] = idx_near[2];
		_MatIdx = 0;
	}
	/**
	 * Copy constructor
	 */
	template<typename T2>
	TRI(const TRI<T2>& in){
		_ea[0] = in.EA(0);
		_ea[1] = in.EA(1);
		_ea[2] = in.EA(2);
		_v0    = VEC<T>(in.V0().x(), in.V0().y(), in.V0().z());
		_v1    = VEC<T>(in.V1().x(), in.V1().y(), in.V1().z());
		_v2    = VEC<T>(in.V2().x(), in.V2().y(), in.V2().z());
		_idx   = in.IDX();
		_idx_near[0] = in.IDX_Near(0);
		_idx_near[1] = in.IDX_Near(1);
		_idx_near[2] = in.IDX_Near(2);
		_MatIdx = in.MatIDX();
	}
	/**
	 * Constructor by giving all memeber variables
	 * @param[in]   V0     [m,m,m] 1st vertice position
	 * @param[in]   V1     [m,m,m] 2nd vertice position
	 * @param[in]   V2     [m,m,m] 3rd vertice position
	 * @param[in]   MatIDX [x] Index of material
	 */
	TRI(const VEC<T>& V0, const VEC<T>& V1, const VEC<T>& V2, const long MatIDX = 0){
		_ea[0] = 0;
		_ea[1] = 0;
		_ea[2] = 0;
		_v0=V0; _v1=V1; _v2=V2;
		_MatIdx = MatIDX;
	}
	/**
	 * Get 1st vertice position (editable)
	 * @return 1st vertice position
	 */
	VEC<T>& V0(){ return _v0; }
	/**
	 * Get 1st vertice position
	 * @return 1st vertice position
	 */
	const VEC<T>& V0()const{ return _v0; }
	/**
	 * Get 2nd vertice position (editable)
	 * @return 2nd vertice position
	 */
	VEC<T>& V1(){ return _v1; }
	/**
	 * Get 2nd vertice position
	 * @return 2nd vertice position
	 */
	const VEC<T>& V1()const{ return _v1; }
	/**
	 * Get 3rd vertice position (editable)
	 * @return 3rd vertice position
	 */
	VEC<T>& V2(){ return _v2; }
	/**
	 * Get 3rd vertice position
	 * @return 3rd vertice position
	 */
	const VEC<T>& V2()const{ return _v2; }
	/**
	 * Get materal index (editable)
	 * @return index of material
	 */
	long MatIDX(){
		return _MatIdx;
	}
	/**
	 * Get materal index (editable)
	 * @return index of material
	 */
	long MatIDX()const{
		return _MatIdx;
	}
	/**
	 * Get wedge angle by giving index, i<=3. (i==0 : v0->v1, i==1 : v1->v2, i==2 : v2->v0) (editable)
	 * @param[in]   idx   Polygon index
	 * @return i-th wedge angle
	 */
	double& EA(const size_t i){ return _ea[i]; }
	/**
	 * Get wedge angle by giving index, i<=3. (i==0 : v0->v1, i==1 : v1->v2, i==2 : v2->v0)
	 * @return i-th wedge angle
	 */
	const double& EA(const size_t i)const{ return _ea[i]; }
	/**
	 * Get self polygon index (editable)
	 * @return Return polygon index
	 */
	size_t& IDX(){ return _idx; }
	/**
	 * Get self polygon index
	 * @return Return polygon index
	 */
	const size_t& IDX()const{ return _idx; }
	/**
	 * Get polygon index by giving index, idx<=3. (editable)
	 * @return Return polygon index
	 */
	long& IDX_Near(const long idx){ return _idx_near[idx]; }
	/**
	 * Get polygon index by giving index, idx<=3.
	 * @param[in]   idx   Polygon index
	 * @return Return polygon index
	 */
	const long& IDX_Near(const long idx)const{ return _idx_near[idx]; }

	double getArea() const {
		return _Area;
	}

	void setArea(double area) {
		_Area = area;
	}

	double getCa() const {
		return _ca;
	}

	void setCa(double ca) {
		_ca = ca;
	}

	double getSa() const {
		return _sa;
	}

	void setSa(double sa) {
		_sa = sa;
	}

	double getCb() const {
		return _cb;
	}

	void setCb(double cb) {
		_cb = cb;
	}

	double getSb() const {
		return _sb;
	}

	void setSb(double sb) {
		_sb = sb;
	}

	/**
	 * Ray-Triangle intersection (virtual)
	 * @param[in]   ray   (Ray class) Ray
	 * @param[out]  I     (IntersectionInfo class) Intersection information class
	 * @reutrn Return a boolean to detect if this is able to intersect. True(Intersection):False(not)
	 */
	virtual bool getIntersection(const Ray& ray, IntersectionInfo* I) const {
		//Find vectors for two edges sharing V1
		VEC<double> e1( _v1 - _v0 );	// Edge 1
		VEC<double> e2( _v2 - _v0 );	// Edge 2
		VEC<double> ray_d( ray.d );
		VEC<double> ray_o( ray.o );
		VEC<double> v0( _v0 );

		//Begin calculating determinant - also used to calculate u parameter
		VEC<double> P = cross(ray_d, e2);
		//if determinant is near zero, ray lies in plane of triangle
		double det = dot(e1, P);
		//NOT CULLING
		if(det > -EPSILON && det < EPSILON) return false;
		double inv_det = 1.0 / det;

		//calculate distance from V0 to ray origin
		VEC<double> TT = ray_o - v0;

		//Calculate u parameter and test bound
		double u = dot(TT, P) * inv_det;

		//The intersection lies outside of the triangle
		if(u < 0.0 || u > 1.0) return false;

		//Prepare to test v parameter
		VEC<double> Q = cross(TT, e1);

		//Calculate V parameter and test bound
		double v = dot(ray_d, Q) * inv_det;

		//The intersection lies outside of the triangle
		if(v < 0.0 || u + v  > 1.0) return false;

		double t = dot(e2, Q) * inv_det;	// Ray distance

		if(t > EPSILON) { //ray intersection
			I->object = this;
			I->t = t;
			I->hit.x() = float( ray_o.x() + (ray_d.x() * t) );
			I->hit.y() = float( ray_o.y() + (ray_d.y() * t) );
			I->hit.z() = float( ray_o.z() + (ray_d.z() * t) );

			return true;
		}

		// No hit, no win
		return false;
	}
	/**
	 * Get a normal vector (virtual)
	 * @return Return the normal vector of this triangle
	 */
	virtual VEC<float> getNormal() const {
		return Unit(cross(_v1 - _v0, _v2 - _v0));
	}
	/**
	 * Get a bounding box for this object (virtual)
	 * @return Return a bounding box class
	 */
	virtual BBox getBBox() const {
		float x_min = min3(_v0.x(), _v1.x(), _v2.x());
		float y_min = min3(_v0.y(), _v1.y(), _v2.y());
		float z_min = min3(_v0.z(), _v1.z(), _v2.z());

		float x_max = max3(_v0.x(), _v1.x(), _v2.x());
		float y_max = max3(_v0.y(), _v1.y(), _v2.y());
		float z_max = max3(_v0.z(), _v1.z(), _v2.z());

		return BBox(VEC<float>(x_min,y_min,z_min), VEC<float>(x_max,y_max,z_max));
	}
	/**
	 * Get the centroid for this object. (Used in BVH Sorting) (virtual)
	 * @return Return the center position of this triangle
	 */
	VEC<float> getCentroid()const{
		return (_v0 + _v1 + _v2) / 3;
	}
	/**
	 * Check the input triangle is equal or not?
	 * @param[in]   tri_in   (TRI class) input triangle class
	 * @return Return true if they are equal.
	 */
	bool Equal(const TRI<T>& tri_in){
		T sum = (_v0 - tri_in.V0()).abs() + (_v1 - tri_in.V1()).abs() + (_v2 - tri_in.V2()).abs();
		return (sum < 1E-15)? true:false;
	}
	/**
	 * Display all memeber variables on the console
	 */
	void Print()const{
		cout<<"+---------------------+"<<endl;
		cout<<"|   Triangle Class    |"<<endl;
		cout<<"+---------------------+"<<endl;
		T deg0 = (_ea[0] == -1)? -1:rad2deg(_ea[0]);
		T deg1 = (_ea[1] == -1)? -1:rad2deg(_ea[1]);
		T deg2 = (_ea[2] == -1)? -1:rad2deg(_ea[2]);
		cout<<" V0    = "; _v0.Print();
		cout<<" V1    = "; _v1.Print();
		cout<<" V2    = "; _v2.Print();
		cout<<" N     = "; Unit(cross(_v1 - _v0, _v2 - _v0)).Print();
		cout<<" EA0   = "<<deg0<<" [deg] -> factor = "<<2-_ea[0]/def::PI<<endl;
		cout<<" EA1   = "<<deg1<<" [deg] -> factor = "<<2-_ea[1]/def::PI<<endl;
		cout<<" EA2   = "<<deg2<<" [deg] -> factor = "<<2-_ea[2]/def::PI<<endl;
		cout<<" IDX           = "<<_idx<<endl;
		cout<<" IDX0 (v0->v1) = "<<_idx_near[0]<<endl;
		cout<<" IDX1 (v1->v2) = "<<_idx_near[1]<<endl;
		cout<<" IDX2 (v2->v0) = "<<_idx_near[2]<<endl;
	}
private:
	double _ea[3];			// Wedge angle, ea0 = {v0->v1}, ea1 = {v1->v2}, ea2 = {v2->v0}
	size_t _idx;			// Self triangle index w.r.t. cad._PL
	long _idx_near[3];	// Near 3 triangle index w.r.t. cad._PL
	VEC<T> _v0, _v1, _v2;	// Vertex position
	long _MatIdx;			// Material index
	// old
	double _Area;				// Area
	double _ca, _sa, _cb, _sb;	// cos(alpha), sin(alpha), cos(beta), sin(beta)

};



#endif

