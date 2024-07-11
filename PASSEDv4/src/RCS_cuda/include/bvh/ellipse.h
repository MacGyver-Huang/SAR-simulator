//
//  ellipse.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 3/15/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef ellipse_h
#define ellipse_h


#include <cmath>
#include "obj.h"


template<typename T>
class ELLIPSE : public Obj{
public:
	ELLIPSE(){
		Center = VEC<T>(0,0,0);
		Rad = Rad2 = VEC<T>(1,1,1);
	}
	ELLIPSE(const string datum_name){
		if(datum_name == "WGS84"){
			double Ea = 6378137;
			double Eb = 6356752.3142;
			Center = VEC<T>(0,0,0);
			Rad = VEC<T>(Ea,Ea,Eb);
			Rad2.x() = Rad.x() * Rad.x();
			Rad2.y() = Rad.y() * Rad.y();
			Rad2.z() = Rad.z() * Rad.z();
		}
	}
	ELLIPSE(const VEC<T>& O, const VEC<T>& RAD){
		Center = O;
		Rad = RAD;
		Rad2.x() = Rad.x() * Rad.x();
		Rad2.y() = Rad.y() * Rad.y();
		Rad2.z() = Rad.z() * Rad.z();
	}
	bool getIntersection(const Ray& ray, IntersectionInfo* I){
		VEC<double> ro = ray.o - Center;
		VEC<double> N  = ray.d;
		double a = Square(N.x()) / Rad2.x() + Square(N.y()) / Rad2.y() + Square(N.z()) / Rad2.z();
		double b = 2*ro.x()*N.x()/ Rad2.x() + 2*ro.y()*N.y()/ Rad2.y() + 2*ro.z()*N.z()/ Rad2.z();
		double c = Square(ro.x())/ Rad2.x() + Square(ro.y())/ Rad2.y() + Square(ro.z())/ Rad2.z() - 1;
		
		double d = ((b*b)-(4*a*c));
		if(d < 0){
			return false;
		}else{
			d = sqrt(d);
		}
		double dis1 = (-b + d)/(2*a);
		double dis2 = (-b - d)/(2*a);
		
		if(dis1 < dis2){
			I->t = dis1;
			I->hit = ray.o + (ray.d * dis1);
		}else{
			I->t = dis2;
			I->hit = ray.o + (ray.d * dis2);
		}
		
		I->object = this;
		return true;
	}
	VEC<float> getNormal(const IntersectionInfo& I) const {
		return Unit(I.hit - Center);
	}
	BBox getBBox() const {
		return BBox(Center-Rad, Center+Rad);
	}
	VEC<float> getCentroid() const {
		return Center;
	}
private:
	VEC<T> Center;	// Center of the ellipse
	VEC<T> Rad;		// Radius
	VEC<T> Rad2;	// Square of Radius
};


#endif
