//
//  sphere.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 1/27/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef sphere_h
#define sphere_h


#include <cmath>
#include "obj.h"

//! For the purposes of demonstrating the BVH, a simple sphere
struct Sphere : public Obj {
	VEC<float> center; // Center of the sphere
	float r, r2; // Radius, Radius^2
	
	Sphere(const VEC<float>& center, float radius) : center(center), r(radius), r2(radius*radius) { }
	
	bool getIntersection(const Ray& ray, IntersectionInfo* I) const {
		VEC<float> s = center - ray.o;
		float sd = dot(s, ray.d);
		float ss = dot(s, s);
		
		// Compute discriminant
		float disc = sd*sd - ss + r2;
		
		// Complex values: No intersection
		if( disc < 0.f ) return false;
		
		// Assume we are not in a sphere... The first hit is the lesser valued
		I->object = this;
		I->t = sd - sqrt(disc);
		return true;
	}
	
	VEC<float> getNormal(const IntersectionInfo& I) const {
		return Unit(I.hit - center);
	}
	
	BBox getBBox() const {
		return BBox(center-VEC<float>(r,r,r), center+VEC<float>(r,r,r));
	}
	
	VEC<float> getCentroid() const {
		return center;
	}
	
};


#endif
