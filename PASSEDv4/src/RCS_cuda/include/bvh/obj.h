//
//  obj.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 1/27/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef obj_h
#define obj_h

#include "intersectioninfo.h"
#include "ray.h"
#include "bbox.h"

class Obj {
public:
	//! All "Objects" must be able to test for intersections with rays.
	virtual bool getIntersection(const Ray& ray, IntersectionInfo* intersection) const = 0;
//	virtual bool getIntersection(const Ray& ray, IntersectionInfo* intersection, long k) const = 0;
	
	//! Return an object normal based on an intersection
//	virtual VEC<float> getNormal(const IntersectionInfo& I) const = 0;
	virtual VEC<float> getNormal() const = 0;
	
	//! Return a bounding box for this object
	virtual BBox getBBox() const = 0;
	
	//! Return the centroid for this object. (Used in BVH Sorting)
	virtual VEC<float> getCentroid() const = 0;

	//! Print
	virtual void Print() const = 0;
};



#endif
