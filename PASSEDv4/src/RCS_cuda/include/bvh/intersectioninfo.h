//
//  intersectioninfo.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 1/27/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef intersectioninfo_h
#define intersectioninfo_h

#include <basic/vec.h>

using namespace vec;

class Obj;

struct IntersectionInfo {
	double t; // Intersection distance along the ray
	const Obj* object; // Object that was hit
	VEC<float> hit; // Location of the intersection
};


#endif
