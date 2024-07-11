//
//  ray.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 1/27/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef ray_h
#define ray_h

#include <basic/vec.h>

using namespace vec;

class Ray {
public:
	Ray(){};
	Ray(const VEC<double>& O, const VEC<double>& D);
//	Ray(const Ray& in):o(in.o),d(in.d),inv_d(in.inv_d){};

	void Print()const;
	// Mics.
	Ray Reflection(const VEC<double>& hitPoint, const VEC<double>& Normal)const;
public:
	VEC<double> o; // Ray Origin
	VEC<double> d; // Ray Direction
	VEC<double> inv_d; // Inverse of each Ray Direction component
};


Ray::Ray(const VEC<double>& O, const VEC<double>& D):o(O),d(D){
//	inv_d = VEC<float>(1,1,1) / d;
	inv_d.x() = 1.0/d.x();
	inv_d.y() = 1.0/d.y();
	inv_d.z() = 1.0/d.z();
}

void Ray::Print()const{
	cout<<"+----------------+"<<endl;
	cout<<"|   Ray Class    |"<<endl;
	cout<<"+----------------+"<<endl;
	cout<<" o     = "; o.Print();
	cout<<" d     = "; d.Print();
	cout<<" inv_d = "; inv_d.Print();
}

Ray Ray::Reflection(const VEC<double>& hitPoint, const VEC<double>& Normal)const{
//		cuVEC<double> d2((double)d.x, (double)d.y, (double)d.z);
//		cuVEC<double> N2((double)Normal.x, (double)Normal.y, (double)Normal.z);
		VEC<double> uv = d - 2.0*dot(d, Normal)*Normal;
//		VEC<double> uv = Unit(d - 2.0*dot(d, Normal)*Normal);


//		cuVEC<float> uv = Unit(d - 2.f*cu::dot(d, Normal)*Normal);
		return Ray(hitPoint, uv);
//	VEC<double> d2((double)d.x(), (double)d.y(), (double)d.z());
//	VEC<double> N2((double)Normal.x(), (double)Normal.y(), (double)Normal.z());
//	VEC<double> uv2 = Unit(d2 - 2.0*dot(d2, N2)*N2);
//	VEC<double>  uv((float)uv2.x(), (float)uv2.y(), (float)uv2.z());
//
////	VEC<float> uv = Unit(d - 2.f*dot(d, Normal)*Normal);
//	return Ray(hitPoint, uv);
}

#endif


////
////  ray.h
////  PhysicalOptics02
////
////  Created by Steve Chiang on 1/27/14.
////  Copyright (c) 2014 Steve Chiang. All rights reserved.
////
//
//#ifndef ray_h
//#define ray_h
//
//#include <basic/vec.h>
//
//using namespace vec;
//
//class Ray {
//public:
//	Ray(){};
//	Ray(const VEC<float>& O, const VEC<float>& D);
////	Ray(const Ray& in):o(in.o),d(in.d),inv_d(in.inv_d){};
//
//	void Print()const;
//	// Mics.
//	Ray Reflection(const VEC<float>& hitPoint, const VEC<float>& Normal)const;
//public:
//	VEC<float> o; // Ray Origin
//	VEC<float> d; // Ray Direction
//	VEC<float> inv_d; // Inverse of each Ray Direction component
//};
//
//
//Ray::Ray(const VEC<float>& O, const VEC<float>& D):o(O),d(D){
////	inv_d = VEC<float>(1,1,1) / d;
//	inv_d.x() = 1.0f/d.x();
//	inv_d.y() = 1.0f/d.y();
//	inv_d.z() = 1.0f/d.z();
//}
//
//void Ray::Print()const{
//	cout<<"+----------------+"<<endl;
//	cout<<"|   Ray Class    |"<<endl;
//	cout<<"+----------------+"<<endl;
//	cout<<" o     = "; o.Print();
//	cout<<" d     = "; d.Print();
//	cout<<" inv_d = "; inv_d.Print();
//}
//
//Ray Ray::Reflection(const VEC<float>& hitPoint, const VEC<float>& Normal)const{
//
//	VEC<double> d2((double)d.x(), (double)d.y(), (double)d.z());
//	VEC<double> N2((double)Normal.x(), (double)Normal.y(), (double)Normal.z());
//	VEC<double> uv2 = Unit(d2 - 2.0*dot(d2, N2)*N2);
//	VEC<float>  uv((float)uv2.x(), (float)uv2.y(), (float)uv2.z());
//
////	VEC<float> uv = Unit(d - 2.f*dot(d, Normal)*Normal);
//	return Ray(hitPoint, uv);
//}
//
//#endif
