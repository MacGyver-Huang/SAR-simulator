//
//  raytrace.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 1/28/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef raytrace_h
#define raytrace_h

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <basic/vec.h>
#include <bvh/obj.h>
#include <bvh/triangle.h>
#include <bvh/ray.h>
#include <bvh/bvh.h>

using namespace std;
using namespace vec;

namespace raytrace {
	/**
	 Find the intersection infomation by BVH(Bounding volume hierarchy) Ray tracing technique.
	 @param[in] bvh  [x] Bounding volume hierarchy class
	 @param[in] ray  [x] Incident ray class
	 @param[in] MaxLevel  [x] Max integer of bouncing level
	 @param[out] DistSet [x] Distance Set form incident Ray.o to intersection point
	 @param[out] RaySet [x] Ray class set for recording reflection ray
	 @param[out] ObjSet [x] Object class set for recoding each intersection triangle polygon
	 @param[out] N [m,m,m] Normal vector set of intersection triangle polygon
	 */
	void ObjectIntersection(const BVH& bvh, const Ray& ray, const long MaxLevel, 
							vector<double>& DistSet, vector<Ray>& RaySet, vector<Obj*>& ObjSet, vector<bool>& Shadow, vector<VEC<float> >& N,
							long& count, long k){
//	void ObjectIntersection(const BVH& bvh, const Ray& ray, const long MaxLevel,
//							vector<double>& DistSet, vector<Ray>& RaySet, vector<Obj*>& ObjSet, vector<VEC<float> >& N,
//							long& count, long k){
		// Intersection
		IntersectionInfo I, I_Shadow;
		bool hit;
		Ray rayRef(ray);
		VEC<double> org = ray.o;
		
		count = 0;
		while(count < MaxLevel){
//			if(k == 24920){
//				printf("\n\n+\n");
//				printf("[CPU] count=%d\n", count);
//				printf("[CPU](Before)    count=%d, MaxLevel=%d, rayRef.o=(%.10f,%.10f,%.10f), rayRef.d=(%.10f,%.10f,%.10f)\n", count, MaxLevel, rayRef.o.x(), rayRef.o.y(), rayRef.o.z(), rayRef.d.x(), rayRef.d.y(), rayRef.d.z());
//			}
//			hit = bvh.getIntersection(rayRef, &I, true);
			hit = bvh.getIntersection(rayRef, &I, false, k);
//			if(k == 208){
//				cout<<"+---------------------+"<<endl;
//				cout<<"|      raytrace       |"<<endl;
//				cout<<"+---------------------+"<<endl;
//				cout<<"k   = "<<k<<endl;
//				cout<<"hit = "<<hit<<endl;
//			}
//			if(k == 108){
//				rayRef.Print();
//				((TRI<float>*)(I.object))->n().Print();
//				char tmp_char = getchar();
//			}
//			if(k == 24920){
//				printf("[CPU](After)     count=%d, MaxLevel=%d, I.hit=(%.10f,%.10f,%.10f)\n", count, MaxLevel, I.hit.x(), I.hit.y(), I.hit.z());
//			}
//			if(k == 286 && count == 0){
//				printf("[CPU] k = %d, rayRef.o = [%.20f,%.20f,%.20f], I.hit = [%.20f,%.20f,%.20f]\n",
//						k, rayRef.o.x(), rayRef.o.y(), rayRef.o.z(), I.hit.x(), I.hit.y(), I.hit.z());
//			}

			if(hit == true){
				// Check Shadow
				// shadow ray
				VEC<double> uv_shadow = Unit(VEC<double>(org.x()-I.hit.x(), org.y()-I.hit.y(), org.z()-I.hit.z()));
				Ray rayShadow(I.hit, uv_shadow);
//				Shadow[count] = bvh.getIntersection(rayShadow, &I_Shadow, false, k);
				bool IsShadow = bvh.getIntersection(rayShadow, &I_Shadow, false, k);
				Shadow[count] = ( IsShadow && (I_Shadow.object != I.object) );

//				N.push_back(I.object->getNormal(I));	// Normal vector of intersection triangle
//				N[count] = I.object->getNormal(I);	// Normal vector of intersection triangle
				N[count] = I.object->getNormal();	// Normal vector of intersection triangle
				// Snell's law (no matter hit point is showdow or not)
				rayRef = rayRef.Reflection(I.hit, N[count]);
				// assignment
				DistSet[count] = I.t;
				RaySet[count]  = rayRef;
				ObjSet[count]  = (Obj*)I.object;

//				if(count == 0){
////					printf("k = %d, rayRef.d     = [%.20f,%.20f,%.20f]\n", k, rayRef.d.x(), rayRef.d.y(), rayRef.d.z());
////					printf("k = %d, rayRef.o     = [%.20f,%.20f,%.20f]\n", k, rayRef.o.x(), rayRef.o.y(), rayRef.o.z());
////					printf("k = %d, rayRef.inv_d = [%.20f,%.20f,%.20f]\n", k, rayRef.inv_d.x(), rayRef.inv_d.y(), rayRef.inv_d.z());
////					printf("k = %d, DistSet      = %.20f\n", k, DistSet[count]);
////					printf("k = %d, I.hit        = [%.20f,%.20f,%.20f]\n", k, I.hit.x(), I.hit.y(), I.hit.z());
////					printf("k = %d, N            = [%.20f,%.20f,%.20f]\n", k, N[count].x(), N[count].y(), N[count].z());
//////					((TRI<float>*)(I.object))->Print();
//////					printf("[CPU](After Ref) count=%d, MaxLevel=%d, I.hit   =(%.10f,%.10f,%.10f), N       =(%.10f,%.10f,%.10f)\n", count, MaxLevel, I.hit.x(), I.hit.y(), I.hit.z(), N[count].x(), N[count].y(), N[count].z());
//////					printf("[CPU](After Ref) count=%d, MaxLevel=%d, rayRef.o=(%.10f,%.10f,%.10f), rayRef.d=(%.10f,%.10f,%.10f)\n", count, MaxLevel, rayRef.o.x(), rayRef.o.y(), rayRef.o.z(), rayRef.d.x(), rayRef.d.y(), rayRef.d.z());
//////					printf("[CPU](After Ref) DistSet[%d]=%.10f\n", count, DistSet[count]);
////////					printf("[CPU](After Ref) RaySet[%d] =%.10f\n", count, RaySet[count]);
//////					printf("+\n\n\n");
//					printf("k = %d, I.hit = [%.20f,%.20f,%.20f], rayRef.o = [%.20f,%.20f,%.20f]\n",
//							k, I.hit.x(), I.hit.y(), I.hit.z(), k, rayRef.o.x(), rayRef.o.y(), rayRef.o.z());
//				}
				
//				N.push_back(I.object->getNormal(I));	// Normal vector of intersection triangle
				// Snell's law
//				rayRef = rayRef.Reflection(I.hit, N[N.size()-1]);
				// assignment
//				DistSet.push_back(I.t);
//				RaySet.push_back(rayRef);
//				ObjSet.push_back((Obj*)I.object);
				count++;
			}else{
//				if(k == 24920){
//					printf("+\n\n\n");
//				}


//				if(k == 25996){
//					printf("\n>\n");
//					printf("------------------------------------>MinLevel[%d] = %d\n", k, count);
//					if(count != 0){
//						for(int i=0;i<count;++i){
//							printf(">>>>>>>>>>>> k=%d, i=%d <<<<<<<<<<<<<<\n", k, i);
//							printf("Ray    = \n"); ray.Print();
//							printf("I.hit  = \n"); I.hit.Print();
//							printf("DisSet = %f\n", DistSet[i]);
//							printf("RaySet = \n"); RaySet[i].Print();
//							printf("ObjSet = \n"); ObjSet[i]->Print();
//						}
//					}
//
//					printf("<\n\n");
//				}



				break;
			}
		}
//		if(k == 2380){
//			printf("\n[CPU] k=%d, MinLevel[k]=%d\n", k, count);
//		}
	}
}


#endif
