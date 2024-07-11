//
//  stl.h
//  PhysicalOptics01
//
//  Created by Steve Chiang on 1/24/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef stleach_h
#define stleach_h

#include <basic/vec.h>

using namespace vec;


class STLEach{
public:
	void Print(){
		cout<<"N   = "; N.Print();
		cout<<"V1  = "; V1.Print();
		cout<<"V2  = "; V2.Print();
		cout<<"V3  = "; V3.Print();
		cout<<"att = "<<att<<endl;
	}
	template<typename T>
	void Translate(const VEC<T>& move){
		N  += move;
		V1 += move; V2 += move; V3 += move;
	}
public:
	VEC<float> N;
	VEC<float> V1, V2, V3;
	unsigned int att;
};

#endif
