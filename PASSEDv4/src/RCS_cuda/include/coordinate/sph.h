//
//  sph.h
//  PhysicalOptics13
//
//  Created by Steve Chiang on 4/21/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//
#ifndef SPH_H_INCLUDED
#define SPH_H_INCLUDED

//#include <basic/def_func.h>
#include <sar/def.h>

//using namespace def_func;

namespace sph {
	template<typename T>
	class SPH {
	public:
		SPH(){}
		SPH(const T R, const T Theta, const T Phi){
			_r = R;
			_theta = Theta;
			_phi = Phi;
		}
		SPH(const SPH<T>& in){
			_r = in._r;
			_theta = in._theta;
			_phi = in._phi;
		}
		SPH<T>& operator=(const SPH<T>& in){
			if(this != &in){
				_r = in._r;
				_theta = in._theta;
				_phi = in._phi;
			}
			return *this;
		}
		const T& R()const{return _r;}
		const T& Theta()const{return _theta;}
		const T& Phi()const{return _phi;}
		T& R(){return _r;}
		T& Theta(){return _theta;}
		T& Phi(){return _phi;}
		void Print(){
			cout<<"+----------------+"<<endl;
			cout<<"|       SPH      |"<<endl;
			cout<<"+----------------+"<<endl;
			cout<<std::setprecision(10);
			cout<<"R     = "<<_r<<" [m]"<<endl;
			cout<<"Theta = "<<_theta*def::DTR<<" [deg]"<<endl;
			cout<<"Phi   = "<<_phi*def::DTR<<" [deg]"<<endl;
		}
	private:
		T _r, _theta, _phi;
	};

}

#endif // SPH_H_INCLUDED

