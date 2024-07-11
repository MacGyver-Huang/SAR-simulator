//
//  roughness.h
//  PhysicalOptics17_cuda
//
//  Created by Steve Chiang on Jan 27, 2015
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//
#ifndef ROUGHNESS_H_
#define ROUGHNESS_H_

#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

namespace roughness {
	class Roughness {
	public:
		void Roughness(const double H_rms, const double L_corr){
			h_rms  = H_rms;
			l_corr = vector<double>[2];
			l_corr[0] = L_corr[0];
			l_corr[1] = L_corr[1];
		}
		void ~Roughness(){}
		const double H_rms()const{ return h_rms; }
		const vector<double> L_corr()const{ return l_corr; }
	private:
		double h_rms;			// RMS height
		vector<double> l_corr;	// Correlation length
	};
}  // namespace roughness





#endif /* ROUGHNESS_H_ */
