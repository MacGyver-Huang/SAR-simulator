//
//  aero.h
//  SARPro
//
//  Created by Chiang Steve on 6/20/12.
//  Copyright (c) 2012 NCU. All rights reserved.
//

#ifndef aero_h
#define aero_h

#include <iostream>
#include <cmath>
#include "sar.h"

using namespace std;
using namespace sar;



namespace aero{
		
	struct StructNOISE {
		D1<VEC<double> > ECR;
		D1<RPY<double> > RPY;
		D1<double> V;
		D1<int> seed;
	};
	
	struct StructAxRPY {
		D1<VEC<double> > AxRl, AxPi, AxYw;
	};
	
	struct StructLPV {
		D1<VEC<double> > LOS, PAL, PER, VEL;
	};
	
	struct StructTrajectory {
		double dt;	// time interval
		D1<VEC<double> > ecr_org, ecr_noise;	// ECR for original and noise
		D1<VEC<double> > vel_org, vel_noise;				// Velocity for original and noise
		StructAxRPY RPY_org, RPY_noise;			// RPY for original and noise (x:Roll, y:Pitch, z:Yaw)
		StructLPV LPV_org, LPV_noise;			// LPV for original and noise
	};
	
	void CreateNoise(const long num, const VEC<double> ECRStd, const VEC<double> RPYStd, const double VStd, StructNOISE& Noise);
	
	void PathTrajectoryToSV(const StructTrajectory& Trj, SV<double>& SV_org, SV<double>& SV_noise);
	
	void PathTrajectorySim(const GEO<double>& LLH_s, const GEO<double>& LLH_e, const double Vs, const double theta_l,
						   const double theta_AEB, const double theta_sq, const def::ORB Datum, const StructNOISE& Noise, 
						   SV<double>& SV_org, SV<double>& SV_noise, StructTrajectory& Trj);
}


// Implements
void aero::CreateNoise(const long num, const VEC<double> ECRStd, const VEC<double> RPYStd, const double VStd,
					   aero::StructNOISE& Noise){
	Noise.ECR = D1<VEC<double> >(num);
	Noise.RPY = D1<RPY<double> >(num);
	Noise.V   = D1<double>(num);
	Noise.seed= D1<int>(7);
	for(int i=0;i<Noise.seed.GetNum()-1L;++i){ Noise.seed[i] = 111+i+900; }
	
	// ECR
	D1<double> ECR_x = mat::Randomu<double>(num, Noise.seed[0]) * ECRStd.x()/(2.0*sqrt(3.0));	
	D1<double> ECR_y = mat::Randomu<double>(num, Noise.seed[1]) * ECRStd.y()/(2.0*sqrt(3.0));
	D1<double> ECR_z = mat::Randomu<double>(num, Noise.seed[2]) * ECRStd.z()/(2.0*sqrt(3.0));
	for(long i=0;i<num;++i){
		Noise.ECR[i].x() = ECR_x[i];
		Noise.ECR[i].y() = ECR_y[i];
		Noise.ECR[i].z() = ECR_z[i];
	}
	ECR_x.clear();
	ECR_y.clear();
	ECR_z.clear();
	
	// RPY
	D1<double> RPY_x = mat::Randomu<double>(num, Noise.seed[3]) * RPYStd.x()/2.0;
	D1<double> RPY_y = mat::Randomu<double>(num, Noise.seed[4]) * RPYStd.y()/2.0;
	D1<double> RPY_z = mat::Randomu<double>(num, Noise.seed[5]) * RPYStd.z()/2.0;
	for(long i=0;i<num;++i){
		Noise.RPY[i].r() = RPY_x[i];
		Noise.RPY[i].p() = RPY_y[i];
		Noise.RPY[i].y() = RPY_z[i];
	}
	RPY_x.clear();
	RPY_y.clear();
	RPY_z.clear();
	
	// Velocity
	Noise.V = mat::Randomu<double>(num, Noise.seed[6]) * VStd/2.0;
}

void aero::PathTrajectoryToSV(const StructTrajectory& Trj, SV<double>& SV_org, SV<double>& SV_noise){
	long num = Trj.ecr_org.GetNum();
	D1<double> t = linspace(0.0, double(num)-1.0, 1.0) * Trj.dt;
	SV_org   = SV<double>(num);
	SV_noise = SV<double>(num);
	SV_org.SetAll(   t, Trj.ecr_org,   Trj.vel_org,   num);
	SV_noise.SetAll( t, Trj.ecr_noise, Trj.vel_noise, num);
}

void aero::PathTrajectorySim(const GEO<double>& LLH_s, const GEO<double>& LLH_e, const double Vs, const double theta_l,
							 const double theta_AEB, const double theta_sq, const def::ORB Datum, const StructNOISE& Noise, 
							 SV<double>& SV_org, SV<double>& SV_noise, StructTrajectory& Trj){
	//=================================================================
	// Find position (independent)
	//=================================================================
	// Position
	long num = Noise.ECR.GetNum();
	D1<double> lon(num), lat(num), h(num);
	linspace(LLH_s.lon(), LLH_e.lon(), lon);
	linspace(LLH_s.lat(), LLH_e.lat(), lat);
	linspace(LLH_s.h(),   LLH_e.h(),   h);
	
	// Transfer form geodetic to ECR
	Trj.ecr_org = D1<VEC<double> >(num);
	for(long i=0;i<num;++i){
		Trj.ecr_org[i] = Gd2ECR(GEO<double>(lon[i], lat[i], h[i]), Datum);
	}
	
	Trj.ecr_noise = Trj.ecr_org + Noise.ECR;
	Trj.dt = (Trj.ecr_org[1] - Trj.ecr_org[0]).abs() / Vs;
	
	
	//=================================================================
	// Allocated memory
	//=================================================================
	// RPY
	Trj.RPY_org.AxRl   = D1<VEC<double> >(num);
	Trj.RPY_org.AxPi   = D1<VEC<double> >(num);
	Trj.RPY_org.AxYw   = D1<VEC<double> >(num);
	Trj.RPY_noise.AxRl = D1<VEC<double> >(num);
	Trj.RPY_noise.AxPi = D1<VEC<double> >(num);
	Trj.RPY_noise.AxYw = D1<VEC<double> >(num);
	// LPV
	Trj.LPV_org.LOS   = D1<VEC<double> >(num);
	Trj.LPV_org.PAL   = D1<VEC<double> >(num);
	Trj.LPV_org.PER   = D1<VEC<double> >(num);
	Trj.LPV_org.VEL   = D1<VEC<double> >(num);
	Trj.LPV_noise.LOS = D1<VEC<double> >(num);
	Trj.LPV_noise.PAL = D1<VEC<double> >(num);
	Trj.LPV_noise.PER = D1<VEC<double> >(num);
	Trj.LPV_noise.VEL = D1<VEC<double> >(num);
	// Velocity for SV
	Trj.vel_org       = D1<VEC<double> >(num);
	Trj.vel_noise     = D1<VEC<double> >(num);
	
	//=================================================================
	// Find attitude (independent)
	//=================================================================
	VEC3MAT<double> M;
	GEO<double> gd0, gd1;
	for(long i=0;i<num-1L;++i){
		// Find AxRl_org
		gd0 = ECR2Gd(Trj.ecr_org[i],   Datum);
		gd1 = ECR2Gd(Trj.ecr_org[i+1], Datum);
		gd1.h() = gd0.h();
		Trj.RPY_org.AxRl[i] = Unit( Gd2ECR(gd1, Datum) - Gd2ECR(gd0, Datum) );
		// Original
		Trj.RPY_org.AxYw[i] = Unit(Trj.ecr_org[i]);
		Trj.RPY_org.AxPi[i] = Unit(cross(Trj.RPY_org.AxRl[i],Trj.RPY_org.AxYw[i]));
		// Refind AxRl_org
		Trj.RPY_org.AxRl[i] = Unit(cross(Trj.RPY_org.AxYw[i],Trj.RPY_org.AxPi[i]));
		// Noise
		// Rotate with AxYw
		Trj.RPY_noise.AxRl[i] = vec::find::ArbitraryRotate(Trj.RPY_org.AxRl[i], Noise.RPY[i].y(), Trj.RPY_org.AxYw[i], M);
		Trj.RPY_noise.AxPi[i] = Multiply(M, Trj.RPY_org.AxPi[i]);
		Trj.RPY_noise.AxYw[i] = Trj.RPY_noise.AxYw[i];
		// Rotate with AxPi
		Trj.RPY_noise.AxRl[i] = vec::find::ArbitraryRotate(Trj.RPY_noise.AxRl[i], Noise.RPY[i].p(), Trj.RPY_noise.AxPi[i], M);
		Trj.RPY_noise.AxYw[i] = vec::Multiply(M, Trj.RPY_noise.AxYw[i]);
		// Rotate with AxRl
		Trj.RPY_noise.AxPi[i] = vec::find::ArbitraryRotate(Trj.RPY_noise.AxPi[i], Noise.RPY[i].r(), Trj.RPY_noise.AxRl[i], M);
		Trj.RPY_noise.AxYw[i] = Multiply(M, Trj.RPY_noise.AxYw[i]);
	}
	// Oringinal
	Trj.RPY_org.AxRl[num-1] = Trj.RPY_org.AxRl[num-2];
	Trj.RPY_org.AxYw[num-1] = Trj.RPY_org.AxYw[num-2];
	Trj.RPY_org.AxPi[num-1] = Trj.RPY_org.AxPi[num-2];
	// Noise
	Trj.RPY_noise.AxRl[num-1] = Trj.RPY_noise.AxRl[num-2];
	Trj.RPY_noise.AxYw[num-1] = Trj.RPY_noise.AxYw[num-2];
	Trj.RPY_noise.AxPi[num-1] = Trj.RPY_noise.AxPi[num-2];
	
	//=================================================================
	// Find LPV (independent)
	//=================================================================
	// initilize
	// original
	Trj.LPV_org.VEL =  Trj.RPY_org.AxRl;
	for(long i=0L;i<num;++i){
		Trj.LPV_org.PAL[i].Setxyz( -Trj.RPY_org.AxYw[i].x(),
								   -Trj.RPY_org.AxYw[i].y(),
								   -Trj.RPY_org.AxYw[i].z());
		Trj.LPV_org.PER[i] = vec::cross(Trj.LPV_org.PAL[i], Trj.LPV_org.VEL[i]);
	}
	// noise
	Trj.LPV_noise.VEL =  Trj.RPY_noise.AxRl;
	for(long i=0L;i<num;++i){
		Trj.LPV_noise.PAL[i].Setxyz( -Trj.RPY_noise.AxYw[i].x(),
									 -Trj.RPY_noise.AxYw[i].y(),
									 -Trj.RPY_noise.AxYw[i].z());
		Trj.LPV_noise.PER[i] = vec::cross(Trj.LPV_noise.PAL[i], Trj.LPV_noise.VEL[i]);
	}
	for(long i=0L;i<num;++i){
		// Original
		Trj.LPV_org.PAL[i] = vec::find::ArbitraryRotate(Trj.LPV_org.PAL[i], -theta_l, Trj.RPY_org.AxRl[i], M);
		Trj.LPV_org.PER[i] = Unit( Multiply(M, Trj.LPV_org.PER[i]) );
		Trj.LPV_org.VEL[i] = Unit( cross(Trj.LPV_org.PAL[i], Trj.LPV_org.PER[i]) );
		Trj.LPV_org.LOS[i] = vec::find::ArbitraryRotate(Trj.LPV_org.PAL[i], theta_sq, Trj.LPV_org.PER[i]);
		// Noise
		Trj.LPV_noise.PAL[i] = vec::find::ArbitraryRotate(Trj.LPV_noise.PAL[i], -theta_l, Trj.RPY_noise.AxRl[i], M);
		Trj.LPV_noise.PER[i] = Unit( Multiply(M, Trj.LPV_noise.PER[i]) );
		Trj.LPV_noise.VEL[i] = Unit( cross(Trj.LPV_noise.PAL[i], Trj.LPV_noise.PER[i]) );
		Trj.LPV_noise.LOS[i] = vec::find::ArbitraryRotate(Trj.LPV_noise.PAL[i], theta_sq, Trj.LPV_noise.PER[i]);
	}
	
	//=================================================================
	// Find Velocity (independent)
	//=================================================================
	for(long i=0;i<num;++i){
		Trj.vel_org[i]   = Trj.LPV_org.VEL[i]  * Vs;
		Trj.vel_noise[i] = Trj.LPV_noise.VEL[i] * (Vs + Noise.V[i]);
	}
	
	//=================================================================
	// Convert to SV
	//=================================================================
	aero::PathTrajectoryToSV(Trj, SV_org, SV_noise);
	
}


#endif
