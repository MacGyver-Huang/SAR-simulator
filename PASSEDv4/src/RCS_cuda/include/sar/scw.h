//
//  scw.h
//  SARSCWPro
//
//  Created by Chiang Steve on 6/4/12.
//  Copyright (c) 2012 NCU. All rights reserved.
//

#ifndef SARSCWPro_scw_h
#define SARSCWPro_scw_h

#include "sar.h"
#include "vec.h"
using namespace sar;


namespace scw {
	typedef struct struct_sar {
		double c,fc,BW_rg,Tr,fs,dm,theta_az,theta_ev,theta_lc;
		double Kr,lambda,h,R0g_max_chamber;
		D1<double> Raz, XYZStd, R0_range;
	} StructSAR;
	
	StructSAR Init();
	
	D1<VEC<double> > InitTargetXYZ(const double X[], const double Y[], const double Z[], const long num_k);
	
	D1<VEC<double> > SCWPathTrajectory(const scw::StructSAR sar, D1<VEC<double> >& Ps_plane);
	
	void SARSim(const StructSAR sar, const D1<VEC<double> >& XYZ, const D1<VEC<double> >& Ps,
				D2<CPLX<double> >& Src, D1<double>& t, D1<double>& m, D2<CPLX<double> >& Sif,
				D1<double>& R0, D1<double>& idx_range);
	
	void MoCo1(const D2<CPLX<double> >& Src, const StructSAR sar, const D1<double>& R0,
			   D1<VEC<double> >& Ps, D1<VEC<double> >& Ps_plane, const double KaiserBeta,
			   D2<CPLX<double> >& Smoco1, D1<VEC<double> >& d, D1<double> d_LOS_ref);
	
	void MoCo2(const D2<CPLX<double> >& Smoco1, const StructSAR sar, const D1<double> d_LOS_ref,
			   const D1<VEC<double> >& d, const D1<double>& R0, const double KaiserBeta, D2<CPLX<double> >& Smoco2);
	
	void CSA(const StructSAR sar, const D2<CPLX<double> >& Src, const D1<double>& R0,
			 D2<CPLX<double> >& Sac);
	
	void RDA(const StructSAR sar, const D2<CPLX<double> >& Src, const D1<double>& R0,
			 D2<CPLX<double> >& Sac);
}

scw::StructSAR scw::Init(){
	// System parameters
	StructSAR sar;
	sar.c  = 2.998E8;				//[m/s] Light speed 
	sar.fc = 90E9;					//[Hz] Carrier freq.
	sar.BW_rg = 13E9;				//[Hz] Transmitted LFM bandwidth
	sar.Tr = 0.05E-6;				//[sec] Duration time
	sar.fs = 15E9;					//[Hz] Recivied ADC sampling rate
	sar.dm = 0.004;					//[m] Distance interval at azimuth
	sar.theta_az = def_func::deg2rad(15.);	//[rad] beamwidth at azimuth
	sar.theta_ev = def_func::deg2rad(35.);	//[rad] beamwidth at elvation
	sar.theta_lc = def_func::deg2rad(50.);	//[rad] look angle of main beam
	sar.h = 1.5;					//[m] Sensor height
	sar.R0g_max_chamber = 5;		//[m] max ground range length (by chamber)
	sar.XYZStd = D1<double>(3);		//[m] XYZ position noise STD
	sar.XYZStd[0] = 0.1;			//
	sar.XYZStd[1] = 0.0;			//
	sar.XYZStd[2] = 0.1;			//
	sar.Raz = D1<double>(2);		//[m] azimuth slow time
	sar.Raz[0] = -1.0;				//
	sar.Raz[1] = 1.0;				//
	
	// Derive
	sar.Kr = sar.BW_rg/sar.Tr;		//[Hz] Chirp rate
	sar.lambda = sar.c/sar.fc;		//[m] wavelength
	
	// Range issue
	//double R0g_min = sar.h*tan(sar.theta_lc - sar.theta_ev/2.0);	//[m] max ground range of targets'
	double R0g_max = sar.h*tan(sar.theta_lc + sar.theta_ev/2.0);	//[m] min ground range of targets'
	double R0_min  = sar.h/cos(sar.theta_lc - sar.theta_ev/2.0);	//[m] max slant range of targets'
	double R0_max  = sar.h/cos(sar.theta_lc + sar.theta_ev/2.0);	//[m] min slant range of targets'
	
	
	if(R0g_max > sar.R0g_max_chamber){
		cout<<"ERROR::Out of chamber range!!"<<endl;
		cout<<" R0g_max = "<<R0g_max<<" [m] > "<<endl;
		cout<<" R0_max_chamber = "<<sar.R0g_max_chamber<<" [m]"<<endl;
		exit(0);
	}
	
	sar.R0_range = D1<double>(2);		//[m] Slant range boundary
	sar.R0_range[0] = R0_min;
	sar.R0_range[1] = R0_max;
	return sar;
};

D1<VEC<double> > scw::InitTargetXYZ(const double X[], const double Y[], const double Z[], const long num_k){
	D1<VEC<double> > xyz(num_k);
	for(long i=0;i<num_k;++i){
		xyz[i] = vec::VEC<double>(X[i],Y[i],Z[i]);
	}
	return xyz;
}


D1<VEC<double> > scw::SCWPathTrajectory(const scw::StructSAR sar, D1<VEC<double> >& Ps_plane){
	D1<double> m = def_func::linspace(sar.Raz[0], sar.Raz[1], sar.dm);
	long num_az = m.GetNum();
	
	//===============================================================
	// Create sensor position with noise
	//===============================================================
	// Noise
	D1<VEC<double> > XYZ_noise(num_az);
	D1<double> tmpX = mat::Randomu<double>(num_az, 11u) * (sar.XYZStd[0]/(2.0*sqrt(3.0)));
	D1<double> tmpY = mat::Randomu<double>(num_az, 12u) * (sar.XYZStd[1]/(2.0*sqrt(3.0)));
	D1<double> tmpZ = mat::Randomu<double>(num_az, 13u) * (sar.XYZStd[2]/(2.0*sqrt(3.0)));
	
	for(long j=0;j<num_az;++j){
		XYZ_noise[j] = vec::VEC<double>(tmpX[j],tmpY[j],tmpZ[j]);
	}
	// sensor position
	Ps_plane = D1<VEC<double> >(num_az);
	D1<VEC<double> > Ps(num_az);
	for(long j=0L;j<num_az;++j){
		Ps_plane[j] = vec::VEC<double>(0.0,m[j],sar.h);
		Ps[j] = Ps_plane[j] + XYZ_noise[j];
	}
		
	return Ps;
}

void scw::SARSim(const StructSAR sar, const D1<VEC<double> >& XYZ, const D1<VEC<double> >& Ps,
				 D2<CPLX<double> >& Src, D1<double>& t, D1<double>& m, D2<CPLX<double> >& Sif, 
				 D1<double>& R0, D1<double>& idx_range){
	// fast time(t) & azimuth position(m)
	t = def_func::linspace(0.0,sar.Tr,1/sar.fs);
	long num_t = t.GetNum();
	m = def_func::linspace(sar.Raz[0],sar.Raz[1],sar.dm);
	
	// Find slant range
	D1<double> ft(num_t);
	def_func::linspace(-sar.fs/2,sar.fs/2,ft);
	
	D1<double> range = ft * sar.c/(2.0*sar.Kr);
	
	// Find effect slant range sample	
	long idx_min = 0L, idx_max = 0L;
	for(long i=0L;i<num_t-1;++i){
		if( (sar.R0_range[0] >= range[i]) && (sar.R0_range[0] <= range[i+1]) ){ idx_min = i+1; }
		if( (sar.R0_range[1] >= range[i]) && (sar.R0_range[1] <= range[i+1]) ){	idx_max = i+1; }
	}
	long num_rg = idx_max - idx_min + 1L;	//[pix] number of slant range sample
	long num_az = m.GetNum();				//[pix] number of azimuth sample

	
	//===============================================================
	// Simulate IF signal
	//===============================================================
	long num_k = XYZ.GetNum();
	Sif = D2<CPLX<double> > (num_az, num_t);
	VEC<double> Pt;
	double Rn, theta_sq_sqc;
	Sif.SetZero();
	
	for(long j=0;j<num_az;++j){
		for(long k=0;k<num_k;++k){
			Pt = XYZ[k];
			// Slant range
			Rn = (Ps[j] - Pt).abs();
			// Find instantaneous squint angle
			theta_sq_sqc = def_func::deg2rad(90.0) - 
						   vec::angle( (Pt - Ps[j]), vec::VEC<double>(0.0,1.0,0.0));
			// Echo signal
			for(long i=0;i<num_t;++i){
				Sif[j][i] = Sif[j][i] + Waz(theta_sq_sqc, sar.theta_az) * 
							exp(CPLX<double>( 0.0, 2*def::PI*( sar.fc*2*Rn/sar.c + sar.Kr*t[i]*2*Rn/sar.c - 
							(2.0*sar.Kr/(sar.c*sar.c))*Rn*Rn ) ));
			}
		}
		if(mat::Mod(j, 100) == 0){ cout<<j<<endl; }
	}
	R0 = D1<double>(idx_max - idx_min + 1L);
	for(long i=0;i<R0.GetNum();++i){ R0[i] = range[idx_min+i]; }	// [m] slant range
	idx_range = D1<double>(2);
	idx_range[0] = idx_min;
	idx_range[1] = idx_max;
	
	//===============================================================
	// Range compression
	//===============================================================
	Src = D2<CPLX<double> >(num_az,num_rg);
	D1<CPLX<double> > tmp(num_t);
	for(long j=0;j<num_az;++j){
		tmp = Sif.GetRow(j);
		mat::FFT(tmp);
		mat::fftshift(tmp);
		for(long i=0;i<num_rg;++i){
			Src[j][i] = tmp[idx_min+i];
		}
	}
}

void scw::MoCo1(const D2<CPLX<double> >& Src, const StructSAR sar, const D1<double>& R0,
				D1<VEC<double> >& Ps, D1<VEC<double> >& Ps_plane, const double KaiserBeta,
				D2<CPLX<double> >& Smoco1, D1<VEC<double> >& d, D1<double> d_LOS_ref){
	long num_rg = Src.GetN();
	long num_az = Src.GetM();
	//===============================================================
	// 1st order MoCo (bulk scaling)
	//===============================================================
	// Restore to pseudo echo signal
	Smoco1 = Src;
	sar::fft::RangeShift(Smoco1);
	sar::fft::RangeInverse(Smoco1);
	
	// Find displacement vector
	d = D1<VEC<double> >(num_az);
	for(long j=0;j<num_az;++j){
		d[j] = Ps[j] - Ps_plane[j];
	}
	
	// Find scene center point
	VEC<double> Pc(R0[num_rg/2],0.0,0.0);			// slant range plane
	Pc.x() = sqrt(Square(Pc.x()) - Square(sar.h));	// ground range plane
	
	// Find scene center look angle
	double theta_l_Pc = atan(Pc.x()/sar.h);
	
	// Definition of LPV coordinate unit vector (from Ps_plane)
	VEC<double> U_m(0.0,1.0,0.0);
	VEC<double> U_LOS = vec::find::ArbitraryRotate(VEC<double>(0.0,0.0,-1.0), -theta_l_Pc, U_m);
	VEC<double> U_per = vec::cross(U_LOS, U_m);
	
	// Find LOS distance
	D1<double> d_m(num_az), d_LOS(num_az), d_per(num_az);
	for(long j=0L;j<num_az;++j){
		d_m[j] = vec::dot(d[j], U_m);
		d_LOS[j] = -vec::dot(d[j], U_LOS);
		d_per[j] = vec::dot(d[j], U_per);
	}
	
	d_LOS_ref = d_LOS;
	
	
	// 1st MoCo
	D1<double> new_t(num_rg);
	def_func::linspace(0.0, sar.Tr, new_t);
	D1<double> win = sar::Kaiser(num_rg, KaiserBeta);
	
	double phs;
	for(long j=0L;j<num_az;++j){
		for(long i=0L;i<num_rg;++i){
			phs = 2.0*def::PI*( sar.fc*2.0*d_LOS_ref[j]/sar.c  +  sar.Kr*new_t[i]*2.0*d_LOS_ref[j]/sar.c);
			Smoco1[j][i] = Smoco1[j][i] * win[i] * mat::exp( (CPLX<double>(0.0,phs)).conj() );
		}
	}
	
	sar::fft::RangeForward(Smoco1);
	sar::fft::RangeShift(Smoco1);
}

void scw::MoCo2(const D2<CPLX<double> >& Smoco1, const StructSAR sar, const D1<double> d_LOS_ref,
				const D1<VEC<double> >& d, const D1<double>& R0, const double KaiserBeta, D2<CPLX<double> >& Smoco2){

	long num_rg = Smoco1.GetN();
	long num_az = Smoco1.GetM();
	
	// Restore to pseudo echo signal
	Smoco2 = Smoco1;
	
	VEC<double> U_m(0.0,1.0,0.0);
	
	// Find displacement at each slant range
	D2<double> d_LOS(num_az,num_rg);
	// d_LOS - d_LOS_ref
	D2<double> diff_LOS(num_az,num_rg);
	
	VEC<double> Pc, U_LOS;
	double theta_l_Pc;
	for(long i=0L;i<num_rg;++i){
		// Find scene center point
		Pc = VEC<double>(R0[i],0.0,0.0);				// Slant range plane
		Pc.x() = sqrt(Square(Pc.x()) - Square(sar.h));	// ground range plane
		
		// find scene center look angle
		theta_l_Pc = atan(Pc.x()/sar.h);
		
		// Definition of LPV coordinate unit vector (from Ps_plane)
		U_LOS = vec::find::ArbitraryRotate(VEC<double>(0.0,0.0,-1.0), -theta_l_Pc, U_m);
		
		// find LOS distance
		for(long j=0L;j<num_az;++j){
			d_LOS[j][i] = -vec::dot(d[j], U_LOS);
			diff_LOS[j][i] = d_LOS[j][i] - d_LOS_ref[j];
		}
	}
	
	// 2nd MoCo
	D1<double> win = sar::Kaiser(num_rg, KaiserBeta);
	
	double phs;
	for(long j=0L;j<num_az;++j){
		for(long i=0L;i<num_rg;++i){
			phs = 2.0*def::PI*( sar.fc*2.0*diff_LOS[j][i]/sar.c );//  +  sar.Kr*new_t[i]*2.0*d_LOS_ref[j]/sar.c);
			Smoco2[j][i] = Smoco2[j][i] * win[i] * mat::exp( (CPLX<double>(0.0,phs)).conj() );
		}
	}
}


void scw::CSA(const StructSAR sar, const D2<CPLX<double> >& Src, const D1<double>& R0,
			  D2<CPLX<double> >& Sac){
	cout<<endl;
	cout<<"================================="<<endl;
	cout<<" CSA focusing "<<endl;
	cout<<"================================="<<endl;
	
	//===============================================================
	// Prepare
	//===============================================================
	long num_rg = Src.GetN();
	long num_az = Src.GetM();
	D1<double> t = def_func::linspace(0.0, sar.Tr, 1.0/sar.fs);
	D1<double> ft(num_rg);
	def_func::linspace(-sar.fs/2, sar.fs/2, ft);
	
//	//===============================================================
//	// Range Compression
//	//===============================================================	
//	D2<CPLX<double> > Src = Sif;
//	sar::fft::RangeForward(Src);
//	sar::fft::RangeShift(Src);
	
	
	//===============================================================
	// Make Instantaneous Doppler frequency
	//===============================================================
	double f_abs = 0.0;
	long shift;
	D1<double> fm = sar::find::SARFindInsFreqOfMatchedFilter(f_abs, 1.0/sar.dm, num_az, shift);
	
	
	
	//
	// CSA
	//
	D2<CPLX<double> > Srd = Src;
	sar::fft::AzimuthForward(Srd);
	//===============================================================
	// 1st Scaling (differential)
	//===============================================================
	// find Reference range time
	double tmp = sar.c * sar::csa::MigrationParameter(fm[num_az/2], 1.0, sar.lambda);
	D1<double> tp(num_rg);
	for(long i=0;i<num_rg;++i){
		tp[i] = t[i] - 2.0/tmp;
	}
	D2<CPLX<double> > Ssc1(num_az,num_rg);
	D1<double> Kr(num_rg);
	
	for(long i=0;i<num_rg;++i){
		if(R0[i] < 1.0E-15){
			Kr[i] = 0.0;
		}else{
			Kr[i] = 2.0/(sar.lambda*R0[i]);
		}
	}
	
	double Km, dd1, dd2;
	CPLX<double> scaling;
	for(long j=0;j<num_az;++j){
		for(long i=0;i<num_rg;++i){
			Km = sar::csa::ModifiedChirpRate(fm[j], 1.0, Kr[i], R0[i], sar.c, sar.fc);
			dd1 = sar::csa::MigrationParameter(fm[num_az/2], 1.0, sar.lambda);
			dd2 = sar::csa::MigrationParameter(fm[j], 1.0, sar.lambda);
			scaling = exp(CPLX<double>( 0.0, def::PI*Km*(dd1/dd2-1.0)*tp[i]*tp[i] ));
			Ssc1[j][i] = Srd[j][i] * scaling;
		}
	}
	
	
	//===============================================================
	// 2nd Scaling (SRC & Bulk RCM)
	//===============================================================
	D2<CPLX<double> > Ssc2 = Ssc1;
	sar::fft::RangeForward(Ssc2);
	//D1<double> ft(num_rg);
	linspace(-sar.fs/2.0, sar.fs/2.0, ft);
	fftshift(ft);
	double Rref = R0[3.0/4.0*num_rg];
	
	for(long j=0;j<num_az;++j){
		for(long i=0;i<num_rg;++i){
			Km = sar::csa::ModifiedChirpRate(fm[j], 1.0, Kr[i], R0[i], sar.c, sar.fc);
			dd1 = sar::csa::MigrationParameter(fm[j], 1.0, sar.lambda);
			dd2 = sar::csa::MigrationParameter(fm[num_az/2], 1.0, sar.lambda);
			//			// SRC
			//			scaling = exp(CPLX<double>( 0.0, def::PI*dd1/(Km*dd2)*ft[i]*ft[i] ));
			//			Ssc2[j][i] = Ssc2[j][i] * scaling;
			// Bulk RCM
			scaling = exp(CPLX<double>( 0.0, 4.0*def::PI/sar.c*(1.0/dd1 - 1.0/dd2)*Rref*ft[i] ));
			Ssc2[j][i] = Ssc2[j][i] * scaling;
		}
	}
	
	
	
	//===============================================================
	// Azimuth compression
	//===============================================================
	Sac = Ssc2;
	sar::fft::RangeInverse(Sac);
	CPLX<double> filter;
	D1<double> wk(num_az);
	
	// window
	wk = KaiserWindow(1.0/sar.dm, num_az, 100.9);
	fftshift(wk);
	
	for(long j=0;j<num_az;++j){
		dd1 = sar::csa::MigrationParameter(fm[j], 1.0, sar.lambda);
		for(long i=0;i<num_rg;++i){
			// Phase
			filter = exp(CPLX<double>( 0.0,-4.0*def::PI*R0[i]*sar.fc*dd1/sar.c ));
			Sac[j][i] = Sac[j][i] * wk[j] * filter;
		}
	}
	sar::fft::AzimuthInverse(Sac);
}

void scw::RDA(const StructSAR sar, const D2<CPLX<double> >& Src, const D1<double>& R0,
			  D2<CPLX<double> >& Sac){
	
	cout<<endl;
	cout<<"================================="<<endl;
	cout<<" RDA focusing "<<endl;
	cout<<"================================="<<endl;
	
	//===============================================================
	// Prepare
	//===============================================================
	long num_rg = Src.GetN();
	long num_az = Src.GetM();
	D1<double> t(num_rg);
	def_func::linspace(0.0, sar.Tr, t);
	D1<double> ft(num_rg);
	def_func::linspace(-sar.fs/2, sar.fs/2, ft);
	
	
//	//===============================================================
//	// Range Compression
//	//===============================================================	
//	D2<CPLX<double> > Src = Sif;
//	sar::fft::RangeForward(Src);
//	sar::fft::RangeShift(Src);
	
	//===============================================================
	// Make Instantaneous Doppler frequency
	//===============================================================
	double f_abs = 0.0;
	long shift;
	D1<double> fm = sar::find::SARFindInsFreqOfMatchedFilter(f_abs, 1.0/sar.dm, num_az, shift);
	
	//===============================================================
	// Range Migration Correction
	//===============================================================
	D1<double> Vr_range(num_rg);
	for(long i=0L;i<num_rg;++i){ Vr_range[i] = 1.0; }
	D2<double> RCM(num_az,num_rg);
	sar::find::RCM(fm, R0, Vr_range, sar.Kr, sar.Tr, sar.lambda, RCM);
	D2<CPLX<double> > Srd = Src; // in place
	sar::fft::AzimuthForward(Srd);
	D2<CPLX<double> > Srcmc = sar::RCMC(Srd, RCM);
	// Azimuth ifft
	//	sar::fft::AzimuthInverse(Srcmc);
	
	
	//===============================================================
	// Azimuth compression
	//===============================================================
	D1<double> Ta(num_rg);
	for(long i=0;i<num_rg;++i){
		Ta[i] = abs( 2.0*R0[i]*tan(sar.theta_az/2.0) / 1.0 );
	}
	
	cout<<endl<<"AZIMUTH MATCHING FILTERING..."<<endl;
	
	// Az. matching filter window function
	D1<double> wk1 = sar::KaiserWindow(1.0/sar.dm, num_az, 2.1);
	fftshift(wk1);
	// Az. matching filter phase function
	CPLX<double> Haz;
	double A;
	
	// Matching filtering
	Sac = D2<CPLX<double> >(num_az, num_rg);
	for(long i=0;i<num_rg;++i){
		A = 2.0/(sar.lambda*R0[i]);
		for(long j=0;j<num_az;++j){
			Haz = exp(CPLX<double>(0.0,def::PI*fm[j]*fm[j]/A)) * wk1[j];
			Sac[j][i] = Srcmc[j][i] * Haz;
		}
		if(Mod(i, 100) == 0){ cout<<i<<" / "<<num_rg<<endl; }
	}
	// Az. IFFT
	sar::fft::AzimuthInverse(Sac);
	
	
	
	
	
	
	//===============================================================
	// For writing data *ONLY*
	//===============================================================
	sar::fft::AzimuthInverse(Srcmc);
	
}


#endif
