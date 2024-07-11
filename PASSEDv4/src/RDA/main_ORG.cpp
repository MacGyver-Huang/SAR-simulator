//
//  main.cpp
//  PhysicalOptics14_SAR
//
//  Created by Steve Chiang on 5/9/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//
//#define FASTPO
#define NORMALH

#ifdef DEBUG
#undef DEBUG
#endif



#include <sar/sar.h>
#include <bvh/obj.h>
#include <bvh/ray.h>
#include <rcs/raytrace.h>
#include <mesh/mesh.h>
#include <rcs/ems.h>
//#include <mesh/cad.h>
#include <basic/new_dir.h>


using namespace std;
using namespace vec;
using namespace mesh;
using namespace ems;
using namespace sv;
//using namespace cad;
using namespace new_dir;




CPLX<float> CPLXdouble2CPLXfloat(const CPLX<double>& in){
	return CPLX<float>(float(in.r()), float(in.i()));
}

void UpdateParameters(const SAR& Sar2, const EF& Ef2, const MultiAngle& MulAng2, const MeshDef& mesh2,	// new
					  SAR& Sar,  EF& Ef,  MultiAngle& MulAng,  MeshDef& mesh){							// old (need to be updated)
	// Check List
	cout<<"+-------------------------------------+"<<endl;
	cout<<"|                 List                |"<<endl;
	cout<<"+-------------------------------------+"<<endl;
	cout<<"|             SAR parameter           |"<<endl;
	cout<<"+-------------------------------------+"<<endl;
	cout<<" Center freq.      : "<<Sar2.f0()<<endl;
	cout<<" ADC sampling rate : "<<Sar2.Fr()<<endl;
	MulAng2.Print();
	cout<<"+-------------------------------------+"<<endl;
	cout<<"|             RCS parameter           |"<<endl;
	cout<<"+-------------------------------------+"<<endl;
	cout<<" Center freq.    : "<<Sar.f0()<<endl;
	cout<<" Chirp bandwidth : "<<Sar.BWrg()<<endl;
	MulAng.Print();
	
	
	//
	// Sar
	//
	if(abs(Sar.f0() - Sar2.f0()) > 0.5E9){
		cout<<"ERROR::There is not sutiable center frequency for this input SAR parameter in RCS database."<<endl;
		cout<<"Nearest center frequency in RCS database : "<<Sar.f0()/1E9<<" [GHz]"<<endl;
		cout<<"Import SAR center frequency              : "<<Sar2.f0()/1E9<<" [GHz]"<<endl;
		exit(EXIT_FAILURE);
	}
	if(abs(Sar.Fr() - Sar2.Fr()) > 100E6){
		cout<<"ERROR::There is not sutiable ADC sampling rate for this input SAR parameter in RCS database."<<endl;
		cout<<"Nearest ADC sampling rate in RCS database: "<<Sar.Fr()/1E6<<" [MHz]"<<endl;
		cout<<"Import SAR ADC sampling rate             : "<<Sar2.Fr()/1E6<<" [MHz]"<<endl;
		exit(EXIT_FAILURE);
	}
	Sar.SetTheta_l_MB(Sar2.theta_l_MB());
	Sar.f0() = Sar2.f0();
	Sar.BWrg() = Sar.BWrg();
	Sar.theta_sqc() = Sar2.theta_sqc();
	Sar.Fr() = Sar2.Fr();
	Sar.Nr() = Sar2.Nr();
	//
	// Ef (Priority in SAR is higher than Ef) -> Check capability
	//
	if(Ef.TxPol() != Ef2.TxPol()){
		cout<<"ERROR::Selected RCS didn't match the SAR parameter (*.parset)"<<endl;
		cout<<"Input TxPol = "<<Ef.TxPol()<<endl;
		cout<<"RCS   TxPol = "<<Ef2.TxPol()<<endl;
		exit(EXIT_FAILURE);
	}
	Ef.Taylor() = Ef2.Taylor();
	// Ef.MaxLevel() = (Ef.MaxLevel() <= Ef2.MaxLevel())? Ef.MaxLevel():Ef2.MaxLevel();
	Ef.MaxLevel() = Ef2.MaxLevel();
	//
	// MulAng
	//
	// Check look angle
	if(MulAng.IsMulLook() == true && MulAng2.IsMulLook() == false){
		cout<<"ERROR::Input is Multi-look angle but RCS is not"<<endl;
		exit(EXIT_FAILURE);
	}
	if(MulAng.IsMulLook() == false && MulAng2.IsMulLook() == true){
		// in the region?
		if(MulAng.LookFrom() < MulAng2.LookFrom() && MulAng.LookFrom() > MulAng2.LookTo()){
			cout<<"ERROR::Input is single Look angle but not within region of RCS"<<endl;
			cout<<"Input Look angle = "<<rad2deg(MulAng.LookFrom())<<endl;
			cout<<"RCS   Look angle Region = ["<<rad2deg(MulAng2.LookFrom())<<", "<<rad2deg(MulAng2.LookTo())<<"]"<<endl;
			exit(EXIT_FAILURE);
		}
	}
//	if(MulAng.IsMulLook() == false && MulAng2.IsMulLook() == false){
//		// match?
//		if(MulAng.LookFrom() != MulAng2.LookFrom()){
//			cout<<"ERROR::The input single Look angle is not mach the single RCS look angle"<<endl;
//			cout<<"Input Look angle = "<<rad2deg(MulAng.LookFrom())<<endl;
//			cout<<"RCS   Look angle = "<<rad2deg(MulAng2.LookFrom())<<endl;
//			exit(EXIT_FAILURE);
//		}
//	}
	if(MulAng.IsMulLook() == true && MulAng2.IsMulLook() == true){
		// in the region?
		if( (MulAng.LookFrom() < MulAng2.LookFrom() || MulAng.LookFrom() > MulAng2.LookTo()) ||
		   (MulAng.LookTo() < MulAng2.LookFrom()   || MulAng.LookTo() > MulAng2.LookTo()) ){
			cout<<"ERROR::The input multi-look angle is not within multi RCS look angle region"<<endl;
			cout<<"Input Look angle Region = ["<<rad2deg(MulAng.LookFrom())<<", "<<rad2deg(MulAng.LookTo())<<"]"<<endl;
			cout<<"RCS   Look angle Region = ["<<rad2deg(MulAng2.LookFrom())<<", "<<rad2deg(MulAng2.LookTo())<<"]"<<endl;
			exit(EXIT_FAILURE);
		}
	}
	// Check look angle
	if(MulAng.IsMulAsp() == true && MulAng2.IsMulAsp() == false){
		cout<<"ERROR::Input is Multi-aspect angle but RCS is not"<<endl;
		exit(EXIT_FAILURE);
	}
	if(MulAng.IsMulAsp() == false && MulAng2.IsMulAsp() == true){
		// in the region?
		if(MulAng.AspFrom() < MulAng2.AspFrom() && MulAng.AspFrom() > MulAng2.AspTo()){
			cout<<"ERROR::Input is single Aspect angle but not within region of RCS"<<endl;
			cout<<"Input Aspect angle = "<<rad2deg(MulAng.AspFrom())<<endl;
			cout<<"RCS   Aspect angle Region = ["<<rad2deg(MulAng2.AspFrom())<<", "<<rad2deg(MulAng2.AspTo())<<"]"<<endl;
			exit(EXIT_FAILURE);
		}
	}
	if(MulAng.IsMulAsp() == false && MulAng2.IsMulAsp() == false){
		// match?
		if(MulAng.AspFrom() != MulAng2.AspFrom()){
			cout<<"ERROR::The input single Aspect angle is not mach the single RCS Aspect angle"<<endl;
			cout<<"Input Aspect angle = "<<rad2deg(MulAng.AspFrom())<<endl;
			cout<<"RCS   Asepct angle = "<<rad2deg(MulAng2.AspFrom())<<endl;
			exit(EXIT_FAILURE);
		}
	}
	if(MulAng.IsMulAsp() == true && MulAng2.IsMulAsp() == true){
		// in the region?
		if( (MulAng.AspFrom() < MulAng2.AspFrom() || MulAng.AspFrom() > MulAng2.AspTo()) ||
		   (MulAng.AspTo() < MulAng2.AspFrom()   || MulAng.AspTo() > MulAng2.AspTo()) ){
			cout<<"ERROR::The input multi-aspect angle is not within multi RCS aspect angle region"<<endl;
			cout<<"Input Aspect angle Region = ["<<rad2deg(MulAng.AspFrom())<<", "<<rad2deg(MulAng.AspTo())<<"]"<<endl;
			cout<<"RCS   Aspect angle Region = ["<<rad2deg(MulAng2.AspFrom())<<", "<<rad2deg(MulAng2.AspTo())<<"]"<<endl;
			exit(EXIT_FAILURE);
		}
	}
	
	
	//
	// mesh
	//
	mesh = mesh2;
}


void GetMapCoordinate(const SAR& Sar, const ORB& Orb, VEC<double>& Ps, VEC<double>& Ps1,	// input
					  const double theta_l_MB, const double SW,								// input
					  GEO<double>& G1n, GEO<double>& G1c, GEO<double>& G1f){				// output
	// 1st
	VEC<double> Pg = sar::find::ProjectPoint(Ps, Orb);
	double Re = Pg.abs();
	double Re_h = Ps.abs();
	double slc1 = sar::find::SlantRange(Ps, Ps1, Sar.theta_sqc(), theta_l_MB, Orb);
	// near
	double slc1n= slc1 - SW/2;
	double theta_l_1n = acos( (slc1n*slc1n + Re_h*Re_h - Re*Re)/(2*slc1n*Re_h) );
	VEC<double> uv_1n = sar::find::LookAngleLineEq(Ps, Pg, Ps1, Sar.theta_sqc(), theta_l_1n);
	VEC<double> P1n   = sar::find::BeamLocationOnSurface(uv_1n, Ps, Orb);
	G1n   = sar::ECR2Gd(P1n, Orb);	// [lon,lat,h]
	// far
	double slc1f= slc1 + SW/2;
	double theta_l_1f = acos( (slc1f*slc1f + Re_h*Re_h - Re*Re)/(2*slc1f*Re_h) );
	VEC<double> uv_1f = sar::find::LookAngleLineEq(Ps, Pg, Ps1, Sar.theta_sqc(), theta_l_1f);
	VEC<double> P1f   = sar::find::BeamLocationOnSurface(uv_1f, Ps, Orb);
	G1f   = sar::ECR2Gd(P1f, Orb);	// [lon,lat,h]
	// Center
	VEC<double> uv_1c = sar::find::LookAngleLineEq(Ps, Pg, Ps1, Sar.theta_sqc(), theta_l_MB);
	VEC<double> P1c   = sar::find::BeamLocationOnSurface(uv_1c, Ps, Orb);
	G1c   = sar::ECR2Gd(P1c, Orb);	// [lon,lat,h]

#ifdef DEBUG
	cout<<rad2deg(theta_l_1n)<<endl;
	cout<<rad2deg(theta_l_MB)<<endl;
	cout<<rad2deg(theta_l_1f)<<endl;
	G1n.PrintDeg();
	G1c.PrintDeg();
	G1f.PrintDeg();
#endif
}


string SearchSuitableRCSFile(const D1<double>& ThetaSeriesRCS, const D1<double>& ThetaSeries, const size_t ProcLevel,
							 const D1<string>& file_RCSData, const EF& Ef, const string POL, const long it){
	// e.g. For total
	//		file_RCSData[0] = "UD07_34.30_rcs_HH.dat"
	//		file_RCSData[1] = "UD07_34.30_rcs_VH.dat"
	//
	// e.g. For certain level
	//		file_RCSData[0] = "UD07_34.30_rcs_HH_Level0.dat"
	//		file_RCSData[1] = "UD07_34.30_rcs_VH_Level0.dat"
	size_t off = 0;
	string StrLevel = "";
	if(ProcLevel != 0){ off = 1; }

	// Find nearest theta(look) angle in RCS
	double tmp = 999999.99;
	double val = ThetaSeriesRCS[0];
	for(size_t i=0;i<ThetaSeriesRCS.GetNum();++i){
		double diff = abs(ThetaSeriesRCS[i] - ThetaSeries[it]);
		if(diff < tmp){
			tmp = diff;
			val = ThetaSeriesRCS[i];
		}
	}
	string StrMatch = StrTruncate(num2str(rad2deg(val)), 2);
	
	// Search suitable file_RCSData
	string file_RCSDataDest;
	for(size_t i=0;i<file_RCSData.GetNum();++i){
		// Get file name
		string name = StrFilename(file_RCSData[i]).name;	// "UD07_34.30_rcs_HH" or "UD07_34.30_rcs_HH_Level0"
		// Get Theta(Look) value in NAME string
		D1<string> tmp2 = StrSplit(name, '_');				// {"UD07","34.30","rcs","HH"} or {"UD07","34.30","rcs","HH","Level0"}
		string StrLook  = tmp2[tmp2.GetNum()-(3+off)];				// "34.30" or "34.30"
		string StrPol   = tmp2[tmp2.GetNum()-(1+off)].substr(0,2);	// "HH" or "HH"
		// Get destination RCS file
		if(off == 1){
			StrLevel = tmp2[tmp2.GetNum()-1];
			if( StrLook == StrMatch && StrPol == (POL + Ef.TxPol()) && StrLevel == "Level"+num2str(ProcLevel) ){
				file_RCSDataDest = file_RCSData[i];
				break;
			}
		}else{
			if( StrLook == StrMatch && StrPol == (POL + Ef.TxPol()) ){
				file_RCSDataDest = file_RCSData[i];
				break;
			}
		}
	}

	return file_RCSDataDest;
}


void GenerateSAR(const SAR& Sar, const SV<double>& sv, const ORB& Orb, const MeshDef& mesh,		// SAR system
				 const D1<double>& ThetaSeriesRCS, const D1<double>& PhiSeriesRCS,				// RCS angle series
				 const double TargetAsp,														// Target angle
				 const D1<double>& R0_in, const D1<double>& freq, const D1<double>& Wr,			// basic series
				 const VEC<double>& Psc, const double Rmin, const double Rc_min,				// Misc.
				 const string file_RCSDataDest,	const size_t ProcLevel,							// input RCS data file (including multi asp angle)
				 // For export
				 const EF& Ef, const double TargetLook, const string POL,						// SAR system & pol and angle information
				 const string dir_out, const string name,										// export name string
				 const int WriteMode, const double Vs, const SV<double>& TER_SV,				// Misc.
				 // Return or keyword
				 D1<double>& theta_sq, const bool ShowSummary, const bool IsCOMPLEX_MESSAGE = false){
	//+----------------------------------------------------------------------------------------+
	//|                                                                                        |
	//|                                 SINGLE SAR IMAGE                                       |
	//|                                                                                        |
	//+----------------------------------------------------------------------------------------+
	// Memories allocation
	D1<CPLX<float> > line(Sar.Nr());			// RAW data from RCS file
	D1<CPLX<double> > Src_tmp(Sar.Nr());		// line + remain PATH distance
	D2<CPLX<double> > Src(Sar.Na(), Sar.Nr());	// 2D SAR echo signal for each theta(look) & phi(asp) angle
												// After range compression
	// Duplicate
	D1<double> R0 = R0_in;
	
	//+-------------------+
	//| Open FILE pointer |
	//+-------------------+
	ifstream fin;
	fin.open(file_RCSDataDest.c_str(), ios_base::binary);
	
	
	
	
	// Nearest index
	double look = Sar.theta_l_MB();	// inst. theta angle
	double asp;		// inst. phi angle
	long idx_look;	// index of theta angle
	long idx_asp;	// index of phi angle
	
	
	//
	// Main Look
	//
	for(long j=0;j<Sar.Na();++j){						// Azimuth samples
		//+-------------------+
		//| Position & squint |
		//+-------------------+
		// Sensor inst. position
		VEC<double> PPs = sv.pos()[j];					// Sesnor position
		VEC<double> PPs1;								// Next by PPs
        VEC<double> PPsC= sv.pos()[(sv.GetNum()-1)/2];	// Center sensor position
		VEC<double> PPt = Psc;							// Target's center position
		// Squint angle
		if(j != Sar.Na()-1){
			PPs1 = sv.pos()[j+1];
			theta_sq[j] = deg2rad(90) - angle(PPs1-PPs, PPt-PPs);
		}else{
			VEC<double> PPsv = sv.pos()[j-1];
			VEC<double> uv   = PPs - PPsv;
			PPs1 = PPs + uv.abs() * Unit(uv);
			theta_sq[j] = -theta_sq[0];
		}
		//+-------------------+
		//|   angle & index   |
		//+-------------------+
		// Find incident angle
		double theta_inc = sar::find::ThetaInc(PPsC, PPt, Orb);		// 38.2809
		// Find Target's global-XYZ axis (xuv,yuv,zuv) converted from local XYZ-axis
		VEC<double> xuv, yuv, zuv;
		sar::find::LocalAxis(PPsC, PPt, xuv, yuv, zuv, Orb);
		// project point PPs to XY-plane
		VEC<double> PPs_xy;
		vec::find::MinDistanceFromPointToPlane(zuv, PPt, PPs, PPs_xy);
		// find min distance from PPs_xy to x-axis
		VEC<double> PPt1 = PPt + 100. * xuv;
		VEC<double> PPs_x;
		vec::find::MinDistanceFromPointToLine(PPt, PPt1, PPs_xy, PPs_x);
		// azimuth angle == phi(azimuth) angle
		double sign = (dot(zuv, cross(xuv, PPs_xy - PPs_x)) > 1E-20)? 1.:-1.;
		double phi_az = atan2(sign * (PPs_xy - PPs_x).abs(), (PPs_x - PPt).abs());
		// Add Target's aspect angle
		phi_az += TargetAsp;
//		phi_az += deg2rad(0);
//		phi_az += deg2rad(45);
//		phi_az += deg2rad(-45);
		
//		cout<<rad2deg(theta_inc)<<endl;
//		cout<<j<<" "<<rad2deg(phi_az)<<endl;
		
//		cout<<"ThetaSeriesRCS[i] : "<<endl;
//		for(size_t idx=0;idx<ThetaSeriesRCS.GetNum();++idx){
//			cout<<rad2deg(ThetaSeriesRCS[idx])<<endl;
//		}
//		cout<<"PhiSeriesRCS[i] : "<<endl;
//		for(size_t idx=0;idx<PhiSeriesRCS.GetNum();++idx){
//			cout<<rad2deg(PhiSeriesRCS[idx])<<endl;
//		}
//		cout<<"theta_inc = "<<rad2deg(theta_inc)<<", phi_az = "<<rad2deg(phi_az)<<endl;
		
		// Nearest index
		sar::find::RCSNearestAngle(ThetaSeriesRCS, PhiSeriesRCS,	// input RCS angle series
								   theta_inc, phi_az,				// desire angles
								   look, asp,						// NEAREST angles
								   idx_look, idx_asp);				// NEAREST angle index
		
		//+-------------------+
		//|     Read Data     |
		//+-------------------+
		// offset
		long off = (idx_asp  * Sar.Nr()) * sizeof(CPLX<float>);
		fin.clear();	// Clean all error state flags
		fin.seekg(off, ios_base::beg);
		// Read
		fin.read(reinterpret_cast<char*>(line.GetPtr()), sizeof(CPLX<float>)*Sar.Nr());
		// If any error occurs, reset the line series to be zero.
		if(fin.fail()){
			line.SetZero();
		}

		//+-------------------+
		//|     Check NaN     |
		//+-------------------+
		for(long i=0;i<line.GetNum();++i){
			if( std::isnan(line[i].abs()) == true ){
				line[i] = CPLX<float>(0,0);
			}
		}		


		// //+-----------------------+
		// //| Range Shift to center |
		// //+-----------------------+
		// D1<CPLX<float> > line2(line.GetNum());
		// // Duplicate
		// line2 = line;
		// // Forward FFT
		// FFT(line2);
		// // Shift
		// long shift_pix = line.GetNum()/4;
		// mat::shift(line2, shift_pix);
		// mat::fftshift(line2);
		// // Backward FFT
		// IFFT(line2);
		// // Assign back
		// line = line2;
		
		
		//+--------------------+
		//| Assign to SAR echo |
		//+--------------------+
		// distance from sensor to sphere incident mesh
		// double Rad  = (PPs - PPt).abs() - mesh.dRad();
		double Rad  = (PPs - PPt).abs();
		// Add remain path distance
		for(long i=0;i<Sar.Nr();++i){
			double k0 = 2*PI/( def::C/freq[i] );
			// Add dis phase
			double phs = k0 * (2 * Rad);
			double cp, sp;
			opt::_sincos(phs, sp, cp);
			CPLX<double> phase(cp,sp);
			Src_tmp[i] = CPLX<double>( line[i].r(), line[i].i() ) * phase ;
			// Src_tmp[i] = CPLX<double>( line[i].r(), line[i].i() );
		}
		
		
		//+-------------------------------------------------------------------------+
		//|                          SAR simulation                                 |
		//+-------------------------------------------------------------------------+
		// Calculate Wa
		double Wa = sar::Waz(theta_sq[j], Sar.theta_az());
		// double Wa = 1;
		
		// Kaiser window
		double KAISER_BETA = 2.1;
		// D1<double> wk = sar::find::Kaiser(Sar.Nr(), KAISER_BETA);
		// fftshift(wk);
		
		// Add range FM
		// Normalize sys.Nr for equalization the ifft output amplitude to be as same as RCS of dihedral
		for(long i=0;i<Sar.Nr();++i){
//			Src_tmp[i].r() *= (Wa * Wr[i]) / Sar.Nr();
//			Src_tmp[i].i() *= (Wa * Wr[i]) / Sar.Nr();
			// Src_tmp[i].r() *= (Wa * wk[i]) / Sar.Nr();
			// Src_tmp[i].i() *= (Wa * wk[i]) / Sar.Nr();
			Src_tmp[i].r() *= (Wa / Sar.Nr());
			Src_tmp[i].i() *= (Wa / Sar.Nr());
		}
		FFT(Src_tmp);
		fftshift(Src_tmp);
		
		// Create Echo Signal
		for(size_t i=0;i<Src.GetN();++i){
			Src[j][i] = Src_tmp[i];
		}
	}// end of azimuth sample
	//+--------------------+
	//| Close FILE pointer |
	//+--------------------+
	fin.close();
	
	
	
	//+-----------------+
	//|  Range Shift    |
	//+-----------------+
	// Calculate shift pixel
	double dR = R0[1]-R0[0];
	// long nss_org, nss_int;
	// double nss_rem;
	// // Rc_min : Nearest Slant scene center range distance
	// // Rmin = Rc_min - sys.SWrg/2 : Min slant range distance
	// sar::cw::ShiftSample(Sar.Nr(), Sar.Fr(), Sar.SWrg(), dR, Rmin, nss_org, nss_int, nss_rem);
	// sar::cw::SignalShift(nss_org, nss_rem, Sar.Nr(), Rc_min, dR, Src, R0);
	for(size_t i=0;i<R0.GetNum();++i){
		R0[i] += Rc_min;
	}

	// printf("Rc_min = %lf, Rc_min_real = 868719.411520\n", Rc_min);
	
	
	//+-----------------+
	//| Check Max Value |
	//+-----------------+
	if(IsCOMPLEX_MESSAGE){
		double Sth  = max(abs(Src));
		cout<<"Sth Max      = "<<Sth<<endl;
		cout<<"Sth Max [dB] = "<<10*log10( 4*def::PI/(Sar.lambda()*Sar.lambda()) * (Sth*Sth) )<<endl;
	}
	
	
	
	
	//+===========================================================================+
	//|                                                                           |
	//|                             Export Data                                   |
	//|                                                                           |
	//+===========================================================================+
	// make directory
	string StrTheta = StrTruncate(num2str(rad2deg(TargetLook)),2);
	string StrPhi   = StrTruncate(num2str(rad2deg(TargetAsp)) ,2);
	string Suffix   = StrTheta + "_" + StrPhi + "_" + POL + Ef.TxPol();
	string FullName = name + "_" + Suffix;
	MKDIR(dir_out + FullName);
	//+---------------------------+
	//| File fullname             |
	//+---------------------------+
	string sar_ant_pattern = ".//constant_antenna.gain";
	string file_SAR_par    = dir_out + FullName + "/SENSOR.par";
	string file_PROC_par   = dir_out + FullName + "/p" + Suffix + ".slc.par";
	string file_Src        = dir_out + FullName + "/" + Suffix + "_Src.raw";
	string file_R0         = dir_out + FullName + "/" + Suffix + "_R0.raw";
	string file_Ra         = dir_out + FullName + "/" + Suffix + "_Ra.raw";
	string file_Sac        = dir_out + FullName + "/" + Suffix + "_focused.raw";

	if(ProcLevel >= 0){
		file_Sac        = dir_out + FullName + "/" + Suffix + "_Level" + num2str(ProcLevel) + "_focused.raw";
	}

	// FFTY(Src);
	// Src.WriteBinary(file_Sac.c_str());
	
	//+---------------------------+
	//| Write with each Mode      |
	//+---------------------------+
	//+------------------+
	//| Sensor Parameter |
	//+------------------+
	char sar_title[]="SAR simulation from Gave Co.\0"; // no effective
	char sar_sensor[]="SENSOR\0";    // no effective
	char sar_chirp[]="UP_CHIRP\0";      // effective
	char sar_mode[]="IQ\0";             // effective
	char sar_type[]="FLOAT\0";          // effective
	char sar_spectrum[]="NORMAL\0";     // effective
	par::SAR_PAR sarp=sar::Init_SAR_par(sar_title, sar_sensor, sar_chirp, sar_mode, sar_type,sar_spectrum,
										sar_ant_pattern.c_str(), Sar.Nr(), Sar);
	
	//+-------------------+
	//| Process Parameter |
	//+-------------------+
	char procp_title[] = "SENSOR\0";   // no effective
	bool procp_SRC=false;               // effective
	bool procp_deskew=false;            // effective
	
	double time_near_range_raw = 2*R0[0]/def::C;
	double time_far_range_raw  = 2*R0[R0.GetNum()-1]/def::C;
	
	
	par::PROC_PAR procp=sar::Init_PROC_par(procp_title, procp_SRC, procp_deskew,
										   Sar, Orb, Ef,
										   Psc, sv.t()[0], sv.pos()[0], TER_SV,
										   time_near_range_raw, time_far_range_raw, Sar.Na());
	
	if((WriteMode == 1) || (WriteMode == 2)){
		//+-----------------+
		//|   Export Data   |
		//+-----------------+
		Src.WriteBinary(file_Src.c_str());
	}// End of write RAW or Both (*_Src.raw)
	
	if((WriteMode == 0) || (WriteMode == 2)){
		//====================================================================================
		// SAR focusing (Simple RDA Focusing)
		//====================================================================================
		//+---------------------------+
		//| Doppler Estimation        |
		//+---------------------------+
		double fabs = 0.;	// Doppler Centroid frequency
		
		//+---------------------------+
		//| slow time inst. frequency |
		//+---------------------------+
		D1<double> fn(Sar.Na());
		double shift_d;
		sar::find::InsFreqOfMatchedFilter(fabs, Sar.PRF(), fn, shift_d);
		
		//+---------------------------+
		//| Find look angle region    |
		//+---------------------------+
		D1<double> theta_l_limit = sar::find::ThetaLookRegion(sv, R0, Orb);
		
		//+---------------------------+
		//| Find Dffective Velocity   |
		//+---------------------------+
		D1<double> R0_rg, Vr_rg;
		sar::find::EffectiveVelocity(sv, Orb, theta_l_limit, Sar.theta_sqc(), Sar.Nr(), R0_rg, Vr_rg);

		// printf("min(R0_rg) = %lf, max(R0_rg) = %lf, min(R0) = %lf, max(R0) = %lf\n", min(R0_rg), max(R0_rg), min(R0), max(R0));
		// printf("PRF = %lf, Fr = %lf, lambda = %lf, mean(Vr_rg) = %lf\n", Sar.PRF(), Sar.Fr(), Sar.lambda(), mean(Vr_rg));

//		sv.Print();
		
		//+---------------------------+
		//| Find RCM values           |
		//+---------------------------+
		D2<double> RCM(Sar.Na(), Sar.Nr());
//		sar::find::RCM(fn, R0_rg, Vr_rg, Sar.Kr(), Sar.Tr(), Sar.lambda(), RCM);
		// sar::find::RCM(fn, R0_rg, Vr_rg, Sar.Fr(), Sar.lambda(), RCM);
		sar::find::RCM(fn, R0_rg, Vr_rg, Sar.Fr(), Sar.lambda(), RCM);
		
		//+---------------------------+
		//| RCMC correction           |
		//+---------------------------+
//		D2<CPLX<double> > Srd0 = Src; FFTX(Srd0);
//		Srd = sar::RCMC(Srd, RCM, false);

//		//
//		// Polynominal
//		//
//		D2<CPLX<double> > Srd = sar::RCMC(Src, RCM, false);
		
		//
		// Sinc interpolator
		//
		D2<CPLX<double> > Srd0 = Src; FFTY(Srd0);
		D2<CPLX<double> > Srd = sar::RCMCSinc(Srd0, RCM, false);
		
		D2<CPLX<double> > Srcmc = Srd;
		IFFTY(Srcmc);

		//+---------------------------+
		//| Remove R0_min             |
		//+---------------------------+
		D1<CPLX<double> > line3(Sar.Nr());
		// Add remain path distance
		for(long j=0;j<Sar.Na();++j){
			// Extract range profile
			line3 = Srcmc.GetRow(j);
			// Forward FFT
			FFT(line3);
			for(long i=0;i<Sar.Nr();++i){
				// Add dis phase
				double k0 = 2*PI/( def::C/freq[i] );
				double phs = k0 * (2 * Rc_min);
				double cp, sp;
				opt::_sincos(phs, sp, cp);
				CPLX<double> phase(cp,sp);
				// Multiplcation
				line3[i] = line3[i] * phase;
			}
			// Backward FFT
			IFFT(line3);
			// Assign back
			Srcmc.SetRow(line3, j);
		}


		// double Rad  = (PPs - PPt).abs();
		// // Add remain path distance
		// for(long i=0;i<Sar.Nr();++i){
		// 	double k0 = 2*PI/( def::C/freq[i] );
		// 	// Add dis phase
		// 	double phs = k0 * (2 * Rad);
		// 	double cp, sp;
		// 	opt::_sincos(-phs, sp, cp);
		// 	CPLX<double> phase(cp,sp);
		// 	Src_tmp[i] = CPLX<double>( line[i].r(), line[i].i() ) * phase ;
		// 	// Src_tmp[i] = CPLX<double>( line[i].r(), line[i].i() );
		// }


		//+---------------------------+
		//| Transfer to RD domain     |
		//+---------------------------+
		Srd = Srcmc;
		FFTY(Srd);

		// // double R0_min = min(R0);
		// // printf("R0_min = %lf\n", R0_min);
		// double sp_rg = def::C/(2 * Sar.Fr());
		// // long shift_pix2 = -2*R0_min/sp_rg;
		// long shift_pix2 = 2*Rc_min/sp_rg;
		// D1<CPLX<double> > line3(Sar.Na());

		// shift_pix2 = shift_pix2 % Sar.Nr();
		// printf("shift_pix2 = %ld\n", shift_pix2);

		// for(size_t j=0;j<Sar.Na();++j){
		// 	// Extract range profile
		// 	line3 = Srcmc.GetRow(j);
		// 	// shift
		// 	mat::shift(line3, shift_pix2);
		// 	mat::fftshift(line3);
		// 	// Assign back
		// 	Srcmc.SetRow(line3, j);
		// }

		
		// printf("min(RCM) = %lf, max(RCM) = %lf\n", min(RCM), max(RCM));
		// RCM.WriteBinary("/root/cuda-workspace/TEST_AREA/ComplexTarget/SAR/RCM.dat");
		// FFTY(Src);
		// Src.WriteBinary("/root/cuda-workspace/TEST_AREA/ComplexTarget/SAR/Src.dat");
		// FFTY(Srcmc);
		// Srcmc.WriteBinary("/root/cuda-workspace/TEST_AREA/ComplexTarget/SAR/Srcmc.dat");
		
		
//		RCM.WriteBinary("/Users/cychiang/Documents/code/C/test/test_RDA01/test_RDA01/data2/res+000/SAR/SAR_res+000_34.06_0.00_HH/RCM.dat");
//		Src.WriteBinary("/Users/cychiang/Documents/code/C/test/test_RDA01/test_RDA01/data2/res+000/SAR/SAR_res+000_34.06_0.00_HH/Src.dat");
//		Srcmc.WriteBinary("/Users/cychiang/Documents/code/C/test/test_RDA01/test_RDA01/data2/res+000/SAR/SAR_res+000_34.06_0.00_HH/Srcmc.dat");
//		Srd.WriteBinary("/Users/cychiang/Documents/code/C/PO/PhysicalOptics16_cuda/TestAreaMSTAR/res_remote/SAR/Srd.dat");
////		RCM.WriteBinary("/Users/cychiang/Documents/code/C/PO/PhysicalOptics17/TestAreaMSTAR/res/rcm.dat");
//		cout<<"Src   = ["<<Src.GetM()<<","<<Src.GetN()<<"]"<<endl;
//		cout<<"Srcmc = ["<<Srcmc.GetM()<<","<<Srcmc.GetN()<<"]"<<endl;
		
		
		
		//+---------------------------------+
		//| Convert ot Range-Doppler domain |
		//+---------------------------------+
//		D2<CPLX<double> > Srd = Srcmc;
//		mat::FFTX(Srd);
//		D2<CPLX<double> > Srd(Src.GetM(), Src.GetN());
//		D1<CPLX<double> > aztmp(Src.GetM());
//		for(long i=0;i<Src.GetN();++i){
//			// copy column
//			aztmp = Srcmc.GetColumn(i);
//			// fft
//			FFT(aztmp);
//			// assign column
//			Srd.SetColumn(aztmp, i);
//		}
		
		
//		Src.WriteBinary("/Users/cychiang/Documents/code/C/PO/PhysicalOptics16_cuda/TestAreaMSTAR/res_remote/SAR/Src.dat");
//		Srd.WriteBinary("/Users/cychiang/Documents/code/C/PO/PhysicalOptics16_cuda/TestAreaMSTAR/res_remote/SAR/Srd.dat");
		
		//+---------------------------+
		//| Azimuth compression       |
		//+---------------------------+
		double R0_mean = mat::mean(R0_rg);
		double Vr_mean = mat::mean(Vr_rg);
		double R_nc = R0_mean;
		double Kamf = 2.0*Vr_mean*Vr_mean/(Sar.lambda()*R_nc);	// Doppler Rate
		
		//+---------------------------+
		//| Azimuth compression       |
		//+---------------------------+
		// Ref. Az. filter
		D1<CPLX<double> > ref_az(Sar.Na());
		for(long i=0;i<Sar.Na();++i){
			// ref_az[i] = cplx::exp( CPLX<double>(0,-def::PI*fn[i]*fn[i]/Kamf) );
			ref_az[i] = cplx::exp( CPLX<double>(0,+def::PI*fn[i]*fn[i]/Kamf) );
		}
		
		//+---------------------------+
		//| Kaiser window             |
		//+---------------------------+
		double KAISER_BETA = 2.1;
		D1<double> wk = sar::find::KaiserWindow(Sar.PRF(), Sar.Na(), KAISER_BETA);
		fftshift(wk);
		
		//+---------------------------+
		//| Matching                  |
		//+---------------------------+
		// double Ratio = (double)Sar.Na() / (double)Sar.Nr();// * (2*sqrt(2));
		// double Ratio = (double)Sar.Na() / (double)Sar.Nr() * (4);
		double Ratio = (double)Sar.Na() / 2;
		D2<CPLX<double> > Sac(Sar.Na(), Sar.Nr());
		D1<CPLX<double> > aztmp(Src.GetM());
		for(long i=0;i<Sar.Nr();++i){
			for(long j=0;j<Sar.Na();++j){
				aztmp[j] = Srd[j][i] * wk[j] * ref_az[j] * Ratio;
			}
			// FFT
			IFFT(aztmp);
			// assign back
			Sac.SetColumn(aztmp, i);
		}
		
		//+---------------------------+
		//| Export data               |
		//+---------------------------+
//		if(WriteMode == 0){
//			cout<<"Src.GetM() : sys.Na  = "<<Sac.GetM()<<endl;
//			cout<<"Src.GetN() : sys.Nr  = "<<Sac.GetN()<<endl;
//		}
		Sac.WriteBinary(file_Sac.c_str());
		
		
//		//+---------------------------+
//		//| Check Max Value           |
		//+---------------------------+
		double MaxSac = mat::max(abs(Sac));
		if(IsCOMPLEX_MESSAGE){
			cout<<"Sac Max      = "<<mat::max(abs(Sac))<<endl;
			cout<<"Sac Max [dB] = "<<10*log10( 4*def::PI/(Sar.lambda()*Sar.lambda()) * (MaxSac*MaxSac) )<<endl<<endl;
		}
		
		//+---------------------------+
		//| Update *.slc.par          |
		//+---------------------------+
//		Sar.Laz() = 1.2;
		double Laz = Sar.Laz();// / 10.0;
		// Doppler coefficience
		D1<double> Vg = sar::find::GroundVelocity(sv, theta_l_limit, Orb);
//		double Vr_mean = (Vr_rg[0] + Vr_rg[Vr_rg.GetNum()-1])/2.0;
		double Vg_mean = (Vg[0] + Vg[1])/2.0;
//		double R0_mean = (R0[0] + R0[R0.GetNum()-1])/2.0;
		// Doppler rate
		double Ka = 2.0*Vr_mean*Vr_mean/(Sar.lambda()*R0_mean);
		// Target exposure time
		double Ta = 0.886*Sar.lambda()*R0_mean/(Laz*Vg_mean*cos(deg2rad(0.0)));
		// Doppler Bandwidth
		double dfd= abs(Ka) * Ta;
		// Azimuth resolution
		double azimres = 0.886*Vg_mean*cos(deg2rad(0.0))/dfd;
		
		
		
		procp.ran_res =	def::C/(2*Sar.BWrg());		// range resolution
		procp.azimres = azimres;				// azimuth resolution
		procp.rpixsp = def::C/(2*Sar.Fr());			// range pixel spacing
		procp.azimsp = Vs/Sar.PRF();			// azimuth pixel spacing
		procp.nrs  = (int)Sar.Nr();				// range sample
		procp.nazs = (int)Sar.Na();				// azimuth sample
		
		// Doppler coefficient
		if(ShowSummary){
			cout<<"+-----------------------+"<<endl;
			cout<<"|    Doppler coeff.     |"<<endl;
			cout<<"+-----------------------+"<<endl;
			cout<<" Azimuth eff. antenna (Laz) = "<<Laz<<" [m]"<<endl;
			cout<<" Doppler Rate          (Ka) = "<<Ka<<" [Hz/sec]"<<endl;
			cout<<" Exposure Time         (Ta) = "<<Ta<<" [sec]"<<endl;
			cout<<" Doppler bandwidth    (dfd) = "<<dfd<<" [Hz]"<<endl;
			cout<<" Azimuth resolution    (az) = "<<azimres<<" [m]"<<endl;
			// Message
			cout<<"+-----------------------+"<<endl;
			cout<<"| Resolution & spacing  |"<<endl;
			cout<<"+-----------------------+"<<endl;
			cout<<" Range resolution   : "<<procp.ran_res<<" [m]"<<endl;
			cout<<" Azimuth resolution : "<<procp.azimres<<" [m]"<<endl;
			cout<<" Range spacing      : "<<procp.rpixsp<<" [m]"<<endl;
			cout<<" Azimuth spacing    : "<<procp.azimsp<<" [m]"<<endl;
			cout<<" Range samples      : "<<procp.nrs<<" [samples]"<<endl;
			cout<<" Azimuth samples    : "<<procp.nazs<<" [samples]"<<endl;
		}
		
		//-------------------------------------------------------------------
		// map coordinate
		double SW = Sar.Nr() * procp.rpixsp;
		GEO<double> Gsn, Gsc, Gsf;
		GEO<double> Gcn, Gcc, Gcf;
		GEO<double> Gen, Gec, Gef;
		
		VEC<double> Ps, Ps1;
		Ps = sv.pos()[0];
		Ps1= sv.pos()[1];
		GetMapCoordinate(Sar, Orb, Ps, Ps1,	look, SW, Gsn, Gsc, Gsf);
		
		Ps = sv.pos()[(sv.GetNum()-1)/2];
		Ps1= sv.pos()[(sv.GetNum()-1)/2+1];
		GetMapCoordinate(Sar, Orb, Ps, Ps1,	look, SW, Gcn, Gcc, Gcf);
		
		Ps = sv.pos()[sv.GetNum()-1];
		Ps1= sv.pos()[sv.GetNum()-2];
		VEC<double> dir = (sv.pos()[sv.GetNum()-1] - sv.pos()[sv.GetNum()-2]);
		double dis = (Ps - Ps1).abs();
		Ps1= Ps + dis * dir;
		GetMapCoordinate(Sar, Orb, Ps, Ps1,	look, SW, Gen, Gec, Gef);
		
		// assign
		procp.map[0] = Gsn;
		procp.map[1] = Gsf;
		procp.map[2] = Gen;
		procp.map[3] = Gef;
		procp.map[4] = Gcc;
		
	}// End of write SLC or Both (*_focused.raw)
	
	if((WriteMode == 0) || (WriteMode == 1) || (WriteMode == 2)){
		//+------------------+
		//| Sensor Parameter |
		//+------------------+
		sar::write_SAR_par(file_SAR_par.c_str(), sarp);
		
		//+-------------------+
		//| Process Parameter |
		//+-------------------+
		sar::write_PROC_par(file_PROC_par.c_str(), procp);
		
		//+-----------------+
		//| Azimuth sample  |
		//+-----------------+
		D1<double> Ra(Sar.Na());
		Ra.Indgen();
		Ra = (Ra - double(Ra.GetNum()-1)/2) * (Vs/Sar.PRF());
		
		
		//+-----------------+
		//|   Export Data   |
		//+-----------------+
		Ra.WriteBinary(file_Ra.c_str());
		R0.WriteBinary(file_R0.c_str());
	}// End of write PARAMETER (p*.slc.par & SENSOR.par,*_R0.raw,*_Ra.raw)
}

void Usage(const string name){
	cout<<"+------------+"<<endl;
	cout<<"|    Usage   |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  "<<name<<" <file_SAR> <file_SV> <dir_RCS> <dir_out> <name>"<<endl;
	cout<<"             [-LV <Level_Index>] [-TX <Tx_Pol>] [-NA <Na>] [-AS <Angle_Squint>]"<<endl;
	cout<<"             [-h] [-CM] [-NINFO] [-WM <mode>] [-NNSV]"<<endl;
}

void Help(){
	cout<<"+------------+"<<endl;
	cout<<"|  Required  |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  file_SAR           [string] file of SAR Simulation Parameter file"<<endl;
	cout<<"  file_SV            [string] file of state vector"<<endl;
	cout<<"  dir_RCS            [string] dirctory of RCS"<<endl;
	cout<<"  dir_out            [string] dirctory of output"<<endl;
	cout<<"  name               [string] output file name"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"|  Optional  |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  -LV <Level_Index>  [x] Level index (default = 0 for total, -LV 1 means 1st reflection)"<<endl;
	cout<<"  -TX <Tx_Pol>       [string] Transmitted Polarization, e.g. 'H' or 'V' "<<endl;
	cout<<"  -NA <Na>           [sample] Azimuth samples"<<endl;
	cout<<"  -AS <Angle_Squint> [deg] SAR Squint angle"<<endl;
	cout<<"  -h                 [x] Show the help menu"<<endl;
	cout<<"  -CM                [x] Display the complete messages on console"<<endl;
	cout<<"  -NINFO             [x] Disable the detail SAR aquisition information? (Default is disable)"<<endl;
	cout<<"  -WM <mode>         [x] Write mode: (0) Focused SAR image(Default), (1) SAR Raw data, (2) Both Raw & focused data"<<endl;
	cout<<"  -NNSV              [x] Disable the re-normalize the altitude of state vector from input data. Default is enable."<<endl;
	cout<<"+------------+"<<endl;
	cout<<"|    Note    |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  1. The options value will replace the values in the file_SAR parameter file."<<endl;
	cout<<"  2. The last option is much higher priority"<<endl;
	cout<<"  3. When enable the '-NNSV' option, this program will NOT re-normalize the altitude of state vector. The all sv position"<<endl;
	cout<<"     will be as same as altitude in geodetic coordinate. Re-normalize state vector is default."<<endl;
}





int main(int argc, char** argv) {
	
	
	cout<<"****************************************************"<<endl;
	cout<<"*                     PASSED                       *"<<endl;
	cout<<"*--------------------------------------------------*"<<endl;
	cout<<"*        Copyright 2018, HomeStudio Taiwan         *"<<endl;
	cout<<"*               Created : 2010/09/06               *"<<endl;
	cout<<"*         Last Modified : 2020/09/18               *"<<endl;
	cout<<"*               Version : 3.0.3 (SAR)              *"<<endl;
	cout<<"*                Author : Cheng-Yen Chiang         *"<<endl;
	cout<<"****************************************************"<<endl;
	
	
	//+----------------------------------+
	//| Detect argument number of -h tag |
	//+----------------------------------+
	// Command Parser
	CmdParser cmp(argc, argv);
	string tmp;
	
	// Print Help
	if(cmp.GetVal("-h",  tmp)){ Usage(argv[0]); Help(); errorexit(); }
	
	// Check input number
    if(argc < 6){ Usage(argv[0]); errorexit(); }
	
	
	
	
	//+---------------------------+
	//| Read Input Parameters     |
	//+---------------------------+
	// required
	string file_Par      = string(argv[1]);
	string file_SV       = string(argv[2]);
	string dir_RCS       = string(argv[3]);
	string dir_out       = string(argv[4]);
	string name          = string(argv[5]);
	
	SAR Sar;				// SAR parameters
	ORB Orb;				// Orbit parameters
	EF Ef;					// Electric field parameters
	MultiAngle MulAng;		// Multi-angle parameters
	MeshDef mesh;			// Incident Mesh parameters
	TAR Tar;				// Target
	int ReduceFactor = 1;	// Reducing Factor in Azimuth
	size_t ProcLevel = 0;	// Processing level (e.g. ProcLevel = 0, for total)
	
	// Read parameters
	if(cmp.GetVal("-LA", tmp)){
		io::read::SimPar(file_Par, Sar, Orb, Ef, MulAng, mesh, ReduceFactor, str2num<double>(tmp));
	}else{
		io::read::SimPar(file_Par, Sar, Orb, Ef, MulAng, mesh, ReduceFactor);
	}
	Tar.ASP() = MulAng.AspFrom();
	

	if(cmp.GetVal("-LV", tmp)){ ProcLevel = str2num<long>(tmp); }			// Processing level
	if(cmp.GetVal("-TX", tmp)){ Ef.SetTxPol(tmp); }							// Tx polarization
	if(cmp.GetVal("-NA", tmp)){ Sar.Na() = str2num<long>(tmp); }			// number of azimuth
	if(cmp.GetVal("-AS", tmp)){ Sar.theta_sqc() = str2num<double>(tmp); }	// Squint Angle

	// Display message
	bool IsCOMPLEX_MESSAGE = false;
	if(cmp.GetVal("-CM", tmp)){ IsCOMPLEX_MESSAGE = true; }
	
	// Disable SAR aquisition infomation
	bool IsDisableInfo = false;
	if(cmp.GetVal("-NINFO", tmp)){ IsDisableInfo = true; }

	// Write RAW data?
	int WriteMode = 0;
	if(cmp.GetVal("-WM", tmp)){ WriteMode = str2num<int>(tmp); }

	// Normalize SV altitude
	bool IsSVNormalize = true;
	if(cmp.GetVal("-NNSV", tmp)){ IsSVNormalize = false; }
	
	
	
	
	new_dir:Dir dr(dir_RCS);
	D1<string> file_RCSMeta = dr.GetSuffix("meta");
	D1<string> file_RCSData = dr.GetSuffix("dat");
	file_RCSMeta = dir_RCS + file_RCSMeta;
	file_RCSData = dir_RCS + file_RCSData;
	
	
	// UPDATE
	D1<long> dim(4);
	D1<double> ThetaSeriesRCS, PhiSeriesRCS;
	D1<double> ThetaSeries, PhiSeries;
	// All tempolary variable will be remove after this processing stage
	{
		// some classess
		SAR Sar2;
		EF Ef2;
		MultiAngle MulAng2;
		MeshDef mesh2;
		int ReduceFactor2;
		
		// Read *.meta
		io::read::SimPar(file_RCSMeta[0], Sar2, Ef2, MulAng2, mesh2, ReduceFactor2);


		cout<<"+-------------------------------------+"<<endl;
		cout<<"|                Level                |"<<endl;
		cout<<"+-------------------------------------+"<<endl;
		cout<<" Max Level = "<<Ef2.MaxLevel()<<endl;
		if(ProcLevel == 0){
			cout<<" Level = Total"<<endl;
		}else{
			cout<<" Level = "<<ProcLevel<<endl;
		}
		if(ProcLevel > Ef2.MaxLevel()){
			cout<<endl;
			cerr<<" ERROR::Cannot found this Processing Level (ProcLevel="<<ProcLevel<<")"<<endl;
			cerr<<"        Condition: ProcLevel <= Ef.MaxLevel()="<<Ef.MaxLevel()<<endl;
			cerr<<"        Note: Ef.MaxLevel=4 means {'Level1', 'Level2', 'Level3', 'Level4'}"<<endl;
			cerr<<"              ProcLevel=4 means processing 'Level4'"<<endl;
			exit(EXIT_FAILURE);
		}
		
		// Update all structures
		UpdateParameters(Sar2, Ef2, MulAng2, mesh2, // new
						 Sar, Ef, MulAng, mesh);	// old (need to be updated)
		//+----------------------+
		//|      Dimension       |
		//+----------------------+
		dim[0] = Sar.Nr();
		dim[1] = MulAng2.GetPhiSeries().size();
		dim[2] = MulAng2.GetThetaSeries().size();
		dim[3] = Ef.RxPol().length();
		ThetaSeriesRCS = vector2D1(MulAng2.GetThetaSeries());
		PhiSeriesRCS   = vector2D1(MulAng2.GetPhiSeries());
		ThetaSeries    = vector2D1(MulAng.GetThetaSeries());
		PhiSeries      = vector2D1(MulAng.GetPhiSeries());
		
		cout<<"+------------------------------------+"<<endl;
		cout<<"|      RCS Dimension Summary         |"<<endl;
		cout<<"+------------------------------------+"<<endl;
		cout<<"Number of range sample = "<<dim[0]<<endl;
		cout<<"Number of Phi_asp      = "<<dim[1]<<endl;
		cout<<"Number of Theta_look   = "<<dim[2]<<endl;
		cout<<"Number of RxPol        = "<<dim[3]<<endl;
	}
	

	
	
	//+===========================================================================+
	//|                                                                           |
	//|                            SAR Simulation                                 |
	//|                                                                           |
	//+===========================================================================+
	
	//====================================================================================
	// State Vector(SV) Interpolation
	//====================================================================================
	// Read SV
	SV<double> TER_SV = io::read::SV(file_SV.c_str());
	
	// New time interval
	double dt = 1/Sar.PRF();
	
	// Create time series
	SV<double> sv(Sar.Na());
	
	// Calculate Central Time [GPS]
	double t_c = (TER_SV.t()[TER_SV.GetNum()-1] + TER_SV.t()[0])/2;
	linspace(t_c - dt*Sar.Na()/2, t_c + dt*Sar.Na()/2, sv.t());
	
	// Interpolation
	sar::sv_func::Interp(TER_SV, sv);
	
	
	// Re-normalize the altitude of SV, replace the original SV position values
	if(IsSVNormalize){
		// Normalize height
		D1<double> h(sv.GetNum());
		for(long i=0;i<sv.GetNum();++i){
			h[i] = sar::ECR2Gd(sv.pos()[i], Orb).h();
		}
		double hmean = mat::total(h)/h.GetNum();
		for(long i=0;i<sv.GetNum();++i){
			GEO<double> gd = sar::ECR2Gd(sv.pos()[i], Orb);
			gd.h() = hmean;
			sv.pos()[i] = sar::Gd2ECR(gd, Orb);
		}
	}
	
	// Calculate distance
	double La=0;
	for(long i=0;i<sv.GetNum()-1;++i){
		La += (sv.pos()[i] - sv.pos()[i+1]).abs();
	}
	double Vs=0;
	for(long i=0;i<sv.GetNum();++i){
		Vs += sv.vel()[i].abs();
	}
	Vs /= sv.GetNum();
	

	//====================================================================================
	// Echo Signal
	//====================================================================================
	//+-----------------+
	//|  Basic Sereis   |
	//+-----------------+
	D1<double> ft(Sar.Nr());
	linspace(-Sar.Fr()/2, Sar.Fr()/2, ft);
	fftshift(ft);
	D1<double> t(Sar.Nr());
	t.Indgen();
	t = (t - Sar.Nr()/2.) / Sar.Fr();
	D1<double> R0 = t * (def::C/2.);
	// Make range freq. series & freq. domain window series
	D1<double> freq = ft + Sar.f0();
	D1<double> Wr = sar::Wr(ft, abs(Sar.Kr())*Sar.Tr());
	
	
	//+------------------+
	//|    Allocation    |
	//+------------------+
	D1<double> theta_sq(Sar.Na());				// Squnit angle
	D1<CPLX<double> > line(Sar.Nr());			// single range line for read RCS
	
	
	
	//+------------------------+
	//|    BIG Loop - Start    |
	//+------------------------+
	clock_t tic = def_func::tic();
	cout<<endl;
	cout<<">> SAR Echo Signal Generation <<"<<endl;

	

	//+------------------+
	//|   Polarization   |
	//+------------------+
	for(unsigned long ipo=0;ipo<Ef.RxPol().length();++ipo){			// Polarization
		// Assign polarization
		string POL = Ef.RxPol().substr(ipo,1);
		//+------------------+
		//|   Theta(Look)    |
		//+------------------+
		for(size_t it=0;it<ThetaSeries.GetNum();++it){		// Theta (Look) angle
			// Set Main beam look angle for SAR
			Sar.SetTheta_l_MB(ThetaSeries[it]);
			//====================================================================================
			// Calculate scene center & boundary
			//====================================================================================
			// Center position of SV (Psc)
			long idx_c = (sv.GetNum()-1)/2;
			VEC<double> Ps  = sv.pos()[idx_c];		// Center position of SV
			VEC<double> Ps1 = sv.pos()[idx_c+1];	// next by Ps in one interval
			VEC<double> Psg = sar::find::ProjectPoint(Ps, Orb);
			VEC<double> uv  = sar::find::LookAngleLineEq(Ps, Psg, Ps1, Sar.theta_sqc(), Sar.theta_l_MB());
			// Scene center (ground)
			VEC<double> Psc = sar::find::BeamLocationOnSurface(uv, Ps, Orb);
			
			//====================================================================================
			// Parameters Check (Slant Range, Nr)
			//====================================================================================
			// Max Slant Range distance, sensor to scene center
			double Rc_min = (sv.pos()[(sv.GetNum()-1)/2] - Psc).abs();
			double Rc_max = (sv.pos()[0] - Psc).abs();
			double Rmax   = Rc_max + Sar.SWrg()/2;
			double Rmin   = Rc_min - Sar.SWrg()/2;
			

			if(!IsDisableInfo && ipo == 0){
				long NrFreq_trad = sar::cw::TraditionalRangeSampleNumber(Rmax, Sar.Fr());
				double SW_eff    = sar::cw::EffectiveSwath(Sar.Nr(), Sar.Fr());
				long Nr_min = sar::cw::TraditionalRangeSampleNumber(Sar.SWrg(), Sar.Fr());
				
				string flag_NrFreq = (Sar.Nr() < NrFreq_trad)? "(NEED_SHIFT)":"(OK)";
				string flag_SW     = (Sar.SWrg() < SW_eff)?    "(ACCEPT)":"(REJECT)";
				flag_NrFreq = (Sar.Nr() < Nr_min)? "(TOO_SMALL)":flag_NrFreq;
				
				//+-------------------------------------------+
				//| Note:                                     |
				//|   The SWrg smaller, the Nr min smaller    |
				//|   The NrFreq larger, the SW_eff bigger    |
				//|   NrFreq(=Nr) Must be larger than Nr min  |
				//+-------------------------------------------+
				cout<<" SW_eff          [m] = "<<SW_eff<<endl;
				cout<<" SWrg user       [m] = "<<Sar.SWrg()<<endl;
				cout<<" Nr Trad.        [#] = "<<NrFreq_trad<<endl;
				cout<<" Nr              [#] = "<<Sar.Nr()<<endl;
				cout<<" Nr min          [#] = "<<Nr_min<<endl;
				string lin = "-";
				string spa = " ";
				string fg1 = (flag_SW == "(ACCEPT)" && flag_NrFreq == "(OK)")? "[>]":spa;
				string fg2 = (flag_SW == "(ACCEPT)" && flag_NrFreq == "(NEED_SHIFT)")? ">":spa;
				string fg3 = (flag_SW == "(ACCEPT)" && flag_NrFreq == "(TOO_SMALL)")? ">":spa;
				string fg4 = (flag_SW == "(REJECT)" && flag_NrFreq == "(OK)")? "[->]":spa;
				string fg5 = (flag_SW == "(REJECT)" && flag_NrFreq == "(TOO_SMALL)")? ">":spa;
				cout<<"+-"<<lin<<"---------------------------------------------------------------------------------+"<<endl;
				cout<<"| "<<spa<<" flag_SW    flag_NrFreq    Flag    Description                                   |"<<endl;
				cout<<"+-"<<lin<<"---------------------------------------------------------------------------------+"<<endl;
				cout<<"| "<<fg1<<" ACCEPT      OK             o      Using tradition FFT                           |"<<endl;
				cout<<"| "<<fg2<<" ACCEPT      NEED_SHIFT     o      Shift must be used                            |"<<endl;
				cout<<"| "<<fg3<<" ACCEPT      TOO_SMALL      x      NrFreq too small                              |"<<endl;
				cout<<"| "<<fg4<<" REJECT      OK             x      Need to enlarge the Sar.SWrg values           |"<<endl;
				cout<<"| "<<fg5<<" REJECT      TOO_SMALL      x      Need to enlarge the Sar.SWrg values & NrFreq  |"<<endl;
				cout<<"+-"<<lin<<"---------------------------------------------------------------------------------+"<<endl;
				if(flag_SW == "(REJECT)" || flag_NrFreq == "(TOO_SMALL)"){
					exit(EXIT_FAILURE);
				}
				
				//====================================================================================
				// Parameters Check (Azimuth, Na)
				//====================================================================================
				//+---------------------------+
				//| Synthetic Aperture Length |
				//+---------------------------+
				D1<double> Ls_rg(2);
				Ls_rg[0] = 2 * Rmin * atan(Sar.theta_az()/2);
				Ls_rg[1] = 2 * Rmax * atan(Sar.theta_az()/2);
				D1<long>  Ls_rg_pix(2);
				Ls_rg_pix[0] = Ls_rg[0]/Vs*Sar.PRF();
				Ls_rg_pix[1] = Ls_rg[1]/Vs*Sar.PRF();
				
				string fg6=" ", fg7=" ", fg8=" ";
				if(Sar.Na() >= Ls_rg_pix[1]){
					fg6 = ">";
				}else if((Sar.Na() >= Ls_rg_pix[0]) && (Sar.Na() < Ls_rg_pix[1])){
					fg7 = ">";
				}else{
					fg8 = ">";
				}
				cout<<endl;
				cout<<" Vs            [m/s] = "<<Vs<<endl;
				cout<<" min(Ls),max(Ls) [m] = "<<Ls_rg[0]<<", "<<Ls_rg[1]<<endl;
				cout<<" min(Ls),max(Ls) [#] = "<<Ls_rg_pix[0]<<", "<<Ls_rg_pix[1]<<endl;
				cout<<"+-"<<lin<<"---------------------------------------------------------------------------------+"<<endl;
				cout<<"| "<<spa<<" flag_Na   Condition   Value                     Description                     |"<<endl;
				cout<<"+-"<<lin<<"---------------------------------------------------------------------------------+"<<endl;
				cout<<"| "<<fg6<<" ACCEPT    GOOD        Na > max(Ls)              Can be a perfect SAR image      |"<<endl;
				cout<<"| "<<fg7<<" ACCEPT    OK          min(Ls) <= Na < max(Ls)   So far so good                  |"<<endl;
				cout<<"| "<<fg8<<" ACCEPT    WARRNING    Na < min(Ls)              Can be a unreasonable SAR image |"<<endl;
				cout<<"+-"<<lin<<"---------------------------------------------------------------------------------+"<<endl;
			}
			
			//====================================================================================
			// Get SINGLE suitable RCS file
			//====================================================================================
			string file_RCSDataDest = SearchSuitableRCSFile(ThetaSeriesRCS, ThetaSeries, ProcLevel, file_RCSData, Ef, POL, it);
			
			cout<<"file_RCSDataDest = '"<<file_RCSDataDest<<"'"<<endl;
			if(file_RCSDataDest == ""){
				cerr<<"ERROR::Cannot found RCSDataset -> '"<<file_RCSDataDest<<"'"<<endl;
				exit(EXIT_FAILURE);
			}
			
			
			//+------------------+
			//|   Phi(Aspect)    |
			//+------------------+
			for(size_t ip=0;ip<MulAng.GetPhiSeries().size();++ip){	// Phi (aspect) angle
																			// Generation
				def::CLOCK clock(tic);
				size_t N1 = ThetaSeries.GetNum()*PhiSeries.GetNum();
				size_t N2 = N1 * Ef.RxPol().length();
				bool ShowSummary = (ipo*N1 + it*PhiSeries.GetNum() + ip + 1 == N2)? true:false;
				// Display Message
				if(IsCOMPLEX_MESSAGE){
					cout<<ipo*N1 + it*PhiSeries.GetNum() + ip + 1<<" / "<<N2<<endl;
					cout<<"Theta_look [deg] = "<<rad2deg(ThetaSeries[it])<<endl;
					cout<<"Phi_asp    [deg] = "<<rad2deg(PhiSeries[ip])<<endl;
				}else{
					def_func::ProgressBar(ipo*N1 + it*PhiSeries.GetNum() + ip, N2, 50, 1, clock);
				}
				GenerateSAR(Sar, sv, Orb, mesh, ThetaSeriesRCS, PhiSeriesRCS, PhiSeries[ip], R0, freq, Wr, Psc, Rmin, Rc_min, file_RCSDataDest, ProcLevel,
							Ef, ThetaSeries[it], POL, dir_out, name, WriteMode, Vs, TER_SV, theta_sq, ShowSummary, IsCOMPLEX_MESSAGE);

			}// end of aspect angle
			
			
		}// end of look angle
	}// end of polarization
	
	cout<<">>      Simulation Finish     <<"<<endl;
	def_func::toc(tic);
	cout<<endl;
	
	//+------------------------+
	//|    BIG Loop - End      |
	//+------------------------+
	
	
		
	
	
	cout<<"Finsih!"<<endl;
#ifdef _WIN32
	system("pause");
#endif
	return 0;
}











