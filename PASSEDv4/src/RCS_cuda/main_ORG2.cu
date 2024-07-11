//
//  main_ORG2.cu
//  PASSEDv4 (RCS)
//
//  Created by Steve Chiang on 11/116/22.
//  Copyright (c) 2022 Steve Chiang. All rights reserved.
//

#define LIMITNUMBER

#ifdef DEBUG
#undef DEBUG
#endif
//#define DEBUG


#include <sar/sar.h>
#include <bvh/obj.h>
#include <bvh/ray.h>
#include <rcs/raytrace.h>
#include <mesh/mesh.h>
#include <mesh/intersect.h>
#include <rcs/ems.h>
#include <rcs/ptd.h>
#include <mesh/cad.h>
#include <cuda/cumain.cuh>
#include <cuda/cumain_ptd.cuh>


using namespace std;
using namespace vec;
using namespace mesh;
using namespace intersect;
using namespace ems;
using namespace ptd;
using namespace sv;
using namespace cad;
using namespace cu;


void Usage(const string& name){
	cout<<"+------------+"<<endl;
	cout<<"|    Usage   |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  "<<name<<" <file_CNF> <file_SV> <file_Material> <file_CAD> <dir_out> <name>"<<endl;
	cout<<"             [-h] [-PEC] [-SC <Scale>] [[-PO] [-PTD]] [-GPU [-GPUCM] [-GPUNST <NSTREAM>]] [-CADUNIT <Unit>] [-SARMODE <mode>]"<<endl;
}

void Help(){
	cout<<"+------------+"<<endl;
	cout<<"|  Required  |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  file_CNF           [string] SAR echo simulation parameter file in json format"<<endl;
	cout<<"  file_SV            [string] State vector file in ASCII format"<<endl;
	cout<<"  file_Material      [string] Material list table file in ASCII format"<<endl;
	cout<<"  file_CAD           [string] 3D model in 3ds format"<<endl;
	cout<<"  dir_out            [string] Output folder"<<endl;
	cout<<"  name               [string] Output file name"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"|  Optional  |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  -h                 [x] Show the help menu"<<endl;
	cout<<"  -PEC               [x] Enable the PEC to be the ONLY ONE material in object"<<endl;
	cout<<"  -SC <Scale>        [x] Mesh generation scale factor, the larger is much more grid in incident mesh, recommand >= 1"<<endl;
	cout<<"  -PO                [x] Enable scattering by PO"<<endl;
	cout<<"  -PTD               [x] Enable diffraction by PTD"<<endl;
	cout<<"  -GPU               [x] Enable CUDA GPU"<<endl;
	cout<<"  -GPUCM             [x] Enable GPU complete message on console (Enable after -GPU)"<<endl;
	cout<<"  -CADUNIT <Unit>    [string] Set CAD unit such as {'mm', 'm', 'cm', 'inch'}"<<endl;
	cout<<"  -SARMODE <mode>    [int] Set the SAR data acquisition mode. Default is Stripmap"<<endl;
	cout<<"                     1: Stripmap (Sensor is linear, constant antenna pointing)"<<endl;
	cout<<"                     2: Spotlight (Sensor is linear, antenna pointing is function of squint)"<<endl;
	cout<<"                     3: ISAR (As same as Spotlight but equal slant range distance)"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"|    Note    |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  1. When -PO is used, the PO calculation is activated ONLY. When -PTD is used, the PTD calculation is activated ONLY."<<endl;
	cout<<"     When -PO and -PTD are present at the same time, the PO, PTD and Total calculations are enabled."<<endl;
	cout<<"     When both -PO and -PTD are not present, the PO, PTD and Total calculations are also started (for compatibility with older versions).."<<endl;
}





//+======================================================+
//|                        Main                          |
//+======================================================+
int main(int argc, char** argv) {

#pragma region start_program_with_arguments
	cout<<"****************************************************"<<endl;
	cout<<"*                    PASSEDv4                      *"<<endl;
	cout<<"*--------------------------------------------------*"<<endl;
	cout<<"*        Copyright 2022, HomeStudio Taiwan         *"<<endl;
	cout<<"*               Created : 2010/09/06               *"<<endl;
	cout<<"*         Last Modified : 2022/12/01               *"<<endl;
	cout<<"*               Version : 4.0.2 (RCS/CUDA)         *"<<endl;
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
    if(argc < 7){ Usage(argv[0]); errorexit(); }



	//+---------------------------+
	//| Read Input Parameters     |
	//+---------------------------+
	string file_Cnf      = string(argv[1]);	// *.json
	string file_SV       = string(argv[2]);	// *.stv
	string file_Material = string(argv[3]); // Material table file
	string file_CAD      = string(argv[4]);	// *.3ds
	string dir_out       = string(argv[5]);	// Export folder of SAR echo simulation results
	string name          = string(argv[6]);	// RCS file name

	// SAR acquisition mode
	//     1: Strimap (Sensor is linear, const antenna pointing)
	//     2: Spotlight (Sensor is linear, antenna pointing is function of squint)
	//     3: ISAR (As same as Spotlight but equal slant range distance)
	int SARMode          = 1;	// 1: Strimap, 2:Spotlight, 3:ISAR
	if(cmp.GetVal("-SARMODE", tmp)){ SARMode = str2num<int>(tmp); }

	// Normalize SV altitude
	bool IsSVNormalize = true;
	if(cmp.GetVal("-NNSV", tmp)){ IsSVNormalize = false; }

	// Mesh Scale factor
	double setScale = -1;
	if(cmp.GetVal("-SC", tmp)){ setScale = str2num<double>(tmp); }

	// Normalize SV altitude
	bool IsPEC = false;
	if(cmp.GetVal("-PEC", tmp)){ IsPEC = true; }

	// CAD Unit
	string CADUnit = "m";
	if(cmp.GetVal("-CADUNIT", tmp)){ CADUnit = tmp; }

	// PO
	bool IsPO = false;
	if(cmp.GetVal("-PO", tmp)){ IsPO = true; }

	// PTD
	bool IsPTD = false;
	if(cmp.GetVal("-PTD", tmp)){ IsPTD = true; }

	// PO + PTD
	if(!IsPO && !IsPTD){
		IsPO  = true;
		IsPTD = true;
	}

	// GPU
	bool IsGPU = false;
	dim3 NThread1, NThread2, NThread3, NThread4;
	bool IsGPUShow = false;
	string GPUName = cu::GetGPUName();
	if(cmp.GetVal("-GPU", tmp)){
		IsGPU = true;
		string IsGPUShowTag = "Disable";
		// NThread setting by GPU device Name
		cu::GetGPUParameters(NThread1, NThread2, NThread3, NThread4, false);
		if(cmp.GetVal("-GPUCM", tmp)){ IsGPUShow = true; IsGPUShowTag = "Enable"; }	// GPU Complete Message
		cout<<"+--------------------------+"<<endl;
		cout<<"|    GPU Device Summary    |"<<endl;
		cout<<"+--------------------------+"<<endl;
		cout<<" Name                    : "<<GPUName<<" [ON]"<<endl;
		cout<<" NThread1 [cuRayTracing] : ("<<NThread1.x<<","<<NThread1.y<<","<<NThread1.z<<")"<<endl;
		cout<<" NThread2 [cuPO]         : ("<<NThread2.x<<","<<NThread2.y<<","<<NThread2.z<<")"<<endl;
		cout<<" NThread3 [cuBlockSum]   : ("<<NThread3.x<<","<<NThread3.y<<","<<NThread3.z<<")"<<endl;
		cout<<" Show complete Message   : "<<IsGPUShowTag<<endl<<endl;
	}else{
		cout<<"+--------------------------+"<<endl;
		cout<<"|    GPU Device Summary    |"<<endl;
		cout<<"+--------------------------+"<<endl;
		cout<<" Name                    : "<<GPUName<<" [OFF]"<<endl<<endl;
	}

	// SAR Mode
	cout<<"+--------------------------+"<<endl;
	cout<<"|    SAR acquisition mode  |"<<endl;
	cout<<"+--------------------------+"<<endl;
	if(SARMode == 1){
		cout<<" SAR Mode                : (1)Stripmap"<<endl;
	}else if(SARMode == 2){
		cout<<" SAR Mode                : (2)Spotlight"<<endl;
	}else if(SARMode == 3){
		cout<<" SAR Mode                : (3)Circular SAR or Inverse SAR"<<endl;
	}else{
		cerr<<"No this kind of SARMode = "<<SARMode<<endl;
		exit(EXIT_FAILURE);
	}

#pragma endregion

	//+==================================================================================+
	//|                     Read ConfJson, Material and SV files                         |
	//+==================================================================================+
#pragma region Read_ConfJson_Material_SV
	//+==================================================================================+
	//|                              Read parameters                                     |
	//+==================================================================================+
	SAR Sar;			// SAR parameters
	EF Ef;				// Electric field parameters
	def::ORB Orb;		// Orbit parameters
	MeshDef mesh;		// Incident Mesh parameters
	double TargetAsp;
	double TargetLon, TargetLatGd;

	// Read SAR.conf
	io::read::SimPar(file_Cnf, Sar, Ef, mesh, Orb, TargetAsp, TargetLon, TargetLatGd);
	if(setScale > 0){
		mesh.Scale() = setScale;
	}

	// Read Material
	MaterialDB MatDB(file_Material);

	// Print out
	Sar.Print();
	MatDB.Print1Line();

//	sar::make::FakePath(Sar.PRF(), file_SV, Sar.Na());

	//+==================================================================================+
	//|                          Get Look, Asp and Squint angles                         |
	//+==================================================================================+
	// Pre-allocation
	double dt = 1/Sar.PRF();
	def::ANG ang(1);

	//+==================================================================================+
	//|                   Read State Vector(SV) & Interpolation to PRF                   |
	//+==================================================================================+
	// Read SV
	SV<double> sv_in = io::read::SV(file_SV.c_str());

	// Create time series
	SV<double> sv(Sar.Na());

	// Calculate Central Time [GPS]
	double t_c = (sv_in.t()[sv_in.GetNum()-1] + sv_in.t()[0])/2;
	linspace(t_c - dt*double(Sar.Na())/2, t_c + dt*double(Sar.Na())/2, sv.t());

	if( abs((sv_in.t()[1] - sv_in.t()[0]) - (1.0/Sar.PRF())) > 1e-7 ){
		cout<<"+-----------------------------+"<<endl;
		cout<<"|    SV need interpolation    |"<<endl;
		cout<<"+-----------------------------+"<<endl;
		cout<<"dt of 1/PRF = "<<1.0/Sar.PRF()<<" [sec]"<<endl;
		cout<<"dt of input = "<<sv_in.t()[1] - sv_in.t()[0]<<" [sec]"<<endl;
		cout<<"diff        = "<<abs((sv_in.t()[1] - sv_in.t()[0]) - (1.0/Sar.PRF()))<<" [sec]"<<endl;

//		cout<<endl<<"Mark-1"<<endl;
		cout<<"sv_in.GetNum()        = "<<sv_in.GetNum()<<endl;
		cout<<"sv.GetNum() (Before)  = "<<sv.GetNum()<<endl;
		// Interpolation
		sar::sv_func::Interp(sv_in, sv);
		cout<<"sv.GetNum() (After)   = "<<sv.GetNum()<<endl;
//		cout<<"Mark-2"<<endl<<endl;;
		// Re-normalize the altitude of SV, replace the original SV position values
		if(IsSVNormalize){
			// Normalize height
			D1<double> h(sv.GetNum());
			for(long i=0;i<sv.GetNum();++i){
				h[i] = sar::ECR2Gd(sv.pos()[i], Orb).h();
			}
			double hmean = mat::total(h)/double(h.GetNum());
			for(long i=0;i<sv.GetNum();++i){
				GEO<double> gd = sar::ECR2Gd(sv.pos()[i], Orb);
				gd.h() = hmean;
				sv.pos()[i] = sar::Gd2ECR(gd, Orb);
			}
		}
		cout<<"dt of after = "<<sv.t()[1] - sv.t()[0]<<" [sec]"<<endl;
	}else{
		cout<<"+-----------------------------+"<<endl;
		cout<<"|  SV NO need interpolation   |"<<endl;
		cout<<"+-----------------------------+"<<endl;
		cout<<"dt of 1/PRF = "<<1.0/Sar.PRF()<<" [sec]"<<endl;
		cout<<"dt of input = "<<sv_in.t()[1] - sv_in.t()[0]<<" [sec]"<<endl;
		cout<<"diff        = "<<abs((sv_in.t()[1] - sv_in.t()[0]) - (1.0/Sar.PRF()))<<" [sec]"<<endl;
		// Duplicate without interpolation (the input t interval is meet the 1/PRF)
		sv = sv_in;
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
	Vs /= double(sv.GetNum());
	cout<<"+-----------------------------+"<<endl;
	cout<<"|      State vector result    |"<<endl;
	cout<<"+-----------------------------+"<<endl;
	{
		D1<double> h(sv.GetNum());
		for(long i=0;i<sv.GetNum();++i){
			h[i] = sar::ECR2Gd(sv.pos()[i], Orb).h();
		}
		cout<<"Vs          = "<<Vs<<" [m/s]"<<endl;
		cout<<"min(h)      = "<<min(h)<<" [m]"<<endl;
		cout<<"max(h)      = "<<max(h)<<" [m]"<<endl;
		cout<<"mean(h)     = "<<mean(h)<<" [m]"<<endl;
	}
#pragma endregion

	//+==================================================================================+
	//|                 Calculate Aspect angle for different SAR mode                    |
	//+==================================================================================+
#pragma region Calculate_Asp_angle_for_different_SAR_mode
	// Target position
	VEC<double> Pt;

	// Calculate aspect angle (ang) by SAR acquisition mode
	//     1: Strimap (Sensor is linear, const antenna pointing)
	//     2: Spotlight (Sensor is linear, antenna pointing is function of squint)
	//     3: ISAR (As same as Spotlight but equal slant range distance)
	if(SARMode == 1 || SARMode == 2){	// Stripmap(1) or Spotlight(2)
		//+==================================================================================+
		//|             Calculate scene center (Target's center) & boundary                  |
		//+==================================================================================+
		{
			long idx_c = (sv.GetNum()-1)/2;
			VEC<double> Ps  = sv.pos()[idx_c];		// Center position of SV
			VEC<double> Ps1 = sv.pos()[idx_c+1];	// next by Ps in one interval
			VEC<double> Psg = sar::find::ProjectPoint(Ps, Orb);
			VEC<double> uv  = sar::find::LookAngleLineEq(Ps, Psg, Ps1, Sar.theta_sqc(), Sar.theta_l_MB());
			// Scene center (ground)
			Pt = sar::find::BeamLocationOnSurface(uv, Ps, Orb);
		}

		//+==================================================================================+
		//|                           Calculate look & squint                                |
		//+==================================================================================+
		// TODO: 當squint沒有切過0時，會出現錯誤
//		def::ANG ang(sv.GetNum());
		ang = def::ANG(sv.GetNum());

		{
			for(size_t i=0;i<ang.GetNum()-1;++i){
				VEC<double> Ps  = sv.pos()[i];
				VEC<double> Ps1 = sv.pos()[i+1];
				VEC<double> Psg = sar::find::ProjectPoint(Ps, Orb);
				double h  = sar::ECR2Gd(Ps).h();
				double R  = (Ps - Pt).abs();
				// Normal vector of zero Doppler plane
				VEC<double> uv_tmp = vec::Unit( vec::cross( (Psg - Ps), (Ps1 - Ps) ) );
				VEC<double> uv0 = vec::Unit( vec::cross(uv_tmp, (Psg-Ps)) );
				// Project point from Pt to Zero Doppler plane
				VEC<double> Pt0;
				vec::find::MinDistanceFromPointToPlane(uv0, Ps, Pt, Pt0);
				// Calculate Look and squint
				VEC<double> Pt0Min;
				vec::find::MinDistanceFromPointToLine(Ps, Psg, Pt0, Pt0Min);
				// Sign
				VEC<double> PsPt  = Unit(Pt  - Ps);
				VEC<double> PsPs1 = Unit(Ps1 - Ps);
				double sgn = (dot(PsPt, PsPs1) > 0)? +1:-1;
				// all angles
				double h2 = (Ps - Pt0Min).abs();
				double R0 = (Ps - Pt0).abs();
				double Look = acos(h2/R0);
				double squint = 0;
				if(abs(R0 - R) > 1e-8){
					squint = sgn * acos(R0/R);
				}
				// Project point of Ps
				double RgPt  = (Pt0Min - Pt).abs();
				double RgPt0 = (Pt0Min - Pt0).abs();
				double asp = sgn * acos(RgPt0/RgPt);
				double RgPt0_div_RgPt = RgPt0/RgPt;
//				if(abs(RgPt0_div_RgPt - 1) < 1e-16){
//					// acos(RgPt0/RgPt) -> 0 [deg]
//					asp = 0;
//				}
				if(RgPt0_div_RgPt > 1.0){ asp = deg2rad(0); }
				if(RgPt0_div_RgPt < 0.0){ asp = deg2rad(90); }
				// Assignment
				ang.Look()[i] = Look;
				ang.Squint()[i] = squint;
				ang.Asp()[i] = asp;
//				printf("uv0 = [%.4f, %.4f, %.4f], Pt0 = [%.4f, %.4f, %.4f], R0 = %.4f\n", uv0.x(), uv0.y(), uv0.z(), Pt0.x(), Pt0.y(), Pt0.z(), R0);
				if(i >= ang.GetNum()/2-1-10 && i <= ang.GetNum()/2-1+10){
					printf("ic = % 3d, Look = %.4f, squint = % .4f, asp = % .4f, theta_sqc_in = % .4f\n", int(double(i)-double(ang.GetNum())/2+1), rad2deg(Look), rad2deg(squint), rad2deg(asp), rad2deg(Sar.theta_sqc()));
				}
			}
			// Last value
			ang.Look()[ang.GetNum()-1]   = ang.Look()[ang.GetNum()-2];
			ang.Squint()[ang.GetNum()-1] = ang.Squint()[ang.GetNum()-2];
			ang.Asp()[ang.GetNum()-1]    = ang.Asp()[ang.GetNum()-2];
		}
	}else if(SARMode == 3){	// ISAR(3) or Circular SAR


		cout<<"1/PRF          = "<<1.0/Sar.PRF()<<endl;
		cout<<"dt             = "<<dt<<endl;
		cout<<"sv_in.GetNum() = "<<sv_in.GetNum()<<endl;
		cout<<"dt             = "<<sv_in.t()[1] - sv_in.t()[0]<<endl;
		cout<<"sv_in:"<<endl;	sv_in.Print();
		cout<<"sv.GetNum()    = "<<sv.GetNum()<<endl;
		cout<<"dt             = "<<sv.t()[1] - sv.t()[0]<<endl;
		cout<<"sv:"<<endl;		sv.Print();

		Pt = sar::Gd2ECR(sar::GEO<double>(TargetLon, TargetLatGd, 0));


		// (0) Prepare: Local Earth's radius
		// Convert from geodetic to geocentric
		sar::GEO<double> gd(0, TargetLatGd, 0);
		sar::GEO<double> gc = sar::Gd2Gc(gd, Orb);
		double TargetLatGc = gc.lat();
		double loc_Re = orb::RadiusGeocentric(TargetLatGc, Orb);

		// (1) Make Euler rotation matrix
		double angZ = TargetLon;
		double angY = deg2rad(90) - TargetLatGc;
		double angX = deg2rad(0);

		D2<double> Mz = Rz(angZ);
		D2<double> My = Ry(angY);
		D2<double> Mx = Rx(angX);
		D2<double> M  = Mz * My * Mx;

		D2<double> M_inv = M.Invert();

		// (2) Apply inverse Euler transformation
		D2<double> xyz1(3,1), xyz2(3,1);
		ang = def::ANG(sv.GetNum());
		for(size_t i=0;i<sv.GetNum();++i){
			// Assign to D2
			xyz2 = VEC2D2(sv.pos()[i], 3, 1);
			// Multiply
			xyz1 = M_inv * xyz2;
			// (3) Shift form North Pole to origin
			xyz1[2][0] -= loc_Re;
			// (4) Cartesian to Spherical coordinate
			double xy = sqrt(Square(xyz1[0][0]) + Square(xyz1[1][0]));
			double r  = sqrt(Square(xyz1[2][0]) + Square(xy));
			double el = atan2(xyz1[2][0], xy);
			double az = atan2(xyz1[1][0], xyz1[0][0]);

			// Assignment
			ang.Look()[i] = el;
			ang.Squint()[i] = deg2rad(0);
			ang.Asp()[i] = az;

		}
	}else{
		cerr<<"ERROR::No this kind of SARMode = "<<SARMode<<endl;
		exit(EXIT_FAILURE);
	}

	// Print ANG class
	ang.Print();


#pragma endregion

	//+==================================================================================+
	//|                                    Read 3ds                                      |
	//+==================================================================================+
#pragma region Read_3ds_CAD
	// Read 3ds & *WITHOUT* Translate the Psc
	// The origin of object is [0,0,0]
	CAD3DX<double> dx3(file_CAD, MatDB, CADUnit, true);

	cout<<"Polygon List size = "<<dx3.size_PL()<<endl;
	cout<<"Vertex  List size = "<<dx3.size_VL()<<endl;
	cout<<"Connect List size = "<<dx3.size_CL()<<endl;
	cout<<"X : ["<<dx3.GetMinX()<<","<<dx3.GetMaxX()<<"]"<<endl;
	cout<<"Y : ["<<dx3.GetMinY()<<","<<dx3.GetMaxY()<<"]"<<endl;
	cout<<"Z : ["<<dx3.GetMinZ()<<","<<dx3.GetMaxZ()<<"]"<<endl;

	// STL -> Obj class
	vector<Obj*> obj = dx3.Convert2Obj();

	//+==================================================================================+
	//|                                    BVH build                                     |
	//+==================================================================================+
	cout<<"Building Target Model..."<<endl;
	BVH bvh(&obj, false, 1);
	cout<<endl;

	cout<<"+------------------------------------+"<<endl;
	cout<<"| SAR echo simulation key parameters |"<<endl;
	cout<<"+------------------------------------+"<<endl;
	cout<<"Transmitted frequency [GHz] = "<<Sar.f0()/1E9<<endl;
	cout<<"ADC sampling rate [MHz]     = "<<Sar.Fr()/1E6<<endl;
	cout<<"# Antenna"<<endl;
	cout<<"Azimuth beam width [deg]    = "<<rad2deg(Sar.theta_az())<<endl;
	cout<<"Elevation beam width [deg]  = "<<rad2deg(Sar.theta_rg())<<endl;
	cout<<"Main beam look angle [deg]  = "<<rad2deg(Sar.theta_l_MB())<<endl;
	cout<<"Central squint angle [deg]  = "<<rad2deg(Sar.theta_sqc())<<endl;
	cout<<"# Data Dimension"<<endl;
	cout<<"Number of Range [sample]    = "<<Sar.Nr()<<endl;
	cout<<"Number of Azimuth [sample]  = "<<Sar.Na()<<endl;
	cout<<"# Electric Field"<<endl;
	cout<<"TX Polarization             = "<<Ef.TxPol()<<endl;
	cout<<"RX Polarization             = "<<Ef.RxPol()<<endl;
	cout<<"# PO/PTD approximation"<<endl;
	if(IsPO && IsPTD){
		if(IsGPU){
			cout<<"EM method                   = PO + PTD (GPU)"<<endl;
		}else{
			cout<<"EM method                   = PO + PTD (CPU)"<<endl;
		}
	}else if(IsPO){
		if(IsGPU) {
			cout<<"EM method                   = PO (GPU)"<<endl;
		}else{
			cout<<"EM method                   = PO (CPU)"<<endl;
		}
	}else if(IsPTD){
		if(IsGPU) {
			cout<<"EM method                   = PTD (GPU)"<<endl;
		}else{
			cout<<"EM method                   = PTD (CPU)"<<endl;
		}
	}else{
		cout<<"EM method                   = No this kind of EM method"<<endl;
	}
	cout<<"Max Bouncing Number         = "<<Ef.MaxLevel()<<endl;
	cout<<"# Incident Mesh"<<endl;
	cout<<"Mesh Scale Factor           = "<<mesh.Scale()<<endl;
	cout<<"# Data acquisition"<<endl;
	cout<<"TargetAsp [deg]             = "<<rad2deg(TargetAsp)<<endl;
	if(SARMode == 1){
		cout<<"SARMode                     = (1)Stripmap"<<endl;
	}else if(SARMode == 2){
		cout<<"SARMode                     = (2)Spotlight"<<endl;
	}else if(SARMode == 3){
		cout<<"SARMode                     = (3)Circular SAR or Inverse SAR"<<endl;
	}else{
		cout<<"SARMode                     = ("<<SARMode<<")Undefined"<<endl;
	}
	cout<<endl;
#pragma endregion

	//+==================================================================================+
	//|                    Preprocessing for Core Calculation                            |
	//+==================================================================================+
#pragma region Preprocessing_for_Core_Calculation
	//+==================================================================================+
	//|                            Range frequency series                                |
	//+==================================================================================+
	// Make range freq. series
	D1<double> freq(Sar.Nr());
	if(Sar.Nr() == 1){
		freq[0] = Sar.f0();
	}else{
		linspace(Sar.f0()-Sar.Fr()/2, Sar.f0()+Sar.Fr()/2, freq);
		fftshift(freq);
	}

	//+==================================================================================+
	//|                             Allocation memory                                    |
	//+==================================================================================+
	// Allocation for Total back-scattering EF, write the simulation raw 2D array into the directory
	D3<CPLX<float> > Echo_H(Ef.MaxLevel() + 1, ang.GetNum(), freq.GetNum());	// Echo_PO_H[#level][#ang][#freq]
	D3<CPLX<float> > Echo_V(Ef.MaxLevel() + 1, ang.GetNum(), freq.GetNum());	// Echo_PO_V[#level][#ang][#freq]
	// Allocation for PO
	D3<CPLX<float> > Echo_PO_H(Ef.MaxLevel() + 1, ang.GetNum(), freq.GetNum());	// Echo_PO_H[#level][#ang][#freq]
	D3<CPLX<float> > Echo_PO_V(Ef.MaxLevel() + 1, ang.GetNum(), freq.GetNum());	// Echo_PO_V[#level][#ang][#freq]
	// Allocation for PTD
	D3<CPLX<float> > Echo_PTD_H(2, ang.GetNum(), freq.GetNum());	// Echo_PTD_H[#ang][#freq]
	D3<CPLX<float> > Echo_PTD_V(2, ang.GetNum(), freq.GetNum());	// Echo_PTD_V[#ang][#freq]
	// Reset data
	Echo_PO_H.SetZero();
	Echo_PO_V.SetZero();
	Echo_PTD_H.SetZero();
	Echo_PTD_V.SetZero();
	Echo_H.SetZero();
	Echo_V.SetZero();
#pragma endregion

#pragma region preprocessing_for_PTD
	// Copy BVH(Host) to cuBVH(Device)
	cu::cuBVH *d_bvh = nullptr;
	cu::cuTRI<float> *d_primes;
	size_t *d_idx_poly;
	cu::cuBVHFlatNode *d_flatTree;

	double* d_k0 = nullptr;

	cuCPLX<double>* d_Echo_freq_H;				// Total diffraction electric field with freq-1D-array
	cuCPLX<double>* d_Echo_freq_V;				// Total diffraction electric field with freq-1D-array
	cuCPLX<double>* d_Echo_triangle_freq_H;		// Total diffraction electric field with freq-triangle-1D-array
	cuCPLX<double>* d_Echo_triangle_freq_V;		// Total diffraction electric field with freq-triangle-1D-array

	cuCPLX<double>* h_Echo_freq_H;	// Store the host mem a.s.a. d_Echo_freq_H
	cuCPLX<double>* h_Echo_freq_V;	// Store the host mem a.s.a. d_Echo_freq_V

	if(IsGPU){
		// define for GPU
		cu::GPUSelect(IsGPUShow);
		// cuBVH malloc & memcpy
		d_bvh->Create(bvh, d_bvh, d_primes, d_idx_poly, d_flatTree);

		if(IsPTD && IsGPU) {
			// Copy k0
			double* h_k0 = new double[freq.GetNum()];
			for(size_t iFreq=0;iFreq<freq.GetNum();++iFreq){
				h_k0[iFreq] = 2*def::PI / (def::C / freq[iFreq]);
			}
			SafeMallocCheck(cudaMalloc(&d_k0, freq.GetNum() * sizeof(double)));
			SafeMallocCheck(cudaMemcpy(d_k0, h_k0, freq.GetNum() * sizeof(double), cudaMemcpyHostToDevice));
			delete [] h_k0;
			//+--------------------+
			//|  Memory Allocation |
			//+--------------------+
			size_t nFreq = freq.GetNum();
			size_t nTriangle = bvh.GetIdxPoly().size();
			// results in device
			SafeMallocCheck(cudaMalloc(&d_Echo_freq_H,   	      nFreq * sizeof(cuCPLX<double>)));
			SafeMallocCheck(cudaMalloc(&d_Echo_freq_V,   	  	  nFreq * sizeof(cuCPLX<double>)));
			SafeMallocCheck(cudaMalloc(&d_Echo_triangle_freq_H,   nFreq * nTriangle * sizeof(cuCPLX<double>)));
			SafeMallocCheck(cudaMalloc(&d_Echo_triangle_freq_V,   nFreq * nTriangle * sizeof(cuCPLX<double>)));
//			SafeMallocCheck(cudaMemset(d_Echo_freq_H, 	   	   0, nFreq * sizeof(cuCPLX<double>)));
//			SafeMallocCheck(cudaMemset(d_Echo_freq_V, 	   	   0, nFreq * sizeof(cuCPLX<double>)));
//			SafeMallocCheck(cudaMemset(d_Echo_triangle_freq_H, 0, nFreq * nTriangle * sizeof(cuCPLX<double>)));
//			SafeMallocCheck(cudaMemset(d_Echo_triangle_freq_V, 0, nFreq * nTriangle * sizeof(cuCPLX<double>)));

			h_Echo_freq_H = new cuCPLX<double>[nFreq];	// Store the host mem a.s.a. d_Echo_freq_H
			h_Echo_freq_V = new cuCPLX<double>[nFreq];	// Store the host mem a.s.a. d_Echo_freq_V
		}
	}
#pragma endregion

	// Start timer (overall)
	clock_t tic = def_func::tic();
	clock_t ticTotal = def_func::tic();
	//+==================================================================================+
	//|               For each angle (Look & Aspect) & frequency (START)                 |
	//+==================================================================================+
	for(size_t i=0;i<ang.GetNum();++i){
//	for(size_t i=0;i<3;++i){
//	for(size_t i=ang.GetNum()/2-25;i<ang.GetNum()/2+25;++i){

#pragma region Preprocessing_for_multiple_angles
		//+-----------------------------------------------------------+
		//|               Make incident spherical surface             |
		//|       (the wavelength using center frequency only, f0)    |
		//+-----------------------------------------------------------+
		// inst. sensor position
		VEC<double> Ps  = sv.pos()[i];
		VEC<double> Ps1 = sv.pos()[i+1];
		VEC<double> PPt(0,0,0);			// Target's Center position
		double TotalAsp = ang.Asp()[i] + TargetAsp;

//		double Rad = 80E4;
		double Rad  = 0;
		double Rad1 = 0;
		if(SARMode == 1 || SARMode == 2){
			Rad  = (Ps - Pt).abs();						// for Stripmap & Spotlight
			Rad1 = (Ps1 - Pt).abs();
		}else if(SARMode == 3){
			Rad = ( sv.pos()[0] - Pt ).abs();			// for ISAR
			Rad1 = Rad;
		}else{
			cerr<<"No this kind of SARMode = "<<SARMode<<" (1:Stripmap, 2:Spotlight, 3:ISAR)"<<endl;
			exit(EXIT_FAILURE);
		}

		VEC<double> PPs(Rad*sin(ang.Look()[i])*cos(TotalAsp),
						Rad*sin(ang.Look()[i])*sin(TotalAsp),
						Rad*cos(ang.Look()[i]));	// Sensor position
		VEC<double> PPs1;
		// Last one issue
		if(i != ang.GetNum()-1){
			// Next Ps
			TotalAsp = ang.Asp()[i+1] + TargetAsp + 1e-16;
			PPs1 = VEC<double>(Rad1*sin(ang.Look()[i+1])*cos(TotalAsp),
							   Rad1*sin(ang.Look()[i+1])*sin(TotalAsp),
							   Rad1*cos(ang.Look()[i+1]));
		}else{
			// Previous Ps
			TotalAsp = ang.Asp()[i-1] + TargetAsp + 1e-16;
			VEC<double> PPs_1(Rad1*sin(ang.Look()[i-1])*cos(TotalAsp),
							  Rad1*sin(ang.Look()[i-1])*sin(TotalAsp),
							  Rad1*cos(ang.Look()[i-1]));
			VEC<double> PPs_1toPPs = PPs - PPs_1;
			// The pesudo next Ps by previous position
			PPs1 = PPs + PPs_1toPPs;
		}

		// Sensor inst. position
		double Rad2 = 80E4;
		VEC<double> PPs2(Rad2*sin(ang.Look()[i])*cos(TotalAsp),
						 Rad2*sin(ang.Look()[i])*sin(TotalAsp),
						 Rad2*cos(ang.Look()[i]));	// Sensor position


		D1<double> MinRadVec(6);
		MinRadVec[0] = abs(dx3.GetMaxX());
		MinRadVec[1] = abs(dx3.GetMaxY());
		MinRadVec[2] = abs(dx3.GetMaxZ());
		MinRadVec[3] = abs(dx3.GetMinX());
		MinRadVec[4] = abs(dx3.GetMinY());
		MinRadVec[5] = abs(dx3.GetMinZ());
//		mesh.dRad() = 2.0*mat::max(MinRadVec);
		mesh.dRad() = 50.0*mat::max(MinRadVec);
//		mesh.dRad() = 20.0*mat::max(MinRadVec);

//		mesh.dRad() = 20.0;
//		mesh.dRad() = 22.0;

//		cout<<"PPs2 = "; PPs2.Print();
//		cout<<"PPt  = "; PPt.Print();
//		cout<<"|PPs2 -> PPt| = "<<(PPs2 - PPt).abs()<<endl;
//		cout<<"mesh.dRad() = "<<mesh.dRad()<<endl;

		// Create incident mesh
		MeshInc inc_mesh(mesh.Scale(), Sar.lambda(), bvh, PPs2, PPt, mesh.dRad());
		if(i == 0){
			inc_mesh.PrintSimple();
			cout<<endl;
		}

		//+----------------------------------------+
		//|    Prepare for Azimuth Antenna gain    |
		//+----------------------------------------+
		// Find main beam uv
		VEC<double> MainBeamUV;
		if(SARMode == 1){
			MainBeamUV = sar::find::MainBeamUVByLookSquintTypeASphericalCoordinate(PPs, PPs1, Sar.theta_l_MB(), Sar.theta_sqc());	// for Stripmap
		}else if(SARMode == 2 || SARMode == 3){
			MainBeamUV = sar::find::MainBeamUVByLookSquintTypeASphericalCoordinate(PPs, PPs1, Sar.theta_l_MB(), ang.Squint()[i]);	// for Spotlight & ISAR
		}else{
			cerr<<"No this kind of SARMode = "<<SARMode<<" (1:Stripmap, 2:Spotlight, 3:ISAR)"<<endl;
			exit(EXIT_FAILURE);
		}


		// Find normal vector on squint plane
		VEC<double> NorSquintPlane = vec::Unit(cross(MainBeamUV, PPs1-PPs));


		//+----------------------------------------+
		//|    define SAR parameters for CPU/GPU   |
		//+----------------------------------------+
		vector<SAR> Sar2(freq.GetNum());
		D1<double> k0(freq.GetNum());
		for(long k=0;k<Sar2.size();++k){
			// Sar2
			Sar2[k].f0()     = freq[k];
			Sar2[k].lambda() = def::C/Sar2[k].f0();
			Sar2[k].k0()     = 2*def::PI/Sar2[k].lambda();
			Sar2[k].SetTheta_l_MB(Sar.theta_l_MB());
			Sar2[k].theta_az() = Sar.theta_az();
			Sar2[k].theta_rg() = Sar.theta_rg();
			// k0
			k0[k] = Sar2[k].k0();
		}
#pragma endregion


		//+-----------------------------------------------------------+
		//|                                                           |
		//|         Echo - Range profile simulation (START)           |
		//|                                                           |
		//+-----------------------------------------------------------+
		if(IsPO){
			if(IsGPU){
				//+-----------------+
				//|   GPU program   |
				//+-----------------+
				// -------- CUDA host memory allocation ---------
				// for results in HOST
				cu::cuSBRElement<float>* res;	// res[#freq].{sumt(for V), sump(for H)}
				// define for GPU
				res = new cu::cuSBRElement<float>[Sar2.size() * Ef.MaxLevel()];
//				cu::GPUSelect(IsGPUShow);

				// 3rd : 一定要是(2^n, 1, 1)，且shared mem size一定要小於規格 e.g. dim(128,1,1) -> 16384 Bytes (GTX 295, 16384 Bytes) 是不可以的
				bool Is1st = (i == 0)? true:false;
//				cu::cuSBRDoIt(Sar2, k0, Ef, bvh, inc_mesh, MatDB, mesh.dRad(), NThread1, NThread2, NThread3, res, IsPEC, IsGPUShow, Is1st);
//				cu::cuSBRDoIt2(Sar2, k0, Ef, bvh, inc_mesh, MatDB, mesh.dRad(), NThread1, NThread2, NThread3, res, IsPEC, MainBeamUV, NorSquintPlane, PPs, PPt, IsGPUShow, Is1st);
				cu::cuSBRDoIt3(Sar2, k0, Ef, d_bvh, inc_mesh, MatDB, mesh.dRad(), NThread1, NThread2, NThread3, res, IsPEC, MainBeamUV, NorSquintPlane, PPs, PPt, IsGPUShow, Is1st);

				// Results storage
				// D3<CPLX<float> > Echo_PO_H(ang.GetNum(), freq.GetNum(), Ef.MaxLevel());	// Echo_PO_H[#ang][#freq][#level]
				// D3<CPLX<float> > Echo_PO_V(ang.GetNum(), freq.GetNum(), Ef.MaxLevel());	// Echo_PO_V[#ang][#freq][#level]
				for(size_t j=0;j<freq.GetNum();++j){
					//
					// Dual export
					//
					for(size_t k=0;k<Ef.MaxLevel();++k){
						size_t idx = j*Ef.MaxLevel() + k;
						cu::cuCPLX<float> cu_cxH = res[idx].sump;
						cu::cuCPLX<float> cu_cxV = res[idx].sumt;
						CPLX<float> cxH(cu_cxH.r, cu_cxH.i);
						CPLX<float> cxV(cu_cxV.r, cu_cxV.i);
						// Level-0 (Total)
						Echo_PO_H[0][i][j] += cxH;		// Phi   (H)
						Echo_PO_V[0][i][j] += cxV;		// Theta (V)
						// Each Level
						Echo_PO_H[k + 1][i][j] = cxH;	// Phi   (H)
						Echo_PO_V[k + 1][i][j] = cxV;	// Theta (V)
						if(j == 0 && i == 0){
							printf("cxH = (% 8.6f,% 8.6f), cxV = (% 8.6f,% 8.6f)\n", cxH.r(), cxH.i(), cxV.r(), cxV.i());
						}
					}
					if(j == 0 && i == 0){
						printf("Total    = (% 8.6f,% 8.6f) -> % 8.6f\n", Echo_PO_H[0][i][j].r(), Echo_PO_H[0][i][j].i(), Echo_PO_H[0][i][j].abs());
						for(size_t p=0;p<Ef.MaxLevel();++p){
							printf("Level %2ld = (% 8.6f,% 8.6f) -> % 8.6f\n", p+1, Echo_PO_H[p + 1][i][j].r(), Echo_PO_H[p + 1][i][j].i(), Echo_PO_H[p + 1][i][j].abs());
						}
					}
				}
				// ------------------ Free res ------------------
				delete [] res;
				// ------------------ Free res ------------------
			}else{
				//+-----------------+
				//|   CPU program   |
				//+-----------------+
				SBR<double> sbr;
				sbr = SBR<double>(Sar2, Ef, bvh, inc_mesh, MatDB, Ef.MaxLevel(), -mesh.dRad());
	//			sbr.DoIt(IsPEC, false);
				sbr.DoIt2(MainBeamUV, NorSquintPlane, PPs, PPt, IsPEC, false);


				// Results storage
				// D3<CPLX<float> > Echo_PO_H(ang.GetNum(), freq.GetNum(), Ef.MaxLevel());	// Echo_PO_H[#ang][#freq][#level]
				// D3<CPLX<float> > Echo_PO_V(ang.GetNum(), freq.GetNum(), Ef.MaxLevel());	// Echo_PO_V[#ang][#freq][#level]
				for(long j=0;j<freq.GetNum();++j){
					//
					// Dual export
					//
					for(size_t k=0;k<Ef.MaxLevel();++k){
						CPLX<float> cxH = def_func::CPLXdouble2CPLXfloat(sbr.GetSBRElement(j).GetSump()[k]);
						CPLX<float> cxV = def_func::CPLXdouble2CPLXfloat(sbr.GetSBRElement(j).GetSumt()[k]);
						// Level-0 (Total)
						Echo_PO_H[0][i][j] += cxH;		// Phi   (H)
						Echo_PO_V[0][i][j] += cxV;		// Theta (V)
						// Each Level
						Echo_PO_H[k + 1][i][j] = cxH;	// Phi   (H)
						Echo_PO_V[k + 1][i][j] = cxV;	// Theta (V)
						if(j == 0 && i == 0){
							printf("cxH = (% 8.6f,% 8.6f), cxV = (% 8.6f,% 8.6f)\n", cxH.r(), cxH.i(), cxV.r(), cxV.i());
						}
					}
					if(j == 0 && i == 0){
						printf("Total    = (% 8.6f,% 8.6f) -> % 8.6f\n", Echo_PO_H[0][i][j].r(), Echo_PO_H[0][i][j].i(), Echo_PO_H[0][i][j].abs());
						for(size_t p=0;p<Ef.MaxLevel();++p){
							printf("Level %2ld = (% 8.6f,% 8.6f) -> % 8.6f\n", p+1, Echo_PO_H[p + 1][i][j].r(), Echo_PO_H[p + 1][i][j].i(), Echo_PO_H[p + 1][i][j].abs());
						}
					}
				} // For effective frequency sample (j)
			}
		} // End if(isPO)
		//+-----------------------------------------------------------+
		//|                                                           |
		//|         Echo - Range profile simulation (END)             |
		//|                                                           |
		//+-----------------------------------------------------------+

		//+-----------------------------------------------------------+
		//|                                                           |
		//|         Diffraction - Range profile simulation (START)    |
		//|                                                           |
		//+-----------------------------------------------------------+
		if(IsPTD){
			if(IsGPU){
				//+-----------------+
				//|   GPU program   |
				//+-----------------+
				// Incident Electric field
				EMWave<double> Ei(Unit(-inc_mesh.Ps), inc_mesh.Ps, Ef.TxPol());

				// Re-assignment
				Ray rayInc(Ei.o(), Ei.k());

				// Get global coordinate of antenna by H E-field vibration direction
				LocalCorrdinate<double> elg(inc_mesh.dirH, Ei.k(), -Ei.k());


				//TODO:: 需打包成一個cu function (START) ------------------------------------------------------------
				bool IsShow = false;
				if(i == 0){ IsShow = true; }

				// Get CUDA memory info
				if(IsShow) { CheckMem("Initial:"); }


				clock_t gpu_dt = def_func::tic();
				clock_t gpu_global_mem_alloc = def_func::tic();
				//+-----------------+
				//|    Dimension    |
				//+-----------------+
				size_t nFreq     = Sar2.size();			// #freq
				size_t nTriangle = bvh.idx_poly.size();	// #triangle

				//+-----------------+
				//|      Copy       |
				//+-----------------+
				// Copy MeshInc(Host) to cuMeshInc(Device)
				cu::cuMeshInc* d_inc_mesh = nullptr;
				cu::cuVEC<double>* d_dirH_disH;
				cu::cuVEC<double>* d_dirV_disV;
				d_inc_mesh->Create(inc_mesh, d_inc_mesh, d_dirH_disH, d_dirV_disV);


				// Copy EF(Host) to cuEF(Device)
				cu::cuEF<float>* d_Ef = nullptr;
				d_Ef->Create(Ef, d_Ef);


				// Copy Ray
				cu::cuRay* d_rayInc = nullptr;
				d_rayInc->Create(rayInc, d_rayInc);

				// Get global coordinate of antenna by H E-field vibration direction
				cu::cuLocalCorrdinate<double>* d_elg = nullptr;
				cu::cuLocalCorrdinate<double> h_elg;
				h_elg.fromLocalCoodinate(elg);
				d_elg->Create(h_elg, d_elg);

				// Copy ElectricField, Ei -> d_Ei
				// Ei.o()
				cu::cuVEC<double>* d_Ei_o = nullptr;
				d_Ei_o->Create(Ei.o(), d_Ei_o);

				// Ei.k()
				cu::cuVEC<double>* d_Ei_k = nullptr;
				d_Ei_k->Create(Ei.k(), d_Ei_k);

				// Ei.cplx()
				cuVEC<cuCPLX<double> >* d_Ei_cplx = nullptr;
				{
					cuVEC<cuCPLX<double> > h_Ei_cplx(cuCPLX<double>(Ei.cplx().x().r(), Ei.cplx().x().i()),
													 cuCPLX<double>(Ei.cplx().y().r(), Ei.cplx().y().i()),
													 cuCPLX<double>(Ei.cplx().z().r(), Ei.cplx().z().i()));
					cudaMalloc(&d_Ei_cplx, sizeof(cuVEC<cuCPLX<double> >));
					cudaMemcpy(d_Ei_cplx, &h_Ei_cplx, sizeof(cuVEC<cuCPLX<double> >), cudaMemcpyHostToDevice);
				}


				//+--------------------+
				//|  Memory reset      |
				//+--------------------+
				SafeMallocCheck(cudaMemset(d_Echo_freq_H, 	   	   0, nFreq * sizeof(cuCPLX<double>)));
				SafeMallocCheck(cudaMemset(d_Echo_freq_V, 	   	   0, nFreq * sizeof(cuCPLX<double>)));
				SafeMallocCheck(cudaMemset(d_Echo_triangle_freq_H, 0, nFreq * nTriangle * sizeof(cuCPLX<double>)));
				SafeMallocCheck(cudaMemset(d_Echo_triangle_freq_V, 0, nFreq * nTriangle * sizeof(cuCPLX<double>)));

				// Get CUDA memory info
				if(IsShow){
					toc(gpu_global_mem_alloc, "Global Mem transfer: ");
					CheckMem("After Malloc:");
				}

				//+--------------------------------------------------------+
				//|                    KERNEL (START)                      |
				//+--------------------------------------------------------+

				// 1. Call PTD calculation (for all triangle in CAD)
				dim3 NThreadPTD = dim3(64,1,1);
				dim3 NBlockPTD(ceil(float(nFreq * nTriangle)/float(NThreadPTD.x)), 1, 1);	// number of block in single patch grid
				cuPTD<<<NBlockPTD, NThreadPTD>>>(*d_inc_mesh, *d_Ei_o, *d_Ei_k, *d_Ei_cplx,
												 d_k0, *d_rayInc, *d_bvh, nTriangle, nFreq, *d_elg, mesh.dRad(),
												 d_Echo_triangle_freq_H, d_Echo_triangle_freq_V);

				// 2. Summation all triangle to one value (for each frequency)
				if(IsShow) {
					cout << "nTriangle = " << nTriangle << endl;
					cout << "size smem = " << 512 << endl;
				}
				for(size_t j=0;j<nFreq;++j){
					cuReductionPTD(d_Echo_triangle_freq_H + j * nTriangle, nTriangle, d_Echo_freq_H, j, (size_t)512); // H
					cuReductionPTD(d_Echo_triangle_freq_V + j * nTriangle, nTriangle, d_Echo_freq_V, j, (size_t)512); // V
				}

				// 3. Copy the d_Echo_freq_H(device, cuCPLX<double>*) into Echo_PTD_H(host, CPLX<float>*)
				//             d_Echo_freq_V(device, cuCPLX<double>*) into Echo_PTD_V(host, CPLX<float>*)
				// 3.1 Copy from device to host
				showCudaError(cudaMemcpy(h_Echo_freq_H, d_Echo_freq_H, nFreq * sizeof(cuCPLX<double>), cudaMemcpyDeviceToHost));
				showCudaError(cudaMemcpy(h_Echo_freq_V, d_Echo_freq_V, nFreq * sizeof(cuCPLX<double>), cudaMemcpyDeviceToHost));
				// 3.2 Write to destination memory for each frequency
				for(size_t j=0;j<nFreq;++j){
					// Level-1
					Echo_PTD_H[1][i][j] = CPLX<float>(float(h_Echo_freq_H[j].r), float(h_Echo_freq_H[j].i));
					Echo_PTD_V[1][i][j] = CPLX<float>(float(h_Echo_freq_V[j].r), float(h_Echo_freq_V[j].i));
					// Total
					Echo_PTD_H[0][i][j] = Echo_PTD_H[1][i][j];
					Echo_PTD_V[0][i][j] = Echo_PTD_V[1][i][j];
				}

				//+--------------------------------------------------------+
				//|                    KERNEL (END)                        |
				//+--------------------------------------------------------+
				cudaDeviceSynchronize();


				//+------------------------------------------------------------------------------------------+
				//|                                  CUDA Free                                               |
				//+------------------------------------------------------------------------------------------+
				d_inc_mesh->Free(d_inc_mesh, d_dirH_disH, d_dirV_disV);
				d_Ef->Free(d_Ef);
				d_rayInc->Free(d_rayInc);
				d_elg->Free(d_elg);
				d_Ei_o->Free(d_Ei_o);
				d_Ei_k->Free(d_Ei_k);

				cudaFree(d_Ei_cplx);

				//TODO:: 需打包成一個cu function (END) ------------------------------------------------------------
			}else{
				//+-----------------+
				//|   CPU program   |
				//+-----------------+
				// Incident Electric field
				EMWave<double> Ei(Unit(-inc_mesh.Ps), inc_mesh.Ps, Ef.TxPol());

				// Re-assignment
				Ray rayInc(Ei.o(), Ei.k());

				// Memory allocate
				vector<VEC<CPLX<double> > > Ed(freq.GetNum());		// Total diffraction electric field
				IntersectionInfo I_Shadow;							// Intersection object for shadow detection

				// Reset to Zero
				for(size_t j=0;j<freq.GetNum();++j){
					Ed[j].SetZero();
				}

				// Get global coordinate of antenna by H E-field vibration direction
				LocalCorrdinate<double> elg(inc_mesh.dirH, Ei.k(), -Ei.k());

				// For all polygon
				for(size_t k=0;k<bvh.GetflatTree()->nPrims;++k){
					//+-------------------+
					//|   For each edge   |
					//+-------------------+
					// Get triangle
					Obj* OBj = (*(bvh.GetBuildPrims()))[k];
					// Force convert
					TRI<double> tri = *((TRI<float>*)(OBj));

					// Get Normal vector
					VEC<double> N0 = tri.getNormal();

					// Edge
					EdgeList<double> EL(tri);

					// Near edge
					long idx_poly_near[3];
					idx_poly_near[0] = tri.IDX_Near(0);
					idx_poly_near[1] = tri.IDX_Near(1);
					idx_poly_near[2] = tri.IDX_Near(2);

					// For loop ( FOR EACH EDGE of EACH TRIANGLE )
					for(size_t p=0; p<3; ++p) {
						// 2.0. Only shared edge need to be calculated
						//      idx_poly_near = -1 means that there is no shared edges.
						if(idx_poly_near[p] < 0){ continue; }
						// 2.1. Find wedge angle
						double WedgeAngle = tri.EA(p);
						double n = ptd::WedgeFactor(WedgeAngle);
						// 2.2 Defined incident & observation unit vector
						// Incident
						VEC<double>& uv_sp = rayInc.d;
						// 2.3 Find Edge vector
						//    右手定則下的 vertex，旋轉方向(大拇指指向)為 normal vector of polygon, 跟diffraction定義的edge方向
						//    由起始點(Start)到終點(End)
						EdgeCoordinate<double> e;
						e.z() = Unit(EL.Edge(p).E() - EL.Edge(p).S() );	// 因為cad reader是右手定則 (Start -> End)
						e.y() = tri.getNormal();
						e.x() = cross(e.y(), e.z());
						//+--------------------------------------------+
						//|       3. For each pieces of segment        |
						//+--------------------------------------------+
						// 3.1 Find segment two end points and centroid point location
						double dL = double(EL.Length(p));
						VEC<double> Q = 0.5 * (EL.Edge(p).S() + EL.Edge(p).E());
						double s  = vec::Norm2(Ei.o() - Q);
						//+-----------------------------------------------------+
						//|   Check it is effective edge with some conditions   |
						//+-----------------------------------------------------+
						// 3.2 The N factor of wedge angle MUST smaller than 1
						if(n <= 1){ continue; }
						// 3.3 The incident angle is between within the two facets boundary
						TRI<double> tri2;
						EdgeCoordinate<double> e2;
						bool isOK = CheckDiffractionIncidentBoundary(bvh, uv_sp, tri, p, tri2, e2, k);
						if(!isOK){ continue; }
						// 3.4 Check the return path is shadow or not?
						//     Check Shadow : If the back ray tracing is shadow, there is no PO result.
						// shadow ray
						VEC<double> uv_shadow = Unit(VEC<double>(Ei.o().x()-Q.x(), Ei.o().y()-Q.y(), Ei.o().z()-Q.z()));
						Ray rayShadow(Q, uv_shadow);
						bool isShadow = bvh.getIntersection(rayShadow, &I_Shadow, false, 9999);
						isShadow = ( isShadow && !(((TRI<double>*)(I_Shadow.object))->Equal(tri)) );
						if(isShadow){ continue; }
						//+--------------------------------+
						//|    Diffraction calculation     |
						//+--------------------------------+
						// 3.5 Observation
						VEC<double> uv_s = VEC<double>(0,0,0) - uv_sp;
						// 3.6 Calculation all angles
						double beta  = acos(dot(uv_s, e.z()));	// to Observation (e.z & s are uv)
						double betap = acos(dot(uv_sp, e.z()));	// from Source (e.z & sp are uv)

						VEC<double> uv_s_proj  = vec::Unit(vec::ProjectOnPlane(uv_s,  e.z()));	// s  projected on local XY-plane
						VEC<double> uv_sp_proj = vec::Unit(vec::ProjectOnPlane(uv_sp, e.z()));	// sp projected on local XY-plane

						double phi  = vec::SignedAngleTwo3DVectors(e.x(),   uv_s_proj, e.z());	// s_proj: observation
						double phip = vec::SignedAngleTwo3DVectors(e.x(), -uv_sp_proj, e.z());	// sp_proj: incident
						if(phi  < 0) {  phi = deg2rad(360) + phi; }
						if(phip < 0) { phip = deg2rad(360) + phip; }

						// 3.7 Avoid Avoid 90 & 180 [deg]
						double eps = 1e-4;
						if(abs(beta - deg2rad(90)) < eps) {
							beta = beta - eps;
	//						betap = deg2rad(180) - beta;    // Backscattering
						}
						if(abs(betap - deg2rad(90)) < eps) {
							betap = betap - eps;
	//						beta = deg2rad(180) - betap;    // Backscattering
						}
						if(abs(phi - deg2rad(90)) < eps || abs(phi - deg2rad(180)) < eps) {
							phi = phi - eps;
						}
						if(abs(phip - deg2rad(90)) < eps || abs(phi - deg2rad(180)) < eps) {
							phip = phip - eps;
						}

						// 3.8 Diffraction fringe calculation: using DiffMichaeliEC2
						CPLX<double> I1, M1, I2, M2, I, M, factor;
						VEC<double> I_vec, M_vec;
						VEC<CPLX<double> > I_comp, M_comp, Ed_local;

						// For each frequency
						for(long j=0; j < freq.GetNum(); ++j){
							// Face-1
							FringeIM(Ei, e.z(), n, k0[j], uv_sp, beta, betap, phi, phip, I1, M1);
							// Face-2
//							FringeIM(Ei, -e.z(), n, k0[j], uv_sp, def::PI-beta, def::PI-betap, n*def::PI-phi, n*def::PI-phip, I2, M2);
							FringeIM(Ei, -e.z(), n, k0[j], uv_sp, def::PI - beta, def::PI - betap, phi, phip, I2, M2);

							I = I1 - I2;
							M = M1 + M2;

							factor = -dL * cplx::exp(CPLX<double>(0, -k0[j] * (s+mesh.dRad()) * 2.0)) / def::PI2;

							I_vec = vec::Unit(cross(uv_sp, cross(uv_sp, e.z())));
							M_vec = vec::Unit(cross(uv_sp, e.z()));

							I_comp = VEC<CPLX<double> >(factor * I * I_vec.x(), factor * I * I_vec.y(), factor * I * I_vec.z());
							M_comp = VEC<CPLX<double> >(factor * M * M_vec.x(), factor * M * M_vec.y(), factor * M * M_vec.z());

							Ed_local = (I_comp + M_comp);

							// 3.9 Avoid Nan & Inf
							if(isnan(Ed_local.x().r()) || isinf(Ed_local.x().r())){ Ed_local.x().r() = 1e-16; }
							if(isnan(Ed_local.x().i()) || isinf(Ed_local.x().i())){ Ed_local.x().i() = 1e-16; }
							if(isnan(Ed_local.y().r()) || isinf(Ed_local.y().r())){ Ed_local.y().r() = 1e-16; }
							if(isnan(Ed_local.y().i()) || isinf(Ed_local.y().i())){ Ed_local.y().i() = 1e-16; }
							if(isnan(Ed_local.z().r()) || isinf(Ed_local.z().r())){ Ed_local.z().r() = 1e-16; }
							if(isnan(Ed_local.z().i()) || isinf(Ed_local.z().i())){ Ed_local.z().i() = 1e-16; }

							// Sum for each triangle(k) and each edge(p)
							Ed[j] += Ed_local;

//							if(j==0 && k==4264) {
//								printf("\n\n\n\n>>>> CPU >>>>\nk=%ld, p=%ld, Ed_local=[(%.10f,%.10f),(%.10f,%.10f),(%.10f,%.10f)]\n>>>>>>>>>>>>>\n\n",
//										k, p,
//										Ed_local.x().r(), Ed_local.x().i(),
//									    Ed_local.y().r(), Ed_local.y().i(),
//									    Ed_local.z().r(), Ed_local.z().i());
////								printf("\n\n\n\n>>>> CPU >>>>\nk=%ld, p=%ld, I=(%.20f,%.20f), M=(%.20f,%.20f), Ei_cplx=[(%f,%f),(%f,%f),(%f,%f)], e.z=[%f,%f,%f], k0=%.10f\n>>>>>>>>>>>>>\n\n",
////									   k, p, I.r(), I.i(), M.r(), M.i(),
////									   Ei.cplx().x().r(), Ei.cplx().x().i(), Ei.cplx().y().r(), Ei.cplx().y().i(), Ei.cplx().z().r(), Ei.cplx().z().i(),
////									   e.z().x(), e.z().y(), e.z().z(), k0[j]);
//							}


						} // End for each frequency(j), j<freq.GetNum()
					} // End for each edge(p), p<3
				} // End of each triangle(k), k<bvh.GetflatTree()->nPrims

				//+----------------------------------------------------------------------------+
				//|                        Split H & V components                              |
				//+----------------------------------------------------------------------------+
				// For each frequency
				for(long j=0; j < freq.GetNum(); ++j){// Level-1
					Echo_PTD_V[1][i][j] = def_func::CPLXdouble2CPLXfloat( vec::dot(Ed[j], elg.es().perp()) );
					Echo_PTD_H[1][i][j] = def_func::CPLXdouble2CPLXfloat( vec::dot(Ed[j], elg.es().para()) );
					// Total
					Echo_PTD_H[0][i][j] = Echo_PTD_H[1][i][j];
					Echo_PTD_V[0][i][j] = Echo_PTD_V[1][i][j];
				} // End for each frequency(j), j<freq.GetNum()

			} // End IsGPU

		} // End if(PTD)
		//+-----------------------------------------------------------+
		//|                                                           |
		//|         Diffraction - Range profile simulation (END)      |
		//|                                                           |
		//+-----------------------------------------------------------+

#pragma region Summation_and_Assign_to_array
		//+----------------------------------------+
		//|       Summation of PO + PTD            |
		//+----------------------------------------+
		// For each level
		for(size_t k=0;k<Ef.MaxLevel();++k){
			// For each frequency
			for(long j=0; j < freq.GetNum(); ++j){
				if(j == 0 || j == 1) {
					// Diffraction occurs in Level-Total & Level-1 only.
					Echo_V[k][i][j] = Echo_PO_V[k][i][j] + Echo_PTD_V[0][i][j];
					Echo_H[k][i][j] = Echo_PO_H[k][i][j] + Echo_PTD_H[0][i][j];
				}else{
					Echo_V[k][i][j] = Echo_PO_V[k][i][j];
					Echo_H[k][i][j] = Echo_PO_H[k][i][j];
				}
			} // End for each frequency(j), j<freq.GetNum()
		} // End for each Level(k), k<Ef.MaxLevel()
		//+----------------------------------------+
		//|            Progress report             |
		//+----------------------------------------+
		def::CLOCK clock(tic);
		def_func::ProgressBar(i+1, ang.GetNum(), 30, 1, clock);
		cout<<endl;
#pragma endregion


	} // end of phi_asp [1100] (i)
	//+==================================================================================+
	//|               For each angle (Look & Aspect) & frequency (END)                   |
	//+==================================================================================+
	// End timer (overall)
	def_func::toc(ticTotal);
	cout<<endl;


#pragma region postprocessing_for_PTD
	if(IsGPU){
		// Free cuBVH(Device)
		d_bvh->Free(d_bvh, d_primes, d_idx_poly, d_flatTree);

		if(IsPTD && IsGPU) {
			cudaFree(d_k0);

			cudaFree(d_Echo_triangle_freq_H);
			cudaFree(d_Echo_triangle_freq_V);
			cudaFree(d_Echo_freq_H);
			cudaFree(d_Echo_freq_V);

			delete[] h_Echo_freq_H;
			delete[] h_Echo_freq_V;
		}
	}
#pragma endregion



	//+==================================================================================+
	//|                           Summary & Export Data                                  |
	//+==================================================================================+
#pragma region Write_results
	printf("Echo_H[0][0][0] = (%.4f, %.4f)\n", Echo_H[0][0][0].r(), Echo_H[0][0][0].i());
	printf("Echo_H[0][1][1] = (%.4f, %.4f)\n", Echo_H[0][1][1].r(), Echo_H[0][1][1].i());
	printf("Echo_H[0][2][2] = (%.4f, %.4f)\n", Echo_H[0][2][2].r(), Echo_H[0][2][2].i());
	printf("Echo_H[0][3][3] = (%.4f, %.4f)\n", Echo_H[0][3][3].r(), Echo_H[0][3][3].i());

	cout<<"+------------------------------------+"<<endl;
	cout<<"|         Dimension Summary          |"<<endl;
	cout<<"+------------------------------------+"<<endl;
	cout<<"Number of Level          = "<<Echo_H.GetP()<<endl;
	cout<<"Number of range sample   = "<<Echo_H.GetQ()<<endl;
	cout<<"Number of azimuth sample = "<<Echo_H.GetR()<<endl;
	cout<<"Number of ang            = "<<ang.GetNum()<<endl;



	//
	// Export Binary
	//
	if(IsPO){ 			io::write::PASSEDEcho(dir_out, name, Ef, Sar.theta_l_MB(), Ef.MaxLevel(), Echo_PO_H,  Echo_PO_V,  "PO"); }
	if(IsPTD){ 			io::write::PASSEDEcho(dir_out, name, Ef, Sar.theta_l_MB(), 1,             Echo_PTD_H, Echo_PTD_V, "PTD"); }
	if(IsPO && IsPTD){ 	io::write::PASSEDEcho(dir_out, name, Ef, Sar.theta_l_MB(), Ef.MaxLevel(), Echo_H,     Echo_V,     ""); }


	//
	// Export Meta
	//
	string file_out_RCS_meta = dir_out + name + "_rcs.meta";
	io::write::Meta(Sar, mesh, Ef, ang, file_Cnf, name, file_out_RCS_meta);
#pragma endregion
	//+-----------------+
	//| Check Max Value |
	//+-----------------+
#pragma region Check_Max_Value
	cout<<"+------------------------------------+"<<endl;
	cout<<"|      SBR + {PO, PTD} Summary       |"<<endl;
	cout<<"+------------------------------------+"<<endl;
	if(IsPO && IsPTD){
		cout<<"Methods for EM   = PO + PTD"<<endl;
	}else if(IsPO){
		cout<<"Methods for EM   = PO"<<endl;
	}else if(IsPTD){
		cout<<"Methods for EM   = PTD"<<endl;
	}else{
		cout<<"Methods for EM   = No method selected {PO, PTD}"<<endl;
	}
	//
	// RCS results
	//
	if(Ef.RxPol().length() == 2){	// Dual
		cout<<"+"<<endl;
		cout<<"  Dual Pol Export: (H" + Ef.TxPol() + ", V" + Ef.TxPol() + ")"<<endl;
		cout<<"+"<<endl;
		double Srcs_H = mat::maxAbs(Echo_H);
		double Srcs_V = mat::maxAbs(Echo_V);
		cout<<std::setprecision(10);
		cout<<"Srcs(H) Max      = "<<Srcs_H<<endl;
		cout<<"Srcs(H) Max [dB] = "<<Intensity2dBsm(Srcs_H, Sar.lambda())<<endl;
		cout<<"Srcs(V) Max      = "<<Srcs_V<<endl;
		cout<<"Srcs(V) Max [dB] = "<<Intensity2dBsm(Srcs_V, Sar.lambda())<<endl;
	}else{
		if(Ef.RxPol() == "H"){		// Single (H)
			cout<<"+"<<endl;
			cout<<"  Single Pol Export: (H" + Ef.TxPol() + ")"<<endl;
			cout<<"+"<<endl;
			double Srcs_H = mat::maxAbs(Echo_H);
			cout<<std::setprecision(10);
			cout<<"Srcs(H) Max      = "<<Srcs_H<<endl;
			cout<<"Srcs(H) Max [dB] = "<<Intensity2dBsm(Srcs_H, Sar.lambda())<<endl;
		}else{						// Single (V)
			cout<<"+"<<endl;
			cout<<"  Single Pol Export: (V" + Ef.TxPol() + ")"<<endl;
			cout<<"+"<<endl;
			double Srcs_V = mat::maxAbs(Echo_V);
			cout<<std::setprecision(10);
			cout<<"Srcs(V) Max      = "<<Srcs_V<<endl;
			cout<<"Srcs(V) Max [dB] = "<<Intensity2dBsm(Srcs_V, Sar.lambda())<<endl;
		}
	}
#pragma endregion


	cout<<endl;
	cout<<">>> Everything is OK, Finish <<<"<<endl;
	return 0;
}


