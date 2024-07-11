#ifndef IO_H
#define IO_H


#include <basic/def_func.h>
#include <sar/envi.h>
#include <sar/sv.h>
#include <mesh/poly.h>
#include <rcs/rcs.h>
#include <json/json.h>

using namespace sv;
using namespace poly;
using namespace rcs;
using namespace def;

namespace io{
	namespace read{
		sv::SV<double> SV(const char* filename);
		sv::SV<double> SV(const char* filename,int &yr,int &mo,int &d,int &h,int &m,double &s); // *NOT* effective
		//void SimPar(const char* filename);
		void SimPar(const string file, SAR& Sar, ORB& Orb, EF& Ef, MultiAngle& MulAng, MeshDef& mesh, int& ReduceFactor, const double theta_l_MB);
		void SimPar(const string file, SAR& Sar, EF& Ef, MultiAngle& MulAng, MeshDef& mesh, int& ReduceFactor, const double theta_l_MB);
		void SimPar(const string file_SAR_conf, SAR& Sar, EF& Ef, MeshDef& mesh, ORB& Orb, double& TargetAsp, double& TargetLon, double& TargetLatGd);
		void SimPar(const string file_SAR_conf, SAR& Sar, EF& Ef, MeshDef& mesh, ORB& Orb, TAR& tar);
//		void SimPar(const char* filename,double inc,const char* pol,double theta_sqc,def::SAR& out_Sar,def::ORB& out_Orb);
        D2<double> RCSTable(const char* filename,double& freq); // *NOT* effective
        POLY<double> CSISTCutPolygons(const char* filename); // *NOT* effective
        template<typename T> void CSISTCutPolygons(const char* filename,const long num,POLY<T>& out);
        void CSISTCutRCS(const char* filename,const long num,RCS<double>& out);
    }
	
	namespace write{
		template<typename T> void ENVI(T& data, const D1<size_t> dim, const string out_filename);
		void SV(const sv::SV<double>& sv, const char* filename);
		void Meta(const SAR& Sar, const MeshDef& mesh, const EF& Ef, const def::ANG& Ang,
				  const string file_Cnf, const string name, const string file_out_RCS_meta);
		void PASSEDEcho(const string& dir_out, const string& name, const EF& Ef, const double theta_l_MB, const size_t MaxLevel,
						D3<CPLX<float> >& Echo_H, D3<CPLX<float> >& Echo_V, const string& suffix = "");
	}
}


//
//
// namespace read
//
//
SV<double> io::read::SV(const char* filename){
	int ST_yr,ST_mo,ST_d,ST_h,ST_m;
	double ST_s;
	sv::SV<double> TER_SV=io::read::SV(filename,ST_yr,ST_mo,ST_d,ST_h,ST_m,ST_s);
	TIME ST(ST_yr,ST_mo,ST_d,ST_h,ST_m,ST_s);
	double t_min = UTC2GPS(ST); // new first SV GPS time
	
	// correct the time in SV to absolute GPS time (the original first time is zero)
    for(long i=0;i<TER_SV.GetNum();++i){
        TER_SV.t()[i]+=t_min;
    }
	return TER_SV;
}

SV<double> io::read::SV(const char* filename,int &yr,int &mo,int &d,int &h,int &m,double &s){
	/*
	 Purpose:
	 Read State vector
	 Input:
	 filename :(string) input filename
	 Return:
	 in :(VEC<T>*)
	 */
	long i=0,line=0;//,buf_len=10000;
	char buf[10000];
	fstream fin(filename,fstream::in);
	if(!fin.good()){
		cout<<"ERROR::[def_func::read::SV]:No this file! -> ";
		cout<<filename<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	double t,x,y,z,vx,vy,vz;
	// Get number of line
	while(fin.getline(buf,10000)){
		line++;
	}
	line--;
//#ifdef _DEBUG
//	cout<<line<<endl;
//#endif
	// back to start of ASCII file
	fin.clear();
	fin.seekg(0,ios::beg);
	// read first line (Start date & time)
	fin.getline(buf,10000);
#ifdef _MSC_VER
	char* next_tok;
	char* split = strtok_s(buf," /:",&next_tok);
#else
	char* split = strtok(buf," /:");
#endif
	yr = atoi(split);
	int count=0;
	while(split != NULL){
#ifdef _MSC_VER
		next_tok;
		split = strtok_s(buf," /:",&next_tok);
#else
		split = strtok(NULL," /:");
#endif
		if(count == 0){ mo=atoi(split); }
		if(count == 1){ d=atoi(split); }
		if(count == 2){ h=atoi(split); }
		if(count == 3){ m=atoi(split); }
		if(count == 4){ s=atof(split); }
		count++;
	}
	// Read data
	D1<double> tt(line);
	D1<VEC<double> > pos(line),vel(line);
	for(i=0;i<line;i++){
		fin>>t>>x>>y>>z>>vx>>vy>>vz;
		//cout<<t<<" "<<x<<" "<<y<<" "<<z<<" "<<vx<<" "<<vy<<" "<<vz<<endl;
		tt[i]=t;
		pos[i].Setxyz(x,y,z);
		vel[i].Setxyz(vx,vy,vz);
	}
	fin.close();
	return sv::SV<double>(tt,pos,vel,line);
}

void io::read::SimPar(const string file_SAR, SAR& Sar, ORB& Orb, EF& Ef, MultiAngle& MulAng, MeshDef& mesh, int& ReduceFactor, const double theta_l_MB = -9999){
	fstream fin(file_SAR.c_str(),fstream::in);
	if(!fin.good()){
		cout<<"ERROR::[def_func::read::SimPar]:No this file_SAR! -> ";
		cout<<file_SAR<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	
	// Name
	string Sen_name;
	// Angle
	bool IsMulLook;
	double Look_From, Look_To, Look_Step;
	bool IsMulAsp;
	double Asp_From, Asp_To, Asp_Step, theta_sq;
	// Sar
	double f0, PRF, Fr, DC, BWrg, SWrg;
	double Laz, Lev, ant_eff;
	// Datum
	string Dat_name;
	double Ea, Eb;
	// Data Dimenison
	long Nr, Na;
	// Electric field
	string TxPol, RxPol;
//	double Rss, DF;
//	double RF_TE, RF_TM;
	// PO approximation
	double TaylorRg;
	int TaylorNt;
	long MaxLevel;
	// Incident mesh
	double MeshScale, MeshdRad;
	
	// initialize values
//	RF_TE = RF_TM = 0;
	TaylorRg = 0;  TaylorNt = 0;
	theta_sq = 0;
	f0 = PRF = Fr = DC = BWrg = Laz = Lev = ant_eff = SWrg = 0;
	Nr = Na = 0;
	Ea = Eb = 0;
//	Rss = DF = 0;
	MaxLevel = 0;
	Look_From = Look_To = Look_Step = 0;
	IsMulAsp = IsMulLook = false;
	Asp_From = Asp_To = Asp_Step = 0;
	MeshScale = MeshdRad = 0;
	
	char buf[1000];
	while(fin.getline(buf,1000)){
		// Skip comment line
		if(buf[0] != '#'){
			// Convert to string
			string str = string(buf);
			// string split
			D1<string> tmp = StrSplit(str, '=');
			string tag = tmp[0];
			tag.erase(tag.end()-1);
			string val = (StrSplit(tmp[1], ' '))[1];
			// Assign
			// Name
			if(tag == "Sensor Name"){ str2num(val, Sen_name); }
			// Angle
			if(tag == "Multi-Look angle"){ IsMulLook = (StrUppercase(val) == "YES")? true:false; }
			if(tag == "Look angle (from) [deg]"){ str2num(val, Look_From); }
			if(tag == "Look angle (to) [deg]"){ str2num(val, Look_To); }
			if(tag == "Look angle (step) [deg]"){ str2num(val, Look_Step); }
			if(tag == "Multi-Aspect angle"){ IsMulAsp = (StrUppercase(val) == "YES")? true:false; }
			if(tag == "Target Aspect angle (from) [deg]"){ str2num(val, Asp_From); }
			if(tag == "Target Aspect angle (to) [deg]"){ str2num(val, Asp_To); }
			if(tag == "Target Aspect angle (step) [deg]"){ str2num(val, Asp_Step); }
			if(tag == "Squint angle [deg]"){ str2num(val, theta_sq); }
			// SAR
			if(tag == "Transmitted frequency [Hz]"){ str2num(val, f0); }
			if(tag == "Pulse Repeat Frequency [Hz]"){ str2num(val, PRF); }
			if(tag == "ADC sampling rate [Hz/sec]"){ str2num(val, Fr); }
			if(tag == "Duty Cycle [%]"){ str2num(val, DC); }
			if(tag == "Chirp bandwidth [Hz]"){ str2num(val, BWrg); }
			if(tag == "Slant Range Swath [m]"){ str2num(val, SWrg); }
			// Antenna
			if(tag == "Length at azimuth [m]"){ str2num(val, Laz); }
			if(tag == "Length at elevation [m]"){ str2num(val, Lev); }
			if(tag == "Antenna effective coefficient"){ str2num(val, ant_eff); }
			// Orbit
			if(tag == "Datum"){ str2num(val, Dat_name); }
			if(tag == "Earth semi-major length [m]"){ str2num(val, Ea); }
			if(tag == "Earth semi-mirror length [m]"){ str2num(val, Eb); }
			// Data Dimension
			if(tag == "Number of Range [sample]"){ str2num(val, Nr); }
			if(tag == "Number of Azimuth [sample]"){ str2num(val, Na); }
			// Electric Field
			if(tag == "TX Polization"){ TxPol = val; }
			if(tag == "RX Polization"){ RxPol = val; }
			if(tag == "Reducing Factor in Azimuth"){ str2num(val, ReduceFactor); }
//			if(tag == "Resistivity"){ str2num(val, Rss); }
//			if(tag == "Divergence factor"){ str2num(val, DF); }
//			if(tag == "Reflection coefficient TE"){ str2num(val, RF_TE); }
//			if(tag == "Reflection coefficient TM"){ str2num(val, RF_TM); }
			// PO approximation coeff.
			if(tag == "Taylor Limit Value"){ str2num(val, TaylorRg); }
			if(tag == "Taylor Series Number"){ str2num(val, TaylorNt); }
			if(tag == "Max Bouncing Number"){ str2num(val, MaxLevel); }
			// Incident mesh
			if(tag == "Mesh Scale Factor"){ str2num(val, MeshScale); }
			if(tag == "Mesh Distance from targets center"){ str2num(val, MeshdRad); }
		}
	}
	
	double Theta_l_MB;
	if(theta_l_MB == -9999){	// undefined
		Theta_l_MB = deg2rad(Look_From);
	}else{					// defined & replace the Look angle to single MB
		IsMulLook = false;
		Look_From = deg2rad(theta_l_MB);
		Look_To   = deg2rad(theta_l_MB);
		Look_Step = 0;
		Theta_l_MB = deg2rad(theta_l_MB);
	}
	
//	vector<double> Raz(2);
//	Raz[0] = Raz_s; Raz[1] = Raz_e;
	
//	RF Rf(RF_TE, RF_TM);
	TAYLOR Taylor(TaylorRg, TaylorNt);
	
	Sar    = SAR(Sen_name, Theta_l_MB, theta_sq, f0, PRF, Fr, DC, BWrg, SWrg, Laz, Lev, ant_eff, Nr, Na, true);	// always up-chirp
	Orb    = ORB(Dat_name, Ea, Eb);
	Ef     = EF(TxPol, RxPol, Taylor, MaxLevel);
	MulAng = MultiAngle(IsMulLook, Look_From, Look_To, Look_Step, IsMulAsp, Asp_From, Asp_To, Asp_Step);
	mesh   = MeshDef(MeshScale, MeshdRad);
}

void io::read::SimPar(const string file_RCS, SAR& Sar, EF& Ef, MultiAngle& MulAng, MeshDef& mesh, int& ReduceFactor, const double theta_l_MB = -9999){
	fstream fin(file_RCS.c_str(),fstream::in);
	if(!fin.good()){
		cout<<"ERROR::[def_func::read::SimPar]:No this file_RCS! -> ";
		cout<<file_RCS<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	
	// Name
	string Sen_name;
	// Angle
	bool IsMulLook;
	double Look_From, Look_To, Look_Step;
	bool IsMulAsp;
	double Asp_From, Asp_To, Asp_Step, theta_sq;
	// Sar
	double f0, PRF, Fr, DC, BWrg, SWrg;
	double Laz, Lev, ant_eff;
	// Datum
	string Dat_name;
	double Ea, Eb;
	// Data Dimenison
	long Nr, Na;
	// Electric field
	string TxPol, RxPol;
//	double Rss, DF;
//	double RF_TE, RF_TM;
	// PO approximation
	double TaylorRg;
	int TaylorNt;
	long MaxLevel;
	// Incident mesh
	double MeshScale, MeshdRad;
	
	// initialize values
//	RF_TE = RF_TM = 0;
	TaylorRg = 0;  TaylorNt = 0;
	theta_sq = 0;
	f0 = PRF = Fr = DC = BWrg = Laz = Lev = ant_eff = SWrg = 0;
	Nr = Na = 0;
	Ea = Eb = 0;
//	Rss = DF = 0;
	MaxLevel = 0;
	Look_From = Look_To = Look_Step = 0;
	IsMulAsp = IsMulLook = false;
	Asp_From = Asp_To = Asp_Step = 0;
	MeshScale = MeshdRad = 0;
	
	char buf[1000];
	while(fin.getline(buf,1000)){
		// Skip comment line
		if(buf[0] != '#'){
			// Convert to string
			string str = string(buf);
			// string split
			D1<string> tmp = StrSplit(str, '=');
			string tag = tmp[0];
			tag.erase(tag.end()-1);
			string val = (StrSplit(tmp[1], ' '))[1];
			// Assign
			// Name
			if(tag == "Sensor Name"){ str2num(val, Sen_name); }
			// Angle
			if(tag == "Multi-Look angle"){ IsMulLook = (StrUppercase(val) == "YES")? true:false; }
			if(tag == "Look angle (from) [deg]"){ str2num(val, Look_From); }
			if(tag == "Look angle (to) [deg]"){ str2num(val, Look_To); }
			if(tag == "Look angle (step) [deg]"){ str2num(val, Look_Step); }
			if(tag == "Multi-Aspect angle"){ IsMulAsp = (StrUppercase(val) == "YES")? true:false; }
			if(tag == "Target Aspect angle (from) [deg]"){ str2num(val, Asp_From); }
			if(tag == "Target Aspect angle (to) [deg]"){ str2num(val, Asp_To); }
			if(tag == "Target Aspect angle (step) [deg]"){ str2num(val, Asp_Step); }
			if(tag == "Squint angle [deg]"){ str2num(val, theta_sq); }
			// SAR
			if(tag == "Transmitted frequency [Hz]"){ str2num(val, f0); }
			if(tag == "Pulse Repeat Frequency [Hz]"){ str2num(val, PRF); }
			if(tag == "ADC sampling rate [Hz/sec]"){ str2num(val, Fr); }
			if(tag == "Duty Cycle [%]"){ str2num(val, DC); }
			if(tag == "Chirp bandwidth [Hz]"){ str2num(val, BWrg); }
			if(tag == "Slant Range Swath [m]"){ str2num(val, SWrg); }
			// Antenna
			if(tag == "Length at azimuth [m]"){ str2num(val, Laz); }
			if(tag == "Length at elevation [m]"){ str2num(val, Lev); }
			if(tag == "Antenna effective coefficient"){ str2num(val, ant_eff); }
			// Orbit
			if(tag == "Datum"){ str2num(val, Dat_name); }
			if(tag == "Earth semi-major length [m]"){ str2num(val, Ea); }
			if(tag == "Earth semi-mirror length [m]"){ str2num(val, Eb); }
			// Data Dimension
			if(tag == "Number of Range [sample]"){ str2num(val, Nr); }
			if(tag == "Number of Azimuth [sample]"){ str2num(val, Na); }
			// Electric Field
			if(tag == "TX Polization"){ TxPol = val; }
			if(tag == "RX Polization"){ RxPol = val; }
			if(tag == "Reducing Factor in Azimuth"){ str2num(val, ReduceFactor); }
//			if(tag == "Resistivity"){ str2num(val, Rss); }
//			if(tag == "Divergence factor"){ str2num(val, DF); }
//			if(tag == "Reflection coefficient TE"){ str2num(val, RF_TE); }
//			if(tag == "Reflection coefficient TM"){ str2num(val, RF_TM); }
			// PO approximation coeff.
			if(tag == "Taylor Limit Value"){ str2num(val, TaylorRg); }
			if(tag == "Taylor Series Number"){ str2num(val, TaylorNt); }
			if(tag == "Max Bouncing Number"){ str2num(val, MaxLevel); }
			// Incident mesh
			if(tag == "Mesh Scale Factor"){ str2num(val, MeshScale); }
			if(tag == "Mesh Distance from targets center"){ str2num(val, MeshdRad); }
		}
	}
	
	double Theta_l_MB;
	if(theta_l_MB == -9999){	// undefined
		Theta_l_MB = deg2rad(Look_From);
	}else{					// defined & replace the Look angle to single MB
		IsMulLook = false;
		Look_From = deg2rad(theta_l_MB);
		Look_To   = deg2rad(theta_l_MB);
		Look_Step = 0;
		Theta_l_MB = deg2rad(theta_l_MB);
	}
	
	//	vector<double> Raz(2);
	//	Raz[0] = Raz_s; Raz[1] = Raz_e;
	
	//	RF Rf(RF_TE, RF_TM);
	TAYLOR Taylor(TaylorRg, TaylorNt);
	
	Sar    = SAR(Sen_name, Theta_l_MB, theta_sq, f0, PRF, Fr, DC, BWrg, SWrg, Laz, Lev, ant_eff, Nr, Na, true);	// always up=chirp
	Ef     = EF(TxPol, RxPol, Taylor, MaxLevel);
	MulAng = MultiAngle(IsMulLook, Look_From, Look_To, Look_Step, IsMulAsp, Asp_From, Asp_To, Asp_Step);
	mesh   = MeshDef(MeshScale, MeshdRad);
}

void io::read::SimPar(const string file_SAR_conf, SAR& Sar, EF& Ef, MeshDef& mesh, ORB& Orb, double& TargetAsp, double& TargetLon, double& TargetLatGd){
	// Read json
	ifstream fin(file_SAR_conf.c_str());
	Json::Value root;
	fin>>root;
	fin.close();

	if(root == Json::nullValue){
		cout<<"ERROR::[def_func::read::SimPar]:No this file_SAR_conf! -> ";
		cout<<file_SAR_conf<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}

	// Name
	string Sen_name;
	// Focused SAR Image Angle
	double theta_l_MB;
	double theta_sqc;
	// SAR
	double f0;
	double PRF;
	double Fr;
	double Tr;
	double BWrg;
	 bool isUpChirp;
	// Data Dimension
	long Nr;
	long Na;
	// Antenna
	double Laz;
	double Lev;
	double ant_eff = 1.0;
	double theta_az;
	double theta_ev;
	// Datum
	string datum_name;
	double Ea;
	double Eb;
	// Electric Field polarization
	string TxPol;
	string RxPol;
	// PO & PTD
	int MaxLevel;
	double MeshScale;

	// Initializa
	TargetLon = 0;
	TargetLatGd = 0;

	// Name
	Sen_name = root["sensor_name"].asString();
	// Focused SAR Image Angle
	theta_l_MB = deg2rad( root["look_angle"].asDouble() );
	theta_sqc = deg2rad( root["squint_angle"].asDouble() );
	TargetAsp = deg2rad( root["target_aspect_angle"].asDouble() );
	// SAR
	// Ref: AN/PPS-5 MSTAR ground radar
	f0 = root["transmit_frequency"].asDouble();
	PRF = root["PRF"].asDouble();
	Fr = root["sampling_rate"].asDouble();
	Tr = root["pulse_width"].asDouble();
	BWrg = root["chirp_bandwidth"].asDouble();
	isUpChirp = root["is_up_chirp"].asBool();
	// Data Dimension
	Nr = root["number_of_range"].asDouble();
	Na = root["number_of_azimuth"].asDouble();
	// Antenna
//	Laz = root["length_at_azimuth"].asDouble();
//	Lev = root["length_at_elevation"].asDouble();
//	ant_eff = root["effective_coefficient"].asDouble();
	theta_az = deg2rad(root["beamwidth_azimuth"].asDouble());
	theta_ev = deg2rad(root["beamwidth_elevation"].asDouble());
	// Datum
	datum_name = root["datum_name"].asString();
	Ea = root["semi_major_length"].asDouble();
	Eb = root["semi_mirror_length"].asDouble();
	// Electric Field polarization
	TxPol = root["TX_polization"].asString();
	RxPol = root["RX_polization"].asString();
	// PO & PTD
	MaxLevel = root["max_bouncing"].asDouble();
	MeshScale = root["ray_density"].asDouble();
	// Target position (for CircularSAR) in degree
	TargetLon = deg2rad(root["target_lon"].asDouble());
	TargetLatGd = deg2rad(root["target_lat"].asDouble());

	double DC = Tr*PRF*100;
	double lambda = def::C/f0;
	Laz = 0.886*lambda/theta_az/ant_eff;
	Lev = 0.886*lambda/theta_ev/ant_eff;

	// Taylor approximation class (dummy)
	TAYLOR Taylor(0.0001, 5);

	// SAR class
	Sar = SAR(Sen_name, theta_l_MB, theta_sqc, f0, PRF, Fr, DC, BWrg, 9999999, Laz, Lev, ant_eff, Nr, Na, isUpChirp);

	// EF class
	Ef = EF(TxPol, RxPol, Taylor, MaxLevel);

	// MeshDef class (MeshRad = 10e3 for default)
	mesh = MeshDef(MeshScale, 10e3);

	// Orbit class
	Orb = ORB(datum_name, Ea, Eb);
}

void io::read::SimPar(const string file_SAR_conf, SAR& Sar, EF& Ef, MeshDef& mesh, ORB& Orb, TAR& tar){
	// Read json
	ifstream fin(file_SAR_conf.c_str());
	Json::Value root;
	fin>>root;
	fin.close();

	if(root == Json::nullValue){
		cout<<"ERROR::[def_func::read::SimPar]:No this file_SAR_conf! -> ";
		cout<<file_SAR_conf<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}

	// Name
	string Sen_name;
	// Focused SAR Image Angle
	double theta_l_MB;
	double theta_sqc;
	// SAR
	double f0;
	double PRF;
	double Fr;
	double Tr;
	double BWrg;
	bool isUpChirp;
	// Data Dimension
	long Nr;
	long Na;
	// Antenna
	double Laz;
	double Lev;
	double ant_eff = 1.0;
	double theta_az;
	double theta_ev;
	// Datum
	string datum_name;
	double Ea;
	double Eb;
	// Electric Field polarization
	string TxPol;
	string RxPol;
	// PO & PTD
	int MaxLevel;
	double MeshScale;

	// Initialization
	tar.Lon() = 0;
	tar.Lat() = 0;

	// Name
	Sen_name = root["sensor_name"].asString();
	// Focused SAR Image Angle
	theta_l_MB = deg2rad( root["look_angle"].asDouble() );
	theta_sqc = deg2rad( root["squint_angle"].asDouble() );
	tar.Asp() = deg2rad( root["target_aspect_angle"].asDouble() );
	// SAR
	// Ref: AN/PPS-5 MSTAR ground radar
	f0 = root["transmit_frequency"].asDouble();
	PRF = root["PRF"].asDouble();
	Fr = root["sampling_rate"].asDouble();
	Tr = root["pulse_width"].asDouble();
	BWrg = root["chirp_bandwidth"].asDouble();
	isUpChirp = root["is_up_chirp"].asBool();
	// Data Dimension
	Nr = root["number_of_range"].asDouble();
	Na = root["number_of_azimuth"].asDouble();
	// Antenna
//	Laz = root["length_at_azimuth"].asDouble();
//	Lev = root["length_at_elevation"].asDouble();
//	ant_eff = root["effective_coefficient"].asDouble();
	theta_az = deg2rad(root["beamwidth_azimuth"].asDouble());
	theta_ev = deg2rad(root["beamwidth_elevation"].asDouble());
	// Datum
	datum_name = root["datum_name"].asString();
	Ea = root["semi_major_length"].asDouble();
	Eb = root["semi_mirror_length"].asDouble();
	// Electric Field polarization
	TxPol = root["TX_polization"].asString();
	RxPol = root["RX_polization"].asString();
	// PO & PTD
	MaxLevel = root["max_bouncing"].asDouble();
	MeshScale = root["ray_density"].asDouble();
	// Target position (for CircularSAR) in degree
	tar.Lon() = deg2rad(root["target_lon"].asDouble());
	tar.Lat() = deg2rad(root["target_lat"].asDouble());

	double DC = Tr*PRF*100;
	double lambda = def::C/f0;
	Laz = 0.886*lambda/theta_az/ant_eff;
	Lev = 0.886*lambda/theta_ev/ant_eff;

	// Taylor approximation class (dummy)
	TAYLOR Taylor(0.0001, 5);

	// SAR class
	Sar = SAR(Sen_name, theta_l_MB, theta_sqc, f0, PRF, Fr, DC, BWrg, 9999999, Laz, Lev, ant_eff, Nr, Na, isUpChirp);

	// EF class
	Ef = EF(TxPol, RxPol, Taylor, MaxLevel);

	// MeshDef class (MeshRad = 10e3 for default)
	mesh = MeshDef(MeshScale, 10e3);

	// Orbit class
	Orb = ORB(datum_name, Ea, Eb);
}


D2<double> io::read::RCSTable(const char* filename,double& freq){
    /*
     Purpose:
	 Read the RCS table
	 */
	fstream fin(filename,fstream::in);
	if(!fin.good()){
		cout<<"ERROR::[def_func::read::RCSTable]:No this file! -> ";
		cout<<filename<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	double idx,theta,phi;
	char buf[10000];
	long line=0;
	// Get number of line
	while(fin.getline(buf,10000)){
		line++;
	}
	D2<double> out(line-1,3);
	
	//back to start of ASCII file
	fin.clear();
	fin.seekg(0, ios::beg);
	
	fin.ignore(12); //"frequency = " 12 bits
	fin>>freq;
//#ifdef DEBUG
//	cout<<" Totla line="<<line<<endl;
//	cout<<"freq="<<freq<<endl;
//#endif
	for(long i=0;i<line-1;++i){
		fin>>idx>>theta>>phi;
		out[i][0]=idx;
		out[i][1]=theta;
		out[i][2]=phi;
//#ifdef DEBUG
//		cout<<idx<<" "<<theta<<" "<<phi<<endl;
//#endif
	}
	fin.close();
	
	return out;
}

POLY<double> io::read::CSISTCutPolygons(const char* filename){
    /*
     Purpose:
	 Read Cutted CSIST Polygon set.
	 */
	ifstream fin(filename,ios::binary);
	if(!fin.good()){
		cout<<"ERROR::[def_func::read::CSISTCutPolygons]:No this file! -> ";
		cout<<filename<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	// Get polygon number
	fin.clear();
	fin.seekg(0,ios::end);
	long num = (long)fin.tellg();
	num = (num-4)/36;
	
	// Go back
	fin.clear();
	fin.seekg(0,ios::beg);
	
	// Initial
	poly::POLY<double> py(num);
	d1::D1<VEC<float> > vec_tmp(3);
	
	// changed from sizeof(long) to sizeof(int) for 64bit system
	//fin.read(reinterpret_cast<char*>(&num),sizeof(long));
	fin.read(reinterpret_cast<char*>(&num),sizeof(int));
	for(long j=0;j<num;++j){ //polygons set number
		for(long i=0;i<3;++i){ //T1 or T2 ot T3
			fin.read(reinterpret_cast<char*>(&vec_tmp[i]),sizeof(vec::VEC<float>));
		}
		py.T0(j)=VEC<double>( double(vec_tmp[0].x()),double(vec_tmp[0].y()),double(vec_tmp[0].z()) );
		py.T1(j)=VEC<double>( double(vec_tmp[1].x()),double(vec_tmp[1].y()),double(vec_tmp[1].z()) );
		py.T2(j)=VEC<double>( double(vec_tmp[2].x()),double(vec_tmp[2].y()),double(vec_tmp[2].z()) );
	}
	
	fin.close();
	
//#ifdef DEBUG
//	cout<<filename<<endl;
//	
//	cout<<num<<endl;
//	py.T0(0).Print();
//	py.T1(0).Print();
//	py.T2(0).Print();
//	
//	cout<<endl;
//	py.T0(1).Print();
//	py.T1(1).Print();
//	py.T2(1).Print();
//	
//	cout<<py.GetNum()<<endl;
//#endif
	
	return py;
}

template<typename T>
void io::read::CSISTCutPolygons(const char* filename,const long num,POLY<T>& out){
    /*
     Purpose:
	 Read Cutted CSIST Polygon set.
	 */
	vec::VEC<float> vec_tmp;
	long tmp;
	
	ifstream fin(filename,ios::binary);
	if(!fin.good()){
		cout<<"ERROR::[def_func::read::CSISTCutPolygons]:No this file! -> ";
		cout<<filename<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	
	fin.read(reinterpret_cast<char*>(&tmp),sizeof(long));
	for(long j=0;j<num;++j){ //polygons set number
		for(long i=0;i<3;++i){ //T1 or T2 ot T3
			fin.read(reinterpret_cast<char*>(&vec_tmp),sizeof(vec::VEC<float>));
			out.SetVal(j,i,vec::VEC<T>( (T)vec_tmp.x(),(T)vec_tmp.y(),(T)vec_tmp.z() ));
		}
	}
	
	fin.close();
}

void io::read::CSISTCutRCS(const char* filename,const long num,RCS<double>& out){
    /*
     Purpose:
	 Read Cutted CSIST RCS set including {HH,HV,VH,VV}
     Input:
	 num :[x] the number of polygon set in this target
	 */
	//string filename=string(dir)+string("01173.dat");
	//ifstream fin(filename.c_str(),ios::binary);
	ifstream fin(filename,ios::binary);
	if(!fin.good()){
		cout<<"ERROR::[def_func::read::CSISTCutRCS]:No this file! -> ";
		cout<<filename<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	
	cplx::CPLX<float> cx;
	
	for(long j=0;j<num;++j){ // number of num
		for(long i=0;i<4;++i){
			fin.read(reinterpret_cast<char*>(&cx),sizeof(cplx::CPLX<float>));
			out.SetVal(j,i,CPLX<double>( (double)cx.r(),(double)cx.i() ));
		}
	}
	
	fin.close();
	
#ifdef DEBUG
	cx.Print();
#endif
}

//
// Write
//
template<typename T>
void io::write::ENVI(T& data, const D1<size_t> dim, const string out_filename){
	// analysis filename
	// Extract file name
	struct Filename str = def_func::StrFilename(string(out_filename));
	string out_filename_hdr = str.path + str.name + string(".hdr");
	
	// Write ENVI header file (ASCII file)
	long sample, line, band;
	string type;
	switch(dim.GetNum()){
		case 1: // 1D
			sample = dim[0];
			line = 1L;
			band = 1L;
			break;
		case 2: // 2D
			sample = dim[0];
			line = dim[1];
			band = 1L;
			break;
		default: // 3D
			sample = dim[0];
			line = dim[1];
			band = dim[2];
			break;
	}
	
	envi::ENVIhdr hdr(line, sample, band, 0, data.GetType(), 0, "BIP");
	hdr.WriteENVIHeader(out_filename_hdr.c_str());
	
	// Write Binary data
	data.WriteBinary(out_filename.c_str());
}

void io::write::SV(const sv::SV<double>& sv, const char* filename){
	ofstream fout(filename);
	double xx,yy,zz,vx,vy,vz;
	double t0 = sv.t()[0];

	TIME t = GPS2UTC(t0);

	if(fout.fail()){
		cout<<"ERROR::[io::write::SV]:Output file path error! -> ";
		cout<<filename<<endl;
		exit(EXIT_FAILURE);
	}

	fout<<" "<<t.GetYear()<<"/"<<t.GetMonth()<<"/"<<t.GetDay()<<" "<<t.GetHour()<<":"<<t.GetMin()<<":"<<t.GetSec()<<endl;

	for(long i=0;i<sv.GetNum();++i){
		xx=sv.pos()[i].x();
		yy=sv.pos()[i].y();
		zz=sv.pos()[i].z();
		vx=sv.vel()[i].x();
		vy=sv.vel()[i].y();
		vz=sv.vel()[i].z();
		fout<<" ";
		fout<<std::setprecision(16)<<std::fixed<<sv.t()[i] - t0<<"\t";
		fout<<std::setprecision(16)<<std::fixed<<xx<<"\t"<<yy<<"\t"<<zz<<"\t";
		fout<<std::setprecision(16)<<std::fixed<<vx<<"\t"<<vy<<"\t"<<vz<<"\n";
	}
	fout.close();
}

void io::write::Meta(const SAR& Sar, const MeshDef& mesh, const EF& Ef, const def::ANG& Ang,
			   const string file_Cnf, const string name, const string file_out_RCS_meta){
	D1<string> meta(33);
	meta[0]  = "#+=============================+";
	meta[1]  = "#|     RCS results (Meta)      |";
	meta[2]  = "#+=============================+";
	meta[3]  = "# Name";
	meta[4]  = "SARCNF Name = " + file_Cnf;
	meta[5]  = "Result Name = " + name + "_rcs.dat";
	meta[6]  = "#";
	meta[7]  = "# (segment from *.parrcs)";
	meta[8]  = "#";
	meta[9]  = "# Angle";
	meta[10] = "Multi-Look angle = No";
	meta[11] = "Look angle (min) [deg] = " + num2str(rad2deg(max(Ang.Look())),7);
	meta[12] = "Look angle (max) [deg] = " + num2str(rad2deg(max(Ang.Look())),7);
	meta[13] = "Look angle (avg. step) [deg] = " + num2str(rad2deg(AvgStep(Ang.Look())),7);
	meta[14] = "Multi-Aspect angle = Yes";
	meta[15] = "Target Aspect angle (min) [deg] = " + num2str(rad2deg(min(Ang.Asp())),7);
	meta[16] = "Target Aspect angle (max) [deg] = " + num2str(rad2deg(max(Ang.Asp())),7);
	meta[17] = "Target Aspect angle (avg. step) [deg] = " + num2str(rad2deg(AvgStep(Ang.Asp())),7);
	meta[18] = "# SAR";
	meta[19] = "Transmitted frequency [Hz] = " + num2str(Sar.f0());
	meta[20] = "ADC sampling rate [Hz/sec] = " + num2str(Sar.Fr());
	meta[21] = "# Data Dimension";
	meta[22] = "Number of Range [sample] = " + num2str(Sar.Nr());
	meta[23] = "# Electric Field";
	meta[24] = "TX Polization = " + Ef.TxPol();
	meta[25] = "RX Polization = " + Ef.RxPol();
	meta[26] = "# PO approximation coeff.";
	meta[27] = "Taylor Limit Value = " + num2str(Ef.Taylor().Rg());
	meta[28] = "Taylor Series Number = " + num2str(Ef.Taylor().Nt());
	meta[29] = "Max Bouncing Number = " + num2str(Ef.MaxLevel());
	meta[30] = "# Incident Mesh";
	meta[31] = "Mesh Scale Factor = " + num2str(mesh.Scale());
	meta[32] = "Mesh Distance from targets center = " + num2str(mesh.dRad());
	meta.WriteASCII(file_out_RCS_meta.c_str());
}

void io::write::PASSEDEcho(const string& dir_out, const string& name, const EF& Ef, const double theta_l_MB, const size_t MaxLevel,
						   D3<CPLX<float> >& Echo_H, D3<CPLX<float> >& Echo_V, const string& suffix){
	string name_RCS_H, name_RCS_V;

	string postString = (suffix == "")? "":"_" + suffix;

	if(Ef.RxPol().length() == 2){//------- Dual -------------------------------------------
		// make name
		string prefix = dir_out + name + "_" + StrTruncate(num2str(rad2deg(theta_l_MB)),2);
		name_RCS_H = prefix + "_rcs_H" + Ef.TxPol() + postString;
		name_RCS_V = prefix + "_rcs_V" + Ef.TxPol() + postString;
		// Write Binary [ Total ]
		// D3<CPLX<float> > Echo_H(ang.GetNum(), freq.GetNum(), Ef.MaxLevel());	// Echo_PO_H[#ang][#freq][#level]
		// D3<CPLX<float> > Echo_V(ang.GetNum(), freq.GetNum(), Ef.MaxLevel());	// Echo_PO_V[#ang][#freq][#level]
		Echo_H.GetD2atP(0).WriteBinary((name_RCS_H + ".dat").c_str());
		Echo_V.GetD2atP(0).WriteBinary((name_RCS_V + ".dat").c_str());
		// Write Binary [ For each Level ]
		for(size_t k=0;k<MaxLevel;++k){
			Echo_H.GetD2atP(k + 1).WriteBinary((name_RCS_H + "_Level" + num2str(k + 1) + ".dat").c_str());
			Echo_V.GetD2atP(k + 1).WriteBinary((name_RCS_V + "_Level" + num2str(k + 1) + ".dat").c_str());
		}
	}else{//------------------------------ Single ------------------------------------------
		if(Ef.RxPol() == "H"){		// H
			// make name
			string prefix = dir_out + name + StrTruncate(num2str(rad2deg(theta_l_MB)),2);
			name_RCS_H = prefix + "_rcs_H" + Ef.TxPol() + postString;
			// Write Binary [ Total ]
			Echo_H.GetD2atP(0).WriteBinary((name_RCS_H + ".dat").c_str());
			// Write Binary [ For each Level ]
			for(size_t k=0;k<MaxLevel;++k){
				Echo_H.GetD2atP(k + 1).WriteBinary((name_RCS_H + "_Level" + num2str(k + 1) + ".dat").c_str());
			}
		}else{						// V
			// make name
			string prefix = dir_out + name + StrTruncate(num2str(rad2deg(theta_l_MB)),2);
			name_RCS_V = prefix + "_rcs_V" + Ef.TxPol() + postString;
			// Write Binary [ Total ]
			Echo_V.GetD2atP(0).WriteBinary((name_RCS_V + ".dat").c_str());
			// Write Binary [ For each Level ]
			for(size_t k=0;k<MaxLevel;++k){
				Echo_V.GetD2atP(k + 1).WriteBinary((name_RCS_V + "_Level" + num2str(k + 1) + ".dat").c_str());
			}
		}
	}
}

#endif
