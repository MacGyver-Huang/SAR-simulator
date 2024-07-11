#ifndef def_h
#define def_h

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstdlib> //EXIT_FAILURE and EXIT_SUCCESS
#include <ctime>
#include <sstream> // for stringstream
#include <cstring> // for strcpy
#include <algorithm> // for std::copy
#include <vector>
#include <string>
#include <basic/cplx.h>

using namespace std;
using namespace cplx;


namespace def{
	//=============================
	// const def
	//=============================
	const double PI=3.141592653589793;		// PI
	const double PI2= 6.283185307179586;	// 2*PI
	const double PI4=12.566370614359172;	// 4*PI
	const double PI2_C=2.095845022e-8;		// 2*PI/C
	const double SQRT2=1.41421356237;		// square root of 2
	const double SQRT2_INV=0.707106781;		// inverse of square root of 2
	//double C=2.99792458e8;
	const double DTR=0.0174532925199;		// 1/180*PI
	const double RTD=57.2957795131;			// 180/PI
	const double C = 2.99792458e8;			// [m/s] light speed
	
	#ifdef _WIN32
		const double clock_ref=1e3;		// CPU time clock for windows 32bits
	#else
		const double clock_ref=1.0e6;		// CPU time clock for Linux
	#endif
	
	const double FACTORIAL[] = {1,1,2,6,24,120,720,5040,40320,362880,3628800,
							    39916800., 479001600.,6227020800.,87178291200.,
							  	1307674368000.,20922789888000.,355687428096000.,
							  	6402373705728000.,121645100408832000.,
							  	2432902008176640000.};
	
	//=============================
	// TYPE define for this lib
	//=============================
	// If modifying this definition, the procedure/function are MUST edited again, 
	// such as "def_prefunc::GetType", "def_prefunc::Type2IDLType", "D1<T>::_error()"
	enum TYPE {
		BOOLEAN,	// Bool
		CHAR,		// char
		SHORT,		// short
		INT,		// int
		FLT,		// float
		DB,			// double
		CPLX_FLT,	// CPLX<float>
		STR,		// string
		CPLX_DB,	// CPLX<double>
		LONG,		// long
		STLEACH,	// STLEach
		SPH_FLT,	// SPH<float>
		SPH_DB,		// SPH<double>
		SIZE_T,		// size_t
		// Mirror pointer
		pBOOLEAN,	// bool*
		pCHAR,		// char*
		pSHORT,		// short*
		pINT,		// int*
		pFLT,		// float*
		pDB,		// double*
		pCPLX_FLT,	// CPLX<float>*
		pSTR,		// string*
		pCPLX_DB,	// CPLX<double>*
		pLONG,		// long*
		pSTLEACH,	// STLEach*
		pSPH_FLT,	// SPH<float>*
		pSPH_DB,	// SPH<double>*
		pSIZE_T,	// size_t*
		// Not belong to IDL data type
		VEC_FLT,	// VEC<float>
		VEC_DB,		// VEC<double>
		GEO_FLT,	// GEO<float> >
		GEO_DB,		// GEO<double> >
		pVEC_FLT,	// VEC<float>*
		pVEC_DB,	// VEC<double>*
		pGEO_FLT,	// GEO<float> >*
		pGEO_DB,	// GEO<double> >*
		RPY_FLT,	// VEC<float>
		RPY_DB,		// VEC<double>
		pRPY_FLT,	// VEC<float>*
		pRPY_DB,	// VEC<double>*
		LPV_FLT,	// VEC<float>
		LPV_DB,		// VEC<double>
		pLPV_FLT,	// VEC<float>*
		pLPV_DB,	// VEC<double>*
		// D1
		D1_pCPLX_FLT,		// D1<CPLX<float>*>
		D1_pCPLX_DB,		// D1<CPLX<double>*>
		// UNKNOW
		UNKNOW
	};
	
	//+===========================+
	//|          Orbit            |
	//+===========================+
	class ORB{
	public:
		// Constructure
		ORB(){
			_name = "WGS84"					;// Datum name
			_E_a = 6378137.0				;// [m] Earth semi-major axis (WGS84)
			_E_b = 6356752.31414			;// [m] Earth semi-mirror axis(WGS84)
			_f  =0.0033528106875095227413	;//(E_a-E_b)/E_a;	//[x] flatness
			_e2 =0.0066943800355127669813	;//1.-E_b*E_b/(E_a*E_a) // [x] Eccentricity;
		}
		ORB(string name, double E_a, double E_b){
			_name= name;
			_E_a = E_a;
			_E_b = E_b;
			_f   = (E_a-E_b)/E_a;
			_e2  = 1.-E_b*E_b/(E_a*E_a);
		}
		ORB(string name, double E_a, double E_b, double f, double e2){
			_name= name;
			_E_a = E_a;
			_E_b = E_b;
			_f   = f;
			_e2  = e2;
		}
		// Get
		const double& E_a() const{return _E_a;};
		const double& E_b() const{return _E_b;};
		const double& f() const{return _f;};
		const double& e2() const{return _e2;};
		double& E_a(){return _E_a;};
		double& E_b(){return _E_b;};
		double& f(){return _f;};
		double& e2(){return _e2;};
		void Print(){
			cout<<"+-------------------------------------+"<<endl;
			cout<<"|           def::ORB class            |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(20);
			cout<<"name      = "<<_name<<endl;
			cout<<"E_a       = "<<_E_a<<endl;
			cout<<"E_b       = "<<_E_b<<endl;
			cout<<"f         = "<<_f<<endl;
			cout<<"e2        = "<<_e2<<endl;
		}
	private:
		string _name;	// Datum name
		double _E_a;	// [m] Earth semi-major axis (WGS84)
		double _E_b;	// [m] Earth semi-mirror axis(WGS84)
		double _f;		//(E_a-E_b)/E_a;	//[x] flatness
		double _e2;		//1.-E_b*E_b/(E_a*E_a) // [x] Eccentricity;
		//double GM = 398600.4405 		;// [km^3 s^-2] G(gravitational constant, M Earth's mass
    };
	
	//+===========================+
	//|          Target           |
	//+===========================+
    class TAR{
	public:
		// Constructure
		TAR():_asp(DTR*90.0),_lon(0),_lat(0){};
		TAR(const double Asp, double Lon, const double LatGd):_asp(Asp),_lon(Lon),_lat(LatGd){};
		// Getter & setter
		const double& Asp() const{return _asp;};
		double& Asp(){return _asp;};
		const double& Lon() const{return _lon;};
		double& Lon(){return _lon;};
		const double& Lat() const{return _lat;};
		double& Lat(){return _lat;};
		// Misc.
		void Print(){
			cout<<"+------------------+"<<endl;
			cout<<"|      Target      |"<<endl;
			cout<<"+------------------+"<<endl;
			cout<<"Asp       = "<<RTD*_asp<<" [deg]"<<endl;
			cout<<"Longitude = "<<RTD*_lon<<" [deg]"<<endl;
			cout<<"Latitude  = "<<RTD*_lat<<" [deg]"<<endl;
		}
	private:
		double _asp;	// [deg] target aspect angle
		double _lon;
		double _lat;
    };
	
	//+===========================+
	//|           Clock           |
	//+===========================+
    class CLOCK{
	public:
		//constructor
		CLOCK():_time(0){};
		CLOCK(clock_t time):_time(time){};
		const clock_t& time() const{return _time;};
		clock_t& time(){return _time;};
	private:
		clock_t _time;
	};
	
	//+===========================+
	//|     Reflection Factor     |
	//+===========================+
	class RF{
	public:
		RF(){};
		RF(const CPLX<double>& TE, const CPLX<double>& TM):_TE(TE),_TM(TM){};
		// const
		const CPLX<double>& TE_perp() const{ return _TE; }
		const CPLX<double>& TM_par() const{ return _TM; }
		// be able to assign
		CPLX<double>& TE_perp(){ return _TE; }
		CPLX<double>& TM_par(){ return _TM; }
		// misc.
		void Print(){
			cout<<"+-------------------------------------+"<<endl;
			cout<<"|  def::RF(Reflection Factor) class   |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(20);
			cout<<"TE (Perpendicular) = "; _TE.Print();
			cout<<"TM (Parallel)      = "; _TM.Print();
		}
	private:
		CPLX<double> _TE, _TM;
	};
	
	//+===========================+
	//|    Taylor Approximation   |
	//+===========================+
	class TAYLOR{
	public:
		TAYLOR():_Rg(0.5),_Nt(5){};
		TAYLOR(const double Rg, const int Nt):_Rg(Rg),_Nt(Nt){};
		// const
		const double& Rg() const{ return _Rg; }
		const int& Nt() const{ return _Nt; }
		// be able to assign
		double& Rg(){ return _Rg; }
		int& Nt(){ return _Nt; }
		// misc.
		void Print(){
			cout<<"+-------------------------------------+"<<endl;
			cout<<"| def::TAYLOR(PO Approximation) class |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(20);
			cout<<"Rg = "<<_Rg<<endl;
			cout<<"Nt = "<<_Nt<<endl;
		}
	public:
		double _Rg;
		int _Nt;
	};
	
	//+===============================+
	//|  E Field Amplitude component  |
	//+===============================+
	// Assign Tx Polarization
	class EAmp{
	public:
		EAmp(){};
		EAmp(const CPLX<double>& Et, const CPLX<double>& Ep):_Et(Et),_Ep(Ep){};
		// const
		const CPLX<double>& Et() const{ return _Et; }
		const CPLX<double>& Ep() const{ return _Ep; }
		// to be able to assign
		CPLX<double>& Et(){ return _Et; }
		CPLX<double>& Ep(){ return _Ep; }
		// Misc.
		void Print(){
			cout<<"+-------------------------------------+"<<endl;
			cout<<"| def::Eamp(Electric Amplitude) class |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(20);
			cout<<"Et (theta, V) = "<<_Et<<endl;
			cout<<"Ep (phi,   H) = "<<_Ep<<endl;
		}
	private:
		CPLX<double> _Et, _Ep;
	};
	
	//+===========================+
	//|   Electric field Class    |
	//+===========================+
	class EF{
	public:
		EF():_MaxLevel(){};
		EF(const string TxPol, const string RxPol, const TAYLOR& Taylor, const long MaxLevel){
			_TxPol = _StrUppercase(TxPol);
			_RxPol = _StrUppercase(RxPol);
			_Taylor = Taylor;
			_MaxLevel = MaxLevel;
			// others
			CombinePol();
			CalculateEamp();
		}
		// Const
		const string& Pol() const{ return _Pol; }
		const string& TxPol() const{ return _TxPol; }
		const string& RxPol() const{ return _RxPol; }
		const TAYLOR& Taylor() const{ return _Taylor; }
		const EAmp& Eamp() const{ return _Eamp; }
		const long& MaxLevel() const{ return _MaxLevel; }
		// be able to assign
		string& Pol(){ return _Pol; }
//		string& TxPol(){ return _TxPol; }
//		string& RxPol(){ return _RxPol; }
		void SetTxPol(const string TxPol){
			_TxPol = _StrUppercase(TxPol);
			CombinePol();
			CalculateEamp();
		}
		void SetRxPol(const string RxPol){
			_RxPol = _StrUppercase(RxPol);
			CombinePol();
		}
		TAYLOR& Taylor(){ return _Taylor; }
		EAmp& Eamp(){ return _Eamp; }
		long& MaxLevel(){ return _MaxLevel; }
		// misc.
		void Print(){
			cout<<"+-------------------------------------+"<<endl;
			cout<<"|    def::EF(Electric Field) class    |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(20);
			cout<<"Pol            = "<<_Pol<<endl;
			cout<<"TxPol          = "<<_TxPol<<endl;
			cout<<"RxPol          = "<<_RxPol<<endl;
			cout<<"Taylor (Rg,Nt) = "<<_Taylor.Rg()<<","<<_Taylor.Nt()<<endl;
			cout<<"MaxLevel       = "<<_MaxLevel<<endl;
		}
	private:
		void CombinePol(){
			_Pol = _RxPol + _TxPol;
		}
		void CalculateEamp(){
			// Assign Tx Polarization
			if(_TxPol == string("V")){
				_Eamp.Et() = CPLX<double>(1,0); // TM(V)
				_Eamp.Ep() = CPLX<double>(0,0);
			}else{
				_Eamp.Et() = CPLX<double>(0,0); // TE(H)
				_Eamp.Ep() = CPLX<double>(1,0);
			}
		}
		string _StrUppercase(const string& in){
			string out = in;
			transform(out.begin(), out.end(), out.begin(), ::toupper);
			return out;
		}
		
		string _StrLowercase(const string& in){
			string out = in;
			transform(out.begin(), out.end(), out.begin(), ::tolower);
			return out;
		}
	private:
		string _Pol;	// Total polarization name
		string _TxPol;	// Transmitted polarization
		string _RxPol;	// Recivied polarization
		TAYLOR _Taylor;	// Taylor approximation
		EAmp _Eamp;		// E-field amplitude
		long _MaxLevel;	// Max bouncing number
	};
	
	//+===========================+
	//|           SAR             |
	//+===========================+
	class SAR{
	public:
		SAR(){
//			// TerraSAR-X
//			_Sen_name = "TerraSAR-X";
//			_theta_l_min = 27.1430492401123047*DTR;// [rad] (min)look angle
//			_theta_l_max = 29.7847690582275391*DTR;// [rad] (max)look angle
//			_theta_l_MB  = (_theta_l_min+_theta_l_max)/2;// [rad] (Main beam)look angle
//			_theta_sq = 0*DTR;					// [rad] SAR squint angle @ beam center
//			_f0 = 9.65e9;						// [Hz] Transmitted frequency
//			_PRF = 3798.60785398230109;			// [Hz] Pulse repeat frequency
//			_Fr = 1.64829192e8;					// [Hz] ADC sampling rate
//			_DC = 16.5;							// [%] Transmit duty cycle
//			_BWrg = 150e6;						// [Hz] bandwidth @ range
//			_SWrg = 50e3;						// [m] Swath width
//			_Laz = 4.784;						// [m] antenna size @ azimuth
//			_Lev = 0.704;						// [m] antenna size @ elevation
//			_ant_eff = 1;						// [x] antenna effective coefficient
//			_Nr = 500;							// [sample] Number of slant range sample
//			_Na = 500;							// [sample] Number of Azimuth sample
//			_Raz = vector<double>(2);			// [m] Start & End point @ azimuth
//			_Raz[0] = -100;
//			_Raz[1] = 100;
//			// others
//			_gain_rg = 9.6516148187220e-4;		// [x] Range gain
//			_Ls=0.;								// [m] Synthetic aperture radar antenna size
//			_lambda=c/_f0;						// [m] Transmitted wavelength
//			_Tr=_DC/100.0/_PRF;					// [sec] duration time
//			_theta_az=0.886*_lambda/_La;		// [rad] beamwidth at azimuth direction
//			_theta_rg=0.886*_lambda/_Lr;		// [rad] beamwidth at slant range direction
//			_Kr=_BW_rg/_Tr;						// [Hz/sec] Chirp rate
			
			// Test
			_Sen_name = "Test";
			_theta_l_min = 27.1430492401123047*DTR;// [rad] (min)look angle
			_theta_l_max = 29.7847690582275391*DTR;// [rad] (max)look angle
			_theta_l_MB  = (_theta_l_min+_theta_l_max)/2;// [rad] (Main beam)look angle
			_theta_sqc = 0*DTR;					// [rad] SAR squint angle @ beam center
			_f0 = 5e9;							// [Hz] Transmitted frequency
			_PRF = 1700;						// [Hz] Pulse repeat frequency
			_Fr = 10e6;							// [Hz] ADC sampling rate
			_DC = 6.8;							// [%] Transmit duty cycle
			_BWrg = 20e6;						// [Hz] bandwidth @ range
			_SWrg = 100;						// [m] Swath width
			_Laz = 10;							// [m] antenna size @ azimuth
			_Lev = 1.5;							// [m] antenna size @ elevation
			_ant_eff = 1;						// [x] antenna effective coefficient
			_Nr = 45;							// [sample] Number of slant range sample
			_Na = 101;							// [sample] Number of Azimuth sample
//			_Raz = vector<double>(2);			// [m] Start & End point @ azimuth
//			_Raz[0] = -300;
//			_Raz[1] = 300;
			// others
			_gain_rg = 1;						// [x] Range gain
			_Ls=0.;								// [m] Synthetic aperture radar antenna size
			Calculate();
		}
		SAR(const string Sen_name, const double theta_l_MB, const double theta_sqc, const double f0, const double PRF, const double Fr, const double DC,
			const double BWrg, const double SWrg, const double Laz, const double Lev, const double ant_eff, const long Nr, const long Na, const bool isUpChirp){
			_Sen_name = Sen_name;
			_theta_l_MB  = theta_l_MB;
			_theta_sqc = theta_sqc;
			_f0 = f0;
			_PRF = PRF;
			_Fr = Fr;
			_DC = DC;
			_BWrg = BWrg;
			_SWrg = SWrg;
			_Laz = Laz;
			_Lev = Lev;
			_ant_eff = ant_eff;
			_Nr = Nr;
			_Na = Na;
			_isUpChirp = isUpChirp;
//			_Raz = Raz;
			// Calculate
			Calculate();
		}
		string& Sen_name(){ return _Sen_name; }
		// const
		const double& theta_l_min() const{ return _theta_l_min; }
		const double& theta_l_max() const{ return _theta_l_max; }
		const double& theta_l_MB() const{ return _theta_l_MB; }
		const double& theta_sqc() const{ return _theta_sqc; }
		const double& f0() const{ return _f0; }
		const double& lambda() const{ return _lambda; }
		const double& k0() const{ return _k0; }
		const double& PRF() const{ return _PRF; }
		const double& Fr() const{ return _Fr; }
		const double& DC() const{ return _DC; }
		const double& BWrg() const{ return _BWrg; }
		const double& SWrg() const{ return _SWrg; }
		const double& Laz() const{ return _Laz; }
		const double& Lev() const{ return _Lev; }
		const double& ant_eff() const{ return _ant_eff; }
		const long& Nr() const{ return _Nr; }
		const long& Na() const{ return _Na; }
		const double& gain_rg() const{ return _gain_rg; }
		const double& Ls() const{ return _Ls; }
		const double& Tr() const{ return _Tr; }
		const double& theta_az() const{ return _theta_az; }
		const double& theta_rg() const{ return _theta_rg; }
		const double& Kr() const{ return _Kr; }
		const bool& isUpChirp() const{ return _isUpChirp; }
		// be able to be assign
//		double& theta_l_min(){ return _theta_l_min; }
//		double& theta_l_max(){ return _theta_l_max; }
//		double& theta_l_MB(){ return _theta_l_MB; }
		double& theta_sqc(){ return _theta_sqc; }
		double& f0(){ return _f0; }
		double& lambda(){ return _lambda; }
		double& k0(){ return _k0; }
		double& PRF(){ return _PRF; }
		double& Fr(){ return _Fr; }
		double& DC(){ return _DC; }
		double& BWrg(){ return _BWrg; }
		double& SWrg(){ return _SWrg; }
		double& Laz(){ return _Laz; }
//		double& Lev(){ return _Lev; }
		double& ant_eff(){ return _ant_eff; }
		long& Nr(){ return _Nr; }
		long& Na(){ return _Na; }
		double& gain_rg(){ return _gain_rg; }
		double& Ls(){ return _Ls; }
		double& Tr(){ return _Tr; }
		double& theta_az(){ return _theta_az; }
		double& theta_rg(){ return _theta_rg; }
		double& Kr(){ return _Kr; }
		bool& isUpChirp(){ return _isUpChirp; }
		void Print(){
			cout<<"+-------------------------------------+"<<endl;
			cout<<"|            def::SAR class           |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(10);
			cout<<"Sen_name          = "<<_Sen_name<<endl;
			cout<<"theta_l_min [deg] = "<<_theta_l_min*RTD<<endl;
			cout<<"theta_l_MB  [deg] = "<<_theta_l_MB*RTD<<endl;
			cout<<"theta_l_max [deg] = "<<_theta_l_max*RTD<<endl;
			cout<<"theta_sqc   [deg] = "<<_theta_sqc*RTD<<endl;
			cout<<"f0          [GHz] = "<<_f0/1E9<<endl;
			cout<<"PRF          [Hz] = "<<_PRF<<endl;
			cout<<"Fr          [MHz] = "<<_Fr/1E6<<endl;
			cout<<"DC            [%] = "<<_DC<<endl;
			cout<<"BWrg        [MHz] = "<<_BWrg/1E6<<endl;
			cout<<"SWrg          [m] = "<<_SWrg<<endl;
			cout<<"Laz           [m] = "<<_Laz<<endl;
			cout<<"Lev           [m] = "<<_Lev<<endl;
			cout<<"ant_eff       [x] = "<<_ant_eff<<endl;
			cout<<"theta_az    [deg] = "<<theta_az()*def::RTD<<endl;
			cout<<"theta_rg    [deg] = "<<theta_rg()*def::RTD<<endl;
			cout<<"Nr      [samples] = "<<_Nr<<endl;
			cout<<"Na      [samples] = "<<_Na<<endl;
//			cout<<"Raz (s,e)     [m] = "<<_Raz[0]<<","<<_Raz[1]<<endl;
			cout<<"gain_rg       [x] = "<<_gain_rg<<endl;
			cout<<"Ls            [m] = "<<_Ls<<endl;
			cout<<"Lambda        [m] = "<<_lambda<<endl;
			cout<<"Tr          [sec] = "<<_Tr<<endl;
			cout<<"theta_az    [deg] = "<<_theta_az*RTD<<endl;
			cout<<"theta_rg    [deg] = "<<_theta_rg*RTD<<endl;
			cout<<"Kr       [Hz/sec] = "<<_Kr<<endl;
		}
		// Set
		void Setf0(const double f0){
			_f0 = f0;
			Calculate();
		}
		void SetTheta_l_MB(const double Theta_l_MB){
			_theta_l_MB = Theta_l_MB;
			double theta_rg = 0.886*_lambda/_Lev;
			_theta_l_min = _theta_l_MB - theta_rg/2.;
			_theta_l_max = _theta_l_MB + theta_rg/2.;
		}
		void SetLev(const double Lev){
			_Lev = Lev;
			double theta_rg = 0.886*_lambda/_Lev;
			_theta_l_min = _theta_l_MB - theta_rg/2.;
			_theta_l_max = _theta_l_MB + theta_rg/2.;
		}
	private:
		void Calculate(){
			_lambda=def::C / _f0;				// [m] Transmitted wavelength
			_k0 = 2*def::PI / _lambda;
			double theta_rg = 0.886*_lambda/_Lev;
			_theta_l_min = _theta_l_MB - theta_rg/2.;
			_theta_l_max = _theta_l_MB + theta_rg/2.;
			_Tr=_DC/100.0/_PRF;							// [sec] duration time
			_theta_az=0.886*_lambda/_Laz/_ant_eff;		// [rad] beamwidth at azimuth direction
			_theta_rg=0.886*_lambda/_Lev/_ant_eff;		// [rad] beamwidth at slant range direction
			if(_isUpChirp == true){
				_Kr = _BWrg/_Tr;						// [Hz/sec] Chirp rate (Up)
			}else{
				_Kr = -_BWrg/_Tr;						// [Hz/sec] Chirp rate (Down)
			}
		}
	private:
		string _Sen_name;	// Sensor name
		double _theta_l_min;// [rad] (min)look angle
		double _theta_l_max;// [rad] (max)look angle
		double _theta_l_MB;	// [rad] (Main beam)look angle
		double _theta_sqc;	// [rad] SAR squint angle @ beam center

		double _f0;			// [Hz] Transmitted frequency
		double _lambda;		// [m] wavelength
		double _k0;			// [1/m] wavenumber
		double _PRF;		// [Hz] Pulse repeat frequency
		double _Fr;			// [Hz] ADC sampling rate
		double _DC;			// [%] Transmit duty cycle
		double _BWrg;		// [Hz] bandwidth @ range
		// Add
		double _SWrg;		// [m] Slant Range Swath width
		bool _isUpChirp;	// [x] Up(true) / Down(false)-chirp
		
		double _Laz;		// [m] antenna size @ azimuth
		double _Lev;		// [m] antenna size @ elevation
		// Add
		double _ant_eff;	// [x] Antenna effect coefficient
		long _Nr;			// [samples] Slant range samples
		long _Na;			// [samples] Azimuth samples
//		vector<double> _Raz;// [m] Start & End point @ azimuth
		// other
		double _gain_rg;	// [x] Range gain
		double _Ls;			// [m] Synthetic aperture radar antenna size
		double _Tr;			// [sec] duration time
		double _theta_az;	// [rad] beamwidth at azimuth direction
		double _theta_rg;	// [rad] beamwidth at slant range direction
		double _Kr;			// [Hz/sec] Chirp rate

	};
	
	//+===========================+
	//|     Multiangle flag       |
	//+===========================+
	class MultiAngle{
	public:
		MultiAngle():_IsMulLook(false),_LookFrom(0),_LookTo(0),_LookStep(0),_IsMulAsp(false),_AspFrom(0),_AspTo(0),_AspStep(0){};
		MultiAngle(const bool IsMulLook, const double LookFrom, const double LookTo, const double LookStep, const bool IsMulAsp, const double AspFrom, const double AspTo, const double AspStep){
			_IsMulLook = IsMulLook;
			_LookFrom  = LookFrom * DTR;
			_LookTo    = LookTo * DTR;
			_LookStep  = LookStep * DTR;
			_IsMulAsp  = IsMulAsp;
			_AspFrom   = AspFrom * DTR;
			_AspTo     = AspTo * DTR;
			_AspStep   = AspStep * DTR;
		}
		// const
		const bool& IsMulLook() const{ return _IsMulLook; }
		const double& LookFrom() const{ return _LookFrom; }
		const double& LookTo() const{ return _LookTo; }
		const double& LookStep() const{ return _LookStep; }
		const bool& IsMulAsp() const{ return _IsMulAsp; }
		const double& AspFrom() const{ return _AspFrom; }
		const double& AspTo() const{ return _AspTo; }
		const double& AspStep() const{ return _AspStep; }
		vector<double> GetThetaSeries(){
			vector<double> out;
			if(IsMulLook()){
				double max_val = LookTo();
				double min_val = LookFrom();
				double step    = LookStep();
				double epsilon = 1E-4;
				long num = long((max_val - min_val)/step + 1L + epsilon);
				// long num = long((max_val - min_val)/step) + 1L;
				out = vector<double>(num);
				for(long i=0;i<num;i++){
					out[i] = min_val+((double)i*step);
				}
			}else{
				out = vector<double>(1);
				out[0] = LookFrom();
			}
			return out;
		}
		vector<double> GetPhiSeries(){
			vector<double> out;
			if(IsMulAsp()){
				double max_val = AspTo();
				double min_val = AspFrom();
				double step    = AspStep();
				double epsilon = 1E-4;
				long num = long((max_val - min_val)/step + 1L + epsilon);
				// long num = long((max_val - min_val)/step) + 1L;
				out = vector<double>(num);
				for(long i=0;i<num;i++){
					out[i] = min_val+((double)i*step);
				}
			}else{
				out = vector<double>(1);
				out[0] = AspFrom();
			}
			return out;
		}
		// be able to assign
		bool& IsMulLook(){ return _IsMulLook; }
		double& LookFrom(){ return _LookFrom; }
		double& LookTo(){ return _LookTo; }
		double& LookStep(){ return _LookStep; }
		bool& IsMulAsp(){ return _IsMulAsp; }
		double& AspFrom(){ return _AspFrom; }
		double& AspTo(){ return _AspTo; }
		double& AspStep(){ return _AspStep; }
		void Print(){
			string flagIsMulLook = (_IsMulLook)? "Yes":"No";
			string flagIsMulAsp  = (_IsMulAsp)?  "Yes":"No";
			cout<<"+-------------------------------------+"<<endl;
			cout<<"|        def::MultiAngle class        |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(8);
			cout<<"IsMulLook       = "<<flagIsMulLook<<endl;
			cout<<"Look_from [deg] = "<<_LookFrom*RTD<<endl;
			cout<<"Look_to   [deg] = "<<_LookTo*RTD<<endl;
			cout<<"Look_step [deg] = "<<_LookStep*RTD<<endl;
			cout<<"IsMulAsp        = "<<flagIsMulAsp<<endl;
			cout<<"Asp_from  [deg] = "<<_AspFrom*RTD<<endl;
			cout<<"Asp_to    [deg] = "<<_AspTo*RTD<<endl;
			cout<<"Asp_step  [deg] = "<<_AspStep*RTD<<endl;
		}
		void Print()const{
			string flagIsMulLook = (_IsMulLook)? "Yes":"No";
			string flagIsMulAsp  = (_IsMulAsp)?  "Yes":"No";
			cout<<"+-------------------------------------+"<<endl;
			cout<<"|        def::MultiAngle class        |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(8);
			cout<<"IsMulLook    = "<<flagIsMulLook<<endl;
			cout<<"Look_from    = "<<_LookFrom*RTD<<endl;
			cout<<"Look_to      = "<<_LookTo*RTD<<endl;
			cout<<"Look_step    = "<<_LookStep*RTD<<endl;
			cout<<"IsMulAsp     = "<<flagIsMulAsp<<endl;
			cout<<"Asp_from     = "<<_AspFrom*RTD<<endl;
			cout<<"Asp_to       = "<<_AspTo*RTD<<endl;
			cout<<"Asp_step     = "<<_AspStep*RTD<<endl;
		}
	private:
		bool _IsMulLook;	//[x] Multilook or not?
		double _LookFrom;	//[rad] Look angle from...
		double _LookTo;		//[rad] Look angle to...
		double _LookStep;	//[rad] Multi Look angle interval
		bool _IsMulAsp;		//[x] Multi aspect angle or not?
		double _AspFrom;	//[rad] Aspect angle from...
		double _AspTo;		//[rad] Aspect angle to...
		double _AspStep;	//[rad] Multi Aspect angle interval
	};
	
	class ANG {
	public:
		ANG():_num(0){};
		ANG(const size_t num){
			_num  = num;
			_look = vector<double>(_num);
			_sq   = vector<double>(_num);
			_asp  = vector<double>(_num);
		}
		ANG(const ANG& in){
			_num  = in._num;
			_look = in._look;
			_sq   = in._sq;
			_asp  = in._asp;
		}
		// Getter & setter
		size_t GetNum()const{return _num;};
		vector<double> Look()const{return _look;};
		vector<double> Squint()const{return _sq;};
		vector<double> Asp()const{return _asp;};
		vector<double>& Look(){return _look;};
		vector<double>& Squint(){return _sq;};
		vector<double>& Asp(){return _asp;};
		// Misc.
		void PrintListAll(){
			cout << "+------------------------------------------+" << endl;
			cout << "|                ANG Class                 |" << endl;
			cout << "+------------------------------------------+" << endl;
			cout << "| Look [deg]  Squint [deg]  Aspect [deg]   |" << endl;
			cout << "+------------------------------------------+" << endl;
			for (size_t i = 0; i < _num; ++i) {
				cout << std::fixed << std::setw(10) << _look[i]*def::RTD << "       " << _sq[i]*def::RTD <<
					 "       " << _asp[i]*def::RTD;
				if (i == (_num - 1) / 2) {
					cout << "  (center)" << endl;
				} else {
					cout << endl;
				}
			}
		}
		void Print(){
			cout<<"+------------------------------------------+"<<endl;
			cout<<"|                ANG Class                 |"<<endl;
			cout<<"+------------------------------------------+"<<endl;
			cout<<"Number = "<<_num<<endl;
			cout<<"Note: [0, center, end]"<<endl;
			double theta_l_min   = def::RTD*_look[0];
			double theta_l_c     = def::RTD*_look[(_num-1)/2];
			double theta_l_max   = def::RTD*_look[_num-1];
			double theta_sq_min  = def::RTD*_sq[0];
			double theta_sq_c    = def::RTD*_sq[(_num-1)/2];
			double theta_sq_max  = def::RTD*_sq[_num-1];
			double theta_asp_min = def::RTD*_asp[0];
			double theta_asp_c   = def::RTD*_asp[(_num-1)/2];
			double theta_asp_max = def::RTD*_asp[_num-1];
			printf("theta_l   = [%f,%f,%f]\n", theta_l_min,   theta_l_c,   theta_l_max);
			printf("theta_sq  = [%f,%f,%f]\n", theta_sq_min,  theta_sq_c,  theta_sq_max);
			printf("theta_asp = [%f,%f,%f]\n", theta_asp_min, theta_asp_c, theta_asp_max);
		}
	private:
		size_t _num;
		vector<double> _look, _sq, _asp;
	};

	//+===========================+
	//|  Incident Mesh parameter  |
	//+===========================+
	// Incident sphere surface definition
	class MeshDef {
	public:
		MeshDef():_Scale(1.0),_dRad(3.0){};
		MeshDef(const double Scale, const double dRad):_Scale(Scale),_dRad(dRad){};
		const double& Scale() const{ return _Scale; }
		const double& dRad() const{ return _dRad; }
		double& Scale(){ return _Scale; }
		double& dRad(){ return _dRad; }
		// misc
		void Print(){
			cout<<"+-------------------------------------+"<<endl;
			cout<<"|          def::MeshDef class         |"<<endl;
			cout<<"+-------------------------------------+"<<endl;
			cout<<std::setprecision(8);
			cout<<"Scale    = "<<_Scale<<endl;
			cout<<"dRad     = "<<_dRad<<endl;
		}
	private:
		double _Scale;		//[x] Incident Mesh number scale, recommand >= 1.0
		double _dRad;		//[m] Distance from target center to sphere surface
	};

	
}



#endif
