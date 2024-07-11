//
//  Created by Steve Chiang on 5/9/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//
#define NORMALH

#ifdef DEBUG
#undef DEBUG
#endif



#include <sar/sar.h>
#include <bvh/obj.h>
#include <bvh/ray.h>
#include <basic/WFds.h>
#include <basic/new_dir.h>
#include <string>


using namespace std;
using namespace vec;
using namespace sv;
using namespace new_dir;
using namespace def_prefunc;
using namespace WFds;


double SARCSAFindMigrationParameter(const double fn, const double Vr, const double lambda) {
	// Find Migration parameter D (7.17)
	double D = sqrt(1 - (fn*fn * lambda*lambda) / (4 * Vr*Vr));

	return D;
}

double SARCSAFindModifiedChirpRate(const double fn, const double Vr, const double Kr, const double R0, const double c, const double f0){
	double D = SARCSAFindMigrationParameter(fn, Vr, c/f0);
	double lambda = c/f0;
	double Km = 1.0/Kr - 2.0*lambda*R0*(D*D - 1.0)/(c*c*D*D*D);
	Km = 1.0/Km;

	return Km;
}

double SARCSAFindRangeScalingFactor(const double fn, const double Vrref, const double lambda, double& a_fa, const double ALPHA=1) {
	a_fa = 1 / SARCSAFindMigrationParameter(fn, Vrref, lambda) - 1;
	double a_scl = a_fa + (1 - ALPHA) * (1 + a_fa) / ALPHA;

	return a_scl;
}

double rect(const double val){
	double out = val * 0.0;
	if(abs(val) <= 0.5){
		out = 1.0;
	}
	return out;
}


void WFEmulatorSARFocusing(const string dir, const Meta& meta, const string Rx_POL, const bool ShowSummary=false){

	// Export data
	string file_PROC_par = dir + "FOCUS_PAR_"+Rx_POL+".meta";
	string file_Sac      = dir + "EXP_FOCUS_SAR_"+Rx_POL+".bin";
	string file_Sac_hdr  = dir + "EXP_FOCUS_SAR_"+Rx_POL+".hdr";
	string file_R0       = dir + "FOCUS_R0.bin";
	string file_Ra       = dir + "FOCUS_Ra.bin";
	// AUX data (for debug)
	string file_RCM      = dir + "FOCUS_RCM_"+Rx_POL+".bin";
	string file_Srcmc    = dir + "FOCUS_Srcmc_"+Rx_POL+".bin";
	string file_Ssrc     = dir + "FOCUS_Ssrc_"+Rx_POL+".bin";
	string file_S0       = dir + "FOCUS_S0_"+Rx_POL+".bin";
	string file_Src      = dir + "FOCUS_Src_"+Rx_POL+".bin";
	string file_Srd      = dir + "FOCUS_Srd_"+Rx_POL+".bin";
	string file_Srd2     = dir + "FOCUS_Srd2_"+Rx_POL+".bin";


	//+--------------------------------------------------+
	//|               Read State Vector                  |
	//+--------------------------------------------------+
	SV<double> sv = ReadWFEmulatorStateVector(dir+"EXP_StateVector.txt");//, true);

	//+--------------------------------------------------+
	//|              Read Target location                |
	//+--------------------------------------------------+
	VEC<double> tar_pos = ReadWFEmulatorTargetLocation(dir+"EXP_Target.txt");
	tar_pos.Print();

	//+--------------------------------------------------+
	//|            [Debug] Read freq_nfreq.txt           |
	//+--------------------------------------------------+
	D1<double> Freq, nFreq;
	ReadWFEmulatorFreq(dir+"EXP_Freq.txt", Freq, nFreq);


	//+--------------------------------------------------+
	//|              Read SAR echo signal                |
	//+--------------------------------------------------+
	string EXPORT_H = dir + "EXP_H"+meta.tx_pol()+".dat";
	string EXPORT_V = dir + "EXP_V"+meta.tx_pol()+".dat";

	string file_s0 = EXPORT_H;
	if(Rx_POL == "V"){
		file_s0 = EXPORT_V;
	}
	D1<float> Coef, Dist, Dopp, AzPat, PhsSingleSin, PhsSingleCos;
	D2<CPLX<double> > s0 = ReadWFEmulatorSAREcho(dir+"EXP.meta", file_s0, Coef, Dist, Dopp, AzPat, PhsSingleSin, PhsSingleCos);

//	// Remove (temporarily)
//	for(size_t j=0;j<s0.GetM();++j){
//		for(size_t i=0;i<s0.GetN();++i){
//			if( isnan(s0[j][i].abs()) ){
//				s0[j][i] = 0;
//			}
//		}
//	}


//	//+-------------------------------------------------------------------------------------+
//	//|                                 Range center shift                                  |
//	//+-------------------------------------------------------------------------------------+
//	cout<<"+----------------------------------------+"<<endl;
//	cout<<"|  Range Center shift: Shift to Rc_slant |"<<endl;
//	cout<<"+----------------------------------------+"<<endl;
//
//	D1<double> freq(meta.Nr());
//	linspace(meta.f0()-meta.BWrg()/2, meta.f0()+meta.BWrg()/2, freq);
//
//	// Srcmc_rd in Range-Doppler domain
//	for (size_t i = 0; i < meta.Na(); ++i) {
//		for (size_t j = 0; j < freq.GetNum(); ++j) {
//			// Add slant range (two-way)
//			s0[i][j].AddDistancePhase( freq[j], -2 * meta.Rc_slant() * cos(meta.theta_sq_mean()) );
//		}
//	}


	//+-------------------------------------------------------------------------------------+
	//|                                      Make SAR                                       |
	//+-------------------------------------------------------------------------------------+
	double sp_rg = def::C/(2*meta.Fr());
	cout<<"+----------------------------------------+"<<endl;
	cout<<"|            Range Compression           |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	D2<CPLX<double> > Src = s0;
	// Add range compression filter
	D1<double> win = sar::find::KaiserWindow(meta.Fr(), meta.Nr(), 2.1);
	mat::fftshift(win);
	for(size_t i=0;i<meta.Na();++i){
		for(size_t j=0;j<meta.Nr();++j) {
			Src[i][j] = Src[i][j] * win[j];
		}
	}

	IFFTX(Src);
	fftshiftx(Src);

	s0.WriteBinary(file_S0.c_str());
	Src.WriteBinary(file_Src.c_str());

	cout<<"+----------------------------------------+"<<endl;
	cout<<"|              Slant range               |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	// Slant range series
	D1<double> Rc_range(meta.Nr());
	linspace(meta.Rn_slant(), meta.Rf_slant(), Rc_range);

	// Slant range series at Zero Doppler
	D1<double> R0_range(meta.Nr());
	for(size_t i=0;i<R0_range.GetNum();++i){
		R0_range[i] = Rc_range[i] * cos(meta.theta_sq_mean());
	}

	cout<<"meta.Rn_slant()="<<Rc_range[0]<<" [m]"<<endl;
	cout<<"meta.Rc_slant()="<<mean(Rc_range)<<" [m]"<<endl;
	cout<<"meta.Rf_slant()="<<Rc_range[Rc_range.GetNum()-1]<<" [m]"<<endl;
	cout<<"meta.R0n_slant()="<<R0_range[0]<<" [m]"<<endl;
	cout<<"meta.R0c_slant()="<<mean(R0_range)<<" [m]"<<endl;
	cout<<"meta.R0f_slant()="<<R0_range[R0_range.GetNum()-1]<<" [m]"<<endl;


	cout<<"+----------------------------------------+"<<endl;
	cout<<"|           Effective velocity           |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	D1<double> vel(meta.Na()-1);

	for(size_t i=1;i<meta.Na();++i){
		double dt = sv.t()[i] - sv.t()[i-1];
		double dis = (sv.pos()[i] - sv.pos()[i-1]).abs();
		vel[i-1] = dis/dt;
	}

	double Vr = mean(vel);
	Vr = meta.Vs_mean();
//	Vr = 1020;
//	double Vr = mean(vel) / mat::Square( cos(meta.theta_sq_mean()) );
//	Vr = 4073;
//	Vr = 3750;
	cout<<"meta.Vs_mean() = "<<meta.Vs_mean()<<endl;
	cout<<"Vr = "<<Vr<<" [m/s]"<<endl;
	double sp_az = Vr/(2*meta.PRF());


	cout<<"+----------------------------------------+"<<endl;
	cout<<"|           Doppler estimation           |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	// double theta_sqc = deg2rad(0);
	double theta_sqc = meta.theta_sq_mean();
	double lambda = def::C/meta.f0();
	double fdc_ideal = 2*Vr*sin(theta_sqc)/lambda;

	double Tr = 1e-6;
	double Kr = meta.BWrg() / Tr;
	double fdc_est = sar::find::AbsolutedCentroidFreq::MLCC(Src, meta.Fr(), meta.f0(), meta.PRF(), Kr, Tr, true);

	double theta_sqc_est = asin(fdc_est*lambda/(2.0*Vr));

	cout<<"fdc_ideal       = "<<fdc_ideal<<endl;
	cout<<"fdc_est         = "<<fdc_est<<endl;
	cout<<"theta_sqc_ideal = "<<rad2deg(theta_sqc)<<" [deg]"<<endl;
	cout<<"theta_sqc_est   = "<<rad2deg(theta_sqc_est)<<" [deg]"<<endl;


	double fdc = fdc_ideal;
	double fdc_shift;
	D1<double> fn(meta.Na());
	sar::find::InsFreqOfMatchedFilter(fdc, meta.PRF(), fn, fdc_shift);
	cout<<"fdc_shift = "<<fdc_shift<<endl;


	cout<<"+----------------------------------------+"<<endl;
	cout<<"|      Transform to Range-Doppler        |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	D2<CPLX<double> > Srd = Src;
	FFTY(Srd);
	Srd.WriteBinary(file_Srd.c_str());

	cout<<"+----------------------------------------+"<<endl;
	cout<<"|      Secondary Range Compression       |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	D1<double> ft(meta.Nr());
	double shift_ft;
	sar::find::InsFreqOfMatchedFilter(0., meta.Fr(), ft, shift_ft);


	// Transform to 2D frequency domain
	D2<CPLX<double> > Sf2 = Srd;
	FFTX(Sf2);

	// Using Option 2 for SRC
	for(size_t i=0;i<meta.Na();++i){
		double D = sqrt( 1.0 - Square(lambda*fn[i])/(4.0*Square(Vr)) );		// (6.24) (P.251)
		for(size_t j=0;j<meta.Nr();++j){
			double Ksrc = 2.0*Square(Vr)*Cubic(meta.f0())*Cubic(D)/(def::C*R0_range[j]*Square(fn[i]));	// (6.22)
			CPLX<double> Hsrc = mat::exp(CPLX<double>(0,-def::PI*Square(ft[j])/Ksrc));

			Sf2[i][j] = Sf2[i][j] * Hsrc;
		}
	}

	// Transform to Range-Doppler domain
	D2<CPLX<double> > Srd2 = Sf2;
	IFFTX(Srd2);
	Srd2.WriteBinary(file_Srd2.c_str());

	D2<CPLX<double> > Ssrc = Srd2;
	IFFTY(Ssrc);
	Ssrc.WriteBinary(file_Ssrc.c_str());


	cout<<"+----------------------------------------+"<<endl;
	cout<<"|                  RCMC                  |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	D1<double> Vr_range(meta.Nr());
	for(size_t j=0;j<meta.Nr();++j){
		Vr_range[j] = Vr;
	}

	// Make series along azimuth
	D1<double> D(meta.Na());
	for(size_t i=0;i<meta.Na();++i){
		D[i] = sqrt( 1.0 - Square(lambda*fn[i])/(4.*Square(Vr)) );	// (6.24)
	}

	// Make RCM 2D-array
	D2<double> RCM(meta.Na(), meta.Nr());
	for(size_t i=0;i<meta.Na();++i){
		for(size_t j=0;j<meta.Nr();++j){
			RCM[i][j] = -R0_range[j] * ((1.-D[i])/D[i]) / sp_rg;	// (6.25)
			// RCM[i][j] -= (R0_range[j] / sp_rg);
		}
	}

	// Remove to center of range
	double RCM_bias = mat::mean(RCM);
//	RCM_bias = -777.8;
//	double RCM_bias = -R0_range[meta.Nr()/2] * ((1.-D[meta.Na()/2])/D[meta.Na()/2]) / (sp_rg);


	for(size_t i=0;i<meta.Na();++i){
		for(size_t j=0;j<meta.Nr();++j){
			RCM[i][j] = RCM[i][j] - RCM_bias;
		}
	}
//	printf("RCM_bias = %.4f [sample] = %.4f [m], cf = %.4f\n", RCM_bias, RCM_bias * sp_rg, mat::mean(RCM));
	printf("mean(RCM) = %.4f, round(mean(RCM)) = %ld\n", RCM_bias, long(RCM_bias) % meta.Nr());
	RCM.WriteBinary(file_RCM.c_str());

//	cout<<"+----------------------------------------+"<<endl;
//	cout<<"|   Apply RCMC (Sinc interpolation)      |"<<endl;
//	cout<<"+----------------------------------------+"<<endl;
////	D2<CPLX<double> > Srcmc_rd = sar::RCMC(Srd2, RCM, true);
//	D2<CPLX<double> > Srcmc_rd = sar::RCMCSinc(Srd2, RCM, true);
////	D2<CPLX<double> > Srcmc_rd = sar::RCMCSinc(Srd, RCM, true);
//	Srcmc_rd.WriteBinary(file_Srcmc.c_str());


	cout<<"+----------------------------------------+"<<endl;
	cout<<"|       Apply RCMC (ChirpScaling)        |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	// (0) Prepare
	double Vrref = meta.Vs_mean();
	double Rref  = meta.Rc_slant()*cos(meta.theta_sq_mean());
	// (1) Transfer to 2-freq domain
	D2<CPLX<double> > S2f = Srd2;
	mat::FFTX(S2f);
	// (2) Make range filter
	D1<double> Kw_rg = sar::find::KaiserWindow(meta.Fr(), meta.Nr(), 2.1);
	// (3) Scaling
	D1<CPLX<double> > H2(meta.Nr());
	for(size_t i=0;i<meta.Na();++i){
		double Km = SARCSAFindModifiedChirpRate(fn[i], Vrref, Kr, Rref, def::C, meta.f0()); // Km
		// 1/Km = 1/Kr - 1/Ksrc
		double Ksrc = 1.0/(1.0/Km - 1.0/Kr);	// Ksrc ONLY

		double a_fa;
		double a_scl = SARCSAFindRangeScalingFactor(fn[i], Vrref, lambda, a_fa);

//		SARFindInsFreqOfMatchedFilter(0, meta.Fr(), meta.Nr());
		double d_shift;
		ft = sar::find::SARFindInsFreqOfMatchedFilter(0.0, meta.Fr(), meta.Nr(), d_shift);

		for(size_t j=0;j<meta.Nr();++j) {
			ft[j] = rect(ft[j] / (abs(Kr) * meta.Tr() * 0.8)) * ft[j];
		}

		// Bulk RCM
		for(size_t j=0;j<meta.Nr();++j) {
			H2[j] = mat::exp(CPLX<double>(0.0, 4.0 * pi * Rref / def::C * a_fa * ft[j]));
		}

		for(size_t j=0;j<meta.Nr();++j) {
			S2f[i][j] = Kw_rg[j] * S2f[i][j] * H2[j];
		}
	}

	D2<CPLX<double> > Srcmc_rd = S2f;
	mat::IFFTX(Srcmc_rd);

	// Shift to center of range
	double swath = sp_rg * double(meta.Nr());
//	long Rc_bais_pix = round( mat::Mod(meta.Rc_slant(), swath) / sp_rg );
//	Rc_bais_pix = round(-Rc_bais_pix + (swath/sp_rg));
	long Rc_bais_pix = round( mat::Mod(RCM_bias, double(meta.Nr())));

//	cout<<"RCM_bias = "<<RCM_bias<<", meta.Nr() = "<<meta.Nr()<<endl;

	for(size_t i=0;i<meta.Na();++i) {
		D1<CPLX<double> > tmp_rg = Srcmc_rd.GetRow(i);
		mat::shift(tmp_rg, -Rc_bais_pix);
		Srcmc_rd.SetRow(tmp_rg, i);
	}


	Srcmc_rd.WriteBinary(file_Srcmc.c_str());



	cout<<"+----------------------------------------+"<<endl;
	cout<<"|           Azimuth focusing             |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	size_t num_rg = Srcmc_rd.GetN();
	size_t num_az = Srcmc_rd.GetM();
	D2<CPLX<double> > Sac(num_az, num_rg);

	// Make series along azimuth
	D = D1<double>(num_az);
	for(size_t i=0;i<num_az;++i){
		D[i] = sqrt( 1.0 - Square(lambda*fn[i])/(4.*Square(Vr)) );	// (6.24)
	}

	// make kaiser window (wk1)
	D1<double> wk1 = sar::find::KaiserWindow(meta.PRF(), num_az, 2.1);
	mat::fftshift(wk1);
	mat::shift( wk1, fdc_shift );
	// make kaiser window (wk2)
	D1<double> wk2 = sar::find::KaiserWindow(meta.PRF(), num_az, 8.9);
	mat::fftshift(wk2);
	mat::shift( wk2, fdc_shift );


	CPLX<double> Haz;


	for(size_t j=0;j<num_rg;++j){	// range
		for(size_t i=0;i<num_az;++i){
			Haz = wk1[i] * mat::exp( CPLX<double>(0, -def::PI4*R0_range[j]*D[i]*meta.f0()/def::C) );	// (6.26) (P.252)
			Sac[i][j] = Srcmc_rd[i][j] * wk2[i] * Haz.conj();
		}
	}


//	cout<<"+----------------------------------------+"<<endl;
//	cout<<"|  Range Center shift: Shift to Rc_slant |"<<endl;
//	cout<<"+----------------------------------------+"<<endl;
//	// (Range-Doppler --> two freq)
//	FFTX(Sac);
//	// (two freq --> Range freq-Azimuth)
//	IFFTY(Sac);
//
//	D1<double> freq(meta.Nr());
//	linspace(meta.f0()-meta.BWrg()/2, meta.f0()+meta.BWrg()/2, freq);
//
//	// Srcmc_rd in Range-Doppler domain
//	for (size_t i = 0; i < meta.Na(); ++i) {
//		for (size_t j = 0; j < freq.GetNum(); ++j) {
//			// Add slant range (two-way)
//			Sac[i][j].AddDistancePhase( freq[j], 2 * meta.Rc_slant() * cos(theta_sqc) );
//		}
//	}
//
//	// (Range freq-Azimuth --> two freq)
//	FFTY(Sac);
//	// (two freq --> Range-Doppler)
//	IFFTX(Sac);


	cout<<"+----------------------------------------+"<<endl;
	cout<<"|   Range-Doppler --> Two time domain    |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	IFFTY(Sac);

	cout<<"+----------------------------------------+"<<endl;
	cout<<"|  Azimuth Center shift: by Squint angle |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	for(size_t i=0;i<num_rg;++i){
		D1<CPLX<double> > tmp_col = Sac.GetColumn(i);
		// mat::shift(tmp_col, round(fdc_shift));
		Sac.SetColumn(tmp_col, i);
	}


	cout<<"+----------------------------------------+"<<endl;
	cout<<"|        Convert to single float         |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	D2<CPLX<float> > Sac_float(Sac.GetM(), Sac.GetN());
	for(size_t i=0;i<Sac_float.GetM();++i){
		for(size_t j=0;j<Sac_float.GetN();++j){
			Sac_float[i][j] = CPLX<float>( Sac[i][j].r(), Sac[i][j].i() );
		}
	}




	cout<<"+----------------------------------------+"<<endl;
	cout<<"|           Export Data & Meta           |"<<endl;
	cout<<"+----------------------------------------+"<<endl;
	//+-------------------+
	//| Process Parameter |
	//+-------------------+
	char procp_title[] = "WaveFidelity SAR Processor\0";   // no effective
	bool procp_SRC=false;               // effective
	bool procp_deskew=false;            // effective

	double time_near_range_raw = 2*R0_range[0]/def::C;
	double time_far_range_raw  = 2*R0_range[R0_range.GetNum()-1]/def::C;

	// Prepare the dominate parameters
	SAR Sar;
	Sar.PRF() = meta.PRF();
	Sar.Fr() = meta.Fr();
	Sar.Setf0( meta.f0() );
	Sar.BWrg() = meta.BWrg();
	Sar.Nr() = meta.Nr();
	Sar.Na() = meta.Na();
	ORB Orb;
	EF Ef;
	Ef.Pol() = Rx_POL + meta.tx_pol();
	VEC<double> Psc = sv.pos()[sv.GetNum()/2];

	// Initialize PROC_par
	par::PROC_PAR procp=sar::Init_PROC_par(procp_title, procp_SRC, procp_deskew,
										   Sar, Orb, Ef,
										   Psc, sv.t()[0], sv.pos()[0], sv,
										   time_near_range_raw, time_far_range_raw, Sar.Na());

	//+---------------------------+
	//| Find look angle region    |
	//+---------------------------+
	D1<double> theta_l_limit = sar::find::ThetaLookRegion(sv, R0_range, Orb);
	double R0_mean = mat::mean(R0_range);

	//+---------------------------+
	//| Update *.slc.par          |
	//+---------------------------+
	double Laz = sp_az*2;	// Pesudo effective azimuth antenna length
	// Doppler coefficient
	D1<double> Vg = sar::find::GroundVelocity(sv, theta_l_limit, Orb);
	double Vg_mean = (Vg[0] + Vg[1])/2.0;
	// Doppler rate
	double Ka = 2.0*Vr*Vr/(Sar.lambda()*R0_mean);
	// Target exposure time
	double Ta = 0.886*Sar.lambda()*R0_mean/(Laz*Vg_mean*cos(deg2rad(0.0)));
	// Doppler Bandwidth
	double dfd= abs(Ka) * Ta;
	// Azimuth resolution
	double azimres = 0.886*Vg_mean*cos(deg2rad(0.0))/dfd;



	procp.ran_res =	def::C/(2*Sar.BWrg());	// range resolution
	procp.azimres = azimres;				// azimuth resolution
	procp.rpixsp = def::C/(2*Sar.Fr());		// range pixel spacing
	procp.azimsp = Vr/meta.PRF();			// azimuth pixel spacing
	procp.nrs  = (int)Sar.Nr();				// range sample
	procp.nazs = (int)Sar.Na();				// azimuth sample
	procp.DAR_dop = fdc_est;				// DAR doppler centroid
	procp.fdp[0] = fdc_est;					// doppler centoid with polynomial fitting
	strcpy(procp.sec_range_mig, "ON\0");	// Enable secondary range migration correction

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

	double look_mean = mean(theta_l_limit);

	VEC<double> Ps, Ps1;
	Ps = sv.pos()[0];
	Ps1= sv.pos()[1];
	sar::GetMapCoordinate(Sar, Orb, Ps, Ps1,	look_mean, SW, Gsn, Gsc, Gsf);

	Ps = sv.pos()[(sv.GetNum()-1)/2];
	Ps1= sv.pos()[(sv.GetNum()-1)/2+1];
	sar::GetMapCoordinate(Sar, Orb, Ps, Ps1,	look_mean, SW, Gcn, Gcc, Gcf);

	Ps = sv.pos()[sv.GetNum()-1];
	Ps1= sv.pos()[sv.GetNum()-2];
	VEC<double> dir_flight_path = (sv.pos()[sv.GetNum()-1] - sv.pos()[sv.GetNum()-2]);
	double dis = (Ps - Ps1).abs();
	Ps1= Ps + dis * dir_flight_path;
	sar::GetMapCoordinate(Sar, Orb, Ps, Ps1,	look_mean, SW, Gen, Gec, Gef);

	// assign
	procp.map[0] = Gsn;
	procp.map[1] = Gsf;
	procp.map[2] = Gen;
	procp.map[3] = Gef;
	procp.map[4] = Gcc;

	//+-------------------+
	//| Process Parameter |
	//+-------------------+
	sar::write_PROC_par(file_PROC_par.c_str(), procp);

	//+-----------------+
	//| Azimuth sample  |
	//+-----------------+
	D1<double> Ra(Sar.Na());
	Ra.Indgen();
	Ra = (Ra - double(Ra.GetNum()-1)/2) * (Vr/Sar.PRF());


	//+-----------------+
	//|   Export Data   |
	//+-----------------+
	Sac_float.WriteBinary(file_Sac.c_str());
	Ra.WriteBinary(file_Ra.c_str());
	R0_range.WriteBinary(file_R0.c_str());
	// envi header
	envi::ENVIhdr hdr_H(Sac_float.GetN(), Sac_float.GetM(), 1, 0, Sac_float.GetType(), 0, "BIP");
	hdr_H.WriteENVIHeader(file_Sac_hdr.c_str());
}

void Usage(const string name){
	cout<<"+------------+"<<endl;
	cout<<"|    Usage   |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  "<<name<<" <dirRoot> <md5> <mission>"<<endl;
	cout<<"             [-h] [-SHOW]"<<endl;
}

void Help(){
	cout<<"+------------+"<<endl;
	cout<<"|  Required  |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  dirRoot            [string] WF Emulator workspace (e.g. /WFEmuRoot/)"<<endl;
	cout<<"  md5                [string] MD5 check sum (e.g. 9acf725f32881e7b2034b61a6c024fd3)"<<endl;
	cout<<"  mission            [string] Mission name (e.g. Mission3)"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"|  Optional  |"<<endl;
	cout<<"+------------+"<<endl;
	cout<<"  -h                 [x] Show the help menu"<<endl;
	cout<<"  -SHOW              [x] Display the summary messages on console"<<endl;
}




int main(int argc, char** argv) {

	cout<<"****************************************************"<<endl;
	cout<<"*              PASSED (WaveFidelity)               *"<<endl;
	cout<<"*--------------------------------------------------*"<<endl;
	cout<<"*        Copyright 2018, HomeStudio Taiwan         *"<<endl;
	cout<<"*               Created : 2010/09/06               *"<<endl;
	cout<<"*         Last Modified : 2022/03/10               *"<<endl;
	cout<<"*               Version : 3.5.10 (SAR)             *"<<endl;
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
	if(argc < 4){ Usage(argv[0]); errorexit(); }



//	// ifstream fin(filename.c_str());
//	Json::Value values;




	//+---------------------------+
	//| Read Input Parameters     |
	//+---------------------------+
	// required
	string dirRoot   = string(argv[1]);
	string md5       = string(argv[2]);
	string mission   = string(argv[3]);


//	string dirRoot = "/Users/cychiang/Documents/code/IdeaProjects/#WFEmuRoot/";
//	string md5     = "9acf725f32881e7b2034b61a6c024fd3";
//
//	string mission = "Mission2";	// T72, Scale=1.0
//	string mission = "Mission3";	// 1-Ball, Scale=1.0
//	string mission = "STEVE003";	// T72_small, Scale=0.1 (example_CAD.3ds)
//	string mission = "STEVE003_2";	// T72_small, Scale=1.0 (example_CAD.3ds)
//	string mission = "STEVE003_3";	// T72, Scale=1.0, PRF=213.4
//	string mission = "STEVE003_4";	// T72_small, Scale=1.0, PRF=1000



	bool ShowSummary = false;
	if(cmp.GetVal("-SHOW", tmp)){ ShowSummary = true; }	// Show summary message

	// Export folder
	string dir = dirRoot + md5 + "/Export/" + mission + "/";


	//+--------------------------------------------------+
	//|                   Read Meta                      |
	//+--------------------------------------------------+
	Meta meta = ReadWFEmulatorMeta(dir+"EXP.meta");
	meta.Print();

	string EXPORT_H = dir + "EXP_H"+meta.tx_pol()+".dat";
	string EXPORT_V = dir + "EXP_V"+meta.tx_pol()+".dat";

	if(FileExist(EXPORT_H)){
		WFEmulatorSARFocusing(dir, meta, "H", ShowSummary);
	}

	if(FileExist(EXPORT_V)){
		WFEmulatorSARFocusing(dir, meta, "V", ShowSummary);
	}



	cout<<endl<<">> Everthing is OK, Finsih! <<"<<endl;
#ifdef _WIN32
	system("pause");
#endif
	return 0;
}





