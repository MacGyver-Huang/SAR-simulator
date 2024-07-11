#ifndef SAR_H_INCLUDED
#define SAR_H_INCLUDED

#include <coordinate/geo.h>
#include <basic/vec.h>
#include <coordinate/rpy.h>
#include <sar/def.h>
#include <basic/d1.h>
#include <basic/d2.h>
#include <sar/sv.h>
#include <basic/cplx.h>
#include <sar/par.h>
#include <basic/new_time.h>
#include <basic/opt.h>
#include <coordinate/orb.h>
#include <basic/mat.h>
#include <basic/io.h>
#include <coordinate/sph.h>
#include <bvh/bvh.h>
#include <mesh/mesh.h>
// #include <basic/new_cv.h>

#ifdef _MSC_VER
// for VC2005 declared deprecated warring
#define strcpy strcpy_s
//#define gets gets_s
//#define strncpy strncpy_s
//#define sprintf sprintf_s
//#define strcat strcat_s
//#define atoi _wtoi_s
#endif

namespace sar{
	using namespace geo;
//	using namespace vec;
	using namespace rpy;
	using namespace def_func;
	using namespace def;
	//using namespace def::ORB;
	using namespace d1;
	using namespace d2;
	using namespace sv;
	using namespace cplx;
	using namespace par;
	using namespace new_time;
	using namespace opt;
	using namespace orb;
	using namespace sph;
	using namespace mesh;
	// using namespace new_cv;
	// ========================================
	// declare functions
	// ========================================
	template<typename T> VEC<T> Gd2ECR(const GEO<T>& gd);
	template<typename T> GEO<T> ECR2Gd(const VEC<T>& ECR);
	template<typename T> VEC<T> Gd2ECR(const GEO<T>& gd,const def::ORB Orb);
	template<typename T> GEO<T> ECR2Gd(const VEC<T>& ECR,const def::ORB Orb);
	template<typename T> GEO<T> Gd2Gc(const GEO<T>& gd,const def::ORB Orb);
	template<typename T> GEO<T> Gc2Gd(const GEO<T>& gc,const def::ORB Orb);


	// ========================================
	// Special class
	// ========================================
	/**
	 * Local XYZ coordinate class
	 */
	class LocalXYZ {
	public:
		LocalXYZ(){};
		LocalXYZ(const VEC<double>& x, const VEC<double>& y, const VEC<double>& z){
			_x = x; _y = y; _z = z;
		}
		const VEC<double>& x()const{ return _x; }
		const VEC<double>& y()const{ return _y; }
		const VEC<double>& z()const{ return _z; }
		VEC<double>& x(){ return _x; }
		VEC<double>& y(){ return _y; }
		VEC<double>& z(){ return _z; }
	private:
		VEC<double> _x, _y, _z;
	};


	// ========================================
	// Functions
	// ========================================
	namespace sv_func{
	    template<typename T> long find_t(const D1<T>& tt,const T& sst);
	    void funk(const double t,const double xk,const double fk,const double* A,double& fkd,double& ans);
	    void sdwang(const double xk,const double xk1,const double fk,const double fk1,
                    const double dfk,const double dfk1,double* out);
        template<typename T> void Interp(const SV<T>& in,const T dt,SV<T>& out);
		template<typename T> void Interp(const SV<T>& in, SV<T>& out);
		template<typename T> void InterpLinear(const SV<T>& in,const T dt,SV<T>& out);
		template<typename T> void InterpSpline(const SV<T>& in,const T dt,SV<T>& out);
        template<typename T1,typename T2> SV<T1> Select(const SV<T1>& in,const D1<T2>& range);
        template<typename T1,typename T2> void Select(const SV<T1>& in,const D1<T2>& range,SV<T1>& out);
	}

	namespace find{
		template<typename T> VEC<T> LookAngleLineEq(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C,const T theta_sq,const T theta_l);
		template<typename T> VEC<T> LookAngleLineEq(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C,const T theta_sq,const T thetq_l,\
													VEC<T>& D);// output
		template<typename T> VEC<T> BeamLocationOnSurface(const VEC<T>& uv,const VEC<T>& P0,const def::ORB Orb);
		template<typename T> VEC<T> ProjectPoint(const VEC<T>& P,const def::ORB Orb);
		template<typename T> double MinDistance(const D1<VEC<T> >& pos,const VEC<T>& Ppy,long& index);
		template<typename T> VEC<T> VerticalPointOnTheSurface(const VEC<T>& Ps,const VEC<T>& Ps_1,const def::ORB Orb);
		template<typename T> VEC<T> MainBeamUniVector(const VEC<T>& Ps,const VEC<T>& Ps_1,const T& theta_l,const T& theta_sqc,const def::ORB Orb);
		template<typename T> VEC<T> MainBeamUniVector(const VEC<T>& Ps,const VEC<T>& Ps_1,const T& theta_l,const T& theta_sqc,VEC<T>& D,const def::ORB Orb);
		template<typename T> double ThetaSqSqc(const VEC<T>& Ps,const VEC<T>& Psp,const VEC<T>& Ppy,const VEC<T>& nv);
		template<typename T> double ThetaInc(const VEC<T>& Ps, const VEC<double>& Poc, const def::ORB& Orb, const bool IsRefGeodetic = true);
		template<typename T> double ThetaLook(const VEC<T>& Ps, const VEC<double>& Poc, const def::ORB& Orb, const bool IsRefGeodetic = true);
		template<typename T> void LocalAxis(const VEC<T>& Ps, const VEC<T>& Psc, VEC<T>& xuv, VEC<T>& yuv, VEC<T>& zuv, const def::ORB& Orb);
		template<typename T> VEC<T> IntersectionPointOnPsPsgAlignTargetXAxis(const VEC<T>& Ps, const VEC<T>& Psc, const def::ORB& Orb);
		template<typename T> D2<T> ASPMatrix(const T& ASP,const T& ASP_North,const GEO<T>& Psg_gd);
		template<typename T> double ASPFromNorth(const VEC<T>& Ps,const VEC<T>& Ps_1,const def::ORB Orb);
		template<typename T> void RCSAngle(const VEC<T>& Ps,const VEC<T>& Ps_1,const VEC<T>& Poc,const T& ASP,double& theta_inc,double& phi,const def::ORB Orb);
		template<typename T> void RCSNearestAngle(const D1<T>& in_theta,const D1<T>& in_phi,const T& theta_val,const T& phi_val,double& theta,double& phi);
		template<typename T> void RCSNearestAngle(const D1<T>& in_theta,const D1<T>& in_phi,const T& theta_val,const T& phi_val,double& theta,double& phi,long& idx1,long& idx2);
        template<typename T> void RCSFile(const char* dir,const D2<T>& table,const long rcs_num,const double& theta_o,const double& phi_o,
                                       const char* pol,D1<CPLX<double> >& out);
		template<typename T> void Geometry(const T theta_l,const T lat_gd,const T height,T& arc_satellite,T& arc_ground,T& Rn,T& theta_e,T& theta_i,const def::ORB Orb);
		template<typename T> T SlantRange(const VEC<T>& Ps,const VEC<T>& Ps_1,const T theta_sq,const T theta_l,const def::ORB Orb);	
		template<typename T> T SlantRange(const VEC<T>& Ps,const VEC<T>& Ps_1,const T theta_sq,const T theta_l,const def::ORB Orb,VEC<T>& Psl);
		template<typename T> VEC<T> SceneCenter(const VEC<T>& Ps,const VEC<T>& Ps_1,const def::SAR Sar,const def::ORB Orb);
		template<typename T> VEC<T> SceneCenter(const VEC<T>& Ps,const VEC<T>& Ps_1,const def::SAR Sar,const def::ORB Orb,GEO<T>& Psl_gd,double& sl);
		template<typename T> D1<T> SARFindInsFreqOfMatchedFilter(const T f_nc_abs, const T PRF, const long num_az, long shift);
//		template<typename T> void RCM(const D1<T> fn, const D1<T> R0_range, const D1<T> Vr_range, const T Kr, const T Tr, const T lambda, D2<T>& RCM);
		template<typename T> void RCM(const D1<T> fn, const D1<T> R0_range, const D1<T> Vr_range, const T Fr, const T lambda, D2<T>& RCM);
		template<typename T> D1<T> Kaiser(const long num, const T beta);
		template<typename T> D1<T> KaiserWindow(const T Tr_Fr, const long num, T beta);
		template<typename T> void SlantRangePosition(const VEC<T>& Ps, const VEC<T>& Ps1, const VEC<T>& Psg, const VEC<T>& Psc, const double theta_l, const double theta_sq, const double SWrg,	const def::ORB& orb, VEC<T>& Ps_new, double& theta_l_new, bool IsNear, double theta_l_step_init=1E-5);
		template<typename T> void InsFreqOfMatchedFilter(const double f_nc_abs, const double PRF, D1<T>& fn, double& shift_d);
		template<typename T> D1<T> ThetaLookRegion(const SV<T>& sv, const D1<T>& R0, const def::ORB& Orb);
		template<typename T> void EffectiveVelocity(const SV<T>& sv, const def::ORB& Orb, const D1<T>& theta_l_limit, const double theta_sqc, const long num_rg, D1<T>& R0, D1<T>& Vr);
		template<typename T> D1<T> GroundVelocity(const SV<T>& sv, const D1<T>& theta_l_range, const def::ORB& Orb);
		void SVNormalizeHeight(SV<double>& sv, const def::ORB& Orb);
		VEC<double> MainBeamUVByLookSquintTypeA(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l,
												const double theta_sq, const def::ORB& Orb, VEC<double>& Pt,
												double& theta_l_new, VEC<double>& uvlp, const bool SHOW=false);
		VEC<double> MainBeamUVByLookSquintTypeA(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l,
												const double theta_sq, const def::ORB& Orb, VEC<double>& Pt, const bool SHOW=false);
		VEC<double> MainBeamUVByLookSquintTypeB(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l,
												const double theta_sq, const def::ORB& Orb, VEC<double>& Pt);
		VEC<double> MainBeamUVByLookSquintTypeASphericalCoordinate(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l, const double theta_sq);
		LocalXYZ LocalCoordinate(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l, const double theta_sq,
								 const def::ORB& Orb, const bool SHOW=false);
		SPH<double> LocalSPH(const VEC<double>& Ps, const VEC<double>& Pt, const LocalXYZ& locXYZ, const bool SHOW=false);
		D1<SPH<double> > LocalSPH(const SV<double>& sv_int, const VEC<double>& Pt, const double PRI, const def::ORB Orb, const bool SHOW=false);
		void AllAngle(const SV<double>& sv_int, const double theta_sqc, const double theta_l_MB,
					  D1<double>& inst_theta_l, D1<double>& inst_theta_sq, D1<double>& inst_theta_az, const bool NormalizeHeight=true);
		D1<double> ThetaAz(const SV<double>& sv_int, const double theta_sqc, const double theta_l_MB, const bool NormalizeHeight = true);
		VEC<double> NearestPs(const SV<double>& sv_int, const VEC<double>& Pt, const double dt_min, const size_t ItrCount=5, const bool SHOW=false);
		template<typename T> CPLX<T> ACCC(D1<CPLX<T> >& array);
		double AzimuthAntennaGain(const SAR& Sar, const VEC<double>& MainBeamUV, const VEC<double>& NorSquintPlane, const VEC<double>& PPs, const VEC<double>& PPt, const VEC<double>& hit);
		double BearingAngle(const GEO<double>& Ps, const GEO<double>& Ps1, const def::ORB& Orb);
		sv::SV<double> CircularSARPathTrajectory(const D1<double>& As, const double Ev, const double R0, const double tar_lon, const double tar_lat_gd, const double PRF, const bool SHOW=false);

		namespace AbsolutedCentroidFreq {
			template<typename T> double MLCC(const D2<CPLX<T> >& rc, const double Fr, const double f0, const double PRF,
											 const double Kr, const double Tr, const bool SHOW=false);
		}
	}
	namespace make {
		void FakePath(const double PRF, const string& file_sv, const int Na);
		void PesudoSAR(const size_t i, const BVH& bvh, const MeshInc& inc_mesh, const string& dir_out, const bool IsWREND=false, const bool IsWPSAR=false);
	}
	namespace cw{
		template<typename T> T EffectiveSwath(const long Nr, const T fs);
		template<typename T> T MaxRange(const long Nr, const T fs);
		template<typename T> long TraditionalRangeSampleNumber(const T Rmax, const T fs);
		void ShiftSample(const long Nr, const double fs, const double SW, const double dR, const double Rmin, long& org, long& nss_int, double& nss_rem);
		void SignalShift(const double nss_org, const double nss_rem, const long Nr, const double Rc, const double dR, D2<CPLX<double> >& s0, D1<double>& R0);

    }

    template<typename T> T Wr(const T& t,const T& Tr);
    template<typename T> D1<T> Wr(const D1<T>& t,const T& Tr);
    template<typename T> double Waz(const T& theta_sq_sqc,const T& theta_az);
    template<typename T> D1<long> InitPpyRCS(const char* RCS_dir,const D2<T>& table,const T& theta_o,const T& phi_o, const char* pol, // input
                                          const VEC<T>& Poc,const double TR_ASP,const double ASP_North,const def::ORB Orb, // input
                                          D1<CPLX<T> >& RCS_val,D1<VEC<T> >& Ppy); // resturn
	void GetMapCoordinate(const SAR& Sar, const def::ORB& Orb, VEC<double>& Ps, VEC<double>& Ps1,	// input
						  const double theta_l_MB, const double SW,								// input
						  GEO<double>& G1n, GEO<double>& G1c, GEO<double>& G1f);				// output
	SAR_PAR Init_SAR_par(const char* sar_title,const char* sar_sensor,const char* sar_chirp,const char* sar_mode,const char* sar_type,
                         const char* sar_spectrum,const char* sar_ant_pattern,const long num_rg,const def::SAR Sar);						 
    template<typename T> PROC_PAR Init_PROC_par(const char* procp_title,bool procp_SRC,bool procp_deskew,
												const def::SAR Sar,const def::ORB Orb,
												const VEC<T>& Poc,const T& t0,const VEC<T>& Ps0,const SV<T>& TER_SV,
												const T tt_min_rg,const T tt_max_rg,const long num_az);
	void write_SAR_par(const char* filename,const SAR_PAR& sarp);
    void write_PROC_par(const char* filename,const PROC_PAR& procp);
    template<typename T1, typename T2> VEC<T1> VecTransform(const VEC<T1>& Ppy_vec,const D2<T2>& Matrix);
    template<typename T> VEC<T> VecTransform(const VEC<T>& Ppy_vec,const D2<T>& Matrix,const VEC<T>& Poc);
    template<typename T> D1<VEC<T> > MakeMovingTarget(const T& V_rg,const T& V_az,const T* t,const long sample,
                                                   const VEC<T>& org);
    template<typename T> D1<VEC<T> > MakeMovingTarget(const T& V_rg,const T& V_az,const T* t,const long sample,
                                                   const VEC<T>& org,const D2<T>& Matrix);
	template<typename T> D2<CPLX<T> > RCMC(D2<CPLX<T> >& Srd, const D2<T>& RCM, const bool SHOW);
	double RCS2dB(const double rcs, const double f0);
	template<typename T> D1<CPLX<T> > RCMCSincInterpCore(D1<CPLX<T> >& Srd, const D1<T>& RCM, const T KAISER_BETA=2.1, const long NUM_KERNEL=8);
	template<typename T> D2<CPLX<T> > RCMCSinc(D2<CPLX<T> >& Srd, const D2<T>& RCM, const bool SHOW);
	
	namespace fft{
		template<typename T> void RangeForward(D2<CPLX<T> >& in_out);
		template<typename T> void RangeInverse(D2<CPLX<T> >& in_out);
		template<typename T> void AzimuthForward(D2<CPLX<T> >& in_out);
		template<typename T> void AzimuthInverse(D2<CPLX<T> >& in_out);
		template<typename T> void RangeShift(D2<CPLX<T> >& in_out_m_az_n_rg);
		template<typename T> void AzimuthShift(D2<CPLX<T> >& in_out_m_az_n_rg);
	}
	
	namespace csa{
		template<typename T> D1<T> MigrationParameter(const D1<T>& fn, const T Vr, const T lambda);
		template<typename T> T MigrationParameter(const T fn, const T Vr, const T lambda);
		template<typename T> D2<T> ModifiedChirpRate(const D1<T>& fn, const T Vr, const T Kr, const D1<T>& R0, const T c, const T f0);
		template<typename T> T ModifiedChirpRate(const T fn, const T Vr, const T Kr, const T R0, const T c, const T f0);
	}
	
	// Implement ********************************************************
	template<typename T>
	VEC<T> Gd2ECR(const GEO<T>& gd){
		def::ORB Orb;
		return Gd2ECR(gd, Orb);
	}
	
	template<typename T>
	GEO<T> ECR2Gd(const VEC<T>& ECR){
		def::ORB Orb;
		return ECR2Gd(ECR, Orb);
	}
	
	template<typename T>
	VEC<T> Gd2ECR(const GEO<T>& gd,const def::ORB Orb){
	/*
	 Purpos:
		Geodetic[Lon,Lat,height] to ECR[X,Y,Z]
	 Input:
		geo_gd :[rad,rad,m,m].[lon,lat_gd,N_gd,h_gd] dim:[4]or[4,1]or[4,n] Geodetic locations
	 Retuen:
		[m,m,m]->[X,Y,Z] ECR location
	 Reference:
		"Theoretucal Basis of the SDP Tookit geolocation package for the ECS Project",
		technical paper (445-TP-002-002), May, 1995.pp.6-20
	*/
		double N;
		VEC<T> out;
		// flatness
		N = Orb.E_a()/sqrt(1.-Orb.f()*(2-Orb.f())*_sin(gd.lat())*_sin(gd.lat()));
		return VEC<T>((N+gd.h())*_cos(gd.lat())*_cos(gd.lon()),
					  (N+gd.h())*_cos(gd.lat())*_sin(gd.lon()),
					  ((1.-Orb.f())*(1.-Orb.f())*N+gd.h())*_sin(gd.lat()));
	}

	template<typename T>
	GEO<T> ECR2Gd(const VEC<T>& ECR,const def::ORB Orb){
	/*
	 Purpose:
		ECR[X,Y,Z] to Geodetic[Lon,Lat_gd,h_gd]
	 Input:
		ECR :[X,Y,Z] ECR locations
	 Return:
		[rad,rad,m,m]->[lon,lat_gd,N_gd,h_gc] Geodetic Location
	 Reference:
		"Theoretucal Basis of the SDP Tookit geolocation package for the ECS Project",
		technical paper (445-TP-002-002), May, 1995.pp.6-21
	*/
		double r0,zn;//,alpha;
		double w=0,wst=1.,tmp;
		long niter=0;
		double zeta,NC;
		double o_lon,o_lat,o_h;

		// Normalized distance from Earth grographic north-south axis
		r0 = sqrt(ECR.x()*ECR.x()+ECR.y()*ECR.y())/Orb.E_a();
		zn = ECR.z()/Orb.E_a();

		o_lon=atan2(ECR.y(),ECR.x());

		// Initial
		//alpha = r0*r0 + (zn*zn/(1-Orb.e2()));
		while( (fabs(wst-w) > 1e-20) && (niter < 20) ){
			wst = w;
			niter += 1;
			tmp = r0-wst;
			w = Orb.e2()*tmp/sqrt( tmp*tmp + zn*zn*(1-Orb.e2()) );
		}
//#ifdef DEBUG
//		cout<<"abs(wst-w)="<<abs(wst-w)<<endl;
//		cout<<"niter="<<niter<<endl;
//#endif

		o_lat = atan(zn/(r0-w));
		// Find height
		// radius of curvature in prime vertical(NC)
		zeta = 1. / ( 1. - Orb.e2() * _sin(o_lat) * _sin(o_lat) );
		NC = Orb.E_a() *sqrt(zeta);
		// height above ellipsoid
		o_h = r0*Orb.E_a() / _cos(o_lat) - NC;	// height of geodetic

		return GEO<T>(o_lon,o_lat,o_h);
	}

	template<typename T>
	GEO<T> Gd2Gc(const GEO<T>& gd,const def::ORB Orb){
	/*
	 Purpos:
		Geodetic[Lon,Lat,height] to Geocentric[Lon,Lat,height]
	 Input:
		geo_gd :[rad,rad,m].[lon,lat_gd,h_gd] Geodetic locations
	 Retuen:
		[rad,rad,m]->[lon,lat_gd,N_gd,h_gc] Geocentric Location
	 Reference:
		MathWorks, "Geocentric to Geodetic Latitude :: Blocks (Aerospace Blockset)"
	*/
		double rs     = Orb.E_b();
		double lat_gd = gd.lat();
		double lon_gd = gd.lon();
		double h_gd   = gd.h();

		double lambda_s = atan(Square(1.0-Orb.f()) * tan(lat_gd));
		double A = h_gd*sin(lat_gd)+rs*sin(lambda_s);
		double B = h_gd*cos(lat_gd)+rs*cos(lambda_s);

		double lat_gc = atan2(A,B);

		double ua = atan2(tan(lat_gc), Square(1.0 - Orb.f()));
		double delta_lambda = ua - lat_gc;
		double h_gc = h_gd/cos(delta_lambda);

		return GEO<T>(lon_gd,lat_gc,h_gc);
	}

	template<typename T>
	GEO<T> Gc2Gd(const GEO<T>& gc,const def::ORB Orb){
	/*
	 Purpos:
		Geocentric[Lon,Lat,height] to Geodetic[Lon,Lat,height]
	 Input:
		geo_gc :[rad,rad,m].[lon,lat_gc,h_gc] Geocentric locations
	 Retuen:
		[rad,rad,m]->[lon,lat_gd,N_gd,h_gc] Geodetic Location
	 Reference:
		MathWorks, "Geocentric to Geodetic Latitude :: Blocks (Aerospace Blockset)"
	*/
		double lat_gc = gc.lat();
		double lon_gc = gc.lon();
		double h_gc   = gc.h();

		double xa = (1.0-Orb.f())*Orb.E_a()/sqrt(Square(tan(lat_gc))+Square(1.0-Orb.f()));

		double ua = atan2(tan(lat_gc),Square(1-Orb.f()));
		double N_gc = xa/cos(lat_gc);
		double d_lambda  = ua-lat_gc;
		double h_gd = h_gc*cos(d_lambda);

		double d_ua = atan2(h_gc*sin(d_lambda),N_gc+h_gc);
		double lat_gd = ua-d_ua;

		return GEO<T>(lon_gc,lat_gd,h_gd);
	}
	 

    //
    // namespace sv_func::
    //
    template<typename T>
    long sv_func::find_t(const D1<T>& tt,const T& sst){
    /*
     Input:
        tt  :[sec](series) Input time series
        sst :[sec] Input time point
     Return:
        The nearest point index of sst in the time series tt.
    */
        long K=0,NP=tt.GetNum();
        for(long i=0;i<NP-1;++i){
            K = ( (sst-tt[i])*(sst-tt[i+1]) <= 0.)? i:K;
//#ifdef DEBUG
//            cout<<"sst="<<sst<<",tt[i]="<<tt[i]<<",tt[i+1]="<<tt[i+1]<<endl;
//            cout<<"k="<<K<<endl;
//#endif
        }
        return ( (K < 0)||(K > NP) )? -1:K;
    }

    void sv_func::funk(const double t,const double xk,const double fk,const double* A,double& fkd,double& ans){
    /*
     Input:
        t   :[sec] Interpolated time point
        xk  :[sec] Original time point at index of "k"
        fk  :[m] Position at any "ONE" direction at time of xk
        A   :[x](3*3 matrix) derived from sdwang
     Return:
        fkd :[m/sec] Interpolated velocity
        ans :[m] Interpolated position at any "ONE" direction
    */
        ans = (t - xk) * (A[0]*t*t + A[1]*t + A[2]) + fk;
        fkd = (A[0]*t*t + A[1]*t + A[2]) + (t - xk)*(2.*A[0]*t + A[1]);
//#ifdef DEBUG
//        cout<<"ans="<<ans<<endl;
//        cout<<"xk=t1[i]="<<xk<<endl;
//        cout<<"fk=tt[k]="<<fk<<endl;
//        cout<<"A[0],A[1],A[2]="<<"["<<A[0]<<","<<A[1]<<","<<A[2]<<"]"<<endl;
//        cout<<"fkd=vel[i]="<<fkd<<endl;
//#endif
        //return ans;
    }

    void sv_func::sdwang(const double xk,const double xk1,const double fk,const double fk1,\
                         const double dfk,const double dfk1,double* out){
    /*
     Purpose:
        Find the interpolation matrix
    */
        double in_a[]={xk1*xk1*xk1 - xk1*xk1*xk, 3.0*xk*xk - 2.0*xk*xk, 3.0*xk1*xk1 - 2.0*xk1*xk};
        double in_b[]={xk1*xk1 - xk1*xk,         2.0*xk - xk,           2.0*xk1 - xk            };
        double in_c[]={xk1 - xk,                 1.0,                   1.0                     };
        double in_d[]={fk1 - fk,                 dfk,                   dfk1                    };
        double tp_det[]={in_a[0],in_a[1],in_a[2],in_b[0],in_b[1],in_b[2],in_c[0],in_c[1],in_c[2]};
        double tp_aa[] ={in_d[0],in_d[1],in_d[2],in_b[0],in_b[1],in_b[2],in_c[0],in_c[1],in_c[2]};
        double tp_bb[] ={in_a[0],in_a[1],in_a[2],in_d[0],in_d[1],in_d[2],in_c[0],in_c[1],in_c[2]};
        double tp_cc[] ={in_a[0],in_a[1],in_a[2],in_b[0],in_b[1],in_b[2],in_d[0],in_d[1],in_d[2]};

        double det_det=def_func::det(tp_det);
        double det_aa =def_func::det(tp_aa);
        double det_bb =def_func::det(tp_bb);
        double det_cc =def_func::det(tp_cc);

        out[0] = det_aa/det_det;
        out[1] = det_bb/det_det;
        out[2] = det_cc/det_det;
    }
	
	template<typename T>
	void sv_func::Interp(const SV<T>& in, SV<T>& out){
	/*
	 Input:
		in :[sec,m,m,m,m/s,m/s,m/s] Input State vector
	 Output:
		out :[sec,m,m,m,m/s,m/s,m/s] Input State vector
	 NOTE:
		The out.t series *MUST* be calculated first
	 */
		long NP=in.GetNum();
        T t_min=mat::min(in.t());
		
        D2<double> a1x(NP,3),a1y(NP,3),a1z(NP,3);
		
        for(int i=0;i<3;++i){
            a1x[NP-1][i]=0;
            a1y[NP-1][i]=0;
            a1z[NP-1][i]=0;
        }
		
        for(long i=0;i<NP-1;++i){
            sdwang(in.t()[i]-t_min, in.t()[i+1]-t_min, in.pos()[i].x(), in.pos()[i+1].x(), in.vel()[i].x(), in.vel()[i+1].x(), a1x[i]);
            sdwang(in.t()[i]-t_min, in.t()[i+1]-t_min, in.pos()[i].y(), in.pos()[i+1].y(), in.vel()[i].y(), in.vel()[i+1].y(), a1y[i]);
            sdwang(in.t()[i]-t_min, in.t()[i+1]-t_min, in.pos()[i].z(), in.pos()[i+1].z(), in.vel()[i].z(), in.vel()[i+1].z(), a1z[i]);
        }
		
		
        long k=19;
		for(long i=0;i<out.GetNum();++i){
            k=find_t(in.t()-t_min,out.t()[i]-t_min);
            if(k != -1){
                // x
                funk(out.t()[i]-t_min, in.t()[k]-t_min, in.pos()[k].x(), a1x[k], out.vel()[i].x(), out.pos()[i].x());
                // y
                funk(out.t()[i]-t_min, in.t()[k]-t_min, in.pos()[k].y(), a1y[k], out.vel()[i].y(), out.pos()[i].y());
                // z
                funk(out.t()[i]-t_min, in.t()[k]-t_min, in.pos()[k].z(), a1z[k], out.vel()[i].z(), out.pos()[i].z());
            }
        }
	}
	

    template<typename T>
    void sv_func::Interp(const SV<T>& in,const T dt,SV<T>& out){
    /*
     Input:
        sv  :[sec,m,m,m,m/s,m/s,m/s] Input State vector
        dt  :[sec] Time difference
     Return:
        out :[sec,m,m,m,m/s,m/s,m/s] Output State vector
    */
        long NP=in.GetNum();
        T t_min=mat::min(in.t());

        D1<double> tt(NP); tt=in.t();

        // Initialize
        for(long i=0;i<NP;++i){
            tt[i] -= t_min;
        }
        double st1=tt[0];
        long kk=(long)round((tt[NP-1]-tt[0])/dt);

        D2<double> a1x(NP,3),a1y(NP,3),a1z(NP,3);

        for(int i=0;i<3;++i){
            a1x[NP-1][i]=0;
            a1y[NP-1][i]=0;
            a1z[NP-1][i]=0;
        }

        for(long i=0;i<NP-1;++i){
            sdwang(tt[i], tt[i+1], in.pos()[i].x(), in.pos()[i+1].x(), in.vel()[i].x(), in.vel()[i+1].x(), a1x[i]);
            sdwang(tt[i], tt[i+1], in.pos()[i].y(), in.pos()[i+1].y(), in.vel()[i].y(), in.vel()[i+1].y(), a1y[i]);
            sdwang(tt[i], tt[i+1], in.pos()[i].z(), in.pos()[i+1].z(), in.vel()[i].z(), in.vel()[i+1].z(), a1z[i]);
        }

		// (Modified: 2020/04/12 Steve Chiang [Start]
		out = SV<T>(kk);
		// (Modified: 2020/04/12 Steve Chiang [End]

        long k=19;
        for(long i=0;i<kk;++i){
            out.t()[i]=st1+(i*dt);
            k=find_t(tt,out.t()[i]);
//#ifdef DEBUG
//            cout<<"k="<<k<<endl;
//#endif
            if(k != -1){
                // x
                funk(out.t()[i], tt[k], in.pos()[k].x(), a1x[k], out.vel()[i].x(), out.pos()[i].x());
                // y
                funk(out.t()[i], tt[k], in.pos()[k].y(), a1y[k], out.vel()[i].y(), out.pos()[i].y());
                // z
                funk(out.t()[i], tt[k], in.pos()[k].z(), a1z[k], out.vel()[i].z(), out.pos()[i].z());
            }
        }
//#ifdef DEBUG
//        cout<<"t_min="<<t_min<<endl;
//#endif
        for(long i=0;i<kk;++i){
            out.t()[i]+=t_min;
        }


//        cout<<"+--------------------------------------------------------+"<<endl;
//		cout<<"|                sv_func::Interp (START)                 |"<<endl;
//		cout<<"+--------------------------------------------------------+"<<endl;
//
//		cout<<"Print (in) [START]: "<<endl;
//		in.Print();
//		cout<<"Print (in) [END]: "<<endl<<endl;
//
//		printf("dt = %.8f\n", dt);
//
//		cout<<"Print (out) [START]: "<<endl;
//		out.Print(0);
//		out.Print(out.GetNum()/2);
//		out.Print(out.GetNum()-1);
//		cout<<"Print (out) [END]: "<<endl<<endl;
//
//		cout<<"+--------------------------------------------------------+"<<endl;
//		cout<<"|                sv_func::Interp (END)                   |"<<endl;
//		cout<<"+--------------------------------------------------------+"<<endl;
    }

    template<typename T>
    void sv_func::InterpLinear(const SV<T>& in,const T dt,SV<T>& out){
    	D1<double> int_t = linspace(in.t()[0], in.t()[in.GetNum()-1], dt);
		out = SV<double>(int_t.GetNum());

		D1<double> org_t(in.GetNum());
		D1<double> org_pos_x(in.GetNum());
		D1<double> org_pos_y(in.GetNum());
		D1<double> org_pos_z(in.GetNum());
		D1<double> org_vel_x(in.GetNum());
		D1<double> org_vel_y(in.GetNum());
		D1<double> org_vel_z(in.GetNum());

		D1<double> int_pos_x(out.GetNum());
		D1<double> int_pos_y(out.GetNum());
		D1<double> int_pos_z(out.GetNum());
		D1<double> int_vel_x(out.GetNum());
		D1<double> int_vel_y(out.GetNum());
		D1<double> int_vel_z(out.GetNum());

		// Read
		D1<VEC<double> > pos = in.pos();
		D1<VEC<double> > vel = in.vel();
		for(size_t i=0;i<in.GetNum();++i){
			org_t[i] = in.t()[i];
			org_pos_x[i] = pos[i].x();
			org_pos_y[i] = pos[i].y();
			org_pos_z[i] = pos[i].z();
			org_vel_x[i] = vel[i].x();
			org_vel_y[i] = vel[i].y();
			org_vel_z[i] = vel[i].z();
		}

		// Calculation
		int_pos_x = mat::PolyInt1(org_t, org_pos_x, int_t);
		int_pos_y = mat::PolyInt1(org_t, org_pos_y, int_t);
		int_pos_z = mat::PolyInt1(org_t, org_pos_z, int_t);


		// Write
		for(size_t i=0;i<out.GetNum();++i){
			out.t()[i] = int_t[i];
			out.pos()[i] = VEC<double>(int_pos_x[i], int_pos_y[i], int_pos_z[i]);
//			out.vel()[i] = VEC<double>(int_vel_x[i], int_vel_y[i], int_vel_z[i]);
			if(i>0 && i<out.GetNum()-1){
				out.vel()[i] = (VEC<double>(int_pos_x[i],   int_pos_y[i],   int_pos_z[i]) -
								VEC<double>(int_pos_x[i-1], int_pos_y[i-1], int_pos_z[i-1])) / dt;
			}
			// 1st point
			if(i == 1){
				out.vel()[0] = out.vel()[1];
			}
			// Last point
			if(i == out.GetNum()-1){
				out.vel()[out.GetNum()-1] = out.vel()[out.GetNum()-2];
			}
		}
    }

    template<typename T>
	void sv_func::InterpSpline(const SV<T>& in,const T dt,SV<T>& out){
		D1<double> int_t = linspace(in.t()[0], in.t()[in.GetNum()-1], dt);
		out = SV<double>(int_t.GetNum());

		D1<double> org_t(in.GetNum());
		D1<double> org_pos_x(in.GetNum());
		D1<double> org_pos_y(in.GetNum());
		D1<double> org_pos_z(in.GetNum());
		D1<double> org_vel_x(in.GetNum());
		D1<double> org_vel_y(in.GetNum());
		D1<double> org_vel_z(in.GetNum());

		D1<double> int_pos_x(out.GetNum());
		D1<double> int_pos_y(out.GetNum());
		D1<double> int_pos_z(out.GetNum());
		D1<double> int_vel_x(out.GetNum());
		D1<double> int_vel_y(out.GetNum());
		D1<double> int_vel_z(out.GetNum());

		// Read
		D1<VEC<double> > pos = in.pos();
		D1<VEC<double> > vel = in.vel();
		for(size_t i=0;i<in.GetNum();++i){
			org_t[i] = in.t()[i];
			org_pos_x[i] = pos[i].x();
			org_pos_y[i] = pos[i].y();
			org_pos_z[i] = pos[i].z();
			org_vel_x[i] = vel[i].x();
			org_vel_y[i] = vel[i].y();
			org_vel_z[i] = vel[i].z();
		}

		// Remove mean
		double org_pos_x_mean = mat::mean(org_pos_x);
		double org_pos_y_mean = mat::mean(org_pos_y);
		double org_pos_z_mean = mat::mean(org_pos_z);
		org_pos_x = org_pos_x - org_pos_x_mean;
		org_pos_y = org_pos_y - org_pos_y_mean;
		org_pos_z = org_pos_z - org_pos_z_mean;

		// Calculation
		int_pos_x = mat::SplInt1(org_t, org_pos_x, int_t);
		int_pos_y = mat::SplInt1(org_t, org_pos_y, int_t);
		int_pos_z = mat::SplInt1(org_t, org_pos_z, int_t);

		// Add mean
		int_pos_x = int_pos_x + org_pos_x_mean;
		int_pos_y = int_pos_y + org_pos_y_mean;
		int_pos_z = int_pos_z + org_pos_z_mean;


		// Write
		for(size_t i=0;i<out.GetNum();++i){
			out.t()[i] = int_t[i];
			out.pos()[i] = VEC<double>(int_pos_x[i], int_pos_y[i], int_pos_z[i]);
//			out.vel()[i] = VEC<double>(int_vel_x[i], int_vel_y[i], int_vel_z[i]);
			if(i>0 && i<out.GetNum()-1){
				out.vel()[i] = (VEC<double>(int_pos_x[i],   int_pos_y[i],   int_pos_z[i]) -
								VEC<double>(int_pos_x[i-1], int_pos_y[i-1], int_pos_z[i-1])) / dt;
			}
			// 1st point
			if(i == 1){
				out.vel()[0] = out.vel()[1];
			}
			// Last point
			if(i == out.GetNum()-1){
				out.vel()[out.GetNum()-1] = out.vel()[out.GetNum()-2];
			}
		}
	}

    template<typename T1,typename T2>
    SV<T1> sv_func::Select(const SV<T1>& in,const D1<T2>& range){
    /*
    */
        long num=range.GetNum();
        //SV<T1> out(num);
        T1* t=new T1[num];
        VEC<T1> *pos=new VEC<T1>[num];
        VEC<T1> *vel=new VEC<T1>[num];
        for(long i=0;i<num;++i){
            t[i]=in.GetT()[range[i]];
            pos[i].Setxyz(in.GetPos()[range[i]]);
            vel[i].Setxyz(in.GetVel()[range[i]]);
#ifdef DEBUG
            cout<<"range[i]="<<range[i]<<", "; (in.GetPos())[range[i]].Print();
#endif
        }
        return SV<T1>(t,pos,vel,num);
    }

    template<typename T1,typename T2>
    void sv_func::Select(const SV<T1>& in,const D1<T2>& range,SV<T1>& out){
    /*
     Purpose:
        Choose the interesting range of state vector
     Input:
        in      : (SV class) State vector
        range   : (D1 class) Range series
        out     : (SV class) Destination
    */
        long num=range.GetNum();
#ifdef DEBUG
        cout<<"in.GetNum()="<<in.GetNum()<<endl;
        cout<<"out.GetNum()="<<out.GetNum()<<endl;
#endif
        for(long i=0;i<num;++i){
            out.t()[i]=in.t()[range[i]];
            out.pos()[i]=in.pos()[range[i]];
            out.vel()[i]=in.vel()[range[i]];
        }
    }

    //
    // namespace find::
    //
	template<typename T>
	VEC<T> find::LookAngleLineEq(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C,const T theta_sq,const T theta_l){
	/*
	 Input:
		A :[m,m,m] ECR, original point at orbit
		B :[m,m,m] ECR, point of A project to ground
		C :[m,m,m] ECR, point next to A at orbit
		theta_sq :[rad] squint angle
		theta_l  :[rad] look angle for specific beam
	 Return:
		uv_AD :[m,m,m] unit vector(direction vector) for destination line eq.
	 Modifeid:
		(20100406) CYChiang : Changing the definitions of look angle and squint angle
		(20100607) CYChiang : Remove (20100406) method and *ADD* theta_az
	*/
		//// REMOVE (20100607)
		//// Modified (20100406)
		//// 1. Rotate theta_l about CA
		//VEC<T> AB=B-A;
		//VEC<T> AC=C-A;
		//VEC<T> uv_AX = Unit(cross(AB,AC));
		//VEC<T> X  = A + uv_AX*1.;
		//VEC<T> AF = vec::find::UnitVector( A,X,B,theta_l );
		//// 2. Find point F on the surface and distance AF
		//VEC<T> F  = sar::find::BeamLocationOnSurface( Unit(AF),A );
		//double m  = (F-A).abs();
		//// 3. Find minimum distance from point F to Line (AB)
		//VEC<T> E(0,0,0);
		//double L  = vec::find::MinDistanceFromPointToLine( A,B,F,E );
		//// 4. Find pesudo theta_sq_fake
		//double theta_sq_fake = 2.*asin( m/L*sin(theta_sq/2.) );
		//// 4. Rotate theta_sq_fake about BA
		//VEC<T> EA = A-E;
		//VEC<T> EF = F-E;
		//VEC<T> uv_AZ = Unit(cross(EA,EF));
		//VEC<T> Z  = E + uv_AZ*1.;
		//VEC<T> ED = vec::find::UnitVector( E,Z,F,theta_sq_fake );
		//VEC<T> uv_ED = Unit(ED);
		//// 6. Find D point
		//VEC<T> D = E + uv_ED*L;
		//// 7. Results
		//VEC<T> uv_AD = vec::Unit(D-A);
		//D = sar::find::BeamLocationOnSurface( uv_AD,A );



		// Find Pseudo point at x' axis
		VEC<T> AC,AB,AX,X,E,D,uv_AE,uv_AD;
		
		AC=C-A;
		AB=B-A;
		AX=cross(AB,AC);
		X=A+Unit(AX);
		
		// Find azimuth angle
		double theta_az = asin( sin(theta_sq)/sin(theta_l) );
		
		// Find (A,E) line equation
		uv_AE = vec::find::UnitVector(A,C,X,theta_az,E);
			
		// Find (A,D) line equation
		uv_AD = vec::find::UnitVector(A,E,B,theta_l,D);

		return uv_AD;
	}

	template<typename T>
	VEC<T> find::LookAngleLineEq(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C,const T theta_sq,const T theta_l,\
								 VEC<T>& D){// output
	/*
	 Input:
		A :[m,m,m] ECR, original point at orbit
		B :[m,m,m] ECR, point of A project to ground
		C :[m,m,m] ECR, point next to A at orbit
		theta_sq :[rad] squint angle
		theta_l  :[rad] look angle for specific beam
	 Return:
		uv_AD :[m,m,m] unit vector(direction vector) for destination line eq.
		uv_AE :[m,m,m] unit vector form A to E point.
		E :[m,m,m] Point E
		D :[m,m,m] Point D
	 Modifeid:
		(20100406) CYChiang : Changing the definitions of look angle and squint angle
		(20100607) CYChiang : Remove (20100406) method and *ADD* theta_az
	*/
		//// REMOVE (20100607)
		//// Modified (20100406)
		//// 1. Rotate about theta_l
		//VEC<T> AF = vec::find::ArbitraryRotate( B,theta_l,Unit(A-C) );
		//// 2. Find point F on the surface
		//VEC<T> F  = sar::find::BeamLocationOnSurface( Unit(AF),A );
		//// length from C to line(AB)
		//double L  = vec::find::MinDistanceFromPointToLine( A,B,F );
		//// length from A to F
		//double m  = AF.abs();
		//// 3. Find pesudo theta_sq
		//double theta_sq_fake = 2.*asin( m/L*sin(theta_sq/2.) );
		//// 4. Rotate about theta_sq_fake
		//VEC<T> AD = vec::find::ArbitraryRotate( F,theta_sq_fake,Unit(A-B) );
		//VEC<T> uv_AD = Unit(AD);
		//
		//D = sar::find::BeamLocationOnSurface( uv_AD,A );

		// Find Pseudo point at x' axis
		VEC<T> AC,AB,AX,X,E,uv_AE,uv_AD;
		
		AC=C-A;
		AB=B-A;
		AX=cross(AB,AC);
		X=A+Unit(AX);

		// Tind azimuth angle
		double theta_az = asin( sin(theta_sq)/sin(theta_l) );
		
		// Find (A,E) line equation
		uv_AE = vec::find::UnitVector(A,C,X,theta_az,E);
		//uv_AE=vec::find::UnitVector(A,C,X,theta_sq,E);
		//uv_AE=vec::find::UnitVector(A,X,B,theta_l,E);
		
		// Find (A,D) line equation
		uv_AD = vec::find::UnitVector(A,E,B,theta_l,D);
		//uv_AD=vec::find::UnitVector(A,E,B,theta_l,D);
		//uv_AD=vec::find::UnitVector(A,C,E,theta_sq,D);

		return uv_AD;
	}

	template<typename T>
	VEC<T> find::BeamLocationOnSurface(const VEC<T>& uv,const VEC<T>& P0,const def::ORB Orb){
	/*
	 Purpose:
		Find the point on the ellipse intersective with a line.
	 Input:
		uv :[m,m,m] Line unit vector
		P0 :[m,m,m] point on the line
		E_a:[m] semi-major axis
		E_b:[m] semi-minor axis
	 Return:
		out:[m,m,m] point on the surface on the line
	*/
		double C3,C4,M,N,I,J,K;
		double x[2],dis[2],tmp;
		VEC<double> out[2],tp;

		C3 = uv.x()*P0.y()/uv.y() - P0.x();
		C4 = uv.x()*P0.z()/uv.z() - P0.x();

		M = uv.y()/(uv.x()*Orb.E_a()); M *= M;
		N = uv.z()/(uv.x()*Orb.E_b()); N *= N;

		I = 1./(Orb.E_a()*Orb.E_a()) + M + N;
		J = 2.*( M*C3 + N*C4 );
		K = M*C3*C3 + N*C4*C4 - 1.;
#ifdef DEBUG
        cout<<"P0.x()=P0[0]="<<P0.x()<<endl;
        cout<<"P0.y()=P0[1]="<<P0.y()<<endl;
        cout<<"P0.z()=P0[2]="<<P0.z()<<endl;
        cout<<"uv.x()=A="<<uv.x()<<endl;
        cout<<"uv.y()=B="<<uv.y()<<endl;
        cout<<"uv.z()=C="<<uv.z()<<endl;
		cout<<"C3="<<C3<<endl;
		cout<<"C4="<<C4<<endl;
		cout<<"M="<<M<<endl;
		cout<<"N="<<N<<endl;
		cout<<"I="<<I<<endl;
		cout<<"J="<<J<<endl;
		cout<<"K="<<K<<endl;
		cout<<"J*J-4.*I*K="<<J*J-4.*I*K<<endl;
#endif
		if( (J*J-4.*I*K) < 0. ){
			cout<<"ERROR(SARFindBeamLocationOnSurface)::sqrt() < 0"<<endl;
			exit(EXIT_FAILURE);
		}

		tmp = sqrt(J*J-4*I*K);
		x[0] = ( -J+tmp )/(2*I);
		x[1] = ( -J-tmp )/(2*I);

		for(long i=0;i<2;i++){
			out[i] = VecLineEq(uv,P0,x[i],'X');
			tp = P0-out[i];
			dis[i] = tp.abs();
		}

		return (dis[0] <= dis[1])? out[0]:out[1];
	}

	template<typename T>
	VEC<T> find::ProjectPoint(const VEC<T>& P,const def::ORB Orb){
	/*
	 Input:
		P :[m,m,m] Position on the space with ECR system
	 Return:
		out :[m,m,m] Point on the earth serface
	*/
		GEO<T> gd=ECR2Gd(P,Orb);
		gd.Seth(0);
		return Gd2ECR(gd,Orb);
	}

	template<typename T>
	double find::MinDistance(const D1<VEC<T> >& pos,const VEC<T>& Ppy,long& index){
	//double find::MinDistance(const VEC<T>* pos,const VEC<T>& Ppy,const long num_sv, long& index){
	/*
	 Input:
		pos :[m,m,m] State vector position
		Ppy :[m,m,m] Point on any location
	 Return:
		dis :[m] Minimun distance between Ppy and pos-series
		index :[x] index of destination point
	*/
		long num_sv=pos.GetNum();
		D1<double> dis(num_sv);
		//double* dis=new double[num_sv];
		VEC<T> tmp;
		for(long i=0;i<num_sv;++i){
			tmp=pos[i]-Ppy;
			dis[i]=tmp.abs();
//#ifdef DEBUG
//			if(i<10){
//				cout<<"pos[i]="; pos[i].Print();
//				cout<<"Ppy="; Ppy.Print();
//				cout<<"pos[i]-Ppy="; tmp.Print();
//				cout<<"dis="<<dis[i]<<endl;
//			}
//#endif
		}
		return mat::min(dis,index);
	}

	template<typename T>
	VEC<T> find::VerticalPointOnTheSurface(const VEC<T>& Ps,const VEC<T>& Ps_1,const def::ORB Orb){
    /*
     Input:
        Ps  :[m,m,m] The First point at the line of state vector.
        Ps_1:[m,m,m] The point next to the point Ps.
     Return:
        The point projected to the Earth surface is perpendicular to the direction of vector(Ps_1-Ps).
    */
	    VEC<T> Psg=find::ProjectPoint(Ps,Orb);
	    VEC<T> uv = cross(Psg-Ps,Ps_1-Ps);
        VEC<T> uv_Psp=vec::cross(Ps_1-Ps,uv);
//#ifdef DEBUG
//	    cout<<"Psg="; Psg.Print();
//	    cout<<"uv="; uv.Print();
//	    cout<<"uv_Psp="; uv_Psp.Print();
//#endif
	    return find::BeamLocationOnSurface(uv_Psp,Ps,Orb);
	}

    template<typename T>
    VEC<T> find::MainBeamUniVector(const VEC<T>& Ps,const VEC<T>& Ps_1,const T& theta_l,const T& theta_sqc,const def::ORB Orb){
    /*
     Modified:
        (20090202) Modified from "Psg" to "Psp"
    */
        //VEC<T> Psg=sar::find::ProjectPoint(Ps);   // skew
        VEC<T> Psp = sar::find::VerticalPointOnTheSurface(Ps,Ps_1,Orb); // no skew
        return sar::find::LookAngleLineEq(Ps,Psp,Ps_1,theta_sqc,theta_l);
    }

	template<typename T>
    VEC<T> find::MainBeamUniVector(const VEC<T>& Ps,const VEC<T>& Ps_1,const T& theta_l,const T& theta_sqc,\
								   VEC<T>& D,const def::ORB Orb){
    /*
     Modified:
        (20090707) Modified LookAngleLineEq optional output
		(20100406) Remove the keyword uv_AD, uv_AE, E
    */
        //VEC<T> Psg=sar::find::ProjectPoint(Ps);   // skew
        VEC<T> Psp = sar::find::VerticalPointOnTheSurface(Ps,Ps_1,Orb); // no skew
        return sar::find::LookAngleLineEq(Ps,Psp,Ps_1,theta_sqc,theta_l,D);
    }

    template<typename T>
    double find::ThetaSqSqc(const VEC<T>& Ps,const VEC<T>& Psp,const VEC<T>& Ppy,const VEC<T>& nv){
    /*
     Purpose:
        Find the angle between the main beam to target location.
     Input:
        Ps  :[m,m,m] Satellite location
        Psp :[m,m,m] The point located by the intersection by main beam and Earth surface
        Ppy :[m,m,m] The target location
        nv  :[m,m,m] Main beam uni-vector
     Return:
        theta_sqsqc :[rad] angle between certain satellite position and certain point target locations
    */
        VEC<T> nv_plane=vec::cross(nv,Psp-Ps);
        double min_dis=vec::find::MinDistanceFromPointToPlane(nv_plane,Ps,Ppy);
        return asin(min_dis/(Ppy-Ps).abs()); // theta_sqsqc
    }

	template<typename T>
	double find::ThetaInc(const VEC<T>& Ps, const VEC<double>& Poc, const def::ORB& Orb, const bool IsRefGeodetic){
		if(IsRefGeodetic){
			// Reference to geodetic high
			// enlarge the height of Poc
			GEO<double> Pocd = ECR2Gd(Poc, Orb);
			Pocd.h() = 100;
			VEC<double> Poch = Gd2ECR(Pocd, Orb);
			return vec::angle(Ps - Poc, Poch - Poc);
		}else{
			// cosine law (reference to gencentric high)
			double n = Ps.abs();
			double a = (Ps - Poc).abs()/n;
			double b = Poc.abs()/n;
			double c = Ps.abs()/n;
			return deg2rad(180) - acos( (a*a + b*b - c*c)/(2*a*b) );
		}
	}
	
	template<typename T>
	double find::ThetaLook(const VEC<T>& Ps, const VEC<double>& Poc, const def::ORB& Orb, const bool IsRefGeodetic){
	/**
	 * Find look angle
	 * @param [in] Ps: [m,m,m] Sensor position
	 * @param [in] Poc: [m,m,m] Target position on the ground
	 */
		if(IsRefGeodetic){
			// Reference to geodetic high
			VEC<T> Pa, Psg = find::ProjectPoint(Ps, Orb);
			double dis = vec::find::MinDistanceFromPointToLine(Ps, Psg, Poc, Pa);
			return atan( dis / (Ps - Pa).abs() );
		}else{
			// cosine law (Reference to geocentric hight)
			double n = Ps.abs();
			double a = (Ps - Poc).abs()/n;
			double b = Ps.abs()/n;
			double c = Poc.abs()/n;
			return acos( (a*a + b*b - c*c)/(2*a*b) );
		}
	}
	
	template<typename T>
	void find::LocalAxis(const VEC<T>& Ps, const VEC<T>& Psc, VEC<T>& xuv, VEC<T>& yuv, VEC<T>& zuv, const def::ORB& Orb){
		VEC<T> Psg = sar::find::ProjectPoint(Ps, Orb);
		VEC<T> Pa;
		vec::find::MinDistanceFromPointToLine(Ps, Psg, Psc, Pa);
		
		GEO<T> Pscg = sar::ECR2Gd(Psc, Orb);
		Pscg.h() = 100;
		VEC<T> Psch = sar::Gd2ECR(Pscg, Orb);
		zuv  = Unit(Psch - Psc);				// local z-axis(normal) vector (plus angle is CCW aspect angle)
		yuv  = Unit(cross( zuv, Pa - Psc));		// local y-axis
		xuv  = Unit(cross(yuv, zuv));			// local x-axis
	}
	
	template<typename T>
	VEC<T> find::IntersectionPointOnPsPsgAlignTargetXAxis(const VEC<T>& Ps, const VEC<T>& Psc, const def::ORB& Orb){
		// Project point
		VEC<double> Psg = sar::find::ProjectPoint(Ps, Orb);
		// find local intersection
		VEC<double> Pa;
		double dis1 = vec::find::MinDistanceFromPointToLine(Ps, Psg, Psc, Pa);
		// find angle difference
		double diff_ang = sar::find::ThetaInc(Ps, Psc, Orb) - sar::find::ThetaLook(Ps, Psc, Orb);
		double dis2 = dis1 * tan(diff_ang);
		return Pa + dis2 * Unit(Ps - Pa);
	}
	
	
	/**
	 * Get Target's aspect angle trnasformation matrix in 3 by 3 matrix in D2<T> class.
	 *
	 * @param ASP : [rad] Target ASPect angle
	 * @param ASP_North : [rad] the angle between vector which projected from state vector and North
	 * @param Psg_gd : [rad,rad,m] Projected point from Ps at state vector
	 * @return Matrix : (3x3) Transformation matrix
	 */
    template<typename T>
    D2<T> find::ASPMatrix(const T& ASP,const T& ASP_North,const GEO<T>& Psg_gd){
    /*
     Input:
        ASP : [rad] Target ASPect angle
        ASP_North : [rad] the angle between vector which projected from state vector and North
        Psg_gd : [rad,rad,m] Projected point from Ps at state vector
     Return:
        Matrix : (3x3) Transformation matrix
    */
        T lon=Psg_gd.lon();
        T lat=Psg_gd.lat();
        return D2<T>( Rz(-ASP-ASP_North)*
					  Rx(-def_func::deg2rad(180))*Ry( sign(lat)*(def_func::deg2rad(90)+abs(lat)) )*
                      Rz(-lon));
    }

    template<typename T>
    double find::ASPFromNorth(const VEC<T>& Ps,const VEC<T>& Ps_1,const def::ORB Orb){
    /*
     Input:
        Ps  :[m,m,m] State vector is respect to scene center
        Ps_1:[m,m,m] Next to Ps point
     Return:
     	angle : [rad] measure from North. PLUS for couterclockwise and NEGTIVE for clockwise
    */
        VEC<T> Psg  =sar::find::ProjectPoint(Ps,Orb);
        VEC<T> Ps_1g=sar::find::ProjectPoint(Ps_1,Orb);
        GEO<T> Psg_gd = sar::ECR2Gd(Psg,Orb);

        D2<T> Matrix=sar::find::ASPMatrix(0.,0.,Psg_gd);
		
		
        T series[]={1000000.0,0.,0.};
        D2<T> tmp_north(series,1,3);
        D2<T> tmp_Pn = tmp_north*Matrix;

        VEC<T> Pn; //Point directed to North
        Pn.Setxyz( tmp_Pn[0][0]+Ps.x(),
                   tmp_Pn[0][1]+Ps.y(),
                   tmp_Pn[0][2]+Ps.z() );

        VEC<T> Png = sar::find::ProjectPoint(Pn,Orb);

        VEC<T> AA = Ps_1g-Psg;
        VEC<T> BB = Png-Psg;
        double angle=vec::angle(AA,BB);

		VEC<T> uv_N=vec::Unit(vec::cross(AA,BB));
        VEC<T> tmp=vec::find::PointFromDis(uv_N,Psg,100);
		

        return (tmp.abs() >= Psg.abs())? -1.*angle:1.*angle;
    }

    template<typename T>
    void find::RCSAngle(const VEC<T>& Ps,const VEC<T>& Ps_1,const VEC<T>& Poc,const T& ASP,double& theta_inc,double& phi,const def::ORB Orb){
    /*
     Input:
        Ps  : [m,m,m] (3) state vector
        Ps_1: [m,m,m] (3) state vector next to Ps
        Poc : [m,m,m] (3) Target Center point with ECR coordinate system
        ASP : [rad] the angle between target with NORTH
     Return:
      	out : [rad,rad] (theta_l,phi) target incdient angle and aspect angle
    */
        // ECR point (origin), first point -> Project to ground
        VEC<T> Psg=sar::find::ProjectPoint(Ps,Orb);
        // Find Min distance point
        T dist;
        VEC<T> A=vec::find::PointMinDistnace(Psg,Ps,Poc,dist);

        // ECR point (origin), second point -> Project to ground
        VEC<T> Psg_1=sar::find::ProjectPoint(Ps_1,Orb);
        // Find Min distance point
        VEC<T> B=vec::find::PointMinDistnace(Psg_1,Ps_1,Poc);

        // Find look angle
        double theta_l=asin(dist/(Ps-Poc).abs()); //[rad]

		// Find incidence angle
		GEO<T> Ps_gd = sar::ECR2Gd(Ps,Orb);

		double arc_satellite,arc_ground,Rn,theta_e;
		Geometry(theta_l,Ps_gd.lat(),Ps_gd.h(),arc_satellite,arc_ground,Rn,theta_e,theta_inc,Orb);
		

        // Find
        double alpha=vec::angle(Poc-A,B-A);//[rad]
        phi=def_func::deg2rad(90)-alpha;       //[rad]
        double new_phi=def_func::deg2rad(90)-(ASP+phi);

        phi = (new_phi > def_func::deg2rad(180))? new_phi-def_func::deg2rad(360):new_phi;
    }

    template<typename T>
    void find::RCSNearestAngle(const D1<T>& in_theta,const D1<T>& in_phi,const T& theta_val,const T& phi_val,
                               double& theta,double& phi){
    /*
     Input:
        in_theta : [deg] input uniq theta angle
        in_phi : [deg] input uniq phi angle
        theta_val : [deg] target incident angle
        phi_val : [deg] aspect angle
     Retrun:
        angle : [2] (theta,phi) the nearest angle comparing to the CSIST table
    */
        long idx1=0,idx2=0;
        double tmp1,torr1=9999999.;
        double tmp2,torr2=9999999.;
        for(long i=0;i<in_theta.GetNum();++i){
            tmp1 = abs(in_theta[i]-theta_val);
            if( tmp1 < torr1 ){
                //cout<<"torr1="<<torr1<<" ";
                idx1=i;
                torr1=tmp1;
                //cout<<"tmp1="<<tmp1<<" in_theta["<<i<<"]="<<in_theta[i]<<endl;
            }
		}
		for(long i=0;i<in_phi.GetNum();++i){
            tmp2 = abs(in_phi[i]-phi_val);
            if( tmp2 < torr2 ){
                idx2=i;
                torr2=tmp2;
            }
        }
        theta=in_theta[idx1];
        phi=in_phi[idx2];
    }
	
	template<typename T>
    void find::RCSNearestAngle(const D1<T>& in_theta,const D1<T>& in_phi,const T& theta_val,const T& phi_val,
                               double& theta,double& phi,long& idx1,long& idx2){
		/*
		 Input:
		 in_theta : [deg] input uniq theta angle
		 in_phi : [deg] input uniq phi angle
		 theta_val : [deg] target incident angle
		 phi_val : [deg] aspect angle
		 Retrun:
		 angle : [2] (theta,phi) the nearest angle comparing to the CSIST table
		 */
        idx1=0;
		idx2=0;
        double tmp1,torr1=9999999.;
        double tmp2,torr2=9999999.;
        for(long i=0;i<in_theta.GetNum();++i){
            tmp1 = abs(in_theta[i]-theta_val);
            if( tmp1 < torr1 ){
                //cout<<"torr1="<<torr1<<" ";
                idx1=i;
                torr1=tmp1;
                //cout<<"tmp1="<<tmp1<<" in_theta["<<i<<"]="<<in_theta[i]<<endl;
            }
		}
		for(long i=0;i<in_phi.GetNum();++i){
            tmp2 = abs(in_phi[i]-phi_val);
            if( tmp2 < torr2 ){
                idx2=i;
                torr2=tmp2;
            }
        }
        theta=in_theta[idx1];
        phi=in_phi[idx2];
    }

    template<typename T>
    void find::RCSFile(const char* dir,const D2<T>& table,const long rcs_num,const double& theta_o,const double& phi_o,
                       const char* pol,D1<CPLX<double> >& out){
    /*
     Input:
        dir     : directory of RCS data
        table   : RCS table
        theta_o : [deg] incident angle
        phi_o   : [deg] aspect angle
        pol     : [char] polization {"HH","HV","VH","VV"}
     Return:
        out     : ( D1<CPLX<double> > )
    */
        //char dir[]="G:\\Chiang_Documents\\Data\\MD80_35\\";
        long num=table.GetM();
        long idx=99999;
        for(long i=0;i<num;++i){
            if((table[i][1]==theta_o) && (table[i][2]==phi_o)){
                idx=long(table[i][0]);
            }
        }
        //cout<<idx<<endl;

        // Finish filename -> "0xxxx.dat"
        string filename = string(dir) + def_func::RCS_num2str(idx) +string(".dat");
        //cout<<filename<<endl;

        RCS<double> rcs(rcs_num);
        io::read::CSISTCutRCS(filename.c_str(),rcs_num,rcs);

        //D1<CPLX<double> > out(rcs_num);
        long idx_pol=999;
        string str_pol(pol);
        if(str_pol==string("HH")){
            idx_pol=0;
        }else if(str_pol==string("HV")){
            idx_pol=1;
        }else if(str_pol==string("VH")){
            idx_pol=2;
        }else if(str_pol==string("VV")){
            idx_pol=3;
        }else{
            cout<<"ERROR::(sar::find::RCSFile)No this type of pol!"<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
        for(long i=0;i<rcs_num;++i){
            out[i]=CPLX<double>(rcs.GetVal(i,idx_pol));
            //out[i].Print();
        }
    }

	template<typename T> 
	void find::Geometry(const T theta_l,const T lat_gd,const T height, // input
					    T& arc_satellite,T& arc_ground,T& Rn,T& theta_e,T& theta_i, //output
						const def::ORB Orb){ // input
	/*
     Input:
        theta_l:    [rad] look angle
		 lat_gd:    [rad] geodetic latitude
		   N_gd:    [m] Earth local radius
	     height:    [m] Satellite local height
     Return:
        ARC_SATELLITE:    [m] Arc distance at satellite orbit
		   ARC_GROUND:    [m] Arc distance at ground
		           Rn:    [m] Distance between satellite and Earth surface at beam center
			  THETA_E:    [rad] angle between satellit & target at center of Earth
		      THETA_I:    [rad] incident angle at elevation = 0
    */
		double Re = orb::RadiusCurvatureMeridian(lat_gd,Orb);
		double H = height;
		theta_i = asin( (H+Re)*sin(theta_l)/Re );	//[rad] incident angle
		theta_e = double(theta_i) - theta_l;	//[rad] angle between satellit & target at center of Earth

		Rn = Re*sin(theta_e)/sin(theta_l);	//[m] distance from satellite to ground point
		arc_satellite = (Re+H)*theta_e;		//[m] distance of arc @ satellite
		arc_ground = Re*theta_e;			//[m] distance of arc @ ground
	}

    
	template<typename T>
	T find::SlantRange(const VEC<T>& Ps,const VEC<T>& Ps_1,const T theta_sq,const T theta_l,const def::ORB Orb){
	/*
	 Purpose:
		Find Slant range distance from State vector(with next one) and look angle
     Input:
        Ps		: [m,m,m] (3) state vector
        Ps_1	: [m,m,m] (3) state vector next to Ps
		theta_sq: [rad] squint angle
        theta_l : [rad] look angle
     Return:
      	out : [rad,rad] Destinated slant range
	 Modified:
		(20100406) CYChiang Replace VerticalPointOnTheSurface() function.
    */
		VEC<T> Psp    = sar::find::VerticalPointOnTheSurface(Ps,Ps_1,Orb);
		//VEC<T> Psg    = sar::find::ProjectPoint(Ps);
		//VEC<T> uv     = vec::Unit( vec::cross(Psg-Ps,Ps_1-Ps) );
		//VEC<T> uv_Psp = vec::Unit( vec::cross(Ps_1-Ps,uv) );
		//VEC<T> Psp    = sar::find::BeamLocationOnSurface(uv_Psp,Ps);
		VEC<T> uv_look= sar::find::LookAngleLineEq(Ps,Psp,Ps_1,theta_sq,theta_l);
		VEC<T> Psl    = sar::find::BeamLocationOnSurface(uv_look,Ps,Orb);
		
		return T( (Ps-Psl).abs() );
	}

	template<typename T>
	T find::SlantRange(const VEC<T>& Ps,const VEC<T>& Ps_1,const T theta_sq,const T theta_l,const def::ORB Orb,VEC<T>& Psl){
	/*
	 Purpose:
		Find Slant range distance from State vector(with next one) and look angle
     Input:
        Ps		: [m,m,m] (3) state vector
        Ps_1	: [m,m,m] (3) state vector next to Ps
		theta_sq: [rad] squint angle
        theta_l : [rad] look angle
     Return:
      	out : [rad,rad] Destinated slant range
	 Modified:
		(20100406) CYChiang Replace VerticalPointOnTheSurface() function.
    */
		VEC<T> Psp    = sar::find::VerticalPointOnTheSurface(Ps,Ps_1,Orb);
		//VEC<T> Psg    = sar::find::ProjectPoint(Ps);
		//VEC<T> uv     = vec::Unit( vec::cross(Psg-Ps,Ps_1-Ps) );
		//VEC<T> uv_Psp = vec::Unit( vec::cross(Ps_1-Ps,uv) );
		//VEC<T> Psp    = sar::find::BeamLocationOnSurface(uv_Psp,Ps);
		VEC<T> uv_look= sar::find::LookAngleLineEq(Ps,Psp,Ps_1,theta_sq,theta_l);
		Psl    = sar::find::BeamLocationOnSurface(uv_look,Ps,Orb);	
		return T( (Ps-Psl).abs() );
	}


	
	template<typename T>
	VEC<T> find::SceneCenter(const VEC<T>& Ps,const VEC<T>& Ps_1,const def::SAR Sar,const def::ORB Orb){
	/*
	 Purpose:
		Find the Scene center within a range of look angle
	 Input:
	  	Ps				:[m,m,m] state vector
	 	Ps_1			:[m,m,m] state vector next to Ps
	 	theta_sqc		:[rad] squint angle
	 	theta_l_range	:[rad] look angle range[min, max]
	 Return:
	 	Scene center location
	*/
		double sl_min = sar::find::SlantRange(Ps,Ps_1,Sar.theta_sqc(),Sar.theta_l_min(),Orb);
		double sl_max = sar::find::SlantRange(Ps,Ps_1,Sar.theta_sqc(),Sar.theta_l_max(),Orb);
		double sl_c   = (sl_min + sl_max)/2.; // rref(MSP)

		VEC<double> Psg   = sar::find::ProjectPoint(Ps,Orb);
		GEO<double> Ps_gd = sar::ECR2Gd(Ps,Orb);
		double height = Ps.abs();
		double H = Ps_gd.h();
		double Re= height - H;
	
		double theta_l = acos( (height*height + sl_c*sl_c - Re*Re)/(2.*height*sl_c) );

		VEC<double> uv = vec::Unit( vec::cross(Psg-Ps, Ps_1-Ps) );
		VEC<double> uv_Psp = vec::Unit( vec::cross(Ps_1-Ps, uv) );
		VEC<double> Psp = sar::find::BeamLocationOnSurface(uv_Psp, Ps, Orb);
		VEC<double> uv_Psl = sar::find::LookAngleLineEq(Ps, Psp, Ps_1, Sar.theta_sqc(), theta_l);
		VEC<double> Psl = sar::find::BeamLocationOnSurface(uv_Psl, Ps, Orb);
		//GEO<double> Psl_gd = sar::ECR2Gd(Psl);

		return Psl;
	}

	template<typename T>
	VEC<T> find::SceneCenter(const VEC<T>& Ps,const VEC<T>& Ps_1,const def::SAR Sar,const def::ORB Orb, // input
						      GEO<T>& Psl_gd,double& sl){ // output
	/*
	 Purpose:
		Find the Scene center within a range of look angle
	 Input:
	  	Ps				:[m,m,m] state vector
	 	Ps_1			:[m,m,m] state vector next to Ps
	 	theta_sqc		:[rad] squint angle
	 	theta_l_range	:[rad] look angle range[min, max]
	 Output:
		Psl_gd			:[rad,rad,m] geolocation of Psl
		sl				:[m] Slant rnage between Ps and Psl
	 Return:
	 	Scene center location
	*/
		double sl_min = sar::find::SlantRange(Ps,Ps_1,Sar.theta_sqc(),Sar.theta_l_min(),Orb);
		double sl_max = sar::find::SlantRange(Ps,Ps_1,Sar.theta_sqc(),Sar.theta_l_max(),Orb);
		double sl_c   = (sl_min + sl_max)/2.; // rref(MSP)

		VEC<double> Psg   = sar::find::ProjectPoint(Ps,Orb);
		GEO<double> Ps_gd = sar::ECR2Gd(Ps,Orb);
		double height = Ps.abs();
		double H = Ps_gd.h();
		double Re= height - H;

		double theta_l = acos( (height*height + sl_c*sl_c - Re*Re)/(2.*height*sl_c) );

		VEC<double> uv = vec::Unit( vec::cross(Psg-Ps,Ps_1-Ps) );
		VEC<double> uv_Psp = vec::Unit( vec::cross(Ps_1-Ps,uv) );
		VEC<double> Psp = sar::find::BeamLocationOnSurface(uv_Psp,Ps,Orb);
		
		VEC<double> uv_Psl = sar::find::LookAngleLineEq(Ps,Psp,Ps_1,Sar.theta_sqc(),theta_l);



		VEC<double> Psl = sar::find::BeamLocationOnSurface(uv_Psl,Ps,Orb);
		Psl_gd = sar::ECR2Gd(Psl,Orb);
		sl = sl_c;

		return Psl;
	}

	template<typename T>
	D1<T> find::SARFindInsFreqOfMatchedFilter(const T f_nc_abs, const T PRF, const long num_az, long shift){
	/*
	 Purpose:
		Find the instataneous frequency of matching filter
	 Input:
		f_nc_abs	:[Hz] absoluted doppler centroid
		PRF			:[Hz] Pulse Repeat Frequency
		num_az		:[x] number of pixel at azimuth direction
	 Output:
		shift		:[pix] Return shift pixels
	*/
		// Find Instantaneous frequency of matched filter (fig 3.12)
		D1<T> fn(num_az);
		linspace(f_nc_abs-PRF/2.0, f_nc_abs+PRF/2.0, fn);
		fftshift(fn);
		// find the Frac
		double Frac = mat::Mod(f_nc_abs, PRF);
		shift = long(Frac * double(num_az)/PRF);
		mat::shift(fn, shift);
		return fn;
	}
	
//	template<typename T>
//	void find::RCM(const D1<T> fn, const D1<T> R0_range, const D1<T> Vr_range, const T Kr, const T Tr, const T lambda, D2<T>& RCM){
//	/*
//	 Purpose:
//		Find the Range Cell Migration
//	 Input:
//		fn			:[Hz](vector) Instantaneous frequency of matched filter
//		R0_range	:[m](vector) Slant range at Doppler plane
//		Vr_range	:[m/s](vector) Effective velocity (function of range)
//		Kr			:[Hz/s] Chirp rate
//		Tr			:[sec] duration time
//		lambda		:[m] wavelength
//	 Output:
//		RCM			:[pix](num_az, num_rg) The array with RCM value
//	*/
//		// Find RCMC
//		double c = 299792458.0;	//[m/sec]
//		long num_az = fn.GetNum();
//		long num_rg = R0_range.GetNum();
//		double rho_r = 1.0/(abs(Kr)*Tr)*(c/2.0);
//		for(long j=0;j<num_az;++j){
//			for(long i=0;i<num_rg;++i){
//				RCM[j][i] = -lambda*lambda*R0_range[i]*fn[j]*fn[j]/
//							 (8.0*Vr_range[i]*Vr_range[i]) / rho_r;
//			}
//		}
//	}
	
	template<typename T>
	void find::RCM(const D1<T> fn, const D1<T> R0_range, const D1<T> Vr_range, const T Fr, const T lambda, D2<T>& RCM){
	/*
	 Purpose:
		Find the Range Cell Migration
	 Input:
		fn			:[Hz](vector) Instantaneous frequency of matched filter
		R0_range	:[m](vector) Slant range at Doppler plane
		Vr_range	:[m/s](vector) Effective velocity (function of range)
		Kr			:[Hz/s] Chirp rate
		Tr			:[sec] duration time
		lambda		:[m] wavelength
	 Output:
		RCM			:[pix](num_az, num_rg) The array with RCM value
	*/
		// Find RCMC
		double c = 299792458.0;	//[m/sec]
		long num_az = fn.GetNum();
		long num_rg = R0_range.GetNum();
//		double rho_r = 1.0/(abs(Kr)*Tr)*(c/2.0);
//		double rho_r = c/( 2 * (abs(Kr)*Tr));
		double sp_r  = c/( 2 * Fr);
		for(long j=0;j<num_az;++j){
			for(long i=0;i<num_rg;++i){
				RCM[j][i] = -lambda*lambda*R0_range[i]*fn[j]*fn[j]/
							 (8.0*Vr_range[i]*Vr_range[i]) / sp_r;
//							 (8.0*Vr_range[i]*Vr_range[i]) / rho_r;
			}
		}
	}
	
	
	//
    // Other
    //
    template<typename T>
    T Wr(const T& t,const T& Tr){
    /*
     Purpose:
        Slant Range Rectangular function
    */
        T val=t/Tr;
        return T((val >= -0.5)&&(val <= 0.5));
    }

    template<typename T>
    D1<T> Wr(const D1<T>& t,const T& Tr){
    /*
     Prupose:
        Slant Range Rectangular function
    */
        long num=t.GetNum();
        T val;
        D1<T> out(num);
        for(long i=0;i<num;++i){
            val=t[i]/Tr;
            out[i]=double((val >= -0.5)&&(val <= 0.5));
        }
        return out;
    }

    template<typename T>
    double Waz(const T& theta_sq_sqc,const T& theta_az){
    /*
     Purpose:
        Find the radar antenna pattern at azimuth direction.
     Input:
        theat_sq_sqc:[rad] the squint angle
        theat_az    :[rad] the azimuth beamwidth
     Return:
        Antenna spectrum at azimuth direction.
    */
        T val=sinc(0.886*theta_sq_sqc/theta_az);
        return (val*val);
    }
	
	template<typename T>
    double Wev(const T& theta_l_lc,const T& theta_ev){
		/*
		 Purpose:
		 Find the radar antenna pattern at azimuth direction.
		 Input:
		 theta_l_lc	 :[rad] angle difference between target and antenna main beam at elevation
		 theta_ev    :[rad] beamwidth at elevation
		 Return:
		 Antenna spectrum at azimuth direction.
		 */
		T val=sinc(0.886*theta_l_lc/theta_ev);
        return (val*val);
    }

    template<typename T>
    D1<long> InitPpyRCS(const char* RCS_dir,const D2<T>& table,const T& theta_o,const T& phi_o, const char* pol, // input
                        const VEC<T>& Poc,const double TR_ASP,const double ASP_North,const def::ORB Orb, // input
                        D1<CPLX<T> >& RCS_val,D1<VEC<T> >& Ppy){ //return
    /*
     Purpose:
        Initialize the Ppy(Mean polygon location) & PCS_val(RCS value for each polygon)
     Input:
        RCS_dir : RCS directory
        table   : RCS table
        theta_o : incident angle of beam to target (Nearest)
        phi_o   : aspect angel with NORTH (Nearest)
        pol     : (char*) polization type {"HH","HV","VH","VV"}
        Poc     : [m,m,m](VEC class) Scene center location
        TR_ASP  : [rad] the angle of target with repective satellite trajection
        ASP_North  : [rad] the angle of state vector and NORTH
     Return:
        series  : (D1<long>)(default) the index series which RCS condition suited to critial.
        RCS_val : (D1<CPLX<T> >) destination RCS value
        Ppy     : the mean location of triangle polygon
    */
        long rcs_num = RCS_val.GetNum();
        sar::find::RCSFile(RCS_dir,table,rcs_num,theta_o,phi_o,pol,RCS_val);

        // Get Transformation matrix
        GEO<double> Poc_gd = sar::ECR2Gd(Poc,Orb);
        D2<double> Matrix = sar::find::ASPMatrix(TR_ASP,ASP_North,Poc_gd);

        for(long k=0;k<rcs_num;++k){
            Ppy[k] = sar::VecTransform(Ppy[k],Matrix,Poc);
			//if(k==0){
			//	cout<<"Ppy[0].Print()="<<endl; Ppy[0].Print();
			//}
        }

        // Find suitable RCS
        long count=0;
        for(long i=0;i<rcs_num;++i){
            if( (RCS_val[i].r() >= 1e-9)||(RCS_val[i].i() >= 1e-9) ){
                count++;
            }
        }

        // Returen value
        D1<long> series(count);
        count=0;
        for(long i=0;i<rcs_num;++i){
            if( (RCS_val[i].r() >= 1e-9)||(RCS_val[i].i() >= 1e-9) ){
                series[count]=i;
//#ifdef DEBUG
//                cout<<"i="<<i<<endl;
//#endif
                count++;
            }
        }
        return series;
    }

	void GetMapCoordinate(const SAR& Sar, const def::ORB& Orb, VEC<double>& Ps, VEC<double>& Ps1,	// input
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


	//
    // Write
    //
    SAR_PAR Init_SAR_par(const char* sar_title,const char* sar_sensor,const char* sar_chirp,const char* sar_mode,const char* sar_type,
                         const char* sar_spectrum,const char* sar_ant_pattern,const long num_rg,const def::SAR Sar){
    /*
     Purpose:
        Initialize the SAR sensor parameters
    */
        par::SAR_PAR sarp;
        strcpy(sarp.title,  sar_title);
        strcpy(sarp.sensor, sar_sensor);
        strcpy(sarp.up_down,sar_chirp);
        strcpy(sarp.s_mode, sar_mode);
        strcpy(sarp.s_type, sar_type);
        strcpy(sarp.spectrum,sar_spectrum);
        sarp.fc = Sar.f0();
        sarp.bw = Sar.BWrg();
        sarp.plen = Sar.Tr();
        sarp.fs = Sar.Fr();
        sarp.file_hdr_sz = 0;
        sarp.nbyte = int(num_rg*8);
        sarp.byte_h = 0;
        sarp.ns_echo = int(num_rg);
        sarp.az_ant_bw = def_func::rad2deg(Sar.theta_az());
        sarp.r_ant_bw = def_func::rad2deg(Sar.theta_rg());
        sarp.az_ang = 90.;
        //sarp.lk_ang = rad2deg(Sar.theta_AMB());
		sarp.lk_ang = def_func::rad2deg(Sar.theta_l_MB());
        sarp.pitch = 0.;
        strcpy(sarp.antpatf,sar_ant_pattern);

        return sarp;
    }

    template<typename T>
    PROC_PAR Init_PROC_par(const char* procp_title,bool procp_SRC,bool procp_deskew,
						   const def::SAR Sar,const def::ORB Orb, const def::EF Ef,
                           const VEC<T>& Poc,const T& t0,const VEC<T>& Ps0,const SV<T>& TER_SV,
                           const T tt_min_rg,const T tt_max_rg,const long num_az){
    /*
     Purpose:
        Initialize the Processing parameters
    */
        PROC_PAR procp;

        GEO<double> Poc_gd=sar::ECR2Gd(Poc,Orb);
        //VEC<double> Psg_0=INT.GetPos()[0];
        VEC<double> Psg_0=Ps0;
        VEC<double> Psg_0_p=sar::find::ProjectPoint(Psg_0,Orb);
        double altitude = (Psg_0-Psg_0_p).abs(); // rough estimation
        //double velocity = Vel0.abs(); // rough estimation
        double R_near = tt_min_rg*def::C/2.;
        double R_center = (tt_min_rg+tt_max_rg)/2.*def::C/2.;
        double R_far = tt_max_rg*def::C/2.;

        procp.MXST = 8;
        procp.nstate = 5;

        // date & time
        TIME d_t=GPS2UTC(t0);
        string d_t_y = num2str(d_t.GetYear());
        string d_t_mo= num2str(d_t.GetMonth());
        string d_t_d = num2str(d_t.GetDay());
        string d_t_h = num2str(d_t.GetHour());
        string d_t_m = num2str(d_t.GetMin());
        string d_t_s = num2str(d_t.GetSec());

        string procp_date = (d_t_d+string(" ")+d_t_mo+string(" ")+d_t_y+string("\0"));
        string procp_time = (d_t_h+string(" ")+d_t_m+string(" ")+d_t_s+string("\0"));

        strcpy(procp.title, procp_title);
        strcpy(procp.date, procp_date.c_str());
        strcpy(procp.time, procp_time.c_str());
        strcpy(procp.pol_ch, Ef.Pol().c_str());
        procp.el_major = Orb.E_a();
        procp.el_minor = Orb.E_b();
        procp.lat = def_func::rad2deg(Poc_gd.lat());
        procp.lon = def_func::rad2deg(Poc_gd.lon());
        procp.track = 0.;
        procp.alt = altitude;
        procp.terra_alt = 0.;
        procp.pos = VEC<double>(0.,0.,altitude);
        //procp.vel = VEC<double>(velocity,0.,0.);
        procp.vel = VEC<double>(0,0.,0.);
        procp.acc = VEC<double>(0.,0.,0.);
        procp.prf = Sar.PRF();
        procp.I_bias = 0.;
        procp.Q_bias = 0.;
        procp.I_sigma = 0.;
        procp.Q_sigma = 0.;
        procp.IQ_corr = 0.;
        procp.SNR_rspec = 0.;
        procp.DAR_dop = 0.;
        procp.DAR_snr = 0.;
        procp.fdp=D1<double>(4); procp.fdp.SetZero();
        procp.td = tt_min_rg;
        procp.rx_gain = 0.;
        procp.cal_gain = 0.;
        procp.r0 = R_near;  // Raw near slant range
        procp.r1 = R_center;// Raw center slant range
        procp.r2 = R_far;   // Raw far slant range
        procp.ir0 = R_near;
        procp.ir1 = R_center;
        procp.ir2 = R_far;
        procp.rpixsp = 0.;
        procp.ran_res = 0.;
        if(procp_SRC == true){
            strcpy(procp.sec_range_mig, "ON\0");
        }else{
            strcpy(procp.sec_range_mig, "OFF\0");
        }
        if(procp_deskew == true){
            strcpy(procp.az_deskew, "ON\0");
        }else{
            strcpy(procp.az_deskew, "OFF\0");
        }
        procp.autofocus_snr = 0.;
        procp.prfrac = 0.8;
        procp.nprs_az = 1;
        procp.loff = 0;
        procp.nl = int(num_az);
        procp.nr_offset = 0;
        procp.nrfft = 8192;
        procp.nlr = 1;
        procp.nlaz = 1;
        procp.azoff = 0.;
        procp.azimsp = 0.;
        procp.azimres = 0.;
        procp.nrs = 0;
        procp.nazs = 0;
        procp.sensor_lat = 0.;
        procp.sensor_lon = 0.;
        procp.sensor_track = 0.;

        procp.map=D1<GEO<double> >(5);

        // Input state vector
        procp.state=SV<double>(procp.nstate);
        int SV_space = int(TER_SV.GetNum()/procp.nstate);
        int first_point = (int)mat::round( double(int(TER_SV.GetNum())-SV_space*procp.nstate)/2.0 );
        //cout<<"first_point="<<first_point<<endl;

        D1<double> state_t_res(procp.nstate);
        D1<double> state_t = TER_SV.t();
        D1<VEC<double> > state_pos_res(procp.nstate);
        D1<VEC<double> > state_pos = TER_SV.pos();
        D1<VEC<double> > state_vel_res=(procp.nstate);
        D1<VEC<double> > state_vel = TER_SV.vel();
#ifdef DEBUG
        cout<<"TER_SV.GetNum()="<<TER_SV.GetNum()<<endl;
        cout<<"SV_space="<<SV_space<<" first_point="<<first_point<<endl;
#endif

        for(long i=0;i<procp.nstate;++i){
#ifdef DEBUG
            cout<<"idx="<<first_point+i*SV_space;
            cout<<" state_t[i]="<<state_t[first_point+i*SV_space]<<endl;
#endif
            state_t_res[i]   = state_t[first_point+i*SV_space];
            state_pos_res[i] = state_pos[first_point+i*SV_space];
            state_vel_res[i] = state_vel[first_point+i*SV_space];
        }

        //procp.t_state = state_t_res[0]; // need convert hr*3600 + min*60 + sec
        //procp.t_state = t_state;//35928;
        //35946.000; // 2007/10/13 9:59:6.00 (first state vector time) 3600*hrs+60*min+sec
        TIME tmp_time = GPS2UTC(state_t_res[0]);
        procp.t_state = 3600.*tmp_time.GetHour() + 60.*tmp_time.GetMin() + tmp_time.GetSec();
#ifdef DEBUG
        cout<<state_t_res[0]<<endl;
        cout<<procp.t_state<<endl;
#endif
        procp.tis = state_t_res[1]-state_t_res[0];
        procp.state.SetAll(state_t_res,state_pos_res,state_vel_res,procp.nstate);

        return procp;
    }

    void write_SAR_par(const char* filename,const SAR_PAR& sarp){
    /*
     Purpose:
        subroutine to write the SAR sensor parameters (with keywords)
        Gamma Remote Sensing v2.0 22-Jul-97 clw
    */
        FILE* sarpf_out;
#ifdef _MSC_VER
		errno_t err;
		err = fopen_s(&sarpf_out,filename,"w");
		if(err!=0){
            cout<<"ERROR::[write_SAR_par] The output file is error! -> ";
            cout<<filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
#else
		sarpf_out = fopen(filename,"w");
		if(sarpf_out==NULL){
            cout<<"ERROR::[write_SAR_par] The output file is error! -> ";
            cout<<filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
#endif

        rewind(sarpf_out);				/* return to start of file and write out */
        fprintf(sarpf_out,"MRSL MSP 2008- Parameter file\n");
        fprintf(sarpf_out,"Contains parameters characterizing the SAR Sensor \n\n");
        fprintf(sarpf_out,"title:                         %s\n",sarp.title);
        fprintf(sarpf_out,"sensor_name:                   %s\n",sarp.sensor);
        fprintf(sarpf_out,"chirp_direction:               %s\n",sarp.up_down);
        fprintf(sarpf_out,"receiver_adc_mode:             %s\n",sarp.s_mode);
        fprintf(sarpf_out,"sample_type:                   %s\n",sarp.s_type);
        fprintf(sarpf_out,"receiver_spectrum_type:        %s\n",sarp.spectrum);
        fprintf(sarpf_out,"SAR_center_frequency:          %13.6e  Hz\n",sarp.fc);
        fprintf(sarpf_out,"chirp_bandwidth:               %13.6e  Hz\n",sarp.bw);
        fprintf(sarpf_out,"chirp_duration:                %13.6e   s\n",sarp.plen);
        fprintf(sarpf_out,"ADC_sampling_frequency:        %13.6e  Hz\n",sarp.fs);
        fprintf(sarpf_out,"file_header_size:              %6d         bytes\n",sarp.file_hdr_sz);
        fprintf(sarpf_out,"record_length:                 %6d         bytes\n",sarp.nbyte);
        fprintf(sarpf_out,"record_header_size:            %6d         bytes\n",sarp.byte_h);
        fprintf(sarpf_out,"samples_per_record:            %6d\n",sarp.ns_echo);
        fprintf(sarpf_out,"antenna_azimuth_3dB_beamwidth: %10.4f     degrees\n",sarp.az_ant_bw);
        fprintf(sarpf_out,"antenna_range_3dB_beamwidth:   %10.4f     degrees\n",sarp.r_ant_bw);
        fprintf(sarpf_out,"nominal_antenna_azimuth_angle: %10.4f     degrees\n",sarp.az_ang);
        fprintf(sarpf_out,"nominal_antenna_look_angle:    %10.4f     degrees\n",sarp.lk_ang);
        fprintf(sarpf_out,"nominal_platform_pitch_angle:  %10.4f     degrees\n",sarp.pitch);
        fprintf(sarpf_out,"antenna_pattern_filename:      %s\n",sarp.antpatf);
        fprintf(sarpf_out,"\n*************** END OF SENSOR PARAMETERS ******************\n");
		fclose(sarpf_out);
    }

    void write_PROC_par(const char* filename,const PROC_PAR& procp){
    /*
     Purpose:
        subroutine to write the MSP processing parameters
        clw 4-Nov-97 v2.1 Gamma Remote Sensing
    */
        FILE* procpf;
#ifdef _MSC_VER
		errno_t err;
		err = fopen_s(&procpf,filename,"w");
		if(err!=0){
            cout<<"ERROR::[write_SAR_par] The output file is error! -> ";
            cout<<filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
#else
		procpf = fopen(filename,"w");
		if(procpf==NULL){
            cout<<"ERROR::[write_SAR_par] The output file is error! -> ";
            cout<<filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
#endif

        rewind(procpf);
        fprintf(procpf,"WaveFidelity SAR processor 2020 Parameter File\n");
        fprintf(procpf,"Contains parameters configuring/documenting the SAR Processing\n\n");
        fprintf(procpf,"title:                  %s\n",procp.title);
        fprintf(procpf,"date:                   %s\n",procp.date);
        fprintf(procpf,"raw_data_start_time:    %s\n",procp.time);
        fprintf(procpf,"channel/mode:           %s\n",procp.pol_ch);

        fprintf(procpf,"earth_semi_major_axis:        %12.4f   m\n",procp.el_major);
        fprintf(procpf,"earth_semi_minor_axis:        %12.4f   m\n",procp.el_minor);
        fprintf(procpf,"scene_center_latitude:        %12.6f   decimal degrees\n",procp.lat);
        fprintf(procpf,"scene_center_longitude:       %12.6f   decimal degrees\n",procp.lon);
        fprintf(procpf,"track_angle:                  %12.6f   degrees\n",procp.track);
        fprintf(procpf,"platform_altitude:            %12.4f   m\n",procp.alt);
        fprintf(procpf,"terrain_height:               %12.4f   m\n",procp.terra_alt);
        fprintf(procpf,"sensor_position_vector:     %14.6f %14.6f %14.6f   m     m      m\n",
                procp.pos.x(), procp.pos.y(), procp.pos.z());
        fprintf(procpf,"sensor_velocity_vector:     %14.6f %14.6f %14.6f   m/s   m/s   m/s  \n",
                procp.vel.x(), procp.vel.y(), procp.vel.z());
        fprintf(procpf,"sensor_acceleration_vector: %14.6f %14.6f %14.6f   m/s^2 m/s^2 m/s^2\n",
                procp.acc.x(), procp.acc.y(), procp.acc.z());
        fprintf(procpf,"pulse_repetition_frequency: %14.6f   Hz\n",procp.prf);
        fprintf(procpf,"I_bias:                     %14.6f\n",procp.I_bias);
        fprintf(procpf,"Q_bias:                     %14.6f\n",procp.Q_bias);
        fprintf(procpf,"I_sigma:                    %14.6f\n",procp.I_sigma);
        fprintf(procpf,"Q_sigma:                    %14.6f\n",procp.Q_sigma);
        fprintf(procpf,"IQ_corr:                    %14.6f\n",procp.IQ_corr);

        fprintf(procpf,"SNR_range_spectrum:         %14.3f\n",procp.SNR_rspec);
        fprintf(procpf,"DAR_doppler:                %14.3f   Hz\n",procp.DAR_dop);
        fprintf(procpf,"DAR_snr:                    %14.3f\n",procp.DAR_snr);

        fprintf(procpf,"doppler_polynomial:  %12.5e %12.5e %12.5e %12.5e   Hz Hz/m Hz/m^2 Hz/m^3\n",
                procp.fdp[0],procp.fdp[1],procp.fdp[2],procp.fdp[3]);
        fprintf(procpf,"echo_time_delay:            %14.6e   s\n",procp.td);

        fprintf(procpf,"receiver_gain:              %14.4f   dB\n",procp.rx_gain);
        fprintf(procpf,"calibration_gain:           %14.4f   dB\n",procp.cal_gain);

        fprintf(procpf,"near_range_raw:             %14.4f   m\n",procp.r0);
        fprintf(procpf,"center_range_raw:           %14.4f   m\n",procp.r1);
        fprintf(procpf,"far_range_raw:              %14.4f   m\n",procp.r2);

        fprintf(procpf,"near_range_slc:             %14.4f   m\n",procp.ir0);
        fprintf(procpf,"center_range_slc:           %14.4f   m\n",procp.ir1);
        fprintf(procpf,"far_range_slc:              %14.4f   m\n",procp.ir2);

        fprintf(procpf,"range_pixel_spacing:          %12.6f   m\n",procp.rpixsp);
        fprintf(procpf,"range_resolution:             %12.3f   m\n",procp.ran_res);

        fprintf(procpf,"sec_range_migration:                 %s\n",procp.sec_range_mig);
        fprintf(procpf,"azimuth_deskew:                      %s\n",procp.az_deskew);
        fprintf(procpf,"autofocus_snr:                    %8.4f\n",procp.autofocus_snr);
        fprintf(procpf,"azimuth_bandwidth_fraction:       %8.4f\n",procp.prfrac);
        fprintf(procpf,"azimuth_presum_factor:              %6d\n", procp.nprs_az);
        fprintf(procpf,"offset_to_first_echo_to_process:    %6d   echoes\n",procp.loff);
        fprintf(procpf,"echoes_to_process:                  %6d   echoes\n",procp.nl);
        fprintf(procpf,"range_offset:                       %6d   samples\n",procp.nr_offset);
        fprintf(procpf,"range_fft_size:                     %6d\n",procp.nrfft);
        fprintf(procpf,"range_looks:                        %6d   looks\n",procp.nlr);
        fprintf(procpf,"azimuth_looks:                      %6d   looks\n",procp.nlaz);
        fprintf(procpf,"azimuth_offset:               %12.5f   s\n", procp.azoff);
        fprintf(procpf,"azimuth_pixel_spacing:        %12.6f   m\n", procp.azimsp);
        fprintf(procpf,"azimuth_resolution:           %12.3f   m\n", procp.azimres);
        fprintf(procpf,"range_pixels:                       %6d   samples\n",procp.nrs);
        fprintf(procpf,"azimuth_pixels:                     %6d   lines\n",procp.nazs);
        fprintf(procpf,"sensor_latitude:              %12.6f   decimal degrees\n",procp.sensor_lat);
        fprintf(procpf,"sensor_longitude:             %12.6f   decimal degrees\n",procp.sensor_lon);
        fprintf(procpf,"sensor_track_angle:           %12.6f   decimal degrees\n",procp.sensor_track);

        fprintf(procpf,"map_coordinate_1:             %12.7f   %12.7f   %12.4f   deg.  deg.  m\n",
                rad2deg(procp.map[0].lat()), rad2deg(procp.map[0].lon()), procp.map[0].h());
        fprintf(procpf,"map_coordinate_2:             %12.7f   %12.7f   %12.4f   deg.  deg.  m\n",
				rad2deg(procp.map[1].lat()), rad2deg(procp.map[1].lon()), procp.map[1].h());
        fprintf(procpf,"map_coordinate_3:             %12.7f   %12.7f   %12.4f   deg.  deg.  m\n",
				rad2deg(procp.map[2].lat()), rad2deg(procp.map[2].lon()), procp.map[2].h());
        fprintf(procpf,"map_coordinate_4:             %12.7f   %12.7f   %12.4f   deg.  deg.  m\n",
				rad2deg(procp.map[3].lat()), rad2deg(procp.map[3].lon()), procp.map[3].h());
        fprintf(procpf,"map_coordinate_5:             %12.7f   %12.7f   %12.4f   deg.  deg.  m\n",
				rad2deg(procp.map[4].lat()), rad2deg(procp.map[4].lon()), procp.map[4].h());

        fprintf(procpf,"number_of_state_vectors:            %6d\n",procp.nstate);

        if (procp.nstate == 0){
			fclose(procpf);
			return;
		}
        if(procp.nstate > procp.MXST){
            fprintf(stderr,"WARNING: #state vectors exceeds structure allocation, MXST=%d\n",procp.MXST);
        }

        fprintf(procpf,"time_of_first_state_vector:   %12.5f   s\n",procp.t_state);
        fprintf(procpf,"state_vector_interval:        %12.5f   s\n",procp.tis);

        for (int i=0; i < mat::min(procp.nstate, procp.MXST); i++){
            fprintf(procpf, "state_vector_position_%s: %14.4f  %14.4f  %14.4f   m   m   m\n",
                    def_func::num2str(i+1).c_str(),
                    procp.state.pos()[i].x(), procp.state.pos()[i].y(), procp.state.pos()[i].z());
            fprintf(procpf, "state_vector_velocity_%s: %14.4f  %14.4f  %14.4f   m/s m/s m/s\n",
                    def_func::num2str(i+1).c_str(),
                    procp.state.vel()[i].x(), procp.state.vel()[i].y(), procp.state.vel()[i].z());
        }
        fprintf(procpf,"\n*************** END OF PROCESSING PARAMETERS ******************\n");
		fclose(procpf);
    }

    template<typename T1, typename T2>
    VEC<T1> VecTransform(const VEC<T1>& Ppy_vec,const D2<T2>& Matrix){
    /*
     Purpose:
        Transform VEC class to destination VEC using transformatin matrix.
     Input:
        Ppy_vec : (VEC class) Vector
        Matrix  : (D2 class) Transformatin matrix
     Return:
        out : a new vector
    */
        D2<T1> tmp(1,3);
        // Convert VEC to D2
        tmp = VEC2D2(Ppy_vec,1,3);
        // Multiply
        tmp = tmp*Matrix;
        // Convert D2 to VEC
        return D22VEC(tmp);
    }

    template<typename T>
    VEC<T> VecTransform(const VEC<T>& Ppy_vec,const D2<T>& Matrix,const VEC<T>& Poc){
    /*
     Purpose:
        Transform VEC class to destination VEC using transformatin matrix.
     Input:
        Ppy_vec : (VEC class) Vector
        Matrix  : (D2 class) Transformatin matrix
        Poc     : (VEC class) Original location
     Return:
        out : a new vector
    */
        //return out;
        D2<T> tmp(1,3);
        D2<T> Poc_D2=def_func::VEC2D2(Poc,1,3);
        // Convert VEC to D2
        tmp = VEC2D2(Ppy_vec,1,3);
		//tmp = tmp*Matrix;
		//tmp = tmp + Poc_D2;
		//tmp.Print();
		//D2<T> tttt = tmp*Matrix;
        // Multiply
        tmp = (tmp*Matrix) + Poc_D2;
		//tttt.Print();
		//Poc_D2.Print();
		//tmp.Print();
        // Convert D2 to VEC
        return D22VEC(tmp);
    }

    template<typename T>
    D1<VEC<T> > MakeMovingTarget(const T& V_rg,const T& V_az,const T* t,const long sample,
                                 const VEC<T>& org){
    /*
     Purpose:
        Make a moving target.
     Input:
        V_rg : [m/s] Velocity at range direction
        V_az : [m/s] Velocity at azimuth direction
        t	 : [sec] time series(slow time) at azimuth direction
        org	 : [m,m,m] Origainl location with ECR system
     Return:
        Moving target : [m,m,m](D1 series
    */
        VEC<T> vel(V_az,V_rg,0.);   //[m/s,m/s,m/s] (x,y,z) velocity
        T dt=t[1]-t[0];             //[sec] Time interval

        // make t-series
        D1<T> tt(sample);
        def_func::linspace(0.,double(sample-1),tt);
        tt = tt/dt;

        cout<<"V_rg="<<def_func::MS2KmHr(V_rg)<<" [km/hr]"<<endl;
        cout<<"V_az="<<def_func::MS2KmHr(V_az)<<" [km/hr]"<<endl;

        D1<VEC<T> > res(sample);
        for(long i=0;i<sample;++i){
            res[i].Setx( vel.x()*tt[i] + org.x() );
            res[i].Sety( vel.y()*tt[i] + org.y() );
            res[i].Setz( vel.z()*tt[i] + org.z() );
        }

        return res;
    }

    template<typename T>
    D1<VEC<T> > MakeMovingTarget(const T& V_rg,const T& V_az,const T* t,const long sample,
                                 const VEC<T>& org,const D2<T>& Matrix){
    /**
     *Purpose:
     *  Make a moving target.
     *Input:
     *  V_rg : [m/s] Velocity at range direction
     *  V_az : [m/s] Velocity at azimuth direction
     *  t	 : [sec] time series(slow time) at azimuth direction
     *  org	 : [m,m,m] Origainl location with ECR system
     *  Matrix:(D2 class) Transformation matrix
     *Return:
     *  Moving target : [m,m,m](D1 series)
     */
        VEC<T> vel(V_az,V_rg,0.);   ///[m/s,m/s,m/s] (x,y,z) velocity
        T dt=t[1]-t[0];             ///[sec] Time interval

        /// make t-series
        D1<T> tt(sample);
        def_func::linspace(0.,double(sample-1),tt);
        tt = tt/dt;

        cout<<"V_rg="<<def_func::MS2KmHr(V_rg)<<" [km/hr]"<<endl;
        cout<<"V_az="<<def_func::MS2KmHr(V_az)<<" [km/hr]"<<endl;

        D1<VEC<T> > res(sample);
        for(long i=0;i<sample;++i){
            res[i].Setx( vel.x()*tt[i] );
            res[i].Sety( vel.y()*tt[i] );
            res[i].Setz( vel.z()*tt[i] );
        }

        for(long i=0;i<sample;++i){
            res[i] = sar::VecTransform(res[i],Matrix,org);
        }

        return res;
    }

	template<typename T>
	D2<CPLX<T> > RCMC(D2<CPLX<T> >& Srd, const D2<T>& RCM, const bool SHOW){
	/**
	 *Purpose:
	 *  Range Cell Migration Correction
	 *Input:
	 *  Srd : Range-Doppler domain
	 *  RCM : [pix] range migration shift pixels
	 *Return:
	 *  The array after RCMC in Range-Doppler domain
	 */
		//
		// Input:
		//	Src_Srd : [rg,az](range-azimuth domain)
		//
		D1<size_t> sz = Srd.GetDim();
		size_t num_rg = sz[1];
		size_t num_az = sz[0];
		
		D2<CPLX<T> > rc_fft_RCMC = Srd;
//		D2<CPLX<T> > Srcmc(num_az, num_rg);

		for(long j=0;j<num_az;++j){
			for(long i=0;i<num_rg;++i){
				if(isinf(Srd[j][i].r()) || isinf(Srd[j][i].i())){
					rc_fft_RCMC[j][i] = CPLX<T>(1e-9, 0);
				}
			}
		}
	    long N_az = num_az;

		if(SHOW){ cout<<endl<<"RANGE CELLS MIGRATION CORRECTION..."<<endl; }
		
		// Transfrom to Range-Doppler domain
//		rc_fft_RCMC.WriteBinary("/Users/cychiang/Documents/code/C/PO/PhysicalOptics16_cuda/TestAreaMSTAR/res_remote/SAR/rc_fft_RCMC0.dat");
		mat::FFTX(rc_fft_RCMC);
//		rc_fft_RCMC.WriteBinary("/Users/cychiang/Documents/code/C/PO/PhysicalOptics16_cuda/TestAreaMSTAR/res_remote/SAR/rc_fft_RCMC1.dat");
		
		
		D1<T> x_idx = mat::Indgen<T>(sz[1]);
		
		
		// extend input data (Range-Doppler domain)
		D1<long> ext_rg2(2);
		ext_rg2[0] = round(min(RCM));
		ext_rg2[1] = round(max(RCM));
		long ext_rg = max(abs(ext_rg2)) + 1;
		D1<CPLX<T> > line(num_rg + ext_rg*2);
		long num_rg_ext = line.GetNum();
		
		D1<double> x_idx_ext(num_rg_ext);
		D1<double> x_idx_new_ext(num_rg_ext);
		
		
		for(long j=0;j<N_az;++j){
			D1<double> x_idx_new(x_idx.GetNum());
			for(long i=0;i<x_idx.GetNum();++i){
				x_idx_new[i] = x_idx[i] - RCM[j][i];
			}
			for(long i=0;i<line.GetNum();++i){
				line[i] = CPLX<T>(0.0,0.0);
			}
			for(long i=0;i<rc_fft_RCMC.GetN();++i){
				line[ext_rg+i] = rc_fft_RCMC[j][i];
			}
			
//			line.WriteBinary("/Users/cychiang/Documents/code/C/PO/PhysicalOptics16_cuda/TestAreaMSTAR/res_remote/SAR/line.dat");
			
			// extended x_idx
			for(long i=0;i<num_rg_ext/2;++i){
				x_idx_ext[i] = x_idx[0];
			}
			for(long i=num_rg_ext/2;i<num_rg_ext;++i){
				x_idx_ext[i] = x_idx[sz[1]-1];
			}
			for(long i=0;i<x_idx.GetNum();++i){
				x_idx_ext[ext_rg+i] = x_idx[i];
			}
			// extended x_idx_new
			for(long i=0;i<num_rg_ext/2;++i){
				x_idx_new_ext[i] = x_idx_new[0];
			}
			for(long i=num_rg_ext/2;i<num_rg_ext;++i){
				x_idx_new_ext[i] = x_idx_new[sz[1]-1];
			}
			for(long i=0;i<x_idx_new.GetNum();++i){
				x_idx_new_ext[ext_rg+i] = x_idx_new[i];
			}
			// Interpolation
			D1<T> line_r(line.GetNum()), line_i(line.GetNum());
			D1<T> line_tmp_r(x_idx_new_ext.GetNum()), line_tmp_i(x_idx_new_ext.GetNum());
			D1<CPLX<T> > line_tmp(x_idx_new_ext.GetNum());
			// assign
			for(long i=0;i<line.GetNum();++i){
				line_r[i] = line[i].r();
				line_i[i] = line[i].i();
			}
//			line_tmp_r = mat::PolyInt1(line_r, x_idx_ext, x_idx_new_ext);
//			line_tmp_i = mat::PolyInt1(line_i, x_idx_ext, x_idx_new_ext);
			mat::INTERPOL(line_r, x_idx_ext, x_idx_new_ext, line_tmp_r);
			mat::INTERPOL(line_i, x_idx_ext, x_idx_new_ext, line_tmp_i);
			// assign back
			for(long i=0;i<x_idx_new_ext.GetNum();++i){
				line_tmp[i].r() = line_tmp_r[i];
				line_tmp[i].i() = line_tmp_i[i];
				// check nan or inf
				if(isnan(line_tmp[i].r()) || isnan(line_tmp[i].i()) ||
				   isinf(line_tmp[i].r()) || isinf(line_tmp[i].i())
				   ){
					line_tmp[i] = CPLX<T>(0.,0.);
				}
			}
			// check nan or inf
//			for(long j=0;j<num_az;++j){
//				for(long i=0;i<num_rg;++i){
//					Srcmc[j][i] = CPLX<T>(0.,0.);
//				}
				for(long i=0;i<num_rg;++i){
					rc_fft_RCMC[j][i] = line_tmp[ext_rg+i];
				}
//			}
//			rc_fft_RCMC[*,j] = line_tmp[ext_rg:ext_rg+num_rg-1L]
			if((j % 200) == 0 && SHOW){
				cout<<j<<" / "<<N_az<<endl;//<<" "<<rc_fft_RCMC[j][89]<<endl;
//				printf("%8d (%16.11f,%16.11f)\n", j, rc_fft_RCMC[j][89].r(), rc_fft_RCMC[j][89].i());
			}
		}
		return rc_fft_RCMC;
/*
		long num_rg = Srd.GetN();
		long num_az = Srd.GetM();
		
		// Sinc interpolator
		double beta = 80.0;
		// kernel size
		long num_kernel = ceil(max(abs(RCM)))*2L + 1;
		num_kernel = (num_kernel > num_rg)? num_rg:num_kernel;
		
		
		long num_half = num_kernel/2;
		D1<double> sinc_x(num_kernel);
		linspace(-double(num_half), double(num_half), sinc_x);
		
		// initialize win
		D1<double> win(num_kernel),intor(num_kernel);
		double tmp;
		double sum;
		D2<CPLX<double> > Srcmc(num_az,num_rg);
		Srcmc.SetZero();

		
		if(SHOW){ cout<<endl<<"RANGE CELLS MIGRATION CORRECTION..."<<endl; }
		for(long j=0;j<num_az;++j){
			for(long i=0;i<num_rg;++i){
				// Kaiser window (2.54) Reduce the ringing effect (AKA. Gibb's phenomenon)
				for(long k=0;k<num_kernel;++k){
					tmp = (2.0*(sinc_x[k] + RCM[j][i])/num_kernel);
					tmp = tmp*tmp;
					if(tmp > 1.0){ tmp = 1.0; }
					win[k] = mat::bessi0(beta * std::sqrt(1.0 - tmp)) / mat::bessi0(beta);
					intor[k] = win[k] * mat::sinc(sinc_x[k] + RCM[j][i]);
				}
				if(i < num_half){
					// normalize factor
					sum = 0.0;
					for(long k=num_half;k<num_kernel;++k){
						sum += intor[k]*intor[k];
						sum = sqrt(sum);
					}
					// Assign value
					if(sum < 1e-5){
						Srcmc[j][i] = CPLX<double>(0.0,0.0);
					}else{
						for(long k=0;k<i+num_half+1;++k){
							Srcmc[j][i] += (Srd[j][k] * intor[k+num_half] / sum);
						}
					}
				}else if(i > (num_rg - num_half - 1L)){
					// normalize factor
					sum = 0.0;
					for(long k=num_half;k<num_half+num_rg-i;++k){
						sum += intor[k]*intor[k];
						sum = sqrt(sum);
					}
					// Assign value
					if(sum < 1e-5){
						Srcmc[j][i] = CPLX<double>(0.0,0.0);
					}else{
						for(long k=i;k<num_rg;++k){
							Srcmc[j][i] += (Srd[j][k] * intor[k+num_half] / sum);
						}
					}
				}else{
					// normalize factor
					sum = 0.0;
					for(long k=0;k<num_kernel;++k){
						sum += (intor[k]*intor[k]);
						sum = sqrt(sum);
					}
					// Assign value (2.59)
					for(long k=i-num_half;k<i+num_half+1;++k){
						Srcmc[j][i] += (Srd[j][k] * intor[k-i+num_half] / sum);
					}
				}
			}
			if(SHOW){ if(mat::Mod(double(j), 100.0) == 0){ cout<<j<<" / "<<num_az<<endl; } }
		}
		return Srcmc;
*/
	}
	
	double RCS2dB(const double rcs, const double f0){
		double lambda = def::C/f0;
		double dB;

		dB = 4*def::PI/(lambda*lambda) * (rcs * rcs);
		if(abs(dB) < 1E-10){
			dB = 1E-2;
		}
		dB = 10*log10(dB);

		return dB;
	}

	template<typename T>
	D1<CPLX<T> > RCMCSincInterpCore(D1<CPLX<T> >& Srd, const D1<T>& RCM, const T KAISER_BETA, const long NUM_KERNEL){
		
		long num_kernel = NUM_KERNEL;
		
		// num_kernel MUST be odd
		if(double(NUM_KERNEL)/2 > 1e-5){
			num_kernel--;
		}
		
		long num_rg   = Srd.GetNum();
		long num_half = floor(double(num_kernel)/2.0);
		D1<double> sinc_x(num_kernel);
		def_func::linspace(-double(num_half), double(num_half), sinc_x);
		
		// extend
		D1<CPLX<T> > Srcmc_ext(num_rg + num_half * 2);
		D1<CPLX<T> > Srd_ext(num_rg + num_half * 2);
		for(size_t i=0;i<num_rg;++i){
			Srd_ext[num_half+i-1] = Srd[i];
		}
		
		// window
		D1<T> win = sar::find::Kaiser(num_kernel, KAISER_BETA);
//		D1<double> win = sar::find::kaKaiser(150, 2.1);
//		D1<double> wk  = sar::Kaiser(Sar.Nr(), KAISER_BETA);
		D1<T> intor(num_kernel);
		T sum;
		
		for(size_t i=num_half;i<(num_half + num_rg);++i){
			sum = 0;
			for(size_t k=0;k<num_kernel;++k){
				intor[k] = win[k] * sinc(sinc_x[k] + RCM[i-num_half]);
//				intor[k] = sinc(sinc_x[k] + RCM[i-num_half]);
				sum += Square(intor[k]);
			}
			sum = sqrt(sum);
			for(size_t k=0;k<num_kernel;++k){
				Srcmc_ext[i] += (Srd_ext[i-num_half+k] * intor[k] / sum);
			}
		}
		
		D1<CPLX<T> > Srcmc(num_rg);
		for(size_t k=0;k<num_rg;++k){
			Srcmc[k] = Srcmc_ext[num_half+k];
		}

		return Srcmc;
	}
	
	template<typename T>
	D2<CPLX<T> > RCMCSinc(D2<CPLX<T> >& Srd, const D2<T>& RCM, const bool SHOW){
		
		long num_rg = Srd.GetN();
		long num_az = Srd.GetM();
		
			
		cout<<"RCMC by Sinc interpolation ..."<<endl;

		//+-------------------------+
		//|    Sinc interpolator    |
		//+-------------------------+
		D1<CPLX<T> > Srd_rcmc1(num_rg);
		D2<CPLX<T> > Srd_rcmc2(num_az, num_rg);
		
		D1<long>   rcm_int(num_rg);
		D1<double> rcm_fra(num_rg);
		
		for(size_t j=0;j<num_az;++j){
			for(size_t i=0;i<num_rg;++i){
				// Get integer part
				rcm_int[i] = floor(RCM[j][i]);
				// Get fractional part
				rcm_fra[i] = RCM[j][i] - double(rcm_int[i]);
			}
			//
			// Integer shift (Step 1)
			//
			for(size_t i=0;i<num_rg;++i){
				long idx = i - rcm_int[i];
				if((idx < 0) || (idx > num_rg-1)){
					Srd_rcmc1[i] = CPLX<T>(0,0);
				}else{
					Srd_rcmc1[i] = Srd[j][idx];
				}
			}
			//
			// Fractional shift (Step 2)
			//
			D1<CPLX<T> > tmp = RCMCSincInterpCore(Srd_rcmc1, rcm_fra, 8.9, 8);
			Srd_rcmc2.SetRow( tmp, j);

//			for(size_t i=0;i<num_rg;++i){
//				Srd_rcmc2[j][i] = Srd_rcmc1[i];
//			}
			
			if((j % 500) == 0){
				cout<<j<<" / "<<num_az<<endl;
			}
		}

		return Srd_rcmc2;
	}

	template<typename T>
	D1<T> find::Kaiser(const long num, const T beta){
		
		T TT = T(num);
		D1<T> t(num);
		linspace(-TT/2,TT/2,t);
		
		D1<T> wk(num);
		T tmp,tp; 
		
		for(long i=0;i<num;++i){
			tmp = t[i]/TT * 2.0;
			tmp = tmp * tmp;
			if(tmp > 1.0){ tmp = 1.0; }
			tp = beta*sqrt( 1.0 - tmp );
			wk[i] = mat::bessi0(tp)/mat::bessi0(beta);
		}
		return wk;
	}
	
	template<typename T>
	D1<T> find::KaiserWindow(const T Tr_Fr, const long num, T beta){
		D1<T> t_f(num);
		linspace(-Tr_Fr/2.0,Tr_Fr/2.0,t_f);
		T tmp, tp;
		D1<T> wk(num);
		
		for(long i=0;i<num;++i){
			tmp = t_f[i]/Tr_Fr * 2.0;
			tmp = tmp * tmp;
			if(tmp > 1.0){ tmp = 1.0; }
			tp = beta*sqrt( 1.0 - tmp );
			wk[i] = mat::bessi0(tp)/mat::bessi0(beta);
		}
		return wk;
	}
	
	template<typename T>
	void find::SlantRangePosition(const VEC<T>& Ps, const VEC<T>& Ps1, const VEC<T>& Psg, const VEC<T>& Psc,			// input
								  const double theta_l, const double theta_sq, const double SWrg, const def::ORB& orb,		// input
								  VEC<T>& Ps_new, double& theta_l_new, bool IsNear, double theta_l_step_init){	// output
		// Near or far
		T sign = (IsNear == true)? 1:-1;
		
		// initial theta_l
		double dis_c = (Ps - Psc).abs();
		theta_l_new = theta_l;
		double theta_l_step = theta_l_step_init;
		double diff;
		bool   type = true;
		vec::VEC<T> uv;
		for(int i=0;i<1000;++i){
			uv     = sar::find::LookAngleLineEq(Ps, Psg, Ps1, theta_sq, theta_l_new);
			Ps_new = sar::find::BeamLocationOnSurface(uv, Ps, orb);
			if( abs((Ps_new - Ps).abs() - dis_c) < SWrg/2 ){
				theta_l_new = theta_l_new - sign*theta_l_step;
				if(i == 0){
					type = true;
				}else if(type == false){
					theta_l_step *= 0.8;
					type = true;
				}
			}else{
				theta_l_new = theta_l_new + sign*theta_l_step;
				if(i == 0){
					type = false;
				}else if(type == true){
					theta_l_step *= 0.8;
					type = false;
				}
			}
			diff = (Ps-Ps_new).abs() - ((Ps-Psc).abs() - sign*SWrg/2);
			if(abs(diff) < 1E-15){
				break;
			}
		}
	}
	
	template<typename T>
	void find::InsFreqOfMatchedFilter(const double f_nc_abs, const double PRF, D1<T>& fn, double& shift_d){
		// Find Instantaneous frequency of matched filter
		linspace(f_nc_abs-PRF/2, f_nc_abs+PRF/2, fn);
		fftshift(fn);
		
		// find the Frac
		double Frac = mat::Mod(f_nc_abs, PRF);
		shift_d = Frac * (double)fn.GetNum()/PRF;
		shift(fn, shift_d);
	}
	
	template<typename T>
	D1<T> find::ThetaLookRegion(const SV<T>& sv, const D1<T>& R0, const def::ORB& Orb){
		/*
		 sv		: State vector class
		 R0		: [m] Slant range distance from near to far of SAR RAW data
		 Orb	: Orbit/datum class
		 */
		VEC<T> Ps   = sv.pos()[(sv.GetNum()-1)/2];
		VEC<T> Psg  = find::ProjectPoint(Ps, Orb);
		double Re   = Psg.abs();			// Local Earth Radius
		double Re_h = Ps.abs();				// Local distance from Earht center (ECEF) to sensor
		double sln  = R0[0];				// [m] Raw data NEAR slant range
		double slf  = R0[R0.GetNum()-1];	// [m] Raw data FAR slant range
		double theta_l_n = acos( (Re_h*Re_h + sln*sln - Re*Re)/(2*Re_h*sln) );	// Look angle of NEAR slant range
		double theta_l_f = acos( (Re_h*Re_h + slf*slf - Re*Re)/(2*Re_h*slf) );	// Look angle of FAR slant range
		D1<T> theta_l_limit(2);
		theta_l_limit[0] = (T)theta_l_n;
		theta_l_limit[1] = (T)theta_l_f;
		return theta_l_limit;
	}
	
	template<typename T>
	void find::EffectiveVelocity(const SV<T>& sv, const def::ORB& Orb, const D1<T>& theta_l_limit, const double theta_sqc, const long num_rg, D1<T>& R0, D1<T>& Vr){
		// Look angle series
		D1<T> look_angle(num_rg);
		linspace(theta_l_limit[0], theta_l_limit[1], look_angle);
//		D1<double> tt = sv.t() - sv.t()[0];
		
		// ===== Find intersection point(beam) on Earth surface ========================
		// find intersection point
		long c_idx = (sv.GetNum()-1)/2;
		VEC<T> Ps = sv.pos()[c_idx];				// point A
		VEC<T> Ps1= sv.pos()[c_idx+1];				// point C
		VEC<T> Psg= find::ProjectPoint(Ps, Orb);	// projected to surface (G)
		VEC<T> uv_tp = cross(Psg-Ps, Ps1-Ps);		// uv = (AG x AC)
		VEC<T> uv_Psp= cross(Ps1-Ps, uv_tp);		// Orthogonal unit vector
		VEC<T> Psp= find::BeamLocationOnSurface(uv_Psp, Ps, Orb);	// Orthogonal point (P)
		
		D1<double> dis2(sv.GetNum());
		R0 = D1<T>(num_rg);
		Vr = D1<T>(num_rg);
		D1<double> coeff;
		
		// Calculate
		for(long i=0;i<num_rg;++i){
			// Find beam unit vector
			VEC<T> uv = find::LookAngleLineEq(Ps, Psp, Ps1, theta_sqc, look_angle[i]);	// beam unit vector
			// Find intersection point(beam) on Earth surface
			VEC<T> SCT = find::BeamLocationOnSurface(uv, Ps, Orb);		// beam intersection point on Earth's surface
			// find distance for all point at orbit
			for(long j=0;j<sv.GetNum();++j){
				dis2[j] = (double)mat::Square((sv.pos()[j] - SCT).abs());
			}
			// curve fitting
			coeff = fit::HYPERBOLIC(sv.t() - sv.t()[(sv.GetNum()-1)/2], dis2);
			R0[i] = (T)sqrt(coeff[0]);
			Vr[i] = (T)sqrt(coeff[1]);
		}
	}
	
	template<typename T>
	D1<T> find::GroundVelocity(const SV<T>& sv, const D1<T>& theta_l_range, const def::ORB& Orb){
		long num_az = sv.GetNum();
		T dt = sv.dt();
		
//		D1<T> Vg(num_az);
//		
//		VEC<T> P1, P1_1, P2, P2_1, Pp;
//		VEC<T> LOS1_n, LOS2_n;
//		VEC<T> ptr_pos1_n, ptr_pos2_n;
//		
//		for(long i=0;i<num_az-1;++i){
//			// Find first point
//			P1   = sv.pos()[num_az/2];			// P1
//			P1_1 = sv.pos()[num_az/2+1];			// Next to P1
//			Pp   = find::ProjectPoint(P1, Orb);	// P1 project point
//			LOS1_n = vec::find::ArbitraryRotate(Pp - P1, theta_l_range[i], vec::Unit(P1_1-P1));
//			ptr_pos1_n = find::BeamLocationOnSurface(LOS1_n, P1, Orb);
//	
//			// Find next point
//			P2   = sv.pos()[num_az/2+1];
//			P2_1 = sv.pos()[num_az/2+2];
//			Pp = find::ProjectPoint(P2, Orb);
//			LOS2_n = vec::find::ArbitraryRotate(Pp - P2, theta_l_range[i], vec::Unit(P2_1-P2));
//			ptr_pos2_n = find::BeamLocationOnSurface(LOS2_n, P2, Orb);
//			
//			Vg[i] = (ptr_pos1_n - ptr_pos2_n).abs()/dt;
//		}
//		Vg[num_az-1] = Vg[num_az-2];
//		return Vg;
		
		
		//=============================== NEAR =====================================
		// Find first point
		VEC<T> P1   = sv.pos()[num_az/2];			// P1
		VEC<T> P1_1 = sv.pos()[num_az/2+1];			// Next to P1
		VEC<T> Pp   = find::ProjectPoint(P1, Orb);	// P1 project point
		VEC<T> LOS1_n = vec::find::ArbitraryRotate(Pp - P1, theta_l_range[0], vec::Unit(P1_1-P1));
		VEC<T> ptr_pos1_n = find::BeamLocationOnSurface(LOS1_n, P1, Orb);
		
		// Find next point
		VEC<T> P2   = sv.pos()[num_az/2+1];
		VEC<T> P2_1 = sv.pos()[num_az/2+2];
		Pp = find::ProjectPoint(P2, Orb);
		VEC<T> LOS2_n = vec::find::ArbitraryRotate(Pp - P2, theta_l_range[0], vec::Unit(P2_1-P2));
		VEC<T> ptr_pos2_n = find::BeamLocationOnSurface(LOS2_n, P2, Orb);
		
		T Vg_n = (ptr_pos1_n - ptr_pos2_n).abs()/dt;
		
		
		//=============================== FAR ======================================
		// Find first point
		P1 = sv.pos()[num_az/2];
		P1_1 = sv.pos()[num_az/2+1];
		Pp = find::ProjectPoint(P1, Orb);
		VEC<T> LOS1_f = vec::find::ArbitraryRotate(Pp - P1, theta_l_range[1], vec::Unit(P1_1-P1));
		VEC<T> ptr_pos1_f = find::BeamLocationOnSurface(LOS1_f, P1, Orb);
		
		// Find next point
		P2 = sv.pos()[num_az/2+1];
		P2_1 = sv.pos()[num_az/2+2];
		Pp = find::ProjectPoint(P2, Orb);
		VEC<T> LOS2_f = vec::find::ArbitraryRotate(Pp - P2, theta_l_range[1], vec::Unit(P2_1-P2));
		VEC<T> ptr_pos2_f = find::BeamLocationOnSurface(LOS2_f, P2, Orb);
		
		T Vg_f = (ptr_pos1_f - ptr_pos2_f).abs()/dt;
		
		D1<T> Vg(2);
		Vg[0] = Vg_n;
		Vg[1] = Vg_f;

		return Vg;
	}

	/**
	 * Equalization the height of state vector by replcaing original data
	 * @param [in/out] sv: State vector
	 * @param [in] Orb : Dataum class
	 */
	void find::SVNormalizeHeight(SV<double>& sv, const def::ORB& Orb){
		D1<double> h(sv.GetNum());
		for (long i = 0; i < sv.GetNum(); ++i) {
			h[i] = sar::ECR2Gd(sv.pos()[i], Orb).h();
		}
		double hmean = mat::total(h) / h.GetNum();
		for (long i = 0; i < sv.GetNum(); ++i) {
			GEO<double> gd = sar::ECR2Gd(sv.pos()[i], Orb);
			gd.h() = hmean;
			sv.pos()[i] = sar::Gd2ECR(gd, Orb);
		}
	}

	/**
	 * Find the main beam unit vector with TYPE-A
	 * @param [in] PsC: [m,m,m] Any position on the path
	 * @param [in] PsC1: [m,m,m] Position next to Ps
	 * @param [in] theta_l: [rad] Look angle
	 * @param [in] theta_sq: [rad] Squint angle
	 * @param [in] Orb: Datum class
	 * @param [out] Pt: [m,m,m] Position intersection to the earth's surface
	 * @param [out] theta_l_new: [rad] Look angle from LOS's uv to NADIR
	 * @param [out] uvlp: Unit vector only occurs by theta_l on zero doppler plane
	 * @param [in] DEBUG: Enable debug message or not.
	 * @return Return a unit vector of LOS by TYPE-A
	 */
	VEC<double> find::MainBeamUVByLookSquintTypeA(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l,
												  const double theta_sq, const def::ORB& Orb, VEC<double>& Pt,
												  double& theta_l_new, VEC<double>& uvlp, const bool SHOW){
		// (0) Prepare
		VEC<double> Pg    = sar::find::ProjectPoint(PsC, Orb);
		VEC<double> PsPg  = Unit(Pg - PsC);
		VEC<double> PsPs1 = Unit(PsC1 - PsC);
		// (1) Find the horizontial UV on the Zero Doppler plane
		VEC<double> uva   = Unit(cross(PsPg, PsPs1));
		// (2) Find the horizontial UV on Ps, Ps1 plane
		VEC<double> uvb   = Unit(cross(uva, PsPg));
		// (3) Find the UV when theta_sqc is zero on Zero Doppler plane
		uvlp  = vec::find::ArbitraryRotate(PsPg, -theta_l, uvb);				// Right-hand
		// (4) Find the squint plane normal UV
		VEC<double> uvN   = Unit(cross(uvlp, uvb));
		// (5) Find destination main beam UV by theta_l & theta_sqc
		VEC<double> uv    = vec::find::ArbitraryRotate(uvlp, +theta_sq, uvN); 	// Right-hand
		// (6) find the new look angle
		theta_l_new = vec::angle(uv, PsPg);
		// (7) Earth's intersection point
		Pt    = sar::find::BeamLocationOnSurface(uv, PsC, Orb);


		if(SHOW){
			cout<<"+------------------------+"<<endl;
			cout<<"|          TEST          |"<<endl;
			cout<<"+------------------------+"<<endl;
			double sign  = (dot(uv, PsPs1) >= 1E-5)? +1:-1;
			cout<<"theta_l      = "<<rad2deg(theta_l)<<" [deg]"<<endl;
			cout<<"theta_sq     = "<<rad2deg(theta_sq)<<" [deg]"<<endl;
			cout<<"theta_l_new  = "<<rad2deg(theta_l_new)<<" [deg]"<<endl;
			cout<<"+"<<endl;
			cout<<"   Recalculated:"<<endl;
			cout<<"+"<<endl;
			cout<<"theta_l      = "<<rad2deg(angle(PsPg, uvlp))<<" [deg]"<<endl;
			cout<<"theta_sq     = "<<rad2deg(sign*angle(uv, uvlp))<<" [deg]"<<endl;
			cout<<"theta_l_new  = "<<rad2deg(angle(PsPg, uv))<<" [deg]"<<endl;
		}

		return uv;
	}

	/**
	 * Find the main beam unit vector with TYPE-A
	 * @param [in] PsC: [m,m,m] Any position on the path
	 * @param [in] PsC1: [m,m,m] Position next to Ps
	 * @param [in] theta_l: [rad] Look angle
	 * @param [in] theta_sq: [rad] Squint angle
	 * @param [in] Orb: Datum class
	 * @param [out] Pt: [m,m,m] Position intersection to the earth's surface
	 * @param [in] DEBUG: Enable debug message or not.
	 * @return Return a unit vector of LOS by TYPE-A
	 */
	VEC<double> find::MainBeamUVByLookSquintTypeA(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l,
												  const double theta_sq, const def::ORB& Orb, VEC<double>& Pt, const bool SHOW){

		double theta_l_new;
		VEC<double> uvlp;
		VEC<double> uv = MainBeamUVByLookSquintTypeA(PsC, PsC1, theta_l, theta_sq, Orb, Pt, theta_l_new, uvlp, SHOW);

		return uv;
	}
	/**
	 * Find the main beam unit vector with TYPE-B
	 * @param [in] PsC: [m,m,m] Any position on the path
	 * @param [in] PsC1: [m,m,m] Position next to Ps
	 * @param [in] theta_l: [rad] Look angle
	 * @param [in] theta_sq: [rad] Squint angle
	 * @param [in] Orb: Datum class
	 * @param [out] Pt: [m,m,m] Position intersection to the earth's surface
	 * @return Return a unit vector of LOS by TYPE-B
	 */
	VEC<double> find::MainBeamUVByLookSquintTypeB(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l,
											const double theta_sq, const def::ORB& Orb, VEC<double>& Pt){
		// (0) Prepare
		VEC<double> Pg    = sar::find::ProjectPoint(PsC, Orb);
		VEC<double> PsPg  = Unit(Pg - PsC);
		VEC<double> PsPs1 = Unit(PsC1 - PsC);
		// (1) Find the horizontial UV on the Zero Doppler plane
		VEC<double> uva   = Unit(cross(PsPg, PsPs1));
		// (2) Find the horizontial UV on Ps, Ps1 plane
		VEC<double> uvb   = Unit(cross(uva, PsPg));
		// (3A) Find the UV when theta_sqc is zero on Zero Doppler plane
		VEC<double> uvl   = vec::find::ArbitraryRotate(PsPg, -theta_l, uvb);	// Right-hand
		//    (A) Find scene center postion on Zero Doppler plane
		VEC<double> Pc    = sar::find::BeamLocationOnSurface(uvl, PsC, Orb);
		//    (B) Find Nearest postion considering Earth's curvature.
		VEC<double> Pca;  vec::find::MinDistanceFromPointToLine(PsC, Pg, Pc, Pca);
		//    (C) Find new look angle on the zero Doppler plane
		double theta_l_p = acos( cos(theta_l)/cos(theta_sq) );
		// (3B) Find the UV when theta_sqc is zero on Zero Doppler plane
		VEC<double> uvlp = vec::find::ArbitraryRotate(PsPg, -theta_l_p, uvb);	// Right-hand
		// (4) Find the squint plane normal UV
		VEC<double> uvN  = cross(uvlp, uvb);
		// (5) Find destination main beam UV by theta_l & theta_sqc
		VEC<double> uv   = vec::find::ArbitraryRotate(uvl, +theta_sq, uvN);		// Right-hand
		// (6) Earth's intersection point
		Pt   = sar::find::BeamLocationOnSurface(uv, PsC, Orb);


		return uv;
	}

	/**
	 * Find the main beam unit vector with TYPE-A
	 * @param [in] PsC: [m,m,m] Any position on the path
	 * @param [in] PsC1: [m,m,m] Position next to Ps
	 * @param [in] theta_l: [rad] Look angle
	 * @param [in] theta_sq: [rad] Squint angle
	 * @param [in] Orb: Datum class
	 * @param [out] Pt: [m,m,m] Position intersection to the earth's surface
	 * @param [out] theta_l_new: [rad] Look angle from LOS's uv to NADIR
	 * @param [out] uvlp: Unit vector only occurs by theta_l on zero doppler plane
	 * @param [in] DEBUG: Enable debug message or not.
	 * @return Return a unit vector of LOS by TYPE-A
	 */
	VEC<double> find::MainBeamUVByLookSquintTypeASphericalCoordinate(const VEC<double> &PsC, const VEC<double> &PsC1,
																	 const double theta_l, const double theta_sq) {
		VEC<double> Pg    = PsC;   Pg.z() = 0;
		VEC<double> PsPg  = Unit(Pg - PsC);
		VEC<double> PsPs1 = Unit(PsC1 - PsC);
		// (1) Find the horizontial UV on the Zero Doppler plane
		VEC<double> uva   = Unit(cross(PsPg, PsPs1));
		// (2) Find the horizontial UV on Ps, Ps1 plane
		VEC<double> uvb   = Unit(cross(uva, PsPg));
		// (3) Find the UV when theta_sqc is zero on Zero Doppler plane
		VEC<double> uvlp  = vec::find::ArbitraryRotate(PsPg, -theta_l, uvb);		// Right-hand
		// (4) Find the squint plane normal UV
		VEC<double> uvN   = Unit(cross(uvlp, uvb));
		// (5) Find destination main beam UV by theta_l & theta_sqc
		VEC<double> MainBeamUV = vec::find::ArbitraryRotate(uvlp, +theta_sq, uvN); 	// Right-hand

		return MainBeamUV;
	}

	/**
	 * Find the local XYZ coordinate
	 * @param [in] PsC: [m,m,m] Center position on the path
	 * @param [in] PsC1: [m,m,m] The position next to the PsC
	 * @param [in] theta_l: [rad] Look angle
	 * @param [in] theta_sq: [rad] Squint angle
	 * @param [in] Orb: Datum class
	 * @param [in] DEBUG: Enable debug message or not.
	 * @return Return the local coordinate with LocalXYZ class
	 */
	LocalXYZ find::LocalCoordinate(const VEC<double>& PsC, const VEC<double>& PsC1, const double theta_l, const double theta_sq,
								   const def::ORB& Orb, const bool SHOW){

		LocalXYZ uv;
		// (0) Prepare
		VEC<double> PsCg = sar::find::ProjectPoint(PsC, Orb);
		// (1) Find center main beam unit vector & scene center position, Pt
		VEC<double> Pt;
		VEC<double> UV   = MainBeamUVByLookSquintTypeA(PsC, PsC1, theta_l, theta_sq, Orb, Pt);
		// (2) Find Zero Doppler intersection position, Pc
		VEC<double> Pc;
		VEC<double> uv2  = MainBeamUVByLookSquintTypeA(PsC, PsC1, theta_l, 0, Orb, Pc);
		// (3) Get above point
		GEO<double> Pthg = sar::ECR2Gd(Pt, Orb);
		Pthg.h() = 100.;
		VEC<double> Pth  = sar::Gd2ECR(Pthg, Orb);
		GEO<double> Pchg = sar::ECR2Gd(Pc, Orb);
		Pchg.h() = 100.;
		VEC<double> Pch  = sar::Gd2ECR(Pchg, Orb);
		// (3) Find uvz
		uv.z()  = Unit(Pth - Pt);
		// (4) Find Pta
		VEC<double> Pta; double dis = vec::find::MinDistanceFromPointToLine(Pch, Pc, Pt, Pta);
		// << Check >>
		//
		//    PsC
		//      +
		//      |\
		//      | \
		// PsCg +  \
		//      |   \
		//      |    \
		//  Pta +-----+ Pt
		//       (dis)
		//
		if(SHOW){
			cout<<"A = "<<(PsC - Pta).abs()<<endl;
			cout<<"B = "<<dis<<endl;
			cout<<"C = "<<(Pt  - PsC).abs()<<endl;
			cout<<"C = "<<sqrt( Square((PsC - Pta).abs()) + Square(dis) )<<" -> sqrt(A^2+B^2)"<<endl;
		}
		// (5) Find uvx & uvy. The "EPSILON" value is very important
		//     To enlarger the "EPSILON" value, to fit the small theta_sq (e.g. deg2rad(1d-5)) to near zero of phi value
		double _EPSILON = 1e-4;
		if( (Pt - Pta).abs() < _EPSILON) {
			// Squint = 0
			uv.y() = Unit(cross(uv.z(), -UV));
			uv.x() = Unit(cross(uv.y(), uv.z()));
		}else{
			// Squint != 0
			VEC<double> PtPta = Unit(Pta - Pt);
			uv.x() = Unit(cross(PtPta, uv.z()));
			uv.y() = Unit(cross(uv.z(), uv.x()));
		}


		if(SHOW){
			cout<<"+------------------------+"<<endl;
			cout<<"|          TEST          |"<<endl;
			cout<<"+------------------------+"<<endl;
			cout<<"uvx = "; uv.x().Print();
			cout<<"uvy = "; uv.y().Print();
			cout<<"uvz = "; uv.z().Print();
			cout<<"+"<<endl;
			cout<<"(if theta_sq = 0)then Pt == Pc == Pta"<<endl;
			cout<<"|Pc - Pt | = "<<(Pc - Pt).abs()<<" [m]"<<endl;
			cout<<"|Pc - Pta| = "<<(Pc - Pta).abs()<<" [m]"<<endl;
			cout<<"+"<<endl;
			cout<<"(if theta_sq = 0)then VecUnit(PcPch) == uvz"<<endl;
			cout<<"((PcPch) - uvz).abs() = "<<(Unit(Pch - Pc) - uv.z()).abs()<<endl;
		}

		return uv;
	}

	/**
	 * Find the local theta & phi angle in sphere coordinate. The phi angle with sign (+/-), the theta is always +.
	 * @param [in] Ps: [m,m,m] Any position on the sensor's paht
	 * @param [in] Pt: [m,m,m] Position of scene center that intersection by main beam unit vector to the earht's surface.
	 * @param [in] locXYZ: {x,y,z} structure, the local coordinate that was calculated from "sar::find::LocalCoordinate" function.
	 * @param [in] DEBUG: Enable debug message or not.
	 * @return Return the SPH class in radian for local sphere coordinate.
	 */
	SPH<double> find::LocalSPH(const VEC<double>& Ps, const VEC<double>& Pt, const LocalXYZ& locXYZ, const bool SHOW){

		SPH<double> sph;
		def::ORB Orb;

		// (0) Unit vector of Ps to Pt
		VEC<double> uv    = Unit(Pt - Ps);
		// (1) Find local theta angle
		sph.Theta() = angle(locXYZ.z(), -uv);
		// (2) Find arbitrary point from Pt along -uv
		VEC<double> Pb   = Pt + 100. * (-uv);
		// (3) Find project point, Pbg, on the uvx & uvy plane
		VEC<double> Pbg; vec::find::MinDistanceFromPointToPlane(locXYZ.z(), Pt, Pb, Pbg);
		// (4) Find local phi angle
		VEC<double> PtPbg= Unit(Pbg - Pt);
		sph.Phi()   = angle(PtPbg, locXYZ.x());
		// (5) Find the sign of phi
		if(sph.Phi() > deg2rad(90.)){ sph.Phi() -= deg2rad(180.); }
		if(dot(-uv, locXYZ.y()) < 0){ sph.Phi() = -sph.Phi(); }
		// (6) Find theta_l
		VEC<double> Psg = sar::find::ProjectPoint(Ps, Orb);
		VEC<double> PsPsg = Unit(Psg - Ps);
		VEC<double> PsPt  = Unit(Pt  - Ps);
		sph.Theta() = angle(PsPsg, PsPt);
		// (7) Slant range
		sph.R() = (Pt - Ps).abs();


		if(SHOW) {
			cout<<"+------------------------+"<<endl;
			cout<<"|          TEST          |"<<endl;
			cout<<"+------------------------+"<<endl;
			cout<<"theta = "<<rad2deg(sph.Theta())<<" [deg]"<<endl;
			cout<<"phi   = "<<rad2deg(sph.Phi())<<" [deg]"<<endl;
			cout<<"+"<<endl;
			cout<<"(if theta_sq = 0)then phi = 0"<<endl;
			cout<<"theta is tiny larger than theta_l"<<endl;
			cout<<",because the Earth curvature"<<endl;
			cout<<"+"<<endl;
			sph.Print();
		}

		return sph;
	}

	/**
	 * Find the SPH class for each sensor's position with target location, Pt.
	 * @param [in] sv_int: State vector after 1st interpolation, the sample MUST larger than 3 points
	 * @param [in] Pt: [m,m,m] Position of scene center that intersection by main beam unit vector to the earht's surface.
	 * @param [in] PRI: [sec] PRI
	 * @param [in] Orb: Datum class
	 * @param [in] SHOW: Display the message or not? (default = false)
	 * @return Return a D1<SPH<double> > sereis for each sv_int position
	 */
	D1<SPH<double> > find::LocalSPH(const SV<double>& sv_int, const VEC<double>& Pt, const double PRI, const def::ORB Orb, const bool SHOW){
		//==========================================================================================
		//|                          Find theta_az by sv_int & Pt                                  |
		//==========================================================================================
		// (1) Find nearest position, PsCn & project point
		VEC<double>	PsCn  = sar::find::NearestPs(sv_int, Pt, PRI/100.);
		VEC<double> PsCng = sar::find::ProjectPoint(PsCn, Orb);
		// (2) Find center main beam unit vector & scene center position, Pt
		VEC<double> uv = Unit(Pt - PsCn);
		// (3) Get above point
		GEO<double> Pth_gd = ECR2Gd(Pt, Orb);
		Pth_gd.h() = 100;
		VEC<double> Pth    = Gd2ECR(Pth_gd, Orb);
		// (4) Find uvz
		VEC<double> uvz = Unit(Pth - Pt);
		// (5) Find uvx & uvy (Squint = 0)
		VEC<double> uvy = Unit(cross(uvz, -uv));
		VEC<double> uvx = Unit(cross(uvy, uvz));
		sar::LocalXYZ locXYZ(uvx, uvy, uvz);
		// (6) Find theta_l, theta_sq, theta_az
		D1<SPH<double> > out(sv_int.GetNum());
		for(size_t i=0;i<sv_int.GetNum();++i){
			VEC<double> Ps = sv_int.pos()[i];
			out[i] = sar::find::LocalSPH(Ps, Pt, locXYZ);
		}

		if(SHOW){
			D1<double> theta_az(out.GetNum());
			for(size_t i=0;i<theta_az.GetNum();++i){
				theta_az[i] = out[i].Phi();
			}
			printf("+----------------------------+\n");
			printf("|         Backward           |\n");
			printf("+----------------------------+\n");
			printf("theta[end] = %f [deg]\n", rad2deg(theta_az[theta_az.GetNum()-1]));
			printf("theta[0]   = %f [deg]\n", rad2deg(theta_az[0]));
		}

		return out;
	}

	/**
	 * Find the theta_az (azimuth angle) on the ground for each interpolated SV position.
	 * @param [in] sv_int: Interpolated state vector
	 * @param [in] theta_sqc: [rad] System center squint angle
	 * @param [in] theta_l_MB: [rad] System center look angle
	 * @return Return a D1<double> theta_az 1D series in radius
	 */
	void find::AllAngle(const SV<double>& sv_int, const double theta_sqc, const double theta_l_MB,
			       		D1<double>& inst_theta_l, D1<double>& inst_theta_sq, D1<double>& inst_theta_az,
			       		const bool NormalizeHeight){

		// Predefined
		def::ORB Orb;
		// Duplicate
		SV<double> sv = sv_int;
		// Allocation
		inst_theta_l  = D1<double>(sv.GetNum());	// Instantaneous Look angle
		inst_theta_sq = D1<double>(sv.GetNum());	// Instantaneous squint angle
		inst_theta_az = D1<double>(sv.GetNum());	// Instantaneous azimuth angle


		//+-----------------------------------------------+
		//|       Re-normalize the altitude of SV         |
		//+-----------------------------------------------+
		if(NormalizeHeight){
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

		//+-----------------------------------------------+
		//|           Calculate each phi angle            |
		//+-----------------------------------------------+
		size_t idx_c = (sv.GetNum()-1)/2;
		VEC<double> PsC  = sv.pos()[idx_c];		// Center position of sensor
		VEC<double> PsC1 = sv.pos()[idx_c+1];	// Position next to the PsC
		VEC<double> Pt;							// Target location by center main beam
		VEC<double> uv   = sar::find::MainBeamUVByLookSquintTypeA(PsC, PsC1, theta_l_MB, theta_sqc, Orb, Pt);
		LocalXYZ locXYZ = sar::find::LocalCoordinate(PsC, PsC1, theta_l_MB, theta_sqc, Orb);

		D1<double> PhiSeriesReal(sv.GetNum());
		for(size_t i=0;i<sv.GetNum();++i){
			VEC<double> Ps  = sv.pos()[i];
			VEC<double> Psg = sar::find::ProjectPoint(Ps, Orb);
			VEC<double> PsPsg = Unit(Psg - Ps);
			VEC<double> PsPt  = Unit(Pt  - Ps);
			SPH<double> sph = sar::find::LocalSPH(Ps, Pt, locXYZ);
			inst_theta_az[i] = sph.Phi();
			inst_theta_l[i]  = angle(PsPsg, PsPt);
			VEC<double> PtPs  = -PsPt;
			VEC<double> PtPsC = Unit(PsC - Pt);
			double sgn = (dot(PtPs, locXYZ.y()) > 0)? +1:-1;
			inst_theta_sq[i] = sgn * angle(PtPsC, PtPs);
		}
	}

	/**
	 * Find the theta_az (azimuth angle) on the ground for each interpolated SV position.
	 * @param [in] sv_int: Interpolated state vector
	 * @param [in] theta_sqc: [rad] System center squint angle
	 * @param [in] theta_l_MB: [rad] System center look angle
	 * @return Return a D1<double> theta_az 1D series in radius
	 */
	D1<double> find::ThetaAz(const SV<double>& sv_int, const double theta_sqc, const double theta_l_MB, const bool NormalizeHeight){

		D1<double> inst_theta_l;
		D1<double> inst_theta_sq;
		D1<double> inst_theta_az;

		AllAngle(sv_int, theta_sqc, theta_l_MB, inst_theta_l, inst_theta_sq, inst_theta_az, NormalizeHeight);

		return inst_theta_az;
	}


	/**
	 * Find nearest sesnor's postion, PsC, from Target's location, Pt
	 * @param [in] sv_int: State vector after 1st interpolation, the sample MUST larger than 3 points
	 * @param [in] Pt: [m,m,m] Target's position
	 * @param [in] dt_min: [sec] Minimum time interval
	 * @param [in] ItrCount: Maximum iteration number (default = 5)
	 * @param [in] SHOW: Display the message or not? (default = false)
	 * @return Return a VEC<double> PsC position
	 */
	VEC<double> find::NearestPs(const SV<double>& sv_int, const VEC<double>& Pt, const double dt_min,
							    const size_t ItrCount, const bool SHOW){

		if(sv_int.GetNum() < 3){
			cerr<<"ERROR::sar::find::NearestPs: The sv_int samples MUST larger than 3"<<endl;
			exit(EXIT_FAILURE);
		}

		// (0) Initialize
		SV<double> sv_org = sv_int;
		double dt = (sv_org.t()[sv_org.GetNum()-1] - sv_org.t()[0]) / (sv_org.GetNum() - 1);
		VEC<double> PsCn;

		size_t count = 0;
		while ((dt > dt_min) && (count < ItrCount)){
			// (1) Find instantaneous slant range, Rn
			D1<double> Rn(sv_org.GetNum());
			for(size_t i=0;i<sv_org.GetNum();++i){
				VEC<double> Ps = sv_org.pos()[i];
				Rn[i] = (Ps - Pt).abs();
			}
			// (2) Find minimum distance & region
			long idx_c;
			min(Rn, idx_c);
			size_t idx_s = idx_c - 1;
			size_t idx_e = idx_c + 1;
			PsCn = sv_org.pos()[idx_c];
			// (3) Crop SV
			SV<double> sv_crop = sv_org.GetRange(idx_s, idx_e);
			// (4) Interpolation
			SV<double> sv_new;
			sar::sv_func::Interp(sv_crop, dt/10., sv_new);
			// (5) Update
			sv_org = sv_new;
			dt = (sv_org.t()[sv_org.GetNum()-1] - sv_org.t()[0]) / (sv_org.GetNum() - 1);
			count++;
			if(SHOW) {
				printf("count = %ld, dt = %f, dt_min = %f\n", count, dt, dt_min);
			}
		}

		return PsCn;
	}


	/**
	 * Calculating the Average Cross correlation coefficience
	 * @tparam T
	 * @param [in] array : one-dimwnsion array
	 * @return The Average Cross correlation coefficience
	 */
	template<typename T>
	CPLX<T> find::ACCC(D1<CPLX<T> >& array){

		CPLX<T> ACCC(0,0), tmp;
		for(size_t i=1;i<array.GetNum();++i){
			tmp = array[i] * array[i-1].conj();
			if(std::isnan(tmp.r()) == false && std::isnan(tmp.i()) == false){
				ACCC += tmp;
			}
		}

		return ACCC;
	}

	/**
	 * Find the instantaneous azimuth antenna gain value
	 * @tparam T
	 * @param [in] Sar : (SAR class)
	 * @param [in] NorSquintPlane : (VEC) Normal vector of instantaneous squint plane
	 * @param [in] PPs : (VEC) Sensor position
	 * @param [in] PPt : (VEC) Antenna main beam pointing position on the surface
	 * @return The instantaneous azimuth antenna gain value
	 */
	double find::AzimuthAntennaGain(const SAR& Sar, const VEC<double>& MainBeamUV, const VEC<double>& NorSquintPlane, const VEC<double>& PPs, const VEC<double>& PPt, const VEC<double>& hit){
		VEC<double> hitProj;
		vec::find::MinDistanceFromPointToPlane(NorSquintPlane, PPs, hit, hitProj);
		// Find azimuth's antenna angle (always positive)
		double theta_sq_inst = angle(MainBeamUV, hitProj - PPs);
		double azGain = sar::Waz(theta_sq_inst, Sar.theta_az());

		return azGain;
//		return theta_sq_inst;
	}

	/**
	 * Find azimuth angle by input two sites
	 * @tparam T
	 * @param [in] Ps_lla  : 1st position in LLA
	 * @param [in] Ps1_lla : 2nd position in LLA
	 * @param [in] Orb : (ORB) Ellipsoid (example: WGS84)
	 * @return The azimuth angle in radian
	 * Ref: https://www.movable-type.co.uk/scripts/latlong-vincenty.html
	 */
	double find::BearingAngle(const GEO<double>& Ps_lla, const GEO<double>& Ps1_lla, const def::ORB& Orb){
		double lat1 = Ps_lla.lat();
		double lon1 = Ps_lla.lon();
		double lat2 = Ps1_lla.lat();
		double lon2 = Ps1_lla.lon();
		double Ea = Orb.E_a();
		double Eb = Orb.E_b();

		double f = (Ea - Eb)/Ea;	// flattening
		double LL = lon2 - lon1; 	// L = difference in longitude, U = reduced latitude, defined by tan U = (1-f)·tanφ.
		double tanU1 = (1.0-f) * tan(lat1);
		double cosU1 = 1.0 / sqrt((1.0 + tanU1*tanU1));
		double sinU1 = tanU1 * cosU1;
		double tanU2 = (1.0-f) * tan(lat2);
		double cosU2 = 1.0 / sqrt((1 + tanU2*tanU2));
		double sinU2 = tanU2 * cosU2;

//		let λ = L, sinλ = null, cosλ = null;    // λ = difference in longitude on an auxiliary sphere
//		let σ = null, sinσ = null, cosσ = null; // σ = angular distance P₁ P₂ on the sphere
//		let cos2σₘ = null;                      // σₘ = angular distance on the sphere from the equator to the midpoint of the line
//		let cosSqα = null;                      // α = azimuth of the geodesic at the equator

		double L = LL;	// λ = L
		double Lp;		// λʹ = null;
		double c2sm;	// cos2σₘ
		double cSqa;	// cosSqα
		double ss;		// sinσ
		double cs;		// cosσ
		double sigma;	// σ
		double sL;		// sinλ
		double cL;		// cosλ
		do {
			sL = sin(L);	// sinλ
			cL = cos(L);	// cosλ
			double sSqs = (cosU2*sL) * (cosU2*sL) + (cosU1*sinU2-sinU1*cosU2*cL)*(cosU1*sinU2-sinU1*cosU2*cL);	// sinSqσ
			ss = sqrt(sSqs);	// sinσ
			cs = sinU1*sinU2 + cosU1*cosU2*cL;	// cosσ
			sigma = atan2(ss, cs);	// σ
			double sa = cosU1 * cosU2 * sL / ss;	// sinα
			cSqa = 1.0 - sa*sa;		// cosSqα
			c2sm = cs - 2.0*sinU1*sinU2/cSqa;	// cos2σₘ
			double C = f/16.0*cSqa*(4.0+f*(4.0-3.0*cSqa));
			Lp = L;		// λʹ = λ
			L = LL + (1-C) * f * sa * (sigma + C*ss*(c2sm+C*cs*(-1+2*c2sm*c2sm)));
		} while (abs(L-Lp) > 1e-12);

//		double uSq = cSqa * (Ea*Ea - Eb*Eb) / (Eb*Eb);
//		double A = 1 + uSq/16384*(4096+uSq*(-768+uSq*(320-175*uSq)));
//		double B = uSq/1024 * (256+uSq*(-128+uSq*(74-47*uSq)));
//		double ds = B*ss*(c2sm+B/4*(cs*(-1+2*c2sm*c2sm)-B/6*c2sm*(-3+4*ss*ss)*(-3+4*c2sm*c2sm)));	// Δσ

//		double s = Eb*A*(sigma-ds); // s = length of the geodesic

//		double a1 = atan2(cosU2*sL,  cosU1*sinU2-sinU1*cosU2*cL); // initial bearing
		double a2 = atan2(cosU1*sL, -sinU1*cosU2+cosU1*sinU2*cL); // final bearing

		return a2;
	}

	/**
	 * Calculate the circular path trajectory for Spotlight & Circular SAR system
	 * @param [in] As : [rad] Aspect angle
	 * @param [in] Ev : [rad] Elevation angle
	 * @param [in] R0 : [m] Center slant range
	 * @param [in] tar_lon : [rad] Target's longitude
	 * @param [in] tar_lat_gd : [rad] Target's latitude in geodetic
	 * @param [in] PRF : [Hz] Pulse repeat frequency
	 * @param [in] SHOW : (bool) Display message or not?
	 * @return [SV<double>] Return the state vector
	 */
	sv::SV<double> find::CircularSARPathTrajectory(const D1<double>& As, const double Ev, const double R0, const double tar_lon, const double tar_lat_gd, const double PRF, const bool SHOW){

		def::ORB Orb;

		// Convert from geodetic to geocentric
		sar::GEO<double> gd(0, tar_lat_gd, 0);
		sar::GEO<double> gc = sar::Gd2Gc(gd, Orb);
		double tar_lat_gc = gc.lat();

		// Local Earth's radius
//		double loc_Re = orb::RadiusCurvatureMeridian(gc.lat(), Orb);
		double loc_Re = orb::RadiusGeocentric(gc.lat(), Orb);

		// (1) Make circular path in Cartesian coordinate
		D2<double> xyz(3,As.GetNum());
		for(size_t i=0;i<As.GetNum();++i){
			xyz[0][i] = R0 * cos(Ev) * cos(As[i]);
			xyz[1][i] = R0 * cos(Ev) * sin(As[i]);
			xyz[2][i] = R0 * sin(Ev);
		}


		// (2) Shift to North pole
		for(size_t i=0;i<xyz.GetN();++i){
			xyz[0][i] = xyz[0][i] + 0;
			xyz[1][i] = xyz[1][i] + 0;
			xyz[2][i] = xyz[2][i] + loc_Re;
		}

		// (3) Eular rotation matrix
		double angZ = tar_lon;
		double angY = deg2rad(90) - tar_lat_gc;
		double angX = deg2rad(0);

		D2<double> Mz = Rz(angZ);
		D2<double> My = Ry(angY);
		D2<double> Mx = Rx(angX);
		D2<double> M  = Mz * My * Mx;

		D2<double> xyz2 = M * xyz;


		// (4) Make SV series
		SV<double> sv(xyz2.GetN());
		double dt = 1/PRF;
		D1<double> t = linspace(0., double(xyz2.GetN()-1), 1.0);
		t = t * dt;
		VEC<double> v;

		// Start date/time
		TIME time(2022, 6, 9, 16, 31, 12.345678);

		for(size_t i=0;i<sv.GetNum();++i){
			sv.pos()[i] = VEC<double>(xyz2[0][i], xyz2[1][i], xyz2[2][i]);
		}

		for(size_t i=0;i<sv.GetNum();++i){
			// time
			t[i] = t[i] + UTC2GPS(time);
			// Velocity
			if(i < sv.GetNum()-1){
				v = (sv.pos()[i+1] - sv.pos()[i])/dt;
			}else{
				v = (sv.pos()[i] - sv.pos()[i-1])/dt;
			}
			// Assignment
			sv.t()[i]   = t[i];
			sv.vel()[i] = v;
		}

//	io::write::SV(sv, file_sv_out.c_str());


		if(SHOW){
			// Check: Convert to Lat(Geodetic)/Lon/Alt
			VEC<double> vec(xyz2[0][0], xyz2[1][0], xyz2[2][0]);
			GEO<double> lla = sar::ECR2Gd(vec);

			printf("Input:\n  Lat_geod = %.4f [deg]\n  Lat_geoc = %.4f [deg]\n", rad2deg(tar_lat_gd), rad2deg(tar_lat_gc));
			printf("Latitude = %.10f [deg]\nLongitude = %.10f [deg]\nAltitude = %.10f [m]\n", rad2deg(lla.lat()), rad2deg(lla.lon()), lla.h());
		}

		return sv;
	}

	/**
	 * Find the absoluted Centroid Frequency the MLCC algorithm
	 * @tparam T
	 * @param [in] rc : [azimuth,range]( D2<CPLX<T> > ) range compression data
	 * @param [in] Fr : [Hz] ADC sampling rate (*NOT* BW_rg)
	 * @param [in] f0 : [Hz] Carrier frequency
	 * @param [in] PRF : [Hz] Pulse Repeat Frequency
	 * @param [in] Kr : [Hz/sec] Chirp rate
	 * @param [in] Tr : [sec] Pulse duration time
	 * @param [in] SHOW : Show some information on console or not? (Default=false)
	 * @return A absoluted Doppler centroid frequency in Hz.
	 *
	 * @Ref "A combined SAR Doppler Centroid Estimation Scheme Based upon Signal Phase", 1996
	 */
	template<typename T>
	double find::AbsolutedCentroidFreq::MLCC(const D2<CPLX<T> >& rc, const double Fr, const double f0, const double PRF,
											 const double Kr, const double Tr, const bool SHOW){

			//
			// Input:
			//	rc : [azimuth,range]( D2<CPLX<T> > ) range compression data
			//	Fr : [Hz] Range bandwidth
			//	f0 : [Hz] Carrier frequency
			//	PRF : [Hz]
			//
			D1<size_t> sz = rc.GetDim();	// sz[0]=az, sz[1]=rg
			D2<CPLX<T> > rc_fftx = rc;
			FFTX(rc_fftx);
			fftshiftx(rc_fftx);

			// Sub-looking
			size_t idx_min_L1 = 0;
			size_t idx_max_L1 = sz[1]/2-1;
			size_t idx_min_L2 = sz[1]/2;
			size_t idx_max_L2 = sz[1]-1;
			double df = Fr/2.0;

			// Look-1
			D2<CPLX<T> > L1(sz[0], idx_max_L1-idx_min_L1+1);
			for(size_t i=0;i<sz[0];++i){
				for(size_t j=0;j<L1.GetN();++j){
					L1[i][j] = rc_fftx[i][idx_min_L1+j];
				}
			}

			// Look-2
			D2<CPLX<T> > L2(sz[0], idx_max_L2-idx_min_L2+1);
			for(size_t i=0;i<sz[0];++i){
				for(size_t j=0;j<L2.GetN();++j){
					L2[i][j] = rc_fftx[i][idx_min_L2+j];
				}
			}

			IFFTX(L1);
			IFFTX(L2);

			size_t num_rg = L1.GetN();
			D1<CPLX<T> > ACCC1(num_rg);
			D1<CPLX<T> > ACCC2(num_rg);
			CPLX<T> ACCC1_2_mean;
			for(size_t j=0;j<num_rg;++j){
				D1<CPLX<T> > tmp1 = L1.GetColumn(j);
				D1<CPLX<T> > tmp2 = L2.GetColumn(j);
				ACCC1[j] = sar::find::ACCC(tmp1);
				ACCC2[j] = sar::find::ACCC(tmp2);

				ACCC1_2_mean += ACCC1[j] * ACCC2[j].conj();
			}
			ACCC1_2_mean = ACCC1_2_mean / double(num_rg);


			double dPhi = atan2(ACCC1_2_mean.i(), ACCC1_2_mean.r() );
			double Fdc_abs = -f0*PRF*dPhi/(def::PI2*df);

			CPLX<T> Phi1_cplx = mean(ACCC1);
			CPLX<T> Phi2_cplx = mean(ACCC2);
			double Phi1 = atan2(Phi1_cplx.i(), Phi1_cplx.r());
			double Phi2 = atan2(Phi2_cplx.i(), Phi2_cplx.r());
			double Fdc_base = PRF/def::PI2*(Phi1+Phi2)/2.0;

			// Refine
			double Mamb = round((Fdc_abs - Fdc_base)/PRF);
			double Fdc_abs_refine = Fdc_base + Mamb*PRF;


			if(SHOW) {
				double remainder = Mamb * PRF - (Fdc_abs - Fdc_base);

				string flag;
				if(abs(remainder) > PRF/3.0){
					flag = " > PRF/3("+num2str(PRF/3.)+") (Reject)";
				}else{
					flag = " < PRF/3("+num2str(PRF/3.)+") (Accept)";
				}

				// scene contrast
				double total1, total2;
				double mean1 = 0, mean2 = 0;

				for(size_t i=0;i<rc.GetM();++i){
					for(size_t j=0;j<rc.GetN();++j){
						total2 = rc[i][j].abs();
						total1 = Square(total2);
						mean1 += total1;
						mean2 += total2;
					}
				}

				mean1 = mean1 / double(rc.GetM()*rc.GetN());
				mean2 = Square( mean2 / double(rc.GetM()*rc.GetN()) );
				double Cs = mean1 / mean2;
				string flag_Cs = (Cs > 1.735)? " (High)":" (Low)";

				cout<<"======================================"<<endl;
				cout<<"            MLCC Analysis             "<<endl;
				cout<<"======================================"<<endl;
				cout<<"Fdc_abs_refine = "<<Fdc_abs_refine<<" [Hz]"<<endl;
				cout<<"Fdc_abs        = "<<Fdc_abs<<" [Hz]"<<endl;
				cout<<"Mamb_new       = "<<Mamb<<endl;
				cout<<"Fdc_base       = "<<Fdc_base<<" [Hz]"<<endl;
				cout<<"Contrast       = "<<Cs<<flag_Cs<<endl;
				cout<<"Remainder      = "<<abs(remainder)<<" [Hz]"<<flag<<endl;
				cout<<endl;
			}


			return Fdc_abs_refine;
		}

	//
	// namespace make::
	//
	/**
	 * Make fake flight path
	 * @param [in] PRF : [Hz] Pulse Repeat Frequency
	 * @param [in] file_sv : (string) Write the fake path trajectory into disk.
	 * @param [in] Na : (samples) Number of azimuth
	 */
	void make::FakePath(const double PRF, const string& file_sv, const int Na){

		def::ORB Orb;	// Datum WGS84

		double As_Start = deg2rad(0.5714286566);	// [rad] Start Azimuth angle (Extract from Backhoe data)
		double dAs = deg2rad(0.0714285374);			// [rad] Azimuth angle interval (Extract from Backhoe data)
		double Ev = deg2rad(15);					// [rad] Elevation angle (const.)
		double R0 = 500;							// [m] Local R0

		double tar_lon = deg2rad(20);				// [rad] Target location (Longitude)
		double tar_lat_gd = deg2rad(60);			// [rad] Target location (Latitude) (Geodetic)

		// Make Azimuth angle series
		D1<double> As(Na);
		linspace(0., dAs*(Na-1), As);
		As = As + As_Start;

		cout<<"Na = "<<Na<<endl;
		cout<<"max(As) = "<<rad2deg(max(As))<<" [deg]"<<endl;
		cout<<"min(As) = "<<rad2deg(min(As))<<" [deg]"<<endl;

		sv::SV<double> sv = find::CircularSARPathTrajectory(As, Ev, R0, tar_lon, tar_lat_gd, PRF, true);

		io::write::SV(sv, file_sv.c_str());
	}

	/**
	 * Pesudo SAR generation
	 * @param i
	 * @param bvh
	 * @param inc_mesh
	 * @param dir_out
	 * @param IsWREND
	 * @param IsWPSAR
	 */
	/*
	void make::PesudoSAR(const size_t i, const BVH& bvh, const MeshInc& inc_mesh, const string& dir_out, const bool IsWREND, const bool IsWPSAR){

		// Allocate space for some image pixels
		const unsigned int width  = inc_mesh.LH * 1.1;
		const unsigned int height = inc_mesh.LV * 1.1;

		// Create a camera from position and focus point
		VEC<float> camera_center( inc_mesh.PLOS.x(), inc_mesh.PLOS.y(), inc_mesh.PLOS.z() );
		float dW = inc_mesh.disH[1] - inc_mesh.disH[0];
		float dH = inc_mesh.disV[1] - inc_mesh.disV[0];

		// Misc.
		VEC<float> U(0,0,1); // up vector, Z direction
		D2<float> mesh(height, width);


		//+------------------------------------------------------------------------------------+
		//|                                                                                    |
		//|                                    Ray tracing                                     |
		//|                                                                                    |
		//+------------------------------------------------------------------------------------+
		for(int ii=0;ii<width;++ii){
			for(int jj=0;jj<height;++jj){
				// find u & v unit vector
				VEC<float> d = Unit(VEC<float>(0,0,0) - camera_center);
				VEC<float> v = cross(U, d);
				VEC<float> u = cross(d, v);

				u = VEC<float>(inc_mesh.dirV.x(), inc_mesh.dirV.y(), inc_mesh.dirV.z());
				v = VEC<float>(inc_mesh.dirH.x(), inc_mesh.dirH.y(), inc_mesh.dirH.z());



				VEC<float> off_v = -(ii-width/2.0) *dW * v;
				VEC<float> off_u =  (jj-height/2.0)*dH * u;
				VEC<float> grid_center = camera_center + off_v + off_u;
				VEC<float> grid_dir    = Unit(VEC<float>(0,0,0) - camera_center);
				Ray ray(grid_center, grid_dir);



				IntersectionInfo I, I_Shadow;
				bool hit = bvh.getIntersection(ray, &I, false, 0);

				// shadow ray
				VEC<double> uv_shadow = Unit(VEC<double>(grid_center.x()-I.hit.x(), grid_center.y()-I.hit.y(), grid_center.z()-I.hit.z()));
				Ray rayShadow(I.hit, uv_shadow);
				bool Shadow = bvh.getIntersection(rayShadow, &I_Shadow, false, 0);

				if(!hit){
					mesh[jj][ii] = 0.f;
				}else{
					// original point
					VEC<double> o = ray.o;
					// hit point
					VEC<double> p = I.hit;
					// Distance
					double dis = (p - o).abs();
					// assign to opencv array
					mesh[jj][ii] = (float)dis;
				}

				if(Shadow){ mesh[jj][ii] = 0.f; }
			}
		}

		// Export
		if(IsWREND && IsWPSAR){
			string file_dist = dir_out + "/exp_dis_" + num2str(i) + ".dat";
			string file_rend = dir_out + "/exp_render_" + num2str(i) + ".jpg";
			string file_pSAR = dir_out + "/exp_pSAR_" + num2str(i) + ".jpg";
			// Export the binary file
			io::write::ENVI(mesh, mesh.GetDim(), file_dist);
			// Transfer from d2::D2 to cv::Mat
			Mat mesh_cv = Mat(height, width, CV_32F, mesh.GetPtr());
			// Make Pesudo SAR and export
			WriteImagePesudoSAR(mesh_cv, file_rend, file_pSAR);
		}
	}*/

	//
	// namespace cw::
	//
	template<typename T>
	T cw::EffectiveSwath(const long Nr, const T fs){
		// Nr : Range sample number in frequency domain
		// dt = 1/fs
		return def::C*(((double)Nr-1) - (double)Nr/2)/fs;
	}
	template<typename T>
	T cw::MaxRange(const long Nr, const T fs){
		// Nr : Range sample number in frequency domain
		// dt = 1/fs
		return def::C*(((double)Nr-1) - (double)Nr/2)/fs;
	}
	template<typename T>
	long cw::TraditionalRangeSampleNumber(const T Rmax, const T fs){
		// dt = 1/fs
		return ceil(2*(2*Rmax/(def::C/fs)+1));
	}
	void cw::ShiftSample(const long Nr, const double fs, const double SW, const double dR, const double Rmin, // input
					 long& org, long& nss_int, double& nss_rem){	// output
		
		double SW_eff = cw::EffectiveSwath(Nr, fs);
		double edge   = (SW_eff - SW)/2;
		double org_d  = (Rmin - edge)/dR;
		org = floor(org_d);
		double nss = org_d * (1 + 1/(double)Nr);
		nss_int = long(nss / Nr);
		nss_rem = Mod(nss, (double)Nr);
	}
	void cw::SignalShift(const double nss_org, const double nss_rem, const long Nr, const double Rc, const double dR, D2<CPLX<double> >& s0, D1<double>& R0){
		for(size_t i=0;i<R0.GetNum();++i){
//			R0[i] += ((double)Nr/2 + nss_org)*dR;
			R0[i] += Rc;
		}
		long bias = nss_rem;// - (double)Nr/2;
//		bias += - ((long)mat::round( Rc/(Nr*dR) / (Nr-1) ) + 1);
//		bias += - ((long)mat::round( Rc/(Nr*dR) / (Nr-1) ));
		// bias += - ((long)mat::round( Rc/(Nr*dR) / (Nr-1) ) - 1);

		printf("-nss_rem = %ld\n", long(-nss_rem));
		printf("bias = %ld\n", bias);
		printf("s0.GetM() = %ld\n", s0.GetM());
		printf("s0.GetN() = %ld\n", s0.GetN());
		printf("Nr = %ld\n", Nr);
		printf("bias %% s0.GetN() = %ld\n", bias % long(s0.GetN()));

		// bias = bias % long(s0.GetN());
		bias = 0;

		
		for(size_t j=0;j<s0.GetM();++j){
			shift(s0.GetPtr()+s0.GetN()*j, s0.GetN(), bias);
		}
	}
	
	
	//
    // namespace fft::
    //
	template<typename T>
	void fft::RangeForward(D2<CPLX<T> >& in_out){
		cout<<endl<<"RANGE FFT(Forward)..."<<endl;
		long num_rg = in_out.GetN();
		long num_az = in_out.GetM();
		for(long j=0;j<num_az;++j){
			mat::FFT(in_out.GetPtr()+j*num_rg, num_rg);
			if(mat::Mod(j, 200L) == 0){ cout<<j<<" / "<<num_az<<endl; }
		}
	}
	
	template<typename T>
	void fft::RangeInverse(D2<CPLX<T> >& in_out){
		cout<<endl<<"RANGE FFT(Inverse)..."<<endl;
		long num_rg = in_out.GetN();
		long num_az = in_out.GetM();
		for(long j=0;j<num_az;++j){
			mat::IFFT(in_out.GetPtr()+j*num_rg, num_rg);
			if(mat::Mod(j, 200L) == 0){ cout<<j<<" / "<<num_az<<endl; }
		}
	}
	
	template<typename T>
	void fft::AzimuthForward(D2<CPLX<T> >& in_out){
		cout<<endl<<"AZIMUTH FFT(Forward)..."<<endl;
		long num_rg = in_out.GetN();
		for (long i=0; i<num_rg; ++i) {
			D1<CPLX<T> > col = in_out.GetColumn(i);
			mat::FFT(col);
			in_out.SetColumn(col, i);
			if(mat::Mod(i, 200L) == 0){ cout<<i<<" / "<<num_rg<<endl; }
		}
	}
	
	template<typename T>
	void fft::AzimuthInverse(D2<CPLX<T> >& in_out){
		cout<<endl<<"AZIMUTH FFT(Inverse)..."<<endl;
		long num_rg = in_out.GetN();
		for (long i=0; i<num_rg; ++i) {
			D1<CPLX<T> > col = in_out.GetColumn(i);
			mat::IFFT(col);
			in_out.SetColumn(col, i);
			if(mat::Mod(i, 200L) == 0){ cout<<i<<" / "<<num_rg<<endl; }
		}
	}
	
	template<typename T>
	void fft::RangeShift(D2<CPLX<T> >& in_out_m_az_n_rg){
		long num_az = in_out_m_az_n_rg.GetM();
		long num_rg = in_out_m_az_n_rg.GetN();
		
		for(long j=0;j<num_az;++j){
			mat::shift(in_out_m_az_n_rg.GetPtr() + j*num_rg, num_rg, num_rg/2);
		}
	}
	
	template<typename T>
	void fft::AzimuthShift(D2<CPLX<T> >& in_out_m_az_n_rg){
		long num_rg = in_out_m_az_n_rg.GetN();
		
		for(long i=0;i<num_rg;++i){
			// Get Azimuth(column) data
			D1<CPLX<T> > col = in_out_m_az_n_rg.GetColumn(i);
			// fftshift
			mat::fftshift(col);
			// Set values
			in_out_m_az_n_rg.SetColumn(col, i);
		}
	}
	
	//
	// namespace csa::
	//
	template<typename T>
	D1<T> csa::MigrationParameter(const D1<T>& fn, const T Vr, const T lambda){
	/*
	 Purpose:
		 Find Migration parameter D (7.17)
	 Input:
		 fn		: [Hz] Instantaneous Doppler frequency
		 Vr		: [m/sec] Effective velocity
		 lambda : [m] Wavelength
	 Return:
		 D		: (D1<T>)Return the Migration parameter D
	 */
		// Find Migration parameter D (7.17)
		long num = fn.GetNum();
		D1<T> D(num);
		for(long i=0;i<num;++i){
			D[i] = sqrt( 1.0 - fn[i]*fn[i]*lambda*lambda / (4.0*Vr*Vr) );
		}
		return D;
	}
	
	template<typename T>
	T csa::MigrationParameter(const T fn, const T Vr, const T lambda){
	/*
	 Purpose:
		 Find Migration parameter D (7.17)
	 Input:
		 fn		: [Hz] Instantaneous Doppler frequency
		 Vr		: [m/sec] Effective velocity
		 lambda : [m] Wavelength
	 Return:
		 D		: (D1<T>)Return the Migration parameter D
	 */
		// Find Migration parameter D (7.17)
		T D = sqrt( 1.0 - fn*fn*lambda*lambda / (4.0*Vr*Vr) );
		return D;
	}
	
	template<typename T>
	D2<T> csa::ModifiedChirpRate(const D1<T>& fn, const T Vr, const T Kr, const D1<T>& R0, const T c, const T f0){
	/*
	 Purpose:
		 Find Migration parameter D (7.17)
	 Input:
		 fn	: [Hz] Instantaneous Doppler frequency
		 Vr	: [m/sec] Effective velocity
		 Kr	: [Hz/s] Original Chirp rate
		 R0	: [m] (D1<T>) Slant range
		 c	: [m/s] Light speed
		 f0	: [m/s] Center frequency
	 Return:
		 Km : (D2<T>)Refine Chirp rate
	 */
		// Find Modified Doppler Rate (7.18)
		long num_rg = R0.GetNum();
		long num_az = fn.GetNum();
		D1<T> D = sar::csa::MigrationParameter(fn, Vr, c/f0);
		D2<T> Km(num_az);
		for(long j=0;j<num_az;++j){
			for(long i=0;i<num_rg;++i){
				Km[j][i] = Kr / (1.0 - Kr*(c*R0[i]*fn[j]*fn[j])/(2.0*Vr*Vr*f0*f0*f0*D[j]*D[j]*D[j]));
			}
		}
		return Km;
	}
	
	template<typename T>
	T csa::ModifiedChirpRate(const T fn, const T Vr, const T Kr, const T R0, const T c, const T f0){
	/*
	 Purpose:
		 Find Migration parameter D (7.17)
	 Input:
		 fn	: [Hz] Instantaneous Doppler frequency
		 Vr	: [m/sec] Effective velocity
		 Kr	: [Hz/s] Original Chirp rate
		 R0	: [m] Slant range
		 c	: [m/s] Light speed
		 f0	: [m/s] Center frequency
	 Return:
		 Km : Refine Chirp rate
	 */
		// Find Modified Doppler Rate (7.18)
		T D = sar::csa::MigrationParameter(fn, Vr, c/f0);
		T Km = Kr / (1.0 - Kr*(c*R0*fn*fn)/(2.0*Vr*Vr*f0*f0*f0*D*D*D));
		return Km;
	}

}


#ifdef _MSC_VER
// for VC2005 declared deprecated warring
#undef strcpy
#endif

#endif // SAR_H_INCLUDED
