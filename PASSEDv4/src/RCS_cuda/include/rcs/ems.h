//
//  ems.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 1/29/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef ems_h
#define ems_h

#include <sar/def.h>
#include <basic/vec.h>
#include <basic/cplx.h>
#include <bvh/triangle.h>
#include <mesh/mesh.h>
#include <basic/opt.h>
#include <coordinate/sph.h>
#include <rcs/material.h>
#include <sar/sar.h>

using namespace vec;
using namespace cplx;
using namespace mesh;
using namespace sph;
using namespace material;

// Electric Magnetic Simulation namespace
namespace ems{
	//+===============================+
	//|  E Field Amplitude component  |
	//+===============================+
	// Assign Tx Polarization
	class EAmp{
	public:
		EAmp(){};
		EAmp(const CPLX<double>& et, const CPLX<double>& ep):Et(et),Ep(ep){};
	public:
		CPLX<double> Et, Ep;
	};
	//+=======================================+
	//|  Theta and Phi value component class  |
	//+=======================================+
	class ThetaPhi{
	public:
		ThetaPhi():theta(0),phi(0){};
		ThetaPhi(const double Theta, const double Phi){
			theta = Theta;
			phi   = Phi;
		}
		void Print(){
			cout<<"["<<theta<<","<<phi<<"]"<<endl;
		}
	public:
		double theta, phi;
	};
	//+========================================+
	//|  Theta and Phi angle / direction class |
	//+========================================+
	class ThetaPhiVec{
	public:
		ThetaPhiVec(){}
		ThetaPhiVec(const VEC<double>& Theta_vec, const VEC<double>& Phi_vec){
			theta_vec = Theta_vec;
			phi_vec   = Phi_vec;
		}
		void Print(){
			cout<<"theta_vec = "; theta_vec.Print();
			cout<<"phi_vec   = "; phi_vec.Print();
		}
	public:
		VEC<double> theta_vec, phi_vec;
	};
	//+===================+
	//|    Scattering     |
	//+===================+
	class Scatter{
	public:
		Scatter(){
			Level = -1;
		};
		Scatter(const VEC<CPLX<double> >& Cplx, const CPLX<double>& ETS, const CPLX<double>& EPS, const long LEVEL){
			cplx = Cplx;
			Ets = ETS;
			Eps = EPS;
			Level = LEVEL;
		}
		void Print(){
			cout<<"+-----------------------+"<<endl;
			cout<<"|    Scatter Results    |"<<endl;
			cout<<"+-----------------------+"<<endl;
			cout<<"Level = "<<Level<<endl;
			cout<<"cplx  = "; cplx.Print();
			cout<<"Ets   = "; Ets.Print();
			cout<<"Eps   = "; Eps.Print();
		}
	public:
		VEC<CPLX<double> > cplx;
		CPLX<double> Ets, Eps;
		long Level;
	};
	//+=======================================+
	//|            Mics. functions            |
	//+=======================================+
	double Intensity2dBsm(const double intensity, const double lambda){
		return 10*log10( def::PI4/(lambda*lambda) * (intensity*intensity) );
	}

	namespace find{
		ThetaPhiVec ThetaPhiVector(const VEC<double>& k){
			VEC<double> kk = -k;
			double xy = sqrt(kk.x()*kk.x() + kk.y()*kk.y());
			double st = xy;
			double ct = kk.z();
			double sp = kk.y()/xy;
			double cp = kk.x()/xy;

			VEC<double> theta_vec(ct*cp, ct*sp, -st);
			VEC<double> phi_vec(-sp, cp, 0);

			ThetaPhiVec out(theta_vec, phi_vec);//, theta, phi);
			return out;
		}

		void LocalIncThetaPhiVector(const VEC<double>& k, const VEC<double>& n,	// input
									ThetaPhiVec& local_inc, VEC<double>& X_vec, VEC<double>& Y_vec, VEC<double>& Z_vec, // output
									double& local_inc_theta // output
		){
			// 反射面的normal vector
			VEC<double> m = Unit(cross(-k,n));
			// 找local XYZ
			X_vec = cross(m,n);
			Y_vec = -m;
			Z_vec = -n;
			// 找local incident angle
			double ct;
			local_inc_theta = vec::angle(-k, n, ct);
			double st = sqrt(1-ct*ct);
			// 找local incident theta 和 phi 向量
			local_inc.theta_vec = Unit(ct * X_vec - st * Z_vec);
			local_inc.phi_vec   = Y_vec;
		}

		void LocalRefThetaPhiVector(const double theta_c_i, const VEC<double>& X_vec, const VEC<double>& Y_vec, const VEC<double>& Z_vec, // input
									ThetaPhiVec& Local_ref	// output
		){
			// Reflection
			double st = sin(theta_c_i);
			double ct = cos(theta_c_i);
			Local_ref.theta_vec = Unit(-ct * X_vec - st * Z_vec);
			Local_ref.phi_vec   = Y_vec;
		}

		D2<double> transfmatrix(const double a, const double b){
			D2<double> T1(3,3), T2(3,3);
			// T1 matrix
			T1[0][0] = cos(a);	T1[0][1] = sin(a);	T1[0][2] = 0;
			T1[1][0] =-sin(a);	T1[1][1] = cos(a);	T1[1][2] = 0;
			T1[2][0] = 0;		T1[2][1] = 0;		T1[2][2] = 1;
			// T2 matrix
			T2[0][0] = cos(b);	T2[0][1] = 0;		T2[0][2] =-sin(b);
			T2[1][0] = 0;		T2[1][1] = 1;		T2[1][2] = 0;
			T2[2][0] = sin(b);	T2[2][1] = 0;		T2[2][2] = cos(b);
			// T21 matrix
			return T2 * T1;
		}

		template<typename T>
		VEC<T> transform(const double sa, const double ca, const double sb, const double cb, const VEC<T> in){
			VEC<T> tmp((ca * in.x()) + (sa * in.y()),
					   (-sa * in.x()) + (ca * in.y()),
					   in.z() );
			return VEC<T>((cb * tmp.x()) - (sb * tmp.z()),
						  tmp.y(),
						  (sb * tmp.x()) + (cb * tmp.z()) );
		}

		VEC<double> spher2cart(const SPH<double> sph){
			double R     = sph.R();
			double theta = sph.Theta();
			double phi   = sph.Phi();

			double st = sin(theta);
			double ct = cos(theta);
			double sp = sin(phi);
			double cp = cos(phi);
			return VEC<double>(R*st*cp, R*st*sp, R*ct);
		}

		VEC<double> spher2cart(const double st, const double ct, const double sp, const double cp, const double R){
			return VEC<double>(R*st*cp, R*st*sp, R*ct);
		}

		SPH<double> cart2spher(const VEC<double> in){
			double x2 = in.x() * in.x();
			double y2 = in.y() * in.y();
			double z2 = in.z() * in.z();
			return SPH<double>(sqrt(x2 + y2 + z2), atan2(sqrt(x2 + y2),in.z()), atan2(in.y(),in.x()));
		}

		void GetSpherTheta(const VEC<double>& in, double& cti, double& sti){
			double x2 = in.x()*in.x();
			double y2 = in.y()*in.y();
			double z2 = in.z()*in.z();
			double inv_r = 1.0/sqrt(x2 + y2 + z2);
			double e  = sqrt(x2 + y2);

			cti = in.z() * inv_r;
			sti = e * inv_r;
		}

		SPH<double> spherglobal2local(const SPH<double>& sph,const D2<double>& T21){
			// convert to cartesian,
			VEC<double> xyz = spher2cart(sph);
			// transform to local coordinates
			VEC<double> tmp = D22VEC(T21 * VEC2D2(xyz, 3, 1));
			// convert to spherical
			return cart2spher(tmp);
		}

		void ReflCoeff_General(const CPLX<double>& er1,const CPLX<double>& mr1,const CPLX<double>& er2,const CPLX<double>& mr2, // input
							   const double& cti, const double& sti,															// input
							   CPLX<double>& gammapar, CPLX<double>& gammaperp){
			//er1,mr1 relative parameters of space of incidence
			//er2,mr2 relative parameters of space of tranmsission
			//thetai: angle of incidence
			//gammapar, gammaperp: reflection coefficients parallel and perpendicular
			//thetat: transmission angle
			//IsTIR=ture when Total Internal Reflection occurs, else TIR=0


			CPLX<double> m0(def::PI4*1e-7, 0);
			CPLX<double> e0(8.854e-12, 0);

//			double thetat = 0;
			double stt = sti*sqrt( er1.r()*mr1.r() / (er2.r()*mr2.r()) );
			double ctt;
			if(stt > 1.0){ // Total Internal Reflection occurs
//				thetat = def::PI/2.0;
				ctt = 0;
			}else{
				ctt = sqrt(1-stt*stt);
			}

//			CPLX<double> n1( sqrt(m0.r()/e0.r()), 0);
//			CPLX<double> n2 = (mr2*m0/(er2*e0)).sqrt();
//			
//			gammaperp = (n2*cti-n1.r()*ctt)/(n2*cti+n1.r()*ctt);
//			gammapar  = (n2*ctt-n1.r()*cti)/(n2*ctt+n1.r()*cti);

//			CPLX<double> tp = (m0/e0).sqrt();
			CPLX<double> tp( sqrt(m0.r()/e0.r()), 0 );
			CPLX<double> n1 = tp * (mr1/er1).sqrt();
			CPLX<double> n2 = tp * (mr2/er2).sqrt();
//			CPLX<double> n1 = (mr1*m0/(er1*e0)).sqrt();
//			CPLX<double> n2 = (mr2*m0/(er2*e0)).sqrt();

			CPLX<double> n2_cti = n2*cti;
			CPLX<double> n2_ctt = n2*ctt;
			CPLX<double> n1_cti = n1*cti;
			CPLX<double> n1_ctt = n1*ctt;

			gammaperp = (n2_cti-n1_ctt)/(n2_cti+n1_ctt);
			gammapar  = (n2_ctt-n1_cti)/(n2_ctt+n1_cti);


//			gammaperp = (n2*cti-n1*ctt)/(n2*cti+n1*ctt);
//			gammapar  = (n2*ctt-n1*cti)/(n2*ctt+n1*cti);

//			CPLX<double> m0_e0 = m0/e0;
//			CPLX<double> n1 = (mr1/er1*m0_e0).sqrt();
//			CPLX<double> n2 = (mr2/er2*m0_e0).sqrt();
//			
//			CPLX<double> n1_cti = n1 * cti;
//			CPLX<double> n1_ctt = n1 * ctt;
//			CPLX<double> n2_cti = n2 * cti;
//			CPLX<double> n2_ctt = n2 * ctt;
//			
//			gammaperp = (n2_cti-n1_ctt)/(n2_cti+n1_ctt);
//			gammapar  = (n2_ctt-n1_cti)/(n2_ctt+n1_cti);
		}

		void ReflCoeff_Air(const CPLX<double>& er2,const CPLX<double>& mr2, // input
						   const double& cti,															// input
						   CPLX<double>& gammapar, CPLX<double>& gammaperp){
			//er1,mr1 relative parameters of space of incidence (restrict in free space = air)
			//er2,mr2 relative parameters of space of tranmsission
			//thetai: angle of incidence
			//gammapar, gammaperp: reflection coefficients parallel and perpendicular
			//thetat: transmission angle
			//IsTIR=ture when Total Internal Reflection occurs, else TIR=0


			CPLX<double> m0(def::PI4*1e-7, 0);
			CPLX<double> e0(8.854e-12, 0);

			double n1 = sqrt(m0.r()/e0.r());
			CPLX<double> n2 = n1 * (mr2/er2).sqrt();
			CPLX<double> n2_cti = n2*cti;
			double n1_cti = n1*cti;


			double stt, ctt;
			if(er2.r() > 1E15){ // Total Internal Reflection


				gammaperp = (n2_cti-n1)/(n2_cti+n1);
				gammapar  = (n2-n1_cti)/(n2+n1_cti);


			}else{
				stt = sqrt( (1 - cti*cti) / (er2.r()*mr2.r()) );
				ctt = sqrt(1-stt*stt);

				CPLX<double> n2_ctt = n2*ctt;
				double n1_ctt = n1*ctt;

				gammaperp = (n2_cti-n1_ctt)/(n2_cti+n1_ctt);
				gammapar  = (n2_ctt-n1_cti)/(n2_ctt+n1_cti);
			}
		}

		RF GetReflectinoFactor(const double st, const double ct, const double sp, const double cp,
							   const Material& MD, const double freq,
							   const double sa, const double ca, const double sb, const double cb,
							   const long IdxLevel, const long idx){

			CPLX<double> er(MD.er_r(), -MD.tang()*MD.er_r());
			CPLX<double> mr(MD.mr(), -MD.mi());

			double t = MD.d()*0.001;		// convert to meters

			// convert to local facet coordinates
			// Reduce to...
			double cti = sb*(st*(ca*cp + sa*sp)) + cb*ct;


			// angle of incidence is thetaloc
			// 1st interface: air to material interface
			CPLX<double> G1par, G1perp;
			ReflCoeff_Air(er, mr, cti, G1par, G1perp);

			// 2nd interface: material to air interface
			CPLX<double> G2par  = -G1par;
			CPLX<double> G2perp = -G1perp;
			// find phase
			double b1 = def::PI2*freq*sqrt(er.r()*mr.r())/def::C;
			double phase = fmod(b1*t, def::PI2);

			// formulate matrices
			double cphs, sphs;
			_sincos(phase, sphs, cphs);
			CPLX<double> exp1(cphs, sphs);
			CPLX<double> exp2(exp1.r(), -exp1.i());

			// compute Reflection Coefficients
			CPLX<double> Mpar00  = exp1 + G1par  * exp2 * G2par;
			CPLX<double> Mpar10  = G1par * exp1  + exp2 * G2par;
			CPLX<double> Mperp00 = exp1 + G1perp * exp2 * G2perp;
			CPLX<double> Mperp10 = G1perp * exp1 + exp2 * G2perp;

			RF Rf;
			Rf.TM_par()  = Mpar10/Mpar00;	// RCpar (parallel) - TM
			Rf.TE_perp() = Mperp10/Mperp00;	// RCperp (parpendicular) - TE

			return Rf;
		}

	}

	template<typename T>
	void CheckComplexZero(CPLX<T>& in){
		in.r() = (in.r() < 1e-15)? 0:in.r();
		in.i() = (in.i() < 1e-15)? 0:in.i();
	}


	//+=========================+
	//|   Electric Field class  |
	//+=========================+
	class ElectricField{
	public:
		ElectricField(){}
		ElectricField(const VEC<double>& K, const VEC<double>& O, const VEC<CPLX<double> >& Cplx, const bool EnableThetaPhiVector=false){
			k = K; o = O; cplx = Cplx;
			// Find Global unit vector
			if(EnableThetaPhiVector == true){
				g = find::ThetaPhiVector(K);
			}
		}
		ElectricField(const VEC<double>& K, const CPLX<double>& Et, const CPLX<double>& Ep, const VEC<double>& O){

			// Find Global unit vector
			g = find::ThetaPhiVector(K);

			cplx = VEC<CPLX<double> >(Ep * g.phi_vec.x(),   Ep * g.phi_vec.y(),   Ep * g.phi_vec.z())   +
				   VEC<CPLX<double> >(Et * g.theta_vec.x(), Et * g.theta_vec.y(), Et * g.theta_vec.z());

			// Create
			k = K;
			o = O;
		}
		void AddPhase(const double phs){
//			double PHS = fmod(phs, def::PI2);
//			CPLX<double> phase = mat::exp(-PHS);
			CPLX<double> phase = mat::exp(+phs);
			cplx.x() = cplx.x() * phase;
			cplx.y() = cplx.y() * phase;
			cplx.z() = cplx.z() * phase;
		}
		CPLX<double> GetCplxThetaComp(){
			CPLX<double> cmp_theta = dot(cplx, g.theta_vec);
			CheckComplexZero(cmp_theta);
			return cmp_theta;
		}
		CPLX<double> GetCplxPhiComp(){
			CPLX<double> cmp_phi = dot(cplx, g.phi_vec);
			CheckComplexZero(cmp_phi);
			return cmp_phi;
		}
		double GetPhaseThetaComp(){
			return GetCplxThetaComp().phase();
		}
		double GetPhasePhiComp(){
			return GetCplxPhiComp().phase();
		}
		double GetAmpThetaComp(){
			// Complex component
			CPLX<double> cmp_theta = GetCplxThetaComp();
			// Phase
			double phs_theta = cmp_theta.phase();
			// signed amplitude
			double theta_sgn = (abs(phs_theta) > deg2rad(90))? -1:1;
			// Amplitude
			return theta_sgn * cmp_theta.abs();
		}
		double GetAmpPhiComp(){
			// Complex component
			CPLX<double> cmp_phi = GetCplxPhiComp();
			// Phase
			double phs_phi = cmp_phi.phase();
			// signed amplitude
			double phi_sgn = (abs(phs_phi) > deg2rad(90))? -1:1;
			// Amplitude
			return phi_sgn * cmp_phi.abs();
		}
		VEC<double> GetElectricFieldDirection(){
			double amp_theta = GetAmpThetaComp();
			double amp_phi   = GetAmpPhiComp();
			return amp_theta * g.theta_vec + amp_phi * g.phi_vec;
		}
		ThetaPhiVec GetThetaPhiInfo(){ return g; }
		void Print(){
			cout<<"+--------------------+"<<endl;
			cout<<"|   E-field Class    |"<<endl;
			cout<<"+--------------------+"<<endl;
			cout<<"k      = "; k.Print();
			cout<<"o      = "; o.Print();
			cout<<"cplx   = "; cplx.Print();
		}
	public:
		VEC<double> k;		// EM wave direction
		VEC<double> o;		// EM wave location inc global XYZ
		VEC<CPLX<double> > cplx;	// complex type of electric field
		ThetaPhiVec g;		// theta_vec & phi_vec infomation
	};


	class PO{
	public:
		PO(){
			Init(1);
		}
		PO(long MaxLevel){
			Init(MaxLevel);
		}
		/**
		 Pre-calculate for PO approximatioin, return a data set of sti2, cti2, cpi2, spi2, u_ui, v_vi, w_wi, 
		 uu, vv, ww, ca, sa, cb, sb, sp, cp
		 @param Ei  [in] Incident electric field class
		 @param tri  [in] Triangle polygon class pointer
		 @param k_obv  [in] Obsercation firection vector is *NOT* same as Ei.k
		 @param i  [in] Index for differenct vector<SBR> index (different lambda)
		 @see SBR
		 */
		template<typename T> void PreCalculate(const ElectricField& Ei, const TRI<T>* tri, const VEC<double>& k_obv,
											   const long i, const long idx){
			// PO
			//------------------------------------------------------------------
			// Pre-calculate
			//------------------------------------------------------------------
			// Incident vector
			VEC<double> D0i = -Ei.k;
			double ui = D0i.x();
			double vi = D0i.y();
			double wi = D0i.z();

			// Observation vector
			VEC<double> Rs = k_obv;
			double u = k_obv.x();
			double v = k_obv.y();
			double w = k_obv.z();
			// cos & sin of theta (angle from z-axis to vector)
//			double theta = vec::angle(VEC<double>(0,0,1), k_obv);
//			double ct = cos(theta);
//			double st = sin(theta);
			double ct = w;
			double st = sqrt(1 - ct*ct);
			// cos & sin of phi (angle from x-axis to projected vector)
			double uv = sqrt(u*u + v*v);
			sp[i] = v/uv;
			cp[i] = u/uv;

			uu[i] = ct*cp[i];
			vv[i] = ct*sp[i];
			ww[i] = -st;


			// Get Transform incidence quantities
			ca[i] = tri->getCa(); sa[i] = tri->getSa();
			cb[i] = tri->getCb(); sb[i] = tri->getSb();
			// Transform the incident vector in GLOBAL coordinate (D0i) to LOCAL coordinate (D2i)
			// T21 * D0i
			VEC<double> D2i = find::transform(sa[i], ca[i], sb[i], cb[i], D0i);
//			VEC<double> D1i( ca[i] * D0i.x() + sa[i] * D0i.y(),
//							-sa[i] * D0i.x() + ca[i] * D0i.y(),
//							 D0i.z() );
//			VEC<double> D2i( cb[i] * D1i.x() - sb[i] * D1i.z(),
//							 D1i.y(),
//							 sb[i] * D1i.x() + cb[i] * D1i.z() );

			// Get D2i component
			double ui2 = D2i.x(); double vi2 = D2i.y(); double wi2 = D2i.z();
			// Find incident spherical angles in local coordinate
			double edge = sqrt(ui2*ui2 + vi2*vi2);
//			cti2[i] = wi2;
//			sti2[i] = sqrt(1- cti2[i]*cti2[i]);
			sti2[i] = edge * sign(wi2);
//			cti2[i] = sqrt(1 - sti2[i]*sti2[i]);
			cti2[i] = abs(wi2);
			cpi2[i] = ui2 / edge;
			spi2[i] = vi2 / edge;

//			// Reflection coefficients (Rss is normalized to Z0)
//			perp[i] = -1/(2*Rss*cti2[i]+1);	// local TE polarization
//			para[i] = 0;
//			// local TM polarization
//			if( (2*Rss+cti2[i]) != 0 ){
//				para[i] = -cti2[i]/(2*Rss+cti2[i]);
//			}

			// P2到P0 向量 dot 半徑向量(Rs)，P2到P0向量 在 半徑向量 的分量
			u_ui[i] = u+ui; v_vi[i] = v+vi; w_wi[i] = w+wi;


//			if(idx == 31 && i == 0){
//				printf("k = %d, sa = %.20f, ca = .20f, sb = %.20f, cb = %.20f\n", idx, sa[i], ca[i], sb[i], cb[i]);
//				printf("k = %d, D0i         = [%.20f,%.20f,%.20f]\n", idx, D0i.x(), D0i.y(), D0i.z());
//				printf("k = %d, D2i         = [%.20f,%.20f,%.20f]\n", idx, D2i.x(), D2i.y(), D2i.z());
//				printf("k = %d, edge        = %.20f\n", idx, edge);
////				printf("k = %d, (u,v,w) = (%.20f,%.20f,%.20f), (ui,vi,wi) = (%.20f,%.20f,%.20f)\n", idx, u, v, w, ui, vi, wi);
//			}


		}
		/**
		 PO approximation claculation in PO class, calling CoreCalculate,
		 return a scatter electric field, Es, in this(PO) class
		 @param Ei  [in] Incident electric field class
		 @param tri  [in] Triangle polygon class pointer
		 @param k0  [in] [Hz/m] wavenumber
		 @param k_obv  [in] Obsercation firection vector is *NOT* same as Ei.k
		 @param o  [in] Intersection point of Ray and triangle polygon
		 @param g  [in] Global angle vector (theta_vec & phi_vec) in ThetaPhiVec class
		 @param Taylor  [in] Taylor series approximation coefficent class
		 @param Rf  [in] Relection Factor class calculating by GetRF
		 @param Level_i  [in] Boucing level index
		 @see CoreCalculate
		 @see GetRf
		 */
		template<typename T> void Calculate(const ElectricField& Ei, const TRI<T>* tri, const double k0,		// input
											const VEC<double>& k_obv, const VEC<double>& o, const ThetaPhiVec g,	// input
											const TAYLOR& Taylor, const RF& Rf, const long Level_i, const long idx){
			Scatter po;
			// PO
//			CoreCalculate(Ei, tri, k0, k_obv, Taylor, po.Ets, po.Eps, Rf, Level_i, idx);
			CoreCalculate2(Ei, tri, k0, po.Ets, po.Eps, Rf, Level_i, idx);
			// Complex form
			VEC<CPLX<double> > Es_cplx( po.Ets * g.theta_vec.x() + po.Eps * g.phi_vec.x(),
										po.Ets * g.theta_vec.y() + po.Eps * g.phi_vec.y(),
										po.Ets * g.theta_vec.z() + po.Eps * g.phi_vec.z() );
			Es = ElectricField(k_obv, o, Es_cplx);
		}
		/**
		 Calculate the Relection Factor TE(Perpendicular) & TM(Parallel) for polarization.
		 @param f0  [in] [Hz] Center frequency
		 @param tri  [in] Triangle polygon class pointer
		 @param MAT  [in] Single Material class
		 @param IdxLevel  [in] Certain bouncing index
		 @returns Return a reflection factor class
		 */
		template<typename T> RF GetRF(const double f0, const TRI<T>* tri, const Material& MAT, const long IdxLevel, const long idx){
//			double ithetar = asin(sti2[IdxLevel]);
//			double iphir   = asin(spi2[IdxLevel]);
//			
//			double st = sin(ithetar);
//			double ct = cos(ithetar);
//			double sp = sin(iphir);
//			double cp = cos(iphir);

			double st = sti2[IdxLevel];
			double ct = cti2[IdxLevel];
			double sp = spi2[IdxLevel];
			double cp = cpi2[IdxLevel];

			return find::GetReflectinoFactor(st, ct, sp, cp, MAT, f0, tri->getSa(), tri->getCa(), tri->getSb(), tri->getCb(), IdxLevel, idx);
//			double ithetar = asin(sti2[IdxLevel]);
//			double iphir   = asin(spi2[IdxLevel]);
//
//			return find::GetReflectinoFactor(ithetar, iphir, MAT, f0, tri->SA(), tri->CA(), tri->SB(), tri->CB());
////			return find::GetReflectinoFactor(ithetar, iphir, MAT, tri->ALPHA(), tri->BETA(), f0);
		}
		ElectricField GetEs(){ return Es; }
	private:
		void Init(const long MaxLevel = 1){
			sti2.resize(MaxLevel);
			cti2.resize(MaxLevel);
			cpi2.resize(MaxLevel);
			spi2.resize(MaxLevel);
//			perp.resize(MaxLevel);
//			para.resize(MaxLevel);
			u_ui.resize(MaxLevel);
			v_vi.resize(MaxLevel);
			w_wi.resize(MaxLevel);
			uu.resize(MaxLevel);
			vv.resize(MaxLevel);
			ww.resize(MaxLevel);
			ca.resize(MaxLevel);
			sa.resize(MaxLevel);
			cb.resize(MaxLevel);
			sb.resize(MaxLevel);
			sp.resize(MaxLevel);
			cp.resize(MaxLevel);
		}
		// Physical Optics function
		VEC<double> D2xVEC(const D2<double>& M, const VEC<double>& V){
			return VEC<double>(M[0][0] * V.x() + M[0][1] * V.y() + M[0][2] * V.z(),
							   M[1][0] * V.x() + M[1][1] * V.y() + M[1][2] * V.z(),
							   M[2][0] * V.x() + M[2][1] * V.y() + M[2][2]* V.z() );
		}
		VEC<CPLX<double> > D2xVEC(const D2<double>& M, const VEC<CPLX<double> >& V){
			return VEC<CPLX<double> >(M[0][0] * V.x() + M[0][1] * V.y() + M[0][2] * V.z(),
									  M[1][0] * V.x() + M[1][1] * V.y() + M[1][2] * V.z(),
									  M[2][0] * V.x() + M[2][1] * V.y() + M[2][2] * V.z() );
		}
		double PowInt(const double val, const int pow){
			double out = val;
			if(pow == 0){
				return 1;
			}else if(pow == 1){
				return val;
			}else{
				for(int i=2;i<=pow;++i){
					out *= out;
				}
				return out;
			}
		}
		CPLX<double> PowCPLX(const CPLX<double>& in, const int pow){
			CPLX<double> out(1,0);	// n = 0
			if(pow > 0){
				for(int i=1;i<pow+1;++i){
					out = out * in;
				}
			}
			return out;
		}
		CPLX<double> PowJPhase(const double img, const int pow){
			if((pow % 2) == 0){	// assign to real_part
				double sign = (((pow-2) % 4) == 0)? -1:1;
				return CPLX<double>(sign*PowInt(img, pow),0);
			}else{				// assign to imag_part
				double sign = (((pow-3) % 4) == 0)? -1:1;
				return CPLX<double>(0,sign*PowInt(img, pow));
			}
		}
		CPLX<double> PowJPhasexCPLX(const double img, const int pow, const CPLX<double>& c){
			if((pow % 2) == 0){	// assign to real_part
				double sign = (((pow-2) % 4) == 0)? -1:1;
				double v = sign*PowInt(img, pow);
				return CPLX<double>(v*c.r(), v*c.i());
			}else{				// assign to imag_part
				double sign = (((pow-3) % 4) == 0)? -1:1;
				double v = sign*PowInt(img, pow);
				return CPLX<double>(-v*c.i(), v*c.r());
			}
		}
		CPLX<double> CPLXxImg(const CPLX<double>& cx, const double img){
			return CPLX<double>(-cx.i()*img, cx.r()*img);
		}
		CPLX<double> G(const int n, const double w, const CPLX<double>& expjw){
			double w1 = 1.0/w;
			double w2 = double(n)/w;
			CPLX<double> expjw_jw = CPLX<double>(expjw.i()*w1, -expjw.r()*w1);
			CPLX<double> g = CPLX<double>(expjw.i()*w1, -(expjw.r()-1.0)*w1);
			CPLX<double> go;
			if(n > 0){
				for(int m=0;m<n;++m){
					go = g;
					g.r() = expjw_jw.r() - go.i()*w2;
					g.i() = expjw_jw.i() + go.r()*w2;
				}
			}
			return g;
		}
		CPLX<double> G3(const int n, const float w){
			CPLX<double> j(0,1);
			CPLX<double> _jw(0,-w);
			CPLX<double> expjw(cos(double(w)), sin(double(w)));
			CPLX<double> sum(0,0);

			for(int i=0;i<=n;++i){
				sum += PowCPLX(_jw, i) / mat::factrl(i);
			}

			CPLX<double> g = mat::factrl(n)/PowCPLX(_jw, n+1) * (double(1) - expjw*sum);
			return g;
		}
		template<typename T> void CoreCalculate(const ElectricField& Ei, const TRI<T>* tri, const double k0,		// input
												const VEC<double>& k_obv, const TAYLOR& Taylor,						// input
												CPLX<double>& Ets, CPLX<double>& Eps, const RF& Rf, const long i, const long idx){				// output

			// residual
			// E-field Amplitude
			double Co = (abs(Ei.cplx)).abs();

			// Incident field in global Cartesian coordinates (Bistatic)
			VEC<CPLX<double> > Rc = Ei.cplx;

			// Incident field in local Cartesian coordinates (stored in e2)
			VEC<CPLX<double> > e2 = find::transform(sa[i], ca[i], sb[i], cb[i], Rc);

			VEC<CPLX<double> > e2_1( ca[i] * Rc.x() + sa[i] * Rc.y(),
									 -sa[i] * Rc.x() + ca[i] * Rc.y(),
									 Rc.z() );

			// Incident field in local spherical coordinates
			CPLX<double> Et2 =        e2.x()*cti2[i]*cpi2[i] + e2.y()*cti2[i]*spi2[i] - e2.z()*sti2[i];
			CPLX<double> Ep2 = (-1.0)*e2.x()*spi2[i]         + e2.y()*cpi2[i];
			// Surface current components in local Cartesian coordinates
			CPLX<double> tp1 = (-1.0)*Et2*Rf.TM_par();
			CPLX<double> tp2 = Ep2*Rf.TE_perp()*cti2[i];
			CPLX<double> Jx2 = tp1*cpi2[i] + tp2*spi2[i];	// cti2 added
			CPLX<double> Jy2 = tp1*spi2[i] - tp2*cpi2[i];	// cti2 added
//			CPLX<double> Jx2 = (-1.0)*Et2*cpi2[i]*Rf.TM_par() + Ep2*spi2[i]*Rf.TE_perp()*cti2[i];	// cti2 added
//			CPLX<double> Jy2 = (-1.0)*Et2*spi2[i]*Rf.TM_par() - Ep2*cpi2[i]*Rf.TE_perp()*cti2[i];	// cti2 added


			// cuVEC<T> P0, P1;		// Edge vector P0=v0-v2, P1=v1-v2, P2=[0,0,0]
			VEC<double> P0 = tri->v0() - tri->v2();
			VEC<double> P1 = tri->v1() - tri->v2();

			double Dp = k0 * ( P0.x()*u_ui[i] + P0.y()*v_vi[i] + P0.z()*w_wi[i] );
			// P2到P1 向量 dot 半徑向量(Rs)，P2到P1向量 在 半徑向量 的分量
			double Dq = k0 * ( P1.x()*u_ui[i] + P1.y()*v_vi[i] + P1.z()*w_wi[i] );
			// 圓點到P2向量(P2位置向量) dot 半徑向量(Rs)
//			double Dr = 0;

			// Area integral for general case
			double DD = Dq-Dp;

			CPLX<double> expDp(cos(Dp),sin(Dp));
			CPLX<double> expDq(cos(Dq),sin(Dq));

			CPLX<double> exp_Dp( expDp.r(), -expDp.i() );
			CPLX<double> exp_Dq( expDq.r(), -expDq.i() );

			// Special case 1
			CPLX<double> sic(0,0);
			CPLX<double> Ic(0,0);

			Ets = CPLX<double>(0,0);
			Eps = CPLX<double>(0,0);


			//----------------------------------------------------------------------------
			// Calcaulte surface current
#ifdef FASTPO
			// case 3
			if((abs(Dp) < Taylor.Rg()) && (abs(Dq) < Taylor.Rg())){
				for(int n=1;n<=Taylor.Nt();++n){
					sic += (PowJPhase(Dp, n) - PowJPhase(Dq, n)) / mat::factrl(n+1);
				}
				Ic = 2.*Co / CPLX<double>(0,DD) * sic;
			// case 1
			}else if((abs(Dp) < Taylor.Rg()) && (abs(Dq) >= Taylor.Rg())){
				for(int n=1;n<=Taylor.Nt();++n){
					sic += PowJPhase(Dp, n-1) / mat::factrl(n);
				}
				Ic = 2.*Co / (Dq * -DD) * ( expDq - CPLXxImg(sic,Dq) - 1. );
			// case 2
			}else if((abs(Dq) < Taylor.Rg()) && (abs(Dp) >= Taylor.Rg())){
				for(int n=1;n<=Taylor.Nt();++n){
					sic += PowJPhase(Dq, n-1) / mat::factrl(n);
				}
				Ic = 2.*Co / (Dp * DD)  * ( expDp - CPLXxImg(sic, Dp) - 1. );
			// case 4
			}else if(abs(DD) < Taylor.Rg()){
				for(int n=1;n<=Taylor.Nt();++n){
					sic += (PowJPhase(Dp, n) - PowJPhase(Dq, n)) / mat::factrl(n+1);
				}
				Ic = 2.*Co / CPLX<double>(0,DD) * sic;
			}else{
				cout<<"PO::CoreCalculate: No this kind of condition."<<endl;
				cout<<"   Dp = "<<Dp<<endl;
				cout<<"   Dq = "<<Dq<<endl;
				def_func::errorexit();
			}
#else
			// Special case 1
			if((abs(Dp) < Taylor.Rg()) && (abs(Dq) >= Taylor.Rg())){
				for(int q=0;q<=Taylor.Nt();++q){
//					sic += PowJPhasexCPLX(Dp, q, (-Co/(q+1) + expDq*(Co*G(q,-Dq,exp_Dq))) ) / mat::factrl(q);
					sic += PowJPhasexCPLX(Dp, q, (-Co/(q+1) + expDq*(Co*G3(q,-Dq))) ) / mat::factrl(q);
				}
				Ic = sic * 2 / CPLX<double>(0,1) / Dq;
				// Special case 2
			}else if ((abs(Dp) < Taylor.Rg()) && (abs(Dq) < Taylor.Rg())){
				for(int q=0;q<=Taylor.Nt();++q){
					for(int nn=0;nn<=Taylor.Nt();++nn){
						sic += PowJPhasexCPLX(Dp, q, PowJPhase(Dq, nn)) / mat::factrl(nn+q+2) * Co;
					}
				}
				Ic = sic * 2;
				// Special case 3
			}else if ((abs(Dp) >= Taylor.Rg()) && (abs(Dq) < Taylor.Rg())){
				for(int q=0;q<=Taylor.Nt();++q){
//					sic += PowJPhasexCPLX(Dq, q, G(q+1,-Dp,exp_Dp)) / (mat::factrl(q)*Co * (q+1));
					sic += PowJPhasexCPLX(Dq, q, G3(q+1,-Dp)) / (mat::factrl(q)*Co * (q+1));
				}
				Ic = sic * 2 * expDp;
				// Special case 4
			}else if ((abs(Dp) >= Taylor.Rg()) && (abs(Dq) >= Taylor.Rg()) && (abs(DD) < Taylor.Rg())){
				for(int q=0;q<=Taylor.Nt();++q){
//					sic += PowJPhasexCPLX(DD, q, (-Co*G(q,Dq,expDq)+expDq*Co/(q+1))/mat::factrl(q) );
					sic += PowJPhasexCPLX(DD, q, (-Co*G3(q,Dq)+expDq*Co/(q+1))/mat::factrl(q) );
				}
				Ic = sic * 2 / CPLX<double>(0,1) / Dq;
				// Default
			}else{
				Ic = 2.*( expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD) - (Co/Dp/Dq) );
			}
#endif
			//----------------------------------------------------------------------------



			// Scattered field components for triangle m in local coordinates
			VEC<CPLX<double> > Es2(Jx2*Ic, Jy2*Ic, CPLX<double>(0,0));
			// Transform back to global coordinates, then sum field
			VEC<CPLX<double> > Es1( cb[i] * Es2.x() + sb[i] * Es2.z(),
									Es2.y(),
									-sb[i] * Es2.x() + cb[i] * Es2.z());
			VEC<CPLX<double> > Es0( ca[i] * Es1.x() - sa[i] * Es1.y(),
									sa[i] * Es1.x() + ca[i] * Es1.y(),
									Es1.z() );

			Ets += uu[i] * Es0.x(); Ets += vv[i] * Es0.y(); Ets += ww[i] * Es0.z();	// 散射場在theta方向分量(H)
			Eps += (-sp[i]*Es0.x() + cp[i]*Es0.y());								// 散射場在phi方向分量(V)


//			if(idx == 207689/2 && i == 0){
//				printf("[CPU] k=%ld, i=%ld, Rc=[(%f,%f),(%f,%f),(%f,%f)]\n", idx, i,
//						Rc.x().r(), Rc.x().i(), Rc.y().r(), Rc.y().i(), Rc.z().r(), Rc.z().i());
//				printf("[CPU] k=%ld, i=%ld, e2=[(%f,%f),(%f,%f),(%f,%f)]\n", idx, i,
//						e2.x().r(), e2.x().i(), e2.y().r(), e2.y().i(), e2.z().r(), e2.z().i());
//				printf("[CPU] k=%ld, i=%ld, Et2=(%f,%f), Ep2=(%f,%f), Dp=%f, Dq=%f\n", idx, i,
//						Et2.r(), Et2.i(), Ep2.r(), Ep2.i(), Dp, Dq);
//				printf("[CPU] k=%ld, i=%ld, P0=(%f,%f,%f), P1=(%f,%f,%f)\n", idx, i,
//						P0.x(), P0.y(), P0.z(), P1.x(), P1.y(), P1.z());
//				printf("[CPU] k=%ld, i=%ld, Jx2=(%f,%f), Jy2=(%f,%f), Ic=(%f,%f)\n", idx, i,
//						Jx2.r(), Jx2.i(), Jy2.r(), Jy2.i(), Ic.r(), Ic.i());
//				printf("[CPU] k=%ld, i=%ld, po_Ets=(%f,%f), po_Eps=(%f,%f)\n", idx, i,
//						Ets.r(), Ets.i(), Eps.r(), Eps.i());
//			}

////#ifdef DEBUG
//			if(idx == 31 && i == 0){
//				printf("\n");
////				printf("Rf.TM = (%.20f,%.20f), Rf.TE = (%.20f,%.20f)\n", Rf.TM_par().r(), Rf.TM_par().i(), Rf.TE_perp().r(), Rf.TE_perp().i());
////				printf("cp   = %.20f, sp   = %.20f\n", cp[i], sp[i]);
////				printf("cpi2 = %.20f, spi2 = %.20f\n", cpi2[i], spi2[i]);
////				printf("cti2 = %.20f, sti2 = %.20f\n", cti2[i], sti2[i]);
////				printf("(u_ui,v_vi,w_wi) = (%.20f,%.20f,%.20f)\n", u_ui[i], v_vi[i], w_wi[i]);
////				printf("(  uu,  vv,  ww) = (%.20f,%.20f,%.20f)\n",   uu[i],   vv[i],   ww[i]);
////				printf("Dp  = %.20f, Dq = %.20f\n", Dp, Dq);
////				printf("Ei.cplx = (%.20f,%.20f) (%.20f,%.20f) (%.20f,%.20f)\n", Ei.cplx.x().r(), Ei.cplx.x().i(), Ei.cplx.y().r(), Ei.cplx.y().i(), Ei.cplx.z().r(), Ei.cplx.z().i());
////				printf("sa  = %.20f, ca = %.20f, sb = %.20f, cb = %.20f\n", sa[i], ca[i], sb[i], cb[i]);
////				printf("Rc.x.r = %.20f, Rc.y.r = %.20f\n", Rc.x().r(), Rc.y().r());
////				printf("ca*x.r          = %.20f\n", ca[i] * Rc.x().r());
////				printf("         sa*y.r = %.20f\n", sa[i] * Rc.y().r());
////				printf("ca*x.r + sa*y.r = %.20f\n", ca[i] * Rc.x().r() + sa[i] * Rc.y().r());
////				printf("ca*x= [%.20f,%.20f]\n", (ca[i] * Rc.x()).r(), (ca[i] * Rc.x()).i());
////				printf("sa*y= [%.20f,%.20f]\n", (sa[i] * Rc.y()).r(), (sa[i] * Rc.y()).i());
////				printf("ca*x + sa*y= [%.20f,%.20f]\n", ((ca[i] * Rc.x())+(sa[i] * Rc.y())).r(), ((ca[i] * Rc.x())+(sa[i] * Rc.y())).i());
////				printf("e2_1= [%.20f,%.20f] [%.20f,%.20f] [%.20f,%.20f]\n", e2_1.x().r(), e2_1.x().i(), e2_1.y().r(), e2_1.y().i(), e2_1.z().r(), e2_1.z().i());
//				printf("e2  = [%.20f,%.20f] [%.20f,%.20f] [%.20f,%.20f]\n", e2.x().r(), e2.x().i(), e2.y().r(), e2.y().i(), e2.z().r(), e2.z().i());
//				printf("Jx2 = (%.20f,%.20f), Jy2 = (%.20f,%.20f)\n", Jx2.r(), Jx2.i(), Jy2.r(), Jy2.i());
////				printf("Dp = %.20f, k0 = %.20f, tri.p0 = (%.20f,%.20f,%.20f), u_ui = %.20f, v_vi = %.20f, w_wi = %.20f\n", Dp, k0, tri->p0().x(), tri->p0().y(), tri->p0().z(), u_ui[i], v_vi[i], w_wi[i]);
//////				printf("k = %d, Ic = (%.20f,%.20f), Dp = %.20f, Dq = %.20f, DD = %.20f, Taylor.Rg = %.20f\n", idx, Ic.r(), Ic.i(), Dp, Dq, DD, Taylor.Rg());
//////				printf("k = %d, expDp*(Co/Dp/DD) = (%.20f,%.20f)\n", idx, (expDp*(Co/Dp/DD)).r(), (expDp*(Co/Dp/DD)).i());
//////				printf("k = %d, expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD) = (%.20f,%.20f)\n", idx, (expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD)).r(), (expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD)).i());
//////				printf("k = %d, (Co/Dp/Dq) = %.20f\n", idx, (Co/Dp/Dq));
//////				printf("k = %d, expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD) - (Co/Dp/Dq) = (%.20f,%.20f)\n", idx, (expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD) - (Co/Dp/Dq)).r(), (expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD) - (Co/Dp/Dq)).i() );
////				printf("k = %d, Ic = (%.20f,%.20f), expDp = (%.20f,%.20f), Co = %.20f, Dp = %.20f, DD = %.20f, expDq = (%.20f,%.20f)\n", idx, Ic.r(), Ic.i(), expDp.r(), expDp.i(), Co, Dp, DD, expDq.r(), expDq.i());
////				printf("k = %d, Jx2 = (%.20f,%.20f), Jy2 = (%.20f,%.20f), Dp = %.20f, Dq = %.20f, Ic = (%.20f,%.20f)\n", idx, Jx2.r(), Jx2.i(), Jy2.r(), Jy2.i(), Dp, Dq, Ic.r(), Ic.i());
////				printf("k = %d, Es2 = (%.20f,%.20f)(%.20f,%.20f)(%.20f,%.20f)\n", idx, Es2.x().r(), Es2.x().i(), Es2.y().r(), Es2.y().i(), Es2.z().r(), Es2.z().i());
////				printf("k = %d, Es1 = (%.20f,%.20f)(%.20f,%.20f)(%.20f,%.20f)\n", idx, Es1.x().r(), Es1.x().i(), Es1.y().r(), Es1.y().i(), Es1.z().r(), Es1.z().i());
////				printf("k = %d, Es0 = (%.20f,%.20f)(%.20f,%.20f)(%.20f,%.20f)\n", idx, Es0.x().r(), Es0.x().i(), Es0.y().r(), Es0.y().i(), Es0.z().r(), Es0.z().i());
//				printf("k = %d, po_Ets  = (%.20f,%.20f), po_Eps = (%.20f,%.20f)\n", idx, Ets.r(), Ets.i(), Eps.r(), Eps.i());
//			}
////#endif

		}
		template<typename T> void CoreCalculate2(const ElectricField& Ei, const TRI<T>* tri, const double k0,		// input
												 CPLX<double>& Ets, CPLX<double>& Eps, const RF& Rf, const long i, const long idx){				// output

			// residual
			// Incident field in global Cartesian coordinates (Bistatic)
			VEC<CPLX<double> > Rc = Ei.cplx;

			// Incident field in local Cartesian coordinates (stored in e2)
			VEC<CPLX<double> > e2 = find::transform(sa[i], ca[i], sb[i], cb[i], Rc);


			// Incident field in local spherical coordinates
			CPLX<double> Et2 =        e2.x()*cti2[i]*cpi2[i] + e2.y()*cti2[i]*spi2[i] - e2.z()*sti2[i];
			CPLX<double> Ep2 = (-1.0)*e2.x()*spi2[i]         + e2.y()*cpi2[i];
			// Surface current components in local Cartesian coordinates
			CPLX<double> tp1 = (-1.0)*Et2*Rf.TM_par();
			CPLX<double> tp2 = Ep2*Rf.TE_perp()*cti2[i];
			CPLX<double> Jx2 = tp1*cpi2[i] + tp2*spi2[i];	// cti2 added
			CPLX<double> Jy2 = tp1*spi2[i] - tp2*cpi2[i];	// cti2 added


			// Scattered field components for triangle m in local coordinates
			VEC<CPLX<double> > Es2(Jx2, Jy2, CPLX<double>(0,0));
			// Transform back to global coordinates, then sum field
			VEC<CPLX<double> > Es1( cb[i] * Es2.x() + sb[i] * Es2.z(),
									Es2.y(),
									-sb[i] * Es2.x() + cb[i] * Es2.z());
			VEC<CPLX<double> > Es0( ca[i] * Es1.x() - sa[i] * Es1.y(),
									sa[i] * Es1.x() + ca[i] * Es1.y(),
									Es1.z() );

			Ets += uu[i] * Es0.x(); Ets += vv[i] * Es0.y(); Ets += ww[i] * Es0.z();	// 散射場在theta方向分量(H)
			Eps += (-sp[i]*Es0.x() + cp[i]*Es0.y());								// 散射場在phi方向分量(V)

		}
	public:
		// Return
		ElectricField Es;
		// temp
		vector<double> sti2, cti2, cpi2, spi2;
		vector<double> u_ui, v_vi, w_wi, uu, vv, ww;
		vector<double> ca, sa, cb, sb, sp, cp;
	};

	/**
	 Calculate the Reflected Electric field, Er
	 @param Ei [in]  Incident electric field
	 @param Er [out] Reflection Electric field
	 @param N [in]  Normal vector of incident surface in global coordinate
	 @param Rf [in]  {TE, TM} Reflection Factor w.r.t. material of surface
	 @param k_next [in]  Reflection firection vector
	 @ref Ling, H., Chou, R. C., & Lee, S. W. (1989). Shooting and bouncing rays: Calculating the RCS of an arbitrarily shaped cavity. Antennas and Propagation, IEEE Transactions on, 37(2), 194–205.
	 */
	void ReflectionElectricField(const ElectricField& Ei, ElectricField& Er, const VEC<double> N, const RF& Rf, const VEC<double>& k_next){

		ThetaPhiVec Local_inc, Local_ref;
		// normal vector of reflection surface
		VEC<double> m = Unit(cross(-Ei.k,N));
		// local XYZ (eq.10)
		VEC<double> X_vec = cross(m,N);
		VEC<double> Y_vec = -m;
		VEC<double> Z_vec = -N;
		// local incident angle (eq.11)
		// theta_i = acos(vec::angle(-Ei.k, N))
		// phi_i   = 0
		double ct = dot(-Ei.k,N)/(Ei.k.abs()*N.abs());
		double st = sqrt(1-ct*ct);
		// local incident theta & phi vector (eq.12)
		Local_inc.theta_vec = Unit(ct * X_vec - st * Z_vec);
		Local_inc.phi_vec   = Y_vec;
		// Reflection
		Local_ref.theta_vec = Unit(-ct * X_vec - st * Z_vec);
		Local_ref.phi_vec   = Y_vec;

		// Amplitude of reflected field (eq.9), [Note: different than ref?]
		EAmp Eamp_cplx(Rf.TM_par()  * dot(Ei.cplx, Local_inc.theta_vec) *  1,	// Et
					   Rf.TE_perp() * dot(Ei.cplx, Local_inc.phi_vec)   * -1);	// Ep (diff?)
		// Ref. E field complex (eq.9)
		Er.cplx.x() = Eamp_cplx.Et * Local_ref.theta_vec.x() + Eamp_cplx.Ep * Local_ref.phi_vec.x();
		Er.cplx.y() = Eamp_cplx.Et * Local_ref.theta_vec.y() + Eamp_cplx.Ep * Local_ref.phi_vec.y();
		Er.cplx.z() = Eamp_cplx.Et * Local_ref.theta_vec.z() + Eamp_cplx.Ep * Local_ref.phi_vec.z();
		// Assign Ref E Field
		Er.k = k_next;
		Er.o = Ei.o;
	}


	double ReflectionElectricField(const ElectricField& Ei, ElectricField& Er,
			  					   const VEC<double> N, const RF& Rf, const VEC<double>& k_next,
			  					   const double RayArea, const VEC<double> Ei_org_k, const double k0){

		ThetaPhiVec Local_inc, Local_ref;
		// normal vector of reflection surface
		VEC<double> m = Unit(cross(-Ei.k,N));
		// local XYZ (eq.10)
		VEC<double> X_vec = cross(m,N);
		VEC<double> Y_vec = -m;
		VEC<double> Z_vec = -N;
		// local incident angle (eq.11)
		// theta_i = acos(vec::angle(-Ei.k, N))
		// phi_i   = 0
		double ct = dot(-Ei.k,N)/(Ei.k.abs()*N.abs());
		double st = sqrt(1-ct*ct);
		// local incident theta & phi vector (eq.12)
		Local_inc.theta_vec = Unit(ct * X_vec - st * Z_vec);
		Local_inc.phi_vec   = Y_vec;
		// Reflection
		Local_ref.theta_vec = Unit(-ct * X_vec - st * Z_vec);
		Local_ref.phi_vec   = Y_vec;

		// Amplitude of reflected field (eq.9), [Note: different than ref?]
		EAmp Eamp_cplx(Rf.TM_par()  * dot(Ei.cplx, Local_inc.theta_vec) *  1,	// Et
					   Rf.TE_perp() * dot(Ei.cplx, Local_inc.phi_vec)   * -1);	// Ep (diff?)
		// Ref. E field complex (eq.9)
		Er.cplx.x() = Eamp_cplx.Et * Local_ref.theta_vec.x() + Eamp_cplx.Ep * Local_ref.phi_vec.x();
		Er.cplx.y() = Eamp_cplx.Et * Local_ref.theta_vec.y() + Eamp_cplx.Ep * Local_ref.phi_vec.y();
		Er.cplx.z() = Eamp_cplx.Et * Local_ref.theta_vec.z() + Eamp_cplx.Ep * Local_ref.phi_vec.z();
		// Assign Ref E Field
		Er.k = k_next;
		Er.o = Ei.o;

		// Calculate sinc_patch
		VEC<double> Kd = (Ei.k + Ei_org_k) * k0;   // kipo == Ei_org.k, kin == Ei_org (when Level = 0)

		double cellSize = sqrt(RayArea);

		double sinc_path = sinc_no_pi( dot(Kd, X_vec)*cellSize/ 2.0 / abs(dot(Ei.k, N)) ) *
						   sinc_no_pi( dot(Kd, Y_vec)*cellSize/ 2.0);  //phase correct

		return sinc_path;

//		VEC<double> tri_N = static_cast<VEC<double> >(tri[i]->n());
//		VEC<double> Ei_k  = static_cast<VEC<double> >(Ei[j][i].k);
//		VEC<double> Ei_org_k  = static_cast<VEC<double> >(Ei_org.k);
//		// normal vector of reflection surface
//		VEC<double> k2 = -Ei_k;
//		VEC<double> m = Unit(cross(k2,tri_N));
//		// local XYZ (eq.10)
//		VEC<double> X_vec = cross(m,tri_N);	// Xc
//		VEC<double> Y_vec = -m;				// Yc
//		VEC<double> Z_vec = -tri_N;			// Zc
//
//		VEC<double> Kd = (Ei_k + Ei_org_k) * k0;   // kipo == Ei_org.k, kin == Ei_org (when Level = 0)
//
//		double cellSize = sqrt(RayArea);
//
//		double sinc_path = sinc_no_pi( dot(Kd, X_vec)*cellSize/ 2.0 / abs(dot(Ei_k, tri_N)) ) *
//						   sinc_no_pi( dot(Kd, Y_vec)*cellSize/ 2.0);  //phase correct
//
//		return sinc_path;
	}

	//+=========================+
	//|          SBR            |
	//+=========================+
	template<typename T>
	class SBRElement{
	public:
		SBRElement():AddDis(0){};
		SBRElement(const SAR& SAr, const long MAxLevel, const double ADdDis=0){
			Sar = SAr;
			// allocation
			sumt = D1<CPLX<double> >(MAxLevel);
			sump = D1<CPLX<double> >(MAxLevel);
			// Add Distance
			AddDis = ADdDis;
		}
		// Get
		D1<CPLX<double> > GetSumt(){ return sumt; }
		D1<CPLX<double> > GetSump(){ return sump; }
		D1<double> GetSthdB(){
			double lambda2 = Sar.lambda() * Sar.lambda();
			double Pi4_lambda2 = def::PI4 / lambda2;
			D1<double> Sth(sumt.GetNum());
			for(long i=0;i<Sth.GetNum();++i){
				Sth[i] = sumt[i].abs() * sumt[i].abs() * Pi4_lambda2;
				Sth[i] = 10.*log10(Sth[i] + 1E-10);
			}
			return Sth;
		}
		D1<double> GetSphdB(){
			double lambda2 = Sar.lambda() * Sar.lambda();
			double Pi4_lambda2 = def::PI4 / lambda2;
			D1<double> Sph(sumt.GetNum());
			for(long i=0;i<Sph.GetNum();++i){
				Sph[i] = sumt[i].abs() * sumt[i].abs() * Pi4_lambda2;
				Sph[i] = 10.*log10(Sph[i] + 1E-10);
			}
			return Sph;
		}
		void Print(){
			double lambda2 = Sar.lambda() * Sar.lambda();
			double Pi4_lambda2 = def::PI4 / lambda2;
			D1<double> Sth(sumt.GetNum()), Sph(sump.GetNum());
			for(long i=0;i<Sth.GetNum();++i){
				Sth[i] = sumt[i].abs() * sumt[i].abs() * Pi4_lambda2;
				Sph[i] = sump[i].abs() * sump[i].abs() * Pi4_lambda2;
			}
			cout<<std::setprecision(10);
			cout<<" Sth   = "<<10.*log10(mat::total(Sth) + 1E-10)<<" [dBsm]"<<endl;
			cout<<" Sph   = "<<10.*log10(mat::total(Sph) + 1E-10)<<" [dBsm]"<<endl;
		}
		void Print(const double IdealArea, const string pol){
			cout<<"+------------------------------------+"<<endl;
			cout<<"|            SBR Summary             |"<<endl;
			cout<<"+------------------------------------+"<<endl;
			double lambda2 = Sar.lambda() * Sar.lambda();
			double Pi4_lambda2 = def::PI4 / lambda2;
			double Sigma0 = 10*log10( (IdealArea*IdealArea) * Pi4_lambda2 );

			D1<double> Sth(sumt.GetNum()), Sph(sump.GetNum());
			for(long i=0;i<Sth.GetNum();++i){
				Sth[i] = sumt[i].abs() * sumt[i].abs() * Pi4_lambda2;
				Sph[i] = sump[i].abs() * sump[i].abs() * Pi4_lambda2;
			}
			cout<<" TX Polarization = "<<pol<<endl;
			cout<<std::setprecision(10);
			cout<<" Ideal   dB      = "<<Sigma0<<" [dBsm]"<<endl;
			cout<<" Sth (V) dB      = "<<10.*log10(mat::total(Sth) + 1E-10)<<" [dBsm]"<<endl;
			cout<<" Sph (H) dB      = "<<10.*log10(mat::total(Sph) + 1E-10)<<" [dBsm]"<<endl;
			cout<<" Ideal   RCS     = "<<(IdealArea*IdealArea) * Pi4_lambda2<<endl;
			cout<<" Sth (V) RCS     = "<<mat::total(Sth)<<endl;
			cout<<" Sph (H) RCS     = "<<mat::total(Sph)<<endl;

			cout<<" Sth (V) Series  = "; Sth.Print();
			cout<<" Sph (H) Series  = "; Sph.Print();
		}
	public:
		// input
		SAR Sar;
		// output
		D1<CPLX<double> > sumt, sump;
		// Add distance
		double AddDis;
	};

	template<typename T>
	class SBR{
	public:
		SBR():MaxLevel(0),MinLevel(0),SHOW(false),HaveDoIt(false){};
		SBR(const vector<SAR>& Sys, const EF& ef, const BVH& Bvh, const MeshInc& Inc_plane, const MaterialDB& MatDb, const long MAxLevel,
			const double ADdDis=0, const bool SHow=false):bvh(Bvh){
			if(MAxLevel > 20){
				cout<<"SBR::[WARRNING]:MaxLevel MUST smaller than 20!"<<endl;
			}
			Sar  = Sys;
			Ef   = ef;
			MaxLevel = MAxLevel;
			MinLevel = 0;
			SHOW = SHow;
			inc_plane = Inc_plane;
			HaveDoIt = false;

			// Initialize SBRElement<T>
			sbr = vector<SBRElement<T> >(Sys.size());
			for(unsigned long i=0;i<sbr.size();++i){
				sbr[i] = SBRElement<T>(Sys[i], MAxLevel, ADdDis);
			}

			// private
			// Scatting structure
			PoEs = vector<vector<Scatter> >(sbr.size());
			for(unsigned long i=0;i<PoEs.size();++i){
				PoEs[i] = vector<Scatter>(MaxLevel);
			}

			// vector reserve
			// Intersection
			DistSet.resize(MaxLevel);
			RaySet.resize(MaxLevel);
			ObjSet.resize(MaxLevel);
			Shadow.resize(MaxLevel);
			N.resize(MaxLevel);
//			DistSet.reserve(MaxLevel);
//			RaySet.reserve(MaxLevel);
//			ObjSet.reserve(MaxLevel);
//			N.reserve(MaxLevel);
			// pre-allocate
			//			Ei.resize(MaxLevel);
			Er.resize(MaxLevel);
			Pr.resize(MaxLevel);
			k_obv.resize(MaxLevel);
			dis.resize(MaxLevel);
			tri.resize(MaxLevel);

			// Ei
			Ei.resize(sbr.size());
			for(unsigned long i=0;i<Ei.size();++i){
				Ei[i].resize(MaxLevel);
			}

			po = PO(MaxLevel);

			// Material
			MatDB = MatDb;
		}
		// Misc.
		void DoIt(const bool IsPEC=false, const bool IsProgressBar=false){

			//==========================================================================
			// SBR for Each Ray
			//==========================================================================
			// Local
			for(unsigned long j=0;j<sbr.size();++j){
				// Set to zero
				for(int i=0;i<MaxLevel;++i){
					sbr[j].sumt[i] = CPLX<double>(0,0);
					sbr[j].sump[i] = CPLX<double>(0,0);
				}
			}

			// Global
			double RayArea;
			Ray RayInc;
			bool IsHit;

			size_t TotalHit = 0;

			clock_t tic = def_func::tic();
			for(long k=0;k<inc_plane.nPy;++k){
				//======================================================================
				// Make a Ray
				//======================================================================
				inc_plane.GetCell(k, RayInc, RayArea);

				for(unsigned long i=0;i<sbr.size();++i){
					for(int j=0;j<MaxLevel;++j){
						PoEs[i][j].Level = -1;
					}
				}


				// SBR for each Ray
				// Return the Es to 'PoEs' private variable
//				IsHit = ForEachRay(k, RayInc, RayArea, IsPEC, true);
				IsHit = ForEachRay(k, RayInc, RayArea, inc_plane.dRad, IsPEC, false);


				if(IsHit == true){
					TotalHit++;
					//==========================
					// for Each SAR_sys
					//==========================
					for(unsigned long i=0;i<sbr.size();++i){
						// Summation for each bounce level [0,MaxLevel]
						for(int j=0;j<MaxLevel;++j){
							if(PoEs[i][j].Level > -1){
								sbr[i].sumt[j] += PoEs[i][j].Ets;
								sbr[i].sump[j] += PoEs[i][j].Eps;
							}
						}
					}
				} // end of IsHit
				if(IsProgressBar && (k%(inc_plane.nPy/100) == 0)){
					def::CLOCK clock(tic);
					def_func::ProgressBar(k+1, inc_plane.nPy, 50, 1, clock);
				}
			}// end of each nPy
			HaveDoIt = true;

#ifdef DEBUG
			printf("+-------------------+\n");
			printf("nPy      = %ld\n", inc_plane.nPy);
			printf("RayArea  = %.16f\n", RayArea);
			printf("TotalHit = %ld\n", TotalHit);
			printf("TotalHit * RayArea = %.16f\n", double(TotalHit)*RayArea);
			printf("+-------------------+\n");
#endif
		}
		void DoIt2(const VEC<double>& MainBeamUV, const VEC<double>& NorSquintPlane, const VEC<double>& PPs, const VEC<double>& PPt,
				   const bool IsPEC=false, const bool IsProgressBar=false){

			//==========================================================================
			// SBR for Each Ray
			//==========================================================================
			// Local
			for(unsigned long j=0;j<sbr.size();++j){
				// Set to zero
				for(int i=0;i<MaxLevel;++i){
					sbr[j].sumt[i] = CPLX<double>(0,0);
					sbr[j].sump[i] = CPLX<double>(0,0);
				}
			}

			// Global
			double RayArea;
			Ray RayInc;
			bool IsHit;

			size_t TotalHit = 0;

			clock_t tic = def_func::tic();
			for(long k=0;k<inc_plane.nPy;++k){
				//======================================================================
				// Make a Ray
				//======================================================================
				inc_plane.GetCell(k, RayInc, RayArea);

				for(unsigned long i=0;i<sbr.size();++i){
					for(int j=0;j<MaxLevel;++j){
						PoEs[i][j].Level = -1;
					}
				}


				// SBR for each Ray
				// Return the Es to 'PoEs' private variable
//				IsHit = ForEachRay(k, RayInc, RayArea, IsPEC, true);
				IsHit = ForEachRay2(k, RayInc, RayArea, inc_plane.dRad, MainBeamUV, NorSquintPlane, PPs, PPt, IsPEC, false);


				if(IsHit == true){
					TotalHit++;
					//==========================
					// for Each SAR_sys
					//==========================
					for(unsigned long i=0;i<sbr.size();++i){
						// Summation for each bounce level [0,MaxLevel]
						for(int j=0;j<MaxLevel;++j){
							if(PoEs[i][j].Level > -1){
								sbr[i].sumt[j] += PoEs[i][j].Ets;
								sbr[i].sump[j] += PoEs[i][j].Eps;
							}
						}
					}
				} // end of IsHit
				if(IsProgressBar && (k%(inc_plane.nPy/100) == 0)){
					def::CLOCK clock(tic);
					def_func::ProgressBar(k+1, inc_plane.nPy, 50, 1, clock);
				}
			}// end of each nPy
			HaveDoIt = true;

#ifdef DEBUG
			printf("+-------------------+\n");
			printf("nPy      = %ld\n", inc_plane.nPy);
			printf("RayArea  = %.16f\n", RayArea);
			printf("TotalHit = %ld\n", TotalHit);
			printf("TotalHit * RayArea = %.16f\n", double(TotalHit)*RayArea);
			printf("+-------------------+\n");
#endif
		}
		SBRElement<T>& GetSBRElement(const long i){
			if(HaveDoIt == false){
				cout<<"ERROR::SBR::GetSBRElement:Haven't DoIt()!"<<endl;
				exit(EXIT_FAILURE);
			}
			return sbr[i];
		}
		vector<SBRElement<T> >& GetSBRElementVector(){
			if(HaveDoIt == false){
				cout<<"ERROR::SBR::GetSBRElement:Haven't DoIt()!"<<endl;
				exit(EXIT_FAILURE);
			}
			return sbr;
		}
	private:
		bool ForEachRay(const long idx, const Ray& RayInc, const double RayArea, const double dRad,
						const bool IsPEC=false, const bool SHOW=false){

			long Level;
			raytrace::ObjectIntersection(bvh, RayInc, MaxLevel, DistSet, RaySet, ObjSet, Shadow, N, Level, idx);

			MinLevel = std::min(Level, MaxLevel);


			if(MinLevel == 0){
				if(SHOW == true){ cout<<"SBR::[ForEachRay]:Incident ray without any intersection["<<idx<<"]."<<endl; }
				return false;
			}
			if(SHOW == true){
				if(MinLevel == MaxLevel){ cout<<"SBR::[ForEachRay]:Terminated at reach MaxLevel condition["<<idx<<"]."<<endl; }
				if(MinLevel < MaxLevel)	{ cout<<"SBR::[ForEachRay]:Terminated before reach MaxLevel condition["<<idx<<"]."<<endl; }
			}


			//======================================================================
			// SBR For Each Ray
			//======================================================================
			// Original Incident E-field
			ElectricField Ei_org = ElectricField(RayInc.d, Ef.Eamp().Et(), Ef.Eamp().Ep(), RayInc.o);
			// Global coordinate
			g = Ei_org.GetThetaPhiInfo();

			//====================================================================
			// SBR
			//====================================================================
			// GO
			// Allocation
			double phs;
			ElectricField Es;
			// double cos_theta_i_obv;
			VEC<double> NN;
			double TotalDis;


			// pre-calculate
			// Final Ref. Point
			for(long i=0;i<MinLevel;++i){ // small loop
				// Final Ref. Point
				Pr[i] = RaySet[i].o;
				// Calculate distance from this point "Pr" to observe plane
				dis[i] = vec::find::MinDistanceFromPointToPlane(Ei_org.k, Ei_org.o, Pr[i]);
				// Observation direction
				k_obv[i] = -Ei_org.k;
				// Get intersection object
				tri[i] = static_cast<TRI<float>*>(ObjSet[i]);
			}

			// Pre-define (Default values, PEC)
			RF Rf(-1,-1);					// Reflection Factor of PEC
			Material MAT;	MAT.SetToPEC();	// PEC material


			//
			// Each lambda
			//
			for(unsigned long j=0;j<sbr.size();++j){ // big loop
				// const parameter
				double f0 = sbr[j].Sar.f0();
				double k0 = sbr[j].Sar.k0();
				// initial phase, the "DistSet[0]" is the distance from incident plane to 1st reflection point
				phs = sbr[j].Sar.k0() * DistSet[0];
				// Initialize inc. E-field -->  Ei[idx_freq][idx_level]
				Ei[j][0] = ElectricField(Ei_org.k, Ei_org.o, Ei_org.cplx);
				Ei[j][0].AddPhase(phs);

				// For each bounce
				for(long i=0;i<MinLevel;++i){
					// PO pre-calculate
					po.PreCalculate(Ei[j][i], tri[i], k_obv[i], i, idx); // for next po.GetRf() function
					//
					// Is PEC ?
					//
					if(!IsPEC && tri[i]->MatIDX() > 0){
						
//						// 修正取用材質時問題: 由材質 id 轉成 MatDB.Mat 的 index
//						// Get material
//						// MAT = MatDB.Get(tri[i]->MatIDX());
//						// MAT = MatDB.Get(2);
//						// Find the index of MatDB.Mat[index]
//						long index = -1;
//
//						for(size_t idx_mat=0;idx_mat<MatDB.Mat.size();++idx_mat){
//							if(tri[i]->MatIDX() == MatDB.Mat[idx_mat].idx()){
//								index = idx_mat;
//								MAT = MatDB.Get(index);
//							}
//						}
//						if(index < 0){
//							cerr<<"WARRNING::The material in triangle is not belong to any material in MatDB dabase"<<endl;
//							cerr<<"          Force assign to be the MatDB.Get(0)"<<endl;
//							MAT = MatDB.Get(0);
//						}
//						// 修正取用材質時問題: 由材質 id 轉成 MatDB.Mat 的 index (END)

						MAT = MatDB.Get(tri[i]->MatIDX());
//						long index = tri[i]->MatIDX()

						// Get Reflection Factor
						Rf = po.GetRF(f0, tri[i], MAT, i, idx);	// Rf is f0 dependent
					}
					// Find Ref. E-field
					// 入射：Ei[j][i] (含原點"o", 波方向"k", 震盪"cplx")
					// 反射：Er[i]    (含原點"o", 波方向"k", 震盪"cplx")
//					ReflectionElectricField(Ei[j][i], Er[i], N[i], Rf, RaySet[i].d);
//
//					// PO approximation
					// ====================== 原始位修改(開始) =====================
					po.Calculate(Ei[j][i], tri[i], k0, k_obv[i], Pr[i], g, Ef.Taylor(), Rf, i, idx);
					Es = po.GetEs();
					// ====================== 原始位修改(結束) =====================
//					// ====================== 修改這裡(開始1) =====================
					VEC<double> tri_N = static_cast<VEC<double> >(tri[i]->getNormal());
//					VEC<double> Ei_k  = static_cast<VEC<double> >(Ei[j][i].k);
//					VEC<double> Ei_org_k  = static_cast<VEC<double> >(Ei_org.k);
//					// normal vector of reflection surface
//					VEC<double> k2 = -Ei_k;
//					VEC<double> m = Unit(cross(k2,tri_N));
//					// local XYZ (eq.10)
//					VEC<double> X_vec = cross(m,tri_N);	// Xc
//					VEC<double> Y_vec = -m;				// Yc
//					VEC<double> Z_vec = -tri_N;			// Zc
//
//					VEC<double> Kd = (Ei_k + Ei_org_k) * k0;   // kipo == Ei_org.k, kin == Ei_org (when Level = 0)
//
//					double cellSize = sqrt(RayArea);
//
//					double sinc_path = sinc_no_pi( dot(Kd, X_vec)*cellSize/ 2.0 / abs(dot(Ei_k, tri_N)) ) *
//							   		   sinc_no_pi( dot(Kd, Y_vec)*cellSize/ 2.0);  //phase correct

					VEC<double> Ei_org_k  = static_cast<VEC<double> >(Ei_org.k);
					double sinc_path = ReflectionElectricField(Ei[j][i], Er[i], tri_N, Rf, RaySet[i].d, RayArea, Ei_org_k, k0);

					// Es.cplx = vec::cross(tri_N, vec::cross(Ei_k, Ei[j][i].cplx)) * sinc_path;
//					Es.cplx = vec::cross(tri_N, vec::cross(Ei[j][i].k, Ei[j][i].cplx)) / fabs(dot(Ei[j][i].k, tri_N));// * sinc_path;
					// ====================== 修改這裡(結束) ======================

//					printf("i = %ld, Es.cplx=[%f,%f]\n")


					// ====================== 修改這裡(原始) ======================
					// double cos_theta_i_obv = dot(cuVEC<double>(tri.N), k_obv);	//(tri.N.abs()*k_obv.abs());
					double cos_theta_i_obv = fabs(dot(VEC<double>(tri[i]->getNormal()), Ei[j][i].k));	//(tri.N.abs()*k_obv.abs());
					float factor = RayArea / cos_theta_i_obv * sinc_path;
					// ====================== 修改這裡(開始1) =====================
//					float factor = sinc_path * RayArea;
					// ====================== 修改這裡(結束) ======================

					// Nomalize
					Es.cplx = Es.cplx * factor;

					// Add distance phase
					TotalDis = dis[i] + (2* sbr[j].AddDis);
					Es.AddPhase(k0 * TotalDis);


					// If this hit point is in shadow, then Ets & Eps is zero
					if(Shadow[i]){
						Es.cplx.x() = CPLX<double>(0.,0.);
						Es.cplx.y() = CPLX<double>(0.,0.);
						Es.cplx.z() = CPLX<double>(0.,0.);
					}

					// Assign values
					PoEs[j][i].cplx = Es.cplx;	// Complex
					PoEs[j][i].Level = i;		// number of bounce
					PoEs[j][i].Ets = dot(Es.cplx, g.theta_vec);	// Theta comp.
					PoEs[j][i].Eps = dot(Es.cplx, g.phi_vec);	// Phi comp.

					// Update the Ei <--- Er with distance between 2 reflection points
					if(i < MinLevel-1){
						phs = sbr[j].Sar.k0() * DistSet[i+1];
						// inc. E-field on the next reflection point
						Ei[j][i+1] = ElectricField(Er[i].k, RaySet[i].o, Er[i].cplx);
						Ei[j][i+1].AddPhase(phs);
					}
				} // End of Level
			} // End of Freq
			return true;
		}
		bool ForEachRay2(const long idx, const Ray& RayInc, const double RayArea, const double dRad,
						 const VEC<double>& MainBeamUV, const VEC<double>& NorSquintPlane, const VEC<double>& PPs, const VEC<double>& PPt,
						 const bool IsPEC=false, const bool SHOW=false){

			long Level;
			raytrace::ObjectIntersection(bvh, RayInc, MaxLevel, DistSet, RaySet, ObjSet, Shadow, N, Level, idx);

//			if(count == 0){
////					printf("k = %d, rayRef.d     = [%.20f,%.20f,%.20f]\n", k, rayRef.d.x(), rayRef.d.y(), rayRef.d.z());
////					printf("k = %d, rayRef.o     = [%.20f,%.20f,%.20f]\n", k, rayRef.o.x(), rayRef.o.y(), rayRef.o.z());
////					printf("k = %d, rayRef.inv_d = [%.20f,%.20f,%.20f]\n", k, rayRef.inv_d.x(), rayRef.inv_d.y(), rayRef.inv_d.z());
////					printf("k = %d, DistSet      = %.20f\n", k, DistSet[count]);
////					printf("k = %d, I.hit        = [%.20f,%.20f,%.20f]\n", k, I.hit.x(), I.hit.y(), I.hit.z());
////					printf("k = %d, N            = [%.20f,%.20f,%.20f]\n", k, N[count].x(), N[count].y(), N[count].z());
//////					((TRI<float>*)(I.object))->Print();
//////					printf("[CPU](After Ref) count=%d, MaxLevel=%d, I.hit   =(%.10f,%.10f,%.10f), N       =(%.10f,%.10f,%.10f)\n", count, MaxLevel, I.hit.x(), I.hit.y(), I.hit.z(), N[count].x(), N[count].y(), N[count].z());
//////					printf("[CPU](After Ref) count=%d, MaxLevel=%d, rayRef.o=(%.10f,%.10f,%.10f), rayRef.d=(%.10f,%.10f,%.10f)\n", count, MaxLevel, rayRef.o.x(), rayRef.o.y(), rayRef.o.z(), rayRef.d.x(), rayRef.d.y(), rayRef.d.z());
//////					printf("[CPU](After Ref) DistSet[%d]=%.10f\n", count, DistSet[count]);
////////					printf("[CPU](After Ref) RaySet[%d] =%.10f\n", count, RaySet[count]);
//////					printf("+\n\n\n");
//					printf("k = %d, I.hit = [%.20f,%.20f,%.20f], rayRef.o = [%.20f,%.20f,%.20f]\n",
//							k, I.hit.x, I.hit.y, I.hit.z, k, rayRef.o.x, rayRef.o.y, rayRef.o.z);
//				}
//			if(idx == 831 && Level == 1){
//				VEC<double> rayRef_o = RaySet[Level-1].o;
//				printf("idx = %ld, rayRef.o = [%.20f,%.20f,%.20f]\n", idx, rayRef_o.x(), rayRef_o.y(), rayRef_o.z());
//			}

			MinLevel = std::min(Level, MaxLevel);


			if(MinLevel == 0){
				if(SHOW == true){ cout<<"SBR::[ForEachRay]:Incident ray without any intersection["<<idx<<"]."<<endl; }
				return false;
			}
			if(SHOW == true){
				if(MinLevel == MaxLevel){ cout<<"SBR::[ForEachRay]:Terminated at reach MaxLevel condition["<<idx<<"]."<<endl; }
				if(MinLevel < MaxLevel)	{ cout<<"SBR::[ForEachRay]:Terminated before reach MaxLevel condition["<<idx<<"]."<<endl; }
			}


			//======================================================================
			// SBR For Each Ray
			//======================================================================
			// Original Incident E-field
			ElectricField Ei_org = ElectricField(RayInc.d, Ef.Eamp().Et(), Ef.Eamp().Ep(), RayInc.o);
			// Global coordinate
			g = Ei_org.GetThetaPhiInfo();

			//====================================================================
			// SBR
			//====================================================================
			// GO
			// Allocation
			double phs;
			ElectricField Es;
			// double cos_theta_i_obv;
			VEC<double> NN;
			double TotalDis;
			D1<double> AzGain(MaxLevel);
			double AzGain1st, AzGainLast;
			AzGain.SetZero();


			// pre-calculate
			// Final Ref. Point
			for(long i=0;i<MinLevel;++i){ // small loop
				// Final Ref. Point
				Pr[i] = RaySet[i].o;
				// Calculate distance from this point "Pr" to observe plane
				dis[i] = vec::find::MinDistanceFromPointToPlane(Ei_org.k, Ei_org.o, Pr[i]);
				// Observation direction
				k_obv[i] = -Ei_org.k;
				// Get intersection object
				tri[i] = static_cast<TRI<float>*>(ObjSet[i]);
				// TODO: Add AzGain
				// Azimuth antenna gain value
				// 1st hit
				if(i == 0){
					// TODO: raytracing2
					AzGain1st = sar::find::AzimuthAntennaGain(sbr[0].Sar, MainBeamUV, NorSquintPlane, PPs, PPt, RaySet[0].o);
				}
				// Last hit
				AzGainLast = sar::find::AzimuthAntennaGain(sbr[0].Sar, MainBeamUV, NorSquintPlane, PPs, PPt, RaySet[MinLevel-1].o);
				// Combine
				AzGain[i] = AzGain1st * AzGainLast;
			}

			// Pre-define (Default values, PEC)
			RF Rf(-1,-1);					// Reflection Factor of PEC
			Material MAT;	MAT.SetToPEC();	// PEC material


			//
			// Each lambda
			//
			for(unsigned long j=0;j<sbr.size();++j){ // big loop
				// const parameter
				double f0 = sbr[j].Sar.f0();
				double k0 = sbr[j].Sar.k0();
				// initial phase, the "DistSet[0]" is the distance from incident plane to 1st reflection point
				phs = sbr[j].Sar.k0() * DistSet[0];
				// Initialize inc. E-field -->  Ei[idx_freq][idx_level]
				Ei[j][0] = ElectricField(Ei_org.k, Ei_org.o, Ei_org.cplx);
				Ei[j][0].AddPhase(phs);

				// For each bounce
				for(long i=0;i<MinLevel;++i){
					// PO pre-calculate
					po.PreCalculate(Ei[j][i], tri[i], k_obv[i], i, idx); // for next po.GetRf() function

//					if(idx == 286 && i == 0 && j == 0){
//						printf("[CPU] k=%ld, i=%ld, ct=%f, st=%f, cti2=%f, sti2=%f\n", idx, i,
//								0.0, 0.0, po.cti2[i], po.sti2[i]);
//					}

					//
					// Is PEC ?
					//
					if(!IsPEC){

//						// 修正取用材質時問題: 由材質 id 轉成 MatDB.Mat 的 index
//						// Get material
//						// MAT = MatDB.Get(tri[i]->MatIDX());
//						// MAT = MatDB.Get(2);
//						// Find the index of MatDB.Mat[index]
//						long index = -1;
//
//						for(size_t idx_mat=0;idx_mat<MatDB.Mat.size();++idx_mat){
//							if(tri[i]->MatIDX() == MatDB.Mat[idx_mat].idx()){
//								index = idx_mat;
//								MAT = MatDB.Get(index);
//							}
//						}
//						if(index < 0){
//							cerr<<"WARRNING::The material in triangle is not belong to any material in MatDB dabase"<<endl;
//							cerr<<"          Force assign to be the MatDB.Get(0)"<<endl;
//							MAT = MatDB.Get(0);
//						}
//						// 修正取用材質時問題: 由材質 id 轉成 MatDB.Mat 的 index (END)

						MAT = MatDB.Get(tri[i]->MatIDX());
//						long index = tri[i]->MatIDX()

						// Get Reflection Factor
						Rf = po.GetRF(f0, tri[i], MAT, i, idx);	// Rf is f0 dependent
					}
					// Find Ref. E-field
					// 入射：Ei[j][i] (含原點"o", 波方向"k", 震盪"cplx")
					// 反射：Er[i]    (含原點"o", 波方向"k", 震盪"cplx")
//					ReflectionElectricField(Ei[j][i], Er[i], N[i], Rf, RaySet[i].d);
//
//					// PO approximation
					// ====================== 原始位修改(開始) =====================
					po.Calculate(Ei[j][i], tri[i], k0, k_obv[i], Pr[i], g, Ef.Taylor(), Rf, i, idx);
					Es = po.GetEs();
					// ====================== 原始位修改(結束) =====================
//					// ====================== 修改這裡(開始1) =====================
					VEC<double> tri_N = static_cast<VEC<double> >(tri[i]->getNormal());
//					VEC<double> Ei_k  = static_cast<VEC<double> >(Ei[j][i].k);
//					VEC<double> Ei_org_k  = static_cast<VEC<double> >(Ei_org.k);
//					// normal vector of reflection surface
//					VEC<double> k2 = -Ei_k;
//					VEC<double> m = Unit(cross(k2,tri_N));
//					// local XYZ (eq.10)
//					VEC<double> X_vec = cross(m,tri_N);	// Xc
//					VEC<double> Y_vec = -m;				// Yc
//					VEC<double> Z_vec = -tri_N;			// Zc
//
//					VEC<double> Kd = (Ei_k + Ei_org_k) * k0;   // kipo == Ei_org.k, kin == Ei_org (when Level = 0)
//
//					double cellSize = sqrt(RayArea);
//
//					double sinc_path = sinc_no_pi( dot(Kd, X_vec)*cellSize/ 2.0 / abs(dot(Ei_k, tri_N)) ) *
//							   		   sinc_no_pi( dot(Kd, Y_vec)*cellSize/ 2.0);  //phase correct

					VEC<double> Ei_org_k  = static_cast<VEC<double> >(Ei_org.k);
					double sinc_path = ReflectionElectricField(Ei[j][i], Er[i], tri_N, Rf, RaySet[i].d, RayArea, Ei_org_k, k0);

					// Es.cplx = vec::cross(tri_N, vec::cross(Ei_k, Ei[j][i].cplx)) * sinc_path;
//					Es.cplx = vec::cross(tri_N, vec::cross(Ei[j][i].k, Ei[j][i].cplx)) / fabs(dot(Ei[j][i].k, tri_N));// * sinc_path;
					// ====================== 修改這裡(結束) ======================

//					printf("i = %ld, Es.cplx=[%f,%f]\n")
//					if(idx == 286 && i == 0 && j == 0){
//						printf("[CPU] k=%ld, Es.cplx = [(%.8f,%.8f),(%.8f,%.8f),(%.8f,%.8f)]\n", idx, Es.cplx.x().r(), Es.cplx.x().i(), Es.cplx.y().r(), Es.cplx.y().i(), Es.cplx.z().r(), Es.cplx.z().i());
//					}


					// ====================== 修改這裡(原始) ======================
					// double cos_theta_i_obv = dot(cuVEC<double>(tri.N), k_obv);	//(tri.N.abs()*k_obv.abs());
					double cos_theta_i_obv = fabs(dot(VEC<double>(tri[i]->getNormal()), Ei[j][i].k));	//(tri.N.abs()*k_obv.abs());
					float factor = RayArea / cos_theta_i_obv * sinc_path;
					// ====================== 修改這裡(開始1) =====================
//					float factor = sinc_path * RayArea;
					// ====================== 修改這裡(結束) ======================


					// =================== 加上 AzGain (開始) ====================
					// TODO: Add AzGain
					factor = factor * AzGain[i];
					// =================== 加上 AzGain (結束) ====================

					// Nomalize
					Es.cplx = Es.cplx * factor;

//					if(idx == 286 && i == 0 && j == 0){
//						printf("[CPU] k=%ld, Es.cplx = [(%.8f,%.8f),(%.8f,%.8f),(%.8f,%.8f)]\n", idx, Es.cplx.x().r(), Es.cplx.x().i(), Es.cplx.y().r(), Es.cplx.y().i(), Es.cplx.z().r(), Es.cplx.z().i());
//					}

					// Add distance phase
					TotalDis = dis[i] + (2* sbr[j].AddDis);
					Es.AddPhase(k0 * TotalDis);


					// If this hit point is in shadow, then Ets & Eps is zero
					if(Shadow[i]){
						Es.cplx.x() = CPLX<double>(0.,0.);
						Es.cplx.y() = CPLX<double>(0.,0.);
						Es.cplx.z() = CPLX<double>(0.,0.);
					}

					// Assign values
					PoEs[j][i].cplx = Es.cplx;	// Complex
					PoEs[j][i].Level = i;		// number of bounce
					PoEs[j][i].Ets = dot(Es.cplx, g.theta_vec);	// Theta comp.
					PoEs[j][i].Eps = dot(Es.cplx, g.phi_vec);	// Phi comp.

//					if(idx == 286 && i == 0 && j == 0){
//						printf("[CPU] k=%ld, Es.cplx = [(%.8f,%.8f),(%.8f,%.8f),(%.8f,%.8f)], Ets=(%.8f,%.8f), Eps=(%.8f,%.8f)\n",
//								idx, Es.cplx.x().r(), Es.cplx.x().i(), Es.cplx.y().r(), Es.cplx.y().i(), Es.cplx.z().r(), Es.cplx.z().i(),
//								PoEs[j][i].Ets.r(), PoEs[j][i].Ets.i(), PoEs[j][i].Eps.r(), PoEs[j][i].Eps.i());
//					}

					// Update the Ei <--- Er with distance between 2 reflection points
					if(i < MinLevel-1){
						phs = sbr[j].Sar.k0() * DistSet[i+1];
						// inc. E-field on the next reflection point
						Ei[j][i+1] = ElectricField(Er[i].k, RaySet[i].o, Er[i].cplx);
						Ei[j][i+1].AddPhase(phs);
					}
				} // End of Level
			} // End of Freq
			return true;
		}
	private:
		// Global
		BVH bvh;
		MeshInc inc_plane;
		long MaxLevel;
		long MinLevel;
		bool SHOW;
		// Local
		vector<SAR> Sar;
		EF Ef;
		vector<SBRElement<T> > sbr;	// 第一層 vector 記錄 Nr(=Sar.size())的element, 第二層 SBRElement.sumt & sump 記錄每個level的複數值
		// Flag
		bool HaveDoIt;
		// private
		// Scatting structure
		vector<vector<Scatter> > PoEs;
		// Intersection
		vector<double> DistSet;
		vector<Ray> RaySet;
		vector<Obj*> ObjSet;
		vector<bool> Shadow;
		vector<VEC<float> > N;
		vector<vector<ElectricField> > Ei;
		vector<ElectricField> Er;
		vector<VEC<double> > Pr, k_obv;
		vector<double> dis;
		vector<TRI<float>*> tri;
		PO po;
		// Material
		MaterialDB MatDB;
		// Global angle vector
		ThetaPhiVec g;
	};

}



#endif
