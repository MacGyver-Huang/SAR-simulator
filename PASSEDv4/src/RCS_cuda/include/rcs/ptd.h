//
//  ptd.h
//  PASSED2
//
//  Created by Steve Chiang on 11/8/17.
//  Copyright (c) 2017 Steve Chiang. All rights reserved.
//

#ifndef ptd_h
#define ptd_h

#include <basic/def_func.h>
#include <mesh/intersect.h>

using namespace def_func;
using namespace intersect;



namespace ptd {
	
	//+-----------------------------------------------------------------------+
	//|                                 Class                                 |
	//+-----------------------------------------------------------------------+
	/**
	 * @brief      Element of Local Coordinate
	 */
	template<typename T>
	class LocalCorrdinateElemt {
	public:
		/**
		 * @brief      Default constructor
		 */
		LocalCorrdinateElemt(){};
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  perp  Perpendicular vector w.r.t. the incident plane
		 * @param[in]  para  Parallel vector w.r.t. the incident plane
		 */
		LocalCorrdinateElemt(const VEC<T>& perp, const VEC<T>& para){
			_perp = perp;
			_para = para;
		}
		/**
		 * @brief      Get    perp
		 */
		const VEC<T>& perp()const{ return _perp; }
		/**
		 * @brief      Get    para
		 */
		const VEC<T>& para()const{ return _para; }
		/**
		 * @brief      Get    perp (editable)
		 */
		VEC<T>& perp(){ return _perp; }
		/**
		 * @brief      Get    para (editable)
		 */
		VEC<T>& para(){ return _para; }
		/**
		 * @brief      Display the class on the console
		 */
		void Print()const{
			cout<<"perp = "; _perp.Print();
			cout<<"para = "; _para.Print();
		}
	private:
		VEC<T> _perp;	// A vector that is perpendicular to the incident plane
		VEC<T> _para;	// A vector that is parallel to the incident plane
	};
	
	/**
	 * @brief      Local Coordinate
	 */
	template<typename T>
	class LocalCorrdinate {
	public:
		/**
		 * @brief      Default constructor
		 */
		LocalCorrdinate(){};
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  ei    Incident local cooridinate element class
		 * @param[in]  es    Scatter(observe) local cooridinate element class
		 */
		LocalCorrdinate(const LocalCorrdinateElemt<T>& ei, const LocalCorrdinateElemt<T>& es){
			_ei = ei;
			_es = es;
		}
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  uv_t  Edge unit vector
		 * @param[in]  uv_sp Incidnet unit vector
		 * @param[in]  uv_s  Scatter(Observe) unit vector
		 *
		 * @return     Incident plane coordinate (ei) and observation plane coordinate (es)
		 *
		 * @ref        E. Knott, “The relationship between Mitzner‘s ILDC and Michaeli’s equivalent currents,
		 *             ” IEEE Trans. Antennas Propagat., vol. 33, no. 1, pp. 112–114, 1985.
		 */
		LocalCorrdinate(const VEC<T>& uv_t, const VEC<T>& uv_sp, const VEC<T>& uv_s){
			// Incident plane (Eq. 12)
			_ei.perp() = Unit(cross(uv_t, uv_sp));
			_ei.para() = cross(uv_sp, _ei.perp());
			
			// Scatter(Observation) plane (Eq. 13)
			_es.perp() = Unit(cross(uv_t, uv_s));
			_es.para() = cross(uv_s, _es.perp());	// [FIX] Why minus symbol?
		}
		/**
		 * @brief      Get    ei
		 */
		const LocalCorrdinateElemt<T>& ei()const{ return _ei; }
		/**
		 * @brief      Get    es
		 */
		const LocalCorrdinateElemt<T>& es()const{ return _es; }
		/**
		 * @brief      Get    ei (editable)
		 */
		LocalCorrdinateElemt<T>& ei(){ return _ei; }
		/**
		 * @brief      Get    es (editable)
		 */
		LocalCorrdinateElemt<T>& es(){ return _es; }
		/**
		 * @brief      Display the class on the console
		 */
		void Print()const{
			cout<<"+-----------------+"<<endl;
			cout<<"|     Summary     |"<<endl;
			cout<<"+-----------------+"<<endl;
			cout<<"ei : "<<endl; _ei.Print();
			cout<<"es : "<<endl; _es.Print();
		}
	private:
		LocalCorrdinateElemt<T> _ei;	// Incident local coordinate element class
		LocalCorrdinateElemt<T> _es;	// Scatter(observe) local coordinate element class
	};
	
	/**
	 * @brief      Edge Coordinate
	 */
	template<typename T>
	class EdgeCoordinate {
	public:
		/**
		 * @brief      Default constructor
		 */
		EdgeCoordinate(){};
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  x     X direction vector in Edge local coordinate
		 * @param[in]  y     Y direction vector in Edge local coordinate
		 * @param[in]  z     Z direction vector in Edge local coordinate
		 */
		EdgeCoordinate(const VEC<T>& x, const VEC<T>& y, const VEC<T>& z){
			_x = x;
			_y = y;
			_z = z;
		}
		/**
		 * @brief      Get    x
		 */
		const VEC<T>& x()const{ return _x; }
		/**
		 * @brief      Get    y
		 */
		const VEC<T>& y()const{ return _y; }
		/**
		 * @brief      Get    z
		 */
		const VEC<T>& z()const{ return _z; }
		/**
		 * @brief      Get    x (editable)
		 */
		VEC<T>& x(){ return _x; }
		/**
		 * @brief      Get    y (editable)
		 */
		VEC<T>& y(){ return _y; }
		/**
		 * @brief      Get    z (editable)
		 */
		VEC<T>& z(){ return _z; }
		/**
		 * @brief      Display the class on the console
		 */
		void Print(){
			cout<<"+----------------+"<<endl;
			cout<<"|     Summary    |"<<endl;
			cout<<"+----------------+"<<endl;
			cout<<"x : "; _x.Print();
			cout<<"y : "; _y.Print();
			cout<<"z : "; _z.Print();
		}
	private:
		VEC<T> _x, _y, _z;	// X,Y,Z direction vector in Edge local coordinate
	};
	
	/**
	 * @brief      Beta, phi with/without prime class
	 */
	template<typename T>
	class BetaPhiPrime {
	public:
		/**
		 * @brief      Default constructor
		 */
		BetaPhiPrime(){}
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  beta  [rad] Beta angle (measure from edge to scatter vector)
		 * @param[in]  betap [rad] Beta prime angle (measure from edge to incident vector)
		 * @param[in]  phi   [rad] Phi angle (measure from Edge X vector to projected scatter vector)
		 * @param[in]  phip  [rad] Phi prime angle (measure from Edge X vector to projected incidnet vector)
		 */
		BetaPhiPrime(const T beta, const T betap, const T phi, const T phip){
			_beta = beta;
			_betap = betap;
			_phi = phi;
			_phip = phip;
		}
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  Ei_d  Incident direction vector
		 * @param[in]  Ed_d  Scatter(observe) direction vector
		 * @param[in]  e     Edge local coordinate
		 */
		BetaPhiPrime(const VEC<T>& Ei_d, const VEC<T>& Ed_d, const EdgeCoordinate<T>& e){
			_Ei_d = Ei_d;
			_Ed_d = Ed_d;
			// Find betap (incidnet) / beta (observation)
			_betap = acos( dot( Ei_d, e.z()) );
			_beta  = acos( dot( Ed_d, e.z()) );
			
			// Find phip (incident)  / phi (observation). Measure from e_x
			_phip  = vec::angle(-Ei_d, e.x());
			_phi   = vec::angle( Ed_d, e.x());
		}
		/**
		 * @brief      Get    Beta angle in rad
		 */
		const T& beta()const{ return _beta; }
		/**
		 * @brief      Get    Beta prime angle in rad
		 */
		const T& betap()const{ return _betap; }
		/**
		 * @brief      Get    Phi angle in rad
		 */
		const T& phi()const{ return _phi; }
		/**
		 * @brief      Get    Phi prime angle in rad
		 */
		const T& phip()const{ return _phip; }
		/**
		 * @brief      Get    Beta angle in rad (editable)
		 */
		T& beta(){ return _beta; }
		/**
		 * @brief      Get    Beta prime angle in rad (editable)
		 */
		T& betap(){ return _betap; }
		/**
		 * @brief      Get    Phi angle in rad (editable)
		 */
		T& phi(){ return _phi; }
		/**
		 * @brief      Get    Phi prime angle in rad (editable)
		 */
		T& phip(){ return _phip; }
		/**
		 * @brief      Display the class on the console
		 */
		void Print(){
			cout<<"+-------------------+"<<endl;
			cout<<"| Beta, Phi & Prime |"<<endl;
			cout<<"+-------------------+"<<endl;
			cout<<"beta   = "<<rad2deg(_beta)<<" [deg]"<<endl;
			cout<<"betap  = "<<rad2deg(_betap)<<" [deg]"<<endl;
			cout<<"phi    = "<<rad2deg(_phi)<<" [deg]"<<endl;
			cout<<"phip   = "<<rad2deg(_phip)<<" [deg]"<<endl;
		}
	private:
		VEC<T> _Ei_d, _Ed_d;
		T _beta;	// [rad] Beta angle (measure from edge to scatter vector)
		T _betap;	// [rad] Beta prime angle (measure from edge to incident vector)
		T _phi;		// [rad] Phi angle (measure from Edge X vector to projected scatter vector)
		T _phip;	// [rad] Phi prime angle (measure from Edge X vector to projected incidnet vector)
	};
	
	/**
	 * @brief      Equivalent Edge Current (EEC)
	 *
	 * @ref        Equivalent Edge Current (EEC) of Michaeli (1984) &
	 *             Incremental Length Diffraction Coefficient (ILDC) from Mitzner (1985)
	 */
	class EEC {
	public:
		/**
		 * @brief      Default constructor
		 */
		EEC(){};
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  v     Beta, Phi with/without prime angle class
		 * @param[in]  n     Wedge factor
		 */
		EEC(const BetaPhiPrime<double> v, const double n){
			_beta = v.beta();
			_betap = v.betap();
			_phi  = v.phi();
			_phip = v.phip();
			_n    = n;
			// Calculate all coefficients
			PreCalA1A2();
			DiffractionCoeffMain();
			UFactor();
			DiffractionCoeffPrime();
		}
		/**
		 * @brief      Get   Michaeli's coefficient on the E component
		 */
		double Dm()const{ return _Dm; }
		/**
		 * @brief      Get   Michaeli's coefficient on the 1st H component
		 */
		double De()const{ return _De; }
		/**
		 * @brief      Get   Michaeli's coefficient on the 2nd H component
		 */
		double Dem()const{ return _Dem; }
		/**
		 * @brief      Get   Mitzner's coefficient on the E component
		 */
		double Dperpp()const{ return _Dperpp; }
		/**
		 * @brief      Get   Mitzner's coefficient on the E component (Editable)
		 */
		double& Dperpp(){ return _Dperpp; }
		/**
		 * @brief      Get   Mitzner's coefficient on the 1st H component
		 */
		double Dparap()const{ return _Dparap; }
		/**
		 * @brief      Get   Mitzner's coefficient on the 1st H component (Editable)
		 */
		double& Dparap(){ return _Dparap; }
		/**
		 * @brief      Get   Mitzner's coefficient on the 2nd H component
		 */
		double Dxp()const{ return _Dxp; }
		/**
		 * @brief      Get   Mitzner's coefficient on the 2nd H component (Editable)
		 */
		double& Dxp(){ return _Dxp; }
		/**
		 * @brief      Get   illumination sign symbol for incident polygon (plus)
		 */
		double Up()const{ return _Up; }
		/**
		 * @brief      Get   illumination sign symbol for incident polygon (plus) (editable)
		 */
		double& Up(){ return _Up; }
		/**
		 * @brief      Get   illumination sign symbol for NOT incident polygon (minus)
		 */
		double Um()const{ return _Um; }
		/**
		 * @brief      Get   illumination sign symbol for NOT incident polygon (minus) (editable)
		 */
		double& Um(){ return _Um; }
		/**
		 * @brief      Display the class on the console
		 */
		void Print(){
			cout<<"+-----------------+"<<endl;
			cout<<"|     Summary     |"<<endl;
			cout<<"+-----------------+"<<endl;
			cout<<"De     = "<<_De<<endl;
			cout<<"Dm     = "<<_Dm<<endl;
			cout<<"Dem    = "<<_Dem<<endl;
			cout<<"Dperpp = "<<_Dperpp<<endl;
			cout<<"Dparap = "<<_Dparap<<endl;
			cout<<"Dxp    = "<<_Dxp<<endl;
		}
	private:
		/**
		 * @brief      Calculate the U factor if the incident ray is illuminate the value is "1"
		 *             otherwise is "0" w.r.t. two side of wedge.
		 *
		 * @return     Return the internal factor _Up(plus side) and _Um(minus side)
		 */
		void UFactor(){
			if(_phip < (PI - (2 - _n)*PI) ){
				_Up = 1;
				_Um = 0;
			}
			if(_phip > PI ){
				_Up = 0;
				_Um = 1;
			}
			if((_phip >= (PI - (2 - _n)*PI)) && (_phip <= PI) ){
				_Up = 1;
				_Um = 1;
			}
			if(_phip >= (2*PI - (2 - _n)*PI) ){
				cerr<<"WARRNING:No this kind of angle, N must larger than 1"<<endl;
				_Up = 0;
				_Um = 0;
			}
		}
		/**
		 * @brief      Pre-calculation of alpha1 & alpha2 angle and w.r.t. sin and cos values. As
		 *             the same process, it is used to beta/betap and phi/phip angles and Q values.
		 */
		void PreCalA1A2(){
			// Alpha 1 & 2
			_ca1 = sin(_beta)*cos(_phi)/sin(_betap);
			_ca2 = sin(_beta)*cos(_n*PI-_phi)/sin(_betap);
			_a1  = acos(_ca1);
			_a2  = acos(_ca2);
			_sa1 = sin(_a1);
			_sa2 = sin(_a2);
			// Beta
			_sb  = sin(_beta);
			_sbp = sin(_betap);
			_cb  = cos(_beta);
			_cbp = cos(_betap);
			// Phi
			_sp = sin(_phi);
			_cp = cos(_phi);
			_spp = sin(_phip);
			_cpp = cos(_phip);
			// Q
			_Q = 2 * (1 + _cb*_cbp)/(_sb*_sbp) * sin(0.5*(_beta+_betap)) * sin(0.5*(_beta-_betap));
		}
		/**
		 * @brief      Calculation of Michaeli's diffraction coefficients (Dm, De, Dem)
		 *
		 * @ref        Equivalent Edge Current (EEC) of Michaeli (1984)
		 */
		void DiffractionCoeffMain(){
			double n_sin_pp_n  = (1/_n)*sin(_phip/_n);
			double cos_pi_a1_n = cos((PI-_a1)/_n);
			double cos_pi_a2_n = cos((PI-_a2)/_n);
			double cos_pp_n    = cos(_phip/_n);
			
			double denominator1 = cos_pi_a1_n - cos_pp_n;
			double denominator2 = cos_pi_a2_n + cos_pp_n;
			
			_De = n_sin_pp_n / denominator1 + n_sin_pp_n / denominator2;
			
			
			double sin_npi_p = sin(_n*PI-_phi);
			double n_sin_pi_a1_n = (1/_n)*sin((PI-_a1)/_n);
			double n_sin_pi_a2_n = (1/_n)*sin((PI-_a2)/_n);
			
			_Dm = _sp/_sa1 * n_sin_pi_a1_n / denominator1 +
				  sin_npi_p/_sa2 * n_sin_pi_a2_n / denominator2;
			
			
			double cos_npi_p = cos(_n*PI-_phi);
			
			
			_Dem = _Q/_sbp * ( _cp/_sa1 * n_sin_pi_a1_n / denominator1 -
						   cos_npi_p/_sa2 * n_sin_pi_a2_n / denominator2);

			// Avoid singularity
			if(abs(_sa1) < 1e-5 || abs(_sa2) < 1e-5){
				_Dm = 0;
				_Dem = 0;
			}
			
			// Avoid singularity
			if((abs(denominator1) < 1e-3) || (abs(denominator2) < 1e-3)){
				_De = 0;
				_Dm = 0;
				_Dem = 0;
			}
		}
		/**
		 * @brief      Calculation of Mitzner's diffraction coefficients (Dperpp, Dparap, Dxp)
		 *
		 * @ref        Incremental Length Diffraction Coefficient (ILDC) from Mitzner (1985)
		 */
		void DiffractionCoeffPrime(){
			double sin_npi_p  = sin(_n*PI - _phi);
			double cos_npi_p  = cos(_n*PI - _phi);
			double sin_npi_pp = sin(_n*PI - _phip);
			double cos_npi_pp = cos(_n*PI - _phip);
			
			
			double denominator1 = _ca1 + _cpp;
			double denominator2 = _ca2 + cos_npi_pp;
			
			_Dperpp = _Up * _sp / denominator1 - _Um * sin_npi_p / denominator2;
			
			_Dparap = -_Up * _spp / denominator1 - _Um * sin_npi_pp /denominator2;
			
			_Dxp = -_Up * ( _Q * _cp / denominator1 - _cbp ) + _Um * ( _Q*cos_npi_p / denominator2 - _cbp );
			
			
			// Avoid singularity
			if((abs(denominator1) < 1e-3) || (abs(denominator2) < 1e-3)){
				_Dperpp = 0;
				_Dparap = 0;
				_Dxp = 0;
			}
		}
	private:
		// Input
		double _beta;	// [rad] Beta angle (measure from edge to scatter vector)
		double _betap;	// [rad] Beta prime angle (measure from edge to incident vector)
		double _phi;	// [rad] Phi angle (measure from Edge X vector to projected scatter vector)
		double _phip;	// [rad] Phi prime angle (measure from Edge X vector to projected incidnet vector)
		double _n;		// Wedge factor
		// Output
		double _Dm;		// Michaeli's coefficient on the E component
		double _De;		// Michaeli's coefficient on the 1st H component
		double _Dem;	// Michaeli's coefficient on the 2nd H component
		double _Dperpp;	// Mitzner's coefficient on the E component
		double _Dparap;	// Mitzner's coefficient on the 1st H component
		double _Dxp;	// Mitzner's coefficient on the 2nd H component
		// Internal parameters
		double _Q;			// Q factor
		double _a1, _a2;	// [rad] Alpha1 & alpha2 angle
		double _ca1, _sa1;	// cos and sin values of alpha1
		double _ca2, _sa2;	// cos and sin values of alpha2
		double _cb, _sb;	// cos and sin values of beta
		double _cbp, _sbp;	// cos and sin values of beta prime
		double _cp, _sp;	// cos and sin values of phi
		double _cpp, _spp;	// cos and sin values of phi prime
		double _Up, _Um;	// Illumination factor
	};

	//+---------------------------+
	//|      Electric Field       |
	//+---------------------------+
	template<typename T>
	class EMWave {
	public:
		/*
		 * Default constructor
		 */
		EMWave(){}
		/*
		 * Constructor with memeber variable copy
		 * @param[in]   k    [m,m,m] Waveform prapogation direction
		 * @param[in]   o    [m,m,m] Origin of wave prapogation
		 * @param[in]   cplx (VEC<CPLX<T>>) Electric field oscilation (xyz direction and phase)
		 */
		EMWave(const VEC<T>& k, const VEC<T>& o, const VEC<CPLX<T> >& cplx){
			_k = k;
			_o = o;
			_cplx = cplx;
		}
		/*
		 * Constructor with memeber variable copy
		 * @param[in]   k    [m,m,m] Waveform prapogation direction
		 * @param[in]   o    [m,m,m] Origin of wave prapogation
		 * @param[in]   pol  (string) Polarization character 'H' or 'V'
		 */
		EMWave(const VEC<T>& k, const VEC<T>& o, const string pol){
			if(pol.length() != 1){
				cerr<<"ERROR::EMWave:The input pol = "<<pol<<" is not single char"<<endl;
				exit(EXIT_FAILURE);
			}
			_txpol = pol[0];
			_k = k;
			_o = o;
			VEC<T> dirV, dirH;
			PolarizationDirection(k, dirH, dirV);
			// _cplx with zero phase and unit(1) amplitude
			if(toupper(_txpol) == 'H'){
				_cplx = VEC<CPLX<T> >(CPLX<T>(dirH.x(),0), CPLX<T>(dirH.y(),0), CPLX<T>(dirH.z(),0));
			}else{
				_cplx = VEC<CPLX<T> >(CPLX<T>(dirV.x(),0), CPLX<T>(dirV.y(),0), CPLX<T>(dirV.z(),0));
			}
		}
		/*
		 * Get k(Waveform direction) value
		 * @return VEC<T> structure of k
		 */
		VEC<T>& k(){ return _k; }
		const VEC<T>& k()const{ return _k; }
		/*
		 * Get o(original position) value
		 * @return VEC<T> structure of o
		 */
		VEC<T>& o(){ return _o; }
		const VEC<T>& o()const{ return _o; }
		/*
		 * Get cplx(complex value of electric filed) value
		 * @return VEC<CPLX<T> > structure of cplx
		 */
		VEC<CPLX<T> >& cplx(){ return _cplx; }
		const VEC<CPLX<T> >& cplx()const{ return _cplx; }
		/**
		 * Add a phase on this structure by giving phase value
		 * @param[in] phs Phase values
		 */
		void AddPhase(const double phs){
			CPLX<double> phase = mat::exp(-phs);
			_cplx.x() = _cplx.x() * phase;
			_cplx.y() = _cplx.y() * phase;
			_cplx.z() = _cplx.z() * phase;
		}
		/**
		 * Calculate the Reflected Electric field and update to the original class
		 * @param[in] N       Normal vector of incident surface in global coordinate
		 * @param[in] Rf      {TE, TM} Reflection Factor w.r.t. material of surface
		 * @param[in] k_next  Reflection firection vector
		 * @ref Ling, H., Chou, R. C., & Lee, S. W. (1989). Shooting and bouncing rays: Calculating the RCS of an arbitrarily
		 *      shaped cavity. Antennas and Propagation, IEEE Transactions on, 37(2), 194–205.
		 * @remark NOTE: This member function will self update to original Ei.
		 */
		void ReflectionElectricField(const VEC<T>& N, const RF& Rf, const VEC<T>& k_next){
			ReflectionElectricField(this, this, N, Rf,k_next);
		}
		/**
		 * Calculate the Reflected Electric field, Er
		 * @param[in] N       Normal vector of incident surface in global coordinate
		 * @param[in] Rf      {TE, TM} Reflection Factor w.r.t. material of surface
		 * @param[in] k_next  Reflection firection vector
		 * @return Reflection Electric field(Er)
		 * @ref Ling, H., Chou, R. C., & Lee, S. W. (1989). Shooting and bouncing rays: Calculating the RCS of an arbitrarily
		 *      shaped cavity. Antennas and Propagation, IEEE Transactions on, 37(2), 194–205.
		 */
		EMWave<T> ReflectionElectricField(const VEC<T>& N, const RF& Rf, const VEC<float>& Ref_o, const VEC<T>& k_next){

			EMWave<T> Er;

			VEC<T> Local_inc_theta_vec, Local_inc_phi_vec;
			VEC<T> Local_ref_theta_vec, Local_ref_phi_vec;
			// normal vector of reflection surface
			VEC<T> m = Unit(cross(-_k ,N));
			// local XYZ (eq.10)
			VEC<T> X_vec = cross(m,N);
			VEC<T> Y_vec = -m;
			VEC<T> Z_vec = -N;
			// local incident angle (eq.11)
			// theta_i = acos(vec::angle(-Ei.k, N))
			// phi_i   = 0
			T ct = dot(-_k ,N)/(_k.abs()*N.abs());
			T st = sqrt(1-ct*ct);
			// local incident theta & phi vector (eq.12)
			Local_inc_theta_vec = Unit(ct * X_vec - st * Z_vec);
			Local_inc_phi_vec   = Y_vec;
			// Reflection
			Local_ref_theta_vec = Unit(-ct * X_vec - st * Z_vec);
			Local_ref_phi_vec   = Y_vec;

			// Amplitude of reflected field (eq.9), [Note: different than ref?]
			CPLX<T> Et = Rf.TM_par()  * dot(_cplx, Local_inc_theta_vec) *  1;
			CPLX<T> Ep = Rf.TE_perp() * dot(_cplx, Local_inc_phi_vec)   * -1;	// Ep (diff?)
			// Ref. E field complex (eq.9)
			Er.cplx().x() = Et * Local_ref_theta_vec.x() + Ep * Local_ref_phi_vec.x();
			Er.cplx().y() = Et * Local_ref_theta_vec.y() + Ep * Local_ref_phi_vec.y();
			Er.cplx().z() = Et * Local_ref_theta_vec.z() + Ep * Local_ref_phi_vec.z();
			// Assign Ref E Field
			Er.k() = k_next;
			Er.o() = VEC<T>(Ref_o.x(), Ref_o.y(), Ref_o.z());

			return Er;
		}
		/**
		 * Display all memeber variable on console
		 */
		void Print(){
			cout<<"+-----------------+"<<endl;
			cout<<"|      EM wave    |"<<endl;
			cout<<"+-----------------+"<<endl;
			cout<<"Pol    = "<<_txpol<<endl;
			cout<<"k      = "; _k.Print();
			cout<<"o      = "; _o.Print();
			cout<<"cplx   = "; _cplx.Print();
		}
	private:
		/**
		 * Definition the polarization by giving the waveform direction (not electric field)
		 * @param[in]  k     [m,m,m]Waveform direction (not electric field direction)
		 * @param[out] dirH  [m,m,m]Return the H definition direction
		 * @param[out] dirV  [m,m,m]Return the V definition direction
		 */
		void PolarizationDirection(const VEC<T>& k, VEC<T>& dirH, VEC<T>& dirV){
			const VEC<T> Z(0,0,1);
			const VEC<T> Y(0,1,0);
			if(vec::angle(-k, Z) < 1E-5){	// k align the Z axis
				dirH = Y;
			}else{
				dirH = Unit(cross(k, Z));
			}
			dirV = Unit(cross(k, dirH));
		}
	private:
		char _txpol;			// Tx polairzation
		VEC<T> _k;				// Wave direction
		VEC<T> _o;				// Start position
		VEC<CPLX<T> > _cplx;	// Electric field
	};
	
	//+-----------------------------------------------------------------------+
	//|                           Other functions                             |
	//+-----------------------------------------------------------------------+
	/**
	 * @brief      Find Diffraction Electric field (without distance phase)
	 *
	 * @param[in]  Ei    Incident Electric filed
	 * @param[in]  dL    [m] Integration small length
	 * @param[in]  cf    {Dm, De, Dem, Dperpp, Dparap, Dxp}, Diffraction coefficent
	 * @param[in]  el    {ei:{perp,para},es:{perp,para}} Edge Local coordinate
	 * @param[in]  betap [rad] beta-prime angle, measure from incident ray to edge
	 * @param[in]  beta  [rad] beta angle, measure from observation ray to edge
	 * @param[in]  SHOW  Display results on the console or not(defulat=false)
	 *
	 * @return     Return the diffraction electric field in VEC<CPLX<double> > format
	 *
	 * @ref        F. Weinmann, “Ray tracing with PO/PTD for RCS modeling of large complex objects,” IEEE 
	 *             Trans. Antennas Propagat., vol. 54, no. 6, pp. 1797–1806, 2006.
	 *
	 * @note       雖然在 WedgeAngle = 180 時，Ed會有值，但是並不表示 Epo + Ed 之後會變大，因為coherence sum之後，部分
	 *             Ed項中的數值會與Epo互相抵消，真正留下來的才是最終的結果，經過IDL plate測試，在 WedgeAngle = 180 時，
	 *             Ed會有明顯的數值(在振幅項明顯增加)但是累加後，主播樹的數值仍維持的明顯，唯在 phi = 0 附近，差異較大，於真實
	 *             環境中，邊緣上的Epo要加回來，如此邊緣的 Epo + Ed 應會顯著被抑制。
	 */
	VEC<CPLX<double> > DiffractionEF(const VEC<CPLX<double> >& Ei, const double dL, EEC& cf, const LocalCorrdinate<double>& el,
									 const double betap, const double beta, const bool SHOW=false){
		CPLX<double> Ei_dot_ei_perp = Ei.x()*el.ei().perp().x() + Ei.y()*el.ei().perp().y() + Ei.z()*el.ei().perp().z();
		CPLX<double> Ei_dot_ei_para = Ei.x()*el.ei().para().x() + Ei.y()*el.ei().para().y() + Ei.z()*el.ei().para().z();
		
//		cf.Dperpp() = 0;
//		cf.Dparap() = 0;
//		cf.Dxp()    = 0;
		
		
		VEC<CPLX<double> > Ed;
		// x
		Ed.x() = 1/(2*PI) * dL * ((cf.Dm() - cf.Dperpp())          * Ei_dot_ei_perp * el.es().perp().x() +							// M
								  (cf.De() - cf.Dparap())          * Ei_dot_ei_para * el.es().para().x() * (sin(beta)/sin(betap)) +	// I1
								  (cf.Dem()*sin(betap) - cf.Dxp()) * Ei_dot_ei_perp * el.es().para().x() * (sin(beta)/sin(betap))	// I2
								  );
		// y
		Ed.y() = 1/(2*PI) * dL * ((cf.Dm() - cf.Dperpp())          * Ei_dot_ei_perp * el.es().perp().y() +							// M
								  (cf.De() - cf.Dparap())          * Ei_dot_ei_para * el.es().para().y() * (sin(beta)/sin(betap)) +	// I1
								  (cf.Dem()*sin(betap) - cf.Dxp()) * Ei_dot_ei_perp * el.es().para().y() * (sin(beta)/sin(betap))	// I2
								  );
		// z
		Ed.z() = 1/(2*PI) * dL * ((cf.Dm() - cf.Dperpp())          * Ei_dot_ei_perp * el.es().perp().z() +							// M
								  (cf.De() - cf.Dparap())          * Ei_dot_ei_para * el.es().para().z() * (sin(beta)/sin(betap)) +	// I1
								  (cf.Dem()*sin(betap) - cf.Dxp()) * Ei_dot_ei_perp * el.es().para().z() * (sin(beta)/sin(betap))	// I2
								  );
		
//		VEC<CPLX<double> > tmp = Ed;
		
		if(SHOW){
			el.Print();
		}
		
		return Ed;
	}
	/**
	 * @brief      Wedge factor (2-angle)/pi
	 *
	 * @param[in]  angle_rad  [rad] Wedge internal angle in rad
	 *
	 * @return     Return the wedge factor, n
	 * 
	 * @example    n = 1 (for pi angle)
	 */
	double WedgeFactor(const double angle_rad){
		return 2 - angle_rad / PI;
	}
	/**
	 * @brief      Add a distance on the phase
	 *
	 * @param[in]  Ef     Electric field
	 * @param[in]  lambda [m] Wavelength
	 * @param[in]  dis    [m] Distance
	 *
	 * @return     Replace the original Ef to a new one
	 */
	template<typename T>
	void AddPhase(VEC<CPLX<T> >& Ef, const T lambda, const T dis){
		// add phase
		CPLX<T> phase = mat::exp(-2*PI*dis/lambda);
		Ef.x() = Ef.x() * phase;
		Ef.y() = Ef.y() * phase;
		Ef.z() = Ef.z() * phase;
	}
	/**
	 * @brief      Find the cross edge length that is cross the origin of the rectangular
	 *
	 * @param[in]  e      Edge class
	 * @param[in]  uv_sp  Incidnet unit vector
	 * @param[in]  area   Original ray grid area
	 *
	 * @return     Return the cross length on the proejcted rectangular
	 */
	double CrossEdgeLength(const EdgeCoordinate<double>& e, const VEC<double>& uv_sp, const double& area){
		// A. Find equivalent edge length
		double W = sqrt(area);
		// B. Find angle between normal and incident ray
		double theta1 = vec::angle(e.y(), -uv_sp);
		// C. Projected rectangular length
		double L = W / cos(theta1);
		// D. Find the vector along the project rectangular, spp = s_prime_peoject
		VEC<double> uv_spp = Unit(cross(e.y(), uv_sp));
		// E. Find the angle between projected vector and W
		double ez_dot_uvssp = abs(dot(e.z(), uv_spp));
		double theta2 = 0;
		if(abs(ez_dot_uvssp - 1) > 1E-7){
			theta2 = acos(ez_dot_uvssp);
		}
		// F. Find the cross length on the projected rectangular
		double theta  = abs(atan(L/W));
		double dL;
		if(theta2 <= theta){
			dL = W / cos(theta2);
		}else{
			dL = L / cos(deg2rad(90)-theta2);
		}
		
		return dL;
	}

	/**
	 * Check the incidnet is can be scattered or not for diffraction
	 * @param[in] uv_sp		 Incident unit vector
	 * @param[in] tri1		 Facet-1 triangle object
	 * @param[in] j_shared1	 Shared edge index of Facet-1
	 * @param[in] tri2		 Facet-2 triangle object
	 * @return Return the boolean result
	 */
	bool CheckDiffractionIncidentBoundary(const BVH& bvh, const VEC<double>& uv_sp, const TRI<double>& tri1, const size_t j_shared1, TRI<double>& tri2, EdgeCoordinate<double>& e2, const size_t kk){
		// uv_sp: Un-projected incident unit vector
		// tri1:  Facet-1 triangle object
		// j_shared1: Shared edge index of Facet-1

		// Edge (Facet-1)
		EdgeList<double> EL1(tri1);

		EdgeCoordinate<double> e1;
		e1.z() = Unit( EL1.Edge(j_shared1).E() - EL1.Edge(j_shared1).S() );	// Along edge
		e1.y() = tri1.getNormal();												// Normal edge
		e1.x() = cross(e1.y(), e1.z());									// On the plate

		VEC<double> uv_sp_proj = vec::ProjectOnPlane(uv_sp, e1.z());
		// Nearest polygon
		// Get triangle
		Obj* obj2 = (*(bvh.GetBuildPrims()))[ bvh.GetIdxPoly()[tri1.IDX_Near(j_shared1)] ];
		// Force convert
		tri2 = *((TRI<float>*)(obj2));

		// Shared edge index for Facet-2
		int j_shared2 = 0;
		for(int k=0;k<3;++k){
//			cout<<"tri2.IDX_Near("<<k<<") = "<<tri2.IDX_Near(k)<<", tri1.IDX() = "<<tri1.IDX()<<endl;
			if(tri2.IDX_Near(k) == tri1.IDX()){
				j_shared2 = k;
			}
		}

		// Edge (Facet-2)
		EdgeList<double> EL2(tri2);

//		EdgeCoordinate<double> e2;
		e2.z() = Unit( EL2.Edge(j_shared2).E() - EL2.Edge(j_shared2).S() );	// Along edge
		e2.y() = tri2.getNormal();											// Normal edge
		e2.x() = cross(e2.y(), e2.z());										// On the plate


		VEC<double> vv1_vv0 = tri2.V1() - tri2.V0();
		VEC<double> vv2_vv0 = tri2.V2() - tri2.V0();
		VEC<double> N2  = cross(vv1_vv0, vv2_vv0);



		bool isOK = vec::CheckEffectiveDiffractionIncident(e1.x(), e1.z(), e2.x(), uv_sp_proj, false, kk);

//		// DEBUG (START) ========================================================================================================================================================================
//		if(kk==60602) {
//			tri2.Print();
////			printf("\n\n\n\n>>>> CPU >>>>\ntri1.IDX=%ld, tri2.idx=%ld, e2.x()=(%f,%f,%f), e2.y()=(%f,%f,%f), e2.z()=(%f,%f,%f), N2=(%f,%f,%f), vv1_vv0=(%f,%f,%f), vv2_vv0=(%f,%f,%f)\n>>>>>>>>>>>>>\n\n",
////				   tri1.IDX(), tri2.IDX(), e2.x().x(), e2.x().y(), e2.x().z(), e2.y().x(), e2.y().y(), e2.y().z(), e2.z().x(), e2.z().y(), e2.z().z(),
////				   N2.x(), N2.y(), N2.z(), vv1_vv0.x(), vv1_vv0.y(), vv1_vv0.z(), vv2_vv0.x(), vv2_vv0.y(), vv2_vv0.z());
////			printf("\n\n\n\n>>>> CPU >>>>\ntri1.IDX=%ld, tri1.IDX_Near=[%ld,%ld,%ld], tri2.IDX=%ld, e1.x=(%f,%f,%f), e1.z=(%f,%f,%f), e2.x=(%f,%f,%f), uv_sp_proj=(%f,%f,%f), isOK=%d\n>>>>>>>>>>>>>\n\n",
////				   tri1.IDX(), tri1.IDX_Near(0), tri1.IDX_Near(1), tri1.IDX_Near(2), tri2.IDX(),
////				   e1.x().x(), e1.x().y(), e1.x().z(), e1.z().x(), e1.z().y(), e1.z().z(), e2.x().z(), e2.x().y(), e2.x().z(),
////				   uv_sp_proj.x(), uv_sp_proj.y(), uv_sp_proj.z(), isOK);
////			printf("\n\n\n\n>>>> CPU >>>>\ntri1.IDX=%ld, tri2.idx=%ld, p=%ld, j_shared2=%d, uv_sp=(%f,%f,%f), e1.z=(%f,%f,%f), e2.x=(%f,%f,%f)\n>>>>>>>>>>>>>\n\n", tri1.IDX(), tri2.IDX(), j_shared1, j_shared2, uv_sp.x(), uv_sp.y(), uv_sp.z(), e1.z().x(), e1.z().y(), e2.x().z(), e2.x().y(), e2.x().z());
//		}
//		// DEBUG (END) ==========================================================================================================================================================================

		return isOK;
	}

	double UnitStep(const double in){
		if(in < 0){
			return 0;
		}else{
			return 1;
		}
	}

	CPLX<double> dot(const VEC<double>& a, const VEC<CPLX<double> >& b){
		CPLX<double> out = 0;
		out += a.x() * b.x();
		out += a.y() * b.y();
		out += a.z() * b.z();
		return out;
	}

	void FringeIM(const EMWave<double>& Ei, const VEC<double>& z, const double n, const double k,
				  const VEC<double>& sp, const double beta, const double betap, const double phi, const double phip,
				  CPLX<double>& I, CPLX<double>& M) {

		double sb = sin(beta);
		double sbp = sin(betap);
		double cotb = mat::Cot(beta);
		double cotbp = mat::Cot(betap);
		double cp = cos(phi);
//		double cpp = cos(phip);

//		double u = sb*cp/sbp;
		double u = cp - 2 * cotb*cotb;


		CPLX<double> a = CPLX<double>(0,-1) * mat::Log( u + CPLX<double>(0,1)*mat::Sqrt(CPLX<double>(1 - u*u, 0)) );
		CPLX<double> sa = mat::Sin(a);


		CPLX<double> jj(0, -1);

		CPLX<double> Eiz = dot(z, Ei.cplx());
		CPLX<double> Hiz = dot(z, cross(sp, Ei.cplx()));

//		CPLX<double> tp11 = -2.0 * jj / (k * sb*sb);
//		CPLX<double> tp12 = (sin(phi) * UnitStep(def::PI - phi) / (cp + u) + (1.0 / n) * sin(phi / n) * (1.0/(mat::Cos((def::PI - a) / n) - cos(phi / n))));
//		CPLX<double> tp13 = 2.0 * jj * mat::Sin((def::PI - a) / n) / (n * k * sb * sa);
//		CPLX<double> tp14 = (u * cotbp - cotb * cp) / (cos(phi / n) - mat::Cos((def::PI - a) / n));
//
//		CPLX<double> tp21 = 2.0 * jj * sin(phi) / (k * sb*sb);
//		CPLX<double> tp22 = UnitStep(def::PI - phi) / (cp + u);
//		CPLX<double> tp23 = (1.0 / n) * mat::Sin((def::PI - a) / n) * mat::Csc(a) * (1.0/(cos(phi / n) - mat::Cos((def::PI - a) / n)));
//
//		CPLX<double> tp200 = (def::PI - a) / n;
//		CPLX<double> tp201= def::PI - a;
//		CPLX<double> tp202 = mat::Sin((def::PI - a) / n);
//		CPLX<double> tp203 = (1.0 / n) * mat::Sin((def::PI - a) / n);
//		CPLX<double> tp204 = mat::Csc(a);
//		CPLX<double> tp205 = (1.0 / n) * mat::Sin((def::PI - a) / n) * mat::Csc(a);
//		CPLX<double> tp206 = mat::Cos((def::PI - a) / n);
//		CPLX<double> tp207 = 1.0/(cos(phi / n) - mat::Cos((def::PI - a) / n));
//		CPLX<double> tp208 = (1.0 / n) * mat::Sin((def::PI - a) / n) * mat::Csc(a) * (1.0/(cos(phi / n) - mat::Cos((def::PI - a) / n)));

		I = -2.0 * jj / (k * sb*sb) *
			(sin(phi) * UnitStep(def::PI - phi) / (cp + u) + (1.0 / n) * sin(phi / n) * (1.0/(mat::Cos((def::PI - a) / n) - cos(phi / n)))) *
			Eiz +
			2.0 * jj * mat::Sin((def::PI - a) / n) / (n * k * sb * sa) *
			(u * cotbp - cotb * cp) / (cos(phi / n) - mat::Cos((def::PI - a) / n)) * Hiz;

		M = 2.0 * jj * sin(phi) / (k * sb*sb) *
			(UnitStep(def::PI - phi) / (cp + u) -
			 (1.0 / n) * mat::Sin((def::PI - a) / n) * mat::Csc(a) * (1.0/(cos(phi / n) - mat::Cos((def::PI - a) / n)))) * Hiz;
	}
}



#endif













