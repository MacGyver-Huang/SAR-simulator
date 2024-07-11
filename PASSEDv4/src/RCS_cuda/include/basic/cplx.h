#ifndef CPLX_H_INCLUDED
#define CPLX_H_INCLUDED

#include <iostream>
#include <complex>
#include <cmath>

using namespace std;

namespace cplx{

	const double PI4=12.566370614359172;	// 4*PI
	const double PI2= 6.283185307179586;	// 2*PI
	const double C = 2.99792458e8;			// [m/s] ligh speed

	// ==============================================
	// Complex value
	// ==============================================
	template<typename T>
	class CPLX
	{
		public:
			// Constructure
			CPLX():_r(0),_i(0){};
			CPLX(T a,T b):_r(a),_i(b){};
			CPLX(const CPLX<T>& in):_r(in._r),_i(in._i){};
			CPLX(const T& in):_r(in),_i((T)0){};
			// Operator overloading
			CPLX<T>& operator=(const CPLX<T>& b);
			CPLX<T> operator-();
			template<typename T2> friend const CPLX<T2> operator+(const CPLX<T2>& L,const CPLX<T2>& R);
			template<typename T2> friend const CPLX<T2> operator-(const CPLX<T2>& L,const CPLX<T2>& R);
			template<typename T2> friend const CPLX<T2> operator*(const CPLX<T2>& L,const CPLX<T2>& R);
			template<typename T2> friend const CPLX<T2> operator/(const CPLX<T2>& L,const CPLX<T2>& R);
			template<typename T2,typename T3> friend const CPLX<T2> operator+(const CPLX<T2>& L,const T3& R);
			template<typename T2,typename T3> friend const CPLX<T2> operator-(const CPLX<T2>& L,const T3& R);
			template<typename T2,typename T3> friend const CPLX<T2> operator*(const CPLX<T2>& L,const T3& R);
			template<typename T2,typename T3> friend const CPLX<T2> operator/(const CPLX<T2>& L,const T3& R);
			template<typename T2,typename T3> friend const CPLX<T2> operator+(const T2& L,const CPLX<T3>& R);
			template<typename T2,typename T3> friend const CPLX<T2> operator-(const T2& L,const CPLX<T3>& R);
			template<typename T2,typename T3> friend const CPLX<T2> operator*(const T2& L,const CPLX<T3>& R);
			template<typename T2,typename T3> friend const CPLX<T2> operator/(const T2& L,const CPLX<T3>& R);
			template<typename T2> friend CPLX<T2>& operator+=(CPLX<T2>& res,const CPLX<T2>& R);
			template<typename T2> friend CPLX<T2>& operator-=(CPLX<T2>& res,const CPLX<T2>& R);
			template<typename T2> friend CPLX<T2>& operator*=(CPLX<T2>& res,const T2& R);
			template<typename T2> friend CPLX<T2>& operator/=(CPLX<T2>& res,const T2& R);
			template<typename T2> friend ostream &operator<<(ostream &stream, CPLX<T2> out);
			// Get
			T& r(){return _r;};
			T& i(){return _i;};
			const T& r()const{return _r;};
			const T& i()const{return _i;};
			// Misc.
			T abs();
			const T abs()const;
			const CPLX<T> sqrt()const{
				double sqrt2 = std::sqrt(2.0);
				double p = 1.0/sqrt2 * std::sqrt(std::sqrt(_r*_r + _i*_i) + _r);
				double sgn = (_i > 0)? 1.0:-1.0;
				double q = sgn/sqrt2 * std::sqrt(std::sqrt(_r*_r + _i*_i) - _r );
				return CPLX<T>(T(p),T(q));
			}
			CPLX<T> conj(){
				CPLX<T> out(_r, -_i);
				return out;
			}
			CPLX<T> pow(const int n);
			void SelfConj(){
				_i = -_i;
			}
			T phase();
			CPLX<T> exp(const CPLX<T>& x);
			CPLX<T> exp(const T& phs);
			void AddDistancePhase(const double f0, const double distance);
			void Print();
		private:
			T _r,_i;
	};

	// ==============================================
	// namespace declare
	// ==============================================
//	template<typename T> T abs(const CPLX<T>& a);
//	template<typename T> CPLX<T> conj(const CPLX<T>& a);
//	template<typename T> void SelfConj();
	template<typename T> T phase(const CPLX<T>& a);

	// Implement ********************************************************
	//
	// Operator overloading
	//
	template<typename T>
	CPLX<T>& CPLX<T>::operator=(const CPLX<T>& b){
		_r = b._r;
		_i = b._i;
		return *this;
	}
	
	template<typename T>
	CPLX<T> CPLX<T>::operator-(){
		return CPLX<T>(-_r,-_i);
	}

	template<typename T>
	const CPLX<T> operator+(const CPLX<T>& L,const CPLX<T>& R){
		return CPLX<T>( L._r+R._r,L._i+R._i );
	}

	template<typename T>
	const CPLX<T> operator-(const CPLX<T>& L,const CPLX<T>& R){
		return CPLX<T>( L._r-R._r,L._i-R._i );
	}

	template<typename T>
	const CPLX<T> operator*(const CPLX<T>& L,const CPLX<T>& R){
		return CPLX<T>( L._r*R._r - L._i*R._i, L._r*R._i + L._i*R._r );
	}

	template<typename T>
	const CPLX<T> operator/(const CPLX<T>& L,const CPLX<T>& R){
		T R_conj_R = R._r*R._r + R._i*R._i;
		return CPLX<T>( (L._r*R._r + L._i*R._i)/R_conj_R, (L._i*R._r - L._r*R._i)/R_conj_R );
	}

    // ----
	template<typename T1,typename T2>
	const CPLX<T1> operator+(const CPLX<T1>& L,const T2& R){
		return CPLX<T1>( L._r+R,L._i );
	}

	template<typename T1,typename T2>
	const CPLX<T1> operator-(const CPLX<T1>& L,const T2& R){
		return CPLX<T1>( L._r-R,L._i );
	}

	template<typename T1,typename T2>
	const CPLX<T1> operator*(const CPLX<T1>& L,const T2& R){
		return CPLX<T1>( L._r*R,L._i*R );
	}

	template<typename T1,typename T2>
	const CPLX<T1> operator/(const CPLX<T1>& L,const T2& R){
		return CPLX<T1>( L._r/R,L._i/R );
	}

    // ----
	template<typename T1,typename T2>
	const CPLX<T1> operator+(const T1& L,const CPLX<T2>& R){
		return CPLX<T1>( L+R._r,R._i );
	}

	template<typename T1,typename T2>
	const CPLX<T1> operator-(const T1& L,const CPLX<T2>& R){
		return CPLX<T1>( L-R._r,-R._i );
	}

	template<typename T1,typename T2>
	const CPLX<T1> operator*(const T1& L,const CPLX<T2>& R){
		return CPLX<T1>( R._r*L,R._i*L );
	}

	template<typename T1,typename T2>
	const CPLX<T1> operator/(const T1& L,const CPLX<T2>& R){
		T1 R_conj_R = R._r*R._r + R._i*R._i;
		return CPLX<T1>( (L*R._r)/R_conj_R, (-L*R._i)/R_conj_R );
	}

	template<typename T1>
    CPLX<T1>& operator+=(CPLX<T1>& res,const CPLX<T1>& R){
        res._r += R._r;
        res._i += R._i;
        return res;
    }
	
	template<typename T1>
	CPLX<T1>& operator-=(CPLX<T1>& res,const CPLX<T1>& R){
		res._r -= R._r;
		res._i -= R._i;
		return res;
	}
	
	template<typename T1>
	CPLX<T1>& operator*=(CPLX<T1>& res,const T1& R){
		res._r *= R;
		res._i *= R;
		return res;
	}

	template<typename T1>
    CPLX<T1>& operator/=(CPLX<T1>& res,const T1& R){
        res._r /= R;
        res._i /= R;
        return res;
    }

	template<typename T1>
	ostream& operator<<(ostream &stream, CPLX<T1> out){
		stream<<"("<<out.r()<<","<<out.i()<<")";
		return stream;
	}
	
	//
	// Misc.
	//
	template<typename T>
	T CPLX<T>::abs(){
		return std::sqrt(_r*_r + _i*_i);
	}
	
	template<typename T>
	const T CPLX<T>::abs()const{
		return std::sqrt(_r*_r + _i*_i);
	}


//	template<typename T>
//	CPLX<T>& CPLX<T>::conj(){
//		_i = -_i;
//		return *this;
//	}
	
	template<typename T>
	CPLX<T> CPLX<T>::pow(const int n){
		complex<T> out = std::pow(complex<T>(_r,_i),n);
		return CPLX<T>(std::real(out),std::imag(out));
	}

	
//	template<typename T>
//	void CPLX<T>::SelfConj(){
//		_i = -_i;
//	}

	template<typename T>
	T CPLX<T>::phase(){
		return T(atan2(double(_i),double(_r)));
	}
	
	template<typename T>
	CPLX<T> exp(const CPLX<T>& x){
		T imag = x.i();
		if(std::isinf(x.r())){
			if(x.r() < T(0)){
				if(!isfinite(imag)){
					imag = T(1);
				}
			}else{
				if(imag == 0 || !isfinite(imag)){
					if(std::isinf(imag)){
						imag = T(NAN);
					}
					return CPLX<T>(x.r(), imag);
				}
			}
		}else{
			if(std::isnan(x.r()) && x.i() == 0){
				return x;
			}
		}
		T e = std::exp(x.r());
		return CPLX<T>(e * cos(imag), e * sin(imag));
	}

	/**
	 * Add additional phase by distance
	 * @param [in] f0 : frequency in [Hz]
	 * @param [in] distance : Distance in [m]
	 */
	template<typename T>
	void CPLX<T>::AddDistancePhase(const double f0, const double distance){
		// Wavenumber
		double lambda = C/f0;
		double k0 = PI2/lambda;
		double phs = k0 * distance;
		// Add distance phase
//		complex<double> tmp = mat::exp(-phs);
//		CPLX<double> phase(tmp.real(), tmp.imag());
//		double cp = cos(+phs);	// real
//		double sp = sin(+phs);	// imag
		double cp = cos(-phs);	// real
		double sp = sin(-phs);	// imag
		double real = _r * cp - _i * sp;
		double imag = _r * sp + _i * cp;
		_r = T(real);
		_i = T(imag);
//		CPLX<double> phase(cp, sp);
//		_r = _r * phase.r() - _i * phase.i();
//		_i = _r * phase.i() + _i * phase.r();
	}

	template<typename T>
	void CPLX<T>::Print(){
		cout<<"("<<_r<<","<<_i<<")"<<endl;
	}
}

#endif // CPLX_H_INCLUDED
