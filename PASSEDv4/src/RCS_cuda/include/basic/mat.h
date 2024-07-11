#ifndef MAT_H_INCLUDED
#define MAT_H_INCLUDED

#include <cmath>
#include <typeinfo>
#include <string>
#include "def_func.h"
#include "d1.h"
#include "d2.h"
#include "d3.h"
#include "cplx.h"
#include "vec.h"

using namespace std;
using namespace d1;
using namespace d2;
using namespace d3;
using namespace cplx;
using namespace vec;


namespace mat{
	// Max Min
	template<typename T> T max(const T& a,const T& b);
	template<typename T> T min(const T& a,const T& b);
	template<typename T> T max(const T* a,const long num);
	template<typename T> T min(const T* a,const long num);
	template<typename T> T max(const T* a,const long num,long& index);
	template<typename T> T min(const T* a,const long num,long& index);
	template<typename T> T max(const D1<T>& a);
	template<typename T> T min(const D1<T>& a);
	template<typename T> T max(const D1<T>& a,long& index);
	template<typename T> T min(const D1<T>& a,long& index);
	template<typename T> T max(const D2<T>& a);
	template<typename T> T min(const D2<T>& a);
	template<typename T> T max(const D2<T>& a,long& idx_M,long& idx_N);
	template<typename T> T min(const D2<T>& a,long& idx_M,long& idx_N);
	template<typename T> T max(const D3<T>& a);
	template<typename T> T min(const D3<T>& a);
	template<typename T> T max(const vector<T>& a);
	template<typename T> T min(const vector<T>& a);
	template<typename T> T max3(const T a1, const T a2, const T a3);
	template<typename T> T min3(const T a1, const T a2, const T a3);
	template<typename T> void SetZero(vector<vector<T> >& in);
	template<typename T> T max(const vector<vector<CPLX<T> > >& in);
	template<typename T> T max(const vector<vector<CPLX<T> > >& in, size_t& ii, size_t& jj);
	template<typename T> T maxAbs(const vector<vector<D2<CPLX<T> > > >& in);
	template<typename T> T maxAbs(const D3<CPLX<T> >& in);
	template<typename T> T AvgStep(const vector<T>& a);
	template<typename T> VEC<T> min(const VEC<T>& a, const VEC<T>& b);
	template<typename T> VEC<T> max(const VEC<T>& a, const VEC<T>& b);
	template<typename T> D1<T> abs(const D1<T>& a);
	template<typename T> D2<T> abs(const D2<T>& a);
	template<typename T> D3<T> abs(const D3<T>& a);
	template<typename T> CPLX<T> Sin(const CPLX<T>& a);
	template<typename T> CPLX<T> Cos(const CPLX<T>& a);
	template<typename T> CPLX<T> Csc(const CPLX<T>& a);
	template<typename T> T Sec(const T rad);
	template<typename T> T Cot(const T rad);
	template<typename T> CPLX<T> Log(const CPLX<T>& in);
	template<typename T> CPLX<T> Sqrt(const CPLX<T>& in);
	template<typename T> T total(const D1<T>& a);
	template<typename T> T total(const D2<T>& a);
	template<typename T> CPLX<T> totalNaN(const D1<CPLX<T> >& a);
	template<typename T> void Scale(D1<T>& in, const T min_val, const T max_val);
	template<typename T> void Scale(D2<T>& in, const T min_val, const T max_val);
	template<typename T> T mean(const D1<T>& a);
	template<typename T> T mean(const D2<T>& a);
	// CPLX cast
	template<typename T> D1<T> abs(const D1<CPLX<T> >& in);
	template<typename T> D2<T> abs(const D2<CPLX<T> >& in);
	template<typename T> D3<T> abs(const D3<CPLX<T> >& in);
	template<typename T> VEC<T> abs(const VEC<CPLX<T> >& in);
	template<typename T> D2<T> phs(const D2<CPLX<T> >& in);
	// Basic or Modified functions
	template<typename T> T Square(const T& val);
	template<typename T> T Cubic(const T& val);
	template<typename T> D1<T> sqrt(const D1<T>& a);
	double sinc(const double in);
	double sinc_no_pi(const double in);
	template<typename T> D1<T> sinc(const D1<T>& in);
	template<typename T> double round(const T& a);
	double Frac(double x);
	template<typename T> T Mod(T x, T y);
	template<typename T> CPLX<T> exp(const T& phs);
	template<typename T> CPLX<T> exp(const CPLX<T>& in);
	template<typename T> void Swap(T& a,T& b);
	template<typename T> void FFT(D1<CPLX<T> >& in_out);
	template<typename T> void FFT(D2<CPLX<T> >& in_out);
	template<typename T> void IFFT(D1<CPLX<T> >& in_out);
	template<typename T> void IFFT(D2<CPLX<T> >& in_out);
	template<typename T> void FFT(CPLX<T>* in_out, const long N);
	template<typename T> void IFFT(CPLX<T>* in_out, const long N);
	template<typename T> void FFTY(D2<CPLX<T> >& in_out);
	template<typename T> void IFFTY(D2<CPLX<T> >& in_out);
	template<typename T> void FFTX(D2<CPLX<T> >& in_out);
	template<typename T> void IFFTX(D2<CPLX<T> >& in_out);
	bool IsPower2(long val);
	namespace fit {
		void FuncPoly(const double x, D1<double>& p);
		void FuncHyperbolic(const double x, D1<double>& p);
		D1<double> POLY(const D1<double>& x, const D1<double>& y,long degree);
		D1<double> POLY(const D1<double>& x, const D1<double>& y,long degree, double& chisq_error);
		D1<double> HYPERBOLIC(const D1<double>& x, const D1<double>& y);
		D1<double> HYPERBOLIC(const D1<double>& x, const D1<double>& y, double& chisq_error);
		D1<double> CORE(const D1<double>& x, const D1<double>& y, size_t NPoly,
						void funcs(const double, D1<double>& ), double& chisq_error);
	};
	template<typename T> D1<CPLX<T> > conv1(const D1<CPLX<T> >& array, const D1<CPLX<T> >& kernel, 
											const char* type);
	template<typename T> D1<T> conv1(const D1<T>& array, const D1<T>& kernel, const char* type);
	template<typename T> void shift(D1<T>& in_out, const long num);
	template<typename T> void shift(T* in_out, const long N, const long num);
	template<typename T> void fftshift(D1<T>& in_out);
	template<typename T> void fftshiftx(D2<CPLX<T> >& in_out);
	template<typename T> void fftshifty(D2<CPLX<T> >& in_out);
	template<typename T> void fftshift(D2<CPLX<T> >& in_out);
	template<typename T> void Flip(D1<T>& in_out);
	template<typename T> void Flip(T* in_out, const long N);
	template<typename T> void FlipLR(D2<CPLX<T> >& in_out);
	template<typename T> void FlipUD(D2<CPLX<T> >& in_out);
	template<typename T> T Randomu();
	template<typename T> D1<T> Randomu(const long num);
	template<typename T> T Randomu(unsigned int seed);
	template<typename T> D1<T> Randomu(const long num, unsigned int seed);
	template<typename T> T Randomu(const T min, const T max);
	template<typename T> T Randomu(const T min, const T max, unsigned int seed);
	D1<double> bessi0(const D1<double>& x);
	double factrl(const int n);
	template<typename T> D1<T> PolyInt1(const D1<T>& x, const D1<T>& y, const D1<T>& x_out);
	template<typename T> D1<T> SplInt1(const D1<T>& x, const D1<T>& y, const D1<T>& x_out);
	template<typename T> void PolyInt2(const D2<T>& in, const D1<T>& x1, const D1<T>& x2, const D1<T>& xout1, const D1<T>& xout2, D2<T>& out);
	template<typename T1, typename T2> void PolyInt2(const D2<CPLX<T1> >& in, const D1<T2>& x1, const D1<T2>& x2, const D1<T2>& xout1, const D1<T2>& xout2, D2<CPLX<T1> >& out);
	template<typename T> D2<T> Interp2(const D2<T>& in, const size_t M, const size_t N);
	template<typename T> D1<T> Indgen(const long sz);
	template<typename T> D1<long> VALUE_LOCATE(const D1<T>& in, const D1<T>& match);
//	void INTERPOL(const D1<double>& v, const D1<double>& x, const D1<double>& XOUT, D1<double>& vout);
	template<typename T> void INTERPOL(const D1<T>& v, const D1<T>& x, const D1<T>& XOUT, D1<T>& vout);
	template<typename T1, typename T2> void INTERPOL(const D1<CPLX<T1> >& v, const D1<T2>& x, const D1<T2>& XOUT, D1<CPLX<T1> >& vout);

	// Add Numerical Recipies(NR) subrutine
	void four1(double* data, unsigned long nn, int isign);
	void lfit(const D1<double>& x, const D1<double>& y, const D1<double>& sig,	// input
			  D1<double>& a, D1<bool>& ia, D2<double>& covar, double &chisq,	// output
			  void funcs(const double, D1<double>& ));							// output
	void gaussj(D2<double>& a, D2<double>& b);
	void covsrt(D2<double>& covar, const D1<bool>& ia, const long mfit);
	double bessi0(double x);
	double bessi(int n, double x);
	void polint(const D1<double>& xa, const D1<double>& ya, const double x, double& y, double& dy);
	void spline(const D1<double>& x, const D1<double>& y, const double yp1, const double ypn, D1<double>& y2);
	void splint(const D1<double>& xa, const D1<double>& ya, const D1<double>& y2a, const double x, double& y);
};


// =====================================================================================
// Basic or Modified functions implements
// =====================================================================================
template<typename T>
T mat::max(const T& a,const T& b){
	/*
	 Find max value
	 */
	return (a > b)? a:b;
}

template<typename T>
T mat::min(const T& a,const T& b){
	/*
	 Find min value
	 */
	return (a < b)? a:b;
}

template<typename T>
T mat::max(const T* a,const long num){
	/*
	 Find max value in a series
	 */
	long index=0;
	for(long i=1;i<num;++i){
		//out=(a[i]>out)? a[i]:out;
		if(a[i]>a[index]){ index=i; }
	}
	return a[index];
}

template<typename T>
T mat::min(const T* a,const long num){
	/*
	 Find min value in a series
	 */
	long index=0;
	for(long i=1;i<num;++i){
		//out=(a[i]<out)? a[i]:out;
		if(a[i]<a[index]){ index=i; }
	}
	return a[index];
}

template<typename T>
T mat::max(const T* a,const long num,long& index){
	/*
	 Find max value in a series (and return index)
	 */
	index=0;
	for(long i=1;i<num;++i){
		if(a[i]>a[index]){
			index=i;
		}
	}
	return a[index];
}

template<typename T>
T mat::min(const T* a,const long num,long& index){
	/*
	 Find min value in a series (and return index)
	 */
	index=0;
	for(long i=1;i<num-1;++i){
		if(a[i]<a[index]){
			index=i;
		}
	}
	return a[index];
}

template<typename T>
T mat::max(const D1<T>& a){
    /*
     Find max value from D1<T> series
	 */
	long num=a.GetNum();
	long index=0;
	for(long i=1;i<num;++i){
		if(a[i]>a[index]){
			index=i;
		}
	}
	return a[index];
}

template<typename T>
T mat::min(const D1<T>& a){
    /*
     Find min value from D1<T> series
	 */
	long num=a.GetNum();
	long index=0;
	for(long i=1;i<num;++i){
		if(a[i]<a[index]){
			index=i;
		}
	}
	return a[index];
}

template<typename T>
T mat::max(const D1<T>& a,long& index){
    /*
     Find max value from D1<T> series & index
	 */
	long num=a.GetNum();
	index=0;
	for(long i=1;i<num;++i){
		if(a[i]>a[index]){
			index=i;
		}
	}
	return a[index];
}

template<typename T>
T mat::min(const D1<T>& a,long& index){
    /*
     Find min value from D1<T> series & index
	 */
	long num=a.GetNum();
	index=0;
	for(long i=1;i<num;++i){
		if(a[i]<a[index]){
			index=i;
		}
	}
	return a[index];
}

template<typename T>
T mat::max(const D2<T>& a){
	long M = a.GetM();
	long N = a.GetN();
	T val=a[0][0];
	for(long i=0;i<N;++i){
		for(long j=0;j<M;++j){
			if(a[j][i]>val){
				val = a[j][i];
			}
		}
	}
	return val;
}

template<typename T>
T mat::min(const D2<T>& a){
	long M = a.GetM();
	long N = a.GetN();
	T val=a[0][0];
	for(long i=0;i<N;++i){
		for(long j=0;j<M;++j){
			if(a[j][i]<val){
				val = a[j][i];
			}
		}
	}
	return val;
}

template<typename T>
T mat::max(const D3<T>& a){
	long P = a.GetP();
	long Q = a.GetQ();
	long R = a.GetR();
	T val=a[0][0][0];
	for(long i=0;i<P;++i){
		for(long j=0;j<Q;++j){
			for(long k=0;k<R;++k){
				if(a[i][j][k]>val){
					val = a[i][j][k];
				}
			}
		}
	}
	return val;
}

template<typename T>
T mat::min(const D3<T>& a){
	long P = a.GetP();
	long Q = a.GetQ();
	long R = a.GetR();
	T val=a[0][0][0];
	for(long i=0;i<P;++i){
		for(long j=0;j<Q;++j){
			for(long k=0;k<R;++k){
				if(a[i][j][k]<val){
					val = a[i][j][k];
				}
			}
		}
	}
	return val;
}

template<typename T>
T mat::max(const vector<T>& a){
	return *(std::max_element(a.begin(), a.end()));
}

template<typename T>
T mat::min(const vector<T>& a){
	return *(std::min_element(a.begin(), a.end()));
}

template<typename T>
T mat::max3(const T a1, const T a2, const T a3){
	T out = a1;
	if(a2 > out){ out = a2; }
	if(a3 > out){ out = a3; }
	return out;
}

template<typename T>
T mat::min3(const T a1, const T a2, const T a3){
	T out = a1;
	if(a2 < out){ out = a2; }
	if(a3 < out){ out = a3; }
	return out;
}

/**
 * Set all elements will be zero
 * @param[in][out]   in   A variable
 */
template<typename T>
void mat::SetZero(vector<vector<T> >& in){
	//	memset(&(in[0]), 0, in.size()*in[0].size()*sizeof(T));
	for(size_t i=0;i<in.size();++i){
		for(size_t j=0;j<in[0].size();++j){
			memset(&(in[i][j]), 0, sizeof(T));
		}
	}
}

/**
 * Find the maximum absoluted value in vector<vector<CPLX<T>>>.
 * @param[in]   in  A vector<vector<CPLX<T>>> variable
 * @return Return a maximum absoluted value
 */
template<typename T>
T mat::max(const vector<vector<CPLX<T> > >& in){
	T max_val = -1E+15;
	for(size_t i=0;i<in.size();++i){
		for(size_t j=0;j<in[0].size();++j){
			max_val = (in[i][j].abs() > max_val)? in[i][j].abs():max_val;
		}
	}
	return max_val;
}

template<typename T>
T mat::max(const vector<vector<CPLX<T> > >& in, size_t& ii, size_t& jj){
	T max_val = -1E+15;
	for(size_t i=0;i<in.size();++i){
		for(size_t j=0;j<in[0].size();++j){
//			max_val = (in[i][j].abs() > max_val)? in[i][j].abs():max_val;
			if(in[i][j].abs() > max_val){
				max_val = in[i][j].abs();
				ii = i;
				jj = j;
			}
		}
	}
	return max_val;
}

template<typename T>
T mat::maxAbs(const vector<vector<D2<CPLX<T> > > >& in){
	// Check ONLY at index of Level = 0 (total part)
	//
	// RCS[pol][theta_look][Level][phi_asp][Nr]
	//     0/1       i        0       j     k
	//
	T out = -999999.999;

//	D1<float> v(in.size());
	for(unsigned long i=0;i<in.size();++i){		// theta_look
		for(long j=0;j<in[0][0].GetM();++j){		// phi_asp
			for(long k=0;k<in[0][0].GetN();++k){	// Nr
				float tmp = in[i][0][j][k].abs();
				if(tmp > out){
					out = tmp;
				}
			}
		}
	}

	return out;
}

template<typename T>
T mat::maxAbs(const D3<CPLX<T> >& in){
	// Echo_H[#level][#ang][#freq] == in
	T out = -999999.999;

	for(size_t p=0;p<in.GetP();++p){
		for(size_t q=0;q<in.GetQ();++q){
			for(size_t r=0;r<in.GetR();++r){
				float tmp = in[p][q][r].abs();
				if(tmp > out){
					out = tmp;
				}
			}
		}
	}

	return out;
}


template<typename T>
T mat::AvgStep(const vector<T>& a){
	return (max(a) - min(a))/a.size();
}

template<typename T>
T mat::max(const D2<T>& a,long& idx_M,long& idx_N){
	long M = a.GetM();
	long N = a.GetN();
	T val=a[0][0];
	idx_M = 0;
	idx_N = 0;
	for(long i=0;i<N;++i){
		for(long j=0;j<M;++j){
			if(a[j][i]>val){
				val = a[j][i];
				idx_M = j;
				idx_N = i;
			}
		}
	}
	return val;
}

template<typename T>
T mat::min(const D2<T>& a,long& idx_M,long& idx_N){
	long M = a.GetM();
	long N = a.GetN();
	T val=a[0][0];
	idx_M = 0;
	idx_N = 0;
	for(long i=0;i<N;++i){
		for(long j=0;j<M;++j){
			if(a[j][i]<val){
				val = a[j][i];
				idx_M = j;
				idx_N = i;
			}
		}
	}
	return val;
}

// Component-wise min
template<typename T>
VEC<T> mat::min(const VEC<T>& a, const VEC<T>& b) {
	return VEC<T>( min(a.x(),b.x()), min(a.y(),b.y()), min(a.z(),b.z()) );
}

// Component-wise max
template<typename T>
VEC<T> mat::max(const VEC<T>& a, const VEC<T>& b) {
	return VEC<T>( max(a.x(),b.x()), max(a.y(),b.y()), max(a.z(),b.z()) );
}

template<typename T>
D1<T> mat::abs(const D1<T>& a){
	D1<T> out = a;
	long num = a.GetNum();
	for(long i=0;i<num;++i){
		out[i] = std::abs(a[i]);
	}
	return out;
}

template<typename T>
D2<T> mat::abs(const D2<T>& a){
	D2<T> out = a;
	long M = a.GetM();
	long N = a.GetN();
	for(long j=0;j<M;++j){
		for(long i=0;i<N;++i){
			out[j][i] = std::abs(a[j][i]);
		}
	}
	return out;
}

template<typename T>
D3<T> mat::abs(const D3<T>& a){
	D3<T> out = a;
	long P = a.GetP();
	long Q = a.GetQ();
	long R = a.GetR();
	
	for(long k=0;k<P;++k){
		for(long j=0;j<Q;++j){
			for(long i=0;i<R;++i){
				out[k][j][i] = std::abs(a[k][j][i]);
			}
		}
	}
	return out;
}

template<typename T>
T mat::total(const D1<T>& a){
	long num = a.GetNum();
	T val = 0;
	for(long i=0;i<num;++i){
		val += a[i];
	}
	return val;
}

template<typename T>
T mat::total(const D2<T>& a){
	long num = a.GetM() * a.GetN();
	T val = 0;
	for(long i=0;i<num;++i){
		val += *(a.GetPtr() + i);
	}
	return val;
}

template<typename T>
CPLX<T> mat::totalNaN(const D1<CPLX<T> >& a){
	long num = a.GetNum();
	CPLX<T> val = 0;
	for(long i=0;i<num;++i){
		if(!std::isnan(a[i].r()) && !std::isnan(a[i].i())){
			val += a[i];
		}
	}
	return val;
}

template<typename T>
void mat::Scale(D1<T>& in, const T min_val, const T max_val){
	T min_in = mat::min(in);
	T max_in = mat::max(in);
	T sub = max_in-min_in;
	for(long i=0;i<in.GetNum();++i){
		in[i] = (max_val-min_val)/sub * (in[i]-min_in) + min_val;
	}
}

template<typename T>
void mat::Scale(D2<T>& in, const T min_val, const T max_val){
	T min_in = mat::min(in);
	T max_in = mat::max(in);
	T sub = max_in-min_in;
	for(long j=0;j<in.GetM();++j){
		for(long i=0;i<in.GetN();++i){
			in[j][i] = (max_val-min_val)/sub * (in[j][i]-min_in) + min_val;
		}
	}
}


template<typename T>
T mat::mean(const D1<T>& a){
	return total(a)/a.GetNum();
}

template<typename T>
T mat::mean(const D2<T>& a){
	return total(a)/(a.GetM() * a.GetN());
}

template<typename T>
D1<T> mat::abs(const D1<CPLX<T> >& in){
	long n = in.GetNum();
	D1<T> out(n);
	for(long i=0;i<n;++i){
		out[i] = in[i].abs();
	}
	return out;
}

template<typename T>
D2<T> mat::abs(const D2<CPLX<T> >& in){
	long m = in.GetM();
	long n = in.GetN();
	D2<T> out(m,n);
	for(long j=0;j<m;++j){
		for(long i=0;i<n;++i){
			out[j][i] = in[j][i].abs();
		}
	}
	return out;
}

template<typename T>
D3<T> mat::abs(const D3<CPLX<T> >& in){
	long p = in.GetP();
	long q = in.GetQ();
	long r = in.GetR();
	D3<T> out(p,q,r);
	for(long i=0;i<p;++i){
		for(long j=0;j<q;++j){
			for(long k=0;k<r;++k){
				out[i][j][k] = in[i][j][k].abs();
			}
		}
	}
	return out;
}

template<typename T>
CPLX<T> mat::Sin(const CPLX<T>& a){
//	CPLX<T> jj(0,1);
//	return (mat::exp(jj*a) - mat::exp(-jj*a)) / CPLX<T>(0.,2.);
	complex<T> tp = std::sin( complex<T>(a.r(), a.i()) );
	return CPLX<T>(tp.real(), tp.imag());
}

template<typename T>
CPLX<T> mat::Cos(const CPLX<T>& a){
//	CPLX<T> jj(0,1);
//	return (mat::exp(jj*a) + mat::exp(-jj*a)) / CPLX<T>(2.,0.);
	complex<T> tp = std::cos( complex<T>(a.r(), a.i()) );
	return CPLX<T>(tp.real(), tp.imag());
}

template<typename T>
CPLX<T> mat::Csc(const CPLX<T>& a){
	return 1.0/mat::Sin(a);
}

template<typename T>
T mat::Sec(const T rad){
	return 1.0/std::cos(rad);
}

template<typename T>
T mat::Cot(const T rad){
	return 1.0/tan(rad);
}

template<typename T>
CPLX<T> mat::Log(const CPLX<T>& in){
	return std::log(in.abs()) + CPLX<T>(0,1) * atan2(in.i(), in.r());
}

template<typename T>
CPLX<T> mat::Sqrt(const CPLX<T>& in){
	return in.sqrt();
}

template<typename T>
VEC<T> mat::abs(const VEC<CPLX<T> >& in){
	VEC<T> out;
	CPLX<T> tmp;
	tmp = in.x(); out.x() = tmp.abs();
	tmp = in.y(); out.y() = tmp.abs();
	tmp = in.z(); out.z() = tmp.abs();
	return out;
}


template<typename T>
D2<T> mat::phs(const D2<CPLX<T> >& in){
	long m = in.GetM();
	long n = in.GetN();

	D2<T> out(m,n);
	for(long j=0;j<m;++j){
		for(long i=0;i<n;++i){
			out[j][i] = in[j][i].phase();
		}
	}
	return out;
}

// =====================================================================================
// Basic or Modified functions
// =====================================================================================
template<typename T>
T mat::Square(const T& val){
	int type_num = def_func::GetTypeNumber(val);
	if(type_num < 200){
		return T(val*val);
	}else{
		def_func::errormsg("ERROR::[mat::Square] The input type is illegal!");
	}
	return 1;
}

template<typename T>
T mat::Cubic(const T& val){
	int type_num = def_func::GetTypeNumber(val);
	if(type_num < 200){
		return T(val*val*val);
	}else{
		def_func::errormsg("ERROR::[mat::Cubic] The input type is illegal!");
	}
	return 1;
}

template<typename T>
D1<T> mat::sqrt(const D1<T>& a){
	long num = a.GetNum();
	D1<T> out(num);
	for(long i=0;i<num;++i){
		out[i] = std::sqrt(a[i]);
	}
	return out;
}

double mat::sinc(const double in){
	// Sinc function
	double tmp = def::PI*(double)in;
	//if((tmp == 0.) || (tmp < 1E-15)){
	if(tmp == 0.){
		return 1;
	}else{
		return _sin(tmp)/(tmp);
	}
}

double mat::sinc_no_pi(const double in){
	// Sinc function
	double tmp = (double)in;
	//if((tmp == 0.) || (tmp < 1E-15)){
	if(tmp == 0.){
		return 1;
	}else{
		return _sin(tmp)/(tmp);
	}
}

template<typename T>
D1<T> mat::sinc(const D1<T>& in){
	// Sinc function
	double tmp;
	long num = in.GetNum();
	D1<T> out(num);
	for(long i=0;i<num;++i){
		tmp = def::PI*(double)in[i];
		if(tmp == 0.){
			out[i] = 1;
		}else{
			out[i] = _sin(tmp)/(tmp);
		}
	}
	return out;
}

template<typename T>
double mat::round(const T& a){
	return floor((double)a+0.5);
}

double mat::Frac(double x){
	// Fractional part of a number (y=x-[x])
	return x-floor(x);
}

template<typename T>
T mat::Mod(T x, T y){
	// x mod y
////	return T( mat::Frac((double)x/(double)y) * (double)y );
//	return T(round((double)y*Frac((double)x/(double)y)));
	return T(fmod((double)x, (double)y));
}

template<typename T>
CPLX<T> mat::exp(const T& phs){
	double cp = cos(phs);
	double sp = sin(phs);
//	double cp, sp;
//	opt::_sincos(phs, sp, cp);
	return CPLX<T>(cp, sp);
}

template<typename T>
CPLX<T> mat::exp(const CPLX<T>& in){
	if(in.r() == 0){
		double sin1,cos1;
		opt::_sincos(in.i(), sin1, cos1);
		return CPLX<T>(cos1,sin1);
//		return CPLX<T>(cos(in.i()),sin(in.i()));
	}else{
		return CPLX<T>( std::exp(in.r())*cos(in.i()),
					    std::exp(in.r())*sin(in.i()) );
	}
}

template<typename T>
void mat::Swap(T& a,T& b){
	// Swap
	T c; c=a; a=b; b=c;
}

template<typename T>
void mat::FFT(D1<CPLX<T> >& in_out){
	long N = in_out.GetNum();
	opt::_fft(int(N), (T*)(in_out.GetPtr()));
	for(long i=0;i<N;++i){
		in_out[i] = in_out[i]/N;
	}
}

template<typename T>
void mat::FFT(D2<CPLX<T> >& in_out){
	// Range FFT
	mat::FFTX(in_out);
	// Azimuth FFT
	mat::FFTY(in_out);
}

template<typename T>
void mat::IFFT(D1<CPLX<T> >& in_out){
	long N = in_out.GetNum();
	// hit : iFFT(x) = 1/N * conj(FFT(conj(x)). 
	// conjugate for every values
	for(long i=0;i<N;++i){ in_out[i].SelfConj(); }
	// FFT
	mat::FFT(in_out);
	// conjugate for every values again & sacling
	for(long i=0;i<N;++i){ in_out[i] = in_out[i].conj()*N; }
}

template<typename T>
void mat::IFFT(D2<CPLX<T> >& in_out){
	// Range FFT
	mat::IFFTX(in_out);
	// Azimuth FFT
	mat::IFFTY(in_out);
}

template<typename T>
void mat::FFT(CPLX<T>* in_out, const long N){
	opt::_fft(int(N), (T*)in_out);
	for(long i=0;i<N;++i){
		in_out[i] = in_out[i]/N;
	}
}

template<typename T>
void mat::IFFT(CPLX<T>* in_out, const long N){
	// hit : iFFT(x) = 1/N * conj(FFT(conj(x)). 
	// conjugate for every values
	for(long i=0;i<N;++i){ in_out[i].SelfConj(); }
	// FFT
	mat::FFT(in_out, N);
	// conjugate for every values again & sacling
	for(long i=0;i<N;++i){ in_out[i] = in_out[i].conj()*N; }
}

template<typename T>
void mat::FFTY(D2<CPLX<T> >& in_out){
	/*
			Azimuth FFT
		+----------------+
		| --> Memory --> | |
		+----------------+ |
		|                | | Azimuth (Y,M)
		+----------------+ |
		|                | |
		+----------------+ v
		----------------->
			Range (X,N)
	 */
	D1<CPLX<T> > tmp(in_out.GetM());
	for(size_t i=0;i<in_out.GetN();++i){
		// copy column
		tmp = in_out.GetColumn(i);
		// FFT
		mat::FFT(tmp);
		// assign column
		in_out.SetColumn(tmp, i);
	}
}

template<typename T>
void mat::IFFTY(D2<CPLX<T> >& in_out){
	/*
			Azimuth IFFT
		+----------------+
		| --> Memory --> | |
		+----------------+ |
		|                | | Azimuth (Y,M)
		+----------------+ |
		|                | |
		+----------------+ v
		----------------->
			Range (X,N)
	 */
	D1<CPLX<T> > tmp(in_out.GetM());
	for(size_t i=0;i<in_out.GetN();++i){
		// copy column
		tmp = in_out.GetColumn(i);
		// FFT
		mat::IFFT(tmp);
		// assign column
		in_out.SetColumn(tmp, i);
	}
}

template<typename T>
void mat::FFTX(D2<CPLX<T> >& in_out){
	/*
			Range FFT
	 	+----------------+
		| --> Memory --> | |
	 	+----------------+ |
	 	|                | | Azimuth (Y,M)
	 	+----------------+ |
	 	|                | |
	 	+----------------+ v
		----------------->
			Range (X,N)
	 */
	long N = in_out.GetN();
	for(long j=0;j<in_out.GetM();++j){
		// FFT
		opt::_fft(int(N), (T*)(in_out.GetPtr()+N*j));
		// Normalize
		for(long i=0;i<N;++i){
			in_out[j][i] /= (T)N;
		}
	}
}

template<typename T>
void mat::IFFTX(D2<CPLX<T> >& in_out){
	/*
			Range IFFT
		+----------------+
		| --> Memory --> | |
		+----------------+ |
		|                | | Azimuth (Y,M)
		+----------------+ |
		|                | |
		+----------------+ v
		----------------->
			Range (X,N)
	 */
	D1<CPLX<T> > tmp(in_out.GetN());
	for(long j=0;j<in_out.GetM();++j){
		// copy row
		tmp = in_out.GetRow(j);
		// FFT
		mat::IFFT(tmp);
		// assign row
		in_out.SetRow(tmp, j);
	}
}

bool mat::IsPower2(long val){
	double tmp = (double)val;
	double frac = Mod(tmp, 2.);
	long fac = long((tmp-frac)/2L);
	while (frac == 0.) {
		if (fac == 1) { return true; }
		tmp = tmp/2.;
		frac = Mod(tmp, 2.);
		fac = long((tmp-frac)/2L);
	}
	return false;
}

void mat::fit::FuncPoly(const double x, D1<double>& p){
	// Function to evaluate Nth degrees polynomial
    p[0] = 1.0;
	for (size_t j = 1; j < p.GetNum(); j++){
        p[j] = p[j - 1] * x;
	}
}

void mat::fit::FuncHyperbolic(const double x, D1<double>& p){
	// Function to evaluate hyperbolic function
	if(p.GetNum() == 2){
		p[0] = 1.0;
		p[1] = x*x;
	}else{
		def_func::errormsg("Error");
	}
}

D1<double> mat::fit::POLY(const D1<double>& x, const D1<double>& y,long degree){
	long NPoly = degree + 1;
	double chisq_error;
	return mat::fit::CORE(x, y, NPoly, mat::fit::FuncPoly, chisq_error);
}

D1<double> mat::fit::POLY(const D1<double>& x, const D1<double>& y,long degree, //input
						  double& chisq_error){
	long NPoly = degree + 1;
	return mat::fit::CORE(x, y, NPoly, mat::fit::FuncPoly, chisq_error);
}

D1<double> mat::fit::HYPERBOLIC(const D1<double>& x, const D1<double>& y){
	double chisq_error;
	return mat::fit::CORE(x, y, 2, mat::fit::FuncHyperbolic, chisq_error);
}

D1<double> mat::fit::HYPERBOLIC(const D1<double>& x, const D1<double>& y, double& chisq_error){
	return mat::fit::CORE(x, y, 2, mat::fit::FuncHyperbolic, chisq_error);
}

D1<double> mat::fit::CORE(const D1<double>& x, const D1<double>& y, size_t NPoly, 
						  void funcs(const double, D1<double>& ), double& chisq_error){

	D1<double> sig(x.GetNum());
	D1<double> coeff(NPoly);
	D1<bool> IsCoeff(NPoly);
	for(size_t i=0;i<x.GetNum();++i){ sig[i] = 0.00002; }
	for(size_t i=0;i<NPoly;++i){ IsCoeff[i] = true; }
	
	D2<double> covar(NPoly,NPoly);
	double chisq;
	
	mat::lfit(x,y,sig,coeff,IsCoeff,covar,chisq,funcs);
	return coeff;
}


template<typename T>
D1<CPLX<T> > mat::conv1(const D1<CPLX<T> >& array, const D1<CPLX<T> >& kernel, const char* type){
	// Convolution with zero padding
	long sz1 = array.GetNum();
	long sz2 = kernel.GetNum();
	long N_tp = sz1 + sz2 - 1L;
	//D1<T> out(N_tp);
	double N = -999.;
	D1<double> frac_2(20);
	for(int i=0;i<20;++i){ frac_2[i] = pow(2., double(i)); }
	
	for(int i=1;i<20;++i){
		if( (N_tp > frac_2[i-1]) && (N_tp < frac_2[i]) ){ N = frac_2[i]; }
	}
	
	// Error return
	if(N == -999.){
		//for(int i;i<N_tp;++i){ out[i] = T(-999); }
		cout<<"ERROR::[mat::conv1]Power of 2 is out of range! -> ";
		cout<<"N_tp="<<N_tp<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	
	// Zero padding
	D1<CPLX<double> > pad_a(N);
	D1<CPLX<double> > pad_k(N);
	for(long i=0;i<sz1;++i){
		if(i < sz1){
			pad_a[i] = CPLX<double>(array[i]);
		}else{
			pad_a[i] = CPLX<double>(0,0);
		}
	}
	for(long i=0;i<sz2;++i){
		if(i < sz2){
			pad_k[i] = CPLX<double>(kernel[i]);
		}else{
			pad_k[i] = CPLX<double>(0,0);
		}
	}
	
	// without normalize
	mat::FFT(pad_a);
	mat::FFT(pad_k);
	D1<CPLX<double> > res(N);
	res = pad_a * pad_k;
	for(long i=0;i<N;++i){ res[i] = res[i] * double(N); }
	mat::IFFT(res);
		
#ifdef _DEBUG
	for(long i=0;i<res.GetNum();++i){
		cout<<"["<<res[i].r()<<","<<res[i].i()<<"]"<<endl;
	}
#endif
	
	string _type(type);
	if( _type == string("same") ){
		long idx = ceil(double(sz2-1)/2.);
		D1<CPLX<T> > out(sz1);
		for(long i=0;i<sz1;++i){
			out[i] = CPLX<T>(res[idx+i]);
		}
		return out;
	}else if( _type == string("valid") ){
		if(sz2-1 > sz1-1){
			D1<CPLX<T> > out(1);
			out[0] = CPLX<T>(res[sz2-1]);
			return out;
		}else{
			D1<CPLX<T> > out(sz1-sz2+1);
			for(long i=0;i<out.GetNum();++i){
				out[i] = CPLX<T>(res[sz2-1+i]);
			}
			return out;
		}
	}else{
		D1<CPLX<T> > out(N_tp);
		for(long i=0;i<N_tp;++i){ out[i] = CPLX<T>(res[i]); }
		return out;
	}
}

template<typename T>
D1<T> mat::conv1(const D1<T>& array, const D1<T>& kernel, const char* type){
	// Convolution with zero padding
	long sz1 = array.GetNum();
	long sz2 = kernel.GetNum();
	long N_tp = sz1 + sz2 - 1L;
	//D1<T> out(N_tp);
	double N = -999.;
	D1<double> frac_2(20);
	for(int i=0;i<20;++i){ frac_2[i] = pow(2., double(i)); }
	
	for(int i=1;i<20;++i){
		if( (N_tp > frac_2[i-1]) && (N_tp < frac_2[i]) ){ N = frac_2[i]; }
	}
	
	// Error return
	if(N == -999.){
		//for(int i;i<N_tp;++i){ out[i] = T(-999); }
		cout<<"ERROR::[mat::conv1]Power of 2 is out of range! -> ";
		cout<<"N_tp="<<N_tp<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	
	// Zero padding
	D1<CPLX<double> > pad_a(N);
	D1<CPLX<double> > pad_k(N);
	for(long i=0;i<sz1;++i){
		if(i < sz1){
			pad_a[i] = CPLX<double>(array[i]);
		}else{
			pad_a[i] = CPLX<double>(0,0);
		}
	}
	for(long i=0;i<sz2;++i){
		if(i < sz2){
			pad_k[i] = CPLX<double>(kernel[i]);
		}else{
			pad_k[i] = CPLX<double>(0,0);
		}
	}
	
	// without normalize
	mat::FFT(pad_a);
	mat::FFT(pad_k);
	D1<CPLX<double> > res(N);
	res = pad_a * pad_k;
	for(long i=0;i<N;++i){ res[i] = res[i] * double(N); }
	mat::IFFT(res);
	
#ifdef _DEBUG
	for(long i=0;i<res.GetNum();++i){
		cout<<"["<<res[i].r()<<","<<res[i].i()<<"]"<<endl;
	}
#endif
	
	string _type(type);
	if( _type == string("same") ){
		long idx = ceil(double(sz2-1)/2.);
		D1<T> out(sz1);
		for(long i=0;i<sz1;++i){
			out[i] = T(res[idx+i].r());
		}
		return out;
	}else if( _type == string("valid") ){
		if(sz2-1 > sz1-1){
			D1<T> out(1);
			out[0] = T(res[sz2-1].r());
			return out;
		}else{
			D1<T> out(sz1-sz2+1);
			for(long i=0;i<out.GetNum();++i){
				out[i] = T(res[sz2-1+i].r());
			}
			return out;
		}
	}else{
		D1<T> out(N_tp);
		for(long i=0;i<N_tp;++i){ out[i] = T(res[i].r()); }
		return out;
	}
}

//template<typename T>
//void mat::shift(D1<T>& in_out, const long num){
//	long sig = def_func::sign(num);
//	long N = in_out.GetNum();
//	long num_shift = (sig < 0)? N+num : num;
//	
//	cout<<"num_shift = "<<num_shift<<endl;
//	cout<<"num, N    = "<<num<<", "<<N<<endl;
//	cout<<"frac      = "<<mat::Frac((double)num/(double)N) * (double)N<<endl;
////	num_shift = Mod(num, N);
//	num_shift = round(mat::Frac((double)num/(double)N) * (double)N);
////	num_shift = N + num_shift;
//	cout<<"num_shift = "<<num_shift<<endl;
//	
//	
//	D1<T> tmp(in_out.GetPtr() + (N - num_shift) ,num_shift);
//	memmove(in_out.GetPtr()+num_shift, in_out.GetPtr(), sizeof(T)*(N - num_shift));
//	memmove(in_out.GetPtr(), tmp.GetPtr(), sizeof(T)*num_shift);
//}
//
//template<typename T>
//void mat::shift(T* in_out, const long N, const long num){
//	long sig = def_func::sign(num);
//	long num_shift = (sig < 0)? N+num : num;
//	
//	num_shift = Mod(num, N);
//	
//	
//	D1<T> tmp(in_out + (N - num_shift) ,num_shift);
//	memmove(in_out+num_shift, in_out, sizeof(T)*(N - num_shift));
//	memmove(in_out, tmp.GetPtr(), sizeof(T)*num_shift);
//	
////	long sig = def_func::sign(num);
////	long num_shift = (sig < 0)? N+num : num;
////	if(std::abs(num) > N){ num_shift = mat::Mod(num_shift, N); }
////	
////	if( (num_shift == N/2) && (mat::Frac(double(N)/2.) == 0) ){
////		for(long i=0;i<N/2;++i){
////			def_prefunc::Swap(in_out[i], in_out[i+N/2]);
////		}
////	}else{
////		D1<T> tmp(in_out + (N - num_shift) ,num_shift);
////		memmove(in_out+num_shift, in_out, sizeof(T)*(N - num_shift));
////		memmove(in_out, tmp.GetPtr(), sizeof(T)*num_shift);
////	}
//}


template<typename T>
void mat::shift(D1<T>& in_out, const long num){
	long sig = def_func::sign(num);
	long N = in_out.GetNum();
	long num_shift = (sig < 0)? N+num : num;
	if(std::abs(num) > N){
		double tmp = mat::Mod((double)num_shift, (double)N);
		num_shift = round(tmp);
	}
	
	if( (num_shift == N/2) && (mat::Frac(double(N)/2.) == 0) ){
		for(long i=0;i<N/2;++i){
			def_prefunc::Swap(in_out[i], in_out[i+N/2]);
		}
	}else{
		D1<T> tmp(in_out.GetPtr() + (N - num_shift) ,num_shift);
		memmove(in_out.GetPtr()+num_shift, in_out.GetPtr(), sizeof(T)*(N - num_shift));
		memmove(in_out.GetPtr(), tmp.GetPtr(), sizeof(T)*num_shift);
	}
}

template<typename T>
void mat::shift(T* in_out, const long N, const long num){
	long sig = def_func::sign(num);
	long num_shift = (sig < 0)? N+num : num;
	if(std::abs(num) > N){ num_shift = mat::Mod(num_shift, N); }
	
	if( (num_shift == N/2) && (mat::Frac(double(N)/2.) == 0) ){
		for(long i=0;i<N/2;++i){
			def_prefunc::Swap(in_out[i], in_out[i+N/2]);
		}
	}else{
		D1<T> tmp(in_out + (N - num_shift) ,num_shift);
		memmove(in_out+num_shift, in_out, sizeof(T)*(N - num_shift));
		memmove(in_out, tmp.GetPtr(), sizeof(T)*num_shift);
	}
}

template<typename T>
void mat::fftshift(D1<T>& in_out){
	mat::shift(in_out, long(in_out.GetNum()/2));
}

template<typename T>
void mat::fftshiftx(D2<CPLX<T> >& in_out){
	/*
		  Range fftshift
		+----------------+
		| --> Memory --> | |
		+----------------+ |
		|                | | Azimuth (Y,j,M)
		+----------------+ |
		|                | |
		+----------------+ v
		----------------->
			Range (X,i,N)
	 */
	long M = in_out.GetM();	// azimuth
	long N = in_out.GetN();	// range
	
	for(size_t j=0;j<M;++j){
		mat::shift((in_out.GetPtr()+j*N), N, N/2);
	}
}

template<typename T>
void mat::fftshifty(D2<CPLX<T> >& in_out){
	/*
		 Azimuth fftshift
		+----------------+
		| --> Memory --> | |
		+----------------+ |
		|                | | Azimuth (Y,j,M)
		+----------------+ |
		|                | |
		+----------------+ v
		----------------->
			Range (X,i,N)
	 */
	long M = in_out.GetM();	// azimuth
	long N = in_out.GetN();	// range
	
	D1<CPLX<T> > tmp(M);
	for(size_t i=0;i<N;++i){
		// Get column
		tmp = in_out.GetColumn(i);
		// shift
		mat::fftshift(tmp);
		// Assign column
		in_out.SetColumn(tmp, i);
	}
}

template<typename T>
void mat::fftshift(D2<CPLX<T> >& in_out){
	// Range fftshift
	mat::fftshiftx(in_out);
	// Azimuth fftshift
	mat::fftshifty(in_out);
}

template<typename T>
void mat::Flip(D1<T>& in_out){
	in_out.Swap();
//	long N = in_out.GetNum();
//	for(size_t i=0;i<N/2;++i){
//		std::swap(in_out.GetPtr()+i, in_out.GetPtr()+N-1-i);
//	}
}

template<typename T>
void mat::Flip(T* in_out, const long N){
	for(size_t i=0;i<N/2;++i){
		std::swap(in_out[i], in_out[N-1-i]);
	}
}

template<typename T>
void mat::FlipLR(D2<CPLX<T> >& in_out){
	/*
		  Flip Left-Right
		+----------------+
		| --> Memory --> | |
		+----------------+ |
		|                | | Azimuth (Y,j,M)
		+----------------+ |
		|                | |
		+----------------+ v
		----------------->
		  Range (X,i,N)
	 */
	long M = in_out.GetM();	// azimuth
	long N = in_out.GetN();	// range
	
	for(size_t j=0;j<M;++j){
		mat::Flip(in_out.GetPtr()+N*j, N);
	}
}

template<typename T>
void mat::FlipUD(D2<CPLX<T> >& in_out){
	/*
	 Flip Left-Right
		+----------------+
		| --> Memory --> | |
		+----------------+ |
		|                | | Azimuth (Y,j,M)
		+----------------+ |
		|                | |
		+----------------+ v
		----------------->
	 Range (X,i,N)
	 */
	long M = in_out.GetM();	// azimuth
	long N = in_out.GetN();	// range
	
	D1<CPLX<T> > tmp(M);
	for(size_t i=0;i<N;++i){
		// Get Column
		tmp = in_out.GetColumn(i);
		// Flip
		tmp.Swap();
		// Assign Column
		in_out.SetColumn(tmp, i);
	}
}

template<typename T>
T mat::Randomu(){
	srand((unsigned int)time(NULL));
	srand(rand());
	srand(rand());
	return T(double(rand())/double(RAND_MAX));
}

template<typename T>
D1<T> mat::Randomu(const long num){
	D1<T> out(num);
	srand((unsigned int)time(NULL));
	srand(rand());
	srand(rand());
	for(long i=0;i<num;++i){
		out[i] = T(double(rand())/double(RAND_MAX));
	}
	return out;
}

template<typename T>
T mat::Randomu(unsigned int seed){
	srand(seed);
	return T(double(rand())/double(RAND_MAX));
}

template<typename T>
D1<T> mat::Randomu(const long num, unsigned int seed){
	D1<T> out(num);
	srand(seed);
	for(long i=0;i<num;++i){
		out[i] = T(double(rand())/double(RAND_MAX));
	}
	return out;
}

template<typename T>
T mat::Randomu(const T min, const T max){
	srand((unsigned int)time(NULL));
	srand(rand());
	srand(rand());
	return T(double(rand())*(max-min)/double(RAND_MAX) + min);
}

template<typename T>
T mat::Randomu(const T min, const T max, unsigned int seed){
	srand(seed);
	return T(double(rand())*(max-min)/double(RAND_MAX) + min);
}

D1<double> mat::bessi0(const D1<double>& x){
	D1<double> out(x.GetNum());
	for(size_t i=0;i<x.GetNum();++i){
		out[i] = mat::bessi0(x[i]);
	}
	return out;
}

double mat::factrl(const int n){
//	if(n==0){
//		return 1;
//	}else{
//		return (n*factrl(n-1));
//	}
	if(n<=20){
		return def::FACTORIAL[n];
	}else{
		return (n*factrl(n-1));
	}
//	if(n < 0 || n > 170){
//		cout<<"factrl out of range"<<endl;
//		return 0;
//	}
//	double a = 1;
//	for(int i=1;i<=n;i++){
//		a *= i;
//	}
//	return a;
}

template<typename T>
D1<T> mat::PolyInt1(const D1<T>& x, const D1<T>& y, const D1<T>& x_out){
	D1<T> y_out(x_out.GetNum());
	D1<double> xa = static_cast<D1<double> >(x);
	D1<double> ya = static_cast<D1<double> >(y);
	double yy,dy;
	for(long i=0;i<x_out.GetNum();++i){
		mat::polint(xa, ya, (double)x_out[i], yy, dy);
		y_out[i] = (T)yy;
	}
	return y_out;
}

template<typename T>
D1<T> mat::SplInt1(const D1<T>& x, const D1<T>& y, const D1<T>& x_out){
	D1<double> y2(x.GetNum());
	D1<double> xa = static_cast<D1<double> >(x);
	D1<double> ya = static_cast<D1<double> >(y);
	double yy;
	D1<T> y_out(x_out.GetNum());
	mat::spline(xa, ya, 9E30, 9E30, y2);
	for(long i=0;i<x_out.GetNum();++i){
		mat::splint(xa, ya, y2, (double)x_out[i], yy);
		y_out[i] = (T)yy;
	}
	return y_out;
}

template<typename T>
void mat::PolyInt2(const D2<T>& in, const D1<T>& x1, const D1<T>& x2, const D1<T>& xout1, const D1<T>& xout2, D2<T>& out){
	
	long m = in.GetM();
//	long n = in.GetN();
	
	long M = xout2.GetNum();
	long N = xout1.GetNum();
	
	out = D2<T>(M,N);
	
	D1<T> tmp_row(N);
	D1<T> tmp_col(M);
	D2<T> tmp(m, N);
	
	// Row interpolation
	for(size_t j=0;j<m;++j){
		INTERPOL(in.GetRow(j), x1, xout1, tmp_row);
		tmp.SetRow(tmp_row, j);
	}
	
	// Column interpolation
	for(size_t i=0;i<N;++i){
		INTERPOL(tmp.GetColumn(i), x2, xout2, tmp_col);
		out.SetColumn(tmp_col, i);
	}
}

template<typename T1, typename T2>
void mat::PolyInt2(const D2<CPLX<T1> >& in, const D1<T2>& x1, const D1<T2>& x2, const D1<T2>& xout1, const D1<T2>& xout2, D2<CPLX<T1> >& out){
	
	long m = in.GetM();
	//	long n = in.GetN();
	
	long M = xout2.GetNum();
	long N = xout1.GetNum();
	
	out = D2<CPLX<T1> >(M,N);
	
	D1<CPLX<T1> > tmp_row(N);
	D1<CPLX<T1> > tmp_col(M);
	D2<CPLX<T1> > tmp(m, N);
	
	// Row interpolation
	for(size_t j=0;j<m;++j){
		INTERPOL(in.GetRow(j), x1, xout1, tmp_row);
		tmp.SetRow(tmp_row, j);
	}
	
	// Column interpolation
	for(size_t i=0;i<N;++i){
		INTERPOL(tmp.GetColumn(i), x2, xout2, tmp_col);
		out.SetColumn(tmp_col, i);
	}
}

template<typename T>
D2<T> mat::Interp2(const D2<T>& in, const size_t M, const size_t N){

	
//	if(typeid(T) == typeid(CPLX<double>)){		// CPLX<double>
//		double Re
//	}else if(typeid(T) == typeid(CPLX<float>)){	// CPLX<float>
//		
//	}else{	// non CPLX<T>
		D2<T> out(M,N);
		
		long m = in.GetM();
		long n = in.GetN();
		
		D1<double> x1 = Indgen<double>(n);	// row index
		D1<double> x2 = Indgen<double>(m);	// col index
		
		D1<double> xout1(N);	// output row index
		D1<double> xout2(M);	// output col index
		def_func::linspace(0., (n-1.), xout1);
		def_func::linspace(0., (m-1.), xout2);
		
		// Interpolation
		PolyInt2(in, x1, x2, xout1, xout2, out);
		
		return out;
//	}
}

template<typename T>
D1<T> mat::Indgen(const long sz){
	D1<T> out(sz);
	for(long i=0;i<sz;++i){
		out[i] = (T)i;
	}
	return out;
}

template<typename T>
D1<long> mat::VALUE_LOCATE(const D1<T>& in, const D1<T>& match){
	D1<long> idx = Indgen<long>(in.GetNum());
	D1<long> out(match.GetNum());
	for(long i=0;i<match.GetNum();++i){
		if(match[i] < in[0]){
			out[i] = -1;
		}else if(match[i] >= in[in.GetNum()-1]){
//			out[i] = idx[idx.GetNum()-1];
			out[i] = idx.GetNum()-1;
		}else{
			for(long j=0;j<in.GetNum();++j){
				if(match[i] >= in[j] && match[i] <= in[j+1]){
//					T val1 = std::abs(match[i] - in[j]);
//					T val2 = std::abs(match[i] - in[j+1]);
//					out[i] = (val1 < val2)? idx[j]:idx[j+1];
					out[i] = idx[j];
				}
			}
		}
	}
	return out;
}

template<typename T>
void mat::INTERPOL(const D1<T>& v, const D1<T>& x, const D1<T>& XOUT, D1<T>& vout){
	vout = D1<T>(XOUT.GetNum());
	
	// Make a copy so we don't overwrite the input arguments.
//	D1<double> v = VV;
//	D1<double> x = XX;
	long m = v.GetNum();	// # of input pnts
	
	// Subscript intervals.
	D1<long> s = VALUE_LOCATE(x, XOUT);
	for(long i=0;i<s.GetNum();++i){
		if(s[i] < 0){
			s[i] = 0;
		}else if(s[i] > m-2){
			s[i] = m-2;
		}
	}
	
	// Linear, not regular
	D1<double> diff(s.GetNum());
	for(long i=0;i<s.GetNum();++i){
		diff[i] = v[s[i]+1] - v[s[i]];
	}
	
	for(long i=0;i<vout.GetNum();++i){
		vout[i] = (XOUT[i]-x[s[i]]) * diff[i] / (x[s[i]+1] - x[s[i]]) + v[s[i]];
	}
}

template<typename T1, typename T2>
void mat::INTERPOL(const D1<CPLX<T1> >& v, const D1<T2>& x, const D1<T2>& XOUT, D1<CPLX<T1> >& vout){
	vout = D1<CPLX<T1> >(XOUT.GetNum());
	
	// Make a copy so we don't overwrite the input arguments.
	//	D1<double> v = VV;
	//	D1<double> x = XX;
	long m = v.GetNum();	// # of input pnts
	
	// Subscript intervals.
	D1<long> s = VALUE_LOCATE(x, XOUT);
	for(long i=0;i<s.GetNum();++i){
		if(s[i] < 0){
			s[i] = 0;
		}else if(s[i] > m-2){
			s[i] = m-2;
		}
	}
	
	// Linear, not regular
	D1<CPLX<T1> > diff(s.GetNum());
	for(long i=0;i<s.GetNum();++i){
		diff[i] = v[s[i]+1] - v[s[i]];
	}
	
	for(long i=0;i<vout.GetNum();++i){
		vout[i] = (T1)((XOUT[i]-x[s[i]]) / (x[s[i]+1] - x[s[i]])) * diff[i] + v[s[i]];
	}
}




// =====================================================================================
// Numerical Recipes implements (from NR C++ 2nd)
// =====================================================================================

///*
void mat::four1(double* data, unsigned long nn, int isign){
	//
	// Purpose:
	// One-Dimension FFT
	// 
	// Description:
	// data -> any type array that represent the array of complex samples
	// in_data -> number of samples (N^2 order number) 
	// isign -> 1 to calculate FFT and -1 to calculate Reverse FFT
	//
	//variables for trigonometric recurrences
	unsigned long n,mmax,m,j,istep,i;
	double wtemp,wr,wpr,wpi,wi,theta;
	double tempr,tempi;
	
	//the complex array is real+complex so the array 
	//as a size n = 2* number of complex samples
	// real part is the data[index] and the complex part is the data[index+1]
	n=nn << 1;
	//binary inversion (note that 
	//the indexes start from 1 witch means that the
	//real part of the complex is on the odd-indexes
	//and the complex part is on the even-indexes
	j=1;
	for (i=1;i<n;i+=2) {
		if (j > i) {
			//swap the real part
			Swap(data[j],data[i]);
			//swap the complex part
			Swap(data[j+1],data[i+1]);
		}
		m=n >> 1;
		while (m >= 2 && j > m) {
			j -= m;
			m >>= 1;
		}
		j += m;
	}
	//Danielson-Lanzcos routine 
	mmax=2;
	//external loop
	while (n > mmax) {
		istep=mmax << 1;
		theta=isign*(6.28318530717959/mmax);
		wtemp=sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi=sin(theta);
		wr=1.0;
		wi=0.0;
		//internal loops
		for (m=1;m<mmax;m+=2) {
			for (i=m;i<=n;i+=istep) {
				j=i+mmax;
				tempr=wr*data[j]-wi*data[j+1];
				tempi=wr*data[j+1]+wi*data[j];
				data[j]=data[i]-tempr;
				data[j+1]=data[i+1]-tempi;
				data[i] += tempr;
				data[i+1] += tempi;
			}
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
		}
		mmax=istep;
	}
}
//*/

void mat::lfit(const D1<double>& x, const D1<double>& y, const D1<double>& sig,
			   D1<double>& a, D1<bool>& ia, D2<double>& covar, double &chisq,
			   void funcs(const double, D1<double>& )){
	long i,j,k,l,m,mfit=0;
	double ym,wt,sum,sig2i;
	
	long ndat=x.GetNum();
	long ma=a.GetNum();
	D1<double> afunc(ma);
	D2<double> beta(ma,1);
	for (j=0;j<ma;j++)
		if (ia[j]) mfit++;
	if (mfit == 0) def_func::errormsg("lfit: no parameters to be fitted");
	for (j=0;j<mfit;j++) {
		for (k=0;k<mfit;k++) covar[j][k]=0.0;
		beta[j][0]=0.0;
	}
	for (i=0;i<ndat;i++) {
		funcs(x[i],afunc);
		ym=y[i];
		if (mfit < ma) {
			for (j=0;j<ma;j++)
				if (!ia[j]) ym -= a[j]*afunc[j];
		}
		sig2i=1.0/Square(sig[i]);
		for (j=0,l=0;l<ma;l++) {
			if (ia[l]) {
				wt=afunc[l]*sig2i;
				for (k=0,m=0;m<=l;m++)
					if (ia[m]) covar[j][k++] += wt*afunc[m];
				beta[j++][0] += ym*wt;
			}
		}
	}
	for (j=1;j<mfit;j++)
		for (k=0;k<j;k++)
			covar[k][j]=covar[j][k];
	D2<double> temp(mfit,mfit);
	for (j=0;j<mfit;j++)
		for (k=0;k<mfit;k++)
			temp[j][k]=covar[j][k];
	gaussj(temp,beta);
	for (j=0;j<mfit;j++)
		for (k=0;k<mfit;k++)
			covar[j][k]=temp[j][k];
	for (j=0,l=0;l<ma;l++)
		if (ia[l]) a[l]=beta[j++][0];
	chisq=0.0;
	for (i=0;i<ndat;i++) {
		funcs(x[i],afunc);
		sum=0.0;
		for (j=0;j<ma;j++) sum += a[j]*afunc[j];
		chisq += Square((y[i]-sum)/sig[i]);
	}
	mat::covsrt(covar,ia,mfit);
}

void mat::gaussj(D2<double>& a, D2<double>& b){
	long icol = 0, irow = 0;
	long i,j,k,l,ll;
	double big,dum,pivinv;

	long n=a.GetM();
	long m=b.GetN();
	D1<long> indxc(n),indxr(n),ipiv(n);
	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) {
		big=0.0;
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) {
					if (ipiv[k] == 0) {
						if (fabs(a[j][k]) >= big) {
							big=fabs(a[j][k]);
							irow=j;
							icol=k;
						}
					}
				}
		++(ipiv[icol]);
		if (irow != icol) {
			for (l=0;l<n;l++) mat::Swap(a[irow][l],a[icol][l]);
			for (l=0;l<m;l++) mat::Swap(b[irow][l],b[icol][l]);
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if (a[icol][icol] == 0.0) def_func::errormsg("gaussj: Singular Matrix");
		pivinv=1.0/a[icol][icol];
		a[icol][icol]=1.0;
		for (l=0;l<n;l++) a[icol][l] *= pivinv;
		for (l=0;l<m;l++) b[icol][l] *= pivinv;
		for (ll=0;ll<n;ll++)
			if (ll != icol) {
				dum=a[ll][icol];
				a[ll][icol]=0.0;
				for (l=0;l<n;l++) a[ll][l] -= a[icol][l]*dum;
				for (l=0;l<m;l++) b[ll][l] -= b[icol][l]*dum;
			}
	}
	for (l=n-1;l>=0;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				mat::Swap(a[k][indxr[l]],a[k][indxc[l]]);
	}
}

void mat::covsrt(D2<double>& covar, const D1<bool>& ia, const long mfit){
	long i,j,k;

	long ma=ia.GetNum();
	for (i=mfit;i<ma;i++)
		for (j=0;j<i+1;j++) covar[i][j]=covar[j][i]=0.0;
	k=mfit-1;
	for (j=ma-1;j>=0;j--) {
		if (ia[j]) {
			for (i=0;i<ma;i++) mat::Swap(covar[i][k],covar[i][j]);
			for (i=0;i<ma;i++) mat::Swap(covar[k][i],covar[j][i]);
			k--;
		}
	}
}

double mat::bessi0(double x){
	double ax,ans;
	double y;
	
	if ((ax=std::abs(x)) < 3.75) {
		y=x/3.75;
		y*=y;
		ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
			+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
	}else{
		y=3.75/ax;
		ans=(std::exp(ax)/std::sqrt(ax))*(0.39894228+y*(0.1328592e-1
			 +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
			 +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
			 +y*0.392377e-2))))))));
	}
	return ans;
}

double mat::bessi(int n, double x){
	double ACC = 40.0;
	double BIGNO = 1.0e10;
	double BIGNI = 1.0e-10;
	int j;
	double bi,bim,bip,tox,ans;
	
	if (x == 0.0)
		return 0.0;
	else {
		tox=2.0/std::abs(x);
		bip=ans=0.0;
		bi=1.0;
		for (j=2*(n+(int) std::sqrt(ACC*n));j>0;j--) {
			bim=bip+j*tox*bi;
			bip=bi;
			bi=bim;
			if (fabs(bi) > BIGNO) {
				ans *= BIGNI;
				bi *= BIGNI;
				bip *= BIGNI;
			}
			if (j == n) ans=bip;
		}
		ans *= bessi0(x)/bi;
		return x < 0.0 && (n & 1) ? -ans : ans;
	}
}

void mat::polint(const D1<double>& xa, const D1<double>& ya, const double x, double& y, double& dy){
	int ns=0;
	double den,dif,dift,ho,hp,w;
	
	int n=(int)xa.GetNum();
	D1<double> c(n),d(n);
	dif=fabs(x-xa[0]);
	for(int i=0;i<n;i++) {
		if((dift=fabs(x-xa[i])) < dif){
			ns=i;
			dif=dift;
		}
		c[i]=ya[i];
		d[i]=ya[i];
	}
	y=ya[ns--];
	for (int m=1;m<n;m++) {
		for (int i=0;i<n-m;i++) {
			ho=xa[i]-x;
			hp=xa[i+m]-x;
			w=c[i+1]-d[i];
			if ((den=ho-hp) == 0.0) def_func::errormsg("Error in routine polint");
			den=w/den;
			d[i]=hp*den;
			c[i]=ho*den;
		}
		y += (dy=(2*(ns+1) < (n-m) ? c[ns+1] : d[ns--]));
	}
}

void mat::spline(const D1<double>& x, const D1<double>& y, const double yp1, const double ypn, D1<double>& y2){
	double p,qn,sig,un;
	
	int n=(int)y2.GetNum();
	D1<double> u(n-1);
	if(yp1 > 0.99e30){
		y2[0]=u[0]=0.0;
	}else{
		y2[0] = -0.5;
		u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
	}
	for(int i=1;i<n-1;i++) {
		sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
		u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
	}
	if(ypn > 0.99e30){
		qn=un=0.0;
	}else{
		qn=0.5;
		un=(3.0/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]));
	}
	y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0);
	for(int k=n-2;k>=0;k--){
		y2[k]=y2[k]*y2[k+1]+u[k];
	}
}

void mat::splint(const D1<double>& xa, const D1<double>& ya, const D1<double>& y2a, const double x, double& y){
	int k;
	double h,b,a;
	
	int n=(int)xa.GetNum();
	int klo=0;
	int khi=n-1;
	while (khi-klo > 1) {
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	if (h == 0.0) def_func::errormsg("Bad xa input to routine splint");
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
}


#endif // MAT_H_INCLUDED
