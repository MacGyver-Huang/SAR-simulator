#ifndef IERS_H_INCLUDED
#define IERS_H_INCLUDED

#include <basic/d1.h>


namespace iers{
	using namespace d1;
	typedef struct IERS_struc{ double d,dEps,dPsi,MJD,UT1_UT1R,UT1_UTC,x,y; }IERS_STRUC;
	// ==============================================
	// IERS
	// ==============================================
	class IERS
	{
		public:
			// Constructure
			IERS(long N);
			IERS(string file);
			// Copy Constructor
			//SV(const SV<T>& in);
			// operator overloading
			//IERS<T>& operator=(const SV<T>& in);
			//~SV();
/*
			// Operator overloading
			// Get Ref.
			D1<T>& t(){return _t;};
			D1VEC& pos(){return _pos;};
			D1VEC& vel(){return _vel;};
*/

			// Get Value
			IERS_STRUC GetIERSBullB(const double MJD_UTC)const;
			//const D1<T>& t()const{return _t;};
			//const D1VEC& pos()const{return _pos;};
			//const D1VEC& vel()const{return _vel;};
			//const long& GetNum()const{return _num;};
			// Set
			//void SetAll(const D1<T>& t,const D1<VEC<T> >& pos,const D1<VEC<T> >& vel,const long num);

			// Misc.
			void Print()const;
			//void Print(long i)const;
			//void WriteBinary(const char* filename);
			//void ReadBinary(const char* filename);
		private:
			void _init(const long N);
			long _num;
			D1<double> _d,_dEps,_dPsi,_MJD,_UT1_UT1R,_UT1_UTC,_x,_y;
	};


	//
	// Constructure
	//
	void IERS::_init(const long N){
	    _num=N;
		_d		 =D1<double>(N);
		_dEps	 =D1<double>(N);
		_dPsi	 =D1<double>(N);
		_MJD	 =D1<double>(N);
		_UT1_UT1R=D1<double>(N);
		_UT1_UTC =D1<double>(N);
		_x		 =D1<double>(N);
		_y		 =D1<double>(N);
	}

	IERS::IERS(const long N){
		_init(N);
	}

	IERS::IERS(string file){
	/*
	 Purpose:
		Read IERS binary data converted from IDL "test_Read_bulletin_b2.pro"
	 Input:
		file	:(string) IERS binary file name
	*/
		ifstream fin(file.c_str(),ios_base::in|ios_base::binary);
		// get vector size
		fin.seekg(0,ios_base::end);
		long N = long( fin.tellg()/(8*sizeof(double)) );
		fin.seekg(0,ios_base::beg);

		// initialize
		_init(N);
		
		// read binary
		double *buf=new double[8*N];
		fin.read(reinterpret_cast<char*>(buf),sizeof(double)*8*N);
		fin.close();

		for(int i=0;i<N;++i){
			_d[i]		= buf[i*8+0];
			_dEps[i]	= buf[i*8+1];
			_dPsi[i]	= buf[i*8+2];
			_MJD[i]		= buf[i*8+3];
			_UT1_UT1R[i]= buf[i*8+4];
			_UT1_UTC[i] = buf[i*8+5];
			_x[i]		= buf[i*8+6];
			_y[i]		= buf[i*8+7];
		}
		fin.close();
	}

	//
	// Get Value
	//
	IERS_STRUC IERS::GetIERSBullB(const double MJD_UTC)const{
	/*
	 Purpose:
		Get the IERS parameter specify the MJD of UTC
	 Intput:
		MJD_UTC	:(double) modified julian day of UTC time
	 Return:
		IERS_STRUC : (structure) IERS structure
	*/
		IERS_STRUC out;
		int i;
		for(i=0;i<_num-1;++i){
			if( (MJD_UTC >= _MJD[i])&&(MJD_UTC <= _MJD[i+1]) ){
				out.d = _d[i];
				out.dEps = _dEps[i];
				out.dPsi = _dPsi[i];
				out.MJD = _MJD[i];
				out.UT1_UT1R = _UT1_UT1R[i];
				out.UT1_UTC = _UT1_UTC[i];
				out.x = _x[i];
				out.y = _y[i];
				return out;
			}
		}
		i--;
		out.d = _d[i];
		out.dEps = _dEps[i];
		out.dPsi = _dPsi[i];
		out.MJD = _MJD[i];
		out.UT1_UT1R = _UT1_UT1R[i];
		out.UT1_UTC = _UT1_UTC[i];
		out.x = _x[i];
		out.y = _y[i];
		return out;
	}

	//
	// Misc.
	//
	void IERS::Print()const{
		long m=(_num >= 20)? 20:_num;
		cout<<"IERS:"<<endl;
		for(long i=0;i<m;++i){
			cout<<i<<": ";
			cout<<" ["<<_d[i]<<", "<<_dEps[i]<<", "<<_dPsi[i]<<", "<<_MJD[i];
			cout<<_UT1_UT1R[i]<<", "<<_UT1_UTC[i]<<", "<<_x[i]<<", "<<_y[i]<<"]";
			cout<<endl;
		}
	}


}

#endif // IERS_H_INCLUDED

