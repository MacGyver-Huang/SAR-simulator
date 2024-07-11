#ifndef GEO_H_INCLUDED
#define GEO_H_INCLUDED

#include <sar/def.h>


namespace geo{
	// ==============================================
	// Geodetic/Geocentric Calss
	// ==============================================
	template<typename T>
	class GEO{
	public:
		// Constructure
		GEO():_lon(0),_lat(0),_h(0){};
		GEO(T a,T b,T c):_lon(a),_lat(b),_h(c){};
		GEO(const GEO<T>& in):_lon(in._lon),_lat(in._lat),_h(in._h){};
		// Operator overloading
		GEO<T>& operator=(const GEO<T>& b);
		template<typename T2> friend const GEO<T2> operator+(const GEO<T2>& b);
		template<typename T2> friend const GEO<T2> operator-(const GEO<T2>& L,const GEO<T2>& R);
		template<typename T2> friend const GEO<T2> operator*(const GEO<T2>& L,const GEO<T2>& R);
		template<typename T2> friend const GEO<T2> operator/(const GEO<T2>& L,const GEO<T2>& R);
		template<typename T2,typename T3> friend const GEO<T2> operator+(const GEO<T2>& L,const T3& R);
		template<typename T2,typename T3> friend const GEO<T2> operator-(const GEO<T2>& L,const T3& R);
		template<typename T2,typename T3> friend const GEO<T2> operator*(const GEO<T2>& L,const T3& R);
		template<typename T2,typename T3> friend const GEO<T2> operator/(const GEO<T2>& L,const T3& R);
		template<typename T2> friend ostream& operator<<(ostream& os, const GEO<T2>& in);
		// Get
		const T lon()const{return _lon;};
		const T lat()const{return _lat;};
		const T h()const{return _h;};
		T& lon(){return _lon;};
		T& lat(){return _lat;};
		T& h(){return _h;};
		// Set
		void Setlon(T in_lon){_lon=in_lon;};
		void Setlat(T in_lat){_lat=in_lat;};
		void Seth(T in_h){_h=in_h;};
		// Misc.
		void Print();
		void PrintDeg();
	private:
		T _lon,_lat,_h;
	};


	// Implement ********************************************************

	//
	// Operator overloading
	//
	template<typename T>
	GEO<T>& GEO<T>::operator=(const GEO<T>& b){
		_lon = b._lon; _lat = b._lat; _h = b._h;
		return *this;
	}

	template<typename T>
	const GEO<T> operator+(const GEO<T>& L,const GEO<T>& R){
		return GEO<T>( L._lon+R._lon,L._lat+R._lat,L._h+R._h );
	}

	template<typename T>
	const GEO<T> operator-(const GEO<T>& L,const GEO<T>& R){
		return GEO<T>( L._lon-R._lon,L._lat-R._lat,L._h-R._h );
	}

	template<typename T>
	const GEO<T> operator*(const GEO<T>& L,const GEO<T>& R){
		return GEO<T>( L._lon*R._lon,L._lat*R._lat,L._h*R._h );
	}

	template<typename T>
	const GEO<T> operator/(const GEO<T>& L,const GEO<T>& R){
		return GEO<T>( L._lon/R._lon,L._lat/R._lat,L._h/R._h );
	}

	template<typename T1,typename T2>
	const GEO<T1> operator+(const GEO<T1>& L,const T2& R){
		return GEO<T1>( L._lon+R,L._lat+R,L._h+R );
	}

	template<typename T1,typename T2>
	const GEO<T1> operator-(const GEO<T1>& L,const T2& R){
		return GEO<T1>( L._lon-R,L._lat-R,L._h-R );
	}

	template<typename T1,typename T2>
	const GEO<T1> operator*(const GEO<T1>& L,const T2& R){
		return GEO<T1>( L._lon*R,L._lat*R,L._h*R );
	}

	template<typename T1,typename T2>
	const GEO<T1> operator/(const GEO<T1>& L,const T2& R){
		return GEO<T1>( L._lon/R,L._lat/R,L._h/R );
	}
	
	template<typename T>
	ostream& operator<<(ostream& os, const GEO<T>& in){
		os<<"{"<<in._lon<<","<<in._lat<<","<<in._h<<"}";
		return os;
	}

	//
	// Misc.
	//
	template<typename T>
	void GEO<T>::Print(){
		cout<<"["<<std::setprecision(8)<<_lon<<","<<_lat<<","<<_h<<"]"<<endl;
	}

	template<typename T>
	void GEO<T>::PrintDeg(){
		cout<<"["<<std::setprecision(8)<<_lon*def::RTD<<","<<_lat*def::RTD<<","<<_h<<"]"<<endl;
	}
}

#endif // GEO_H_INCLUDED
