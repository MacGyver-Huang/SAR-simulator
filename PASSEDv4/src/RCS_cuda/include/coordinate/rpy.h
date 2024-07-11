#ifndef RPY_H_INCLUDED
#define RPY_H_INCLUDED

#include <sar/def.h>
#include <basic/opt.h>

namespace rpy{
    using namespace def;
	using namespace opt;
	// ==============================================
	// RPY class
	// ==============================================
	template<typename T>
	class RPY
	{
	public:
		// Constructure
		RPY():_r(0),_p(0),_y(0){};
		RPY(T r,T p,T y):_r(r),_p(p),_y(y){};
		RPY(const RPY<T>& in):_r(in._r),_p(in._p),_y(in._y){};
		// Set Method
		void SetZero();
		// Operator overloading
		RPY<T>& operator=(const RPY<T>& R);
		template<typename T2> friend const RPY<T2> operator+(const RPY<T2>& L,const RPY<T2>& R);
		template<typename T2> friend const RPY<T2> operator-(const RPY<T2>& L,const RPY<T2>& R);
		template<typename T2> friend const RPY<T2> operator*(const RPY<T2>& L,const RPY<T2>& R);
		template<typename T2> friend const RPY<T2> operator/(const RPY<T2>& L,const RPY<T2>& R);
		template<typename T2,typename T3> friend const RPY<T2> operator+(const RPY<T2>& L,const T3& R);
		template<typename T2,typename T3> friend const RPY<T2> operator-(const RPY<T2>& L,const T3& R);
		template<typename T2,typename T3> friend const RPY<T2> operator*(const RPY<T2>& L,const T3& R);
		template<typename T2,typename T3> friend const RPY<T2> operator/(const RPY<T2>& L,const T3& R);
		template<typename T2> friend istream& operator>>(istream &is,RPY<T2>& in);
		template<typename T2> friend ostream& operator<<(ostream &os,const RPY<T2>& in);
		// Get
		const T& r() const{return _r;};
		const T& p() const{return _p;};
		const T& y() const{return _y;};
		T& r(){return _r;};
		T& p(){return _p;};
		T& y(){return _y;};
		// Set
		void SetRPY(T r,T p,T y){_r=r;_p=p;_y=y;};
		// Misc.
		void Print()const;
	private:
		T _r,_p,_y;
	};
	
	
	
	// Implement ********************************************************
	
	//
	// Set Method
	//
	template<typename T>
	void RPY<T>::SetZero(){
		_r=0;_p=0;_y=0;
	}
	
	//
	// Operator overloading
	//
	template<typename T>
	RPY<T>& RPY<T>::operator=(const RPY<T>& R){
		_r=R._r; _p=R._p; _y=R._y;
		return *this;
	}
	
	template<typename T>
	const RPY<T> operator+(const RPY<T>& L,const RPY<T>& R){
		return RPY<T>( L._r+R._r,L._p+R._p,L._y+R._y );
	}
	
	template<typename T>
	const RPY<T> operator-(const RPY<T>& L,const RPY<T>& R){
		return RPY<T>( L._r-R._r,L._p-R._p,L._y-R._y );
	}
	
	template<typename T>
	const RPY<T> operator*(const RPY<T>& L,const RPY<T>& R){
		return RPY<T>( L._r*R._r,L._p*R._p,L._y*R._y );
	}
	
	template<typename T>
	const RPY<T> operator/(const RPY<T>& L,const RPY<T>& R){
		return RPY<T>( L._r/R._r,L._p/R._p,L._y/R._y );
	}
	
	template<typename T1,typename T2>
	const RPY<T1> operator+(const RPY<T1>& L,const T2& R){
		return RPY<T1>( L._r+R,L._p+R,L._y+R );
	}
	
	template<typename T1,typename T2>
	const RPY<T1> operator-(const RPY<T1>& L,const T2& R){
		return RPY<T1>( L._r-R,L._p-R,L._y-R );
	}
	
	template<typename T1,typename T2>
	const RPY<T1> operator*(const RPY<T1>& L,const T2& R){
		return RPY<T1>( L._r*R,L._p*R,L._y*R );
	}
	
	template<typename T1,typename T2>
	const RPY<T1> operator/(const RPY<T1>& L,const T2& R){
		return RPY<T1>( L._r/R,L._p/R,L._y/R );
	}
	
	// -----
	template<typename T>
	istream& operator>>(istream &is,RPY<T>& in){
        is>>in._r>>in._p>>in._y;
        return is;
    }
	
    template<typename T>
    ostream& operator<<(ostream &os,const RPY<T>& in){
        os<<in._r<<" "<<in._p<<" "<<in._y;
        return os;
    }
	
	
	
	
	
	
} // namespace

#endif // RPY_H_INCLUDED
