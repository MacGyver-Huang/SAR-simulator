#ifndef VEC_H_INCLUDED
#define VEC_H_INCLUDED

#include <sar/def.h>
#include <basic/opt.h>
#include <basic/cplx.h>

namespace vec{
    using namespace def;
	using namespace opt;
	using namespace cplx;
	// Temp object
	template<typename T>
	class VEC3MAT
	{
		public:
			VEC3MAT();
			~VEC3MAT();
			T& operator[](const long i){return _v[i];};
			const T& operator[](const long i)const{return _v[i];};
		private:
			T* _v;
	};
	// ==============================================
	// Vector class
	// ==============================================
	template<typename T>	
	class VEC
	{
		public:
			// Constructure
			VEC():_x(0),_y(0),_z(0){};
			VEC(T a,T b,T c):_x(a),_y(b),_z(c){};
			VEC(const VEC<T>& in):_x(in._x),_y(in._y),_z(in._z){};
			template<typename T2> VEC(const VEC<T2>& in){
				_x = (T)in.x();
				_y = (T)in.y();
				_z = (T)in.z();
			}
			template<typename T2> VEC(VEC<T2>& in){
				_x = (T)in.x();
				_y = (T)in.y();
				_z = (T)in.z();
			}
			// Set Method
			void SetZero();
			void SetRandFloat();
			// Operator overloading
			VEC<T>& operator=(const VEC<T>& R);
			VEC<T>& operator+=(const VEC<T>& R);
			VEC<T>& operator-=(const VEC<T>& R);
			VEC<T>& operator*=(const VEC<T>& R);
			VEC<T>& operator/=(const VEC<T>& R);
//			friend VEC<T> operator-();
			VEC<T> operator-()const{ return VEC<T>(-_x,-_y,-_z); }
			template<typename T2> friend const VEC<T2> operator+(const VEC<T2>& L,const VEC<T2>& R);
			template<typename T2> friend const VEC<T2> operator-(const VEC<T2>& L,const VEC<T2>& R);
			template<typename T2> friend const VEC<T2> operator*(const VEC<T2>& L,const VEC<T2>& R);
			template<typename T2> friend const VEC<T2> operator/(const VEC<T2>& L,const VEC<T2>& R);
			template<typename T2,typename T3> friend const VEC<T2> operator+(const VEC<T2>& L,const T3& R);
			template<typename T2,typename T3> friend const VEC<T2> operator-(const VEC<T2>& L,const T3& R);
			template<typename T2,typename T3> friend const VEC<T2> operator*(const VEC<T2>& L,const T3& R);
			template<typename T2,typename T3> friend const VEC<T2> operator/(const VEC<T2>& L,const T3& R);
			template<typename T2,typename T3> friend const VEC<T2> operator+(const T2& L, const VEC<T3>& R);
			template<typename T2,typename T3> friend const VEC<T2> operator-(const T2& L, const VEC<T3>& R);
			template<typename T2,typename T3> friend const VEC<T2> operator*(const T2& L, const VEC<T3>& R);
			template<typename T2,typename T3> friend const VEC<T2> operator/(const T2& L, const VEC<T3>& R);
			template<typename T2> friend istream& operator>>(istream &is,VEC<T2>& in);
			template<typename T2> friend ostream& operator<<(ostream &os,const VEC<T2>& in);
			const T& operator[](const long i)const;
			T& operator[](const long i);
			// Get
			const T& x() const{return _x;};
			const T& y() const{return _y;};
			const T& z() const{return _z;};
			T& x(){return _x;};
			T& y(){return _y;};
			T& z(){return _z;};
			T abs() const;
			void Unit();
			// Set
			void Setx(T x){_x=x;};
			void Sety(T y){_y=y;};
			void Setz(T z){_z=z;};
			void Setxyz(T x,T y,T z){_x=x;_y=y;_z=z;};
			void Setxyz(const VEC<T>& in){_x=in.x();_y=in.y();_z=in.z();};
			// Misc.
			void Print()const;
			void WriteASCII(const char* filename);
		private:
			T _x,_y,_z;
	};
	
	// ==============================================
	// namespace declare
	// ==============================================
	template<typename T1, typename T2> void dot(const VEC<T1>& a, const VEC<T2>& b, T2& out);
	template<typename T> void dot(const VEC<T>& a, const VEC<T>& b, T& out);
	template<typename T> T dot(const VEC<T>& a,const VEC<T>& b);
	double dot(const VEC<double>& a,const VEC<float>& b);
	template<typename T> CPLX<T> dot(const VEC<CPLX<T> >& a, const VEC<T>& b);
	template<typename T> VEC<T> cross(const VEC<T>& a,const VEC<T>& b);
	template<typename T1, typename T2> VEC<CPLX<T1> > cross(const VEC<T1>& a,const VEC<CPLX<T2> >& b);
	template<typename T> T angle(const VEC<T>& a,const VEC<T>& b);
	template<typename T> T angle(const VEC<T>& a,const VEC<T>& b,double& cos_val);
	template<typename T> VEC<T> Unit(const VEC<T>& a);
	template<typename T> VEC<T> VecLineEq(const VEC<T>& uv,const VEC<T>& p,const T in,const char type);
	template<typename T> VEC<T> Multiply(const VEC3MAT<T>& Mat, const VEC<T>& Vec);
	template<typename T> T Norm2(const VEC<T>& Vec);
	template<typename T> VEC<T> ProjectOnPlane(const VEC<T>& Vec, const VEC<T>& N);
	template<typename T> VEC<T> Unit3DNonZeroVector(const VEC<T>& vectorMatrix);
	template<typename T> T SignedAngleTwo3DVectors(const VEC<T>& startVec, const VEC<T>& endVec, const VEC<T>& rotateVec);
	template<typename T> bool CheckEffectiveDiffractionIncident(const VEC<T>& e1_x, const VEC<T>& e1_z, const VEC<T>& e2_x, const VEC<T>& sp, const bool SHOW=false);
namespace find{
	template<typename T> void ArbitraryRotateMatrix(const double theta,const VEC<T>& uv, VEC3MAT<T>& M);
	template<typename T> VEC<T> ArbitraryRotate(const VEC<T>& p,const double theta,const VEC<T>& uv);
	template<typename T> VEC<T> ArbitraryRotate(const VEC<T>& p,const double theta,const VEC<T>& uv, VEC3MAT<T>& M);
	template<typename T> VEC<T> PointFromDis(const VEC<T>& n,const VEC<T>& P,const double dis);
	template<typename T> double MinDistanceFromPointToPlane(const VEC<T>& n,const VEC<T>& A,const VEC<T>& P);
	template<typename T> double MinDistanceFromPointToPlane(const VEC<T>& n,const VEC<T>& A,const VEC<T>& P, VEC<T>& POINT);
	template<typename T> double MinDistanceFromPointToLine(const VEC<T>& LP1,const VEC<T>& LP2,const VEC<T>& P);
	template<typename T> double MinDistanceFromPointToLine(const VEC<T>& LP1,const VEC<T>& LP2,const VEC<T>& P,VEC<T>& P_res);
	template<typename T> VEC<T> UnitVector(const VEC<T>& A,const VEC<T>& C,const VEC<T>& X,const T theta);
	template<typename T> VEC<T> UnitVector(const VEC<T>& A,const VEC<T>& C,const VEC<T>& X,const T theta,VEC<T>& E);
	template<typename T> VEC<T> PointMinDistnace(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C);
	template<typename T> VEC<T> PointMinDistnace(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C,double& dist);
	template<typename T> T TriangleArea(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C);
}

	// Implement ********************************************************
//	// ==============================================
//	// constructure
//	// ==============================================
//	template<typename T2, typename T>
//	VEC<T>(const VEC<T2>& in){
//		_x = (T)in.x();
//		_y = (T)in.y();
//		_z = (T)in.z();
//	}
	
	// VEC3MATRIX
	template<typename T>
	VEC3MAT<T>::VEC3MAT(){
		_v = new T[6]; // Row major
	}
	
	template<typename T>
	VEC3MAT<T>::~VEC3MAT(){
	    if(_v != NULL){
	        delete [] _v;
	    }
	}
	
	//
	// Set Method
	//
	template<typename T>
	void VEC<T>::SetZero(){
		_x=0;_y=0;_z=0;
	}
	
	template<typename T>
	void VEC<T>::SetRandFloat(){
		_x = rand() * (2.f / RAND_MAX) - 1.f;
		_y = rand() * (2.f / RAND_MAX) - 1.f;
		_z = rand() * (2.f / RAND_MAX) - 1.f;
	}

	//
	// Operator overloading
	//
	template<typename T>
	VEC<T>& VEC<T>::operator=(const VEC<T>& R){
		_x=R._x; _y=R._y; _z=R._z;
		return *this;
	}
	
	template<typename T>
	VEC<T>& VEC<T>::operator+=(const VEC<T>& R){
		_x+=R._x; _y+=R._y; _z+=R._z;
		return *this;
	}
	
	template<typename T>
	VEC<T>& VEC<T>::operator-=(const VEC<T>& R){
		_x-=R._x; _y-=R._y; _z-=R._z;
		return *this;
	}
	
	template<typename T>
	VEC<T>& VEC<T>::operator*=(const VEC<T>& R){
		_x*=R._x; _y*=R._y; _z*=R._z;
		return *this;
	}
	
	template<typename T>
	VEC<T>& VEC<T>::operator/=(const VEC<T>& R){
		_x/=R._x; _y/=R._y; _z/=R._z;
		return *this;
	}

	template<typename T>
	const VEC<T> operator+(const VEC<T>& L,const VEC<T>& R){
		return VEC<T>( L._x+R._x,L._y+R._y,L._z+R._z );
	}

	template<typename T>
	const VEC<T> operator-(const VEC<T>& L,const VEC<T>& R){
		return VEC<T>( L._x-R._x,L._y-R._y,L._z-R._z );
	}

	template<typename T>
	const VEC<T> operator*(const VEC<T>& L,const VEC<T>& R){
		return VEC<T>( L._x*R._x,L._y*R._y,L._z*R._z );
	}

	template<typename T>
	const VEC<T> operator/(const VEC<T>& L,const VEC<T>& R){
		return VEC<T>( L._x/R._x,L._y/R._y,L._z/R._z );
	}

	template<typename T1,typename T2>
	const VEC<T1> operator+(const VEC<T1>& L,const T2& R){
		return VEC<T1>( L._x+R,L._y+R,L._z+R );
	}

	template<typename T1,typename T2>
	const VEC<T1> operator-(const VEC<T1>& L,const T2& R){
		return VEC<T1>( L._x-R,L._y-R,L._z-R );
	}

	template<typename T1,typename T2>
	const VEC<T1> operator*(const VEC<T1>& L,const T2& R){
		return VEC<T1>( L._x*R,L._y*R,L._z*R );
	}

	template<typename T1,typename T2>
	const VEC<T1> operator/(const VEC<T1>& L,const T2& R){
		return VEC<T1>( L._x/R,L._y/R,L._z/R );
	}
		
	template<typename T1,typename T2>
	const VEC<T1> operator+(const T1& L, const VEC<T2>& R){
		return VEC<T1>( R._x+L,R._y+L,R._z+L );
	}
	
	template<typename T1,typename T2>
	const VEC<T1> operator-(const T1& L, const VEC<T2>& R){
		return VEC<T1>( R._x-L,R._y-L,R._z-L );
	}
	
	template<typename T1,typename T2>
	const VEC<T1> operator*(const T1& L, const VEC<T2>& R){
		return VEC<T1>( R._x*L,R._y*L,R._z*L );
	}
	
	template<typename T1,typename T2>
	const VEC<T1> operator/(const T1& L, const VEC<T2>& R){
		return VEC<T1>( R._x/L,R._y/L,R._z/L );
	}
	
	
	
	
	

	// -----
	template<typename T>
	istream& operator>>(istream &is,VEC<T>& in){
        is>>in._x>>in._y>>in._z;
        return is;
    }

    template<typename T>
    ostream& operator<<(ostream &os,const VEC<T>& in){
		os<<"["<<in.x()<<","<<in.y()<<","<<in.z()<<"]";
		return os;
	}
	
	template<typename T>
	const T& VEC<T>::operator[](const long i)const{
		if(i == 0){
			return _x;
		}else if(i == 1){
			return _y;
		}else if(i == 2){
			return _z;
		}else{
			cout<<"ERROR::The input index i="<<i<<", must smaller than 2"<<endl;
			return _x;
		}
	}
	
	template<typename T>
	T& VEC<T>::operator[](const long i){
		if(i == 0){
			return _x;
		}else if(i == 1){
			return _y;
		}else if(i == 2){
			return _z;
		}else{
			cout<<"ERROR::The input index i="<<i<<", must smaller than 2"<<endl;
			return _x;
		}
	}


	//
	// Get
	//
	template<typename T>
	T VEC<T>::abs() const{
		return sqrt(_x*_x+_y*_y+_z*_z);
	}
	
	template<typename T>
	void VEC<T>::Unit(){
		T inv = 1/sqrt(_x*_x+_y*_y+_z*_z);
		_x *= inv;
		_y *= inv;
		_z *= inv;
	}

	//
	// Misc.
	//
	template<typename T>
	void VEC<T>::Print()const{
		cout<<std::setprecision(10)<<"["<<_x<<","<<_y<<","<<_z<<"]"<<endl;
//		printf("[%.10f,%.10f,%.10f]\n",_x,_y,_z);
	}

	template<typename T>
	void VEC<T>::WriteASCII(const char* filename){
	/*
	 Purpose:
		Write a seires to ascii file in the disk.
	*/
		ofstream fout;
		fout.open(filename);
		if(fout.fail()){
			cout<<"ERROR::[WriteAscii]Input filename! -> ";
			cout<<filename<<endl;
			exit(EXIT_FAILURE);
		}
		fout<<std::setprecision(10)<<std::fixed<<_x<<"\t"<<_y<<"\t"<<_z<<endl;
		fout.close();
	}


	//
	// namespace
	//
	template<typename T1, typename T2>
	void dot(const VEC<T1>& a, const VEC<T2>& b, T2& out){
		out = 0;
		out += a.x() * b.x();
		out += a.y() * b.y();
		out += a.z() * b.z();
	}
	
	template<typename T>
	void dot(const VEC<T>& a, const VEC<T>& b, T& out){
		out = 0;
		out += a.x() * b.x();
		out += a.y() * b.y();
		out += a.z() * b.z();
	}
	
	template<typename T>
	T dot(const VEC<T>& a,const VEC<T>& b){
		return (a.x()*b.x() + a.y()*b.y() + a.z()*b.z());
	}

	double dot(const VEC<double>& a,const VEC<float>& b){
		return (a.x()*b.x() + a.y()*b.y() + a.z()*b.z());
	}
	
	template<typename T>
	CPLX<T> dot(const VEC<CPLX<T> >& a, const VEC<T>& b){
		T r = a.x().r()*b.x() + a.y().r()*b.y() + a.z().r()*b.z();
		T i = a.x().i()*b.x() + a.y().i()*b.y() + a.z().i()*b.z();
		return CPLX<T>(r,i);
	}

	template<typename T>
	VEC<T> cross(const VEC<T>& a,const VEC<T>& b){
		return VEC<T>(a.y()*b.z() - a.z()*b.y(), \
					  a.z()*b.x() - a.x()*b.z(), \
					  a.x()*b.y() - a.y()*b.x());
	}

	template<typename T1, typename T2>
	VEC<CPLX<T1> > cross(const VEC<T1>& a,const VEC<CPLX<T2> >& b){
		return VEC<CPLX<T1> >(a.y()*b.z() - a.z()*b.y(),
				a.z()*b.x() - a.x()*b.z(),
				a.x()*b.y() - a.y()*b.x());
	}

	template<typename T>
	T angle(const VEC<T>& a,const VEC<T>& b){
		T a_dot_b = dot(a,b);
		T ab = a.abs()*b.abs();
//		return acos(a_dot_b/ab);
		double cos_val=a_dot_b/ab;

		if(abs(a_dot_b - ab) < 1E-14){
			return (T)0;
		}

		if(abs(cos_val-1) < 1E-14){
			return (T)0;
		}else{
			return acos(cos_val);
		}
	}
	
	template<typename T>
	T angle(const VEC<T>& a,const VEC<T>& b,double& cos_val){
		T a_dot_b = dot(a,b);
		T ab = a.abs()*b.abs();
		cos_val = a_dot_b/ab;
//		return acos(cos_val);
		if(abs(cos_val-1) < 1E-14){
			return (T)0;
		}else{
			return acos(cos_val);
		}
	}

	template<typename T>
	VEC<T> Unit(const VEC<T>& a){
	/*
	 Input:
		a	:[m,m,m] any space vector
	 Return:
		out	:[m,m,m] uni-vector of a
	*/
		T inv = 1/a.abs();
		return VEC<T>(a.x()*inv, a.y()*inv, a.z()*inv);
//		return a/a.abs();
		//double abs_a = a.abs();
		//return VEC<T>(a.x()/abs_a,
		//			  a.y()/abs_a,
		//			  a.z()/abs_a);
	}

	template<typename T>
	VEC<T> VecLineEq(const VEC<T>& uv,const VEC<T>& p,const T in,const char type){
	/*
	 Puepose:
		Equation: (x-x0)/A = (Y-Y0)/B = (Z-Z0)/C (Line-equation)
		Find the other two parameter if one of them is known.
		ex: x(known) and line vecotr[A,B,C](known) find y,z(unknown).
	 Input:
		uv	:[x,x,x] uni-vector
		p	:[x] point on this line
		in	:[x] alternative vlaue in [x,y,z]
	 Return:
		[x,y,z] : [x,x,x] total vector value
	*/
		//VEC<T> out;
		switch(type){
			case 'X':
				return VEC<T>(in, \
					   		  (uv.y()/uv.x())*(in-p.x())+p.y(), \
					   		  (uv.z()/uv.x())*(in-p.x())+p.z());
			case 'Y':
				return VEC<T>((uv.x()/uv.y())*(in-p.y())+p.x(),\
					   		  in, \
							  (uv.z()/uv.y())*(in-p.y())+p.z());
			case 'Z':
				return VEC<T>((uv.x()/uv.z())*(in-p.z())+p.x(), \
							  (uv.y()/uv.z())*(in-p.z())+p.y(), \
							  in);
			default:
				cout<<"ERROR(VecLinEq)::Input error!"<<endl;
				break;
		}
		//return out;
		exit(EXIT_FAILURE);
	}
	
	template<typename T> 
	VEC<T> Multiply(const VEC3MAT<T>& Mat, const VEC<T>& Vec){
		VEC<T> out;
		T tmp[] = {0,0,0};
		for(int i=0;i<3;++i){
			tmp[i] = Mat[i*3+0]*Vec.x() + Mat[i*3+1]*Vec.y() + Mat[i*3+2]*Vec.z();
		}
		out.Setxyz(tmp[0],tmp[1],tmp[2]);
		return out;
	}

	template<typename T>
	T Norm2(const VEC<T>& Vec){
		return std::sqrt(Vec.x()*Vec.x() + Vec.y()*Vec.y() + Vec.z()*Vec.z());
	}

	template<typename T>
	VEC<T> ProjectOnPlane(const VEC<T>& Vec, const VEC<T>& N){
		// Vec: Arbitrary vector
		// N  : Normal vector
		VEC<T> Vec_proj = Vec - dot(Vec, N)/Norm2(N) * N;
		return vec::Unit(Vec_proj);
	}

	template<typename T>
	VEC<T> Unit3DNonZeroVector(const VEC<T>& vectorMatrix){
		double eps = 1e-16;

		// unitLengthVec3D is what the function returns
		double EuclideanNorm3D =  vec::Norm2(vectorMatrix);
		VEC<T> unitLengthVec3D;
		if(EuclideanNorm3D <= eps) {    // Ensure vector does not have zero length
			cerr<<"Found a vector of length zero; cannot continue processing."<<endl;
		} else {
			unitLengthVec3D = vectorMatrix / EuclideanNorm3D;
		}

		return unitLengthVec3D;
	}

	template<typename T>
	T SignedAngleTwo3DVectors(const VEC<T>& startVec, const VEC<T>& endVec, const VEC<T>& rotateVec){
		double eps = 1e-16;

		if( abs(dot(rotateVec, cross(startVec, endVec))) < (2.5 * eps) ){
//			cerr<<"WARNNING: rotateVec is nearly coplanar with startVec and endVec."<<endl;
			return 0;
		}

		VEC<T> temp = cross(startVec, endVec); // temp is normal to both startVec & endVec

		if( dot(rotateVec, temp) < 0){
			temp = -temp;
		}

		// Get normal vector
		VEC<T> rotateVec2;

		if( (abs(temp.x()) + abs(temp.y()) + abs(temp.z())) > eps ) {
			rotateVec2 = Unit3DNonZeroVector(temp);
		}else{
			rotateVec2 = rotateVec;
		}

		// Numerator of tan(beta) = sin(beta)
		double sineBeta = dot(rotateVec2, cross(startVec, endVec));

		// Denominator of tan(beta) = cos(beta)
		double cosineBeta = dot(startVec, endVec);

		// Two-argument (four-quadrant) arctangent takes sine & cosine as arguments
		double beta = atan2(sineBeta, cosineBeta);

		return beta;
	}

	template<typename T>
	bool CheckEffectiveDiffractionIncident(const VEC<T>& e1_x, const VEC<T>& e1_z, const VEC<T>& e2_x, const VEC<T>& sp, const bool SHOW, const size_t k) {
		// e1_x: Edge x component (on the plate for facet-1)
		// e1_z: Edge z component (along the edge for facet-1)
		// e2_x: Edge x component (on the plate for facet-2)
		// sp: Incident unit vector
		// SHOW: Display results or not? (default = false)

		double ang1 = SignedAngleTwo3DVectors(e1_x, -sp, e1_z);
		double ang2 = SignedAngleTwo3DVectors(e1_x, e2_x, e1_z);

//		double ang10 = ang1;
//		double ang20 = ang2;

		const double PI=3.1415926535897931086245;	// PI

		if (ang1 < 0) {
			ang1 = 360./180.*PI + ang1;
		}

		if (ang2 < 0) {
			ang2 = 360./180.*PI + ang2;
		}

		double WA = 360./180.*PI - ang2;


		if (SHOW) {
			printf("Angle from e1.x to -sp  = %f [deg]\n", 180./PI*ang1);
			printf("Angle from e1.x to e2.x = %f [deg]\n", 180./PI*ang2);
			printf("Wedge angle             = %f [deg]\n", 180./PI*WA);

			printf("\n==================\n");
			if (ang2 > ang1) {
				cout << "  Result: OK" << endl;
			} else {
				cout << "  Result: Fail" << endl;
			}
			printf("==================\n");
		}

//		// DEBUG (START) ========================================================================================================================================================================
//		if(k==60602) {
//			printf("\n\n\n\n>>>> CPU >>>>\ne1_x=(%f,%f,%f), e1_z=(%f,%f,%f), e2_x=(%f,%f,%f), sp=(%f,%f,%f), ang10=%.10f, ang1=%.10f, ang20=%.10f, ang2=%.10f\n>>>>>>>>>>>>>\n\n",
//				   e1_x.x(), e1_x.y(), e1_x.z(), e1_z.x(), e1_z.y(), e1_z.z(), e2_x.x(), e2_x.y(), e2_x.z(), sp.x(), sp.y(), sp.z(), ang10, ang1, ang20, ang2);
//
//		}
//		// DEBUG (END) ==========================================================================================================================================================================


		if (ang2 > ang1) {
			return true;
		} else {
			return false;
		}
	}

	//
	// namepsace find::
	//
	template<typename T>
	void find::ArbitraryRotateMatrix(const double theta,const VEC<T>& uv, VEC3MAT<T>& M){
		VEC<T> r = Unit(uv);
		double C = cos(theta);
		double S = sin(theta);
		
		M[0] = (C + (1 - C) * r.x() * r.x());
		M[1] = ((1 - C) * r.x() * r.y() - r.z() * S);
		M[2] = ((1 - C) * r.x() * r.z() + r.y() * S);
		
		M[3] = ((1 - C) * r.x() * r.y() + r.z() * S);
		M[4] = (C + (1 - C) * r.y() * r.y());
		M[5] = ((1 - C) * r.y() * r.z() - r.x() * S);
		
		M[6] = ((1 - C) * r.x() * r.z() - r.y() * S);
		M[7] = ((1 - C) * r.y() * r.z() + r.x() * S);
		M[8] = (C + (1 - C) * r.z() * r.z());
	}
	
	template<typename T>
	VEC<T> find::ArbitraryRotate(const VEC<T>& p,const double theta,const VEC<T>& uv){
	/*
	 Purpose:
		Rotate about a arbitrary axis
		*NOTE* : The Rotation axis and point p is reference to the original point [0,0,0]
	Input:
		p		:[x,x,x] The point needed to be rotated
		theta	:[rad] angle
		uv		:[x,x,x] Axis reference to the original point [0,0,0]
	Return:
		The point after rotating
	Reference:
		Aurthor : Ronald Goldman
	 	http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
	*/
		VEC<T> out(0,0,0);
		VEC<T> r = Unit(uv);
		double C = cos(theta);
		double S = sin(theta);
		
//		VEC3MAT<T> M;
//		vec::find::ArbitraryRotateMatrix(theta, uv, M);
//
//		return vec::Multiply(M, p);
		
		out.x() += (C + (1 - C) * r.x() * r.x()) * p.x();
		out.x() += ((1 - C) * r.x() * r.y() - r.z() * S) * p.y();
		out.x() += ((1 - C) * r.x() * r.z() + r.y() * S) * p.z();
		
		out.y() += ((1 - C) * r.x() * r.y() + r.z() * S) * p.x();
		out.y() += (C + (1 - C) * r.y() * r.y()) * p.y();
		out.y() += ((1 - C) * r.y() * r.z() - r.x() * S) * p.z();
		
		out.z() += ((1 - C) * r.x() * r.z() - r.y() * S) * p.x();
		out.z() += ((1 - C) * r.y() * r.z() + r.x() * S) * p.y();
		out.z() += (C + (1 - C) * r.z() * r.z()) * p.z();

		return out;
	}
	
	template<typename T>
	VEC<T> find::ArbitraryRotate(const VEC<T>& p,const double theta,const VEC<T>& uv, VEC3MAT<T>& M){
		/*
		 Purpose:
		 Rotate about a arbitrary axis
		 *NOTE* : The Rotation axis and point p is reference to the original point [0,0,0]
		 Input:
		 p		:[x,x,x] The point needed to be rotated
		 theta	:[rad] angle
		 uv		:[x,x,x] Axis reference to the original point [0,0,0]
		 Return:
		 The point after rotating
		 Reference:
		 Aurthor : Ronald Goldman
		 http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
		 */
		VEC<T> out(0,0,0);
		VEC<T> r = Unit(uv);
		
		vec::find::ArbitraryRotateMatrix(theta, uv, M);
		
		out = vec::Multiply(M, p);
		return out;
	}


	template<typename T>
	VEC<T> find::PointFromDis(const VEC<T>& n,const VEC<T>& P,const double dis){
	/*
	 Original name: "VecFindPoint"
	 Input:
		n 	:[m,m,m] line dirctive vector(or Uni-vector)
		P 	:[m,m,m] point at
		dis	:[m] distance from point P
	 Return:
		out :[m,m,m] Destination point
	*/
		double abs_n = n.abs();
		//uv = VecMulScl(n,1/abs_n); // convert plane vector to unit vector

		return VEC<T>(P.x() + dis*n.x()/abs_n, \
					  P.y() + dis*n.y()/abs_n, \
					  P.z() + dis*n.z()/abs_n);
	}

	template<typename T>
	double find::MinDistanceFromPointToPlane(const VEC<T>& n,const VEC<T>& A,const VEC<T>& P){
	/*
	 Input:
		n :[m,m,m] plane normal vector(or Uni-vector)
		A :[m,m,m] point is loacted on the plane
		P :[m,m,m] point is *NOT* located on the plane
	 Return:
		min_dis :[m] minimun distance between plane and P-point
	*/
//		VEC<T> uv=Unit(n);	// convert plane vector to unit vector
//		VEC<T> uv = n;
//		double sum_min_dis=0;
//
//		sum_min_dis = (P.x()-A.x())*uv.x() + (P.y()-A.y())*uv.y() + (P.z()-A.z())*uv.z();
		double sum_min_dis = (P.x()-A.x())*n.x() + (P.y()-A.y())*n.y() + (P.z()-A.z())*n.z();

		return fabs(sum_min_dis);
	}
	
	template<typename T>
	double find::MinDistanceFromPointToPlane(const VEC<T>& n,const VEC<T>& A,const VEC<T>& P,VEC<T>& POINT){
		/*
		 Input:
		 n :[m,m,m] plane normal vector(or Uni-vector)
		 A :[m,m,m] point is loacted on the plane
		 P :[m,m,m] point is *NOT* located on the plane
		 Return:
		 min_dis :[m] minimun distance between plane and P-point
		 */
		VEC<T> uv=Unit(n);	// convert plane vector to unit vector
		double min_dis=0;
		
		min_dis = (P.x()-A.x())*uv.x() + (P.y()-A.y())*uv.y() + (P.z()-A.z())*uv.z();
		POINT = P - min_dis*n;

		
		return fabs(min_dis);
	}

	template<typename T>
	double find::MinDistanceFromPointToLine(const VEC<T>& LP1,const VEC<T>& LP2,const VEC<T>& P){
	/*
	 Purpose:
		Find the minimum distance from point P to Lin (LP1,LP2)
	 Input:
		LP1 : [x,x,x] The first point on the line
		LP2 : [x,x,x] The second point on the line
		P   : [x,x,x] The point is outside of line
	 Return:
		The minmum distance
	 Reference:
		http://local.wasp.uwa.edu.au/~pbourke/geometry/planeeq/
	*/
		VEC<T> mag = LP2-LP1;
		double u = ( (P.x()-LP1.x())*(LP2.x()-LP1.x()) + 
		             (P.y()-LP1.y())*(LP2.y()-LP1.y()) + 
		             (P.z()-LP1.z())*(LP2.z()-LP1.z()) ) /
				   ( mag.x()*mag.x() + mag.y()*mag.y() + mag.z()*mag.z() );
		VEC<T> P_res = LP1 + (LP2-LP1)*u;
		return (P-P_res).abs();
	}

	template<typename T>
	double find::MinDistanceFromPointToLine(const VEC<T>& LP1,const VEC<T>& LP2,const VEC<T>& P,VEC<T>& P_res){
	/*
	 Purpose:
		Find the minimum distance from point P to Lin (LP1,LP2)
	 Input:
		LP1 : [x,x,x] The first point on the line
		LP2 : [x,x,x] The second point on the line
		P   : [x,x,x] The point is outside of line
	 Return:
		The minmum distance
	 Reference:
		http://local.wasp.uwa.edu.au/~pbourke/geometry/planeeq/
	*/
		VEC<T> mag = LP2-LP1;
		double u = ( (P.x()-LP1.x())*(LP2.x()-LP1.x()) + 
		             (P.y()-LP1.y())*(LP2.y()-LP1.y()) + 
		             (P.z()-LP1.z())*(LP2.z()-LP1.z()) ) /
				   ( mag.x()*mag.x() + mag.y()*mag.y() + mag.z()*mag.z() );
		P_res = LP1 + (LP2-LP1)*u;
		return (P-P_res).abs();
	}

	template<typename T>
	VEC<T> find::UnitVector(const VEC<T>& A,const VEC<T>& C,const VEC<T>& X,const T theta){
	/*
	 Input:
		A : [x,x,x] original point @ plane
		X : [x,x,x] second point @ plane
		C : [x,x,x] third point @ plane
		theta : [rad] angle from vector AX (counter-clockwise is positive)
	 Retun:
		uv_AE : [x,x,x] unit vector(direction vector) for destination line eq.
	 Modified:
        (20100401) using ArbitraryRotate function and no limit to the XAC angle.
	*/
		VEC<double>	A_new(0,0,0);
		VEC<double>	X_new = X-A;
		VEC<double> C_new = C-A;
	
		VEC<double> AX = X_new;
		VEC<double> AC = C_new;
	
		VEC<double> uv = cross(AX,AC);
	
		return Unit(vec::find::ArbitraryRotate(X_new,theta,uv));
/*
		VEC<T> AX=X-A;
		VEC<T> AC=C-A;
		VEC<T> uv1=Unit(AC);
		VEC<T> out(-1,-1,-1);
		double ang=angle(AC,AX); // rad
		double L1,L2,L3,dis=0;
		double rad90=def::PI/2.;
		double theta1;

		// find distance
		// angle between AC & AX is 90 deg ?
		if(ang >= def::PI){ // rad2deg(ang) >= 180 deg
			cout<<"ERROR(UnitVector)::angle(CAX) >= 180 deg"<<endl;
			exit(EXIT_FAILURE);
		}

		if(theta > ang){ // rad
			cout<<"ERROR(UnitVector)::theta > angle(CAX)"<<endl;
			exit(EXIT_FAILURE);
		}

		if(theta == ang){ // rad
			out = C;
		}else{
			theta1 = fabs(rad90-ang); // rad
			L1 = AX.abs()*_cos(theta1);
			if(ang < rad90){
				L2=L1/_cos(theta1+theta);
				L3=L2*_sin(theta);
				dis=L3/_cos(theta1);
			}
			if(ang > rad90){
				if(theta > theta1){
					dis=L1*( tan(theta1) + tan(theta-theta1) );
				}else if(theta < theta1){
					dis=L1*( tan(theta1) - tan(theta1-theta) );
				}else{
					dis=L1*tan(theta);
				}
			}
			if(ang == rad90){
				dis=AX.abs()*tan(theta);
			}

			// Find the point with it's distance is "dis"
			out=PointFromDis(uv1,X,dis);
		}
		// find result unit vector of line eq.
		out=out-A;
		out=Unit(out);
		return out;
*/
	}

	template<typename T>
	VEC<T> find::UnitVector(const VEC<T>& A,const VEC<T>& C,const VEC<T>& X,const T theta,VEC<T>& E){
	/*
	 Input:
		A :[x,x,x] original point @ plane
		X :[x,x,x] second point @ plane
		C :[x,x,x] third point @ plane
		theta : [rad] angle from vector AX (counter-clockwise is positive)
	 Retun:
		uv  :[x,x,x] unit vector(direction vector) for destination line eq.
		E	:[x,x,x] point on AE vector
	*/
		VEC<T> uv_AE=UnitVector(A,C,X,theta);
		E = A+uv_AE*1.;
		return uv_AE;
	}

    template<typename T>
    VEC<T> find::PointMinDistnace(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C){
    /*
     Purpose:
        Find the min distance from point "C" to AB line
     Example:
        VEC<double> a( 4.,0.,0.);
        VEC<double> b(-5.,0.,0.);
        VEC<double> c( 0.,3.,0.);
        VEC<double> D=sar::find::PointMinDistance(a,b,c);
        double ang=vec::angle(C-D,A-D);
        cout<<ang<<endl; //MUST be 90[deg]
    */
        VEC<T> AB=B-A;
        VEC<T> AC=C-A;
        double alpha=vec::angle(AB,AC); //[rad]
        double m = AC.abs()*_cos(alpha);
#ifdef DEBUG
        cout<<alpha<<endl;
        cout<<"Unit(AB)="; vec::Unit(AB).Print();
#endif
        return vec::find::PointFromDis(vec::Unit(AB),A,m);
    }

    template<typename T>
    VEC<T> find::PointMinDistnace(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C,double& dist){
    /*
     Purpose:
        Find the min distance from point "C" to AB line
     Example:
        VEC<double> a( 4.,0.,0.);
        VEC<double> b(-5.,0.,0.);
        VEC<double> c( 0.,3.,0.);
        VEC<double> D=sar::find::PointMinDistance(a,b,c);
        double ang=vec::angle(C-D,A-D);
        cout<<ang<<endl; //MUST be 90[deg]
    */
        VEC<T> AB=B-A;
        VEC<T> AC=C-A;
        double alpha=vec::angle(AB,AC); //[rad]
        dist=AC.abs()*_sin(alpha);//[m]
        double m = AC.abs()*_cos(alpha);
        return vec::find::PointFromDis(vec::Unit(AB),A,m);
    }

    template<typename T>
    T find::TriangleArea(const VEC<T>& A,const VEC<T>& B,const VEC<T>& C){
    /*
     Purpose:
        Find the triangle area
     Input:
        A	:[m,m,m] End point 1
        B	:[m,m,m] End point 2
        C	:[m,m,m] End point 3
     Return:
        out :[m^2] triangle area
     Ref:
        using Heron's formula
    */
        T AB = (B-A).abs();
        T AC = (C-A).abs();
        T BC = (B-C).abs();

        T s = 0.5*(AB+AC+BC);

        return sqrt(s*(s-AB)*(s-AC)*(s-BC));
    }




} // namespace

#endif // VEC_H_INCLUDED
