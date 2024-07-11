#ifndef D1_H_INCLUDED
#define D1_H_INCLUDED

//#include "def.h"
#include <cstdlib>
#include <fstream>
#include "def_prefunc.h"
#include "cplx.h"
#include "vec.h"

using namespace std;
using namespace cplx;
using namespace vec;

namespace d1{
	// ==============================================
	// Series (1D vector)
	// ==============================================
	template<typename T>
	class D1
	{
		public:
			// Constructure
			D1();//:_n(1),_v(new T[1]){};
			D1(const size_t n);//:_n(n),_v(new T[_n]){};
			D1(const T* series,const size_t n);
			// Copy constructor
			D1(const D1<T>& in);
			// de-constructre *MUST*
			~D1();
			// Operator overloading
			// *Do NOT* overloading other operator for the effectivity of execution.
			D1<T>& operator=(const D1<T>& in);
			//
			T& operator[](const size_t i){return _v[i];};
			const T& operator[](const size_t i)const{return _v[i];};
			//
			template<typename T2> friend const D1<T2> operator+(const D1<T2>& L,const D1<T2>& R);
			template<typename T2> friend const D1<T2> operator-(const D1<T2>& L,const D1<T2>& R);
			template<typename T2> friend const D1<T2> operator*(const D1<T2>& L,const D1<T2>& R);
			template<typename T2> friend const D1<T2> operator/(const D1<T2>& L,const D1<T2>& R);
			template<typename T2> friend const D1<T2> operator+(const D1<T2>& L,const T2& R);
			template<typename T2> friend const D1<T2> operator-(const D1<T2>& L,const T2& R);
			template<typename T2> friend const D1<T2> operator*(const D1<T2>& L,const T2& R);
			template<typename T2> friend const D1<T2> operator/(const D1<T2>& L,const T2& R);
			template<typename T2> friend const D1<T2> operator+(const T2& L,const D1<T2>& R);
//			template<typename T2> friend const D1<T2> operator-(const T2& L,const D1<T2>& R);
			template<typename T2> friend const D1<T2> operator*(const T2& L,const D1<T2>& R);
//			template<typename T2> friend const D1<T2> operator/(const T2& L,const D1<T2>& R);
			// Set
			void Set(const T* series,const size_t n);
			// Misc.
			void SetZero();
			size_t GetNum() const;
			TYPE GetType() const{ return _type; };
			void PrintAll();
			void Print();
			void Print()const;
			void Print(const size_t start_idx,const size_t end_idx);
			void WriteBinary(const char* filename);
			void WriteBinary(const char* filename)const;
			void WriteASCII(const char* filename, const int Precision=4);
			void Crop(const size_t start_idx, const size_t end_idx);
			D1<T> GetRange(const size_t start_idx, const size_t end_idx);
			D1<T>& Square();
			void SelfSquare();
			void Indgen();
			void clear();
			void Swap();
			// Set & Get Pointer
			T* GetPtr() const{return _v;};
		private:
			void _error();
			TYPE _type;
			size_t _n;
			T* _v;
	};
	// ==============================================
	// namespace declare
	// ==============================================

	// Implement ********************************************************

	//
	// Error
	//
	template<typename T>
	void D1<T>::_error(){
		if(_type == UNKNOW){
			if(typeid(T).name() == typeid(D1<bool*>).name()){ _type = pBOOLEAN;
			}else if(typeid(T).name() == typeid(D1<char*>).name()){ _type = pCHAR;
			}else if(typeid(T).name() == typeid(D1<short*>).name()){ _type = pSHORT;
			}else if(typeid(T).name() == typeid(D1<int*>).name()){ _type = pINT;
			}else if(typeid(T).name() == typeid(D1<float*>).name()){ _type = pFLT;
			}else if(typeid(T).name() == typeid(D1<double*>).name()){ _type = pDB;
			}else if(typeid(T).name() == typeid(D1<CPLX<float*> >).name()){ _type = pCPLX_FLT;
			}else if(typeid(T).name() == typeid(D1<CPLX<double*> >).name()){ _type = pCPLX_DB;
			}else if(typeid(T).name() == typeid(D1<VEC<float*> >).name()){ _type = pVEC_FLT;
			}else if(typeid(T).name() == typeid(D1<VEC<double*> >).name()){ _type = pVEC_DB;
			}else if(typeid(T).name() == typeid(D1<GEO<float*> >).name()){ _type = pGEO_FLT;
			}else if(typeid(T).name() == typeid(D1<GEO<double*> >).name()){ _type = pGEO_DB;
			// D1
			}else if(typeid(T).name() == typeid(D1<CPLX<double>*>).name()){ _type = D1_pCPLX_DB;
			}else if(typeid(T).name() == typeid(D1<CPLX<float>*>).name()){  _type = D1_pCPLX_FLT;
			}else{
				_type = UNKNOW;
			}
		}
		if(_type == UNKNOW){
			cout<<"ERROR::[D1() data type MUST be bool, char, short, int, float, double, "<<endl;
			cout<<"        CPLX<float>, CPLX<double>, VEC<float>, VEC<double>, "<<endl;
			cout<<"        GEO<float>, GEO<double>, D1<CPLX<double>*>, D1<CPLX<float>*>, D1<D1<float> >, D1<D1<float*> >]"<<endl;
			cout<<typeid(T).name()<<"   "<<_type<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
			exit(EXIT_FAILURE);
		}
	}
	
    //
    // Constructor
    //
	template<typename T>
	D1<T>::D1(){
		_n = 0L;
		_v = NULL;
		//_v = new T[0L];
		_type = def_prefunc::GetType(typeid(T).name());
		_error();
	}
	
	template<typename T>
	D1<T>::D1(const size_t n){
		_n = n;
		_v = new T[_n];
		_type = def_prefunc::GetType(typeid(T).name());
		_error();
	}
	
    template<typename T>
    D1<T>::D1(const T* series,const size_t n){
		_type = def_prefunc::GetType(typeid(T).name());
		_error();
        _n=n;
        _v=new T[_n];
        for(size_t i=0;i<_n;++i){
            _v[i]=series[i];
        }
    }

    //
    // Copy constructor
    //
    template<typename T>
    D1<T>::D1(const D1<T>& in){
		_type = def_prefunc::GetType(typeid(T).name());
		_error();
        if(this != &in){
            _n=in._n;
            _v=new T[_n];
            //std::copy(in._v,in._v+in._n,_v);  // #include <algorithm> for std::copy
            for(size_t i=0;i<in._n;++i){
                _v[i]=in._v[i];
            }
        }
    }


	//
	// De-constructor
	//
	template<typename T>
	D1<T>::~D1(){
	    if(_v != NULL){
//		if(_n != 0 & _v != NULL){
//		if(_n > 1){
	        delete [] _v;
	    }
	    _n=0L;
	}

	//
	// Operator overloading
	//
	template<typename T>
	D1<T>& D1<T>::operator=(const D1<T>& in){
	    if(this != &in){
//            // 1: allocate new memory and copy the elements
//            T* new_v = new T[in._n];
//            //std::copy(in._v, in._v + in._n, new_v);
//            for(size_t i=0;i<in._n;++i){
//                new_v[i]=in._v[i];
//            }
//            // 2: deallocate old memory
////            delete [] _v;
//            // 3: assign the new memory to the object
//            _v = new_v;
//            _n = in._n;
//			_type = in._type;
			if(_n != 0){
				delete [] _v;
			}
			_v = new T[in._n];
			_n = in._n;
			_type = in._type;
			for(size_t i=0;i<_n;++i){
				_v[i] = in._v[i];
			}
        }
        return *this;
	}

	template<typename T>
	const D1<T> operator+(const D1<T>& L,const D1<T>& R){
	    size_t num_L=L.GetNum();
	    size_t num_R=R.GetNum();
	    if(num_L!=num_R){
	        cout<<"ERROR::[D1+]:The elements of lvalue and rvalue *MUST* be same! -> ";
	        cout<<"L=["<<L.GetNum()<<"], R=["<<R.GetNum()<<"]"<<endl;
	        exit(EXIT_FAILURE);
	    }
	    D1<T> out(num_L);
	    for(size_t i=0;i<num_L;++i){
	        out[i]=L[i]+R[i];
	    }
		return out;
	}

	template<typename T>
	const D1<T> operator-(const D1<T>& L,const D1<T>& R){
	    size_t num_L=L.GetNum();
	    size_t num_R=R.GetNum();
	    if(num_L!=num_R){
	        cout<<"ERROR::[D1+]:The elements of lvalue and rvalue *MUST* be same! -> ";
	        cout<<"L=["<<L.GetNum()<<"], R=["<<R.GetNum()<<"]"<<endl;
	        exit(EXIT_FAILURE);
	    }
	    D1<T> out(num_L);
	    for(size_t i=0;i<num_L;++i){
	        out[i]=L[i]-R[i];
	    }
		return out;
	}

	template<typename T>
	const D1<T> operator*(const D1<T>& L,const D1<T>& R){
	    size_t num_L=L.GetNum();
	    size_t num_R=R.GetNum();
	    if(num_L!=num_R){
	        cout<<"ERROR::[D1+]:The elements of lvalue and rvalue *MUST* be same! -> ";
	        cout<<"L=["<<L.GetNum()<<"], R=["<<R.GetNum()<<"]"<<endl;
	        exit(EXIT_FAILURE);
	    }
	    D1<T> out(num_L);
	    for(size_t i=0;i<num_L;++i){
	        out[i]=L[i]*R[i];
	    }
		return out;
	}

	template<typename T>
	const D1<T> operator/(const D1<T>& L,const D1<T>& R){
	    size_t num_L=L.GetNum();
	    size_t num_R=R.GetNum();
	    if(num_L!=num_R){
	        cout<<"ERROR::[D1+]:The elements of lvalue and rvalue *MUST* be same! -> ";
	        cout<<"L=["<<L.GetNum()<<"], R=["<<R.GetNum()<<"]"<<endl;
	        exit(EXIT_FAILURE);
	    }
	    D1<T> out(num_L);
	    for(size_t i=0;i<num_L;++i){
	        out[i]=L[i]/R[i];
	    }
		return out;
	}

	
	template<typename T>
	const D1<T> operator+(const D1<T>& L,const T& R){
	    size_t num_L=L.GetNum();
	    D1<T> out(num_L);
	    for(size_t i=0;i<num_L;++i){
	        out[i]=L[i]+R;
	        //cout<<"out[i]="<<out[i]<<endl;
	    }
		return out;
	}

	template<typename T>
	const D1<T> operator-(const D1<T>& L,const T& R){
	    size_t num_L=L.GetNum();
	    D1<T> out(num_L);
	    for(size_t i=0;i<num_L;++i){
	        out[i]=L[i]-R;
	        //cout<<"out[i]="<<out[i]<<endl;
	    }
		return out;
	}

    template<typename T>
	const D1<T> operator*(const D1<T>& L,const T& R){
	    size_t num_L=L.GetNum();
	    D1<T> out(num_L);
	    for(size_t i=0;i<num_L;++i){
	        out[i]=L[i]*R;
	        //cout<<"out[i]="<<out[i]<<endl;
	    }
		return out;
	}

	template<typename T>
	const D1<T> operator/(const D1<T>& L,const T& R){
	    size_t num_L=L.GetNum();
	    D1<T> out(num_L);
	    for(size_t i=0;i<num_L;++i){
	        out[i]=L[i]/R;
	        //cout<<"out[i]="<<out[i]<<endl;
	    }
		return out;
	}

	
	
	template<typename T>
	const D1<T> operator+(const T& L,const D1<T>& R){
	    size_t num_R=R.GetNum();
	    D1<T> out(num_R);
	    for(size_t i=0;i<num_R;++i){
	        out[i]=L+R[i];
	        //cout<<"out[i]="<<out[i]<<endl;
	    }
		return out;
	}
	
//	template<typename T>
//	const D1<T> operator-(const T& L,const D1<T>& R){
//	    size_t num_R=R.GetNum();
//	    D1<T> out(num_R);
//	    for(size_t i=0;i<num_R;++i){
//	        out[i]=L-R[i];
//	        //cout<<"out[i]="<<out[i]<<endl;
//	    }
//		return out;
//	}
	
    template<typename T>
	const D1<T> operator*(const T& L,const D1<T>& R){
	    size_t num_R=R.GetNum();
	    D1<T> out(num_R);
	    for(size_t i=0;i<num_R;++i){
	        out[i]=L*R[i];
	        //cout<<"out[i]="<<out[i]<<endl;
	    }
		return out;
	}
	
//	template<typename T>
//	const D1<T> operator/(const T& L,const D1<T>& R){
//	    size_t num_R=R.GetNum();
//	    D1<T> out(num_R);
//	    for(size_t i=0;i<num_R;++i){
//	        out[i]=L[i]/R;
//	        //cout<<"out[i]="<<out[i]<<endl;
//	    }
//		return out;
//	}

    //
    // Set
    //
    template<typename T>
    void D1<T>::Set(const T* series,const size_t n){
        _n=n;
        for(size_t i=0;i<_n;++i){
            _v[i]=series[i];
        }
    }

    //
	// Misc.
	//
	template<typename T>
	void D1<T>::SetZero(){
		for(size_t i=0;i<_n;i++){
			_v[i]=0;
		}
	}

	template<typename T>
	size_t D1<T>::GetNum() const{
		return _n;
	}
	
	template<typename T>
	void D1<T>::PrintAll(){
		size_t i;
		cout<<"[";
		for(i=0;i<_n-1;i++){
			cout<<_v[i]<<",";
		}
		cout<<_v[i]<<"]"<<endl;
	}

	template<typename T>
	void D1<T>::Print(){
		size_t m=(_n > 20)? 20:_n;
		size_t i;
		
		cout<<"[";
		for(i=0;i<m-1;i++){
			cout<<_v[i]<<",";
		}
		cout<<_v[i]<<"]"<<endl;
	}
	
	template<typename T>
	void D1<T>::Print()const{
		size_t m=(_n > 20)? 20:_n;
		size_t i;
		
		cout<<"[";
		for(i=0;i<m-1;i++){
			cout<<_v[i]<<",";
		}
		cout<<_v[i]<<"]"<<endl;
	}

	template<typename T>
	void D1<T>::Print(const size_t start_idx,const size_t end_idx){
		size_t i;

		//cout<<"Type : "<<typeid(T).name()<<"["<<_type<<"]"<<endl;
		if( ((end_idx - start_idx) > _n-1) || (start_idx > _n-1) ||
		    (end_idx > _n-1) || (end_idx < start_idx) ){
            cout<<"ERROR::[D1:Print]:Input:[ start_idx = "<<start_idx<<", end_idx = "<<end_idx
				<<", _n = "<<_n<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
		cout<<"[";
		for(i=start_idx;i<end_idx;i++){
			cout<<_v[i]<<",";
		}
		cout<<_v[i]<<"]"<<endl;
	}

	template<typename T>
	void D1<T>::WriteBinary(const char* filename){
	    //FILE* out;
	    //out=fopen(filename,"wb");
	    //fwrite(_v,sizeof(T),_n,out);
	    //fclose(out);
	    ofstream fout(filename,ios::binary);
	    if(!fout.good()){
	        cout<<"ERROR::[D1<T>::WriteBinary] The output filename is error! -> ";
	        cout<<filename<<endl;
	        exit(EXIT_FAILURE);
	    }
	    fout.write(reinterpret_cast<char*>(_v),sizeof(T)*_n);
	    fout.close();
	}
	
	template<typename T>
	void D1<T>::WriteBinary(const char* filename)const{
	    //FILE* out;
	    //out=fopen(filename,"wb");
	    //fwrite(_v,sizeof(T),_n,out);
	    //fclose(out);
	    ofstream fout(filename,ios::binary);
	    if(!fout.good()){
	        cout<<"ERROR::[D1<T>::WriteBinary] The output filename is error! -> ";
	        cout<<filename<<endl;
	        exit(EXIT_FAILURE);
	    }
	    fout.write(reinterpret_cast<char*>(_v),sizeof(T)*_n);
	    fout.close();
	}

	template<typename T>
	void D1<T>::WriteASCII(const char* filename, const int Precision){
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
		for(size_t i=0;i<_n;++i){
			fout<<std::setprecision(Precision)<<std::fixed<<_v[i]<<endl;
		}
		fout.close();
    }

	template<typename T>
	void D1<T>::Crop(const size_t start_idx, const size_t end_idx){
		if( ((end_idx - start_idx) > _n-1) || (start_idx > _n-1) ||
		   (end_idx > _n-1) || (end_idx < start_idx) ){
            cout<<"ERROR::[D1:Crop]:Input:[ start_idx = "<<start_idx<<", end_idx = "<<end_idx
			<<", _n = "<<_n<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
		
		_n = end_idx - start_idx + 1L;
		_v = &(_v[start_idx]);
	}

	template<typename T>
	D1<T> D1<T>::GetRange(const size_t start_idx, const size_t end_idx){
		if( ((end_idx - start_idx) > _n-1) || (start_idx > _n-1) ||
			(end_idx > _n-1) || (end_idx < start_idx) ){
			cout<<"ERROR::[D1:Crop]:Input:[ start_idx = "<<start_idx<<", end_idx = "<<end_idx
				<<", _n = "<<_n<<"]"<<endl;
			exit(EXIT_FAILURE);
		}

		D1<T> out(end_idx - start_idx + 1L);
		for(size_t i=0;i<out.GetNum();++i){
			out[i] = _v[start_idx + i];
		}

		return out;
	}

	template<typename T>
	D1<T>& D1<T>::Square(){
		D1<T> out(_n);
		for(size_t i=0;i<_n;++i){
			out[i] = _v[i]*_v[i];
		}
		return out;
	}
	
	template<typename T>
	void D1<T>::SelfSquare(){
		for(size_t i=0;i<_n;++i){
			_v[i] = _v[i]*_v[i];
		}
	}
	
	template<typename T>
	void D1<T>::Indgen(){
		for(size_t i=0;i<_n;++i){ _v[i] = T(i); }
	}
	
	template<typename T>
	void D1<T>::clear(){
		D1<T>();
	}
	
	template<typename T>
	void D1<T>::Swap(){
		for(size_t i=0;i<_n/2;++i){
			std::swap(_v[i], _v[_n-1-i]);
		}
	}
}

#endif // D1_H_INCLUDED
