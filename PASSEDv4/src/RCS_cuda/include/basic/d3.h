#ifndef D3_H_INCLUDED
#define D3_H_INCLUDED

#include <sar/def.h>
#include <basic/d1.h>
#include <basic/d2.h>
#include <basic/vec.h>


namespace d3{
    using namespace d1;
	using namespace d2;
    using namespace vec;
    // ==============================================
	// Series (3D Matrix)
	// ==============================================
	template<typename T>
	class D3
	{
		public:
			// Constructure
			D3();//:_m(0),_n(0),_v(D1<T>(_m*_n)){};
			D3(const size_t p,const size_t q,const size_t r);
			D3(const T* series,const size_t p,const size_t q,const size_t r,const string type);
			D3(const D3<T>& in);
			// Operator
			D1<T*> operator[](const size_t i){return _v[i];};
			D1<T*> operator[](const size_t i)const{return _v[i];};
			D3<T>& operator=(const D3<T>& in);
			// Set
			void SetZero(){ _data.SetZero(); };
			// Get
			size_t GetP()const{return _p;};
			size_t GetQ()const{return _q;};
			size_t GetR()const{return _r;};
			D1<size_t> GetDim()const{ size_t dim[3]={_p,_q,_r}; return D1<size_t>(dim,3); };
			int GetType()const{ return _type; };
			D2<T> GetD2atP(const size_t P){
				if(P > _p){
					cout<<"ERROR::D3:Input number P("<<P<<") is out of range, _p("<<_p<<"!"<<endl;
					exit(EXIT_FAILURE);
				}
				return D2<T>(_data.GetPtr() + P * (_q*_r), _q, _r);
			}
			// Misc.
			void Print();
			void WriteBinary(const char* filename);
			D3<T>& Square();
			void SelfSquare();
			void Indgen();
			void clear();
			// Set & Get Pointer
			T* GetPtr() const{return _data.GetPtr();};
		private:
			void _error();
            void _init(const size_t p,const size_t q,const size_t r);
			size_t _p,_q,_r;
			TYPE _type;
			string _inter;
			D1<T> _data;
			D1<D1<T*> > _v;
	};
	// ==============================================
	// namespace declare
	// ==============================================

	// Implement ********************************************************
    //
	// Private
    //
	
	//
	// Error
	//
	template<typename T>
	void D3<T>::_error(){
		if(_type == UNKNOW){
			cout<<"ERROR::[D3() data type MUST be char, short, int, float, double, CPLX<float>, CPLX<double> !"<<endl;
			cout<<typeid(T).name()<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
			exit(EXIT_FAILURE);
		}
	}
	
    template<typename T>
    void D3<T>::_init(const size_t p,const size_t q,const size_t r){
        _p=p;
        _q=q;
		_r=r;
		_inter="BIP";
		_type = _data.GetType();
	

		_data=D1<T>(_p*_q*_r);
		_error();
		
		_v=D1<D1<T*> >(_p);
		

		// BIP
		for(size_t i=0;i<_p;++i){
			_v[i] = D1<T*>(_q);
			for(size_t j=0;j<_q;++j){
				_v[i][j]=&(_data[i*(_q*_r)+j*_r]);
			}
		}
    }


    //
    // Constructor
    //
    template<typename T>
    D3<T>::D3():_p(0),_q(0),_r(0){
        _init(_p,_q,_r);
    }

    template<typename T>
    D3<T>::D3(const size_t p,const size_t q,const size_t r){
		_init(p,q,r);
    }

	template<typename T>
    D3<T>::D3(const T* series,const size_t p,const size_t q,const size_t r,const string interleave){
		_init(p,q,r);
		_inter=interleave;

		if(_inter == "BIP"){
			// BIP
			size_t num = _p*_q*_r;
			for(size_t i=0;i<num;++i){
				_data[i]=series[i];
			}
		}else if(_inter == "BIL"){
			// BIL
			size_t count=0;
			for(size_t i=0;i<_p;++i){
				for(size_t j=0;j<_q;++j){
					for(size_t k=0;k<_r;++k){
						_data[count] = series[i*(_q*_r) + k*_q + j];
						count++;
					}
				}
			}
		}else{
			// BSQ
			size_t count=0,num=_p*_q;
			for(size_t i=0;i<num;++i){
				for(size_t k=0;k<_r;++k){
					_data[count] = series[k*num+i];
					count++;
				}
			}
		}
    }

	template<typename T>
    D3<T>::D3(const D3<T>& in){
        if(this != &in){
			_init(in._p,in._q,in._r);
            for(size_t i=0;i<_p*_q*_r;++i){
                _data[i]=in._data[i];
            }
        }
    }

	//
	// Operator overloading
	//

	template<typename T>
	D3<T>& D3<T>::operator=(const D3<T>& in){
	    if(this != &in){
			_init(in._p,in._q,in._r);
			for(size_t i=0;i<_p*_q*_r;++i){
				_data[i]=in._data[i];
			}
        }
        return *this;
	}

	//
    // Misc.
    //
    template<typename T>
	void D3<T>::Print(){
	    size_t lim=20;
	    size_t p=_p;
	    size_t q=_q;
		size_t r=_r;


	    if((_p > lim)&(_q > lim)&(_r > lim)){
			cout<<"WARRING:: [D3::Print()] : Partial #column(p) & #row(q) & #band(r)"<<endl;
	        p=q=r=lim;
		}else if((_p > lim)&(_q > lim)){
			cout<<"WARRING:: [D3::Print()] : Partial #column(p) & #row(q))"<<endl;
	        p=q=lim;
		}else if((_p > lim)&(_r > lim)){
			cout<<"WARRING:: [D3::Print()] : Partial #column(p) & #band(r))"<<endl;
	        p=r=lim;
		}else if((_q > lim)&(_r > lim)){
			cout<<"WARRING:: [D3::Print()] : Partial #row(q) & #band(r))"<<endl;
	        q=r=lim;
	    }else if(_p > lim){
	        cout<<"WARRING:: [D3::Print()] : Partial #column(p)"<<endl;
	        p=lim;
	    }else if(_q > lim){
	        cout<<"WARRING:: [D3::Print()] : Partial #row(q)"<<endl;
	        q=lim;
		}else if(_r > lim){
	        cout<<"WARRING:: [D3::Print()] : Partial #band(r)"<<endl;
	        r=lim;
	    }

		
		cout<<"Type : "<<typeid(T).name()<<"["<<_type<<"] & Interleave : "<<_inter<<endl;
		size_t i,j,k;
		for(k=0;k<r;++k){
			cout<<"band"<<k<<endl;
			for(i=0;i<p;++i){
				cout<<"[";
				for(j=0;j<q-1;++j){
					cout<<_data[i*(_q*_r)+j*_r+k]<<",";
				}
				cout<<_data[i*(_q*_r)+j*_r+k]<<"]"<<endl;
			}
		}
		cout<<endl;
	}

	template<typename T>
	void D3<T>::WriteBinary(const char* filename){
		_data.WriteBinary(filename);
	}
	
	template<typename T>
	D3<T>& D3<T>::Square(){
		D3<T> out(_p,_q,_r);
		for(size_t i=0;i<_p*_q*_r;++i){
			(*(out.GetPtr()+i)) = _data[i] * _data[i];
		}
		return out;
	}
	
	template<typename T>
	void D3<T>::SelfSquare(){
		_data.SelfSquare();
	}
	
	template<typename T>
	void D3<T>::Indgen(){
		_data.Indgen();
	}
	
	template<typename T>
	void D3<T>::clear(){
		D3<T>();
	}
}

#endif // D2_H_INCLUDED
