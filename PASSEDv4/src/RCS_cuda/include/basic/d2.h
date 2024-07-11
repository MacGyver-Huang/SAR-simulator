#ifndef D2_H_INCLUDED
#define D2_H_INCLUDED

#include <basic/def_prefunc.h>
#include <sar/def.h>
#include <basic/d1.h>
#include <basic/vec.h>
#include <basic/opt.h>
#include <algorithm>


namespace d2{
    using namespace d1;
    using namespace vec;
    // ==============================================
	// Series (2D Matrix)
	// ==============================================
	template<typename T>
	class D2
	{
		public:
			// Constructure
			D2();//:_m(0),_n(0),_v(D1<T>(_m*_n)){};
			D2(const size_t m,const size_t n);//:_typenamem(m),_n(n),_v(_m*_n){};
			D2(const T* series,const size_t m,const size_t n);
			D2(const D1<T>& series,const size_t m,const size_t n);
			D2(const VEC<T>& in,const size_t m,const size_t n);
			D2(const D2<T>& in);
			// Operator
			T* operator[](const size_t i){return _v[i];};
			T* operator[](const size_t i)const{return _v[i];};
			D2<T>& operator=(const D2<T>& in);
			template<typename T2,typename T3> friend D2<T2> operator*(const D2<T2>& L,const D2<T3>& R);
			template<typename T2> friend D2<T2> operator*(const D2<T2>& L,const D2<T2>& R);
			template<typename T2> friend D2<T2> operator*(const D2<T2>& L,const T2& R);
			template<typename T2> friend D2<CPLX<T2> > operator*(const D2<T2>& L,const D2<CPLX<T2> >& R);
			template<typename T2> friend D2<T2> operator+(const D2<T2>& L,const D2<T2>& R);
            template<typename T2> friend D2<T2> operator-(const T2& L,const D2<T2>& R);
			// Transpose
			void SelfTranspose();
			D2<T> Transpose();
			// Invert
			double determinant(const D2<T>& a, double k);
			D2<T> cofactor(D2<T>& num, T f);
			D2<T> transpose(const D2<T>& num, const D2<T>& fac, T r);
			D2<T> Invert();
			// Get
			size_t GetM()const{return _m;};
			size_t GetN()const{return _n;};
			D1<size_t> GetDim()const{ size_t dim[2]={_m,_n}; return D1<size_t>(dim,2); };
			TYPE GetType()const{ return _type; };
			//T& GetVal(const size_t j_m,const size_t i_n)const{return _data[j_m*_n+i_n];};
			D1<T> GetColumn(const size_t i_n);
			D1<T> GetRow(const size_t j_m);
			const D1<T> GetColumn(const size_t i_n)const;
			const D1<T> GetRow(const size_t j_m)const;
			// Set
			//void SetVal(const size_t i_m,const size_t j_n,const T& val){_v[i_m][j_n]=val;};
			void SetVal(const size_t start_m,const size_t start_n,const size_t num,const T* val);
			void SetVal(const size_t start_m,const size_t start_n,const D1<T>& val);
			void SetZero(){ _data.SetZero(); };
			void SetColumn(const D1<T>& in, const size_t i_n);
			void SetRow(const D1<T>& in, const size_t j_m);
			// Misc.
			void Print();
			void WriteASCII(const char* filename, const int Precision=4);
			void WriteBinary(const char* filename);
			void WriteBMP(const char* filename);
			T det();
			D2<T>& Square();
			void SelfSquare();
			void Indgen();
			void clear();
			// Get data pointer
			T* GetPtr() const{return _data.GetPtr();};
	private:
			void _error();
            void _init(const size_t m,const size_t n);
			size_t _m,_n;//for math.algebra(_m,_n)
			TYPE _type;
			d1::D1<T> _data;
			d1::D1<T*> _v;
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
	void D2<T>::_error(){
		if(_type == UNKNOW){
			cout<<"ERROR::[D2() data type MUST be char, short, int, float, double, CPLX<float>, CPLX<double> !"<<endl;
			cout<<typeid(T).name()<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
			exit(EXIT_FAILURE);
		}
	}

	//<editor-fold desc="Description">
    template<typename T>
    void D2<T>::_init(const size_t m,const size_t n){
		_type = _data.GetType();
		_error();
        _m=m;
        _n=n;
        _data=D1<T>(_m*_n);
        _v=D1<T*>(_m);
        for(size_t i=0;i<_m;++i){
            _v[i]=&(_data[i*_n]);
        }
    }
	//</editor-fold>


    //
    // Constructor
    //
    template<typename T>
    D2<T>::D2():_m(0),_n(0){
        _init(_m,_n);
    }

    template<typename T>
    D2<T>::D2(const size_t m,const size_t n):_m(m),_n(n){
        _init(_m,_n);
    }


    template<typename T>
    D2<T>::D2(const T* series,const size_t m,const size_t n):_m(m),_n(n){//,_v(_m*_n){
        _init(m,n);
        for(size_t i=0;i<_m*_n;++i){
            _data[i]=series[i];
        }
        //_v.Set(series,_n*_m);
    }

    template<typename T>
    D2<T>::D2(const D1<T>& series,const size_t m,const size_t n):_m(m),_n(n){//,_v(_m*_n){
        _init(m,n);
        for(size_t i=0;i<_m*_n;++i){
            _data[i]=series[i];
        }
    }

    template<typename T>
    D2<T>::D2(const VEC<T>& in,const size_t m,const size_t n){
    /*
     Purpose:
        Convert VEC class to column or row 1-D D2 class.
    */
        _init(m,n);
        _data[0]=in.x();
        _data[1]=in.y();
        _data[2]=in.z();
    }

    template<typename T>
    D2<T>::D2(const D2<T>& in){
        if(this != &in){
            _init(in._m,in._n);
            for(size_t i=0;i<_m*_n;++i){
                _data[i]=in._data[i];
            }
        }
    }

    //
	// Operator overloading
	//
	template<typename T1, typename T2>
	D2<T1> operator*(const D2<T1>& L,const D2<T2>& R){
		/*
		 Purpose:
		 2D array multiplication
		 */
        size_t m_L=L.GetM(),n_L=L.GetN();
        size_t m_R=R.GetM(),n_R=R.GetN();
        if(n_L != m_R){
            cout<<"ERROR::[D2::operator*]:Martix Matliply error! -> ";
            cout<<"L=["<<m_L<<"x"<<n_L<<"], R=["<<m_R<<"x"<<n_R<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        D2<T1> out(m_L,n_R);
        for(size_t i=0;i<m_L;++i){
            for(size_t j=0;j<n_R;++j){
                out[i][j]=0;
                for(size_t k=0;k<m_R;++k){
                    out[i][j] += ( L[i][k]*R[k][j] );
                }
            }
        }
        return out;
		//return D2<T>(series,m,n);
	}
	
	template<typename T>
	D2<T> operator*(const D2<T>& L,const D2<T>& R){
    /*
	 Purpose:
        2D array multiplication
	*/
        size_t m_L=L.GetM(),n_L=L.GetN();
        size_t m_R=R.GetM(),n_R=R.GetN();
        if(n_L != m_R){
            cout<<"ERROR::[D2::operator*]:Martix Matliply error! -> ";
            cout<<"L=["<<m_L<<"x"<<n_L<<"], R=["<<m_R<<"x"<<n_R<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        D2<T> out(m_L,n_R);
        for(size_t i=0;i<m_L;++i){
            for(size_t j=0;j<n_R;++j){
                out[i][j]=0;
                for(size_t k=0;k<m_R;++k){
                    out[i][j] += ( L[i][k]*R[k][j] );
                }
            }
        }
        return out;
		//return D2<T>(series,m,n);
	}

	template<typename T>
	D2<T> operator*(const D2<T>& L,const T& R){
	/*
	 Purpose:
		Multiple a scalar
	*/
		D2<T> out(L.GetM(),L.GetN());
        for(size_t i=0;i<L.GetM();++i){
            for(size_t j=0;j<L.GetN();++j){
                out[i][j]=L[i][j]*R;
            }
        }
        return out;
	}

	template<typename T>
	D2<CPLX<T> > operator*(const D2<T>& L,const D2<CPLX<T> >& R){
	/*
	 Purpose:
		 Multiple a scalar
	 */
		size_t m_L=L.GetM(),n_L=L.GetN();
        size_t m_R=R.GetM(),n_R=R.GetN();
        if(n_L != m_R){
            cout<<"ERROR::[D2::operator*]:Martix Matliply error! -> ";
            cout<<"L=["<<m_L<<"x"<<n_L<<"], R=["<<m_R<<"x"<<n_R<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        D2<CPLX<T> > out(m_L,n_R);
        for(size_t i=0;i<m_L;++i){
            for(size_t j=0;j<n_R;++j){
                out[i][j] = CPLX<T>(0,0);
                for(size_t k=0;k<m_R;++k){
                    out[i][j] += CPLX<T>( L[i][k]*R[k][j].r(), L[i][k]*R[k][j].i() );
                }
            }
        }
        return out;
	}


	template<typename T>
	D2<T> operator+(const D2<T>& L,const D2<T>& R){
    /*
	 Purpose:
        2D array add
	*/
        size_t m_L=L.GetM(),n_L=L.GetN();
        size_t m_R=R.GetM(),n_R=R.GetN();
        if((m_L != m_R)&&(n_L != n_R)){
            cout<<"ERROR::[D2::operator+]:Input dimension MUST be same! -> ";
            cout<<"L=["<<m_L<<"x"<<n_L<<"], R=["<<m_R<<"x"<<n_R<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        D2<T> out(m_L,n_R);
        for(size_t i=0;i<m_L*n_R;++i){
            out._data[i]=L._data[i]+R._data[i];
        }
        return out;
	}

    template<typename T>
    D2<T> operator-(const T& L,const D2<T>& R){
    /*
     Purpose:
        2D array add
    */
        size_t m_R=R.GetM(),n_R=R.GetN();
        D2<T> out(m_R,n_R);
        for(size_t i=0;i<m_R*n_R;++i){
            out._data[i]=L-R._data[i];
        }
        return out;
    }

	template<typename T>
	D2<T>& D2<T>::operator=(const D2<T>& in){
	    if(this != &in){
			/*
            // 1: allocate new memory and copy the elements
            T* new_v = new T[in._n];
            //std::copy(in._v, in._v + in._n, new_v);
            for(size_t i=0;i<in._n;++i){
                new_v[i]=in._v[i];
            }
            // 2: deallocate old memory
            delete [] _v;
            // 3: assign the new memory to the object
            _v = new_v;
            _n = in._n;
			*/
			_init(in._m,in._n);
			for(size_t i=0;i<_m*_n;++i){
				_data[i]=in._data[i];
			}
        }
        return *this;
	}

	//
	// Tranpose
	//
	template<typename T>
	void D2<T>::SelfTranspose(){
	/*
	 Purpose:
	 Transpose matrix
	 */
	
		if(_n == _m){ // swap
			for(size_t i=1;i<_m;++i){
				for(size_t j=0;j<_n-1;++j){
					std::swap(_data[i*_n+j],_data[j*_n+i]);
				}
			}
		}else{
			opt::_fnMatrixTransposeInplace(_data.GetPtr(), _m, _n);
			std::swap(_m,_n);
		}
	}
	
	template<typename T>
	D2<T> D2<T>::Transpose(){
	/*
	 Purpose:
		Transpose matrix
	*/
		D2<T> out(_n,_m);
		T* tmp = out.GetPtr();
		
#ifdef _DEBUG
		cout<<tmp<<endl;
		cout<<out.GetPtr()<<endl;
#endif
		
		size_t count=0;
		for(size_t i=0;i<_n;++i){
			for(size_t j=0;j<_m;++j){
				tmp[count] = _data[j*_n+i];
				count++;
			}
		}
		return out;
	}

	template<typename T>
	double D2<T>::determinant(const D2<T>& a, double k) {
		double s = 1, det = 0;
		size_t M = a.GetM();
		size_t N = a.GetN();
		D2<double> b(M, N);
		int i, j, m, n, c;
		if (k == 1){
			return (a[0][0]);
		} else {
			det = 0;
			for (c = 0; c < k; c++) {
				m = 0;
				n = 0;
				for (i = 0;i < k; i++) {
					for (j = 0 ;j < k; j++) {
						b[i][j] = 0;
						if (i != 0 && j != c) {
							b[m][n] = a[i][j];
							if (n < (k - 2)){
								n++;
							}else{
								n = 0;
								m++;
							}
						}
					}
				}
				det = det + s * (a[0][c] * this->determinant(b, k - 1));
				s = -1 * s;
			}
		}

		return (det);
	}

	template<typename T>
	D2<T> D2<T>::cofactor(D2<T>& num, T f){

		size_t M = this->GetM();
		size_t N = this->GetN();
		D2<double> b(M, N), fac(M, N);

		int p, q, m, n, i, j;
		for (q = 0;q < f; q++){
			for (p = 0;p < f; p++){
				m = 0;
				n = 0;
				for (i = 0;i < f; i++){
					for (j = 0;j < f; j++){
						if (i != q && j != p){
							b[m][n] = num[i][j];
							if (n < (f - 2)){
								n++;
							}else{
								n = 0;
								m++;
							}
						}
					}
				}
				fac[q][p] = pow(-1, q + p) * determinant(b, f - 1);
			}
		}
		return transpose(num, fac, f);
	}

	template<typename T>
	D2<T> D2<T>::transpose(const D2<T>& num, const D2<T>& fac, T r){

		size_t M = num.GetM();
		size_t N = num.GetN();

		int i, j;
		D2<T> inverse(M, N), b(M, N);
		double d;

		for (i = 0;i < r; i++){
			for (j = 0;j < r; j++){
				b[i][j] = fac[j][i];
			}
		}
		d = determinant(num, r);
		for (i = 0;i < r; i++){
			for (j = 0;j < r; j++){
				inverse[i][j] = b[i][j] / d;
			}
		}
//		printf("\n\n\nThe inverse of matrix is : \n");
//
//		for (i = 0;i < r; i++){
//			for (j = 0;j < r; j++){
//				printf("\t%f", inverse[i][j]);
//			}
//			printf("\n");
//		}
		return inverse;
	}

	template<typename T>
	D2<T> D2<T>::Invert(){

		size_t M = this->GetM();
		size_t N = this->GetN();

		if(M != N){
			cerr<<"ERROR::D2:Invet: This is not square matrix"<<endl;
			exit(EXIT_FAILURE);
		}

		// Duplicate
		D2<T> P(this->_data, M, N);

		T d = determinant(P, M);
		D2<T> inv(M,N);
		if (d == 0){
			printf("\nInverse of Entered Matrix is not possible\n");
		}else{
			inv = cofactor(P, M);
		}

		return inv;
	}


    //
    // Get
    //
    template<typename T>
    D1<T> D2<T>::GetColumn(const size_t i_n){
    /*
     Purpose:
        Get a clomun vector(vertical)
    */
        D1<T> out(_m);
        if(i_n > _n-1){
            cout<<"ERROR::[D2:GetColumn]:Input:["<<i_n<<"]exceed the range:[0~"<<_n-1<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        for(size_t i=0;i<_m;++i){
            out[i]=_v[i][i_n];
        }
        return out;
    }

    template<typename T>
    D1<T> D2<T>::GetRow(const size_t j_m){
    /*
     Purpose:
        Get a clomun vector(horizontial)
    */
        D1<T> out(_n);
        if(j_m > _m-1){
            cout<<"ERROR::[D2:GetRow]:["<<j_m<<"] exceed the range:[0~"<<_m-1<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        for(size_t j=0;j<_n;++j){
            out[j]=_v[j_m][j];
        }
        return out;
    }

    template<typename T>
    const D1<T> D2<T>::GetColumn(const size_t i_n)const{
    /*
     Purpose:
        Get a clomun vector(vertical)
    */
        D1<T> out(_m);
        if(i_n > _n-1){
            cout<<"ERROR::[D2:GetColumn]:Input:["<<i_n<<"]exceed the range:[0~"<<_n-1<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        for(size_t i=0;i<_m;++i){
            out[i]=_v[i][i_n];
        }
        return out;
    }

    template<typename T>
    const D1<T> D2<T>::GetRow(const size_t j_m)const{
    /*
     Purpose:
        Get a clomun vector(horizontial)
    */
        D1<T> out(_n);
        if(j_m > _m-1){
            cout<<"ERROR::[D2:GetRow]:["<<j_m<<"] exceed the range:[0~"<<_m-1<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        for(size_t j=0;j<_n;++j){
            out[j]=_v[j_m][j];
        }
        return out;
    }


    //
    // Set
    //
    template<typename T>
    void D2<T>::SetVal(const size_t start_m,const size_t start_n,const size_t num,const T* val){
        for(size_t i=0;i<num;++i){
            _data[start_m*_n+start_n+i]=val[i];
        }
    }

    template<typename T>
    void D2<T>::SetVal(const size_t start_m,const size_t start_n,const D1<T>& val){
        size_t num=val.GetNum();
        for(int i=0;i<num;++i){
            _data[start_m*_n+start_n+i]=val[i];
        }
    }
	
	template<typename T>
    void D2<T>::SetColumn(const D1<T>& in, const size_t i_n){
		/*
		 Purpose:
		 Get a clomun vector(vertical)
		 */
		if(in.GetNum() != _m){
			cout<<"ERROR::[D2:SetColumn]:Input:["<<in.GetNum()<<"]exceed the range:["<<_m<<"]"<<endl;
            exit(EXIT_FAILURE);
		}
        if(i_n > _n-1){
            cout<<"ERROR::[D2:SetColumn]:Input:["<<i_n<<"]exceed the range:[0~"<<_n-1<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        for(size_t i=0;i<_m;++i){
            _v[i][i_n] = in[i];
        }
    }
	
    template<typename T>
    void D2<T>::SetRow(const D1<T>& in, const size_t j_m){
		/*
		 Purpose:
		 Get a clomun vector(horizontial)
		 */
        if(in.GetNum() != _n){
			cout<<"ERROR::[D2:SetColumn]:Input:["<<in.GetNum()<<"]exceed the range:["<<_n<<"]"<<endl;
            exit(EXIT_FAILURE);
		}
        if(j_m > _m-1){
            cout<<"ERROR::[D2:GetRow]:["<<j_m<<"] exceed the range:[0~"<<_m-1<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        for(size_t j=0;j<_n;++j){
            _v[j_m][j] = in[j];
        }
    }

    //
    // Misc.
    //
    template<typename T>
	void D2<T>::Print(){
	    size_t lim=20;
	    size_t m=_m;
	    size_t n=(_n > lim)? lim:_n;
		
	    if((_m > lim)&(_n > lim)){
	        cout<<"WARRING:: [D2::Print()] : Partial #column(m) & #row(n)"<<endl;
	        m=n=lim;
	    }else if(_m > lim){
	        cout<<"WARRING:: [D2::Print()] : Partial #column(m)"<<endl;
	        m=lim;
	    }else if(_n > lim){
	        cout<<"WARRING:: [D2::Print()] : Partial #row(n)"<<endl;
	        n=lim;
	    }
		
		
		//cout<<"Type : "<<typeid(T).name()<<"["<<_type<<"]"<<endl;
		for(size_t i=0;i<m;++i){
		    _data.Print(i*_n,i*_n+n-1);
		}

	}

    template<typename T>
    void D2<T>::WriteASCII(const char* filename, const int Precision){
    	ofstream fout;
		fout.open(filename);
		if(fout.fail()){
			cout<<"ERROR::[WriteAscii]Input filename! -> ";
			cout<<filename<<endl;
			exit(EXIT_FAILURE);
		}
		for(size_t i=0;i<_m;++i){
			for(size_t j=0;j<_n;++j){
				fout<<std::setprecision(Precision)<<std::fixed<<_data[i*_n+j];
				if(j == _n-1){
					fout<<endl;
				}else{
					fout<<"\t";
				}
			}
		}
		fout.close();
    }

    template<typename T>
	void D2<T>::WriteBinary(const char* filename){
	    _data.WriteBinary(filename);
	}
	
	template<typename T>
	T D2<T>::det(){
		/*
		 Purpose:
		 Caliculate the determinant of input matrix.
		 */
        if( (_m != 3L)&&(_n != 3L) ){
            cout<<"ERROR::[vec(det)]:Only supported 3 by 3 matrix -> ";
            //cout<<"["<<this.GetM()<<"x"<<this.GetN()<<"]"<<endl;
			cout<<"["<<_m<<"x"<<_n<<"]"<<endl;
            exit(EXIT_FAILURE);
        }
        return _data[0]*_data[4]*_data[8] +
		_data[2]*_data[3]*_data[7] +
		_data[1]*_data[5]*_data[6] -
		_data[2]*_data[4]*_data[6] -
		_data[1]*_data[3]*_data[8] -
		_data[0]*_data[5]*_data[7];
	}
	
	template<typename T>
	D2<T>& D2<T>::Square(){
		D2<T> out(_m,_n);
		for(size_t i=0;i<_m*_n;++i){
			(*(out.GetPtr()+i)) = _data[i] * _data[i];
			//(*(out.GetPtr()+i)) = (*(in_ptr+i)) * (*(in_ptr+i));
		}
		return out;
	}
	
	template<typename T>
	void D2<T>::SelfSquare(){
		_data.SelfSquare();
	}
	
	template<typename T>
	void D2<T>::Indgen(){
		_data.Indgen();
	}

	template<typename T>
	void D2<T>::clear(){
		this->~D2<T>();
	}
}


#endif // D2_H_INCLUDED
