#ifndef POLY_H_INCLUDED
#define POLY_H_INCLUDED

#include <sar/def.h>
#include <basic/vec.h>
#include <basic/d1.h>

namespace poly{
    using namespace vec;
    using namespace d1;
    // ==============================================
	// Polygon class
	// ==============================================
	template<typename T>
	class POLY
	{
	    typedef D1<VEC<T> > D1VEC;
		public:
            POLY():_num(0){};
            POLY(const long num):_num(num),_p(D1VEC(3*_num)){};
            //POLY(const POLY<T>& in):_num(in._num),_p(D1VEC(3*_num)){};
            POLY(const POLY<T>& in):_num(in._num),_p(in._p){};
            // Overloading
			//D1<VEC<T> >& operator=(const D1<VEC<T> >& in);
			POLY<T>& operator=(const POLY<T>& in);
            const D1<VEC<T> > operator[](const long i)const;
            // Get
            long GetNum()const{return _num;};
            //VEC<T> GetVal(const long idx,const long T123){return _p[3*idx+T123];};
            VEC<T>& T0(const long idx){return _p[3*idx  ];};
            VEC<T>& T1(const long idx){return _p[3*idx+1];};
            VEC<T>& T2(const long idx){return _p[3*idx+2];};
            D1VEC GetMean();  // Get center location of each polygon
            T GetArea(const long i);// Get area of each polygon
            // Set
            //void SetVal(const long idx,const long T123,const VEC<T>& value){_p[3*idx+T123]=value;};
            // Misc.
            void Print();
            void Print(const long idx_num);
		private:
            long _num;
            D1VEC _p;
	};

    //
    // Overloading
    //
	template<class T>
	POLY<T>& POLY<T>::operator=(const POLY<T>& in){
	//D1<VEC<T> >& POLY<T>::operator=(const D1<VEC<T> >& in){
	    if(this != &in){
			_num=in._num;
			_p=in._p;
			/*
            // 1: allocate new memory and copy the elements
            T* new_v = new T[in._n];
            //std::copy(in._v, in._v + in._n, new_v);
            for(long i=0;i<in._n;++i){
                new_v[i]=in._v[i];
            }
            // 2: deallocate old memory
            delete [] _v;
            // 3: assign the new memory to the object
            _v = new_v;
            _n = in._n;
			*/
        }
        return *this;
	}

    template<typename T>
    const D1<VEC<T> > POLY<T>::operator[](const long i)const{
        D1<VEC<T> > out(3);
        out[0] = _p[3*i  ];
        out[1] = _p[3*i+1];
        out[2] = _p[3*i+2];
        return out;
    }

    //
    // Get
    //
    template<typename T>
    D1<VEC<T> > POLY<T>::GetMean(){
    /*
     Purpose:
        Get the center location for T1, T2  and T3.
    */
        D1<VEC<T> > out(_num);
        VEC<T> tmp;
        for(long i=0;i<_num;++i){
            tmp=_p[3*i]+_p[3*i+1]+_p[3*i+2];
            out[i]=VEC<T>( tmp.x()/3.,tmp.y()/3.,tmp.z()/3. );
        }
        return out;
    }

    template<typename T>
    T POLY<T>::GetArea(const long i){
        return vec::find::TriangleArea(_p[3*i  ],_p[3*i+1],_p[3*i+2]);
    }


	//
	// Misc.
	//
	template<typename T>
	void POLY<T>::Print(){
	    if(_num>0){
	        Print(0);
	    }
	}

	template<typename T>
	void POLY<T>::Print(const long idx_num){
	    cout<<idx_num<<" : [";
	    cout<<T0(idx_num).x()<<","<<T0(idx_num).y()<<","<<T0(idx_num).z()<<"] [";
	    cout<<T1(idx_num).x()<<","<<T1(idx_num).y()<<","<<T1(idx_num).z()<<"] [";
	    cout<<T2(idx_num).x()<<","<<T2(idx_num).y()<<","<<T2(idx_num).z()<<"]"<<endl;
	}
}



#endif // POLY_H_INCLUDED
