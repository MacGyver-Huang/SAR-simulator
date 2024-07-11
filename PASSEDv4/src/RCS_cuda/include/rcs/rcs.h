#ifndef RCS_H_INCLUDED
#define RCS_H_INCLUDED

#include <basic/cplx.h>
#include <basic/d1.h>

namespace rcs{
    using namespace cplx;
    using namespace d1;
    // ==============================================
	// RCS class
	// ==============================================
	template<typename T>
	class RCS
	{
		public:
            RCS():_num(0){};
            RCS(const long num):_num(num),_rcs(D1<CPLX<T> >(4*num)){};
            // Get
            long GetNum()const{return _num;};
            CPLX<T> GetVal(const long idx,const long idx_pol){return _rcs[4*idx+idx_pol];};
            CPLX<T> GetHH(const long idx){return _rcs[4*idx  ];};
            CPLX<T> GetHV(const long idx){return _rcs[4*idx+1];};
            CPLX<T> GetVH(const long idx){return _rcs[4*idx+2];};
            CPLX<T> GetVV(const long idx){return _rcs[4*idx+3];};
            // Set
            void SetVal(const long idx,const long idx_pol,const CPLX<T>& value){_rcs[4*idx+idx_pol]=value;};
            // Misc.
            void Print(const long idx_num);
		private:
            long _num;
            D1<CPLX<T> > _rcs;
	};

	//
	// Misc.
	//
	template<typename T>
	void RCS<T>::Print(const long idx_num){
	    GetHH(idx_num).Print();
        GetHV(idx_num).Print();
        GetVH(idx_num).Print();
        GetVV(idx_num).Print();
	}
}


#endif // RCS_H_INCLUDED
