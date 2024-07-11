#ifndef SV_H_INCLUDED
#define SV_H_INCLUDED

#include <basic/vec.h>
#include <basic/d1.h>

namespace sv{
	using namespace vec;
	using namespace d1;
	// ==============================================
	// State vector
	// ==============================================
	template<typename T>
	class SV
	{
	    typedef D1<VEC<T> > D1VEC;
		public:
			// Constructure
			SV();
			SV(long num);
			//SV(T* t,VEC<T>* pos,VEC<T>* vel,const long num);
			SV(const D1<T>& t,const D1<VEC<T> >& pos,const D1<VEC<T> >& vel,const long num);
			// Copy Constructor
			//SV(const SV<T>& in);
			// operator overloading
			SV<T>& operator=(const SV<T>& in);
			//~SV();
			// Operator overloading
			// Get Ref.
			D1<T>& t(){return _t;};
			D1VEC& pos(){return _pos;};
			D1VEC& vel(){return _vel;};
			// Get Value
			const D1<T>& t()const{return _t;};
			const D1VEC& pos()const{return _pos;};
			const D1VEC& vel()const{return _vel;};
			const long& GetNum()const{return _num;};
			const T dt()const{ return _t[1]-_t[0];};
			// Set
			void SetAll(const D1<T>& t,const D1<VEC<T> >& pos,const D1<VEC<T> >& vel,const long num);
			// Misc.
			void Print()const;
			void Print(long i)const;
			void WriteASCII(const char* filename);
			void WriteBinary(const char* filename);
			void ReadBinary(const char* filename);
			const VEC<T> NextPs(const size_t i)const;
			SV<T> GetRange(const size_t idx_start, const size_t idx_end);
		private:
            void _init(const long num);
			long _num;
			D1<T> _t;
			D1VEC _pos,_vel;
	};
	//
	// Private
	//
	template<typename T>
	void SV<T>::_init(const long num){
	    _num=num;
	    _t=D1<T>(_num);
		_pos=D1<VEC<T> >(_num);
		_vel=D1<VEC<T> >(_num);
	}
	//
	// Constructure
	//
	template<typename T>
	SV<T>::SV(){
	    _init(1);
	}

	template<typename T>
	SV<T>::SV(long num){
	    _init(num);
	}

	template<typename T>
	SV<T>::SV(const D1<T>& t,const D1<VEC<T> >& pos,const D1<VEC<T> >& vel,const long num){
	    _num=num;
        _t=t;
        _pos=pos;
        _vel=vel;
	}

	//
	// Copy Constructor
	//
/*	template<typename T>
	SV<T>::SV(const SV<T>& in){
	    if(this != &in){
	        cout<<"test"<<endl;
            _num=in._num;
            _t=D1<T>(in._t);
            _pos=D1VEC(in._pos);
            _vel=D1VEC(in._vel);
        }
	    //_init(in.GetNum());
	    //for(long i=0;i<in.GetNum();++i){
	    //    _t[i]=in._t[i];
	    //    _pos[i]=in._pos[i];
	    //    _vel[i]=in._vel[i];
	    //}
	}
*/
	//
	// Operator overloading
	//
	template<typename T>
	SV<T>& SV<T>::operator=(const SV<T>& in){
        if(this != &in){
            _num=in._num;
            _t=D1<T>(in._t);
            _pos=D1VEC(in._pos);
            _vel=D1VEC(in._vel);
        }
        return *this;
	}

/*
	template<typename T>
	SV<T>::SV(T* t,VEC<T>* pos,VEC<T>* vel,const long num){
		_num=num;
		_t=t;
		_pos=pos;
		_vel=vel;
	}
*/
/*
    template<typename T>
	SV<T>::~SV(){
		_num=0;
		delete _t;
		delete _pos;
		delete _vel;
	}
*/
	//
	// Operator overloading
	//

	//
	// Set
	//
	template<typename T>
	void SV<T>::SetAll(const D1<T>& t,const D1<VEC<T> >& pos,const D1<VEC<T> >& vel,const long num){
	    for(long i=0;i<num;++i){
	        _t[i]=t[i];
	        _pos[i]=pos[i];
	        _vel[i]=vel[i];
	    }
	}

	//
	// Misc.
	//
	template<typename T>
	void SV<T>::Print()const{
		long m=(_num >= 20)? 20:_num;
		cout<<"+-------------------------------------+"<<endl;
		cout<<"|             sv::SV class            |"<<endl;
		cout<<"+-------------------------------------+"<<endl;
		for(long i=0;i<m;++i){
			printf("%ld: %.8f [%.6f, %.6f, %.6f] [%.6f, %.6f, %.6f]\n",
					i, _t[i],
					_pos[i].x(), _pos[i].y(), _pos[i].z(),
					_vel[i].x(), _vel[i].y(), _vel[i].z() );
//			cout<<i<<": "; cout<<_t[i];
//			cout<<" ["<<_pos[i].x()<<", "<<_pos[i].y()<<", "<<_pos[i].z()<<"]";
//			cout<<" ["<<_vel[i].x()<<", "<<_vel[i].y()<<", "<<_vel[i].z()<<"]";
//			cout<<endl;
		}
	}

	template<typename T>
	void SV<T>::Print(long i)const{
		cout<<"SV:"<<endl;
		printf("%ld: %.8f [%.6f, %.6f, %.6f] [%.6f, %.6f, %.6f]\n",
				i, _t[i],
				_pos[i].x(), _pos[i].y(), _pos[i].z(),
				_vel[i].x(), _vel[i].y(), _vel[i].z() );
//		cout<<i<<": "; cout<<_t[i];
//		cout<<" ["<<_pos[i].x()<<", "<<_pos[i].y()<<", "<<_pos[i].z()<<"]";
//		cout<<" ["<<_vel[i].x()<<", "<<_vel[i].y()<<", "<<_vel[i].z()<<"]";
//		cout<<endl;
	}

	template<typename T>
	void SV<T>::WriteASCII(const char* filename){
		ofstream fout(filename);
		double xx,yy,zz,vx,vy,vz;
		if(fout.fail()){
			cout<<"ERROR::[SV::WriteASCII]:Output file path error! -> ";
			cout<<filename<<endl;
			exit(EXIT_FAILURE);
		}
		for(long i=0;i<_num;++i){
			xx=_pos[i].x();
			yy=_pos[i].y();
			zz=_pos[i].z();
			vx=_vel[i].x();
			vy=_vel[i].y();
			vz=_vel[i].z();
			fout<<std::setprecision(6)<<std::fixed<<_t[i]<<"\t";
			fout<<std::setprecision(10)<<std::fixed<<xx<<"\t"<<yy<<"\t"<<zz<<"\t";
			fout<<std::setprecision(10)<<std::fixed<<vx<<"\t"<<vy<<"\t"<<vz<<"\n";
		}
		fout.close();
	}

    template<typename T>
    void SV<T>::WriteBinary(const char* filename){
        ofstream fout(filename,ios::binary);
        double xx,yy,zz,vx,vy,vz;
        if(fout.fail()){
            cout<<"ERROR::[SV::WriteBinary]:Output file path error! -> ";
            cout<<filename<<endl;
            exit(EXIT_FAILURE);
        }
        for(long i=0;i<_num;++i){
            xx=_pos[i].x();
            yy=_pos[i].y();
            zz=_pos[i].z();
            vx=_vel[i].x();
            vy=_vel[i].y();
            vz=_vel[i].z();
            fout.write(reinterpret_cast<char*>( &_t[i] ),sizeof(double));
            fout.write(reinterpret_cast<char*>( &xx ),sizeof(double));
            fout.write(reinterpret_cast<char*>( &yy ),sizeof(double));
            fout.write(reinterpret_cast<char*>( &zz ),sizeof(double));
            fout.write(reinterpret_cast<char*>( &vx ),sizeof(double));
            fout.write(reinterpret_cast<char*>( &vy ),sizeof(double));
            fout.write(reinterpret_cast<char*>( &vz ),sizeof(double));
        }
        fout.close();
    }

    template<typename T>
    void SV<T>::ReadBinary(const char* filename){
        ifstream fin(filename,ios::binary);
        double xx,yy,zz,vx,vy,vz;
        if(fin.fail()){
            cout<<"ERROR::[SV::WriteBinary]:Output file path error! -> ";
            cout<<filename<<endl;
            exit(EXIT_FAILURE);
        }
        for(long i=0;i<_num;++i){
            fin.read(reinterpret_cast<char*>( &_t[i] ),sizeof(double));
            fin.read(reinterpret_cast<char*>( &xx ),sizeof(double));
            fin.read(reinterpret_cast<char*>( &yy ),sizeof(double));
            fin.read(reinterpret_cast<char*>( &zz ),sizeof(double));
            fin.read(reinterpret_cast<char*>( &vx ),sizeof(double));
            fin.read(reinterpret_cast<char*>( &vy ),sizeof(double));
            fin.read(reinterpret_cast<char*>( &vz ),sizeof(double));
            _pos[i].Setxyz(xx,yy,zz);
            _vel[i].Setxyz(vx,vy,vz);
        }
        fin.close();
    }

	template<typename T>
	const VEC<T> SV<T>::NextPs(const size_t idx_Ps)const{
		VEC<double> Ps = _pos[idx_Ps];
		VEC<double> Ps1;
		if (idx_Ps != _num - 1) {
			Ps1 = _pos[idx_Ps + 1];
		} else {
			VEC<double> uv_Ps1 = Ps - _pos[idx_Ps - 1];
			Ps1 = Ps + uv_Ps1;
		}

		return Ps1;
	}

	template<typename T>
	SV<T> SV<T>::GetRange(const size_t idx_start, const size_t idx_end){
		D1<double>       sv_crop_t   = _t.GetRange(idx_start, idx_end);
		D1<VEC<double> > sv_crop_pos = _pos.GetRange(idx_start, idx_end);
		D1<VEC<double> > sv_crop_vel = _vel.GetRange(idx_start, idx_end);
		SV<double> sv_crop(sv_crop_t, sv_crop_pos, sv_crop_vel, sv_crop_t.GetNum());

		return sv_crop;
	}
}

#endif // SV_H_INCLUDED
