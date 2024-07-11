#ifndef DEF_FUNC_H_INCLUDED
#define DEF_FUNC_H_INCLUDED

#include <sar/def.h>
#include <basic/def_prefunc.h>
#include <basic/vec.h>
#include <sar/sv.h>
#include <basic/d1.h>
#include <basic/d2.h>
#include <basic/d3.h>
#include <mesh/poly.h>
#include <rcs/rcs.h>
#include <basic/opt.h>
#include <coordinate/geo.h>
#include <basic/cplx.h>
//#include <sar/envi.h>
#include <basic/new_time.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>


namespace def_func{
	using namespace def_prefunc;
	using namespace vec;
	using namespace sv;
	using namespace d1;
	using namespace d2;
	using namespace d3;
	using namespace poly;
	using namespace rcs;
	using namespace opt;
	using namespace geo;
	using namespace cplx;

	// ========================================
	// declare functions
	// ========================================
	D1<double> deg2rad(const D1<double>& deg);
	D1<double> rad2deg(const D1<double>& rad);
	double deg2rad(const double deg);
	double rad2deg(const double rad);
	double ft2m(const double ft);
	double m2ft(const double m);
	double knot2ms(const double knot);
	double ms2knot(const double ms);
	template<typename T> void linspace(const T min_val,const T max_val,d1::D1<T>& out);
	template<typename T> d1::D1<T> linspace(const T min_val,const T max_val,const T step);

	clock_t tic();
    void toc(const clock_t& start, const string);
    template<typename T> void WriteAscii(const D1<T>& in,char* filename);
    template<typename T1,typename T2> void WriteAscii(const D1<T1>& in1,const D1<T2>& in2,char* filename);
    template<typename T1,typename T2,typename T3> void WriteAscii(const D1<T1>& in1,const D1<T2>& in2,const D1<T3>& in3,char* filename);
    template<typename T> void WriteBinary(const T* in,const long num,const char* filename);
    template<typename T> T det(const T* a);
    template<typename T> D2<T> Rx(const T& rad);
    template<typename T> D2<T> Ry(const T& rad);
    template<typename T> D2<T> Rz(const T& rad);
    template<typename T> T sign(const T& val);
    string RCS_num2str(const long idx);
    template<typename T> T MS2KmHr(const T& m_s);
    template<typename T> D2<T> VEC2D2(const VEC<T>& in,const long m,const long n);
    template<typename T> VEC<T> D22VEC(const D2<T>& in);
	template<typename T> VEC<T> GEO2VEC(const GEO<T>& geo);
	template<typename T> GEO<T> VEC2GEO(const VEC<T>& vec);
    void COPY(const char* in,const char* out);
	void Delay(const double msec);
	void ProgressBar(const size_t k, const size_t num_idx,const long progress_len,const long step,def::CLOCK& ProgClock);
	string PathSepString();
	char PathSep();
	D1<string> StrSplit(const string& s,char pattern);
	string StrTruncate(const string s, const unsigned long Truncate_num);
	struct Filename StrFilename(const string file_full);
	long file_line(ifstream& fin);
	template<typename T> void Print(T msg);
	void errormsg(string msg);
	void errorexit();
	template<typename T> int GetTypeNumber(const T& in);
	template<typename T> bool IsCPLX(const T& in);
	template<typename T> bool IsReal(const T& in);
	string StrUppercase(const string& in);
	string StrLowercase(const string& in);
	template<typename T> D1<T> vector2D1(const vector<T>& in);
	CPLX<float> CPLXdouble2CPLXfloat(const CPLX<double>& in);

	// Command Parser class
	class CmdParser{
	public:
		CmdParser():_count(0){};
		CmdParser(int argc, char* argv[]){
			int i;
			_count = 0;
			for(i=1;i<argc;++i){
				if(i != argc-1){
					string tag = argv[i];
					string val = argv[i+1];
					string pre_tag = tag.substr(0,1);
					string pre_val = val.substr(0,1);
					if(((pre_tag == "-") && (pre_val != "-")) ||
						(tag == "-AZS") || (tag == "-AZE") || (tag == "-AZT") ||
						(tag == "-ELS") || (tag == "-ELE") || (tag == "-ELT")){
						_tag.push_back(tag);
						_val.push_back(val);
						i++;
						_count++;
					}else if((pre_tag == "-") && (pre_val == "-")){
						_tag.push_back(tag);
						_val.push_back("YES");
						_count++;
					}
				}else{
					string tag = argv[i];

					_tag.push_back(tag);
					_val.push_back("YES");
					_count++;
				}
			}
			// Totla command
			for(int i=0;i<argc;++i){
				_TotalCmd += string(" ")+string(argv[i]);
			}
		}
		vector<string> tag(){ return _tag; }
		vector<string> val(){ return _val; }
		long GetNum() const{ return _count; }
		bool GetVal(const string& pattern, string& out_str){
			for(int i=0;i<_count;++i){
				if(_tag[i] == pattern){
					out_str = _val[i];
//					cout<<"Option "<<_tag[i]<<" : "<<_val[i]<<endl;
					return true;
				}
			}
			out_str = "";
			return false;
		}
		string GetTotalCmd(){ return _TotalCmd; }
		void Print(){
			cout<<"+---------------------+"<<endl;
			cout<<"|    Command Parser   |"<<endl;
			cout<<"+---------------------+"<<endl;
			cout<<std::setprecision(5);
			for(int i=0;i<_count;++i){
				cout<<_tag[i]<<" : "<<_val[i]<<endl;
			}
		}
	private:
		long _count;
		vector<string> _tag;
		vector<string> _val;
		string _TotalCmd;
	};

	
	
	namespace Cast{
		template<typename T> T* CPLXTptr2Tptr(CPLX<T>* in){ return reinterpret_cast<T*>(in); }
		template<typename T> CPLX<T>* Tptr2CPLXTptr(T* in){ return reinterpret_cast<CPLX<T>*>(in); }
		template<typename T> D1<CPLX<T> > CPLXTptr2D1CPLXT(CPLX<T>* in, const long n){return D1<CPLX<T> >(in,n); }
		template<typename T> T* D1CPLXT2Tptr(D1<CPLX<T> >& in){ return reinterpret_cast<T*>(in.GetPtr()); }
	}
		
	namespace StrColor{
		string Prefix(const int color_code);
		string Suffix();
		string Add2Str(string& in_str,const int color_code);
	}



	// Implement ********************************************************
	D1<double> deg2rad(const D1<double>& deg){
		/*
		 Convert degree to radius
		 */
		D1<double> out(deg.GetNum());
		for(size_t i=0;i<out.GetNum();++i){
			out[i] = deg[i]*def::DTR;
		}
		return out;
	}
	
	D1<double>  rad2deg(const D1<double>& rad){
		/*
		 Convert radius to degree
		 */
		D1<double> out(rad.GetNum());
		for(size_t i=0;i<out.GetNum();++i){
			out[i] = rad[i]*def::RTD;
		}
		return out;
	}
	
	double deg2rad(const double deg){
	/*
	 Convert degree to radius
	*/
		return deg*def::DTR;
	}

	double rad2deg(const double rad){
	/*
	 Convert radius to degree
	*/
		return rad*def::RTD;
	}
	
	double ft2m(const double ft){
		return 0.3048*ft;
	}
	
	double m2ft(const double m){
		return m/0.3048;
	}
	
	double knot2ms(const double knot){
		return knot*0.514444;
	}
	
	double ms2knot(const double ms){
		return ms/0.514444;
	}

	template<typename T>
	void linspace(const T min_val,const T max_val,d1::D1<T>& out){
	/*
	 Create series number
	 Input:
		min_val :[x] minimun value in this series
		max_val :[x] maximun value in this series
		num		:[x] number of point in this series
	 Return:
		out		:[x] series
	*/
		//out = (double *) malloc( ((long)num)*sizeof(double) );
		long num=out.GetNum();
		if(num == 1){
			out[0] = min_val;
		}else{
			double step=(max_val-min_val)/(num-1L);
			for(long i=0;i<num;i++){
				out[i] = T(min_val+((double)i*step));
			}
		}
	}
	
	template<typename T>
	d1::D1<T> linspace(const T min_val,const T max_val,const T step){
		/*
		 Create series number
		 Input:
		 min_val :[x] minimun value in this series
		 max_val :[x] maximun value in this series
		 num		:[x] number of point in this series
		 Return:
		 out		:[x] series
		 */
		//out = (double *) malloc( ((long)num)*sizeof(double) );
		double epsilon = 1E-4;
		long num = long((max_val - min_val)/step + 1L + epsilon);
		d1::D1<T> out(num);
		for(long i=0;i<num;i++){
			out[i] = T(min_val+((double)i*step));
		}
		return out;
	}

    clock_t tic(){
        return clock();
    }

    void toc(const clock_t& start, const string title=""){
        clock_t end = clock();
        cout<<title<<"Escaped: "<<fixed<<double(end-start)/def::clock_ref<<" sec"<<endl;
    }

    template<typename T>
    void WriteAscii(D1<T>& in,char* filename){
    /*
     Purpose:
        Write a seires to ascii file in the disk.
    */
		ofstream fout;
		long num=in.GetNum();
		fout.open(filename);
		if(fout.fail()){
			cout<<"ERROR::[WriteAscii]Input filename! -> ";
            cout<<filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
		for(long i=0;i<num;++i){
			fout<<in[i]<<endl;
		}
		fout.close();
    }

    template<typename T1,typename T2>
    void WriteAscii(const D1<T1>& in1,const D1<T2>& in2,char* filename){
    /*
     Purpose:
        Write a seires to ascii file in the disk.
    */
		ofstream fout;
		long num=in1.GetNum();
		fout.open(filename);
		if(fout.fail()){
			cout<<"ERROR::[WriteAscii]Input filename! -> ";
            cout<<filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
		if(in1.GetNum()!=in2.GetNum()){
            cout<<"ERROR::[WriteAscii]The input array elements *MUST* be same! -> ";
            cout<<"in1=["<<in1.GetNum()<<"], in2=["<<in2.GetNum()<<"]"<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
		for(long i=0;i<num;++i){
			fout<<in1[i]<<"  "<<in2[i]<<endl;
		}
		fout.close();
    }

    template<typename T1,typename T2,typename T3>
    void WriteAscii(const D1<T1>& in1,const D1<T2>& in2,const D1<T3>& in3,char* filename){
    /*
     Purpose:
        Write a seires to ascii file in the disk.
    */
		ofstream fout;
		long num=in1.GetNum();
		fout.open(filename);
		if(fout.fail()){
			cout<<"ERROR::[WriteAscii]Input filename! -> ";
            cout<<filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
        if( (in1.GetNum()!=in2.GetNum()) || (in1.GetNum()!=in3.GetNum()) ){
            cout<<"ERROR::[WriteAscii]The input array elements *MUST* be same!";
            cout<<"in1=["<<in1.GetNum()<<"], in2=["<<in2.GetNum()<<"]"<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
		for(long i=0;i<num;++i){
			fout<<in1[i]<<"  "<<in2[i]<<"  "<<in3[i]<<endl;
		}
		fout.close();
    }

    template<typename T>
    void WriteBinary(const T* in,const long num,const char* filename){
    /*
     Purpose:
        Wrtie a series to a file.
    */
		ofstream fout;
		fout.open(filename);
		fout.write(reinterpret_cast<char*>(in),sizeof(T));
		fout.close();
    }

    template<typename T>
    T det(const T* a){
    /*
     Purpose:
       Caliculate the determinant of input matrix.
	   Note: *ONLY* For 3x3 matrix
    */
        return a[0]*a[4]*a[8] +
               a[2]*a[3]*a[7] +
               a[1]*a[5]*a[6] -
               a[2]*a[4]*a[6] -
               a[1]*a[3]*a[8] -
               a[0]*a[5]*a[7];
	}

    template<typename T> D2<T> Rx(const T& rad){ //*NOT* same as IDL
    /*
     Purpose:
        Calculate the roatetion matrix with X axis
     Input:
        rad :[rad] angle(radius)
     Return:
        The rotation matrix
    */
        double cos1,sin1;
		_sincos(rad,sin1,cos1);
        T series[]={1.,   0.,    0.,
                    0., cos1, -sin1,
                    0., sin1,  cos1 };
        return d2::D2<T>(series,3,3);
    }

    template<typename T> D2<T> Ry(const T& rad){ //*NOT* same as IDL
    /*
     Purpose:
        Calculate the roatetion matrix with Y axis
     Input:
        rad :[rad] angle(radius)
     Return:
        The rotation matrix
    */
        double cos1,sin1;
		_sincos(rad,sin1,cos1);
        T series[]={ cos1, 0., sin1,
					   0., 1.,   0.,
                    -sin1, 0., cos1 };
        return d2::D2<T>(series,3,3);
    }

    template<typename T> D2<T> Rz(const T& rad){ //*NOT* same as IDL
    /*
     Purpose:
        Calculate the roatetion matrix with Y axis
     Input:
        rad :[rad] angle(radius)
     Return:
        The rotation matrix
    */
        double cos1,sin1;
		_sincos(rad,sin1,cos1);
        T series[]={ cos1, -sin1, 0.,
                     sin1,  cos1, 0.,
					   0.,    0., 1 };
        return d2::D2<T>(series,3,3);
    }

    template<typename T>
    T sign(const T& val){
        return (double(val) < 0.)? -T(1):T(1);
    }

    string RCS_num2str(const long idx){
    /*
     Purpose:
        Convert filename -> "0xxxx.dat"
    */
        string str_num;
        if(idx >= 10000){
            str_num = num2str(long(idx));
        }else if(idx >= 1000){
            str_num = string("0");
        }else if(idx >= 100){
            str_num = string("00");
        }else if(idx >= 10){
            str_num = string("000");
        }else{
            str_num = string("0000");
        }

        str_num+=num2str(long(idx));

        return str_num;
    }

    template<typename T>
    T MS2KmHr(const T& m_s){
    /*
     Purpose:
        Convert speed from m/s to km/hour.
    */
        return m_s/1000.*3600.;
    }

    template<typename T>
    D2<T> VEC2D2(const VEC<T>& in,const long m,const long n){
    /*
     Purpose:
        Convert VEC class to 1-D D2 class.
    */
        if((m*n) != 3){
            cout<<"ERROR::[def_func::VEC2D2]:(m,n) MUST be (3,1) or (1,3)! -> ";
            cout<<"in=["<<m<<"x"<<n<<"]"<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
		return D2<T>(in,m,n);
        //T series[3] = {in.x(),in.y(),in.z()};
        //return D2<T>(series,m,n);
    }

    template<typename T>
    VEC<T> D22VEC(const D2<T>& in){
    /*
     Purpose:
        Convert 1-D D2 class to VEC class.
    */
        long m=in.GetM();
        long n=in.GetN();
        if((m > 1)&&(n > 1)){
            cout<<"ERROR::[D2::D22VEC]:Input D2 MUST be 1-D -> ";
            cout<<"in=["<<m<<"x"<<n<<"]"<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
        if( ((m == 1)&&(n != 3)) || ((m != 3)&&(n == 1)) ){
            cout<<"ERROR::[D2::D22VEC]:The 1-D MUST ONLY 3 elements -> ";
            cout<<"in=["<<in.GetM()<<"x"<<in.GetN()<<"]"<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
        T x,y,z;
        if(m == 1){
            x = in[0][0]; // row
            y = in[0][1];
            z = in[0][2];
        }else{
            x = in[0][0]; // column
            y = in[1][0];
            z = in[2][0];
        }
        return VEC<T>(x,y,z);
    }

	template<typename T>
	VEC<T> GEO2VEC(const GEO<T>& geo){
	/*
	 Purpose:
		Convert from GEO to VEC class
	 Input:
		geo :[rad,rad,m](lon,lat,h)
	 Return:
		VEC class
	*/
		return VEC<T>(geo.lon(),geo.lat(),geo.h());
	}

	template<typename T>
	GEO<T> VEC2GEO(const VEC<T>& vec){
	/*
	 Purpose:
		Convert from VEC to GEO class
	 Input:
		vec :[m,m,m]
	 Return:
		GEO class
	*/
		return GEO<T>(vec.x(),vec.y(),vec.z());
	}

    void COPY(const char* in,const char* out){
    /*
     Purpose:
        Copy data from "in" to "out"
    */
        long length;
        char* buffer;

        ifstream f_in;
        ofstream f_out;

        f_in.open(in,ios::binary );
        if(f_in.fail()){
            cout<<"ERROR::[def_func::COPY]:Input file error! -> ";
            cout<<in<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }

        f_out.open(out,ios::binary);
        if(f_in.fail()){
            cout<<"ERROR::[def_func::COPY]:Output file error! -> ";
            cout<<out<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }

        // get length of file:
        f_in.seekg(0,ios::end);
        length = (long)f_in.tellg();
        f_in.seekg(0,ios::beg);

        // allocate memory:
        buffer = new char[length];

        // read data as a block:
        f_in.read(buffer,length);
        f_out.write(buffer,length);

        f_in.close();
        f_out.close();

        delete[] buffer;
    }
	
	void Delay(const double msec){
		clock_t now, start;
		start = clock();
		do{
			now = clock();
		}
		while(now - start < msec*CLOCKS_PER_SEC/1E3);
	}

    void ProgressBar(const size_t k,const size_t num_idx,const long progress_len,const long step,def::CLOCK& ProgClock){
    /*
     Purpose:
        Display a prgress bar on the console screen
     Input:
        k           : counting index
        num_idx     : total number of index
        progress_len: display progressbar length
        step        : step of index
    */
        if(k==0){
			ProgClock = def_func::tic();
            //def::func_time = def_func::tic();
            return;
        }
        if(k==num_idx-1){
            cout<<"\r[";
            for(long m=0;m<progress_len;++m){
                cout<<"=";
            }
            cout<<"] ( "<<k+1<<" / "<<num_idx<<" ) 100%";
            cout<<"                         "<<endl;
        }else if((k%step)==0){
            float progress = float(k+1)/float(num_idx)*float(100.);
			if(progress < 100){
				long p;
				
				double pass= double(def_func::tic()-ProgClock.time())/def::clock_ref;
				//double pass= double(def_func::tic()-def::func_time)/def::clock_ref;
				double rem = pass/(double(k)/double(num_idx))-pass;
				int rem_min=int(rem/60.);
				double rem_sec=rem-rem_min*60.;
				string rem_sec_str=(rem_sec<10)? string("0")+num2str(int(rem_sec)) : num2str(int(rem_sec));
				
				
				cout.flush();
				cout<<"[";
				for(p=0;p<long(progress)*progress_len/100;++p){
					cout<<"=";
				}
				
				if(p<progress_len){
					cout<<">";
					p++;
					for(long m=0;m<progress_len-p;++m){
						cout<<" ";
					}
				}
				cout<<"] ( "<<k+1<<" / "<<num_idx<<" ) "<<fixed<<std::setprecision(1)<<progress<<"% ";
				cout<<"Remaining : "<<rem_min<<"m"<<rem_sec_str<<"s           \r";
				cout.flush();
            }
        }
    }

	string PathSepString(){
		string Sep=string("//");	// for Linux & Mac OS


		#ifdef _MSC_VER		// for Windows
			Sep = string("\\");
		#endif

		return Sep;
	}

	char PathSep(){
		char Sep=char('/');	// for Linux & Mac OS


		#ifdef _MSC_VER		// for Windows
			Sep = char('\\');
		#endif

		return Sep;
	}

	D1<string> StrSplit(const string& s,char pattern){
		stringstream ss(s);
		string token[100];
		int count=0;

		for(int i=0;i<100;++i){
			getline(ss, token[count], pattern);
			if (ss.fail()){ break; }
			count++;
		}
		D1<string> out(count);
		for(int i=0;i<count;++i){
			out[i] = token[i];
		}
		return out;
	}
	
	string StrTruncate(const string s, const unsigned long Truncate_num){
		D1<string> tmp = StrSplit(s, '.');
		string zero = "";
		// create zero
		if(tmp.GetNum() == 1){
			for(unsigned int i=0;i<Truncate_num;++i){
				zero += "0";
			}
		}else{
			zero = tmp[1].substr(0,Truncate_num);
			if(zero.length() < Truncate_num){
				for(unsigned long i=zero.length();i<Truncate_num;++i){
					zero += "0";
				}
			}
		}
		return tmp[0] + "." + zero;
	}

	struct Filename StrFilename(const string file_full){
		// Extract file name
		D1<string> split = StrSplit(file_full,PathSep());
		// combine path
		string path;
		for(size_t i=0;i<split.GetNum()-1;++i){
			path += split[i];
			path += PathSepString();
		}
		// extract name & type
		D1<string> name_type = StrSplit(split[split.GetNum()-1],'.');
		// construct output structure
		struct Filename str;
		str.filename = split[split.GetNum()-1];
		str.foldername = split[split.GetNum()-2];
		str.name = name_type[0];
		if(name_type.GetNum() > 2){
			for(size_t i=0;i<name_type.GetNum()-2;++i){
				str.name += "." + name_type[i+1];
			}
		}
		str.path = path;
		str.type = name_type[name_type.GetNum()-1];
	
		return str;
	}
	
	long file_line(ifstream& fin){
		fin.seekg(0, ios::beg);
		string str;
		long line = 0;
		while (!fin.eof()){
			getline(fin, str);
			line++;
		}
		return line;
	}

	template<typename T> void Print(T msg){
		if( (typeid(msg) != typeid(string("123"))) ||	// string
		    (typeid(msg) != typeid("123")) ){			// char*
			cout<<setprecision(12)<<msg<<endl;
		}else{
			cout<<msg<<endl;
		}
	}

	void errormsg(string msg){
		cout<<"ERROR::"<<msg<<"!"<<endl;
		cout<<"<<Press Enter to Stop>>"; getchar();
		exit(EXIT_FAILURE);
	}
	
	void errorexit(){
#ifdef _WIN32
		system("pause");
#endif
		exit(EXIT_FAILURE);
	}
	
	template<typename T>
	int GetTypeNumber(const T& in){
		string type = typeid(in).name();
		int val = Type2IDLType( GetType(type) );
		if(val != 0){ return val; }
		//
		// Number     C++_type			IDL_type
		//
		//    0       undefined			undefined
		//   10       char				byte
		//   20       short				integer
		//   30       int				long
		//   40       float				float
		//   50       double			double
		//   60       CPLX<float>		complex
		//   70	      string			string
		//   90       CPLX<double>		dcomplex
		//  210       D1<char>			na
		//  220       D1<short>			na
		//  230       D1<int>			na
		//  240       D1<float>			na
		//  250       D1<double>		na
		//  260       D1<CPLX<float> >  na
		//  290       D1<CPLX<double> > na
		//  310       D2<char>			na
		//  320       D2<short>			na
		//  330       D2<int>			na
		//  340       D2<float>			na
		//  350       D2<double>		na
		//  360       D2<CPLX<float> >  na
		//  390       D2<CPLX<double> > na
		//  410       D3<char>			na
		//  420       D3<short>			na
		//  430       D3<int>			na
		//  440       D3<float>			na
		//  450       D3<double>		na
		//  460       D3<CPLX<float> >  na
		//  490       D3<CPLX<double> > na
		if(type == typeid(D1<char>).name()){
			return 210;
		}else if(type == typeid(D1<short>).name()){
			return 220;
		}else if(type == typeid(D1<int>).name()){
			return 230;
		}else if(type == typeid(D1<float>).name()){
			return 240;
		}else if(type == typeid(D1<double>).name()){
			return 250;
		}else if(type == typeid(D1<CPLX<float> >).name()){
			return 260;
		}else if(type == typeid(D1<string>).name()){
			return 270;
		}else if(type == typeid(D1<CPLX<double> >).name()){
			return 290;
		}else if(type == typeid(D2<char>).name()){
			return 310;
		}else if(type == typeid(D2<short>).name()){
			return 320;
		}else if(type == typeid(D2<int>).name()){
			return 330;
		}else if(type == typeid(D2<float>).name()){
			return 340;
		}else if(type == typeid(D2<double>).name()){
			return 350;
		}else if(type == typeid(D2<CPLX<float> >).name()){
			return 360;
		}else if(type == typeid(D2<string>).name()){
			return 370;
		}else if(type == typeid(D2<CPLX<double> >).name()){
			return 390;
		}else if(type == typeid(D3<char>).name()){
			return 410;
		}else if(type == typeid(D3<short>).name()){
			return 420;
		}else if(type == typeid(D3<int>).name()){
			return 430;
		}else if(type == typeid(D3<float>).name()){
			return 440;
		}else if(type == typeid(D3<double>).name()){
			return 450;
		}else if(type == typeid(D3<CPLX<float> >).name()){
			return 460;
		}else if(type == typeid(D3<string>).name()){
			return 470;
		}else if(type == typeid(D3<CPLX<double> >).name()){
			return 490;
		//
		// +1 (for single pointer)
		//
		}else if(type == typeid(D1<char*>).name()){
			return 211;
		}else if(type == typeid(D1<short*>).name()){
			return 221;
		}else if(type == typeid(D1<int*>).name()){
			return 231;
		}else if(type == typeid(D1<float*>).name()){
			return 241;
		}else if(type == typeid(D1<double*>).name()){
			return 251;
		}else if(type == typeid(D1<CPLX<float*> >).name()){
			return 261;
		}else if(type == typeid(D1<string*>).name()){
			return 261;
		}else if(type == typeid(D1<CPLX<double*> >).name()){
			return 291;
		}else if(type == typeid(D2<char*>).name()){
			return 311;
		}else if(type == typeid(D2<short*>).name()){
			return 321;
		}else if(type == typeid(D2<int*>).name()){
			return 331;
		}else if(type == typeid(D2<float*>).name()){
			return 341;
		}else if(type == typeid(D2<double*>).name()){
			return 351;
		}else if(type == typeid(D2<CPLX<float*> >).name()){
			return 361;
		}else if(type == typeid(D2<string*>).name()){
			return 371;
		}else if(type == typeid(D2<CPLX<double*> >).name()){
			return 391;
		}else if(type == typeid(D3<char*>).name()){
			return 411;
		}else if(type == typeid(D3<short*>).name()){
			return 421;
		}else if(type == typeid(D3<int*>).name()){
			return 431;
		}else if(type == typeid(D3<float*>).name()){
			return 441;
		}else if(type == typeid(D3<double*>).name()){
			return 451;
		}else if(type == typeid(D3<CPLX<float*> >).name()){
			return 461;
		}else if(type == typeid(D3<string*>).name()){
			return 461;
		}else if(type == typeid(D3<CPLX<double*> >).name()){
			return 491;
		}else{
			return 0;
		}
	}
	
	template<typename T>
	bool IsCPLX(const T& in){
		int type = GetTypeNumber(in);
		if( (type == 6)  || (type == 9)  || 
		    (type == 26) || (type == 29) || 
		    (type == 36) || (type == 39) || 
		    (type == 46) || (type == 49) ){
			return true;
		}else{
			return false;
		}
	}
	
	template<typename T>
	bool IsReal(const T& in){
		int type = GetTypeNumber(in);
		if( (type == 6)  || (type == 9)  || 
		   (type == 26) || (type == 29) || 
		   (type == 36) || (type == 39) || 
		   (type == 46) || (type == 49) ){
			return false;
		}else{
			return true;
		}
	}
	
	string StrUppercase(const string& in){
		string out = in;
		transform(out.begin(), out.end(), out.begin(), ::toupper);
		return out;
	}
	
	string StrLowercase(const string& in){
		string out = in;
		transform(out.begin(), out.end(), out.begin(), ::tolower);
		return out;
	}
	
	template<typename T>
	D1<T> vector2D1(const vector<T>& in){
		D1<T> out(in.size());
		for(unsigned long i=0;i<in.size();++i){
			out[i] = in[i];
		}
		return out;
	}

	CPLX<float> CPLXdouble2CPLXfloat(const CPLX<double>& in){
		return CPLX<float>(float(in.r()), float(in.i()));
	}
	
	//	
	//
    // namespace StrColor
    //
	//
	string StrColor::Prefix(const int color_code){
	/*
	 Purpose:
		Print out the message on console in color (Prefix)
	*/
		// convert color_code from int to string
		std::ostringstream out;
		string str_prefix;
		out<<str_prefix<<color_code;
		str_prefix = out.str();
		str_prefix = "\033[" + str_prefix + "m";
		return str_prefix;
	}
	
	string StrColor::Suffix(){
	/*
	 Purpose:
		Print out the message on console in color (Suffix)
	*/
		string str_suffix("\033[0m");
		return str_suffix;
	}
	
	string StrColor::Add2Str(string& in_str,const int color_code){
	/*
	 Purpose:
		Print out the message on console in color
	   =============================================
	     COLOR CODE
	   =============================================
	   0 = default colour
	   1 = bold
	   4 = underlined
	   5 = flashing text
	   7 = reverse field
	   31 = red
	   32 = green
	   33 = orange
	   34 = blue
	   35 = purple
	   36 = cyan
	   37 = grey
	   40 = black background
	   41 = red background
	   42 = green background
	   43 = orange background
	   44 = blue background
	   45 = purple background
	   46 = cyan background
	   47 = grey background
	   90 = dark grey
	   91 = light red
	   92 = light green
	   93 = yellow
	   94 = light blue
	   95 = light purple
	   96 = turquoise
	   100 = dark grey background
	   101 = light red background
	   102 = light green background
	   103 = yellow background
	   104 = light blue background
	   105 = light purple background
	   106 = turquoise background
	   =============================================
	*/
		// convert color_code from int to string
		std::ostringstream out;
		string str_prefix;
		out<<str_prefix<<color_code;
		str_prefix = out.str();
		str_prefix = "\033[" + str_prefix + "m";
		string str_suffix("\033[0m");
		// add string
		return str_prefix + in_str + str_suffix;
	}

}


#endif // DEF_FUNC_H_INCLUDED
