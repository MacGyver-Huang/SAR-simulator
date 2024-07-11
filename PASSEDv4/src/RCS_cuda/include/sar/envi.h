#ifndef ENVI_H_INCLUDED
#define ENVI_H_INCLUDED


#include <typeinfo>
#include <basic/def_prefunc.h>
#include <basic/def_func.h>
#include <basic/d1.h>
#include <basic/d2.h>
#include <basic/d3.h>
#include <basic/cplx.h>
#include <basic/new_time.h>
#include <fstream>

using namespace std;
using namespace def_prefunc;
using namespace d1;
using namespace d2;
using namespace d3;
using namespace cplx;


namespace envi{
	// Constant structure
	class ENVIhdr{ // ENVI header structure
		public:
			ENVIhdr();
			ENVIhdr(const char* file);
			ENVIhdr(const long in_xsz, const long in_ysz, const long in_zsz, const long in_offset,
					const long in_type, const long in_byte, const string in_interleave);
			// Utilities
			void ReadENVIHeader();
			void WriteENVIHeader(const char* filename);
			void Print();
			// Set
			void xsz(long in_xsz){ _xsz=in_xsz; };
			void ysz(long in_ysz){ _xsz=in_ysz; };
			void zsz(long in_zsz){ _xsz=in_zsz; };
			void offset(long in_offset){ _xsz=in_offset; };
			void type(long in_type){ _xsz=in_type; };
			void byte(long in_byte){ _xsz=in_byte; };
			void interleave(string in_interleave){ _interleave = in_interleave; };
			void filename(const char* file){ _filename = string(file); };
			// Get
			long xsz()const{return _xsz;}
			long ysz()const{return _ysz;}
			long zsz()const{return _zsz;}
			long offset()const{return _offset;}
			long type()const{return _type;}
			long byte()const{return _byte;}
			string interleave()const{return _interleave;}
			string filename()const{return _filename; };
		private:
			string _filename;
			long _xsz;			// samples
			long _ysz;			// lines
			long _zsz;			// bands
			long _offset;		// header offset in Bytes
			long _type;			// data type (ex. 4 == float, 6 == float complex, 9 == double complex)
			long _byte;			// byte order (0 == PC, 1 == Workstation)
			string _interleave;	// pixel interleave ["BIP", "BIL", "BSQ"]
	};
	
	// Main class
	template<typename T>
	class ENVI{
		public:
			// Constructure
			ENVI();
			ENVI(const string file);
			ENVI(const string file, const long in_xsz, const long in_ysz, const long in_zsz, const long in_offset,
				 const long in_type, const long in_byte, const string in_interleave);
			// Utilities
			void SetENVIheader(const ENVIhdr& in_hdr);
			//template<T2> void SetData(const T2& in);
			T& ReadENVI();
			void WriteENVI(const char* out_filename);
			//template<typename T2> void WriteENVI(const T2* out);
			void PrintENVIHeader();
			ENVIhdr& GetENVIhdr(){ return _hdr; };
		private:
			string _filename;
			ENVIhdr _hdr;
			struct def_prefunc::Filename _str;
			// define array
			D3<char> _raw_c;	//1(Byte)
			D3<short> _raw_s;	//2(integer)
			D3<int> _raw_i;		//3(long)
			D3<float> _raw_f;	//4(float)
			D3<double> _raw_d;	//5(double)
			D2<CPLX<float> > _raw_cxf;	//6(float complex)
			D2<CPLX<double> > _raw_cxd;	//9(double complex)
			// instance
			char* _raw;
			// Utilities
			template<typename T1> void _ReadENVICore3D(D3<T1>& raw);
			template<typename T1> void _ReadENVICoreCPLX(D2<CPLX<T1> >& raw);
			T& _GetDataRef();
	};
	
	// ========================================================
	// Implement for ENVIhdr
	// ========================================================
	ENVIhdr::ENVIhdr():_xsz(0),_ysz(0),_zsz(0),_offset(0),_type(0),_byte(0){
		_filename = "";
	}

	ENVIhdr::ENVIhdr(const char* file):_xsz(0),_ysz(0),_zsz(0),_offset(0),_type(0),_byte(0){
		_filename = string(file);
		this->ReadENVIHeader();
	}
	
	ENVIhdr::ENVIhdr(const long in_xsz, const long in_ysz, const long in_zsz, const long in_offset,
					 const long in_type, const long in_byte, const string in_interleave){
		_xsz = in_xsz;
		_ysz = in_ysz;
		_zsz = in_zsz;
		_offset = in_offset;
		_type = in_type;
		_byte = in_byte;
		_interleave = in_interleave;
		_filename = string("");
	}

	void ENVIhdr::ReadENVIHeader(){
		// (for general : Read ASCII file)
		char buf[10000];
        ifstream fin(_filename.c_str(),ios::binary);
		if(!fin.good()){
			cout<<"ERROR::[ENVI::ENVIhdr::ReadENVIHeader]:No this file! -> ";
			cout<<_filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
			exit(EXIT_FAILURE);
		}
		long line = def_prefunc::file_line(fin);
		for(int i=0;i<line;++i){
			fin.getline(buf,10000);

#ifdef _MSC_VER
			char* next_tok;
			char* split = strtok_s(buf,"=",&next_tok);
#else 
			char* split = strtok(buf,"=");
#endif

			int count=0;
			while(split != NULL){
				//cout<<"'"<<split<<"'"<<endl;
				string str = string(split);

#ifdef _MSC_VER
				next_tok;
				split = strtok_s(NULL,"=",&next_tok);
#else 
				split = strtok(NULL,"=");
#endif

				if(str == "samples "){_xsz = atoi(split);};
				if(str == "lines   "){_ysz = atoi(split);};
				if(str == "bands   "){_zsz = atoi(split);};
				if(str == "header offset "){_offset = atoi(split);};
				if(str == "data type "){_type = atoi(split);};
				if(str == "interleave "){
					/*
					string tmp_str1;
					char buffer[20];
					tmp_str1 = string(split+1);
					long length = tmp_str1.copy(buffer,0,3);
					_interleave = string(buffer);
					*/
					_interleave = string(split+1,3);
				};
				if(str == "byte order "){_byte = atoi(split);};
				count++;
			}
		}
		fin.close();
	}
	
	void ENVIhdr::WriteENVIHeader(const char* filename){
		ofstream fout;
		fout.open(filename);
		if(fout.fail()){
			cout<<"ERROR::[ENVIhdr::WriteENVIHeader]Input filename! -> ";
            cout<<filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
            exit(EXIT_FAILURE);
        }
//		if(mat::Frac(double(_type)/10.) > 1E-3){
//			cout<<"ERROR::[ENVIhdr::WriteENVIHeader]Input file type is invalid for ENVI standard! -> ";
//            cout<<_type<<endl;
//			cout<<"<<Press Enter to Stop>>"; getchar();
//            exit(EXIT_FAILURE);
//		}
		// Set Today time
		new_time::TIME tm;
		tm.SetTodayLocal();
		// Write
		fout<<"ENVI"<<endl;
		fout<<"description = {"<<endl;
		fout<<"  Writed from C++.(Created at:"<<tm.GetTimeString()<<"}"<<endl;
		fout<<"samples = "<<num2str(_xsz)<<endl;
		fout<<"lines   = "<<num2str(_ysz)<<endl;
		fout<<"bands   = "<<num2str(_zsz)<<endl;
		fout<<"header offset = "<<num2str(_offset)<<endl;
		fout<<"file type = ENVI Standard"<<endl;
//		fout<<"data type = "<<num2str(_type/10)<<endl;
		fout<<"data type = "<<_type<<endl;
		fout<<"interleave = "<<_interleave<<endl;
		fout<<"sensor type = Unknown"<<endl;
		fout<<"byte order = "<<num2str(_byte)<<endl;
		fout<<"wavelength units = Unknown"<<endl;
		fout.close();
	}
	
	void ENVIhdr::Print(){
		cout<<"================================"<<endl;
		cout<<"          ENVI Header "<<endl;
		cout<<"================================"<<endl;
		cout<<"samples : "<<"'"<<_xsz<<"'"<<endl;
		cout<<"lines   : "<<"'"<<_ysz<<"'"<<endl;
		cout<<"bands   : "<<"'"<<_zsz<<"'"<<endl;
		cout<<"Header offset : "<<"'"<<_offset<<"'"<<endl;
		cout<<"Data type  : "<<"'"<<_type<<"'"<<endl;
		cout<<"Interleave : "<<"'"<<_interleave<<"'"<<endl;
		cout<<"Byte order : "<<"'"<<_byte<<"'"<<endl;
	}

	// ========================================================
	// Implement for ENVI
	// ========================================================
	//
	// Public
	//
	// Constructor
	template<typename T>
	ENVI<T>::ENVI():_raw(){
		_filename = "";
		_hdr = ENVIhdr();
	}
	
	template<typename T>
	ENVI<T>::ENVI(const string file):_raw(){
		_filename = file;
		
		// Extract file name
		_str = StrFilename(_filename);
		string filename_hdr = _str.path + _str.name + string(".hdr");
		
		// Read ENVI header file (ASCII file)
		_hdr = ENVIhdr(filename_hdr.c_str());
	}
	
	template<typename T>
	ENVI<T>::ENVI(const string file, const long in_xsz, const long in_ysz, const long in_zsz, const long in_offset,
				  const long in_type, const long in_byte, const string in_interleave):_raw(){
		_filename = file;
		
		// Extract file name
		_str = StrFilename(_filename);
		string filename_hdr = _str.path + _str.name + string(".hdr");
		
		// Assign ENVIhdr values
		_hdr = ENVIhdr(in_xsz, in_ysz, in_zsz, in_offset, in_type, in_byte, in_interleave);
		_hdr.filename(filename_hdr.c_str());		
	}
	
	// Utilities
	template<typename T>
	void ENVI<T>::SetENVIheader(const ENVIhdr& in_hdr){
		_hdr.xsz(in_hdr.xsz());
		_hdr.ysz(in_hdr.ysz());
		_hdr.zsz(in_hdr.zsz());
		_hdr.offset(in_hdr.offset());
		_hdr.byte(in_hdr.byte());
		_hdr.interleave(in_hdr.interleave());
	}
	
	template<typename T>
	T& ENVI<T>::ReadENVI(){
		// Recognite data type
		switch(_hdr.type()){
			//
			// using _ReadENVICore3D
			//
			case 1: //1(Byte)
				_ReadENVICore3D(_raw_c); // Read binary
				_raw = (char*)&_raw_c;
				break;
			case 2: //2(integer)
				_ReadENVICore3D(_raw_s); // Read binary
				_raw = (char*)&_raw_s;
				break;
			case 3: //3(long)
				_ReadENVICore3D(_raw_i); // Read binary
				_raw = (char*)&_raw_i;
				break;
			case 4: //4(float)
				_ReadENVICore3D(_raw_f); // Read binary
				_raw = (char*)&_raw_f;
				break;
			case 5: //5(double)
				_ReadENVICore3D(_raw_d); // Read binary
				_raw = (char*)&_raw_d;
				break;
			//
			// using _ReadENVICoreCPLX
			//
			case 6: //6(float complex)
				_ReadENVICoreCPLX(_raw_cxf); // Read binary
				_raw = (char*)&_raw_cxf;
				break;
			case 9: //9(double complex)
				_ReadENVICoreCPLX(_raw_cxd); // Read binary
				_raw = (char*)&_raw_cxd;
				break;
			default:
				cout<<"ERROR::[ENVI::ReadENVI]:No this data type! -> ";
				cout<<_hdr.type()<<endl;
				cout<<"<<Press Enter to Stop>>"; getchar();
				exit(EXIT_FAILURE);
		}
		return _GetDataRef();
	}
	
	template<typename T>
	void ENVI<T>::WriteENVI(const char* out_filename){
		
		// Extract file name
		struct Filename str = def_func::StrFilename(string(out_filename));
		string out_filename_hdr = str.path + str.name + string(".hdr");
		
		// Write ENVI header file (ASCII file)
		_hdr.WriteENVIHeader(out_filename_hdr.c_str());
		
		
		// Write BINARY file
		// Recognite data type
		
		switch(_hdr.type()){
				//
				// using _ReadENVICore3D
				//
			case 1: //1(Byte)
				_raw_c.WriteBinary(out_filename);
				break;
			case 2: //2(intege)
				_raw_s.WriteBinary(out_filename);
				break;
			case 3: //3(long)
				_raw_i.WriteBinary(out_filename);
				break;
			case 4: //4(float)
				_raw_f.WriteBinary(out_filename);
				break;
			case 5: //5(double)
				_raw_d.WriteBinary(out_filename);
				break;
				//
				// using _ReadENVICoreCPLX
				//
			case 6: //6(float complex)
				_raw_cxf.WriteBinary(out_filename);
				break;
			case 9: //9(double complex)
				_raw_cxd.WriteBinary(out_filename);
				break;
			default:
				cout<<"ERROR::[ENVI::WriteENVI]:No this data type! -> ";
				cout<<_hdr.type()<<endl;
				cout<<"<<Press Enter to Stop>>"; getchar();
				exit(EXIT_FAILURE);
		}
	}

	//
	// Private
	//
	template<typename T>
	template<typename T1>
	void ENVI<T>::_ReadENVICore3D(D3<T1>& raw){
		cout<<"ENVI::_ReadENVICore3D, waiting..."<<endl;
		// Prepare binary files
		ifstream fin(_filename.c_str(),ios::binary);
		fin.seekg(_hdr.offset(), ios::beg);
		if(!fin.good()){
			cout<<"ERROR::[ENVI::_ReadENVICore3D]:No this file! -> ";
			cout<<_filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
			exit(EXIT_FAILURE);
		}

		
		if(_hdr.interleave() == string("bsq")){
			// bsq
			long num = _hdr.ysz()*_hdr.xsz()*_hdr.zsz();
			T1* tmp = new T1[num];
			fin.read(reinterpret_cast<char*>(tmp),sizeof(T1)*num);
			raw = D3<T1>(tmp,_hdr.ysz(),_hdr.xsz(),_hdr.zsz(),"BSQ");
		}else if(_hdr.interleave() == string("bil")){
			// bil
			long num = _hdr.ysz()*_hdr.xsz()*_hdr.zsz();
			T1* tmp = new T1[num];
			fin.read(reinterpret_cast<char*>(tmp),sizeof(T1)*num);
			raw = D3<T1>(tmp,_hdr.ysz(),_hdr.xsz(),_hdr.zsz(),"BIL");
		}else{
			// BIP (default)
			raw = D3<T1>(_hdr.ysz(),_hdr.xsz(),_hdr.zsz());
			fin.read(reinterpret_cast<char*>(raw.GetPtr()),sizeof(T1)*_hdr.xsz()*_hdr.ysz()*_hdr.zsz());
		}
		fin.close();
	}

	template<typename T>
	template<typename T1>
	void ENVI<T>::_ReadENVICoreCPLX(D2<CPLX<T1> >& raw){
		cout<<"ENVI::_ReadENVICoreCPLX, waiting..."<<endl;
		// Prepare binary files
		ifstream fin(_filename.c_str(),ios::binary);
		fin.seekg(_hdr.offset(), ios::beg);
		if(!fin.good()){
			cout<<"ERROR::[ENVI::_ReadENVICoreCPLX]:No this file! -> ";
			cout<<_filename<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
			exit(EXIT_FAILURE);
		}
		// allocated memories
		raw = D2<CPLX<T1> >(_hdr.ysz(),_hdr.xsz());
		// Read
		fin.read(reinterpret_cast<char*>(raw.GetPtr()),sizeof(CPLX<T1>)*_hdr.xsz()*_hdr.ysz());
		
		fin.close();
	}

	template<typename T>
	void ENVI<T>::PrintENVIHeader(){
		_hdr.Print();
	}

	template<typename T>
	T& ENVI<T>::_GetDataRef(){
#ifdef _DEBUG
#ifndef _MSC_VER
		int ColorCode;
		ColorCode = (_hdr.type() == 4)? 31:0;
		//cout<<def_func::StrColor::
		cout<<def_func::AddStringColor(string("&(_raw_f)="),ColorCode)<<_raw_f.GetPtr()<<endl;
		ColorCode = (_hdr.type() == 6)? 31:0;
		cout<<def_func::AddStringColor(string("&(_raw_cxf)="),ColorCode)<<_raw_cxf.GetPtr()<<endl;
#else
		cout<<"&(_raw_f)="<<_raw_f.GetPtr()<<endl;
		cout<<"&(_raw_cxf)="<<_raw_cxf.GetPtr()<<endl;
#endif
#endif
		return *((T*)_raw);
	}
}
#endif // ENVI_H_INCLUDED
