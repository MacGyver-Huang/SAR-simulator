#ifndef DEF_PREFUNC_H
#define DEF_PREFUNC_H


#include <iostream>
#include <sstream>
#include <typeinfo>
#include <basic/cplx.h>
#include <basic/vec.h>
#include <coordinate/geo.h>
#include <coordinate/rpy.h>
#include <coordinate/sph.h>
#include <sar/def.h>
#include <mesh/stleach.h>
#include <string>

using namespace std;
using namespace cplx;
using namespace vec;
using namespace geo;
using namespace rpy;
using namespace def;
using namespace sph;

namespace def_prefunc{
	struct Filename{ // file name structure
		string filename,foldername, name, path, type;
	};
	
	template<typename T> void Swap(T& a,T& b){ T c; c=a; a=b; b=c; }
	template<typename T> size_t NumDigits(const T& in);
	template<typename T> size_t Find1stNonZeroDigit(const T val);
	template<typename T> string num2str(const T& i, const int Precision=4);
//	template<typename T> string num2str(const T& i, const int num, const bool IsDecimalPoint=false);
	template<typename T> void str2num(const string& str,T& out);
	template<typename T> T str2num(const string& str);
	TYPE GetType(string type);
	int Type2IDLType(TYPE type);
	long file_line(ifstream& fin);
}

template<typename T>
size_t def_prefunc::NumDigits(const T& in){
	int digits = 0;
	T number = in;
	if (number < 0) digits = 1; // remove this line if '-' counts as a digit
	while (number) {
		number /= 10;
		digits++;
	}
	return digits;
}

template<typename T>
size_t def_prefunc::Find1stNonZeroDigit(const T val){
	// Input value condition
	if( (typeid(T) != typeid(double)) && (typeid(T) != typeid(float)) ){
		cerr<<"ERROR::Find1stNonZeroDigit:The input data type MUST be double or float!"<<endl;
		cerr<<"                           Input type is "<<typeid(T).name()<<endl;
		exit(EXIT_FAILURE);
	}
	
	// Find 1st number after zero
	double tmp = val;
	size_t count = 0;
	for(size_t i=0;i<308;++i){
		tmp *= 10;
		if(int(tmp) > 0){
			break;
		}else{
			count++;
		}
	}
	
	return count;
}

template<typename T>
string def_prefunc::num2str(const T& i, const int Precision){
    /*
     Purpose:
	 Convert number to string.
	 */
	string s;
	stringstream ss(s);
	ss << std::setprecision(Precision) << std::fixed << i;
	
	return ss.str();
}

//template<typename T>
//string def_prefunc::num2str(const T& i, const int num, const bool IsDecimalPoint){
//	/*
//	 Purpose:
//	 Convert number to string.
//
//	 IsDecimalPoint : if the "num" is count for decimal point only.
//	 (e.g.
//	     cout<< num2str(1234.123456789, 4, true) <<endl;
//		 1234.1234
//
//		 cout<< num2str(1234.123456789, 4, false) <<endl;
//		 1234
//	 */
//	string s;
//	stringstream ss(s);
//
//	int NUM;
//
//	if(IsDecimalPoint){
//		NUM = def_prefunc::NumDigits(long(i))+num;
//	}else{
//		NUM = num;
//	}
//
//	ss << std::setprecision(NUM) << i;
//
//	return ss.str();
//}

template<typename T>
void def_prefunc::str2num(const string& str,T& out){
    /*
     Purpose:
	 Convert string to number(float, int, double....)
	 */
	stringstream ss(str);
	ss >> out;
}

template<typename T>
T def_prefunc::str2num(const string& str){
	/*
     Purpose:
	 Convert string to number(float, int, double....)
	 */
	T out;
	stringstream ss(str);
	ss >> out;
	return out;
}

TYPE def_prefunc::GetType(string type){
	if(type == typeid(bool).name()){
		return BOOLEAN;
	}else if(type == typeid(char).name()){
		return CHAR;
	}else if(type == typeid(short).name()){
		return SHORT;
	}else if(type == typeid(int).name()){
		return INT;
	}else if(type == typeid(float).name()){
		return FLT;
	}else if(type == typeid(double).name()){
		return DB;
	}else if(type == typeid(CPLX<float>).name()){
		return CPLX_FLT;
	}else if(type == typeid(string).name()){
		return STR;
	}else if(type == typeid(CPLX<double>).name()){
		return CPLX_DB;
	}else if(type == typeid(long).name()){
		return LONG;
	}else if(type == typeid(STLEach).name()){
		return STLEACH;
	}else if(type == typeid(SPH<float>).name()){
		return SPH_FLT;
	}else if(type == typeid(SPH<double>).name()){
		return SPH_DB;
	}else if(type == typeid(size_t).name()){
		return SIZE_T;
	// Mirror pointer
	}else if(type == typeid(bool*).name()){
		return pBOOLEAN;
	}else if(type == typeid(char*).name()){
		return pCHAR;
	}else if(type == typeid(short*).name()){
		return pSHORT;
	}else if(type == typeid(int*).name()){
		return pINT;
	}else if(type == typeid(float*).name()){
		return pFLT;
	}else if(type == typeid(double*).name()){
		return pDB;
	}else if(type == typeid(CPLX<float>*).name()){
		return pCPLX_FLT;
	}else if(type == typeid(string*).name()){
		return pSTR;
	}else if(type == typeid(CPLX<double>*).name()){
		return pCPLX_DB;
	}else if(type == typeid(long*).name()){
		return pLONG;
	}else if(type == typeid(STLEach*).name()){
		return pSTLEACH;
	}else if(type == typeid(SPH<float>*).name()){
		return pSPH_FLT;
	}else if(type == typeid(SPH<double>*).name()){
		return pSPH_DB;
	}else if(type == typeid(size_t*).name()){
		return pSIZE_T;
	// Not belong to IDL data type
	}else if(type == typeid(VEC<float>).name()){
		return VEC_FLT;
	}else if(type == typeid(VEC<double>).name()){
		return VEC_DB;
	}else if(type == typeid(GEO<float>).name()){
		return GEO_FLT;
	}else if(type == typeid(GEO<double>).name()){
		return GEO_DB;
	}else if(type == typeid(VEC<float>*).name()){
		return pVEC_FLT;
	}else if(type == typeid(VEC<double>*).name()){
		return pVEC_DB;
	}else if(type == typeid(GEO<float>*).name()){
		return pGEO_FLT;
	}else if(type == typeid(GEO<double>*).name()){
		return pGEO_DB;
	}else if(type == typeid(RPY<float>).name()){
		return RPY_FLT;
	}else if(type == typeid(RPY<double>).name()){
		return RPY_DB;
	}else if(type == typeid(RPY<float>*).name()){
		return pRPY_FLT;
	}else if(type == typeid(RPY<double>*).name()){
		return pRPY_DB;
	// Unknow
	}else{
		return UNKNOW;
	}
}

int def_prefunc::Type2IDLType(TYPE type){
	if(type == BOOLEAN){
		return 5;
	}else if(type == CHAR){
		return 10;
	}else if(type == SHORT){
		return 20;
	}else if(type == INT){
		return 30;
	}else if(type == FLT){
		return 40;
	}else if(type == DB){
		return 50;
	}else if(type == CPLX_FLT){
		return 60;
	}else if(type == STR){
		return 70;
	}else if(type == CPLX_DB){
		return 90;
	}else if(type == LONG){
		return 140;
	}else if(type == STLEACH){
		return 150;
	// Mirror pointer
	}else if(type == pBOOLEAN){
		return 6;
	}else if(type == pCHAR){
		return 11;
	}else if(type == pSHORT){
		return 21;
	}else if(type == pINT){
		return 31;
	}else if(type == pFLT){
		return 41;
	}else if(type == pDB){
		return 51;
	}else if(type == pCPLX_FLT){
		return 61;
	}else if(type == pSTR){
		return 71;
	}else if(type == pCPLX_DB){
		return 91;
	}else if(type == pLONG){
		return 141;
	}else if(type == pSTLEACH){
		return 151;
	// Not belong to IDL data type
	}else if(type == VEC_FLT){
		return 210;
	}else if(type == VEC_DB){
		return 220;
	}else if(type == pVEC_FLT){
		return 211;
	}else if(type == pVEC_DB){
		return 221;
	}else if(type == GEO_FLT){
		return 310;
	}else if(type == GEO_DB){
		return 320;
	}else if(type == pGEO_FLT){
		return 311;
	}else if(type == pGEO_DB){
		return 321;
	}else if(type == RPY_FLT){
		return 410;
	}else if(type == RPY_DB){
		return 420;
	}else if(type == pRPY_FLT){
		return 411;
	}else if(type == pRPY_DB){
		return 421;
	// Unknow	
	}else{
		return 0;
	}
}

long def_prefunc::file_line(ifstream& fin){
	long line=0;
	char buf[10000];
	// Get number of line
	while(fin.getline(buf,10000)){
		line++;
	}
	// back to start of ASCII file
	fin.clear();
	fin.seekg(0,ios::beg);
	return line;
}


#endif
