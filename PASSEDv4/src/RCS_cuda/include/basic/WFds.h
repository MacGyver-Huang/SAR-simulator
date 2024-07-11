#ifndef WFDS_H_
#define WFDS_H_
//
//  WFds.h
//  test_WF_data_structure01
//
//  Created by Steve Chiang on 2020/3/20.
//  Copyright © 2020 Steve Chiang. All rights reserved.
//

//#include <map>
#include <set>
//#include <unordered_set>
//#include <vector>
//#include <string>
//#include <algorithm>
//#include <cmath>
//#include <iterator>
//#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
//#include <sys/types.h>
#include <sys/stat.h>
#include "dirent.h"
//#include <basic/def_prefunc.h>
#include <json/json.h>
#include <openssl/md5.h>

using namespace std;


namespace WFds {

	const double FPGA_TX_BANDWIDTH = 491.52e6;
	const double FPGA_CABLE_DISTANCE = 631;		// [m] Cable delay length (for FPGA calibration ONLY)

	//===========================================================================================
	//|                                                                                         |
	//|                                      Functions                                          |
	//|                                                                                         |
	//===========================================================================================
	/**
	 * Define the slash symbol in different OS
	 *
	 * @return Return slash symbol in string
	 */
	string Slash(){
#if defined(_WIN32) || defined(_WIN64)
		return string("\\");
#else
		return string("/");
#endif
	}
	/**
	 * Remote the suffix after "." symbol of filename
	 *
	 * @param [in] filename - File name
	 * @return Return string without suffix
	 */
	string remove_extension(const string& filename){
		const string::size_type p(filename.find_last_of('.'));
		return p > 0 && p != string::npos ? filename.substr(0, p) : filename;
	}
	/**
	 * Execute the "cmd" and feedback with "result"
	 *
	 * @param [in] cmd - (stinrg) Command string
	 * @param [out] result - (string) return message
	 * @return Return a linux code (0 = correct, !=0 incorrect)
	 */
	int Exec(const string cmd, string result) {
		char buffer[128];
		result = "";
		FILE* pipe = popen(cmd.c_str(), "r");
		if (!pipe) throw std::runtime_error("popen() failed!");
		try {
			while (!feof(pipe)) {
				if (fgets(buffer, 128, pipe) != NULL){
					result += buffer;
					cout<<buffer;
				}
			}
		} catch (...) {
			pclose(pipe);
			throw;
		}
		return pclose(pipe);
	}
	/**
	 * Execute the "cmd" without any message and return strings
	 *
	 * @param [in] cmd - (string) Command string
	 */
	void Exec(const string cmd) {
		char buffer[128];
		string result = "";
		FILE* pipe = popen(cmd.c_str(), "r");
		if (!pipe) throw std::runtime_error("popen() failed!");
		try {
			while (!feof(pipe)) {
				if (fgets(buffer, 128, pipe) != NULL){
					result += buffer;
				}
			}
		} catch (...) {
			pclose(pipe);
			throw;
		}
		int ExitCode = pclose(pipe);
		if(ExitCode != 0){
			cerr<<result<<endl;
			exit(EXIT_FAILURE);
		}
	}
	/**
	 * Check the file is exist or not?
	 *
	 * @param [in] file - (string) file name
	 * @return Return true (Exist) or false (not exist)
	 */
	bool FileExist(const string file) {
		struct stat buffer;
		return (stat (file.c_str(), &buffer) == 0);
	}
	/**
	 * Display a prgress bar on the console screen
	 *
	 * @param [in] k - counting index
	 * @param [in] num_idx - total number of index
	 * @param [in] progress_len - display progressbar length
	 * @param [in] step - step of index
	 * @param [out] ProgClock - Return the clock_t class
	 * @param [out] out - Return a message string
	 */
	void ProgressBar(const long k,const long num_idx,const long progress_len,const long step, clock_t& ProgClock, string& out){
		
		out = "";
		if(k==0){
			ProgClock = tic();
			//def::func_time = def_func::tic();
			return;
		}
		if(k==num_idx-1){
			cout<<"\r[";
			for(long m=0;m<progress_len;++m){
				cout<<"=";
			}
			cout<<"] ( "<<k+1<<" / "<<num_idx<<" ) 100.0% Remaining : 0m00s   "<<endl;
		}else if((k%step)==0){
			float progress = float(k+1)/float(num_idx)*float(100.);
			if(progress < 100){
				long p;
				
				double pass= double(tic()-ProgClock)/CLOCKS_PER_SEC;
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
				// export remaining time string
				out = "Remaining : " + num2str(rem_min) + "m" + rem_sec_str + "s";
			}
		}
	}
	/**
	 * Display a prgress bar on the console screen
	 *
	 * @param [in] k - counting index
	 * @param [in] num_idx - total number of index
	 * @param [in] progress_len - display progressbar length
	 * @param [in] step - step of index
	 * @param [out] ProgClock - Return the clock_t class
	 * @param [out] out - Return a message string
	 */
	void GetRemainTime(const long k,const long num_idx,const long step, clock_t& ProgClock, string& out){

		if(k==0){
			ProgClock = tic();
			return;
		}
		if(k==num_idx-1){
			out = "Remaining : ( " + num2str(k+1) + " / " + num2str(num_idx) + " ) 100.0% Remaining : 0m00s   ";
		}else if((k%step)==0){
			float progress = float(k+1)/float(num_idx)*float(100.);
			if(progress < 100){
				double pass= double(tic()-ProgClock)/CLOCKS_PER_SEC;
				//double pass= double(def_func::tic()-def::func_time)/def::clock_ref;
				double rem = pass/(double(k)/double(num_idx))-pass;
				int rem_min=int(rem/60.);
				double rem_sec=rem-rem_min*60.;
				string rem_sec_str=(rem_sec<10)? string("0")+num2str(int(rem_sec)) : num2str(int(rem_sec));
				// export remaining time string
				out = "Remaining : " + num2str(rem_min) + "m" + rem_sec_str + "s";
			}
		}
	}
	/**
	 * Search the files within the folder
	 *
	 * @param [in] folder - folder name
	 * @param [in] prefix - prefix
	 * @param [in] suffix - suffix
	 * @param [in] IsFullPath - Is the input folder is full path?
	 * @return If the condition is matched, return a string vector
	 */
	vector<string> SearchFileInFolder(const string& folder, const string& prefix, const string& suffix, const bool IsFullPath = false){
		DIR* dir;
		struct dirent* ent;
		
		vector<string> out;
		
		if ((dir = opendir(folder.c_str())) != NULL) {
			// print all the files and directories within directory
			while((ent = readdir(dir)) != NULL) {
				string name = ent->d_name;
				D1<string> str_p = StrSplit(name, '_');
				D1<string> str_s = StrSplit(name, '.');
				if(str_p.GetNum() > 1 && str_s.GetNum() > 1){
					if(str_p[0] == prefix && str_s[1] == suffix){
						if(IsFullPath){
							out.push_back(folder + name);
						}else{
							out.push_back(name);
						}
					}
				}
			}
			closedir(dir);
		} else {
			// could not open directory
			cerr<<"ERROR::SearchFileInFolder:Cound not open directory"<<endl;
			exit(EXIT_FAILURE);
		}
		
		return out;
	}
	/**
	 * Search the files within the dolfer without prefix
	 *
	 * @param [in] folder - folder name
	 * @param [in] suffix - suffix
	 * @param [in] IsFullPath - Is the input folder is full path?
	 * @return If the condition is matched, return a string vector
	 */
	vector<string> SearchFileInFolder(const string& folder, const string& suffix, const bool IsFullPath = false){
		DIR *dir;
		struct dirent* ent;
		
		vector<string> out;
		
		if ((dir = opendir(folder.c_str())) != NULL) {
			// print all the files and directories within directory
			while((ent = readdir(dir)) != NULL) {
				string name = ent->d_name;
				D1<string> str_s = StrSplit(name, '.');
				if(str_s.GetNum() > 1){
					if(str_s[1] == suffix){
						if(IsFullPath){
							out.push_back(folder + name);
						}else{
							out.push_back(name);
						}
					}
				}
			}
			closedir(dir);
		} else {
			// could not open directory
			cerr<<"ERROR::SearchFileInFolder:Cound not open directory"<<endl;
			exit(EXIT_FAILURE);
		}
		
		return out;
	}
	/**
	 * Make directory
	 *
	 * @param [in] dir - Directory
	 * @param [in] mode - Permission code (default = "777")
	 * @return Return a linux command code
	 */
	int MakeDir(const string dir, const string mode="777"){
		string cmd = "mkdir -m " + mode + " -p " + dir;
		string result;
		return Exec(cmd, result);
	}
	//===========================================================================================
	//|                                                                                         |
	//|                                        Class                                            |
	//|                                                                                         |
	//===========================================================================================
	/*
	 * LAF (Look/Aspect/Freq) class
	 */
	class LAF {
	public:
		// ctor
		LAF():_look(0),_asp(0),_freq(0){};
		LAF(const LAF& in){
			_look = in._look;
			_asp  = in._asp;
			_freq = in._freq;
		}
		LAF(const double& look, const double& asp, const double& freq){
			_look = look;
			_asp  = asp;
			_freq = freq;
		}
		// Operator overloading
		bool operator<(const LAF& rhs) const{
			return tie(this->_look, this->_asp, this->_freq) < tie(rhs._look, rhs._asp, rhs._freq);
		}
		double operator[](const size_t idx) const {
			if(idx == 0){ return _look; }
			if(idx == 1){ return _asp; }
			if(idx == 2){ return _freq; }
			return 0;
		}
		friend ostream& operator<<(ostream &out, const LAF& in){
			out<<"["<<in._look<<","<<in._asp<<","<<in._freq<<"]";
			return out;
		}
		// Getter & Setter
		double Look()const{return _look;};
		double Asp()const{return _asp;};
		double Freq()const{return _freq;};
		double& Look(){return _look;};
		double& Asp(){return _asp;};
		double& Freq(){return _freq;};
		// Misc.
		void Print(){
			cout<<"+---------------+"<<endl;
			cout<<"|   LAF Class   |"<<endl;
			cout<<"+---------------+"<<endl;
			cout<<" Look   = "<<_look<<" [deg]"<<endl;
			cout<<" Aspect = "<<_asp <<" [deg]"<<endl;
			cout<<" Freq   = "<<_freq<<" [Hz]"<<endl;
		}
	private:
		double _look, _asp, _freq;
	};
	/*
	 * FileName class
	 */
	class FileName {
	public:
		/*
		 * Default constructor
		 */
		FileName(){};
		/**
		 * Check the input string is exist or not?
		 *
		 * @param [in] filename - (string) Input file with full path
		 * @return Return true (exist) or false (not exist)
		 */
		bool IsExist(const string filename){
			if(std::find(_file.begin(), _file.end(), filename) != _file.end()){
				return true;
			}else{
				return false;
			}
		}
		/**
		 * Push back to file dataset
		 *
		 * @param [in] filename - (string) File name with full path
		 * @return Return a pair<size_t, bool>, if the filename is not exist,
		 *		   return true, oterwise return false. The size_t is the
		 *		   index of this filename in _file.
		 */
		pair<size_t, bool> PushBack(const string filename){
			// Search
			pair<size_t, bool> p = Search(filename);
			
			if(!p.second){		// Not exist
				// if not exist
				_file.push_back(filename);
				// return
				return pair<size_t, bool>(_file.size()-1, true);
			}else{				// Exist
				// return
				return pair<size_t, bool>(p.first, false);
			}
		}
		/**
		 * Find the filename is exist or not?
		 *
		 * @param [in] filename (string) File name with full path
		 * @return Return a pair<size_t, bool>, if the filename is not exist,
		 *		   return true, oterwise return false. The size_t is the
		 *		   index of this filename in _file.
		 */
		pair<size_t, bool> Search(const string filename){
			vector<string>::iterator it = std::find(_file.begin(), _file.end(), filename);
			bool IsExist = (it != _file.end())? true:false;
			
			size_t idx = 0;
			idx = std::distance(_file.begin(), it);
			
			return pair<size_t, bool>(idx, IsExist);
		}
		/**
		 * Get value by index
		 *
		 * @param [in] idx - (size_t) The index (MUST < _file.size())
		 * @return Return a value in string by input idx
		 */
		string GetValue(const size_t idx){
			if(idx >= _file.size()){
				cerr<<"ERROR::FileName:GetValue: The input idx larger than the size of _file\n";
				cerr<<"       _file.size = "<<_file.size()<<", idx = "<<idx<<"\n";
				exit(EXIT_FAILURE);
			}
			return _file[idx];
		}
		/**
		 * Remove all elements
		 */
		void Clear(){
			_file.clear();
		}
		/**
		 * Get the size of this data structure
		 *
		 * @return Return the size of unordered_set<string> _file;
		 */
		size_t GetSize()const{
			return _file.size();
		}
		/**
		 * Write this filename dataset into a file
		 *
		 * @param [in] filename_out - (stinrg) Output file name
		 * @return Return a boolean, success (true) or fail (false)
		 */
		bool Write(const string filename_out){
			//
			// Write vector of string into a file
			//
			ofstream fout;
			fout.open(filename_out.c_str(), ios::binary);
			if(!fout.is_open()){
				cerr<<"ERROR::Write:Cannot open into the file\n";
				return false;
			}
			
			string str;
			size_t sz;
			
			// write number of vector
			sz = _file.size();
			fout.write((char*)&sz, sizeof(size_t));
			
			for(auto it=_file.begin(); it!=_file.end(); ++it){
				str = *it;
				sz  = str.size();
				// Write string size
				fout.write((char*)&sz, sizeof(size_t));
				// Write the string contain
				fout.write(&str[0], sz*sizeof(char));
			}
			
			fout.close();
			return true;
		}
		/**
		 * Read this filename dataset from a file
		 *
		 * @param [in] filename_in - (stinrg) Input file name
		 * @return Return a boolean, success (true) or fail (false)
		 */
		bool Read(const string filename_in){
			//
			// Read vector of string into a file
			//
			string str;
			size_t sz_elem, sz;
			
			// Earse all contain in _file
			_file.clear();
			
			
			ifstream fin;
			fin.open(filename_in.c_str(), ios::binary);
			if(!fin.is_open()){
				cerr<<"ERROR::Write:Cannot open from the file\n";
				return false;
			}
			
			// Read number of vector
			fin.read((char*)&sz_elem, sizeof(size_t));
			
			// Read into _file
			for(size_t i=0;i<sz_elem;++i){
				// Read size
				fin.read((char*)&sz, sizeof(sz));
				// Read string
				str.resize(sz);
				fin.read((char*)&str[0], sz*sizeof(char));
				// Push back (no need to check unique
				_file.push_back(str);
			}
			fin.close();
			
			return true;
		}
		/**
		 * Print the first 10 elements
		 */
		void Print(){
			cout<<"+----------------+\n";
			cout<<"|     Print      |\n";
			cout<<"+----------------+\n";
			cout<<"| Print 10 items |\n";
			cout<<"+----------------+\n";
			size_t count = 0;
			for(size_t i=0;i<_file.size();++i){
				if(count < 10){
					cout<<"idx = "<<i<<", '"<<_file[i]<<"'\n";
				}
				count++;
			}
		}
	public:
		vector<string> _file;
	};
	/*
	 * "IsMatchKey1" Functor
	 */
	class IsMatchKey1 {
	public:
		// functor
		template< typename T1, typename T2 >
		bool operator()( T1 const& lhs, T2 const& rhs ) const {
			return asInt(lhs) < asInt(rhs);
		}
	private:
		int asInt( const pair<LAF, size_t>& v ) const {
			return v.first.Look();
		}
		int asInt( LAF t ) const {
			return t.Look();
		}
	};
	/*
	 * Dict class
	 * 		Look = [0,360,step=2], num_look = 180
	 * 		Asp,  num_asp  = 10000
	 * 		freq, num_freq = 10000
	 *
	 * 		Total size:
	 *    		num_look * num_asp * num_freq = 180*10000*10000*8 = 144 [GBytes] (worest case, dense matrix)
	 */
	class Dict {
	public:
		typedef typename multimap<LAF, size_t>::iterator Itr;
	public:
		/**
		 * Default constructor
		 */
		Dict(){}
		/**
		 * Insert a value with key
		 *
		 * @param [in] key1_look - Key 1 look angle
		 * @param [in] key2_asp - Key 2 aspect angle
		 * @param [in] key3_freq - Key3 frequency
		 * @param [in] value - File name string
		 * @return If a insert enable return "true", otherwise return "false"
		 */
		bool Insert(const double key1_look, const double key2_asp, const double key3_freq, const string& value){
			LAF key(key1_look, key2_asp, key3_freq);
			
			bool isExist = IsExist(key);
			if(isExist == false){
				//
				// (2) Push back the value into the "_filename"
				//
				pair<size_t, bool> p = _filename.PushBack(value);
				//
				// (3) Insert the key and index into "_data"
				//
				// _data : multimap<Triple, size_t>
				_data.insert(make_pair(key, p.first));
				return true;
			}else{
				return false;
			}
		}
		/**
		 * Return the number in this "_data"
		 *
		 * @return Return the data size
		 */
		size_t GetDataSize()const{
			return _data.size();
		}
		/**
		 * Return the number in this "_file"
		 *
		 * @return Return the file data size
		 */
		size_t GetFileSize()const{
			return _filename.GetSize();
		}
		/**
		 * Search and extract the Key1 matched all multimap
		 *
		 * @param [in] Key1 - (double) Input key1 value
		 * @return Return a range with pair<Itr, Itr> type
		 */
		pair<Itr, Itr> Search(const double Key1){
			auto rg = std::equal_range(_data.begin(), _data.end(),
									   pair<LAF, size_t>(LAF(Key1,0,0),12345),
									   IsMatchKey1());
			return rg;
		}
		/**
		 * Search a list for all "key1~key3"
		 *
		 * @param [in] KeyVal - 2 Keys {look, aspect, frequency} [deg,deg,Hz]
		 * @return Return this key is exist(true) or not(false)?
		 */
		bool IsExist(const LAF KeyVal){
			Itr it = _data.find(KeyVal);
			if(it != _data.end()){
				return true;	// found
			}else{
				return false;	// not found
			}
		}
		/**
		 * Get the unique set of look angle series
		 *
		 * @return A unique set of look angle series
		 */
		set<double> GetUniqueLook(){
			set<double> look;
			for(auto it = _data.begin(); it != _data.end(); it = _data.upper_bound(it->first)){
				look.insert(it->first.Look());
			}
			return look;
		}
		/**
		 * Write this Dict to file
		 *
		 * @param [in] dir - Output directory
		 * @return Return the write procedure is correct (true) or not (false)?
		 * @note
		 *		"Filename.dat"	: Save the unique full path file
		 *		"Look.dat"		: Save the look angle series
		 *		"data_Look#.dat": Save all {key2,key3,idx} set for each look#
		 */
		bool Write(const string dir){
			ofstream fout;
			//
			// (A) Write _file
			//		+--------+------------+---------+
			//		|  size  |    char*   | ...     |
			//		+--------+------------+---------+
			//		  size_t    dynamic
			//
			_filename.Write(dir+Slash()+"Filename.dat");
			//
			// (B) Write _data
			//
			set<double> look = GetUniqueLook();
			//
			// (1) Write look table
			//		+--------+-------+-------+---------+
			//		|  size  | Look1 | Look1 | ...     |
			//		+--------+-------+-------+---------+
			//		  size_t   double  double
			//
			fout.open(dir+Slash()+"Look.dat", ios::binary);
			if(!fout.is_open()){
				cerr<<"ERROR::Write:Cannot open Look.dat file\n";
				return false;
			}
			size_t sz = look.size();
			fout.write((char*)&(sz), sizeof(size_t));
			for(auto it = look.begin(); it != look.end(); ++it){
				fout.write((char*)&(*it), sizeof(double));
			}
			fout.close();
			//
			// (2) Write _data by each Look integer
			//
			//		For each "data_LookXX.dat"
			//		+-------+------+-------+------------+
			//		|  Asp  | Freq | index |     ...    | .....
			//		+-------+------+-------+------------+
			//		 double  double  size_t
			//      <--------- #1 --------> <-- #2 -->...	 [iit]
			//      <----------- Look #1 --------------->... [it]
			//
			double Asp, Freq;
			for(auto it = look.begin(); it != look.end(); ++it){	// each look angle
				// Get filename with full path
				string file_name = "data_Look"+num2str(*it)+".dat";
				string file_data = dir+Slash()+file_name;
				// Get the range
				auto rg = Search(*it);
				// Write into a file for each look angle
				fout.open(file_data.c_str(), ios::out | ios::binary );
				if(!fout){
					cout<<"ERROR::Dict:Write: Cannot open a file\n";
					return false;
				}
				for(Itr iit=rg.first; iit != rg.second; ++iit){
					Asp = iit->first.Asp();
					Freq= iit->first.Freq();
					// Save key
					fout.write((char*)&Asp,	 sizeof(double));
					fout.write((char*)&Freq, sizeof(double));
					// Save value
					fout.write((char*)&(iit->second), sizeof(size_t));
				}
				if(!fout.is_open()){
					cout<<"ERROR::Dict:Write: Cannot wrtie a file\n";
					return false;
				}
				fout.close();
			}
			return true;
		}
		/**
		 * Read this Dict from file
		 *
		 * @param [in] dir - Input directory
		 * @return Return the read procedure is correct (true) or not (false)?
		 */
		bool Read(const string dir){
			ifstream fin;
			//
			// (0) Prepare
			//
			string file_Filename = dir+Slash()+"Filename.dat";
			string file_Look     = dir+Slash()+"Look.dat";
			//
			// (1) Clean all & check exist
			//
			_filename.Clear();
			_data.clear();
			if(!IsExist(file_Filename) || FileSize(file_Filename) == 0){
				cerr<<"WARNNING::Dict:Read:The Filename.dat is empty"<<endl;
				cerr<<"                    No this file or file size is zero"<<endl;
				return false;
			}
			//
			// (2) Read into _file "Filename.dat"
			//
			_filename.Read(file_Filename);
			//
			// (3) Read Look sereis "Look.dat"
			//
			// Read "Look.dat"
			fin.open(file_Look);
			if(!fin.is_open()){
				cerr<<"ERROR::Dict::Read:Cannot open Look.dat file\n";
				return false;
			}
			size_t sz;
			double num;
			vector<double> Look;
			fin.read((char*)&(sz), sizeof(size_t));
			for(size_t i=0;i<sz;++i){
				fin.read((char*)&num, sizeof(double));
				Look.push_back(num);
			}
			fin.close();
			//
			// (4) Read "data_Look*.dat" into _data
			//
			for(size_t i=0;i<Look.size();++i){
				// Get filename with full path
				string file_name = "data_Look"+num2str(Look[i])+".dat";
				string file_data = dir+Slash()+file_name;
				// Open
				fin.open(file_data.c_str(), ios::binary);
				if(!fin.is_open()){
					cerr<<"ERROR::Dict::Read:Cannot open from the file\n";
					return false;
				}
				// Get file size
				size_t FileSz = FileSize(file_data);
				// Read
				double asp, freq;
				size_t idx, pos;
				while(!fin.eof()){
					// Read key
					fin.read((char*)&asp,  sizeof(double));
					fin.read((char*)&freq, sizeof(double));
					// Read idx
					fin.read((char*)&idx, sizeof(size_t));
					// Insert
					_data.insert(make_pair(LAF(Look[i], asp, freq), idx));
					// check EOF
					pos = fin.tellg();
					if(pos == FileSz){ break; }
				}
				fin.close();
			}
			return true;
		}
		/**
		 * Print results by iterator
		 *
		 * @param [in] it - The vector of iterator
		 */
		void Print(const Itr& it){
			int N_dig = DigitCount(_filename.GetSize()-1);
			cout.setf(ios::fixed, ios::floatfield);
			cout.precision(4);
			cout<<"key=["<<setw(8)<<it->first.Look()<<", "<<
							 setw(8)<<it->first.Asp()<<", "<<
							 setw(16)<<it->first.Freq()<<"], "<<
				  "idx="<<setw(N_dig)<<it->second<<", "<<
				  "val='"<<_filename.GetValue(it->second)<<"'\n";
		}
		/**
		 * Display the elements in this Dict 
		 *
		 * @param [in] num - Maximum number to display (default = 20)
		 */
		void Print(const size_t num=20){
			cout<<"+----------------+\n";
			cout<<"|     Print      |\n";
			cout<<"+----------------+\n";
			cout<<"| Print "<<num<<" items |\n";
			cout<<"+----------------+\n";
			cout<<"| Key 1 = Look   |\n";
			cout<<"| Key 2 = Aspect |\n";
			cout<<"| Key 3 = Freq   |\n";
			cout<<"+----------------+\n";
			size_t count = 0;
			for(auto it=_data.begin();it!=_data.end();++it){
				if(count < num){
					Print(it);
				}
				count++;
			}
		}
		/**
		 * Print the file vector
		 */
		void PrintFile(){
			_filename.Print();
		}
	private:
		/**
		 * To get the file size
		 *
		 * @param [in] filename - (string) file name with full path
		 * @return Return the size of file in bytes
		 */
		size_t FileSize(const string& filename){
			ifstream in(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
			return in.tellg();
		}
		/**
		 * Get digital count of integer
		 *
		 * @param [in] n - (size_t) A unsigned integer
		 * @return Return the number of digit
		 */
		int DigitCount(size_t n){
			int count = 0;
			while(n != 0){
				n = n / 10;
				++count;
			}
			return count;
		}
		/**
		 * Check the file is exist or not?
		 *
		 * @param [in] filename - (string) A file name with full path
		 * @return True (exist) or False (not exist)
		 */
		bool IsExist(const string& filename) {
			ifstream f(filename.c_str());
			return f.good();
		}

	private:
		FileName _filename;				// filename vector
		multimap<LAF, size_t> _data;	// data storage
	};
	/*
	 * FLP (Fligh path Class)
	 *		time
	 *		llh : Longitude/Latitude/Height
	 *		look & squint angle
	 */
	class Flp {
	public:
		Flp():_look(0),_sq(0){};
		Flp(const TIME& time, const GEO<double>& llh, const double look, const double sq){
			_time = time;
			_llh  = llh;
			_look = look;
			_sq   = sq;
		}
		Flp(const Flp& in){
			_time = in._time;
			_llh  = in._llh;
			_look = in._look;
			_sq   = in._sq;
		}
		// Getter & setter
		TIME Time()const{return _time;};
		double Look()const{return _look;};
		double Squint()const{return _sq;};
		GEO<double> LLH()const{return _llh;};
		TIME& Time(){return _time;};
		double& Look(){return _look;};
		double& Squint(){return _sq;};
		GEO<double>& LLH(){return _llh;};
		// Misc.
		void Print(){
			cout<<"+------------------------+"<<endl;
			cout<<"|        Flp Class       |"<<endl;
			cout<<"+------------------------+"<<endl;
			cout<<std::setprecision(20);
			cout<<_time<<endl;
			cout<<_llh<<endl;
			cout<<"Look    = "<<_look<<endl;
			cout<<"Squint  = "<<_sq<<endl;
		}
	private:
		TIME _time;
		GEO<double>  _llh;
		double _look, _sq;
	};
	/*
	 * ANG (Angle class)
	 *		look, squint, aspect angle series
	 *		isCal  (true or false)
	 */
	class ANG {
	public:
		ANG():_num(0){};
		ANG(const size_t num){
			_num  = num;
			_look = D1<double>(_num);
			_sq   = D1<double>(_num);
			_asp  = D1<double>(_num);
		}
		ANG(const ANG& in){
			_num  = in._num;
			_look = in._look;
			_sq   = in._sq;
			_asp  = in._asp;
		}
		ANG(const SV<double>& sv_int, const VEC<double>& Pt, const double PRF, const bool NormalizeHeight=false, const bool SHOW=false){

			// Predefined
			ORB Orb;
			// Duplicate
			SV<double> sv_int2 = sv_int;
			//+-----------------------------------------------+
			//|       Re-normalize the altitude of SV         |
			//+-----------------------------------------------+
			if(NormalizeHeight){
				// Normalize height
				D1<double> h(sv_int2.GetNum());
				for(long i=0;i<sv_int2.GetNum();++i){
					h[i] = sar::ECR2Gd(sv_int2.pos()[i], Orb).h();
				}
				double hmean = mat::total(h)/h.GetNum();
				for(long i=0;i<sv_int2.GetNum();++i){
					GEO<double> gd = sar::ECR2Gd(sv_int2.pos()[i], Orb);
					gd.h() = hmean;
					sv_int2.pos()[i] = sar::Gd2ECR(gd, Orb);
				}
			}

			D1<SPH<double> > locSPH = sar::find::LocalSPH(sv_int2, Pt, 1./PRF, Orb, SHOW);

			//+-----------------------------------------------+
			//|             Assign to ANG class               |
			//+-----------------------------------------------+
			_num = locSPH.GetNum();
			_look = D1<double>(_num);
			_sq   = D1<double>(_num);
			_asp  = D1<double>(_num);

			// Nearest Ps
			VEC<double> PsCn = sar::find::NearestPs(sv_int2, Pt, (1./PRF)/10);
			VEC<double> PsCng= sar::find::ProjectPoint(PsCn, Orb);
			VEC<double> PsCnPsCng = Unit(PsCng - PsCn);
			VEC<double> PtPsCn    = Unit(PsCn  - Pt);
			for(size_t i=0;i<_num;++i){
				VEC<double> Ps  = sv_int2.pos()[i];
				VEC<double> Ps1 = sv_int2.NextPs(i);
				VEC<double> Psg= sar::find::ProjectPoint(Ps, Orb);
				VEC<double> PsPt  = Unit(Pt  - Ps);
				VEC<double> PsPsg = Unit(Psg - Ps);
				VEC<double> PtPs  = -PsPt;
				VEC<double> PsPs1 = Unit(Ps1 - Ps);
				// theta_l
				_look[i] = angle(PsPsg, PsPt);
				// theta_sq
				double sgn = (dot(PsPt, PsPs1) > 0)? +1:-1;
				_sq[i] = sgn * angle(PtPsCn, PtPs);
				// theta_az
				_asp[i] = locSPH[i].Phi();
			}

			if(SHOW){ Print(); }
		}
		// Getter & setter
		size_t GetNum()const{return _num;};
		D1<double> Look()const{return _look;};
		D1<double> Squint()const{return _sq;};
		D1<double> Asp()const{return _asp;};
		D1<double>& Look(){return _look;};
		D1<double>& Squint(){return _sq;};
		D1<double>& Asp(){return _asp;};
		// Misc.
		void PrintListAll(){
			cout << "+------------------------------------------+" << endl;
			cout << "|                ANG Class                 |" << endl;
			cout << "+------------------------------------------+" << endl;
			cout << "| Look [deg]  Squint [deg]  Aspect [deg]   |" << endl;
			cout << "+------------------------------------------+" << endl;
			for (size_t i = 0; i < _num; ++i) {
				cout << std::fixed << std::setw(10) << rad2deg(_look[i]) << "       " << rad2deg(_sq[i]) <<
					 "       " << rad2deg(_asp[i]);
				if (i == (_num - 1) / 2) {
					cout << "  (center)" << endl;
				} else {
					cout << endl;
				}
			}
		}
		void Print(){
			cout<<"+------------------------------------------+"<<endl;
			cout<<"|                ANG Class                 |"<<endl;
			cout<<"+------------------------------------------+"<<endl;
			cout<<"Number = "<<_num<<endl;
			cout<<"Note: [0, center, end]"<<endl;
			double theta_l_min   = rad2deg(_look[0]);
			double theta_l_c     = rad2deg(_look[(_num-1)/2]);
			double theta_l_max   = rad2deg(_look[_num-1]);
			double theta_sq_min  = rad2deg(_sq[0]);
			double theta_sq_c    = rad2deg(_sq[(_num-1)/2]);
			double theta_sq_max  = rad2deg(_sq[_num-1]);
			double theta_asp_min = rad2deg(_asp[0]);
			double theta_asp_c   = rad2deg(_asp[(_num-1)/2]);
			double theta_asp_max = rad2deg(_asp[_num-1]);
			printf("theta_l   = [%f,%f,%f]\n", theta_l_min,   theta_l_c,   theta_l_max);
			printf("theta_sq  = [%f,%f,%f]\n", theta_sq_min,  theta_sq_c,  theta_sq_max);
			printf("theta_asp = [%f,%f,%f]\n", theta_asp_min, theta_asp_c, theta_asp_max);
		}
	private:
		size_t _num;
		D1<double> _look, _sq, _asp;
	};
	/*
	 * RCSFolder (RCS folder class)
	 *		look folder name
	 *		aspect catalog folder name
	 *		aspect file name (*.dat)
	 *		full path
	 */
	class RCSFolder {
	public:
		RCSFolder(){};
		RCSFolder(const RCSFolder& in){
			_look = in._look;
			_asp_catlg = in._asp_catlg;
			_asp  = in._asp;
			_full = in._full;
		}
		RCSFolder(const string& folder_look, const string& folder_asp_catlg,
				  const string& filename_asp, const string& folder_filename_full){
			_look = folder_look;
			_asp_catlg = folder_asp_catlg;
			_asp  = filename_asp;
			_full = folder_filename_full;
		}
		string Look()const{return _look;};
		string Asp_catlg()const{return _asp_catlg;};
		string Asp()const{return _asp;};
		string Full()const{return _full;};
		string& Look(){return _look;};
		string& Asp_catlg(){return _asp_catlg;};
		string& Asp(){return _asp;};
		string& Full(){return _full;};
		// Misc
		string GetRCSPath(const string& dir_dbs){
			return dir_dbs + Slash() + _look + Slash() + _asp_catlg + Slash() + _asp + ".rcs";
		}
		void Print(){
			cout<<"+---------------------+"<<endl;
			cout<<"|   RCSFolder Class   |"<<endl;
			cout<<"+---------------------+"<<endl;
			cout<<"Look : "<<_look<<endl;
			cout<<"Asp_Catalog : "<<_asp_catlg<<endl;
			cout<<"Asp  : "<<_asp<<endl;
			cout<<"Full : "<<_full<<endl;
		}
	private:
		string _look, _asp_catlg, _asp, _full;
	};
	/**
	 * TargetType parameter class
	 */
	class TargetTYPE {
	public:
		TargetTYPE():_target_type(""),_single_rcs(0),_waveform_type(""){};
		TargetTYPE(const string targetType, const double singleRcs, const string waveformType, const string codeType, const string codeString){
			_target_type   = targetType;
			_single_rcs    = singleRcs;
			_waveform_type = waveformType;
			_code_type     = codeType;
			_code_string   = codeString;
		}
		string targetType()const{ return _target_type; }
		double SingleRCS()const{ return _single_rcs; }
		string WaveformType()const{ return _waveform_type; }
		string CodeType()const{ return _code_type; }
		string CodeString()const{ return _code_string; }
		string& targetType(){ return _target_type; }
		double& SingleRCS(){ return _single_rcs; }
		string& WaveformType(){ return _waveform_type; }
		string& CodeType(){ return _code_type; }
		string& CodeString(){ return _code_string; }
		void Print(){
			cout<<"+---------------------+"<<endl;
			cout<<"|   TargetType Class  |"<<endl;
			cout<<"+---------------------+"<<endl;
			cout<<"Target type       : '"<<_target_type<<"'"<<endl;
			cout<<"Single RCS [dB]   : "<<_single_rcs<<endl;
			cout<<"Waveform type     : '"<<_waveform_type<<"'"<<endl;
			cout<<"Code type         : '"<<_code_type<<"' (if the WaveformType = LFM, this is empty)"<<endl;
			cout<<"Code string       : '"<<_code_string<<"' (if the WaveformType = LFM, this is empty)"<<endl;
		}
	private:
		string _target_type;		// (string) "complex" or "single"
		double _single_rcs;			// [dB] Single target RCS value
		string _waveform_type;		// (string) Waveform type: "coding", "LFM"
		string _code_type;			// (string) Waveform type "type0", "type1", "type2", "type3", "type4"
		string _code_string;		// (string) Coding string
	};
	/**
	 * FPGA parameter class
	 */
	class FPGA {
	public:
		FPGA(): _inputGain(0), _outputGain(0), _powerLevel(0), _fpgaMode(1), _signalOffset(0), _isRealTimeConv(false),
					 _attenuation_type("single"), _Pt_dBm(0), _Gt_dBi(0), _imageGain(0), _distance(0), _PathLoss(0),
					 _distanceCnt(0), _increasing_time(0){};
		FPGA(const long InputGain, const long OutputGain, const long PowerLevel, const long FpgaMode, const long SignalOffset, const bool IsRealTimeConv,
			 // Attenuation
			 const string Attenuation_type, const double Pt_dBm, const double Gt_dBi, const long ImageGain, const double Distance, const double PathLoss,
			 // other
			 const long distanceCnt, const long Increasing_time){
			// FPGA options
			_outputGain = OutputGain;
			_powerLevel = PowerLevel;
			_inputGain = InputGain;
			_fpgaMode = FpgaMode;
			_signalOffset = SignalOffset;
			_isRealTimeConv = IsRealTimeConv;
			// Attenuation
			_attenuation_type = Attenuation_type;
			_Pt_dBm = Pt_dBm;
			_Gt_dBi = Gt_dBi;
			_imageGain = ImageGain;
			_distance  = Distance;
			_PathLoss  = PathLoss;
			// others
			_distanceCnt = distanceCnt;
			_increasing_time = Increasing_time;
		}
		// get
		long InputGain()const{return _inputGain;};
		long OutputGain()const{return _outputGain;};
		long PowerLevel()const{return _powerLevel;};
		long FpgaMode()const{return _fpgaMode;};
		long SignalOffset()const{return _signalOffset;};
		bool IsRealTimeConv()const{return _isRealTimeConv;};
		string AttenuationType()const{return _attenuation_type;}
		double Pt_dBm()const{ return _Pt_dBm; }
		double Gt_dBi()const{return _Gt_dBi;};
		long ImageGain()const{return _imageGain;};
		double Distance()const{return _distance;};
		double PathLoss()const{return _PathLoss;};
		long DistanceCnt()const{return _distanceCnt;};
		long Increasing_time()const{return _increasing_time;};
		// set
		long& InputGain(){return _inputGain;};
		long& OutputGain(){return _outputGain;};
		long& PowerLevel(){return _powerLevel;};
		long& FpgaMode(){return _fpgaMode;};
		long& SignalOffset(){return _signalOffset;};
		bool& IsRealTimeConv(){return _isRealTimeConv;};
		string& AttenuationType(){return _attenuation_type;}
		double& Pt_dBm(){ return _Pt_dBm; }
		double& Gt_dBi(){return _Gt_dBi; }
		long& ImageGain(){return _imageGain;};
		double& Distance(){return _distance;};
		double& PathLoss(){return _PathLoss;};
		long& DistanceCnt(){return _distanceCnt;};
		long& Increasing_time(){return _increasing_time;};
		// Misc.
		void Print(){
			string isRealTimeConv = (_isRealTimeConv)? "true":"false";
			cout<<"+---------------------+"<<endl;
			cout<<"|      FPGA Class     |"<<endl;
			cout<<"+---------------------+"<<endl;
			cout<<"+----------------+"<<endl;
			cout<<"|  FPGA options  |"<<endl;
			cout<<"+----------------+"<<endl;
			cout<<"Input Gain [2^n]         : "<<_inputGain<<endl;
			cout<<"Output Gain [2^n]        : "<<_outputGain<<endl;
			cout<<"Power Level [2^n]        : "<<_powerLevel<<endl;
			cout<<"FPGA mode (1~5)          : "<<_fpgaMode<<endl;
			cout<<"Signal offset (0~255)    : "<<_signalOffset<<endl;
			cout<<"Is Real Time Convolution : "<<isRealTimeConv<<endl;
			cout<<"+----------------+"<<endl;
			cout<<"|  Attenuation   |"<<endl;
			cout<<"+----------------+"<<endl;
			cout<<"Attenuation type         : "<<_attenuation_type<<endl;
			cout<<"Pt_dBm [dBm]             : "<<_Pt_dBm<<endl;
			cout<<"Gt_dBi [dBi]             : "<<_Gt_dBi<<endl;
			cout<<"Image Gain [2^n]         : "<<_imageGain<<endl;
			cout<<"Distance [m]             : "<<_distance<<endl;
			cout<<"PathLoss [dB]            : "<<_PathLoss<<endl;
			cout<<"+----------------+"<<endl;
			cout<<"|     Others     |"<<endl;
			cout<<"+----------------+"<<endl;
			cout<<"Distance count [sample]  : "<<_distanceCnt<<endl;
			cout<<"Increasing_time [sample] : "<<_increasing_time<<endl;
		}
	private:
		// FPGA options
		long   _inputGain;			// [2^n] Input gain
		long   _outputGain;			// [2^n] Output gain
		long   _powerLevel;			// [2^n] Power Level
		long   _fpgaMode;			// [x] (1~25) FPGA mode
		long   _signalOffset;		// [x] (0~255) Signal offset
		bool   _isRealTimeConv;		// (boolean) Is real time convolution?
		// Attenuation
		string _attenuation_type;	// (string) "free_space" or "cable"
		double _Pt_dBm;				// [dBm] Transmit power
		double _Gt_dBi;				// [dBi] Transmit gain
		long _imageGain;			// [2^n] FPGA gain
		double _distance;			// [m] Physical distance between antenna of Emulator and seeker's antenna
		double _PathLoss;			// [dB] Path loss
		// Others
		long   _distanceCnt;		// [sample] Radar distance in count for zero calibration
		long   _increasing_time;	// [sample] Calculate from pulse width floor(Tr/245.5M)
	};


//	class MetaFPGA {
//	public:
//		MetaFPGA(){}
//		MetaFPGA(string attenuation_type, double increasing_time, double distance, double path_loss, long RadarDistCnt){
//			_attenuation_type = attenuation_type;
//			_increasing_time  = increasing_time;
//			_distance = distance;
//			_path_loss = path_loss;
//			_RadarDistCnt = RadarDistCnt;
//		}
//		// Getter
//		const string& attenuation_type() const { return _attenuation_type; }
//		double increasing_time() const { return _increasing_time; }
//		double distance() const { return _distance; }
//		double path_loss() const { return _path_loss; }
//		long   RadarDistCnt() const { return _RadarDistCnt; }
//		// Setter
//		string& attenuation_type() { return _attenuation_type; }
//		double& increasing_time() { return _increasing_time; }
//		double& distance() { return _distance; }
//		double& path_loss() { return _path_loss; }
//		long&   RadarDistCnt() { return _RadarDistCnt; }
//		// Misc.
//		void Print(){
//			cout<<"+--------------+"<<endl;
//			cout<<"|  Fpga Class  |"<<endl;
//			cout<<"+--------------+"<<endl;
//			cout<<"attenuation_type = "<<_attenuation_type<<endl;
//			cout<<"increasing_time  = "<<_increasing_time<<" [sec]"<<endl;
//			cout<<"distance         = "<<_distance<<" [m]"<<endl;
//			cout<<"path_loss        = "<<_path_loss<<" [dB]"<<endl;
//			cout<<"Radar Dist Cnt   = "<<_RadarDistCnt<<" [cnt]"<<endl;
//		}
//	private:
////		string _attenuation_type;
//		double _increasing_time;
//		double _distance;
////		double _path_loss;
////		long   _RadarDistCnt;
//	};

	class Meta {
	public:
		Meta(){}
		Meta(double look_min, double look_max, double asp_min, double asp_max, double f0, double Fr, double BWrg, double PRF,
			 double Rn_slant, double Rc_slant, double Rf_slant, double theta_sq_mean, double theta_l_mean, double Vs_mean,
			 const FPGA& fpga, size_t Nr, size_t Na, const string& tx_pol,
			 const string& rx_pol, double Maxlevel, double scale, double dR,
			 // Add
			 double Tr, double dAsp, double df, bool isUpChirp,
			 const TargetTYPE& targetType){
//			 string& targetType, double singleRCS, string& waveformType, string& codeType,
//			 string codeString){
//			 int peakPower, int powerWidth, int triggerTrhreshold, int fpgaMode, int signalshift, bool realTimeConv) {
			_look_min = look_min;
			_look_max = look_max;
			_asp_min = asp_min;
			_asp_max = asp_max;
			_f0 = f0;
			_Fr = Fr;
			_BWrg = BWrg;
			_PRF = PRF;
			_Rn_slant = Rn_slant;
			_Rc_slant = Rc_slant;
			_Rf_slant = Rf_slant;
			_theta_sq_mean = theta_sq_mean;
			_theta_l_mean = theta_l_mean;
			_Vs_mean = Vs_mean;
			_fpga = fpga;
			_Nr = Nr;
			_Na = Na;
			_tx_pol = tx_pol;
			_rx_pol = rx_pol;
			_Maxlevel = Maxlevel;
			_scale = scale;
			_dR = dR;
			// Add
			_Tr = Tr;
			_dAsp = dAsp;
			_df = df;
			_isUpChirp = isUpChirp;
			_targetType = targetType;
//			_singleRCS = singleRCS;
//			_waveformType = waveformType;
//			_codeType = codeType;
//			_codeString = codeString;
//			_peakPower = peakPower;
//			_powerWidth = powerWidth;
//			_triggerTrhreshold = triggerTrhreshold;
//			_fpgaMode = fpgaMode;
//			_signalShift = signalshift;
//			_realTimeConv = realTimeConv;
		}
		// Getter
		double look_min() const { return _look_min; }
		double look_max() const { return _look_max; }
		double asp_min() const { return _asp_min; }
		double asp_max() const { return _asp_max; }
		double f0() const { return _f0; }
		double Fr() const { return _Fr; }
		double BWrg() const { return _BWrg; }
		double PRF() const { return _PRF; }
		double Rn_slant() const { return _Rn_slant; }
		double Rc_slant() const { return _Rc_slant; }
		double Rf_slant() const { return _Rf_slant; }
		double theta_sq_mean() const { return _theta_sq_mean; }
		double theta_l_mean() const { return _theta_l_mean; }
		double Vs_mean() const { return _Vs_mean; }
		const FPGA& Fpga() const { return _fpga; }
		size_t Nr() const { return _Nr; }
		size_t Na() const { return _Na; }
		const string& tx_pol() const { return _tx_pol; }
		const string& rx_pol() const { return _rx_pol; }
		double Maxlevel() const { return _Maxlevel; }
		double scale() const { return _scale; }
		double dR() const { return _dR; }
		double Tr() const { return _Tr; }
		double dAsp() const { return _dAsp; }
		double df() const { return _df; }
		bool isUpChirp() const { return _isUpChirp; }
		const TargetTYPE targetType() const { return _targetType; }
//		double singleRCS() const { return _singleRCS; }
//		string waveformType() const { return _waveformType; }
//		string codeType() const { return _codeType; }
//		string codeString() const { return _codeString; }
//		int peakPower() const { return _peakPower; }
//		int powerWidth() const { return _powerWidth; }
//		int triggerTrhreshold() const { return _triggerTrhreshold; }
//		int fpgaMode() const { return _fpgaMode; }
//		int signalShift() const { return _signalShift; }
//		bool   realTimeConv() const { return _realTimeConv; }
		// Setter
		double& look_min() { return _look_min; }
		double& look_max() { return _look_max; }
		double& asp_min() { return _asp_min; }
		double& asp_max() { return _asp_max; }
		double& f0() { return _f0; }
		double& Fr() { return _Fr; }
		double& BWrg() { return _BWrg; }
		double& PRF() { return _PRF; }
		double& Rn_slant() { return _Rn_slant; }
		double& Rc_slant() { return _Rc_slant; }
		double& Rf_slant() { return _Rf_slant; }
		double& theta_sq_mean() { return _theta_sq_mean; }
		double& theta_l_mean() { return _theta_l_mean; }
		double& Vs_mean() { return _Vs_mean; }
		class FPGA& Fpga() { return _fpga; }
		size_t& Nr() { return _Nr; }
		size_t& Na() { return _Na; }
		string& tx_pol() { return _tx_pol; }
		string& rx_pol() { return _rx_pol; }
		double& Maxlevel() { return _Maxlevel; }
		double& scale() { return _scale; }
		double& dR() { return _dR; }
		double& Tr() { return _Tr; }
		double& dAsp() { return _dAsp; }
		double& df() { return _df; }
		bool& isUpChirp() { return _isUpChirp; }
		TargetTYPE& targetType() { return _targetType; }
//		double& singleRCS() { return _singleRCS; }
//		string& waveformType() { return _waveformType; }
//		string& codeType() { return _codeType; }
//		string& codeString() { return _codeString; }
//		int& peakPower() { return _peakPower; }
//		int& powerWidth() { return _powerWidth; }
//		int& triggerTrhreshold() { return _triggerTrhreshold; }
//		int& fpgaMode() { return _fpgaMode; }
//		int& signalShift() { return _signalShift; }
//		bool&   realTimeConv() { return _realTimeConv; }
		// Misc.
		void Print(){
			cout<<"+--------------+"<<endl;
			cout<<"|  Meta Class  |"<<endl;
			cout<<"+--------------+"<<endl;
			cout<<"look_min     = "<<_look_min<<" [deg]"<<endl;
			cout<<"look_max     = "<<_look_max<<" [deg]"<<endl;
			cout<<"asp_min      = "<<_asp_min<<" [deg]"<<endl;
			cout<<"asp_max      = "<<_asp_max<<" [deg]"<<endl;
			cout<<"f0           = "<<_f0/1e+9<<" [GHz]"<<endl;
			cout<<"Fr           = "<<_Fr/1e+6<<" [MHz]"<<endl;
			cout<<"BWrg         = "<<_BWrg/1e+6<<" [MHz]"<<endl;
			cout<<"PRF          = "<<_PRF<<" [Hz]"<<endl;
			cout<<"Rn_slant     = "<<_Rn_slant<<" [m]"<<endl;
			cout<<"Rc_slant     = "<<_Rc_slant<<" [m]"<<endl;
			cout<<"Rf_slant     = "<<_Rf_slant<<" [m]"<<endl;
			cout<<"theta_sq_mean= "<<rad2deg(_theta_sq_mean)<<" [deg]"<<endl;
			cout<<"theta_l_mean = "<<rad2deg(_theta_l_mean)<<" [deg]"<<endl;
			cout<<"Vs_mean      = "<<_Vs_mean<<" [m/s]"<<endl;
			cout<<"Nr           = "<<_Nr<<" [samples]"<<endl;
			cout<<"Na           = "<<_Na<<" [samples]"<<endl;
			cout<<"tx_pol       = "<<_tx_pol<<endl;
			cout<<"rx_pol       = "<<_rx_pol<<endl;
			cout<<"Maxlevel     = "<<_Maxlevel<<endl;
			cout<<"scale        = "<<_scale<<endl;
			cout<<"dR           = "<<_dR<<" [m]"<<endl;
			cout<<"Tr           = "<<_Tr<<" [sec]"<<endl;
			cout<<"dAsp         = "<<rad2deg(_dAsp)<<" [deg]"<<endl;
			cout<<"df           = "<<_df/1e6<<" [MHz]"<<endl;
			if(_isUpChirp == true){
				cout<<"isUpChirp    = true"<<endl;
			}else{
				cout<<"isUpChirp    = false"<<endl;
			}
			cout<<"+"<<endl;
			cout<<"    Target Type"<<endl;
			cout<<"+"<<endl;
			_targetType.Print();
//			cout<<"targetType   = "<<_targetType<<endl;
//			cout<<"singleRCS    = "<<_singleRCS<<" [dB]"<<endl;
//			cout<<"waveformType = "<<_waveformType<<endl;
//			cout<<"codeType     = "<<_codeType<<endl;
//			cout<<"codeString   = "<<_codeString<<endl;
			cout<<"+"<<endl;
			cout<<"    FPGA"<<endl;
			cout<<"+"<<endl;
			_fpga.Print();
//			cout<<"peakPower    = "<<_peakPower<<" [dB]"<<endl;
//			cout<<"powerWidth   = "<<_powerWidth<<" [sample]"<<endl;
//			cout<<"triggerTrhreshold = "<<_triggerTrhreshold<<" [dB]"<<endl;
//			cout<<"fpgaMode     = "<<_fpgaMode<<endl;
//			cout<<"signalShift  = "<<_signalShift<<endl;
//			cout<<"realTimeConv = "<<_realTimeConv<<endl;
//			_fpga.Print();
		}
	private:
		double _look_min;
		double _look_max;
		double _asp_min;
		double _asp_max;
		double _f0;
		double _Fr;
		double _BWrg;
		double _PRF;
		double _Rn_slant;
		double _Rc_slant;
		double _Rf_slant;
		double _theta_sq_mean;
		double _theta_l_mean;
		double _Vs_mean;
		class FPGA _fpga;
		size_t _Nr;
		size_t _Na;
		string _tx_pol;
		string _rx_pol;
		double _Maxlevel;
		double _scale;
		double _dR;
		// Add
		double _Tr;
		double _dAsp;
		double _df;
		bool _isUpChirp;
		class TargetTYPE _targetType;
//		string _targetType;
//		double _singleRCS;
//		string _waveformType;
//		string _codeType;
//		string _codeString;

//		int    _peakPower;
//		int    _powerWidth;
//		int    _triggerTrhreshold;
//		int    _fpgaMode;
//		int    _signalShift;
//		bool   _realTimeConv;
	};


	//===========================================================================================
	//|                                                                                         |
	//|                                      Functions                                          |
	//|                                                                                         |
	//===========================================================================================
	/**
	 * Read the json file in fo Json::Value class
	 *
	 * @param filename - Input json file name
	 * @return Return a JSon::Value class
	 */
	Json::Value ReadJson(const string filename){
		ifstream fin(filename.c_str());
		Json::Value values;
		fin>>values;
		fin.close();
		
		if(values == Json::nullValue){
			cerr<<"ERROR::ReadJson:Cannot read this Json file : "<<filename<<endl;
			exit(EXIT_FAILURE);
		}
		
		return values;
	}
	/**
	 * Convert number with "0" fill
	 *
	 * @param num - Input number
	 * @param tot_len - The length of you want to dsiplay
	 * @return Return a string with zero filling
	 */
	template<typename T>
	string num2strfill(const T& num, const size_t tot_len){
		string num_str = num2str(num);
		size_t num_len = num_str.length();
		
		if(tot_len < num_len){ return "-1"; }
		
		string out = "";
		for(size_t i=0;i<tot_len-num_len;++i){ out += "0"; }
		
		out += num_str;
		
		return out;
	}
	/**
	 * Convert number with space " " fill
	 *
	 * @param num - Input number
	 * @param tot_len - The length of you want to dsiplay
	 * @return Return a string with zero filling
	 */
	template<typename T>
	string num2strfillSpace(const T& num, const size_t tot_len){
		string num_str = num2str(num);
		size_t num_len = num_str.length();

		if(tot_len < num_len){ return "-1"; }

		string out = "";
		for(size_t i=0;i<tot_len-num_len;++i){ out += " "; }

		out += num_str;

		return out;
	}
	/**
	 * Convert hte Time string into TIME class
	 *
	 * @param str - Input time string
	 * @return Return the TIME class
	 */
	TIME TimeString2TIME(const string str){
		// e.g. str = "2020/08/12 00:00:23:683" (SQL definition = DATATIME(3)) or
		//      str = "2020-08-12 00:00:23:683123" (definition = varchar(255))
		int yr  = str2num<int>(str.substr( 0, 4));
		int mo  = str2num<int>(str.substr( 5, 2));
		int day = str2num<int>(str.substr( 8, 2));
		int hr  = str2num<int>(str.substr(11, 2));
		int min = str2num<int>(str.substr(14, 2));
		int sec1= str2num<int>(str.substr(17, 2));
		int sec2= str2num<int>(str.substr(20, 3));
		double sec = 0;

		if(str.length() >= 23){
			int sec3 = str2num<int>(str.substr(23, str.length()-23));
			sec = double(sec1) + double(sec2)/1e3 + double(sec3)/1e6; // varchar(255) case
		}else{
			sec = double(sec1) + double(sec2)/1e3; // DATATIME(3) case
		}

		// string sec_str = str.substr(12, str.length()-12);
		// double sec1 = str2num<double>(sec_str.substr(0,2));
		// double sec2 = str2num<double>(sec_str.substr(2, sec_str.length()-2))/pow(10,(double(sec_str.length()-2)));
		// double sec  = sec1 + sec2;

		// // e.g. str = "2020030806542368326"
		// int yr = str2num<int>(str.substr( 0, 4));
		// int mo = str2num<int>(str.substr( 4, 2));
		// int day= str2num<int>(str.substr( 6, 2));
		// int hr = str2num<int>(str.substr( 8, 2));
		// int min= str2num<int>(str.substr(10, 2));

		// string sec_str = str.substr(12, str.length()-12);
		// double sec1 = str2num<double>(sec_str.substr(0,2));
		// double sec2 = str2num<double>(sec_str.substr(2, sec_str.length()-2))/pow(10,(double(sec_str.length()-2)));
		// double sec  = sec1 + sec2;
		
//		cout<<"+---------------------------+"<<endl;
//		cout<<"|  TimeString2TIME (START)  |"<<endl;
//		cout<<"+---------------------------+"<<endl;
//		cout<<"Input  : "<<str<<endl;
//		TIME ttime(yr, mo, day, hr, min, sec);
//		cout<<"Output : "; ttime.Print();
//		cout<<"+---------------------------+"<<endl;
//		cout<<"|  TimeString2TIME (END)    |"<<endl;
//		cout<<"+---------------------------+"<<endl;

		return TIME(yr, mo, day, hr, min, sec);
	}
	/**
	 * Convert the TIME class to Time string
	 *
	 * @param [in] time - TIME class
	 * @return Return a time string
	 */
	string TIME2TimeString(const TIME& time){

		int sec_int = int(time.GetSec());
		int sec_rem = int((time.GetSec() - double(sec_int))*1e3);

		string str = "";
		str += num2strfill(time.GetYear(),4);
		str += "/";
		str += num2strfill(time.GetMonth(),2);
		str += "/";
		str += num2strfill(time.GetDay(),2);
		str += " ";
		str += num2strfill(time.GetHour(),2);
		str += ":";
		str += num2strfill(time.GetMin(),2);
		str += ":";
		str += num2strfill(sec_int, 2);
		str += ".";
		str += num2strfill(sec_rem, 3);
		
		return str;
	}
	/**
	 * Read flight path json file without convert ot SV<double> class
	 *
	 * @param [in] file_FLP - The flight path json file
	 * @return Return the vector<Flp> class
	 */
	vector<Flp> ReadFLP(const string file_FLP){
		Json::Value FLP_json = ReadJson(file_FLP);
		
		Json::Value pts = FLP_json["flightPoints"];
		
		vector<Flp> flp(pts.size());
		
		// (1) Reformat json -> D1<FLP>
		for(int i=0;i<flp.size();++i){
			// (1) Get all elements
			string time = pts[i]["time"].asString();
			double lon  = deg2rad(pts[i]["longitude"].asDouble());
			double lat  = deg2rad(pts[i]["latitude"].asDouble());
			double h    = pts[i]["height"].asDouble();
			double look = deg2rad(pts[i]["look"].asDouble());
			double sq   = deg2rad(pts[i]["squint"].asDouble());
			// (2) Insert
			TIME Time = TimeString2TIME(time);
			GEO<double> llh(lon, lat, h);
			flp[i] = Flp(Time, llh, look, sq);
		}
		
		return flp;
	}
	/**
	 * Make the Fligh path json file by RS1 data
	 *
	 * @param [out] file_FLP - Return flight path json file
	 * @param [out] file_TAR - Return Target location json file
	 */
	void SimOrb2Json(const string file_FLP, const string file_TAR, const bool SHOW=false) {
		//+---------------------------+
		//| Read Input Parameters     |
		//+---------------------------+
//		TIME Time = TimeString2TIME("20200308065408");
		TIME Time = TimeString2TIME("2020/03/08 06:54:08:683");
		vector<double> Pos = {
				0.0000000000000000, 298909.2464330748771317, -5243954.0254650777205825, 3608560.5922599136829376, 13.8722490773568268, -29.2566528008248241, -43.3719332913320983,
				1.5366579836221312, 298930.5634486944763921, -5243998.9820097498595715, 3608493.9438450359739363, 13.8723228244660444, -29.2560512171808931, -43.3723154976937550,
				3.0733154882136331, 298951.8805743071134202, -5244043.9376229802146554, 3608427.2948532374575734, 13.8723965661121387, -29.2554496294069502, -43.3726976967510325,
				4.6099725137057135, 298973.1978099031839520, -5244088.8923047613352537, 3608360.6452845297753811, 13.8724703017381330, -29.2548480370334616, -43.3730798889988165,
				6.1466290603083360, 298994.5151554756448604, -5244133.8460550867021084, 3608293.9951389213092625, 13.8725440323815850, -29.2542464393397559, -43.3734620745913162,
				7.6832851278036625, 299015.8326110159396194, -5244178.7988739488646388, 3608227.3444164241664112, 13.8726177577042158, -29.2536448370833178, -43.3738442531258528,
				9.2199407163102300, 299037.1501765159773640, -5244223.7507613413035870, 3608160.6931170490570366, 13.8726914777184707, -29.2530432302574148, -43.3742264246029947,
				10.7565958257827372, 299058.4678519684821367, -5244268.7017172602936625, 3608094.0412408094853163, 13.8727651930414222, -29.2524416208632232, -43.3746085874757838,
				12.2932504563311866, 299079.7856373640825041, -5244313.6517416955903172, 3608027.3887877129018307, 13.8728389021775289, -29.2518400043801563, -43.3749907452713401,
				13.8299046076986514, 299101.1035326941637322, -5244358.6008346416056156, 3607960.7357577723450959, 13.8729126059140224, -29.2512383844601942, -43.3753728952749782,
				15.3665582804190048, 299122.4215379519737326, -5244403.5489960936829448, 3607894.0821509980596602, 13.8729863049259947, -29.2506367605770166, -43.3757550376257512
		};

		double theta_l_MB = deg2rad(72.636719);// [deg]
		double theta_sqc = deg2rad(0.);        // [deg]
		size_t Na = 1101;                        	// [samples]
		double PRF = 213.4;                         // [Hz]
		bool IsSVNormalize = true;
		ORB Orb;

		//+===========================================================================+
		//|                                                                           |
		//|                               Prediction                                  |
		//|                                                                           |
		//+===========================================================================+

		//+==========================================================+
		//|                     SV generation                        |
		//+==========================================================+
		unsigned long num = Pos.size() / 7;
		D1<GEO<double> > Gd(num);
		D1<string> TimeStr(num);
		// State vector
		D1<double> sv_t(num);
		D1<VEC<double> > sv_pos(num);
		D1<VEC<double> > sv_vel(num);
		for (size_t i = 0; i < num; ++i) {
			double gps = Pos[i * 7 + 0] + UTC2GPS(Time);
			VEC<double> pos(Pos[i * 7 + 1], Pos[i * 7 + 2], Pos[i * 7 + 3]);
			VEC<double> vel(Pos[i * 7 + 4], Pos[i * 7 + 5], Pos[i * 7 + 6]);
			// ECR to Geodetic
			Gd[i] = sar::ECR2Gd(pos);
			// Misc.
			TimeStr[i] = TIME2TimeString(GPS2UTC(gps));
			// Assign
			sv_t[i] = gps;
			sv_pos[i] = pos;
			sv_vel[i] = vel;
		}
		// Make original SV (num=11)
		SV<double> sv_org(sv_t, sv_pos, sv_vel, num);


		// Re-normalize the altitude of SV, replace the original SV position values
		if (IsSVNormalize) {
			// Normalize height
			D1<double> h(sv_org.GetNum());
			for (long i = 0; i < sv_org.GetNum(); ++i) {
				h[i] = sar::ECR2Gd(sv_org.pos()[i], Orb).h();
			}
			double hmean = mat::total(h) / h.GetNum();
			for (long i = 0; i < sv_org.GetNum(); ++i) {
				GEO<double> gd = sar::ECR2Gd(sv_org.pos()[i], Orb);
				gd.h() = hmean;
				sv_org.pos()[i] = sar::Gd2ECR(gd, Orb);
			}
		}


		//+==========================================================+
		//|        State Vector(SV) Interpolation & selection        |
		//+==========================================================+
		// New time interval
		double dt = 1 / PRF;

		// Create time series
		SV<double> sv(Na);

		// Calculate Central Time [GPS]
		double t_c = (sv_org.t()[sv_org.GetNum() - 1] + sv_org.t()[0]) / 2;
		linspace(t_c - dt * Na / 2, t_c + dt * Na / 2, sv.t());

		// Interpolation
		sar::sv_func::Interp(sv_org, sv);


		// Downsampling (num=3)
		SV<double> sv_down(3);
		linspace(sv.t()[0], sv.t()[sv.GetNum()-1], sv_down.t());
		sar::sv_func::Interp(sv, sv_down);
		D1<string> TimeStr_down(sv_down.GetNum());
		for(size_t i=0;i<sv_down.GetNum();++i){
			TimeStr_down[i] = TIME2TimeString(GPS2UTC(sv_down.t()[i]));
		}

		// Convert to geodetic
		D1<GEO<double> > Gd_down(sv_down.GetNum());
		for (size_t i = 0; i < sv_down.GetNum(); ++i) {
			Gd_down[i] = sar::ECR2Gd(sv_down.pos()[i]);
		}

		// sv_org : Original SV
		// sv : Selected & interpolated from sv_org (dt = 1/PRF)
		// sv_down : down-sampling from sv
		// sv_int : interpolation from sv_down (dt = 1/PRF)

		if(SHOW){
			printf("sv.t()[0]          = %12f\n", sv.t()[0]);
			printf("sv.t()[end]        = %12f\n", sv.t()[sv.GetNum()-1]);
			cout<<"sv.pos()[0]        = "; sv.pos()[0].Print();
			cout<<"sv.pos()[end]      = "; sv.pos()[sv.GetNum()-1].Print();
			printf("sv_down.t()[0]          = %12f\n", sv_down.t()[0]);
			printf("sv_down.t()[end]        = %12f\n", sv_down.t()[sv_down.GetNum()-1]);
			cout<<"sv_down.pos()[0]   = "; sv_down.pos()[0].Print();
			cout<<"sv_down.pos()[end] = "; sv_down.pos()[sv_down.GetNum()-1].Print();
		}

		//+==========================================================+
		//|                   Write to FLP.json                      |
		//+==========================================================+
		Json::Value FLP;

//		"flightPath":{
//			"id":12345,
//			"number": 3,
//			"points":[
//				{
//					"time": "20200312081454",
//					"longitude": 120.01234567,
//					"latitude": 23.01234567,
//					"height": 1000.01234567,
//					"angle": 0
//				},
//				{
//					"time": "20200312091819",
//					"longitude": 120.21234567,
//					"latitude": 23.21234567,
//					"height": 1000.21234567,
//					"angle": 0
//				}
//				{
//					"time": "20200312101945",
//					"longitude": 120.21234567,
//					"latitude": 23.21234567,
//					"height": 1000.21234567,
//					"angle": 0
//				}
//			]
//		}

		FLP["id"] = 123456789;
		FLP["number"] = (int) sv_down.GetNum();

		for (int i = 0; i < sv_down.GetNum(); ++i) {
			FLP["points"][i]["time"] = TimeStr_down[i];
			FLP["points"][i]["longitude"] = rad2deg(Gd_down[i].lon());
			FLP["points"][i]["latitude"] = rad2deg(Gd_down[i].lat());
			FLP["points"][i]["height"] = Gd_down[i].h();
			FLP["points"][i]["look"] = rad2deg(theta_l_MB);
			FLP["points"][i]["squint"] = 0;
		}
		cout << FLP << endl;

		ofstream fout(file_FLP.c_str());
		if(SHOW){
			fout << FLP << endl;
		}
		fout.close();



		//+==========================================================+
		//|                   Find each theta_az                     |
		//+==========================================================+
		// SV interpolation
		dt = 1 / PRF;
		D1<double> sv_int_t = linspace(sv_down.t()[0], sv_down.t()[sv_down.GetNum()-1], dt);
		SV<double> sv_int(sv_int_t.GetNum());
		sv_int.t() = sv_int_t;
		sar::sv_func::Interp(sv_down, sv_int);


//		// Find Scene center
//		size_t idx_c = (sv_int.GetNum() - 1) / 2;
//		VEC<double> PsC = sv_int.pos()[idx_c];
//		VEC<double> PsC1 = sv_int.pos()[idx_c + 1];
		// Find center position with smallest interval
		size_t idx_c = (sv_int.GetNum() - 1) / 2;
		SV<double> sv_crop = sv_int.GetRange(idx_c-2, idx_c+2);
		SV<double> sv_int_high;
		sar::sv_func::Interp(sv_crop, dt/50., sv_int_high);
		idx_c = (sv_int_high.GetNum() - 1) / 2;
		VEC<double> PsC   = sv_int_high.pos()[idx_c];
		VEC<double> PsC1  = sv_int_high.pos()[idx_c+1];

		// Find target position
		VEC<double> Pt;
		VEC<double> uv = sar::find::MainBeamUVByLookSquintTypeA(PsC, PsC1, theta_l_MB, theta_sqc, Orb, Pt);

		if(SHOW){
			// Find local coordinate
			sar::LocalXYZ locXYZ = sar::find::LocalCoordinate(PsC, PsC1, theta_l_MB, theta_sqc, Orb);
			// Calculate each theta_az
			D1<double> theta_az(sv_int.GetNum());
			for (size_t i = 0; i < sv_int.GetNum(); ++i) {
				VEC<double> Ps = sv_int.pos()[i];
				sph::SPH<double> sph = sar::find::LocalSPH(Ps, Pt, locXYZ);
				theta_az[i] = sph.Phi();
			}

			printf("+---------------------------+\n");
			printf("|            CHECK          |\n");
			printf("+---------------------------+\n");
			printf("theta_az[end]  = %f [deg]\n", rad2deg(theta_az[theta_az.GetNum() - 1]));
			printf("theta_az[0]    = %f [deg]\n", rad2deg(theta_az[0]));
			//theta_az[end]  = -1.850180 [deg]
			//theta_az[0]    = 1.846818 [deg]
			printf("Na             = %ld\n", Na);
			printf("sv_int.GetNum()= %ld\n", sv_int.GetNum());
			printf("+\n");
			printf("sv_int.t()[0]    = %12f\n", sv_int.t()[0]);
			printf("sv_int.t()[end]  = %12f\n", sv_int.t()[sv_int.GetNum()-1]);
			cout<<"sv_int.pos()[0]    = "; sv_int.pos()[0].Print();
			cout<<"sv_int.pos()[end]  = "; sv_int.pos()[sv_int.GetNum()-1].Print();
			printf("PsC = [%.8f,%.8f,%.8f]\n", PsC.x(), PsC.y(), PsC.z());
		}

		//+==========================================================+
		//|                   Write to TAR.json                      |
		//+==========================================================+
		GEO<double> Tar_Gd = sar::ECR2Gd(Pt);

		Json::Value TAR_json;
		TAR_json["longitude"] = rad2deg(Tar_Gd.lon());
		TAR_json["latitude"] = rad2deg(Tar_Gd.lat());
		TAR_json["height"] = 0;

		cout<<"file_FLT = "<<file_FLP<<endl;
		cout<<"file_TAR = "<<file_TAR<<endl;

		ofstream fout2(file_TAR.c_str());
		fout2 << TAR_json << endl;
		fout2.close();
	}

	/**
	 * Make the Fligh path json file by RS1 data
	 *
	 * @param [out] file_FLP - Return flight path json file
	 * @param [out] file_TAR - Return Target location json file
	 */
	void SimOrb2JsonOrg(const string file_FLP_TAR, const bool SHOW=false) {
		//+---------------------------+
		//| Read Input Parameters     |
		//+---------------------------+
//		TIME Time = TimeString2TIME("20200308065408");
		TIME Time = TimeString2TIME("2020/04/16 14:54:08:683");
		vector<double> Pos = {
				0.0000000000000000, 298909.2464330748771317, -5243954.0254650777205825, 3608560.5922599136829376, 13.8722490773568268, -29.2566528008248241, -43.3719332913320983,
				1.5366579836221312, 298930.5634486944763921, -5243998.9820097498595715, 3608493.9438450359739363, 13.8723228244660444, -29.2560512171808931, -43.3723154976937550,
				3.0733154882136331, 298951.8805743071134202, -5244043.9376229802146554, 3608427.2948532374575734, 13.8723965661121387, -29.2554496294069502, -43.3726976967510325,
				4.6099725137057135, 298973.1978099031839520, -5244088.8923047613352537, 3608360.6452845297753811, 13.8724703017381330, -29.2548480370334616, -43.3730798889988165,
				6.1466290603083360, 298994.5151554756448604, -5244133.8460550867021084, 3608293.9951389213092625, 13.8725440323815850, -29.2542464393397559, -43.3734620745913162,
				7.6832851278036625, 299015.8326110159396194, -5244178.7988739488646388, 3608227.3444164241664112, 13.8726177577042158, -29.2536448370833178, -43.3738442531258528,
				9.2199407163102300, 299037.1501765159773640, -5244223.7507613413035870, 3608160.6931170490570366, 13.8726914777184707, -29.2530432302574148, -43.3742264246029947,
				10.7565958257827372, 299058.4678519684821367, -5244268.7017172602936625, 3608094.0412408094853163, 13.8727651930414222, -29.2524416208632232, -43.3746085874757838,
				12.2932504563311866, 299079.7856373640825041, -5244313.6517416955903172, 3608027.3887877129018307, 13.8728389021775289, -29.2518400043801563, -43.3749907452713401,
				13.8299046076986514, 299101.1035326941637322, -5244358.6008346416056156, 3607960.7357577723450959, 13.8729126059140224, -29.2512383844601942, -43.3753728952749782,
				15.3665582804190048, 299122.4215379519737326, -5244403.5489960936829448, 3607894.0821509980596602, 13.8729863049259947, -29.2506367605770166, -43.3757550376257512
		};

		//+==========================================================+
		//|                     SV generation                        |
		//+==========================================================+
		unsigned long num = Pos.size()/7;
		D1<GEO<double> > Gd(num);
		D1<string> TimeStr(num);
		// State vector
		D1<double> sv_t(num);
		D1<VEC<double> > sv_pos(num);
		D1<VEC<double> > sv_vel(num);
		for (size_t i = 0; i < num; ++i) {
			double gps = Pos[i * 7 + 0] + UTC2GPS(Time);
			VEC<double> pos(Pos[i * 7 + 1], Pos[i * 7 + 2], Pos[i * 7 + 3]);
			VEC<double> vel(Pos[i * 7 + 4], Pos[i * 7 + 5], Pos[i * 7 + 6]);
			// ECR to Geodetic
			Gd[i] = sar::ECR2Gd(pos);
			// Misc.
			TimeStr[i] = TIME2TimeString(GPS2UTC(gps));
			// Assign
			sv_t[i] = gps;
			sv_pos[i] = pos;
			sv_vel[i] = vel;
		}
		// Make original SV (num=11)
		SV<double> sv_org(sv_t, sv_pos, sv_vel, num);

		// Convert to geodetic
		D1<GEO<double> > Gd_org(sv_org.GetNum());
		for (size_t i = 0; i < sv_org.GetNum(); ++i) {
			Gd_org[i] = sar::ECR2Gd(sv_org.pos()[i]);
		}


		//+==========================================================+
		//|                   Find each theta_az                     |
		//+==========================================================+
		double theta_l_MB = deg2rad(72.6367);
		double theta_sqc  = deg2rad(0);
		ORB Orb;

//		// Find Scene center
//		size_t idx_c = (sv_int.GetNum() - 1) / 2;
//		VEC<double> PsC = sv_int.pos()[idx_c];
//		VEC<double> PsC1 = sv_int.pos()[idx_c + 1];
		// Find center position with smallest interval
		size_t idx_c = (sv_org.GetNum() - 1) / 2;
		SV<double> sv_crop = sv_org.GetRange(idx_c-2, idx_c+2);
		SV<double> sv_int_high;
		double dt = sv_org.t()[2] -sv_org.t()[1];
		sar::sv_func::Interp(sv_org, dt/500., sv_int_high);
		idx_c = (sv_int_high.GetNum() - 1) / 2;
		VEC<double> PsC   = sv_int_high.pos()[idx_c];
		VEC<double> PsC1  = sv_int_high.pos()[idx_c+1];

		// Find target position
		VEC<double> Pt;
		VEC<double> uv = sar::find::MainBeamUVByLookSquintTypeA(PsC, PsC1, theta_l_MB, theta_sqc, Orb, Pt);


		//+==========================================================+
		//|                   Write to TAR.json                      |
		//+==========================================================+
		GEO<double> Tar_Gd = sar::ECR2Gd(Pt);

		cout<<"+==========================================================+"<<endl;
		cout<<"|                   Write to FLP.json                      |"<<endl;
		cout<<"+==========================================================+"<<endl;
		Json::Value FLP_TAR;

//		{
//		  "targetLocation" : {
//			"longitude" : -86.7825400816,
//			"latitude" : 34.6572955075,
//			"height" : 0.0000
//		  },
//		  "flightPoints" : [ {
//			"time" : "2020/12/07 00:00:21:103",
//			"longitude" : -86.7369501531,
//			"latitude" : 34.6672987077,
//			"height" : 1349.7501,
//			"look" : 72.6367,
//			"squint" : 0.0000
//		  }, {
//			"time" : "2020/12/07 00:00:23:683",
//			"longitude" : -86.7366072844,
//			"latitude" : 34.6660726574,
//			"height" : 1349.7500,
//			"look" : 72.6367,
//			"squint" : 0.0000
//		  }, {
//			"time" : "2020/12/07 00:00:26:262",
//			"longitude" : -86.7362644205,
//			"latitude" : 34.6648466063,
//			"height" : 1349.7500,
//			"look" : 72.6367,
//			"squint" : 0.0000
//		  } ]
//		}

		FLP_TAR["targetLocation"]["longitude"] = rad2deg(Tar_Gd.lon());
		FLP_TAR["targetLocation"]["latitude"] = rad2deg(Tar_Gd.lat());
		FLP_TAR["targetLocation"]["height"] = 0;

		for (int i = 0; i < sv_org.GetNum(); ++i) {
			FLP_TAR["flightPoints"][i]["index"] = i+1;
			FLP_TAR["flightPoints"][i]["time"] = TimeStr[i];
			FLP_TAR["flightPoints"][i]["longitude"] = rad2deg(Gd_org[i].lon());
			FLP_TAR["flightPoints"][i]["latitude"] = rad2deg(Gd_org[i].lat());
			FLP_TAR["flightPoints"][i]["height"] = Gd_org[i].h();
			FLP_TAR["flightPoints"][i]["look"] = rad2deg(theta_l_MB);
			FLP_TAR["flightPoints"][i]["squint"] = rad2deg(theta_sqc);
		}
		cout << FLP_TAR << endl;

		ofstream fout(file_FLP_TAR.c_str());
		if(SHOW){
			fout << FLP_TAR << endl;
		}
		fout.close();

		cout<<"+---------------------------------------------------------+"<<endl;
		cout<<"|  This file is generated by original MSTAR state vector  |"<<endl;
		cout<<"+---------------------------------------------------------+"<<endl;
	}
	/**
	 * Make the Fligh path json file by RS1 data, the maximum flight length is 15 sec.
	 *
	 * @param [in] file_FLP_TAR - Return flight path with target location json file
	 * @param [in] theta_l_MB - Main beam look angle
	 * @param [in] theta_sqc - Center squint angle
	 * @param [in] Na - Azimuth number of sample
	 * @param [in] PRF - Pulse repeat frequency
	 * @param [in] Nsv - Number of state vector for output file (default = 3)
	 * @param [in] SHOW - Display the message (default = false)
	 */
	void SimOrb2Json(const string file_FLP_TAR, const double theta_l_MB, const double theta_sqc, const double PRF, const size_t Na, const size_t Nsv = 3, const bool SHOW=false) {
		//+---------------------------+
		//| Read Input Parameters     |
		//+---------------------------+
		TIME Time = TimeString2TIME("2020/03/08 06:54:08:683");

//		// Height = 1349.7500652605668 [m]
//		vector<double> Pos = {
//				0.0000000000000000, 298909.2464330748771317, -5243954.0254650777205825, 3608560.5922599136829376, 13.8722490773568268, -29.2566528008248241, -43.3719332913320983,
//				1.5366579836221312, 298930.5634486944763921, -5243998.9820097498595715, 3608493.9438450359739363, 13.8723228244660444, -29.2560512171808931, -43.3723154976937550,
//				3.0733154882136331, 298951.8805743071134202, -5244043.9376229802146554, 3608427.2948532374575734, 13.8723965661121387, -29.2554496294069502, -43.3726976967510325,
//				4.6099725137057135, 298973.1978099031839520, -5244088.8923047613352537, 3608360.6452845297753811, 13.8724703017381330, -29.2548480370334616, -43.3730798889988165,
//				6.1466290603083360, 298994.5151554756448604, -5244133.8460550867021084, 3608293.9951389213092625, 13.8725440323815850, -29.2542464393397559, -43.3734620745913162,
//				7.6832851278036625, 299015.8326110159396194, -5244178.7988739488646388, 3608227.3444164241664112, 13.8726177577042158, -29.2536448370833178, -43.3738442531258528,
//				9.2199407163102300, 299037.1501765159773640, -5244223.7507613413035870, 3608160.6931170490570366, 13.8726914777184707, -29.2530432302574148, -43.3742264246029947,
//				10.7565958257827372, 299058.4678519684821367, -5244268.7017172602936625, 3608094.0412408094853163, 13.8727651930414222, -29.2524416208632232, -43.3746085874757838,
//				12.2932504563311866, 299079.7856373640825041, -5244313.6517416955903172, 3608027.3887877129018307, 13.8728389021775289, -29.2518400043801563, -43.3749907452713401,
//				13.8299046076986514, 299101.1035326941637322, -5244358.6008346416056156, 3607960.7357577723450959, 13.8729126059140224, -29.2512383844601942, -43.3753728952749782,
//				15.3665582804190048, 299122.4215379519737326, -5244403.5489960936829448, 3607894.0821509980596602, 13.8729863049259947, -29.2506367605770166, -43.3757550376257512
//		};

		// Height =3488.6055 [m]
		vector<double> Pos = {
				0.0000000000000000, 299009.3533916563028470, -5245710.2651756266131997, 3609777.2695676144212484, 13.8722490773568268, -29.2566528008248241, -43.3719332913320983,
				1.5366579836221312, 299030.6775505115510896, -5245755.2368468344211578, 3609710.5987299443222582, 13.8723228244660444, -29.2560512171808931, -43.3723154976937550,
				3.0733154882136331, 299052.0018193936557509, -5245800.2075862875208259, 3609643.9273151564411819, 13.8723965661121387, -29.2554496294069502, -43.3726976967510325,
				4.6099725137057135, 299073.3261982974945568, -5245845.1773939784616232, 3609577.2553232642821968, 13.8724703017381330, -29.2548480370334616, -43.3730798889988165,
				6.1466290603083360, 299094.6506872170139104, -5245890.1462699044495821, 3609510.5827542725019157, 13.8725440323815850, -29.2542464393397559, -43.3734620745913162,
				7.6832851278036625, 299115.9752861404558644, -5245935.1142140561714768, 3609443.9096081973984838, 13.8726177577042158, -29.2536448370833178, -43.3738442531258528,
				9.2199407163102300, 299137.2999950601952150, -5245980.0812264261767268, 3609377.2358850464224815, 13.8726914777184707, -29.2530432302574148, -43.3742264246029947,
				10.7565958257827372, 299158.6248139712261036, -5246025.0473070107400417, 3609310.5615848340094090, 13.8727651930414222, -29.2524416208632232, -43.3746085874757838,
				12.2932504563311866, 299179.9497428619652055, -5246070.0124558005481958, 3609243.8867075690068305, 13.8728389021775289, -29.2518400043801563, -43.3749907452713401,
				13.8299046076986514, 299201.2747817233321257, -5246114.9766727890819311, 3609177.2112532621249557, 13.8729126059140224, -29.2512383844601942, -43.3753728952749782,
				15.3665582804190048, 299222.5999305518344045, -5246159.9399579716846347, 3609110.5352219245396554, 13.8729863049259947, -29.2506367605770166, -43.3757550376257512
		};

//		double theta_l_MB = deg2rad(72.636719);// [deg]
//		double theta_sqc = deg2rad(0.);        // [deg]
//		size_t Na = 1101;                        	// [samples]
//		double PRF = 213.4;                         // [Hz]
		bool IsSVNormalize = true;
		ORB Orb;

		//+===========================================================================+
		//|                                                                           |
		//|                               Prediction                                  |
		//|                                                                           |
		//+===========================================================================+

		//+==========================================================+
		//|                     SV generation                        |
		//+==========================================================+
		unsigned long num = Pos.size() / 7;
		D1<GEO<double> > Gd(num);
		D1<string> TimeStr(num);
		// State vector
		D1<double> sv_t(num);
		D1<VEC<double> > sv_pos(num);
		D1<VEC<double> > sv_vel(num);
		for (size_t i = 0; i < num; ++i) {
			double gps = Pos[i * 7 + 0] + UTC2GPS(Time);
			VEC<double> pos(Pos[i * 7 + 1], Pos[i * 7 + 2], Pos[i * 7 + 3]);
			VEC<double> vel(Pos[i * 7 + 4], Pos[i * 7 + 5], Pos[i * 7 + 6]);
			// ECR to Geodetic
			Gd[i] = sar::ECR2Gd(pos);
			// Misc.
			TimeStr[i] = TIME2TimeString(GPS2UTC(gps));
			// Assign
			sv_t[i] = gps;
			sv_pos[i] = pos;
			sv_vel[i] = vel;
		}
		// Make original SV (num=11)
		SV<double> sv_org(sv_t, sv_pos, sv_vel, num);


		// Re-normalize the altitude of SV, replace the original SV position values
		if (IsSVNormalize) {
			// Normalize height
			D1<double> h(sv_org.GetNum());
			for (long i = 0; i < sv_org.GetNum(); ++i) {
				h[i] = sar::ECR2Gd(sv_org.pos()[i], Orb).h();
			}
			double hmean = mat::total(h) / h.GetNum();
			for (long i = 0; i < sv_org.GetNum(); ++i) {
				GEO<double> gd = sar::ECR2Gd(sv_org.pos()[i], Orb);
				gd.h() = hmean;
				sv_org.pos()[i] = sar::Gd2ECR(gd, Orb);
			}
		}


		//+==========================================================+
		//|        State Vector(SV) Interpolation & selection        |
		//+==========================================================+
		// New time interval
		double dt = 1 / PRF;

		// Create time series
		SV<double> sv(Na);

		// Calculate Central Time [GPS]
		double t_c = (sv_org.t()[sv_org.GetNum() - 1] + sv_org.t()[0]) / 2;
		linspace(t_c - dt * Na / 2, t_c + dt * Na / 2, sv.t());

		// Interpolation
		sar::sv_func::Interp(sv_org, sv);


		// Downsampling (default num=3)
		SV<double> sv_down(Nsv);
		linspace(sv.t()[0], sv.t()[sv.GetNum()-1], sv_down.t());
		sar::sv_func::Interp(sv, sv_down);
		D1<string> TimeStr_down(sv_down.GetNum());
		for(size_t i=0;i<sv_down.GetNum();++i){
			TimeStr_down[i] = TIME2TimeString(GPS2UTC(sv_down.t()[i]));
			cout<<"TimeStr_down = "<<TimeStr_down[i]<<endl;
		}

		// Convert to geodetic
		D1<GEO<double> > Gd_down(sv_down.GetNum());
		for (size_t i = 0; i < sv_down.GetNum(); ++i) {
			Gd_down[i] = sar::ECR2Gd(sv_down.pos()[i]);
		}

		// sv_org : Original SV
		// sv : Selected & interpolated from sv_org (dt = 1/PRF)
		// sv_down : down-sampling from sv
		// sv_int : interpolation from sv_down (dt = 1/PRF)

		if(SHOW){
			printf("sv.t()[0]          = %12f\n", sv.t()[0]);
			printf("sv.t()[end]        = %12f\n", sv.t()[sv.GetNum()-1]);
			cout<<"sv.pos()[0]        = "; sv.pos()[0].Print();
			cout<<"sv.pos()[end]      = "; sv.pos()[sv.GetNum()-1].Print();
			printf("sv_down.t()[0]          = %12f\n", sv_down.t()[0]);
			printf("sv_down.t()[end]        = %12f\n", sv_down.t()[sv_down.GetNum()-1]);
			cout<<"sv_down.pos()[0]   = "; sv_down.pos()[0].Print();
			cout<<"sv_down.pos()[end] = "; sv_down.pos()[sv_down.GetNum()-1].Print();
		}

		//+==========================================================+
		//|                   Find each theta_az                     |
		//+==========================================================+
		// SV interpolation
		dt = 1 / PRF;
		D1<double> sv_int_t = linspace(sv_down.t()[0], sv_down.t()[sv_down.GetNum()-1], dt);
		SV<double> sv_int(sv_int_t.GetNum());
		sv_int.t() = sv_int_t;
		sar::sv_func::Interp(sv_down, sv_int);


//		// Find Scene center
//		size_t idx_c = (sv_int.GetNum() - 1) / 2;
//		VEC<double> PsC = sv_int.pos()[idx_c];
//		VEC<double> PsC1 = sv_int.pos()[idx_c + 1];
		// Find center position with smallest interval
		size_t idx_c = (sv_int.GetNum() - 1) / 2;
		SV<double> sv_crop = sv_int.GetRange(idx_c-2, idx_c+2);
		SV<double> sv_int_high;
		sar::sv_func::Interp(sv_crop, dt/50., sv_int_high);
		idx_c = (sv_int_high.GetNum() - 1) / 2;
		VEC<double> PsC   = sv_int_high.pos()[idx_c];
		VEC<double> PsC1  = sv_int_high.pos()[idx_c+1];

		// Find target position
		VEC<double> Pt;
		VEC<double> uv = sar::find::MainBeamUVByLookSquintTypeA(PsC, PsC1, theta_l_MB, theta_sqc, Orb, Pt);

		if(SHOW){
			// Find local coordinate
			sar::LocalXYZ locXYZ = sar::find::LocalCoordinate(PsC, PsC1, theta_l_MB, theta_sqc, Orb);
			// Calculate each theta_az
			D1<double> theta_az(sv_int.GetNum());
			for (size_t i = 0; i < sv_int.GetNum(); ++i) {
				VEC<double> Ps = sv_int.pos()[i];
				sph::SPH<double> sph = sar::find::LocalSPH(Ps, Pt, locXYZ);
				theta_az[i] = sph.Phi();
			}

			printf("+---------------------------+\n");
			printf("|            CHECK          |\n");
			printf("+---------------------------+\n");
			printf("theta_az[end]  = %f [deg]\n", rad2deg(theta_az[theta_az.GetNum() - 1]));
			printf("theta_az[0]    = %f [deg]\n", rad2deg(theta_az[0]));
			//theta_az[end]  = -1.850180 [deg]
			//theta_az[0]    = 1.846818 [deg]
			printf("Na             = %ld\n", Na);
			printf("sv_int.GetNum()= %ld\n", sv_int.GetNum());
			printf("+\n");
			printf("sv_int.t()[0]    = %12f\n", sv_int.t()[0]);
			printf("sv_int.t()[end]  = %12f\n", sv_int.t()[sv_int.GetNum()-1]);
			cout<<"sv_int.pos()[0]    = "; sv_int.pos()[0].Print();
			cout<<"sv_int.pos()[end]  = "; sv_int.pos()[sv_int.GetNum()-1].Print();
			printf("PsC = [%.8f,%.8f,%.8f]\n", PsC.x(), PsC.y(), PsC.z());
		}

		//+==========================================================+
		//|                   Write to TAR.json                      |
		//+==========================================================+
		GEO<double> Tar_Gd = sar::ECR2Gd(Pt);

		cout<<"+==========================================================+"<<endl;
		cout<<"|                   Write to FLP.json                      |"<<endl;
		cout<<"+==========================================================+"<<endl;
		Json::Value FLP_TAR;

//		{
//		  "targetLocation" : {
//			"longitude" : -86.7825400816,
//			"latitude" : 34.6572955075,
//			"height" : 0.0000
//		  },
//		  "flightPoints" : [ {
//			"time" : "2020/12/07 00:00:21:103",
//			"longitude" : -86.7369501531,
//			"latitude" : 34.6672987077,
//			"height" : 1349.7501,
//			"look" : 72.6367,
//			"squint" : 0.0000
//		  }, {
//			"time" : "2020/12/07 00:00:23:683",
//			"longitude" : -86.7366072844,
//			"latitude" : 34.6660726574,
//			"height" : 1349.7500,
//			"look" : 72.6367,
//			"squint" : 0.0000
//		  }, {
//			"time" : "2020/12/07 00:00:26:262",
//			"longitude" : -86.7362644205,
//			"latitude" : 34.6648466063,
//			"height" : 1349.7500,
//			"look" : 72.6367,
//			"squint" : 0.0000
//		  } ]
//		}

		FLP_TAR["targetLocation"]["longitude"] = rad2deg(Tar_Gd.lon());
		FLP_TAR["targetLocation"]["latitude"] = rad2deg(Tar_Gd.lat());
		FLP_TAR["targetLocation"]["height"] = 0;

		for (int i = 0; i < sv_down.GetNum(); ++i) {
			FLP_TAR["flightPoints"][i]["index"] = i+1;
			FLP_TAR["flightPoints"][i]["time"] = TimeStr_down[i];
			FLP_TAR["flightPoints"][i]["longitude"] = rad2deg(Gd_down[i].lon());
			FLP_TAR["flightPoints"][i]["latitude"] = rad2deg(Gd_down[i].lat());
			FLP_TAR["flightPoints"][i]["height"] = Gd_down[i].h();
			FLP_TAR["flightPoints"][i]["look"] = rad2deg(theta_l_MB);
			FLP_TAR["flightPoints"][i]["squint"] = rad2deg(theta_sqc);
		}
		cout << FLP_TAR << endl;

		ofstream fout(file_FLP_TAR.c_str());
		if(SHOW){
			fout << FLP_TAR << endl;
		}
		fout.close();

		cout<<"+==========================================================+"<<endl;
		cout<<"|         Check instantaneous slant range distance         |"<<endl;
		cout<<"+==========================================================+"<<endl;
		D1<double> dist(sv.GetNum());
		for (size_t i = 0; i < sv.GetNum(); ++i) {
			dist[i] = (sv.pos()[i] - Pt).abs();
		}
		cout<<"Min(dist)  = "<<mat::min(dist)<<endl;
		cout<<"Max(dist)  = "<<mat::max(dist)<<endl;
		cout<<"Mean(dist) = "<<mat::mean(dist)<<endl;
	}
	/**
	 * Read the Fligh path json file (FLP.json)
	 *
	 * @param [in] file_FLP - Fligh path json file
	 * @return Return the SV<double> state vector class
	 */
	SV<double> ReadFLP2SV(const string file_FLP, D1<double>& theta_l, D1<double>& theta_sq){
		// (1) Reformat json -> D1<FLP>
		vector<Flp> flp = ReadFLP(file_FLP);
		
		// (2) Convert Lon/Lat/H to ECR & store into the SV class
		//
		// SV constructor:
		// 		SV(const D1<T>& t,const D1<VEC<T> >& pos,const D1<VEC<T> >& vel,const long num);
		//
		size_t num = flp.size();
		// Convert to MJD time in double seconds
		D1<double> t(flp.size());
		// Convert to Position D1<VEC<double> >
		D1<VEC<double> > pos(flp.size());
		// Convert to Position D1<VEC<double> >
		D1<VEC<double> > vel(flp.size());
		// Define datum
		ORB Orb;
		
		for(long i=num-1;i>=0;--i){
			t[i] = UTC2GPS(flp[i].Time());
//			printf("t[%ld] = %f25.15\n", i, t[i]);
			pos[i] = sar::Gd2ECR(flp[i].LLH(), Orb);
			if(i != num-1){
				vel[i] = (pos[i+1] - pos[i])/(t[i+1] - t[i]);
			}
		}
		// assgin last elements equals to previous one
		vel[num-1] = vel[num-2];

		// for theta_l & theta_sq
		theta_l  = D1<double>(num);
		theta_sq = D1<double>(num);
		for(size_t i=0;i<num;++i){
			theta_l[i]  = flp[i].Look();
			theta_sq[i] = flp[i].Squint();
		}
		
		// Assign to SV class & return
		return SV<double>(t, pos, vel, num);
	}
	/**
	 * Read RCS parameter json file (RCS.json)
	 *
	 * @param [in] file_RCS - RCS parameter json file
	 * @param [out] Ef - Return EF class
	 * @param [out] mesh - Return MeshDef class
	 * @param [out] Sar - Return Sar class
	 * @param [in] SHOW - Display the message on console or not?
	 */
	void ReadRCSpar(const string file_RCS, EF& Ef, MeshDef& mesh, SAR& Sar, FPGA& Fpga, TargetTYPE& TargetType, bool& IsPEC, bool& IsOverwrite, const bool SHOW=false){
		// Read RCS parameter
		Json::Value RCS_json = ReadJson(file_RCS);

		//
		// EF class
		//
		// Electric field parameters (PO approximation coefficient)
		//	Taylor Limit Value = 0.5 (fix)
		//	Taylor Series Number = 2 (fix)
		//	Max Bouncing Number = 4 (user)
		string TxPol = RCS_json["tx_pol"].asString();
		string RxPol = RCS_json["rx_pol"].asString();
		long MaxLevel= RCS_json["max_bounce"].asUInt64();
		Ef = EF(TxPol, RxPol, TAYLOR(1.0, 0), MaxLevel);

		if(RxPol == "H&V"){
			RxPol = "VH";
		}

		//
		// MeshDef class
		//
		// Incident Mesh parameters
		//  Scale = 3.0 (user)
		//  dRad = 3000 (fix)
		double scale = RCS_json["scale"].asDouble();
		mesh = MeshDef(scale, 3000);
		
		// SAR parameters
		//  theta_l_MB = 30 (under defined) v
		//  theta_sqc = 0 (under defined) v
		//  f0 = 9600000000 (user) v
		//  PRF = 213.4 (user) v
		//  Fr = 591000000 (user) v
		//  DC = 6.8 (user) v
		//  BWrg = 591000000 (user) v
		//  SWrg = ? (calculated)
		//  Laz = ? (calculated)
		//  Lev = ? (calculated)
		//  ant_eff = 1 (fix) v
		//  Nr = 180 (user) v
		//  Na = 1 (under defined)


		// {
		//   "id": null,
		//   "transmitted_frequency": 9.6,
		//   "pulse_Repeat_Frequency": 213.4,
		//   "adc_sampling_rate": 591.0, (remove)
		//   "duty_cycle": 6.8,
		//   "chirp_bandwidth": 591.0,
		//   "ant_azimuth_3db": 2.0,
		//   "ant_elevation_3db": 10.0,
		//   "number_of_range": 180,
		//   "tx_pol": "H",
		//   "rx_pol": "H&V",
		//   "max_bounce": 4,
		//   "scale": 0.1,
		//   "pec": true,
		//   "tx_power": 12.0,
		//   "fpgagain": null,
		//   "rdelta": null,
		//   "pathLoss": null,
		//	 // add
		//   "target_type":"complex",
		//   "waveform_type":"type3",
		//   "single_rcs":1919.0,
		//   "force_overwrite":false
		// }


		double f0  = RCS_json["transmitted_frequency"].asDouble() * 1e+9;	// [GHz]
		double PRF = RCS_json["pulse_Repeat_Frequency"].asDouble();
//		double Fr  = RCS_json["adc_sampling_rate"].asDouble() * 1e+6;
//		double Fr  = RCS_json["chirp_bandwidth"].asDouble() * 1e+6;		// set "adc_sampling_rate" equal to "chirp_bandwidth"
//		double DC  = RCS_json["duty_cycle"].asDouble();
		double Tr  = RCS_json["duration_time"].asDouble() * 1e-6;		// [us]
		double BWrg= RCS_json["chirp_bandwidth"].asDouble() * 1e+6;		// [MHz]
		double Fr  = BWrg;
		double EL3 = deg2rad(RCS_json["ant_elevation_3db"].asDouble());	// [deg]
		double AZ3 = deg2rad(RCS_json["ant_azimuth_3db"].asDouble());	// [deg]
		long   Nr  = RCS_json["number_of_range"].asUInt64();
		IsPEC       = RCS_json["pec"].asBool();
		IsOverwrite = RCS_json["force_overwrite"].asBool();
		// Find Lev & Laz
		double ant_eff = 1.0;
		double lambda = def::C / f0;
		double Lev = 0.886*lambda/EL3/ant_eff;
		double Laz = 0.886*lambda/AZ3/ant_eff;
		// Find SWrg
		double theta_l_near = deg2rad(45.0) - EL3/2; // main beam look angle = 45 [deg]
		double theta_l_far  = deg2rad(45.0) + EL3/2;
		double h = 6000;	// 6 [km]
		ORB orb;
		double Re = (orb.E_a()+orb.E_b())/2;
		double theta_i_near = asin((Re+h)/Re*sin(theta_l_near));
		double beta_e_near  = theta_i_near - theta_l_near;
		double R_near       = (Re + h)*sin(beta_e_near)/sin(theta_i_near);
		double theta_i_far  = asin((Re+h)/Re*sin(theta_l_far));
		double beta_e_far   = theta_i_far - theta_l_far;
		double R_far        = (Re + h)*sin(beta_e_far)/sin(theta_i_far);
		double SWrg         = R_far - R_near;
		double DC			= 100 * PRF * Tr;		// Duty cycle

		string updownChirp  = RCS_json["updown_chirp"].asString();

		cout<<endl<<endl;
		cout<<"+-----------------------+"<<endl;
		cout<<"|           Hi          |"<<endl;
		cout<<"+-----------------------+"<<endl;
		cout<<"updownChirp = "<<updownChirp<<endl;
		cout<<endl<<endl;

		bool isUpChirp = false;
		if(updownChirp == "up"){
			isUpChirp = true;
		}

		
		const long Na = 1;
		//
		// SAR class
		//
		Sar = SAR("WF_RCS_SIM", deg2rad(45.0), deg2rad(0.0), f0, PRF, Fr, DC, BWrg, SWrg, Laz, Lev, 1.0, Nr, Na, isUpChirp);

		//
		// Target Type
		//
		string Target_type  = RCS_json["target_type"].asString();	// (string) Attenuation type ("complex" or "single")
		double SingleRCS    = RCS_json["single_rcs"].asDouble();	// [dB] Single target RCS value for "single"
		string WaveformType = RCS_json["waveform_type"].asString();	// (string) Waveform type ("coding", "LFM")
		string CodeType     = RCS_json["code_type"].asString();	    // (string) Coding type ("type0", "type1", "type2", "type3", "type4")
		string CodeString   = RCS_json["code_string"].asString();	// (string) Coding string
		TargetType = TargetTYPE(Target_type, SingleRCS, WaveformType, CodeType, CodeString);

		//
		// FPGA class
		//
		// (1) FPGA options (For trigger FPGA conditions)
		long inputGain         = RCS_json["input_gain"].asInt64();		// [2^n] Input gain
		double outputGain      = RCS_json["output_gain"].asInt64();		// [2^n] Output gain
		double powerLevel      = RCS_json["power_level"].asInt64();		// [2^n] Power level
		long fpgaMode          = RCS_json["fpga_mode"].asInt64();		// [x] (1~4) FPGA mode
		long signalOffset      = RCS_json["signal_offset"].asInt64();	// [x] (0~255) Signal offset
		bool IsRealTimeConv    = RCS_json["real_time_conv"].asBool();
		// (2) Attenuation
		string AttenuationType = RCS_json["attenuation_type"].asString();	// (string) Attenuation type ("free_space" or "cable")
		double Pt_dBm          = RCS_json["tx_power"].asDouble();		// [dBm] Transmit power
		double Gt_dBi          = RCS_json["tx_gain"].asDouble();		// [dBi] Transmit gain
		double imageGain       = RCS_json["image_gain"].asDouble();		// [2^n] Image gain
		double Distance        = RCS_json["distance"].asDouble();		// [m] Physical distance between antenna of Emulator and seeker's antenna
		double PathLoss        = RCS_json["path_loss"].asDouble();		// [dB] Path loss
		// (3) Others
		long DistanceCnt 	   = std::floor( (Distance + FPGA_CABLE_DISTANCE) * 2.0/def::C*(245.76e+6) );	// TODO: [count]
		long increasing_time = (unsigned int)(std::ceil( Sar.Tr() * FPGA_TX_BANDWIDTH ));


		Fpga = FPGA(inputGain, outputGain, powerLevel, fpgaMode, signalOffset, IsRealTimeConv,
					AttenuationType, Pt_dBm, Gt_dBi, imageGain, Distance, PathLoss, DistanceCnt, increasing_time);


		
		if(SHOW){
			cout<<RCS_json<<endl;
			Ef.Print();
			mesh.Print();
			Sar.Print();
			TargetType.Print();
			Fpga.Print();
		}
	}
	/**
	 * Read the Target location file (TAR.json)
	 *
	 * @param [in] file_TAR - Target location file
	 * @return Return a VEC<double> of target location
	 */
	VEC<double> ReadTargetLocation(const string file_TAR){
		Json::Value TAR_json = ReadJson(file_TAR);
		double lon = deg2rad(TAR_json["longitude"].asDouble());
		double lat = deg2rad(TAR_json["latitude"].asDouble());
		double h   = TAR_json["height"].asDouble();
		GEO<double> TAR_gd(lon, lat, h);
		ORB Orb;
		VEC<double> TAR = sar::Gd2ECR(TAR_gd, Orb);
		
		return TAR;
	}
	/**
	 * Rearrange the folder name by Look and Aspect angle
	 *
	 * @param [in] Look - Near Look angle in degree
	 * @param [in] Asp - Near Aspect angle in degree
	 * @return Return RCSFolder class
	 */
	RCSFolder MakeRCSDatabaseFolderName(const double Look, const double Asp){
		// Input: Look [deg] & Asp [deg]
		RCSFolder folder;
		// (1) Make Look folder name (e.g. delta[0] = 2 [deg])
		//     e.g. "Look02", "Look04" ... "Look88" (44 folders)
		folder.Look() = "Look" + num2strfill(Look, 6);
		// (2) Make Aspect angle catalog folder name
		//     e.g. "001", "002" ... "359" (359 folders)
		//          "002" means 2 [deg]
		string sgn = (Asp < 0)? "A-":"A+";
		folder.Asp_catlg() = sgn + num2strfill(int(abs(Asp)), 3);
		// (3) Make freq filename (e.g. delta[1] = 0.00001 [deg])
		//     e.g. "00100000.dat", "00100001.dat" ... "00199999.dat" (within "001" folder, 100000 files)
		//          "35900000.dat", "35900001.dat" ... "35999999.dat" (within "359" folder, 100000 files)
		size_t asp_val = int(abs(Asp) * 100000);
		folder.Asp() = sgn + num2strfill(asp_val, 8);
		// (4) Combine all
		folder.Full() = folder.Look() + "_" + folder.Asp_catlg() + "_" + folder.Asp();
		
		return folder;
	}
	/**
	 * Read the Delta json file (Delta.json)
	 *
	 * @param [in] file_DLT - Delta json file
	 * @return Return a LAF class
	 */
	LAF ReadDelta(const string& file_DLT){
		// Read Interval.json
		Json::Value DLT = ReadJson(file_DLT);
		double d_look = DLT["look"].asDouble();
		double d_asp  = DLT["aspect"].asDouble();
		double d_freq = DLT["frequency"].asDouble();
		return LAF(d_look, d_asp, d_freq);
	}
	/**
	 * Find the near look angle in degree, and return near value in degree
	 *
	 * @param [in] look_deg - Look angle in degree
	 * @param [in] d_look_deg - Look angle small interval in degree
	 * @return Return the near value of look in degree
	 *         e.g. look_deg = 72.64 [deg], d_look_deg = 0.05 [deg]
	 *         Return "726500" double values
	 */
	double FindNearLookDeg(const double& look_deg, const double& d_look_deg){
		return round(look_deg/d_look_deg)*round(d_look_deg*10000);
	}
	/**
	 * Find the near aspect angle in degree, and return near value in degree
	 *
	 * @param [in] asp_deg - Aspect angle in degree
	 * @param [in] d_asp_deg - Aspect angle small interval in degree
	 * @return Return the near value of aspect in degree
	 * 		   e.g. asp_deg = -1.720812345 [deg], d_asp_deg = 0.000001 [deg]
	 * 		   Return "-1.720812" double value
	 */
	double FindNearAspDeg(const double& asp_deg, const double& d_asp_deg){
		return round(asp_deg/d_asp_deg)*d_asp_deg;
	}
	/**
	 * Find the near frequency, and return near value
	 *
	 * @param [in] freq - Frequency value
	 * @param [in] d_freq - Frequency small interval
	 * @return Return the near value of frequency
	 */
	double FindNearFreq(const double& freq, const double& d_freq){
		return round(freq/d_freq)*d_freq;
	}
	/**
	 * Find the near Look angle/Aspect angle/Frequency class with delta value
	 *
	 * @param [in] real - Real value in LAF class
	 * @param [in] delta - Delta value in LAF class
	 * @return Return the Near LAF class
	 */
	LAF FindNearLAF(const LAF& real, const LAF& delta){
		return LAF(FindNearLookDeg(rad2deg(real.Look()), delta.Look()),
				   FindNearAspDeg(rad2deg(real.Asp()), delta.Asp()),
				   FindNearFreq(real.Freq(), delta.Freq()));
	}
	/**
	 * Convert the CPLX<double> to CPLX<float>
	 *
	 * @param [in] in - Input value
	 * @return Return a CPLX<float> value
	 */
	CPLX<float> CPLXdouble2CPLXfloat(const CPLX<double>& in){
		return CPLX<float>(float(in.r()), float(in.i()));
	}
	/**
	 * Write the RCS meta data file
	 *
	 * @param [in] in_meta - Meta class
	 * @param [in] file_out_RCS_meta - (String) RCS meta file name with path
	 */
	void WriteMeta(const Meta& in_meta, const string file_out_RCS_meta){

		D1<string> meta(58);
		meta[0]  = "#+=============================+";
		meta[1]  = "#|     RCS results (Meta)      |";
		meta[2]  = "#+=============================+";
		meta[3]  = "# Angle";
		meta[4]  = "Look angle (Min) [deg] = " + num2str(rad2deg(in_meta.look_min()));
		meta[5]  = "Look angle (Max) [deg] = " + num2str(rad2deg(in_meta.look_max()));
		meta[6]  = "Target Aspect angle (Min) [deg] = " + num2str(rad2deg(in_meta.asp_min()));
		meta[7]  = "Target Aspect angle (Max) [deg] = " + num2str(rad2deg(in_meta.asp_max()));
		meta[8]  = "# SAR";
		meta[9]  = "Transmitted frequency [Hz] = " + num2str(in_meta.f0());
		meta[10] = "ADC sampling rate [kHz] = " + num2str(in_meta.Fr()/1e3);
		meta[11] = "Bandwidth [kHz] = " + num2str(in_meta.BWrg()/1e3);
		meta[12] = "PRF [Hz] = " + num2str(in_meta.PRF());
		meta[13] = "Slant range at near [m] = " + num2str(in_meta.Rn_slant(), 6);
		meta[14] = "Slant range at center [m] = " + num2str(in_meta.Rc_slant(), 6);
		meta[15] = "Slant range at far [m] = " + num2str(in_meta.Rf_slant(), 6);
		meta[16] = "Pulse width [sec] = " + num2str(in_meta.Tr(), 16);
		meta[17] = "Aspect angle interval [deg] = " + num2str(rad2deg(in_meta.dAsp()), 8);
		meta[18] = "Frequency interval [Hz] = " + num2str(in_meta.df(), 8);
		meta[19] = "# State vector";
		meta[20] = "Mean squint [deg] = " + num2str(rad2deg(in_meta.theta_sq_mean()), 8);
		meta[21] = "Mean look [deg] = " + num2str(rad2deg(in_meta.theta_l_mean()), 8);
		meta[22] = "Mean velocity [m/s] = " + num2str(in_meta.Vs_mean(), 8);
		meta[23] = "# Data Dimension";
		meta[24] = "Number of Range [sample] = " + num2str(in_meta.Nr());
		meta[25] = "Number of Azimuth [sample] = " + num2str(in_meta.Na());
		meta[26] = "# Electric Field";
		meta[27] = "TX Polarization = " + in_meta.tx_pol();
		meta[28] = "RX Polarization = " + in_meta.rx_pol();
		meta[29] = "# PO approximation coeff.";
		meta[30] = "Max Bouncing Number = " + num2str(in_meta.Maxlevel());
		meta[31] = "# Incident Mesh";
		meta[32] = "Mesh Scale Factor = " + num2str(in_meta.scale());
		meta[33] = "Mesh Distance from targets center = " + num2str(in_meta.dR());
		meta[34] = "# Target Type";
		meta[35] = "Target Type = " + in_meta.targetType().targetType();
		meta[36] = "Single RCS = " + num2str(in_meta.targetType().SingleRCS());
		meta[37] = "# Waveform Type";
		meta[38] = "Waveform Type = " + in_meta.targetType().WaveformType();
		meta[39] = "Code Type = " + in_meta.targetType().CodeType();
		meta[40] = "Code String = " + in_meta.targetType().CodeString();
		if(in_meta.isUpChirp() == true){
			meta[41] = "Updown Chirp = up";
		}else{
			meta[41] = "Updown Chirp = down";
		}
		// FPGA
		// (1) FGPA options
		meta[42] = "# FPGA";
		meta[43] = "Input gain = " + num2str(in_meta.Fpga().InputGain());
		meta[44] = "Output gain = " + num2str(in_meta.Fpga().OutputGain());
		meta[45] = "Power level = " + num2str(in_meta.Fpga().PowerLevel());
		meta[46] = "FPGA mode = " + num2str(in_meta.Fpga().FpgaMode());
		meta[47] = "Signal offset = " + num2str(in_meta.Fpga().SignalOffset());
		if(in_meta.Fpga().IsRealTimeConv()){
			meta[48] = "Real time convolution = true";
		}else{
			meta[48] = "Real time convolution = false";
		}
		// (2) Attenuation
		meta[49] = "Attenuation_type = " + in_meta.Fpga().AttenuationType();
		meta[50] = "Transmit power [dBm] = " + num2str(in_meta.Fpga().Pt_dBm());
		meta[51] = "Transmit gain [dBi] = " + num2str(in_meta.Fpga().Gt_dBi());

		if(in_meta.targetType().targetType() == "complex" && in_meta.targetType().WaveformType() == "code"){
			meta[52] = "Transform = FFT";
		}else{
			meta[52] = "Transform = ";
		}

		meta[53] = "Image gain [pow] = " + num2str(in_meta.Fpga().ImageGain());
		meta[54] = "Slant range difference [m] = " + num2str(in_meta.Fpga().Distance());
		meta[55] = "Path loss [dB] = " + num2str(in_meta.Fpga().PathLoss());
		// (3) Others
		meta[56] = "Radar dist cnt = " + num2str(in_meta.Fpga().DistanceCnt());
		meta[57] = "Increasing time [samples] = " + num2str(in_meta.Fpga().Increasing_time());


		meta.WriteASCII(file_out_RCS_meta.c_str());
	}
	/**
	 * Write the RCS meta data file
	 *
	 * @param [in] Sar - Sar class
	 * @param [in] mesh - MeshDef class
	 * @param [in] Ef - EF class
	 * @param [in] ang - ANG class
	 * @param [in] Rn_slant - Slant range at near
	 * @param [in] Rc_slant - Slant range at center
	 * @param [in] Rf_slant - Slant range at far
	 * @param [in] file_out_RCS_meta - (String) RCS meta file name with path
	 */
	void WriteMeta(const SAR& Sar, const MeshDef& mesh, const EF& Ef, const ANG& ang, const FPGA& Fpga, const TargetTYPE& TargetType,
				   const double Rn_slant, const double Rc_slant, const double Rf_slant, const D1<double>& freq,
				   const double theta_sq_mean, const double theta_l_mean, const double Vs_mean,
				   const string file_out_RCS_meta){
//		string flagIsMulLook = (MulAng.IsMulLook())? "YES":"NO";
//		string flagIsMulAsp  = (MulAng.IsMulAsp())?  "YES":"NO";
		double LookMin = mat::min(ang.Look());
		double LookMax = mat::max(ang.Look());
		double AspMin  = mat::min(ang.Asp());
		double AspMax  = mat::max(ang.Asp());
		double AspInterval = abs(ang.Asp()[1] - ang.Asp()[0]);
		double FreqInterval = abs(freq[1] - freq[0]);



		unsigned int increasing_time = (unsigned int)(std::ceil( Sar.Tr() * FPGA_TX_BANDWIDTH ));
		string isRealTimeConv = (Fpga.IsRealTimeConv())? "true":"false";


		Meta in_meta(LookMin, LookMax, AspMin, AspMax, Sar.f0(), Sar.Fr(), Sar.BWrg(), Sar.PRF(),
					 Rn_slant, Rc_slant, Rf_slant,
					 theta_sq_mean, theta_l_mean, Vs_mean,
					 Fpga, Sar.Nr(), ang.GetNum(), Ef.TxPol(),
					 Ef.RxPol(), Ef.MaxLevel(), mesh.Scale(), mesh.dRad(),
					 // Add
					 Sar.Tr(), AspInterval, FreqInterval, Sar.isUpChirp(), TargetType);


		WriteMeta(in_meta, file_out_RCS_meta);
	}
	/**
	 * Calculate the maximum amplitude value for CPLX<float> array
	 *
	 * @param [in] in - (vector<D2<CPLX<float> >)Input array
	 * @param [in] i - (long) The number of bouncing level
	 * @return Return a maximum floating value
	 */
	float MaxAbs(const D2<CPLX<float> >& in, const long i){
		float out = -999999.999;		
		for(long j=0;j<in.GetN();++j){
			float tmp = in[i][j].abs();
			if(tmp > out){
				out = tmp;
			}
		}
		return out;
	}
	/**
	 * Calculate the maximum amplitude value for CPLX<float> array
	 *
	 * @param [in] in - (vector<D2<CPLX<float> >)Input array
	 * @return Return a maximum floating value
	 */
	float MaxAbs(const D2<CPLX<float> >& in){
		float out = -999999.999;
		for(size_t j=0;j<in.GetN();++j){
			for(size_t i=0;i<in.GetM();++i){
				float tmp = in[i][j].abs();
				if(tmp > out){
					out = tmp;
				}
			}
		}
		return out;
	}
	/**
	 * Convert the CPLX<float> into complex<float>
	 *
	 * @param [in] in - CPLX<float> class
	 * @return Return a complex<float> class
	 */
	complex<float> CPLXFloat2ComplexFloat(const CPLX<float>& in){
		return complex<float>(in.r(), in.i());
	}
	/**
	 * Save RCS dataset into file system (*.rcs)
	 *
	 * @param RCS - D2<CPLX<float> > RCS data
	 * @param FREQ - vector<double> Near frequency series
	 * @param file_RCS_save - RCS dataset file name
	 */
	void WriteRCSData(const D2<CPLX<float> >& RCS, const vector<double>& FREQ, const string& file_RCS_save){
		//
		// Assign to memory
		//
		// map<freq[i], RCS[*][j]>
		// RCS.GetN() == FREQ.size()
		map<double, pair<CPLX<float>, CPLX<float>> > RCS_save;
		for(size_t j=0;j<FREQ.size();++j){
			size_t FREQ_10e6 = round(FREQ[j]*1000000);
			RCS_save[FREQ_10e6] = make_pair(RCS[0][j],	// Phi   (H)
										  	RCS[1][j]);	// Theta (V)
		}

//		cout<<"(0) Input data:"<<endl;
//		for(size_t kk=0;kk<5;++kk){
//			printf("kk = %ld, freq = %f, RCS = (%f,%f)\n", kk, FREQ[kk], RCS[0][kk].r(), RCS[0][kk].i());
//		}

		//
		// Store into a file system
		//
		// +----------+----------------+------------------------------+
		// |   size   | freq | IQ | IQ | freq | IQ | IQ |.............|
		// +----------+----------------+------------------------------+
		//                      H   V            H    V
		//             <----- #1 -----> <----- #2 -----> ....
		//
		//+------------------------+
		//|  (1) Read old data     |
		//+------------------------+
		size_t num;
		size_t ffreq_10e6;
		float cplx_r_h, cplx_i_h, cplx_r_v, cplx_i_v;

		if(FileExist(file_RCS_save)){
			ifstream fin_RCS_save(file_RCS_save.c_str(), ios::binary);
			if(!fin_RCS_save.is_open()){ cerr<<"ERROR::Cannot open fin_RCS_save file"<<endl; exit(EXIT_FAILURE); }

			fin_RCS_save.read((char*)&num, sizeof(size_t));	// 8 [Bytes]

			for(size_t j=0;j<num;++j){
				fin_RCS_save.read((char*)&ffreq_10e6, sizeof(size_t));// 8 [Bytes]
				fin_RCS_save.read((char*)&cplx_r_h,   sizeof(float));	// 4 [Bytes]
				fin_RCS_save.read((char*)&cplx_i_h,   sizeof(float));	// 4 [Bytes]
				fin_RCS_save.read((char*)&cplx_r_v,   sizeof(float));	// 4 [Bytes]
				fin_RCS_save.read((char*)&cplx_i_v,   sizeof(float));	// 4 [Bytes]
				//+--------------------+
				//|  Combine data      |
				//+--------------------+
				RCS_save[ffreq_10e6] = make_pair(CPLX<float>(cplx_r_h,cplx_i_h),
												 CPLX<float>(cplx_r_v,cplx_i_v));
			}

			fin_RCS_save.close();
		}

//		cout<<"(1) After read:"<<endl;
//		for(size_t kk=0;kk<5;++kk){
//			double freq9 = FREQ[kk];
//			auto it = RCS_save.find(freq9);
//			printf("kk = %ld, freq = %f, RCS = (%f,%f)\n", kk, it->first, it->second.first.r(), it->second.first.i());
//		}

		//+------------------------+
		//|  (2) Write new data    |
		//+------------------------+
		ofstream fout_RCS_save(file_RCS_save.c_str(), ios::binary);
		
		if(!fout_RCS_save.is_open()){ cerr<<"ERROR::Cannot write fout_RCS_save file"<<endl; exit(EXIT_FAILURE); }
		num = RCS_save.size();
		fout_RCS_save.write((char*)&num, sizeof(size_t));	// 8 [Bytes]
		
		for(auto it=RCS_save.begin();it!=RCS_save.end();++it){
			ffreq_10e6 = it->first;
			cplx_r_h   = it->second.first.r();
			cplx_i_h   = it->second.first.i();
			cplx_r_v   = it->second.second.r();
			cplx_i_v   = it->second.second.i();
			fout_RCS_save.write((char*)&ffreq_10e6, sizeof(size_t));// 8 [Bytes]
			fout_RCS_save.write((char*)&cplx_r_h,   sizeof(float));	// 4 [Bytes]
			fout_RCS_save.write((char*)&cplx_i_h,   sizeof(float));	// 4 [Bytes]
			fout_RCS_save.write((char*)&cplx_r_v,   sizeof(float));	// 4 [Bytes]
			fout_RCS_save.write((char*)&cplx_i_v,   sizeof(float));	// 4 [Bytes]
		}
		
		fout_RCS_save.close();	
	}
	/**
	 * Read RCS dataset (*.rcs)
	 *
	 * @param file_RCS_from - (string) RCS dataset file name (*.rcs)
	 * @return Return a map<double, pair<complex<float>, complex<float> > container
	 */
	map<double, pair<CPLX<float>, CPLX<float>> > ReadRCSData(const string file_RCS_from){
		// Read *.rcs in Database (all frequency series)
		ifstream fin(file_RCS_from.c_str(), ios::binary);
		if(!fin.is_open()){ cerr<<"ERROR::Cannot open file : "<<file_RCS_from<<endl; exit(EXIT_FAILURE); }
		// allocation
		size_t num;
		size_t ffreq_10e6;
		float cplx_r_h, cplx_i_h, cplx_r_v, cplx_i_v;
		map<double, pair<CPLX<float>, CPLX<float>> > RCS;
		// Read
		//
		// RCS dataset file system
		//
		// +----------+----------------+------------------------------+
		// |   size   | freq | IQ | IQ | freq | IQ | IQ |.............|
		// +----------+----------------+------------------------------+
		//                      H   V            H    V
		//             <----- #1 -----> <----- #2 -----> ....
		//
		fin.read((char*)&num, sizeof(size_t));
		for(size_t k=0;k<num;++k){
			fin.read((char*)&ffreq_10e6, sizeof(size_t));
			fin.read((char*)&cplx_r_h,   sizeof(float));
			fin.read((char*)&cplx_i_h,   sizeof(float));
			fin.read((char*)&cplx_r_v,   sizeof(float));
			fin.read((char*)&cplx_i_v,   sizeof(float));
			// insert into map container
			RCS[ffreq_10e6] = make_pair(CPLX<float>(cplx_r_h,cplx_i_h),
								   	    CPLX<float>(cplx_r_v,cplx_i_v));
		}
		fin.close();
		
		return RCS;
	}
	/**
	 * Get file size
	 *
	 * @param filename - (string) Input file name
	 * @return Return a file size in bytes
	 */
	size_t FileSize(const string filename){
		std::ifstream fin(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
		if(!fin.is_open()){ cerr<<"ERROR::FileSize:Cannot open file : "<<filename<<endl; exit(EXIT_FAILURE); }
		
		size_t sz = fin.tellg();
		
		fin.close();
		
		return sz; 
	}
	/**
	 * Get the MD5 string
	 *
	 * @param filename - (string) Input file name
	 * @return Return a MD5 string
	 */
	string GetMd5String(const string filename){
		ifstream fin(filename, ios::binary);
		if(!fin.is_open()){ cerr<<"ERROR::GetMd5String:Cannot open file : "<<filename<<endl; exit(EXIT_FAILURE); }
		
		size_t file_sz = FileSize(filename);
		unsigned char* tmp = new unsigned char[file_sz];
		
		fin.read((char*)tmp, file_sz);
		fin.close();
		
		
		unsigned char result[MD5_DIGEST_LENGTH];
		MD5(tmp, file_sz, result);
		
		std::ostringstream sout;
		sout<<std::hex<<std::setfill('0');
		for(long long c: result){
			sout<<std::setw(2)<<(long long)c;
		}
		string MD5String = sout.str();
		
		delete[] tmp;
		
		return MD5String;
	}
	/**
	 * Print the real & near look & aspect angle on console
	 * @param [in] look_deg: [deg] Real Look angle
	 * @param [in] n_look_deg: [deg] Near look angle
	 * @param [in] asp_deg: [deg] Real aspect angle
	 * @param [in] n_asp_deg: [deg] Near aspect angle
	 * @param [in] i: For loop index
	 */
	void PrintRealNearLookAsp(const double look_deg, const double n_look_deg, const double asp_deg, const double n_asp_deg, const size_t i){
		if(i == 0){
			printf("+-----------------------------------------------+\n");
			printf("|   Real & Near for Look & aspect angle [deg]   |\n");
			printf("+------------+-----------+----------+-----------+\n");
			printf("|    Look    |   n_look  |    asp   |   n_asp   |\n");
			printf("+------------+-----------+----------+-----------+\n");
		}
		printf("   %f   %f   %f   %f\n", rad2deg(look_deg), rad2deg(n_look_deg), rad2deg(asp_deg), rad2deg(n_asp_deg));
	}
//	/**
//	 * Calculate the WF hardware Gain coefficient by FPGA G0 [dB] & Rd [m]
//	 * @param [in] FPGA_G0dB: [dB] WF emulator simuation antenna gain
//	 * @param [in] FPGA_Rd: [m] WF physical distance from Tx to Rx
//	 * @param [in] lambda: [m] Wavelength
//	 * @param [in] R_real: [m] Physical distance
//	 * @return Return the gain coefficient in dB
//	 */
//	double FindWFCoefficientIndB(const double FPGA_G0dB, const double FPGA_Rd, const double lambda, const double R_real){
//		double G0 = std::pow(10, (FPGA_G0dB/10.));
//		double coeff = 1./(def::PI4 * mat::Square(R_real)) *
//					   (G0*mat::Square(lambda)/(def::PI4 * mat::Square(FPGA_Rd)));
//		return float(10.*std::log10(coeff));	// [dB]
//	}
	/**
	 * Convert Linear to dBsm
	 * @param [in] rcs: single RCS value
	 * @param [in] f0: [Hz] Center frequency
	 * @return Return a value in dB
	 */
	double RCS2dBsm(const double& rcs, const double f0){
		double lambda = def::C/f0;
		double factor = 4*PI/(lambda*lambda);
		double m2 = factor * (rcs * rcs);

		if(abs(m2) < 1E-20){
			m2 = 1E-2;
		}
		return 10*log10(m2);
	}
	/**
	 * Convert Linear to m2
	 * @param [in] rcs: single RCS value
	 * @param [in] f0: [Hz] Center frequency
	 * @return Return a value in dB
	 */
	double RCS2m2(const double& rcs, const double f0){
		double lambda = def::C/f0;
		double factor = 4*PI/(lambda*lambda);
		return factor * (rcs * rcs);
	}
	/**
	 * Calculate the WF hardware Gain coefficient by FPGA G0 [dB] & Rd [m]
	 * @param [in] FPGA_G0dB: [dB] WF emulator simuation antenna gain
	 * @param [in] FPGA_Rd: [m] WF physical distance from Tx to Rx
	 * @param [in] lambda: [m] Wavelength
	 * @param [in] R_real: [m] Physical distance
	 * @param [in] PathLoss: [dB] Cable loss
	 * @return Return the gain coefficient in dB
	 */
	double FindWFCoefficientIndB(const FPGA& fpga, const double R_real){


		double Gt = std::pow(10., fpga.Gt_dBi()/10.);
		double G0 = std::pow(10., fpga.ImageGain()/10.);
		double Pt = std::pow(10., fpga.Pt_dBm()/10.);

		double coeff = Pt*Gt/(def::PI4*Square(R_real)*Square(R_real)) * (Square(fpga.Distance())/G0);	// Linear

		float out = float(10.*std::log10(coeff));

		if(fpga.AttenuationType() == "free_space"){
			return out;	// [dB]
		}else{	// "cable"
			return out + fpga.PathLoss();	// [dB]
		}

//		double G0 = std::pow(10, (FPGA_G0dB/10.));
//		double coeff = 1./(def::PI4 * mat::Square(R_real)) *
//					   (G0*mat::Square(lambda)/(def::PI4 * mat::Square(FPGA_Rd)));
//		return float(10.*std::log10(coeff));	// [dB]
	}

	Meta ReadWFEmulatorMeta(const string file_meta){
		ifstream fin(file_meta);

		if(fin.fail()){
			cerr<<"ERROR::Cannot open file -> "<<file_meta<<endl;
			exit(EXIT_FAILURE);
		}

		string buffer;
		Meta meta;
		while(fin){
			std::getline(fin, buffer);
			D1<string> sub = StrSplit(buffer, '=');
			if(sub.GetNum() >= 2) {
				if (sub[0] == "Look angle (Min) [deg] ") { meta.look_min() = deg2rad(str2num<double>(sub[1])); }
				if (sub[0] == "Look angle (Max) [deg] ") { meta.look_max() = deg2rad(str2num<double>(sub[1])); }
				if (sub[0] == "Target Aspect angle (Min) [deg] ") { meta.asp_min() = deg2rad(str2num<double>(sub[1])); }
				if (sub[0] == "Target Aspect angle (Max) [deg] ") { meta.asp_max() = deg2rad(str2num<double>(sub[1])); }
				if (sub[0] == "Transmitted frequency [Hz] ") { meta.f0() = str2num<double>(sub[1]); }
				if (sub[0] == "ADC sampling rate [kHz] ") { meta.Fr() = str2num<double>(sub[1]) * 1e3; }
				if (sub[0] == "Bandwidth [kHz] ") { meta.BWrg() = str2num<double>(sub[1]) * 1e3; }
				if (sub[0] == "PRF [Hz] ") { meta.PRF() = str2num<double>(sub[1]); }
				if (sub[0] == "Slant range at near [m] ") { meta.Rn_slant() = str2num<double>(sub[1]); }
				if (sub[0] == "Slant range at center [m] ") { meta.Rc_slant() = str2num<double>(sub[1]); }
				if (sub[0] == "Slant range at far [m] ") { meta.Rf_slant() = str2num<double>(sub[1]); }
				if (sub[0] == "Pulse width [sec] ") { meta.Tr() = str2num<double>(sub[1]); }
				if (sub[0] == "Aspect angle interval [deg] ") { meta.dAsp() = deg2rad(str2num<double>(sub[1])); }
				if (sub[0] == "Frequency interval [Hz] ") { meta.df() = str2num<double>(sub[1]); }
				if (sub[0] == "Mean squint [deg] ") { meta.theta_sq_mean() = deg2rad(str2num<double>(sub[1])); }
				if (sub[0] == "Mean look [deg] ") { meta.theta_l_mean() = deg2rad(str2num<double>(sub[1])); }
				if (sub[0] == "Mean velocity [m/s] ") { meta.Vs_mean() = str2num<double>(sub[1]); }
				if (sub[0] == "Number of Range [sample] ") { meta.Nr() = str2num<size_t>(sub[1]); }
				if (sub[0] == "Number of Azimuth [sample] ") { meta.Na() = str2num<size_t>(sub[1]); }
				if (sub[0] == "TX Polarization ") { meta.tx_pol() = string(sub[1].substr(1, sub[1].length() - 1)); }
				if (sub[0] == "RX Polarization ") { meta.rx_pol() = string(sub[1].substr(1, sub[1].length() - 1)); }
				if (sub[0] == "Max Bouncing Number ") { meta.Maxlevel() = str2num<double>(sub[1]); }
				if (sub[0] == "Mesh Scale Factor ") { meta.scale() = str2num<double>(sub[1]); }
				if (sub[0] == "Mesh Distance from targets center ") { meta.dR() = str2num<double>(sub[1]); }
				if (sub[0] == "Target Type ") { meta.targetType().targetType() = string(sub[1].substr(1, sub[1].length() - 1)); }
				if (sub[0] == "Single RCS ") { meta.targetType().SingleRCS() = str2num<double>(sub[1]); }
				if (sub[0] == "Waveform Type ") { meta.targetType().WaveformType() = string(sub[1].substr(1, sub[1].length() - 1)); }
				if (sub[0] == "Code Type ") { meta.targetType().CodeType() = string(sub[1].substr(1, sub[1].length() - 1)); }
				if (sub[0] == "Code String ") { meta.targetType().CodeString() = string(sub[1].substr(1, sub[1].length() - 1)); }
				if (sub[0] == "Updown Chirp ") {
					string updown_chirp = string(sub[1].substr(1, sub[1].length() - 1));
					if(updown_chirp == "up"){
						meta.isUpChirp() = true;
					}else{
						meta.isUpChirp() = false;
					}
				}
				// FPGA
				// (1) FGPA options
				if (sub[0] == "Input gain ") { meta.Fpga().InputGain() = str2num<long>(sub[1]); }
				if (sub[0] == "Output gain ") { meta.Fpga().OutputGain() = str2num<long>(sub[1]); }
				if (sub[0] == "Power level ") { meta.Fpga().PowerLevel() = str2num<long>(sub[1]); }
				if (sub[0] == "FPGA mode ") { meta.Fpga().FpgaMode() = str2num<long>(sub[1]); }
				if (sub[0] == "Signal offset ") { meta.Fpga().SignalOffset() = str2num<long>(sub[1]); }
				if (sub[0] == "Real time convolution ") {
					if(sub[1] == "true"){
						meta.Fpga().IsRealTimeConv() = true;
					}else{
						meta.Fpga().IsRealTimeConv() = false;
					}
				}
				// (2) Attenuation
				if (sub[0] == "Attenuation_type ") {
					meta.Fpga().AttenuationType() = string(sub[1].substr(1, sub[1].length() - 1));
				}
				if (sub[0] == "Transmit power [dBm] ") { meta.Fpga().Pt_dBm() = str2num<double>(sub[1]); }
				if (sub[0] == "Transmit gain [dBi] ") { meta.Fpga().Gt_dBi() = str2num<double>(sub[1]); }
				if (sub[0] == "Image gain [pow] ") { meta.Fpga().ImageGain() = str2num<long>(sub[1]); }
				if (sub[0] == "Slant range difference [m] ") { meta.Fpga().Distance() = str2num<double>(sub[1]); }
				if (sub[0] == "Path loss [dB] ") { meta.Fpga().PathLoss() = str2num<double>(sub[1]); }
				// (3) Others
				if (sub[0] == "Radar dist cnt ") { meta.Fpga().DistanceCnt() = str2num<long>(sub[1]); }
				if (sub[0] == "Increasing time [samples] ") { meta.Fpga().Increasing_time() = str2num<long>(sub[1]); }
			}
		}

		fin.close();

//		meta.Print();

		return meta;
	}

	D1<double> ReadWFEmulatorCoeff(const string file_coeff){
		ifstream fin(file_coeff);

		if(fin.fail()){
			cerr<<"ERROR::Cannot open file -> "<<file_coeff<<endl;
			exit(EXIT_FAILURE);
		}

		vector<double> data;

		string buffer;

		while(fin){
			std::getline(fin, buffer);
			if(buffer.length() > 0){
				data.push_back(str2num<double>(buffer));
			}
		}

		fin.close();

		// Assgin to D1<double>
		D1<double> out(&(data[0]), data.size());

		return out;
	}

	SV<double> ReadWFEmulatorStateVector(const string file_sv, const bool SHOW=false){
		ifstream fin(file_sv);

		if(fin.fail()){
			cerr<<"ERROR::Cannot open file -> "<<file_sv<<endl;
			exit(EXIT_FAILURE);
		}

		vector<double> t;
		vector<VEC<double> > pos, vel;

		string buffer;

		while(fin){
			std::getline(fin, buffer);
			if(buffer.length() > 0){
				D1<string> tmp = StrSplit(buffer, '\t');

				t.push_back( str2num<double>(tmp[0]) );
				pos.push_back( VEC<double>(str2num<double>(tmp[1]), str2num<double>(tmp[2]), str2num<double>(tmp[3])) );
				vel.push_back( VEC<double>(str2num<double>(tmp[4]), str2num<double>(tmp[5]), str2num<double>(tmp[6])) );
			}
		}

		fin.close();

		// Assgin to D1<double>
		D1<double> t2(&(t[0]), t.size());
		D1<VEC<double> > pos2(&(pos[0]), pos.size());
		D1<VEC<double> > vel2(&(vel[0]), vel.size());

		SV<double> out(t2, pos2, vel2, t.size());

		// SHOW
		if(SHOW){
			cout<<"#First  -> "; out.Print(0);
			cout<<"#Center -> "; out.Print(out.GetNum()/2);
			cout<<"#End    -> "; out.Print(out.GetNum()-1);
		}

		return out;
	}

	VEC<double> ReadWFEmulatorTargetLocation(const string file_Tar){
		ifstream fin(file_Tar);

		if(fin.fail()){
			cerr<<"ERROR::Cannot open file -> "<<file_Tar<<endl;
			exit(EXIT_FAILURE);
		}

		VEC<double> data;

		string buffer;

		while(fin){
			std::getline(fin, buffer);
			if(buffer.length() > 0){
				D1<string> tmp = StrSplit(buffer, '\t');
				data = VEC<double>(str2num<double>(tmp[0]), str2num<double>(tmp[1]), str2num<double>(tmp[2]));
			}
		}

		fin.close();

		return data;
	}

	void ReadWFEmulatorFreq(const string file_FreqNFreq, D1<double>& Freq, D1<double>& nFreq){
		ifstream fin(file_FreqNFreq);

		if(fin.fail()){
			cerr<<"ERROR::Cannot open file -> "<<file_FreqNFreq<<endl;
			exit(EXIT_FAILURE);
		}

		vector<double> freq, nfreq;

		string buffer;

		while(fin){
			std::getline(fin, buffer);
			if(buffer.length() > 0){
				D1<string> tmp = StrSplit(buffer, '\t');
				freq.push_back(str2num<double>(tmp[0]));
				nfreq.push_back(str2num<double>(tmp[1]));
			}
		}

		fin.close();

		// Assgin to D1<double>
		Freq = D1<double>(&(freq[0]), freq.size());
		nFreq = D1<double>(&(nfreq[0]), nfreq.size());
	}

	D2<CPLX<double> > ReadWFEmulatorSAREcho(const string file_meta, const string file_s0, D1<float>& Coef, D1<float>& Dist, D1<float>& Dopp, D1<float>& AzPt, D1<float>& PhsSin, D1<float>& PhsCos){
		Meta meta = ReadWFEmulatorMeta(file_meta);

		D1<CPLX<float> > tmp(meta.Nr());
		D2<CPLX<double> > s0(meta.Na(), meta.Nr());

		ifstream fin(file_s0, ios::binary);

		if(fin.fail()){
			cerr<<"ERROR::Cannot open file -> "<<file_s0<<endl;
			exit(EXIT_FAILURE);
		}

		Coef = D1<float>(meta.Na());	// Coefficient of range center for EXPORT by each angle [dB]
		Dist = D1<float>(meta.Na());	// Instantaneous slant range distance (one-way) from sensor to ZERO squint direction at Zero Doppler plane [m]
		Dopp = D1<float>(meta.Na());	// Doppler value at slant range direction
		AzPt = D1<float>(meta.Na());	// Azimuth beam pattern in linear (NOT in dB)
		PhsSin = D1<float>(meta.Na());	// Additional phase need add in FPGA and ONLY for SINGLE target (Sin)
		PhsCos = D1<float>(meta.Na());	// Additional phase need add in FPGA and ONLY for SINGLE target (Cosin)


		float coeff;
		float dis;
		float dop;
		float azPt;
		float phsSin;
		float phsCos;
		for(size_t i=0;i<meta.Na();++i){
			// Read coeff & dis & dop;
			fin.read(reinterpret_cast<char*>(&coeff), 1*sizeof(float));
			fin.read(reinterpret_cast<char*>(&dis),   1*sizeof(float));
			fin.read(reinterpret_cast<char*>(&dop),   1*sizeof(float));
			fin.read(reinterpret_cast<char*>(&azPt),  1*sizeof(float));
			fin.read(reinterpret_cast<char*>(&phsSin),  1*sizeof(float));
			fin.read(reinterpret_cast<char*>(&phsCos),  1*sizeof(float));
			// Read RCS
			fin.read(reinterpret_cast<char*>(tmp.GetPtr()), tmp.GetNum()*sizeof(CPLX<float>));
			// Assign
			Coef[i] = coeff;
			Dist[i] = dis;
			Dopp[i] = dop;
			AzPt[i] = azPt;
			PhsSin[i] = phsSin;
			PhsCos[i] = phsCos;
			// Assign
			for(size_t j=0;j<meta.Nr();++j){
				s0[i][j] = CPLX<double>(tmp[j].r(), tmp[j].i());
			}
		}

		fin.close();

		return s0;
	}


	vector<string> ReadStatusLog(const string file_LOG){
		//+---------------------------+
		//|     Read STATUS.log       |
		//+---------------------------+
		vector<string> STATUS;
		// 2nd line in STATUS.log will be
		// - RUNNING PAUSE STOP DELETE (control by TaskQueue.sh)
		// - FINISH ERROR (control by this program)
		fstream fin(file_LOG, fstream::in);
		if(!fin.good()){
			cout<<"ERROR::[WFds::ReadStatusLog]:No this file! -> ";
			cout<<file_LOG<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
			exit(EXIT_FAILURE);
		}
		char buf[10000];
		while(fin.getline(buf, 10000)){
			STATUS.push_back(string(buf));
		}
		fin.close();

		return STATUS;
	}


	void WriteStatusLog(const string file_LOG, const vector<string>& STATUS){
		//+---------------------------+
		//|     Write STATUS.log      |
		//+---------------------------+
		// 2nd line in STATUS.log will be
		// - RUNNING PAUSE STOP DELETE (control by TaskQueue.sh)
		// - FINISH ERROR (control by this program)
		ofstream fout;
		fout.open(file_LOG);
		if(fout.fail()){
			cout<<"ERROR::Input filename! -> ";
			cout<<file_LOG<<endl;
			cout<<"<<Press Enter to Stop>>"; getchar();
			exit(EXIT_FAILURE);
		}
		for(size_t i=0;i<STATUS.size();++i){
			fout<<STATUS[i]<<endl;
		}
		fout.close();
	}

}


#endif /* WFDS_H_ */
