//
//  material.h
//  PhysicalOptics13
//
//  Created by Steve Chiang on 4/21/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//
#ifndef MATERIAL_H_
#define MATERIAL_H_

#include <iostream>
#include <vector>
#include <iomanip>
#include <json/json.h>

using namespace std;

namespace material {
	class Material {
	public:
		Material():_idx(0),_er_r(0),_tang(0),_mr(0),_mi(0),_d(0){};
		Material(const long Idx, const string Name, const string Freq, const double Er_r, const double Tang, const double Mr,
				 const double Mi, const double D, const string Remark){
			_idx = Idx;
			_name = Name;
			_freq = Freq;
			_er_r = Er_r;
			_tang = Tang;
			_mr = Mr;
			_mi = Mi;
			_d = D;
			_remark = Remark;
		}
		void Print(){
			cout<<"+--------------+"<<endl;
			cout<<"|   Material   |"<<endl;
			cout<<"+--------------+"<<endl;
			cout<<"Idx    = "<<_idx<<endl;
			cout<<"name   = '"<<_name<<"'"<<endl;
			cout<<"Freq   = '"<<_freq<<"'"<<endl;
			cout<<"er_r   = "<<_er_r<<endl;
			cout<<"tang   = "<<_tang<<endl;
			cout<<"mr     = "<<_mr<<endl;
			cout<<"mi     = "<<_mi<<endl;
			cout<<"d      = "<<_d<<endl;
			cout<<"Remark = '"<<_remark<<"'"<<endl;
		}
		long& idx(){return _idx;}
		string& name(){return _name;}
		string& freq(){return _freq;}
		double& er_r(){return _er_r;}
		double& tang(){return _tang;}
		double& mr(){return _mr;}
		double& mi(){return _mi;}
		double& d(){return _d;}
		string& remark(){return _remark;}
		void SetToPEC(){
			_er_r = 1.0E+30;
			_tang = 0.0;
			_mr   = 1.0;
			_mi   = 0.0;
			_d    = 0.0000001;
		}
		// const
		const long& idx()const{return _idx;}
		const string& name()const{return _name;}
		const string& freq()const{return _freq;}
		const double& er_r()const{return _er_r;}
		const double& tang()const{return _tang;}
		const double& mr()const{return _mr;}
		const double& mi()const{return _mi;}
		const double& d()const{return _d;}
		const string& remark()const{return _remark;}
	private:
		long _idx;		// 1. (integer) [x] Index
		string _name;	// 2. (string)  [x] Name
		string _freq;	// 3. (string)  [x] Frequency
		double _er_r;	// 4. (float)   [x] Real part of relative permittivity, relative to Air [F/m]
		double _tang;	// 5. (float)   [x] Loss tangent (x10^-4)
		double _mr;		// 6. (float)   [x] real part of relative permeability, relative to Air [H/m]
		double _mi;		// 7. (float)   [x] imag part of relative permeability, relative to Air [H/m]
		double _d;		// 8. (float)   [mm] Depth of this Layer
		string _remark;	// 9. (string)  [x] Remark
	};
	
	
	class MaterialDB {
	public:
		MaterialDB(){};
		MaterialDB(const string Filename, const bool IsJson=false, const bool IsSINGLE=false, const string file_LOG=""){
			filename = Filename;
			if(IsJson){
				ReadFromJson(IsSINGLE, file_LOG);
			}else{
				Read();
			}
		}
		long size(){
			return Mat.size();
		}
		Material Get(const size_t MatIdx){
			if(MatIdx > Mat.size()){
				cerr<<"ERROR::MaterialDB:Get:The MatIdx is out of range, MatIdx = "<<
						MatIdx<<", Mat.size() = "<<Mat.size()<<endl;
				exit(EXIT_FAILURE);
			}
			return Mat[MatIdx];
		}
		void Print(){
			for(unsigned long i=0;i<Mat.size();++i){
				Mat[i].Print();
			}
		}
		void Print1Line(){
			printf("+--------------------------+\n");
			printf("|       Material List      |\n");
			printf("+--------------------------+\n");
			for(unsigned long i=0;i<Mat.size();++i){
				printf("Idx=%ld, Name='%s', er_r=%f, tang=%f, mr=%f, mi=%f, d=%f\n",
						Mat[i].idx(), Mat[i].name().c_str(), Mat[i].er_r(),
						Mat[i].tang(), Mat[i].mr(), Mat[i].mi(), Mat[i].d());
			}
		}
	private:
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
		void ReadFromJson(const bool IsSINGLE, const string file_LOG){
			// (1) Read from json & store in json format
			Json::Value MAT_json = ReadJson(filename);
			// (2) Reformat to MaterialDB structure
			Json::Value mats = MAT_json["materials"];
			
			long idx;
			string name;
			string freq;
			double er_r, tang, mr, mi, d;
			string remark;
			
			if(mats.size() == 0){
				if(IsSINGLE){
					// TargetType = "single"
					cerr<<"WARNING::MaterialDB:ReadFromJson:Cannot read any material, size = 0. Please re-assigned the material in 3D editor."<<endl;
				}else{
					// TargetType = "complex"
					cerr<<"ERROR::MaterialDB:ReadFromJson:Cannot read any material, size = 0. Please re-assigned the material in 3D editor."<<endl;
					if(file_LOG != ""){
						vector<string> STATUS = ReadStatusLog(file_LOG);
						// 只要沒跑完，都是error
						STATUS[1] = "ERROR";
						STATUS[2] = "ERROR::MaterialDB:ReadFromJson:Cannot read any material, size = 0. Please re-assigned the material in 3D editor.";
						// Write this file first for default
						WriteStatusLog(file_LOG, STATUS);
					}
					exit(EXIT_FAILURE);
				}
			}
			
			for(int i=0;i<int(mats.size());++i){
				idx  = mats[i]["index"].asInt();			// Count index
				name = mats[i]["name"].asString();
				freq = mats[i]["frequency"].asString();
				er_r = mats[i]["realPartFm"].asDouble();
				tang = mats[i]["lossTangent"].asDouble();
				mr   = mats[i]["realPartHm"].asDouble();
				mi   = mats[i]["imagPartHm"].asDouble();
				d    = mats[i]["depthLayer"].asDouble();
				remark = mats[i]["remark"].asString();
				Mat.push_back( Material(idx, name, freq, er_r, tang, mr, mi, d, remark) );
			}
		}
		void Read(){
			fstream fin(filename.c_str(),fstream::in);
			if(!fin.good()){
				cerr<<"Can not read this file: "<<filename<<endl;
				exit(EXIT_FAILURE);
			}
			
			
			long idx;
			string name;
			string freq;
			double er_r, tang, mr, mi, d;
			string remark;
			
			
			char buf[1000];
			while(fin.getline(buf,1000)){
				// Skip comment line
				if(buf[0] != '#'){
					// Convert to string
					string str = string(buf);
					str = str.substr(1,str.length()-2);
					// string split
					D1<string> tmp = StrSplit(str, ',');
					idx  = str2num<long>(tmp[0]);
					name = tmp[1].substr(2,tmp[1].length()-3);
					freq = tmp[2].substr(2,tmp[2].length()-3);
					er_r = str2num<double>(tmp[3]);
					tang = str2num<double>(tmp[4]);
					mr   = str2num<double>(tmp[5]);
					mi   = str2num<double>(tmp[6]);
					d    = str2num<double>(tmp[7]);
					remark = tmp[8].substr(2,tmp[8].length()-3);
					Mat.push_back( Material(idx,name,freq,er_r,tang,mr,mi,d,remark) );
				}
			}
			
			fin.close();
		}
	public:
		string filename;
		vector<Material> Mat;
	};

}


#endif /* MATERIAL_H_ */



