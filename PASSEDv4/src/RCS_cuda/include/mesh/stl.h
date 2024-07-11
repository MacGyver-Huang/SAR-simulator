//
//  stl.h
//  PhysicalOptics01
//
//  Created by Steve Chiang on 1/24/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef stl_h
#define stl_h

#include <mesh/stleach.h>
#include <basic/d1.h>
#include <bvh/obj.h>
#include <bvh/triangle.h>
#include <vector>
#include <basic/vec.h>

using namespace d1;
using namespace std;
using namespace vec;

namespace stl{
	class STL{
	public:
		STL(const long Num=1){
			num = Num;
			Data = D1<STLEach>(num);
		}
		// Overload
		const STLEach& operator[](const long i)const{ return Data[i]; }
		STLEach& operator[](const long i){ return Data[i]; }
		template<typename T> void Rx(const double rad){
			D2<double> M0 = def_func::Rx(rad);
			D2<T> M(M0.GetM(),M0.GetN());
			for(long j=0;j<M.GetN();++j){
				for(long i=0;i<M.GetM();++i){
					M[i][j] = (T)M0[i][j];
				}
			}
			// vert & N
			for(long i=0;i<num;++i){
				Data[i].V1 = D22VEC(M * VEC2D2(Data[i].V1, 3, 1));
				Data[i].V2 = D22VEC(M * VEC2D2(Data[i].V2, 3, 1));
				Data[i].V3 = D22VEC(M * VEC2D2(Data[i].V3, 3, 1));
				Data[i].N  = D22VEC(M * VEC2D2(Data[i].N,  3, 1));
			}
		}
		template<typename T> void Ry(const double rad){
			D2<double> M0 = def_func::Ry(rad);
			D2<T> M(M0.GetM(),M0.GetN());
			for(long j=0;j<M.GetN();++j){
				for(long i=0;i<M.GetM();++i){
					M[i][j] = (T)M0[i][j];
				}
			}
			// vert & N
			for(long i=0;i<num;++i){
				Data[i].V1 = D22VEC(M * VEC2D2(Data[i].V1, 3, 1));
				Data[i].V2 = D22VEC(M * VEC2D2(Data[i].V2, 3, 1));
				Data[i].V3 = D22VEC(M * VEC2D2(Data[i].V3, 3, 1));
				Data[i].N  = D22VEC(M * VEC2D2(Data[i].N,  3, 1));
			}
		}
		template<typename T> void Rz(const double rad){
			D2<double> M0 = def_func::Rz(rad);
			D2<T> M(M0.GetM(),M0.GetN());
			for(long j=0;j<M.GetN();++j){
				for(long i=0;i<M.GetM();++i){
					M[i][j] = (T)M0[i][j];
				}
			}
			// vert & N
			for(long i=0;i<num;++i){
				Data[i].V1 = D22VEC(M * VEC2D2(Data[i].V1, 3, 1));
				Data[i].V2 = D22VEC(M * VEC2D2(Data[i].V2, 3, 1));
				Data[i].V3 = D22VEC(M * VEC2D2(Data[i].V3, 3, 1));
				Data[i].N  = D22VEC(M * VEC2D2(Data[i].N,  3, 1));
			}
		}
		template<typename T> void Translate(const VEC<T>& move){
			for(long i=0;i<Data.GetNum();++i){
				Data[i].Translate(move);
			}
		}
		template<typename T> void ReDirection(const VEC<T>& Ps, const VEC<T>& Pt){
			VEC<T> uv = Unit(Ps - Pt);
			double theta_y = deg2rad(90.) - vec::angle(uv, VEC<T>(0,0,1));	// Elevation angle (from XY-plane)
			double theta_z = acos(uv.x()/cos(theta_y));						// Azimuth angle (from X-Axis)
			// Rotation with Y-Axis(Elevation) & Z-Axis(Azimuth)
			Ry<T>(-theta_y);
			Rz<T>(theta_z);
		}
		template<typename T> void Match2Coordinate(const VEC<T>& xuv, const VEC<T>& yuv, const VEC<T>& zuv){
			// Ref: http://www.eng.fsu.edu/~chandra/courses/egm5611/04/topics/chap2/B/2b11.html
			VEC<double> xx(1,0,0);
			VEC<double> yy(0,1,0);
			VEC<double> zz(0,0,1);
			
			D2<double> Q(3,3);
			Q[0][0] = dot(xx, xuv); Q[0][1] = dot(xx, yuv); Q[0][2] = dot(xx, zuv);
			Q[1][0] = dot(yy, xuv); Q[1][1] = dot(yy, yuv); Q[1][2] = dot(yy, zuv);
			Q[2][0] = dot(zz, xuv); Q[2][1] = dot(zz, yuv); Q[2][2] = dot(zz, zuv);
			
			// vert & N
			for(long i=0;i<num;++i){
				Data[i].V1 = D22VEC(Q * VEC2D2(Data[i].V1, 3, 1));
				Data[i].V2 = D22VEC(Q * VEC2D2(Data[i].V2, 3, 1));
				Data[i].V3 = D22VEC(Q * VEC2D2(Data[i].V3, 3, 1));
				Data[i].N  = D22VEC(Q * VEC2D2(Data[i].N,  3, 1));
			}
		}
		// Mics.
		void ReadBinary(const string file){
			// Read STL Binary
			ifstream fin(file.c_str(),ios::in | ios::binary);
			if(fin.good()){
				unsigned int Num;
				D1<VEC<float> > tmp(4);
				uint16_t att;
				fin.read(hdr, 80);
				fin.read((char*)&Num, sizeof(Num));
				
				// assignment
				num = Num;
				Data = D1<STLEach>(num);
				
				
				for(int i=0;i<num;++i){
					fin.read((char*)tmp.GetPtr(), sizeof(VEC<float>)*4);
					fin.read((char*)&att, sizeof(uint16_t));
					// assignment
					Data[i].N  = tmp[0];
					Data[i].V1 = tmp[1];
					Data[i].V2 = tmp[2];
					Data[i].V3 = tmp[3];
					Data[i].att= att;
				}
			}else{
				cout<<"ERROR::str::ReadBinary:Check the input file : "<<file<<endl;
			}
			fin.close();
		}
		void WriteBinary(const string file){
			// Wrtie STL binary
			ofstream fout(file.c_str(), ios::out | ios::binary);
			if(fout.good()){
				// Header
				fout.write((char*)hdr, sizeof(hdr));
				// number
				fout.write((char*)&num, sizeof(unsigned int));
				// STLEach
				for(int i=0;i<num;++i){
					// Normal vector
					fout.write((char*)&(Data[i].N),  sizeof(float)*3);
					// Vertex 1
					fout.write((char*)&(Data[i].V1), sizeof(float)*3);
					// Vertex 2
					fout.write((char*)&(Data[i].V2), sizeof(float)*3);
					// Vertex 3
					fout.write((char*)&(Data[i].V3), sizeof(float)*3);
					// Attribute count
					fout.write((char*)&(Data[i].att), sizeof(uint16_t));
				}
			}else{
				cout<<"ERROR::str::WriteBinary:Check the output file : "<<file<<endl;
			}
			fout.close();
		}
		void Print(){
			cout<<"Num  = "<<num<<endl;
			for(long i=0;i<num;++i){
				cout<<" #"<<i<<endl;
				Data[i].Print();
			}
		}
		template<typename T> void RotArbitrary(const VEC<T>& uv, const VEC<T>& Pt, const double Theta){
			vector<double> theta;
			theta.push_back(Theta);
			
			if(abs(theta[0]) > deg2rad(90.)){
				double tmp = theta[0];
				theta[0] = sign(theta[0]) * deg2rad(90.);
				theta.push_back( sign(theta[0]) * (std::abs(tmp - deg2rad(90.))) );
			}
			// vert
			for(long i=0;i<num;++i){
				Data[i].V1 -= Pt;
				Data[i].V2 -= Pt;
				Data[i].V3 -= Pt;
				for(long j=0;j<theta.size();++j){
					Data[i].V1 = vec::find::ArbitraryRotate(VEC<double>(Data[i].V1), theta[j], uv);
					Data[i].V2 = vec::find::ArbitraryRotate(VEC<double>(Data[i].V2), theta[j], uv);
					Data[i].V3 = vec::find::ArbitraryRotate(VEC<double>(Data[i].V3), theta[j], uv);
					Data[i].N  = vec::find::ArbitraryRotate(VEC<double>(Data[i].N), theta[j], uv);
				}
				Data[i].V1 += Pt;
				Data[i].V2 += Pt;
				Data[i].V3 += Pt;
			}
		}
		vector<Obj*> Convert2Obj(){
			vector<Obj*> obj;
			obj.reserve(num);
			for(long i=0;i<num;++i){
				obj.push_back( new TRI<float>(Data[i].V1, Data[i].V2, Data[i].V3, 0) );
			}
			return obj;
		}
	public:
		long num;
		char hdr[80];
		D1<STLEach> Data;
	};	
}

#endif
