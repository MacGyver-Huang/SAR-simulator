//
//  mesh.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 1/29/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef mesh_h
#define mesh_h

#include <cmath>
#include <basic/vec.h>
#include <basic/d1.h>
#include <basic/d2.h>
#include <basic/def_func.h>
#include <mesh/stl.h>
#include <vector>

using namespace vec;
using namespace d1;
using namespace d2;
using namespace def_func;
using namespace stl;
using namespace std;

namespace mesh {
	
	//=============================================================================
	//   Base Mesh
	//=============================================================================
	template<typename T>
	class MeshBase {
	public:
		D1<long> GetPolyLong(){
			return D1<long>(poly.GetPtr(),poly.GetM()*poly.GetN());
		}
		void Print(){
			cout<<"nPy = "<<nPy<<endl;
			cout<<"nVx = "<<nVx<<endl;
			cout<<"CC:"<<endl;
			for(long i=0;i<nPy;++i){
				cout<<"  ";
				CC[i].Print();
			}
			cout<<"N :"<<endl;
			for(long i=0;i<nPy;++i){
				cout<<"  ";
				N[i].Print();
			}
			cout<<"Area = "<<endl;
			for(long i=0;i<nPy;++i){
				cout<<"  ";
				cout<<Area[i]<<endl;
			}
		}
	public:
		long nPy;	// # of polygon
		long nVx;	// # of vertex
		D1<VEC<T> > vert;
		D2<long> poly;
		D1<T> Area;
		D1<VEC<T> > CC;
		D1<VEC<T> > N;
	};
	
	//=============================================================================
	//   Triangular Mesh
	//=============================================================================
	template<typename T>
	class MeshTri : public MeshBase<T> {
	public:
		MeshTri():W(),L(),nL(),nW(){};
		MeshTri(const double L, const double W, const long nL, const long nW)
				:L(L),W(W),nL(nL),nW(nW){
			// Triangle of Plate
			// plate mesh generation
			MeshBase<T>::nVx = (nL+1)*(nW+1);
			MeshBase<T>::nPy = nL*nW*2;
			
			double dL = L/(double)nL;
			double dW = W/(double)nW;
			
			// vertex
			MeshBase<T>::vert = D1<VEC<T> >(MeshBase<T>::nVx);
			long count = 0;
			for(long j=0;j<nW+1;++j){
				for(long i=0;i<nL+1;++i){
					MeshBase<T>::vert[count++] = VEC<T>(dL*i,dW*j,0.f);
				}
			}
			
			// polygon (connection)
			MeshBase<T>::poly = D2<long>(MeshBase<T>::nPy,4);
			count = 0;
			for(long j=0;j<nW;++j){
				for(long i=0;i<nL;++i){
					// Up triangle
					MeshBase<T>::poly[count][0] = (long)3;
					MeshBase<T>::poly[count][1] = i+j*(nL+1);
					MeshBase<T>::poly[count][2] = i+j*(nL+1)+1;
					MeshBase<T>::poly[count][3] = i+(j+1)*(nL+1)+1;
					count++;
					// Down triangle
					MeshBase<T>::poly[count][0] = (long)3;
					MeshBase<T>::poly[count][1] = i+j*(nL+1);
					MeshBase<T>::poly[count][2] = i+(j+1)*(nL+1)+1;
					MeshBase<T>::poly[count][3] = i+(j+1)*(nL+1);
					count++;
				}
			}
			
			// Area & CC & N
			MeshBase<T>::Area = D1<T>(MeshBase<T>::nPy);
			MeshBase<T>::CC   = D1<VEC<T> >(MeshBase<T>::nPy);
			MeshBase<T>::N    = D1<VEC<T> >(MeshBase<T>::nPy);
			for(long i=0;i<MeshBase<T>::nPy;++i){
				VEC<T> P0 = MeshBase<T>::vert[MeshBase<T>::poly[i][1]];
				VEC<T> P1 = MeshBase<T>::vert[MeshBase<T>::poly[i][2]];
				VEC<T> P2 = MeshBase<T>::vert[MeshBase<T>::poly[i][3]];
				D1<T> EdgeLength(3);
				EdgeLength[0] = (P1-P0).abs();
				EdgeLength[1] = (P2-P1).abs();
				EdgeLength[2] = (P0-P2).abs();
				T s = total( EdgeLength )/2;
				MeshBase<T>::Area[i] = sqrt( s * (s-EdgeLength[0]) * (s-EdgeLength[1]) * (s-EdgeLength[2]) );
				// CC
				MeshBase<T>::CC[i] = (P0+P1+P2)/3;
				// N
				MeshBase<T>::N[i]  = Unit( cross(P1-P0, P2-P1) );
			}
		}
		// Member functions
		void UpdateNCC(){
			// Update CC, N
			for(long i=0;i<MeshBase<T>::nPy;++i){
				VEC<T> P0 = MeshBase<T>::vert[MeshBase<T>::poly[i][1]];
				VEC<T> P1 = MeshBase<T>::vert[MeshBase<T>::poly[i][2]];
				VEC<T> P2 = MeshBase<T>::vert[MeshBase<T>::poly[i][3]];
				// CC
				MeshBase<T>::CC[i] = (P0+P1+P2)/3;
				// N
				MeshBase<T>::N[i]  = Unit( cross(P1-P0, P2-P1) );
			}
		}
		void Rx(const double rad){
			D2<double> M0 = def_func::Rx(rad);
			D2<T> M(M0.GetM(),M0.GetN());
			for(long j=0;j<M.GetN();++j){
				for(long i=0;i<M.GetM();++i){
					M[i][j] = (T)M0[i][j];
				}
			}
			// vert
			for(long i=0;i<MeshBase<T>::nVx;++i){
				MeshBase<T>::vert[i] = D22VEC(M * VEC2D2(MeshBase<T>::vert[i], 3, 1));
			}
			// Update N, CC
			UpdateNCC();
		}
		void Ry(const double rad){
			D2<double> M0 = def_func::Ry(rad);
			D2<T> M(M0.GetM(),M0.GetN());
			for(long j=0;j<M.GetN();++j){
				for(long i=0;i<M.GetM();++i){
					M[i][j] = (T)M0[i][j];
				}
			}
			// vert
			for(long i=0;i<MeshBase<T>::nVx;++i){
				MeshBase<T>::vert[i] = D22VEC(M * VEC2D2(MeshBase<T>::vert[i], 3, 1));
			}
			// Update N, CC
			UpdateNCC();
		}
		void Rz(const double rad){
			D2<double> M0 = def_func::Rz(rad);
			D2<T> M(M0.GetM(),M0.GetN());
			for(long j=0;j<M.GetN();++j){
				for(long i=0;i<M.GetM();++i){
					M[i][j] = (T)M0[i][j];
				}
			}
			// vert
			for(long i=0;i<MeshBase<T>::nVx;++i){
				MeshBase<T>::vert[i] = D22VEC(M * VEC2D2(MeshBase<T>::vert[i], 3, 1));
			}
			// Update N, CC
			UpdateNCC();
		}
		void Translate(const VEC<T>& move){
			// vert
			for(long i=0;i<MeshBase<T>::nVx;++i){
				MeshBase<T>::vert[i] += move;
			}
			// CC
			for(long i=0;i<MeshBase<T>::nPy;++i){
				MeshBase<T>::CC[i] += move;
			}
		}
		void ReDirection(const VEC<T>& Ps, const VEC<T>& Pt){
			VEC<T> uv = Unit(Ps - Pt);
			double theta_y = deg2rad(90.) - vec::angle(uv, VEC<T>(0,0,1));	// Elevation angle (from XY-plane)
			double theta_z = acos(uv.x()/cos(theta_y));						// Azimuth angle (from X-Axis)
			// Rotation with Y-Axis(Elevation) & Z-Axis(Azimuth)
			Ry(-theta_y);
			Rz(theta_z);
//			Translate(Pt);
		}
//		void ReDirection(const VEC<T>& dir, const VEC<T>& dir_old, const VEC<T> Pt){
//			VEC<T> uv = cross(dir_old, dir);
//			double total = uv.x() + uv.y() + uv.z();
//			double theta;
//			
//			if(abs(total - 0) < 1E-20){ // total -> 0
//				VEC<T> N2 = dir_old + 1E-100;
//				uv = cross(N2, dir);
//				theta = vec::angle(N2, dir);
//			}else{
//				theta = vec::angle(dir_old, dir);
//			}
//			RotArbitrary(uv, Pt, theta);
//		}
		void RotArbitrary(const VEC<T>& uv, const VEC<T>& Pt, const double Theta){
			vector<double> theta;
			theta.push_back(Theta);
			
			if(abs(theta[0]) > deg2rad(90.)){
				double tmp = theta[0];
				theta[0] = sign(theta[0]) * deg2rad(90.);
				theta.push_back( sign(theta[0]) * (std::abs(tmp - deg2rad(90.))) );
			}
			// vert
			for(long i=0;i<MeshBase<T>::nVx;++i){
				MeshBase<T>::vert[i] -= Pt;
				for(long j=0;j<theta.size();++j){
					MeshBase<T>::vert[i] = vec::find::ArbitraryRotate(MeshBase<T>::vert[i], theta[j], uv);
				}
				MeshBase<T>::vert[i] += Pt;
			}
			// CC
			for(long i=0;i<MeshBase<T>::nPy;++i){
				MeshBase<T>::CC[i] -= Pt;
				for(long j=0;j<theta.size();++j){
					MeshBase<T>::CC[i] = vec::find::ArbitraryRotate(MeshBase<T>::CC[i], theta[j], uv);
				}
				MeshBase<T>::CC[i] += Pt;
			}
			// N
			for(long i=0;i<MeshBase<T>::nPy;++i){
				for(long j=0;j<theta.size();++j){
					MeshBase<T>::N[i] = vec::find::ArbitraryRotate(MeshBase<T>::N[i], theta[j], uv);
				}
				
			}
		}
		vector<Obj*> Convert2Obj(){
			return GetSTL().Convert2Obj();
		}
		template<typename T2>
		friend const MeshTri<T2> operator+(const MeshTri<T2>& L,const MeshTri<T2>& R){
		
			MeshTri<T2> out;
			out.nVx = L.nVx + R.nVx;
			out.nPy = L.nPy + R.nPy;
			
			out.vert = D1<VEC<T2> >(out.nVx);
			out.poly = D2<long>(out.nPy,4);
			out.CC   = D1<VEC<T2> >(out.nPy);
			out.N    = D1<VEC<T2> >(out.nPy);
			out.Area = D1<double>(out.nPy);
			
			// vert
			for(long i=0;i<L.nVx;++i){
				out.vert[i] = L.vert[i];
			}
			for(long i=0;i<R.nVx;++i){
				out.vert[L.nVx+i] = R.vert[i];
			}
			// poly
			for(long i=0;i<L.nPy;++i){
				for(long j=0;j<4;++j){
					out.poly[i][j] = L.poly[i][j];
				}
			}
			for(long i=0;i<R.nPy;++i){
				out.poly[L.nPy-1+i][0] = 3;
				for(long j=1;j<4;++j){
					out.poly[L.nPy+i][j] = R.poly[i][j] + L.nVx;
				}
			}
			// N & CC
			for(long i=0;i<L.nPy;++i){
				out.N [i] = L.N [i];
				out.CC[i] = L.CC[i];
			}
			for(long i=0;i<R.nPy;++i){
				out.N [L.nPy+i] = R.N [i];
				out.CC[L.nPy+i] = R.CC[i];
			}
			// Area
			for(long i=0;i<out.nPy;++i){
				VEC<T> P0 = out.vert[out.poly[i][1]];
				VEC<T> P1 = out.vert[out.poly[i][2]];
				VEC<T> P2 = out.vert[out.poly[i][3]];
				D1<T> EdgeLength(3);
				EdgeLength[0] = (P1-P0).abs();
				EdgeLength[1] = (P2-P1).abs();
				EdgeLength[2] = (P0-P2).abs();
				T s = total( EdgeLength )/2;
				out.Area[i] = sqrt( s * (s-EdgeLength[0]) * (s-EdgeLength[1]) * (s-EdgeLength[2]) );
			}
			return out;
		}
		STL GetSTL(){// Wrtie to STL Binary (ONLY support triangle mesh)
			stl::STL cad(MeshBase<T>::nPy);
			
			VEC<float> V1, V2, V3;
			for(long i=0;i<MeshBase<T>::nPy;++i){
				D1<long> idx = MeshBase<T>::poly.GetRow(i);
				V1 = VEC<float>(MeshBase<T>::vert[idx[1]]);
				V2 = VEC<float>(MeshBase<T>::vert[idx[2]]);
				V3 = VEC<float>(MeshBase<T>::vert[idx[3]]);
				cad[i].N  = Unit(cross(V2-V1, V3-V1));
				cad[i].V1 = V1;
				cad[i].V2 = V2;
				cad[i].V3 = V3;
			}
			return cad;
		}
	public:
		double L, W;
		long nL, nW;
	};
	
	//=============================================================================
	//   Grid Mesh
	//=============================================================================
	template<typename T>
	class MeshGrid : public MeshBase<T> {
	public:
		MeshGrid(){};
		MeshGrid(const D1<T>& x, const D1<T>& y, const D2<T>& z){		// Grid of Plate
			// vert
			MeshBase<T>::vert = D1<VEC<T> >(x.GetNum() * y.GetNum());
			MeshBase<T>::nVx = 0;
			for(long j=0;j<y.GetNum();++j){
				for(long i=0;i<x.GetNum();++i){
					MeshBase<T>::vert[MeshBase<T>::nVx++] = VEC<T>(x[i],y[j],z[i][j]);
				}
			}
			// poly
			MeshBase<T>::nPy  = (x.GetNum()-1) * (y.GetNum()-1);
			MeshBase<T>::poly = D2<long>(5,MeshBase<T>::nPy);
			long count = 0;
			for(long j=0;j<y.GetNum()-1;++j){
				for(long i=0;i<x.GetNum()-1;++i){
					long v0 = j*x.GetNum()+i;
					long v1 = v0 + 1;
					long v3 = (j+1)*x.GetNum()+i;
					long v2 = v3 + 1;
					MeshBase<T>::poly[0][count] = (long)4;
					MeshBase<T>::poly[1][count] = v0;
					MeshBase<T>::poly[2][count] = v1;
					MeshBase<T>::poly[3][count] = v2;
					MeshBase<T>::poly[4][count] = v3;
					count++;
				}
			}
			// Area & CC & N
			MeshBase<T>::Area = D1<T>(MeshBase<T>::nPy);
			MeshBase<T>::CC   = D1<VEC<T> >(MeshBase<T>::nPy);
			MeshBase<T>::N    = D1<VEC<T> >(MeshBase<T>::nPy);
			for(long i=0;i<MeshBase<T>::nPy;++i){
				VEC<T> P0 = MeshBase<T>::vert[MeshBase<T>::poly[1][i]];
				VEC<T> P1 = MeshBase<T>::vert[MeshBase<T>::poly[2][i]];
				VEC<T> P2 = MeshBase<T>::vert[MeshBase<T>::poly[3][i]];
				VEC<T> P3 = MeshBase<T>::vert[MeshBase<T>::poly[4][i]];
				MeshBase<T>::Area[i] = (P1-P0).abs() * (P2-P1).abs();
				// CC
				MeshBase<T>::CC[i] = (P0+P1+P2+P3)/4;
				// N
				MeshBase<T>::N[i]  = Unit( cross(P1-P0, P2-P1) );
			}
			
		}
	};
	
	
	//=============================================================================
	//   Dihedral Mesh
	//=============================================================================
	template<typename T>
	class MeshDihedral {
	public:
		MeshDihedral(){
			Init(1.0, 1.0, 1, 1, VEC<T>(0,0,0));
		}
		MeshDihedral(const double L, const double W, const long nL, const long nW, const double RotAngle=0){
			Init(L, W, nL, nW, VEC<T>(0,0,0));
			if(RotAngle != 0){
				Rx(RotAngle);
			}
		}
		MeshDihedral(const double L, const double W, const long nL, const long nW, const VEC<T> O, const double RotAngle=0){
			Init(L, W, nL, nW, O);
			if(RotAngle != 0){
				Rx(RotAngle);
			}
			Translate(O);
		}
		// Member functions
		void Rx(const double rad){
			mesh.Rx(rad);
		}
		void Ry(const double rad){
			mesh.Ry(rad);
		}
		void Rz(const double rad){
			mesh.Rz(rad);
		}
		void Translate(const VEC<T>& move){
			mesh.Translate(move);
			o += move;
		}
		void ReDirection(const VEC<T>& Ps, const VEC<T>& Pt){

			VEC<T> uv = Unit(Ps - Pt);
			double theta_y = deg2rad(90.) - vec::angle(uv, VEC<T>(0,0,1));	// Elevation angle (from XY-plane)
			double theta_z = acos(uv.x()/cos(theta_y));						// Azimuth angle (from X-Axis)
			// Rotation with Y-Axis(Elevation) & Z-Axis(Azimuth)
			Ry(-theta_y);
			Rz(theta_z);
			Translate(Pt);
		}
		STL GetSTL(){
			return mesh.GetSTL();
		}
		void Write2STLBinary(const string file){
			mesh.GetSTL().WriteBinary(file);
		}
		double GetL(){ return mesh.L; }
		double GetW(){ return mesh.W; }
	private:
		void Init(const double L, const double W, const long nL, const long nW, const VEC<T>& O){
			// Construction Dihedral
			MeshTri<T> P1(L,W,nL,nW); // on XY plane
			P1.Rx(deg2rad(90));
			P1.Rz(deg2rad(45));
			MeshTri<T> P2(L,W,nL,nW); // on XY plane
			P2.Rx(deg2rad(-90));
			P2.Translate(VEC<T>(0,0,W));
			P2.Rz(deg2rad(-45));
			mesh = P1 + P2;
			mesh.L = L;
			mesh.W = W;
			mesh.nL = nL;
			mesh.nW = nW;
			Translate(VEC<T>(0,0,-W/2));
			o = O;
		}
	public:
		VEC<T> o;
		MeshTri<T> mesh;
	};
	
//	//=============================================================================
//	//   Plate Mesh
//	//=============================================================================
//	template<typename T>
//	class MeshPlate {
//	public:
//		MeshPlate(){}
//	private:
//		void Init(const double L, const double W){
//			// Construction plate
//			MeshTri<T> P1(L,W,nL,nW); // on XY plane
//			P1.Rx(deg2rad(90));
//			P1.Rz(deg2rad(45));
//		}
//	public:
//		MeshTri<T> mesh;
//	};

	
	//=============================================================================
	//   New Inicident Mesh
	//=============================================================================
	class MeshInc{
	public:
		MeshInc():Scale(0),dRad(0),lambda(0),LH(0),LV(0),nPy(0),Rad(0),dLW(0),Area(0){};
		MeshInc(const double SCALE, const double Lambda, const BVH& bvh, const VEC<double>& PS, const VEC<double>& PT, const double dRAD){
			Scale  = SCALE;
			dRad   = dRAD;
			lambda = Lambda;
			Ps = PS - PT;
			
			// Ref. direction vector
			VEC<double> Z(0,0,1);
			if(rad2deg(angle(Ps, Z)) < 1E-7){
				dirH = VEC<double>(0,-1,0);	// H-ref direction
			}else{
				dirH = Unit(cross(Ps, Z));	// H-ref direction
			}
			dirV = Unit(cross(dirH, Ps));	// V-ref direction
			// Calculate suitable dLW
			VEC<double> O(0,0,0);
			double dP = lambda/SCALE;
			double Rad_org = (Ps - O).abs();
			// Small angle of each cell, both in H & V direction
			dLW = 2 * atan((dP/2)/Rad_org);
			
			// Direction unit vector from Ps to [0,0,0]
			dir = Unit(O - Ps);
			
			// sensor to curve surface distance
			//
			//                 +--- Pseudo incident surface
			//                 V
			//                 |
			// Ps              |                          Pt
			//  +--------------+---------------------------+
			//                 |
			//  <------------->|<-------------------------->
			//       dRad      |            Rad
			//  <------------------------------------------>
			//                    Rad_org

			if(Rad_org < dRad){
				cerr<<"ERROR::The Ps & Pt distance("<<Rad_org<<") is smaller than the dRad("<<dRad<<")"<<endl;
				exit(EXIT_FAILURE);
			}
			Rad = Rad_org - dRad;
			
			// intersection point form LOS(main beam) to incident surface
			PLOS = Ps + dir * Rad;
			
			// LOS unit vector
			NN = Unit(PLOS);

			
			// Find angle region suit for object
			VEC<double> NH = dirV;	// normal vector of H-plane
			VEC<double> NV = dirH;	// normal vector of V-plane
			
			// extract 8 points of BVH
			BVHFlatNode* node = bvh.GetflatTree();
			VEC<double> Min = VEC<double>(node->bbox.min);
			VEC<double> Max = VEC<double>(node->bbox.max);

			// Calcuate outline frame
			D1<VEC<double> > overt(8);
			overt[0].x() = Min.x(); overt[0].y() = Min.y(); overt[0].z() = Min.z();
			overt[1].x() = Min.x(); overt[1].y() = Min.y(); overt[1].z() = Max.z();
			overt[2].x() = Min.x(); overt[2].y() = Max.y(); overt[2].z() = Max.z();
			overt[3].x() = Min.x(); overt[3].y() = Max.y(); overt[3].z() = Min.z();
			overt[4].x() = Max.x(); overt[4].y() = Min.y(); overt[4].z() = Min.z();
			overt[5].x() = Max.x(); overt[5].y() = Min.y(); overt[5].z() = Max.z();
			overt[6].x() = Max.x(); overt[6].y() = Max.y(); overt[6].z() = Max.z();
			overt[7].x() = Max.x(); overt[7].y() = Max.y(); overt[7].z() = Min.z();
			
			
//			double void_dis;
			VEC<double> PH0, PV0, PH, PV;
			D1<double> AngleH(8), AngleV(8);
			
			for(int i=0;i<8;++i){
				// Intersection POINT to H-plane
				vec::find::MinDistanceFromPointToPlane(NH, Ps, overt[i], PH0);
				vec::find::MinDistanceFromPointToPlane(NN, PLOS, PH0, PH);
				// Intersection POINT to V-plane
				vec::find::MinDistanceFromPointToPlane(NV, Ps, overt[i], PV0);
				vec::find::MinDistanceFromPointToPlane(NN, PLOS, PV0, PV);
				// Pre-processing
				VEC<double> Ps_O  = O  - Ps;
				VEC<double> Ps_PH = PH - Ps;
				VEC<double> Ps_PV = PV - Ps;
				VEC<double> Ps_PLOS = PLOS - Ps;
				VEC<double> PLOS_PH = PH - PLOS;
				VEC<double> PLOS_PV = PV - PLOS;
				// Angle range at H-plane w.r.t LOS (main beam) on dRad plane
				double sigH  = sign(dot(Unit(NH), Unit(cross(Ps_O, Ps_PH))));
				if( PLOS_PH.abs() < 1E-5 ){
					AngleH[i] = 0;
				}else{
					// Note: Replace the AngleH[i] = sigH * abs(vec::angle(Ps_O, Ps_PH)) to be below
					//       Because if the (Ps_O, Ps_PH) angle is too small the error will be increasing,
					//       we replace the vecangle to combine the two edge and find the residual angle (180 - ang1 - ang2)
					double ang1 = vec::angle(Ps_PLOS, PLOS_PH);
					double ang2 = vec::angle(Ps_PH,   PLOS_PH);
					AngleH[i] = sigH * (def::PI - ang1 - ang2);
				}
				// Angle range at V-plane w.r.t LOS (main beam) on dRad plane
				double sigV  = sign(dot(Unit(NV), Unit(cross(Ps_O, Ps_PV))));
				if( PLOS_PV.abs() < 1E-5 ){
					AngleV[i] = 0;
				}else{
					// Note: Replace the AngleH[i] = sigH * abs(vec::angle(Ps_O, Ps_PH)) to be below
					//       Because if the (Ps_O, Ps_PH) angle is too small the error will be increasing,
					//       we replace the vecangle to combine the two edge and find the residual angle (180 - ang1 - ang2)
					double ang1 = vec::angle(Ps_PLOS, PLOS_PV);
					double ang2 = vec::angle(Ps_PV,   PLOS_PV);
					AngleV[i] = sigV * (def::PI - ang1 - ang2);
				}
				// Nan/Inf detection
				if(std::isnan(AngleH[i])){ AngleH[i] = 0; }
				if(std::isnan(AngleV[i])){ AngleV[i] = 0; }


#ifdef DEBUG
				if(i==0){
					printf("ooooooooooooooo i=%d oooooooooooooooooooo\n",i);
					printf("overt[i] : "); overt[i].Print();
					printf("PH0      : "); PH0.Print();
					printf("PV0      : "); PV0.Print();
					printf("PH       : "); PH.Print();
					printf("PV       : "); PV.Print();
					printf("PLOS     : "); PLOS.Print();
					printf("Ps       : "); Ps.Print();
					printf("Rad = %.15f\n", Rad);
					printf("dis(PH-PLOS) = %.15f\n", (PH-PLOS).abs());
					printf("angle(Ps_PLOS, Ps_PH) = %.15f (%.15f deg)\n", vec::angle(PLOS-Ps, PH-Ps), rad2deg(vec::angle(PLOS-Ps, PH-Ps)));
					printf("dis(Ps_PLOS) = %.15f, dis(PLOS_PH) = %.15f\n", (PLOS-Ps).abs(), (PH-PLOS).abs());
					printf("atan((PH-PLOS).abs()/Rad) = %.15f (%.15f deg)\n", atan((PH-PLOS).abs()/Rad), rad2deg(atan((PH-PLOS).abs()/Rad)));
					printf("dis1 = %.15f, dis2 = %.15f\n", Rad*tan(vec::angle(PLOS-Ps, PH-Ps)), Rad*tan(atan((PH-PLOS).abs()/Rad)));
					printf("AngleH[i] = %.15f, AngleV[i] = %.15f\n\n", AngleH[i], AngleV[i]);
				}
#endif
			}

			// find min & max angle at H/V- plane
			AngH = vector<double>(2);
			AngV = vector<double>(2);
			AngH[0] = min(AngleH) - dLW;	AngH[1] = max(AngleH) + dLW;
			AngV[0] = min(AngleV) - dLW;	AngV[1] = max(AngleV) + dLW;
			
			// Cell number at each edge
			LH = ceil(abs(AngH[1] - AngH[0]) / dLW);
			LV = ceil(abs(AngV[1] - AngV[0]) / dLW);
						
			// Number of polygon
			nPy = LH * LV;

#ifdef DEBUG
			printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
			printf("dP = %.15f, dLW = %.15f, Rad = %.15f\n", dP, dLW, Rad);
			printf("NH : "); NH.Print();
			printf("NV : "); NV.Print();
			printf("Min : "); Min.Print();
			printf("Max : "); Max.Print();
			printf("min(AngleH) = %.15f, max(AngleH) = %.15f\n", min(AngleH), max(AngleH));
			printf("min(AngleV) = %.15f, max(AngleV) = %.15f\n", min(AngleV), max(AngleV));
			printf("LH = %ld(%.15f), LV = %ld(%.15f)\n", LH, abs(AngH[1] - AngH[0]) / dLW, LV, abs(AngV[1] - AngV[0]) / dLW);
			printf("Edge length = %.15f\n", 2*Rad*tan(dLW/2));
			printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
#endif

			// Distance of cell generation at H-drection
			double angleH;
			disH = D1<double>(LH);
			dH   = D1<double>(LH);
			for(long i=0;i<LH;++i){
				// H cell's angle
				angleH = AngH[0] + dLW*i;
				// distance from point to V-plane
				disH[i] = Rad * sin(angleH);
				dH[i]   = Rad - Rad * cos(angleH);
			}
			// Distance of cell generation at V-drection
			double angleV;
			disV = D1<double>(LV);
			dV   = D1<double>(LV);
			for(long j=0;j<LV;++j){
				// V cell's angle
				angleV = AngV[0] + dLW*j;
				// distance from point to H-plane
				disV[j] = Rad * sin(angleV);
				dV[j]   = Rad - Rad * cos(angleV);
			}

			// Add
			dirH_disH = D1<VEC<double> >(LH);
			dirV_disV = D1<VEC<double> >(LV);
			for(long i=0;i<LH;++i){
				dirH_disH[i] = dirH*disH[i] - dH[i] * NN + PLOS;
			}
			for(long j=0;j<LV;++j){
				dirV_disV[j] = dirV*disV[j] + dV[j] * NN ;
			}
			// Area
			Area = 2*Rad*tan(dLW/2) * 2*Rad*tan(dLW/2);	// Area
		}
		void GetCell(const long k, Ray& ray, double& area){
			long i = k % LH;	// H index
			long j = k/LH;		// V index
			
//			VEC<double> CC = PLOS + dirH*disH[i] - dH[i]*NN - dirV*disV[j] - dV[j]*NN;	// temp point
			VEC<double> CC = dirH_disH[i] - dirV_disV[j];	// temp point
			VEC<double> N  = Unit(CC - Ps);					// Normal vector
//			VEC<float> N = Unit(VEC<float>(CC - Ps));
			
//			VEC<double> PTemp = PLOS + dirH*disH[i] - dirV*disV[j];	// temp point
//			VEC<double> N    = Unit(PTemp - Ps);				// Normal vector
//			VEC<double> CC   = Ps + Rad * N;					// Center
			
			
			ray  = Ray(CC, N);
			area = Area;
		}
//		void GetCell(const long k, VEC<double>& N, VEC<double>& CC, double& area){
//			long i = k % LH;	// H index
//			long j = k/LH;		// V index
//			// combine
////			N.x() = PLOS.x() + dirH.x() * disH[i] - dirV.x() * disV[j] - Ps.x();
////			N.y() = PLOS.y() + dirH.y() * disH[i] - dirV.y() * disV[j] - Ps.y();
////			N.z() = PLOS.z() + dirH.z() * disH[i] - dirV.z() * disV[j] - Ps.z();
////			double inv = 1.0/N.abs();
////			N.x() = N.x() * inv;
////			N.y() = N.y() * inv;
////			N.z() = N.z() * inv;
//			VEC<double> PTemp = PLOS + dirH*disH[i] - dirV*disV[j];	// temp point
//////			VEC<double> N1    = Unit(PTemp - Ps);				// Normal vector
////			N    = PTemp - Ps;
////			N.Unit();								// Normal vector
////			
//////			cout<<"N1 = "; N1.Print();
//////			cout<<"N  = "; N.Print();
//			N     = Unit(PTemp - Ps);				// Normal vector
//			
//			CC   = Ps + Rad * N;					// Center
//			area = Area;
//		}
		void Print(){
			cout<<std::setprecision(10);
			cout<<"+------------------------------------+"<<endl;
			cout<<"|          MeshInc Summary           |"<<endl;
			cout<<"+------------------------------------+"<<endl;
			cout<<"nPy        = "<<nPy<<endl;
			cout<<"Rad    [m] = "<<Ps.abs()<<endl;
			cout<<"dRad   [m] = "<<dRad<<endl;
			cout<<"AngH [deg] = "<<rad2deg(abs(AngH[1]-AngH[0]))<<endl;
			cout<<"AngV [deg] = "<<rad2deg(abs(AngV[1]-AngV[0]))<<endl;
			cout<<"LH     [#] = "<<LH<<endl;
			cout<<"LV     [#] = "<<LV<<endl;
			cout<<"dLW  [deg] = "<<rad2deg(dLW)<<endl;
		}
		void PrintSimple(){
			cout<<std::setprecision(10);
			cout<<"+------------------------------------+"<<endl;
			cout<<"|          MeshInc Summary           |"<<endl;
			cout<<"+------------------------------------+"<<endl;
			cout<<"nPy        = "<<nPy<<endl;
			cout<<"Rad    [m] = "<<Ps.abs()<<endl;
			cout<<"dRad   [m] = "<<dRad<<endl;
		}
		void Print2(const long N=1){
			cout<<std::setprecision(10);
			cout<<"+------------------------------------+"<<endl;
			cout<<"|          MeshInc Summary           |"<<endl;
			cout<<"+------------------------------------+"<<endl;
			cout<<"Rad    [m] = "<<Ps.abs()<<endl;
			cout<<"LH     [#] = "<<LH<<endl;
			cout<<"Area  [m2] = "<<Area<<endl;
			cout<<"+             "<<endl;
			cout<<"   dirH - disH"<<endl;
			cout<<"+             "<<endl;
			for(int i=0;i<N;++i){
				dirH_disH[i].Print();
			}
			cout<<"+             "<<endl;
			cout<<"   dirV - disV"<<endl;
			cout<<"+             "<<endl;
			for(int i=0;i<N;++i){
				dirV_disV[i].Print();
			}
		}
	public:
		// Target *MUST* be [0,0,0]
		double Scale;		// [x] Mesh scale vector
		double dRad;		//
		double lambda;		// [m] Wavelength
		long LH;			// [#] H direction samples
		long LV;			// [#] V direction samples
		long nPy;			// [#] Number of polygon (#ray)
		VEC<double> Ps;		// [m,m,m] Sensor position
		double Rad;			// [m] Distance from sensor to surface of incident curve plane
		VEC<double> dir;	// [m,m,m] Direction from sensor to target's origin [0,0,0]
		vector<double> AngH;	// [rad] H direction angle region
		vector<double> AngV;	// [rad] V direction angle region
		VEC<double> dirH;	// [m,m,m] H plane normal vector
		VEC<double> dirV;	// [m,m,m] V plane normal vector
		double dLW;			// [rad] Cell angle interval
		VEC<double> PLOS;	// intersection point form LOS(main beam) to incident surface
		D1<double> disH;	// [m] Distance of cell at H direction
		D1<double> disV;	// [m] Distance of cell at V direction
		// add
		VEC<double> NN;
		D1<double> dH;
		D1<double> dV;
		D1<VEC<double> > dirH_disH;
		D1<VEC<double> > dirV_disV;
		double Area;		// [m^2] Area of each cell
	};
	
}



#endif
