//
//  test.h
//  PhysicalOptics14
//
//  Created by Steve Chiang on 5/8/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef test_h
#define test_h


#include <rcs/ems.h>
#include <cuda/cumain.cuh>

#ifdef OPENCV
#include "new_cv.h"
#endif

using namespace ems;


namespace test{
	// Test for Dihedral
	class Di {
	public:
		Di():L(1),W(0),ROT(0){};
		Di(const SAR& Sar, const EF& Ef, const MeshDef& mesh, const MaterialDB& MatDB,
			   const double L, const double W):
		Sar(Sar),Ef(Ef),mesh(mesh),MatDB(MatDB),L(L),W(W),ROT(0){
		}
		// Misc.
		double RCS2dB(const double rcs){
			double lambda = def::C/Sar.f0();
			double dB;
			
			dB = 4*PI/(lambda*lambda) * (rcs * rcs);
			if(abs(dB) < 1E-20){
				dB = 1E-2;
			}
			dB = 10*log10(dB);
			
			return dB;
		}
		D1<double> RCS2dB(const D1<double>& rcs){
			double lambda = def::C/Sar.f0();
			D1<double> dB(rcs.GetNum());
			double factor = 4*PI/(lambda*lambda);
			for(long i=0;i<rcs.GetNum();++i){
				dB[i] = factor * (rcs[i] * rcs[i]);
				if(abs(dB[i]) < 1E-20){
					dB[i] = 1E-2;
				}
				dB[i] = 10*log10(dB[i]);
			}
			return dB;
		}
		double GetMaxH(){
			D1<double> rcs_H = abs(res_H);
			return max(rcs_H);
		}
		double GetMaxV(){
			D1<double> rcs_V = abs(res_V);
			return max(rcs_V);
		}
		double GetMaxHdB(){
			D1<double> rcs_H = abs(res_H);
			return max(RCS2dB(rcs_H));
		}
		double GetMaxVdB(){
			D1<double> rcs_V = abs(res_V);
			return max(RCS2dB(rcs_V));
		}
		D1<double> GetMaxIdeal(){
			D1<double> out(2);
			out[0] = RCS2dB(L*W);
			out[1] = RCS2dB(sqrt(2.)*L*W);
			return out;
		}
		void DoIt(const long nL=1, const long nW=1, const bool IsGPU=false, const bool IsShow=false){

			string GPUName = cu::GetGPUName();
			dim3 NThread1, NThread2, NThread3, NThread4;
			cu::GetGPUParameters(NThread1, NThread2, NThread3, NThread4);


			double Rad = 1000;
			//		double Rad = 1000;
			VEC<double> PPs(600E4,0,0);	// sensor location
			VEC<double> PSc(0,0,0);		// scene center
//			VEC<double> PPs(-2816704.1234734389, 5608473.8766335594, 2832347.5953361401);	// sesnor location
//			VEC<double> PSc(-2909748.1413439866, 4996551.2089265306, 2683272.1304448019);	// scene center
			
			//
			// Object
			//
			MeshDihedral<double> Di(L, W, nL, nW, ROT);	// origin is (0,0,0)
			Di.ReDirection(PPs, PSc);					// origin is PSc
			Di.Translate(-PSc);							// *MUST* be shift to (0,0,0)
			
			// Convert to Obj* vector
			vector<Obj*> obj = Di.GetSTL().Convert2Obj();
			
			// BVH build
			BVH bvh(&obj, true);
			
			// Memory allocate
			res_H = D1<CPLX<double> >(phi.GetNum());
			res_V = D1<CPLX<double> >(phi.GetNum());
			SBR<double> sbr;
			vector<SAR> Sar2(1);
			
			if(IsGPU){
				size_t Nr = Sar2.size();
				size_t MaxLevel = Ef.MaxLevel();
				cu::cuSBRElement<float>* res = new cu::cuSBRElement<float>[Nr*MaxLevel];
				for(long i=0;i<res_H.GetNum();++i){
					double Phi = deg2rad(phi[i]);
					PPs = VEC<double>(Rad*cos(Phi), Rad*sin(Phi), 0);
					// Sphere surface
					MeshInc inc2(mesh.Scale(), Sar.lambda(), bvh, PPs, PSc, mesh.dRad());
					// SBR
					Sar2[0] = Sar;
					D1<double> k0(Sar2.size());
					for(size_t j=0;j<k0.GetNum();++j){
						k0[j] = def::PI2_C * Sar2[j].f0();
					}
					// CUDA SBR Do It
					cu::cuSBRDoIt(Sar2, k0, Ef, bvh, inc2, MatDB, 0, NThread1, NThread2, NThread3, res, true, false);


					// Re-initialize results
					res_H[i] = CPLX<double>(0,0);
					res_V[i] = CPLX<double>(0,0);
					// Only one element(one wavelength) in res pointer
					for(size_t p=0;p<MaxLevel;++p){
						cu::cuCPLX<float> cu_cxH = res[p].sump;
						cu::cuCPLX<float> cu_cxV = res[p].sumt;
						CPLX<double> cxH(cu_cxH.r, cu_cxH.i);
						CPLX<double> cxV(cu_cxV.r, cu_cxV.i);
						res_H[i] += cxH;
						res_V[i] += cxV;
					}
					// Show
					if(IsShow){
						double MaxH = res_H[i].abs();
						double MaxV = res_V[i].abs();
						double MaxVal = (MaxH > MaxV)? MaxH:MaxV;
						printf("%3ld, %3.0f, %.7f\n", i, phi[i], MaxVal);
					}

				}
				delete [] res;
			}else{
				for(long i=0;i<res_H.GetNum();++i){
					double Phi = deg2rad(phi[i]);
					PPs = VEC<double>(Rad*cos(Phi), Rad*sin(Phi), 0);
					// Sphere surface
					MeshInc inc2(mesh.Scale(), Sar.lambda(), bvh, PPs, PSc, mesh.dRad());
					// SBR
					Sar2[0] = Sar;
					sbr = SBR<double>(Sar2, Ef, bvh, inc2, MatDB, Ef.MaxLevel(), -mesh.dRad());
					sbr.DoIt(true);
					// result
					res_H[i] = total(sbr.GetSBRElement(0).GetSump());
					res_V[i] = total(sbr.GetSBRElement(0).GetSumt());
					// Show
					if(IsShow){
						double MaxH = res_H[i].abs();
						double MaxV = res_V[i].abs();
						double MaxVal = (MaxH > MaxV)? MaxH:MaxV;
						printf("%3ld, %3.0f, %.7f\n", i, phi[i], MaxVal);
					}
				}
			}
		}
		// Main function
		void SingleAsp(const double PHI, const double Rot, const bool IsGPU, const long nL=1, const long nW=1){
			phi = D1<double>(1);
			phi[0] = PHI;
			ROT = Rot;
			DoIt(nL, nW, IsGPU);
		}
		void MultiAsp(const D1<double>& PHI, const bool IsGPU, const bool IsShow, const long nL=1, const long nW=1){
			phi = PHI;
			ROT = 0;
			DoIt(nL, nW, IsGPU, IsShow);
		}
		// Self test
		void Test(const bool IsGPU=false, const bool IsShow=false, const string file_phi="", const string file_res_H="", const string file_res_V=""){
			//+===========================================================================+
			//|                                                                           |
			//|                         Single Dihedral Test                              |
			//|                                                                           |
			//+===========================================================================+
			Ef.SetTxPol("V");
			
			//-----------------------------------------------------------------------------
			// Single ASP
			//-----------------------------------------------------------------------------
			Di TD(Sar, Ef, mesh, MatDB, L, W);
			D2<double> Max(2,3);
			if(IsShow){
				cout<<"+--------------------------------------+"<<endl;
				cout<<"|  ROT = 0 [deg]                       |"<<endl;
				cout<<"+--------------------------------------+"<<endl;
			}else{
				cout<<"Rotation Test!";
			}
			TD.SingleAsp(0, deg2rad(0), IsGPU);
			Max[0][0] = TD.GetMaxHdB();
			Max[1][0] = TD.GetMaxVdB();
			if(IsShow){ TD.ShowResults(); }
			if(IsShow){
				cout<<"+--------------------------------------+"<<endl;
				cout<<"|  ROT = 22.5 [deg]                    |"<<endl;
				cout<<"+--------------------------------------+"<<endl;
			}
			TD.SingleAsp(0, deg2rad(22.5), IsGPU);
			Max[0][1] = TD.GetMaxHdB();
			Max[1][1] = TD.GetMaxVdB();
			if(IsShow){ TD.ShowResults(); }
			if(IsShow){
				cout<<"+--------------------------------------+"<<endl;
				cout<<"|  ROT = 45 [deg]                      |"<<endl;
				cout<<"+--------------------------------------+"<<endl;
			}
			TD.SingleAsp(0., deg2rad(45.), IsGPU);
			Max[0][2] = TD.GetMaxHdB();
			Max[1][2] = TD.GetMaxVdB();
			if(IsShow){ TD.ShowResults(); }
			
			// Summary
			string CheckRot, CheckPeak, CheckSide;
			double MaxIdealdB = TD.GetMaxIdeal()[1];
//			cout<<"Max[1][0]   : "<<Max[1][0]<<endl;
//			cout<<"Max[0][0]   : "<<Max[0][0]<<endl;
//			cout<<"Max[1][1]   : "<<Max[1][1]<<endl;
//			cout<<"Max[0][1]   : "<<Max[0][1]<<endl;
//			cout<<"Max[0][2]   : "<<Max[0][2]<<endl;
//			cout<<"Max[1][2]   : "<<Max[1][2]<<endl;
//			cout<<"double(abs(total(TD.GetMaxIdeal())))                           : "<<MaxIdealdB<<endl;
//			cout<<"(Max[1][0] - Max[0][0]) > double(abs(total(TD.GetMaxIdeal()))) : "<<((Max[1][0] - Max[0][0]) > MaxIdealdB)<<endl;
//			cout<<"(abs(Max[1][1] - Max[0][1])) < 0.05                            : "<<((abs(Max[1][1] - Max[0][1])) < 0.05)<<endl;
//			cout<<"(Max[0][2] - Max[1][2]) > double(abs(total(TD.GetMaxIdeal()))) : "<<((Max[0][2] - Max[1][2]) > MaxIdealdB)<<endl;
			// Rotation
			if((Max[1][0] - Max[0][0]) > MaxIdealdB &&
			   (abs(Max[1][1] - Max[0][1])) < 0.05 &&
			   (Max[0][2] - Max[1][2]) > MaxIdealdB ){
				CheckRot = "Ok";
			}else{
				CheckRot = "Failure";
			}
			if(!IsShow){
				cout<<" "<<CheckRot<<endl;
			}
			
			
			//-----------------------------------------------------------------------------
			// Multi ASP
			//-----------------------------------------------------------------------------
			if(IsShow){
				cout<<"+--------------------------------------+"<<endl;
				cout<<"|  Multi ASP = [-48, 48]/1 [deg]       |"<<endl;
				cout<<"+--------------------------------------+"<<endl;
			}else{
				cout<<"Multi-Angle Test!"<<endl;
			}
			Di TD2(Sar, Ef, mesh, MatDB, L, W);
			D1<double> phi = linspace(-48., 48., 1.);
			TD2.MultiAsp(phi, IsGPU, IsShow);
			if(IsShow){ TD2.ShowResults(); }
			if(file_phi != ""){
				cout<<"Write:"<<endl;
				cout<<"  file_phi   : "<<file_phi<<endl;
				cout<<"  file_res_H : "<<file_res_H<<endl;
				cout<<"  file_res_V : "<<file_res_V<<endl;
				TD2.WriteResults(file_phi, file_res_H, file_res_V);
			}
			
			// Summary
			double Max48 = TD2.RCS2dB(TD2.res_V[48].abs());
			double Max03 = TD2.RCS2dB(TD2.res_V[3].abs());
			double Max93 = TD2.RCS2dB(TD2.res_V[93].abs());
			double MaxLevel1 = abs(TD2.GetMaxIdeal()[0]);		// 45 deg ideal RCS
			double MaxLevel2 = abs(TD2.GetMaxIdeal()[1]);		// 0 deg ideal RCS
			
			CheckPeak = (abs(Max48 - MaxLevel2) < 0.2)? "Ok":"Failure";
			CheckSide = (abs(Max03 - MaxLevel1) < 0.4 && abs(Max93 - MaxLevel1) < 0.4)? "Ok":"Failure";
			
//			cout<<abs(Max48 - MaxLevel2)<<endl;
//			cout<<abs(Max03 - MaxLevel1)<<endl;
//			cout<<abs(Max93 - MaxLevel1)<<endl;
			
			if(IsShow){
				cout<<"+--------------------------+"<<endl;
				cout<<"|          Summary         |"<<endl;
				cout<<"+--------------------------+"<<endl;
				cout<<"Dihedral Rotation res. check is [ "<<CheckRot<<" ]"<<endl;
				cout<<"Multi-ASP 0  deg value check is [ "<<CheckPeak<<" ]"<<endl;
				cout<<"Multi-ASP 45 deg value check is [ "<<CheckSide<<" ]"<<endl;
			}else{
				cout<<"Multi-ASP 0  deg value check is [ "<<CheckPeak<<" ]"<<endl;
				cout<<"Multi-ASP 45 deg value check is [ "<<CheckSide<<" ]"<<endl;
			}
			
			//-----------------------------------------------------------------------------
			// OpenCV
			//-----------------------------------------------------------------------------
#ifdef OPENCV
			D1<double> rcs(TD2.res_V.GetNum());
			for(long i=0;i<rcs.GetNum();++i){
				rcs[i] = -TD2.res_V[i].abs();
			}
			new_cv::Plot(rcs);
#endif
		}
		// results
		void ShowResults(){
			D1<double> rcs_H  = abs(res_H);
			D1<double> rcs_V  = abs(res_V);
			double maxRCSH    = max(rcs_H);
			double maxRCSHdB  = max(RCS2dB(rcs_H));
			double maxRCSV    = max(rcs_V);
			double maxRCSVdB  = max(RCS2dB(rcs_V));
			double maxIdealdB = GetMaxIdeal()[1];
			cout<<"Tx : "<<Ef.TxPol()<<endl;
			cout<<"max RCS (H)        = "<<maxRCSH<<endl;
			cout<<"max RCS (V)        = "<<maxRCSV<<endl;
			cout<<"max RCS (H) [dB]   = "<<maxRCSHdB<<endl;
			cout<<"max RCS (V) [dB]   = "<<maxRCSVdB<<endl;
			cout<<"max Ideal RCS [dB] = "<<maxIdealdB<<endl;
		}
		void ShowResultsDetail(){
			double maxRCSIdea1 = RCS2dB(L*W);
			double maxRCSIdea2 = RCS2dB(sqrt(2.)*L*W);
			cout<<"+----------------------------+"<<endl;
			cout<<"|         Simulation         |"<<endl;
			cout<<"+----------------------------+"<<endl;
			ShowResults();
			cout<<"+----------------------------+"<<endl;
			cout<<"|           Ideal            |"<<endl;
			cout<<"+----------------------------+"<<endl;
			cout<<"max RCS Ideal (Level=1) [dB] = "<<maxRCSIdea1<<endl;
			cout<<"max RCS Ideal (Level=2) [dB] = "<<maxRCSIdea2<<endl;
		}
		void WriteResults(const string file_phi, const string file_res_H, const string file_res_V){
			phi.WriteASCII(file_phi.c_str());
			res_H.WriteASCII(file_res_H.c_str());
			res_V.WriteASCII(file_res_V.c_str());
		}
	private:
		SAR Sar;
		EF Ef;
		MeshDef mesh;
		MaterialDB MatDB;
		double L, W;
		D1<double> phi;
		double ROT;
	public:
		D1<CPLX<double> > res_H, res_V;
	};
}


#endif
