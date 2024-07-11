/*
 * cukernel.cuh
 *
 *  Created on: Oct 16, 2014
 *      Author: cychiang
 */

#ifndef CUSUB_CUH_
#define CUSUB_CUH_

#include <cmath>
#include <cuda/cumisc.cuh>
#include <cuda/cuvec.cuh>
#include <cuda/cuclass.cuh>
#include <cuda/cuopt.cuh>
#include <cuda/cusar.cuh>
#include <sar/sar.h>
#include <cuda_runtime.h>


namespace cu {

	//+======================================================+
	//|                   CUDA subroutine                    |
	//+======================================================+
	__global__
// #if __CUDA_ARCH__ == 350
// 	__launch_bounds__(128, 10)
// #endif
	void cuRayTracing(const size_t nPy, const size_t MaxLevel, cuBVH& bvh, cuMeshInc& inc, 						// input
					  size_t* MinLevel, double* DistSet, cuRay* RaySet, cuTRI<float>** ObjSet, bool* Shadow,	// output
					  const size_t OffsetPatch = 0, const bool SHOW=false){																	// keyword
		//---------------------------------------------------------------------------------------------------------+
		// Purpose:                                                                                                |
		//    Calculate "ALL wavelength" results and store it in the sbr with multu bouncing level.                |
		//    The 1 wavelength(lambda) was stroed in the cuSAR class.                                              |
		//---------------------------------------------------------------------------------------------------------+
		// Memory (Global):                                                                                        |
		//	- Output:                                                                                              |
		//		1. MinLevel: GPU results strage = { x,x,x,......,x } --> total nPy sets (x=Level, x=0 no hit)      |
		//		2. DistSet : GPU results strage = { [L1,L2,...]....[L1,L2,..,MaxLevel] } --> total nPy sets        |
		//		3. RaySet  : GPU results strage = { [L1,L2,...]....[L1,L2,..,MaxLevel] } --> total nPy sets        |
		//		4. ObjSet  : GPU results strage = { [L1,L2,...]....[L1,L2,..,MaxLevel] } --> total nPy pointer sets|
		//---------------------------------------------------------------------------------------------------------+
		size_t k = blockIdx.x * blockDim.x + threadIdx.x + OffsetPatch;
		if(k >= nPy){
			return;
		}

		if(SHOW && k == 0){
//			printf("+-------------------------+\n");
//			printf("| cuRayTracing Capability |\n");
//			printf("+-------------------------+\n");
//			printf("# MAXLEVEL   = %d\n", MAXLEVEL);
			printf("+-------------------------+\n");
			printf("|            Loop         |\n");
			printf("+-------------------------+\n");
			printf("# Mesh       = %ld\n", nPy);
			printf("+-------------------------+\n");
			printf("|            CUDA         |\n");
			printf("+-------------------------+\n");
			printf("# blocks  in 1 grid  = %d\n", gridDim.x);
			printf("# thresds in 1 block = %d\n", blockDim.x);
			printf("#Total thresds       = %d\n", gridDim.x*blockDim.x);
			__syncthreads();
		}


		//+===============================================+
		//| Ray Tracing (When Level-1)                    |
		//+-----------------------------------------------+
		//|     SBR For Each Ray (k)                      |
		//+===============================================+
		cuRay rayRef;
		double RayArea;
		inc.GetCell(k, rayRef, RayArea);

		cuVEC<double> org = rayRef.o;

		// SBR for each Ray
		// Find intersection
		// offset between each k
		size_t off = (k - OffsetPatch)*MaxLevel;

		// Intersection
		cuIntersectionInfo I, I_Shadow;
		bool hit = true;

		size_t count = 0;
		while(count < MaxLevel && hit){

			hit = bvh.getIntersection(rayRef, &I, false);

			if(hit){
				// Check Shadow
				// shadow ray
				cuVEC<double> uv_shadow = cu::Unit(cuVEC<double>(org.x-I.hit.x, org.y-I.hit.y, org.z-I.hit.z));
				cuRay rayShadow(I.hit, uv_shadow);
				bool IsShadow = bvh.getIntersection(rayShadow, &I_Shadow, false);
				*(Shadow  + off + count) = ( IsShadow && (I_Shadow.object != I.object) );
				// Snell's law (no matter hit point is showdow or not)
				rayRef = rayRef.Reflection(I.hit, I.object->N);
				// assignment
				*(DistSet + off + count) = I.t;
				*(RaySet  + off + count) = rayRef;
				*(ObjSet  + off + count) = (cuTRI<float>*)(I.object);

				count++;
			}
		}

		MinLevel[k-OffsetPatch] = count;
	}

	__global__
// #if __CUDA_ARCH__ == 350
// 	__launch_bounds__(128, 10)
// #endif
	void cuRayTracing2(const size_t nPy, const size_t MaxLevel, cuBVH& bvh, cuMeshInc& inc, 												// input
					   const double theta_sqc, const double theta_az, 																		// input
					   const cuVEC<double>& MainBeamUV, const cuVEC<double>& NorSquintPlane, const cuVEC<double>& PPs, const cuVEC<double>& PPt,	// input
					   size_t* MinLevel, double* DistSet, cuRay* RaySet, cuTRI<float>** ObjSet, bool* Shadow, double* AzGain,				// output
					   const size_t OffsetPatch = 0, const bool SHOW=false){																// keyword
		//---------------------------------------------------------------------------------------------------------+
		// Purpose:                                                                                                |
		//    Calculate "ALL wavelength" results and store it in the sbr with multu bouncing level.                |
		//    The 1 wavelength(lambda) was stroed in the cuSAR class.                                              |
		//---------------------------------------------------------------------------------------------------------+
		// Memory (Global):                                                                                        |
		//	- Output:                                                                                              |
		//		1. MinLevel: GPU results strage = { x,x,x,......,x } --> total nPy sets (x=Level, x=0 no hit)      |
		//		2. DistSet : GPU results strage = { [L1,L2,...]....[L1,L2,..,MaxLevel] } --> total nPy sets        |
		//		3. RaySet  : GPU results strage = { [L1,L2,...]....[L1,L2,..,MaxLevel] } --> total nPy sets        |
		//		4. ObjSet  : GPU results strage = { [L1,L2,...]....[L1,L2,..,MaxLevel] } --> total nPy pointer sets|
		//---------------------------------------------------------------------------------------------------------+
		size_t k = blockIdx.x * blockDim.x + threadIdx.x + OffsetPatch;
		if(k >= nPy){
			return;
		}

//		if(SHOW && k == 22180){
////			printf("+-------------------------+\n");
////			printf("| cuRayTracing Capability |\n");
////			printf("+-------------------------+\n");
////			printf("# MAXLEVEL   = %d\n", MAXLEVEL);
////			printf("+-------------------------+\n");
////			printf("|            Loop         |\n");
////			printf("+-------------------------+\n");
////			printf("# Mesh       = %ld\n", nPy);
////			printf("+-------------------------+\n");
////			printf("|            CUDA         |\n");
////			printf("+-------------------------+\n");
////			printf("# blocks  in 1 grid  = %d\n", gridDim.x);
////			printf("# thresds in 1 block = %d\n", blockDim.x);
////			printf("#Total thresds       = %d\n", gridDim.x*blockDim.x);
//			printf("k = %ld\n", k);
//			__syncthreads();
//		}


		//+===============================================+
		//| Ray Tracing (When Level-1)                    |
		//+-----------------------------------------------+
		//|     SBR For Each Ray (k)                      |
		//+===============================================+
		cuRay rayRef;
		double RayArea;
		inc.GetCell(k, rayRef, RayArea);

		cuVEC<double> org = rayRef.o;

		// SBR for each Ray
		// Find intersection
		// offset between each k
		size_t off = (k - OffsetPatch)*MaxLevel;

		// Intersection
		cuIntersectionInfo I, I_Shadow;
		bool hit = true;

		// Azimuth antenna gain value (input)
		double AzGain1st, AzGainLast;

		size_t count = 0;
		while(count < MaxLevel && hit){

			hit = bvh.getIntersection(rayRef, &I, false);

//			// Check incident ray
//			if(k == 286 && count == 0){
//				printf("[GPU] k = %d, rayRef.o = [%.20f,%.20f,%.20f], I.hit = [%.20f,%.20f,%.20f]\n",
//						k, rayRef.o.x, rayRef.o.y, rayRef.o.z, I.hit.x, I.hit.y, I.hit.z);
//				__syncthreads();
//			}

			if(hit){
				// Check Shadow
				// shadow ray
				cuVEC<double> uv_shadow = cu::Unit(cuVEC<double>(org.x-I.hit.x, org.y-I.hit.y, org.z-I.hit.z));
				cuRay rayShadow(I.hit, uv_shadow);
				bool IsShadow = bvh.getIntersection(rayShadow, &I_Shadow, false);
				*(Shadow  + off + count) = ( IsShadow && (I_Shadow.object != I.object) );
				// Snell's law (no matter hit point is showdow or not)
				rayRef = rayRef.Reflection(I.hit, I.object->N);
				// assignment
				*(DistSet + off + count) = I.t;
				*(RaySet  + off + count) = rayRef;
				*(ObjSet  + off + count) = (cuTRI<float>*)(I.object);
				// Azimuth antenna gain  value
				// 1st hit
				if(count == 0){
					// TODO: raytracing2
					AzGain1st = cu::AzimuthAntennaGain(theta_az, MainBeamUV, NorSquintPlane, PPs, PPt, I.hit);
				}
				// Last hit
				AzGainLast = cu::AzimuthAntennaGain(theta_az, MainBeamUV, NorSquintPlane, PPs, PPt, I.hit);
//				if(count != 0){
//					printf("count = %ld, PPt = [%.4f,%.4f,%.4f], I.hit = [%.4f,%.4f,%.4f], AzGain1st = %.4f, AzGainLast = %.4f\n", count, PPt.x, PPt.y, PPt.z, I.hit.x, I.hit.y, I.hit.z, AzGain1st, AzGainLast);
//				}
//				if(k == 7488){
//					printf("count = %ld, I.hit = [%.4f,%.4f,%.4f], MainBeamUV = [%.4f, %.4f, %.4f], NorSquintPlane = [%.4f, %.4f, %.4f], AzGain1st = %.4f, AzGainLast = %.4f\n",
//							count,
//							MainBeamUV.x, MainBeamUV.y, MainBeamUV.z,
//							NorSquintPlane.x, NorSquintPlane.y, NorSquintPlane.z,
//							I.hit.x, I.hit.y, I.hit.z, AzGain1st, AzGainLast);
//				}
				*(AzGain + off + count) = AzGain1st * AzGainLast;
//				*(AzGain + off + count) = 1;

//				if(isnan(AzGain1st) || isnan(AzGainLast)){
//					printf("[GPU] k=%ld, count=%ld, AzGain1st=%.8f, AzGainLast=%.8f\n",
//							k, count, AzGain1st, AzGainLast);
//				}

//				if(count == 0){
////					printf("+-------------------------+\n");
////					printf("|            Info         |\n");
////					printf("+-------------------------+\n");
////					printf("# MaxLevel   = %d\n", MaxLevel);
////					printf("# MinLevel   = %d\n", MinLevel[k-OffsetPatch]);
////					printf("# AzGain     = %.4f\n", AzGain[off+count]);
////					printf("k = %d, rayRef.d     = [%.20f,%.20f,%.20f]\n", k, rayRef.d.x, rayRef.d.y, rayRef.d.z);
////					printf("k = %d, rayRef.o     = [%.20f,%.20f,%.20f]\n", k, rayRef.o.x, rayRef.o.y, rayRef.o.z);
////					printf("k = %d, rayRef.inv_d = [%.20f,%.20f,%.20f]\n", k, rayRef.inv_d.x, rayRef.inv_d.y, rayRef.inv_d.z);
////					printf("k = %d, DistSet      = %.20f\n", k, DistSet[count]);
//					printf("k = %d, MinLevel = %d, AzGain = %.4f, I.hit = [%.20f,%.20f,%.20f], rayRef.o = [%.20f,%.20f,%.20f]\n",
//							k, MinLevel[k-OffsetPatch], AzGain[off+count], I.hit.x, I.hit.y, I.hit.z, k, rayRef.o.x, rayRef.o.y, rayRef.o.z);
//					__syncthreads();
//				}

				count++;
			}
		}

		MinLevel[k-OffsetPatch] = count;



//		if(SHOW && k == 0){
//			printf("+-------------------------+\n");
//			printf("|            Info         |\n");
//			printf("+-------------------------+\n");
//			printf("# MaxLevel   = %d\n", MaxLevel);
//			printf("# MinLevel   = %d\n", MinLevel[k-OffsetPatch]);
//			printf("I.hit  = [%.4f, %.4f, %.4f]\n", I.hit.x, I.hit.y, I.hit.z);
//			printf("rayRef = [%.4f, %.4f, %.4f]\n", rayRef.x, rayRef.y, rayRef.z);
//			printf("# AzGain[0]  = %d\n", AzGain[off+0]);
//			printf("# AzGain[1]  = %d\n", AzGain[off+1]);
//			printf("# AzGain[2]  = %d\n", AzGain[off+2]);
//			printf("# AzGain[3]  = %d\n", AzGain[off+3]);
//		}
	}



	__global__
	void cuGetTotalHit(const size_t nPy, const size_t* MinLevel, size_t* TotalHit){
		// This kernel can be lunch with Nthread = 1 & NBlock = 1
		// **************************************************************************************
		// *   IMPORTANT: This kernel is very SLOW. Remove it before release this application   *
		// **************************************************************************************
		*TotalHit = 0;
		for(int k=0;k<nPy;++k){
			if(MinLevel[k] > 0){
				(*TotalHit)++;
			}
		}
	}


	__global__
// #if __CUDA_ARCH__ >= 350
//	// __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
//// 	__launch_bounds__(32, 32) // 64 -> 64*16*4
//	__launch_bounds__(64, 32) // 64 -> 64*16*4
// #endif
	void cuPO(const double k0, cuEF<float>& Ef, cuBVH& bvh, cuMeshInc& inc, cuMaterialDB& MatDB,	// input
			  const size_t nPy, const size_t Nr, const size_t MaxLevel,								// input
			  size_t* MinLevel, double* DistSet, cuRay* RaySet, cuTRI<float>** ObjSet, bool* Shadow,// input (from Ray Tracing)
			  cuSBRElement<float>* sbr_partial,														// output
			  const size_t OffsetPatch = 0, const bool IsPEC=true, const double AddDis=0, 
			  const bool SHOW=false){				// keyword
#if 1
		//---------------------------------------------------------------------------------------------------------+
		// Purpose:                                                                                                |
		//    Calculate "ALL wavelength" results and store it in the sbr with multu bouncing level.                |
		//    The 1 wavelength(lambda) was stroed in the cuSAR class.                                              |
		//---------------------------------------------------------------------------------------------------------+
		// Memory (Global):                                                                                        |
		//	- Input:                                                                                               |
		//      1. Sar : cuSAR class copied from CPU results                                                       |
		//      2. Ef  : cuEF<float> class copied from CPU results                                                 |
		//		3. bvh : cuBVH class copied from CPU results                                                       |
		//		4. inc : cuMeshInc class copied from CPU results                                                   |
		//		5. MatDB : cuMAterialDB class copied from CPU results (combined MaterialDB & Material classes)     |
		//	- Output:                                                                                              |
		//  	1. sbr : [L1,L2,...MaxLevel]...[L1,L2,...MaxLevel(i)] --> Nr(j) sets                               |
		//                                               :                :                                        |
		//                                               :                v                                        |
		//  	         [L1,L2,...MaxLevel]...[L1,L2,...MaxLevel   ]    nPy(k)                                    |
		//                                                                                                         |
		//  	2. sbr_partial : [L1,L2,...MaxLevel]...[L1,L2,...MaxLevel] --> Nr sets                             |
		//                                                       :                :                                |
		//                                                       :                v                                |
		//  	                 [L1,L2,...MaxLevel]...[L1,L2,...MaxLevel]     NBlock                              |
		//                                                                                                         |
		//                                                                                                         |
		// [ Start ]                                                                                               |
		//---------------------------------------------------------------------------------------------------------+

		// Set share memory
		// size = NThread * MaxLevel
		extern __shared__ cuSBRElement<float> sdata[];



		// Index
		size_t k   = blockIdx.x * blockDim.x + threadIdx.x + OffsetPatch; 	// mesh grid
//		size_t j   = blockIdx.y * blockDim.y + threadIdx.y;		// lambda
		// Offset
		size_t off = (k - OffsetPatch)*MaxLevel;								// Offset

		// Initialize share memory to zero
		for(size_t ii=0;ii<MaxLevel;++ii){
			size_t idx = threadIdx.x*MaxLevel + ii;
			sdata[idx].sump = cuCPLX<float>(0,0);
			sdata[idx].sumt = cuCPLX<float>(0,0);
		}

		if(k >= nPy){
			return;
		}

		//+===============================================+
		//| SBR For Each Ray (k)                          |
		//+===============================================+
		cuRay RayInc;
		double RayArea;
		inc.GetCell(k, RayInc, RayArea);
		// Original Incident E-field
		cuElectricField Ei_org(RayInc.d, RayInc.o, Ef.Et, Ef.Ep);
		// Global coordinate
		cuThetaPhiVec g = Ei_org.g;

		// Pre-define (Default values, PEC)
		cuRF Rf(-1,-1);				// Reflection Factor of PEC


		//+===============================================+
		//| Each lambda (j)                               |
		//+===============================================+
		double phs = 0;

		// Incident / Reflection Electric field
		cuElectricField Ei, Er;
		// Scattering Electric field
		cuScatter PoEs;
		// inital PoEs.Level = 0
		PoEs.Level = 0;

		//+===============================================+
		//| Each bounce (i)								  |
		//+===============================================+
		for(size_t i=0;i<MinLevel[k-OffsetPatch];++i){
			// Scatter point
			cuVEC<double> Pr = RaySet[off+i].o;
			// Distance from Scatter point to Reciver plane
			double dis = cu::MinDistanceFromPointToPlane(Ei_org.k, Ei_org.o, Pr);
			// Invert direction of incident wave direction
			cuVEC<double> k_obv = -Ei_org.k;
			// Triangle plane which Scatter point is on
			const cuTRI<float> tri = *(ObjSet[off+i]);

			if(i == 0){
				// initial phase
#if __CUDA_ARCH__ >= 300
				phs = k0 * (double)__ldg(&DistSet[off+0]);
#else
				phs = k0 * (double)DistSet[off+0];
#endif
				// Initialize inc. E-field
				Ei = cuElectricField(Ei_org.k, Ei_org.o, Ei_org.cplx);
				Ei.AddPhase(phs);
			}


			//+===========================================+
			//| Pre-calculate                             |
			//+===========================================+
			// ====================== 原始位修改(開始) =====================
			 // Observation vector
			 float u = k_obv.x;
			 float v = k_obv.y;
//			 float w = k_obv.z;
//			 // Incident vector
//			 float ui = Ei.k.x;
//			 float vi = Ei.k.y;
//			 float wi = Ei.k.z;
			 // cos & sin of theta (angle from z-axis to vector)
			 float ct = k_obv.z;
			 float st = sqrtf(1.f - ct*ct);
			 // cos & sin of phi (angle from x-axis to projected vector)
			 float uv_inv = rsqrtf(u*u + v*v);
			 float sp = v*uv_inv;
			 float cp = u*uv_inv;

			 float uu = ct*cp;
			 float vv = ct*sp;
			 float ww = -st;

			 // P2到P0 向量 dot 半徑向量(Rs)，P2到P0向量 在 半徑向量 的分量
			 double u_ui = k_obv.x-Ei.k.x;
			 double v_vi = k_obv.y-Ei.k.y;
			 double w_wi = k_obv.z-Ei.k.z;
			 double w_wi2= k_obv.z+Ei.k.z;
			 // ====================== 原始位修改(結束) =====================

			// Transform the incident vector in GLOBAL coordinate (D0i) to LOCAL coordinate (D2i)
			cuVEC<float> D2i = cu::transform(tri.sa, tri.ca, tri.sb, tri.cb, cuVEC<float>(-Ei.k));

			// Get D2i component
			// Find incident spherical angles in local coordinate
			float edge = sqrtf(D2i.x*D2i.x + D2i.y*D2i.y);
			float sti2 = ::copysign(edge, D2i.z);
			float cti2 = std::abs(D2i.z);
			float cpi2 = D2i.x / edge;
			float spi2 = D2i.y / edge;

//			if(k == 1217642/2 && i == 0){
//				printf("k=%ld, i=%ld, ct=%f, st=%f, cti2=%f, sti2=%f\n", k, i,
//						ct, st, cti2, sti2);
//			}


			// ====================================== Material (Start) ============================================
			// Is PEC? If it is not PEC do something following:
			if(!IsPEC && tri.MatIdx > 0){
				long index = tri.MatIdx;

//				// 單純介質材料 (air -> material -> air)
//				Rf = cu::GetReflectinoFactor(sti2, cti2, spi2, cpi2, MatDB, uint32_t(index), k0,
//											 tri.sa, tri.ca, tri.sb, tri.cb, k, i);

				// Coating在金屬上  (air -> 金屬material)
				Rf = cu::GetReflectinoFactorOnAlumina(sti2, cti2, spi2, cpi2, MatDB, uint32_t(index), k0,
												  tri.sa, tri.ca, tri.sb, tri.cb, k, i);
			}
			// ======================================= Material (End) =============================================

		


			// Find Ref. E-field
//			Er = ReflectionElectricField(Ei, tri.N, Rf, RaySet[off+i].d);
			double sinc_patch;
			Er = ReflectionElectricField(Ei, tri.N, Rf, RaySet[off+i].d,
										 Ei_org.k, k0, RayArea, sinc_patch);






			// ====================== 原始未修改(開始) =====================
//			PoEs.cplx = cu::POCalculate(Ei, tri, k0, Rf, Ef.Taylor,
//										sp, cp, sti2, cti2, spi2, cpi2,
//										uu, vv, ww, u_ui, v_vi, w_wi,
//										k_obv, Pr, g, k, i);
			// ====================== 修改這裡(方法1) =====================
//			PoEs.cplx = cross(tri.N, cross(Ei.k, Ei.cplx));// / fabs(dot(Ei.k, tri.N)) * sinc_path;
			// ====================== 修改這裡(方法2) =====================
//			PoEs.cplx = cu::POCalculateCross(Ei, tri, k0, Rf, Ef.Taylor,
//											 sp, cp, st, ct, sti2, cti2, spi2, cpi2,
//											 uu, vv, ww, u_ui, v_vi, w_wi,
//											 k_obv, Pr, g, k, i);
			// ====================== 修改這裡(方法3) =====================
//			cuVEC<cuCPLX<float> > H = cross(Ei.k, Ei.cplx);
//			PoEs.cplx = cross(tri.N, H);
			// TODO: ====================== 修改這裡(方法4) ===================== << 用在WF上，MSTAR正常，CR正常，Ball會擴散
			PoEs.cplx = cu::POCalculate2(Ei, tri, k0, Rf,
//										 Ef.Taylor,
//										 sinc_path,
										 sp, cp, //st, ct,
										 sti2, cti2, spi2, cpi2,
										 uu, vv, ww,
//										 u_ui, v_vi, w_wi,
//										 k_obv, Pr,
										 g, k, i);
			// TODO: ======== 修改這裡(方法5: NTOU material check OK) =========== << 用在 NTOU 測試Ball正常
//			PoEs.cplx = cu::POCalculate1(Ei, tri, k0, Rf, Ef.Taylor,
//										sp, cp, sti2, cti2, spi2, cpi2,
//										uu, vv, ww, u_ui, v_vi, w_wi, w_wi2,
//										k_obv, Pr, g, k, i);
			// ====================== 修改這裡(結束) ======================


			// ====================== 修改這裡(原始) ======================
//			// double cos_theta_i_obv = dot(cuVEC<double>(tri.N), k_obv);	//(tri.N.abs()*k_obv.abs());
//			double cos_theta_i_obv = fabs(dot(cuVEC<double>(tri.N), Ei.k));	//(tri.N.abs()*k_obv.abs());
//			float factor = RayArea / cos_theta_i_obv * sinc_path;
			// ====================== 修改這裡(開始1) =====================
			float factor = RayArea / fabs(dot(Ei.k, tri.N)) * sinc_patch;
//			float factor = RayArea;
//			float factor = RayArea / fabs(dot(Ei.k, tri.N));
			// ====================== 修改這裡(結束) ======================


			PoEs.cplx.x *= factor;
			PoEs.cplx.y *= factor;
			PoEs.cplx.z *= factor;


			// Add distance phase
			double TotalDis = dis + (2* AddDis);
			AddPhase(PoEs.cplx, k0*TotalDis);
			// Assign values
			PoEs.Level = i+1;						// number of bounce (!= MinLevel)

			// If this hit point is in shadow, then Ets & Eps is zero
			if(Shadow[off+i]){
				PoEs.cplx.x = cuCPLX<float>(0.,0.);
				PoEs.cplx.y = cuCPLX<float>(0.,0.);
				PoEs.cplx.z = cuCPLX<float>(0.,0.);
			}

			// tranfrom component to theta_vec & phi_vec from observation point
			PoEs.Ets = dot(PoEs.cplx, g.theta_vec);	// Theta comp.
			PoEs.Eps = dot(PoEs.cplx, g.phi_vec);	// Phi comp.

			//
			// Duplicate inc. E-field & update phase
			// to next bounce
			//
			if(i < MinLevel[k-OffsetPatch]-1){
				phs = k0 * DistSet[off+i+1];
				// inc. E-field on the next reflection point
				Ei = cuElectricField(Er.k, RaySet[off+i].o, Er.cplx);
				Ei.AddPhase(phs);
			}

			// Assign share data
			//
			// sdata size = blockDim.x * BlockDim.y * MaxLevel
			//
			size_t b_idx = threadIdx.x*MaxLevel + i;

			sdata[b_idx].sump = PoEs.Eps;
			sdata[b_idx].sumt = PoEs.Ets;
		}// End of Level i=[0,MaxLevel-1]
		__syncthreads();
#endif

		// 在這個 block 內，切成兩半，把右半邊的壘加到左半邊
		for(size_t offset = blockDim.x / 2; offset > 0; offset >>= 1){ // x 方向的 offset
			if(threadIdx.x < offset){ // 若 threadIdx 在左半邊，才要計算
				// add a partial sum upstream to our own
				for(int ii=0;ii<MaxLevel;++ii){
					size_t idx_org = (threadIdx.x         ) * MaxLevel + ii;
					size_t idx_off = (threadIdx.x + offset) * MaxLevel + ii;
					sdata[idx_org].sump += sdata[idx_off].sump;
					sdata[idx_org].sumt += sdata[idx_off].sumt;
				}
			}
			// 當所有threadIdx都做完後(都寫入左半邊後)，再作下一輪
			__syncthreads();
		}

		// 最後結果block內最左邊的元素內
		if(threadIdx.x == 0){
			for(size_t ii=0;ii<MaxLevel;++ii){
				size_t idx_sdata =                         ii;
				size_t idx_sbr_p = blockIdx.x * MaxLevel + ii;
				sbr_partial[idx_sbr_p].sump = sdata[idx_sdata].sump;
				sbr_partial[idx_sbr_p].sumt = sdata[idx_sdata].sumt;

			}
		}



		//---------------------------------------------------------------------------------------------------------+
		// [ End ]                                                                                                 |
		//                                                                                                         |
		//---------------------------------------------------------------------------------------------------------+
	}

	__global__
// #if __CUDA_ARCH__ >= 350
//	// __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
//// 	__launch_bounds__(32, 32) // 64 -> 64*16*4
//	__launch_bounds__(64, 32) // 64 -> 64*16*4
// #endif
	void cuPO2(const double k0, cuEF<float>& Ef, cuBVH& bvh, cuMeshInc& inc, cuMaterialDB& MatDB,						// input
			   const size_t nPy, const size_t Nr, const size_t MaxLevel,												// input
			   size_t* MinLevel, double* DistSet, cuRay* RaySet, cuTRI<float>** ObjSet, bool* Shadow, double* AzGain,	// input (from Ray Tracing)
			   size_t idx, cuSBRElement<float>* sbr_partial,																		// output
			   const size_t OffsetPatch = 0, const bool IsPEC=true, const double AddDis=0, const bool SHOW=false){		// keyword
#if 1
		//---------------------------------------------------------------------------------------------------------+
		// Purpose:                                                                                                |
		//    Calculate "ALL wavelength" results and store it in the sbr with multu bouncing level.                |
		//    The 1 wavelength(lambda) was stroed in the cuSAR class.                                              |
		//---------------------------------------------------------------------------------------------------------+
		// Memory (Global):                                                                                        |
		//	- Input:                                                                                               |
		//      1. Sar : cuSAR class copied from CPU results                                                       |
		//      2. Ef  : cuEF<float> class copied from CPU results                                                 |
		//		3. bvh : cuBVH class copied from CPU results                                                       |
		//		4. inc : cuMeshInc class copied from CPU results                                                   |
		//		5. MatDB : cuMAterialDB class copied from CPU results (combined MaterialDB & Material classes)     |
		//	- Output:                                                                                              |
		//  	1. sbr : [L1,L2,...MaxLevel]...[L1,L2,...MaxLevel(i)] --> Nr(j) sets                               |
		//                                               :                :                                        |
		//                                               :                v                                        |
		//  	         [L1,L2,...MaxLevel]...[L1,L2,...MaxLevel   ]    nPy(k)                                    |
		//                                                                                                         |
		//  	2. sbr_partial : [L1,L2,...MaxLevel]...[L1,L2,...MaxLevel] --> Nr sets                             |
		//                                                       :                :                                |
		//                                                       :                v                                |
		//  	                 [L1,L2,...MaxLevel]...[L1,L2,...MaxLevel]     NBlock                              |
		//                                                                                                         |
		//                                                                                                         |
		// [ Start ]                                                                                               |
		//---------------------------------------------------------------------------------------------------------+

		// Set share memory
		// size = NThread * MaxLevel
		extern __shared__ cuSBRElement<float> sdata[];



		// Index
		size_t k   = blockIdx.x * blockDim.x + threadIdx.x + OffsetPatch; 	// mesh grid
//		size_t j   = blockIdx.y * blockDim.y + threadIdx.y;		// lambda
		// Offset
		size_t off = (k - OffsetPatch)*MaxLevel;								// Offset

		// Initialize share memory to zero
		for(size_t ii=0;ii<MaxLevel;++ii){
			size_t idx = threadIdx.x*MaxLevel + ii;
			sdata[idx].sump = cuCPLX<float>(0,0);
			sdata[idx].sumt = cuCPLX<float>(0,0);
		}

		if(k >= nPy){
			return;
		}

		//+===============================================+
		//| SBR For Each Ray (k)                          |
		//+===============================================+
		cuRay RayInc;
		double RayArea;
		inc.GetCell(k, RayInc, RayArea);
		// Original Incident E-field
		cuElectricField Ei_org(RayInc.d, RayInc.o, Ef.Et, Ef.Ep);
		// Global coordinate
		cuThetaPhiVec g = Ei_org.g;

		// Pre-define (Default values, PEC)
		cuRF Rf(-1,-1);				// Reflection Factor of PEC


		//+===============================================+
		//| Each lambda (j)                               |
		//+===============================================+
		double phs = 0;

		// Incident / Reflection Electric field
		cuElectricField Ei, Er;
		// Scattering Electric field
		cuScatter PoEs;
		// inital PoEs.Level = 0
		PoEs.Level = 0;

		//+===============================================+
		//| Each bounce (i)								  |
		//+===============================================+
		for(size_t i=0;i<MinLevel[k-OffsetPatch];++i){
			// Scatter point
			cuVEC<double> Pr = RaySet[off+i].o;
			// Distance from Scatter point to Reciver plane
			double dis = cu::MinDistanceFromPointToPlane(Ei_org.k, Ei_org.o, Pr);
			// Invert direction of incident wave direction
			cuVEC<double> k_obv = -Ei_org.k;
			// Triangle plane which Scatter point is on
			const cuTRI<float> tri = *(ObjSet[off+i]);

			if(i == 0){
				// initial phase
#if __CUDA_ARCH__ >= 300
				phs = k0 * (double)__ldg(&DistSet[off+0]);
#else
				phs = k0 * (double)DistSet[off+0];
#endif
				// Initialize inc. E-field
				Ei = cuElectricField(Ei_org.k, Ei_org.o, Ei_org.cplx);
				Ei.AddPhase(phs);
			}


			//+===========================================+
			//| Pre-calculate                             |
			//+===========================================+
			// ====================== 原始位修改(開始) =====================
			 // Observation vector
			 float u = k_obv.x;
			 float v = k_obv.y;
//			 float w = k_obv.z;
//			 // Incident vector
//			 float ui = Ei.k.x;
//			 float vi = Ei.k.y;
//			 float wi = Ei.k.z;
			 // cos & sin of theta (angle from z-axis to vector)
			 float ct = k_obv.z;
			 float st = sqrtf(1.f - ct*ct);
			 // cos & sin of phi (angle from x-axis to projected vector)
			 float uv_inv = rsqrtf(u*u + v*v);
			 float sp = v*uv_inv;
			 float cp = u*uv_inv;

			 float uu = ct*cp;
			 float vv = ct*sp;
			 float ww = -st;

			 // P2到P0 向量 dot 半徑向量(Rs)，P2到P0向量 在 半徑向量 的分量
			 double u_ui = k_obv.x-Ei.k.x;
			 double v_vi = k_obv.y-Ei.k.y;
			 double w_wi = k_obv.z-Ei.k.z;
			 double w_wi2= k_obv.z+Ei.k.z;
			 // ====================== 原始位修改(結束) =====================

			// Transform the incident vector in GLOBAL coordinate (D0i) to LOCAL coordinate (D2i)
			cuVEC<float> D2i = cu::transform(tri.sa, tri.ca, tri.sb, tri.cb, cuVEC<float>(-Ei.k));

			// Get D2i component
			// Find incident spherical angles in local coordinate
			float edge = sqrtf(D2i.x*D2i.x + D2i.y*D2i.y);
			float sti2 = ::copysign(edge, D2i.z);
			float cti2 = std::abs(D2i.z);
			float cpi2 = D2i.x / edge;
			float spi2 = D2i.y / edge;

//			if(k == 286 && i == 0){
//				printf("[GPU] k=%ld, i=%ld, ct=%f, st=%f, cti2=%f, sti2=%f\n", k, i,
//						ct, st, cti2, sti2);
//			}


			// ====================================== Material (Start) ============================================
			// Is PEC? If it is not PEC do something following:
			if(!IsPEC){
				long index = tri.MatIdx;

				// 單純介質材料 (air -> material -> air)
//				Rf = cu::GetReflectinoFactor(sti2, cti2, spi2, cpi2, MatDB, uint32_t(index), k0,
//											 tri.sa, tri.ca, tri.sb, tri.cb, k, i);

//				// Coating在金屬上  (air -> material -> Alumina)
//				Rf = cu::GetReflectinoFactorOnAlumina(sti2, cti2, spi2, cpi2, MatDB, uint32_t(index), k0,
//												  tri.sa, tri.ca, tri.sb, tri.cb, k, i);

				// Coating在金屬上  (air -> material -> PEC)
				Rf = cu::GetReflectinoFactorOnPEC(sti2, cti2, spi2, cpi2, MatDB, uint32_t(index), k0,
												  tri.sa, tri.ca, tri.sb, tri.cb, k, i);

//				if(idx == 339 && (k == 423 || k == 424) && i == 0){
//					printf("[GPU] idx=%ld, k=%ld, i=%ld, sti2=%.4f, cti2=%.4f, spi2=%.4f, cpi2=%.4f, index=%ld, k0=%f, sa=%.4f, ca=%.4f, sb=%.4f, cb=%.4f, MatDB.ER=(%f,%f), MatDB.MR=(%f.%f), MatDB.d=%f, Rf(TE,TM) = ((%.10f, %.10f), (%.10f, %.10f))\n",
//							idx, k, i, sti2, cti2, spi2, cpi2, index, k0,
//							tri.sa, tri.ca, tri.sb, tri.cb,
//							MatDB.ER[1].r, MatDB.ER[1].i, MatDB.MR[1].r, MatDB.MR[1].i, MatDB.d[1],
//							Rf.TE.r, Rf.TE.i, Rf.TM.r, Rf.TM.i);
//				}
			}
			// ======================================= Material (End) =============================================




			// Find Ref. E-field
//			Er = ReflectionElectricField(Ei, tri.N, Rf, RaySet[off+i].d);
			double sinc_patch;
			Er = ReflectionElectricField(Ei, tri.N, Rf, RaySet[off+i].d,
										 Ei_org.k, k0, RayArea, sinc_patch);






			// ====================== 原始未修改(開始) =====================
//			PoEs.cplx = cu::POCalculate(Ei, tri, k0, Rf, Ef.Taylor,
//										sp, cp, sti2, cti2, spi2, cpi2,
//										uu, vv, ww, u_ui, v_vi, w_wi,
//										k_obv, Pr, g, k, i);
			// ====================== 修改這裡(方法1) =====================
//			PoEs.cplx = cross(tri.N, cross(Ei.k, Ei.cplx));// / fabs(dot(Ei.k, tri.N)) * sinc_path;
			// ====================== 修改這裡(方法2) =====================
//			PoEs.cplx = cu::POCalculateCross(Ei, tri, k0, Rf, Ef.Taylor,
//											 sp, cp, st, ct, sti2, cti2, spi2, cpi2,
//											 uu, vv, ww, u_ui, v_vi, w_wi,
//											 k_obv, Pr, g, k, i);
			// ====================== 修改這裡(方法3) =====================
//			cuVEC<cuCPLX<float> > H = cross(Ei.k, Ei.cplx);
//			PoEs.cplx = cross(tri.N, H);
			// TODO: ====================== 修改這裡(方法4) ===================== << 用在WF上，MSTAR正常，CR正常，Ball會擴散
			PoEs.cplx = cu::POCalculate2(Ei, tri, k0, Rf,
//										 Ef.Taylor,
//										 sinc_path,
										 sp, cp, //st, ct,
										 sti2, cti2, spi2, cpi2,
										 uu, vv, ww,
//										 u_ui, v_vi, w_wi,
//										 k_obv, Pr,
										 g, k, i);
			// TODO: ======== 修改這裡(方法5: NTOU material check OK) =========== << 用在 NTOU 測試Ball正常
//			PoEs.cplx = cu::POCalculate1(Ei, tri, k0, Rf, Ef.Taylor,
//										sp, cp, sti2, cti2, spi2, cpi2,
//										uu, vv, ww, u_ui, v_vi, w_wi, w_wi2,
//										k_obv, Pr, g, k, i);
			// ====================== 修改這裡(結束) ======================


//			if(k == 286 && i == 0 && idx == 0){
//				printf("[GPU] k=%ld, Es.cplx = [(%.8f,%.8f),(%.8f,%.8f),(%.8f,%.8f)]\n", k, PoEs.cplx.x.r, PoEs.cplx.x.i, PoEs.cplx.y.r, PoEs.cplx.y.i, PoEs.cplx.z.r, PoEs.cplx.z.i);
//			}


			// ====================== 修改這裡(原始) ======================
//			// double cos_theta_i_obv = dot(cuVEC<double>(tri.N), k_obv);	//(tri.N.abs()*k_obv.abs());
//			double cos_theta_i_obv = fabs(dot(cuVEC<double>(tri.N), Ei.k));	//(tri.N.abs()*k_obv.abs());
//			float factor = RayArea / cos_theta_i_obv * sinc_path;
			// ====================== 修改這裡(開始1) =====================
			double factor = RayArea / fabs(dot(Ei.k, tri.N)) * sinc_patch;
//			float factor = RayArea;
//			float factor = RayArea / fabs(dot(Ei.k, tri.N));
			// ====================== 修改這裡(結束) ======================


			// =================== 加上 AzGain (開始) ====================
			// TODO: Add AzGain
			double factor2 = factor * AzGain[off+i];
			// =================== 加上 AzGain (結束) ====================


			PoEs.cplx.x *= factor2;
			PoEs.cplx.y *= factor2;
			PoEs.cplx.z *= factor2;
//			PoEs.cplx.x = AzGain[off+i];
//			PoEs.cplx.y = AzGain[off+i];
//			PoEs.cplx.z = AzGain[off+i];


			// Add distance phase
			double TotalDis = dis + (2* AddDis);
			AddPhase(PoEs.cplx, k0*TotalDis);
			// Assign values
			PoEs.Level = i+1;						// number of bounce (!= MinLevel)

			// If this hit point is in shadow, then Ets & Eps is zero
			if(Shadow[off+i]){
				PoEs.cplx.x = cuCPLX<float>(0.,0.);
				PoEs.cplx.y = cuCPLX<float>(0.,0.);
				PoEs.cplx.z = cuCPLX<float>(0.,0.);
			}

			// tranfrom component to theta_vec & phi_vec from observation point
			PoEs.Ets = dot(PoEs.cplx, g.theta_vec);	// Theta comp.
			PoEs.Eps = dot(PoEs.cplx, g.phi_vec);	// Phi comp.

//			PoEs.Ets = cuCPLX<float>(AzGain[off+i], 0);
//			PoEs.Eps = cuCPLX<float>(AzGain[off+i], 0);

			// TODO: 偵測位置
//			if( isnan(PoEs.Ets.r) || isnan(PoEs.Ets.i) ||
//				isnan(PoEs.Eps.r) || isnan(PoEs.Eps.i) ){
//				printf("[GPU] k=%ld, i=%ld, Ets=(%.8f,%.8f), Eps=(%.8f,%.8f), AzGain[off+i]=%.8f, factor2=%.8f, factor=%.8f, RayArea=%.8f, sinc_patch=%.4f\n",
//						k, i, PoEs.Ets.r, PoEs.Ets.i, PoEs.Eps.r, PoEs.Eps.i, AzGain[off+i], factor2, factor, RayArea, sinc_patch);
//			}
//			if(k == 286 && i == 0 && idx == 0){
//			if(k == 7488 && idx == 0){
//				printf("[GPU] k=%ld, i=%ld, idx=%ld, Es.cplx=[(%.8f,%.8f),(%.8f,%.8f),(%.8f,%.8f)], Ets=(%.8f,%.8f), Eps=(%.8f,%.8f), factor=%.8f, factor2=%.8f, Shadow=%d\n",
//						k, i, idx, PoEs.cplx.x.r, PoEs.cplx.x.i, PoEs.cplx.y.r, PoEs.cplx.y.i, PoEs.cplx.z.r, PoEs.cplx.z.i,
//						PoEs.Ets.r, PoEs.Ets.i, PoEs.Eps.r, PoEs.Eps.i, factor, factor2, Shadow[off+i]);
//			}


			//
			// Duplicate inc. E-field & update phase
			// to next bounce
			//
			if(i < MinLevel[k-OffsetPatch]-1){
				phs = k0 * DistSet[off+i+1];
				// inc. E-field on the next reflection point
				Ei = cuElectricField(Er.k, RaySet[off+i].o, Er.cplx);
				Ei.AddPhase(phs);
			}

			// Assign share data
			//
			// sdata size = blockDim.x * BlockDim.y * MaxLevel
			//
			size_t b_idx = threadIdx.x*MaxLevel + i;

			sdata[b_idx].sump = PoEs.Eps;
			sdata[b_idx].sumt = PoEs.Ets;
		}// End of Level i=[0,MaxLevel-1]
		__syncthreads();
#endif

		// 在這個 block 內，切成兩半，把右半邊的壘加到左半邊
		for(size_t offset = blockDim.x / 2; offset > 0; offset >>= 1){ // x 方向的 offset
			if(threadIdx.x < offset){ // 若 threadIdx 在左半邊，才要計算
				// add a partial sum upstream to our own
				for(int ii=0;ii<MaxLevel;++ii){
					size_t idx_org = (threadIdx.x         ) * MaxLevel + ii;
					size_t idx_off = (threadIdx.x + offset) * MaxLevel + ii;
					sdata[idx_org].sump += sdata[idx_off].sump;
					sdata[idx_org].sumt += sdata[idx_off].sumt;
				}
			}
			// 當所有threadIdx都做完後(都寫入左半邊後)，再作下一輪
			__syncthreads();
		}

		// 最後結果block內最左邊的元素內
		if(threadIdx.x == 0){
			for(size_t ii=0;ii<MaxLevel;++ii){
				size_t idx_sdata =                         ii;
				size_t idx_sbr_p = blockIdx.x * MaxLevel + ii;
				sbr_partial[idx_sbr_p].sump = sdata[idx_sdata].sump;
				sbr_partial[idx_sbr_p].sumt = sdata[idx_sdata].sumt;

			}
		}



		//---------------------------------------------------------------------------------------------------------+
		// [ End ]                                                                                                 |
		//                                                                                                         |
		//---------------------------------------------------------------------------------------------------------+
	}

	__global__
	void cuSumSingleThread(const cuSBRElement<float>* sbr_partial, const size_t MaxLevel, 	// input
						   const dim3 NBlock, cuSBRElement<float>* d_res){					// input + output
		// sbr_partial : cuSBRElement<float> [NBlock2.x * MaxLevel]
		// d_res：Nr * MaxLevel
		for(size_t j=0;j<MaxLevel;++j){
			// reset to zero
			d_res[j].sumt = cuCPLX<float>(0,0);
			d_res[j].sump = cuCPLX<float>(0,0);
			for(size_t i=0;i<NBlock.x;++i){
				size_t off = i*MaxLevel;
				d_res[j].sumt += sbr_partial[off+j].sumt;
				d_res[j].sump += sbr_partial[off+j].sump;
			}
		}
	}

	/**
	 * @param [in] sbr_partial
	 */
	__global__
	void cuBlockSum(const cuSBRElement<float>* sbr_partial, cuSBRElement<float>* per_block_results, const size_t nPy,
					const size_t MaxLevel, const dim3 NBlock, const size_t off=0, cuSBRElement<float>* d_res=NULL){

		// Shared memory size is:
		// NThread3.x * MaxLevel * sizeof(cuSBRElement<float>);
		extern __shared__ cuSBRElement<float> sdata1[];

		// sbr_partial size is:
		// NBlock2.x * MaxLevel * sizeof(cuSBRElement<float>)

		// per_block_results size is:
		// NThread3.x * MaxLevel * sizeof(cuSBRElement<float>);

		// Index
		size_t k   = blockIdx.x * blockDim.x + threadIdx.x;		// mesh grid


		// 對於這個 block 而言，把每個thread index對應的 input[i] 寫入 __shared__ sdata 內
		if(k < NBlock.x){
			for(size_t i=0;i<MaxLevel;++i){
				size_t idx_sbr_p = k*MaxLevel + i;
				sdata1[threadIdx.x * MaxLevel + i].sump = sbr_partial[idx_sbr_p].sump;
				sdata1[threadIdx.x * MaxLevel + i].sumt = sbr_partial[idx_sbr_p].sumt;
			}
		}else{
			for(size_t i=0;i<MaxLevel;++i){
				sdata1[threadIdx.x * MaxLevel + i].sump = cuCPLX<float>(0,0);
				sdata1[threadIdx.x * MaxLevel + i].sumt = cuCPLX<float>(0,0);
			}
		}
		__syncthreads();

		// 在這個 block 內，切成兩半，把右半邊的壘加到左半邊
		for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
			if(threadIdx.x < offset) { // 若 threadIdx 在左半邊，才要計算
				for(size_t i=0;i<MaxLevel;++i){
					// add a partial sum upstream to our own
					size_t idx_org =  threadIdx.x           * MaxLevel + i;
					size_t idx_off = (threadIdx.x + offset) * MaxLevel + i;
					sdata1[idx_org].sump += sdata1[idx_off].sump;
					sdata1[idx_org].sumt += sdata1[idx_off].sumt;
				}
			}
			// 當所有threadIdx都做完後(都寫入左半邊後)，再作下一輪
			__syncthreads();
		}

		// 最後結果block內最左邊的元素內
		if(threadIdx.x == 0) {
			if(d_res == NULL){
				for(size_t i=0;i<MaxLevel;++i){
					size_t idx_sdata =                       i;
					size_t idx_sbr_p = blockIdx.x*MaxLevel + i;
					per_block_results[idx_sbr_p].sump = sdata1[idx_sdata].sump;
					per_block_results[idx_sbr_p].sumt = sdata1[idx_sdata].sumt;
				}
			}else{
				// 寫到暫存的 d_res
				for(size_t i=0;i<MaxLevel;++i){
					(d_res + off + i)->sump = sdata1[i].sump;
					(d_res + off + i)->sumt = sdata1[i].sumt;
				}
			}
		}

	}



	void cuSum(const dim3 NBlock2, const dim3 NThread3, const size_t nPy, const size_t MaxLevel, const size_t off,
			   const cuSBRElement<float>* d_sbr_partial, cuSBRElement<float>* d_res,
			   vector<cuSBRElement<float>*> d_sbr_per_block,
			   const dim3* NBlock3, const size_t* SharedSize3, const cudaStream_t ST=0){

		if(NBlock2.x <= NThread3.x){
			// cout<<"CASE 1"<<endl;
			// 如果 NBlock.x <= NThread3.x 只需要執行一次 cuBlockSum
			cuBlockSum<<<NBlock3[0], NThread3, SharedSize3[0], ST>>>
						 (d_sbr_partial, d_sbr_per_block[0], nPy, MaxLevel, NBlock2, off, d_res);
		}else if(NBlock2.x/NThread3.x <= NThread3.x){
			// cout<<"CASE 2"<<endl;
			// 如果 NBlock.x/NThread3.x <= NThread3.x 需要執行二次 cuBlockSum
			// It only needs to twice processes because that : (256^2 = 65536 max block number)
			//
			// Round 1
			//
			cuBlockSum<<<NBlock3[0], NThread3, SharedSize3[0], ST>>>
						 (d_sbr_partial, d_sbr_per_block[0], nPy, MaxLevel, NBlock2);
			//
			// Round 2
			//
			cuBlockSum<<<NBlock3[1], NThread3, SharedSize3[1], ST>>>
						 (d_sbr_per_block[0], d_sbr_per_block[1], nPy, MaxLevel, NBlock3[0], off, d_res);
		}else if(NBlock2.x/NThread3.x/NThread3.x <= NThread3.x){
			// cout<<"CASE 3"<<endl;
			// 如果 NBlock.x/NThread3.x/NThread3.x <= NThread3.x 需要執行三次 cuBlockSum
			// It only needs to twice processes because that : (256^2 = 65536 max block number)
			//
			// Round 1
			//
			cuBlockSum<<<NBlock3[0], NThread3, SharedSize3[0], ST>>>
						 (d_sbr_partial, d_sbr_per_block[0], nPy, MaxLevel, NBlock2);
			//
			// Round 2
			//
			cuBlockSum<<<NBlock3[1], NThread3, SharedSize3[1], ST>>>
						 (d_sbr_per_block[0], d_sbr_per_block[1], nPy, MaxLevel, NBlock3[0]);
			//
			// Round 3
			//
			cuBlockSum<<<NBlock3[2], NThread3, SharedSize3[2], ST>>>
						 (d_sbr_per_block[1], d_sbr_per_block[2], nPy, MaxLevel, NBlock3[1], off, d_res);
		}else{
			// cout<<"CASE 4"<<endl;
			// 如果 NBlock.x/NThread3.x/NThread3.x/NThread3.x <= NThread3.x 需要執行四次 cuBlockSum
			// It only needs to twice processes because that : (256^2 = 65536 max block number)
			//
			// Round 1
			//
			cuBlockSum<<<NBlock3[0], NThread3, SharedSize3[0], ST>>>
						 (d_sbr_partial, d_sbr_per_block[0], nPy, MaxLevel, NBlock2);
			//
			// Round 2
			//
			cuBlockSum<<<NBlock3[1], NThread3, SharedSize3[1], ST>>>
						 (d_sbr_per_block[0], d_sbr_per_block[1], nPy, MaxLevel, NBlock3[0]);
			//
			// Round 3
			//
			cuBlockSum<<<NBlock3[2], NThread3, SharedSize3[2], ST>>>
						 (d_sbr_per_block[1], d_sbr_per_block[2], nPy, MaxLevel, NBlock3[1]);
			//
			// Round 4
			//
			cuBlockSum<<<NBlock3[3], NThread3, SharedSize3[3], ST>>>
						 (d_sbr_per_block[2], d_sbr_per_block[3], nPy, MaxLevel, NBlock3[2], off, d_res);
		}// Block Sum
	}






}  // namespace cu




#endif /* CUSUB_CUH_ */
