/*
 * cukernel.cuh
 *
 *  Created on: Oct 16, 2014
 *      Author: cychiang
 */

#ifndef CUMAIN_CUH_
#define CUMAIN_CUH_

//#define nPyMax 33554432	// Maxmium nPy (4096^2 * 2)
//#define nPyMax 29360128	// Maxmium nPy (4096^2 * 1.75)
//#define nPyMax 25165824	// Maxmium nPy (4096^2 * 1.5)
#define nPyMax 16777216	// Maxmium nPy (4096^2)
//#define nPyMax 4194304	// Maxmium nPy (2048^2)
//#define nPyMax 1048576	// Maxmium nPy (1024^2)
//#define nPyMax 262144	// Maxmium nPy (512^2)
//#define nPyMax 65536	// Maxmium nPy (256^2)
//#define nPyMax 16384	// Maxmium nPy (256^2)

#include <cmath>
#include <cuda/cumisc.cuh>
#include <cuda/cusub.cuh>
#include <sar/sar.h>


namespace cu {


	//+=======================================================+
	//|                     CUDA Main                         |
	//+=======================================================+
	/**
	 * @param [in] OffsetPatch: Number of offset for each patch (default = 0)
	 */
	void cuDoIt(const vector<SAR>& Sar2, const D1<double>& k0, const EF& Ef, const BVH& bvh, const MeshInc& inc2, const MaterialDB& MatDB,
				const double mesh_dRad, const dim3 NThread1, const dim3 NThread2, const dim3 NThread4,
				cuSBRElement<float>* res, const bool IsPEC, size_t nPyPatch = 0, const size_t OffsetPatch = 0, 
				const bool IsShow=false){

		//+------------------------------------------------------------------------------------------+
		//|                                  Host -> Device Memory                                   |
		//+------------------------------------------------------------------------------------------+
		// Get CUDA memory info
		if(IsShow){ CheckMem("Initial:"); }


		clock_t gpu_dt = tic();
		clock_t gpu_dt0 = tic();
		//+-----------------+
		//|    Dimension    |
		//+-----------------+
		size_t Nr  = Sar2.size();
		size_t nPy = inc2.nPy;
		size_t MaxLevel = Ef.MaxLevel();

		// For defaul, processing full scene, no patch
		if(nPyPatch == 0){
			nPyPatch = nPy;
		}

		//+-----------------+
		//|      Copy       |
		//+-----------------+
		// Copy BVH(Host) to cuBVH(Device)
		cu::cuBVH* d_bvh = NULL;
		cu::cuTRI<float>* d_primes;
		size_t* d_idx_poly;
		cu::cuBVHFlatNode* d_flatTree;
		d_bvh->Create(bvh, d_bvh, d_primes, d_idx_poly, d_flatTree);


		// Copy MeshInc(Host) to cuMeshInc(Device)
		cu::cuMeshInc* d_inc = NULL;
		cu::cuVEC<double>* d_dirH_disH;
		cu::cuVEC<double>* d_dirV_disV;
		d_inc->Create(inc2, d_inc, d_dirH_disH, d_dirV_disV);


		// Copy MaterialDB(Host) to cuMaterialDB(Device)
		cu::cuMaterialDB* d_MatDB = NULL;
		size_t* d_Idx;
		double* d_Er_r;
		double* d_Tang;
		double* d_Mr;
		double* d_Mi;
		double* d_D;
		cuCPLX<double>* d_ER = NULL;
		cuCPLX<double>* d_MR = NULL;
		double* d_ERr_MRr_Sqrt = NULL;
		d_MatDB->Create(MatDB, d_MatDB, d_Idx, d_Er_r, d_Tang, d_Mr, d_Mi, d_D, d_ER, d_MR, d_ERr_MRr_Sqrt);


		// Copy EF(Host) to cuEF(Device)
		cu::cuEF<float>* d_Ef = NULL;
		d_Ef->Create(Ef, d_Ef);


		// Copy SAR(Host) to cuSAR(Device)
		cu::cuSAR* d_Sar = NULL;
		d_Sar->Create(Sar2[0], d_Sar);


		//+--------------------+
		//|  Memoy Allocation  |
		//+--------------------+
		// Malloc Tray Tracing Sets
		size_t* d_MinLevel;			// Store the minimum bounceing level for each grid
		double*  d_DisSet;			// Store the distance for each grid & level
		cuRay*  d_RaySet;			// Store the Ray class set for each grid & level
		bool* d_Shadow;				// Store the shadow or not for each grid & level (it could be deleted)
		cuTRI<float>** d_ObjSet;	// Store the intersect object set for each grid & level
		SafeMallocCheck(cudaMalloc(&d_MinLevel, nPyPatch * sizeof(size_t)));
		SafeMallocCheck(cudaMalloc(&d_DisSet,   MaxLevel * nPyPatch * sizeof(double)));
		SafeMallocCheck(cudaMalloc(&d_RaySet,   MaxLevel * nPyPatch * sizeof(cuRay)));
		SafeMallocCheck(cudaMalloc(&d_ObjSet,   MaxLevel * nPyPatch * sizeof(cuTRI<float>*)));
		SafeMallocCheck(cudaMalloc(&d_Shadow,   MaxLevel * nPyPatch * sizeof(bool)));

#ifdef CUDEBUG
		if(IsShow){
			printf("nPyPatch              = %ld\n", nPyPatch);
			printf("MaxLevel              = %ld\n", MaxLevel);
			printf("MaxLevel * nPyPatch   = %ld\n", MaxLevel * nPyPatch);
			printf("sizeof(size_t)        = %ld\n", sizeof(size_t));
			printf("sizeof(double)        = %ld\n", sizeof(double));
			printf("sizeof(cuRay)         = %ld\n", sizeof(cuRay));
			printf("sizeof(cuTRI<float>*) = %ld\n", sizeof(cuTRI<float>*));
			printf("sizeof(bool)          = %ld\n", sizeof(bool));
		}
#endif


		// results in device
		cuSBRElement<float>* d_res;
		SafeMallocCheck(cudaMalloc(&d_res,   Nr * MaxLevel * sizeof(cuSBRElement<float>)));
		cudaMemset(d_res, 0, Nr * MaxLevel * sizeof(cuSBRElement<float>));

		// Get CUDA memory info
		if(IsShow){
			toc(gpu_dt0, "Global Mem transfer: ");
			CheckMem("After Malloc:");
		}


		//+------------------------------------------------------------------------------------------+
		//|                                  Ray Tracing                                             |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt1 = tic();
		//+--------------------+
		//|       Kernel       |
		//+--------------------+
		//
		// Ray Tracing
		//
		// dim3 NBlock1(ceil(float(inc2.nPy)/NThread1.x));
		dim3 NBlock1(ceil(float(nPyPatch)/NThread1.x));

		if(IsShow){ CheckDim(NThread1, NBlock1, 0); }

		// 1st parameter "nPy" stands for full scene ray number
		cuRayTracing<<<NBlock1,NThread1>>>(nPy, MaxLevel, *d_bvh, *d_inc, d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow, OffsetPatch);
		cudaDeviceSynchronize();

		// Show the ray tracing information
#ifdef CUDEBUG
		size_t TotalHit;
		size_t* d_TotalHit;
		SafeMallocCheck(cudaMalloc(&d_TotalHit, sizeof(size_t)));
		cuGetTotalHit<<<1,1>>>(nPyPatch, d_MinLevel, d_TotalHit);
		cudaMemcpy(&TotalHit, d_TotalHit, sizeof(size_t), cudaMemcpyDeviceToHost);
		printf("+-------------------+\n");
		printf("nPyPatch = %ld\n", nPyPatch);
		printf("RayArea  = %.16f\n", inc2.Area);
		printf("TotalHit = %ld\n", TotalHit);
		printf("TotalHit * RayArea = %.16f\n", double(TotalHit)*inc2.Area);
		printf("+-------------------+\n");
#endif

		if(IsShow){ toc(gpu_dt1, "cuRayTracing: "); }
		//+------------------------------------------------------------------------------------------+
		//|                                  PO + Block Sum                                          |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt2 = tic();
		//+-----------------+
		//|  Set & Alloc.   |
		//+-----------------+
		// In the 1srt round, the

		// Threads / Blocks Setting
		dim3 NBlock2(ceil(float(nPyPatch)/NThread2.x), 1, 1);	// number of block in single patch grid
		size_t sbr_partial_sz = NBlock2.x * MaxLevel * sizeof(cuSBRElement<float>);

		// Allocation partial SBR algorithm results array
		cuSBRElement<float>* d_sbr_partial_array = NULL;
		cudaMalloc(&d_sbr_partial_array,     sbr_partial_sz);
		cudaMemset(d_sbr_partial_array,   0, sbr_partial_sz);



		// Get CUDA memory info
		size_t SharedSize2 = NThread2.x*MaxLevel*sizeof(cuSBRElement<float>);


		// Pre-define memory (The number of reduction round MUST <= 4 times)
		// Because the
		dim3 NBlock4[4];
		size_t SharedSize4[4]    = {0,0,0,0};
		size_t per_block_size[4] = {0,0,0,0};

		// cuBlockSum pre-calculate
		size_t rem  = ceil(float(NBlock2.x)/NThread4.x);
		size_t rem2 = rem;
		size_t count = 0;
		bool check = true;

		do{
			NBlock4[count] 		  = dim3(rem, 1, 1);
			SharedSize4[count]    = NThread4.x * MaxLevel * sizeof(cuSBRElement<float>);
			per_block_size[count] = NBlock4[count].x * SharedSize4[count];
			// Next step
			rem = ceil(float(rem)/NThread4.x);
			check = (rem != rem2);
			rem2 = rem;
			count++;
		}while(check);

		// cuBlockSum temp variables
		vector<cuSBRElement<float>*> d_sbr_per_block(4);

		cudaMalloc(&d_sbr_per_block[0], per_block_size[0]);
		cudaMalloc(&d_sbr_per_block[1], per_block_size[1]);
		cudaMalloc(&d_sbr_per_block[2], per_block_size[2]);
		cudaMalloc(&d_sbr_per_block[3], per_block_size[3]);

		//+-----------------+
		//| Wavelength Loop |
		//+-----------------+
		for(size_t idx=0;idx<Nr;++idx){
			// each stream
			size_t off = idx * MaxLevel;
#ifdef CUDEBUG
			printf("NBlock2.x = %ld, NThread2.x = %ld, SharedSize2 = %ld\n",
				    NBlock2.x, NThread2.x, SharedSize2);
#endif
			// kernel excution

//			printf("inc2.Rad = %lf, (8E4 - 4) = %lf\n", inc2.Rad, 8E4 - 4);

			cuPO<<<NBlock2, NThread2, SharedSize2>>>
				   (k0[idx], *d_Ef, *d_bvh, *d_inc, *d_MatDB, nPy, Nr, MaxLevel,
				   d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow, d_sbr_partial_array,
				   OffsetPatch, IsPEC, -mesh_dRad, false);
			// Run, d_res : Nr * MaxLevel * sizeof(cuSBRElement<float>)
			cuSum(NBlock2, NThread4, nPyPatch, MaxLevel, off, d_sbr_partial_array, d_res,
				  d_sbr_per_block, NBlock4, SharedSize4);
			// cuSumSingleThread<<<1,1>>>(d_sbr_partial_array, MaxLevel, NBlock2, d_res);
		}// f0 block
		//+-----------------+
		//|  Store to host  |
		//+-----------------+
		cudaMemcpy(res, d_res, Nr*MaxLevel*sizeof(cuSBRElement<float>), cudaMemcpyDeviceToHost);
		if(IsShow){ toc(gpu_dt2, "cuPO + cuBlockSum: "); }

		//+------------------------------------------------------------------------------------------+
		//|                                  CUDA Free                                               |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt3 = tic();

		d_bvh->Free(d_bvh, d_primes, d_idx_poly, d_flatTree);
		d_inc->Free(d_inc, d_dirH_disH, d_dirV_disV);
		d_MatDB->Free(d_MatDB, d_Idx, d_Er_r, d_Tang, d_Mr, d_Mi, d_D, d_ER, d_MR, d_ERr_MRr_Sqrt);
		d_Ef->Free(d_Ef);
		d_Sar->Free(d_Sar);

		cudaFree(d_MinLevel);
		cudaFree(d_DisSet);
		cudaFree(d_RaySet);
		cudaFree(d_ObjSet);
		cudaFree(d_Shadow);
		cudaFree(d_res);

		cudaFree(d_sbr_partial_array);
		cudaFree(d_sbr_per_block[0]);
		cudaFree(d_sbr_per_block[1]);
		cudaFree(d_sbr_per_block[2]);
		cudaFree(d_sbr_per_block[3]);

		// if(IsShow){
		// 	toc(gpu_dt3, "cudaFree: ");
		// 	toc(gpu_dt,  "GPU Total: ");
		// 	CheckMem("PO Free:");
		// }


	} // End of cuDoIt()


	/**
	 * @param [in] NThread1: Thread number of cuRayTracing
	 * @param [in] NThread2: Thread number of cuPO
	 * @param [in] NThread4: Thread number of cuSum
	 * @param [out] res: Return each level summing all of ray, cuSBRElement<float>[N_Freq * MaxLevel]
	 */
	void cuSBRDoIt(const vector<SAR>& Sar2, const D1<double>& k0, const EF& Ef, const BVH& bvh, const MeshInc& inc2, const MaterialDB& MatDB,
				   const double mesh_dRad, const dim3 NThread1, const dim3 NThread2, const dim3 NThread4,
				   cuSBRElement<float>* res, const bool IsPEC, const bool IsShow=false, const bool Is1st=false){
		//+------------------------------------------------------------------------------------------+
		//|                                  Check split number                                      |
		//+------------------------------------------------------------------------------------------+
		size_t nPy = inc2.nPy;
		size_t nPyProcess = 0;
		size_t nPatch = std::ceil(double(nPy) / nPyMax);
		size_t nFreq = Sar2.size();
		size_t MaxLevel = Ef.MaxLevel();

		if(Is1st){
			printf("=============================\n");
			printf("|      Patch Processing     |\n");
			printf("=============================\n");
			printf("nPy    = %ld\n", nPy);
			printf("nPyMax = %d\n",  nPyMax);
			printf("nPatch = %ld\n", nPatch);
		}

		clock_t gpu_dt = tic();

		if(nPy > nPyMax){
			// Reset destination array to be zero
			memset(res, 0, Sar2.size() * Ef.MaxLevel() * sizeof(cuSBRElement<float>));

			// Host Memory allocation
			cuSBRElement<float>* res_tmp = new cuSBRElement<float>[nFreq * MaxLevel];
			// Reset to zero
			for(size_t j=0;j<nFreq;++j){
				for(int i=0;i<MaxLevel;++i){
					res_tmp[j*MaxLevel+i].sump = cuCPLX<float>(0,0);
					res_tmp[j*MaxLevel+i].sumt = cuCPLX<float>(0,0);
				}
			}

			// For each patch
			for(size_t p=0;p<nPatch;++p){
				size_t OffsetPatch = p*nPyMax;	// nPy offset [samples]

				if(p != nPatch-1){
					nPyProcess = nPyMax;
				}else{
					nPyProcess = nPy - (nPatch-1)*nPyMax;
				}

				cuDoIt(Sar2, k0, Ef, bvh, inc2, MatDB, 						// input
					   mesh_dRad, NThread1, NThread2, NThread4,				// input
					   res_tmp, IsPEC, nPyProcess, OffsetPatch, IsShow); 	// output + keywords

#ifdef CUDEBUG
				printf("================================\n");
				printf("#Patch = %ld, nPy = %ld, nPatch = %ld, nPyProcess = %ld, OffsetPatch = %ld\n", p, nPy, nPatch, nPyProcess, OffsetPatch);
				CheckMem("After call cuDoIt["+num2str<size_t>(p)+"]:");
				printf("================================\n");

				printf("\n**************** Patch = %ld ******************\n", p);
				printf("Tx = %s\n", Ef.TxPol().c_str());
				for(int i=0;i<MaxLevel;++i){
					cu::cuCPLX<float> RxH = res[i].sump;
					cu::cuCPLX<float> RxV = res[i].sumt;
					printf("Level = %d\n", i);
					printf("  H%s[%d] = (% .16f,% .16f), abs = %.16f\n", Ef.TxPol().c_str(), i, RxH.r, RxH.i, RxH.abs());
					printf("  V%s[%d] = (% .16f,% .16f), abs = %.16f\n", Ef.TxPol().c_str(), i, RxV.r, RxV.i, RxV.abs());
				}
#endif				

				for(size_t j=0;j<nFreq;++j){
					for(int i=0;i<MaxLevel;++i){
						res[j*MaxLevel+i].sump += res_tmp[j*MaxLevel+i].sump;
						res[j*MaxLevel+i].sumt += res_tmp[j*MaxLevel+i].sumt;
					}
				}
			}

			// delete
			delete[] res_tmp;
		}else{
			cuDoIt(Sar2, k0, Ef, bvh, inc2, MatDB, 				// input
				   mesh_dRad, NThread1, NThread2, NThread4,		// input
				   res, IsPEC, nPy, 0, IsShow); 				// output + keywords
#ifdef CUDEBUG
			CheckMem("After call cuDoIt:");
#endif			
		}


		// toc(gpu_dt,  "GPU Total: ");
	} // End of cuSBRDoIt





	//+=======================================================+
	//|          CUDA Main (With Az Ant. Gain)                |
	//+=======================================================+
	/**
	 * @param [in] OffsetPatch: Number of offset for each patch (default = 0)
	 */
	void cuDoIt2(const vector<SAR>& Sar2, const D1<double>& k0, const EF& Ef, const BVH& bvh, const MeshInc& inc2, const MaterialDB& MatDB,
				 const double mesh_dRad, const dim3 NThread1, const dim3 NThread2, const dim3 NThread4,
				 cuSBRElement<float>* res, const bool IsPEC,
				 const VEC<double>& MainBeamUV, const VEC<double>& NorSquintPlane, const VEC<double>& PPs, const VEC<double>& PPt,
				 size_t nPyPatch = 0, const size_t OffsetPatch = 0, const bool IsShow=false){

		//+------------------------------------------------------------------------------------------+
		//|                                  Host -> Device Memory                                   |
		//+------------------------------------------------------------------------------------------+
		// Get CUDA memory info
		if(IsShow){ CheckMem("Initial:"); }


		clock_t gpu_dt = tic();
		clock_t gpu_dt0 = tic();
		//+-----------------+
		//|    Dimension    |
		//+-----------------+
		size_t Nr  = Sar2.size();
		size_t nPy = inc2.nPy;
		size_t MaxLevel = Ef.MaxLevel();

		// For default, processing full scene, no patch
		if(nPyPatch == 0){
			nPyPatch = nPy;
		}

		//+-----------------+
		//|      Copy       |
		//+-----------------+
		// Copy BVH(Host) to cuBVH(Device)
		cu::cuBVH* d_bvh = NULL;
		cu::cuTRI<float>* d_primes;
		size_t* d_idx_poly;
		cu::cuBVHFlatNode* d_flatTree;
		d_bvh->Create(bvh, d_bvh, d_primes, d_idx_poly, d_flatTree);


		// Copy MeshInc(Host) to cuMeshInc(Device)
		cu::cuMeshInc* d_inc = NULL;
		cu::cuVEC<double>* d_dirH_disH;
		cu::cuVEC<double>* d_dirV_disV;
		d_inc->Create(inc2, d_inc, d_dirH_disH, d_dirV_disV);


		// Copy MaterialDB(Host) to cuMaterialDB(Device)
		cu::cuMaterialDB* d_MatDB = NULL;
		size_t* d_Idx;
		double* d_Er_r;
		double* d_Tang;
		double* d_Mr;
		double* d_Mi;
		double* d_D;
		cuCPLX<double>* d_ER = NULL;
		cuCPLX<double>* d_MR = NULL;
		double* d_ERr_MRr_Sqrt = NULL;
		d_MatDB->Create(MatDB, d_MatDB, d_Idx, d_Er_r, d_Tang, d_Mr, d_Mi, d_D, d_ER, d_MR, d_ERr_MRr_Sqrt);


		// Copy EF(Host) to cuEF(Device)
		cu::cuEF<float>* d_Ef = NULL;
		d_Ef->Create(Ef, d_Ef);


		// Copy SAR(Host) to cuSAR(Device)
		cu::cuSAR* d_Sar = NULL;
		d_Sar->Create(Sar2[0], d_Sar);

		// Copy some parameters for Azimuth antenna pattern gain
		cuVEC<double>* d_MainBeamUV = NULL;
		cuVEC<double>* d_NorSquintPlane = NULL;
		cuVEC<double>* d_PPs = NULL;
		cuVEC<double>* d_PPt = NULL;

		d_MainBeamUV->Create(MainBeamUV, d_MainBeamUV);
		d_NorSquintPlane->Create(NorSquintPlane, d_NorSquintPlane);
		d_PPs->Create(PPs, d_PPs);
		d_PPt->Create(PPt, d_PPt);


		//+--------------------+
		//|  Memory Allocation |
		//+--------------------+
		// Malloc Tray Tracing Sets
		size_t* d_MinLevel;			// Store the minimum bounceing level for each grid
		double*  d_DisSet;			// Store the distance for each grid & level
		cuRay*  d_RaySet;			// Store the Ray class set for each grid & level
		bool* d_Shadow;				// Store the shadow or not for each grid & level (it could be deleted)
		cuTRI<float>** d_ObjSet;	// Store the intersect object set for each grid & level
		double* d_AzGain;			// Store the Azimuth antenna gain value
		SafeMallocCheck(cudaMalloc(&d_MinLevel, nPyPatch * sizeof(size_t)));
		SafeMallocCheck(cudaMalloc(&d_DisSet,   MaxLevel * nPyPatch * sizeof(double)));
		SafeMallocCheck(cudaMalloc(&d_RaySet,   MaxLevel * nPyPatch * sizeof(cuRay)));
		SafeMallocCheck(cudaMalloc(&d_ObjSet,   MaxLevel * nPyPatch * sizeof(cuTRI<float>*)));
		SafeMallocCheck(cudaMalloc(&d_Shadow,   MaxLevel * nPyPatch * sizeof(bool)));
		SafeMallocCheck(cudaMalloc(&d_AzGain,	MaxLevel * nPyPatch * sizeof(double)));

#ifdef CUDEBUG
		if(IsShow){
			printf("nPyPatch              = %ld\n", nPyPatch);
			printf("MaxLevel              = %ld\n", MaxLevel);
			printf("MaxLevel * nPyPatch   = %ld\n", MaxLevel * nPyPatch);
			printf("sizeof(size_t)        = %ld\n", sizeof(size_t));
			printf("sizeof(double)        = %ld\n", sizeof(double));
			printf("sizeof(cuRay)         = %ld\n", sizeof(cuRay));
			printf("sizeof(cuTRI<float>*) = %ld\n", sizeof(cuTRI<float>*));
			printf("sizeof(bool)          = %ld\n", sizeof(bool));
		}
#endif


		// results in device
		cuSBRElement<float>* d_res;
		SafeMallocCheck(cudaMalloc(&d_res,   Nr * MaxLevel * sizeof(cuSBRElement<float>)));
		cudaMemset(d_res, 0, Nr * MaxLevel * sizeof(cuSBRElement<float>));

		// Get CUDA memory info
		if(IsShow){
			toc(gpu_dt0, "Global Mem transfer: ");
			CheckMem("After Malloc:");
		}


		//+------------------------------------------------------------------------------------------+
		//|                                  Ray Tracing                                             |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt1 = tic();
		//+--------------------+
		//|       Kernel       |
		//+--------------------+
		//
		// Ray Tracing
		//
		// dim3 NBlock1(ceil(float(inc2.nPy)/NThread1.x));
		dim3 NBlock1(ceil(float(nPyPatch)/NThread1.x));

		if(IsShow){ CheckDim(NThread1, NBlock1, 0); }

		// 1st parameter "nPy" stands for full scene ray number
//		cuRayTracing<<<NBlock1,NThread1>>>(nPy, MaxLevel, *d_bvh, *d_inc, d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow, OffsetPatch);
		cuRayTracing2<<<NBlock1,NThread1>>>(nPy, MaxLevel, *d_bvh, *d_inc,
											Sar2[0].theta_sqc(), Sar2[0].theta_az(),
											*d_MainBeamUV, *d_NorSquintPlane, *d_PPs, *d_PPt,
											d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow, d_AzGain, OffsetPatch, true);
		cudaDeviceSynchronize();

		// Show the ray tracing information
#ifdef CUDEBUG
		size_t TotalHit;
		size_t* d_TotalHit;
		SafeMallocCheck(cudaMalloc(&d_TotalHit, sizeof(size_t)));
		cuGetTotalHit<<<1,1>>>(nPyPatch, d_MinLevel, d_TotalHit);
		cudaMemcpy(&TotalHit, d_TotalHit, sizeof(size_t), cudaMemcpyDeviceToHost);
		printf("+-------------------+\n");
		printf("nPyPatch = %ld\n", nPyPatch);
		printf("RayArea  = %.16f\n", inc2.Area);
		printf("TotalHit = %ld\n", TotalHit);
		printf("TotalHit * RayArea = %.16f\n", double(TotalHit)*inc2.Area);
		printf("+-------------------+\n");
		cudaFree(d_TotalHit);
#endif

		if(IsShow){ toc(gpu_dt1, "cuRayTracing: "); }
		//+------------------------------------------------------------------------------------------+
		//|                                  PO + Block Sum                                          |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt2 = tic();
		//+-----------------+
		//|  Set & Alloc.   |
		//+-----------------+
		// In the 1srt round, the

		// Threads / Blocks Setting
		dim3 NBlock2(ceil(float(nPyPatch)/NThread2.x), 1, 1);	// number of block in single patch grid
		size_t sbr_partial_sz = NBlock2.x * MaxLevel * sizeof(cuSBRElement<float>);

		// Allocation partial SBR algorithm results array
		cuSBRElement<float>* d_sbr_partial_array = NULL;
		cudaMalloc(&d_sbr_partial_array,     sbr_partial_sz);
		cudaMemset(d_sbr_partial_array,   0, sbr_partial_sz);



		// Get CUDA memory info
		size_t SharedSize2 = NThread2.x*MaxLevel*sizeof(cuSBRElement<float>);


		// Pre-define memory (The number of reduction round MUST <= 4 times)
		// Because the
		dim3 NBlock4[4];
		size_t SharedSize4[4]    = {0,0,0,0};
		size_t per_block_size[4] = {0,0,0,0};

		// cuBlockSum pre-calculate
		size_t rem  = ceil(float(NBlock2.x)/NThread4.x);
		size_t rem2 = rem;
		size_t count = 0;
		bool check = true;

		do{
			NBlock4[count] 		  = dim3(rem, 1, 1);
			SharedSize4[count]    = NThread4.x * MaxLevel * sizeof(cuSBRElement<float>);
			per_block_size[count] = NBlock4[count].x * SharedSize4[count];
			// Next step
			rem = ceil(float(rem)/NThread4.x);
			check = (rem != rem2);
			rem2 = rem;
			count++;
		}while(check);

		// cuBlockSum temp variables
		vector<cuSBRElement<float>*> d_sbr_per_block(4);

		cudaMalloc(&d_sbr_per_block[0], per_block_size[0]);
		cudaMalloc(&d_sbr_per_block[1], per_block_size[1]);
		cudaMalloc(&d_sbr_per_block[2], per_block_size[2]);
		cudaMalloc(&d_sbr_per_block[3], per_block_size[3]);

		//+-----------------+
		//| Wavelength Loop |
		//+-----------------+
		for(size_t idx=0;idx<Nr;++idx){
			// each stream
			size_t off = idx * MaxLevel;
#ifdef CUDEBUG
			printf("NBlock2.x = %ld, NThread2.x = %ld, SharedSize2 = %ld\n",
					NBlock2.x, NThread2.x, SharedSize2);
#endif
			// kernel excution

//			printf("inc2.Rad = %lf, (8E4 - 4) = %lf\n", inc2.Rad, 8E4 - 4);

//			cuPO<<<NBlock2, NThread2, SharedSize2>>>
//				   (k0[idx], *d_Ef, *d_bvh, *d_inc, *d_MatDB, nPy, Nr, MaxLevel,
//				   d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow,
//				   d_sbr_partial_array,
//				   OffsetPatch, IsPEC, -mesh_dRad, false);
			cuPO2<<<NBlock2, NThread2, SharedSize2>>>
				   (k0[idx], *d_Ef, *d_bvh, *d_inc, *d_MatDB, nPy, Nr, MaxLevel,
				   d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow, d_AzGain,
				   idx, d_sbr_partial_array,
				   OffsetPatch, IsPEC, -mesh_dRad, false);
			// Run, d_res : Nr * MaxLevel * sizeof(cuSBRElement<float>)
			cuSum(NBlock2, NThread4, nPyPatch, MaxLevel, off, d_sbr_partial_array, d_res,
				  d_sbr_per_block, NBlock4, SharedSize4);
			// cuSumSingleThread<<<1,1>>>(d_sbr_partial_array, MaxLevel, NBlock2, d_res);
		}// f0 block
		//+-----------------+
		//|  Store to host  |
		//+-----------------+
		cudaMemcpy(res, d_res, Nr*MaxLevel*sizeof(cuSBRElement<float>), cudaMemcpyDeviceToHost);
		if(IsShow){ toc(gpu_dt2, "cuPO + cuBlockSum: "); }
		//+------------------------------------------------------------------------------------------+
		//|                                  CUDA Free                                               |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt3 = tic();

		d_bvh->Free(d_bvh, d_primes, d_idx_poly, d_flatTree);
		d_inc->Free(d_inc, d_dirH_disH, d_dirV_disV);
		d_MatDB->Free(d_MatDB, d_Idx, d_Er_r, d_Tang, d_Mr, d_Mi, d_D, d_ER, d_MR, d_ERr_MRr_Sqrt);
		d_Ef->Free(d_Ef);
		d_Sar->Free(d_Sar);

		d_MainBeamUV->Free(d_MainBeamUV);
		d_NorSquintPlane->Free(d_NorSquintPlane);
		d_PPs->Free(d_PPs);
		d_PPt->Free(d_PPt);

		cudaFree(d_MinLevel);
		cudaFree(d_DisSet);
		cudaFree(d_RaySet);
		cudaFree(d_ObjSet);
		cudaFree(d_Shadow);
		cudaFree(d_AzGain);
		cudaFree(d_res);

		cudaFree(d_sbr_partial_array);
		cudaFree(d_sbr_per_block[0]);
		cudaFree(d_sbr_per_block[1]);
		cudaFree(d_sbr_per_block[2]);
		cudaFree(d_sbr_per_block[3]);

		// if(IsShow){
		// 	toc(gpu_dt3, "cudaFree: ");
		// 	toc(gpu_dt,  "GPU Total: ");
		// 	CheckMem("PO Free:");
		// }


	} // End of cuDoIt()


	/**
	 * @param [in] NThread1: Thread number of cuRayTracing
	 * @param [in] NThread2: Thread number of cuPO
	 * @param [in] NThread4: Thread number of cuSum
	 * @param [out] res: Return each level summing all of ray, cuSBRElement<float>[N_Freq * MaxLevel]
	 */
	void cuSBRDoIt2(const vector<SAR>& Sar2, const D1<double>& k0, const EF& Ef, const BVH& bvh, const MeshInc& inc2, const MaterialDB& MatDB,
				    const double mesh_dRad, const dim3 NThread1, const dim3 NThread2, const dim3 NThread4,
				    cuSBRElement<float>* res, const bool IsPEC,
				    const VEC<double>& MainBeamUV, const VEC<double>& NorSquintPlane, const VEC<double>& PPs, const VEC<double>& PPt,
				    const bool IsShow=false, const bool Is1st=false){
		//+------------------------------------------------------------------------------------------+
		//|                                  Check split number                                      |
		//+------------------------------------------------------------------------------------------+
		size_t nPy = inc2.nPy;
		size_t nPyProcess = 0;
		size_t nPatch = std::ceil(double(nPy) / nPyMax);
		size_t nFreq = Sar2.size();
		size_t MaxLevel = Ef.MaxLevel();

		if(Is1st){
			printf("=============================\n");
			printf("|      Patch Processing     |\n");
			printf("=============================\n");
			printf("nPy    = %ld\n", nPy);
			printf("nPyMax = %d\n",  nPyMax);
			printf("nPatch = %ld\n", nPatch);
		}

		clock_t gpu_dt = tic();

		if(nPy > nPyMax){
			// Reset destination array to be zero
			memset(res, 0, Sar2.size() * Ef.MaxLevel() * sizeof(cuSBRElement<float>));

			// Host Memory allocation
			cuSBRElement<float>* res_tmp = new cuSBRElement<float>[nFreq * MaxLevel];
			// Reset to zero
			for(size_t j=0;j<nFreq;++j){
				for(int i=0;i<MaxLevel;++i){
					res_tmp[j*MaxLevel+i].sump = cuCPLX<float>(0,0);
					res_tmp[j*MaxLevel+i].sumt = cuCPLX<float>(0,0);
				}
			}

			// For each patch
			for(size_t p=0;p<nPatch;++p){
				size_t OffsetPatch = p*nPyMax;	// nPy offset [samples]

				if(p != nPatch-1){
					nPyProcess = nPyMax;
				}else{
					nPyProcess = nPy - (nPatch-1)*nPyMax;
				}

				cuDoIt2(Sar2, k0, Ef, bvh, inc2, MatDB, 			// input
					    mesh_dRad, NThread1, NThread2, NThread4,	// input
					    res_tmp, IsPEC,								// input
					    MainBeamUV, NorSquintPlane, PPs, PPt,		// input
					    nPyProcess, OffsetPatch, IsShow); 			// output + keywords

#ifdef CUDEBUG
				printf("================================\n");
				printf("#Patch = %ld, nPy = %ld, nPatch = %ld, nPyProcess = %ld, OffsetPatch = %ld\n", p, nPy, nPatch, nPyProcess, OffsetPatch);
				CheckMem("After call cuDoIt["+num2str<size_t>(p)+"]:");
				printf("================================\n");

				printf("\n**************** Patch = %ld ******************\n", p);
				printf("Tx = %s\n", Ef.TxPol().c_str());
				for(int i=0;i<MaxLevel;++i){
					cu::cuCPLX<float> RxH = res[i].sump;
					cu::cuCPLX<float> RxV = res[i].sumt;
					printf("Level = %d\n", i);
					printf("  H%s[%d] = (% .16f,% .16f), abs = %.16f\n", Ef.TxPol().c_str(), i, RxH.r, RxH.i, RxH.abs());
					printf("  V%s[%d] = (% .16f,% .16f), abs = %.16f\n", Ef.TxPol().c_str(), i, RxV.r, RxV.i, RxV.abs());
				}
#endif

				for(size_t j=0;j<nFreq;++j){
					for(int i=0;i<MaxLevel;++i){
						res[j*MaxLevel+i].sump += res_tmp[j*MaxLevel+i].sump;
						res[j*MaxLevel+i].sumt += res_tmp[j*MaxLevel+i].sumt;
					}
				}
			}

			// delete
			delete[] res_tmp;
		}else{
			cuDoIt2(Sar2, k0, Ef, bvh, inc2, MatDB, 			// input
				    mesh_dRad, NThread1, NThread2, NThread4,	// input
				    res, IsPEC,									// input
				    MainBeamUV, NorSquintPlane, PPs, PPt,		// input
				    nPy, 0, IsShow); 							// output + keywords
#ifdef CUDEBUG
			CheckMem("After call cuDoIt:");
#endif
		}


		// toc(gpu_dt,  "GPU Total: ");
	} // End of cuSBRDoIt


	//+=======================================================+
	//|          CUDA Main (With Az Ant. Gain)                |
	//+=======================================================+
	/**
	 * @param [in] OffsetPatch: Number of offset for each patch (default = 0)
	 */
	void cuDoIt3(const vector<SAR>& Sar2, const D1<double>& k0, const EF& Ef, cuBVH*& d_bvh, const MeshInc& inc2, const MaterialDB& MatDB,
				 const double mesh_dRad, const dim3 NThread1, const dim3 NThread2, const dim3 NThread4,
				 cuSBRElement<float>* res, const bool IsPEC,
				 const VEC<double>& MainBeamUV, const VEC<double>& NorSquintPlane, const VEC<double>& PPs, const VEC<double>& PPt,
				 size_t nPyPatch = 0, const size_t OffsetPatch = 0, const bool IsShow=false){

		//+------------------------------------------------------------------------------------------+
		//|                                  Host -> Device Memory                                   |
		//+------------------------------------------------------------------------------------------+
		// Get CUDA memory info
		if(IsShow){ CheckMem("Initial:"); }


		clock_t gpu_dt = tic();
		clock_t gpu_dt0 = tic();
		//+-----------------+
		//|    Dimension    |
		//+-----------------+
		size_t Nr  = Sar2.size();
		size_t nPy = inc2.nPy;
		size_t MaxLevel = Ef.MaxLevel();

		// For default, processing full scene, no patch
		if(nPyPatch == 0){
			nPyPatch = nPy;
		}

		//+-----------------+
		//|      Copy       |
		//+-----------------+
//		// Copy BVH(Host) to cuBVH(Device)
//		cu::cuBVH* d_bvh = NULL;
//		cu::cuTRI<float>* d_primes;
//		size_t* d_idx_poly;
//		cu::cuBVHFlatNode* d_flatTree;
//		d_bvh->Create(bvh, d_bvh, d_primes, d_idx_poly, d_flatTree);


		// Copy MeshInc(Host) to cuMeshInc(Device)
		cu::cuMeshInc* d_inc = NULL;
		cu::cuVEC<double>* d_dirH_disH;
		cu::cuVEC<double>* d_dirV_disV;
		d_inc->Create(inc2, d_inc, d_dirH_disH, d_dirV_disV);


		// Copy MaterialDB(Host) to cuMaterialDB(Device)
		cu::cuMaterialDB* d_MatDB = NULL;
		size_t* d_Idx;
		double* d_Er_r;
		double* d_Tang;
		double* d_Mr;
		double* d_Mi;
		double* d_D;
		cuCPLX<double>* d_ER = NULL;
		cuCPLX<double>* d_MR = NULL;
		double* d_ERr_MRr_Sqrt = NULL;
		d_MatDB->Create(MatDB, d_MatDB, d_Idx, d_Er_r, d_Tang, d_Mr, d_Mi, d_D, d_ER, d_MR, d_ERr_MRr_Sqrt);


		// Copy EF(Host) to cuEF(Device)
		cu::cuEF<float>* d_Ef = NULL;
		d_Ef->Create(Ef, d_Ef);


		// Copy SAR(Host) to cuSAR(Device)
		cu::cuSAR* d_Sar = NULL;
		d_Sar->Create(Sar2[0], d_Sar);

		// Copy some parameters for Azimuth antenna pattern gain
		cuVEC<double>* d_MainBeamUV = NULL;
		cuVEC<double>* d_NorSquintPlane = NULL;
		cuVEC<double>* d_PPs = NULL;
		cuVEC<double>* d_PPt = NULL;

		d_MainBeamUV->Create(MainBeamUV, d_MainBeamUV);
		d_NorSquintPlane->Create(NorSquintPlane, d_NorSquintPlane);
		d_PPs->Create(PPs, d_PPs);
		d_PPt->Create(PPt, d_PPt);


		//+--------------------+
		//|  Memory Allocation |
		//+--------------------+
		// Malloc Tray Tracing Sets
		size_t* d_MinLevel;			// Store the minimum bounceing level for each grid
		double*  d_DisSet;			// Store the distance for each grid & level
		cuRay*  d_RaySet;			// Store the Ray class set for each grid & level
		bool* d_Shadow;				// Store the shadow or not for each grid & level (it could be deleted)
		cuTRI<float>** d_ObjSet;	// Store the intersect object set for each grid & level
		double* d_AzGain;			// Store the Azimuth antenna gain value
		SafeMallocCheck(cudaMalloc(&d_MinLevel, nPyPatch * sizeof(size_t)));
		SafeMallocCheck(cudaMalloc(&d_DisSet,   MaxLevel * nPyPatch * sizeof(double)));
		SafeMallocCheck(cudaMalloc(&d_RaySet,   MaxLevel * nPyPatch * sizeof(cuRay)));
		SafeMallocCheck(cudaMalloc(&d_ObjSet,   MaxLevel * nPyPatch * sizeof(cuTRI<float>*)));
		SafeMallocCheck(cudaMalloc(&d_Shadow,   MaxLevel * nPyPatch * sizeof(bool)));
		SafeMallocCheck(cudaMalloc(&d_AzGain,	MaxLevel * nPyPatch * sizeof(double)));

#ifdef CUDEBUG
		if(IsShow){
			printf("nPyPatch              = %ld\n", nPyPatch);
			printf("MaxLevel              = %ld\n", MaxLevel);
			printf("MaxLevel * nPyPatch   = %ld\n", MaxLevel * nPyPatch);
			printf("sizeof(size_t)        = %ld\n", sizeof(size_t));
			printf("sizeof(double)        = %ld\n", sizeof(double));
			printf("sizeof(cuRay)         = %ld\n", sizeof(cuRay));
			printf("sizeof(cuTRI<float>*) = %ld\n", sizeof(cuTRI<float>*));
			printf("sizeof(bool)          = %ld\n", sizeof(bool));
		}
#endif


		// results in device
		cuSBRElement<float>* d_res;
		SafeMallocCheck(cudaMalloc(&d_res,   Nr * MaxLevel * sizeof(cuSBRElement<float>)));
		cudaMemset(d_res, 0, Nr * MaxLevel * sizeof(cuSBRElement<float>));

		// Get CUDA memory info
		if(IsShow){
			toc(gpu_dt0, "Global Mem transfer: ");
			CheckMem("After Malloc:");
		}


		//+------------------------------------------------------------------------------------------+
		//|                                  Ray Tracing                                             |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt1 = tic();
		//+--------------------+
		//|       Kernel       |
		//+--------------------+
		//
		// Ray Tracing
		//
		// dim3 NBlock1(ceil(float(inc2.nPy)/NThread1.x));
		dim3 NBlock1(ceil(float(nPyPatch)/NThread1.x));

		if(IsShow){ CheckDim(NThread1, NBlock1, 0); }

		// 1st parameter "nPy" stands for full scene ray number
//		cuRayTracing<<<NBlock1,NThread1>>>(nPy, MaxLevel, *d_bvh, *d_inc, d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow, OffsetPatch);
		cuRayTracing2<<<NBlock1,NThread1>>>(nPy, MaxLevel, *d_bvh, *d_inc,
											Sar2[0].theta_sqc(), Sar2[0].theta_az(),
											*d_MainBeamUV, *d_NorSquintPlane, *d_PPs, *d_PPt,
											d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow, d_AzGain, OffsetPatch, true);
		cudaDeviceSynchronize();

		// Show the ray tracing information
#ifdef CUDEBUG
		size_t TotalHit;
		size_t* d_TotalHit;
		SafeMallocCheck(cudaMalloc(&d_TotalHit, sizeof(size_t)));
		cuGetTotalHit<<<1,1>>>(nPyPatch, d_MinLevel, d_TotalHit);
		cudaMemcpy(&TotalHit, d_TotalHit, sizeof(size_t), cudaMemcpyDeviceToHost);
		printf("+-------------------+\n");
		printf("nPyPatch = %ld\n", nPyPatch);
		printf("RayArea  = %.16f\n", inc2.Area);
		printf("TotalHit = %ld\n", TotalHit);
		printf("TotalHit * RayArea = %.16f\n", double(TotalHit)*inc2.Area);
		printf("+-------------------+\n");
		cudaFree(d_TotalHit);
#endif

		if(IsShow){ toc(gpu_dt1, "cuRayTracing: "); }
		//+------------------------------------------------------------------------------------------+
		//|                                  PO + Block Sum                                          |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt2 = tic();
		//+-----------------+
		//|  Set & Alloc.   |
		//+-----------------+
		// In the 1srt round, the

		// Threads / Blocks Setting
		dim3 NBlock2(ceil(float(nPyPatch)/NThread2.x), 1, 1);	// number of block in single patch grid
		size_t sbr_partial_sz = NBlock2.x * MaxLevel * sizeof(cuSBRElement<float>);

		// Allocation partial SBR algorithm results array
		cuSBRElement<float>* d_sbr_partial_array = NULL;
		cudaMalloc(&d_sbr_partial_array,     sbr_partial_sz);
		cudaMemset(d_sbr_partial_array,   0, sbr_partial_sz);



		// Get CUDA memory info
		size_t SharedSize2 = NThread2.x*MaxLevel*sizeof(cuSBRElement<float>);


		// Pre-define memory (The number of reduction round MUST <= 4 times)
		// Because the
		dim3 NBlock4[4];
		size_t SharedSize4[4]    = {0,0,0,0};
		size_t per_block_size[4] = {0,0,0,0};

		// cuBlockSum pre-calculate
		size_t rem  = ceil(float(NBlock2.x)/NThread4.x);
		size_t rem2 = rem;
		size_t count = 0;
		bool check = true;

		do{
			NBlock4[count] 		  = dim3(rem, 1, 1);
			SharedSize4[count]    = NThread4.x * MaxLevel * sizeof(cuSBRElement<float>);
			per_block_size[count] = NBlock4[count].x * SharedSize4[count];
			// Next step
			rem = ceil(float(rem)/NThread4.x);
			check = (rem != rem2);
			rem2 = rem;
			count++;
		}while(check);

		// cuBlockSum temp variables
		vector<cuSBRElement<float>*> d_sbr_per_block(4);

		cudaMalloc(&d_sbr_per_block[0], per_block_size[0]);
		cudaMalloc(&d_sbr_per_block[1], per_block_size[1]);
		cudaMalloc(&d_sbr_per_block[2], per_block_size[2]);
		cudaMalloc(&d_sbr_per_block[3], per_block_size[3]);

		//+-----------------+
		//| Wavelength Loop |
		//+-----------------+
		for(size_t idx=0;idx<Nr;++idx){
			// each stream
			size_t off = idx * MaxLevel;
#ifdef CUDEBUG
			printf("NBlock2.x = %ld, NThread2.x = %ld, SharedSize2 = %ld\n",
					NBlock2.x, NThread2.x, SharedSize2);
#endif
			// kernel excution

//			printf("inc2.Rad = %lf, (8E4 - 4) = %lf\n", inc2.Rad, 8E4 - 4);

//			cuPO<<<NBlock2, NThread2, SharedSize2>>>
//				   (k0[idx], *d_Ef, *d_bvh, *d_inc, *d_MatDB, nPy, Nr, MaxLevel,
//				   d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow,
//				   d_sbr_partial_array,
//				   OffsetPatch, IsPEC, -mesh_dRad, false);
			cuPO2<<<NBlock2, NThread2, SharedSize2>>>
				   (k0[idx], *d_Ef, *d_bvh, *d_inc, *d_MatDB, nPy, Nr, MaxLevel,
				   d_MinLevel, d_DisSet, d_RaySet, d_ObjSet, d_Shadow, d_AzGain,
				   idx, d_sbr_partial_array,
				   OffsetPatch, IsPEC, -mesh_dRad, false);
			// Run, d_res : Nr * MaxLevel * sizeof(cuSBRElement<float>)
			cuSum(NBlock2, NThread4, nPyPatch, MaxLevel, off, d_sbr_partial_array, d_res,
				  d_sbr_per_block, NBlock4, SharedSize4);
			// cuSumSingleThread<<<1,1>>>(d_sbr_partial_array, MaxLevel, NBlock2, d_res);
		}// f0 block
		//+-----------------+
		//|  Store to host  |
		//+-----------------+
		cudaMemcpy(res, d_res, Nr*MaxLevel*sizeof(cuSBRElement<float>), cudaMemcpyDeviceToHost);
		if(IsShow){ toc(gpu_dt2, "cuPO + cuBlockSum: "); }
		//+------------------------------------------------------------------------------------------+
		//|                                  CUDA Free                                               |
		//+------------------------------------------------------------------------------------------+
		clock_t gpu_dt3 = tic();

//		d_bvh->Free(d_bvh, d_primes, d_idx_poly, d_flatTree);
		d_inc->Free(d_inc, d_dirH_disH, d_dirV_disV);
		d_MatDB->Free(d_MatDB, d_Idx, d_Er_r, d_Tang, d_Mr, d_Mi, d_D, d_ER, d_MR, d_ERr_MRr_Sqrt);
		d_Ef->Free(d_Ef);
		d_Sar->Free(d_Sar);

		d_MainBeamUV->Free(d_MainBeamUV);
		d_NorSquintPlane->Free(d_NorSquintPlane);
		d_PPs->Free(d_PPs);
		d_PPt->Free(d_PPt);

		cudaFree(d_MinLevel);
		cudaFree(d_DisSet);
		cudaFree(d_RaySet);
		cudaFree(d_ObjSet);
		cudaFree(d_Shadow);
		cudaFree(d_AzGain);
		cudaFree(d_res);

		cudaFree(d_sbr_partial_array);
		cudaFree(d_sbr_per_block[0]);
		cudaFree(d_sbr_per_block[1]);
		cudaFree(d_sbr_per_block[2]);
		cudaFree(d_sbr_per_block[3]);

		// if(IsShow){
		// 	toc(gpu_dt3, "cudaFree: ");
		// 	toc(gpu_dt,  "GPU Total: ");
		// 	CheckMem("PO Free:");
		// }


	} // End of cuDoIt()


	/**
	 * @param [in] NThread1: Thread number of cuRayTracing
	 * @param [in] NThread2: Thread number of cuPO
	 * @param [in] NThread4: Thread number of cuSum
	 * @param [out] res: Return each level summing all of ray, cuSBRElement<float>[N_Freq * MaxLevel]
	 */
	void cuSBRDoIt3(const vector<SAR>& Sar2, const D1<double>& k0, const EF& Ef, cuBVH*& d_bvh, const MeshInc& inc2, const MaterialDB& MatDB,
					const double mesh_dRad, const dim3 NThread1, const dim3 NThread2, const dim3 NThread4,
					cuSBRElement<float>* res, const bool IsPEC,
					const VEC<double>& MainBeamUV, const VEC<double>& NorSquintPlane, const VEC<double>& PPs, const VEC<double>& PPt,
					const bool IsShow=false, const bool Is1st=false){
		//+------------------------------------------------------------------------------------------+
		//|                                  Check split number                                      |
		//+------------------------------------------------------------------------------------------+
		size_t nPy = inc2.nPy;
		size_t nPyProcess = 0;
		size_t nPatch = std::ceil(double(nPy) / nPyMax);
		size_t nFreq = Sar2.size();
		size_t MaxLevel = Ef.MaxLevel();

		if(Is1st){
			printf("=============================\n");
			printf("|      Patch Processing     |\n");
			printf("=============================\n");
			printf("nPy    = %ld\n", nPy);
			printf("nPyMax = %d\n",  nPyMax);
			printf("nPatch = %ld\n", nPatch);
		}

		clock_t gpu_dt = tic();

		if(nPy > nPyMax){
			// Reset destination array to be zero
			memset(res, 0, Sar2.size() * Ef.MaxLevel() * sizeof(cuSBRElement<float>));

			// Host Memory allocation
			cuSBRElement<float>* res_tmp = new cuSBRElement<float>[nFreq * MaxLevel];
			// Reset to zero
			for(size_t j=0;j<nFreq;++j){
				for(int i=0;i<MaxLevel;++i){
					res_tmp[j*MaxLevel+i].sump = cuCPLX<float>(0,0);
					res_tmp[j*MaxLevel+i].sumt = cuCPLX<float>(0,0);
				}
			}

			// For each patch
			for(size_t p=0;p<nPatch;++p){
				size_t OffsetPatch = p*nPyMax;	// nPy offset [samples]

				if(p != nPatch-1){
					nPyProcess = nPyMax;
				}else{
					nPyProcess = nPy - (nPatch-1)*nPyMax;
				}

				cuDoIt3(Sar2, k0, Ef, d_bvh, inc2, MatDB, 			// input
						mesh_dRad, NThread1, NThread2, NThread4,	// input
						res_tmp, IsPEC,								// input
						MainBeamUV, NorSquintPlane, PPs, PPt,		// input
						nPyProcess, OffsetPatch, IsShow); 			// output + keywords

#ifdef CUDEBUG
				printf("================================\n");
				printf("#Patch = %ld, nPy = %ld, nPatch = %ld, nPyProcess = %ld, OffsetPatch = %ld\n", p, nPy, nPatch, nPyProcess, OffsetPatch);
				CheckMem("After call cuDoIt["+num2str<size_t>(p)+"]:");
				printf("================================\n");

				printf("\n**************** Patch = %ld ******************\n", p);
				printf("Tx = %s\n", Ef.TxPol().c_str());
				for(int i=0;i<MaxLevel;++i){
					cu::cuCPLX<float> RxH = res[i].sump;
					cu::cuCPLX<float> RxV = res[i].sumt;
					printf("Level = %d\n", i);
					printf("  H%s[%d] = (% .16f,% .16f), abs = %.16f\n", Ef.TxPol().c_str(), i, RxH.r, RxH.i, RxH.abs());
					printf("  V%s[%d] = (% .16f,% .16f), abs = %.16f\n", Ef.TxPol().c_str(), i, RxV.r, RxV.i, RxV.abs());
				}
#endif

				for(size_t j=0;j<nFreq;++j){
					for(int i=0;i<MaxLevel;++i){
						res[j*MaxLevel+i].sump += res_tmp[j*MaxLevel+i].sump;
						res[j*MaxLevel+i].sumt += res_tmp[j*MaxLevel+i].sumt;
					}
				}
			}

			// delete
			delete[] res_tmp;
		}else{
			cuDoIt3(Sar2, k0, Ef, d_bvh, inc2, MatDB, 			// input
					mesh_dRad, NThread1, NThread2, NThread4,	// input
					res, IsPEC,									// input
					MainBeamUV, NorSquintPlane, PPs, PPt,		// input
					nPy, 0, IsShow); 							// output + keywords
#ifdef CUDEBUG
			CheckMem("After call cuDoIt:");
#endif
		}


		// toc(gpu_dt,  "GPU Total: ");
	} // End of cuSBRDoIt




}  // namespace cu




#endif /* CUMAIN_CUH_ */
