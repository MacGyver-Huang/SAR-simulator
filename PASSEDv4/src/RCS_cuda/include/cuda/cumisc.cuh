/*
 * cukernel.cuh
 *
 *  Created on: Oct 16, 2014
 *      Author: cychiang
 */

#ifndef CUMISC_CUH_
#define CUMISC_CUH_

#include <cmath>
#include <cuda/cuvec.cuh>
#include <cuda/cuclass.cuh>
#include <cuda/cuopt.cuh>
#include <sar/sar.h>



namespace cu {
	//+======================================================+
	//|                      __host__                        |
	//+======================================================+
	__device__ double sinc(const double in){
		// Sinc function
		double tmp = def::PI*(double)in;
		//if((tmp == 0.) || (tmp < 1E-15)){
		if(tmp == 0.){
			return 1;
		}else{
			return sin(tmp)/(tmp);
		}
	}

	__device__ double sinc_no_pi(const double in){
		// Sinc function
		double tmp = (double)in;
		//if((tmp == 0.) || (tmp < 1E-15)){
		if(tmp == 0.){
			return 1;
		}else{
			return sin(tmp)/(tmp);
		}
	}

	void GPUSelect(const bool SHOW=false){
		int num_devices, device;
		cudaGetDeviceCount(&num_devices);
		int max_device = 0;
		if(num_devices > 1){
			int max_multiprocessors = 0;
			for(device = 0; device < num_devices; device++){
				cudaDeviceProp properties;
				cudaGetDeviceProperties(&properties, device);
				if(max_multiprocessors < properties.multiProcessorCount){
					max_multiprocessors = properties.multiProcessorCount;
					max_device = device;
				}
			}
		}
		cudaSetDevice(max_device);
		cudaDeviceReset();

		if(SHOW){
			cout<<"GPU:"<<max_device<<" was selected!"<<endl;
		}
	}

	string GetGPUName(){
		cudaDeviceProp properties;
		int device;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&properties, device);
		return string(properties.name);
	}

	void GetGPUParameters(dim3& NThread1, dim3& NThread2, dim3& NThread3, dim3& NThread4, const bool IsShow=false){
		// Get GPU name
		string GPUName = GetGPUName();

		// Default
		NThread1 = dim3(64,1,1);
		NThread2 = dim3(64,1,1);
		NThread3 = dim3(64,1,1);
		NThread4 = dim3(64,1,1);

		// GeForce GTX 295
		if(GPUName == string("GeForce GTX 295")){
			NThread1 = dim3(64,1,1);
			NThread2 = dim3(64,1,1);
			NThread3 = dim3(64,1,1);
			NThread4 = dim3(64,1,1);
		}

// 		// GeForce GTX TITAN
// 		if(GPUName == string("GeForce GTX TITAN")){
// 			NThread1 = dim3(128,1,1);	// cuRayTracing
// 			NThread2 = dim3(64,1,1);	// cuPOApprox(cuPO1)
// //			NThread2 = dim3(128,1,1);	// cuPOApprox(cuPO1)
// //			NThread2 = dim3(256,1,1);	// cuPOApprox(cuPO1)
// 			NThread3 = dim3(128,1,1);	// cuPreCal, cuPORefl
// //			NThread3 = dim3(256,1,1);	// cuPreCal, cuPORefl
// //			NThread3 = dim3(512,1,1);	// cuPreCal, cuPORefl
// //			NThread4 = dim3(64,1,1);	// cuSum2(cuBlockSum3)
// 			NThread4 = dim3(128,1,1);	// cuSum2(cuBlockSum3)
// //			NThread4 = dim3(256,1,1);	// cuSum2(cuBlockSum3)
// //			NThread4 = dim3(256,1,1);	// cuSum2(cuBlockSum3)
// //			NThread4 = dim3(512,1,1);	// cuSum2(cuBlockSum3)
// //			NThread4 = dim3(1024,1,1);	// cuSum2(cuBlockSum3)
// 		}

		// GeForce GTX 1080Ti
		if(GPUName == string("GeForce GTX 1080 Ti")){
			NThread1 = dim3(128,1,1);
			NThread2 = dim3(64,1,1);
			NThread3 = dim3(128,1,1);
			NThread4 = dim3(128,1,1);
		}

		// GeForce GTX TITAN
		if(GPUName == string("TITAN X (Pascal)")){
			NThread1 = dim3(128,1,1);	// cuRayTracing
			NThread2 = dim3(32,1,1);	// cuPO
//			NThread1 = dim3(64,1,1);	// cuRayTracing
//			NThread2 = dim3(64,1,1);	// cuPO
//			NThread2 = dim3(128,1,1);	// cuPO
			NThread4 = dim3(128,1,1);	// cuSum(cuBlockSum)
		}

		// Quadro GV100
		if(GPUName == string("Quadro GV100")){
			NThread1 = dim3(128,1,1);	// cuRayTracing
			NThread2 = dim3(128,1,1);	// cuPO
			NThread4 = dim3(128,1,1);	// cuSum(cuBlockSum)
		}

		if(IsShow){
			cout<<"+--------------------------+"<<endl;
			cout<<"|    GPU Device Summary    |"<<endl;
			cout<<"+--------------------------+"<<endl;
			cout<<" Name                    : "<<GPUName<<endl;
			cout<<" NThread1 [cuRayTracing] : ("<<NThread1.x<<","<<NThread1.y<<","<<NThread1.z<<")"<<endl;
			cout<<" NThread2 [cuPO]         : ("<<NThread2.x<<","<<NThread2.y<<","<<NThread2.z<<")"<<endl;
			cout<<" NThread3 [cuBlockSum]   : ("<<NThread3.x<<","<<NThread3.y<<","<<NThread3.z<<")"<<endl;
		}
	}

	void CheckDim(const dim3 NThread, const dim3 NBlock, int DevID=0, bool DevMessage=false, bool DetailMessage=false){
		cout<<"+--------------------------+"<<endl;
		cout<<"|   Check CUDA Dimension   |"<<endl;
		cout<<"+--------------------------+"<<endl;
		//
		// Get device properties
		//
		cudaDeviceProp dev;
		cudaGetDeviceProperties(&dev, DevID);
		if(DevMessage){
			cout<<"+"<<endl;
			cout<<" Device Spec."<<endl;
			cout<<"+"<<endl;
			cout<<"maxThreadsPerBlock = "<<dev.maxThreadsPerBlock<<endl;
			cout<<"maxThreadsDim      = ("<<dev.maxThreadsDim[0]<<","<<dev.maxThreadsDim[1]<<","<<dev.maxThreadsDim[2]<<")"<<endl;
			cout<<"maxGridSize        = ("<<dev.maxGridSize[0]<<","<<dev.maxGridSize[1]<<","<<dev.maxGridSize[2]<<")"<<endl;
			cout<<"+"<<endl;
			cout<<"shared Mem Per Block          = "<<dev.sharedMemPerBlock<<" [bytes]"<<endl;
			cout<<"shared Mem Per Multiprocessor = "<<dev.sharedMemPerMultiprocessor<<" [bytes]"<<endl;
		}
		//
		// Check block/thread dimension
		//
		string TagTotalThread, TagThreadSize, TagBlockSize;
		bool IsTotalThread, IsThreadSize, IsBlockSize;
		size_t TotalThreads = NThread.x * NThread.y * NThread.z;
		if(TotalThreads > dev.maxThreadsPerBlock){
			if(DetailMessage)cout<<"ERROR::Total design threads("<<TotalThreads<<") exceeds device max thread per block("<<dev.maxThreadsPerBlock<<")"<<endl;
			TagTotalThread = "[Failure]";
			IsTotalThread  = false;
		}else{
			if(DetailMessage)cout<<"OK::Total design threads("<<TotalThreads<<") lower than device max thread per block("<<dev.maxThreadsPerBlock<<")"<<endl;
			TagTotalThread = "[OK]";
			IsTotalThread  = true;
		}
		if(NThread.x > dev.maxThreadsDim[0] ||
		   NThread.y > dev.maxThreadsDim[1] ||
		   NThread.z > dev.maxThreadsDim[2]){
			if(DetailMessage)cout<<"ERROR::Thread size("<<NThread.x<<","<<NThread.y<<","<<NThread.z<<") is exceeds max size in device"<<
				  "("<<dev.maxThreadsDim[0]<<","<<dev.maxThreadsDim[1]<<","<<dev.maxThreadsDim[2]<<")"<<endl;
			TagThreadSize = "[Failure]";
			IsThreadSize  = false;
		}else{
			if(DetailMessage)cout<<"OK::Thread size("<<NThread.x<<","<<NThread.y<<","<<NThread.z<<") is lower than max size in device"<<
				  "("<<dev.maxThreadsDim[0]<<","<<dev.maxThreadsDim[1]<<","<<dev.maxThreadsDim[2]<<")"<<endl;
			TagThreadSize = "[OK]";
			IsThreadSize  = true;
		}
		if(NBlock.x > dev.maxGridSize[0] ||
		   NBlock.y > dev.maxGridSize[1] ||
		   NBlock.z > dev.maxGridSize[2]){
			if(DetailMessage)cout<<"ERROR::Block size("<<NBlock.x<<","<<NBlock.y<<","<<NBlock.z<<") is exceeds max size in device"<<
				  "("<<dev.maxGridSize[0]<<","<<dev.maxGridSize[1]<<","<<dev.maxGridSize[2]<<")"<<endl;
			TagBlockSize = "[Failure]";
			IsBlockSize  = false;
		}else{
			if(DetailMessage)cout<<"OK::Block size("<<NBlock.x<<","<<NBlock.y<<","<<NBlock.z<<") is lower than max size in device"<<
				  "("<<dev.maxGridSize[0]<<","<<dev.maxGridSize[1]<<","<<dev.maxGridSize[2]<<")"<<endl;
			TagBlockSize = "[OK]";
			IsBlockSize  = true;
		}

		cout<<"+"<<endl;
		cout<<" User Allocated"<<endl;
		cout<<"+"<<endl;
		cout<<"Total thread       = "<<TotalThreads<<" "<<TagTotalThread<<endl;
		cout<<"NThread            = ("<<NThread.x<<","<<NThread.y<<","<<NThread.z<<") "<<TagThreadSize<<endl;
		cout<<"NBlock             = ("<<NBlock.x<<","<<NBlock.y<<","<<NBlock.z<<") "<<TagBlockSize<<endl;
		cout<<"-"<<endl;
		cout<<"Total Threads      = ("<<NBlock.x*NThread.x<<","<<NBlock.y*NThread.y<<","<<NBlock.z*NThread.z<<")"<<endl;
		if((!IsTotalThread) || (!IsThreadSize) || (!IsBlockSize)){
			cout<<">> Abort - Failure <<"<<endl;
			exit(EXIT_FAILURE);
		}
		cout<<endl;
	}

	void CheckMem(const string title=""){
		size_t mem_free, mem_total, mem_use;
		cudaMemGetInfo(&mem_free, &mem_total);
		mem_use = mem_total - mem_free;
		if(title != ""){
			cout<<"+"<<endl;
			cout<<" "<<title<<endl;
		}
		cout<<"+"<<endl;
		cout<<"mem Free  = "<<double(mem_free)/1024./1024.<<" [MBytes]"<<endl;
		cout<<"mem usage = "<<double(mem_use) /1024./1024.<<" [MBytes]"<<endl;
		cout<<"+"<<endl;
	}

	//+======================================================+
	//|                     __device__                       |
	//+======================================================+
	template<typename T>
	__device__
	void AddPhase(cuVEC<cuCPLX<T> >& cplx, const double phs){
		double sp, cp;
//		sincos(+phs, &sp, &cp);
		sincos(-phs, &sp, &cp);

		cuCPLX<T> phase(cp, sp);

		cplx.x = cplx.x * phase;
		cplx.y = cplx.y * phase;
		cplx.z = cplx.z * phase;

//		cplx.x.r *= cp;
//		cplx.x.i *= sp;
//		cplx.y.r *= cp;
//		cplx.y.i *= sp;
//		cplx.z.r *= cp;
//		cplx.z.i *= sp;
	}

	template<typename T>
	__device__
	void cuCPLXMulSelf(T& L_r, T& L_i, const T& R_r, const T& R_i){
		T L2_r = L_r;
		T L2_i = L_i;
		L_r = L2_r*R_r - L2_i*R_i;
		L_i = L2_r*R_i + L2_i*R_r;
	}

	template<typename T>
	__device__
	void cuCPLXMul(const T& L_r, const T& L_i, const T& R_r, const T& R_i, T& out_r, T& out_i){
		out_r = L_r*R_r - L_i*R_i;
		out_i = L_r*R_i + L_i*R_r;
	}

	__device__
	void AddPhase(float& cplx_x_r, float& cplx_x_i,
				  float& cplx_y_r, float& cplx_y_i,
				  float& cplx_z_r, float& cplx_z_i,
				  const double phs){
		double sp, cp;
//		sincos(+phs, &sp, &cp);
		sincos(-phs, &sp, &cp);

		cuCPLX<float> phase(cp, sp);

//		cplx.x = cplx.x * phase;
//		cplx.y = cplx.y * phase;
//		cplx.z = cplx.z * phase;
		cuCPLXMulSelf(cplx_x_r, cplx_x_i, phase.r, phase.i);
		cuCPLXMulSelf(cplx_y_r, cplx_y_i, phase.r, phase.i);
		cuCPLXMulSelf(cplx_z_r, cplx_z_i, phase.r, phase.i);

//		cplx.x.r *= cp;
//		cplx.x.i *= sp;
//		cplx.y.r *= cp;
//		cplx.y.i *= sp;
//		cplx.z.r *= cp;
//		cplx.z.i *= sp;
	}

	__device__
	void AddPhaseMul(cuVEC<cuCPLX<float> >& cplx, const double phs, const float factor){
		double sp, cp;
		sincos(+phs, &sp, &cp);

		cuCPLX<float> phase(cp*factor, sp*factor);

		cplx.x = cplx.x * phase;
		cplx.y = cplx.y * phase;
		cplx.z = cplx.z * phase;
	}

	template<typename T>
	__device__
	void cuCPLXDiv(const cuCPLX<T>& L,const cuCPLX<T>& R, T& out_r, T& out_i){
		T R_conj_R = R.r*R.r + R.i*R.i;
		out_r = (L.r*R.r + L.i*R.i)/R_conj_R;
		out_i = (L.i*R.r - L.r*R.i)/R_conj_R;
	}

	__device__
	cuThetaPhiVec ThetaPhiVector(const cuVEC<double>& k){
		cuThetaPhiVec out;
		cuVEC<float> kk = -k;
		float xy = sqrtf(kk.x*kk.x + kk.y*kk.y);
		float st = xy;
		float ct = kk.z;
		float sp = kk.y/xy;
		float cp = kk.x/xy;

		out.theta_vec = cuVEC<float>(ct*cp, ct*sp, -st);
		out.phi_vec   = cuVEC<float>(-sp, cp, 0);

		return out;
	}

	__device__
	cuThetaPhiVec ThetaPhiVector(const double kx, const double ky, const double kz){
		cuThetaPhiVec out;
//		cuVEC<float> kk = -k;
//		float xy = sqrtf(kk.x*kk.x + kk.y*kk.y);
//		float st = xy;
//		float ct = kk.z;
//		float sp = kk.y/xy;
//		float cp = kk.x/xy;
		float xy = sqrtf(kx*kx + ky*ky);
		float st = xy;
		float ct = -kz;
		float sp = -ky/xy;
		float cp = -kx/xy;

		out.theta_vec = cuVEC<float>(ct*cp, ct*sp, -st);
		out.phi_vec   = cuVEC<float>(-sp, cp, 0);

		return out;
	}

	__device__
	void ThetaPhiVector(const double kx, const double ky, const double kz, cuThetaPhiVec& out){
		float xy = sqrtf(kx*kx + ky*ky);
		float st = xy;
		float ct = -kz;
		float sp = -ky/xy;
		float cp = -kx/xy;

		out.theta_vec.x =  ct*cp;
		out.theta_vec.y =  ct*sp;
		out.theta_vec.z = -st;

		out.phi_vec.x = -sp;
		out.phi_vec.y =  cp;
		out.phi_vec.z =   0;
	}

	template<typename T>
	__device__
	cuVEC<T> transform(float sa, float ca, float sb, float cb, const cuVEC<T>& in){
		cuVEC<T> tmp( ca * in.x + sa * in.y,
					 -sa * in.x + ca * in.y,
					  in.z );

		cuVEC<T> out( cb * tmp.x - sb * tmp.z,
					  tmp.y,
					  sb * tmp.x + cb * tmp.z );
		return out;
	}

	template<typename T, typename T2>
	__device__
	void transform(float sa, float ca, float sb, float cb, const T2 inx, const T2 iny, const T2 inz, cuVEC<T>& out){

		T tmpx =  ca * inx + sa * iny;
//		T tmpy = -sa * inx + ca * iny;
//		T tmpz = inz;

		out.x = cb * tmpx - sb * inz;
//		out.y = tmpy;
		out.y = -sa * inx + ca * iny;
		out.z = sb * tmpx + cb * inz;
//		return out;
	}

	template<typename T>
	__device__
	cuVEC<T> spher2cart(const T st, const T ct, const T sp, const T cp, const T R){
		return cuVEC<T>(R*st*cp, R*st*sp, R*ct);
	}

	template<typename T>
	__device__
	cuVEC<T> spher2cartOne(const T st, const T ct, const T sp, const T cp){
		return cuVEC<T>(st*cp, st*sp, ct);
	}

	__device__
	float sign(const float in){
		if(in < 0.f){
			return -1.f;
		}else{
			return 1.f;
		}
	}

	template<typename T>
	__device__
	cuCPLX<T> dot(const cuVEC<cuCPLX<T> >& a, const cuVEC<T>& b){
		T r = a.x.r*b.x + a.y.r*b.y + a.z.r*b.z;
		T i = a.x.i*b.x + a.y.i*b.y + a.z.i*b.z;
		return cuCPLX<T>(r,i);
	}

	template<typename T>
	__device__
	void dot(const cuVEC<cuCPLX<T> >& a, const cuVEC<T>& b, cuCPLX<T>& out){
		out.r = a.x.r*b.x + a.y.r*b.y + a.z.r*b.z;
		out.i = a.x.i*b.x + a.y.i*b.y + a.z.i*b.z;
	}

	template<typename T>
	__device__
	void dot(const cuVEC<cuCPLX<T> >& a, const cuVEC<T>& b, T& out_r, T& out_i){
		out_r = a.x.r*b.x + a.y.r*b.y + a.z.r*b.z;
		out_i = a.x.i*b.x + a.y.i*b.y + a.z.i*b.z;
	}

	template<typename T>
	__device__
	void dot(const cuVEC<cuCPLX<T> >& a, const T bx, const T by, const T bz, T& out_r, T& out_i){
		out_r = a.x.r*bx + a.y.r*by + a.z.r*bz;
		out_i = a.x.i*bx + a.y.i*by + a.z.i*bz;
	}

	template<typename T>
	__device__
	void dot(const cuVEC<cuCPLX<T> >& a, const T bx, const T by, T& out_r, T& out_i){
		out_r = a.x.r*bx + a.y.r*by;
		out_i = a.x.i*bx + a.y.i*by;
	}

	template<typename T>
	__device__
	void dot(const T& axr, const T& axi,
			 const T& ayr, const T& ayi,
			 const T& azr, const T& azi,
			 const cuVEC<T>& b, cuCPLX<T>& out){
		out.r = axr*b.x + ayr*b.y + azr*b.z;
		out.i = axi*b.x + ayi*b.y + azi*b.z;
	}

	template<typename T>
	__device__
	void dot(const T& axr, const T& axi,
			 const T& ayr, const T& ayi,
			 const T& azr, const T& azi,
			 const T& bx, const T& by, const T& bz,
			 T& out_r, T& out_i){
		out_r = axr*bx + ayr*by + azr*bz;
		out_i = axi*bx + ayi*by + azi*bz;
	}

	template<typename T>
	__device__
	cuCPLX<T> dot(const cuCPLX<T>& ax, const cuCPLX<T>& ay, const cuCPLX<T>& az, const cuVEC<T>& b){
		T r = ax.r*b.x + ay.r*b.y + az.r*b.z;
		T i = ax.i*b.x + ay.i*b.y + az.i*b.z;
		return cuCPLX<T>(r,i);
	}

	template<typename T>
	__device__
	cuVEC<T> abs(const cuVEC<cuCPLX<T> >& in){
		cuVEC<T> out;
		cuCPLX<T> tmp;
		tmp = in.x; out.x = tmp.abs();
		tmp = in.y; out.y = tmp.abs();
		tmp = in.z; out.z = tmp.abs();
		return out;
	}

	__device__
	double factrl(const int n){
//		const double d_FACTORIAL[] = {1,1,2,6,24,120,720,5040,40320,362880,3628800};
		if(n<=20){
			return cu::opt::c_FACTORIAL[n];
		}else{
			printf("ERROR::cu::factrl:The input number MUST little than 10.\n>>> Abort - Failure <<<\n");
			return 1.0;
//			return (n*factrl(n-1));
		}
	}

	__device__
	float PowInt(const float val, const int pow){
//		const float table[] = {val, 1, 1, 1, 1, 1, 1, 1, 1, 1};
		float out = 1.f;
		for(int i=1;i<=pow;++i){
			out *= val;
		}
//		for(int i=0;i<pow;++i){
//			out *= (out * ((i<1)? val:1) );
//		}
		return out;

//		float out = val;
//		if(pow == 0){
//			return 1;
//		} else if(pow == 1){
//			return val;
//		}else{
//			for(int i=2;i<=pow;++i){
////				out *= out;
//				out *= val;
//			}
//			return out;
//		}
	}

	template<typename T>
	__device__
	cuCPLX<T> PowJPhase(const T img, const int pow){
#ifdef LIMITNUMBER
		T tmp = PowInt(img, pow);
		return cuCPLX<T>(( cu::opt::c_Is[pow]) * cu::opt::c_ev[pow]*tmp,
						 (!cu::opt::c_Is[pow]) * cu::opt::c_od[pow]*tmp);
#else
//		if((pow % 2) == 0){	// assign to real_part
//			T sign = (((pow-2) % 4) == 0)? -1:1;
//			return cuCPLX<T>(sign*PowInt(img, pow),0);
//		}else{				// assign to imag_part
//			T sign = (((pow-3) % 4) == 0)? -1:1;
//			return cuCPLX<T>(0,sign*PowInt(img, pow));
//		}
		bool IsEven = ((pow % 2) == 0);
		T sign_even = (((pow-2) % 4) == 0)? -1:1;
		T sign_odd  = (((pow-3) % 4) == 0)? -1:1;
		return cuCPLX<T>((IsEven)*(sign_even*PowInt(img, pow)) + (!IsEven)*0,
						 (IsEven)*0                            + (!IsEven)*(sign_odd*PowInt(img, pow)) );
#endif
	}

	template<typename T>
	__device__
	void PowJPhase(const T img, const int pow, T& out_r, T& out_i){
#ifdef LIMITNUMBER
		T tmp = PowInt(img, pow);
		out_r = ( cu::opt::c_Is[pow]) * cu::opt::c_ev[pow]*tmp;
		out_i = (!cu::opt::c_Is[pow]) * cu::opt::c_od[pow]*tmp;
#else
//		if((pow % 2) == 0){	// assign to real_part
//			T sign = (((pow-2) % 4) == 0)? -1:1;
//			return cuCPLX<T>(sign*PowInt(img, pow),0);
//		}else{				// assign to imag_part
//			T sign = (((pow-3) % 4) == 0)? -1:1;
//			return cuCPLX<T>(0,sign*PowInt(img, pow));
//		}
		bool IsEven = ((pow % 2) == 0);
		T sign_even = (((pow-2) % 4) == 0)? -1:1;
		T sign_odd  = (((pow-3) % 4) == 0)? -1:1;

		out_r = ( IsEven) * sign_even*PowInt(img, pow);
		out_i = (!IsEven) * sign_odd*PowInt(img, pow);
#endif
	}

	template<typename T>
	__device__
	cuCPLX<T> PowJPhasexCPLX(const T img, const int pow, const cuCPLX<T>& c){
#ifdef LIMITNUMBER
		bool IsEven = cu::opt::c_Is[pow];
		T v = cu::opt::c_ev[pow]*cu::opt::c_od[pow]*PowInt(img, pow);
		T v_cr = v*c.r;
		T v_ci = v*c.i;
		return cuCPLX<T>((IsEven)*(v_cr) + (!IsEven)*(-v_ci),
						 (IsEven)*(v_ci) + (!IsEven)*( v_cr) );
#else
//		T sign;
//		if((pow % 2) == 0){	// assign to real_part
//			sign = (((pow-2) % 4) == 0)? -1:1;
//			T v = sign*PowInt(img, pow);
//			return cuCPLX<T>(v*c.r, v*c.i);
//		}else{				// assign to imag_part
//			sign = (((pow-3) % 4) == 0)? -1:1;
//			T v = sign*PowInt(img, pow);
//			return cuCPLX<T>(-v*c.i, v*c.r);
//		}
		bool IsEven = ((pow % 2) == 0);
		T sign_even = (((pow-2) % 4) == 0)? -1:1;
		T sign_odd  = (((pow-3) % 4) == 0)? -1:1;
		T v = sign_even*sign_odd*PowInt(img, pow);
		T v_cr = v*c.r;
		T v_ci = v*c.i;
		return cuCPLX<T>((IsEven)*(v_cr) + (!IsEven)*(-v_ci),
						 (IsEven)*(v_ci) + (!IsEven)*( v_cr) );
#endif
	}

	template<typename T>
	__device__
	void PowJPhasexCPLX(const T img, const int pow, const cuCPLX<T>& c, T& out_r, T& out_i){
#ifdef LIMITNUMBER
		T v = cu::opt::c_ev[pow]*cu::opt::c_od[pow]*PowInt(img, pow);

		T vcr = v*c.r;
		T vci = v*c.i;

		out_r = (cu::opt::c_Is[pow]) * vcr - (!cu::opt::c_Is[pow]) * vci;
		out_i = (cu::opt::c_Is[pow]) * vci + (!cu::opt::c_Is[pow]) * vcr;
#else
//		T sign;
//		if((pow % 2) == 0){	// assign to real_part
//			sign = (((pow-2) % 4) == 0)? -1:1;
//			T v = sign*PowInt(img, pow);
//			return cuCPLX<T>(v*c.r, v*c.i);
//		}else{				// assign to imag_part
//			sign = (((pow-3) % 4) == 0)? -1:1;
//			T v = sign*PowInt(img, pow);
//			return cuCPLX<T>(-v*c.i, v*c.r);
//		}
		bool IsEven = ((pow % 2) == 0);
		T sign_even = (((pow-2) % 4) == 0)? -1:1;
		T sign_odd  = (((pow-3) % 4) == 0)? -1:1;
		T v = sign_even*sign_odd*PowInt(img, pow);
		T vcr = v*c.r;
		T vci = v*c.i;

		out_r = (sign_even) * vcr - (!sign_even) * vci;
		out_i = (sign_even) * vci + (!sign_even) * vcr;
#endif
	}

	template<typename T>
	__device__
	cuCPLX<T> PowCPLX(const cuCPLX<T>& in, const int pow){
		cuCPLX<T> out(1,0);	// n = 0
		if(pow > 0){
			for(int i=1;i<pow+1;++i){
				out = out * in;
			}
		}
		return out;
	}

	__device__
	cuCPLX<double> PowCPLXImg(const double img, const int Pow){
		// pow = [0,1,2,3,...19,20]
//		__constant__ double pow_i[] = {  1, -1, -1, 1,
//										 1, -1, -1, 1,
//										 1, -1, -1, 1,
//										 1, -1, -1, 1,
//										 1, -1, -1, -1,
//										 1};
		double tmp = pow(img,double(Pow)) * cu::opt::c_powi[Pow];
//		double tmp = pow(img,double(Pow));
//		tmp *= cu::opt::c_powi[Pow];

		if((Pow % 2) != 0){	// IsOdd
			return cuCPLX<double>(0, tmp);
		}else{
			return cuCPLX<double>(tmp, 0);
		}
	}

	template<typename T>
	__device__
	cuCPLX<T> G(const int n, const T w, const cuCPLX<T>& expjw){
		T w1 = 1.f/w;
		T w2 = T(n) * w1;
		cuCPLX<T> expjw_jw(expjw.i*w1, -expjw.r*w1);
		cuCPLX<T> g(expjw_jw.r, expjw_jw.i + w1);
		T g_r;
		for(int m=0;m<n;++m){
			g_r = g.r;
			// fmaf(x,y,z) == x*y+z
			g.r = fmaf(g.i, (-w2), expjw_jw.r);
			g.i = fmaf(g_r, ( w2), expjw_jw.i);
		}
		return g;
	}

//	__device__
//	cuCPLX<float> G2(const int n, const float w){
//		cuCPLX<double> jw(0,w);
//		double cw, sw;
//		sincos(double(w), &sw, &cw);
//		cuCPLX<double> expjw(cw, sw);
//
//		(expjw - cuCPLX<double>(1,0))
//
//		cuCPLX<double> sum(0,0);
//
//		for(int i=0;i<=n;++i){
//			sum += cu::PowCPLX(_jw, i) / cu::factrl(i);
////			sum += cu::PowCPLXImg(-w, i) / cu::factrl(i);
//		}
//
//		cuCPLX<double> g = cu::factrl(n)/cu::PowCPLX(_jw, n+1) * (double(1) - expjw*sum);
////		cuCPLX<double> g = cu::factrl(n)/cu::PowCPLXImg(-w, n+1) * (double(1) - expjw*sum);
//		return cuCPLX<float>(g.r, g.i);
//	}

	__device__
	cuCPLX<float> G2(const int n, const float w){
//		double w = -Dq;
//		int n = 5;
		cuCPLX<double> jw(0,w);
		double cw, sw;
		sincos(double(w), &sw, &cw);
		cuCPLX<double> expjw(cw, sw);

		cuCPLX<double> g0 = (expjw - cuCPLX<double>(1,0));
		cuCPLX<double> g = g0/jw;
		cuCPLX<double> go;

		if(n > 0){
			for(int m=1; m<=n; ++m){
				go = g;
				g = (expjw - (double)n*go)/jw;
			}
		}

		return cuCPLX<float>(g.r, g.i);
	}

	__device__
	cuCPLX<float> G3(const int n, const float w){
//		cuCPLX<double> j(0,1);
		cuCPLX<double> _jw(0,-w);
		double cw, sw;
		sincos(double(w), &sw, &cw);
		cuCPLX<double> expjw(cw, sw);
		cuCPLX<double> sum(0,0);

		for(int i=0;i<=n;++i){
			sum += cu::PowCPLX(_jw, i) / cu::factrl(i);
//			sum += cu::PowCPLXImg(-w, i) / cu::factrl(i);
		}

		cuCPLX<double> g = cu::factrl(n)/cu::PowCPLX(_jw, n+1) * (double(1) - expjw*sum);
//		cuCPLX<double> g = cu::factrl(n)/cu::PowCPLXImg(-w, n+1) * (double(1) - expjw*sum);
		return cuCPLX<float>(g.r, g.i);
	}

	template<typename T>
	__device__
	cuCPLX<T> G(const int n, const T w, const T expjw_r, const T expjw_i){
		T w1 = 1.f/w;
		T w2 = T(n) * w1;
		cuCPLX<T> expjw_jw(expjw_i*w1, -expjw_r*w1);
		cuCPLX<T> g(expjw_jw.r, expjw_jw.i + w1);
		T G_r;
		for(int m=0;m<n;++m){
			G_r = g.r;
			// fmaf(x,y,z) == x*y+z
			g.r = fmaf(g.i, (-w2), expjw_jw.r);
			g.i = fmaf(G_r, ( w2), expjw_jw.i);
		}
		return g;
	}

	template<typename T>
	__device__
	void G(const int n, const T w, const T expjw_r, const T expjw_i, cuCPLX<T>& g){
		T w1 = 1.f/w;
		T w2 = T(n) * w1;
		cuCPLX<T> expjw_jw(expjw_i*w1, -expjw_r*w1);
		g.r = expjw_jw.r;
		g.i = expjw_jw.i + w1;
		T G_r;
#pragma unroll
		for(int m=0;m<n;++m){
			G_r = g.r;
			// fmaf(x,y,z) == x*y+z
			g.r = fmaf(g.i, (-w2), expjw_jw.r);
			g.i = fmaf(G_r, ( w2), expjw_jw.i);
		}
	}

	template<class T>
	__device__
	cuCPLX<T> Exp(cuCPLX<T>& __x){
	    T __i = __x.i;
	    if (::isinf(__x.r)){
	        if (__x.r < T(0)){
	            if (!::isfinite(__i))
	                __i = T(1);
	        }else if (__i == 0 || !::isfinite(__i)){
	            if (::isinf(__i))
	                __i = T(NAN);
	            return cuCPLX<T>(__x.r, __i);
	        }
	    }else if (::isnan(__x.r) && __x.i == 0){
	        return __x;
	    }
	    T __e = exp(__x.r);
	    return cuCPLX<T>(__e * cos(__i), __e * sin(__i));
	}

	template<typename T>
	__device__
	cuCPLX<T> G(const int n, const T w){
		cuCPLX<T> j(0,1);
		cuCPLX<T> jw = j*w;
		cuCPLX<T> g  = (Exp(jw)-1)/jw;

		if(n>0){
			for(int m=1;m<=n;++m){
				g=(Exp(jw)-n*g)/jw;
			}
		}
		
		return g;
	}

	template<typename T>
	__device__
	cuCPLX<T> G3(const int n, const T w){
		cuCPLX<T> j(0,1);
		cuCPLX<T> jw = j*w;
		cuCPLX<T> sum(0,0);

		for(int i=0;i<n;++i){
			sum +=  PowCPLX(-jw, i)/factrl(i);
		}

		cuCPLX<T> g = factrl(n)/PowCPLX(-jw, n+1) * (1 - exp(jw)*sum);
		return g;
	}

	__device__
	double MinDistanceFromPointToPlane(const cuVEC<double>& n,const cuVEC<double>& A,const cuVEC<double>& P){

//		cuVEC<double> P2((double)P.x, (double)P.y, (double)P.z);
//		cuVEC<double> A2((double)A.x, (double)A.y, (double)A.z);
//		cuVEC<double> n2((double)n.x, (double)n.y, (double)n.z);
//
//		cuVEC<double> uv=Unit(n2);	// convert plane vector to unit vector
//		double sum_min_dis = (P2.x-A2.x)*uv.x + (P2.y-A2.y)*uv.y + (P2.z-A2.z)*uv.z;
////		double sum_min_dis = (P2.x-A2.x)*n2.x + (P2.y-A2.y)*n2.y + (P2.z-A2.z)*n2.z;


//		cuVEC<float> uv=Unit(n);	// convert plane vector to unit vector
		double sum_min_dis = (P.x-A.x)*n.x + (P.y-A.y)*n.y + (P.z-A.z)*n.z;

		return fabs(sum_min_dis);
	}


	__device__
	void MinDistanceFromPointToPlane(const cuVEC<double>& n,const cuVEC<double>& A,		// input
									 const double Px, const double Py, const double Pz, // input
									 double& out){										// output

		double sum_min_dis = (Px-A.x)*n.x + (Py-A.y)*n.y + (Pz-A.z)*n.z;
		out = fabs(sum_min_dis);
	}

	__device__
	void MinDistanceFromPointToPlane(const double& nx, const double& ny, const double& nz,
									 const double& Ax, const double& Ay, const double& Az,
									 const double Px, const double Py, const double Pz, double& out){

		double sum_min_dis = (Px-Ax)*nx + (Py-Ay)*ny + (Pz-Az)*nz;
		out = fabs(sum_min_dis);
	}

	__device__
	void ReflCoeff_Air(const cuCPLX<float>& er2,const cuCPLX<float>& mr2,	// input
					   const float& cti_org,									// input
					   cuCPLX<float>& gammapar, cuCPLX<float>& gammaperp){
		//er1,mr1 relative parameters of space of incidence (restrict in free space = air)
		//er2,mr2 relative parameters of space of tranmsission
		//thetai: angle of incidence
		//gammapar, gammaperp: reflection coefficients parallel and perpendicular
		//thetat: transmission angle
		//IsTIR=ture when Total Internal Reflection occurs, else TIR=0

		double theta = acos(cti_org);
		double cti = cti_org;
		if(theta > def::PI/2.0){
			cti = cos(theta - def::PI/2);
		}

//		cuCPLX<float> m0(def::PI4*1e-7, 0);
//		cuCPLX<float> e0(8.854e-12, 0);

//		double n1 = sqrt(m0.r/e0.r);
		float n1 = 376.734309182;


		cuCPLX<float> mr2_er2 = mr2/er2;
		cuCPLX<float> n2 = n1 * mr2_er2.sqrt();
		cuCPLX<float> n2_cti = n2*cti;
		float n1_cti = n1*cti;


		if(er2.r > 1E15){ // Total Internal Reflection


			gammaperp = (n2_cti-n1)/(n2_cti+n1);
			gammapar  = (n2-n1_cti)/(n2+n1_cti);


		}else{
			float stt = sqrtf( (1 - cti*cti) / (er2.r*mr2.r) );
			float ctt = sqrtf(1-stt*stt);

			cuCPLX<float> n2_ctt = n2*ctt;
			float n1_ctt = n1*ctt;

			gammaperp = (n2_cti-n1_ctt)/(n2_cti+n1_ctt);
			gammapar  = (n2_ctt-n1_cti)/(n2_ctt+n1_cti);
		}
	}

	__device__
	void ReflCoeff(const cuCPLX<float>& er1,const cuCPLX<float>& mr1,	// input
			   	   const cuCPLX<float>& er2,const cuCPLX<float>& mr2,	// input
			   	   const float& cti_org, 									// input
			   	   float& thetat,										// output
			   	   cuCPLX<float>& gammapar, cuCPLX<float>& gammaperp){	// output
			// er1,mr1 relative parameters of space of incidence
			// er2,mr2 relative parameters of space of tranmsission
			// cti = cos(thetai): Cosine of incidence angle
			// gammapar, gammaperp: reflection coefficients parallel and perpendicular
			// thetat: transmission angle
			// TIR=1 when Total Internal Reflection occurs, else TIR=0

		double theta = acos(cti_org);
		double cti = cti_org;
		if(theta > def::PI/2.0){
			cti = cos(theta - def::PI/2);
		}

		cuCPLX<double> Er1(double(er1.r), double(er1.i));
		cuCPLX<double> Mr1(double(mr1.r), double(mr1.i));
		cuCPLX<double> Er2(double(er2.r), double(er2.i));
		cuCPLX<double> Mr2(double(mr2.r), double(mr2.i));

		cuCPLX<double> m0(def::PI4*1e-7, 0);
		cuCPLX<double> e0(8.8541878128e-12, 0);	// Real-part: vacuum permittivity (absolute dielectric permittivity)

		double sti = sqrt(1 - cti*cti);
		double sinthetat = sti*sqrt(Er1.r*Mr1.r/(Er2.r*Mr2.r));
		thetat = asin(sinthetat);
		double ctt = cos(thetat);
		
		cuCPLX<double> n1=(Mr1*m0/(Er1*e0)).sqrt();
		cuCPLX<double> n2=(Mr2*m0/(Er2*e0)).sqrt();

		gammaperp = (n2*cti-n1*ctt)/(n2*cti+n1*ctt);
		gammapar  = (n2*ctt-n1*cti)/(n2*ctt+n1*cti);
	}

	__device__
	cuRF GetReflectinoFactor(const float st, const float ct, const float sp, const float cp,
						     const cuMaterialDB& MatDB, const uint32_t& IdxMat, const double k0,
						     const float sa, const float ca, const float sb, const float cb,
						     const size_t k=0, const size_t i=0){

//		double PI2 = def::PI2;

//		cuCPLX<float> er(MatDB.er_r[IdxMat], -MatDB.tang[IdxMat]*MatDB.er_r[IdxMat]);
//		cuCPLX<float> mr(MatDB.mr[IdxMat],   -MatDB.mi[IdxMat]);
//		cuCPLX<float> er = MatDB.ER[IdxMat];
//		cuCPLX<float> mr = MatDB.MR[IdxMat];


//		float t = MatDB.d[IdxMat]*0.001;		// convert to meters

//		// convert to local facet coordinates
//		cuVEC<float> xyz_loc = transform(sa, ca, sb, cb, spher2cartOne(st, ct, sp, cp));
//		// Becaus xyz_loc is unit vector
//		float cti = xyz_loc.z;
		// Reduce to...
		float cti = sb*(st*(ca*cp + sa*sp)) + cb*ct;


		// angle of incidence is thetaloc
		// 1st interface: air to material interface
		cuCPLX<float> G1par, G1perp;
		ReflCoeff_Air(MatDB.ER[IdxMat], MatDB.MR[IdxMat], cti, G1par, G1perp);

////		// 2nd interface: material to air interface
////		cuCPLX<float> G2par  = -G1par;
////		cuCPLX<float> G2perp = -G1perp;
//		// find phase
////		double b1 = def::PI2_C*freq*MatDB.ERr_MRr_Sqrt[IdxMat];
//		double b1 = k0*MatDB.ERr_MRr_Sqrt[IdxMat];
////		double b1 = f0*MatDB.ERr_MRr_Sqrt[IdxMat];
////		double b1 = freq*MatDB.ERr_MRr_Sqrt[IdxMat];
//		double phase = std::fmod(b1*MatDB.d[IdxMat], def::PI2);
////		float phase = modff(b1*t, &PI2);



////		double phase = fmod(k0_mod_PI2*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
////		double phase = fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
//		double phase = fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
////		double phase = cu::opt::fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);

		double b1 = k0*sqrt(MatDB.ER[IdxMat].r*MatDB.MR[IdxMat].r);
//		double phase = fmod(b1*MatDB.d[IdxMat], def::PI2);
		float phase = b1*MatDB.d[IdxMat];


		// formulate matrices
//		float cphs, sphs;
//		sincosf((float)phase, &sphs, &cphs);
//		sincosf(phase, &sphs, &cphs);
//		double cphs, sphs;
//		sincos(phase, &sphs, &cphs);
		cuCPLX<float> exp1;
		sincosf(phase, &(exp1.i), &(exp1.r));
		cuCPLX<float> exp2(exp1.r, -exp1.i);
		cuCPLX<float> exp1_exp2(0, 2.f*exp1.i);


//		if(k == 31 && i == 0){
//			printf("k = %ld, sa = %.20f, ca = %.20f, sb = %.20f, cb = %.20f, st = %.20f, ct = %.20f, sp = %.20f, cp = %.20f\n", k, sa, ca, sb, cb, st, ct, sp, cp);
//			printf("k = %ld, cti = %.20f\n", k, cti);
//			printf("k = %ld, G1par = (%.20f,%.20f), G1perp = (%.20f,%.20f)\n", k, G1par.r, G1par.i, G1perp.r, G1perp.i);
//			printf("k = %ld, phase = %.20f\n", k, phase);
//		}



		// compute Reflection Coefficients
		cuCPLX<float> Mpar00  = exp1 + G1par  * exp2 * (-G1par);
		cuCPLX<float> Mpar10  = G1par * (exp1_exp2); //G1par * exp1  + exp2 * (-G1par);
		cuCPLX<float> Mperp00 = exp1 + G1perp * exp2 * (-G1perp);
		cuCPLX<float> Mperp10 = G1perp * (exp1_exp2); //G1perp * exp1 + exp2 * (-G1perp);
//		cuCPLX<float> Mpar00  = exp1 + G1par  * exp2 * G2par;
//		cuCPLX<float> Mpar10  = exp1 * G1par  + exp2 * G2par;
//		cuCPLX<float> Mperp00 = exp1 + G1perp * exp2 * G2perp;
//		cuCPLX<float> Mperp10 = exp1 * G1perp + exp2 * G2perp;

		cuCPLX<float> Rf_TM = Mpar10/Mpar00;	// RCpar (parallel) - TM
		cuCPLX<float> Rf_TE = Mperp10/Mperp00;	// RCperp (parpendicular) - TE

//		cuCPLX<float> Rf_TM = exp1_exp2 / (exp1/G1par  - G1par  * exp2);	// RCpar (parallel) - TM
//		cuCPLX<float> Rf_TE = exp1_exp2 / (exp1/G1perp - G1perp * exp2);	// RCperp (parpendicular) - TE


//		if(k == 13788){
//			printf("k=%d\n", k);
//			printf("st=%.10f, ct=%.10f, sp=%.10f, cp=%.10f\n", st, ct, sp, cp);
//			printf("er=(%.10f,%.10f), mr=(%.10f,%.10f), t=%.10f\n", er.r, er.i, mr.r, mr.i, t);
//			printf("freq=%.10f, xyz_loc=(%.10f,%.10f,%.10f)\n", freq, xyz_loc.x, xyz_loc.y, xyz_loc.z);
//			printf("G1par=(%.10f,%.10f), G1perp=(%.10f,%.10f)\n", G1par.r, G1par.i, G1perp.r, G1perp.i);
//			printf("b1=%.10f, phase=%.10f, cphs=%.10f, sphs=%.10f\n", b1, phase, cphs, sphs);
//			printf("exp1=(%.10f,%.10f), exp2=(%.10f,%.10f)\n", exp1.r, exp1.i, exp2.r, exp2.i);
//			printf("Mpar00=(%.10f,%.10f),  Mpar10=(%.10f,%.10f)\n", Mpar00.r,  Mpar00.i,  Mpar10.r,  Mpar10.i);
//			printf("Mperp00=(%.10f,%.10f), Mperp10=(%.10f,%.10f)\n", Mperp00.r, Mperp00.i, Mperp10.r, Mperp10.i);
//			printf("Rf.TE=(%.10f,%.10f), Rf.TE=(%.10f,%.10f)\n\n", Rf.TE.r, Rf.TE.i, Rf.TM.r, Rf.TM.i);
//		}

		// if(k == 562 && i == 0){
		// 	printf("k = %ld, i = %ld, cti = %f, Gpar = (%f,%f), Gperp = (%f,%f), Rf_TM = (%f,%f), Rf_TE = (%f,%f), k0 = %f\n", k, i, cti, G1par.r, G1par.i, G1perp.r, G1perp.i, Rf_TM.r, Rf_TM.i, Rf_TE.r, Rf_TE.i, k0);
		// }

		return cuRF( Rf_TE, Rf_TM );
	}

	__device__
	void GetReflectinoFactor1(const float st, const float ct, const float sp, const float cp,
						      const cuMaterialDB& MatDB, const uint32_t& IdxMat, const double k0,
						      const float sa, const float ca, const float sb, const float cb,
						      cuRF& Rf,
						      const size_t k=0, const size_t i=0){

//		double PI2 = def::PI2;

//		cuCPLX<float> er(MatDB.er_r[IdxMat], -MatDB.tang[IdxMat]*MatDB.er_r[IdxMat]);
//		cuCPLX<float> mr(MatDB.mr[IdxMat],   -MatDB.mi[IdxMat]);
//		cuCPLX<float> er = MatDB.ER[IdxMat];
//		cuCPLX<float> mr = MatDB.MR[IdxMat];


//		float t = MatDB.d[IdxMat]*0.001;		// convert to meters

//		// convert to local facet coordinates
//		cuVEC<float> xyz_loc = transform(sa, ca, sb, cb, spher2cartOne(st, ct, sp, cp));
//		// Becaus xyz_loc is unit vector
//		float cti = xyz_loc.z;
		// Reduce to...
		float cti = sb*(st*(ca*cp + sa*sp)) + cb*ct;


		// angle of incidence is thetaloc
		// 1st interface: air to material interface
		cuCPLX<float> G1par, G1perp;
		ReflCoeff_Air(MatDB.ER[IdxMat], MatDB.MR[IdxMat], cti, G1par, G1perp);

////		// 2nd interface: material to air interface
////		cuCPLX<float> G2par  = -G1par;
////		cuCPLX<float> G2perp = -G1perp;
//		// find phase
////		double b1 = def::PI2_C*freq*MatDB.ERr_MRr_Sqrt[IdxMat];
//		double b1 = k0*MatDB.ERr_MRr_Sqrt[IdxMat];
////		double b1 = f0*MatDB.ERr_MRr_Sqrt[IdxMat];
////		double b1 = freq*MatDB.ERr_MRr_Sqrt[IdxMat];
//		double phase = std::fmod(b1*MatDB.d[IdxMat], def::PI2);
////		float phase = modff(b1*t, &PI2);



////		double phase = fmod(k0_mod_PI2*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
////		double phase = fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
//		double phase = fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
////		double phase = cu::opt::fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);

		double b1 = k0*sqrt(MatDB.ER[IdxMat].r*MatDB.MR[IdxMat].r);
//		double phase = fmod(b1*MatDB.d[IdxMat], def::PI2);
		float phase = b1*MatDB.d[IdxMat];


		// formulate matrices
//		float cphs, sphs;
//		sincosf((float)phase, &sphs, &cphs);
//		sincosf(phase, &sphs, &cphs);
//		double cphs, sphs;
//		sincos(phase, &sphs, &cphs);
		cuCPLX<float> exp1;
		sincosf(phase, &(exp1.i), &(exp1.r));
		cuCPLX<float> exp2(exp1.r, -exp1.i);
		cuCPLX<float> exp1_exp2(0, 2.f*exp1.i);


//		if(k == 31 && i == 0){
//			printf("k = %ld, sa = %.20f, ca = %.20f, sb = %.20f, cb = %.20f, st = %.20f, ct = %.20f, sp = %.20f, cp = %.20f\n", k, sa, ca, sb, cb, st, ct, sp, cp);
//			printf("k = %ld, cti = %.20f\n", k, cti);
//			printf("k = %ld, G1par = (%.20f,%.20f), G1perp = (%.20f,%.20f)\n", k, G1par.r, G1par.i, G1perp.r, G1perp.i);
//			printf("k = %ld, phase = %.20f\n", k, phase);
//		}



		// compute Reflection Coefficients
		cuCPLX<float> Mpar00  = exp1 + G1par  * exp2 * (-G1par);
		cuCPLX<float> Mpar10  = G1par * (exp1_exp2); //G1par * exp1  + exp2 * (-G1par);
		cuCPLX<float> Mperp00 = exp1 + G1perp * exp2 * (-G1perp);
		cuCPLX<float> Mperp10 = G1perp * (exp1_exp2); //G1perp * exp1 + exp2 * (-G1perp);
//		cuCPLX<float> Mpar00  = exp1 + G1par  * exp2 * G2par;
//		cuCPLX<float> Mpar10  = exp1 * G1par  + exp2 * G2par;
//		cuCPLX<float> Mperp00 = exp1 + G1perp * exp2 * G2perp;
//		cuCPLX<float> Mperp10 = exp1 * G1perp + exp2 * G2perp;

//		cuCPLX<float> Rf_TM = Mpar10/Mpar00;	// RCpar (parallel) - TM
//		cuCPLX<float> Rf_TE = Mperp10/Mperp00;	// RCperp (parpendicular) - TE
		Rf.TM = Mpar10/Mpar00;	// RCpar (parallel) - TM
		Rf.TE = Mperp10/Mperp00;	// RCperp (parpendicular) - TE

//		cuCPLX<float> Rf_TM = exp1_exp2 / (exp1/G1par  - G1par  * exp2);	// RCpar (parallel) - TM
//		cuCPLX<float> Rf_TE = exp1_exp2 / (exp1/G1perp - G1perp * exp2);	// RCperp (parpendicular) - TE


//		if(k == 13788){
//			printf("k=%d\n", k);
//			printf("st=%.10f, ct=%.10f, sp=%.10f, cp=%.10f\n", st, ct, sp, cp);
//			printf("er=(%.10f,%.10f), mr=(%.10f,%.10f), t=%.10f\n", er.r, er.i, mr.r, mr.i, t);
//			printf("freq=%.10f, xyz_loc=(%.10f,%.10f,%.10f)\n", freq, xyz_loc.x, xyz_loc.y, xyz_loc.z);
//			printf("G1par=(%.10f,%.10f), G1perp=(%.10f,%.10f)\n", G1par.r, G1par.i, G1perp.r, G1perp.i);
//			printf("b1=%.10f, phase=%.10f, cphs=%.10f, sphs=%.10f\n", b1, phase, cphs, sphs);
//			printf("exp1=(%.10f,%.10f), exp2=(%.10f,%.10f)\n", exp1.r, exp1.i, exp2.r, exp2.i);
//			printf("Mpar00=(%.10f,%.10f),  Mpar10=(%.10f,%.10f)\n", Mpar00.r,  Mpar00.i,  Mpar10.r,  Mpar10.i);
//			printf("Mperp00=(%.10f,%.10f), Mperp10=(%.10f,%.10f)\n", Mperp00.r, Mperp00.i, Mperp10.r, Mperp10.i);
//			printf("Rf.TE=(%.10f,%.10f), Rf.TE=(%.10f,%.10f)\n\n", Rf.TE.r, Rf.TE.i, Rf.TM.r, Rf.TM.i);
//		}



//		return cuRF( Rf_TE, Rf_TM );
	}

	__device__
	void GetReflectinoFactor2(const float st, const float ct, const float sp, const float cp,
						      const cuMaterialDB& MatDB, const uint32_t& IdxMat, const double k0,
						      const float sa, const float ca, const float sb, const float cb,
						      float& Rf_TE_r, float& Rf_TE_i, float& Rf_TM_r, float& Rf_TM_i){

//		double PI2 = def::PI2;

//		cuCPLX<float> er(MatDB.er_r[IdxMat], -MatDB.tang[IdxMat]*MatDB.er_r[IdxMat]);
//		cuCPLX<float> mr(MatDB.mr[IdxMat],   -MatDB.mi[IdxMat]);
//		cuCPLX<float> er = MatDB.ER[IdxMat];
//		cuCPLX<float> mr = MatDB.MR[IdxMat];


//		float t = MatDB.d[IdxMat]*0.001;		// convert to meters

//		// convert to local facet coordinates
//		cuVEC<float> xyz_loc = transform(sa, ca, sb, cb, spher2cartOne(st, ct, sp, cp));
//		// Becaus xyz_loc is unit vector
//		float cti = xyz_loc.z;
		// Reduce to...
		float cti = sb*(st*(ca*cp + sa*sp)) + cb*ct;


		// angle of incidence is thetaloc
		// 1st interface: air to material interface
		cuCPLX<float> G1par, G1perp;
		ReflCoeff_Air(MatDB.ER[IdxMat], MatDB.MR[IdxMat], cti, G1par, G1perp);

////		// 2nd interface: material to air interface
////		cuCPLX<float> G2par  = -G1par;
////		cuCPLX<float> G2perp = -G1perp;
//		// find phase
////		double b1 = def::PI2_C*freq*MatDB.ERr_MRr_Sqrt[IdxMat];
//		double b1 = k0*MatDB.ERr_MRr_Sqrt[IdxMat];
////		double b1 = f0*MatDB.ERr_MRr_Sqrt[IdxMat];
////		double b1 = freq*MatDB.ERr_MRr_Sqrt[IdxMat];
//		double phase = std::fmod(b1*MatDB.d[IdxMat], def::PI2);
////		float phase = modff(b1*t, &PI2);



////		double phase = fmod(k0_mod_PI2*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
////		double phase = fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
//		double phase = fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);
////		double phase = cu::opt::fmod(k0*MatDB.ERr_MRr_Sqrt[IdxMat], def::PI2);

		double b1 = k0*sqrt(MatDB.ER[IdxMat].r*MatDB.MR[IdxMat].r);
//		double phase = fmod(b1*MatDB.d[IdxMat], def::PI2);
		float phase = b1*MatDB.d[IdxMat];


		// formulate matrices
//		float cphs, sphs;
//		sincosf((float)phase, &sphs, &cphs);
//		sincosf(phase, &sphs, &cphs);
//		double cphs, sphs;
//		sincos(phase, &sphs, &cphs);
		cuCPLX<float> exp1;
		sincosf(phase, &(exp1.i), &(exp1.r));
		cuCPLX<float> exp2(exp1.r, -exp1.i);
		cuCPLX<float> exp1_exp2(0, 2.f*exp1.i);


//		if(k == 31 && i == 0){
//			printf("k = %ld, sa = %.20f, ca = %.20f, sb = %.20f, cb = %.20f, st = %.20f, ct = %.20f, sp = %.20f, cp = %.20f\n", k, sa, ca, sb, cb, st, ct, sp, cp);
//			printf("k = %ld, cti = %.20f\n", k, cti);
//			printf("k = %ld, G1par = (%.20f,%.20f), G1perp = (%.20f,%.20f)\n", k, G1par.r, G1par.i, G1perp.r, G1perp.i);
//			printf("k = %ld, phase = %.20f\n", k, phase);
//		}



		// compute Reflection Coefficients
		cuCPLX<float> Mpar00  = exp1 + G1par  * exp2 * (-G1par);
		cuCPLX<float> Mpar10  = G1par * (exp1_exp2); //G1par * exp1  + exp2 * (-G1par);
		cuCPLX<float> Mperp00 = exp1 + G1perp * exp2 * (-G1perp);
		cuCPLX<float> Mperp10 = G1perp * (exp1_exp2); //G1perp * exp1 + exp2 * (-G1perp);
//		cuCPLX<float> Mpar00  = exp1 + G1par  * exp2 * G2par;
//		cuCPLX<float> Mpar10  = exp1 * G1par  + exp2 * G2par;
//		cuCPLX<float> Mperp00 = exp1 + G1perp * exp2 * G2perp;
//		cuCPLX<float> Mperp10 = exp1 * G1perp + exp2 * G2perp;

//		cuCPLX<float> Rf_TM = Mpar10/Mpar00;	// RCpar (parallel) - TM
//		cuCPLX<float> Rf_TE = Mperp10/Mperp00;	// RCperp (parpendicular) - TE

		cuCPLXDiv(Mpar10,  Mpar00,  Rf_TM_r, Rf_TM_i);
		cuCPLXDiv(Mperp10, Mperp00, Rf_TE_r, Rf_TE_i);




//		cuCPLX<float> Rf_TM = exp1_exp2 / (exp1/G1par  - G1par  * exp2);	// RCpar (parallel) - TM
//		cuCPLX<float> Rf_TE = exp1_exp2 / (exp1/G1perp - G1perp * exp2);	// RCperp (parpendicular) - TE


//		if(k == 13788){
//			printf("k=%d\n", k);
//			printf("st=%.10f, ct=%.10f, sp=%.10f, cp=%.10f\n", st, ct, sp, cp);
//			printf("er=(%.10f,%.10f), mr=(%.10f,%.10f), t=%.10f\n", er.r, er.i, mr.r, mr.i, t);
//			printf("freq=%.10f, xyz_loc=(%.10f,%.10f,%.10f)\n", freq, xyz_loc.x, xyz_loc.y, xyz_loc.z);
//			printf("G1par=(%.10f,%.10f), G1perp=(%.10f,%.10f)\n", G1par.r, G1par.i, G1perp.r, G1perp.i);
//			printf("b1=%.10f, phase=%.10f, cphs=%.10f, sphs=%.10f\n", b1, phase, cphs, sphs);
//			printf("exp1=(%.10f,%.10f), exp2=(%.10f,%.10f)\n", exp1.r, exp1.i, exp2.r, exp2.i);
//			printf("Mpar00=(%.10f,%.10f),  Mpar10=(%.10f,%.10f)\n", Mpar00.r,  Mpar00.i,  Mpar10.r,  Mpar10.i);
//			printf("Mperp00=(%.10f,%.10f), Mperp10=(%.10f,%.10f)\n", Mperp00.r, Mperp00.i, Mperp10.r, Mperp10.i);
//			printf("Rf.TE=(%.10f,%.10f), Rf.TE=(%.10f,%.10f)\n\n", Rf.TE.r, Rf.TE.i, Rf.TM.r, Rf.TM.i);
//		}
	}

	__device__
	cuRF GetReflectinoFactorOnAlumina(const float st, const float ct, const float sp, const float cp,
						    		  const cuMaterialDB& MatDB, const uint32_t& IdxMat, const double k0,
						     		  const float sa, const float ca, const float sb, const float cb,
						     		  const size_t k=0, const size_t i=0){
		// initialize matrices
		//     Mpar00,   Mpar01
		//     Mpar10,   Mpar11
		cuCPLX<float> Mpar00(1,0);	cuCPLX<float> Mpar01(0,0);
		cuCPLX<float> Mpar10(0,0);	cuCPLX<float> Mpar11(1,0);
		//     Mperp00,   Mperp01
		//     Mperp10,   Mperp11
		cuCPLX<float> Mperp00(1,0);	cuCPLX<float> Mperp01(0,0);
		cuCPLX<float> Mperp10(0,0);	cuCPLX<float> Mperp11(1,0);
		
		// Reduce to...
		float cti = sb*(st*(ca*cp + sa*sp)) + cb*ct;

		// if(k == 1109 && i == 0){
		// 	printf("k=%ld, i=%ld, cti=%f, st=%f, ct=%f, sp=%f, cp=%f, sa=%f, ca=%f, sb=%f, cb=%f\n", k, i, cti, st, ct, sp, cp, sa, ca, sb, cb);
		// }
		
		// allocation
		float thetat, phase;
		double b1;
		double t;
		cuCPLX<float> Gpar, Gperp;
		cuCPLX<float> exp1, exp2;
		cuCPLX<float> Mpr00, Mpr01, Mpr10, Mpr11;
		cuCPLX<float> Mpp00, Mpp01, Mpp10, Mpp11;
		cuCPLX<float> er0, mr0, er, mr;


		// 1st layer (air -> material) -----------------------------------------------------
		// 1st interface: air to material interface
		// ReflCoeff_Air(MatDB.ER[IdxMat], MatDB.MR[IdxMat], cti, Gpar, Gperp);
		// previous transmission angle becomes incidence angle (Air)
		er0 = cuCPLX<float>(1, 0);
		mr0 = cuCPLX<float>(1, 0);
		ReflCoeff(er0, mr0, MatDB.ER[IdxMat], MatDB.MR[IdxMat], cti, thetat, Gpar, Gperp);



		b1 = k0*sqrt(MatDB.ER[IdxMat].r*MatDB.MR[IdxMat].r);
		phase = b1*MatDB.d[IdxMat]/cti;


		// formulate matrices
		sincosf(phase, &(exp1.i), &(exp1.r));
		exp2 = cuCPLX<float>(exp1.r, -exp1.i);
		cuCPLX<float> exp1_exp2(0, 2.f*exp1.i);

		// // compute Reflection Coefficients
		// Mpar00  = exp1 + Gpar  * exp2 * (-Gpar);
		// Mpar10  = Gpar * (exp1_exp2); //Gpar * exp1  + exp2 * (-Gpar);
		// Mperp00 = exp1 + Gperp * exp2 * (-Gperp);
		// Mperp10 = Gperp * (exp1_exp2); //Gperp * exp1 + exp2 * (-Gperp);

		// Mpar=Mpar*[exp(j*phase), Gpar*exp(-j*phase);Gpar*exp(j*phase), exp(-j*phase)];
		Mpr00 = Mpar00*exp1 + Mpar01*Gpar*exp1;
		Mpr01 = Mpar00*Gpar*exp2 + Mpar01*exp2;
		Mpr10 = Mpar10*exp1 + Mpar11*Gpar*exp1;
		Mpr11 = Mpar10*Gpar*exp2 + Mpar11*exp2;
		// Assign back
		Mpar00 = Mpr00;		Mpar01 = Mpr01;
		Mpar10 = Mpr10;		Mpar11 = Mpr11;
		// Mperp=Mperp*[exp(j*phase), Gperp*exp(-j*phase);Gperp*exp(j*phase), exp(-j*phase)];
		Mpp00 = Mperp00*exp1 + Mperp01*Gperp*exp1;
		Mpp01 = Mperp00*Gperp*exp2 + Mperp01*exp2;
		Mpp10 = Mperp10*exp1 + Mperp11*Gperp*exp1;
		Mpp11 = Mperp10*Gperp*exp2 + Mperp11*exp2;
		// Assign back
		Mperp00 = Mpp00;	Mperp01 = Mpp01;
		Mperp10 = Mpp10;	Mperp11 = Mpp11;


		// End layer (material -> Alumina) -------------------------------------------------
		
		// Alumina
		// er = cuCPLX<float>(9.6, -0.003*9.6);
		// mr = cuCPLX<float>(1.0, -0.0);
		er = cuCPLX<float>(3.7, -0.0045*3.7);
		mr = cuCPLX<float>(1.0, -0.0);

		// // PEC
		// er = cuCPLX<float>(1E30, 0.0);
		// mr = cuCPLX<float>(1.0, 0.0);


		t  = 1e-9;
		// previous transmission angle becomes incidence angle
		// ReflCoeff(MatDB.ER[IdxMat], MatDB.MR[IdxMat], er, mr, cti, thetat, Gpar, Gperp);
		ReflCoeff(MatDB.ER[IdxMat], MatDB.MR[IdxMat], er, mr, cos(thetat), thetat, Gpar, Gperp);
		// find phase
		b1 = k0*sqrt(er.r*mr.r);
		phase = b1*t/cti;
		// form matrices
		exp1 = cuCPLX<float>(cos(phase),sin(phase));		// exp(j*phase)
		exp2 = cuCPLX<float>(exp1.r, -exp1.i);	// exp(-j*phase)

		// Mpar=Mpar*[exp(j*phase), Gpar*exp(-j*phase);Gpar*exp(j*phase), exp(-j*phase)];
		Mpr00 = Mpar00*exp1 + Mpar01*Gpar*exp1;
		Mpr10 = Mpar10*exp1 + Mpar11*Gpar*exp1;
		// Assign back
		Mpar00 = Mpr00;//		Mpar01 = Mpr01;
		Mpar10 = Mpr10;//		Mpar11 = Mpr11;
		Mpp00 = Mperp00*exp1 + Mperp01*Gperp*exp1;
		Mpp10 = Mperp10*exp1 + Mperp11*Gperp*exp1;
		// Assign back
		Mperp00 = Mpp00;//	Mperp01 = Mpp01;
		Mperp10 = Mpp10;//	Mperp11 = Mpp11;




		cuCPLX<float> Rf_TM = Mpar10/Mpar00;	// RCpar (parallel) - TM
		cuCPLX<float> Rf_TE = Mperp10/Mperp00;	// RCperp (parpendicular) - TE

		// // if(k == 562 && i == 0){
		// if(k == 75 && i == 0){
		// 	// printf("k=%ld, i=%ld, k0=%f, cti=%f, MatDB.ER[IdxMat]=(%f,%f), MatDB.MR[IdxMat]=(%f,%f), Gpar=(%f,%f), Gperp=(%f,%f), n1=(%f,%f), n2=(%f,%f), sti=%f, sinthetat=%f, thetat=%f, ctt=%f\n", k, i, k0, cti, MatDB.ER[IdxMat].r, MatDB.ER[IdxMat].i, MatDB.MR[IdxMat].r, MatDB.MR[IdxMat].i, Gpar.r, Gpar.i, Gperp.r, Gperp.i, n1.r, n1.i, n2.r, n2.i, sti, sinthetat, thetat, ctt);
		// 	printf("k = %ld, i = %ld, cti = %f, Gpar = (%f,%f), Gperp = (%f,%f), Rf_TM = (%f,%f), Rf_TE = (%f,%f), k0 = %f\n", k, i, cti, Gpar.r, Gpar.i, Gperp.r, Gperp.i, Rf_TM.r, Rf_TM.i, Rf_TE.r, Rf_TE.i, k0);
		// }

		return cuRF( Rf_TE, Rf_TM );
	}

	__device__
	cuRF GetReflectinoFactorOnPEC(const float st, const float ct, const float sp, const float cp,
								  const cuMaterialDB& MatDB, const uint32_t& IdxMat, const double k0,
								  const float sa, const float ca, const float sb, const float cb,
								  const size_t k=0, const size_t i=0){
		// initialize matrices
		//     Mpar00,   Mpar01
		//     Mpar10,   Mpar11
		cuCPLX<float> Mpar00(1,0);	cuCPLX<float> Mpar01(0,0);
		cuCPLX<float> Mpar10(0,0);	cuCPLX<float> Mpar11(1,0);
		//     Mperp00,   Mperp01
		//     Mperp10,   Mperp11
		cuCPLX<float> Mperp00(1,0);	cuCPLX<float> Mperp01(0,0);
		cuCPLX<float> Mperp10(0,0);	cuCPLX<float> Mperp11(1,0);

		// Reduce to...
		float cti = sb*(st*(ca*cp + sa*sp)) + cb*ct;

		// if(k == 1109 && i == 0){
		// 	printf("k=%ld, i=%ld, cti=%f, st=%f, ct=%f, sp=%f, cp=%f, sa=%f, ca=%f, sb=%f, cb=%f\n", k, i, cti, st, ct, sp, cp, sa, ca, sb, cb);
		// }

		// allocation
		float thetat, phase;
		double b1;
		double t;
		cuCPLX<float> Gpar, Gperp;
		cuCPLX<float> exp1, exp2;
		cuCPLX<float> Mpr00, Mpr01, Mpr10, Mpr11;
		cuCPLX<float> Mpp00, Mpp01, Mpp10, Mpp11;
		cuCPLX<float> er0, mr0, er, mr;


		// 1st layer (air -> material) -----------------------------------------------------
		// 1st interface: air to material interface
		// ReflCoeff_Air(MatDB.ER[IdxMat], MatDB.MR[IdxMat], cti, Gpar, Gperp);
		// previous transmission angle becomes incidence angle (Air)
		er0 = cuCPLX<float>(1, 0);
		mr0 = cuCPLX<float>(1, 0);
		ReflCoeff(er0, mr0, MatDB.ER[IdxMat], MatDB.MR[IdxMat], cti, thetat, Gpar, Gperp);



		b1 = k0*sqrt(MatDB.ER[IdxMat].r*MatDB.MR[IdxMat].r);
		phase = b1*MatDB.d[IdxMat]/cti;


		// formulate matrices
		sincosf(phase, &(exp1.i), &(exp1.r));
		exp2 = cuCPLX<float>(exp1.r, -exp1.i);
		cuCPLX<float> exp1_exp2(0, 2.f*exp1.i);

		// // compute Reflection Coefficients
		// Mpar00  = exp1 + Gpar  * exp2 * (-Gpar);
		// Mpar10  = Gpar * (exp1_exp2); //Gpar * exp1  + exp2 * (-Gpar);
		// Mperp00 = exp1 + Gperp * exp2 * (-Gperp);
		// Mperp10 = Gperp * (exp1_exp2); //Gperp * exp1 + exp2 * (-Gperp);

		// Mpar=Mpar*[exp(j*phase), Gpar*exp(-j*phase);Gpar*exp(j*phase), exp(-j*phase)];
		Mpr00 = Mpar00*exp1 + Mpar01*Gpar*exp1;
		Mpr01 = Mpar00*Gpar*exp2 + Mpar01*exp2;
		Mpr10 = Mpar10*exp1 + Mpar11*Gpar*exp1;
		Mpr11 = Mpar10*Gpar*exp2 + Mpar11*exp2;
		// Assign back
		Mpar00 = Mpr00;		Mpar01 = Mpr01;
		Mpar10 = Mpr10;		Mpar11 = Mpr11;
		// Mperp=Mperp*[exp(j*phase), Gperp*exp(-j*phase);Gperp*exp(j*phase), exp(-j*phase)];
		Mpp00 = Mperp00*exp1 + Mperp01*Gperp*exp1;
		Mpp01 = Mperp00*Gperp*exp2 + Mperp01*exp2;
		Mpp10 = Mperp10*exp1 + Mperp11*Gperp*exp1;
		Mpp11 = Mperp10*Gperp*exp2 + Mperp11*exp2;
		// Assign back
		Mperp00 = Mpp00;	Mperp01 = Mpp01;
		Mperp10 = Mpp10;	Mperp11 = Mpp11;


		// End layer (material -> Alumina) -------------------------------------------------

//		// Alumina
//		// er = cuCPLX<float>(9.6, -0.003*9.6);
//		// mr = cuCPLX<float>(1.0, -0.0);
//		er = cuCPLX<float>(3.7, -0.0045*3.7);
//		mr = cuCPLX<float>(1.0, -0.0);

		// PEC
		er = cuCPLX<float>(1E30, 0.0);
		mr = cuCPLX<float>(1.0, 0.0);


		t  = 1e-9;
		// previous transmission angle becomes incidence angle
		// ReflCoeff(MatDB.ER[IdxMat], MatDB.MR[IdxMat], er, mr, cti, thetat, Gpar, Gperp);
		ReflCoeff(MatDB.ER[IdxMat], MatDB.MR[IdxMat], er, mr, cos(thetat), thetat, Gpar, Gperp);
		// find phase
		b1 = k0*sqrt(er.r*mr.r);
		phase = b1*t/cti;
		// form matrices
		exp1 = cuCPLX<float>(cos(phase),sin(phase));		// exp(j*phase)
		exp2 = cuCPLX<float>(exp1.r, -exp1.i);	// exp(-j*phase)

		// Mpar=Mpar*[exp(j*phase), Gpar*exp(-j*phase);Gpar*exp(j*phase), exp(-j*phase)];
		Mpr00 = Mpar00*exp1 + Mpar01*Gpar*exp1;
		Mpr10 = Mpar10*exp1 + Mpar11*Gpar*exp1;
		// Assign back
		Mpar00 = Mpr00;//		Mpar01 = Mpr01;
		Mpar10 = Mpr10;//		Mpar11 = Mpr11;
		Mpp00 = Mperp00*exp1 + Mperp01*Gperp*exp1;
		Mpp10 = Mperp10*exp1 + Mperp11*Gperp*exp1;
		// Assign back
		Mperp00 = Mpp00;//	Mperp01 = Mpp01;
		Mperp10 = Mpp10;//	Mperp11 = Mpp11;




		cuCPLX<float> Rf_TM = Mpar10/Mpar00;	// RCpar (parallel) - TM
		cuCPLX<float> Rf_TE = Mperp10/Mperp00;	// RCperp (parpendicular) - TE

		// // if(k == 562 && i == 0){
		// if(k == 75 && i == 0){
		// 	// printf("k=%ld, i=%ld, k0=%f, cti=%f, MatDB.ER[IdxMat]=(%f,%f), MatDB.MR[IdxMat]=(%f,%f), Gpar=(%f,%f), Gperp=(%f,%f), n1=(%f,%f), n2=(%f,%f), sti=%f, sinthetat=%f, thetat=%f, ctt=%f\n", k, i, k0, cti, MatDB.ER[IdxMat].r, MatDB.ER[IdxMat].i, MatDB.MR[IdxMat].r, MatDB.MR[IdxMat].i, Gpar.r, Gpar.i, Gperp.r, Gperp.i, n1.r, n1.i, n2.r, n2.i, sti, sinthetat, thetat, ctt);
		// 	printf("k = %ld, i = %ld, cti = %f, Gpar = (%f,%f), Gperp = (%f,%f), Rf_TM = (%f,%f), Rf_TE = (%f,%f), k0 = %f\n", k, i, cti, Gpar.r, Gpar.i, Gperp.r, Gperp.i, Rf_TM.r, Rf_TM.i, Rf_TE.r, Rf_TE.i, k0);
		// }

		return cuRF( Rf_TE, Rf_TM );
	}


	__device__
	cuElectricField ReflectionElectricField(const cuElectricField& Ei, const cuVEC<float>& N, const cuRF& Rf, const cuVEC<double>& k_next,
											const cuVEC<double>& Ei_org_k, const double k0, const double RayArea, double& sinc_patch){

		cuVEC<float> Local_inc_theta_vec, Local_ref_theta_vec;
//		cuThetaPhiVec Local_inc, Local_ref;
		// normal vector of reflection surface
		cuVEC<float> k = -Ei.k;
//		cuVEC<double> N2 = N;
//		cuVEC<float> tmp = cross(-Ei.k,N);
//		float inv = rsqrtf(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
//		cuVEC<float> m(tmp.x*inv, tmp.y*inv, tmp.z*inv);
		cuVEC<float> m = Unit(cross(k,N));
		// local XYZ (eq.10)
		cuVEC<float> X_vec = cross(m,N);
		cuVEC<float> Y_vec = -m;
		cuVEC<float> Z_vec = -N;
		// local incident angle (eq.11)
		float ct = dot(k,N);
//		float st = sqrtf(1.f-ct*ct);
		float val_1_ct2 = 1.0-ct*ct;
		float st = (val_1_ct2 > 0) * sqrtf(val_1_ct2) + 0.0;
		// local incident theta & phi vector (eq.12)
		Local_inc_theta_vec = Unit(ct * X_vec - st * Z_vec);
//		Local_inc.phi_vec   = Y_vec;
		// Reflection
		Local_ref_theta_vec = Unit(-ct * X_vec - st * Z_vec);
//		Local_ref.phi_vec   = Y_vec;

		// Amplitude of reflected field (eq.9), [Note: different than ref?]
		cuCPLX<float> Et = Rf.TM * dot(Ei.cplx, Local_inc_theta_vec) *  1;	// Et
		cuCPLX<float> Ep = Rf.TE * dot(Ei.cplx, Y_vec)   			 * -1;	// Ep (diff?)

		cuElectricField Er;
		// Ref. E field complex (eq.9)
		Er.cplx.x = Et * Local_ref_theta_vec.x + Ep * Y_vec.x;
		Er.cplx.y = Et * Local_ref_theta_vec.y + Ep * Y_vec.y;
		Er.cplx.z = Et * Local_ref_theta_vec.z + Ep * Y_vec.z;
		// Assign Ref E Field
		Er.k = k_next;
		Er.o = Ei.o;


		cuVEC<double> Kd = (Ei.k + Ei_org_k) * k0;   // kipo == Ei_org.k, kin == Ei_org (when Level = 0)
	
		double cellSize = sqrt(RayArea);

		sinc_patch = sinc_no_pi( dot(Kd, X_vec)*cellSize/ 2.0 / fabs(dot(Ei.k, N)) ) *
		   		     sinc_no_pi( dot(Kd, Y_vec)*cellSize/ 2.0);  //phase correct


		return Er;
	}

	__device__
	cuElectricField ReflectionElectricField(const cuElectricField& Ei, const cuVEC<float>& N, const cuRF& Rf, const cuVEC<double>& k_next){

		cuThetaPhiVec Local_inc, Local_ref;
		// normal vector of reflection surface
		cuVEC<float> k = -Ei.k;
//		cuVEC<double> N2 = N;
//		cuVEC<float> tmp = cross(-Ei.k,N);
//		float inv = rsqrtf(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
//		cuVEC<float> m(tmp.x*inv, tmp.y*inv, tmp.z*inv);
		cuVEC<float> m = Unit(cross(k,N));
		// local XYZ (eq.10)
		cuVEC<float> X_vec = cross(m,N);
		cuVEC<float> Y_vec = -m;
		cuVEC<float> Z_vec = -N;
		// local incident angle (eq.11)
		float ct = dot(k,N);
//		float st = sqrtf(1.f-ct*ct);
		float val_1_ct2 = 1.0-ct*ct;
		float st = (val_1_ct2 > 0) * sqrtf(val_1_ct2) + 0.0;
		// local incident theta & phi vector (eq.12)
		Local_inc.theta_vec = Unit(ct * X_vec - st * Z_vec);
		Local_inc.phi_vec   = Y_vec;
		// Reflection
		Local_ref.theta_vec = Unit(-ct * X_vec - st * Z_vec);
		Local_ref.phi_vec   = Y_vec;

		// Amplitude of reflected field (eq.9), [Note: different than ref?]
		cuCPLX<float> Et = Rf.TM * dot(Ei.cplx, Local_inc.theta_vec) *  1;	// Et
		cuCPLX<float> Ep = Rf.TE * dot(Ei.cplx, Local_inc.phi_vec)   * -1;	// Ep (diff?)

		cuElectricField Er;
		// Ref. E field complex (eq.9)
		Er.cplx.x = Et * Local_ref.theta_vec.x + Ep * Local_ref.phi_vec.x;
		Er.cplx.y = Et * Local_ref.theta_vec.y + Ep * Local_ref.phi_vec.y;
		Er.cplx.z = Et * Local_ref.theta_vec.z + Ep * Local_ref.phi_vec.z;
		// Assign Ref E Field
		Er.k = k_next;
		Er.o = Ei.o;

		return Er;
	}

	__device__
	void ReflectionElectricField(const cuVEC<double>& Ei_k, cuVEC<cuCPLX<float> >& Ei_cplx,
								 const cuVEC<float>& N, const cuRF& Rf){

		cuThetaPhiVec Local_inc, Local_ref;
		// normal vector of reflection surface
		cuVEC<float> k = -Ei_k;
	//		cuVEC<double> N2 = N;
	//		cuVEC<float> tmp = cross(-Ei.k,N);
	//		float inv = rsqrtf(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
	//		cuVEC<float> m(tmp.x*inv, tmp.y*inv, tmp.z*inv);
		cuVEC<float> m = Unit(cross(k,N));
		// local XYZ (eq.10)
		cuVEC<float> X_vec = cross(m,N);
//		cuVEC<float> Y_vec = -m;
//		cuVEC<float> Z_vec = -N;
		// local incident angle (eq.11)
		float ct = dot(k,N);
//		float st = sqrtf(1.f-ct*ct);
		float val_1_ct2 = 1.0-ct*ct;
		float st = (val_1_ct2 > 0) * sqrtf(val_1_ct2) + 0.0;
		// local incident theta & phi vector (eq.12)
		Local_inc.theta_vec = Unit(ct * X_vec + st * N);
//		Local_inc.phi_vec   = Y_vec;
		// Reflection
		Local_ref.theta_vec = Unit(-ct * X_vec + st * N);
//		Local_ref.phi_vec   = Y_vec;

		// Amplitude of reflected field (eq.9), [Note: different than ref?]
		cuCPLX<float> Et = Rf.TM * dot(Ei_cplx, Local_inc.theta_vec);// *  1;	// Et
		cuCPLX<float> Ep = Rf.TE * dot(Ei_cplx, m);//   * -1;	// Ep (diff?)

//		cuElectricField Er;
//		cuVEC<cuCPLX<float> > Er_cplx;
		// Ref. E field complex (eq.9)
		Ei_cplx.x = Et * Local_ref.theta_vec.x - Ep * m.x;
		Ei_cplx.y = Et * Local_ref.theta_vec.y - Ep * m.y;
		Ei_cplx.z = Et * Local_ref.theta_vec.z - Ep * m.z;
//		// Assign Ref E Field
//		Er.k = k_next;
//		Er.o = Ei.o;
//
//		return Er;
//		return Er_cplx;
	}

	__device__
	void ReflectionElectricFieldAddDis(const cuVEC<double>& Ei_k, cuVEC<cuCPLX<float> >& Ei_cplx,
								 	   const cuVEC<float>& N, const cuRF& Rf,
								 	   const double k0, const double DistSet){

		cuThetaPhiVec Local_inc, Local_ref;
		// normal vector of reflection surface
		cuVEC<float> k = -Ei_k;
	//		cuVEC<double> N2 = N;
	//		cuVEC<float> tmp = cross(-Ei.k,N);
	//		float inv = rsqrtf(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
	//		cuVEC<float> m(tmp.x*inv, tmp.y*inv, tmp.z*inv);
		cuVEC<float> m = Unit(cross(k,N));
		// local XYZ (eq.10)
		cuVEC<float> X_vec = cross(m,N);
//		cuVEC<float> Y_vec = -m;
//		cuVEC<float> Z_vec = -N;
		// local incident angle (eq.11)
		float ct = dot(k,N);
//		float st = sqrtf(1.f-ct*ct);
		float val_1_ct2 = 1.0-ct*ct;
		float st = (val_1_ct2 > 0) * sqrtf(val_1_ct2) + 0.0;
		// local incident theta & phi vector (eq.12)
		Local_inc.theta_vec = Unit(ct * X_vec + st * N);
//		Local_inc.phi_vec   = Y_vec;
		// Reflection
		Local_ref.theta_vec = Unit(-ct * X_vec + st * N);
//		Local_ref.phi_vec   = Y_vec;

		// Amplitude of reflected field (eq.9), [Note: different than ref?]
		cuCPLX<float> Et = Rf.TM * dot(Ei_cplx, Local_inc.theta_vec);// *  1;	// Et
		cuCPLX<float> Ep = Rf.TE * dot(Ei_cplx, m);//   * -1;	// Ep (diff?)



		// Phase induced by distance
		double phs = k0 * DistSet;
		double sp, cp;
		sincos(-phs, &sp, &cp);

		cuCPLX<float> phase(cp, sp);

//		cuElectricField Er;
//		cuVEC<cuCPLX<float> > Er_cplx;
		// Ref. E field complex (eq.9)
		Ei_cplx.x = (Et * Local_ref.theta_vec.x - Ep * m.x) * phase;
		Ei_cplx.y = (Et * Local_ref.theta_vec.y - Ep * m.y) * phase;
		Ei_cplx.z = (Et * Local_ref.theta_vec.z - Ep * m.z) * phase;
//		// Assign Ref E Field
//		Er.k = k_next;
//		Er.o = Ei.o;
//
//		return Er;
//		return Er_cplx;
	}

	template<typename T1, typename T2>
	__device__
	cuVEC<cuCPLX<T2> > cross(const cuVEC<T1>& a,const cuVEC<cuCPLX<T2> >& b){
		return cuVEC<cuCPLX<T2> >(a.y*b.z - a.z*b.y, \
								  a.z*b.x - a.x*b.z, \
								  a.x*b.y - a.y*b.x);
	}

 	__device__ // TODO: 需要確定 case 4 是否正確
	cuVEC<cuCPLX<float> > POCalculate1(const cuElectricField& Ei, const cuTRI<float>& tri, const double k0,			// input
								const cuRF& Rf, const cuTAYLOR<float>& Taylor,								// input
								const float sp, const float cp,												// input
								const float sti2, const float cti2, const float spi2, const float cpi2,		// input
								const float uu, const float vv, const float ww,								// input
								const double u_ui, const double v_vi, const double w_wi, const double w_wi2,		// input
								const cuVEC<double>& k_obv, const cuVEC<float>& o, const cuThetaPhiVec g, const size_t k, const size_t i){	// input

		// residual
		// E-field Amplitude
//		cuVEC<float> Ei_abs = abs(Ei.cplx);
//		float Co = sqrtf(Ei_abs.x*Ei_abs.x + Ei_abs.y*Ei_abs.y + Ei_abs.z*Ei_abs.z);
		float Co = (abs(Ei.cplx)).abs();

		// Incident field in global Cartesian coordinates (Bistatic)
		cuVEC<cuCPLX<float> > Rc = Ei.cplx;

		// Incident field in local Cartesian coordinates (stored in e2)
		cuVEC<cuCPLX<float> > e2 = cu::transform(tri.sa, tri.ca, tri.sb, tri.cb, Rc);

		cuVEC<cuCPLX<float> > e2_1( (tri.ca * Rc.x) + (tri.sa * Rc.y),
								   (-tri.sa * Rc.x) + (tri.ca * Rc.y),
									 Rc.z );

		// Incident field in local spherical coordinates
		cuCPLX<float> Et2 =        e2.x*cti2*cpi2 + e2.y*cti2*spi2 - e2.z*sti2;
		cuCPLX<float> Ep2 = (-1.0)*e2.x*spi2      + e2.y*cpi2;
		// Surface current components (Jx2, Jy2) in local Cartesian coordinates (no z component)
		cuCPLX<float> tp1 = (-1.0)*Et2*Rf.TM;
		cuCPLX<float> tp2 = Ep2*Rf.TE*cti2;
		cuCPLX<float> Jx2 = tp1*cpi2 + tp2*spi2;	// cti2 added
		cuCPLX<float> Jy2 = tp1*spi2 - tp2*cpi2;	// cti2 added


//		float Dp = k0 * ( tri.P0.x*u_ui + tri.P0.y*v_vi + tri.P0.z*w_wi );
//		// P2到P1 向量 dot 半徑向量(Rs)，P2到P1向量 在 半徑向量 的分量
//		float Dq = k0 * ( tri.P1.x*u_ui + tri.P1.y*v_vi + tri.P1.z*w_wi );
//		// 圓點到P2向量(P2位置向量) dot 半徑向量(Rs)
////		float Dr = 0;

		float Dp = k0 * ( tri.P0.x*u_ui + tri.P0.y*v_vi + tri.P0.z*w_wi );	// P0 = V0 - V2
		// P2到P1 向量 dot 半徑向量(Rs)，P2到P1向量 在 半徑向量 的分量
		float Dq = k0 * ( tri.P1.x*u_ui + tri.P1.y*v_vi + tri.P1.z*w_wi );	// P1 = V1 - V2
		// 圓點到P2向量(P2位置向量) dot 半徑向量(Rs)
		float D0 = k0 * ( tri.V2.x*u_ui + tri.V2.y*v_vi + tri.V2.z*w_wi );
		// TODO: Debug 用的Release後要刪除
		D0 = 0;


		// Area integral for general case
		float DD = Dq-Dp;
//		float cDp = cos(Dp);
//		float sDp = sin(Dp);
//		float cDq = cos(Dq);
//		float sDq = sin(Dq);
		float cDp, sDp, cDq, sDq, cD0, sD0;
		sincosf(Dp, &sDp, &cDp);
		sincosf(Dq, &sDq, &cDq);
		sincosf(D0, &sD0, &cD0);

		cuCPLX<float> expDp(cDp,sDp);
		cuCPLX<float> expDq(cDq,sDq);
		cuCPLX<float> expD0(cD0,sD0);

		cuCPLX<float> exp_Dp( expDp.r, -expDp.i );
		cuCPLX<float> exp_Dq( expDq.r, -expDq.i );

		// Special case 1
		cuCPLX<float> sic(0,0);
		cuCPLX<float> Ic(0,0);


		cuCPLX<float> po_Ets(0,0);
		cuCPLX<float> po_Eps(0,0);


		//----------------------------------------------------------------------------
		// Calcaulte surface current
		// Special case 1
		float Dp_abs = fabsf(Dp);
		float Dq_abs = fabsf(Dq);
		float DD_abs = fabsf(DD);

//		CASE-1, (k=244, i=0)
//		CASE-3, (k=123, i=0)
//		CASE-4, (k=122, i=0)
//		CASE-0, (k=252, i=0)

		cuCPLX<float> Ic1, Ic2;


		if((Dp_abs < Taylor.Rg) && (Dq_abs >= Taylor.Rg)){
			for(int q=0;q<=Taylor.Nt;++q){
				sic += cu::PowJPhasexCPLX(Dp, q, (-Co/(q+1) + expDq*(Co*cu::G(q,-Dq,exp_Dq))) ) / cu::factrl(q);
			}
//			Ic = sic * 2 / cuCPLX<float>(0,1) / Dq;
//			Ic  = sic * cuCPLX<float>(0,-2.0/Dq);
//			Ic1 = sic * 2 / cuCPLX<float>(0,1) / Dq;
//			Ic2 = sic * 2 * expD0 / cuCPLX<float>(0,1) / Dq;
			Ic = sic * 2 * expD0 / cuCPLX<float>(0,1) / Dq;

//			printf("CASE-1, (k=%d, i=%d)\n", k, i);

			// 121, 256, 237, 122,184,145
//			if(k==11353){
//				printf("CASE-1, (k=%ld, i=%ld), (Rg, Nt)=(%f,%f), u_ui=%f, v_vi=%f, w_wi=%f, w_wi2=%f, V0=[%f,%f,%f], V1=[%f,%f,%f], V2=[%f,%f,%f], k0=%f, Co=%f, Dp=%f, Dq=%f, D0=%f, Ic=(%f,%f), Ic1=(%f,%f), Ic2=(%f,%f)\n",
//						k, i, Taylor.Rg, Taylor.Nt,
//						u_ui, v_vi, w_wi, w_wi2,
//						tri.V0.x, tri.V0.y, tri.V0.z,
//						tri.V1.x, tri.V1.y, tri.V1.z,
//						tri.V2.x, tri.V2.y, tri.V2.z,
//						k0, Co, Dp, Dq, D0,
//						Ic.r, Ic.i,
//						Ic1.r, Ic1.i,
//						Ic2.r, Ic2.i);
//			}

		// Special case 2
		}else if ((Dp_abs < Taylor.Rg) && (Dq_abs < Taylor.Rg)){
//			printf("CASE-2, (k=%d, i=%d)\n", k, i);
			for(int q=0;q<=Taylor.Nt;++q){
				for(int nn=0;nn<=Taylor.Nt;++nn){
					sic += cu::PowJPhasexCPLX(Dp, q, cu::PowJPhase(Dq, nn)) / cu::factrl(nn+q+2) * Co;
				}
			}
//			Ic = sic * 2;
			Ic = sic * 2 * expD0;
		// Special case 3
		}else if ((Dp_abs >= Taylor.Rg) && (Dq_abs < Taylor.Rg)){
//			if(k==123 && i==0){ printf("CASE-3, (k=%d, i=%d)\n", k, i); }
			for(int q=0;q<=Taylor.Nt;++q){
				sic += cu::PowJPhasexCPLX(Dq, q, cu::G(q+1,-Dp,exp_Dp)) / (cu::factrl(q)*Co * (q+1));
			}
//			Ic = sic * 2 * expDp;
			Ic = sic * 2 * expD0 * expDp;
		// Special case 4
		}else if ((Dp_abs >= Taylor.Rg) && (Dq_abs >= Taylor.Rg) && (DD_abs < Taylor.Rg)){
//			if(k==122 && i==0){ printf("CASE-4, (k=%d, i=%d)\n", k, i); }
			for(int q=0;q<=Taylor.Nt;++q){
				sic += cu::PowJPhasexCPLX(DD, q, (-Co*cu::G(q,Dq,expDq)+expDq*Co/(q+1))/cu::factrl(q) );
			}
//			Ic = sic * 2 / cuCPLX<float>(0,1) / Dq;
//			Ic = sic * cuCPLX<float>(0,-2.0/Dq);
			Ic = sic * 2 * expD0 / cuCPLX<float>(0,1) / Dq;
		// Default
		}else{
//			if(k==252 && i==0){ printf("CASE-0, (k=%d, i=%d)\n", k, i); }
//			if(std::abs(DD) < 0.0001){
//				Ic = 2.f*( (expDp*(Co/Dp/(10000.f*DD)) - expDq*(Co/Dq/(10000.f*DD)))/10000.f - (Co/Dp/Dq) );
//			}else{
//				Ic = 2.f*( expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD) - (Co/Dp/Dq) );
				Ic = 2.f * expD0 * ( expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD) - (Co/Dp/Dq) );
//			}
		}
		//----------------------------------------------------------------------------




		// Scattered field components for triangle m in local coordinates
		cuVEC<cuCPLX<float> > Es2(Jx2*Ic, Jy2*Ic, cuCPLX<float>(0,0));
		// Transform back to global coordinates, then sum field
		cuVEC<cuCPLX<float> > Es1( tri.cb * Es2.x + tri.sb * Es2.z,
								   Es2.y,
							      -tri.sb * Es2.x + tri.cb * Es2.z);
		cuVEC<cuCPLX<float> > Es0( tri.ca * Es1.x - tri.sa * Es1.y,
								   tri.sa * Es1.x + tri.ca * Es1.y,
								   Es1.z );

		po_Ets += uu * Es0.x; po_Ets += vv * Es0.y; po_Ets += ww * Es0.z;	// 散射場在theta方向分量(H)
		po_Eps += (-sp*Es0.x + cp*Es0.y);									// 散射場在phi方向分量(V)


		// Complex form
		cuVEC<cuCPLX<float> > Es_cplx( po_Ets * g.theta_vec.x + po_Eps * g.phi_vec.x,
									   po_Ets * g.theta_vec.y + po_Eps * g.phi_vec.y,
									   po_Ets * g.theta_vec.z + po_Eps * g.phi_vec.z );


//		if(k == 207689/2 && i == 0){
//			printf("[GPU] k=%ld, i=%ld, Rc=[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
//					Rc.x.r, Rc.x.i, Rc.y.r, Rc.y.i, Rc.z.r, Rc.z.i);
//			printf("[GPU] k=%ld, i=%ld, e2=[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
//					e2.x.r, e2.x.i, e2.y.r, e2.y.i, e2.z.r, e2.z.i);
//			printf("[GPU] k=%ld, i=%ld, Et2=(%f,%f), Ep2=(%f,%f), Dp=%f, Dq=%f\n", k, i,
//					Et2.r, Et2.i, Ep2.r, Ep2.i, Dp, Dq);
//			printf("[GPU] k=%ld, i=%ld, P0=(%f,%f,%f), P1=(%f,%f,%f)\n", k, i,
//					P0.x, P0.y, P0.z, P1.x, P1.y, P1.z);
//			printf("[GPU] k=%ld, i=%ld, Jx2=(%f,%f), Jy2=(%f,%f), Ic=(%f,%f)\n", k, i,
//					Jx2.r, Jx2.i, Jy2.r, Jy2.i, Ic.r, Ic.i);
//			printf("[GPU] k=%ld, i=%ld, po_Ets=(%f,%f), po_Eps=(%f,%f)\n", k, i,
//					po_Ets.r, po_Ets.i, po_Eps.r, po_Eps.i);
////			printf("[GPU] k=%ld, i=%ld, Es_cplx=[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
////					Es_cplx.x.r, Es_cplx.x.i, Es_cplx.y.r, Es_cplx.y.i, Es_cplx.z.r, Es_cplx.z.i);
//		}


		return Es_cplx;
	}

 	__device__
	cuVEC<cuCPLX<float> > POCalculate(const cuElectricField& Ei, const cuTRI<float>& tri, const double k0,			// input
								const cuRF& Rf, const cuTAYLOR<float>& Taylor,								// input
								const float sp, const float cp,												// input
								const float sti2, const float cti2, const float spi2, const float cpi2,		// input
								const float uu, const float vv, const float ww,								// input
								const double u_ui, const double v_vi, const double w_wi,						// input
								const cuVEC<double>& k_obv, const cuVEC<float>& o, const cuThetaPhiVec g, const size_t k, const size_t i){	// input

		// residual
		// E-field Amplitude
 //		cuVEC<float> Ei_abs = abs(Ei.cplx);
 //		float Co = sqrtf(Ei_abs.x*Ei_abs.x + Ei_abs.y*Ei_abs.y + Ei_abs.z*Ei_abs.z);
		float Co = (abs(Ei.cplx)).abs();

		// Incident field in global Cartesian coordinates (Bistatic)
		cuVEC<cuCPLX<float> > Rc = Ei.cplx;

		// Incident field in local Cartesian coordinates (stored in e2)
		cuVEC<cuCPLX<float> > e2 = cu::transform(tri.sa, tri.ca, tri.sb, tri.cb, Rc);

		cuVEC<cuCPLX<float> > e2_1( (tri.ca * Rc.x) + (tri.sa * Rc.y),
								   (-tri.sa * Rc.x) + (tri.ca * Rc.y),
									 Rc.z );

		// Incident field in local spherical coordinates
		cuCPLX<float> Et2 =        e2.x*cti2*cpi2 + e2.y*cti2*spi2 - e2.z*sti2;
		cuCPLX<float> Ep2 = (-1.0)*e2.x*spi2      + e2.y*cpi2;
		// Surface current components (Jx2, Jy2) in local Cartesian coordinates (no z component)
		cuCPLX<float> tp1 = (-1.0)*Et2*Rf.TM;
		cuCPLX<float> tp2 = Ep2*Rf.TE*cti2;
		cuCPLX<float> Jx2 = tp1*cpi2 + tp2*spi2;	// cti2 added
		cuCPLX<float> Jy2 = tp1*spi2 - tp2*cpi2;	// cti2 added

		// cuVEC<T> P0, P1;		// Edge vector P0=v0-v2, P1=v1-v2, P2=[0,0,0]
		cuVEC<float> P0 = tri.V0 - tri.V2;
		cuVEC<float> P1 = tri.V1 - tri.V2;

		float Dp = k0 * ( P0.x*u_ui + P0.y*v_vi + P0.z*w_wi );
		// P2到P1 向量 dot 半徑向量(Rs)，P2到P1向量 在 半徑向量 的分量
		float Dq = k0 * ( P1.x*u_ui + P1.y*v_vi + P1.z*w_wi );
		// 圓點到P2向量(P2位置向量) dot 半徑向量(Rs)
 //		float Dr = 0;

		// Area integral for general case
		float DD = Dq-Dp;
 //		float cDp = cos(Dp);
 //		float sDp = sin(Dp);
 //		float cDq = cos(Dq);
 //		float sDq = sin(Dq);
		float cDp, sDp, cDq, sDq;
		sincosf(Dp, &sDp, &cDp);
		sincosf(Dq, &sDq, &cDq);

		cuCPLX<float> expDp(cDp,sDp);
		cuCPLX<float> expDq(cDq,sDq);

		cuCPLX<float> exp_Dp( expDp.r, -expDp.i );
		cuCPLX<float> exp_Dq( expDq.r, -expDq.i );

		// Special case 1
		cuCPLX<float> sic(0,0);
		cuCPLX<float> Ic(0,0);


		cuCPLX<float> po_Ets(0,0);
		cuCPLX<float> po_Eps(0,0);


		//----------------------------------------------------------------------------
		// Calcaulte surface current
		// Special case 1
		float Dp_abs = fabsf(Dp);
		float Dq_abs = fabsf(Dq);
		float DD_abs = fabsf(DD);


		if((Dp_abs < Taylor.Rg) && (Dq_abs >= Taylor.Rg)){
			for(int q=0;q<=Taylor.Nt;++q){
				sic += cu::PowJPhasexCPLX(Dp, q, (-Co/(q+1) + expDq*(Co*cu::G3(q,-Dq))) ) / cu::factrl(q);
			}
 //			Ic = sic * 2 / cuCPLX<float>(0,1) / Dq;
			Ic = sic * cuCPLX<float>(0,-2.0/Dq);
		// Special case 2
		}else if ((Dp_abs < Taylor.Rg) && (Dq_abs < Taylor.Rg)){
			for(int q=0;q<=Taylor.Nt;++q){
				for(int nn=0;nn<=Taylor.Nt;++nn){
					sic += cu::PowJPhasexCPLX(Dp, q, cu::PowJPhase(Dq, nn)) / cu::factrl(nn+q+2) * Co;
				}
			}
			Ic = sic * 2;
		// Special case 3
		}else if ((Dp_abs >= Taylor.Rg) && (Dq_abs < Taylor.Rg)){
			for(int q=0;q<=Taylor.Nt;++q){
				sic += cu::PowJPhasexCPLX(Dq, q, cu::G3(q+1,-Dp)) * Co / (cu::factrl(q) * (q+1));
// 				sic += cu::PowJPhasexCPLX(Dq, q, cu::G(q+1,-Dp,exp_Dp)) / (cu::factrl(q)*Co * (q+1));
			}
			Ic = sic * 2 * expDp;
		// Special case 4
		}else if ((Dp_abs >= Taylor.Rg) && (Dq_abs >= Taylor.Rg) && (DD_abs < Taylor.Rg)){
			for(int q=0;q<=Taylor.Nt;++q){
				sic += cu::PowJPhasexCPLX(DD, q, (-Co*cu::G3(q,Dq)+expDq*Co/(q+1))/cu::factrl(q) );
			}
 //			Ic = sic * 2 / cuCPLX<float>(0,1) / Dq;
			Ic = sic * cuCPLX<float>(0,-2.0/Dq);
		// Default
		}else{
 //			if(std::abs(DD) < 0.0001){
 //				Ic = 2.f*( (expDp*(Co/Dp/(10000.f*DD)) - expDq*(Co/Dq/(10000.f*DD)))/10000.f - (Co/Dp/Dq) );
 //			}else{
				Ic = 2.f*( expDp*(Co/Dp/DD) - expDq*(Co/Dq/DD) - (Co/Dp/Dq) );
 //			}
		}
		//----------------------------------------------------------------------------

// 		cuVEC<cuCPLX<float> > IcXYZ = cross(tri.N, cross(Ei.k, Ei.cplx));
// 		Ic.r = sqrt(IcXYZ.x.r*IcXYZ.x.r + IcXYZ.y.r*IcXYZ.y.r + IcXYZ.z.r*IcXYZ.z.r);
// 		Ic.i = sqrt(IcXYZ.x.i*IcXYZ.x.i + IcXYZ.y.i*IcXYZ.y.i + IcXYZ.z.i*IcXYZ.z.i);


		// Scattered field components for triangle m in local coordinates
		cuVEC<cuCPLX<float> > Es2(Jx2*Ic, Jy2*Ic, cuCPLX<float>(0,0));
		// Transform back to global coordinates, then sum field
		cuVEC<cuCPLX<float> > Es1( tri.cb * Es2.x + tri.sb * Es2.z,
								   Es2.y,
								  -tri.sb * Es2.x + tri.cb * Es2.z);
		cuVEC<cuCPLX<float> > Es0( tri.ca * Es1.x - tri.sa * Es1.y,
								   tri.sa * Es1.x + tri.ca * Es1.y,
								   Es1.z );

//		po_Ets += uu * Es0.x; po_Ets += vv * Es0.y; po_Ets += ww * Es0.z;	// 散射場在theta方向分量(H)
		po_Ets += (uu * Es0.x + vv * Es0.y + ww * Es0.z);					// 散射場在theta方向分量(H)
		po_Eps += (-sp*Es0.x + cp*Es0.y);									// 散射場在phi方向分量(V)


		// Complex form
		cuVEC<cuCPLX<float> > Es_cplx( po_Ets * g.theta_vec.x + po_Eps * g.phi_vec.x,
									   po_Ets * g.theta_vec.y + po_Eps * g.phi_vec.y,
									   po_Ets * g.theta_vec.z + po_Eps * g.phi_vec.z );


		if(k == 207689/2 && i == 0){
			printf("[GPU] k=%ld, i=%ld, Rc=[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
					Rc.x.r, Rc.x.i, Rc.y.r, Rc.y.i, Rc.z.r, Rc.z.i);
			printf("[GPU] k=%ld, i=%ld, e2=[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
					e2.x.r, e2.x.i, e2.y.r, e2.y.i, e2.z.r, e2.z.i);
			printf("[GPU] k=%ld, i=%ld, Et2=(%f,%f), Ep2=(%f,%f), Dp=%f, Dq=%f\n", k, i,
					Et2.r, Et2.i, Ep2.r, Ep2.i, Dp, Dq);
			printf("[GPU] k=%ld, i=%ld, P0=(%f,%f,%f), P1=(%f,%f,%f)\n", k, i,
					P0.x, P0.y, P0.z, P1.x, P1.y, P1.z);
			printf("[GPU] k=%ld, i=%ld, Jx2=(%f,%f), Jy2=(%f,%f), Ic=(%f,%f)\n", k, i,
					Jx2.r, Jx2.i, Jy2.r, Jy2.i, Ic.r, Ic.i);
			printf("[GPU] k=%ld, i=%ld, po_Ets=(%f,%f), po_Eps=(%f,%f)\n", k, i,
					po_Ets.r, po_Ets.i, po_Eps.r, po_Eps.i);
//			printf("[GPU] k=%ld, i=%ld, Es_cplx=[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
//					Es_cplx.x.r, Es_cplx.x.i, Es_cplx.y.r, Es_cplx.y.i, Es_cplx.z.r, Es_cplx.z.i);
		}


		return Es_cplx;
	}

 	__device__
	cuVEC<cuCPLX<float> > POCalculate2(const cuElectricField& Ei, const cuTRI<float>& tri, const double k0,			// input
								const cuRF& Rf,
//								const cuTAYLOR<float>& Taylor,								// input
//								const double sinc_path,
								const float sp, const float cp, //const float st, const float ct,				// input
								const float sti2, const float cti2, const float spi2, const float cpi2,		// input
								const float uu, const float vv, const float ww,								// input
//								const double u_ui, const double v_vi, const double w_wi,						// input
//								const cuVEC<double>& k_obv, const cuVEC<float>& o,
								const cuThetaPhiVec g, const size_t k, const size_t i){	// input

		// Incident field in global Cartesian coordinates (Bistatic)
		cuVEC<cuCPLX<float> > Rc = Ei.cplx;	// == e0

		// Incident field in local Cartesian coordinates (stored in e2)
		cuVEC<cuCPLX<float> > e2 = cu::transform(tri.sa, tri.ca, tri.sb, tri.cb, Rc);

		// Incident field in local spherical coordinates
		cuCPLX<float> Et2 =        e2.x*cti2*cpi2 + e2.y*cti2*spi2 - e2.z*sti2;
		cuCPLX<float> Ep2 = (-1.0)*e2.x*spi2      + e2.y*cpi2;
		// Surface current components (Jx2, Jy2) in local Cartesian coordinates (no z component)
		cuCPLX<float> tp1 = (-1.0)*Et2*Rf.TM;
		cuCPLX<float> tp2 = Ep2*Rf.TE;
		cuCPLX<float> Jx2 = tp1*cpi2 + tp2*spi2*cti2;	// cti2 added
		cuCPLX<float> Jy2 = tp1*spi2 - tp2*cpi2*cti2;	// cti2 added


		// Scattered field components for triangle m in local coordinates
		cuVEC<cuCPLX<float> > Es2(Jx2, Jy2, cuCPLX<float>(0,0));
		// Transform back to global coordinates, then sum field
		cuVEC<cuCPLX<float> > Es1( tri.cb * Es2.x + tri.sb * Es2.z,
								   Es2.y,
								  -tri.sb * Es2.x + tri.cb * Es2.z);
		cuVEC<cuCPLX<float> > Es0( tri.ca * Es1.x - tri.sa * Es1.y,
								   tri.sa * Es1.x + tri.ca * Es1.y,
								   Es1.z );

		cuCPLX<float> po_Ets = (uu * Es0.x + vv * Es0.y + ww * Es0.z);					// 散射場在theta方向分量(H)
		cuCPLX<float> po_Eps = (-sp*Es0.x + cp*Es0.y);									// 散射場在phi方向分量(V)


		// Complex form
		cuVEC<cuCPLX<float> > Es_cplx( po_Ets * g.theta_vec.x + po_Eps * g.phi_vec.x,
									   po_Ets * g.theta_vec.y + po_Eps * g.phi_vec.y,
									   po_Ets * g.theta_vec.z + po_Eps * g.phi_vec.z );

		return Es_cplx;
	}

 	__device__
	cuVEC<cuCPLX<float> > POCalculateCross(const cuElectricField& Ei, const cuTRI<float>& tri, const double k0,			// input
								const cuRF& Rf, const cuTAYLOR<float>& Taylor,								// input
								const float sp, const float cp,	const float st, const float ct,				// input
								const float sti2, const float cti2, const float spi2, const float cpi2,		// input
								const float uu, const float vv, const float ww,								// input
								const double u_ui, const double v_vi, const double w_wi,						// input
								const cuVEC<double>& k_obv, const cuVEC<float>& o, const cuThetaPhiVec g, const size_t k, const size_t i){	// input


 		// New method using cross product
 		cuVEC<cuCPLX<float> > Es_cplx_in = cross(tri.N, cross(Ei.k, Ei.cplx));


// 		//+--------------------------------------------------------------------+
//		//|   (1) Find Jx2/Jy2 & Jx2_PEC&Jy2_PEC                               |
//		//+--------------------------------------------------------------------+
// 		// Incident field in global Cartesian coordinates (Bistatic)
// 		cuVEC<float> D0(st*cp, st*sp, ct);
// 		// Incident field in local Cartesian coordinates (stored in e2)
// 		cuVEC<float> e2 = cu::transform(tri.sa, tri.ca, tri.sb, tri.cb, D0);

// 		//+--------------------------------------------------------------------+
// 		//|   (1) Find Jx2/Jy2 & Jx2_PEC&Jy2_PEC                               |
// 		//+--------------------------------------------------------------------+
// 		// Incident field in global Cartesian coordinates (Bistatic)
//		cuVEC<cuCPLX<float> > Rc = Ei.cplx;
//
//		// Incident field in local Cartesian coordinates (stored in e2)
//		cuVEC<cuCPLX<float> > e2 = cu::transform(tri.sa, tri.ca, tri.sb, tri.cb, Rc);
//
//		// Incident field in local spherical coordinates
//		cuCPLX<float> Et2 =        e2.x*cti2*cpi2 + e2.y*cti2*spi2 - e2.z*sti2;
//		cuCPLX<float> Ep2 = (-1.0)*e2.x*spi2      + e2.y*cpi2;
//		// Surface current components (Jx2, Jy2) in local Cartesian coordinates (no z component)
//		cuCPLX<float> tp1 = (-1.0)*Et2*Rf.TM;
//		cuCPLX<float> tp2 = Ep2*Rf.TE*cti2;
//		cuCPLX<float> Jx2 = tp1*cpi2 + tp2*spi2;
//		cuCPLX<float> Jy2 = tp1*spi2 - tp2*cpi2;
//
//		// Find surface current, Jx2 & Jy2 for PEC
//		cuCPLX<float> Rf_TE_PEC(-1., 0.);
//		cuCPLX<float> Rf_TM_PEC(-1., 0.);
//		cuCPLX<float> tp1_PEC = (-1.0)*Et2*Rf_TM_PEC;
//		cuCPLX<float> tp2_PEC = Ep2*Rf_TE_PEC*cti2;
//		cuCPLX<float> Jx2_PEC = tp1_PEC*cpi2 + tp2_PEC*spi2;	// cti2 added
//		cuCPLX<float> Jy2_PEC = tp1_PEC*spi2 - tp2_PEC*cpi2;	// cti2 added
//
//		//+--------------------------------------------------------------------+
//		//|   (2) Decomposite total E-field to global theta and phi component  |
//		//+--------------------------------------------------------------------+
//		cuCPLX<float> po_Ets = (Es_cplx_in.x*g.phi_vec.y - Es_cplx_in.y*g.phi_vec.x) / (g.theta_vec.x*g.phi_vec.y - g.theta_vec.y*g.phi_vec.x);
//		cuCPLX<float> po_Eps = (Es_cplx_in.x - po_Ets*g.theta_vec.x)/g.phi_vec.x;
//
//		//+--------------------------------------------------------------------+
//		//|   (3) Find Scattering E-field at local coordinate                  |
//		//+--------------------------------------------------------------------+
//		float A = (uu*tri.ca*tri.cb+vv*tri.sa*tri.cb-ww*tri.sb);
//		float B = (-uu*tri.sa+vv*tri.ca);
//		float C = (-sp*tri.ca*tri.cb+cp*tri.sa*tri.cb);
//		float D = (sp*tri.sa+cp*tri.ca);
//
//		cuCPLX<float> Jx2Ic = (po_Ets*D-po_Eps*B) / (A*D-C*B);	// == Jx2*Ic, cuCPLX<float> (Jx2Ic)
//		cuCPLX<float> Jy2Ic = (po_Ets/B-(A/B)*Jx2Ic); 			// == Jy2*Ic, cuCPLX<float> (Jy2Ic)
//
//		//+--------------------------------------------------------------------+
//		//|   (4) Remove PEC surface current coeffienet                        |
//		//+--------------------------------------------------------------------+
//		cuCPLX<float> Ic = Jx2Ic / Jx2_PEC;
//		cuCPLX<float> Ic2 = Jy2Ic / Jy2_PEC;
//
//		//+--------------------------------------------------------------------+
//		//|   (5) Add someone matrial                                          |
//		//+--------------------------------------------------------------------+
////		// Scattered field components for triangle m in local coordinates
////		cuVEC<cuCPLX<float> > Es2(Jx2*Ic, Jy2*Ic, cuCPLX<float>(0,0));
////		// Transform back to global coordinates, then sum field
////		cuVEC<cuCPLX<float> > Es1( tri.cb * Es2.x + tri.sb * Es2.z,
////								   Es2.y,
////								  -tri.sb * Es2.x + tri.cb * Es2.z);
////		cuVEC<cuCPLX<float> > Es0( tri.ca * Es1.x - tri.sa * Es1.y,
////								   tri.sa * Es1.x + tri.ca * Es1.y,
////								   Es1.z );
////
////		po_Ets = (uu * Es0.x + vv * Es0.y + ww * Es0.z);					// 散射場在theta方向分量(H)
////		po_Eps = (-sp*Es0.x + cp*Es0.y);									// 散射場在phi方向分量(V)
////
////		// 散射場在theta方向分量(H)
////		po_Ets = (uu*tri.ca*tri.cb+vv*tri.sa-ww*tri.sb)*(Jx2*Ic) + (-uu*tri.sa+vv*tri.ca)*(Jy2*Ic);
////		po_Eps = (-sp*tri.ca*tri.cb+cp*tri.sa)*(Jx2*Ic) + (sp*tri.sa+cp*tri.ca)*(Jy2*Ic);
//
//		po_Ets = A*(Jx2*Ic) + B*(Jy2*Ic);
//		po_Eps = C*(Jx2*Ic) + D*(Jy2*Ic);
//
//		// Complex form
//		cuVEC<cuCPLX<float> > Es_cplx( po_Ets * g.theta_vec.x + po_Eps * g.phi_vec.x,
//									   po_Ets * g.theta_vec.y + po_Eps * g.phi_vec.y,
//									   po_Ets * g.theta_vec.z + po_Eps * g.phi_vec.z );

////		if(k == 207689/2 && i == 0){
//		if(k == 1217642/2 && i == 0){
//			printf("k=%ld, i=%ld, Ex_cplx_in=[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
//					Es_cplx_in.x.r, Es_cplx_in.x.i, Es_cplx_in.y.r, Es_cplx_in.y.i, Es_cplx_in.z.r, Es_cplx_in.z.i);
//			printf("k=%ld, i=%ld, Ex_cplx   =[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
//					Es_cplx.x.r, Es_cplx.x.i, Es_cplx.y.r, Es_cplx.y.i, Es_cplx.z.r, Es_cplx.z.i);
//			printf("k=%ld, i=%ld, Rf.TE=(%f,%f), Rf.TM=(%f,%f) <--- for someone\n", k, i,
//					Rf.TE.r, Rf.TE.i, Rf.TM.r, Rf.TM.i);
//			printf("k=%ld, i=%ld, Rf.TE=(%f,%f), Rf.TM=(%f,%f) <--- for PEC\n", k, i,
//					Rf_TE_PEC.r, Rf_TE_PEC.i, Rf_TM_PEC.r, Rf_TM_PEC.i);
//			printf("k=%ld, i=%ld, Ic=(%f,%f), Ic2=(%f,%f)\n", k, i,
//					Ic.r, Ic.i, Ic2.r, Ic2.i);
//			printf("k=%ld, i=%ld, Jx2=(%f,%f), Jy2=(%f,%f), Jx2_PEC=(%f,%f), Jy2_PEC=(%f,%f)\n", k, i,
//					Jx2.r, Jx2.i, Jy2.r, Jy2.i, Jx2_PEC.r, Jx2_PEC.i, Jy2_PEC.r, Jy2_PEC.i);
//			printf("k=%ld, i=%ld, g.theta_vec=(%f,%f,%f), g.phi_vec=(%f,%f,%f)\n", k, i,
//					g.theta_vec.x, g.theta_vec.y, g.theta_vec.z, g.phi_vec.x, g.phi_vec.y, g.phi_vec.z);
//			printf("k=%ld, i=%ld, po_Ets=(%f,%f), po_Eps=(%f,%f)\n", k, i,
//					po_Ets.r, po_Ets.i, po_Eps.r, po_Eps.i);
//			printf("k=%ld, i=%ld, (g.theta_vec.x*g.phi_vec.y - g.theta_vec.y*g.phi_vec.x)=%f\n", k, i,
//					(g.theta_vec.x*g.phi_vec.y - g.theta_vec.y*g.phi_vec.x));
//			printf("k=%ld, i=%ld, (A,B,C,D)=(%f,%f,%f,%f)\n", k, i,
//					A, B, C, D);
////			printf("k=%ld, i=%ld, tri.cb=%f\n", k, i,
////					tri.cb);
////			tri.Print();
//			printf("k=%ld, i=%ld, Rc=[(%f,%f),(%f,%f),(%f,%f)]\n", k, i,
//					Rc.x.r, Rc.x.i, Rc.y.r, Rc.y.i, Rc.z.r, Rc.z.i);
//		}


 		cuVEC<cuCPLX<float> > Es_cplx;
		return Es_cplx;
	}

	__device__
	void warpReduce(volatile cuSBRElement<float>* sdata, size_t tid, size_t MaxLevel){
		size_t idx;
		for(size_t i=0;i<MaxLevel;++i){
			idx = tid * MaxLevel + i;
//			sdata[idx].sump.r += sdata[idx + 32*MaxLevel].sump.r;
//			sdata[idx].sump.i += sdata[idx + 32*MaxLevel].sump.i;
//			sdata[idx].sumt.r += sdata[idx + 32*MaxLevel].sumt.r;
//			sdata[idx].sumt.i += sdata[idx + 32*MaxLevel].sumt.i;
			sdata[idx].sump.r += sdata[idx + 16*MaxLevel].sump.r;
			sdata[idx].sump.i += sdata[idx + 16*MaxLevel].sump.i;
			sdata[idx].sumt.r += sdata[idx + 16*MaxLevel].sumt.r;
			sdata[idx].sumt.i += sdata[idx + 16*MaxLevel].sumt.i;
			sdata[idx].sump.r += sdata[idx + 8*MaxLevel].sump.r;
			sdata[idx].sump.i += sdata[idx + 8*MaxLevel].sump.i;
			sdata[idx].sumt.r += sdata[idx + 8*MaxLevel].sumt.r;
			sdata[idx].sumt.i += sdata[idx + 8*MaxLevel].sumt.i;
			sdata[idx].sump.r += sdata[idx + 4*MaxLevel].sump.r;
			sdata[idx].sump.i += sdata[idx + 4*MaxLevel].sump.i;
			sdata[idx].sumt.r += sdata[idx + 4*MaxLevel].sumt.r;
			sdata[idx].sumt.i += sdata[idx + 4*MaxLevel].sumt.i;
			sdata[idx].sump.r += sdata[idx + 2*MaxLevel].sump.r;
			sdata[idx].sump.i += sdata[idx + 2*MaxLevel].sump.i;
			sdata[idx].sumt.r += sdata[idx + 2*MaxLevel].sumt.r;
			sdata[idx].sumt.i += sdata[idx + 2*MaxLevel].sumt.i;
			sdata[idx].sump.r += sdata[idx + 1*MaxLevel].sump.r;
			sdata[idx].sump.i += sdata[idx + 1*MaxLevel].sump.i;
			sdata[idx].sumt.r += sdata[idx + 1*MaxLevel].sumt.r;
			sdata[idx].sumt.i += sdata[idx + 1*MaxLevel].sumt.i;
		}
	}




	//+======================================================+
	//|                     __global__                       |
	//+======================================================+
	__global__
	void cuScanPerBlock(int* d_b, int* d_a, size_t n){
		//
		// 注意：所有累加只在block內
		//
		// d_a 標記0或1的值，為輸入，並把後面的值累加到前面 (replace)
		// d_b 與d_a友相同的結果
		// n: d_a的大小
		// Example:
		//			0 1 1 0 0 1 1 1 1 1 0 1 0 1
		//          _ (finished)
		// off = 1  0 1 2 1 0 1 2 2 2 2 1 1 1 1
		//          ___
		// off = 2  0 1 2 2 2 2 2 3 4 4 3 3 2 2
		//          _______
		// off = 4  0 1 2 2 2 3 4 5 6 6 5 6 6 6
		//          _______________
		// off = 8  0 1 2 2 2 3 4 5 6 7 7 8 8 9
		int k = threadIdx.x + blockDim.x*blockIdx.x;
		int tid = threadIdx.x;

		int temp;

		if(k >= n){
			return;
		}


		for(int offset = 1; offset < blockDim.x; offset *= 2){
			if(tid >= offset){
				temp = d_a[k - offset];
				__syncthreads();
				d_a[k] += temp;
			}
			__syncthreads();
		}
		d_b[k] = d_a[k];

	}

	__global__
	void cuScan(int *d_d, int *d_b, size_t n, size_t nc, size_t* number){
		int tid = threadIdx.x;
		int nt, temp=0, x=0;

		for(nt=0;nt<nc;nt++){
			if(tid<n){
				if(tid > 0){
					d_d[tid] = d_b[tid];
					d_d[tid] += temp;
					__syncthreads();
				}else{
					d_d[tid] = d_b[tid];
				}
				tid = tid + blockDim.x;
				x   = x   + blockDim.x;
				temp = d_d[x-(1)];
				__syncthreads();
			}
		}
		if((tid-blockDim.x) == (n-1)){
			(*number) = d_d[n-1];
//			printf("Number = %lu, d_d[n-1] = %d, n = %lu\n", *number, d_d[n-1], n);
		}
	}

	__global__
	void cuScanExtract(int *d_valuex, int *newd_valuex, size_t arraysize){
		// Example:
		//	    thread: 0 1 2 3 4 5 6
		//    d_valuex: 0 1 1 2 3 4 4
		// newd_valuex: 1 3 4 5

		int k = threadIdx.x + blockIdx.x * blockDim.x;
		int val;

		if(k >= arraysize){
			return;
		}


		if(k == 0){
			if( d_valuex[0] == 1 ){
				newd_valuex[k] = 0;
			}
		}


		if(k > 0){
			val =  d_valuex[k-1];
			__syncthreads();
			if (d_valuex[k] != val){
				newd_valuex[d_valuex[k]-1]=k;
			}
			__syncthreads();
		}
	}

	void cuPOApproxScan(int* d_in_out, int* d_temp1, int* d_temp2, size_t& number, size_t size, int NThread){
		// Purpose:
		// 		Extract the index when the value is "1" otherwise is "0"
		//
		// Example:
		//		Original :
		//			0 1 1 0 0 1 1 1 1 0 0 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 1 1 0 0 0 0 1 0 0 1 0 1
		//		Extraction index :
		//			1 2 5 6 7 8 11 12 14 15 18 19 21 23 24 25 26 27 32 35 37
		//		Total number = 21
		//
		size_t* d_number;
		cudaMalloc(&d_number, sizeof(size_t));
		int NBlock = ceil((float)size/NThread);
		cuScanPerBlock<<<NBlock, NThread>>>(d_temp1, d_in_out, size);
		cuScan<<<1, NThread>>>(d_temp2, d_temp1, size, NBlock, d_number);
		cudaMemcpy(&number, d_number, sizeof(size_t), cudaMemcpyDeviceToHost);
		if(number > 0){
			cuScanExtract<<<NBlock, NThread>>>(d_temp2, d_in_out, size);
		}
		cudaFree(d_number);
	}

	void cuPOApproxScan(int* d_tag, int* d_idx, int* d_temp1, int* d_temp2, size_t& number, size_t size, int NThread){
		// Purpose:
		// 		Extract the index when the value is "1" otherwise is "0"
		//
		// Example:
		//		Original :
		//			0 1 1 0 0 1 1 1 1 0 0 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 1 1 0 0 0 0 1 0 0 1 0 1
		//		Extraction index :
		//			1 2 5 6 7 8 11 12 14 15 18 19 21 23 24 25 26 27 32 35 37
		//		Total number = 21
		//
		size_t* d_number;
		cudaMalloc(&d_number, sizeof(size_t));
		int NBlock = ceil((float)size/NThread);
		cuScanPerBlock<<<NBlock, NThread>>>(d_temp1, d_tag, size);
		cuScan<<<1, NThread>>>(d_temp2, d_temp1, size, NBlock, d_number);
		cuScanExtract<<<NBlock, NThread>>>(d_temp2, d_idx, size);
		cudaMemcpy(&number, d_number, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(d_number);
	}

}  // namespace cu



#endif /* CUMISC_CUH_ */
