/*
 * cuopt.cuh
 *
 *  Created on: Dec 25, 2014
 *      Author: cychiang
 */

#ifndef CUOPT_CUH_
#define CUOPT_CUH_


void showCudaError(const cudaError_t err){
	if(err != cudaSuccess){
		cout<<"showCudaError::"<<cudaGetErrorString(err)<<endl;
	}
}

void SafeMallocCheck(const cudaError_t err){
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

	if(err != cudaSuccess){
		cerr<<"+-----------------------------------------------------------+"<<endl;
		cerr<<"|  ERROR::Memory allocate out of device size                |"<<endl;
		cerr<<"|         Please turn down the 'Mesh Scale Factor' value.   |"<<endl;
		cerr<<"+-----------------------------------------------------------+"<<endl;
		cerr<<"The total global memory is "<<double(prop.totalGlobalMem)/1024/1024/1024<<" [GB]"<<endl;
		cerr<<"Message: "<<cudaGetErrorString(err)<<endl;
		// Return memory usage
		size_t mem_free, mem_total, mem_use;
		cudaMemGetInfo(&mem_free, &mem_total);
		mem_use = mem_total - mem_free;
		cout<<"+"<<endl;
		cout<<" mem Total = "<<double(mem_total)/1024./1024.<<" [MBytes]"<<endl;
		cout<<" mem Free  = "<<double(mem_free)/1024./1024. <<" [MBytes]"<<endl;
		cout<<" mem usage = "<<double(mem_use) /1024./1024. <<" [MBytes]"<<endl;
		cout<<"+"<<endl;
		exit(EXIT_FAILURE);
	}
}

// fast truncation of double-precision to integers
#define CUMP_D2I_TRUNC (double)(3ll << 51)
// computes r = a + b subop c unsigned using extended precision
#define VADDx(r, a, b, c, subop) \
    asm volatile("vadd.u32.u32.u32." subop " %0, %1, %2, %3;" :  \
            "=r"(r) : "r"(a) , "r"(b), "r"(c));
// Check CUDA RETURN ERROR
#define CUDA_CHECK_RETURN(value) {                                      \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                                                            \
} }


namespace cu {
namespace opt {

	//declare constant memory
	__constant__ double c_FACTORIAL[] = {1,1,2,6,24,120,720,5040,40320,362880,3628800,
							  			 39916800., 479001600.,6227020800.,87178291200.,
							  			 1307674368000.,20922789888000.,355687428096000.,
							  			 6402373705728000.,121645100408832000.,
							  			 2432902008176640000.};
	__constant__ double c_EPSILON[] = { 0.000001, -0.000001 };
	__constant__ double c_ONE[] = { 1, -1 };
	__constant__ double c_powi[]= { 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1};

#ifdef LIMITNUMBER
	__constant__ bool c_Is[] = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
	__constant__ int  c_ev[] = { 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1};
	__constant__ int  c_od[] = { 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1};
#endif

// 	//declare constant memory
// 	__constant__ double c_FACTORIAL[21];
// 	__constant__ double c_EPSILON[2];
// 	__constant__ double c_ONE[2];

// #ifdef LIMITNUMBER
// 	__constant__ bool c_Is[16];
// 	__constant__ int  c_ev[16];
// 	__constant__ int  c_od[16];
// #endif


// 	void init_const_variable(){
// #ifdef LIMITNUMBER
// 		bool Is[] = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};	// ((pow % 2) == 0)
// 		int  ev[] = { 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1};	// (((pow-2) % 4) == 0)? -1:1
// 		int  od[] = { 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1};	// (((pow-3) % 4) == 0)? -1:1
// 		//copy host angle data to constant memory
// 		cudaMemcpyToSymbol(c_Is, Is, 16*sizeof(bool));
// 		cudaMemcpyToSymbol(c_ev, ev, 16*sizeof(int));
// 		cudaMemcpyToSymbol(c_od, od, 16*sizeof(int));
// #endif
// 		double FACTORIAL[] = {1,1,2,6,24,120,720,5040,40320,362880,3628800,
// 							  39916800., 479001600.,6227020800.,87178291200.,
// 							  1307674368000.,20922789888000.,355687428096000.,
// 							  6402373705728000.,121645100408832000.,
// 							  2432902008176640000.};
// 		//copy host angle data to constant memory
// 		cudaMemcpyToSymbol(c_FACTORIAL, FACTORIAL, 21*sizeof(double));
// 		// EPSILON
// 		double Epsilon[] = { 0.000001, -0.000001 };
// 		cudaMemcpyToSymbol(c_EPSILON, Epsilon, 2*sizeof(double));
// 		// ONE
// 		double One[] = { 1, -1 };
// 		cudaMemcpyToSymbol(c_ONE, One, 2*sizeof(double));
// 		// Check error
// 		ChkErr("cu::opt::init_const_variable");
// 	}

//#ifdef LIMITNUMBER
//		const __device__ bool Is[] = { 1, 0, 1, 0, 1, 0};
//		const __device__ int  ev[] = { 1, 1,-1, 1, 1, 1};
//		const __device__ int  od[] = { 1, 1, 1,-1, 1, 1};
//#endif

//	// computes a * b mod m; invk = (double)(1<<30) / m
//	__device__ __forceinline__
//	unsigned mod(unsigned a, unsigned b, volatile unsigned m, volatile double invk) {
//
//	   unsigned hi = __umulhi(a*2, b*2); // 3 flops
//	   // 2 double instructions
//	   double rf = __uint2double_rn(hi) * invk + CUMP_D2I_TRUNC;
//	   unsigned r = (unsigned)__double2loint(rf);
//	   r = a * b - r * m; // 2 flops
//
//	   // can also be replaced by: VADDx(r, r, m, r, "min") // == umin(r, r + m);
//	   if((int)r < 0)
//		  r += m;
//	   return r;
//	}
	__host__ __device__
	double fmod(double x,double y){
		double a = x/y;
		return (a-(int)a)*y;
//		double a;
//		return ((a=x/y)-(int)a)*y;
	};

//	template<int N> __host__ __device__ struct power_impl;
//
//	template<int N>
//	__host__ __device__
//	struct power_impl {
//	    template<typename T>
//	    static T calc(const T &x) {
//	        if (N%2 == 0)
//	            return power_impl<N/2>::calc(x*x);
//	        else if (N%3 == 0)
//	            return power_impl<N/3>::calc(x*x*x);
//	        return power_impl<N-1>::calc(x)*x;
//	    }
//	};
//
//	template<>
//	__host__ __device__
//	struct power_impl<0> {
//	    template<typename T>
//	    static T calc(const T &) { return 1; }
//	};
//
//	template<int N, typename T>
//	__host__ __device__
//	inline T pow(const T &x) {
//	    return power_impl<N>::calc(x);
//	}

}  // namespace opt
}  // namespace cu



#endif /* CUOPT_CUH_ */
