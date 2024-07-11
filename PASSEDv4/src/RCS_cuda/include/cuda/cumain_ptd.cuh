//
// Created by Steve Chiang on 2022/11/17.
//
#ifndef CUMAIN_PTD_CUH_
#define CUMAIN_PTD_CUH_

#include <cuda/cumisc.cuh>
#include <openssl/ssl.h>

namespace cu {

	//+=======================================================+
	//|                     CUDA Main                         |
	//+=======================================================+
	/**
	 * Line segment calss
	 * @tparam[in] T: typename
	 */
	template<typename T>
	class cuSEGMENT {
	public:
		// Constructor
		__host__ __device__ cuSEGMENT(){};
		__host__ __device__ cuSEGMENT(const cuVEC<T>& Start, const cuVEC<T>& End){ _S=Start; _E=End; }
		template<typename T2>
		__host__ __device__ cuSEGMENT(const cuVEC<T2>& Start, const cuVEC<T2>& End){
			_S=cuVEC<T>(T(Start.x), T(Start.y), T(Start.z));
			_E=cuVEC<T>(T(End.x),   T(End.y),   T(End.z));
		}
		// IO
		__host__ __device__ const cuVEC<T>& S() const{return _S;};
		__host__ __device__ const cuVEC<T>& E() const{return _E;};
		__host__ __device__ cuVEC<T>& S(){return _S;};
		__host__ __device__ cuVEC<T>& E(){return _E;};
		// Misc.
		__host__ __device__ T Length(){ return (_E-_S).abs(); }
		__host__ void Print(){
			cout<<"+-----------------+"<<endl;
			cout<<"|   Segment (3D)  |"<<endl;
			cout<<"+-----------------+"<<endl;
			cout<<"Start : "; _S.Print();
			cout<<"End   : "; _E.Print();
		}
	private:
		cuVEC<T> _S;	// Start point
		cuVEC<T> _E;	// End point
	};

	/**
	 * 3 Edges set for triangle
	 * @tparam[in] T: typename
	 */
	template<typename T>
	class cuEdgeList {
	public:
		// Constructor
		__host__ __device__ cuEdgeList(){};
		__host__ __device__ cuEdgeList(const cuSEGMENT<T>& edge0, const cuSEGMENT<T>& edge1, const cuSEGMENT<T>& edge2){
			_edge[0] = edge0;
			_edge[1] = edge1;
			_edge[2] = edge2;
		}
		__host__ __device__ cuEdgeList(const cuTRI<T>& tri){
			_edge[0] = cuSEGMENT<T>(tri.V0, tri.V1);
			_edge[1] = cuSEGMENT<T>(tri.V1, tri.V2);
			_edge[2] = cuSEGMENT<T>(tri.V2, tri.V0);
		}
		template<typename T2>
		__host__ __device__ cuEdgeList(const cuTRI<T2>& tri){
			_edge[0] = cuSEGMENT<T>(tri.V0, tri.V1);
			_edge[1] = cuSEGMENT<T>(tri.V1, tri.V2);
			_edge[2] = cuSEGMENT<T>(tri.V2, tri.V0);
		}
		// IO
		__host__ __device__ const cuSEGMENT<T>& Edge(const size_t i) const{ return _edge[i]; };
		__host__ __device__ const cuSEGMENT<T>& Edge0() const{ return _edge[0]; };
		__host__ __device__ const cuSEGMENT<T>& Edge1() const{ return _edge[1]; };
		__host__ __device__ const cuSEGMENT<T>& Edge2() const{ return _edge[2]; };
		__host__ __device__ cuSEGMENT<T>& Edge(const size_t i){ return _edge[i]; };
		__host__ __device__ cuSEGMENT<T>& Edge0(){ return _edge[0]; };
		__host__ __device__ cuSEGMENT<T>& Edge1(){ return _edge[1]; };
		__host__ __device__ cuSEGMENT<T>& Edge2(){ return _edge[2]; };
		// Misc.
		__host__ __device__ T Length(const size_t i){ return _edge[i].Length(); }
		__host__ void Print(){
			cout<<"+--------------------+"<<endl;
			cout<<"|    EdgeList (3D)   |"<<endl;
			cout<<"+--------------------+"<<endl;
			cout<<"Edge 0 : "<<endl; _edge[0].Print();
			cout<<"Edge 1 : "<<endl; _edge[1].Print();
			cout<<"Edge 2 : "<<endl; _edge[2].Print();
		}
	private:
		cuSEGMENT<T> _edge[3]; // Edge set (ONLY 3 for triangle polygon)
	};

	/**
	 * Coordinate element for storing perpendicular and parallel unit vector
	 * @tparam[in] T: typename
	 */
	template<typename T>
	class cuLocalCorrdinateElemt {
	public:
		/**
		 * @brief      Default constructor
		 */
		__host__ __device__ cuLocalCorrdinateElemt(){};
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  perp  Perpendicular vector w.r.t. the incident plane
		 * @param[in]  para  Parallel vector w.r.t. the incident plane
		 */
		__host__ __device__ cuLocalCorrdinateElemt(const cuVEC<T>& perp, const cuVEC<T>& para){
			_perp = perp;
			_para = para;
		}
		/**
		 * @brief      Get    perp
		 */
		__host__ __device__ const cuVEC<T>& perp()const{ return _perp; }
		/**
		 * @brief      Get    para
		 */
		__host__ __device__ const cuVEC<T>& para()const{ return _para; }
		/**
		 * @brief      Get    perp (editable)
		 */
		__host__ __device__ cuVEC<T>& perp(){ return _perp; }
		/**
		 * @brief      Get    para (editable)
		 */
		__host__ __device__ cuVEC<T>& para(){ return _para; }
		/**
		 * @brief      Display the class on the console
		 */
		__host__ __device__ void Print()const{
			cout<<"perp = "; _perp.Print();
			cout<<"para = "; _para.Print();
		}
	private:
		cuVEC<T> _perp;	// A vector that is perpendicular to the incident plane
		cuVEC<T> _para;	// A vector that is parallel to the incident plane
	};

	/**
	 * Local coordinate for storing incident (ei) and scattering (es) unit vector
	 * @tparam[in] T: typename
	 */
	template<typename T>
	class cuLocalCorrdinate {
	public:
		/**
		 * @brief      Default constructor
		 */
		__host__ __device__ cuLocalCorrdinate(){};
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  ei    Incident local cooridinate element class
		 * @param[in]  es    Scatter(observe) local cooridinate element class
		 */
		__host__ __device__ cuLocalCorrdinate(const cuLocalCorrdinate<T>& ei, const cuLocalCorrdinate<T>& es){
			_ei = ei;
			_es = es;
		}
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  uv_t  Edge unit vector
		 * @param[in]  uv_sp Incidnet unit vector
		 * @param[in]  uv_s  Scatter(Observe) unit vector
		 *
		 * @return     Incident plane coordinate (ei) and observation plane coordinate (es)
		 *
		 * @ref        E. Knott, “The relationship between Mitzner‘s ILDC and Michaeli’s equivalent currents,
		 *             ” IEEE Trans. Antennas Propagat., vol. 33, no. 1, pp. 112–114, 1985.
		 */
		__host__ __device__ cuLocalCorrdinate(const cuVEC<T>& uv_t, const cuVEC<T>& uv_sp, const cuVEC<T>& uv_s){
			// Incident plane (Eq. 12)
			_ei.perp() = Unit(cross(uv_t, uv_sp));
			_ei.para() = cross(uv_sp, _ei.perp());

			// Scatter(Observation) plane (Eq. 13)
			_es.perp() = Unit(cross(uv_t, uv_s));
			_es.para() = cross(uv_s, _es.perp());	// [FIX] Why minus symbol?
		}
		/**
		 * Inport from LocalCorrdinate<T>
		 * @param[in] in: Input source LocalCorrdinate<T> class
		 */
		__host__ void fromLocalCoodinate(const ptd::LocalCorrdinate<T>& in){
			cuVEC<double> es_perp(in.es().perp().x(), in.es().perp().y(), in.es().perp().z());
			cuVEC<double> es_para(in.es().para().x(), in.es().para().y(), in.es().para().z());
			cuVEC<double> ei_perp(in.ei().perp().x(), in.ei().perp().y(), in.ei().perp().z());
			cuVEC<double> ei_para(in.ei().para().x(), in.ei().para().y(), in.ei().para().z());
			_es = cuLocalCorrdinateElemt<double>(es_perp, es_para);
			_ei = cuLocalCorrdinateElemt<double>(ei_perp, ei_para);
		}
		/**
		 * Creation
		 * @param[in] h_elg: Source cuLocalCorrdinate<T> class
		 * @param[out] d_elg: Destination cuLocalCorrdinate<T> class
		 */
		__host__ void Create(const cuLocalCorrdinate<T>& h_elg, cuLocalCorrdinate<T>*& d_elg){
			cuLocalCorrdinateElemt<T> tmp_ei( h_elg._ei );
			cuLocalCorrdinateElemt<T> tmp_es( h_elg._es );
			//
			// memory allocation
			//
			cudaMalloc(&d_elg, sizeof(cuLocalCorrdinate<T>));
			//
			// copy
			//
			cudaMemcpy(&(d_elg->_ei),      	&tmp_ei, 	  sizeof(cuLocalCorrdinateElemt<T>),  cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_elg->_es),      	&tmp_es, 	  sizeof(cuLocalCorrdinateElemt<T>),  cudaMemcpyHostToDevice);
			//
			// Check Error
			//
			ChkErr("cu::cuLocalCorrdinate<T>::Create");
		}
		__host__ void Free(cuLocalCorrdinate<T>* d_elg){
			cudaFree(d_elg);
			//
			// Check Error
			//
			ChkErr("cu::cuLocalCorrdinate<T>::Free");
		}
		/**
		 * @brief      Get    ei
		 */
		__host__ __device__ const cuLocalCorrdinateElemt<T>& ei()const{ return _ei; }
		/**
		 * @brief      Get    es
		 */
		__host__ __device__ const cuLocalCorrdinateElemt<T>& es()const{ return _es; }
		/**
		 * @brief      Get    ei (editable)
		 */
		__host__ __device__ cuLocalCorrdinateElemt<T>& ei(){ return _ei; }
		/**
		 * @brief      Get    es (editable)
		 */
		__host__ __device__ cuLocalCorrdinateElemt<T>& es(){ return _es; }
		/**
		 * @brief      Display the class on the console
		 */
		__host__ __device__ void Print()const{
			cout<<"+-----------------+"<<endl;
			cout<<"|     Summary     |"<<endl;
			cout<<"+-----------------+"<<endl;
			cout<<"ei : "<<endl; _ei.Print();
			cout<<"es : "<<endl; _es.Print();
		}
	private:
		cuLocalCorrdinateElemt<T> _ei;	// Incident local coordinate element class
		cuLocalCorrdinateElemt<T> _es;	// Scatter(observe) local coordinate element class
	};

	/**
	 * 3 Edge unit vector for tirangle
	 * @tparam[in] T: typename
	 */
	template<typename T>
	class cuEdgeCoordinate {
	public:
		/**
		 * @brief      Default constructor
		 */
		__host__ __device__ cuEdgeCoordinate(){};
		/**
		 * @brief      Constructor with input arguments
		 *
		 * @param[in]  x     X direction vector in Edge local coordinate
		 * @param[in]  y     Y direction vector in Edge local coordinate
		 * @param[in]  z     Z direction vector in Edge local coordinate
		 */
		__host__ __device__ cuEdgeCoordinate(const cuVEC<T>& x, const cuVEC<T>& y, const cuVEC<T>& z){
			_x = x;
			_y = y;
			_z = z;
		}
		/**
		 * @brief      Get    x
		 */
		__host__ __device__ const cuVEC<T>& x()const{ return _x; }
		/**
		 * @brief      Get    y
		 */
		__host__ __device__ const cuVEC<T>& y()const{ return _y; }
		/**
		 * @brief      Get    z
		 */
		__host__ __device__ const cuVEC<T>& z()const{ return _z; }
		/**
		 * @brief      Get    x (editable)
		 */
		__host__ __device__ cuVEC<T>& x(){ return _x; }
		/**
		 * @brief      Get    y (editable)
		 */
		__host__ __device__ cuVEC<T>& y(){ return _y; }
		/**
		 * @brief      Get    z (editable)
		 */
		__host__ __device__ cuVEC<T>& z(){ return _z; }
		/**
		 * @brief      Display the class on the console
		 */
		__host__ void Print(){
			cout<<"+----------------+"<<endl;
			cout<<"|     Summary    |"<<endl;
			cout<<"+----------------+"<<endl;
			cout<<"x : "; _x.Print();
			cout<<"y : "; _y.Print();
			cout<<"z : "; _z.Print();
		}
	private:
		cuVEC<T> _x, _y, _z;	// X,Y,Z direction vector in Edge local coordinate
	};

	//+=======================================================+
	//|            device function (__device__)               |
	//+=======================================================+
	/**
	 * Wedge factor (2-angle)/pi
	 * @param [in] angle_rad [rad] Wedge internal angle in rad
	 * @return Return the wedge factor, n
	 * @Example n = 1 (for pi angle)
	 */
	 __device__
	double WedgeFactor(const double angle_rad){
		return 2 - angle_rad / cu::PId;
	}

	/**
	 * Calculate Sine value of complex input
	 * @tparam [in] T: typename
	 * @param [in] a: Input complex value
	 * @return Return a Sine of complex value
	 */
	template<typename T>
	__device__
	cuCPLX<T> Sin(const cuCPLX<T>& a){
		cuCPLX<T> jj(0,1);
		cuCPLX<T> tmp1 =  jj*a;
		cuCPLX<T> tmp2 = -jj*a;
		return (Exp(tmp1) - Exp(tmp2)) / cuCPLX<T>(0.,2.);
//		complex<T> tp = std::sin( complex<T>(a.r(), a.i()) );
//		return CPLX<T>(tp.real(), tp.imag());
	}

	/**
	 * Calculate Cosine value of complex input
	 * @tparam [in] T: typename
	 * @param [in] a: Input complex value
	 * @return Return a Cosine of complex value
	 */
	template<typename T>
	__device__
	cuCPLX<T> Cos(const cuCPLX<T>& a){
		cuCPLX<T> jj(0,1);
		cuCPLX<T> tmp1 =  jj*a;
		cuCPLX<T> tmp2 = -jj*a;
		return (Exp(tmp1) + Exp(tmp2)) / cuCPLX<T>(2.,0.);
//		complex<T> tp = std::cos( complex<T>(a.r(), a.i()) );
//		return CPLX<T>(tp.real(), tp.imag());
	}

	/**
	 * Calculate Cosecant value of complex input
	 * @tparam [in] T: typename
	 * @param [in] a: Input complex value
	 * @return Return a Cosecant of complex value
	 */
	template<typename T>
	__device__
	cuCPLX<T> Csc(const cuCPLX<T>& a){
		return 1.0/Sin(a);
	}

	/**
	 * Calculate Cotangent value of complex input
	 * @tparam [in] T: typename
	 * @param [in] a: Input complex value
	 * @return Return a Cotangent of complex value
	 */
	template<typename T>
	__device__
	T Cot(const T rad){
		return 1.0/tan(rad);
	}

	/**
	 * Calculate the square root of input complex value
	 * @tparam [in] T: typename
	 * @param [in] in: input complex value
	 * @return return a square root of complex value
	 */
	template<typename T>
	__device__
	cuCPLX<T> Sqrt(const cuCPLX<T>& in){
		return in.sqrt();
	}

	/**
	 * Calculate the log of input complex value
	 * @tparam [in] T: typename
	 * @param [in] in: input complex value
	 * @return return a log of complex value
	 */
	template<typename T>
	__device__
	cuCPLX<T> Log(const cuCPLX<T>& in){
		return log(in.abs()) + cuCPLX<T>(0,1) * atan2(in.i, in.r);
	}

	/**
	 * Calculate dot product of a vector and complex vector
	 * @tparam [in] T: typename
	 * @param [in] a: input cuVEC<T>
	 * @param [in] b: input cuVEC<cuCPLX<T> >
	 * @return return a cuCPLX<T> after dot product
	 */
	template<typename T>
	__device__
	cuCPLX<T> dot(const cuVEC<T>& a,const cuVEC<cuCPLX<T> >& b){
		return (a.x*b.x + a.y*b.y + a.z*b.z);
	}

	/**
	 * Check the incidnet is can be scattered or not for diffraction
	 * @param[in] uv_sp		 Incident unit vector
	 * @param[in] tri1		 Facet-1 triangle object
	 * @param[in] j_shared1	 Shared edge index of Facet-1
	 * @param[in] tri2		 Facet-2 triangle object
	 * @return Return the boolean result
	 */
	 __device__
	bool CheckDiffractionIncidentBoundary(const cuBVH& bvh, const cuVEC<double>& uv_sp, const cuTRI<double>& tri1, const long j_shared1, cuTRI<double>& tri2, cuEdgeCoordinate<double>& e2, const size_t iFrq, const size_t kk){
		// uv_sp: Un-projected incident unit vector
		// tri1:  Facet-1 triangle object
		// j_shared1: Shared edge index of Facet-1

		// Edge (Facet-1)
		cuEdgeList<double> EL1(tri1);

		cuEdgeCoordinate<double> e1;
		e1.z() = Unit( EL1.Edge(j_shared1).E() - EL1.Edge(j_shared1).S() );	// Along edge
		e1.y() = tri1.getNormalDouble();									// Normal edge
		e1.x() = cross(e1.y(), e1.z());										// On the plate

		cuVEC<double> uv_sp_proj = ProjectOnPlane(uv_sp, e1.z());
		// Nearest polygon
		// Get triangle
		cuTRI<float> tmp = (bvh.build_primes)[ bvh.idx_poly[tri1.IDX_Near(j_shared1)] ];
		tri2 = cuTRI<double>(cuVEC<double>(tmp.V0.x, tmp.V0.y, tmp.V0.z), cuVEC<double>(tmp.V1.x, tmp.V1.y, tmp.V1.z), cuVEC<double>(tmp.V2.x, tmp.V2.y, tmp.V2.z), tmp.IDX(), tmp.ea, tmp.IDX(), tmp.idx_near);


		// Shared edge index for Facet-2
		long j_shared2 = 0;
		for(long k=0;k<3;++k){
//			cout<<"tri2.IDX_Near("<<k<<") = "<<tri2.IDX_Near(k)<<", tri1.IDX() = "<<tri1.IDX()<<endl;
			if(tri2.IDX_Near(k) == tri1.IDX()){
				j_shared2 = k;
			}
		}

		// Edge (Facet-2)
		cuEdgeList<double> EL2(tri2);

//		EdgeCoordinate<double> e2;
		e2.z() = Unit( EL2.Edge(j_shared2).E() - EL2.Edge(j_shared2).S() );	// Along edge
		e2.y() = tri2.getNormalDouble();									// Normal edge
		e2.x() = cross(e2.y(), e2.z());										// On the plate


		bool isOK = CheckEffectiveDiffractionIncident(e1.x(), e1.z(), e2.x(), uv_sp_proj, kk);


//		// DEBUG (START) ========================================================================================================================================================================
//		if(iFrq==0 && kk==60602) {
//			tri2.Print();
////			printf("\n\n\n\n>>>> GPU >>>>\ntri1.IDX=%ld, tri2.idx=%ld, e2.x()=(%f,%f,%f), e2.y()=(%f,%f,%f), e2.z()=(%f,%f,%f), N2=(%f,%f,%f), vv1_vv0=(%f,%f,%f), vv2_vv0=(%f,%f,%f)\n>>>>>>>>>>>>>\n\n",
////				   tri1.IDX(), tri2.IDX(), e2.x().x, e2.x().y, e2.x().z, e2.y().x, e2.y().y, e2.y().z, e2.z().x, e2.z().y, e2.z().z,
////				   N2.x, N2.y, N2.z, vv1_vv0.x, vv1_vv0.y, vv1_vv0.z, vv2_vv0.x, vv2_vv0.y, vv2_vv0.z);
////			printf("\n\n\n\n>>>> GPU >>>>\ntri1.IDX=%ld, tri1.IDX_Near=[%ld,%ld,%ld], tri2.IDX=%ld, e1.x=(%f,%f,%f), e1.z=(%f,%f,%f), e2.x=(%f,%f,%f), uv_sp_proj=(%f,%f,%f), isOK=%d\n>>>>>>>>>>>>>\n\n",
////				   tri1.IDX(), tri1.IDX_Near(0), tri1.IDX_Near(1), tri1.IDX_Near(2), tri2.IDX(),
////				   e1.x().x, e1.x().y, e1.x().z, e1.z().x, e1.z().y, e1.z().z, e2.x().x, e2.x().y, e2.x().z,
////				   uv_sp_proj.x, uv_sp_proj.y, uv_sp_proj.z, isOK);
////			printf("\n\n\n\n>>>> GPU >>>>\ntri1.IDX=%ld, tri2.IDX=%ld, p=%ld, j_shared2=%ld, uv_sp=(%f,%f,%f), e1.z=(%f,%f,%f), e2.x=(%f,%f,%f)\n>>>>>>>>>>>>>\n\n", tri1.IDX(), tri2.IDX(), j_shared1, j_shared2, uv_sp.x, uv_sp.y, uv_sp.z, e1.z().x, e1.z().y, e1.z().z, e2.x().x, e2.x().y, e2.x().z);
//		}
//		// DEBUG (END) ==========================================================================================================================================================================


		return isOK;
	}

	/**
	 * Reduction kernel function for single call by reduce3. The results is partial reduction. For all reduction, it need to be call multiple times.
	 * Ref: HARRIS, Mark, et al. Optimizing parallel reduction in CUDA. Nvidia developer technology, 2007, 2.4: 70.
	 * @param[in] g_idata: Input 1D array in global memory
	 * @param[out] g_odata: Output 1D array in global memory (for partial summation)
	 * @param[in] N: Size of input array (g_idata)
	 */
	__global__
	void reduce3(cuCPLX<double>* g_idata, cuCPLX<double>* g_odata, const size_t nTriangle) {
		//
		// g_idata (Ed_triangle_freq):
		//
		//   (in 1D)
		//   111111..........1 2........2 ....... o.............o
		//   <-- nTriangle -->
		//   <---------------- nFreq sets ---------------------->
		//
		//   (in 2D)
		//   <------ nTriangle ------>
		//   1111111111111111111111111  ^
		//   2         ....          2  |
		//   :         ....          :  nFreq
		//   :         ....          :  |
		//   ooooooooooooooooooooooooo  V
		//
		// g_odata (Ed_freq):
		//   (in 1D)
		//   123...............o
		//   <----- nFreq ----->
		//
		extern __shared__ cuCPLX<double> s_data[];
		// each thread loads one element from global to shared mem
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;


		s_data[tid] = g_idata[i] * (i < nTriangle);

		__syncthreads();
		// do reduction in shared mem
		for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
			if (tid < s) {
				s_data[tid] += s_data[tid + s];
			}
			__syncthreads();
		}
		// write result for this block to global mem
		if (tid == 0){
			g_odata[blockIdx.x] = s_data[0];
		}
	}


	/**
	 * Unit step function
	 * @param[in] in Input any real value
	 * @return Return 0 if input smaller then 0, otherwise return 1.
	 */
	__device__
	double UnitStep(const double in){
		if(in < 0){
			return 0;
		}else{
			return 1;
		}
	}

	/**
	 * Calculate the Fringe current components by Michaeli's method.
	 * Ref: A. Michaeli, “Elimination of infinities in equivalent edge currents, Part I: Fringe current components,” IEEE Trans. Antennas Propagat., vol. 34, no. 7, pp. 912–918, Jul. 1986, doi: 10.1109/TAP.1986.1143913.
	 * @param[in] Ei_cplx [x,x,x] Complex vector of incident electric field
	 * @param[in] z       [m,m,m] Unit vector along edge
	 * @param[in] n       [x] Wedge n factor
	 * @param[in] k       [1/m] Wavenumber
	 * @param[in] sp      [m,m,m] Incident unit vector
	 * @param[in] beta    [rad] Angle between observation and edge (e.z) vector
	 * @param[in] betap   [rad] Angle between incident and edge (e.z) vector
	 * @param[in] phi     [rad] Angle between observation of projected on the (e.x & e.y) plane  and edge on the plane (e.x) vector
	 * @param[in] phip    [rad] Angle between incident of projected on the (e.x & e.y) plane and edge on the plane (e.x) vector
	 * @param[out] I      [?] Electric edge current
	 * @param[out] M      [?] Magnetic edge current
	 */
	__device__
	void FringeIM(const cuVEC<cuCPLX<double> >& Ei_cplx, const cuVEC<double>& z, const double n, const double k,
				  const cuVEC<double>& sp, const double beta, const double betap, const double phi, const double phip,
				  cuCPLX<double>& I, cuCPLX<double>& M) {

//		const double PId=3.141592653589793;		// PI

		double sb = sin(beta);
//		double sbp = sin(betap);
		double cotb = Cot(beta);
		double cotbp = Cot(betap);
		double cp = cos(phi);
//		double cpp = cos(phip);

//		double u = sb*cp/sbp;
		double u = cp - 2 * cotb*cotb;


		cuCPLX<double> a = cuCPLX<double>(0,-1) * Log( u + cuCPLX<double>(0,1)*Sqrt(cuCPLX<double>(1 - u*u, 0)) );
		cuCPLX<double> sa = Sin(a);


		cuCPLX<double> jj(0, -1);

		cuCPLX<double> Eiz = dot(z, Ei_cplx);
		cuCPLX<double> Hiz = dot(z, cross(sp, Ei_cplx));

		I = -2.0 * jj / (k * sb*sb) *
			(sin(phi) * UnitStep(cu::PId - phi) / (cp + u) + (1.0 / n) * sin(phi / n) * (1.0/(Cos((cu::PId - a) / n) - cos(phi / n)))) *
			Eiz +
			2.0 * jj * Sin((cu::PId - a) / n) / (n * k * sb * sa) *
			(u * cotbp - cotb * cp) / (cos(phi / n) - Cos((cu::PId - a) / n)) * Hiz;

		M = 2.0 * jj * sin(phi) / (k * sb*sb) *
			(UnitStep(cu::PId - phi) / (cp + u) -
			 (1.0 / n) * Sin((cu::PId - a) / n) * Csc(a) * (1.0/(cos(phi / n) - Cos((cu::PId - a) / n)))) * Hiz;
	}


	//+=======================================================+
	//|                CPU program (__host__)                 |
	//+=======================================================+
	/**
	 * Reduction for any size of input
	 * @param[in] in: input 1D array
	 * @param[in] N: Size of 1D array
	 * @param[in] sum: Results single value
	 * @param[in] Nthread: Number of thread need to be used. (Default = 512)
	 */
	/**
	 * Summation all triangle into only one value for each frequency.
	 * @param[in] Ed_triangle_freq (cuCPLX<double>*) The inout 1D array. The pointer was offset by j*nTriangle
	 * @param[in] nTriangle 	   [sample] Number of triangle in the CAD.
	 * @param[out] Ed_freq 		   (cuCPLX<double>*) The destination for each frequency.
	 * @param[in] idx_freq         [x] Index of output frequency for Ed_freq. e.g. Ed_freq[idx_freq].
	 * @param[in] Nthread
	 */
	__host__
	void cuReductionPTD(const cuCPLX<double>* Ed_triangle_freq, const size_t nTriangle,
						cuCPLX<double>* Ed_freq, const size_t idx_freq, const size_t Nthread = 512){

		// Ed_triangle_freq: The pointer was offset.
		// nTriangle
		// idx_freq: the start index of Ed_freq need to be written

		// Duplicate the Ed_triangle_freq + (offset idx)
		// This memory will be modified, the final results after reduction was stored in the 1st element of cuin.
		cuCPLX<double>* cuin;

		size_t sz = nTriangle;


		// Execute multi times of device function caller
		SafeMallocCheck(cudaMalloc(&cuin, nTriangle * sizeof(cuCPLX<double>)));
		SafeMallocCheck(cudaMemcpy(cuin, Ed_triangle_freq, nTriangle * sizeof(cuCPLX < double >), cudaMemcpyHostToDevice));


		size_t smem = Nthread * sizeof(cuCPLX<double>);

		size_t Nblock;

		// Detect number of __global__ call
		size_t NCall = 1;
		size_t rem = nTriangle;
		do {
			rem = ceil(double(rem) / double(Nthread));
			if(rem > 1){ NCall++; }
		}while(rem > 1);

//		cout<<"NCall = "<<NCall<<endl;
//		cout<<"rem   = "<<rem<<endl;

		for(size_t i=0;i<NCall;++i){
			Nblock = ceil(double(sz)/double(Nthread));
//			printf("%3ld, sz = %3ld, Nblock = %3ld\n", i, sz, Nblock);
			//
			// Execute (CPU time : 0.60 [sec])
			//
//			reduce0<<< Nblock, Nthread, smem >>>(cuin, cuin, sz);	// 0.70 [sec]
//			reduce1<<< Nblock, Nthread, smem >>>(cuin, cuin, sz);	// 0.69 [sec]
//			reduce2<<< Nblock, Nthread, smem >>>(cuin, cuin, sz);	// 0.67 [sec]
			reduce3<<< Nblock, Nthread, smem >>>(cuin, cuin, sz);	// 0.66 [sec]
//			reduce4<<< Nblock, Nthread, smem >>>(cuin, cuin, sz);	// ? [sec] (Error)
//			reduce5<<< Nblock, Nthread, smem >>>(cuin, cuin, sz);	// ? [sec] (Error)
			sz = Nblock;// + itr;
		}

		// Copy from cuin[0] into Ed_freq[idx_freq]
		cudaMemcpy(Ed_freq + idx_freq, cuin, 1 * sizeof(cuCPLX < double >), cudaMemcpyDeviceToDevice);
//		cudaMemcpy(&sum, cuin, 1*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(cuin);
	}


	//+=======================================================+
	//|              CUDA Kernel (__global__)                 |
	//+=======================================================+
	/**
	 * Calculate the diffraction scattering electric field for each triangle (sum 3 edges) and each frequency
	 * @param[in] inc_mesh     [x] (cuMeshInc) Incident mesh
	 * @param[in] Ei_o         [m,m,m] Origin of incident electric field
	 * @param[in] Ei_k         [m,m,m] Origin of incident electric field
	 * @param[in] Ei_cplx      [x,x,x] Complex vector of incident electric field
	 * @param[in] k0           [1/m] Wavenumber
	 * @param[in] rayInc       [x] (cuRay) Incident ray
	 * @param[in] bvh          [x] (cuBVH)
	 * @param[in] nTriangle    [x] Number of triangle
	 * @param[in] nFreq		   [x] Number of frequency
	 * @param[out] Ed_triangle [x] Scattering electric field after summation 3 edges for each triangle and for each frequency
	 * @param[out] Ed          [x] Scattering electric field after summation 3 edges for each frequency
	 */
	__global__
	void cuPTD(const cuMeshInc& inc_mesh, const cuVEC<double>& Ei_o, const cuVEC<double>& Ei_k, const cuVEC<cuCPLX<double> >& Ei_cplx,
			   const double* k0, const cuRay& rayInc, const cuBVH& bvh, const size_t nTriangle, const size_t nFreq, cuLocalCorrdinate<double>& elg, const double dRad,
			   cuCPLX<double>* Echo_triangle_freq_H, cuCPLX<double>* Echo_triangle_freq_V){
		//
		// Echo_triangle_freq_H:
		//
		//   (in 1D)
		//   111111..........1 2........2 ....... o.............o
		//   <-- nTriangle -->
		//   <---------------- nFreq sets ---------------------->
		//
		//   (in 2D)
		//   <------ nTriangle ------>
		//   1111111111111111111111111  ^
		//   2         ....          2  |
		//   :         ....          :  nFreq
		//   :         ....          :  |
		//   ooooooooooooooooooooooooo  V
		//
		// Echo_freq_H:
		//   (in 1D)
		//   123...............o
		//   <----- nFreq ----->
		//
		volatile unsigned int k = threadIdx.x + blockDim.x*blockIdx.x;
		volatile unsigned int iFrq = k / nTriangle;
		volatile unsigned int iTri = k % nTriangle;

		if(k >= nTriangle*nFreq ){ return; }

		// Get triangle
		cuTRI<float>  tmp = bvh.build_primes[iTri];
		// Force convert to double
		cuTRI<double> tri;
		tri.fromTRI(tmp);
//		cuTRI<double> tri(cuVEC<double>(double(tmp.V0.x), double(tmp.V0.y), double(tmp.V0.z)),
//						  cuVEC<double>(double(tmp.V1.x), double(tmp.V1.y), double(tmp.V1.z)),
//						  cuVEC<double>(double(tmp.V2.x), double(tmp.V2.y), double(tmp.V2.z)),
//						  tmp.MatIdx, tmp.ea, tmp.idx, tmp.idx_near);

		// Get Normal vector
//		cuVEC<double> N0 = tri.getNormal();

		// Intersection object for shadow detection
		cuIntersectionInfo I_Shadow;

		// Edge
		cuEdgeList<double> EL(tri);

		// Near edge
//		long idx_poly_near[3];
//		idx_poly_near[0] = tri.IDX_Near(0);
//		idx_poly_near[1] = tri.IDX_Near(1);
//		idx_poly_near[2] = tri.IDX_Near(2);

//		if(iFrq==23 && iTri==117641) {
//			printf("iFrq=%d, iTri=%d, idx_poly_near=[%ld,%ld,%ld]\n", iFrq, iTri, idx_poly_near[0], idx_poly_near[1], idx_poly_near[2]);
//			bvh.build_primes[iTri].Print();
//			tri.Print();
//		}

		// declare
		cuCPLX<double> I1, M1, I2, M2, I, M, factor;
		cuVEC<double> I_vec, M_vec;
		cuVEC<cuCPLX<double> > I_comp, M_comp, Ed_local;


		// TODO: 測試使用 azimuth line -------------------- (START) -----------------------------
		cuVEC<double> Q(0.,0.,0.);
		double s  = Norm2(Ei_o - Q);
//		cuCPLX<double> tmp_exp0(0, -k0[iFrq] * (s) * 2.0);
//		factor = -0.01 * Exp(tmp_exp0) / PI2d;
//
//		Ed_local.x = factor;
//		Ed_local.y = factor;
//		Ed_local.z = factor;
//
//		double TotalDis = dis + (2* AddDis);
//		AddPhase(PoEs.cplx, k0*TotalDis);

		Ed_local.x = cuCPLX<double>(1.,0.);
		Ed_local.y = cuCPLX<double>(1.,0.);
		Ed_local.z = cuCPLX<double>(1.,0.);

		double TotalDis = (2* s);
		AddPhase(Ed_local, k0[iFrq]*TotalDis);

		Echo_triangle_freq_V[iFrq * nTriangle + iTri] += dot(Ed_local, elg.es().perp()); // V
		Echo_triangle_freq_H[iFrq * nTriangle + iTri] += dot(Ed_local, elg.es().para()); // H
		// TODO: 測試使用 azimuth line -------------------- (END) -------------------------------

/*
#pragma unroll
		// For loop ( FOR EACH EDGE of EACH TRIANGLE )
		for(long p=0; p<3; ++p) {
			// 2.0. Only shared edge need to be calculated
			//      tri.IDX_Near(p) = -1 means that there is no shared edges.
			if(tri.IDX_Near(p) < 0){ continue; }
			// 2.1. Find wedge angle
			volatile double WedgeAngle = tri.ea[p];
			volatile double n = WedgeFactor(WedgeAngle);
			// 2.2 Defined incident & observation unit vector
			// Incident
			cuVEC<double> uv_sp = rayInc.d;
			// 2.3 Find Edge vector
			//    右手定則下的 vertex，旋轉方向(大拇指指向)為 normal vector of polygon, 跟diffraction定義的edge方向
			//    由起始點(Start)到終點(End)
			cuEdgeCoordinate<double> e;
			e.z() = Unit(EL.Edge(p).E() - EL.Edge(p).S() );	// 因為cad reader是右手定則 (Start -> End)
			e.y() = tri.getNormal();
			e.x() = cross(e.y(), e.z());
			//+--------------------------------------------+
			//|       3. For each pieces of segment        |
			//+--------------------------------------------+
			// 3.1 Find segment two end points and centroid point location
			double dL = double(EL.Length(p));
			cuVEC<double> Q = 0.5 * (EL.Edge(p).S() + EL.Edge(p).E());
			double s  = Norm2(Ei_o - Q);
//			printf("s = %.10f\n", s);
			//+-----------------------------------------------------+
			//|   Check it is effective edge with some conditions   |
			//+-----------------------------------------------------+
			// 3.2 The N factor of wedge angle MUST smaller than 1
			if(n <= 1){ continue; }
			// 3.3 The incident angle is between within the two facets boundary
			cuTRI<double> tri2;
			cuEdgeCoordinate<double> e2;
			bool isOK = CheckDiffractionIncidentBoundary(bvh, uv_sp, tri, p, tri2, e2, iFrq, k);
			if(!isOK){ continue; }
			// 3.4 Check the return path is shadow or not?
			//     Check Shadow : If the back ray tracing is shadow, there is not PO result.
			// shadow ray
			cuVEC<double> uv_shadow = Unit(cuVEC<double>(Ei_o.x-Q.x, Ei_o.y-Q.y, Ei_o.z-Q.z));
			cuRay rayShadow(Q, uv_shadow);
			bool isShadow = bvh.getIntersection(rayShadow, &I_Shadow, false);
			cuTRI<float>* tmp_tri_ptr = (cuTRI<float>*)(I_Shadow.object);
			isShadow = ( isShadow && !(tmp_tri_ptr->Equal(tri)) );
			if(isShadow){ continue; }
			//+--------------------------------+
			//|    Diffraction calculation     |
			//+--------------------------------+
			// 3.5 Observation
			cuVEC<double> uv_s = cuVEC<double>(0,0,0) - uv_sp;
			// 3.6 Calculation all angles
			double beta  = acos(dot(uv_s, e.z()));	// to Observation (e.z & s are uv)
			double betap = acos(dot(uv_sp, e.z()));	// from Source (e.z & sp are uv)

			cuVEC<double> uv_s_proj  = Unit(ProjectOnPlane(uv_s,  e.z()));	// s  projected on local XY-plane
			cuVEC<double> uv_sp_proj = Unit(ProjectOnPlane(uv_sp, e.z()));	// sp projected on local XY-plane

			double phi  = SignedAngleTwo3DVectors(e.x(),   uv_s_proj, e.z());	// s_proj: observation
			double phip = SignedAngleTwo3DVectors(e.x(), -uv_sp_proj, e.z());	// sp_proj: incident
			if(phi  < 0) {  phi = (360*DTRd) + phi; }
			if(phip < 0) { phip = (360*DTRd) + phip; }

			// 3.7 Avoid Avoid 90 & 180 [deg]
			double eps = 1e-4;
			if(std::abs(beta - (90*DTRd)) < eps) {
				beta = beta - eps;
//				betap = (180*DTRd) - beta;    // Backscattering
			}
			if(std::abs(betap - (90*DTRd)) < eps) {
				betap = betap - eps;
//				beta = (180*DTRd) - betap;    // Backscattering
			}
			if(std::abs(phi - (90*DTRd)) < eps || std::abs(phi - (180*DTRd)) < eps) {
				phi = phi - eps;
			}
			if(std::abs(phip - (90*DTRd)) < eps || std::abs(phi - (180*DTRd)) < eps) {
				phip = phip - eps;
			}

			// 3.8 Diffraction fringe calculation: using DiffMichaeliEC2
			// For each frequency
//			for(long j=0; j < nFreq; ++j){
				// Face-1
				FringeIM(Ei_cplx,  e.z(), n, k0[iFrq], uv_sp, beta, betap, phi, phip, I1, M1);
				// Face-2
//				FringeIM(Ei_cplx, -e.z(), n, k0[iFrq], uv_sp, PId - beta, PId - betap, n*def::PI-phi, n*def::PI-phip, I2, M2);
				FringeIM(Ei_cplx, -e.z(), n, k0[iFrq], uv_sp, PId - beta, PId - betap, phi, phip, I2, M2);

				I = I1 - I2;
				M = M1 + M2;

				// TODO: 不知為何要加上20m？
//				cuCPLX<double> tmp_exp0(0, -k0[iFrq] * (s) * 2.0);
				cuCPLX<double> tmp_exp0(0, -k0[iFrq] * (s+54.0) * 2.0);		// 合適於 L-band
//				cuCPLX<double> tmp_exp0(0, -k0[iFrq] * (s+20.0) * 2.0);		// 合適於 X-band
//				cuCPLX<double> tmp_exp0(0, -k0[iFrq] * (s-dRad) * 2.0);
//				cuCPLX<double> tmp_exp0(0, +k0[iFrq] * (s+dRad) * 2.0);
//				cuCPLX<double> tmp_exp0(0, -k0[iFrq] * (s+2.0*dRad) );
				factor = -dL * Exp(tmp_exp0) / PI2d;

				I_vec = Unit(cross(uv_sp, cross(uv_sp, e.z())));
				M_vec = Unit(cross(uv_sp, e.z()));

//				I_comp = cuVEC<cuCPLX<double> >(factor * I * I_vec.x, factor * I * I_vec.y, factor * I * I_vec.z);
//				M_comp = cuVEC<cuCPLX<double> >(factor * M * M_vec.x, factor * M * M_vec.y, factor * M * M_vec.z);
				I_comp.x = factor * I * I_vec.x;
				I_comp.y = factor * I * I_vec.y;
				I_comp.z = factor * I * I_vec.z;
				M_comp.x = factor * M * M_vec.x;
				M_comp.y = factor * M * M_vec.y;
				M_comp.z = factor * M * M_vec.z;

				Ed_local = (I_comp + M_comp);

				// 3.9 Avoid Nan & Inf
				if(isnan(Ed_local.x.r) || isinf(Ed_local.x.r)){ Ed_local.x.r = 1e-16; }
				if(isnan(Ed_local.x.i) || isinf(Ed_local.x.i)){ Ed_local.x.i = 1e-16; }
				if(isnan(Ed_local.y.r) || isinf(Ed_local.y.r)){ Ed_local.y.r = 1e-16; }
				if(isnan(Ed_local.y.i) || isinf(Ed_local.y.i)){ Ed_local.y.i = 1e-16; }
				if(isnan(Ed_local.z.r) || isinf(Ed_local.z.r)){ Ed_local.z.r = 1e-16; }
				if(isnan(Ed_local.z.i) || isinf(Ed_local.z.i)){ Ed_local.z.i = 1e-16; }

				// Sum for each triangle(k) and each edge(p)
				Echo_triangle_freq_V[iFrq * nTriangle + iTri] += dot(Ed_local, elg.es().perp()); // V
				Echo_triangle_freq_H[iFrq * nTriangle + iTri] += dot(Ed_local, elg.es().para()); // H


//				if(iFrq==23 && iTri==117641) {
//					printf("iFrq=%d, iTri=%d, I=(%f,%f), M=(%f,%f)\n", iFrq, iTri, I.r, I.r, M.r, M.i);
//				}

//				if(iFrq==0 && k==4264) {
//					printf("\n\n\n\n>>>> GPU >>>>\nk=%ld, p=%ld, Ed_local=[(%.10f,%.10f),(%.10f,%.10f),(%.10f,%.10f)]\n>>>>>>>>>>>>>\n\n",
//							k, p,
//							Ed_local.x.r, Ed_local.x.i,
//						    Ed_local.y.r, Ed_local.y.i,
//						    Ed_local.z.r, Ed_local.z.i);
////					Ed_local.x.Print();
////					Ed_local.y.Print();
////					Ed_local.z.Print();
//////					printf("\n\n\n\n>>>> GPU >>>>\nk=%ld, p=%ld, I=(%.20f,%.20f), M=(%.20f,%.20f), Ei_cplx=[(%f,%f),(%f,%f),(%f,%f)], e.z=[%f,%f,%f], k0=%.10f, Ed_local=[(%f,%f),(%f,%f),(%f,%f)]\n>>>>>>>>>>>>>\n\n",
//////						   k, p, I.r, I.i, M.r, M.i,
//////						   Ei_cplx.x.r, Ei_cplx.x.i, Ei_cplx.y.r, Ei_cplx.y.i, Ei_cplx.z.r, Ei_cplx.z.i,
//////						   e.z().x, e.z().y, e.z().z, k0[iFrq],
//////						   Ed_local.x.r, Ed_local.x.i, Ed_local.y.r, Ed_local.y.i, Ed_local.z.r, Ed_local.z.i);
//				}



//			} // End for each frequency(j), j<freq.GetNum()
		} // End for each edge(p), p<3
*/
	}

	__global__
	void cuCheck(const cuBVH& bvh){
		bvh.build_primes[0].Print();
	}

}  // namespace cu

#endif // CUMAIN_PTD_CUH_

