/*
 * cusar.cuh
 *
 *  Created on: Sep 11, 2021
 *      Author: cychiang
 */

#ifndef CUSAR_CUH_
#define CUSAR_CUH_

#include <cmath>
#include <cuda/cuvec.cuh>
#include <cuda/cuclass.cuh>
#include <cuda/cuopt.cuh>
#include <sar/sar.h>



namespace cu {
	//+======================================================+
	//|                      __host__                        |
	//+======================================================+

	//+======================================================+
	//|                     __device__                       |
	//+======================================================+
	/**
	 * ECR[X,Y,Z] to Geodetic[Lon,Lat_gd,h_gd]
	 * @param [in] ECR : [X,Y,Z] ECR locations
	 * @param [in] Orb : (cuORB class)
	 * @return [rad,rad,m,m]->[lon,lat_gd,N_gd,h_gc] Geodetic Location
	 * @Ref "Theoretucal Basis of the SDP Tookit geolocation package for the ECS Project", technical paper (445-TP-002-002), May, 1995.pp.6-21
	 */
	__device__
	cuGEO ECR2Gd(const cuVEC<double>& ECR,const cuORB Orb){
		double r0,zn;//,alpha;
		double w=0,wst=1.,tmp;
		long niter=0;
		double zeta,NC;
		double o_lon,o_lat,o_h;

		// Normalized distance from Earth grographic north-south axis
		r0 = sqrt(ECR.x*ECR.x+ECR.y*ECR.y)/Orb.E_a;
		zn = ECR.z/Orb.E_a;

		o_lon=atan2(ECR.y,ECR.x);

		// Initial
		//alpha = r0*r0 + (zn*zn/(1-Orb.e2()));
		while( (fabs(wst-w) > 1e-20) && (niter < 20) ){
			wst = w;
			niter += 1;
			tmp = r0-wst;
			w = Orb.e2*tmp/sqrt( tmp*tmp + zn*zn*(1-Orb.e2) );
		}
//#ifdef DEBUG
//		cout<<"abs(wst-w)="<<abs(wst-w)<<endl;
//		cout<<"niter="<<niter<<endl;
//#endif

		o_lat = atan(zn/(r0-w));
		// Find height
		// radius of curvature in prime vertical(NC)
		zeta = 1. / ( 1. - Orb.e2 * sin(o_lat) * sin(o_lat) );
		NC = Orb.E_a *sqrt(zeta);
		// height above ellipsoid
		o_h = r0*Orb.E_a / cos(o_lat) - NC;	// height of geodetic

		return cuGEO(o_lon,o_lat,o_h);
	}

	/**
	 * Find the instantaneous azimuth antenna gain value
	 * @param [in] n : [m,m,m] plane normal vector(or Uni-vector)
	 * @param [in] A : [m,m,m] point is loacted on the plane
	 * @param [in] P : [m,m,m] point is *NOT* located on the plane
	 * @param [out] POINT : [m,m,m] Intersection point
	 * @return min_dis : [m] minimun distance between plane and P-point
	 */
	__device__
	double MinDistanceFromPointToPlane(const cuVEC<double>& n,const cuVEC<double>& A,const cuVEC<float>& P, cuVEC<double>& POINT){
		cuVEC<double> uv=Unit(n);	// convert plane vector to unit vector
		double min_dis=0;
		cuVEC<double> p(P.x, P.y, P.z);

		min_dis = (p.x-A.x)*uv.x + (p.y-A.y)*uv.y + (p.z-A.z)*uv.z;
		POINT = p - min_dis*n;

		return fabs(min_dis);
	}

	/**
	 * @param [in] theta_sq_sqc : [rad] Central squint angle
	 * @param [in] theta_az : [rad] Azimuth beamwidth
	 */
	__device__
	double Waz(const double& theta_sq_sqc,const double& theta_az){
        double val=sinc(0.886*theta_sq_sqc/theta_az);
        return (val*val);
    }

	/**
	 * Find the instantaneous azimuth antenna gain value
	 * @param [in] Sar : (cuSAR class)
	 * @param [in] NorSquintPlane : (cuVEC) Normal vector of instantaneous squint plane
	 * @param [in] PPs : (cuVEC) Sensor position
	 * @param [in] PPt : (cuVEC) Antenna main beam pointing position on the surface
	 * @return The instantaneous azimuth antenna gain value
	 */
	__device__
	double AzimuthAntennaGain(const double theta_az, const cuVEC<double>& MainBeamUV, const cuVEC<double>& NorSquintPlane,
							  const cuVEC<double>& PPs, const cuVEC<double>& PPt, const cuVEC<float>& hit){
		cuVEC<double> hitProj;
		MinDistanceFromPointToPlane(NorSquintPlane, PPs, hit, hitProj);
		// Find azimuth's antenna angle (always positive)
		double theta_sq_inst = angle(MainBeamUV, hitProj - PPs);
		double azGain = Waz(theta_sq_inst, theta_az);

//		if(isnan(azGain)){
////			printf("[GPU:AzimuthAntennaGain] azGain=%.4f, theta_sq_inst=%.4f, theta_az=%.4f\n",
////					azGain, theta_sq_inst, theta_az);
////			printf("MainBeamUV: "); MainBeamUV.Print();
////			printf("hitProj: "); hitProj.Print();
////			printf("hit: "); hit.Print();
////			printf("PPs: "); PPs.Print();
//
//			cuVEC<double> a = MainBeamUV;
//			cuVEC<double> b = hitProj - PPs;
//			double a_dot_b = dot(a,b);
//			double ab = a.abs()*b.abs();
//			double cos_val=a_dot_b/ab;
//			printf("[GPU:AzimuthAntennaGain] azGain=%.8f, a_dot_b=%.8f, ab=%.8f, cos_val=%.20f, fabs(cos_val-1)=%.20f\n", azGain, a_dot_b, ab, cos_val, fabs(cos_val-1));
//		}

		return azGain;
//		return theta_sq_inst;
	}

	//+======================================================+
	//|                     __global__                       |
	//+======================================================+

}  // namespace cu



#endif /* CUMISC_CUH_ */
