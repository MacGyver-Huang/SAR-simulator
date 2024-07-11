#ifndef ORB_H_INCLUDED
#define ORB_H_INCLUDED

#include <sar/def.h>
#include <basic/new_time.h>
#include <basic/def_func.h>
#include <basic/d1.h>
#include <basic/d2.h>
#include <basic/vec.h>
#include <coordinate/iers.h>
#include <basic/mat.h>

using namespace d1;
using namespace d2;
using namespace new_time;
using namespace mat;
using namespace def_func;

namespace orb{
	//using namespace geo;
	//using namespace vec;
	//using namespace def_func;
	//using namespace def;
	//using namespace def::ORB;
	// ========================================
	// declare functions
	// ========================================
	template<typename T> T RadiusCurvatureMeridian(const T lat_gd,const def::ORB Orb);
	template<typename T> T RadiusGeocentric(const T lat_gc, const def::ORB Orb);
	double Arcsec2Rad(const double asec);
	double MeanObliquity(const double Mjd_TT);
	void NutAngles(const double Mjd_TT, double& dpsi, double& deps);
	double GMST(const double Mjd_UT1);
	double EqnEquinox(const double Mjd_TT);
	double GAST(const double Mjd_UT1);
	D2<double> PrecMatrix(const double MJD_TT_1,const double MJD_TT_2);
	D2<double> NutMatrix(const double MJD_TT);
	D2<double> GHAMatrix(const double Mjd_UT1);
	D2<double> PoleMatrix(double Mjd_UTC,IERS IERS_table);
	D2<double> ECI2ECRMatrix(const TIME& UTC, const IERS& IERS_table);
	D2<double> ECI2ECRMatrix(const TIME& UTC, const IERS& IERS_table,D2<double>& dU);
	D2<double> ECR2ECIMatrix(const TIME& UTC, const IERS& IERS_table);
	D2<double> ECR2ECIMatrix(const TIME& UTC, const IERS& IERS_table,D2<double>& dU);
	void ECR2ECI(const VEC<double>& ECR_pos,const VEC<double>& ECR_vel,	//input
				 const TIME& UTC,const IERS& IERS_table,				//input
				 VEC<double>& ECI_pos,VEC<double>& ECI_vel);			//output
	void ECI2ECR(const VEC<double>& ECI_pos,const VEC<double>& ECI_vel,	//input
				 const TIME& UTC,const IERS& IERS_table,				//input
				 VEC<double>& ECR_pos,VEC<double>& ECR_vel);			//output
}


// Implement ********************************************************
template<typename T>
T orb::RadiusCurvatureMeridian(const T lat_gd,const def::ORB Orb){
/*
 Purpose:
	Find local radius form geodetic centre to surface of Earth
 Input:
	lat_gd:    [rad] geodetic latitude
 Return:
	[m] radius of curvature in meridian
*/
	//cout<<E_a<<" "<<E_b<<endl;
	double e = sqrt( 1.-(Orb.E_b()/Orb.E_a())*(Orb.E_b()/Orb.E_a()) );
	//cout<<"E_a/E_b="<<E_a/E_b<<endl;
	//cout<<"1.-(E_a/E_b)*(E_a/E_b)="<<1.-(E_a/E_b)*(E_a/E_b)<<endl;
	//cout<<"e="<<e<<endl;
	double e2 = e*e;
	double up = Orb.E_a()*(1-e2);
	double dn = (1-e2*sin(lat_gd)*sin(lat_gd));
	dn = sqrt(dn*dn);

	return T(up/dn);
}

template<typename T>
T orb::RadiusGeocentric(const T lat_gc, const def::ORB Orb){
/*
 Purpose:
	Find local radius form geocentric centre to surface of Earth
 Input:
	lat_gc:    [rad] geocentric latitude
 Return:
	[m] radius of curvature of geocentric coordinate
*/
	double lambda = lat_gc;
	double sinlam = sin( lambda );
	double R = Orb.E_a();
	double radius  = sqrt(( Square(R) )/( 1.0 + (1.0/Square( 1.0 - Orb.f() ) - 1.0) * Square(sinlam) ));

	return T(radius);
}



double orb::Arcsec2Rad(const double asec){
/*
 Purpose:
	Convert from arcsecond to radius
*/
	return asec / (3600.*180./def::PI);
}

double orb::MeanObliquity(const double Mjd_TT){
/*
 Purpose:
	Find the mean obliquity
*/
	double T = (Mjd_TT - new_time::MJD_J2000)/36525.;
//	double eps = (23.43929111*60.*60.) + (46.8150 * T) + (0.00059 * T*T) + (0.001813*T*T*T);	//["] arcsecond
//	eps = Arcsec2Rad(eps);	//[rad]
	double eps = 23.43929111 -(46.8150 +(0.00059 -0.001813*T)*T)*T/3600.;	// [deg]
	eps = def_func::deg2rad(eps);	// [rad]

	return eps;
}

void orb::NutAngles(const double Mjd_TT, double& dpsi, double& deps){

  // Constants

  const double T  = (Mjd_TT-new_time::MJD_J2000)/36525.0;
  const double T2 = T*T;
  const double T3 = T2*T;
  const double rev = 360.0*3600.0;  // arcsec/revolution

  const int  N_coeff = 106;
  const long C[N_coeff][9] =
  {
   //
   // l  l' F  D Om    dpsi    *T     deps     *T       #
   //
    {  0, 0, 0, 0, 1,-1719960,-1742,  920250,   89 },   //   1
    {  0, 0, 0, 0, 2,   20620,    2,   -8950,    5 },   //   2
    { -2, 0, 2, 0, 1,     460,    0,    -240,    0 },   //   3
    {  2, 0,-2, 0, 0,     110,    0,       0,    0 },   //   4
    { -2, 0, 2, 0, 2,     -30,    0,      10,    0 },   //   5
    {  1,-1, 0,-1, 0,     -30,    0,       0,    0 },   //   6
    {  0,-2, 2,-2, 1,     -20,    0,      10,    0 },   //   7
    {  2, 0,-2, 0, 1,      10,    0,       0,    0 },   //   8
    {  0, 0, 2,-2, 2, -131870,  -16,   57360,  -31 },   //   9
    {  0, 1, 0, 0, 0,   14260,  -34,     540,   -1 },   //  10
    {  0, 1, 2,-2, 2,   -5170,   12,    2240,   -6 },   //  11
    {  0,-1, 2,-2, 2,    2170,   -5,    -950,    3 },   //  12
    {  0, 0, 2,-2, 1,    1290,    1,    -700,    0 },   //  13
    {  2, 0, 0,-2, 0,     480,    0,      10,    0 },   //  14
    {  0, 0, 2,-2, 0,    -220,    0,       0,    0 },   //  15
    {  0, 2, 0, 0, 0,     170,   -1,       0,    0 },   //  16
    {  0, 1, 0, 0, 1,    -150,    0,      90,    0 },   //  17
    {  0, 2, 2,-2, 2,    -160,    1,      70,    0 },   //  18
    {  0,-1, 0, 0, 1,    -120,    0,      60,    0 },   //  19
    { -2, 0, 0, 2, 1,     -60,    0,      30,    0 },   //  20
    {  0,-1, 2,-2, 1,     -50,    0,      30,    0 },   //  21
    {  2, 0, 0,-2, 1,      40,    0,     -20,    0 },   //  22
    {  0, 1, 2,-2, 1,      40,    0,     -20,    0 },   //  23
    {  1, 0, 0,-1, 0,     -40,    0,       0,    0 },   //  24
    {  2, 1, 0,-2, 0,      10,    0,       0,    0 },   //  25
    {  0, 0,-2, 2, 1,      10,    0,       0,    0 },   //  26
    {  0, 1,-2, 2, 0,     -10,    0,       0,    0 },   //  27
    {  0, 1, 0, 0, 2,      10,    0,       0,    0 },   //  28
    { -1, 0, 0, 1, 1,      10,    0,       0,    0 },   //  29
    {  0, 1, 2,-2, 0,     -10,    0,       0,    0 },   //  30
    {  0, 0, 2, 0, 2,  -22740,   -2,    9770,   -5 },   //  31
    {  1, 0, 0, 0, 0,    7120,    1,     -70,    0 },   //  32
    {  0, 0, 2, 0, 1,   -3860,   -4,    2000,    0 },   //  33
    {  1, 0, 2, 0, 2,   -3010,    0,    1290,   -1 },   //  34
    {  1, 0, 0,-2, 0,   -1580,    0,     -10,    0 },   //  35
    { -1, 0, 2, 0, 2,    1230,    0,    -530,    0 },   //  36
    {  0, 0, 0, 2, 0,     630,    0,     -20,    0 },   //  37
    {  1, 0, 0, 0, 1,     630,    1,    -330,    0 },   //  38
    { -1, 0, 0, 0, 1,    -580,   -1,     320,    0 },   //  39
    { -1, 0, 2, 2, 2,    -590,    0,     260,    0 },   //  40
    {  1, 0, 2, 0, 1,    -510,    0,     270,    0 },   //  41
    {  0, 0, 2, 2, 2,    -380,    0,     160,    0 },   //  42
    {  2, 0, 0, 0, 0,     290,    0,     -10,    0 },   //  43
    {  1, 0, 2,-2, 2,     290,    0,    -120,    0 },   //  44
    {  2, 0, 2, 0, 2,    -310,    0,     130,    0 },   //  45
    {  0, 0, 2, 0, 0,     260,    0,     -10,    0 },   //  46
    { -1, 0, 2, 0, 1,     210,    0,    -100,    0 },   //  47
    { -1, 0, 0, 2, 1,     160,    0,     -80,    0 },   //  48
    {  1, 0, 0,-2, 1,    -130,    0,      70,    0 },   //  49
    { -1, 0, 2, 2, 1,    -100,    0,      50,    0 },   //  50
    {  1, 1, 0,-2, 0,     -70,    0,       0,    0 },   //  51
    {  0, 1, 2, 0, 2,      70,    0,     -30,    0 },   //  52
    {  0,-1, 2, 0, 2,     -70,    0,      30,    0 },   //  53
    {  1, 0, 2, 2, 2,     -80,    0,      30,    0 },   //  54
    {  1, 0, 0, 2, 0,      60,    0,       0,    0 },   //  55
    {  2, 0, 2,-2, 2,      60,    0,     -30,    0 },   //  56
    {  0, 0, 0, 2, 1,     -60,    0,      30,    0 },   //  57
    {  0, 0, 2, 2, 1,     -70,    0,      30,    0 },   //  58
    {  1, 0, 2,-2, 1,      60,    0,     -30,    0 },   //  59
    {  0, 0, 0,-2, 1,     -50,    0,      30,    0 },   //  60
    {  1,-1, 0, 0, 0,      50,    0,       0,    0 },   //  61
    {  2, 0, 2, 0, 1,     -50,    0,      30,    0 },   //  62
    {  0, 1, 0,-2, 0,     -40,    0,       0,    0 },   //  63
    {  1, 0,-2, 0, 0,      40,    0,       0,    0 },   //  64
    {  0, 0, 0, 1, 0,     -40,    0,       0,    0 },   //  65
    {  1, 1, 0, 0, 0,     -30,    0,       0,    0 },   //  66
    {  1, 0, 2, 0, 0,      30,    0,       0,    0 },   //  67
    {  1,-1, 2, 0, 2,     -30,    0,      10,    0 },   //  68
    { -1,-1, 2, 2, 2,     -30,    0,      10,    0 },   //  69
    { -2, 0, 0, 0, 1,     -20,    0,      10,    0 },   //  70
    {  3, 0, 2, 0, 2,     -30,    0,      10,    0 },   //  71
    {  0,-1, 2, 2, 2,     -30,    0,      10,    0 },   //  72
    {  1, 1, 2, 0, 2,      20,    0,     -10,    0 },   //  73
    { -1, 0, 2,-2, 1,     -20,    0,      10,    0 },   //  74
    {  2, 0, 0, 0, 1,      20,    0,     -10,    0 },   //  75
    {  1, 0, 0, 0, 2,     -20,    0,      10,    0 },   //  76
    {  3, 0, 0, 0, 0,      20,    0,       0,    0 },   //  77
    {  0, 0, 2, 1, 2,      20,    0,     -10,    0 },   //  78
    { -1, 0, 0, 0, 2,      10,    0,     -10,    0 },   //  79
    {  1, 0, 0,-4, 0,     -10,    0,       0,    0 },   //  80
    { -2, 0, 2, 2, 2,      10,    0,     -10,    0 },   //  81
    { -1, 0, 2, 4, 2,     -20,    0,      10,    0 },   //  82
    {  2, 0, 0,-4, 0,     -10,    0,       0,    0 },   //  83
    {  1, 1, 2,-2, 2,      10,    0,     -10,    0 },   //  84
    {  1, 0, 2, 2, 1,     -10,    0,      10,    0 },   //  85
    { -2, 0, 2, 4, 2,     -10,    0,      10,    0 },   //  86
    { -1, 0, 4, 0, 2,      10,    0,       0,    0 },   //  87
    {  1,-1, 0,-2, 0,      10,    0,       0,    0 },   //  88
    {  2, 0, 2,-2, 1,      10,    0,     -10,    0 },   //  89
    {  2, 0, 2, 2, 2,     -10,    0,       0,    0 },   //  90
    {  1, 0, 0, 2, 1,     -10,    0,       0,    0 },   //  91
    {  0, 0, 4,-2, 2,      10,    0,       0,    0 },   //  92
    {  3, 0, 2,-2, 2,      10,    0,       0,    0 },   //  93
    {  1, 0, 2,-2, 0,     -10,    0,       0,    0 },   //  94
    {  0, 1, 2, 0, 1,      10,    0,       0,    0 },   //  95
    { -1,-1, 0, 2, 1,      10,    0,       0,    0 },   //  96
    {  0, 0,-2, 0, 1,     -10,    0,       0,    0 },   //  97
    {  0, 0, 2,-1, 2,     -10,    0,       0,    0 },   //  98
    {  0, 1, 0, 2, 0,     -10,    0,       0,    0 },   //  99
    {  1, 0,-2,-2, 0,     -10,    0,       0,    0 },   // 100
    {  0,-1, 2, 0, 1,     -10,    0,       0,    0 },   // 101
    {  1, 1, 0,-2, 1,     -10,    0,       0,    0 },   // 102
    {  1, 0,-2, 2, 0,     -10,    0,       0,    0 },   // 103
    {  2, 0, 0, 2, 0,      10,    0,       0,    0 },   // 104
    {  0, 0, 2, 4, 2,     -10,    0,       0,    0 },   // 105
    {  0, 1, 0, 1, 0,      10,    0,       0,    0 }    // 106
   };

  // Variables

  double  l, lp, F, D, Om;
  double  arg;
  

  // Mean arguments of luni-solar motion
  //
  //   l   mean anomaly of the Moon
  //   l'  mean anomaly of the Sun
  //   F   mean argument of latitude
  //   D   mean longitude elongation of the Moon from the Sun 
  //   Om  mean longitude of the ascending node
  
  l  = Mod(  485866.733 + (1325.0*rev +  715922.633)*T + 31.310*T2 + 0.064*T3, rev );
  lp = Mod( 1287099.804 + (  99.0*rev + 1292581.224)*T -  0.577*T2 - 0.012*T3, rev );
  F  = Mod(  335778.877 + (1342.0*rev +  295263.137)*T - 13.257*T2 + 0.011*T3, rev );
  D  = Mod( 1072261.307 + (1236.0*rev + 1105601.328)*T -  6.891*T2 + 0.019*T3, rev );
  Om = Mod(  450160.280 - (   5.0*rev +  482890.539)*T +  7.455*T2 + 0.008*T3, rev );

  // Nutation in longitude and obliquity [rad]

  deps = dpsi = 0.0;
  for (int i=0; i<N_coeff; i++) {
    arg  =  Arcsec2Rad( ( C[i][0]*l+C[i][1]*lp+C[i][2]*F+C[i][3]*D+C[i][4]*Om ) );
    dpsi += ( C[i][5]+C[i][6]*T ) * sin(arg);
    deps += ( C[i][7]+C[i][8]*T ) * cos(arg);
  };
      
  dpsi = Arcsec2Rad( 1.0E-5 * dpsi);
  deps = Arcsec2Rad( 1.0E-5 * deps);

}




double orb::GMST(const double Mjd_UT1){
/*
 Purpose:
	Greenwich Mean Sidereal Time
 Input:
	Mjd_UT1   Modified Julian Date UT1
 Return:
	GMST in [rad]
*/
	// Constants
	const double Secs = 86400.0;        // Seconds per day
	// Variables
	double Mjd_0,UT1,T_0,T,gmst;

	// Mean Sidereal Time
	Mjd_0 = floor(Mjd_UT1);
	UT1   = Secs*(Mjd_UT1-Mjd_0);          // [s]
	T_0   = (Mjd_0  -MJD_J2000)/36525.0; 
	T     = (Mjd_UT1-MJD_J2000)/36525.0; 

	gmst  = 24110.54841 + 8640184.812866*T_0 + 1.002737909350795*UT1 + (0.093104-6.2e-6*T)*T*T; // [s]
	return  2.*def::PI*Frac(gmst/Secs); // [rad], 0..2pi
}



double orb::EqnEquinox(const double Mjd_TT){
  double dpsi, deps;              // Nutation angles
  // Nutation in longitude and obliquity 
  NutAngles(Mjd_TT, dpsi,deps);
  // Equation of the equinoxes
  return  dpsi * cos ( MeanObliquity(Mjd_TT) );
};



double orb::GAST(const double Mjd_UT1){
/*
 Purpose:
	Greenwich Apparent Sidereal Time
 Input:
	Mjd_UT1   Modified Julian Date UT1
 Return:
	GAST in [rad]
*/
	return Mod( GMST(Mjd_UT1) + EqnEquinox(Mjd_UT1), 2*def::PI );
}




D2<double> orb::PrecMatrix(const double MJD_TT_1,const double MJD_TT_2){
/*
 Purpose:
	Precession matrix
*/

	// Constants
	const double T  = (MJD_TT_1-new_time::MJD_J2000)/36525.0;
	const double dT = (MJD_TT_2-MJD_TT_1)/36525.0;

	// Variables
	double zeta,z,theta;

	// Precession angles
	zeta  =  ( (2306.2181+(1.39656-0.000139*T)*T)+((0.30188-0.000344*T)+0.017998*dT)*dT )*dT;
	z     =  zeta + ( (0.79280+0.000411*T)+0.000205*dT)*dT*dT;
	theta =  ( (2004.3109-(0.85330+0.000217*T)*T)-((0.42665+0.000217*T)+0.041833*dT)*dT )*dT;

	zeta = Arcsec2Rad(zeta);
	z    = Arcsec2Rad(z);
	theta= Arcsec2Rad(theta);


	d2::D2<double> out = (Rz(z) * Ry(-theta) * Rz(zeta));
	return out.Transpose();
}

D2<double> orb::NutMatrix(const double MJD_TT){
/*
 Purpose:
	Nutation matrix
*/
	// Mean obliquity of the ecliptic
	double eps = MeanObliquity(MJD_TT);
	// Nutation in longitude and obliquity
	double dpsi,deps;
	NutAngles(MJD_TT,dpsi,deps);
	// Transformation from mean to true equator and equinox
	return Rx(eps+deps) * Rz(dpsi) * Rx(-eps);
}

D2<double> orb::GHAMatrix(const double Mjd_UT1){
/*
 Purpose:
	Earth rotation matrix
*/
	return Rz(-GAST(Mjd_UT1));
}

D2<double> orb::PoleMatrix(double Mjd_UTC,IERS IERS_table){
/*
 Purpose:
	Polar motion matrix
*/
	iers::IERS_STRUC table = IERS_table.GetIERSBullB(Mjd_UTC);
	double xp = Arcsec2Rad(table.x);
	double yp = Arcsec2Rad(table.y);

	return Ry(xp) * Rx(yp);
}


D2<double> orb::ECI2ECRMatrix(const TIME& UTC, const IERS& IERS_table){
/*
 Purpose:
	Tind the tranformation matrix from ECI to ECR.
 Input:
	UTC : [Time] Calendar Date
 Ref:
 	Satellite Orbits, models, methods, and applications, 2000
*/
	double MJD_UTC = UTC.MJD();
	double MJD_TT  = (UTC2TT(UTC)).MJD();
	double MJD_UT1 = (UTC2UT1(UTC,IERS_table)).MJD();
	d2::D2<double> Prec = PrecMatrix( MJD_TT, new_time::MJD_J2000 );
	d2::D2<double> Nut  = NutMatrix ( MJD_TT );
	d2::D2<double> GHA  = GHAMatrix ( MJD_UT1 );
	d2::D2<double> Pole = PoleMatrix( MJD_UTC, IERS_table );

	return Pole * GHA * Nut * Prec;
}

D2<double> orb::ECI2ECRMatrix(const TIME& UTC, const IERS& IERS_table,D2<double>& dU){
/*
 Purpose:
	Find the tranformation matrix from ECI to ECR.
 Input:
	UTC : [Time] Calendar Date
 Ref:
 	Satellite Orbits, models, methods, and applications, 2000
*/
	double MJD_UTC = UTC.MJD();
	double MJD_TT  = (UTC2TT(UTC)).MJD();
	double MJD_UT1 = (UTC2UT1(UTC,IERS_table)).MJD();
	d2::D2<double> Prec = PrecMatrix( MJD_TT, new_time::MJD_J2000 );
	d2::D2<double> Nut  = NutMatrix ( MJD_TT );
	d2::D2<double> GHA  = GHAMatrix ( MJD_UT1 );
	d2::D2<double> Pole = PoleMatrix( MJD_UTC, IERS_table );

	double we = 7.2921158553e-5; // Earth rotaion angular velocity
	
	double series[]={0., 1., 0.,
					 -1.,0., 0.,
					 0., 0., 0.};
	D2<double> tmp(series,3,3);
	D2<double> dGHA = tmp * GHA * we;	//(5.92)
	dU = Pole * dGHA * Nut * Prec;

	return Pole * GHA * Nut * Prec;
}

D2<double> orb::ECR2ECIMatrix(const TIME& UTC, const IERS& IERS_table){
/*
 Purpose:
	Tind the tranformation matrix from ECR to ECI.
 Input:
	UTC : [Time] Calendar Date
 Ref:
 	Satellite Orbits, models, methods, and applications, 2000
*/
	D2<double> U = ECI2ECRMatrix(UTC,IERS_table);
	return U.Transpose();
}

D2<double> orb::ECR2ECIMatrix(const TIME& UTC, const IERS& IERS_table,D2<double>& dU){
/*
 Purpose:
	Tind the tranformation matrix from ECR to ECI.
 Input:
	UTC : [Time] Calendar Date
 Ref:
 	Satellite Orbits, models, methods, and applications, 2000
*/
	D2<double> U = ECI2ECRMatrix(UTC,IERS_table,dU);
	dU = dU.Transpose();
	return U.Transpose();
}


void orb::ECR2ECI(const VEC<double>& ECR_pos,const VEC<double>& ECR_vel,	//input
			 const TIME& UTC,const IERS& IERS_table,				//input
			 VEC<double>& ECI_pos,VEC<double>& ECI_vel){			//output
/*
 Purpose:
	Convert position and velocity form ECR to ECI.
 Input:
	UTC			: UTC time
	IERS_Table	: IERS reference matrix table
 Return:
	ECR_pos	: ECR position
	ECR_vel	: ECR velocity
*/
	D2<double> dU;
	D2<double> U = ECR2ECIMatrix(UTC,IERS_table,dU);

	ECI_pos = def_func::D22VEC(  U * def_func::VEC2D2(ECR_pos,3,1) );
	ECI_vel = def_func::D22VEC( (U * def_func::VEC2D2(ECR_vel,3,1)) + (dU * def_func::VEC2D2(ECR_pos,3,1)) );
}
void orb::ECI2ECR(const VEC<double>& ECI_pos,const VEC<double>& ECI_vel,	//input
			 const TIME& UTC,const IERS& IERS_table,				//input
			 VEC<double>& ECR_pos,VEC<double>& ECR_vel){			//output
/*
 Purpose:
	Convert position and velocity form ECI to ECR.
 Input:
	UTC			: UTC time
	IERS_Table	: IERS reference matrix table
 Return:
	ECR_pos	: ECR position
	ECR_vel	: ECR velocity
*/
	D2<double> dU;
	D2<double> U = ECI2ECRMatrix(UTC,IERS_table,dU);

	ECR_pos = def_func::D22VEC(  U * def_func::VEC2D2(ECI_pos,3,1) );
	ECR_vel = def_func::D22VEC( (U * def_func::VEC2D2(ECI_vel,3,1)) + (dU * def_func::VEC2D2(ECI_pos,3,1)) );
}






#endif
