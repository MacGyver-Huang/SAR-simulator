#ifndef NEW_TIME_H_INCLUDED
#define NEW_TIME_H_INCLUDED

#include <basic/def_func.h>
#include <basic/def_prefunc.h>
#include <coordinate/iers.h>

using namespace iers;
using namespace def_prefunc;


namespace new_time{
    //using namespace def_func;
	//using namespace def_prefunc;
	//using namespace iers;
    // ==============================================
	// Time Calss
	// ==============================================

	class TIME{
        public:
            // Constructor
			TIME();
            TIME(const int y,const int mo,const int d,const int h,const int m,const double s);
            TIME(const int y,const int mo,const int d);
            TIME(const TIME& in);
            // Operator
            friend const TIME operator+(const TIME& L,const double sec);
			friend ostream& operator<<(ostream& os, const TIME& in);
			TIME& operator=(const TIME& in);
            // Get
            int GetYear()const{return _y;};
            int GetMonth()const{return _mo;};
            int GetDay()const{return _d;};
            int GetHour()const{return _h;};
            int GetMin()const{return _m;};
            double GetSec()const{return _s;};
			string GetTimeString();
            // Misc.
			void SetTodayUTC();
			void SetTodayLocal();
            void Print()const;
            // Convert
            double JD()const;
            double MJD()const;
        private:
            int _y,_mo,_d,_h,_m;
            double _s;
    };

	// J2000
	const double JD_J2000 = 2451545.;
	const double MJD_J2000= 51544.5;
	const TIME TT_J2000 = TIME(2000,1,1,12,0,0.);
	const TIME TAI_J2000= TIME(2000,1,1,11,59,27.816);
	const TIME UTC_J2000= TIME(2000,1,1,11,58,55.816);

    //
    // Constructor
    //
	TIME::TIME(){
		_y=1981;_mo=2;_d=25;_h=11;_m=1;_s=1.;
    }

    TIME::TIME(const int y,const int mo,const int d,const int h,const int m,const double s){
        _y=y;_mo=mo;_d=d;_h=h;_m=m;_s=s;
    }

    TIME::TIME(const int y,const int mo,const int d){
        _y=y;_mo=mo;_d=d;
        _h=0;_m=0;_s=0.;
    }

    TIME::TIME(const TIME& in){
        _y = in._y;
        _mo= in._mo;
        _d = in._d;
        _h = in._h;
        _m = in._m;
        _s = in._s;
    }

    //
    // Operator
    //
	const TIME operator+(const TIME& L,const double sec){
	    int i_y,i_mo,i_d,i_h,i_m;
        double i_s;

        i_s = L.GetSec() + sec;

        i_m = L.GetMin();
        if(i_s >= 60){
            i_s-=60;
            i_m++;
        }

        i_h = L.GetHour();
        if(i_m >= 60){
            i_m-=60;
            i_h++;
        }

        i_d = L.GetDay();
        if(i_h >= 24){
            i_h-=24;
            i_d++;
        }

        i_mo = L.GetMonth();
        i_y  = L.GetYear();

        switch(i_mo){
            case 1:
                if(i_d >= 31){i_mo=2;} break;
            case 2:
                if((i_y%4)!=0){
                    if(i_d >= 28){i_mo=3;}
                }else{
                    if(i_d >= 29){i_mo=3;}
                }
                break;
            case 3:
                if(i_d >= 31){i_mo=4;} break;
            case 4:
                if(i_d >= 30){i_mo=5;} break;
            case 5:
                if(i_d >= 31){i_mo=6;} break;
            case 6:
                if(i_d >= 30){i_mo=7;} break;
            case 7:
                if(i_d >= 31){i_mo=8;} break;
            case 8:
                if(i_d >= 31){i_mo=9;} break;
            case 9:
                if(i_d >= 30){i_mo=10;} break;
            case 10:
                if(i_d >= 31){i_mo=11;} break;
            case 11:
                if(i_d >= 30){i_mo=12;} break;
            case 12:
                if(i_d >= 31){
                    i_mo=1;
                    i_y++;
                }
                break;
        }
        return TIME(i_y,i_mo,i_d,i_h,i_m,i_s);
	}
	
	ostream& operator<<(ostream& os, const TIME& in){
		string c_mo,c_d,c_h,c_m,c_s;
		
		c_mo = (in._mo<10)? string("0")+num2str(in._mo):num2str(in._mo);
		c_d  = (in._d<10)?  string("0")+num2str(in._d) :num2str(in._d);
		c_h  = (in._h<10)?  string("0")+num2str(in._h) :num2str(in._h);
		c_m  = (in._m<10)?  string("0")+num2str(in._m) :num2str(in._m);
		c_s  = (in._s<10)?  string("0")+num2str(in._s) :num2str(in._s);
		
		os<<"{"<<in._y<<"/"<<c_mo<<"/"<<c_d<<" "<<c_h<<":"<<c_m<<":"<<c_s<<"}";
		
		return os;
	}
	
	TIME& TIME::operator=(const TIME& in){
		_y = in._y;
		_mo= in._mo;
		_d = in._d;
		_h = in._h;
		_m = in._m;
		_s = in._s;
	
		return *this;
	}

	// Get
	string TIME::GetTimeString(){
		// return: e.g. "{2020/05/20 23:53:12.3457}"
		string c_mo,c_d,c_h,c_m,c_s;
		
        c_mo = (_mo<10)? string("0")+num2str(_mo):num2str(_mo);
        c_d  = (_d<10)?  string("0")+num2str(_d) :num2str(_d);
        c_h  = (_h<10)?  string("0")+num2str(_h) :num2str(_h);
        c_m  = (_m<10)?  string("0")+num2str(_m) :num2str(_m);
        c_s  = (_s<10)?  string("0")+num2str(_s) :num2str(_s);
		
        string out = string("{")+num2str(_y)+string("/")+c_mo+string("/")+
					 c_d+(" ")+c_h+string(":")+c_m+string(":")+c_s+string("}");
		return out;
	}
	
    //
    // Misc.
    //
	void TIME::SetTodayUTC(){
		time_t rawtime;
		struct tm tinfo;
		time( &rawtime );

#ifdef _MSC_VER
		gmtime_s(&tinfo,&rawtime);
#else
		tinfo = *(gmtime( &rawtime ));	// To UTC time
#endif
		
		_y = tinfo.tm_year+1900;
		_mo= tinfo.tm_mon+1;
		_d = tinfo.tm_mday;
		_h = tinfo.tm_hour;
		_m = tinfo.tm_min;
		_s = double(tinfo.tm_sec);
	}
	
	void TIME::SetTodayLocal(){
		time_t rawtime;
		struct tm tinfo;
		time( &rawtime );

#ifdef _MSC_VER
		localtime_s(&tinfo,&rawtime);
#else
		tinfo = *(localtime( &rawtime )); // To Local time
#endif
		
		_y = tinfo.tm_year+1900;
		_mo= tinfo.tm_mon+1;
		_d = tinfo.tm_mday;
		_h = tinfo.tm_hour;
		_m = tinfo.tm_min;
		_s = double(tinfo.tm_sec);
	}
	
    void TIME::Print()const{
        cout<<"+---------------+"<<endl;
        cout<<"|      TIME     |"<<endl;
		cout<<"+---------------+"<<endl;
		cout<<"Year  = "<<_y<<endl;
		cout<<"Month = "<<_mo<<endl;
		cout<<"Day   = "<<_d<<endl;
		cout<<"Hour  = "<<_h<<endl;
		cout<<"Min   = "<<_m<<endl;
		cout<<"Sec   = "<<_s<<endl;
    }

    //
    // Convert
    //
    double TIME::JD()const{
    /*
     Purpose:
        Convert to Julian Date (sec)
    */
        //cout<<double(  int( 7.*( _y + double(int((_mo+9.)/12.)) )/4. )  )<<endl;
        //cout<<double( int( 275.*_mo/9. ) )<<endl;
        //cout<<( (_s/60. + double(_m))/60. + _h )/24.<<endl;
        return 367.*_y - double( int( 7.*double(_y+int((_mo+9.)/12.))/4. ) ) +
               double( int( 275.*_mo/9. ) ) + _d + 1721013.5 +
               ( (_s/60. + double(_m))/60. + _h )/24.;
    }

    double TIME::MJD()const{
    /*
     Purpose:
        Convert to Modified Julian Date (sec)
    */
        return this->JD()-2400000.5;
    }

    // ===========================
    // Convert
    // ===========================
    // Declare
    template<typename T> double Day2Sec(const T days);
    double CD2Sec(const TIME& cd);
    double GetIERSBullC(const double MJD_UTC);
    template<typename T> double Sec2Day(const T sec_MJD);
    template<typename T> TIME MJD2CD(const T mjd);
    template<typename T> TIME Sec2CD(const T sec_MJD);
    TIME GPS2TAI(const double GPS);	
    TIME TAI2UTC(const TIME& TAI);
    template<typename T> TIME GPS2UTC(const T GPS);
	double TAI2GPS(const TIME& TAI);
	TIME UTC2TAI(const TIME& UTC);
	double UTC2GPS(const TIME& UTC);
	TIME UTC2UT1(const TIME& UTC,const IERS IERS_table);
	TIME TAI2TT(const TIME& TAI);
	TIME UTC2TT(const TIME& UTC);
	double JD2MJD(double JD);
	double MJD2JD(double MJD);


    // Implement
    template<typename T>
    double Day2Sec(const T days){
    /*
     Purpose:
        Convert days to secons
    */
        return double(days)*(24.*60.*60.);
    }

    double CD2Sec(const TIME& cd){
    /*
     Purpose:
        Calendar Date to sec based on MJD
     Input:
        cd : (TIME class)
    */
        double mjd=cd.MJD();
        return Day2Sec(mjd);
    }

    double GetIERSBullC(const double MJD_UTC){
    /*
     Purpose:
        Get the time distance of TAI-UTC
     Input:
        MJD_UTC :(mjd of UTC) ex:UTC.MJD()
     Return:
        The time distance of TAI-UTC
    */
        // Construct the table C
        double bull_c[]={
            TIME(1961, 1,1).MJD(), TIME(1961, 7,31).MJD(), 1.4228180 + (MJD_UTC - 37300.) * 0.001296,
            TIME(1961, 8,1).MJD(), TIME(1961,12,31).MJD(), 1.3728180 + (MJD_UTC - 37300) * 0.001296,
            TIME(1962, 1,1).MJD(), TIME(1963,10,31).MJD(), 1.8458580 + (MJD_UTC - 37665) * 0.001123,
            TIME(1963,11,1).MJD(), TIME(1963,12,31).MJD(), 1.9458580 + (MJD_UTC - 37665) * 0.001123,
            TIME(1964, 1,1).MJD(), TIME(1964, 3,31).MJD(), 3.2401300 + (MJD_UTC - 38761) * 0.001296,
            TIME(1964, 4,1).MJD(), TIME(1964, 8,31).MJD(), 3.3401300 + (MJD_UTC - 38761) * 0.001296,
            TIME(1964, 9,1).MJD(), TIME(1964,12,31).MJD(), 3.4401300 + (MJD_UTC - 38761) * 0.001296,
            TIME(1965, 1,1).MJD(), TIME(1965, 2,28).MJD(), 3.5401300 + (MJD_UTC - 38761) * 0.001296,
            TIME(1965, 3,1).MJD(), TIME(1965, 6,30).MJD(), 3.6401300 + (MJD_UTC - 38761) * 0.001296,
            TIME(1965, 7,1).MJD(), TIME(1965, 8,31).MJD(), 3.7401300 + (MJD_UTC - 38761) * 0.001296,
            TIME(1965, 9,1).MJD(), TIME(1965,12,31).MJD(), 3.8401300 + (MJD_UTC - 38761) * 0.001296,
            TIME(1966, 1,1).MJD(), TIME(1968, 1,31).MJD(), 4.3131700 + (MJD_UTC - 39126) * 0.002592,
            TIME(1968, 2,1).MJD(), TIME(1971,12,31).MJD(), 4.2131700 + (MJD_UTC - 39126) * 0.002592,
            TIME(1972, 1,1).MJD(), TIME(1972, 6,30).MJD(), 10.,
            TIME(1972, 7,1).MJD(), TIME(1972,12,31).MJD(), 11.,
            TIME(1973, 1,1).MJD(), TIME(1973,12,31).MJD(), 12.,
            TIME(1974, 1,1).MJD(), TIME(1974,12,31).MJD(), 13.,
            TIME(1975, 1,1).MJD(), TIME(1975,12,31).MJD(), 14.,
            TIME(1976, 1,1).MJD(), TIME(1976,12,31).MJD(), 15.,
            TIME(1977, 1,1).MJD(), TIME(1977,12,31).MJD(), 16.,
            TIME(1978, 1,1).MJD(), TIME(1978,12,31).MJD(), 17.,
            TIME(1979, 1,1).MJD(), TIME(1979,12,31).MJD(), 18.,
            TIME(1980, 1,1).MJD(), TIME(1981, 6,30).MJD(), 19.,
            TIME(1981, 7,1).MJD(), TIME(1982, 6,30).MJD(), 20.,
            TIME(1982, 7,1).MJD(), TIME(1983, 6,30).MJD(), 21.,
            TIME(1983, 7,1).MJD(), TIME(1985, 6,30).MJD(), 22.,
            TIME(1985, 7,1).MJD(), TIME(1987,12,31).MJD(), 23.,
            TIME(1988, 1,1).MJD(), TIME(1989,12,31).MJD(), 24.,
            TIME(1990, 1,1).MJD(), TIME(1990,12,31).MJD(), 25.,
            TIME(1991, 1,1).MJD(), TIME(1992, 6,30).MJD(), 26.,
            TIME(1992, 7,1).MJD(), TIME(1993, 6,30).MJD(), 27.,
            TIME(1993, 7,1).MJD(), TIME(1994, 6,30).MJD(), 28.,
            TIME(1994, 7,1).MJD(), TIME(1995,12,31).MJD(), 29.,
            TIME(1996, 1,1).MJD(), TIME(1997, 6,30).MJD(), 30.,
            TIME(1997, 7,1).MJD(), TIME(1998,12,31).MJD(), 31.,
            TIME(1999, 1,1).MJD(), TIME(2005,12,31).MJD(), 32.,
            TIME(2006, 1,1).MJD(), TIME(2999,12,31).MJD(), 33.
        };

        double out_TAI_UTC=0.;

        for(long i=0;i<37;++i){
            if( (MJD_UTC >= bull_c[i*3])&&(MJD_UTC <= bull_c[i*3+1]) ){
                out_TAI_UTC = bull_c[i*3+2];
            }
        }
        return out_TAI_UTC;
    }

    template<typename T>
    double Sec2Day(const T sec_MJD){
    /*
     Purpose:
        Convert from sec to days
     Input:
        sec_MJD : MJD using sec
    */
        return double(sec_MJD)/(24.*60.*60.);
    }

    template<typename T>
    TIME MJD2CD(const T mjd){
    /*
     Purpose:
        Convert Modified Julian Date to Calendar Date
    */
        double a = double(int(mjd)) + 2400001.;
        double q = double(mjd) - double(int(mjd));
        double b,c,d,e,f;
        double day_tp,day,month,year;
        double hour_tp,hour,min_tp,min,sec;

        if(a < 2299161.){
            b = 0.;
            c = a + 1524.;
        }else{
            b = double( int((a-1867216.25)/36524.25) );
            c = a + b - double(int(b/4.)) + 1525.;
        }

        d = double( int( (c-121.1)/365.25 ) );
        e = double( int( 365.25*d ) );
        f = double( int( (c-e)/30.6001 ) );

        day_tp = c-e-double( int(30.6001*f) )+q;
        day = double(int(day_tp));
        month = f-1. -12.*double(int(f/14.));
        year = d-4715. - double(int((7. + month)/10.));

        hour_tp = (day_tp - day)*24.;
        hour = double(int(hour_tp));
        min_tp = (hour_tp - hour)*60.;
        min = double(int(min_tp));
        sec = (min_tp-min)*60.;

        //cout<<"mjd="<<mjd<<endl;
        //cout<<"a="<<a<<" b="<<b<<" c="<<c<<" d="<<d<<" day="<<day<<endl;
        //cout<<"day_tp="<<day_tp<<" e="<<e<<" f="<<f<<" hour="<<hour<<endl;
        //cout<<"hour_tp="<<hour_tp<<" min="<<min<<" min_tp="<<min_tp<<endl;
        //cout<<"month="<<month<<" Q="<<q<<" sec="<<sec<<" year="<<year<<endl;
        return TIME(int(year),int(month),int(day),int(hour),int(min),sec);
    }

    template<typename T>
    TIME Sec2CD(const T sec_MJD){
    /*
     Purpose:
        Convert from sec to Calandar date
     Input:
        sec_MJD : MJD using sec
    */
        double mjd_days = Sec2Day(sec_MJD); // [days]
        return MJD2CD(mjd_days);
    }

    TIME GPS2TAI(const double GPS){
    /*
     Purpose:
        Convert from GPS to UTC
     Input:
        GPS :[sec] GPS time
    */
        double org = Day2Sec( TIME(1980,1,6).MJD() ); // [sec]
        double TAI_sec = GPS + org + 19.;   // [sec]

        return Sec2CD(TAI_sec);
    }

    TIME TAI2UTC(const TIME& TAI){
    /*
     Purpose:
        Convert from TAI to UTC
    */
        double in = CD2Sec(TAI);
        double TAI_UTC = GetIERSBullC(Sec2Day(in)); // [sec]
        double UTC_sec = in - TAI_UTC;

        //cout<<"in="<<in-4698986400.<<" TAI_UTC="<<TAI_UTC<<" UTC_sec="<<UTC_sec<<endl;

        return Sec2CD(UTC_sec);
    }

    template<typename T>
    TIME GPS2UTC(const T GPS){
    /*
     Purpose:
        Convert from GPS to UTC
     Input:
        GPS :[sec] GPS time
    */
        TIME TAI=GPS2TAI(GPS);
        return TAI2UTC(TAI);
    }
	
	double TAI2GPS(const TIME& TAI){
	/*
 	 Purpose:
		Convert from TAI to GPS
	 Input:
		TAI : [CD]
	*/
		double in = CD2Sec(TAI);
		TIME ref(1980,1,6,0,0,0);
		double org = Day2Sec(ref.MJD());
		return in - org - 19.; 
	}
	
	TIME UTC2TAI(const TIME& UTC){
	/*
	 Purpose:
		Convert UTC to TAI
	 Input:
		UTC	:[CD] UTC time
	*/
		
		double mjd_UTC = UTC.MJD();
		double TAI_UTC = new_time::GetIERSBullC(mjd_UTC);
		double mjd_TAI = mjd_UTC + TAI_UTC/86400.;
		return new_time::MJD2CD(mjd_TAI);
	}

	double UTC2GPS(const TIME& UTC){
	/*
 	 Purpose:
		Convert from UTC to GPS
	 Input:
		UTC : [CD]
	*/
		TIME TAI=UTC2TAI(UTC);
		return TAI2GPS(TAI);
	}

	TIME UTC2UT1(const TIME& UTC,const IERS IERS_table){
	/*
	 Purpose:
		Convert UTC to UT1
	 Input:
		UTC	:[CD] UTC time
	*/
		double mjd_UTC = UTC.MJD();
		IERS_STRUC IERS1 = IERS_table.GetIERSBullB(mjd_UTC);
		double UT1_UTC = IERS1.UT1_UTC;
		double mjd_UT1 = mjd_UTC + UT1_UTC/86400.;
		return new_time::MJD2CD(mjd_UT1);
	}

	TIME TAI2TT(const TIME& TAI){
	/*
	 Purpose:
		Convert TAI to TT
	 Input:
		UTC	:[CD] UTC time
	*/
		double mjd_TAI = TAI.MJD();
		return new_time::MJD2CD(mjd_TAI + 32.184/86400.);
	}

	TIME UTC2TT(const TIME& UTC){
	/*
	 Purpose:
		Convert UTC to TAI
	 Input:
		UTC	:[CD] UTC time
	*/
		TIME TAI = new_time::UTC2TAI(UTC);
		return new_time::TAI2TT(TAI);
	}


	double JD2MJD(double JD){
	/*
	 Purpose:
		Convert from Julian day to Modified Julian Day
	*/
		return JD - 2400000.5;
	}

	double MJD2JD(double MJD){
	/*
	 Purpose:
		Convert from Modified Julian day to Julian Day
	*/
		return MJD + 2400000.5;
	}
}



#endif // NEW_TIME_H_INCLUDED
