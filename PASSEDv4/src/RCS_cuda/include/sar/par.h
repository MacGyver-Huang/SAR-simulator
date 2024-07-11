#ifndef PAR_H_INCLUDED
#define PAR_H_INCLUDED

#include <basic/vec.h>
#include <coordinate/geo.h>
#include <sar/sv.h>
#include <basic/d1.h>


namespace par{
    using namespace vec;
    using namespace geo;
    using namespace sv;
    using namespace d1;
    // ==============================================
	// SAR sensor parameter
	// ==============================================
	class SAR_PAR{
	    public:
            char title[132];	// ascii text description of parameter file
            char sensor[16];	// sensor name (RADARSAT, SRL-1, SRL-2, ERS1, ERS2, JERS-1,...)
            char up_down[16];	// chirp direction flag values: (UP_CHIRP, DOWN_CHIRP)
            char s_mode[16];	// Receiver ADC mode: (REAL, IQ), REAL denotes offset video sampling, while
                                // IQ is in-phase/quadrature ADC sampling, 2 samples/point
            char s_type[16];	// sample type (FLOAT, BYTE), floats are 4 bytes/value and bytes
                                // are of type unsigned char, 1 byte/value
            char spectrum[16];	// SAR receiver spectrum: (NORMAL, INVERT), inverted spectrum caused by the
                                // recevier L.O. frequency above the chirp signal spectrum
            double fc;		// SAR center frequency (Hz)
            double bw;		// chirp bandwidth (Hz)
            double plen;	// chirp duration (sec)
            double fs;		// range ADC sampling frequency (Hz)
            int file_hdr_sz;// SAR raw data file header size (bytes)
            int nbyte;      // length of each record in bytes
            int byte_h;		// record header size in bytes
            int ns_echo;	// number of samples/record (IQ pair counts as one sample)
            double az_ant_bw;	// azimuth antenna 3 dB (half-power) beamwidth (decimal deg.)
            double r_ant_bw;	// range antenna 3 dB (half-power) beamwidth (decimal deg.)

            //
            // COORDINATE SYSTEM FOR AZIMUTH, LOOK AND PITCH ANGLES:
            //
            // N = DOWN
            // C = N x V /|N x V|
            // T = C x N
            //
            // ROLL is measured about T axis CW + (right wing down)
            // PITCH is measured CW about the temp C axis (nose up +)
            // YAW is measured CW about +N axis (looking down).
            //

            double az_ang;	// nominal azimuth antenna angle (including ave. squint for simulation)
                            // (decimal degrees CW about N, right looking SAR 90.0, left looking: -90.0)
            double lk_ang;	// nominal antenna look angle (decimal degrees CW (includes ave. roll for simulation)
                            // about the final T axis, rotation dependent on right or left looking
            double pitch;		// nominal platform pitch angle, CW rot. about the temp. C axis, nose up +
            char antpatf[500];	// antenna pattern filename (relative to peak, one-way gain) (not in dB!)
	};

    // ==============================================
	// Processing parameter
	// ==============================================
	class PROC_PAR{
	    public:
            int MXST;          // Max number of state vector
            char title[128];    // ascii text description of scene and processing parameters
            char date[40];	    // date data acquired/generated in decimal numbers (YYYY MM DD)
            char time[40];	    // time of first record in SAR signal data file (HH MM SS.SSSS)
            char pol_ch[40];	// polarization/channel to be processed out of the set {HH, HV, VH, VV, CH1, CH2}
            double el_major, el_minor;	// earth ellipsoid semi-major, semi-minor axises (m)
            double lat,lon;	    // latitude and longitude of scene center (decimal degrees)
            double track;		// track angle of radar platform (decimal degrees)
            double alt;		    // average altitude above geoid of the radar platform (m)
            double terra_alt;	// average reference height of terrain in the scene above the geoid (m)
            VEC<double> pos;	// position of platform at center of the
                                // prefiltered data in (X,Y,Z/T,C,N) coordinates (m)
            VEC<double> vel;	// velocity vector (X,Y,Z/T,C,N) along track (m/s)
                                // at the center of the prefiltered data
            VEC<double> acc;	// acceleration vector (X,Y,Z/T,C,N) along track (m/s^2)
                                // at the center of prefiltered data
            double prf;		// radar pulse repetition frequency PRF (Hz)
            double I_bias;	// DC bias of the raw data I channel
            double Q_bias;	// DC bias of the raw data Q channel
            double I_sigma;	// standard deviation of the raw data I channel
            double Q_sigma;	// standard deviation of the raw data Q channel
            double IQ_corr;	// correlation of the I and Q channels
            double SNR_rspec;	// average SNR determined from range spectrum
            double DAR_dop;	// doppler ambiguity resolver estimate of the doppler centroid at center swath
            double DAR_snr;	// unambiguous doppler estimate signal to noise ratio
            D1<double> fdp;	// doppler centroid polynomial coefficients as a function of range
                            // fd=fdp[0]+fdp[1]*r+fdp[2]*r**2+fdp[3]*r**3 (Hz)/m
            double td;		    // time delay between transmission of pulse and first sample of echo (s)
            double rx_gain;	// receiver gain dB
            double cal_gain;	// calibration gain dB
            double r0,r1,r2;	// raw SAR data near, center, far slant ranges (m)
            double ir0,ir1,ir2;	// output image near, center and far range (m)
            double rpixsp,ran_res;  // slant-range pixel spacing, slant-range resolution (m)
            char sec_range_mig[4];  // secondary range migration correction, values: (ON,OFF)
            char az_deskew[4];	    // azimuth SAR processing, deskew on or off (ON, OFF)
            double autofocus_snr;   // autofocus signal to noise ratio
            double prfrac;	// fraction of doppler bandwidth to process (0.0 -- 1.0, nominal=.8)
            int nprs_az;	// azimuth prefilter decimation factor (4 for a 4:1 decimation)
            int loff;		// offset in echos from first SAR data record to begin processing
            int nl;		    // number echoes to process
            int nr_offset;	// offset to first  range sample, for IQ data, 1 complex pair = 1 sample
            int nrfft;		// size of range FFT, power of 2, 2048,4096,8192,16384..
            int nlr,nlaz;	// number of range looks, number of azimuth looks
            double azoff;	// along-track azimuth offset of the first image line from the
                            // start of the SAR raw data (s)
            double azimsp,azimres;// azimuth image pixel spacing, azimuth image resolution (m)
            int nrs, nazs;	// image width in range pixels, length in azimuth pixels
            double sensor_lat;  // SAR geodetic latitude at image center
            double sensor_lon;	// SAR geodetic longitude at image center
            double sensor_track;// track angle of the sensor at image center
            D1<GEO<double> > map;// latitude and longitude of the 4 image corners and image center:
                                 // 1. first image record, near range (WGS-84 coordinates)
                                 // 2. first image record, far range
                                 // 3. last image record, near range
                                 // 4. last image record, far range
                                 // 5. image center at (int)(nlr/2), (int)(nlaz/2)
            int nstate;		// number of state vectors
            double t_state;	// UTC time (sec) since start of day for first state vector
            double tis;		// time interval between state vectors (s)
            SV<double> state;	// maximum number of MXST state vectors (X,Y,Z)
                                // earth fixed coordinates (e.g. Conventional Terrestrial System)
	};
}


#endif // PAR_H_INCLUDED
