//
//  stopwatch.h
//  PhysicalOptics02
//
//  Created by Steve Chiang on 1/27/14.
//  Copyright (c) 2014 Steve Chiang. All rights reserved.
//

#ifndef stopwatch_h
#define stopwatch_h


#include <sys/time.h>

class Stopwatch {
private:
	double start;
	double _stopwatch() const
	{
		struct timeval time;
		gettimeofday(&time, 0 );
		return 1.0 * time.tv_sec + time.tv_usec / (double)1e6;
	}
public:
	Stopwatch() { reset(); }
	void reset() { start = _stopwatch(); }
	double read() const {	return _stopwatch() - start; }
};


#endif
