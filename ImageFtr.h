// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef H_IMGFTR
#define H_IMGFTR

#include "Matrix.h"
#include "Public.h"
#include "Sample.h"

class Ftr;
typedef vector<Ftr*> vecFtr;

//////////////////////////////////////////////////////////////////////////////////////////////////////////

class FtrParams {
public:
	int					_width, _height;
	int					_maxNumRect, _minNumRect;
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////
class Ftr {
public:
							Ftr() {_width=0; _height=0;};
	float					compute( const Sample &sample ) const;
	void					generate( FtrParams *params );

	static void				compute( SampleSet &samples, const vecFtr &ftrs );
	static vecFtr			generate( FtrParams *params, uint num );
	uint					_width, _height;
	vectorf					_weights;
	vector<IppiRect>		_rects;
};




//////////////////////////////////////////////////////////////////////////////////////////////////////////

inline float Ftr::compute( const Sample &sample ) const
{
	if( !sample._img->isInitII() ) abortError(__LINE__,__FILE__,"Integral image not initialized before called compute()");
	IppiRect r;
	float sum = 0.0f;

	//#pragma omp parallel for
	for( int k=0; k<(int)_rects.size(); k++ )
	{
		r = _rects[k];
		r.x = sample._col + r.x; 
		r.y = sample._row + r.y;
		sum += _weights[k]*sample._img->sumRect(r,0);///_rsums[k];
	}

	return (float)(sum);
}





#endif
