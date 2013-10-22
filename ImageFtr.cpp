// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "ImageFtr.h"
#include "Sample.h"



//////////////////////////////////////////////////////////////////////////////////////////////////////////
void Ftr::generate(FtrParams *p)
{
	_width = p->_width;
	_height = p->_height;
	int numrects = randint(p->_minNumRect,p->_maxNumRect);
	_rects.resize(numrects);
	_weights.resize(numrects);

	for( int k=0; k<numrects; k++ )
	{
		//_weights[k] = randfloat()*2-1;
		int wt = (randfloat()*16);
		_weights[k] = (float)(wt - 8);
		_rects[k].x = randint(0,(uint)(p->_width-3));
		_rects[k].y = randint(0,(uint)(p->_height-3));
		_rects[k].width = randint(1,(p->_width-_rects[k].x-2));
		_rects[k].height = randint(1 ,(p->_height-_rects[k].y-2));
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
void Ftr::compute( SampleSet &samples, const vecFtr &ftrs)
{
	int numftrs = ftrs.size();
	int numsamples = samples.size();
	if( numsamples==0 ) return;

	samples.resizeFtrs(numftrs);

	#pragma omp parallel for
	for( int ftr=0; ftr<numftrs; ftr++ ){
		//#pragma omp parallel for
		for( int k=0; k<numsamples; k++ ){
			samples.getFtrVal(k,ftr) = ftrs[ftr]->compute(samples[k]);
		}
	}

}


vecFtr Ftr::generate( FtrParams *params, uint num )
{
	vecFtr ftrs;

	ftrs.resize(num);
	for( uint k=0; k<num; k++ ){
		ftrs[k] = new Ftr();
		ftrs[k]->generate(params);
	}
	return ftrs;
}


