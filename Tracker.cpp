// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "OnlineBoost.h"
#include "Tracker.h"
#include "Public.h"
#include "Sample.h"


bool Tracker::init(Matrixu *img, TrackerParams *p, ClfParams *clfparams)
{
	SampleSet posx, negx;
	int iiFree;

	iiFree = (!img->isInitII());

	if (!img->isInitII())
		img->initII();

	_clf = new ClfStrong();
	_clf->init(clfparams);

	_lost = 0;
	_x = p->_initX;
	_y = p->_initY;
	_w = p->_initW;
	_h = p->_initH;
	_scale = p->_initScale;

	fprintf(stderr,"Initializing Tracker..\n");

	// sample positives and negatives from first frame
	posx.sampleImage(img, _x, _y, _w, _h, p->_init_postrainrad);
	negx.sampleImage(img, _x, _y, _w, _h, 2.0f*p->_srchwinsz, (1.5f*p->_init_postrainrad), p->_init_negnumtrain);
	if( posx.size()<1 || negx.size()<1 ) return false;

	// train
	_clf->update(posx, negx);
	posx.clear();
	negx.clear();

	if (iiFree)
		img->FreeII();

	_trparams = p;
	_clfparams = clfparams;

	return true;
}


double Tracker::update_location(Matrixu *img)
{
	static SampleSet detectx;
	static vectorf prob;
	double resp;

	if (!img->isInitII())
		abortError(__LINE__,__FILE__,"Integral image not initialized before calling update_location");

	// run current clf on search window
	detectx.sampleImage(img, _x, _y, _w, _h, (float)_trparams->_srchwinsz);
	prob = _clf->classify(detectx,_trparams->_useLogR);

	// find best location
	int bestind = max_idx(prob);
	resp=prob[bestind];
	_x = (float)detectx[bestind]._col;
	_y = (float)detectx[bestind]._row; 

	// clean up
	detectx.clear();
	return resp;
}

void Tracker::update_classifier(Matrixu *img)
{
	static SampleSet posx, negx;

	if (!img->isInitII())
		abortError(__LINE__,__FILE__,"Integral image not initialized before calling update_classifier");

	// train location clf (negx are randomly selected from image, posx is just the current tracker location)
	negx.sampleImage(img, _x, _y, _w, _h, (1.5f*_trparams->_srchwinsz), _trparams->_posradtrain+5, _trparams->_negnumtrain);
	posx.sampleImage(img, _x, _y, _w, _h, _trparams->_posradtrain, 0, _trparams->_posmaxtrain);
	_clf->update(posx, negx);

	// clean up
	posx.clear(); negx.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
TrackerParams::TrackerParams()
{
	_srchwinsz		= 30;
	_negsamplestrat	= 1;
	_boxcolor.resize(3);
	_boxcolor[0]	= 204;
	_boxcolor[1]	= 25;
	_boxcolor[2]	= 204;
	_lineWidth		= 2;
	_negnumtrain	= 15;
	_posradtrain	= 1;
	_posmaxtrain	= 100000;
	_init_negnumtrain = 1000;
	_init_postrainrad = 3;
	_initX			= 0;
	_initY			= 0;
	_initW			= 0;
	_initH			= 0;
	_initScale		= 1.0;
	_debugv			= false;
	_useLogR		= true;
	_disp			= true;
	_initWithFace	= true;
}


