// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "OnlineBoost.h"
#include "Sample.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////
void ClfWeak::init(int id, float lRate, Ftr *ftr)
{
	_lRate=lRate;
	_ind=id;
	_ftr = ftr;
	_mu0	= 0;
	_mu1	= 0;
	_sig0	= 1;
	_sig1	= 1;
	_lRate	= 0.85f;
	_trained = false;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
void ClfStrong::init(ClfParams *params)
{
	// initialize model
	_params		= params;
	_numsamples = 0;

	_ftrs = Ftr::generate(_params->_ftrParams, _params->_numFeat);
	_weakclf.resize(_params->_numFeat);
	for( int k=0; k<_params->_numFeat; k++ ) {
		_weakclf[k] = new ClfWeak();
		_weakclf[k]->init(k, _params->_lRate, _ftrs[k]);
	}
}


void ClfStrong::update(SampleSet &posx, SampleSet &negx)
{
	int numneg = negx.size();
	int numpos = posx.size();

	// compute ftrs
	if( !posx.ftrsComputed() ) Ftr::compute(posx, _ftrs);
	if( !negx.ftrsComputed() ) Ftr::compute(negx, _ftrs);

	// initialize H
	static vectorf Hpos, Hneg;
	Hpos.clear(); Hneg.clear();
	Hpos.resize(posx.size(),0.0f), Hneg.resize(negx.size(),0.0f);

	_selectors.clear();
	vectorf posw(posx.size()), negw(negx.size());
	vector<vectorf> pospred(_weakclf.size()), negpred(_weakclf.size());

	// train all weak classifiers without weights
	#pragma omp parallel for
	for( int m=0; m<_params->_numFeat; m++ ){
		_weakclf[m]->update(posx, negx);
		pospred[m] = _weakclf[m]->classifySetF(posx);
		negpred[m] = _weakclf[m]->classifySetF(negx);
	}


	// pick the best features
	for( int s=0; s<_params->_numSel; s++ ){

		// compute errors/likl for all weak clfs
		vectorf poslikl(_weakclf.size(),1.0f), neglikl(_weakclf.size()), likl(_weakclf.size());
		#pragma omp parallel for
		for( int w=0; w<(int)_weakclf.size(); w++) {
			float lll=1.0f;
			//#pragma omp parallel for reduction(*: lll)
			for( int j=0; j<numpos; j++ )
				lll *= oneminussigmoid(Hpos[j]+pospred[w][j]);
			poslikl[w] = (float)-logf(1-lll+1e-5);

			lll=1.0f;
			for( int j=0; j<numneg; j++ )
				lll *= oneminussigmoid(Hneg[j]+negpred[w][j]);
			neglikl[w]= (float)-logf(lll+1e-5);

			likl[w] = poslikl[w]/numpos + neglikl[w]/numneg;
		}

		// pick best weak clf
		vectori order;
		sort_order_asc(likl,order);

		// find best weakclf that isn't already included
		for( uint k=0; k<order.size(); k++ )
			if( count( _selectors.begin(), _selectors.end(), order[k])==0 ){
				_selectors.push_back(order[k]);
				break;
			}

		// update H = H + h_m
		#pragma omp parallel for
		for( int k=0; k<posx.size(); k++ )
			Hpos[k] += pospred[_selectors[s]][k];
		#pragma omp parallel for
		for( int k=0; k<negx.size(); k++ )
			Hneg[k] += negpred[_selectors[s]][k];
	}

	return;
}



