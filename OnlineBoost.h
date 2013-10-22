// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef ONLINEBOOST_H
#define ONLINEBOOST_H

#include "Public.h"
#include "ImageFtr.h"


class ClfWeak
{
public:
	void				init(int id, float lRate, Ftr *ftr);
	void				update(SampleSet &posx, SampleSet &negx);
	float				classifyF(SampleSet &x, int i);
	vectorf				classifySetF(SampleSet &x);

	float				ftrcompute(const Sample &x) {return _ftr->compute(x);};
	float				getFtrVal(const SampleSet &x,int i) { return (x.ftrsComputed()) ? x.getFtrVal(i,_ind) : _ftr->compute(x[i]); };

	float				_mu0, _mu1, _sig0, _sig1;
	float				_n1, _n0;
	float				_e1, _e0;
	bool				_trained;
	Ftr					*_ftr;
	vecFtr				*_ftrs;
	int					_ind;
	float				_lRate;
};


class ClfParams {
public:
						ClfParams(){_lRate=0.85f;_numSel=50;_numFeat=250;};

	FtrParams			*_ftrParams;
	float				_lRate; // learning rate for weak learners;
	int					_numFeat;
	int					_numSel;
};


class ClfStrong {
public:
	ClfParams			*_params;
	vecFtr				_ftrs;
	vectori				_selectors;
	vector<ClfWeak*>	_weakclf;
	uint				_numsamples;

public:
	int					nFtrs() {return _ftrs.size();};

	void				init(ClfParams *params);
	void				update(SampleSet &posx, SampleSet &negx);
	vectorf				classify(SampleSet &x, bool logR=true);

};



//////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void ClfWeak::update(SampleSet &posx, SampleSet &negx)
{
	float posmu=0.0,negmu=0.0;
	if( posx.size()>0 ) posmu=posx.ftrVals(_ind).Mean();
	if( negx.size()>0 ) negmu=negx.ftrVals(_ind).Mean();

	if( _trained){
		if( posx.size()>0 ){
			_mu1	= ( _lRate*_mu1  + (1-_lRate)*posmu );
			_sig1	= ( _lRate*_sig1 + (1-_lRate)* ( (posx.ftrVals(_ind)-_mu1).Sqr().Mean() ) );
		}
		if( negx.size()>0 ){
			_mu0	= ( _lRate*_mu0  + (1-_lRate)*negmu );
			_sig0	= ( _lRate*_sig0 + (1-_lRate)* ( (negx.ftrVals(_ind)-_mu0).Sqr().Mean() ) );
		}

		_n0 = logf(1e-5*(1.0f/pow(_sig0,0.5f)));
		_n1 = logf(1e-5*(1.0f/pow(_sig1,0.5f)));
		_e1 = -1.0f/(2.0f*_sig1+1e-99);
		_e0 = -1.0f/(2.0f*_sig0+1e-99);
	}
	else{
		_trained = true;
		if( posx.size()>0 ){
			_mu1 = posmu;
			_sig1 = posx.ftrVals(_ind).Var()+1e-9f;
		}

		if( negx.size()>0 ){
			_mu0 = negmu;
			_sig0 = negx.ftrVals(_ind).Var()+1e-9f;
		}

		_n0 = logf(1e-5*(1.0f/pow(_sig0,0.5f)));
		_n1 = logf(1e-5*(1.0f/pow(_sig1,0.5f)));
		_e1 = -1.0f/(2.0f*_sig1+1e-99);
		_e0 = -1.0f/(2.0f*_sig0+1e-99);
	}
}

inline vectorf ClfWeak::classifySetF(SampleSet &x)
{
	vectorf res(x.size());
	
	#pragma omp parallel for
	for( int k=0; k<(int)res.size(); k++ ){
		res[k] = classifyF(x,k);
	}
	return res;
}

inline float ClfWeak::classifyF(SampleSet &x, int i)
{
	float xx = getFtrVal(x,i);
	//float r = ((xx-_mu1)*(xx-_mu1)*_e1) - ((xx-_mu0)*(xx-_mu0)*_e0) + logf(1e-5*_n1) - logf(1e-5+_n0);
	float r = ((xx-_mu1)*(xx-_mu1)*_e1) - ((xx-_mu0)*(xx-_mu0)*_e0) + _n1 - _n0;

	return r;
}







inline vectorf ClfStrong::classify(SampleSet &x, bool logR)
{
	int numsamples = x.size();
	vectorf res(numsamples);
	vectorf tr;

	for( uint w=0; w<_selectors.size(); w++ ){
		tr = _weakclf[_selectors[w]]->classifySetF(x);
		#pragma omp parallel for
		for( int j=0; j<numsamples; j++ ){
			res[j] += tr[j];
		}
	}

	return res;
}


#endif
