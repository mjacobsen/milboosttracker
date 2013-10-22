#include "Matrix.h"
#include "ImageFtr.h"
#include "Tracker.h"
#include "Public.h"
#include <string.h>
#include <algorithm>

#define NUM_TRACKERS 1
#define NUM_SEL_FEATS 50
#define NUM_FEATS 250

static CvHaarClassifierCascade	*facecascade;

GET_TIME_INIT(2);


bool initFace(Matrixu *frame, int *x, int *y, int *w, int *h) {
	const char* cascade_name = "haarcascade_frontalface_alt_tree.xml";
	const int minsz = 20;
	if( facecascade == NULL )
		facecascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );

	frame->createIpl();
	IplImage *img = frame->getIpl();
	IplImage* gray = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U, 1 );
    cvCvtColor(img, gray, CV_BGR2GRAY );
	frame->freeIpl();
	cvEqualizeHist(gray, gray);

	CvMemStorage* storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);
	CvSeq* faces = cvHaarDetectObjects(gray, facecascade, storage, 1.05, 3, CV_HAAR_DO_CANNY_PRUNING ,cvSize(minsz, minsz));
	
	int index = faces->total-1;
	CvRect* r = (CvRect*)cvGetSeqElem( faces, index );
	
	while(r && (r->width<minsz || r->height<minsz || 
		(r->y+r->height+10)>frame->rows() || (r->x+r->width)>frame->cols() ||
		r->y<0 || r->x<0)){
		r = (CvRect*)cvGetSeqElem( faces, --index);
	}

	if( r==NULL )
		return false;

	*x = r->x;
	*y = r->y;
	*w = r->width;
	*h = r->height+10;

	return true;
}


vector<Matrixu> loadFromMov(const char *filepath)
{
    //filepath should be something like /home/sid/Downloads/a1.mov;
	vector<Matrixu> res;
	cv::Mat frame;
    Matrixu frameu;

	cv::VideoCapture cap(filepath);
	while(cap.read(frame)) {
        IplImage img = frame;
        frameu.Resize(img.height, img.width, 1);

        static IplImage *img2;
		if( img2 == NULL ) img2 = cvCreateImage( cvSize(frameu._cols, frameu._rows), IPL_DEPTH_8U, 1 );
		cvCvtColor( &img, img2, CV_RGB2GRAY );
		img2->origin = 0;
		frameu.GrayIplImage2Matrix(img2);
		res.push_back(frameu);
	}
	cap.release();

	return res;
}


float distCenters(Tracker *t0, Tracker *t1) {
	float centerX0 = (t0->_x + t0->_w/2.0)*t0->_scale;
	float centerY0 = (t0->_y + t0->_h/2.0)*t0->_scale;
	float centerX1 = (t1->_x + t1->_w/2.0)*t1->_scale;
	float centerY1 = (t1->_y + t1->_h/2.0)*t1->_scale;
	return (centerX0 - centerX1)*(centerX0 - centerX1) + (centerY0 - centerY1)*(centerY0 - centerY1);
}


bool trackersAgree(Tracker *tr[], int numTrackers) {
	if (numTrackers < 3)
		return true;

	float distThresh = (0.8*tr[0]->_w)*(0.8*tr[0]->_w);
	float dist01 = distCenters(tr[0], tr[1]);
	float dist02 = distCenters(tr[0], tr[2]);
	float dist12 = distCenters(tr[1], tr[2]);
	if (dist01 > distThresh || dist02 > distThresh || dist12 > distThresh) {
		if (dist01 < dist02) {
			if (dist01 < dist12) {
				float centerX = ((tr[0]->_x*tr[0]->_scale)+(tr[1]->_x*tr[1]->_scale))/2.0;
				float centerY = ((tr[0]->_y*tr[0]->_scale)+(tr[1]->_y*tr[1]->_scale))/2.0;
				tr[2]->_x = centerX/tr[2]->_scale;
				tr[2]->_y = centerY/tr[2]->_scale;
			}
			else {
				float centerX = ((tr[1]->_x*tr[1]->_scale)+(tr[2]->_x*tr[2]->_scale))/2.0;
				float centerY = ((tr[1]->_y*tr[1]->_scale)+(tr[2]->_y*tr[2]->_scale))/2.0;
				tr[0]->_x = centerX/tr[0]->_scale;
				tr[0]->_y = centerY/tr[0]->_scale;
			}
		}
		else {
			if (dist02 < dist12) {
				float centerX = ((tr[0]->_x*tr[0]->_scale)+(tr[2]->_x*tr[2]->_scale))/2.0;
				float centerY = ((tr[0]->_y*tr[0]->_scale)+(tr[2]->_y*tr[2]->_scale))/2.0;
				tr[1]->_x = centerX/tr[1]->_scale;
				tr[1]->_y = centerY/tr[1]->_scale;
			}
			else {
				float centerX = ((tr[1]->_x*tr[1]->_scale)+(tr[2]->_x*tr[2]->_scale))/2.0;
				float centerY = ((tr[1]->_y*tr[1]->_scale)+(tr[2]->_y*tr[2]->_scale))/2.0;
				tr[0]->_x = centerX/tr[0]->_scale;
				tr[0]->_y = centerY/tr[0]->_scale;
			}
		}
		return false;
	}
	else {
		return true;
	}
}


void initParams(TrackerParams *trparams[], FtrParams *ftrparams[], ClfParams *clfparams[],
	float scale, int x, int y, int w, int h, int searchRad, int numTrackers, 
	int numFeats, int numSelFeats) {
	
	for (int i=0; i<numTrackers; i++) {
		ftrparams[i]->_minNumRect = 2;
		ftrparams[i]->_maxNumRect = 6;
		ftrparams[i]->_width = w/scale;
		ftrparams[i]->_height = h/scale;

		clfparams[i]->_numSel = numSelFeats;
		clfparams[i]->_numFeat = numFeats;
		clfparams[i]->_ftrParams = ftrparams[i];

		trparams[i]->_init_negnumtrain	= 65;
		trparams[i]->_init_postrainrad	= 3.0f;
		trparams[i]->_srchwinsz = searchRad;
		trparams[i]->_negnumtrain = 65;
		trparams[i]->_posradtrain = 4.0f;
		trparams[i]->_initScale = scale;
		trparams[i]->_initX = x/scale;
		trparams[i]->_initY = y/scale;
		trparams[i]->_initW = w/scale;
		trparams[i]->_initH = h/scale;
	}

	if (numTrackers > 0) {
		trparams[0]->_boxcolor[0] = 25;
		trparams[0]->_boxcolor[1] = 25;
		trparams[0]->_boxcolor[2] = 204;
	}

	if (numTrackers > 1) {
		trparams[1]->_boxcolor[0] = 25;
		trparams[1]->_boxcolor[1] = 204;
		trparams[1]->_boxcolor[2] = 25;
	}
}


bool withinRad(vectori *xPos, vectori *yPos, int rad, int numFrames) {
	int xdist, ydist;
	int zdist = rad*rad;
	if (xPos->size() >= numFrames) {
		for (int i=xPos->size()-2; i >= 0 && i >= xPos->size()-numFrames; i--) {
			xdist = ((*xPos)[i+1] - (*xPos)[i])*((*xPos)[i+1] - (*xPos)[i]);
			ydist = ((*yPos)[i+1] - (*yPos)[i])*((*yPos)[i+1] - (*yPos)[i]);
			if (xdist + ydist > zdist)
				return false;
		}
	}
	return true;
}


void demo() {
	Tracker *tr[NUM_TRACKERS];
	TrackerParams *trparams[NUM_TRACKERS];
	ClfParams *clfparams[NUM_TRACKERS];
	FtrParams *ftrparams[NUM_TRACKERS];
	Matrixu f[NUM_TRACKERS];
	Matrixu frame, framedisp;
	int initX=100, initY=100, initW=20, initH=20;
	double ttime=0.0;

	// print usage
	printf("MILTRACK FACE DEMO\n===============================\n");
	printf("This demo uses the OpenCV face detector to initialize the tracker.\n");
	printf("Commands:\n");
	printf("\tPress 'q' to QUIT\n");
	printf("\tPress 'r' to RE-INITIALIZE\n\n");

	// Tracker and parameters
	for (int i=0; i<NUM_TRACKERS; i++) {
		ftrparams[i] = new FtrParams();
		clfparams[i] = new ClfParams();
		trparams[i] = new TrackerParams();
		tr[i] = new Tracker();
	}

	// Set up video
	CvCapture* capture = cvCaptureFromCAM( 0 );
	if( capture == NULL ){
		abortError(__LINE__,__FILE__,"Camera not found!");
		return;
	}

	// Register location
	do {
		Matrixu::CaptureImage(capture, frame, 0, 1);
		frame.display(1,1);
		cvWaitKey(2);
	}
	while( !initFace(&frame, &initX, &initY, &initW, &initH) );

	// Initialize location on first frame
	initParams(trparams, ftrparams, clfparams, 1.0, initX, initY, initW, initH, 25, NUM_TRACKERS, NUM_FEATS, NUM_SEL_FEATS);
	for (int t=0; t<NUM_TRACKERS; t++) {
		tr[t]->init(&frame, trparams[t], clfparams[t]);
	}

	// Track
	GET_TIME_VAL(0);
	for (int cnt = 0; 1; cnt++) {
		Matrixu::CaptureImage(capture, frame, 0, 1);

		// Update location
		for (int t=0; t<NUM_TRACKERS; t++) {
			f[t] = frame.imResize(frame._rows/tr[t]->_scale, frame._cols/tr[t]->_scale);
			f[t].initII();
			tr[t]->update_location(&f[t]);  // grab tracker confidence
		}

		// Check if all trackers agree
		if (trackersAgree(tr, NUM_TRACKERS)) {
			for (int t=0; t<NUM_TRACKERS; t++)
				tr[t]->update_classifier(&f[t]);
		}

		for (int t=0; t<NUM_TRACKERS; t++) {
			f[t].FreeII();
		}

		// Draw locations
		float cX = 0;
		float cY = 0;
		frame.conv2RGB(framedisp);
		framedisp._keepIpl=true;
		for (int t=0; t<NUM_TRACKERS; t++) {
			cX += (tr[t]->_x + tr[t]->_w/2)*tr[t]->_scale;
			cY += (tr[t]->_y + tr[t]->_h/2)*tr[t]->_scale;
			framedisp.drawRect(tr[t]->_w*tr[t]->_scale, tr[t]->_h*tr[t]->_scale, tr[t]->_x*tr[t]->_scale, tr[t]->_y*tr[t]->_scale,
				1, 0, trparams[t]->_lineWidth, trparams[t]->_boxcolor[0], trparams[t]->_boxcolor[1], trparams[t]->_boxcolor[2] );
		}
		cX = cX/NUM_TRACKERS;
		cY = cY/NUM_TRACKERS;

		framedisp.display(1,1);
		framedisp._keepIpl=false;
		framedisp.freeIpl();
		char q=cvWaitKey(2);

		GET_TIME_VAL(1);
		ttime = (ELAPSED_TIME_MS(1,0)/1000.0);
		fprintf(stderr,"%sframes:%d sec:%f FPS:%f", ERASELINE, cnt, ttime, ((double)cnt)/ttime);
	
		// quit
		if( q == 'q' )
			break;
	}

	// clean up
	cvReleaseCapture( &capture );
	printf("\n");
}


int	main(int argc, char * argv[]) {
	facecascade = NULL;
	demo();
}


