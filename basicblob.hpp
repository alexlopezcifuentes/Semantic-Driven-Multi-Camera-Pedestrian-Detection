//Video Analysis Techniques for Surveillance (VATS)
//class description
/**
 * \class BasicBlob
 * \brief Class to describe a basic blob
 *
 * 
 */

#ifndef BASICBLOB_H_INCLUDE
#define BASICBLOB_H_INCLUDE

#include "opencv2/opencv.hpp"
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)

// Maximun number of char in the blob's format
const int MAX_FORMAT = 1024;

/// Type of labels for blobs
typedef enum {	
	UNKNOWN=0, 
	PERSON=1, 
	GROUP=2, 
	CAR=3, 
	OBJECT=4
} CLASS;

struct cvBlob {
    int PID = 200;          /*       Pedestrian   ID              */
    int DetecID;            /*       Detection   ID               */
    Rect Blob;              /*       Coordinates of the blob      */
    float Score;            /*       Blob score                   */
    int OriginalCamera;     /*       Original blob camera         */
    Point2f FloorPoint;     /*       Projection in the 3D         */
    float Heigth3D;         /*       3D Heigth in the World       */
    bool Fix = 0;           /*       Bool to fix BB to the mask   */
};

//inline cvBlob initBlob(int id, Rect Blob, float score, int OriginalCamera, Point2f FloorPoint)
//{
    //cvBlob B = { id, Blob, score, OriginalCamera, FloorPoint};
    //return B;
//}
#endif
