#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#include <QObject>
#include "barrier.h"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <fstream>
#include <string>
#include "camerastream.h"
#include "peopledetector.h"
#include "basicblob.hpp"

using namespace cv;
using namespace std;

class CameraWorker : public QObject
{
    Q_OBJECT
public:
    CameraWorker(CameraStream Camera, Barrier barrier, String ParamDatasetName, String ParamPDDetector, bool ParamSemanticFitlering, bool ParamMultiCamera, int NumCams, String Mode);
    ~CameraWorker();

    // Dataset Name
    String DatasetName;

    // Operation mode
    String Mode;

    // Number of Analysed Cameras in the execution
    int NumCams;

    // Camera Class
    CameraStream Camera;
    // People detector class
    PeopleDetector PeopleDetec;

    // Display Widget Variables
    int WidgetWidth, WidgetHeight;

    // UI Variables
    String CBOption;
    // Representation Method
    String RepresentationOption;
    // Pedestrian Filtering
    bool MultiCameraFiltering;
    bool SemanticFiltering;

    // Cenital Frame
    Mat CenitalPlane, ImageWarping;

    // Txt file to extract and save selected views
    ofstream SelectedViewsFile;

    // Final Pedestrian Projected Bounding Boxes from the other cameras
    volatile bool HapFinished = 0;
    vector<Point2f> ProjCenterPoints, ProjLeftPoints, ProjRightPoints;
    vector<double> ProjScores;

    // NMS Checking
    volatile bool NMSFinished = 0;

    // Number of clusters for K-Means
    int NMaxClusters, NMinClusters;

    void processVideo();

signals:
    // frame and index of label which frame will be displayed
    void frameFinished(Mat frame, Mat CenitalPlane, int CameraNumber);
    void finished();
    void PedestrianDetectionFinished(int CameraNumber);
    void HAPAlgorithmSignal(int CameraNumber, String FrameNumber);

public slots:
    void preProcessVideo();

private:
    Barrier barrier;
    // Txt file to extract and save information
    ofstream VideoStatsFile;
};

#endif // CAMERAWORKER_H
