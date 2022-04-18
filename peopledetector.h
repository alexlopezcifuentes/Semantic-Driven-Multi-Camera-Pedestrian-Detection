#ifndef PEOPLEDETECTOR_H
#define PEOPLEDETECTOR_H

#include <QMainWindow>
#include <QObject>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <DPM/dpm.hpp>
#include <ACF/ACFDetector.h>
#include "ACF/ACFFeaturePyramid.h"
#include <ACF/Core/DetectionList.h>
#include <ACF/Core/NonMaximumSuppression.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "camerastream.h"
#include "basicblob.hpp"

using namespace std;
using namespace cv;
using namespace cv::dpm;

class PeopleDetector
{
public:
    PeopleDetector();
    ~PeopleDetector();

    // Main People Detection Function
    void MainPeopleDetection(CameraStream &Camera, String DatasetName, String FrameNumber, String CBOption, bool MultiCameraFiltering, bool SemanticFiltering);
    double Threshold;
    void ThresholdDetections(const vector<cvBlob> &srcBlobStructure,  vector<cvBlob> &resBlobStructure, double ThresholdMin, double ThresholdMax);

    // HOG People Detection
    HOGDescriptor HOG;
    void HOGPeopleDetection(CameraStream &Camera);

    // DPM People Detector
    Ptr<DPMDetector> DPMdetector = DPMDetector::create(vector<string>(1, "/home/alex/Desktop/IPCV-MasterThesis-master/MultithreadParameters/DPM/DPM_InriaPerson.xml"));
    void DPMPeopleDetection(CameraStream &Camera);
    void paintBoundingBoxes(Mat &ActualFrame, vector<cvBlob> BlobStructure, int CameraNumber, int Thickness);

    // ACF People Detector
    ACFDetector ACFdetector;
    void ACFPeopleDetection(CameraStream &Camera);

    // PSP-Net Detector
    void PSPNetScores(String DatasetName, int CameraNumber, String FrameNumber);

    // Fast-RCNN
    void decodeBlobFile(CameraStream &Camera, string FileName, string FrameNumber);
    void FastRCNNPeopleDetection(String VideoPath, CameraStream &Camera, string FrameNumber);

    // YOLOv3
    void YOLOPeopleDetection(String VideoPath, CameraStream &Camera, String DatasetName, string FrameNumber);

    // Gaussians creation
    void projectBlobs(vector<cvBlob> &BlobStructure, Mat Homography, Mat HomographyBetweenViews, int CameraNumber, bool Cilinder, String DatasetName);
    void meshgrid(Mat &X, Mat &Y, int rows, int cols);
    void gaussianFunction(Mat &Gaussian3C, Mat X, Mat Y, Point2f center, double score, int CameraNumber);

    // Semantic Pedestrian Constraining
    vector<int> SupressedIndices;
    void SemanticConstraining(vector<cvBlob> &BlobStructure, Mat &ActualFrame, String VideoPath);

    // Final Pedestrian Projected Bounding Boxes from the camera.
    vector<Point2f> ProjectedCenterPoints, ProjectedLeftPoints, ProjectedRightPoints;

    vector<Rect> AllPedestrianVector, AllPedestrianVectorNMS, AllPedestrianVectorNMS2, AllPedestrianVectorHAP;
    vector<double> AllPedestrianVectorScore, AllPedestrianVectorScoreNMS, AllPedestrianVectorScoreNMS2;
    vector<cvBlob> BlobStructurePD, BlobStructurePDNMS, BlobStructureHAP, BlobStructureHAPNMS;

    // Low Score bounding boxes not used for HAP procedure
    vector<cvBlob> BlobStructureLowScores;

    // Statistical Data Usage
    void ExtractDataUsage(int CameraNumber, String FrameNumber, Mat Homography, Mat HomographyBetweenViews);
    ofstream StatisticalBlobFile;

    // Clustering Functions. K-Means Fucntion and Distance Function
    void KMeans(vector<Point2f> Points, vector<Point2f> &OutputPoints, vector<int> &Labels, int NMinClusters, int NMaxClusters);
    void DistanceClustering(vector<cvBlob> &BlobStructure, float Distance);
    float SilhoutteIndex(const vector<Point2f> &Points, const Mat &Labels, int NClusters);

    // Blob Saving
    ofstream BoundingBoxesFile;
    void blobSavingTXT(vector<cvBlob> BlobStructure, String FrameNumber, int CameraNumber);
};

#endif // PEOPLEDETECTOR_H
