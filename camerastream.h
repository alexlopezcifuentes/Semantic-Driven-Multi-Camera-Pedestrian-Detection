#ifndef CAMERASTREAM_H
#define CAMERASTREAM_H

#include <QObject>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "basicblob.hpp"

using namespace std;
using namespace cv;

class CameraStream
{
public:
    CameraStream();
    ~CameraStream();

    // Video Variables
    string GlobalPath;
    string InputPath;
    string VideoPath;
    VideoCapture cap;
    void VideoOpenning(int NumCams);
    int Width, Height, FrameRate, FrameNumber;
    int CameraNumber;

    // Mat to store the frame to process
    Mat ActualFrame;
    Mat ActualSemFrame;
    void getActualSemFrame(string FrameNumber);

    // Mixture Of Gaussians Background Substractor
    Mat BackgroundMask;
    bool EmptyBackground;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    // Homography and Image Wrapping
    bool CameraPanning = 0;
    int NViews = 9;
    Mat HomographyBetweenViews;
    vector<Mat> CameraViewsVector;
    vector<Mat> HomographyVector;
    Mat Homography;
    int SelectedView;
    void computeHomography();
    void ViewSelection(vector<Mat> HomographyVector);
    void ViewSelectionFromTXT(String VideoPath, vector<Mat> HomographyVector, String FrameNumber, String DatasetName);
    void saveWarpImages(Mat ActualFrame, String FrameNumber, Mat ImageWarping);

    // Semantic Projection
    vector<Mat> ProjectedFullSemanticVector;
    Mat CommonSemantic12, CommonSemantic23, CommonSemantic13;
    Mat CommonSemanticAllCameras;
    //vector<Point2f> ProjectedFloorVector;
    //int NumberFloorPoints;
    void ProjectSemanticPoints(Mat CenitalPlane, String FrameNumber);

    // Induced Plane Homography
    void SemanticCommonPoints(String DatasetName);
    void ProjectCommonSemantic(Mat &Frame);

    // Pedestrian mask, blobs and images
    Mat PedestrianMask;
    vector<Rect> FGBlobs;
    vector<Mat> FGImages;
    void extractPDMask(Mat ActualSemFrame, string DatasetName);
    void extractFGBlobs(Mat fgmask, string CBOption);
    void CheckSemanticInBlobs(vector<cvBlob> &srcBlobStructure, Mat ForegroundMask);
    void non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects);
    void non_max_suppresion_scores(String CBOption, const vector<Rect> &srcRects, const vector<double> &srcScores, vector<Rect> &resRects, vector<double> &resScores);
    void selectBB(const vector<cvBlob> &srcStructure, cvBlob &resBlob, Mat ForegroundMask);

    // AKAZE
    Ptr<AKAZE> akazeDescriptor;
    vector<vector<KeyPoint>> AKAZEKeyPointsVector;
    vector<Mat> AKAZEDescriptorsVector;
    void AkazePointsForViewImages();
    void Akaze(Mat Image1, vector<KeyPoint> kpts1, Mat desc1, Mat Image2, int &NMatches, vector<Point2f> &GoodMatchesPoints1, vector<Point2f> &GoodMatchesPoints2, int CameraView);

    // HOG Vectors
    vector<Rect> HOGBoundingBoxes;
    vector<Rect> HOGBoundingBoxesNMS;
    vector<double> HOGScores;

    // Fast RCNN Vectors
    vector<Rect> RCNNBoundingBoxes;
    vector<double> RCNNScores;

    // DPM Vectors
    vector<Rect> DPMBoundingBoxes;
    vector<double> DPMScores;

    // ACF Vectors
    vector<Rect> ACFBoundingBoxes;
    vector<double> ACFScores;
};

#endif // CAMERASTREAM_H
