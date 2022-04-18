#include "camerastream.h"
#include <string>
#include <QMutex>
#include <fstream>
#include <QThread>
#include <stdio.h>
#include <numeric>
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <list>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "basicblob.hpp"

CameraStream::CameraStream(){}
CameraStream::~CameraStream(){}

using namespace cv;
using namespace std;

void CameraStream::VideoOpenning(int NumCams)
{
    // Open the videofile to check if it exists
    cap.open(InputPath);
    if (!cap.isOpened()) {
        cout << "Could not open video file " << InputPath << endl;
        exit(EXIT_FAILURE);
    }

    // Extract information from VideoCapture
    Width = cap.get(CAP_PROP_FRAME_WIDTH);
    Height = cap.get(CAP_PROP_FRAME_HEIGHT);
    FrameRate = cap.get(CAP_PROP_FPS);
    FrameNumber = cap.get(CAP_PROP_FRAME_COUNT);

    // Extract the dataset path
    vector<size_t> characterLocations;
    for(size_t i =0; i < InputPath.size(); i++){
        if(InputPath[i] == '/')
            characterLocations.push_back(i);
    }

    size_t Pos = characterLocations[characterLocations.size() - 2];
    VideoPath = InputPath.substr(0, Pos);

    if(CameraPanning){
        // Load and save CameraViews to a vector
        for(int i = 1; i <= NViews; i++){
            Mat CameraFrame = imread(VideoPath + "/Homography Images/Camera " + to_string(CameraNumber) + "/View " + to_string(i) + ".jpg");

            // Check for invalid input
            if(! CameraFrame.data ){
                cout << "Could not open Camera Views files with the following path:" << endl;
                cout << (VideoPath + "/Homography Images/Camera " + to_string(CameraNumber) + "/View " + to_string(i) + ".jpg") << endl;
                exit(EXIT_FAILURE);
            }
            CameraViewsVector.push_back(CameraFrame);
        }
    }

    // Save projected semantic into vector
    for(int i = 1; i <= NumCams; i++){
        Mat SemProjectFrame = imread(VideoPath + "/Projected Semantic Frames/Sem" + to_string(i) + "Median.png");

        // Check if image is not loaded properly
        if(!SemProjectFrame.data){
            cout << "Could not open the temporally filtered maps with the following path:" << endl;
            cout << (VideoPath + "/Projected Semantic Frames/Sem" + to_string(i) + "Median.png") << endl;
        }
        else{
            ProjectedFullSemanticVector.push_back(SemProjectFrame);
        }
    }
}

void CameraStream::getActualSemFrame(string FrameNumber)
{
    int FrameNumber2 = atoi(FrameNumber.c_str()) - 1;
    string SemImagesPath;

    if (FrameNumber2 < 10){
        // Add 000 to the path string
        SemImagesPath = VideoPath + "/Semantic Images/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "000" + to_string(FrameNumber2) + ".png";
    }
    else if (FrameNumber2 < 100){
        // Add 00 to the path string
        SemImagesPath = VideoPath + "/Semantic Images/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "00" + to_string(FrameNumber2) + ".png";
    }
    else if (FrameNumber2 < 1000){
        // Add 0 to the path string
        SemImagesPath = VideoPath + "/Semantic Images/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "0" + to_string(FrameNumber2) + ".png";
    }
    else{
        SemImagesPath = VideoPath + "/Semantic Images/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + to_string(FrameNumber2) + ".png";
    }
    ActualSemFrame = imread(SemImagesPath);

    // Check for invalid input
    if(! ActualSemFrame.data ){
        cout << "Could not open the actual semantic frame with the following path:" << endl;
        cout << SemImagesPath << endl;
        exit(EXIT_FAILURE);
    }
}

void CameraStream::extractPDMask(Mat ActualSemFrame, string DatasetName)
{
    Mat SemanticImageGray;

    // Find pedestrian mask
    // - TERRACE DATASET (label 7)
    // - PETS 2012 DATASET (label 13)
    // - WILDTRACK DATASET (label 13)
    // - RLC DATASET (label 15)
    int PedestrianLabel;
    if (!DatasetName.compare("Terrace"))
        PedestrianLabel = 7;
    else if((DatasetName.find("PETS2012") != string::npos) || !DatasetName.compare("Wildtrack"))
        PedestrianLabel = 13;
    else if(!DatasetName.compare("RLC"))
        PedestrianLabel = 15;

    cvtColor(ActualSemFrame, SemanticImageGray , COLOR_BGR2GRAY);
    compare(SemanticImageGray, PedestrianLabel, PedestrianMask, CMP_EQ);
}

void CameraStream::computeHomography()
{
    if(CameraPanning){
        // Compute homogrpahy by relating points from image and world
        string MainPath = VideoPath + "/Homography Images";

        for (int CameraView = 1; CameraView <= NViews; CameraView++){
            vector<Point2f> pts_src, pts_dst;
            string XCoord, YCoord;

            // CAMERA FRAME POINTS
            string FileName = MainPath + "/Camera " + to_string(CameraNumber) + "/View " + to_string(CameraView) + "_PtsSrcFile.txt";
            ifstream input(FileName);

            if (!input) {
                // The file does not exists
                cout << "Problem with the following path:" << endl;
                cout << FileName << endl;
                cout << "The file that should contain homography points for Camera " + to_string(CameraNumber) + " Frame do not exist" << endl;
                exit(EXIT_FAILURE);
            }

            // Start decoding the file with src points
            while (input >> XCoord){
                input >> YCoord;
                Point2f pt;
                pt.x = atoi(XCoord.c_str());
                pt.y = atoi(YCoord.c_str());
                pts_src.push_back(pt);
            }

            // CENITAL FRAME POINTS
            FileName = MainPath + "/Camera " + to_string(CameraNumber) + "/View " + to_string(CameraView) + "_PtsDstFile.txt";
            ifstream input2(FileName);

            if (!input2) {
                // The file does not exists
                cout << "Problem with the following path:" << endl;
                cout << FileName << endl;
                cout << "The file that should contain homography points for Cenital Frame for camera " + to_string(CameraNumber) + " do not exist" << endl;
                exit(EXIT_FAILURE);
            }

            // Start decoding the file with dst points
            while (input2 >> XCoord){
                input2 >> YCoord;
                Point2f pt;
                pt.x = atoi(XCoord.c_str());
                pt.y = atoi(YCoord.c_str());
                pts_dst.push_back(pt);
            }

            if (pts_dst.size() != pts_src.size()){
                cout << "The number of homography points for Camera " + to_string(CameraNumber) + " is not the same in source and destiny" << endl;
                exit(EXIT_FAILURE);
            }

            // Calculate Homography and store it in the vector
            HomographyVector.push_back(findHomography(pts_src, pts_dst, LMEDS));

            // Check the homography by warping the view and drawing points //

            Mat View = imread(VideoPath + "/Homography Images/Camera " + to_string(CameraNumber) + "/View " + to_string(CameraView) + ".jpg");

            // Check for invalid input
            if(! View.data ){
                cout << "Could not open homography view files with the following path:" << endl;
                cout << (VideoPath + "/Homography Images/Camera " + to_string(CameraNumber) + "/View " + to_string(CameraView) + ".jpg") << endl;
                exit(EXIT_FAILURE);
            }

            Mat ImageWarping = Mat::zeros(986, 1606, CV_8UC1);
            Mat Homografia = HomographyVector[CameraView-1];
            warpPerspective(View, ImageWarping, Homografia, ImageWarping.size());

            // Apply Homography to vectors of Points to find the projection
            vector<Point2f> pts_src_projected;
            perspectiveTransform(pts_src, pts_src_projected, Homografia);

            for(int i = 0; i < pts_src.size(); i++){
                Point PuntoImagen = pts_src_projected[i];
                Point PuntoCenital = pts_dst[i];

                // Puntos de la imagen seleccionados proyectados
                circle(ImageWarping, PuntoImagen, 2, Scalar(255,0,0), 4);
                // Puntos de la imagen cenital seleccionados
                circle(ImageWarping, PuntoCenital, 4, Scalar(0,0,255), 4);
            }
            String ImageName = "/Users/alex/Desktop/Vistas Proyectadas/Camera " + to_string(CameraNumber) + "_Vista" + to_string(CameraView) + ".jpg";
            imwrite(ImageName, ImageWarping);
        }
    }
    else{
        // Read and decode homography matrix
        string FileName = VideoPath + "/Homography " + to_string(CameraNumber) + ".txt";
        ifstream input(FileName);

        string AuxString;
        vector<double> MatrixValues;
        Mat HomographyMatrix = Mat::zeros(3, 3, CV_64F);
        // Start decoding the file with homography matrix points
        while (input >> AuxString){
            MatrixValues.push_back(stod(AuxString.c_str()));
        }

        for(int row = 0; row < HomographyMatrix.rows; row++){
            for(int col = 0; col < HomographyMatrix.cols; col++){
                HomographyMatrix.at<double>(row,col) = MatrixValues[row*3 + col];
            }
        }
        HomographyVector.push_back(HomographyMatrix);
    }
}

void CameraStream::ViewSelection(vector<Mat> HomographyVector)
{
    // Compare Actual Frame with all the frames used to extract homographies with AKAZE
    // Extract number of correspondant view to index the homography vectors
    int NMatches;
    vector<Point2f> GoodMatchesPoints1, GoodMatchesPoints2;
    vector<Point2f> GoodMatchesPoints1Def, GoodMatchesPoints2Def;
    vector<vector<Point2f>> VectorGoodMatches1, VectorGoodMatches2;
    vector<int> VectorNMaches;

    for (int CameraView = 0; CameraView < NViews; CameraView++){
        GoodMatchesPoints1.clear();
        GoodMatchesPoints2.clear();

        Mat CameraViewImage = CameraViewsVector[CameraView];
        Mat ViewDescriptor = AKAZEDescriptorsVector[CameraView];
        vector<KeyPoint> ViewKeypoints = AKAZEKeyPointsVector[CameraView];

        Akaze(CameraViewImage, ViewKeypoints, ViewDescriptor, ActualFrame, NMatches, GoodMatchesPoints1, GoodMatchesPoints2, CameraView);

        VectorNMaches.push_back(NMatches);
        VectorGoodMatches1.push_back(GoodMatchesPoints1);
        VectorGoodMatches2.push_back(GoodMatchesPoints2);
    }

    // Sort NMatches vector
    vector<int> SortedNMatches;
    SortedNMatches = VectorNMaches;
    sort(SortedNMatches.begin(), SortedNMatches.end());

    // Extract the first maximum number of matches
    auto MaxNMatches = SortedNMatches.at(NViews-1);

    // Extract maximum positon
    SelectedView = find(VectorNMaches.begin(), VectorNMaches.end(), MaxNMatches) - VectorNMaches.begin();

    // Get the maximum view points for the homography
    GoodMatchesPoints1Def = VectorGoodMatches1[SelectedView];
    GoodMatchesPoints2Def = VectorGoodMatches2[SelectedView];

    if (GoodMatchesPoints1Def.size() > 4){
        // Number of match points between images when selecting homography is more than 4 so we can compute
        // an homography

        // Now that we know the nearest view with respect with the ActualFrame we have to
        // interpolate/trasnform the homography so it is more accurate
        // Convert the ActualFrame to the view perspective
        HomographyBetweenViews = findHomography(GoodMatchesPoints2Def, GoodMatchesPoints1Def, LMEDS);
    }
    Homography = HomographyVector[SelectedView];
}

void CameraStream::ViewSelectionFromTXT(String VideoPath, vector<Mat> HomographyVector, String FrameNumber, String DatasetName)
{
    String ViewsPath = VideoPath + "/SelectedViews" + to_string(CameraNumber) + ".txt";
    ifstream input(ViewsPath);

    if (!input) {
        // The file does not exists
        cout << "The file containing the FastRCNN blobs does not exist. Path: " << ViewsPath << endl;
        exit(EXIT_FAILURE);
    }

    // Auxiliary variables to store the information
    string AuxString;
    int Counter = 0;
    int LineCounter = 0;

    HomographyBetweenViews = Mat::zeros(3, 3, CV_64FC1);

    // Start decoding the file
    while (input >> AuxString){
        switch(Counter)
        {
        case 0:
            // Case for frame number
            if (LineCounter == atoi(FrameNumber.c_str())){
                // Convert ActualSemFrame with the computed homography to be similar to the semantic image from the view
                warpPerspective(ActualSemFrame, ActualSemFrame, HomographyBetweenViews, ActualSemFrame.size());
                return;
            }
            LineCounter++;
            Counter++;
            break;
        case 1:
            // Case for Selected View
            SelectedView = atoi(AuxString.c_str());
            Counter++;
            break;
        case 2:
            // Case for Homography(0,0)
            HomographyBetweenViews.at<double>(0,0) = atof(AuxString.c_str());
            Counter++;
            break;
        case 3:
            // Case for Homography(0,1)
            HomographyBetweenViews.at<double>(0,1) = atof(AuxString.c_str());
            Counter++;
            break;
        case 4:
            // Case for Homography(0,2)
            HomographyBetweenViews.at<double>(0,2) = atof(AuxString.c_str());
            Counter++;
            break;
        case 5:
            // Case for Homography(1,0)
            HomographyBetweenViews.at<double>(1,0) = atof(AuxString.c_str());
            Counter++;
            break;
        case 6:
            // Case for Homography(1,1)
            HomographyBetweenViews.at<double>(1,1) = atof(AuxString.c_str());
            Counter++;
            break;
        case 7:
            // Case for Homography(1,2)
            HomographyBetweenViews.at<double>(1,2) = atof(AuxString.c_str());
            Counter++;
            break;
        case 8:
            // Case for Homography(2,0)
            HomographyBetweenViews.at<double>(2,0) = atof(AuxString.c_str());
            Counter++;
            break;
        case 9:
            // Case for Homography(2,1)
            HomographyBetweenViews.at<double>(2,1) = atof(AuxString.c_str());
            Counter++;
            break;
        case 10:
            // Case for Homography(2,2)
            HomographyBetweenViews.at<double>(2,2) = atof(AuxString.c_str());
            Homography = HomographyVector[SelectedView];

            Counter++;

            // Restart the couter to read frame
            Counter = 0;
            break;
        }
    }
}

void CameraStream::saveWarpImages(Mat ActualFrame, String FrameNumber, Mat ImageWarping)
{
    // Extract image warping
    warpPerspective(ActualFrame, ImageWarping, HomographyBetweenViews, ImageWarping.size());
    warpPerspective(ImageWarping, ImageWarping, Homography, ImageWarping.size());

    String ImageName = VideoPath + "/Wrapped Images/Camera " + to_string(CameraNumber) + "/Frame" + FrameNumber + ".png";
    imwrite(ImageName, ImageWarping);
}

void CameraStream::SemanticCommonPoints(String DatasetName)
{
    // Check if Temporally Filtered Maps were loaded
    if(ProjectedFullSemanticVector.empty()){
        CommonSemanticAllCameras = Mat::zeros(5000, 5000, CV_8UC3);
    }
    else{
        // Extract common points between cameras with the offline projected semantic images
        int Rows = ProjectedFullSemanticVector[0].rows;
        int Cols = ProjectedFullSemanticVector[0].cols;
        int GrayLevel1, GrayLevel2, GrayLevel3, GrayLevel4, GrayLevel5, GrayLevel6, GrayLevel7;

        // Common semantic between the four cameras
        CommonSemanticAllCameras = Mat::zeros(Rows, Cols, ProjectedFullSemanticVector[0].type());
        for (int i = 0; i < Rows; i++){
            for (int j = 0; j < Cols; j++){
                // Extract Semantic Label for pixel. Number of labels depending on the dataset.
                GrayLevel1 = ProjectedFullSemanticVector[0].at<Vec3b>(i,j)[0];
                GrayLevel2 = ProjectedFullSemanticVector[1].at<Vec3b>(i,j)[0];
                if(!DatasetName.compare("Terrace") || !DatasetName.compare("PETS2012_S2_L1")){
                    GrayLevel3 = ProjectedFullSemanticVector[2].at<Vec3b>(i,j)[0];
                    GrayLevel4 = ProjectedFullSemanticVector[3].at<Vec3b>(i,j)[0];
                }
                else if((DatasetName.find("EPS_Hall") != string::npos) || !DatasetName.compare("RLC"))
                    GrayLevel3 = ProjectedFullSemanticVector[2].at<Vec3b>(i,j)[0];
                else if(!DatasetName.compare("Wildtrack")){
                    GrayLevel3 = ProjectedFullSemanticVector[2].at<Vec3b>(i,j)[0];
                    GrayLevel4 = ProjectedFullSemanticVector[3].at<Vec3b>(i,j)[0];
                    GrayLevel5 = ProjectedFullSemanticVector[4].at<Vec3b>(i,j)[0];
                    GrayLevel6 = ProjectedFullSemanticVector[5].at<Vec3b>(i,j)[0];
                    GrayLevel7 = ProjectedFullSemanticVector[6].at<Vec3b>(i,j)[0];
                }

                // Check if all the labels are equal. Depending on the dataset
                if (!DatasetName.compare("Terrace")){
                    if((GrayLevel1 == 3) || (GrayLevel2 == 3) || (GrayLevel3 == 3) || (GrayLevel4 == 3)){
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[0] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[1] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[2] = 3;
                    }
                }
                else if(!DatasetName.compare("PETS2012_S2_L1")){
                    if((GrayLevel1 == 10) || (GrayLevel2 == 10) || (GrayLevel3 == 10) || (GrayLevel4 == 10) ||
                            (GrayLevel1 == 7) || (GrayLevel2 == 7) || (GrayLevel3 == 7) || (GrayLevel4 == 7)){
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[0] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[1] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[2] = 3;
                    }
                }
                else if(!DatasetName.compare("PETS2012_CC")){
                    if((GrayLevel1 == 10) || (GrayLevel2 == 10) || (GrayLevel1 == 7) || (GrayLevel2 == 7) ||
                            (GrayLevel2 == 53)){
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[0] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[1] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[2] = 3;
                    }
                }
                else if((DatasetName.find("EPS_Hall") != string::npos) || !DatasetName.compare("RLC")){
                    if((GrayLevel1 == 4) || (GrayLevel2 == 4) || (GrayLevel3 == 4)){
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[0] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[1] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[2] = 3;
                    }
                }
                else if(!DatasetName.compare("Wildtrack")){
                    if((GrayLevel1 == 12) || (GrayLevel2 == 12) || (GrayLevel3 == 12) || (GrayLevel4 == 12)
                            || (GrayLevel5 == 12) || (GrayLevel6 == 12) || (GrayLevel7 == 12)){
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[0] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[1] = 3;
                        CommonSemanticAllCameras.at<Vec3b>(i,j)[2] = 3;
                    }
                }
            }
        }
    }
    // Save image Results
    String ImageName = VideoPath + "/Projected Semantic Frames/CommonSemantic.png";
    Mat Aux = CommonSemanticAllCameras*20;
    imwrite(ImageName, Aux);
}

void CameraStream::ProjectSemanticPoints(Mat CenitalPlane, String FrameNumber)
{
    // Project all semantic image
    warpPerspective(ActualSemFrame, CenitalPlane, HomographyBetweenViews, CenitalPlane.size());
    warpPerspective(CenitalPlane, CenitalPlane, Homography, CenitalPlane.size());

    String ImageName = VideoPath + "/Projected Semantic Frames/Projected Frames " + to_string(CameraNumber) + "/Frame" + FrameNumber + ".png";
    imwrite(ImageName, CenitalPlane);

    /*
    Mat FloorMask;
    Mat SemanticImageGray;
    vector<Point> FloorPoints;
    vector<Point2f> ProjectedFloor;

    // Find floor mask (label 3) and extract floor coordinates (Point format)
    cvtColor(ActualSemFrame, SemanticImageGray , CV_BGR2GRAY);
    // Floor label
    // - PETS 10,7
    // - Terrace 3
    // - EPS & RLC 4
    compare(SemanticImageGray, 4, FloorMask, CMP_EQ);
    findNonZero(FloorMask == 255, FloorPoints);

    // Convert from Point to Point2f floor coordinates. Auxiliar vector.
    vector<Point2f> FloorPoints2(FloorPoints.begin(), FloorPoints.end());

    if(FloorPoints2.empty())
        return;

    // Apply Homography to vector of Points2f to find the projection of the floor
    perspectiveTransform(FloorPoints2, ProjectedFloor, HomographyBetweenViews);
    perspectiveTransform(ProjectedFloor, ProjectedFloor, Homography);

    // Fill the global vector
    ProjectedFloorVector = ProjectedFloor;
    // Extract number of Floor Points
    NumberFloorPoints = static_cast<int>(ProjectedFloorVector.size());

    // Extract projected floor mask
    Mat ProjectedFloorMask = Mat::zeros(CenitalPlane.rows, CenitalPlane.cols, CV_8U);

    for (int i = 0 ; i < NumberFloorPoints ; i++){
        Point punto = ProjectedFloorVector[i];
        if ((punto.y > 0 && punto.y < ProjectedFloorMask.rows) && (punto.x > 0 && punto.x < ProjectedFloorMask.cols)){
            ProjectedFloorMask.at<uchar>(punto.y, punto.x) = 255;
        }
    }

    // Create the mask that will be filled
    Mat FilledFloorMask = Mat::zeros(CenitalPlane.rows, CenitalPlane.cols, CV_8U);

    // Dilatation and Erosion kernels to fill the mask
    Mat kernel_di = getStructuringElement(MORPH_ELLIPSE, Size(9, 9), Point(-1, -1));
    Mat kernel_ero = getStructuringElement(MORPH_ELLIPSE, Size(9, 9), Point(-1, -1));

    // First dilate to fill then erode to keep the original contour
    dilate(ProjectedFloorMask, FilledFloorMask, kernel_di, Point(-1, -1));
    erode(FilledFloorMask, FilledFloorMask, kernel_ero, Point(-1, -1));

    // Erase previous points
    FloorPoints.clear();
    ProjectedFloor.clear();
    ProjectedFloorVector.clear();

    // Find floor mask and extract floor coordinates (Point format)
    findNonZero(FilledFloorMask == 255, FloorPoints);

    // Convert from Point to Point2f floor coordinates. Auxiliar vector.
    vector<Point2f> FloorPoints3(FloorPoints.begin(), FloorPoints.end());

    // Fill the global vector
    ProjectedFloorVector = FloorPoints3;
    // Extract number of Floor Points
    NumberFloorPoints = static_cast<int>(ProjectedFloorVector.size());
    */
}

void CameraStream::ProjectCommonSemantic(Mat &Frame)
{
    Vec3b Color;
    vector<Point> CommonPoints;
    vector<Point2f> ReProjectedCommonPoints;
    Mat overlay;
    double alpha = 0.2;

    // Select color depending on the CameraNumber
    if (CameraNumber == 1){
        Color.val[0] = 0;
        Color.val[1] = 255;
        Color.val[2] = 0;
    }
    if (CameraNumber == 2){
        Color.val[0] = 255;
        Color.val[1] = 0;
        Color.val[2] = 0;
    }
    if (CameraNumber == 3){
        Color.val[0] = 0;
        Color.val[1] = 0;
        Color.val[2] = 255;
    }

//    // COMMON AREA BETWEEN ALL THE CAMERAS
//    Mat CommonImage1 = imread(VideoPath + "/Projected Semantic Frames/CommonSemantic.png", IMREAD_GRAYSCALE);
//    compare(CommonImage1, 60, CommonImage1, CMP_EQ);

    // Authors ROI
    Mat CommonImage1 = imread(VideoPath + "/Projected Semantic Frames/CommonSemanticAuthors.png", IMREAD_GRAYSCALE);
    compare(CommonImage1, 60, CommonImage1, CMP_EQ);

    warpPerspective(CommonImage1, overlay, Homography.inv(DECOMP_LU), Frame.size());
    warpPerspective(overlay, overlay, HomographyBetweenViews.inv(DECOMP_LU), Frame.size());

    cvtColor(overlay, overlay, cv::COLOR_GRAY2BGR);

    Mat ch1, ch2, ch3; // declare three matrices
    // "channels" is a vector of 3 Mat arrays:
    vector<Mat> channels(3);
    // split img:
    split(overlay, channels);
    // get the channels (follow BGR order in OpenCV)
    ch1 = channels[0];
    ch2 = channels[1];
    ch3 = channels[2];
    // modify channel// then merge
    ch1.setTo(0);
    ch3.setTo(0);

    merge(channels, overlay);

    // Create the convex poligon from array of Point and add transparency to the final image
    addWeighted(overlay, alpha, Frame, 1, 0, Frame);
}

void CameraStream::AkazePointsForViewImages()
{
    for (int CameraView = 0; CameraView < NViews; CameraView++){
        Mat ViewImage = CameraViewsVector[CameraView];
        vector<KeyPoint> kpts1;
        Mat desc1;

        // Compute AKAZE points for the selected view image
        akazeDescriptor = AKAZE::create();
        akazeDescriptor->detectAndCompute(ViewImage, noArray(), kpts1, desc1);

        // Save descriptors for the view image
        AKAZEDescriptorsVector.push_back(desc1);
        AKAZEKeyPointsVector.push_back(kpts1);
    }
}

void CameraStream::Akaze(Mat Image1, vector<KeyPoint> kpts1, Mat desc1, Mat Image2, int &NMatches, vector<Point2f> &GoodMatchesPoints1, vector<Point2f> &GoodMatchesPoints2, int CameraView)
{
    vector<KeyPoint> kpts2;
    Mat desc2;

    akazeDescriptor->setNOctaves(2);
    akazeDescriptor->setNOctaveLayers(1);
    akazeDescriptor->detectAndCompute(Image2, noArray(), kpts2, desc2);

    //  ------------------  //
    // BRUTE FORCE MATCHER  //
    //  ------------------  //
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 5);

    vector<DMatch> good_matches;
    for (size_t i = 0; i < nn_matches.size(); ++i) {
        const float ratio = 0.8; // As in Lowe's paper; can be tuned
        if (nn_matches[i][0].distance < ratio * nn_matches[i][1].distance) {
            good_matches.push_back(nn_matches[i][0]);
        }
    }

    //  ------------------  //
    //  MATCHES COORDINATES //
    //  ------------------  //
    NMatches = good_matches.size();

    // Extract point coordinates from the good matches between ActualFrame and the correspondant ViewFrame
    for (size_t j = 0; j < NMatches; ++j) {
        DMatch Match = good_matches[j];

        int Index1 = Match.queryIdx;
        int Index2 = Match.trainIdx;

        Point2f Point1 = kpts1[Index1].pt;
        Point2f Point2 = kpts2[Index2].pt;

        GoodMatchesPoints1.push_back(Point1);
        GoodMatchesPoints2.push_back(Point2);
    }

    Mat res;
    drawMatches(Image1, kpts1, Image2, kpts2, good_matches, res);
    String ImageName = "/Users/alex/Desktop/AKAZE/Camera " + to_string(CameraNumber) + "/View " + to_string(CameraView) + ".png";
    imwrite(ImageName, res);
}

void CameraStream::extractFGBlobs(Mat fgmask, string CBOption)
{
    // Required variables for connected component analysis
    Point pt;
    Rect RectangleOutput;
    Scalar NewValue = 254;
    Scalar MaxMin = 1;
    int Flag = 8;

    // Clear blob list (to fill with this function)
    vector<Rect> bloblist;
    vector<Rect> bloblist_joined;

    bloblist.clear();
    bloblist_joined.clear();

    // Connected component analysis
    // Scan the FG mask to find blob pixels
    for (int x = 0; x < fgmask.rows; x++){
        for (int y = 0; y < fgmask.cols; y++){

            // Extract connected component (blob)
            // We only analyze foreground pixels
            if ((fgmask.at<uchar>(x,y)) == 255.0) {
                pt.x = y;
                pt.y = x;

                // We use the function to obtain the blob.
                floodFill(fgmask, pt, NewValue, &RectangleOutput, MaxMin, MaxMin, Flag);

                // Increse Rectangle size if method is not Semantic
                if(CBOption.compare("PSPNet")){
                    int PixelIncrease = 25;
                    RectangleOutput.x -= PixelIncrease;
                    RectangleOutput.y -= PixelIncrease;
                    RectangleOutput.width += PixelIncrease * 4;
                    RectangleOutput.height += PixelIncrease * 4;
                }

                // Include blob in 'bloblist'
                bloblist.push_back(RectangleOutput);
            }
        }
    }

    // Iterate through nms until the number of blob do not change
    vector<Rect> resRectsAux1, resRectsAux2;
    resRectsAux1 = bloblist;

    int SizeRectsAux1 = resRectsAux1.size();
    int SizeRectsAux2 = resRectsAux2.size();

    while(SizeRectsAux1 != SizeRectsAux2){
        SizeRectsAux2 = resRectsAux2.size();
        non_max_suppresion(resRectsAux1, resRectsAux2);
        resRectsAux1 = resRectsAux2;
        SizeRectsAux1 = resRectsAux1.size();
    }

    bloblist_joined = resRectsAux2;

    vector<Rect> bloblist_joined_filtered;
    // Suppress small boxes
    for (size_t i = 0; i < bloblist_joined.size(); i++) {
        Rect rect = bloblist_joined[i];
        //if (rect.area() > 5000)
        bloblist_joined_filtered.push_back(rect);
    }
    FGBlobs = bloblist_joined_filtered;
    return;
}

void CameraStream::CheckSemanticInBlobs(vector<cvBlob> &srcBlobStructure, Mat ForegroundMask)
{
    if(srcBlobStructure.empty())
        return;

    for(size_t n = 0; n < srcBlobStructure.size(); n++){
        cvBlob SBB = srcBlobStructure[n];
        // If detection is obtained by multi-camera fusion
        if(SBB.OriginalCamera != 15){
            Mat CroppedMask = ForegroundMask(SBB.Blob);
            float Sum = sum(CroppedMask)[0];
            if(Sum <= 0){
                srcBlobStructure.erase(srcBlobStructure.begin() + n);
                n--;
            }
        }
        // If detection is obtained from only one camera
        else{
            Mat CroppedMask = ForegroundMask(SBB.Blob);
            float Sum = sum(CroppedMask)[0];
            if(Sum <= 0){
                srcBlobStructure[n].Score *= 0.75;
            }
        }
    }
    return;
}

void CameraStream::non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects)
{
    resRects.clear();
    const size_t size = srcRects.size();
    if (srcRects.empty())
        return;

    float thresh = 0.2;
    int neighbors = 0;

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i) {
        idxs.insert(pair<int, size_t>(srcRects[i].br().y, i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0) {
        // grab the last rectangle
        auto lastElem = --end(idxs);
        const Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;
        idxs.erase(lastElem);

        for (auto pos = begin(idxs); pos != end(idxs);){
            // grab the current rectangle
            const Rect& rect2 = srcRects[pos->second];

            float intArea = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh) {
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
        {
            resRects.push_back(rect1);
        }
    }
}

void CameraStream::non_max_suppresion_scores(String CBOption, const vector<Rect> &srcRects, const vector<double> &srcScores, vector<Rect> &resRects, vector<double> &resScores)
{
    resRects.clear();
    resScores.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    if (srcRects.size() != srcScores.size()) {
        cout << "NMS  ERROR. Sizes of detection vector and score vector are not the same." << endl;
        exit(EXIT_FAILURE);
    }

    float minScoresSum;
    if (!CBOption.compare("HOG")){
        minScoresSum = 0.5104;
    }
    else if(!CBOption.compare("FastRCNN")){
        minScoresSum = 0.3;
    }
    else if(!CBOption.compare("DPM")){
        minScoresSum = -3.6239;
    }
    else if(!CBOption.compare("ACF")){
        minScoresSum = -7.3918;
    }
    else if(!CBOption.compare("PSPNet")){
        minScoresSum = 0.5924;
    }

    float thresh = 0.2;
    int neighbors = 0;

    // Sort the bounding boxes by the detection score. Higher score first
    multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i){
        idxs.insert(pair<float, size_t>(srcScores[i], i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0){
        // grab the last rectangle
        auto lastElem = --end(idxs);
        const Rect& rect1 = srcRects[lastElem->second];
        const double& score1 = srcScores[lastElem->second];

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = begin(idxs); pos != end(idxs); ){
            // grab the current rectangle
            const Rect& rect2 = srcRects[pos->second];

            float intArea = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh){
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else{
                ++pos;
            }
        }
        if (neigborsCount >= neighbors && scoresSum >= minScoresSum){
            // Save bounding box and score if conditions are fulfilled
            resRects.push_back(rect1);
            resScores.push_back(score1);
        }
    }
}

void CameraStream::selectBB(const vector<cvBlob> &srcStructure, cvBlob &resBlob, Mat ForegroundMask)
{
    double FunctionValue = -1;
    for(int BBIndex = 0; BBIndex < srcStructure.size(); BBIndex++){
        Rect AuxiliarBB = srcStructure[BBIndex].Blob;
        Mat SelectedForeground = ForegroundMask(AuxiliarBB);

        double Ratio = 0;
        for(int  row = 0; row < SelectedForeground.rows; row++){
            int col = SelectedForeground.cols/2;
            Ratio = Ratio + SelectedForeground.at<uchar>(row,col);
        }

        if(Ratio > FunctionValue){
            FunctionValue = Ratio;
            resBlob = srcStructure[BBIndex];
        }
    }
}
