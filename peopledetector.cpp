#include "peopledetector.h"
#include "camerastream.h"
#include <string>
#include <numeric>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "basicblob.hpp"

PeopleDetector::PeopleDetector(){}
PeopleDetector::~PeopleDetector(){}

void PeopleDetector::MainPeopleDetection(CameraStream &Camera, String DatasetName, String FrameNumber, String CBOption,
                                         bool MultiCameraFiltering, bool SemanticFiltering)
{
    BlobStructurePD.clear();
    BlobStructurePDNMS.clear();
    BlobStructureLowScores.clear();

    if (!CBOption.compare("HOG")){
        // HOG Detector
        HOGPeopleDetection(Camera);
    }
    else if(!CBOption.compare("FastRCNN")){
        // FastRCNN Detector
        FastRCNNPeopleDetection(Camera.VideoPath, Camera, FrameNumber);
    }
    else if(!CBOption.compare("YOLOv3")){
        // YOLOv3 Detector
        YOLOPeopleDetection(Camera.VideoPath, Camera, DatasetName, FrameNumber);
    }
    else if(!CBOption.compare("DPM")){
        // DPM Detector
        DPMPeopleDetection(Camera);
    }
    else if(!CBOption.compare("ACF")){
        // ACF Detector
        ACFPeopleDetection(Camera);
    }
    else if(!CBOption.compare("PSPNet")){
        // People detection using labels from semantic information.
        AllPedestrianVector = Camera.FGBlobs;
        PSPNetScores(DatasetName, Camera.CameraNumber, FrameNumber);
    }
    else if(!CBOption.compare("None")){
        return;
    }

    if(!MultiCameraFiltering && !SemanticFiltering){
        // All detections are taken into account for the raw detector
        BlobStructurePDNMS = BlobStructurePD;
    }
    else{
        // Project blobs to the ground
        projectBlobs(BlobStructurePD, Camera.Homography, Camera.HomographyBetweenViews, Camera.CameraNumber, 1, DatasetName);

        // Semantic filtering for all the set of detections
        if(SemanticFiltering){
            SemanticConstraining(BlobStructurePD, Camera.ActualFrame, Camera.VideoPath);
        }

        // Treshold some detections for the multicamera fusion
        if(!CBOption.compare("FastRCNN")){
            if (!DatasetName.compare("Terrace")){
                Threshold = 0.5;
            }
            else if (!DatasetName.compare("RLC") || !DatasetName.compare("Wildtrack")){
                Threshold = 0.8;
            }
            else
                Threshold = 0.75;

            ThresholdDetections(BlobStructurePD, BlobStructurePDNMS, Threshold, 1);
            ThresholdDetections(BlobStructurePD, BlobStructureLowScores, 0, Threshold);
        }
        else if(!CBOption.compare("YOLOv3")){
            if (!DatasetName.compare("Terrace")){
                Threshold = 0.5;
            }
            else if (!DatasetName.compare("RLC") || !DatasetName.compare("Wildtrack")){
                Threshold = 0.8;
            }
            else
                Threshold = 0.70;
            ThresholdDetections(BlobStructurePD, BlobStructurePDNMS, Threshold, 1);
            ThresholdDetections(BlobStructurePD, BlobStructureLowScores, 0, Threshold);
        }
        else{
            BlobStructurePDNMS.clear();
            BlobStructurePDNMS = BlobStructurePD;
        }

        // Code to plot detections to Actual Frame and save into disk
        Mat ActualFrameCopy;
        Camera.ActualFrame.copyTo(ActualFrameCopy);
        paintBoundingBoxes(ActualFrameCopy, BlobStructurePDNMS, Camera.CameraNumber, 1);
        // Draw common flow into the Actual Frame
        Camera.ProjectCommonSemantic(ActualFrameCopy);

        // Save Actual Frame to disk
        String SavingPath = "/home/alex/Desktop/Resultados Auxiliares/Raw Detections/Camera " + to_string(Camera.CameraNumber) + "/Image" + FrameNumber + ".png";
        imwrite(SavingPath, ActualFrameCopy);
    }
}

void PeopleDetector::HOGPeopleDetection(CameraStream &Camera)
{
    // Clear vectors
    Camera.HOGBoundingBoxes.clear();
    Camera.HOGBoundingBoxesNMS.clear();
    Camera.HOGScores.clear();

    // Initialice the SVM
    HOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // HOG Detector
    HOG.detectMultiScale(Camera.ActualFrame, Camera.HOGBoundingBoxes, Camera.HOGScores, 0, Size(8, 8), Size(32, 32), 1.1, 2);

    Camera.HOGBoundingBoxesNMS = Camera.HOGBoundingBoxes;
    AllPedestrianVector = Camera.HOGBoundingBoxesNMS;
    AllPedestrianVectorScore = Camera.HOGScores;
}

void PeopleDetector::DPMPeopleDetection(CameraStream &Camera)
{
    Camera.DPMBoundingBoxes.clear();
    Camera.DPMScores.clear();

    // Auxiliar ActualFrame
    Mat AuxiliarFrame = Camera.ActualFrame.clone();

    // Local detection vector
    vector<DPMDetector::ObjectDetection> DPMBoundingBoxesAux;
    // DPM detector with NMS
    DPMdetector->detect(AuxiliarFrame, DPMBoundingBoxesAux);

    // Convert from vector<ObjectDetection> to vector<Rect>
    for (unsigned int i = 0; i < DPMBoundingBoxesAux.size(); i++){
        Rect Aux1 = DPMBoundingBoxesAux[i].rect;
        float score = DPMBoundingBoxesAux[i].score;
        Camera.DPMScores.push_back(score);
        Camera.DPMBoundingBoxes.push_back(Aux1);
    }

    AllPedestrianVector = Camera.DPMBoundingBoxes;
    AllPedestrianVectorScore = Camera.DPMScores;
}

void PeopleDetector::paintBoundingBoxes(Mat &ActualFrame, vector<cvBlob> BlobStructure, int CameraNumber, int Thickness)
{
    Scalar Color = Scalar (255, 0, 0);
    /*
    if (CameraNumber == 1)
        Color = Scalar (0, 255, 0);
    else if (CameraNumber == 2)
        Color = Scalar (255, 0, 0);
    else if (CameraNumber == 3)
        Color = Scalar (0, 0, 255);
    else
        Color = Scalar (255, 255, 255);
        */

    for (size_t i = 0; i < BlobStructure.size(); i++) {
        Rect r = BlobStructure[i].Blob;

        if(r.x <= 0)
            r.x = 1;
        if(r.y <= 0)
            r .y = 1;

        if(r.height > 0 && r.width > 0){
            rectangle(ActualFrame, r.tl(), r.br(), Color, Thickness);
        }
    }
}

void PeopleDetector::ACFPeopleDetection(CameraStream &Camera)
{
    // Auxiliar ActualFrame
    Mat AuxiliarFrame = Camera.ActualFrame.clone();

    // Local detection vector
    DetectionList ACFDetectionsList;
    // ACF detector
    ACFDetectionsList = ACFdetector.applyDetector(AuxiliarFrame);

    //NMS
    DetectionList ACFDetectionsListNMS;
    NonMaximumSuppression NMS;
    ACFDetectionsListNMS = NMS.dollarNMS(ACFDetectionsList);

    // Convert from DetectionList to vector<Rect>
    for (unsigned int i = 0; i < ACFDetectionsListNMS.Ds.size(); i++){
        Detection Ds = ACFDetectionsListNMS.Ds[i];

        cvBlob Blob;
        Blob.Blob.x = Ds.m_x;
        Blob.Blob.y = Ds.m_y;
        Blob.Blob.width = Ds.m_width;
        Blob.Blob.height = Ds.m_height;
        Blob.Score = Ds.m_score;
        Blob.OriginalCamera = Camera.CameraNumber;
        Blob.DetecID = i;
        BlobStructurePD.push_back(Blob);
    }
}

void PeopleDetector::PSPNetScores(String DatasetName, int CameraNumber, String FrameNumber)
{
    int FrameNumber2 = atoi(FrameNumber.c_str()) - 1;
    string SemScoresPath;

    if (FrameNumber2 < 10){
        // Add 000 to the path string
        SemScoresPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/" + DatasetName + "/Scores PSP/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "000" + to_string(FrameNumber2) + ".png";
    }
    else if (FrameNumber2 < 100){
        // Add 00 to the path string
        SemScoresPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/" + DatasetName + "/Scores PSP/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "00" + to_string(FrameNumber2) + ".png";
    }
    else if (FrameNumber2 < 1000){
        // Add 0 to the path string
        SemScoresPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/" + DatasetName + "/Scores PSP/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "0" + to_string(FrameNumber2) + ".png";
    }
    else{
        SemScoresPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/" + DatasetName + "/Scores PSP/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + to_string(FrameNumber2) + ".png";
    }

    Mat ScoresImage = imread(SemScoresPath, IMREAD_GRAYSCALE);

    // Check for invalid input
    if(! ScoresImage.data ){
        cout << "Could not open the PSP Score file with the following path:" << endl;
        cout << SemScoresPath << endl;
        exit(EXIT_FAILURE);
    }

    for(size_t i = 0; i < AllPedestrianVector.size(); i++){
        Rect Blob = AllPedestrianVector[i];
        Mat Aux = ScoresImage(Blob);

        int rows = Aux.rows;
        int cols = Aux.cols;

        double s = 0;
        int counter = 0;

        for(int row = 0; row < rows; row++){
            for(int col = 0; col < cols; col++){
                double pixel = Aux.at<uchar>(row, col);
                if(pixel > 15){
                    s = s + pixel;
                    counter++;
                }
            }
        }

        if(counter != 0){
            s = s / counter;
            s = s / 100;
            AllPedestrianVectorScore.push_back(s);
        }
        else{
            s = 0;
            AllPedestrianVectorScore.push_back(s);
        }
    }
}

void PeopleDetector::FastRCNNPeopleDetection(String VideoPath, CameraStream &Camera, string FrameNumber)
{
    // Clear vectors
    Camera.RCNNBoundingBoxes.clear();
    Camera.RCNNScores.clear();

    String FileName = VideoPath + "/Fast-RCNN/Camera" + to_string(Camera.CameraNumber) + "Syncfast.txt";

    // Decodes txt file and saves results in RCNNBoundingBoxes variable
    decodeBlobFile(Camera, FileName, FrameNumber);

    AllPedestrianVector = Camera.RCNNBoundingBoxes;
    AllPedestrianVectorScore = Camera.RCNNScores;

    BlobStructurePD.clear();
    for(size_t i = 0; i< Camera.RCNNBoundingBoxes.size(); i++){
        cvBlob Blob;
        Blob.Blob = Camera.RCNNBoundingBoxes[i];
        Blob.Score = Camera.RCNNScores[i];
        Blob.OriginalCamera = Camera.CameraNumber;
        Blob.DetecID = i;
        BlobStructurePD.push_back(Blob);
    }
}

void PeopleDetector::YOLOPeopleDetection(String VideoPath, CameraStream &Camera, String DatasetName, string FrameNumber)
{
    // Clear vectors
    Camera.RCNNBoundingBoxes.clear();
    Camera.RCNNScores.clear();

    String FileName = VideoPath + "/YOLOv3/" + DatasetName + "Camera" + to_string(Camera.CameraNumber) + "_YOLOv3.txt";

    // Decodes txt file and saves results in RCNNBoundingBoxes variable
    decodeBlobFile(Camera, FileName, FrameNumber);

    BlobStructurePD.clear();
    for(size_t i = 0; i< Camera.RCNNBoundingBoxes.size(); i++){
        cvBlob Blob;
        Blob.Blob = Camera.RCNNBoundingBoxes[i];
        Blob.Score = Camera.RCNNScores[i];
        Blob.OriginalCamera = Camera.CameraNumber;
        Blob.DetecID = i;
        BlobStructurePD.push_back(Blob);
    }
}

void PeopleDetector::decodeBlobFile(CameraStream &Camera, string FileName, string FrameNumber)
{
    ifstream input(FileName);

    if (!input) {
        // The file does not exists
        cout << "The file containing the FastRCNN blobs does not exist. Path: " << FileName << endl;
        exit(EXIT_FAILURE);
    }

    // Auxiliary variables to store the information
    string AuxString;
    int x2, y2;
    double Score;
    Rect RectAux;
    size_t found;
    int Counter = 0;
    int LineCounter = 0;

    // Start decoding the file
    while (input >> AuxString){

        if (AuxString.find("Frame") != std::string::npos) {
            // Check if the desired line has been read and so
            // exit the function
            if (LineCounter == atoi(FrameNumber.c_str()))
                return;
            LineCounter++;
        }

        if (LineCounter == atoi(FrameNumber.c_str())) {
            switch(Counter)
            {
            case 0:
                Counter++;
                break;
            case 1:
                // Case for x1
                found = AuxString.find(',');
                AuxString = AuxString.substr(1, found - 1 );
                RectAux.x = atoi(AuxString.c_str());
                Counter++;
                break;
            case 2:
                // Case for y1
                found = AuxString.find(',');
                AuxString = AuxString.substr(0, found);
                RectAux.y = atoi(AuxString.c_str());
                Counter++;
                break;
            case 3:
                // Case for x2
                found = AuxString.find(',');
                AuxString = AuxString.substr(0, found);
                x2 = atoi(AuxString.c_str());
                Counter++;
                break;
            case 4:
                // Case for y2
                found = AuxString.find(']');
                AuxString = AuxString.substr(0, found);
                y2 = atoi(AuxString.c_str());
                Counter++;
                break;
            case 5:
                // Case for "Score:"
                Counter++;
                break;
            case 6:
                // Case for score
                Score = boost::lexical_cast<double>(AuxString);

                // Save blob information into class variables
                RectAux.width = x2 - RectAux.x;
                RectAux.height = y2 - RectAux.y;
                Camera.RCNNBoundingBoxes.push_back(RectAux);
                Camera.RCNNScores.push_back(Score);

                // Restart the couter to read another blob
                Counter = 1;
                break;

            }
        }
    }
}

void PeopleDetector::ThresholdDetections(const vector<cvBlob> &srcBlobStructure, vector<cvBlob> &resBlobStructure, double ThresholdMin, double ThresholdMax)
{
    if(srcBlobStructure.empty())
        return;

    vector<int> SuppresedBlobs;
    vector<cvBlob> srcCopy = srcBlobStructure;

    for(size_t Index = 0; Index < srcBlobStructure.size(); Index++){
        double Score = srcBlobStructure[Index].Score;
        if(!(Score >= ThresholdMin && Score <= ThresholdMax))
            SuppresedBlobs.push_back(Index);
    }

    sort(SuppresedBlobs.begin(), SuppresedBlobs.end(), greater<int>());

    for(size_t i = 0; i < SuppresedBlobs.size(); i++){
        int Index = SuppresedBlobs[i];
        srcCopy.erase(srcCopy.begin() + Index);
    }

    resBlobStructure.clear();
    resBlobStructure = srcCopy;
}

void PeopleDetector::projectBlobs(vector<cvBlob> &BlobStructure, Mat Homography, Mat HomographyBetweenViews, int CameraNumber, bool Cilinder, String DatasetName)
{
    if (BlobStructure.empty()){
        ProjectedLeftPoints.clear();
        ProjectedRightPoints.clear();
        ProjectedCenterPoints.clear();
        return;
    }

    // Each dataset has a different multicamera configuration
    Point Rotation;
    if (!DatasetName.compare("Terrace")){
        if(CameraNumber == 1){
            Rotation.x = -1;
            Rotation.y = 1;
        }
        else if(CameraNumber == 2){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 3){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 4){
            Rotation.x = -1;
            Rotation.y = -1;
        }
    }
    else if (!DatasetName.compare("PETS2012_S2_L1")){
        if(CameraNumber == 1){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 2){
            Rotation.x = -1;
            Rotation.y = 1;
        }
        else if(CameraNumber == 3){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 4){
            Rotation.x = -1;
            Rotation.y = -1;
        }
    }
    else if (!DatasetName.compare("PETS2012_CC")){
        if(CameraNumber == 1){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 2){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 3){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 4){
            Rotation.x = -1;
            Rotation.y = -1;
        }
    }
    else if (!DatasetName.compare("Wildtrack")){
        if(CameraNumber == 1){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 2){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 3){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 4){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 5){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 6){
            Rotation.x = 1;
            Rotation.y = -1;
        }
        else if(CameraNumber == 7){
            Rotation.x = 1;
            Rotation.y = -1;
        }
    }
    else if (!DatasetName.compare("RLC")){
        if(CameraNumber == 1){
            Rotation.x = -1;
            Rotation.y = 1;
        }
        else if(CameraNumber == 2){
            Rotation.x = -1;
            Rotation.y = 1;
        }
        else if(CameraNumber == 3){
            Rotation.x = -1;
            Rotation.y = 1;
        }
    }


    //Mat ImageAux = imread("/Volumes/ALEXHD/Datasets/" + DatasetName + "/Wrapped Images/RGB" + to_string(CameraNumber) + "Median.png", IMREAD_COLOR);

    vector<Point2f> LeftCornerVectors, RightCornerVectors;
    ProjectedLeftPoints.clear();
    ProjectedRightPoints.clear();

    // Extract bottom bounding box segment
    for (size_t i = 0; i < BlobStructure.size(); i++) {
        // Extract the corresponding rectangle
        Rect r = BlobStructure[i].Blob;
        Point2f LeftCorner, RightCorner;

        // Extract Coordinates of the bottom segment
        LeftCorner.x = cvRound(r.x);
        LeftCorner.y = cvRound(r.y + r.height);
        RightCorner.x = cvRound(r.x + r.width);
        RightCorner.y = cvRound(r.y + r.height);

        // Same coordinates in vectors
        LeftCornerVectors.push_back(LeftCorner);
        RightCornerVectors.push_back(RightCorner);
    }
    // Apply Homography to vectors of Points to find the projection in the view
    perspectiveTransform(LeftCornerVectors, ProjectedLeftPoints, HomographyBetweenViews);
    perspectiveTransform(RightCornerVectors, ProjectedRightPoints, HomographyBetweenViews);
    // Apply Homography to vectors of Points to find the projection in the cenital plane
    perspectiveTransform(ProjectedLeftPoints, ProjectedLeftPoints, Homography);
    perspectiveTransform(ProjectedRightPoints, ProjectedRightPoints, Homography);

    // Vector to save the coordinates of projected squares for gaussians
    ProjectedCenterPoints.clear();

    for (size_t i = 0; i < ProjectedLeftPoints.size(); i++) {
        // Left Projected Point
        Point2f LeftProjected = ProjectedLeftPoints[i];
        // Rigth Projected Point
        Point2f RightProjected = ProjectedRightPoints[i];

        // Middle Segment Point
        Point2f MiddleSegmentPoint;
        MiddleSegmentPoint.x = cvRound((RightProjected.x + LeftProjected.x) / 2);
        MiddleSegmentPoint.y = cvRound((RightProjected.y + LeftProjected.y) / 2);

        // ------------------------------- //
        // APROXIMACION CILINDRICA DE RAFA //
        // ------------------------------- //
        Point2f C;
        if(Cilinder){
            // Direction Vector From Left Point to Rigth Point
            Point2f VectorLeft2Rigth;
            VectorLeft2Rigth.x = LeftProjected.x - RightProjected.x;
            VectorLeft2Rigth.y = LeftProjected.y - RightProjected.y;

            // Normalize Direction Vector
            float mag = sqrt (VectorLeft2Rigth.x * VectorLeft2Rigth.x + VectorLeft2Rigth.y * VectorLeft2Rigth.y);
            VectorLeft2Rigth.x = VectorLeft2Rigth.x / mag;
            VectorLeft2Rigth.y = VectorLeft2Rigth.y / mag;

            // Depending on the camera the direction of the perpendicular line rotation is
            // different
            float temp = VectorLeft2Rigth.x;
            VectorLeft2Rigth.x = Rotation.x * VectorLeft2Rigth.y;
            VectorLeft2Rigth.y = Rotation.y * temp;

            // Length of the new perpedicular line
            float length = sqrt(pow((RightProjected.x - LeftProjected.x),2) + pow((RightProjected.y - LeftProjected.y),2)) / 2;

            // Center of the projected square
            C.x = MiddleSegmentPoint.x + VectorLeft2Rigth.x * length;
            C.y = MiddleSegmentPoint.y + VectorLeft2Rigth.y * length;

            //line(ImageAux,C,MiddleSegmentPoint,Scalar(255,0,0));
            //line(ImageAux,LeftProjected,RightProjected,Scalar(0,255,0));
            //imwrite(("/Users/alex/Desktop/" + to_string(CameraNumber) + ".png"), ImageAux);
        }
        // ------------------------------- //
        // APROXIMACION CILINDRICA DE RAFA //
        // ------------------------------- //

        // Save projected square central point
        if(Cilinder){
            ProjectedCenterPoints.push_back(C);
            BlobStructure[i].FloorPoint = C;
        }
        else{
            ProjectedCenterPoints.push_back(MiddleSegmentPoint);
            BlobStructure[i].FloorPoint = MiddleSegmentPoint;
        }
    }
}

void PeopleDetector::meshgrid(Mat &X, Mat &Y, int rows, int cols)
{
    X = Mat::zeros(1, cols, CV_32FC1);
    Y = Mat::zeros(rows, 1, CV_32FC1);

    // Create incrementing row and column vector
    for (int i = 0; i < cols; i++)
        X.at<float>(0,i) = i;

    for (int i = 0; i < rows; i++)
        Y.at<float>(i,0) = i;

    // Create matrix repiting row and column
    X = repeat(X, rows, 1);
    Y = repeat(Y, 1, cols);
}

void PeopleDetector::gaussianFunction(Mat &Gaussian3C, Mat X, Mat Y, Point2f center, double score, int CameraNumber)
{
    Mat Gaussian;
    Mat Fra1, Fra2, Powx1, Powx2, Powy1, Powy2;
    double A = 1;
    double MeanX, MeanY, sigmaX, sigmaY;

    // Gaussian Parameters
    MeanX = center.x;
    MeanY = center.y;
    sigmaX = score;
    sigmaY = score;

    // X Equation
    pow((X - MeanX), 2, Powx1);
    pow(sigmaX, 2, Powx2);
    Powx2 = 2*Powx2;
    divide(Powx1, Powx2, Fra1);

    // Y Equation
    pow((Y - MeanY), 2, Powy1);
    pow(sigmaY, 2, Powy2);
    Powy2 = 2*Powy2;
    divide(Powy1, Powy2, Fra2);

    // Combine X and Y fractions
    Gaussian = -(Fra1 + Fra2);
    exp(Gaussian, Gaussian);
    Gaussian = A*Gaussian;

    // Convert Gaussian to 3-channel matrix
    vector<cv::Mat> GaussianChannels(3);
    GaussianChannels.at(0) = Mat::zeros(X.rows, X.cols, CV_32FC1);
    GaussianChannels.at(1) = Mat::zeros(X.rows, X.cols, CV_32FC1);
    GaussianChannels.at(2) = Mat::zeros(X.rows, X.cols, CV_32FC1);

    if (CameraNumber == 1)
        GaussianChannels.at(1) = Gaussian;

    if (CameraNumber == 2)
        GaussianChannels.at(0) = Gaussian;

    if (CameraNumber == 3)
        GaussianChannels.at(2) = Gaussian;

    merge(GaussianChannels, Gaussian3C);
}

void PeopleDetector::SemanticConstraining(vector<cvBlob> &BlobStructure, Mat &ActualFrame, String VideoPath)
{
    if(BlobStructure.empty())
        return;

//    // Load the SemanticImage
//    Mat SemanticImage = imread(VideoPath + "/Projected Semantic Frames/CommonSemantic.png", IMREAD_GRAYSCALE);
//    if((!SemanticImage.data)) {
//        cout <<  "Could not open the common semantic images for SemanticConstraining2 function with path:" + VideoPath + "/Projected Semantic Frames/CommonSemantic.png" << endl ;
//        exit(EXIT_FAILURE);
//    }

    // Load the Authors ROI
    Mat SemanticImage = imread(VideoPath + "/Projected Semantic Frames/CommonSemanticAuthors.png", IMREAD_GRAYSCALE);
    if((!SemanticImage.data)) {
        cout <<  "Could not open the common semantic images for SemanticConstraining2 function with path:" + VideoPath + "/Projected Semantic Frames/CommonSemanticAuthors.png" << endl ;
        exit(EXIT_FAILURE);
    }

    // Vector for supressed points due to semantic constrains
    SupressedIndices.clear();
    int Counter = 0;

    for(size_t i = 0; i < BlobStructure.size(); i++){
        Point2f BottomCenter, BottomCenterProjected;
        BottomCenterProjected = BlobStructure[i].FloorPoint;

        if((BottomCenterProjected.x > 0) && (BottomCenterProjected.y > 0) && (BottomCenterProjected.x < SemanticImage.cols) && (BottomCenterProjected.y < SemanticImage.rows)) {
            int Label = SemanticImage.at<uchar>(cvRound(BottomCenterProjected.y), cvRound(BottomCenterProjected.x)) / 20;

            if (!(Label == 3)){
                SupressedIndices.push_back(Counter);
                putText(ActualFrame, "BLOB SUPRESSED", BottomCenter, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.5);
            }
        }
        else{
            SupressedIndices.push_back(Counter);
            putText(ActualFrame, "BLOB SUPRESSED", BottomCenter, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.5);
        }
        Counter++;
    }

    sort(SupressedIndices.begin(), SupressedIndices.end(), greater<int>());

    // Suppress the blobs from the vector
    for(size_t k = 0; k < SupressedIndices.size(); k++){
        int Index = SupressedIndices[k];
        BlobStructure.erase(BlobStructure.begin() + Index);
    }
}

void PeopleDetector::ExtractDataUsage(int CameraNumber, String FrameNumber, Mat Homography, Mat HomographyBetweenViews)
{
    if(AllPedestrianVectorNMS.empty() || Homography.empty() || HomographyBetweenViews.empty()){
        StatisticalBlobFile << "0" << endl;
        return;
    }

    Mat StatisticalMap = imread("/Users/alex/Desktop/StatisticalMap.png", IMREAD_GRAYSCALE);

    // Check for invalid input
    if(! StatisticalMap.data ){
        cout << "Could not open Statistical Map for ExtractDataUsage fucntion with the following path: /Users/alex/Desktop/StatisticalMap.png" << endl;
        exit(EXIT_FAILURE);
    }

    int FloorCount = 0, DoorCount = 0, ChairCount = 0;

    for(int i = 0; i < AllPedestrianVectorNMS.size(); i++){
        Rect Blob = AllPedestrianVectorNMS[i];
        int MapLabel;
        Point2f BottomCenter, BottomCenterProjected;
        vector<Point2f> VectorAux;

        // Compute bottom middle part of the blob and save it in an auxiliar vector
        BottomCenter.x = cvRound(Blob.x + Blob.width/2);
        BottomCenter.y = Blob.y + Blob.height;
        VectorAux.push_back(BottomCenter);

        // Transform the point to the cenital plane
        perspectiveTransform(VectorAux, VectorAux, HomographyBetweenViews);
        perspectiveTransform(VectorAux, VectorAux, Homography);

        // Extract the projected point
        BottomCenterProjected = VectorAux[0];

        StatisticalBlobFile << BottomCenterProjected.x << " " << BottomCenterProjected.y << " ";

        if((BottomCenterProjected.x > 0) && (BottomCenterProjected.y > 0) && (BottomCenterProjected.x < StatisticalMap.cols) && (BottomCenterProjected.y < StatisticalMap.rows)) {
            // Check label of Statistical Map
            MapLabel = StatisticalMap.at<uchar>(cvRound(BottomCenterProjected.y), cvRound(BottomCenterProjected.x));

            //cout << "Pos: " << BottomCenterProjected << " with label " << to_string(MapLabel) << endl;
            // If floor, door or chair sum 1
            if(MapLabel == 3){
                FloorCount++;
            }
            else if(MapLabel == 8){
                DoorCount++;
            }
            else if(MapLabel == 9){
                ChairCount++;
            }
        }
    }
    StatisticalBlobFile << endl;
    cout << "Frame: " << FrameNumber << ". Camera " << to_string(CameraNumber) << ". " << to_string(FloorCount) << " on the floor. "
         << to_string(DoorCount) << " usign the doors. " << to_string(ChairCount) << " seated on chairs." << endl;
}

void PeopleDetector::KMeans(vector<Point2f> Points, vector<Point2f> &OutputPoints, vector<int> &Labels, int NMinClusters, int NMaxClusters)
{
    if(Points.empty())
        return;

    if(Points.size() == 1){
        OutputPoints.push_back(Points[0]);
        Labels.push_back(0);
        return;
    }

    Mat labels1, labels2;
    Mat OutputPoints1, OutputPoints2;
    TermCriteria Criteria = TermCriteria(TermCriteria::COUNT, 10000, 0.0001);
    int NAttempts = 3;
    vector<double> SilhoutteVector;

    int FinalNClusters = 0;
    if(Points.size() <= 4)
        FinalNClusters = 1;
    else{
        if(NMinClusters != NMaxClusters && NMaxClusters > NMinClusters){
            // Iterate over all possible number of clusters
            for(int K = NMinClusters; K <= NMaxClusters ; K++){
                float Comp = kmeans(Points, K, labels1, Criteria, NAttempts, KMEANS_PP_CENTERS, OutputPoints1 );
                SilhoutteVector.push_back(SilhoutteIndex(Points, labels1, K));
            }
            FinalNClusters = distance(SilhoutteVector.begin(), max_element(SilhoutteVector.begin(), SilhoutteVector.end())) + NMinClusters;
        }
        else
            FinalNClusters = NMinClusters;
    }

    // Compute the final clusters
    double C = kmeans(Points, FinalNClusters, labels2, Criteria, NAttempts, KMEANS_PP_CENTERS, OutputPoints2 );

    // Convert labels2 and outputpoints2
    for(int i = 0; i < OutputPoints2.rows; i++){
        OutputPoints.push_back(OutputPoints2.at<Point2f>(i));
    }
    for(int i = 0; i < labels2.rows; i++){
        Labels.push_back(labels2.at<int>(i));
    }
}

void PeopleDetector::DistanceClustering(vector<cvBlob> &BlobStructure, float Distance)
{
    for (size_t i = 0; i < BlobStructure.size(); i++){
        BlobStructure[i].PID = 200;
    }

    int GroupCounter = 0;
    for (size_t i = 0; i < BlobStructure.size(); i++){
        bool Flag = 1;
        Point2f Pos1 = BlobStructure[i].FloorPoint;

        if(BlobStructure[i].PID == 200){
            for (size_t j = 0; j < BlobStructure.size(); j++){
                Point2f Pos2 = BlobStructure[j].FloorPoint;
                if((j != i)){
                    // Check distance between Pos1 and Pos2
                    if(abs(Pos1.x - Pos2.x) <= Distance && abs(Pos1.y - Pos2.y) <= Distance){
                        if(BlobStructure[i].PID == 200 && BlobStructure[j].PID == 200){
                            BlobStructure[i].PID = GroupCounter;
                            BlobStructure[j].PID = GroupCounter;
                        }
                        else if(BlobStructure[j].PID == 200){
                            BlobStructure[j].PID = BlobStructure[i].PID;
                        }
                        else{
                            BlobStructure[i].PID = BlobStructure[j].PID;
                            Flag = 0;
                        }
                    }
                }
            }
            if(BlobStructure[i].PID == 200){
                BlobStructure[i].PID = GroupCounter;
            }
            if(Flag)
                GroupCounter++;
        }
    }
    return;
}

float PeopleDetector::SilhoutteIndex(const vector<Point2f> &Points, const Mat &Labels, int NClusters)
{
    // Function to compute the Silhoutte Index for a set of points divided in NClusters
    float Silhoutte = 0;
    vector<float> SilPoints;

    vector<float> PointsInClusters(NClusters);
    for (size_t i = 0; i < Labels.rows; i++){
        PointsInClusters[Labels.at<int>(i)] = PointsInClusters[Labels.at<int>(i)] + 1;
    }

    for(size_t i = 0 ; i < Points.size(); i++){
        float a = 0;
        vector<float> b(NClusters);
        float CounterA = 0;
        float CounterB = 0;
        Point2f PointI = Points[i];
        int LabelPointI = Labels.at<int>(i);

        for(size_t j = 0 ; j < Points.size(); j++){
            Point2f PointJ = Points[j];
            int LabelPointJ = Labels.at<int>(j);

            // Do not compute distance between the same point
            if(i != j){
                float distance = (PointI.x - PointJ.x) * (PointI.x - PointJ.x) + (PointI.y - PointJ.y) * (PointI.y - PointJ.y);
                if(LabelPointI == LabelPointJ){
                    a+= distance;
                    CounterA++;
                }
                else{
                    b[LabelPointJ] = b[LabelPointJ] + distance;
                }
            }
        }
        if(CounterA != 0 )
            a /= CounterA;

        for(size_t i = 0; i < b.size(); i++){
            if(LabelPointI != i){
                b[i] = b[i] / PointsInClusters[i];
            }
            else
                b[i] = 100000000000000;
        }
        float bfinal = *min_element(begin(b), end(b));

        // Compute the value S for one data point
        float SAux = (bfinal-a) / max(a,bfinal);
        SilPoints.push_back(SAux);
    }

    // Compute the average of SilPoints to obtain the final Silhoutte Index
    for(size_t i = 0; i< SilPoints.size(); i++){
        Silhoutte += SilPoints[i];
    }
    Silhoutte /= SilPoints.size();
    return Silhoutte;
}

void PeopleDetector::blobSavingTXT(vector<cvBlob> BlobStructure, String FrameNumber, int CameraNumber)
{
    int FrameNumber2 = atoi(FrameNumber.c_str()) - 1;
    String Filename;
    if (FrameNumber2 < 10){
        // Add 000 to the path string
        Filename = "/Camera" + to_string(CameraNumber) + "000" + to_string(FrameNumber2) + ".jpg";
        Filename = '\"' + Filename + '\"';
    }
    else if (FrameNumber2 < 100){
        // Add 00 to the path string
        Filename = "/Camera" + to_string(CameraNumber) + "00" + to_string(FrameNumber2) + ".jpg";
        Filename = '\"' + Filename + '\"';
    }
    else if (FrameNumber2 < 1000){
        // Add 0 to the path string
        Filename = "/Camera" + to_string(CameraNumber) + "0" + to_string(FrameNumber2) + ".jpg";
        Filename = '\"' + Filename + '\"';
    }
    else{
        Filename = "/Camera" + to_string(CameraNumber) + to_string(FrameNumber2) + ".jpg";
        Filename = '\"' + Filename + '\"';
    }

    BoundingBoxesFile << Filename << ";";
    for(int i = 0; i< BlobStructure.size(); i++){
        Rect Blob = BlobStructure[i].Blob;
        float Score = BlobStructure[i].Score;

        if(Blob.x > 0 && Blob.y > 0 && Blob.width > 0 && Blob.height > 0)
            BoundingBoxesFile << " (" << Blob.x << ", " << Blob.y << ", " << Blob.width << ", " << Blob.height << "):" << Score << ",";
    }
    BoundingBoxesFile << endl;
}
