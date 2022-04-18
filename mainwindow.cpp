#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "barrier.h"
#include "camerastream.h"
#include "cameraworker.h"
#include "aboutdialog.h"
#include <QDebug>
#include <QLabel>
#include <string>
#include <QThread>
#include <QMutex>
#include <iostream>
#include <QGridLayout>
#include <QFileDialog>
#include <math.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>
#include "CameraModel/cameraModel.h"
#include <QElapsedTimer>
#include <QFuture>
#include <QtConcurrent/QtConcurrent>
#include <qtconcurrentrun.h>
#include <QApplication>
#include "basicblob.hpp"
#include "graph.h"

using namespace std;
using namespace cv;
using namespace QtConcurrent;

// Vector of camera streams
vector<CameraStream> Cameras;
// Vector of camera calibrations
vector<Etiseo::CameraModel> CameraModelVector;

// Distinguisble colors
Scalar colorTab[] = {
    Scalar(0,0,255),
    Scalar(255,0,0),
    Scalar(0,255,0),
    Scalar(0,0,44),
    Scalar(255,	26,185),
    Scalar(255,	211,0),
    Scalar(0,88,0),
    Scalar(132,	132,255),
    Scalar(158,	79,70),
    Scalar(0,255,193),
    Scalar(0,132,149),
    Scalar(0,0,123),
    Scalar(149,211,	79),
    Scalar(246,158,	220),
    Scalar(211, 18,	255),
    Scalar(123,	26,	106),
    Scalar(246,	18,	97),
    Scalar(255,	193,	132),
    Scalar(35,	35,	9),
    Scalar(141,	167,	123),
    Scalar(246,	132,	9),
    Scalar(132,	114,	0),
    Scalar(114,	246,	255),
    Scalar(158,	193,	255),
    Scalar(114,	97,	123),
    Scalar(158,	0,	0),
    Scalar(0,	79,	255),
    Scalar(0,	70,	149),
    Scalar(211,	255,	0),
    Scalar(185,	79,	211)};

MainWindow::MainWindow(String ModeCons, String ParamDatasetNameCons, String ParamPDDetectorCons, bool ParamSemanticFitleringCons, bool ParamMultiCameraCons, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Global Path variable should be change if used in other computer
    GlobalPath = "/home/alex/Desktop/IPCV-MasterThesis/MultithreadParameters";

    // Save the operation mode
    Mode = ModeCons;

    if(!Mode.compare("Parameters")){
        ParamDatasetName = ParamDatasetNameCons;
        ParamPDDetector = ParamPDDetectorCons;
        ParamSemanticFitlering = ParamSemanticFitleringCons;
        ParamMultiCamera = ParamMultiCameraCons;

        // Select video file extension depending on the dataset
        string VideoExtension;
        if (!ParamDatasetName.compare("Terrace"))
            VideoExtension = ".m4v";
        else if (!ParamDatasetName.compare("Wildtrack"))
            VideoExtension = ".mpg";
        else
            VideoExtension = ".m2v";

        // Video Paths
        for(int i = 1; i <= numCams; i++){
            filenames << QString::fromStdString("/home/alex/Desktop/Pedestrian Detection Datasets/" + ParamDatasetName + "/Videos/Camera" + to_string(i) + "Sync" + VideoExtension);
        }
        threadStarting();
    }

    // MetaType register for connection between signal and slots
    qRegisterMetaType<Mat>("Mat");
    qRegisterMetaType<String>("String");
    qRegisterMetaType<CameraStream>("CameraStream");
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_File_triggered()
{
    // Messages for the interface
    ui->textBrowser->append("Interface thread started");
    ui->textBrowser->append("Press File and Open video files");

    // Opens the QFileDialog to get files
    filenames = QFileDialog::getOpenFileNames(this, tr("Open video files for the cameras"), GlobalPath, tr("Video Files (*.mpg *.avi *.m4v *.ts *.m2v)"));
    threadStarting();
}

void MainWindow::threadStarting()
{
    // Create a barrier for thread syncronization during the execution
    Barrier barrier(numCams, numCams);

    // Thread creating loop. One thread per camera
    for (int i = 0; i < numCams; i++){
        // Create threads
        threads[i] = new QThread;

        // Create the camera class for workers
        CameraStream Camera;
        // Fill some camera initial variables
        fillCameraVariables(Camera, i);

        // New CameraWorker initialize with Camera(CameraStream) and the barrier
        CameraWorkers[i] = new CameraWorker(Camera, barrier, ParamDatasetName, ParamPDDetector, ParamSemanticFitlering, ParamMultiCamera, numCams, Mode);

        if(!Mode.compare("GUI")){
            // Fill UI widget size in CameraWorkers
            CameraWorkers[i]->WidgetHeight = ui->CVWidget1->height();
            CameraWorkers[i]->WidgetWidth = ui->CVWidget1->width();
            CameraWorkers[i]->CBOption = ui->PeopleDetectorCB->currentText().toStdString();
            CameraWorkers[i]->DatasetName = "EPS_Hall/EPS_Hall_6";
            cout << "COMPLETAR LINEA 120 MAINWINDOW" << endl;
            exit(EXIT_FAILURE);
            CameraWorkers[i]->SemanticFiltering = ui->PDFiltering;
            CameraWorkers[i]->MultiCameraFiltering = ui->MultiCameraFiltering;
            ui->textBrowser->append(QString::fromStdString("Thread from camera " + to_string(i+1) + " started"));
        }

        // Move CameraWorker to thread
        CameraWorkers[i]->moveToThread(threads[i]);

        // Connect signals to slot between thread and CameraWorkers
        connectSignals2Slots(threads[i], CameraWorkers[i]);

        // Thread is started
        threads[i]->start();
    }
}

void MainWindow::connectSignals2Slots(QThread *thread, CameraWorker *worker)
{
    // THREAD SIGNAL CONECTION
    // Thread starting with processVideo slot
    connect(thread, SIGNAL(started()), worker, SLOT(preProcessVideo()));
    // Thread finished with delete slot
    connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));

    // WORKER SIGNAL CONNECTIONS
    connect(worker, SIGNAL(HAPAlgorithmSignal(int, String)), this, SLOT(HAPAlgorithm(int, String)));
    if(!Mode.compare("GUI")){
        connect(worker, SIGNAL(frameFinished(Mat, Mat, int)), this, SLOT(updateVariables(Mat, Mat, int)));
        connect(worker, SIGNAL(frameFinished(Mat, Mat, int)), this, SLOT(displayFrame(Mat, Mat, int)));
    }

    // finished signal with quit and deleteLater slots
    connect(worker, SIGNAL(finished()), thread, SLOT(quit()));
    connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater()));
}

void MainWindow::fillCameraVariables(CameraStream &Camera, int i)
{
    // Camera Number
    Camera.CameraNumber = i + 1;
    // Global path
    Camera.GlobalPath = GlobalPath.toStdString();
    // Input Video Path
    Camera.InputPath = filenames.at(i).toStdString();
}

void MainWindow::LoadOpenCVCalibration(String ExtrinsicPath, String IntrinsicPath)
{
    for(int Camera = 1; Camera <= numCams; Camera++){

        // EXTRINSIC PARAMETERS //
        string Extrinsicfilename = ExtrinsicPath + "extr_" + to_string(Camera) + ".xml";
        FileStorage fs;
        fs.open(Extrinsicfilename, FileStorage::READ);

        if (!fs.isOpened())
            cerr << "Failed to open " << Extrinsicfilename << endl;

        vector<float> RVecTemp;
        FileNode n = fs["rvec"]; // Read string sequence - Get node
        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it){
            float aux = (float)*it;
            RVecTemp.push_back(aux);
        }
        RVec.push_back(RVecTemp);

        vector<float> TVecTemp;
        n = fs["tvec"]; // Read string sequence - Get node
        it = n.begin();
        it_end = n.end(); // Go through the node
        for (; it != it_end; ++it){
            float aux = (float)*it;
            TVecTemp.push_back(aux);
        }
        TVec.push_back(TVecTemp);

        // INTRINSIC PARAMETERS //
        string filenameIntrinsic = IntrinsicPath + "intr_" + to_string(Camera) + ".xml";
        FileStorage fsIntrinsic;
        fsIntrinsic.open(filenameIntrinsic, FileStorage::READ);

        Mat CameraMatrix = Mat::zeros(3,3,CV_8UC1);
        fsIntrinsic["camera_matrix"] >> CameraMatrix;
        CameraMatrixVector.push_back(CameraMatrix);

        Mat Distortion = Mat::zeros(5,1,CV_8UC1);
        fsIntrinsic["distortion_coefficients"] >> Distortion;
        DistortionVector.push_back(Distortion);
    }
}

vector<Rect> MainWindow::ObtainBBFromP(vector<cvBlob> &PS, int CameraNumber)
{
    vector<Rect> BBVector;
    double WorldHeight;
    Rect BB;
    Point2f GroundPosition;

    size_t InitialIndex = 0;
    size_t FinalIndex = 0;
    if(CameraNumber == 0){
        InitialIndex = 0;
        FinalIndex = OriginalNPoints - 1;
    }
    else if(CameraNumber == 1){
        InitialIndex = OriginalNPoints;
        FinalIndex = OriginalNPoints*2 - 1;
    }
    else if(CameraNumber == 2){
        InitialIndex = OriginalNPoints*2;
        FinalIndex = OriginalNPoints*3 - 1;
    }
    else if(CameraNumber == 3){
        InitialIndex = OriginalNPoints*3;
        FinalIndex = OriginalNPoints*4 - 1;
    }

    for(size_t k = InitialIndex; k <= FinalIndex; k++){
        GroundPosition = PS[k].FloorPoint;
        WorldHeight = PS[k].Heigth3D;

        bool outside;
        BB = BuildBB(GroundPosition, WorldHeight, CameraNumber, HomographyVector[CameraNumber], HomographyBetweenViewsVector[CameraNumber], outside);

        BBVector.push_back(BB);
    }
    return BBVector;
}

float MainWindow::HLossFunction(const vector<cvBlob> &PS, const vector<Mat> &ForegroundVector, const vector<Etiseo::CameraModel> &CameraModelVector)
{
    float H = 0;
    float Denominator = ForegroundVector[0].rows * ForegroundVector[0].cols;

    // BB Computations
    vector<QFuture<vector<Rect>>> BBComputations;
    for(int cam = 0; cam < numCams; cam++){
        BBComputations.push_back(run(this, &MainWindow::ObtainBBFromP, PS, cam));
    }

    vector<vector<Rect>> BBVectors;
    for(int cam = 0; cam < numCams; cam++){
        QFuture<vector<Rect>> BBComputing = BBComputations[cam];
        BBVectors.push_back(BBComputing.result());
    }

    // H Computations
    vector<QFuture<float>> HComputations;
    for(int cam = 0; cam < numCams; cam++){
        HComputations.push_back(run(this, &MainWindow::ExtractHNumerator, ForegroundVector[cam], BBVectors[cam]));
    }

    vector<float> HVector;
    for(int cam = 0; cam < numCams; cam++){
        QFuture<float> HAux = HComputations[cam];
        HVector.push_back(HAux.result());
        H += HAux.result();
    }
    H /= Denominator;
    return -H;
}

float MainWindow::ExtractHNumerator(Mat FG, const vector<Rect> &BBVector)
{
    float Numerator = 0;
    int PixelV;
    float Gamma = 0.3;
    float LhValue;

    // Extract maximum value from FG
    double min, maxFG;
    minMaxLoc(FG, &min, &maxFG);

    for(int row = 0; row < FG.rows; row++){
        uchar* RowPointer = FG.ptr<uchar>(row);
        for(int col = 0; col < FG.cols; col++){
            PixelV = RowPointer[col];
            LhValue = LhFunction(row, col, BBVector, PixelV, maxFG);

            if(PixelV == maxFG){
                Numerator += LhValue;
            }
            else{
                LhValue = 1 - LhValue;
                Numerator += (Gamma * LhValue);
            }
        }
    }
    return Numerator;
}

float MainWindow::LhFunction(int Vrow, int Vcol, const vector<Rect> &BBVector, int PixelV, double maxFG)
{
    float LhValue = 1;
    float Distance, PhiD;
    Rect BB;
    bool Inside = 0;

    for(size_t DetectionIndex = 0; DetectionIndex < BBVector.size(); DetectionIndex++){
        BB = BBVector[DetectionIndex];
        if((Vcol > BB.x) && (Vrow > BB.y) && (Vcol < (BB.x + BB.width)) && (Vrow < (BB.y + BB.height))){
            // Compute distance between BB vertical middle axis and Point(Vrow, Vcol)
            Distance = abs(Vcol - BB.x - (BB.width/2));
            if(Distance > 0)
                PhiD = 2/Distance;
            else
                PhiD = 1;

            LhValue *= (1 - PhiD);
            Inside = 1;
        }
    }

    if(!Inside && (PixelV == maxFG))
        LhValue = 3;

    return LhValue;
}

void MainWindow::PreAssignment(vector<cvBlob> &PS, int Radius)
{
    // Extract connected components in PS by graph theory
    int GroupCounter;
    GraphConnectedComponents(PS, Radius, GroupCounter);

    // Check whether a blob is alone or not and update Fix variable
    for(size_t n = 0; n < PS.size(); n++){
        bool Alone = 1;
        for(size_t m = 0; m < PS.size(); m++){
            if(n != m){
                if(PS[n].PID == PS[m].PID)
                    Alone = 0;
            }
        }
        if(Alone)
            PS[n].Fix = 1;
    }
}

void MainWindow::GraphConnectedComponents(vector<cvBlob> &PS, int Radius, int &GroupCounter)
{
    int NDetections = PS.size();

    // Index vector
    vector<int> indx;
    for(int i = 0; i < NDetections; i++){
        indx.push_back(i);
    }

    // Sort PS vector comparing with (0,0)
    vector<cvBlob> PSSorted;
    Point2f PosI;
    PosI.x = 0;
    PosI.y = 0;
    float DistanceI = 10000000;
    int P1 = 0;
    int P2 = 0;
    while(!indx.empty()){
        for(size_t p = 0; p < indx.size(); p++ ){
            Point2f Pos = PS[indx[p]].FloorPoint;
            float Distance = sqrt(pow((Pos.x - PosI.x) ,2) + pow((Pos.y - PosI.y),2));
            if(Distance < DistanceI){
                DistanceI = Distance;
                P1 = indx[p];
                P2 = p;
            }
        }
        PSSorted.push_back(PS[P1]);

        assert(P2 < indx.size());
        indx.erase(indx.begin() + P2);
        DistanceI = 10000000;
    }
    PS.clear();
    PS = PSSorted;

    // Create a graph with NDetections nodes numbered from 0 to NDetections
    Graph g(NDetections);
    Mat Connected;

    for (int i = 0; i < NDetections; i++){
        Point2f Pos1 = PS[i].FloorPoint;
        int Cam1 = PS[i].OriginalCamera;
        for (int j = 0; j < NDetections; j++){
            Point2f Pos2 =  PS[j].FloorPoint;
            int Cam2 = PS[j].OriginalCamera;
            if((j != i)){
                // Check if they are from the same camera
                if(Cam1 != Cam2){
                    // Check distance between Pos1 and Pos2
                    if(abs(Pos1.x - Pos2.x) <= Radius && abs(Pos1.y - Pos2.y) <= Radius){
                        // Check connected components
                        g.connectedComponents(Connected, NDetections);
                        // Check if j has already a connected component
                        int CJ = 200;
                        for(int c = 0; c < Connected.rows ; c++){
                            int RowSum = sum(Connected.row(c))[0];
                            if(Connected.at<uchar>(c,j) == 1 && RowSum > 1)
                                CJ = c;
                        }
                        // Check if i has already a connected component
                        int CI = 200;
                        for(int c = 0; c < Connected.rows ; c++){
                            int RowSum = sum(Connected.row(c))[0];
                            if(Connected.at<uchar>(c,i) == 1 && RowSum > 1)
                                CI = c;
                        }

                        // Case 1. Both have connected components
                        if(CJ != 200 && CI != 200){
                            bool Add = true;
                            for(int m = 0; m < NDetections ; m++){
                                if(Connected.at<uchar>(CJ,m) == 1 && m != j){
                                    if(PS[m].OriginalCamera == Cam1)
                                        Add = false;
                                }
                            }
                            for(int m = 0; m < NDetections ; m++){
                                if(Connected.at<uchar>(CI,m) == 1 && m != i){
                                    if(PS[m].OriginalCamera == Cam2)
                                        Add = false;
                                }
                            }
                            if(Add)
                                g.addEdge(i, j);
                        }
                        // Case 2. j Point has connected component
                        else if(CJ != 200){
                            bool Add = true;
                            for(int m = 0; m < NDetections ; m++){
                                if(Connected.at<uchar>(CJ,m) == 1 && m != j){
                                    if(PS[m].OriginalCamera == Cam1)
                                        Add = false;
                                }
                            }
                            if(Add)
                                g.addEdge(i, j);
                        }
                        // Case 3. i Point has connected component
                        else if(CI != 200){
                            bool Add = true;
                            for(int m = 0; m < NDetections ; m++){
                                if(Connected.at<uchar>(CI,m) == 1 && m != i){
                                    if(PS[m].OriginalCamera == Cam2)
                                        Add = false;
                                }
                            }
                            if(Add)
                                g.addEdge(i, j);
                        }
                        // Case 4. Neither i nor j have connected components
                        else
                            g.addEdge(i, j);
                    } // If of radius check
                } // If of same camera check
            }
        }
    }

    // Final connected components
    g.connectedComponents(Connected, NDetections);

    // Change PID labels from PS
    for(int n = 0; n < NDetections; n++){
        for(int c = 0; c < Connected.rows; c++){
            if(Connected.at<uchar>(c,n) == 1)
                PS[n].PID = c;
        }
    }
    // Update the number of groups
    GroupCounter = Connected.rows;
}

void MainWindow::AssignHeigths(vector<cvBlob> &PS, double InitialHeigth, int Radius, int &NDetections, String FrameNumber)
{
    // Fill a vector with the initial heigth
    for(size_t i = 0; i < PS.size(); i++){
        PS[i].Heigth3D = InitialHeigth;
    }

    // Extract connected components in PS by graph theory
    int GroupCounter;
    GraphConnectedComponents(PS, Radius, GroupCounter);

    // If there is more than one point in the cluster select one
    vector<cvBlob> Aux;
    vector<cvBlob> NewPS;
    for(int p = 0; p < GroupCounter; p++){
        Aux.clear();

        // Check how many points we have for a group C
        for(size_t j = 0; j < PS.size(); j++){
            if(PS[j].PID == p)
                Aux.push_back(PS[j]);
        }
        int NPoints = Aux.size();

        // If more than one detection in the cluster select centroid
        if(NPoints > 1){
            Point2f MiddlePoint;
            MiddlePoint.x = 0;
            MiddlePoint.y = 0;
            float Score = 0;
            for(int n = 0; n< NPoints; n++){
                MiddlePoint.x += Aux[n].FloorPoint.x;
                MiddlePoint.y += Aux[n].FloorPoint.y;
                Score += Aux[n].Score;
            }
            // Average of middle point and score
            MiddlePoint.x /= NPoints;
            MiddlePoint.y /= NPoints;
            Score /= NPoints;

            // Final fusion detection
            cvBlob NewBlob;
            NewBlob.FloorPoint = MiddlePoint;
            NewBlob.PID = p;
            NewBlob.Score = Score;
            NewBlob.Heigth3D = InitialHeigth;
            NewBlob.OriginalCamera = 15;

            NewPS.push_back(NewBlob);
        }
        else if(NPoints == 1)
            NewPS.push_back(Aux[0]);
    }

    String Path = "/home/alex/Desktop/Pedestrian Detection Datasets/" + ParamDatasetName + "/Wrapped Images/RGBComplete.png";
    Mat ImageWarping = imread(Path);

    if(!ImageWarping.data ){
        cout << "Could not open ImageWarping in AssignHeigths functions with the following path:" << endl;
        cout << Path << endl;
        exit(EXIT_FAILURE);
    }

    // Draw points

    for(size_t i = 0; i < PS.size(); i++){
        Point2f Point = PS[i].FloorPoint;
        circle(ImageWarping, Point, 2, colorTab[PS[i].OriginalCamera - 1]);
        circle(ImageWarping, Point, Radius, colorTab[PS[i].OriginalCamera - 1]);
    }


    // Draw cluster centers
    for(size_t i = 0; i < NewPS.size(); i++){
        Point2f Point = NewPS[i].FloorPoint;
        circle(ImageWarping, Point, 10, colorTab[i], -1);
    }

    String aux2 = "/home/alex/Desktop/IPCV-MasterThesis-master/HAP Method/Suelo/" + FrameNumber + "Suelo.png";
    imwrite(aux2, ImageWarping);

    PS.clear();
    PS = NewPS;
    NDetections = PS.size();
    return;
}

void MainWindow::FixDetections(vector<cvBlob> &PS, const vector<Mat> &ForegroundVector, const vector<Etiseo::CameraModel> &CameraModelVector)
{
    double IncreasingPercentage;
    if (!ParamDatasetName.compare("Terrace")){
        IncreasingPercentage = 0.2;
    }
    else if ((ParamDatasetName.find("PETS2012") != string::npos) || !ParamDatasetName.compare("RLC") || !ParamDatasetName.compare("Wildtrack")){
        IncreasingPercentage = 0.1;
    }

    for(size_t i = 0; i < PS.size(); i++){
        cvBlob BlobS = PS[i];

        // Only fix a blob if it is alone in the PreAssigment
        if(BlobS.Fix == 1){
            int CameraIndex = BlobS.OriginalCamera - 1;
            if((CameraIndex >= ForegroundVector.size()) || (CameraIndex >= HomographyVector.size())
                    || (CameraIndex >= HomographyBetweenViewsVector.size())){
                // Error 1 when indexing vectors. Finish program
                cout << "Error 1 in function FixDetections. CameraIndex = " << to_string(CameraIndex) << endl;
                exit(EXIT_FAILURE);
            }

            Mat Mask = ForegroundVector[CameraIndex];
            Mat Homography = HomographyVector[CameraIndex];
            Mat HomographyBetweenViews = HomographyBetweenViewsVector[CameraIndex];
            bool Outside;

            // Create BB based on FloorPoint and 3DHeigth
            Rect Aux = BuildBB(BlobS.FloorPoint, BlobS.Heigth3D, CameraIndex, Homography, HomographyBetweenViews, Outside);
            if(!Outside){
                int OriginalHeigth = Aux.height;
                Mat CroppedMask = Mask(Aux);
                bool Fixed = 0;
                bool Increase = 0;

                // If the BB contains mask
                if(sum(CroppedMask)[0] > 0){
                    // Check if we have to increase o decrese the heigth
                    for(int Col = 0; Col < CroppedMask.cols; Col++){
                        if(CroppedMask.at<uchar>(CroppedMask.rows, Col) == 254){
                            Increase = 1;
                        }
                    }
                    // Increase Case
                    if(Increase){
                        while(!Fixed){
                            Fixed = 1;
                            Aux.height += 1;
                            if((Aux.height + Aux.y) < Mask.rows){
                                CroppedMask = Mask(Aux);
                                for(int Col = 0; Col < CroppedMask.cols; Col++){
                                    if(CroppedMask.at<uchar>(CroppedMask.rows, Col) == 254)
                                        Fixed = 0;
                                }
                            }
                            if(Aux.height > (OriginalHeigth + OriginalHeigth * IncreasingPercentage))
                                Fixed = 1;
                        }
                    }
                    // Decrese Case
                    else{
                        while(!Fixed){
                            Fixed = 0;
                            Aux.height -= 1;
                            if(Aux.height > 0){
                                CroppedMask = Mask(Aux);
                                for(int Col = 0; Col < CroppedMask.cols; Col++){
                                    if(CroppedMask.at<uchar>(CroppedMask.rows, Col) == 254)
                                        Fixed = 1;
                                }
                            }
                            else{
                                Fixed = 1;
                                Aux.height = 1;
                            }
                        }
                    }
                    // Project and save the final blob
                    BlobS.Blob = Aux;
                    vector<cvBlob> AuxVector;
                    AuxVector.push_back(BlobS);
                    CameraWorkers[CameraIndex]->PeopleDetec.projectBlobs(AuxVector, Homography, HomographyBetweenViews, BlobS.OriginalCamera, 0, ParamDatasetName);

                    if(AuxVector.size() < 1){
                        // Error 2. Finish program
                        cout << "Error 2 in function FixDetections" << endl;
                        exit(EXIT_FAILURE);
                    }

                    PS[i] = AuxVector[0];
                }

                // If the BB does not contain mask delete it
                else{
                    if(i >= PS.size()){
                        // Error 3 when erasing element from vector
                        cout << "Error 3 in function FixDetections" << endl;
                        exit(EXIT_FAILURE);
                    }
                    else{
                        //PS.erase(PS.begin() + i);
                        //i--;
                    }
                }
            }
        }
        PS[i].PID = 200;
    }
}

Rect MainWindow::BuildBB(Point2f GroundPosition, float WorldHeight, int CameraNumber, Mat HomographyIter, Mat HomographyBetweenViewsIter, bool &Outside)
{
    Rect BBResult;
    double Offset = 0;
    double Scale = 1;
    if (!ParamDatasetName.compare("Terrace")){
        if(CameraNumber == 0)
            Offset = 10;
        if(CameraNumber == 1)
            Offset = 20;
        if(CameraNumber == 2)
            Offset = 20;
        if(CameraNumber == 3)
            Offset = 20;
    }
    if (!ParamDatasetName.compare("RLC")){
        Scale = 4;
    }
    int FrameCols = ImageCols;
    if (!ParamDatasetName.compare("PETS2012_S2_L1")){
        if(CameraNumber == 0)
            FrameCols = 768;
    }

    // Project ground position to image coordinates
    vector<Point2f> GroundPositionVector;
    GroundPositionVector.push_back(GroundPosition);
    perspectiveTransform(GroundPositionVector, GroundPositionVector, HomographyIter.inv(DECOMP_LU));
    perspectiveTransform(GroundPositionVector, GroundPositionVector, HomographyBetweenViewsIter.inv(DECOMP_LU));
    Point2f ProjectedCenter = GroundPositionVector[0];

    double ycHead;
    // OpenCV Camera Model Calibration
    if (!ParamDatasetName.compare("Wildtrack")){
        // Rescale and move to original world coordinates the GroundPosition
        Point3f NewProjectedFeet;
        NewProjectedFeet.x = (GroundPosition.x - 1000)*3;
        NewProjectedFeet.y = (GroundPosition.y - 1000)*3;
        NewProjectedFeet.z = WorldHeight;

        vector<Point3f> NewProjectedFeetVector;
        NewProjectedFeetVector.push_back(NewProjectedFeet);
        vector<Point2f> HeadPointVector;
        projectPoints(NewProjectedFeetVector, RVec[CameraNumber], TVec[CameraNumber], CameraMatrixVector[CameraNumber], DistortionVector[CameraNumber], HeadPointVector);
        ycHead = HeadPointVector[0].y;
    }

    // TSAI Camera Model
    else{
        Etiseo::CameraModel CameraModel = CameraModelVector[CameraNumber];
        // Scale coordinates back if necessary
        Point2f ProjectedCenterScaled = ProjectedCenter * Scale;

        // Extract BB Top with CameraModel, BB Bottom and WorldHeigth
        double XWFeet, YWFeet, xcHead;
        CameraModel.imageToWorld(ProjectedCenterScaled.x, ProjectedCenterScaled.y, 0, XWFeet, YWFeet);
        CameraModel.worldToImage(XWFeet, YWFeet, WorldHeight, xcHead, ycHead);

        // Scale back the coordinates
        ycHead = ycHead / Scale;
    }

    BBResult.height = (ProjectedCenter.y - ycHead) + Offset;
    BBResult.width = 0.35 * BBResult.height;
    BBResult.x = ProjectedCenter.x - BBResult.width/2;
    BBResult.y = ProjectedCenter.y - BBResult.height;

    Outside = 0;
    if(BBResult.x <= 0){
        BBResult.width = BBResult.width - abs(BBResult.x);
        BBResult.x = 1;
    }
    if(BBResult.y <= 0){
        BBResult.height = BBResult.height - abs(BBResult.y);
        BBResult.y = 1;
        Outside = 1;
    }
    if(BBResult.x  >= FrameCols){
        BBResult.x = 0;
        BBResult.y = 0;
        BBResult.width = 0;
        BBResult.height = 0;
    }
    else if((BBResult.x + BBResult.width) >= FrameCols){
        BBResult.width = (FrameCols - BBResult.x) - 3;
    }
    if(BBResult.y >= ImageRows){
        BBResult.x = 0;
        BBResult.y = 0;
        BBResult.width = 0;
        BBResult.height = 0;
    }
    else if((BBResult.y + BBResult.height) >= ImageRows){
        BBResult.height = (ImageRows - BBResult.y) - 3;
    }

    if(BBResult.width <= 0 || BBResult.height <= 0)
        Outside  = 1;

    return BBResult;
}

void MainWindow::saveGroundDetectionsXML(vector<cvBlob> PS, vector<cvBlob> ExtraDetections, int GroundHeigth, int GroundWidth)
{
    String XMLPath = "/home/alex/Desktop/IPCV-MasterThesis-master/Matlab/Evaluation Code/Video/" + CameraWorkers[0]->DatasetName + "/GroundDetections.xml";

    // First sequence frame
    if (OpeningFlag) {
        // Open file
        fs.open(XMLPath, FileStorage::WRITE);
        if(!fs.isOpened()){
            cout << "ERROR: Ground detections file not correctly opened." << endl;
            exit(EXIT_FAILURE);
        }

        // Initial Information
        fs << "GroundHeigth"  << GroundHeigth;
        fs << "GroundWidth"  << GroundWidth;
        OpeningFlag = 0;
    }

    fs << "FrameNumber" << "{";
    // First HAP Detections
    for(size_t i = 0; i < PS.size(); i++){
        int x = round(PS[i].FloorPoint.x);
        int y = round(PS[i].FloorPoint.y);
        double Score = PS[i].Score;

        if(x > 0 && x < GroundWidth && y > 0 && y < GroundHeigth)
            fs << "NDetection" << "{" << "x" << x << "y" << y << "Score" << Score << "}";
    }
    // Extra Detections (Low Score)
    for(size_t i = 0; i < ExtraDetections.size(); i++){
        int x = round(ExtraDetections[i].FloorPoint.x);
        int y = round(ExtraDetections[i].FloorPoint.y);
        double Score = ExtraDetections[i].Score;

        if(x > 0 && x < GroundWidth && y > 0 && y < GroundHeigth)
            fs << "NDetection" << "{" << "x" << x << "y" << y << "Score" << Score << "}";
    }
    fs << "}"; // End of FrameNumber Node
}

void MainWindow::HAPAlgorithm(int CameraNumber, String FrameNumber)
{
    if(CameraNumber == 1){
        // Saving and auxiliar variables for the procedure
        vector<Mat> ForegroundVector;
        vector<cvBlob> PS, PSOriginal;

        String SavingPath = "/home/alex/Desktop/IPCV-MasterThesis-master/";

        // Iteration Counter
        int j = 0;

        /* -------------- */
        /* INITIALIZATION */
        /* -------------- */
        // Fill P, AuxScores and OrginalBBVector and the number of detections coming for each camera
        OriginalNPoints = 0;
//        ofstream AuxiliarFile2;
//        AuxiliarFile2.open(SavingPath + "FasterDetecciones.txt");
        for(int n = 0; n < numCams; n++){
            // New structure PS
            PS.insert(PS.end(), CameraWorkers[n]->PeopleDetec.BlobStructurePDNMS.begin(), CameraWorkers[n]->PeopleDetec.BlobStructurePDNMS.end());
            OriginalNPoints += CameraWorkers[n]->PeopleDetec.BlobStructurePDNMS.size();

//            // Save
//            for(int i = 0; i < CameraWorkers[n]->PeopleDetec.BlobStructurePDNMS.size(); i++){
//                cvBlob aux = CameraWorkers[n]->PeopleDetec.BlobStructurePDNMS[i];
//                AuxiliarFile2 << aux.FloorPoint << colorTab[n] << endl;
//            }
        }

        // Check if Detections are empty and exit function
        if(PS.empty()){
            for(int n = 0; n < numCams; n++){
                CameraWorkers[n]->PeopleDetec.BlobStructureHAP.clear();
                CameraWorkers[n]->HapFinished = 1;
            }
            return;
        }

        // Fill vectors that are used in HAP algorithm
        HomographyVector.clear();
        HomographyBetweenViewsVector.clear();
        for(int n = 0; n < numCams; n++){
            CameraWorkers[n]->PeopleDetec.BlobStructureHAP.clear();
            ForegroundVector.push_back(CameraWorkers[n]->Camera.PedestrianMask);
            HomographyVector.push_back(CameraWorkers[n]->Camera.Homography);
            HomographyBetweenViewsVector.push_back(CameraWorkers[n]->Camera.HomographyBetweenViews);
        }

        // Common Pedestrian Mask Extraction
        CommonPeople = Mat::zeros(CameraWorkers[0]->CenitalPlane.rows, CameraWorkers[0]->CenitalPlane.cols, CV_8UC3);

        // Read Calibration
        if (!ParamDatasetName.compare("Wildtrack")){
            // OpenCV Camera Calibration
            String ExtrinsicPath = "/home/alex/Desktop/Pedestrian Detection Datasets/" + ParamDatasetName + "/Calibration/Extrinsic/";
            String IntrinsicPath = "/home/alex/Desktop/Pedestrian Detection Datasets/" + ParamDatasetName + "/Calibration/Intrinsic/";
            LoadOpenCVCalibration(ExtrinsicPath, IntrinsicPath);
        }
        else{
            // Save TSAI Camera Models in vector
            for(int n = 0; n < numCams; n++){
                Etiseo::CameraModel CameraModel("tsai");
                CameraModel.fromXml("/home/alex/Desktop/Pedestrian Detection Datasets/" + ParamDatasetName + "/Calibration/tsai-c" + to_string(n + 1) +".xml");
                CameraModelVector.push_back(CameraModel);
            }
        }

        /* --------------------- */
        /*  DISTANCE CLUSTERING  */
        /* --------------------- */
        vector<Mat> ImagenAux, ImageAux2;
        for(int n = 0; n < numCams; n++){
            Mat Aux;
            cvtColor(CameraWorkers[n]->Camera.PedestrianMask.clone(), Aux, COLOR_GRAY2BGR);
            ImagenAux.push_back(Aux);
            ImageRows = Aux.rows;
            ImageCols = Aux.cols;
        }

        // Assign real initial heigths for the detections, radius for the fusion and boolean variable for optimization
        double InitialHeigth = 0;
        int Radius = 0;
        bool Optimizaction = 1;

        if (!ParamDatasetName.compare("Terrace")){
            InitialHeigth = 2000; // Terrace Dataset
            if((stoi(FrameNumber) >= 0 && stoi(FrameNumber) <= 500) || (stoi(FrameNumber) >= 4500 && stoi(FrameNumber) <= 5500))
                Radius = 100;
            else
                Radius = 40;
        }
        else if (!ParamDatasetName.compare("PETS2012_S2_L1")){
            InitialHeigth = 1700; // PETS Dataset
            Radius = 60;
        }
        else if (!ParamDatasetName.compare("PETS2012_CC")){
            InitialHeigth = 1700; // PETS Dataset
            Radius = 30;
        }
        else if (!ParamDatasetName.compare("RLC")){
            InitialHeigth = -1700; // RLC Dataset
            if((stoi(FrameNumber) >= 0 && stoi(FrameNumber) <= 275))
                Radius = 30;
            else
                Radius = 75;
        }
        else if (!ParamDatasetName.compare("Wildtrack")){
            InitialHeigth = 175; // Wildtrack Dataset
            Optimizaction = 0; // Variable to do or not optimization in Wildtrack
            Radius = 30;
        }

        // Fill PS initial heigths
        for(size_t i = 0; i < PS.size(); i++){
            PS[i].Heigth3D = InitialHeigth;
        }

        // Preassigment to group points and check if there are blobs alone
        PreAssignment(PS, Radius);

        // Fix those detections that have been clasified as alone
        FixDetections(PS, ForegroundVector, CameraModelVector);

        // Assigment to group points into the final detections
        AssignHeigths(PS, InitialHeigth, Radius, OriginalNPoints, FrameNumber);

        // Save final ground detections to XML in addition to extra low-score detections from each camera
        if(!CameraWorkers[0]->DatasetName.compare("Wildtrack")){
            vector<cvBlob> ExtraDetections;
            for(int n = 0; n < numCams; n++){
                ExtraDetections.insert(ExtraDetections.end(), CameraWorkers[n]->PeopleDetec.BlobStructureLowScores.begin(), CameraWorkers[n]->PeopleDetec.BlobStructureLowScores.end());
            }
            saveGroundDetectionsXML(PS, ExtraDetections, CameraWorkers[0]->CenitalPlane.rows, CameraWorkers[0]->CenitalPlane.cols);
        }

        // Replicate PS structure numCams-1 times.
        PSOriginal = PS;
        for(int n = 0; n < numCams - 1; n++){
            PS.insert(PS.end(), PSOriginal.begin(), PSOriginal.end());
        }
        // Save the original points for distance constrainings replicated
        PSOriginal.clear();
        PSOriginal = PS;

        // Check if Detections are empty and exit function
        if(PS.empty()){
            for(int n = 0; n < numCams; n++){
                CameraWorkers[n]->PeopleDetec.BlobStructureHAP.clear();
                CameraWorkers[n]->HapFinished = 1;
            }
            return;
        }

        /* ---------------- */
        /* DRAW INITIAL BBs */
        /* ---------------- */
        for(int Camera = 0; Camera < numCams; Camera++){
            size_t InitialIndex = OriginalNPoints * Camera;
            size_t FinalIndex = OriginalNPoints * (Camera + 1) - 1;

            int Counter = 0;
            for(size_t i = InitialIndex; i <= FinalIndex; i++){
                Point2f GroundPosition = PS[i].FloorPoint;
                double Height3D = PS[i].Heigth3D;
                int OriginalCamera = PS[i].OriginalCamera;

                bool outside;
                Rect BB = BuildBB(GroundPosition, Height3D, Camera, HomographyVector[Camera], HomographyBetweenViewsVector[Camera], outside);

                rectangle(ImagenAux[Camera], BB.tl(), BB.br(), colorTab[OriginalCamera - 1], 2);
                Counter++;
            }
        }

        String FolderPath = SavingPath + "HAP Method/Iterative/" + FrameNumber + "/";
        const char* path = FolderPath.c_str();
        boost::filesystem::path dir(path);
        boost::filesystem::remove_all(dir);
        boost::filesystem::create_directory(dir);

        for(int n = 0; n < numCams; n++){
            String aux = FolderPath + "Cam" + to_string(n+1) + "_Image" + to_string(j) + ".png";
            imwrite(aux, ImagenAux[n]);
        }
        /* ---------------- */
        /* DRAW INITIAL BBs */
        /* ---------------- */


        /* ----------------- */
        /*   HAP ALGORITHM   */
        /* ----------------- */
        float H0 = 0;
        float H1 = 0;
        vector<float> HVector;
        bool Convergence = 0;

        j++;
        if(Optimizaction){
            H0 = HLossFunction(PS, ForegroundVector, CameraModelVector);
            HVector.push_back(H0);

            //cout << endl;
            while (!Convergence) {
                // cout << "H value: " << to_string(H0) << " iteration " << to_string(j) << endl;
                // Update Tao. Linesearch
                if (j <= 3){
                    TaoX = 5;
                    TaoY = TaoX;
                    TaoH = 3;
                }
                else if (j <= 5){
                    TaoX = 3;
                    TaoY = TaoX;
                    TaoH = 2;
                }
                else{
                    TaoX = 1;
                    TaoY = TaoX;
                    TaoH = 1;
                }

                // Update P, PHeigths and PWidths
                UpdateDetections(PS, PSOriginal, H0, ForegroundVector);

                // Compute new H
                H1 = HLossFunction(PS, ForegroundVector, CameraModelVector);

                /* ---------------- */
                /* DRAW PROCESS BBs */
                /* ---------------- */
                ImageAux2.clear();
                for(int n = 0; n < numCams; n++){
                    Mat Aux;
                    cvtColor(CameraWorkers[n]->Camera.PedestrianMask.clone(), Aux, COLOR_GRAY2BGR);
                    ImageAux2.push_back(Aux);
                }

                for(int Camera = 0; Camera < numCams; Camera++){
                    size_t InitialIndex = OriginalNPoints * Camera;
                    size_t FinalIndex = OriginalNPoints * (Camera + 1) - 1;

                    int Counter = 0;
                    for(size_t i = InitialIndex; i <= FinalIndex; i++){
                        Point2f GroundPositionAux = PS[i].FloorPoint;
                        double WorldHeight = PS[i].Heigth3D;
                        int OriginalCamera = PS[i].OriginalCamera;

                        bool outside;
                        Rect BB = BuildBB(GroundPositionAux, WorldHeight, Camera, HomographyVector[Camera], HomographyBetweenViewsVector[Camera], outside);

                        rectangle(ImageAux2[Camera], BB.tl(), BB.br(), colorTab[OriginalCamera - 1], 2);
                        Counter++;
                    }
                }

                for(int n = 0; n < numCams; n++){
                    String aux = FolderPath + "/Cam" + to_string(n+1) + "_Image" + to_string(j) + ".png";
                    imwrite(aux, ImageAux2[n]);
                }
                /* ---------------- */
                /* DRAW PROCESS BBs */
                /* ---------------- */

                // Update previous H
                H0 = H1;
                HVector.push_back(H0);

                // Convergence check. Minimum 8 iterations
                if(j > 5){
                    float HC1 = *(HVector.rbegin());
                    float HC2 = *(HVector.rbegin() + 2);
                    float HC3 = *(HVector.rbegin() + 4);
                    if(((HC1 == HC2) && (HC1 == HC3)) || j >= 8)
                        Convergence = 1;
                }
                j++;
            }
        }

        /* ---------------------- */
        /*   SAVE FINAL RESULTS   */
        /* ---------------------- */
        ofstream AuxiliarFile;
        AuxiliarFile.open(SavingPath + "Detecciones.txt");

        // DRAW BB ON IMAGE
        ImageAux2.clear();
        vector<Mat> ActualFrameVector;
        for(int n = 0; n < numCams; n++){
            Mat Aux;
            cvtColor(CameraWorkers[n]->Camera.PedestrianMask.clone(), Aux, COLOR_GRAY2BGR);
            ImageAux2.push_back(Aux);

            ActualFrameVector.push_back(CameraWorkers[n]->Camera.ActualFrame.clone());
        }

        for(int Camera = 0; Camera < numCams; Camera++){
            size_t InitialIndex = OriginalNPoints * Camera;
            size_t FinalIndex = OriginalNPoints * (Camera + 1) - 1;

            int Counter = 0;
            for(size_t i = InitialIndex; i <= FinalIndex; i++){
                cvBlob BlobStructure = PS[i];
                int OriginalCamera = BlobStructure.OriginalCamera;

                bool outside;
                BlobStructure.Blob = BuildBB(BlobStructure.FloorPoint, BlobStructure.Heigth3D, Camera, HomographyVector[Camera], HomographyBetweenViewsVector[Camera], outside);
                if(BlobStructure.Blob.height > 10 && BlobStructure.Blob.width > 10){
                    CameraWorkers[Camera]->PeopleDetec.BlobStructureHAP.push_back(BlobStructure);
                    rectangle(ImageAux2[Camera], BlobStructure.Blob.tl(), BlobStructure.Blob.br(), colorTab[OriginalCamera - 1], 2);
                    //rectangle(ActualFrameVector[Camera], BlobStructure.Blob.tl(), BlobStructure.Blob.br(), colorTab[Counter], 2);
                    rectangle(CameraWorkers[Camera]->Camera.ActualFrame, BlobStructure.Blob.tl(), BlobStructure.Blob.br(), colorTab[Counter], 3);

                    AuxiliarFile << BlobStructure.FloorPoint << colorTab[Counter] << endl;
                }
                Counter++;
            }
        }

        AuxiliarFile.close();

        for(int n = 0; n < numCams; n++){
            String aux = FolderPath + "Cam" + to_string(n+1) + "_Image" + to_string(j) + ".png";
            imwrite(aux, ImageAux2[n]);

            CameraWorkers[n]->Camera.ProjectCommonSemantic(CameraWorkers[n]->Camera.ActualFrame);
            String aux2 = SavingPath + "HAP Method/Camera " + to_string(n+1) + " RGB/Image" + FrameNumber + ".png";
            //imwrite(aux2, ActualFrameVector[n]);
            imwrite(aux2, CameraWorkers[n]->Camera.ActualFrame);
        }

        // HAP Finished
        for(int n = 0; n < numCams; n++)
            CameraWorkers[n]->HapFinished = 1;

        return;
    }
}

void MainWindow::UpdateDetections(vector<cvBlob> &PS, const vector<cvBlob> &PSOriginal, float H, vector<Mat> ForegroundVector)
{
    float EpsilonX = 5;
    float EpsilonY = 5;
    //    float EpsilonH = 20;
    float EpsilonH = 2;
    float mu = 0.0000000000000001;

    // Gradient Variables
    float GradientXY, GradientH;

    // Clear previous vectors
    VectorNormX.clear();
    VectorNormY.clear();
    VectorNormH.clear();

    Point2f pointCombination[] = {
        Point2f(mu,-EpsilonY),
        Point2f(EpsilonX,-EpsilonY),
        Point2f(EpsilonX,mu),
        Point2f(EpsilonX,EpsilonY),
        Point2f(mu,EpsilonY),
        Point2f(-EpsilonX,EpsilonY),
        Point2f(-EpsilonX,mu),
        Point2f(-EpsilonX,-EpsilonY)};

    // Compute gradient for X and Y coordinates
    for(size_t i = 0; i < PS.size(); i++){
        Point2f Aux;
        vector<cvBlob> PAuxX;
        int Index;
        float GFinal = -10000000;

        for(int c = 0; c < 8; c++){
            Aux = pointCombination[c];

            // Forward
            //PAuxX.clear();
            PAuxX = PS;
            PAuxX[i].FloorPoint = PS[i].FloorPoint + Aux;
            float HAux_plus = HLossFunction(PAuxX, ForegroundVector, CameraModelVector);

            // Backward
            //PAuxX.clear();
            PAuxX = PS;
            PAuxX[i].FloorPoint = PS[i].FloorPoint - Aux;
            float HAux_minus = HLossFunction(PAuxX, ForegroundVector, CameraModelVector);

            GradientXY = HAux_plus - HAux_minus;

            if(GradientXY > GFinal){
                GFinal = GradientXY;
                Index = c;
            }
        }
        Point2f Final = pointCombination[Index];
//        VectorNormX.push_back(Final.x/EpsilonX);
//        VectorNormY.push_back(Final.y/EpsilonX);
        VectorNormX.push_back(Final.x / abs(Final.x));
        VectorNormY.push_back(Final.y / abs(Final.y));
    }

    // Compute gradient norms for Heigth
    int NDetections = 0;
    for(size_t n = 0; n < PS.size(); n++){
        if(PS[n].PID >= NDetections)
            NDetections = PS[n].PID;
    }

    for(int Detection = 0; Detection <= NDetections; Detection++){
        vector<cvBlob> PHeigthAux = PS;
        for(size_t i = 0; i < PS.size(); i++){
            if(PS[i].PID == Detection)
                PHeigthAux[i].Heigth3D = PS[i].Heigth3D + EpsilonH;
        }

        float HAuxH = HLossFunction(PHeigthAux, ForegroundVector, CameraModelVector);
        GradientH = ((HAuxH - H)) / EpsilonH;
        if(GradientH == 0)
            VectorNormH.push_back(0);
        else
            VectorNormH.push_back(GradientH / abs(GradientH));
    }

    // Update the new values
    for(int Camera = 0; Camera < numCams; Camera++){
        size_t InitialIndex = OriginalNPoints * Camera;
        size_t FinalIndex = OriginalNPoints * (Camera + 1) - 1;

        for(size_t i = InitialIndex; i <= FinalIndex; i++){
            Point2f Displacement;
            Displacement.x = (TaoX * VectorNormX[i]);
            Displacement.y = (TaoY * VectorNormY[i]);

            Point2f NewGroundPoint;
            //int DistanceConstraint = 20;
            int DistanceConstraint = 50;
            Point Pos1;
            Pos1.x = PS[i].FloorPoint.x + Displacement.x;
            Pos1.y = PS[i].FloorPoint.y + Displacement.y;

            if(abs(Pos1.x - PSOriginal[i].FloorPoint.x) <= DistanceConstraint){
                // Compute posible new ground point
                NewGroundPoint.x = PS[i].FloorPoint.x + Displacement.x;
            }
            else
                NewGroundPoint.x = PS[i].FloorPoint.x - Displacement.x;
            if(abs(Pos1.y - PSOriginal[i].FloorPoint.y) <= DistanceConstraint){
                // Compute posible new ground point
                NewGroundPoint.y = PS[i].FloorPoint.y + Displacement.y;
            }
            else
                NewGroundPoint.y = PS[i].FloorPoint.y - Displacement.y;

            // Compute posible new 3D heigth
            float NewHeigth = PS[i].Heigth3D + (TaoH * VectorNormH[PS[i].PID]);

            // Check if the new updated blob is coherent
            bool Outside;
            Rect BB = BuildBB(NewGroundPoint, NewHeigth, Camera, HomographyVector[Camera], HomographyBetweenViewsVector[Camera], Outside);

            // If the top left corner is outside the image reduce the heigth
            if(Outside){
                VectorNormH[PS[i].PID] = -1;
                // Compute new 3D heigth
                NewHeigth = PS[i].Heigth3D + (TaoH * VectorNormH[PS[i].PID]);
            }
            PS[i].FloorPoint = NewGroundPoint;
            PS[i].Heigth3D = NewHeigth;
        }
    }
}

void MainWindow::updateVariables(Mat Frame, Mat CenitalPlane, int CameraNumber)
{
    if (CameraNumber == 1){
        // Update messages only for one camera
        if(CameraWorkers[CameraNumber-1]->CBOption.compare(ui->PeopleDetectorCB->currentText().toStdString()))
            ui->textBrowser->append(ui->PeopleDetectorCB->currentText() + " People Detector in use");
    }

    // Widget size variables
    CameraWorkers[CameraNumber-1]->WidgetWidth  = ui->CVWidget1->width();
    CameraWorkers[CameraNumber-1]->WidgetHeight = ui->CVWidget1->height();

    // People detection options
    CameraWorkers[CameraNumber-1]->CBOption = ui->PeopleDetectorCB->currentText().toStdString();
    if(!CameraWorkers[CameraNumber-1]->CBOption.compare("Semantic Detector"))
        CameraWorkers[CameraNumber-1]->SemanticFiltering = 1;
    else
        CameraWorkers[CameraNumber-1]->SemanticFiltering = ui->PDFiltering->isChecked();

    // People detection representation methods
    CameraWorkers[CameraNumber-1]->RepresentationOption = ui->RepresentationCB->currentText().toStdString();

    // Pedestrian score threshold
    CameraWorkers[CameraNumber-1]->PeopleDetec.Threshold = ui->PDThreshold->value();

    // Pedestrian filtering with multicamera or semantic information
    CameraWorkers[CameraNumber-1]->MultiCameraFiltering = ui->MultiCameraFiltering->isChecked();
    CameraWorkers[CameraNumber-1]->SemanticFiltering = ui->SemanticFiltering->isChecked();
}

void MainWindow::displayFrame(Mat Frame, Mat CenitalPlane, int CameraNumber)
{
    if (CameraNumber == 1) {
        // Camera 1
        ui->CVWidget1->showImage(Frame);
    }
    else if (CameraNumber == 2) {
        // Camera 2
        ui->CVWidget2->showImage(Frame);
    }
    else if (CameraNumber == 3) {
        // Camera 3
        ui->CVWidget3->showImage(Frame);
        // Cenital Plane is only displayed once
        ui->CVWidgetCenital->showImage(CenitalPlane);
    }
}

void MainWindow::on_actionAbout_triggered()
{
    // Funcion to display the About Window from the interface
    AboutDialog Aboutwindow;
    Aboutwindow.setModal(true);
    Aboutwindow.exec();
}
