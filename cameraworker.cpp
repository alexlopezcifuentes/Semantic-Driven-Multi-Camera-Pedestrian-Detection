#include "cameraworker.h"
#include "camerastream.h"
#include "barrier.h"
#include "peopledetector.h"
#include <QDebug>
#include <string>
#include <QThread>
#include <iostream>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "basicblob.hpp"

using namespace cv;
using namespace std;

CameraWorker::CameraWorker(CameraStream Camera, Barrier barrier, String ParamDatasetName, String ParamPDDetector, bool ParamSemanticFitlering, bool ParamMultiCamera, int NumCams, String Mode) :
    Camera(Camera), barrier(barrier), DatasetName(ParamDatasetName),CBOption(ParamPDDetector), MultiCameraFiltering(ParamMultiCamera), SemanticFiltering(ParamSemanticFitlering), NumCams(NumCams), Mode(Mode){
}

CameraWorker::~CameraWorker(){}

void CameraWorker::preProcessVideo()
{
    String DetectionsPath;

    if(Camera.CameraNumber == 1){
        cout << endl;
        cout << endl;
        cout << "VIDEO PROCESSING INFORMATION: " << endl;
        cout << endl;
        cout << "Processing Dataset: " << DatasetName << endl;
        cout << "Pedestrian Detector: " << CBOption << " People Detector." << endl;
        if(SemanticFiltering & MultiCameraFiltering){
            cout << "Using Multicamera and semantic" << endl;
            cout << endl;
        }
        else if(MultiCameraFiltering){
            cout << "Using Multicamera" << endl;
            cout << endl;
        }
        else if(SemanticFiltering){
            cout << "Using semantic" << endl;
            cout << endl;
        }
        else{
            cout << "Raw Detector" << endl;
            cout << endl;
        }
        cout << "Processing the following video path:" << endl;
        cout << Camera.InputPath << endl;
    }

    // Check Dataset name in order to know if there is panning or not. Only video sequences from EPS_Hall Dataset use panning
    if(DatasetName.find("EPS_Hall") != string::npos){
        Camera.CameraPanning = 1;
    }

    if(SemanticFiltering & MultiCameraFiltering){
        DetectionsPath = "/home/alex/Desktop/IPCV-MasterThesis-master/Matlab/Evaluation Code/Video/" + DatasetName + "/4MultiSemantic/BoundingBoxes "+ CBOption + " " + to_string(Camera.CameraNumber) + ".idl";
    }
    else if(MultiCameraFiltering){
        DetectionsPath = "/home/alex/Desktop/IPCV-MasterThesis-master/Matlab/Evaluation Code/Video/" + DatasetName + "/2Multi/BoundingBoxes "+ CBOption + " " + to_string(Camera.CameraNumber) + ".idl";
    }
    else if(SemanticFiltering){
        DetectionsPath = "/home/alex/Desktop/IPCV-MasterThesis-master/Matlab/Evaluation Code/Video/" + DatasetName + "/3Semantic/BoundingBoxes "+ CBOption + " " + to_string(Camera.CameraNumber) + ".idl";
    }
    else{
        DetectionsPath = "/home/alex/Desktop/IPCV-MasterThesis-master/Matlab/Evaluation Code/Video/" + DatasetName + "/1Raw/BoundingBoxes "+ CBOption + " " + to_string(Camera.CameraNumber) + ".idl";
    }
    cout << "Detection results path: " << DetectionsPath << endl;

    // Create and open the file to save detections
    PeopleDetec.BoundingBoxesFile.open(DetectionsPath);

    // Open video file
    Camera.VideoOpenning(NumCams);

    // Create and open the statistics file
    VideoStatsFile.open("/home/alex/Desktop/IPCV-MasterThesis-master/MultithreadParameters/VideoProcessingStats" + to_string(Camera.CameraNumber) + ".txt");
    VideoStatsFile << "Frame  Computational Time" << endl;

    // Compute camera homographies by reading view points from files
    Camera.computeHomography();

    if(Camera.CameraPanning){
        // Compute AKAZE points for camera views
        Camera.AkazePointsForViewImages();
    }

    // Extract common projected semantic points from offline images
    Camera.SemanticCommonPoints(DatasetName);

    // Main video processing function
    barrier.wait();
    processVideo();
}

void CameraWorker::processVideo()
{
    double AccumulatedTime = 0;

    // Sizes of Ground Plane Images
    // - Terrace 700x700
    // - PETS 1500x1500
    // - APIDIS 950x950
    // - EPS HALL 1606×986
    // - RLC 1200x1300
    int Cols, Rows;
    if (!DatasetName.compare("Terrace")){
        Cols = 700;
        Rows = 700;
    }
    else if(DatasetName.find("PETS2012") != string::npos){
        Cols = 1500;
        Rows = 1500;
    }
    else if(!DatasetName.compare("APIDIS")){
        Cols = 950;
        Rows = 950;
    }
    else if(!DatasetName.compare("RLC")){
        Cols = 1300;
        Rows = 1200;
    }
    else if(DatasetName.find("EPS_Hall") != string::npos){
        Cols = 1606;
        Rows = 986;
    }
    else if(!DatasetName.compare("Wildtrack")){
        Cols = 2500;
        Rows = 2000;
    }

    // Main Video Loop
    while (true) {
        // Start the clock for measuring time consumption/frame
        QElapsedTimer timer;
        timer.start();

        // Read ActualFrame from video
        Camera.cap >> Camera.ActualFrame;

        // Check if video end is reached
        if(Camera.ActualFrame.empty()) {
            cout << "Camera " << to_string(Camera.CameraNumber) << " Video finished" << endl;
            // Close txt files
            VideoStatsFile.close();
            SelectedViewsFile.close();
            PeopleDetec.BoundingBoxesFile.close();
            barrier.wait();
            exit(EXIT_FAILURE);
            break;
        }

        // Get frame number
        stringstream ss;
        ss << Camera.cap.get(CAP_PROP_POS_FRAMES);
        String FrameNumber = ss.str().c_str();

        if(Camera.CameraNumber == 1)
            cout << "\r" << "Processing Frame: " << FrameNumber << ". ";

        /* -----------------------*/
        /*      MAIN ALGORITHM    */
        /* -----------------------*/
        if(stoi(FrameNumber) < 5){
            // Load actual semantic frame from the folder
            Camera.getActualSemFrame(FrameNumber);

            // -------------------------------- //
            //   MASK EXTRACTION AND FILTERING  //
            // -------------------------------- //
            if(MultiCameraFiltering){
                // Extract a pedestrian binary mask for a given semantic segmentation label image
                Camera.extractPDMask(Camera.ActualSemFrame, DatasetName);

                // Extract blobs from the previous mask images. ¿Only used for PSPNet detetor?
//                Camera.extractFGBlobs(Camera.PedestrianMask, CBOption);
            }

            // ------------------------------ //
            //   HOMOGRAPHY & VIEW SELECTION  //
            // ------------------------------ //
            ImageWarping = Mat::zeros(Rows, Cols, CV_8UC3);
            if(Camera.CameraPanning){
                // Process to select the view and compute HomographyBetweenViews
                Camera.ViewSelection(Camera.HomographyVector);

                // Select the view from the views txt file if previously saved
                //Camera.ViewSelectionFromTXT(Camera.HomographyVector, FrameNumber, DatasetName);
            }
            else{
                // For Non Panning Datasets (almost every one), Homogrpahy is just the homography to
                // ground plane
                Camera.Homography = Camera.HomographyVector[0];
                Camera.HomographyBetweenViews = Mat::eye(3, 3, Camera.Homography.type());
            }

            // Project RGB Images to the cenital plane and saves them to folder
            // This is used later to compute the Mean RGB Map
            //Camera.saveWarpImages(Camera.ActualFrame, FrameNumber, ImageWarping);

            // ----------------------- //
            //   SEMANTIC PROJECTION   //
            // ----------------------- //
            // Clear Cenital Plane
            CenitalPlane = Mat::zeros(Rows, Cols, CV_8UC3);
            if(MultiCameraFiltering || SemanticFiltering){
                // Project Semantic Images to the cenital plane and saves them to folder
                // This is used later to compute the Mean Semantic Ground Map
                //Camera.ProjectSemanticPoints(CenitalPlane, FrameNumber);
            }

            // --------------------------------------------------- //
            //     PEOPLE DETECTION & BLOBS PROJECTION & FUSION    //
            // --------------------------------------------------- //
            PeopleDetec.MainPeopleDetection(Camera, DatasetName, FrameNumber, CBOption, MultiCameraFiltering, SemanticFiltering);
            if(MultiCameraFiltering){
                barrier.wait();

                // Emit signal to start the HAP method in the main thread
                emit HAPAlgorithmSignal(Camera.CameraNumber, FrameNumber);

                // Infinite loop until HAP is ended
                while(!HapFinished){}
                HapFinished = 0;

                // Delete or reduce score of the bounding boxes that do not contain semantic
                Camera.CheckSemanticInBlobs(PeopleDetec.BlobStructureHAP, Camera.PedestrianMask);

                // Perform Again Sematnic Constraining ?¿?¿¿
                PeopleDetec.SemanticConstraining(PeopleDetec.BlobStructureHAP, Camera.ActualFrame, Camera.VideoPath);

                // Copy of ActualFrame to draw Bounding Boxes
                Mat ActualFrameCopy;
                Camera.ActualFrame.copyTo(ActualFrameCopy);
                if(!PeopleDetec.BlobStructureHAP.empty()){
                    // Draw final detections from HAP + WnMS
                    PeopleDetec.paintBoundingBoxes(ActualFrameCopy, PeopleDetec.BlobStructureHAP, Camera.CameraNumber, 2);
                }
                // Draws the common floor into the actual frame
                Camera.ProjectCommonSemantic(ActualFrameCopy);
                String SavingPath = "/home/alex/Desktop/IPCV-MasterThesis-master/HAP Method/Camera " + to_string(Camera.CameraNumber) + "/Image" + FrameNumber + ".png";
                imwrite(SavingPath, ActualFrameCopy);
            }

            // -------------------- //
            //     BLOBS SAVING     //
            // -------------------- //
            if(MultiCameraFiltering){
                PeopleDetec.BlobStructureHAP.insert(PeopleDetec.BlobStructureHAP.end(), PeopleDetec.BlobStructureLowScores.begin(), PeopleDetec.BlobStructureLowScores.end());
                PeopleDetec.blobSavingTXT(PeopleDetec.BlobStructureHAP, FrameNumber, Camera.CameraNumber);
            }
            else if(SemanticFiltering){
                PeopleDetec.BlobStructurePDNMS.insert(PeopleDetec.BlobStructurePDNMS.end(), PeopleDetec.BlobStructureLowScores.begin(), PeopleDetec.BlobStructureLowScores.end());
                PeopleDetec.blobSavingTXT(PeopleDetec.BlobStructurePDNMS, FrameNumber, Camera.CameraNumber);
            }
            else
                PeopleDetec.blobSavingTXT(PeopleDetec.BlobStructurePDNMS, FrameNumber, Camera.CameraNumber);
        }

        // Compute the processing time (sec) per frame
        double seconds = timer.elapsed();
        seconds /= 1000;

        // Compute the average time
        AccumulatedTime += seconds;

        // Save measures to .txt file
        VideoStatsFile << FrameNumber << "       " << seconds << endl;

        // cout << "\r" << "Processing Frame: " << FrameNumber << ". Average Time: " << (AccumulatedTime/stod(FrameNumber)) << endl;
        if(Camera.CameraNumber == 1)
            cout << "Frame Time: " << seconds << "s. Average Time: " << (AccumulatedTime/stod(FrameNumber)) << "s." << endl;

        // Threads must wait here until all of them have reached the barrier
        barrier.wait();
        if(!Mode.compare("GUI")){
            Mat ActualFrameAux;
            cv::resize(Camera.ActualFrame, ActualFrameAux, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
            emit frameFinished(ActualFrameAux, CenitalPlane, Camera.CameraNumber);
        }
    }
    emit finished();
}
