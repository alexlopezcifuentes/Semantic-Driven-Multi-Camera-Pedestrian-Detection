#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cameraworker.h"
#include "camerastream.h"
#include "CameraModel/cameraModel.h"
#include "basicblob.hpp"

#define MAX_NUM_CAM 7

using namespace cv;

namespace Ui {
class MainWindow;
}

class QThread;
class QLabel;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(String Mode, String ParamDatasetNameCons, String ParamPDDetectorCons, bool ParamSemanticFitleringCons, bool ParamMultiCameraCons, QWidget *parent = 0);
    ~MainWindow();

    void threadStarting();
    void connectSignals2Slots(QThread *thread, CameraWorker *worker);
    void fillCameraVariables(CameraStream &Camera, int i);

    // Mode of operation
    String Mode;

    // Camera number of sampled views
    int NViews = 9;

    // Program Parameter Variables
    String ParamDatasetName, ParamPDDetector;
    double ParamPDThreshold;
    bool ParamSemanticFitlering, ParamMultiCamera;

    // OpenCV Calibration Variables
    vector<Mat> CameraMatrixVector, DistortionVector;
    vector<vector<float>> RVec, TVec;
    // OpenCV Calibration reading function
    void LoadOpenCVCalibration(String ExtrinsicPath, String IntrinsicPath);

    // HAP Algorithm Variables
    int OriginalNPoints;
    int ImageRows, ImageCols;
    Mat CommonPeople;
    vector<Mat> HomographyVector, HomographyBetweenViewsVector;
    Mat HomographyIter, HomographyBetweenViewsIter;
    vector<float> VectorNormX, VectorNormY, VectorNormH;
    float TaoX, TaoY, TaoH;

    // HAP Algorithm Functions
    void GraphConnectedComponents(vector<cvBlob> &PS, int Radius, int &GroupCounter);
    vector<Rect> ObtainBBFromP(vector<cvBlob> &PS, int CameraNumber);
    void UpdateDetections(vector<cvBlob> &PS, const vector<cvBlob> &PSOriginal, float H, vector<Mat> ForegroundVector);
    float HLossFunction(const vector<cvBlob> &PS, const vector<Mat> &ForegroundVector, const vector<Etiseo::CameraModel> &CameraModelVector);
    float ExtractHNumerator(Mat FG, const vector<Rect> &BBVector);
    float LhFunction(int row, int col, const vector<Rect> &BBVector, int PixelV, double maxFG);
    void PreAssignment(vector<cvBlob> &PS, int Radius);
    void AssignHeigths(vector<cvBlob> &PS, double InitialHeigth, int Radius, int &NDetections, String FrameNumber);
    void FixDetections(vector<cvBlob> &PS, const vector<Mat> &ForegroundVector, const vector<Etiseo::CameraModel> &CameraModelVector);
    Rect BuildBB(Point2f GroundPosition, float WorldHeight, int CameraNumber, Mat HomographyIter, Mat HomographyBetweenViewsIter, bool &Outside);

    // Saving function and variables
    void saveGroundDetectionsXML(vector<cvBlob> PS, vector<cvBlob> ExtraDetections, int GroundHeigth, int GroundWidth);
    FileStorage fs;
    bool OpeningFlag = 1;

signals:

private slots:
    void HAPAlgorithm(int CameraNumber, String FrameNumber);
    void updateVariables(Mat Frame, Mat CenitalPlane, int CameraNumber);
    void displayFrame(Mat Frame, Mat CenitalPlane, int CameraNumber);
    void on_actionAbout_triggered();

    void on_actionOpen_File_triggered();

private:
    Ui::MainWindow *ui;
    // Number of cameras that will be used
    int numCams = MAX_NUM_CAM;
    QLabel *labels[MAX_NUM_CAM];
    QThread* threads[MAX_NUM_CAM];
    CameraWorker* CameraWorkers[MAX_NUM_CAM];
    String VideoPaths[MAX_NUM_CAM];
    QStringList filenames;
    QString GlobalPath;
};

#endif // MAINWINDOW_H
