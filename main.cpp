#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // SELECT BETWEEN GUI OR PARAMETERS MODE
    //String Mode = "GUI";
    String Mode = "Parameters";

    std::cout << cv::getBuildInformation() << std::endl;

    // Variables for Paramters Mode
    String ParamDatasetName, ParamPDDetector;
    bool ParamSemanticFitlering, ParamMultiCamera;

    if(!Mode.compare("Parameters")){
        // We expect 5 arguments: DatasetName, Pedestrian Detector, Semantic Filtering Enable, Multicamera Fusion Enable.
        if (argc < 5){
            std::cerr << "Usage: " << argv[0] << ". Not enough arguments for the program" << std::endl;
            return 1;
        }

        // Dataset (First Argument)
        ParamDatasetName = argv[1];
        // People Detector (Second Argument)
        ParamPDDetector = argv[2];
        // Semantic Filtering (Third Argument)
        istringstream(argv[3]) >> ParamSemanticFitlering;
        // Multicamera Fusion (Fourth Argument)
        istringstream(argv[4]) >> ParamMultiCamera;
    }

    MainWindow w(Mode,ParamDatasetName, ParamPDDetector, ParamSemanticFitlering, ParamMultiCamera);

    if (!Mode.compare("GUI")){
        w.show();
    }

    return a.exec();
}
