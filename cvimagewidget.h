#pragma once
#include <QWidget>
#include <QImage>
#include <QPainter>
#include <opencv2/opencv.hpp>
#include <unistd.h>

using namespace cv;

class CVImageWidget : public QWidget
{
    Q_OBJECT

public:

    explicit CVImageWidget(QWidget *parent = 0) : QWidget(parent) {}

    QSize sizeHint() const { return qImage.size(); }
    QSize minimumSizeHint() const { return qImage.size(); }

public slots:

    void showImage(const Mat& image) {
        // Convert the image to the RGB888 format
        switch (image.type()) {
        case CV_8UC1:
            cvtColor(image, tmpImage, COLOR_GRAY2RGB);
            break;
        case CV_8UC3:
            cvtColor(image, tmpImage, COLOR_BGR2RGB);
            break;
        }

        // QImage needs the data to be stored continuously in memory
        assert(tmpImage.isContinuous());
        // Assign OpenCV's image buffer to the QImage. Note that the bytesPerLine parameter
        // (http://qt-project.org/doc/qt-4.8/qimage.html#QImage-6) is 3*width because each pixel
        // has three bytes.
        qImage = QImage(tmpImage.data, tmpImage.cols, tmpImage.rows, tmpImage.cols*3, QImage::Format_RGB888);
        repaint();
    }

protected:

    void paintEvent(QPaintEvent* /*event*/) {
        // Display the image
        QPainter painter(this);
        painter.drawImage(QPoint(0,0), qImage);
        painter.end();
    }

    QImage qImage;
    Mat tmpImage;
};
