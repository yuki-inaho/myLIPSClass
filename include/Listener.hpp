#include <iostream>
#include <stdexcept>
#include <mutex>

#include <pcl/io/pcd_io.h>
#include <pcl/common/common_headers.h>
#include <PS1080.h>
#include <OpenNI.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <armadillo>

using namespace cv;
using namespace std;
using namespace openni;

class MyListener 
{
public:    
    void initialize()
    {
        // デバイスを取得する

        openni::Status ret = device.open( openni::ANY_DEVICE );
        if ( ret != openni::STATUS_OK ) {
            throw std::runtime_error( "openni::Device::open() failed." );
        }

        if( !device.isImageRegistrationModeSupported( openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR ) )
        {
            throw std::runtime_error( "ERROR: ImageRegistration mode is not supported" );
        }
        device.setDepthColorSyncEnabled(true);
        device.setImageRegistrationMode( IMAGE_REGISTRATION_DEPTH_TO_COLOR );

        // カラーストリームを有効にする
        colorStream.create( device, openni::SENSOR_COLOR );
        colorStream.setMirroringEnabled(false);    
        changeResolution( colorStream );


        irStream.create( device, openni::SENSOR_IR );
        irStream.start();
        changeResolution( irStream );

        // Depth ストリームを有効にする
        depthStream.create( device, openni::SENSOR_DEPTH );
        depthStream.setMirroringEnabled(false);
        changeResolution( depthStream );

//        cout << depthStream.isCroppingSupported() << endl; 
//        cout << depthStream.getMaxPixelValue() << endl; 
        pcl::PointCloud<pcl::PointXYZ>::Ptr _pc (new pcl::PointCloud<pcl::PointXYZ>);
        pc_dat = _pc;
        colorStream.start();
        irStream.start();
        depthStream.start();

        auto colorSettings = colorStream.getCameraSettings();
        if(colorSettings != 0){
            colorSettings->setAutoExposureEnabled(true);
            colorSettings->setAutoWhiteBalanceEnabled(true);
        }

    }

    void update()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        openni::VideoFrameRef colorFrame;
        openni::VideoFrameRef depthFrame;
        openni::VideoFrameRef irFrame;

        // 更新されたフレームを取得する
        if ( colorStream.isValid() ) {
            if ( STATUS_OK == colorStream.readFrame( &colorFrame) ) {
                colorImage = cv::Mat( colorFrame.getHeight(), colorFrame.getWidth(), CV_8UC3, (void*)colorFrame.getData() );
            }
        }

        if ( depthStream.isValid() ) {
            if ( STATUS_OK == depthStream.readFrame( &depthFrame) ) {
                depthImage = cv::Mat( depthFrame.getHeight(), depthFrame.getWidth(), CV_16UC1, (void*)depthFrame.getData() );
            }
        }
         
        if ( irStream.isValid() ) {
            if ( STATUS_OK == irStream.readFrame( &irFrame) ) {
                irImage = cv::Mat( irFrame.getHeight(), irFrame.getWidth(), CV_16UC1, (void*)irFrame.getData() );
            }
        }
        irImage.convertTo( irImage, CV_8U );

        if( !depthImage.empty() && !colorImage.empty() )
        {
            pc_dat = niComputeCloud( depthImage, depthStream );
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr 
    getPointCloud()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        return pc_dat;
    }

    cv::Mat
    getColorImage()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        return colorImage;
    }

    cv::Mat
    getDepthImage()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        return depthImage;
    }

    cv::Mat
    getIRImage()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        return irImage;
    }

    cv::Mat
    getPointIndexMat()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        return PointIndexMat;
    }

    void close()
    {
        colorStream.destroy();  // カラーストリーム
        depthStream.destroy();  // Depth ストリーム
        irStream.destroy();  // Depth ストリーム
        device.close();
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr 
    _niComputeCloud( const cv::Mat depthMap)
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        const DepthPixel* pDepthArray = (const DepthPixel*)depthMap.data;
        pcl::PointCloud<pcl::PointXYZ>::Ptr _pc (new pcl::PointCloud<pcl::PointXYZ>);
        cv::Mat PointIndexMat_tmp = Mat::zeros(depthMap.rows * depthMap.cols, 3, CV_16UC1);
        int count =0;
        for( int y = 0; y < depthMap.rows; y++ )
        {
            for( int x = 0; x < depthMap.cols; x++ )
            {
                if(std::isfinite(depthMap.ptr(y)[x]) ){
//                if(true){
                    pcl::PointXYZ _point;
                    float fX, fY, fZ;
                    CoordinateConverter::convertDepthToWorld( depthStream,
                                                            x, y, *pDepthArray++,
                                                            &fX, &fY, &fZ );
                    _point.x = double(fX)/1000;
                    _point.y = double(fY)/1000;
                    _point.z = double(fZ)/1000;
                    _pc->points.push_back(_point);
                    PointIndexMat_tmp.at<unsigned short>(count,0) = x;
                    PointIndexMat_tmp.at<unsigned short>(count,1) = y;
                    PointIndexMat_tmp.at<unsigned short>(count,2) = 0;
                    count ++;
                }else{
                    *pDepthArray++;
                }
            }
        }
        PointIndexMat = Mat::zeros(count, 3, CV_16UC1);
        for(int i=0;i<count;i++){
            PointIndexMat.at<unsigned short>(i,0) = PointIndexMat_tmp.at<unsigned short>(i,0);
            PointIndexMat.at<unsigned short>(i,1) = PointIndexMat_tmp.at<unsigned short>(i,1);
            PointIndexMat.at<unsigned short>(i,2) = PointIndexMat_tmp.at<unsigned short>(i,2);
        }

        return(_pc);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr 
    _niComputeCloud2d( const cv::Mat depthMap)
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        const DepthPixel* pDepthArray = (const DepthPixel*)depthMap.data;
        pcl::PointCloud<pcl::PointXYZ>::Ptr _pc (new pcl::PointCloud<pcl::PointXYZ>);
        cv::Mat PointIndexMat_tmp = Mat::zeros(depthMap.rows * depthMap.cols, 3, CV_16UC1);
        int count =0;

        _pc->width = depthMap.cols;
        _pc->height = depthMap.rows;
        _pc->points.resize( _pc->width * _pc->height );

        for( int y = 0; y < depthMap.rows; y++ )
        {
            for( int x = 0; x < depthMap.cols; x++ )
            {
                if(std::isfinite(depthMap.ptr(y)[x]) && depthMap.ptr(y)[x]>0 ){
//                if(true){
                    pcl::PointXYZ _point;
                    float fX, fY, fZ;
                    CoordinateConverter::convertDepthToWorld( depthStream,
                                                            x, y, *pDepthArray++,
                                                            &fX, &fY, &fZ );
                    _point.x = double(fX)/1000;
                    _point.y = double(fY)/1000;
                    _point.z = double(fZ)/1000;
                    //_pc->points.push_back(_point);
                    _pc->points[x + y * _pc->width].x = _point.x;
                    _pc->points[x + y * _pc->width].y = _point.y;
                    _pc->points[x + y * _pc->width].z = _point.z;                    
                    PointIndexMat_tmp.at<unsigned short>(count,0) = x;
                    PointIndexMat_tmp.at<unsigned short>(count,1) = y;
                    PointIndexMat_tmp.at<unsigned short>(count,2) = 0;
                    count ++;
                }else{
                    *pDepthArray++;
                }
            }
        }
        PointIndexMat = Mat::zeros(count, 3, CV_16UC1);
        for(int i=0;i<count;i++){
            PointIndexMat.at<unsigned short>(i,0) = PointIndexMat_tmp.at<unsigned short>(i,0);
            PointIndexMat.at<unsigned short>(i,1) = PointIndexMat_tmp.at<unsigned short>(i,1);
            PointIndexMat.at<unsigned short>(i,2) = PointIndexMat_tmp.at<unsigned short>(i,2);
        }
        return(_pc);
    }


    arma::mat getXYMat()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        return xy_mat;
    }

    vector<int> x_vec, y_vec;    
    arma::mat xy_mat;
    

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr 
    niComputeCloud( const cv::Mat depthMap, const VideoStream &depthStream)
    {

        const DepthPixel* pDepthArray = (const DepthPixel*)depthMap.data;
        cv::Mat PointIndexMat_tmp = Mat::zeros(depthMap.rows * depthMap.cols, 3, CV_16UC1);
        x_vec.clear();
        y_vec.clear();

        pcl::PointCloud<pcl::PointXYZ>::Ptr _pc (new pcl::PointCloud<pcl::PointXYZ>);
        int count = 0;
        for( int y = 0; y < depthMap.rows; y++ )
        {
            for( int x = 0; x < depthMap.cols; x++ )
            {
                //if(std::isfinite(depthMap.ptr(y)[x]) && depthMap.ptr(y)[x]>0 ){
                if(true){
                    pcl::PointXYZ _point;
                    float fX, fY, fZ;
                    CoordinateConverter::convertDepthToWorld( depthStream,
                                                            x, y, *pDepthArray++,
                                                            &fX, &fY, &fZ );
                    _point.x = double(fX)/1000;
                    _point.y = double(fY)/1000;
                    _point.z = double(fZ)/1000;
                    _pc->points.push_back(_point);
                    PointIndexMat_tmp.at<unsigned short>(count,0) = x;
                    PointIndexMat_tmp.at<unsigned short>(count,1) = y;
                    PointIndexMat_tmp.at<unsigned short>(count,2) = 0;
                    x_vec.push_back(x);
                    y_vec.push_back(y);

                    count ++;
                }else{
                    *pDepthArray++;
                }
            }
        }

        xy_mat = arma::zeros(y_vec.size(), 2);
        for(int i=0;i<x_vec.size();i++){
            xy_mat(i,0) = x_vec[i];
            xy_mat(i,1) = y_vec[i];
        }

        PointIndexMat = Mat::zeros(count, 3, CV_16UC1);
        for(int i=0;i<count;i++){
            PointIndexMat.at<unsigned short>(i,0) = PointIndexMat_tmp.at<unsigned short>(i,0);
            PointIndexMat.at<unsigned short>(i,1) = PointIndexMat_tmp.at<unsigned short>(i,1);
            PointIndexMat.at<unsigned short>(i,2) = PointIndexMat_tmp.at<unsigned short>(i,2);
        }
        return(_pc);
    }

    void 
    changeResolution( openni::VideoStream& stream )
    {
        openni::VideoMode mode = stream.getVideoMode();
        mode.setResolution( 320, 240 );
        mode.setFps( 30 );
        stream.setVideoMode( mode );
    }

    void 
    changeHighResolution( openni::VideoStream& stream )
    {
        openni::VideoMode mode = stream.getVideoMode();
        mode.setResolution( 640, 480 );
        mode.setFps( 30 );
        stream.setVideoMode( mode );
    }


    openni::Device device;            // 使用するデバイス
    openni::VideoStream colorStream;  // カラーストリーム
    openni::VideoStream depthStream;  // Depth ストリーム
    openni::VideoStream irStream;  // Depth ストリーム    
    
    cv::Mat colorImage;               // 表示用データ
    cv::Mat depthImage;               // Depth 表示用データ
    cv::Mat irImage;               // 表示用データ    
    cv::Mat PointIndexMat;               // index of pixel-wised relation of a point on point cloud
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_dat;
    std::mutex flagMutex;
};
