#include <iostream>
#include <stdexcept>

#include <PS1080.h>
#include <OpenNI.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace openni;
using namespace cv;

Mat niComputeCloud( const Mat depthMap, const VideoStream& depthStream )
{
    Size nsize = depthMap.size();
    vector<Mat> output( 3 );
    output[0] = Mat( nsize, CV_32F );
    output[1] = Mat( nsize, CV_32F );
    output[2] = Mat( nsize, CV_32F );

    const DepthPixel* pDepthArray = (const DepthPixel*)depthMap.data;

    for( int y = 0; y < depthMap.rows; y++ )
    {
        for( int x = 0; x < depthMap.cols; x++ )
        {
            float fX, fY, fZ;
            openni::CoordinateConverter::convertDepthToWorld( depthStream,
                                                    x, y, *pDepthArray++,
                                                    &fX, &fY, &fZ );
            output[0].at<float>(y,x) = fX;
            output[1].at<float>(y,x) = fY;
            output[2].at<float>(y,x) = fZ;
        }
    }

    Mat outMat;
    merge( output, outMat );
    return outMat;
}

class DepthSensor
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
//    device.setImageRegistrationMode( openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR );

    // カラーストリームを有効にする
    colorStream.create( device, openni::SENSOR_COLOR );
    colorStream.setMirroringEnabled(false);    
    changeResolution( colorStream );
    colorStream.start();

    // Depth ストリームを有効にする
    depthStream.create( device, openni::SENSOR_DEPTH );
    depthStream.setMirroringEnabled(false);
    changeResolution( depthStream );
    cout << depthStream.isCroppingSupported() << endl; 
    cout << depthStream.getMaxPixelValue() << endl; 

    openni::VideoMode mode = depthStream.getVideoMode();
    cout << "Depth VideoMode: " << mode.getResolutionX() << " x " << mode.getResolutionY() << " @ " << mode.getFps() << " FPS";
    cout << ", Unit is ";
    if(mode.getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_1_MM)
        cout << "1mm";
    else if (mode.getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_100_UM)
        cout << "100um";
    cout << endl;

//    depthStream.setProperty(XN_STREAM_PROPERTY_CLOSE_RANGE, true);
//    openni::CameraSettings* cs = depthStream.getCameraSettings();

//    if(cs){
//      int test = cs->getExposure();
//      cs->setAutoExposureEnabled(false);
//      bool test = cs->getAutoExposureEnabled();
//      cout << test << endl;
//    }
//

    depthStream.start();
  }
  
  void update()
  {
    openni::VideoFrameRef colorFrame;
    openni::VideoFrameRef depthFrame;

    // 更新されたフレームを取得する
    colorStream.readFrame( &colorFrame );
    depthStream.readFrame( &depthFrame );

//    bool CloseRange;    
//    depthStream.getProperty(XN_STREAM_PROPERTY_CLOSE_RANGE, &CloseRange);
//    printf("\nClose range: %s", CloseRange?"On":"Off");    
    
    // フレームのデータを表示できる形に変換する
    colorImage = showColorStream( colorFrame );
    depthImage = showDepthStream( depthFrame );
    
    if( !depthImage.empty() && !colorImage.empty() )
    {
        mPointCloud = niComputeCloud( depthImage, depthStream );
    }

/*
    for(int i=0;i<10;i++){
      cout << mPointCloud.at<float>(i,0) << " ";
      cout << mPointCloud.at<float>(i,1) << " ";
      cout << mPointCloud.at<float>(i,2) << endl;            
    }
*/    

//    cvtColor(depthImage_color, depthImage_color, COLOR_HSV2BGR);
    // フレームのデータを表示する
    cv::imshow( "Color Stream", colorImage );
    cv::imshow( "Depth Stream", depthImage );
//    cv::imshow( "Depth Stream Color", depthImage_color );
  }

  cv::Mat
  getPointCloud()
  {
    return mPointCloud;
  }

  cv::Mat
  getColorImage()
  {
    return colorImage;
  }

  cv::Mat
  getDepthImage()
  {
    return depthImage;
  }

private:

  void changeResolution( openni::VideoStream& stream )
  {
    openni::VideoMode mode = stream.getVideoMode();
    mode.setResolution( 640, 480 );
    mode.setFps( 30 );
    stream.setVideoMode( mode );
  }
  
  // カラーストリームを表示できる形に変換する
  cv::Mat showColorStream( const openni::VideoFrameRef& colorFrame )
  {
    // OpenCV の形に変換する
    cv::Mat colorImage = cv::Mat( colorFrame.getHeight(),
                                 colorFrame.getWidth(),
                                 CV_8UC3, (unsigned char*)colorFrame.getData() );
    
    // BGR の並びを RGB に変換する
    cv::cvtColor( colorImage, colorImage, CV_RGB2BGR );
    
    return colorImage;
  }
  
  // Depth ストリームを表示できる形に変換する
  cv::Mat showDepthStream( const openni::VideoFrameRef& depthFrame )
  {
    // 距離データを画像化する(16bit)
    cv::Mat depthImage = cv::Mat( depthFrame.getHeight(),
                                 depthFrame.getWidth(),
                                 CV_16UC1, (unsigned short*)depthFrame.getData() );

    for(int y=0;y< depthImage.rows;y++){
      for(int x=0;x< depthImage.cols;x++){
        if(depthImage.at<float>(y,x)>2500)
        depthImage.at<float>(y,x) = 0;
      }
    }

/*
    for(int y=0;y< depthImage.rows;y++){
      for(int x=0;x< depthImage.cols;x++){
        depthImage_color.at<Vec3b>(y,x)[0] = static_cast<uchar>(depthImage.at<float>(y,x)/1500*255.0);        
        depthImage_color.at<Vec3b>(y,x)[1] = 255;        
        depthImage_color.at<Vec3b>(y,x)[2] = 255;
      }
    }
*/

    // 0-10000mmまでのデータを0-255(8bit)にする
    depthImage.convertTo( depthImage, CV_8U, 255.0 / 2500.0 );
//    depthImage.convertTo( depthImage_color, CV_32FC3, 255.0 / 1500 );
    
    // 中心点の距離を表示する
    showCenterDistance( depthImage, depthFrame );
    return depthImage;
  }

  // 中心点の距離を表示する
  void showCenterDistance( cv::Mat& depthImage, const openni::VideoFrameRef& depthFrame)
  {
    // 中心点の距離を表示する
	  openni::VideoMode videoMode = depthStream.getVideoMode();

    int centerX = videoMode.getResolutionX() / 2;
    int centerY = videoMode.getResolutionY() / 2;
    int centerIndex = (centerY * videoMode.getResolutionX()) + centerX;

    unsigned short* depth = (unsigned short*)depthFrame.getData();

    std::stringstream ss;
    ss << "Center Point :" << depth[centerIndex];
    cv::putText( depthImage, ss.str(), cv::Point( 0, 50 ),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar( 255 ) );
  }

  void getColorImage( cv::Mat& depthImage, const openni::VideoFrameRef& depthFrame)
  {
    // 中心点の距離を表示する
	  openni::VideoMode videoMode = depthStream.getVideoMode();

    int centerX = videoMode.getResolutionX() / 2;
    int centerY = videoMode.getResolutionY() / 2;
    int centerIndex = (centerY * videoMode.getResolutionX()) + centerX;

    unsigned short* depth = (unsigned short*)depthFrame.getData();

    std::stringstream ss;
    ss << "Center Point :" << depth[centerIndex];
    cv::putText( depthImage, ss.str(), cv::Point( 0, 50 ),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar( 255 ) );
  }
  
private:
  openni::Device device;            // 使用するデバイス
  openni::VideoStream colorStream;  // カラーストリーム
  openni::VideoStream depthStream;  // Depth ストリーム
  
  cv::Mat colorImage;               // 表示用データ
  cv::Mat depthImage;               // Depth 表示用データ
  //cv::Mat depthImage_color;               // Depth 表示用データ  
  cv::Mat mPointCloud;  
};

int main(int argc, const char * argv[])
{
  try {
    // OpenNI を初期化する
    openni::OpenNI::initialize();
  
    // センサーを初期化する
    Mat depthImage;
    DepthSensor sensor;
    sensor.initialize();
  
    // メインループ
    while ( 1 ) {
      sensor.update();
      depthImage = sensor.getDepthImage();
      cv::Mat depthImage_color = cv::Mat::zeros( depthImage.rows,
                                  depthImage.cols,
                                  CV_8UC3);

      for(int y=0;y< depthImage.rows;y++){
        for(int x=0;x< depthImage.cols;x++){
          if(depthImage.at<uchar>(y,x) == 255 || depthImage.at<uchar>(y,x) == 0){
            depthImage_color.at<Vec3b>(y,x)[0] = depthImage.at<uchar>(y,x);
            depthImage_color.at<Vec3b>(y,x)[1] = 0;
            depthImage_color.at<Vec3b>(y,x)[2] = 0;
          }else{
            depthImage_color.at<Vec3b>(y,x)[0] = depthImage.at<uchar>(y,x);
            depthImage_color.at<Vec3b>(y,x)[1] = 255;
            depthImage_color.at<Vec3b>(y,x)[2] = 255;
          }
        }
      }

      cvtColor(depthImage_color, depthImage_color, COLOR_HSV2BGR_FULL);
      cv::imshow("gtest", depthImage_color);
      int key = cv::waitKey( 10 );
      if ( key == 'q' ) {
        break;
      }

      if (key == 's'){
        imwrite("../img/depth_col.png", depthImage_color);
      }

    }
  }
  catch ( std::exception& ) {
    std::cout << openni::OpenNI::getExtendedError() << std::endl;
  }
  return 0;
}

