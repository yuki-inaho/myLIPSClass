#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <boost/thread/thread.hpp>
#include <chrono>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_cloud.h>
//#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <iostream>
#include <pcl/surface/mls.h>

#include <pcl/features/integral_image_normal.h>

using namespace std;

cv::Mat
generateSurfaceNormalImg(pcl::PointCloud<pcl::PointXYZ>::Ptr pc_dat, cv::Mat index_map, int width, int height)
{
    cv::Mat sn_img = cv::Mat::zeros(height, width, CV_32FC3);

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    cout << pc_dat->points.size() << endl; 

    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);    
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::PointCloud <pcl::PointXYZ>::Ptr pc_tmp (new pcl::PointCloud <pcl::PointXYZ>);

/*
    pcl::MovingLeastSquaresOMP<pcl::PointXYZ, pcl::PointXYZ> mls_ground(16);
    mls_ground.setComputeNormals(true);
    mls_ground.setInputCloud(pc_dat);
    mls_ground.setSearchMethod(tree);
    mls_ground.setSearchRadius(0.01);
    mls_ground.process(*pc_tmp);
*/

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator(16);
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (pc_dat);
    normal_estimator.setKSearch (10);
    normal_estimator.compute (*normals);

/*
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setNormalEstimationMethod (normal_estimator.AVERAGE_3D_GRADIENT);
    normal_estimator.setMaxDepthChangeFactor(0.01f);
    normal_estimator.setNormalSmoothingSize(0.05f);
    normal_estimator.setInputCloud(pc_dat);
    normal_estimator.compute(*normals);
*/

    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << elapsed << endl;

    cout <<  pc_dat->points.size() << endl;
    cout <<  index_map.rows << endl;

    cv::parallel_for_(cv::Range(0, index_map.rows), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int x = index_map.ptr<unsigned short>(r)[0];
            int y = index_map.ptr<unsigned short>(r)[1];
            if(std::isfinite(normals->points[r].normal_x)){
                cv::Vec3f *sn_img_ptr = sn_img.ptr<cv::Vec3f>(y);
                sn_img_ptr[x][0] = normals->points[r].normal_x;
                sn_img_ptr[x][1] = normals->points[r].normal_y;
                sn_img_ptr[x][2] = normals->points[r].normal_z;
            }
        }
    });

    return sn_img;
}

cv::Mat
generateIntegralSurfaceNormalImg(pcl::PointCloud<pcl::PointXYZ>::Ptr pc_dat, cv::Mat index_map, int width, int height)
{
    cv::Mat sn_img = cv::Mat::zeros(height, width, CV_32FC3);

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    cout << pc_dat->points.size() << endl; 

    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);    
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::PointCloud <pcl::PointXYZ>::Ptr pc_tmp (new pcl::PointCloud <pcl::PointXYZ>);

/*
    pcl::MovingLeastSquaresOMP<pcl::PointXYZ, pcl::PointXYZ> mls_ground(16);
    mls_ground.setComputeNormals(true);
    mls_ground.setInputCloud(pc_dat);
    mls_ground.setSearchMethod(tree);
    mls_ground.setSearchRadius(0.01);
    mls_ground.process(*pc_tmp);
*/

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator(16);
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (pc_dat);
    normal_estimator.setKSearch (10);
    normal_estimator.compute (*normals);



    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << elapsed << endl;

    cout <<  pc_dat->points.size() << endl;
    cout <<  index_map.rows << endl;

    cv::parallel_for_(cv::Range(0, index_map.rows), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int x = index_map.ptr<unsigned short>(r)[0];
            int y = index_map.ptr<unsigned short>(r)[1];
            if(std::isfinite(normals->points[r].normal_x)){
                cv::Vec3f *sn_img_ptr = sn_img.ptr<cv::Vec3f>(y);
                sn_img_ptr[x][0] = normals->points[r].normal_x;
                sn_img_ptr[x][1] = normals->points[r].normal_y;
                sn_img_ptr[x][2] = normals->points[r].normal_z;
            }
        }
    });

    return sn_img;
}
