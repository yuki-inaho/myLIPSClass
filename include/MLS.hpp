#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>


//removeNan: NaN要素を点群データから除去するメソッド
//input : target(NaN要素を除去する対象の点群)
//output: cloud(除去を行った点群)
pcl::PointCloud<pcl::PointXYZ>::Ptr removeNan(pcl::PointCloud<pcl::PointXYZ>::Ptr target){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  int n_point = target->points.size();

  for(int i=0;i<n_point; i++){
    pcl::PointXYZ tmp_point;
    if(std::isfinite(target->points[i].x) && std::isfinite(target->points[i].y) && std::isfinite(target->points[i].z && target->points[i].z>0)){
      tmp_point.x = target->points[i].x;
      tmp_point.y = target->points[i].y;
      tmp_point.z = target->points[i].z;
      cloud->points.push_back(tmp_point);
    }
  }
//  cout << "varid points:" << cloud->points.size() << endl;
  return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr 
MyMLS(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);    
    pcl::PointCloud<pcl::PointNormal> mls_points_ground;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smooth (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::MovingLeastSquaresOMP<pcl::PointXYZ, pcl::PointXYZ> mls_ground(8);
    mls_ground.setComputeNormals(true);
    mls_ground.setInputCloud(cloud);
    mls_ground.setSearchMethod(tree);
    mls_ground.setSearchRadius(0.01);
    mls_ground.process(*cloud_smooth);
    return cloud_smooth;
}


