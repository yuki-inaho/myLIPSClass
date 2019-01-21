#include <iostream>
#include <OpenNI.h>

#include <boost/thread/thread.hpp>
#include "MedianFilter.hpp"
#include "MLS.hpp"
#include <pcl/filters/voxel_grid.h>
#include "NormalImage.hpp"
#include "Visualizer.hpp"
#include "Listener.hpp" 

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <pcl/features/integral_image_normal.h>

using namespace cv;
using namespace std;
using namespace openni;

pcl::PointCloud<pcl::PointXYZ>::Ptr deepcopyPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc_input){
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_output (new pcl::PointCloud<pcl::PointXYZ>);
    int n_pc  = pc_input->points.size();
    for(int i=0;i<n_pc;i++){
        pcl::PointXYZ point;
        point.x = pc_input->points[i].x;
        point.y = pc_input->points[i].y;
        point.z = pc_input->points[i].z;
        pc_output->points.push_back(point);
    }
    return pc_output;
}

cv::Mat
DepthToColorMat(cv::Mat depth_mat){
    cv::Mat depth_color = cv::Mat::zeros(depth_mat.rows, depth_mat.cols, CV_8UC3);
    for(int y =0 ; y< depth_mat.rows; y++){
        unsigned short *ptr = depth_mat.ptr<unsigned short>(y);
        for(int x =0 ; x< depth_mat.cols; x++){
            unsigned short depth_var = ptr[x];
            unsigned char hi = (depth_var >> 8) & 0xff;
            unsigned char low = (depth_var >> 0) & 0xff;
            depth_color.at<cv::Vec3b>(y, x)[0] = hi; // blue
            depth_color.at<cv::Vec3b>(y, x)[1] = low; // green
            depth_color.at<cv::Vec3b>(y, x)[2] = 0; // red
        }
    }
    return depth_color;
}

cv::Mat
ColorToDepthMat(cv::Mat color_mat){
    cv::Mat depth_mat = cv::Mat::zeros(color_mat.rows, color_mat.cols, CV_16U);
    for(int y =0 ; y< color_mat.rows; y++){
        cv::Vec3b *ptr = color_mat.ptr<cv::Vec3b>(y);
        for(int x =0 ; x< color_mat.cols; x++){
            cv::Vec3b col_var = ptr[x];
            unsigned short depth_var =  (col_var[0]  << 8)  + col_var[1] ;
            depth_mat.at<unsigned short>(y,x) = depth_var; // green
        }
    }
    return depth_mat;
}

cv::Mat
normalizePseudoSurfaceNormal(cv::Mat psn_img){
    Mat sn_img_normalized = Mat::zeros(psn_img.rows, psn_img.cols, CV_8UC3);

    cv::parallel_for_(cv::Range(0, psn_img.rows*psn_img.cols), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int y = r / psn_img.cols;
            int x = r % psn_img.cols;
            float psn_norm = std::sqrt(psn_img.at<cv::Vec3b>(y,x)[0]*psn_img.at<cv::Vec3b>(y,x)[0] + 
                                psn_img.at<cv::Vec3b>(y,x)[1]*psn_img.at<cv::Vec3b>(y,x)[1]+ 
                                psn_img.at<cv::Vec3b>(y,x)[2]*psn_img.at<cv::Vec3b>(y,x)[2]);
            if(psn_norm > 0.99 ){
                float n_x = float(psn_img.at<cv::Vec3b>(y,x)[0])/255.0;
                float n_y = float(psn_img.at<cv::Vec3b>(y,x)[1])/255.0;
                float n_z = float(psn_img.at<cv::Vec3b>(y,x)[2])/255.0;
                
                float n_norm = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);

                sn_img_normalized.at<cv::Vec3b>(y,x)[0] = (unsigned char)(n_x/n_norm*255.0);
                sn_img_normalized.at<cv::Vec3b>(y,x)[1] = (unsigned char)(n_y/n_norm*255.0);
                sn_img_normalized.at<cv::Vec3b>(y,x)[2] = (unsigned char)(n_z/n_norm*255.0);
            }
        }
    });
    return sn_img_normalized;
}

cv::Mat
Float2CharSurfaceNormal(cv::Mat sn_img)
{
    cv::Mat sn_img_char = cv::Mat::zeros(sn_img.rows, sn_img.cols, CV_8UC3);
    cv::parallel_for_(cv::Range(0, sn_img.rows*sn_img.cols), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int y = r / sn_img.cols;
            int x = r % sn_img.cols;
            if(sn_img.at<cv::Vec3f>(y,x)[0] != 0){
                sn_img_char.at<cv::Vec3b>(y,x)[0] = (unsigned char)(sn_img.at<cv::Vec3f>(y,x)[0]*127.0 + 127.0);
                sn_img_char.at<cv::Vec3b>(y,x)[1] = (unsigned char)(sn_img.at<cv::Vec3f>(y,x)[1]*127.0 + 127.0);
                sn_img_char.at<cv::Vec3b>(y,x)[2] = (unsigned char)(sn_img.at<cv::Vec3f>(y,x)[2]*127.0 + 127.0);
            }
        }
    });
    return sn_img_char;
}



void
drawContours(cv::Mat &img, const cv::Mat &mask)
{
    for(int y = 0;y<img.rows;y++){
        for(int x = 0;x<img.cols;x++){
            if(mask.at<unsigned char>(y,x) == 255){
                img.at<cv::Vec3b>(y,x)[0] = 255;
                img.at<cv::Vec3b>(y,x)[1] = 0;
                img.at<cv::Vec3b>(y,x)[2] = 0;
            }
        }
    }
}

int
getLabelNum(cv::Mat label){
    int max_label_num = 0;
    for(int y = 0;y<label.rows;y++){
        for(int x = 0;x<label.cols;x++){    
            if(label.at<int>(y,x) >max_label_num )
                max_label_num = label.at<int>(y,x);
        }
    }
    return max_label_num+1;
}

vector<vector<vector<int>>>
label2ind(cv::Mat label){
    int label_num = getLabelNum(label);
    vector<vector<vector<int>>> label_wised_index;

    for(int l =0;l<label_num;l++){
        vector<vector<int>> _label_set;
        label_wised_index.push_back(_label_set);
    }

    for(int y = 0;y<label.rows;y++){
        for(int x = 0;x<label.cols;x++){
            vector<int> _ind;
            _ind.push_back(x);
            _ind.push_back(y);
            label_wised_index[label.at<int>(y,x)].push_back(_ind);
        }
    }

    return label_wised_index;
}

cv::Mat
smoothingOnLabel(cv::Mat img, vector<vector<vector<int>>> label_list){
    cv::Mat img_smoothed = cv::Mat::zeros(img.rows, img.cols, CV_16UC1);
    int num_label = label_list.size();
    cout << num_label << endl;
    for(int l=0; l<num_label; l++){
        int num_elem = label_list[l].size();
        if(num_elem==0) continue;
        unsigned short min_pixel;
        min_pixel = 10000;

        for(int e=0; e<num_elem; e++){
            int x = label_list[l][e][0];
            int y = label_list[l][e][1];
            if(min_pixel > img.at<unsigned short>(y,x) && img.at<unsigned short>(y,x) >0)
                min_pixel = img.at<unsigned short>(y,x);
        }

        for(int e=0; e<num_elem; e++){
            int x = label_list[l][e][0];
            int y = label_list[l][e][1];
            img_smoothed.at<unsigned short>(y,x) = min_pixel;
        }
    }
    return img_smoothed;
}


cv::Mat
PaddingColor(cv::Mat img, int kernel_size)
{
    cv::Mat img_smoothed = cv::Mat::zeros(img.rows+kernel_size-1, img.cols+kernel_size-1, CV_8UC3);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            img_smoothed.at<cv::Vec3b>(y+kernel_size/2, x+kernel_size/2) = img.at<cv::Vec3b>(y, x);
        }
    }
    return img_smoothed;
}

cv::Mat
PaddingDepth(cv::Mat img, int kernel_size)
{
    cv::Mat img_smoothed = cv::Mat::zeros(img.rows+kernel_size-1, img.cols+kernel_size-1, CV_16UC1);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            img_smoothed.at<unsigned short>(y+kernel_size/2, x+kernel_size/2) = img.at<unsigned short>(y, x);
        }
    }
    return img_smoothed;
}

unsigned short
_JointBilateralFilterInpainting(cv::Mat colorImage_padded, cv::Mat depthImage_padded, int x, int y, int kernel_size, double sigma_pos, double sigma_col, double sigma_depth)
{
    double _kernel_var, kernel_var, W;
    _kernel_var=0;
    W=0;

    for(int k_y=-kernel_size/2; k_y<=kernel_size/2; k_y++){
        for(int k_x=-kernel_size/2; k_x<=kernel_size/2; k_x++){
            if(depthImage_padded.at<unsigned short>(y+k_y,x+k_x)!=0){
                cv::Vec3b centor_col = colorImage_padded.at<cv::Vec3b>(y,x);
                cv::Vec3b perf_col = colorImage_padded.at<cv::Vec3b>(y+k_y,x+k_x);
                unsigned short centor_depth = depthImage_padded.at<unsigned short>(y,x);
                unsigned short perf_depth = depthImage_padded.at<unsigned short>(y+k_y,x+k_x);

                double diff_pos = std::sqrt(double(k_x)*double(k_x) + double(k_y)*double(k_y));
                double diff_col_r = double(centor_col[0]) - double(perf_col[0]);
                double diff_col_g = double(centor_col[1]) - double(perf_col[1]);
                double diff_col_b = double(centor_col[2]) - double(perf_col[2]);
                double diff_col = std::sqrt(diff_col_r*diff_col_r + diff_col_g*diff_col_g + diff_col_b*diff_col_b);
                double _diff_depth = double(centor_depth) - double(perf_depth);
                double diff_depth = std::sqrt(_diff_depth*_diff_depth);
                double kernel_pos = std::exp(-diff_pos*diff_pos/(2*sigma_pos*sigma_pos));
                double kernel_col = std::exp(-diff_col*diff_col/(2*sigma_col*sigma_col));
                double kernel_depth = std::exp(-diff_depth*diff_depth/(2*sigma_depth*sigma_depth));
                _kernel_var += kernel_pos * kernel_col * kernel_depth * double(perf_depth);
                W += kernel_pos * kernel_col*kernel_depth;
            }
        }
    }
    
    if(W >0){
        kernel_var = static_cast<unsigned short>(_kernel_var/(W));
    }else{
        kernel_var = 0;
    }

    return kernel_var;
}

cv::Mat
JointBilateralFilterInpaintingOMP(cv::Mat colorImage, cv::Mat depthImage, vector<vector<int>> &inpainting_index, int kernel_size, double sigma_pos, double sigma_col, double sigma_depth)
{
    cv::Mat depthImage_smoothed = cv::Mat::zeros(depthImage.rows, depthImage.cols, CV_16U);
    cv::Mat depthImage_padded = PaddingDepth(depthImage, kernel_size);
    cv::Mat colorImage_padded = PaddingColor(colorImage, kernel_size);
    cv::Mat depthImage_mask = cv::Mat::zeros(depthImage.rows, depthImage.cols, CV_8UC1);

    cv::parallel_for_(cv::Range(0, depthImage.rows*depthImage.cols), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int y = r / depthImage.cols;
            int x = r % depthImage.cols;
            if(depthImage.at<unsigned short>(y, x) == 0){
                unsigned short kernel_var = _JointBilateralFilterInpainting(colorImage_padded, depthImage_padded, x+kernel_size/2, y+kernel_size/2, kernel_size, sigma_pos, sigma_col, sigma_depth);
                depthImage_smoothed.at<unsigned short>(y, x) = kernel_var;
                if(kernel_var>0){
                    depthImage_mask.at<unsigned char>(y, x) = 255;
                }
            }else{
                depthImage_smoothed.at<unsigned short>(y, x) = depthImage.at<unsigned short>(y, x);
            }
        }
    });

    for(int y=0;y<depthImage.rows;y++){
        for(int x=0;x<depthImage.cols;x++){
            if(depthImage_mask.at<unsigned char>(y, x)>0){
                vector<int> _inpainting_index;
                _inpainting_index.push_back(x);
                _inpainting_index.push_back(y);
                inpainting_index.push_back(_inpainting_index);
            }
        }
    }

    return depthImage_smoothed;
}

cv::Mat
RefineInpaintingArea(cv::Mat color, cv::Mat depthImage, vector<vector<int>> inpainting_index)
{
    cv::Mat _depth = depthImage.clone();

    cv::ximgproc::amFilter(color, depthImage, _depth, 5, 0.01, true);
    for(size_t i=0;i<inpainting_index.size();i++){
        vector<int> _inpainting_index = inpainting_index[i];
        int x = _inpainting_index[0];
        int y = _inpainting_index[1];
        depthImage.at<unsigned short>(_inpainting_index[1],_inpainting_index[0]) = _depth.at<unsigned short>(_inpainting_index[1],_inpainting_index[0]);
    }
    return depthImage;
}

float
frobenius_norm(cv::Mat X)
{
    int nrow = X.rows;
    int ncol = X.cols;
    float norm = 0;
    float _norm;

    for(int y=0;y<nrow;y++){
        for(int x=0;x<ncol;x++){
            _norm = X.at<float>(y,x);
            norm += _norm*_norm;
        }
    }
    norm = std::sqrt(norm);
    return norm;
}

cv::Mat 
So(cv::Mat X, float tau, int nrow, int ncol)
{
    int ndiag = nrow;
    cv::Mat r= cv::Mat::zeros(nrow,ncol, CV_32FC1);   

    for(int i=0;i<ndiag;i++){
        float sign;
        if(X.at<float>(0,i) >0){
            sign = 1;
        }else{
            sign = -1;
        }

        if(std::abs(X.at<float>(0,i)) > tau){
            r.at<float>(i,i) = sign * std::abs(X.at<float>(0,i));
        }else{
            r.at<float>(i,i) = 0;
        }
    }
    return r;
}

cv::Mat 
Do(cv::Mat X, float tau)
{
    int nrow = X.rows;
    int ncol = X.cols;
    cv::Mat w, u, vt;
    cv::SVD::compute(X, w, u, vt, cv::SVD::FULL_UV);
//    cout << "test" << endl;
    cv::Mat _So = So(w,tau,nrow,ncol);
    cv::Mat r= u*_So*vt;
    return r;
}

cv::Mat 
RobustPCA(cv::Mat X, float lambda, float mu, float tol, int max_iter)
{
    cv::Mat L, S, Y, _Y, Z;
    int nrow = X.rows;
    int ncol = X.cols;
    L = cv::Mat::zeros(nrow,ncol, CV_32FC1);    
    cv::Mat _X;
    X.convertTo(_X, CV_32FC1);

    float normX = frobenius_norm(_X);

    /*
    % default arguments
    if nargin < 2
        lambda = 1 / sqrt(max(M,N));
    end
    if nargin < 3
        mu = 10*lambda;
    end
    if nargin < 4
        tol = 1e-6;
    end
    if nargin < 5
        max_iter = 1000;
    end
    */
    
    L = cv::Mat::zeros(nrow,ncol, CV_32FC1);
    S = cv::Mat::zeros(nrow,ncol, CV_32FC1);
    _Y = cv::Mat::zeros(nrow,ncol, CV_32FC1);
    Z = cv::Mat::zeros(nrow,ncol, CV_32FC1);    
    float err;

    for(int iter=0;iter<max_iter;iter++){
        cv::Mat temp = _X - S + (1/mu)*_Y;
        L = Do(temp, 1/mu);
        S = So(_X - L + (1/mu)*_Y, lambda/mu,nrow, ncol);
        Z = _X - L - S;
        _Y = _Y + mu*Z;
        err = frobenius_norm(Z)/normX;
        cout << err << endl;
    }
    Z.convertTo(Y, CV_8UC1);
    return Y;
}

int main(int argc, const char * argv[])
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = simpleVis ();
    
    try {
        // OpenNI を初期化する
        openni::OpenNI::initialize();
    
        // センサーを初期化する
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_dat (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_tmp (new pcl::PointCloud<pcl::PointXYZ>);        
        Mat depthImage, colorImage, point_index_mat;
        MyListener listener;
        listener.initialize();
        listener.update();
        depthImage = listener.getDepthImage();
    
        int filter_length = 5;
        DepthMedianFilter dfilter(filter_length, depthImage.cols, depthImage.rows);

        // メインループ
        int count = 0;
        cv::Mat depthImage_test, IRImage;
        int key = cv::waitKey( 30 );                    
        while ( 1 ) {
            listener.update();
            depthImage = listener.getDepthImage();
            colorImage = listener.getColorImage();
            IRImage = listener.getIRImage();
            dfilter.enqueue(depthImage);
            if(dfilter.get_init_flag()){
                depthImage = dfilter.process();
            }else{
                continue;
            }

            cv::Mat depthImageClone = depthImage.clone();
            cv::Mat colorImageClone = colorImage.clone();                
            cv::cvtColor(colorImageClone, colorImageClone, cv::COLOR_BGR2RGB);
            cv::Mat IRImageClone = IRImage.clone();                
            cv::flip(IRImageClone, IRImageClone, 1);

            std::chrono::steady_clock::time_point start_solver = std::chrono::steady_clock::now();
            IRImageClone = RobustPCA(IRImageClone, 0.1, 0.001, 1e-5, 100);
      	    std::chrono::steady_clock::time_point end_solver = std::chrono::steady_clock::now();
      	    std::cout << "calc time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_solver - start_solver).count() << "ms" << std::endl;

//            cv::ximgproc::amFilter(colorImageClone, IRImageClone, IRImageClone, 5, 0.2, true);
            
/*
            cv::Mat depthImageRaw = depthImage.clone();
            vector<vector<int>> inpainting_index;
            cv::Mat depthImage_smoothed = JointBilateralFilterInpaintingOMP(colorImageClone, depthImageClone, inpainting_index, 3, 2.0, 3.0, 100.0);
            //cout << inpainting_index.size() << endl;
            for(int i=0;i<2;i++){
                depthImage_smoothed = JointBilateralFilterInpaintingOMP(colorImage, depthImage_smoothed, inpainting_index, 3, 2.0, 3.0, 100.0);
            }
            depthImageClone = depthImage_smoothed;
            */

/*            
            auto  slic = cv::ximgproc::createSuperpixelSLIC(colorImageClone, cv::ximgproc::SLIC, 3);
            slic->iterate(3);
            slic->enforceLabelConnectivity(3);
            cv::Mat label;
            slic->getLabels(label);
            cout << label.type() << endl;
            vector<vector<vector<int>>> label_wised_index;
            label_wised_index = label2ind(label);
            Mat depthImageSmoothed = smoothingOnLabel(depthImageClone , label_wised_index);
            Mat mask = Mat::zeros(depthImageSmoothed.rows, depthImageSmoothed.cols, CV_8UC1);
*/

/*
            for(int y = 0; y<depthImageSmoothed.rows; y++){
                for(int x = 0; x<depthImageSmoothed.cols; x++){
                    cout << depthImageSmoothed.at<unsigned short>(y,x) << endl;
                }
            }
            */
//            ximgproc::amFilter(depthImageClone, depthImageSmoothed, depthImageSmoothed, 5, 0.00001, false);
            /*
            for(int y = 0; y<mask.rows; y++){
                for(int x = 0; x<mask.cols; x++){
                   if(mask.at<unsigned char>(y,x)>0){
                       depthImage.at<unsigned short>(y,x)=0;
                   }
                }
            }
            */
//            pc_dat = listener.getPointCloud();
            pc_dat = listener._niComputeCloud(depthImageClone);
            point_index_mat = listener.getPointIndexMat();

//            Mat sn_img_float = generateSurfaceNormalImg(pc_dat, point_index_mat, depthImage.cols, depthImage.rows);
//            Mat sn_img;
//            sn_img = Float2CharSurfaceNormal(sn_img_float);
/*            
*/            

            //sn_img_float.convertTo(sn_img, CV_8UC3, 255.0);
            //sn_img = normalizePseudoSurfaceNormal(sn_img);

            //depthImage_test = dfilter.process();
            //Mat dc = DepthToColorMat(depthImageClone);

/*
            Mat mask = Mat::zeros(sn_img.rows, sn_img.cols, CV_8UC1);
            for(int y = 0; y<sn_img.rows; y++){
                for(int x = 0; x<sn_img.cols; x++){
                   if(sn_img.at<cv::Vec3b>(y,x)[0]==0){
                       mask.at<unsigned char>(y,x)=255;
                   }
                }
            }
            cv::inpaint(sn_img, mask, sn_img, 3, INPAINT_NS);
            */
//            ximgproc::amFilter(colorImageClone, sn_img, sn_img, 5, 0.001, true);
//            ximgproc::amFilter(sn_img, sn_img, sn_img, 16, 0.2, true);
//            ximgproc::amFilter(colorImageClone, sn_img, sn_img, 5, 0.05, true);
//            sn_img = normalizePseudoSurfaceNormal(sn_img);            
//            ximgproc::guidedFilter(sn_img, sn_img, sn_img, 5, 0.01);
            //ximgproc::jointBilateralFilter(colorImageClone, sn_img, sn_img, 3, 1, 1);
//            sn_img = normalizePseudoSurfaceNormal(sn_img);
            
            //depthImage = ColorToDepthMat(dc);
//            pc_dat = listener._niComputeCloud(depthImage);
//                pc_tmp = removeNan(pc_dat);
//                cout << pc_tmp->points.size() << endl;
//                pc_dat = MyMLS(pc_tmp);

/*
            if(dfilter.get_init_flag()){
                //depthImage_test = dfilter.process();
                Mat depthImageClone = depthImage.clone();
                Mat colorImageClone = colorImage.clone();                
                Mat dc = DepthToColorMat(depthImageClone);
                ximgproc::amFilter(colorImageClone, sn_img, sn_img, 16, 0.2, true);
                //ximgproc::jointBilateralFilter(colorImageClone, sn_img, sn_img, 16, 1, 1);
                depthImage = ColorToDepthMat(dc);
                pc_dat = listener._niComputeCloud(depthImage);
//                pc_tmp = removeNan(pc_dat);
//                cout << pc_tmp->points.size() << endl;
//                pc_dat = MyMLS(pc_tmp);
            }
*/

            /*
            if(queue_color.size() == queue_color.queue_size){
                cout << queue_color[0] << endl;
            }
            */

/*           
            cv::Mat depthImage_color = cv::Mat::zeros( depthImage.rows,
                                        depthImage.cols,
                                        CV_8UC3);

            for(int y=0;y< depthImage.rows;y++){
                for(int x=0;x< depthImage.cols;x++){
                    if(depthImage.at<unsigned short>(y,x)*255.0 == 255 || depthImage.at<unsigned short>(y,x)*255.0 == 0.0){
                        depthImage_color.at<Vec3b>(y,x)[0] = (unsigned char)(depthImage.at<unsigned short>(y,x)*255);
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
*/

/*
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud (pc_dat);
            sor.setLeafSize (0.001f, 0.001f, 0.001f);
            sor.filter (*pc_dat);
*/
            auto f_update = viewer->updatePointCloud(pc_dat, "sample cloud");
            if (!f_update){
                viewer->addPointCloud<pcl::PointXYZ> (pc_dat, "sample cloud");
            }
            viewer->spinOnce(10);


            count++;
            cv::imshow("ctest", colorImageClone);
            cv::imshow("irtest", IRImage);
            cv::Mat depthImage_color = cv::Mat::zeros( depthImage.rows,
                                        depthImage.cols,
                                        CV_8UC3);
            Mat depthImage8, IRImage8;
            depthImageClone.convertTo( depthImage8, CV_8U, 255.0 / 4096.0 );
            cv::applyColorMap(depthImage8, depthImage_color, COLORMAP_JET);    

/*
            cv::Mat depthImageRaw_color = cv::Mat::zeros( depthImage.rows,
                                        depthImage.cols,
                                        CV_8UC3);
*/

//            depthImageRaw.convertTo( depthImage8, CV_8U, 255.0 / 4096 );
//            cv::applyColorMap(depthImage8, depthImageRaw_color, COLORMAP_JET);    


            //cv::cvtColor(depthImage8, depthImage_color, COLOR_HSV2BGR_FULL);

            cv::imshow("gtest", depthImage_color);
            cv::imshow("ir", IRImageClone);
//            cv::imshow("sn", sn_img);
            /*
            if(colorImage_test.rows >0){
                cv::imshow("ctest", colorImage_test);
            }
            */

            if ( key == 's' ) {
                cv::imwrite("../img/colorImage.png", colorImageClone);
                cv::imwrite("../img/depthImage.png", depthImage_color);
                cv::imwrite("../img/irImage.png", IRImageClone);
//                cv::imwrite("../img/depthImageRaw.png", depthImageRaw_color);                

            }

            if ( key == 'q' ) {
                listener.close();                
                break;
            }
            key = cv::waitKey( 30 );                    
        }

    }catch ( std::exception& ) {
        cout << "got error" << endl;
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
    }

    std::cout << "shutdown" << std::endl;
    OpenNI::shutdown();
    return 0;
}

