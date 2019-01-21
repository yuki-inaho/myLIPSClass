#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <vector>
#include <deque>  
#include <iostream>

using namespace std;

class FixedSizeQueue
{
    public:    
        FixedSizeQueue()
        {

        }

        FixedSizeQueue(int _size)
        {
            queue_size = _size;
        }

    void enqueue(cv::Mat obj)
    {
        _queue.push_front(obj);
        while(_queue.size() > queue_size){
            _queue.pop_back();
        }
    }

    cv::Mat
    &operator[](int i)
    {
        return _queue[i];
    };

    int
    size()
    {
        return _queue.size();
    };

    int queue_size;
    private:
        std::deque<cv::Mat> _queue;        
};

class MedianFilter
{
    public:
        MedianFilter(int filter_length, int _frame_width, int _frame_height);
        void enqueue(cv::Mat obj);
        cv::Mat process();
        bool get_init_flag();
        int get_queue_size();        

    private:
        cv::Mat frame;        
        int filter_length;
        int frame_width, frame_height;
        bool pop_front;
        bool is_init;
        FixedSizeQueue median_lists;
};

// MedianFilter public methods
MedianFilter::MedianFilter(int _filter_length, int _frame_width, int _frame_height)
{
    filter_length = _filter_length;
    FixedSizeQueue median_lists(filter_length);
    frame_width = _frame_width;
    frame_height = _frame_height;
    is_init = false;
}

void
MedianFilter::enqueue(cv::Mat obj)
{
    if(obj.rows == frame_height && obj.cols == frame_width){
        median_lists.enqueue(obj);
    }else{
        std::cout << "Input image has invalid sizes" << std::endl;
    }

    if(get_queue_size() >= filter_length){
        is_init = true;
    }else{
        is_init = false;
    }
}

int
MedianFilter::get_queue_size()
{
    return median_lists.size();
}

cv::Mat
MedianFilter::process()
{
    cv::Mat medianMat ;
    medianMat = cv::Mat::zeros(cv::Size(frame_width,frame_height), CV_8UC3);    

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間

/*
    for(int y=0; y<medianMat.rows; y++) {
        cv::Vec3b *cptr = medianMat.ptr<cv::Vec3b>(y);
        for(int x=0; x<medianMat.cols; x++) {
            for(int c=0; c<3; c++){
                std::vector<int> _pixel;                
                for(int l=0;l<filter_length;l++){
                    _pixel.push_back(median_lists[l].at<cv::Vec3b>(y,x)[c]);
                }
                int med_idx = filter_length/2;
                std::nth_element(_pixel.begin(), _pixel.begin()+med_idx, _pixel.end());
                cptr[x][c] = _pixel[_pixel.size()/2];
            }
        }
    }    
*/
    cv::parallel_for_(cv::Range(0, medianMat.rows*medianMat.cols), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int y = r / medianMat.cols;
            int x = r % medianMat.cols;
            for(int c=0; c<3; c++){
                std::vector<int> _pixel;                
                for(int l=0;l<filter_length;l++){
                    _pixel.push_back(median_lists[l].at<cv::Vec3b>(y,x)[c]);
                }
                int med_idx = filter_length/2;
                std::nth_element(_pixel.begin(), _pixel.begin()+med_idx, _pixel.end());
                medianMat.ptr<cv::Vec3b>(y)[x][c] = _pixel[_pixel.size()/2];
            }
        }
    });

    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << elapsed << endl;

    return medianMat;
}

bool
MedianFilter::get_init_flag()
{
    return is_init;
}

class DepthMedianFilter
{
    public:
        DepthMedianFilter(int filter_length, int _frame_width, int _frame_height);
        void enqueue(cv::Mat obj);
        cv::Mat process();
        bool get_init_flag();
        int get_queue_size();        

    private:
        cv::Mat frame;        
        int filter_length;
        int frame_width, frame_height;
        bool pop_front;
        bool is_init;
        FixedSizeQueue median_lists;
};

// DepthMedianFilter public methods
DepthMedianFilter::DepthMedianFilter(int _filter_length, int _frame_width, int _frame_height)
{
    filter_length = _filter_length;
    FixedSizeQueue median_lists(filter_length);
    frame_width = _frame_width;
    frame_height = _frame_height;
    is_init = false;
}

void
DepthMedianFilter::enqueue(cv::Mat obj)
{
    if(obj.rows == frame_height && obj.cols == frame_width){
        median_lists.enqueue(obj);
    }else{
        std::cout << "Input image has invalid sizes" << std::endl;
    }

    if(get_queue_size() >= filter_length){
        is_init = true;
    }else{
        is_init = false;
    }
}

int
DepthMedianFilter::get_queue_size()
{
    return median_lists.size();
}

cv::Mat
DepthMedianFilter::process()
{
    cv::Mat medianMat ;
    medianMat = cv::Mat::zeros(cv::Size(frame_width,frame_height), CV_16U);    

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    cv::parallel_for_(cv::Range(0, medianMat.rows*medianMat.cols), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int y = r / medianMat.cols;
            int x = r % medianMat.cols;
            std::vector<int> _pixel;                
            for(int l=0;l<filter_length;l++){
                if(std::isfinite(median_lists[l].at<unsigned short>(y,x)) && median_lists[l].at<unsigned short>(y,x)>0)
                _pixel.push_back(median_lists[l].at<unsigned short>(y,x));
            }
            if(_pixel.size()==0)_pixel.push_back(0);
            int med_idx = filter_length/2;
            std::nth_element(_pixel.begin(), _pixel.begin()+med_idx, _pixel.end());
            medianMat.ptr<unsigned short>(y)[x] = _pixel[0];
        }
    });

    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << elapsed << endl;
    return medianMat;
}

bool
DepthMedianFilter::get_init_flag()
{
    return is_init;
}
