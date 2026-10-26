#pragma once
#include <vector>
#include <deque>
#include <queue>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp> // Для OpenCV 5.0
#include "FrameProcessor.hpp"
#include "WaveDescriptor.hpp"

class Detector : public FrameProcessor {
public:
    Detector(double fps,
             int    window       = 5,      
             double sigma        = 50.0,   
             double threshold    = 5.0,    
             double minArea      = 2000.0, 
             int    boxPadding   = 25)     
        : tracker_(fps),
          window_(window), sigma_(sigma), threshold_(threshold),
          minArea_(minArea), boxPadding_(boxPadding),
          kernelOpen_(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))),
          kernelClose_(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(17, 17))) {}

    void process(cv::Mat& frame, int idx) override {
        lastFrame_ = idx;

        cv::Mat gray, blur, local, mask;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(0, 0), sigma_);
        cv::subtract(gray, blur, local);
        
        cv::threshold(local, mask, threshold_, 255, cv::THRESH_BINARY);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernelOpen_);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelClose_);

        buffer_.push_back(mask.clone());
        while ((int)buffer_.size() > window_) buffer_.pop_front();

        cv::Mat validated = cv::Mat::zeros(mask.size(), CV_8U);
        if ((int)buffer_.size() == window_) {
            validated = cv::Mat::ones(mask.size(), CV_8U) * 255;
            for (auto& m : buffer_) cv::bitwise_and(validated, m, validated);
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(validated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::vector<std::vector<cv::Point>> validContours;
        for (auto& c : contours) {
            if (cv::contourArea(c) > minArea_) validContours.push_back(c);
        }

        cv::Mat merged = cv::Mat::zeros(mask.size(), CV_8U);
        if (!validContours.empty()) {
            int n = (int)validContours.size();
            std::vector<cv::Rect> boxes(n);
            for (int i = 0; i < n; ++i) {
                cv::Rect r = cv::boundingRect(validContours[i]);
                int x = std::max(0, r.x - boxPadding_);
                int y = std::max(0, r.y - boxPadding_);
                int w = std::min(frame.cols - x, r.width  + 2 * boxPadding_);
                int h = std::min(frame.rows - y, r.height + 2 * boxPadding_);
                boxes[i] = cv::Rect(x, y, w, h);
            }

            std::vector<bool> visited(n, false);
            for (int i = 0; i < n; ++i) {
                if (visited[i]) continue;
                std::queue<int> q; q.push(i); visited[i] = true;
                std::vector<int> component;
                while (!q.empty()) {
                    int curr = q.front(); q.pop(); component.push_back(curr);
                    for (int j = 0; j < n; ++j) {
                        if (visited[j]) continue;
                        if ((boxes[curr] & boxes[j]).area() > 0) { 
                            visited[j] = true; q.push(j);
                        }
                    }
                }
                for (int c : component) {
                    cv::drawContours(merged, validContours, c, cv::Scalar(255), -1);
                }
            }
        }

        std::vector<std::vector<cv::Point>> finalContours;
        cv::findContours(merged, finalContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::vector<cv::Point2d> centroids;
        std::vector<double> areas;
        for (auto& c : finalContours) {
            double a = cv::contourArea(c);
            if (a <= minArea_) continue;
            cv::Moments m = cv::moments(c);
            if (m.m00 == 0) continue;
            
            centroids.push_back(cv::Point2d(m.m10 / m.m00, m.m01 / m.m00));
            areas.push_back(a);
        }

        activeWaves_ = tracker_.update(centroids, areas, idx);
        
        // Сохраняем контуры для main.cpp. Отрисовку убрали!
        lastContours_ = finalContours;
    }

    void finalize() override {
        tracker_.closeAll(lastFrame_);
    }

    // Геттеры для main.cpp
    const WaveTracker& tracker() const { return tracker_; }
    const std::vector<std::vector<cv::Point>>& getLastContours() const { return lastContours_; }
    int getActiveWaves() const { return activeWaves_; }
    
    double getTotalArea() const { 
        double area = 0; 
        for(auto& c : lastContours_) area += cv::contourArea(c); 
        return area; 
    }

private:
    WaveTracker tracker_;
    int    window_;
    double sigma_, threshold_, minArea_;
    int    boxPadding_;
    cv::Mat kernelOpen_, kernelClose_;
    std::deque<cv::Mat> buffer_;
    int    activeWaves_ = 0;
    int    lastFrame_   = 0;
    std::vector<std::vector<cv::Point>> lastContours_;
};