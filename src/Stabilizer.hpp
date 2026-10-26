#pragma once
#include <opencv2/opencv.hpp>
#include "FrameProcessor.hpp"
#include <cmath>

class Stabilizer : public FrameProcessor {
public:
    void process(cv::Mat& frame, int idx) override {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F);

        if (ref_.empty()) {
            ref_ = gray.clone();
            lastStabilized_ = frame.clone(); // Сохраняем цветной кадр
            return;
        }

        cv::Point2d shift = cv::phaseCorrelate(ref_, gray);

        if (std::abs(shift.x) > 0.01 || std::abs(shift.y) > 0.01) {
            cv::Matx23d M(1.0, 0.0, -shift.x,
                          0.0, 1.0, -shift.y);
                         
            cv::Mat stabilized;
            cv::warpAffine(frame, stabilized, M, frame.size(),
                           cv::INTER_LINEAR, cv::BORDER_REFLECT);
            frame = stabilized; 
        }
        
        // Сохраняем цветной стабилизированный кадр ДО того, как его испортит вычитание фона
        lastStabilized_ = frame.clone(); 
    }

    void train(cv::Mat& frame, int idx) override { process(frame, idx); }

    // Геттер для получения чистого цветного кадра
    const cv::Mat& getLastStabilized() const { return lastStabilized_; }

private:
    cv::Mat ref_;
    cv::Mat lastStabilized_;
};