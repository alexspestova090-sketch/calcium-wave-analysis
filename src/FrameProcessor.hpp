#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <iostream>

class FrameProcessor {
public:
    virtual ~FrameProcessor() = default;
    virtual void process(cv::Mat& frame, int frameIndex) = 0;
    virtual void train(cv::Mat& frame, int frameIndex) {}
    virtual void finalizeTraining(int totalFrames) {}
    virtual void finalize() {}
};

class Pipeline {
public:
    // ИСПОЛЬЗУЕМ shared_ptr, чтобы можно было обращаться к модулям извне
    void addProcessor(std::shared_ptr<FrameProcessor> p) {
        processors_.push_back(p);
    }

    void process(cv::Mat& frame, int idx) {
        for (auto& p : processors_) p->process(frame, idx);
    }

    void train(cv::Mat& frame, int idx) {
        for (auto& p : processors_) p->train(frame, idx);
    }
    void finalizeTraining(int total) {
        for (auto& p : processors_) p->finalizeTraining(total);
    }
    void finalize() {
        for (auto& p : processors_) p->finalize();
    }

private:
    std::vector<std::shared_ptr<FrameProcessor>> processors_;
};