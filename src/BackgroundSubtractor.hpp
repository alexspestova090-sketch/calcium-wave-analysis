#pragma once
#include "FrameProcessor.hpp"

class BackgroundSubtractor : public FrameProcessor {
public:
    BackgroundSubtractor(double scaleFactor = 0.25, int sampleStep = 10)
        : scaleFactor_(scaleFactor), sampleStep_(sampleStep) {}

    // 1. ФАЗА ОБУЧЕНИЯ: собираем уменьшенные кадры для анализа
    void train(cv::Mat& frame, int idx) override {
        if (idx % sampleStep_ != 0) return; // Берем каждый 10-й кадр (экономим память)
        
        cv::Mat gray, down;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, down, cv::Size(), scaleFactor_, scaleFactor_, cv::INTER_AREA);
        
        if (trainData_.empty()) {
            downRows_ = down.rows;
            downCols_ = down.cols;
        }
        // Вытягиваем кадр в одну строку и сохраняем
        trainData_.push_back(down.reshape(1, 1).clone());
    }

    // 2. ВЫЧИСЛЕНИЕ ВСПЫШКИ (вызывается 1 раз после обучения)
    void finalizeTraining(int totalFrames) override {
        if (trainData_.empty()) return;
        
        cv::Mat dataMat;
        cv::vconcat(trainData_, dataMat); // Собираем все строки в матрицу
        trainData_.clear();

        // PCA находит главные компоненты. В OpenCV pca.mean — это и есть усредненный кадр (вспышка)
        cv::PCA pca(dataMat, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
        
        // Восстанавливаем форму картинки из плоского массива
        flashImg_ = pca.mean.reshape(1, downRows_).clone();
    }

    // 3. РАБОЧАЯ ФАЗА: вычитаем вспышку из каждого кадра
    void process(cv::Mat& frame, int idx) override {
        if (flashImg_.empty()) return;

        cv::Mat gray, flashResized;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Растягиваем маленькую вспышку до полного размера кадра
        cv::resize(flashImg_, flashResized, frame.size(), 0, 0, cv::INTER_LINEAR);

        // Вычитаем
        cv::Mat grayF, flashF;
        gray.convertTo(grayF, CV_32F);
        flashResized.convertTo(flashF, CV_32F);
        grayF -= flashF;
        
        // Ограничиваем значения (как в Python)
        cv::threshold(grayF, grayF, 255.0, 255.0, cv::THRESH_TRUNC); // Не больше 255
        cv::threshold(grayF, grayF, 0.0, 0.0, cv::THRESH_TOZERO);    // Не меньше 0

        // Обратно в 3-канальный кадр для следующих модулей
        cv::Mat out;
        grayF.convertTo(out, CV_8U);
        cv::cvtColor(out, frame, cv::COLOR_GRAY2BGR);
    }

private:
    double scaleFactor_;
    int sampleStep_;
    std::vector<cv::Mat> trainData_;
    cv::Mat flashImg_;
    int downRows_ = 0, downCols_ = 0;
};