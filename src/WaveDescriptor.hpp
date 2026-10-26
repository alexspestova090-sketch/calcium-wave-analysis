#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>

// ============================================================
// 1. Структура для сохранения итогового события (для CSV)
// ============================================================
struct WaveEvent {
    int         id            = 0;
    int         startFrame    = 0;
    int         endFrame      = 0;
    double      startTimeSec  = 0.0;
    double      endTimeSec    = 0.0;
    int         durationFrames= 0;
    double      durationSec   = 0.0;
    double      peakArea      = 0.0;
    cv::Point2d peakCentroid  {};
};

// ============================================================
// 2. Класс "Живой" волны. Обновляется каждый кадр.
// ============================================================
class Wave {
public:
    Wave() = default;
    Wave(int id, int startFrame, cv::Point2d initialCentroid, double initialArea)
        : id_(id), startFrame_(startFrame), lastSeen_(startFrame),
          lastCentroid_(initialCentroid), peakCentroid_(initialCentroid), maxArea_(initialArea) {}

    int         id()             const { return id_; }
    int         startFrame()     const { return startFrame_; }
    int         lastSeen()       const { return lastSeen_; }
    cv::Point2d lastCentroid()   const { return lastCentroid_; }
    double      maxArea()        const { return maxArea_; }
    int         durationFrames() const { return lastSeen_ - startFrame_ + 1; }

    // Метод обновления, когда волна найдена на новом кадре
    void update(cv::Point2d centroid, double area, int frame) {
        lastCentroid_ = centroid;
        lastSeen_     = frame;
        if (area > maxArea_) { 
            maxArea_ = area; 
            peakCentroid_ = centroid; 
        }
    }

    // Конвертация в итоговое событие (когда волна исчезла)
    WaveEvent toEvent(double fps) const {
        WaveEvent ev;
        ev.id             = id_;
        ev.startFrame     = startFrame_;
        ev.endFrame       = lastSeen_;
        ev.startTimeSec   = startFrame_ / fps;
        ev.endTimeSec     = lastSeen_   / fps;
        ev.durationFrames = lastSeen_ - startFrame_ + 1;
        ev.durationSec    = ev.durationFrames / fps;
        ev.peakArea       = maxArea_;
        ev.peakCentroid   = peakCentroid_;
        return ev;
    }

private:
    int          id_          = 0;
    int          startFrame_  = 0;
    int          lastSeen_    = 0;
    cv::Point2d  lastCentroid_{};
    cv::Point2d  peakCentroid_{};
    double       maxArea_     = 0.0;
};

// ============================================================
// 3. Трекер. Связывает контуры между кадрами (по расстоянию).
// ============================================================
class WaveTracker {
public:
    WaveTracker(double fps,
                double distThreshold      = 60.0, // Макс. расстояние смещения волны за кадр
                int    maxMissingFrames   = 5,    // Сколько кадров ждем, если волна пропала
                int    minDurationFrames  = 15)   // Мин. длительность, чтобы записать событие (отсев шума)
        : fps_(fps), distThreshold_(distThreshold),
          maxMissingFrames_(maxMissingFrames),
          minDurationFrames_(minDurationFrames) {}

    // Вызывается каждый кадр детектором
    int update(const std::vector<cv::Point2d>& centroids,
               const std::vector<double>&      areas,
               int currentFrame)
    {
        std::vector<bool> matched(centroids.size(), false);
        std::map<int, Wave> updated;

        // 1. Ищем совпадения для уже существующих волн
        for (auto& [id, wave] : active_) {
            int    bestIdx  = -1;
            double bestDist = distThreshold_;
            
            for (size_t i = 0; i < centroids.size(); ++i) {
                if (matched[i]) continue;
                double d = cv::norm(wave.lastCentroid() - centroids[i]);
                if (d < bestDist) { bestDist = d; bestIdx = (int)i; }
            }

            if (bestIdx >= 0) {
                // Волна продолжается
                wave.update(centroids[bestIdx], areas[bestIdx], currentFrame);
                updated[id] = wave;
                matched[bestIdx] = true;
            } else {
                // Волна пропала. Ждем maxMissingFrames кадров.
                if (currentFrame - wave.lastSeen() > maxMissingFrames_) {
                    // Если она жила достаточно долго — сохраняем!
                    if (wave.durationFrames() > minDurationFrames_)
                        completed_.push_back(wave.toEvent(fps_));
                } else {
                    updated[id] = wave; // Еще держим в буфере
                }
            }
        }

        // 2. Добавляем новые волны (которым не нашлось пары)
        for (size_t i = 0; i < centroids.size(); ++i) {
            if (!matched[i]) {
                updated[nextId_] = Wave(nextId_, currentFrame, centroids[i], areas[i]);
                ++nextId_;
            }
        }

        active_ = std::move(updated);
        return (int)active_.size();
    }

    // Вызывается в самом конце видео (закрывает волны, которые еще горят)
    void closeAll(int finalFrame) {
        for (auto& [id, wave] : active_)
            if (wave.durationFrames() > minDurationFrames_)
                completed_.push_back(wave.toEvent(fps_));
        active_.clear();
    }

    const std::vector<WaveEvent>& events() const { return completed_; }

private:
    double fps_, distThreshold_;
    int    maxMissingFrames_, minDurationFrames_;
    int    nextId_ = 1;
    std::map<int, Wave> active_;
    std::vector<WaveEvent> completed_;
};