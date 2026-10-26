#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdlib> // Для std::system (вызов Питона)
#include <opencv2/opencv.hpp>

// Подключаем наши модули конвейера
#include "FrameProcessor.hpp"
#include "Stabilizer.hpp"
#include "BackgroundSubtractor.hpp"
#include "Detector.hpp"

int main() {
    std::cout << "=== ЗАПУСК ПОЛНОГО КОНВЕЙЕРА (СЕРИЯ T) ===" << std::endl;

    // --- ПУТИ К ФАЙЛАМ ---
    // Входное сырое видео (серия T)
    std::string videoPath = "/Users/a1/Documents/видео_анализ/220702_T.mp4"; 
    
    // Куда сохраняем результаты (везде добавили _T)
    std::string outPath   = "build/final_vis_T.mp4";
    std::string csvPath   = "build/wave_summary_T.csv";
    // ВОТ ЭТА СТРОЧКА БЫЛА ПРОПУЩЕНА - чистое видео для Питона:
    std::string cleanOutPath = "build/svd_clean_T.mp4"; 

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть видео! Проверь путь: " << videoPath << std::endl;
        return -1;
    }

    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    int totalFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    // СОБИРАЕМ КОНВЕЙЕР 
    auto stab = std::make_shared<Stabilizer>();
    auto bgSub = std::make_shared<BackgroundSubtractor>();
    auto det = std::make_shared<Detector>(fps, 5, 50.0, 5.0, 2000.0, 25);

    Pipeline pipe;
    pipe.addProcessor(stab);
    pipe.addProcessor(bgSub);
    pipe.addProcessor(det);

    // ПРОХОД 1: ОБУЧЕНИЕ (Поиск вспышки)
    std::cout << "Проход 1: Обучение..." << std::endl;
    cv::Mat frame;
    int idx = 0;
    while (cap.read(frame)) { 
        pipe.train(frame, idx++); 
    }
    pipe.finalizeTraining(idx);
    std::cout << "Обучение завершено." << std::endl;

    // ПРОХОД 2: ОБРАБОТКА И ВИЗУАЛИЗАЦИЯ 
    std::cout << "Проход 2: Детекция и наложение маски..." << std::endl;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Перематываем в начало
    idx = 0;
    
    // Открываем запись для видео с маской
    cv::VideoWriter writer(outPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(W, H));
    // Открываем запись для ЧИСТОГО видео (для Питона)
    cv::VideoWriter cleanWriter(cleanOutPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(W, H));

    while (cap.read(frame)) {
        // Прогоняем через конвейер (внутри кадр станет черно-белым и очищенным)
        pipe.process(frame, idx++); 
        
        // ЗАПИСЫВАЕМ ЧИСТЫЙ КАДР (без маски) в отдельный файл для Питона!
        cleanWriter.write(frame);

        // Берем оригинальный (но стабилизированный) ЦВЕТНОЙ кадр для отрисовки
        cv::Mat display = stab->getLastStabilized().clone();
        
        // Получаем контуры, найденные детектором
        const auto& contours = det->getLastContours();
        
        // Рисуем маску поверх цветного кадра
        if (!contours.empty()) {
            cv::Mat overlay = display.clone();
            for (auto& c : contours) {
                std::vector<std::vector<cv::Point>> single = { c };
                cv::drawContours(overlay, single, -1, cv::Scalar(0, 0, 255), -1); // Заливка
                cv::drawContours(display, single, -1, cv::Scalar(0, 255, 0), 2);  // Обводка
            }
            cv::addWeighted(overlay, 0.25, display, 0.75, 0, display);
        }

        // Текст с информацией
        std::string text = "Waves: " + std::to_string(det->getActiveWaves()) + 
                           " | Area: " + std::to_string((int)det->getTotalArea());
        cv::rectangle(display, cv::Rect(5, 5, 380, 40), cv::Scalar(0, 0, 0), -1);
        cv::putText(display, text, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        writer.write(display);
        
        if (idx % 200 == 0) std::cout << "\rОбработано: " << idx << " / " << totalFrames << std::flush;
    }
    
    pipe.finalize();
    cleanWriter.release(); // Закрываем запись чистого видео
    std::cout << "\n\nОбработка видео завершена!" << std::endl;

    //ЗАПИСЬ СОБЫТИЙ В CSV 
    std::ofstream csv(csvPath);
    csv << "Wave_Event_ID,Start_Frame,Start_Time_sec,End_Frame,End_Time_sec,Duration_Frames,Duration_Sec,Peak_Area_px\n";
    const auto& events = det->tracker().events();
    for (const auto& ev : events) {
        csv << ev.id << "," << ev.startFrame << "," << ev.startTimeSec << ","
            << ev.endFrame << "," << ev.endTimeSec << "," << ev.durationFrames << ","
            << ev.durationSec << "," << ev.peakArea << "\n";
    }
    csv.close();

    std::cout << "Найдено событий: " << events.size() << "\nВидео сохранено: " << outPath << std::endl;

    
    // АВТОМАТИЧЕСКИЙ ЗАПУСК PYTHON-СКРИПТА (ПУНКТ 2)
    
    std::cout << "\n=== ЗАПУСК КИНЕМАТИЧЕСКОГО АНАЛИЗА (PYTHON) ===" << std::endl;
    
    // ВАЖНО: Мы передаем путь к новому чистому видео прямо в Питон!
    std::string py_cmd = "python3 hilbert_analysis.py " + cleanOutPath;
    int py_result = std::system(py_cmd.c_str());

    if (py_result == 0) {
        std::cout << "✅ Python скрипт успешно отработал!" << std::endl;
    } else {
        std::cerr << "❌ Ошибка при запуске Python скрипта." << std::endl;
    }

    return 0;
}