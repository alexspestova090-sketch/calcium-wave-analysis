import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter
import os
import sys # <--- Добавили sys для приема пути из C++

# 1. НАСТРОЙКИ ПУТЕЙ
# Если C++ передал путь (sys.argv[1]), берем его. Если нет — берем дефолтный.
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = '/Users/a1/Documents/oop_project/build/svd_clean_test.mp4'

# Автоматически делаем имя для графика на основе имени видео
# (Если видео svd_clean_T.mp4, график будет svd_clean_T_kinematics.png)
base_name = os.path.basename(VIDEO_PATH).replace('.mp4', '')
OUTPUT_PLOT = f'/Users/a1/Documents/oop_project/build/{base_name}_kinematics.png'

if not os.path.exists(VIDEO_PATH):
    print(f"❌ ОШИБКА: Видео не найдено по пути {VIDEO_PATH}")
    exit(1)

print(f"1. Извлечение сигнала из видео: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0: 
    fps = 30.0

signal = []
while True:
    ret, frame = cap.read()
    if not ret: 
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Считаем площадь загоревшихся пикселей на каждом кадре
    signal.append(np.sum(gray > 10)) 
cap.release()

signal = np.array(signal, dtype=np.float32)
t = np.arange(len(signal)) / fps 

if len(signal) < 100:
    print("❌ ОШИБКА: Слишком мало кадров в видео.")
    exit(1)
# 2. УДАЛЕНИЕ КРАЕВЫХ АРТЕФАКТОВ
cut_start = int(5 * fps)
cut_end = int(len(signal) - 5 * fps)
t_cut = t[cut_start:cut_end]
signal_cut = signal[cut_start:cut_end]
# 3. ПРЕОБРАЗОВАНИЕ ГИЛЬБЕРТА
signal_smooth = savgol_filter(signal_cut, window_length=31, polyorder=3)
analytic_signal = hilbert(signal_smooth)

envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# 4. КИНЕМАТИКА (Гладкие производные)
win_len = 51 
velocity = savgol_filter(envelope, window_length=win_len, polyorder=3, deriv=1, delta=1/fps)
acceleration = savgol_filter(envelope, window_length=win_len, polyorder=3, deriv=2, delta=1/fps)
# 5. ПОСТРОЕНИЕ ГРАФИКОВ
print("2. Построение графиков...")
plt.figure(figsize=(14, 10))

# График 1: Сигнал и Огибающая
plt.subplot(3, 1, 1)
plt.plot(t_cut, signal_cut, label='Исходный сигнал', color='gray', alpha=0.5)
plt.plot(t_cut, envelope, label='Огибающая (Амплитуда Гильберта)', color='red', linewidth=2)
plt.title('Преобразование Гильберта: Сигнал и Огибающая')
plt.ylabel('Площадь волны (px)')
plt.legend()
plt.grid(True)

# График 2: Фаза
plt.subplot(3, 1, 2)
plt.plot(t_cut, instantaneous_phase, label='Мгновенная фаза', color='blue')
plt.title('Мгновенная фаза сигнала')
plt.ylabel('Фаза (радианы)')
plt.legend()
plt.grid(True)

# График 3: Кинематика (Зум на одну волну)
zoom_start = int(10 * fps) - cut_start
zoom_end = int(35 * fps) - cut_start
if zoom_start < 0: zoom_start = 0
if zoom_end > len(t_cut): zoom_end = len(t_cut)

plt.subplot(3, 1, 3)
plt.plot(t_cut[zoom_start:zoom_end], velocity[zoom_start:zoom_end], label='Скорость (dA/dt)', color='green', linewidth=2)
plt.plot(t_cut[zoom_start:zoom_end], acceleration[zoom_start:zoom_end], label='Ускорение (d²A/dt²)', color='purple', linestyle='--')
plt.title('Кинематика волны')
plt.xlabel('Время (с)')
plt.ylabel('Скорость / Ускорение')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(OUTPUT_PLOT, dpi=150)

plt.close() 

print(f"✅ Готово! График сохранен в: {OUTPUT_PLOT}")