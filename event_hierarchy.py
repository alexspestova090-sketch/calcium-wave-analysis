from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class DetectedEvent(ABC):
    """Базовый абстрактный класс «Обнаруженное событие»."""
    event_id: int
    start_frame: int
    end_frame: int
    duration_sec: float
    peak_area_px: float
    trajectory: List[Tuple[float, float]]  # Список координат (x, y) по кадрам

    def path_length(self) -> float:
        """Общая длина пройденного пути (в пикселях)."""
        if len(self.trajectory) < 2:
            return 0.0
        pts = np.array(self.trajectory, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def net_displacement(self) -> float:
        """Прямое расстояние от точки старта до точки финиша (в пикселях)."""
        if len(self.trajectory) < 2:
            return 0.0
        start = np.array(self.trajectory[0], dtype=np.float32)
        end = np.array(self.trajectory[-1], dtype=np.float32)
        return float(np.linalg.norm(end - start))

    def straightness_index(self) -> float:
        """Индекс прямолинейности движения (от 0 до 1).
        Близко к 0 — хаотичное дергание на месте (шум).
        Близко к 1 — направленное распространение волны.
        """
        total = self.path_length()
        if total < 1e-5:
            return 0.0
        return self.net_displacement() / total

    @abstractmethod
    def is_valid_wave(self) -> bool:
        """Абстрактный метод: является ли событие целевой волной."""
        pass

    @abstractmethod
    def summary(self) -> str:
        """Абстрактный метод: текстовое описание события."""
        pass


@dataclass
class TrueWave(DetectedEvent):
    """Наследник: «Настоящая кальциевая волна»."""
    def is_valid_wave(self) -> bool:
        return True

    def summary(self) -> str:
        return (
            f"[ВОЛНА #{self.event_id}] Длительность: {self.duration_sec:.2f} с | "
            f"Пик. площадь: {self.peak_area_px:.0f} px | "
            f"Направленность: {self.straightness_index():.2f}"
        )


@dataclass
class Artifact(DetectedEvent):
    """Наследник: «Артефакт / Помеха»."""
    reason: str = "Неизвестная помеха"

    def is_valid_wave(self) -> bool:
        return False

    def summary(self) -> str:
        return (
            f"[АРТЕФАКТ #{self.event_id}] Отброшен ({self.reason}) | "
            f"Площадь: {self.peak_area_px:.0f} px | Длит.: {self.duration_sec:.2f} с"
        )


def classify_event(
    event_id: int,
    start_frame: int,
    end_frame: int,
    duration_sec: float,
    peak_area_px: float,
    trajectory: List[Tuple[float, float]],
    min_area_px: float = 2000.0,
    min_duration_sec: float = 0.5,
    min_straightness: float = 0.25,
) -> DetectedEvent:
    """Классификатор: решает, создать объект TrueWave или Artifact."""
    base_kwargs = dict(
        event_id=event_id,
        start_frame=start_frame,
        end_frame=end_frame,
        duration_sec=duration_sec,
        peak_area_px=peak_area_px,
        trajectory=trajectory,
    )

    if peak_area_px < min_area_px:
        return Artifact(**base_kwargs, reason=f"малая площадь < {min_area_px:.0f} px")

    if duration_sec < min_duration_sec:
        return Artifact(**base_kwargs, reason=f"короткая жизнь < {min_duration_sec:.2f} с")

    # Проверяем хаотичность только если объект успел заметно сдвинуться
    temp = TrueWave(**base_kwargs)
    if temp.path_length() > 15.0 and temp.straightness_index() < min_straightness:
        return Artifact(**base_kwargs, reason=f"хаотичное движение (индекс {temp.straightness_index():.2f})")

    return temp


# Пример запуска поверх готового CSV от C++ конвейера
if __name__ == "__main__":
    csv_file = "build/wave_summary_T.csv"
    df = pd.read_csv(csv_file)
    print(f"Загружено событий из C++: {len(df)}\n")

    for _, row in df.iterrows():
        # Для демонстрации по CSV задаем условную направленную траекторию
        demo_traj = [(0.0, 0.0), (25.0, 30.0), (50.0, 60.0)]
        ev = classify_event(
            event_id=int(row["Wave_Event_ID"]),
            start_frame=int(row["Start_Frame"]),
            end_frame=int(row["End_Frame"]),
            duration_sec=float(row["Duration_Sec"]),
            peak_area_px=float(row["Peak_Area_px"]),
            trajectory=demo_traj,
        )
        print(ev.summary())
