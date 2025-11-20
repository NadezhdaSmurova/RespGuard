"""
RespiraGuard Audio Dashboard (prototype)

⚠️ ВАЖНО:
Это НЕ медицинское устройство и НЕ ставит диагноз.
Код предназначен только для демонстрации анализа дыхания / кашля / хрипов по звуку
в исследовательских и хакатон-проектах.
"""

import os
import cv2
import numpy as np
from IPython.display import Video, display  # удобно в Colab / Jupyter

#import moviepy.editor as mp
import moviepy as mp
import librosa


# -------------------
# Параметры
# -------------------

# Окна для подсчёта эпизодов (сек)
WINDOW_SEC_COUGH = 60.0
WINDOW_SEC_WHEEZE = 60.0

# Пороги "эпизода" по уровню
COUGH_LEVEL_THRESHOLD = 0.6
WHEEZE_LEVEL_THRESHOLD = 0.7  # порог хрипов

# Пороги эпизодов за минуту (обновлено)
COUGH_HIGH_EPISODES = 6   # >=6 эпизодов кашля за 60с → HIGH
COUGH_MED_EPISODES = 3    # 3–5 → MEDIUM

WHEEZE_HIGH_EPISODES = 2  # >=2 эпизода хрипов за 60с → HIGH
WHEEZE_MED_EPISODES = 1   # 1 → MEDIUM

# Пороги дыхания (очень грубо, для прототипа)
BREATH_MED_MIN = 30.0      # 30–35: MEDIUM
BREATH_MED_MAX = 35.0
BREATH_HIGH = 35.0         # >35: HIGH

# Аудио-настройки
TARGET_SR = 16000
HOP_LENGTH = 512


# -------------------
# Вспомогательные функции (замена SciPy)
# -------------------

def simple_find_peaks(x, distance=1, height=None):
    """
    Простейшая замена scipy.signal.find_peaks на NumPy.

    - x: 1D массив
    - distance: минимальное расстояние между пиками (в индексах)
    - height: минимальная высота пика (если None — не учитываем)

    Возвращает:
    - peaks: np.ndarray индексов пиков
    - props: dict с ключом 'peak_heights' (как в SciPy, упрощённо)
    """
    x = np.asarray(x)
    if x.size < 3:
        return np.array([], dtype=int), {"peak_heights": np.array([], dtype=float)}

    # локальные максимумы: x[i-1] < x[i] > x[i+1]
    mid = x[1:-1]
    left = x[:-2]
    right = x[2:]
    candidate_mask = (mid > left) & (mid > right)
    candidate_indices = np.where(candidate_mask)[0] + 1  # сдвиг на 1

    if candidate_indices.size == 0:
        return np.array([], dtype=int), {"peak_heights": np.array([], dtype=float)}

    # фильтр по высоте
    if height is not None:
        h = x[candidate_indices]
        height_mask = h >= height
        candidate_indices = candidate_indices[height_mask]

        if candidate_indices.size == 0:
            return np.array([], dtype=int), {"peak_heights": np.array([], dtype=float)}

    # применяем минимальное расстояние: берём первые подходящие, оставляя >= distance
    if distance is None or distance <= 1:
        peaks = candidate_indices
    else:
        peaks = []
        last_idx = -np.inf
        for idx in candidate_indices:
            if idx - last_idx >= distance:
                peaks.append(idx)
                last_idx = idx
        peaks = np.array(peaks, dtype=int)

    return peaks, {"peak_heights": x[peaks]}


def local_max_filter_1d(x, size):
    """
    Упрощённая замена scipy.ndimage.maximum_filter1d на чистом NumPy.

    Для каждого i берём максимум в окне [i - size//2, i + size//2].
    Края "обрезаем" аккуратно.

    - x: 1D массив
    - size: размер окна (целое число >=1)
    """
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return x
    size = max(int(size), 1)
    half = size // 2

    out = np.empty_like(x)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        out[i] = np.max(x[start:end])
    return out


# -------------------
# Обобщённый счётчик эпизодов с минимальным интервалом
# -------------------

class EpisodeCounter:
    """
    Счётчик эпизодов для аудио-сигнала:
    - prob_threshold     — порог "всплеска" (0..1)
    - window_sec         — окно по времени (сек), в котором считаем эпизоды
    - min_separation_sec — минимальное расстояние между эпизодами (чтобы
                           не считать вдох+выдох как два разных и
                           не ловить дрожание около порога)

    Хранит:
    - self.episodes       — времена эпизодов (для окна)
    - self.total_episodes — общее число эпизодов с начала видео
    """

    def __init__(self, prob_threshold=0.6, window_sec=60.0, min_separation_sec=0.0):
        self.prob_threshold = prob_threshold
        self.window_sec = window_sec
        self.min_separation_sec = min_separation_sec
        self.episodes = []           # времена начала эпизодов (для окна)
        self.total_episodes = 0      # общий счётчик
        self.active = False          # сейчас внутри эпизода или нет
        self.last_episode_time = None

    def update(self, t, p):
        """
        t — время (сек)
        p — "уровень" (0..1) для текущего сигнала (кашель/хрипы)

        Возвращает:
        - n_last_min     — число эпизодов в окне [t - window_sec, t]
        - total_episodes — общее число эпизодов с начала
        """
        # Новый эпизод: переход через порог вверх
        if p > self.prob_threshold and not self.active:
            can_add = (
                self.last_episode_time is None or
                (t - self.last_episode_time) >= self.min_separation_sec
            )
            if can_add:
                self.episodes.append(t)
                self.last_episode_time = t
                self.total_episodes += 1

            self.active = True

        elif p <= self.prob_threshold and self.active:
            # выходим из активного эпизода
            self.active = False

        # чистим старые эпизоды (для скользящего окна)
        self.episodes = [te for te in self.episodes if t - te <= self.window_sec]

        n_last_min = len(self.episodes)
        return n_last_min, self.total_episodes


# -------------------
# Аудио-пайплайн
# -------------------

def extract_audio_features(audio_path, sr=TARGET_SR, hop_length=HOP_LENGTH):
    """
    Считает:
    - times          — времена аудио-кадров
    - cough_level    — нормированный onset envelope (резкие всплески)
    - wheeze_level   — показатель "хрипов"
    - breath_peaks   — моменты "вдохов" по огибающей RMS
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) == 0:
        return {
            "times": np.array([0.0]),
            "cough_level": np.array([0.0]),
            "wheeze_level": np.array([0.0]),
            "breath_peaks": np.array([]),
        }

    # --- базовые вещи: RMS и времена ---
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # 1) Cough level — onset envelope (резкие короткие всплески)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    min_len = min(len(onset_env), len(rms), len(times))
    onset_env = onset_env[:min_len]
    rms = rms[:min_len]
    times = times[:min_len]

    if onset_env.max() > onset_env.min():
        cough_level = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min())
    else:
        cough_level = np.zeros_like(onset_env)

    # 2) Wheeze level — энергия в 400–2000 Гц относительно низких частот, с локальной нормализацией
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_mask = freqs < 400                      # "базовый" низкочастотный фон
    mid_mask = (freqs >= 400) & (freqs <= 2000) # зона хрипов/стридора

    low_energy = (S[low_mask] ** 2).sum(axis=0)
    mid_energy = (S[mid_mask] ** 2).sum(axis=0)

    # подгоним длину спектральных рядов под times/cough_level
    min_len_spec = min(len(mid_energy), len(times))
    mid_energy = mid_energy[:min_len_spec]
    low_energy = low_energy[:min_len_spec]

    wheeze_raw = mid_energy / (low_energy + 1e-8)

    # локальная нормализация по максимуму в окне (около 1 секунды)
    win = int(1.0 * sr / hop_length)  # ~1 секунда
    if win < 3:
        win = 3
    local_max = local_max_filter_1d(wheeze_raw, size=win)
    wheeze_norm = wheeze_raw / (local_max + 1e-8)
    wheeze_norm = np.clip(wheeze_norm, 0.0, 1.0)

    # снижение чувствительности: сжимаем диапазон степенью 2
    wheeze_level = wheeze_norm ** 2

    # 3) дыхание — по сглаженному RMS
    N = 20
    kernel = np.ones(N) / N
    rms_smooth = np.convolve(rms, kernel, mode="same")

    min_distance_frames = int(0.4 * sr / hop_length)   # минимум 0.4 c между вдохами
    height_thresh = np.percentile(rms_smooth, 70)

    peaks, _ = simple_find_peaks(
        rms_smooth,
        distance=max(1, min_distance_frames),
        height=height_thresh,
    )
    breath_peak_times = times[peaks] if len(peaks) > 0 else np.array([])

    # Подравниваем длины всех массивов
    min_len_all = min(len(times), len(cough_level), len(wheeze_level))
    times = times[:min_len_all]
    cough_level = cough_level[:min_len_all]
    wheeze_level = wheeze_level[:min_len_all]

    return {
        "times": times,
        "cough_level": cough_level,
        "wheeze_level": wheeze_level,
        "breath_peaks": breath_peak_times,
    }


def value_at_time(arr_times, arr_values, t):
    """
    Берём значение фичи в момент времени t по ближайшему индексу.
    """
    if len(arr_times) == 0:
        return 0.0
    idx = np.searchsorted(arr_times, t)
    idx = np.clip(idx, 0, len(arr_times) - 1)
    return float(arr_values[idx])


def breath_rate_at_time(breath_peak_times, t, window_sec=20.0):
    """
    Оценка частоты дыхания (вдохов/мин) вокруг момента t.
    """
    if len(breath_peak_times) == 0:
        return 0.0
    t_min = t - window_sec / 2
    t_max = t + window_sec / 2
    mask = (breath_peak_times >= t_min) & (breath_peak_times <= t_max)
    n_peaks = mask.sum()
    if n_peaks <= 1:
        return 0.0
    return n_peaks * 60.0 / window_sec


# -------------------
# Медицинская логика ALERT (упрощённая)
# -------------------

def compute_risk_alert(n_cough_last_min, n_wheeze_last_min, breath_rate):
    """
    Возвращает "LOW" / "MEDIUM" / "HIGH" по упрощённой логике.
    Использует только эпизоды за последнюю минуту.
    """
    high = (
        (n_cough_last_min >= COUGH_HIGH_EPISODES) or
        (n_wheeze_last_min >= WHEEZE_HIGH_EPISODES) or
        (breath_rate > BREATH_HIGH)
    )

    medium = (
        (n_cough_last_min >= COUGH_MED_EPISODES) or
        (n_wheeze_last_min >= WHEEZE_MED_EPISODES) or
        (BREATH_MED_MIN <= breath_rate <= BREATH_MED_MAX)
    )

    if high:
        return "HIGH"
    elif medium:
        return "MEDIUM"
    else:
        return "LOW"


def risk_bar_ratio(alert):
    if alert == "HIGH":
        return 1.0
    elif alert == "MEDIUM":
        return 0.66
    else:
        return 0.33


# -------------------
# Визуальный оверлей
# -------------------

def draw_overlay(frame, t,
                 cough_level, wheeze_level,
                 breath_rate,
                 n_cough_last_min, n_cough_total,
                 n_wheeze_last_min, n_wheeze_total,
                 alert):
    """
    Рисует на кадре информационную панель:
    - уровни кашля/хрипов
    - дыхание
    - эпизоды за последнюю минуту
    - суммарные эпизоды
    - индикатор RISK ALERT
    """
    if alert == "HIGH":
        alert_color = (0, 0, 255)
    elif alert == "MEDIUM":
        alert_color = (0, 255, 255)
    else:
        alert_color = (0, 255, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2

    cv2.putText(frame, f"t = {t:5.1f}s",                           (10, 30),  font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Cough level:        {cough_level:0.2f}",  (10, 60),  font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Wheeze level:       {wheeze_level:0.2f}", (10, 90),  font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Breathing:          {breath_rate:0.1f}/min",
                (10, 120), font, scale, (255, 255, 255), thickness)

    cv2.putText(frame, f"Cough (last 60s):   {n_cough_last_min}",  (10, 150), font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Cough total:        {n_cough_total}",     (10, 180), font, scale, (255, 255, 255), thickness)

    cv2.putText(frame, f"Wheeze (last 60s):  {n_wheeze_last_min}", (10, 210), font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Wheeze total:       {n_wheeze_total}",    (10, 240), font, scale, (255, 255, 255), thickness)

    cv2.putText(frame, "RISK ALERT:", (10, 275), font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, alert,         (190, 275), font, scale, alert_color, thickness)

    # Полоска риска
    bar_x, bar_y, bar_h, bar_max_w = 10, 305, 25, 220
    ratio = risk_bar_ratio(alert)
    bar_w = int(bar_max_w * ratio)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_max_w, bar_y + bar_h),
                  (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  alert_color, -1)
    if bar_w < bar_max_w:
        cv2.rectangle(frame,
                      (bar_x + bar_w, bar_y),
                      (bar_x + bar_max_w, bar_y + bar_h),
                      (50, 50, 50), -1)

    return frame


# -------------------
# Главная функция обработки видео
# -------------------

def process_video(video_path, output_path="output_med_dashboard.mp4",
                  print_wheeze_debug=False, show_in_notebook=True):
    """
    Полный пайплайн:
    1) читаем видео и аудио
    2) считаем аудио-фичи
    3) идём по кадрам, обновляем счётчики эпизодов
    4) рисуем оверлей
    5) склеиваем готовое видео с исходным аудио
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео-файл не найден: {video_path}")

    print("Шаг 1: читаем видео и аудио...")
    clip = mp.VideoFileClip(video_path)
    audio = clip.audio

    temp_audio_path = "temp_audio.wav"
    temp_video_path = "temp_no_audio.mp4"

    # сохраняем аудио
    audio.write_audiofile(temp_audio_path, fps=TARGET_SR)
    #audio.write_audiofile(temp_audio_path, fps=TARGET_SR, verbose=False, logger=None)

    print("Шаг 2: извлекаем аудио-фичи...")
    feats = extract_audio_features(temp_audio_path, sr=TARGET_SR, hop_length=HOP_LENGTH)
    times_a = feats["times"]
    cough_level_arr = feats["cough_level"]
    wheeze_level_arr = feats["wheeze_level"]
    breath_peaks = feats["breath_peaks"]

    print("Шаг 3: обрабатываем видео покадрово...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        clip.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # разные минимальные интервалы для кашля и хрипов
    cough_counter = EpisodeCounter(
        prob_threshold=COUGH_LEVEL_THRESHOLD,
        window_sec=WINDOW_SEC_COUGH,
        min_separation_sec=0.3,   # кашель может быть частым
    )
    wheeze_counter = EpisodeCounter(
        prob_threshold=WHEEZE_LEVEL_THRESHOLD,
        window_sec=WINDOW_SEC_WHEEZE,
        min_separation_sec=1.0,   # хрип/стридор считаем не чаще раза в ~секунду
    )

    frame_idx = 0

    if print_wheeze_debug:
        print("time(s), WHEEZE_LEVEL  (печатаю только > 0.01)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        cough_level = value_at_time(times_a, cough_level_arr, t)
        wheeze_level = value_at_time(times_a, wheeze_level_arr, t)
        breath_rate = breath_rate_at_time(breath_peaks, t, window_sec=20.0)

        # печать для анализа чувствительности WHEEZE_LEVEL
        if print_wheeze_debug and wheeze_level > 0.01:
            print(f"time={t:.3f}, WHEEZE_LEVEL={wheeze_level:.3f}")

        # обновляем счётчики
        n_cough_last_min, n_cough_total = cough_counter.update(t, cough_level)
        n_wheeze_last_min, n_wheeze_total = wheeze_counter.update(t, wheeze_level)

        alert = compute_risk_alert(n_cough_last_min, n_wheeze_last_min, breath_rate)

        vis = draw_overlay(
            frame.copy(), t,
            cough_level, wheeze_level,
            breath_rate,
            n_cough_last_min, n_cough_total,
            n_wheeze_last_min, n_wheeze_total,
            alert
        )
        out.write(vis)
        frame_idx += 1

    cap.release()
    out.release()

    print("Шаг 4: склеиваем с аудио и сохраняем...")
    processed = mp.VideoFileClip(temp_video_path)
    #final = processed.set_audio(audio)
    final = processed.with_audio(audio)

    final.write_videofile(output_path,
                          codec="libx264",
                          audio_codec="aac")

    # чистим временные файлы (по желанию)
    try:
        os.remove(temp_audio_path)
    except OSError:
        pass
    try:
        os.remove(temp_video_path)
    except OSError:
        pass

    clip.close()
    processed.close()
    final.close()

    print("Готово.")
    if show_in_notebook:
        try:
            print("Показываю результат в ноутбуке:")
            display(Video(output_path, embed=True))
        except Exception:
            # если нет IPython окружения — просто пропускаем показ
            pass


# -------------------
# CLI-запуск
# -------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "RespiraGuard Audio Dashboard prototype: "
            "анализ кашля / хрипов / дыхания по аудио в видео.\n"
            "⚠️ НЕ является медицинским устройством."
        )
    )
    parser.add_argument(
        "input_video",
        help="Путь к входному видеофайлу (mp4).",
    )
    parser.add_argument(
        "--output",
        default="output_med_dashboard.mp4",
        help="Путь для сохранения обработанного видео (mp4).",
    )
    parser.add_argument(
        "--no-notebook",
        action="store_true",
        help="Не пытаться отображать результат через IPython.display (для чистого CLI).",
    )
    parser.add_argument(
        "--wheeze-debug",
        action="store_true",
        help="Печатать значения WHEEZE_LEVEL > 0.01 для отладки.",
    )

    args = parser.parse_args()

    process_video(
        video_path=args.input_video,
        output_path=args.output,
        print_wheeze_debug=args.wheeze_debug,
        show_in_notebook=not args.no_notebook,
    )


if __name__ == "__main__":
    main()
