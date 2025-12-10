"""
RespiraGuard Audio Dashboard (prototype)

⚠️ Not a medical device and does not provide any diagnosis.

Simple demo: audio-based cough / wheeze / breathing analysis
for research and hackathon experiments.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from IPython.display import Video, display  # handy in Colab / Jupyter / удобно в Colab / Jupyter

#import moviepy.editor as mp
import moviepy as mp
import librosa

# try to import tqdm for  progress bar over videos
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# -------------------
# Parameters
# -------------------

# Windows for counting episodes (sec)
WINDOW_SEC_COUGH = 60.0
WINDOW_SEC_WHEEZE = 60.0

# Episode level thresholds
COUGH_LEVEL_THRESHOLD = 0.6
WHEEZE_LEVEL_THRESHOLD = 0.7

# Episode thresholds per minute
COUGH_HIGH_EPISODES = 6   # HIGH
COUGH_MED_EPISODES = 3    # MEDIUM

WHEEZE_HIGH_EPISODES = 5  # HIGH
WHEEZE_MED_EPISODES = 3   # MEDIUM

# Very rough breathing thresholds for prototype
BREATH_MED_MIN = 30.0      # 30–35 → MEDIUM
BREATH_MED_MAX = 35.0
BREATH_HIGH = 35.0         # >35 → HIGH

# Audio settings
TARGET_SR = 16000
HOP_LENGTH = 512


# -------------------
# Small SciPy-free helpers
# -------------------

def simple_find_peaks(x, distance=1, height=None):
    """
    This function is a tiny NumPy-only replacement for scipy.signal.find_peaks
    so the project doesn’t depend on SciPy and can run easily in lightweight
    or minimal environments.

    x: 1D array
    distance: min distance between peaks
    height: minimal peak height (or None)
    """
    x = np.asarray(x)
    if x.size < 3:
        return np.array([], dtype=int), {"peak_heights": np.array([], dtype=float)}

    # local maxima / локальные максимумы
    mid = x[1:-1]
    left = x[:-2]
    right = x[2:]
    candidate_mask = (mid > left) & (mid > right)
    candidate_indices = np.where(candidate_mask)[0] + 1

    if candidate_indices.size == 0:
        return np.array([], dtype=int), {"peak_heights": np.array([], dtype=float)}

    # optional height filter / фильтр по высоте
    if height is not None:
        h = x[candidate_indices]
        height_mask = h >= height
        candidate_indices = candidate_indices[height_mask]

        if candidate_indices.size == 0:
            return np.array([], dtype=int), {"peak_heights": np.array([], dtype=float)}

    # enforce minimal distance
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
    Simple 1D moving max filter on NumPy.
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
# EpisodeCounter with minimal separation
# -------------------

class EpisodeCounter:
    """
    Episode counter for audio signal (cough / wheeze).

    prob_threshold: level threshold (0..1) / порог уровня
    window_sec: sliding window in seconds / окно по времени
    min_separation_sec: minimal gap between episodes / мин. пауза между эпизодами
    """

    def __init__(self, prob_threshold=0.6, window_sec=60.0, min_separation_sec=0.0):
        self.prob_threshold = prob_threshold
        self.window_sec = window_sec
        self.min_separation_sec = min_separation_sec
        self.episodes = []           # times of episodes
        self.total_episodes = 0      # total count
        self.active = False          # inside episode or not
        self.last_episode_time = None

    def update(self, t, p):
        """
        t — time in seconds
        p — current level 0..1

        Returns:
        n_last_min — episodes in [t - window_sec, t]
        total_episodes — total count from start
        """
        # new episode: crossing threshold upwards
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
            # leaving episode
            self.active = False

        # keep only episodes inside window
        self.episodes = [te for te in self.episodes if t - te <= self.window_sec]

        n_last_min = len(self.episodes)
        return n_last_min, self.total_episodes


# -------------------
# Audio pipeline
# -------------------

def extract_audio_features(audio_path, sr=TARGET_SR, hop_length=HOP_LENGTH):
    """
    Compute simple audio features for prototype.

    Returns dict with:
    - times: frame times
    - cough_level: 0..1 onset envelope
    - wheeze_level: 0..1 wheeze proxy
    - breath_peaks: times of inhalation peaks
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) == 0:
        return {
            "times": np.array([0.0]),
            "cough_level": np.array([0.0]),
            "wheeze_level": np.array([0.0]),
            "breath_peaks": np.array([]),
        }

    # basic RMS and time axis
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # 1) cough level from onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    min_len = min(len(onset_env), len(rms), len(times))
    onset_env = onset_env[:min_len]
    rms = rms[:min_len]
    times = times[:min_len]

    if onset_env.max() > onset_env.min():
        cough_level = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min())
    else:
        cough_level = np.zeros_like(onset_env)

    # 2) wheeze level: mid freq energy vs low freq
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_mask = freqs < 400                      # low band  низкие частоты
    mid_mask = (freqs >= 400) & (freqs <= 2000) # mid band with wheeze средние частоты (хрипы)

    low_energy = (S[low_mask] ** 2).sum(axis=0)
    mid_energy = (S[mid_mask] ** 2).sum(axis=0)

    # align spectral length with times
    min_len_spec = min(len(mid_energy), len(times))
    mid_energy = mid_energy[:min_len_spec]
    low_energy = low_energy[:min_len_spec]

    wheeze_raw = mid_energy / (low_energy + 1e-8)

    # local normalization (≈1 s window) / локальная нормализация (окно ~1 сек)
    win = int(1.0 * sr / hop_length)
    if win < 3:
        win = 3
    local_max = local_max_filter_1d(wheeze_raw, size=win)
    wheeze_norm = wheeze_raw / (local_max + 1e-8)
    wheeze_norm = np.clip(wheeze_norm, 0.0, 1.0)

    # compress sensitivity / немного сжимаю диапазон чувствительности
    wheeze_level = wheeze_norm ** 2

    # 3) breathing from smoothed RMS / дыхание по сглаженному RMS
    N = 20
    kernel = np.ones(N) / N
    rms_smooth = np.convolve(rms, kernel, mode="same")

    min_distance_frames = int(0.4 * sr / hop_length)   # min 0.4 s between breaths / мин. 0.4 с между вдохами
    height_thresh = np.percentile(rms_smooth, 70)

    peaks, _ = simple_find_peaks(
        rms_smooth,
        distance=max(1, min_distance_frames),
        height=height_thresh,
    )
    breath_peak_times = times[peaks] if len(peaks) > 0 else np.array([])

    # align lengths of arrays / подравниваю длины массивов
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
    Get feature value at time t via nearest index.
    Беру значение фичи в момент t по ближайшему индексу.
    """
    if len(arr_times) == 0:
        return 0.0
    idx = np.searchsorted(arr_times, t)
    idx = np.clip(idx, 0, len(arr_times) - 1)
    return float(arr_values[idx])


def breath_rate_at_time(breath_peak_times, t, window_sec=20.0):
    """
    Approximate breathing rate (breaths/min) around time t.
    Приближённая частота дыхания (вдохов/мин) около момента t.
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
# Simple ALERT logic
# -------------------

def compute_risk_alert(n_cough_last_min, n_wheeze_last_min, breath_rate):
    """
    Return "LOW" / "MEDIUM" / "HIGH" by simple thresholds.
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


def compute_cough_risk(n_cough_last_min):
    """
    Per-parameter risk for cough.
    Уровень риска по кашлю только по эпизодам за последнюю минуту.
    """
    if n_cough_last_min >= COUGH_HIGH_EPISODES:
        return "HIGH"
    elif n_cough_last_min >= COUGH_MED_EPISODES:
        return "MEDIUM"
    else:
        return "LOW"


def compute_wheeze_risk(n_wheeze_last_min):
    """
    Per-parameter risk for wheeze.
    Уровень риска по хрипам только по эпизодам за последнюю минуту.
    """
    if n_wheeze_last_min >= WHEEZE_HIGH_EPISODES:
        return "HIGH"
    elif n_wheeze_last_min >= WHEEZE_MED_EPISODES:
        return "MEDIUM"
    else:
        return "LOW"


def compute_breath_risk(breath_rate):
    """
    Per-parameter risk for breathing.
    Уровень риска по частоте дыхания.
    """
    if breath_rate > BREATH_HIGH:
        return "HIGH"
    elif BREATH_MED_MIN <= breath_rate <= BREATH_MED_MAX:
        return "MEDIUM"
    else:
        return "LOW"


def risk_to_ratio_and_color(risk_level):
    """
    Map per-parameter risk to bar ratio and color.
    Перевожу риск параметра в длину полоски и цвет.
    """
    if risk_level == "HIGH":
        return 1.0, (0, 0, 255)      # красный
    elif risk_level == "MEDIUM":
        return 0.66, (0, 255, 255)   # жёлтый
    else:
        return 0.33, (0, 255, 0)     # зелёный


# -------------------
# Visual overlay
# -------------------

def draw_overlay(frame, t,
                 cough_level, wheeze_level,
                 breath_rate,
                 n_cough_last_min, n_cough_total,
                 n_wheeze_last_min, n_wheeze_total,
                 alert):
    """
    Рисую простой дашборд на кадре.

    - Под BREATH / COUGH / WHEEZE свои полоски риска
    - Больше вертикальных отступов, полоски шире
    - Крупный ALERT / STATUS OK внизу
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2

    # --- базовый текст ---
    y_breath = 80
    y_cough1 = 155
    y_cough2 = 185
    y_wheeze1 = 260
    y_wheeze2 = 290

    cv2.putText(frame, f"t = {t:5.1f}s", (10, 40),
                font, scale, (255, 255, 255), thickness)

    # BREATHING
    cv2.putText(frame, f"Breathing: {breath_rate:0.1f}/min",
                (10, y_breath), font, scale, (255, 255, 255), thickness)

    # COUGH
    cv2.putText(frame, f"Cough (last 60s): {n_cough_last_min}",
                (10, y_cough1), font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Cough total: {n_cough_total}",
                (10, y_cough2), font, scale, (255, 255, 255), thickness)

    # WHEEZE
    cv2.putText(frame, f"Wheeze (last 60s): {n_wheeze_last_min}",
                (10, y_wheeze1), font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Wheeze total: {n_wheeze_total}",
                (10, y_wheeze2), font, scale, (255, 255, 255), thickness)

    # --- вычисляем риск для полосок ---
    breath_risk = compute_breath_risk(breath_rate)
    cough_risk = compute_cough_risk(n_cough_last_min)
    wheeze_risk = compute_wheeze_risk(n_wheeze_last_min)

    # --- настройки полосок ---
    bar_x = 10
    bar_max_w = 320   # шире
    bar_h = 22

    def draw_param_bar(risk, base_y):
        ratio, color = risk_to_ratio_and_color(risk)
        bar_w = int(bar_max_w * ratio)

        # фон
        cv2.rectangle(frame,
                      (bar_x, base_y),
                      (bar_x + bar_max_w, base_y + bar_h),
                      (40, 40, 40), -1)

        # заполненная часть
        cv2.rectangle(frame,
                      (bar_x, base_y),
                      (bar_x + bar_w, base_y + bar_h),
                      color, -1)

        # рамка
        cv2.rectangle(frame,
                      (bar_x, base_y),
                      (bar_x + bar_max_w, base_y + bar_h),
                      (0, 0, 0), 2)

        # текст LOW / MEDIUM / HIGH
        text = risk
        text_scale = 0.6
        text_thick = 2
        text_x = bar_x + 5
        text_y = base_y + bar_h - 5
        cv2.putText(frame, text, (text_x, text_y),
                    font, text_scale, (255, 255, 255), text_thick)

    # полоска под BREATH
    draw_param_bar(breath_risk, base_y=y_breath + 10)

    # полоска под COUGH
    draw_param_bar(cough_risk, base_y=y_cough2 + 10)

    # полоска под WHEEZE
    draw_param_bar(wheeze_risk, base_y=y_wheeze2 + 10)

    # --- крупный общий статус внизу ---
    # смещаем пониже и делаем крупнее
    alert_y = y_wheeze2 + 80
    if alert in ("HIGH", "MEDIUM"):
        cv2.putText(frame, f"ALERT: {alert}",
                    (10, alert_y),
                    font, 1.4, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "STATUS: OK",
                    (10, alert_y),
                    font, 1.2, (0, 255, 0), 3)

    # --- маленькая подпись внизу в углу ---
    h, w = frame.shape[:2]
    disclaimer = "*Experimental demo. Not for medical use."
    small_scale = 0.45
    small_thick = 1
    text_size, _ = cv2.getTextSize(disclaimer, font, small_scale, small_thick)
    text_w, text_h = text_size

    margin = 10
    text_x = w - text_w - margin
    text_y = h - margin

    cv2.putText(frame,
                disclaimer,
                (text_x, text_y),
                font,
                small_scale,
                (200, 200, 200),
                small_thick)

    return frame


# -------------------
# Main video processing
# -------------------

def process_video(video_path, output_path="output_med_dashboard.mp4",
                  print_wheeze_debug=False, show_in_notebook=True):
    """
    Full prototype pipeline for one video.
    Полный пайплайн прототипа для одного видео.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео-файл не найден: {video_path}")

    print(f"\n=== Processing: {video_path} ===")
    print("Шаг 1: читаю видео и аудио...")
    clip = mp.VideoFileClip(video_path)
    audio = clip.audio

    temp_audio_path = "temp_audio.wav"
    temp_video_path = "temp_no_audio.mp4"

    # save audio track / сохраняю аудио-дорожку
    audio.write_audiofile(temp_audio_path, fps=TARGET_SR)

    print("Шаг 2: извлекаю аудио-фичи...")
    feats = extract_audio_features(temp_audio_path, sr=TARGET_SR, hop_length=HOP_LENGTH)
    times_a = feats["times"]
    cough_level_arr = feats["cough_level"]
    wheeze_level_arr = feats["wheeze_level"]
    breath_peaks = feats["breath_peaks"]

    print("Шаг 3: обрабатываю видео покадрово...")
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    cough_counter = EpisodeCounter(
        prob_threshold=COUGH_LEVEL_THRESHOLD,
        window_sec=WINDOW_SEC_COUGH,
        min_separation_sec=0.3,
    )
    wheeze_counter = EpisodeCounter(
        prob_threshold=WHEEZE_LEVEL_THRESHOLD,
        window_sec=WINDOW_SEC_WHEEZE,
        min_separation_sec=1.0,
    )

    frame_idx = 0

    if print_wheeze_debug:
        print("time(s), WHEEZE_LEVEL  (печатаю только > 0.01)")

    # простой прогрессбар по кадрам / simple per-frame progress bar
    def print_frame_progress(idx):
        if total_frames:
            pct = 100.0 * idx / total_frames
            print(f"\rКадр {idx}/{total_frames} ({pct:5.1f}%)", end="")
        else:
            print(f"\rКадр {idx}", end="")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        cough_level = value_at_time(times_a, cough_level_arr, t)
        wheeze_level = value_at_time(times_a, wheeze_level_arr, t)
        breath_rate = breath_rate_at_time(breath_peaks, t, window_sec=20.0)

        if print_wheeze_debug and wheeze_level > 0.01:
            print(f"\n time={t:.3f}, WHEEZE_LEVEL={wheeze_level:.3f}")

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

        print_frame_progress(frame_idx)

    print()

    cap.release()
    out.release()

    print("Шаг 4: склеиваю с аудио и сохраняю...")
    processed = mp.VideoFileClip(temp_video_path)
    final = processed.with_audio(audio)

    final.write_videofile(output_path,
                          codec="libx264",
                          audio_codec="aac")

    # clean up temp files
    for p in (temp_audio_path, temp_video_path):
        try:
            os.remove(p)
        except OSError:
            pass

    clip.close()
    processed.close()
    final.close()

    print(f"Готово: {output_path}")
    if show_in_notebook:
        try:
            print("Показываю результат в ноутбуке:")
            display(Video(output_path, embed=True))
        except Exception:
            pass


# -------------------
# Batch processing / Пакетная обработка
# -------------------

def process_all_videos(input_dir="examples/input",
                       output_dir="examples/output",
                       print_wheeze_debug=False,
                       show_in_notebook=False):
    """
    I run all .mp4 files from input_dir and save results to output_dir.
    Прогоняю все .mp4 из input_dir и сохраняю в output_dir.

    input:  video.mp4  →  output_video.mp4
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(input_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"В папке {input_dir} нет .mp4 файлов.")
        return

    print(f"Нашла {len(mp4_files)} видео в {input_dir}.")

    iterable = mp4_files
    if tqdm is not None:
        iterable = tqdm(mp4_files, desc="Обработка видео", unit="video")

    for video_path in iterable:
        out_name = "output_" + video_path.name
        out_path = output_dir / out_name

        process_video(
            video_path=str(video_path),
            output_path=str(out_path),
            print_wheeze_debug=print_wheeze_debug,
            show_in_notebook=show_in_notebook,
        )


# -------------------
# CLI entry point
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
        "--input-dir",
        default="examples/input",
        help="Folder with input .mp4, files examples/input",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/output",
        help="Folder for processed .mp4, examples/output",
    )
    parser.add_argument(
        "--no-notebook",
        action="store_true",
        help="Skip IPython display / Не показывать результат через IPython.display.",
    )
    parser.add_argument(
        "--wheeze-debug",
        action="store_true",
        help="Print WHEEZE_LEVEL > 0.01 for debug / Печатать значения WHEEZE_LEVEL > 0.01 для отладки.",
    )

    args = parser.parse_args()

    process_all_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        print_wheeze_debug=args.wheeze_debug,
        show_in_notebook=not args.no_notebook,
    )


if __name__ == "__main__":
    main()
