# File: video_splitter.py
"""
视频分割工具 (Python 3.14 / tkinter GUI)
功能: 选择视频 -> 设置每段最大时长/大小 -> 可选倍速(整段 / 仅无人声部分) -> 分割为 MP4
      无人声判定内置智能人声识别(VAD), 可把嗡嗡声/风扇声等稳态噪声与人说话区分开
依赖: pip install imageio-ffmpeg   (自带 ffmpeg 二进制, 无需安装系统软件)
      numpy (已安装, 用于人声识别; 缺失时自动退回音量阈值方案)
输出: 与源视频同一目录, 命名为 "原文件名-1.mp4"、"原文件名-2.mp4" ...
"""

from __future__ import annotations

import math
import os
import queue
import re
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

GB = 1024 ** 3
SIZE_MARGIN = 0.97           # 体积安全余量 (容器封装开销 + 码率波动)
DUR_TOLERANCE = 1.02         # 时长校验容差 (关键帧对齐造成的微小超出)
MAX_RETRY = 4                # 分割校验失败后的最大重切次数

SPEED_MIN = 0.5              # 倍速滑块下限
SPEED_MAX = 8.0              # 倍速滑块上限
SPEED_STEP = 0.1             # 倍速精度
MAX_OUTPUT_FPS = 60.0        # 倍速后输出帧率上限 (超出则丢帧, 体积才会真正下降)

# ---- 人声识别 (VAD) 参数 ----
VAD_SR = 16000               # 分析用采样率
VAD_FRAME = 1024             # 帧长 64 ms
VAD_HOP = 512                # 帧移 32 ms
VAD_NFFT = 2048              # FFT 点数 (零填充, 用于线性自相关)

SENSITIVITY_DB = {"低（只保留清晰人声）": 9.0,
                  "中（推荐）": 6.0,
                  "高（宁可多保留）": 4.0}
AC_THRESH = 0.30             # 自相关峰阈值: 高于此值认为存在谐波(浊音)
FLAT_THRESH = 0.38           # 谱平坦度阈值: 低于此值认为有共振峰结构
LOUD_OVERRIDE_DB = 14.0      # 信噪比高出阈值这么多时无条件视为人声
STATIONARY_STD_DB = 1.2      # 能量起伏低于此值 -> 稳态噪声(嗡嗡声)
STATIONARY_AC_MAX = 0.50     # 稳态否决规则的谐波性上限
MOD_WINDOW_SEC = 0.5         # 计算能量起伏的滑动窗长
FLOOR_BLOCK_SEC = 5.0        # 噪声底噪估计分块长度
FLOOR_PCT = 10               # 噪声底噪取块内百分位

SPEECH_MIN_DUR = 0.20        # 短于此长度的"人声"视为杂音丢弃 (秒)
SPEECH_PAD = 0.25            # 人声前后各保留的缓冲 (秒), 防止吞字头字尾
MIN_SPEEDUP_DUR = 1.5        # 默认: 只有超过这么长的无人声片段才加速 (秒)

DENOISE_FILTER = "highpass=f=85,afftdn=nr=12:nf=-25:tn=1"

PIECE_CAPS = (200, 80, 30)   # 变速片段数量上限 (失败后依次降级重试)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
TIME_RE = re.compile(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Video:\s*([\w\d]+)")
AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:\s*([\w\d]+)")
FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
TBR_RE = re.compile(r"(\d+(?:\.\d+)?)k?\s*tbr")
SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


# --------------------------------------------------------------------------- #
# FFmpeg 定位与通用工具
# --------------------------------------------------------------------------- #

def find_ffmpeg() -> str | None:
    """优先使用 imageio-ffmpeg 自带的二进制, 其次使用系统 PATH 中的 ffmpeg。"""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


FFMPEG = find_ffmpeg()


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} TB"


def human_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def sanitize_base(name: str) -> str:
    """'%' 是 ffmpeg 输出模板的特殊字符, 同时清理非法文件名字符。"""
    for ch in '%<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name.strip() or "output"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# 媒体信息探测
# --------------------------------------------------------------------------- #

def probe_media(path: str) -> dict:
    """通过解析 ffmpeg 的输出获取时长、帧率与流信息 (imageio-ffmpeg 不含 ffprobe)。"""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    text = proc.stderr or ""

    duration = None
    m = DURATION_RE.search(text)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        if duration <= 0:
            duration = None

    vm = VIDEO_RE.search(text)
    am = AUDIO_RE.search(text)
    if vm is None:
        raise RuntimeError("该文件中未检测到视频流, 请确认选择的是视频文件。")

    if duration is None:
        duration = _probe_duration_by_decode(path)

    line = text[vm.start():vm.start() + 400]
    fps = None
    fm = FPS_RE.search(line)
    if fm:
        fps = float(fm.group(1))
    if not fps or fps <= 0 or fps > 1000:
        tm = TBR_RE.search(line)
        if tm:
            fps = float(tm.group(1))
    if not fps or fps <= 0 or fps > 1000:
        fps = 25.0

    size = os.path.getsize(path)
    return {
        "duration": duration,
        "size": size,
        "bitrate": size / duration,
        "fps": fps,
        "vcodec": vm.group(1),
        "acodec": am.group(1) if am else None,
    }


def _probe_duration_by_decode(path: str) -> float:
    """容器缺少时长元数据时, 空解码一遍取最后的 time= 值。"""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-i", path,
         "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    last = None
    for m in TIME_RE.finditer(proc.stderr or ""):
        last = m
    if last is None:
        raise RuntimeError("无法获取视频时长, 文件可能已损坏或格式不受支持。")
    total = int(last.group(1)) * 3600 + int(last.group(2)) * 60 + float(last.group(3))
    if total <= 0:
        raise RuntimeError("无法获取视频时长 (解析结果为 0)。")
    return total


# --------------------------------------------------------------------------- #
# FFmpeg 执行封装
# --------------------------------------------------------------------------- #

def run_ffmpeg(cmd: list[str], total_duration: float, progress,
               cancel_event: threading.Event,
               keep_stderr: bool = False) -> tuple[int, str, list[str]]:
    """执行 ffmpeg, 通过 -progress 更新进度, 返回 (返回码, 错误摘要, stderr 全部行)。"""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_NO_WINDOW,
    )
    err_lines: list[str] = []

    def read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                err_lines.append(line)

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith(("out_time_us=", "out_time_ms=")):
            value = line.split("=", 1)[1]
            if value.isdigit() and total_duration > 0:
                div = 1_000_000 if line.startswith("out_time_us=") else 1_000
                progress(min(1.0, (int(value) / div) / total_duration))
        if cancel_event.is_set():
            proc.terminate()
            break

    proc.wait()
    t.join(timeout=1.0)
    return proc.returncode, "\n".join(err_lines[-15:]), (err_lines if keep_stderr else [])


# --------------------------------------------------------------------------- #
# 人声识别 (VAD): 音频特征提取
# --------------------------------------------------------------------------- #

def _moving_mean_std(x, win: int):
    """滑动均值与标准差 (基于累加和, O(n))。"""
    if win < 2:
        return x.copy(), np.zeros_like(x)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge").astype(np.float64)
    c1 = np.concatenate(([0.0], np.cumsum(xp)))
    c2 = np.concatenate(([0.0], np.cumsum(xp * xp)))
    s1 = c1[win:] - c1[:-win]
    s2 = c2[win:] - c2[:-win]
    mean = s1 / win
    var = np.maximum(s2 / win - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _median_filter1d(x, k: int):
    if k <= 1 or len(x) <= 2:
        return x.copy()
    k = min(k, len(x))
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.median(win, axis=1).astype(np.float32)


def compute_audio_features(path: str, duration: float, progress,
                           cancel_event: threading.Event) -> dict:
    """解码音频并逐帧提取特征。返回各特征数组 (长度 = 帧数)。"""
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-i", path,
           "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(VAD_SR),
           "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=_NO_WINDOW)
    err_chunks: list[bytes] = []

    def read_err() -> None:
        assert proc.stderr is not None
        err_chunks.append(proc.stderr.read() or b"")

    te = threading.Thread(target=read_err, daemon=True)
    te.start()

    freqs = np.fft.rfftfreq(VAD_NFFT, 1.0 / VAD_SR)

    def bidx(f: float) -> int:
        return int(np.searchsorted(freqs, f))

    sp_lo, sp_hi = bidx(300), bidx(3400)          # 语音主带
    lo_lo, lo_hi = bidx(20), bidx(180)            # 工频嗡嗡声区
    fl_lo, fl_hi = bidx(200), bidx(6000)          # 计算谱平坦度的区间
    pt_lo, pt_hi = bidx(80), bidx(1200)           # 求基频自相关的带限区间
    lag_min, lag_max = int(VAD_SR / 400), int(VAD_SR / 70)   # 基频 70~400 Hz

    window = np.hanning(VAD_FRAME).astype(np.float32)
    acc = {k: [] for k in ("e_tot", "e_sp", "e_low", "flat", "ac")}
    carry = np.zeros(0, dtype=np.float32)
    total_samples = 0
    batch_target = VAD_SR * 15                    # 每 15 秒批量做一次 FFT

    def process_buffer(buf) -> np.ndarray:
        nonlocal total_samples
        if len(buf) < VAD_FRAME:
            return buf
        n_frames = 1 + (len(buf) - VAD_FRAME) // VAD_HOP
        view = np.lib.stride_tricks.sliding_window_view(buf, VAD_FRAME)[::VAD_HOP]
        frames = view[:n_frames] * window
        spec = np.fft.rfft(frames, n=VAD_NFFT, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

        e_tot = power.sum(axis=1) + 1e-20
        e_sp = power[:, sp_lo:sp_hi].sum(axis=1) + 1e-20
        e_low = power[:, lo_lo:lo_hi].sum(axis=1) + 1e-20

        band = power[:, fl_lo:fl_hi] + 1e-20
        flat = np.exp(np.log(band).mean(axis=1)) / band.mean(axis=1)

        pb = np.zeros_like(power)
        pb[:, pt_lo:pt_hi] = power[:, pt_lo:pt_hi]
        corr = np.fft.irfft(pb, n=VAD_NFFT, axis=1)[:, :VAD_FRAME]
        r0 = np.maximum(corr[:, 0], 1e-20)
        ac = corr[:, lag_min:lag_max].max(axis=1) / r0

        acc["e_tot"].append(10.0 * np.log10(e_tot))
        acc["e_sp"].append(10.0 * np.log10(e_sp))
        acc["e_low"].append(10.0 * np.log10(e_low))
        acc["flat"].append(flat.astype(np.float32))
        acc["ac"].append(np.clip(ac, 0.0, 1.0).astype(np.float32))

        consumed = n_frames * VAD_HOP
        total_samples += consumed
        if duration > 0:
            progress(min(0.99, (total_samples / VAD_SR) / duration))
        return buf[consumed:]

    try:
        assert proc.stdout is not None
        pending = [carry]
        pending_len = len(carry)
        while True:
            if cancel_event.is_set():
                proc.terminate()
                raise RuntimeError("用户已取消")
            raw = proc.stdout.read(1 << 18)
            if not raw:
                break
            pending.append(np.frombuffer(raw, dtype=np.float32))
            pending_len += len(pending[-1])
            if pending_len >= batch_target:
                carry = process_buffer(np.concatenate(pending))
                pending, pending_len = [carry], len(carry)
        if pending_len:
            process_buffer(np.concatenate(pending))
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
        te.join(timeout=1.0)

    if proc.returncode not in (0, None) and not acc["e_sp"]:
        msg = (b"".join(err_chunks)).decode("utf-8", "replace")[-400:]
        raise RuntimeError("音频解码失败:\n" + (msg or f"返回码 {proc.returncode}"))
    if not acc["e_sp"]:
        raise RuntimeError("音频过短或无法解码, 无法进行人声识别。")

    feat = {k: np.concatenate(v) for k, v in acc.items()}
    feat["hop_sec"] = VAD_HOP / VAD_SR
    feat["n"] = len(feat["e_sp"])
    feat["duration"] = duration
    return feat


def estimate_noise_floor(e_sp, hop_sec: float):
    """分块百分位 + 中值平滑, 自适应跟踪底噪(含嗡嗡声)电平。"""
    n = len(e_sp)
    blk = max(1, int(round(FLOOR_BLOCK_SEC / hop_sec)))
    nb = max(1, math.ceil(n / blk))
    pad_len = nb * blk - n
    padded = np.pad(e_sp, (0, pad_len), mode="edge") if pad_len else e_sp
    blocks = padded.reshape(nb, blk)
    floor_b = np.percentile(blocks, FLOOR_PCT, axis=1).astype(np.float32)
    floor_b = _median_filter1d(floor_b, 5)

    if nb == 1:
        floor = np.full(n, floor_b[0], dtype=np.float32)
    else:
        centers = (np.arange(nb) + 0.5) * blk
        floor = np.interp(np.arange(n), centers, floor_b).astype(np.float32)

    # 安全钳制: 底噪不得高于整体中值 6 dB 以内, 避免"整段几乎都是人声"时漏检
    floor = np.minimum(floor, float(np.median(e_sp)) - 6.0)
    return floor


def frames_to_speech_mask(feat: dict, snr_thresh: float):
    """逐帧判定是否为人声, 返回布尔掩码。"""
    e_sp, ac, flat = feat["e_sp"], feat["ac"], feat["flat"]
    hop = feat["hop_sec"]
    floor = estimate_noise_floor(e_sp, hop)
    snr = e_sp - floor

    mod_win = max(3, int(round(MOD_WINDOW_SEC / hop)))
    _, mod_std = _moving_mean_std(e_sp, mod_win)

    structured = (ac >= AC_THRESH) | (flat <= FLAT_THRESH)
    speech = (snr >= snr_thresh) & structured
    speech |= snr >= (snr_thresh + LOUD_OVERRIDE_DB)

    # 稳态噪声否决: 电平几乎不起伏且无明显谐波 -> 嗡嗡声/风扇声
    stationary = (mod_std < STATIONARY_STD_DB) & (ac < STATIONARY_AC_MAX)
    speech &= ~stationary
    return speech


def mask_to_intervals(mask, hop_sec: float, duration: float) -> list[list[float]]:
    """布尔掩码 -> 时间区间列表。"""
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    starts, ends = idx[0::2], idx[1::2]
    out = []
    for s, e in zip(starts, ends):
        t0 = max(0.0, s * hop_sec)
        t1 = min(duration, e * hop_sec + VAD_FRAME / VAD_SR)
        out.append([t0, t1])
    return out


def refine_speech_intervals(intervals: list[list[float]], duration: float,
                            min_speedup: float) -> list[list[float]]:
    """丢弃过短人声 -> 前后加缓冲 -> 合并短于 min_speedup 的间隙。"""
    kept = [iv for iv in intervals if iv[1] - iv[0] >= SPEECH_MIN_DUR]
    if not kept:
        return []
    padded = [[max(0.0, s - SPEECH_PAD), min(duration, e + SPEECH_PAD)]
              for s, e in kept]
    merged: list[list[float]] = []
    for s, e in padded:
        if merged and s - merged[-1][1] < min_speedup:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def complement_intervals(intervals: list[list[float]],
                         duration: float) -> list[list[float]]:
    gaps, cur = [], 0.0
    for s, e in intervals:
        if s - cur > 1e-6:
            gaps.append([cur, s])
        cur = max(cur, e)
    if duration - cur > 1e-6:
        gaps.append([cur, duration])
    return gaps


def analyze_speech(feat: dict, snr_thresh: float, min_speedup: float) -> dict:
    """由缓存特征推导人声区间与待加速区间 (毫秒级, 参数变动可即时重算)。"""
    duration = feat["duration"]
    raw_mask = frames_to_speech_mask(feat, snr_thresh)
    raw = mask_to_intervals(raw_mask, feat["hop_sec"], duration)
    speech = refine_speech_intervals(raw, duration, min_speedup)
    fast = complement_intervals(speech, duration)
    all_gaps = complement_intervals(
        refine_speech_intervals(raw, duration, 0.0), duration)
    return {
        "speech": speech,
        "fast": fast,
        "all_gaps": all_gaps,
        "speech_total": sum(e - s for s, e in speech),
        "fast_total": sum(e - s for s, e in fast),
        "duration": duration,
    }


def gap_histogram(all_gaps: list[list[float]]) -> list[str]:
    """输出实际间隙长度分布, 便于用户为自己的视频挑选阈值。"""
    buckets = [(0.0, 0.3, "< 0.3 s  词内间隙/塞音, 绝不能加速"),
               (0.3, 0.6, "0.3-0.6 s  句内呼吸停顿, 建议保留"),
               (0.6, 1.0, "0.6-1.0 s  句子之间, 建议保留"),
               (1.0, 2.0, "1.0-2.0 s  段落/话题切换, 可加速"),
               (2.0, 5.0, "2.0-5.0 s  明显空档, 应加速"),
               (5.0, float("inf"), "> 5 s   大段空白, 应加速")]
    lines = []
    for lo, hi, name in buckets:
        sel = [g[1] - g[0] for g in all_gaps if lo <= g[1] - g[0] < hi]
        total = sum(sel)
        bar = "█" * min(30, int(total / max(1e-9, sum(
            g[1] - g[0] for g in all_gaps)) * 60)) if all_gaps else ""
        lines.append(f"    {name:<34} {len(sel):>5} 处  合计 {human_time(total)} {bar}")
    return lines


# --------------------------------------------------------------------------- #
# 简单音量阈值方案 (numpy 不可用时的退路)
# --------------------------------------------------------------------------- #

def detect_silence_simple(path: str, duration: float, min_dur: float,
                          progress, cancel_event: threading.Event) -> list[list[float]]:
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-i", path, "-map", "0:a:0",
           "-af", f"silencedetect=noise=-30dB:d={max(0.1, min_dur):.3f}",
           "-f", "null", "-",
           "-progress", "pipe:1", "-nostats", "-loglevel", "info"]
    code, tail, lines = run_ffmpeg(cmd, duration, progress, cancel_event,
                                   keep_stderr=True)
    if cancel_event.is_set():
        raise RuntimeError("用户已取消")
    if code != 0:
        raise RuntimeError("静音检测失败:\n" + (tail or f"返回码 {code}"))

    intervals: list[list[float]] = []
    pending: float | None = None
    for line in lines:
        ms = SILENCE_START_RE.search(line)
        if ms:
            pending = max(0.0, float(ms.group(1)))
            continue
        me = SILENCE_END_RE.search(line)
        if me and pending is not None:
            intervals.append([pending, min(duration, float(me.group(1)))])
            pending = None
    if pending is not None and duration - pending > 0.05:
        intervals.append([pending, duration])
    return intervals


# --------------------------------------------------------------------------- #
# 倍速: 分片与滤镜图
# --------------------------------------------------------------------------- #

def limit_fast_intervals(fast: list[list[float]], cap: int) -> list[list[float]]:
    """片段过多时只保留最长的若干段, 控制滤镜图规模。"""
    max_fast = max(1, cap // 2)
    if len(fast) <= max_fast:
        return fast
    longest = sorted(fast, key=lambda x: x[1] - x[0], reverse=True)[:max_fast]
    return sorted(longest)


def intervals_to_pieces(fast: list[list[float]],
                        duration: float) -> list[tuple[float, float | None, bool]]:
    """待加速区间 -> 分片列表 [(start, end 或 None, 是否倍速), ...]。"""
    pieces: list[tuple[float, float | None, bool]] = []
    cur = 0.0
    for s, e in fast:
        s, e = max(cur, s), min(duration, e)
        if e - s <= 0.05:
            continue
        if s - cur > 0.05:
            pieces.append((cur, s, False))
        else:
            s = cur
        pieces.append((s, e, True))
        cur = e
    if duration - cur > 0.05:
        pieces.append((cur, None, False))
    if not pieces:
        pieces = [(0.0, None, False)]
    ls, le, lf = pieces[-1]
    if le is not None and le >= duration - 0.05:
        pieces[-1] = (ls, None, lf)
    return pieces


def piece_length(piece, duration: float) -> float:
    s, e, _ = piece
    return max(0.0, (duration if e is None else e) - s)


def expected_output_duration(pieces, duration: float, speed: float) -> float:
    total = 0.0
    for p in pieces:
        length = piece_length(p, duration)
        total += length / speed if p[2] else length
    return max(total, 0.1)


def atempo_chain(speed: float) -> str:
    """atempo 单实例范围有限, 拆成多级串联以保证兼容性。"""
    parts: list[str] = []
    s = speed
    while s > 2.0 + 1e-9:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        parts.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-9:
        parts.append(f"atempo={s:.6f}")
    return ",".join(parts) if parts else "anull"


def build_speed_graph(pieces, speed: float, out_fps: float,
                      has_audio: bool, denoise: bool) -> str:
    """构建 filter_complex 滤镜图 (单行文本)。"""
    n = len(pieces)
    tail_a = ("," + DENOISE_FILTER) if denoise else ""
    parts: list[str] = []

    if n == 1:
        _, _, fast = pieces[0]
        vchain = ([f"setpts=PTS/{speed:.6f}"] if fast else []) + [f"fps={out_fps:.5f}"]
        parts.append("[0:v]" + ",".join(vchain) + "[vout]")
        if has_audio:
            achain = atempo_chain(speed) if fast else "anull"
            parts.append(f"[0:a]{achain}{tail_a},asetpts=N/SR/TB[aout]")
        return ";".join(parts)

    parts.append("[0:v]split=%d%s" % (n, "".join(f"[sv{i}]" for i in range(n))))
    if has_audio:
        parts.append("[0:a]asplit=%d%s" % (n, "".join(f"[sa{i}]" for i in range(n))))

    for i, (s, e, fast) in enumerate(pieces):
        rng = f"start={s:.3f}" + (f":end={e:.3f}" if e is not None else "")
        vchain = [f"trim={rng}", "setpts=PTS-STARTPTS"]
        if fast:
            vchain.append(f"setpts=PTS/{speed:.6f}")
        parts.append(f"[sv{i}]" + ",".join(vchain) + f"[v{i}]")
        if has_audio:
            achain = [f"atrim={rng}", "asetpts=PTS-STARTPTS"]
            if fast:
                achain.append(atempo_chain(speed))
            parts.append(f"[sa{i}]" + ",".join(achain) + f"[a{i}]")

    if has_audio:
        joined = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{joined}concat=n={n}:v=1:a=1[vc][ac]")
        parts.append(f"[vc]fps={out_fps:.5f}[vout]")
        parts.append(f"[ac]aresample=async=1:first_pts=0{tail_a}[aout]")
    else:
        joined = "".join(f"[v{i}]" for i in range(n))
        parts.append(f"{joined}concat=n={n}:v=1:a=0[vc]")
        parts.append(f"[vc]fps={out_fps:.5f}[vout]")
    return ";".join(parts)


def run_speed_pass(src: str, dst: str, graph: str, out_fps: float,
                   exp_duration: float, maxrate_bits: int, has_audio: bool,
                   progress, cancel_event: threading.Event) -> tuple[int, str]:
    """执行倍速转码, 滤镜图写入脚本文件传入 (规避命令行长度限制)。"""
    script = dst + ".filter.txt"
    with open(script, "w", encoding="utf-8", newline="") as fh:
        fh.write(graph)
    try:
        cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", src,
               "-filter_complex_script", script, "-map", "[vout]"]
        if has_audio:
            cmd += ["-map", "[aout]"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-maxrate", str(maxrate_bits), "-bufsize", str(maxrate_bits * 2),
                "-g", str(max(12, int(out_fps * 2))),
                "-keyint_min", str(max(6, int(out_fps)))]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += ["-movflags", "+faststart", "-f", "mp4",
                "-progress", "pipe:1", "-nostats", "-loglevel", "error", dst]
        code, tail, _ = run_ffmpeg(cmd, exp_duration, progress, cancel_event)
        return code, tail
    finally:
        try:
            os.remove(script)
        except OSError:
            pass


def run_denoise_only(src: str, dst: str, duration: float, progress,
                     cancel_event: threading.Event) -> tuple[int, str]:
    """仅降噪不变速: 视频流直接复制, 只重编码音频。"""
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", src,
           "-map", "0:v:0", "-map", "0:a:0", "-sn", "-dn",
           "-c:v", "copy", "-af", DENOISE_FILTER,
           "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-f", "mp4",
           "-progress", "pipe:1", "-nostats", "-loglevel", "error", dst]
    code, tail, _ = run_ffmpeg(cmd, duration, progress, cancel_event)
    return code, tail


# --------------------------------------------------------------------------- #
# 分割
# --------------------------------------------------------------------------- #

def plan_segment_count(duration: float, size: float,
                       dur_limit: float, size_limit: float) -> int:
    n_by_dur = math.ceil(duration / dur_limit - 1e-9)
    n_by_size = math.ceil(size / (size_limit * SIZE_MARGIN) - 1e-9)
    return max(1, n_by_dur, n_by_size)


def build_split_command(src: str, out_pattern: str, cut_times: list[float],
                        mode: str, seg_dur: float, size_limit: float,
                        has_audio: bool) -> list[str]:
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", src, "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a?"]
    cmd += ["-sn", "-dn"]

    if mode == "copy":
        cmd += ["-c", "copy"]
    elif mode == "copy_audio_aac":
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    else:
        budget_bps = int(size_limit * 0.92 * 8 / max(seg_dur, 1.0))
        audio_bps = 192_000 if has_audio else 0
        vmax = max(300_000, budget_bps - audio_bps)
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                "-pix_fmt", "yuv420p",
                "-maxrate", str(vmax), "-bufsize", str(vmax * 2)]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        if cut_times:
            cmd += ["-force_key_frames", ",".join(f"{t:.3f}" for t in cut_times)]

    if cut_times:
        cmd += ["-f", "segment",
                "-segment_times", ",".join(f"{t:.3f}" for t in cut_times),
                "-segment_start_number", "1",
                "-reset_timestamps", "1",
                "-segment_format", "mp4",
                "-movflags", "+empty_moov"]
    else:
        cmd += ["-f", "mp4", "-movflags", "+faststart"]

    cmd += ["-progress", "pipe:1", "-nostats", "-loglevel", "error", out_pattern]
    return cmd


def cleanup_outputs(outdir: str, base: str, protect: str) -> None:
    """删除同名旧分段; 绝不删除源文件或正在使用的临时文件。"""
    pattern = re.compile(re.escape(base) + r"-\d+\.mp4$", re.IGNORECASE)
    protect_abs = os.path.abspath(protect)
    for name in os.listdir(outdir):
        if not pattern.match(name):
            continue
        full = os.path.join(outdir, name)
        if os.path.abspath(full) == protect_abs:
            continue
        try:
            os.remove(full)
        except OSError:
            pass


def collect_outputs(outdir: str, base: str) -> list[str]:
    pattern = re.compile(re.escape(base) + r"-(\d+)\.mp4$", re.IGNORECASE)
    found = []
    for name in os.listdir(outdir):
        m = pattern.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(outdir, name)))
    return [p for _, p in sorted(found)]


def split_media(work_src: str, base: str, outdir: str, info: dict,
                dur_limit: float, size_limit: float, force_reencode: bool,
                movable: bool, log, progress, status,
                cancel_event: threading.Event) -> list[str]:
    duration: float = info["duration"]
    size: int = info["size"]
    has_audio = info["acodec"] is not None

    out_pattern = os.path.join(outdir, f"{base}-%d.mp4")
    out_single = os.path.join(outdir, f"{base}-1.mp4")
    n = plan_segment_count(duration, size, dur_limit, size_limit)

    if n == 1 and movable and size <= size_limit and duration <= dur_limit:
        cleanup_outputs(outdir, base, protect=work_src)
        os.replace(work_src, out_single)
        log(f"  仅需 1 段, 直接输出: {os.path.basename(out_single)}")
        progress(1.0)
        return [out_single]

    modes = ["reencode"] if force_reencode else ["copy", "copy_audio_aac", "reencode"]
    mode_idx = 0

    for attempt in range(1, MAX_RETRY + 1):
        if cancel_event.is_set():
            raise RuntimeError("用户已取消")

        seg = duration / n
        cut_times = [seg * i for i in range(1, n)]
        mode = modes[mode_idx]
        mode_name = {"copy": "流复制(无损/最快)",
                     "copy_audio_aac": "视频复制 + 音频转 AAC",
                     "reencode": "重新编码 (H.264/AAC)"}[mode]

        status(f"分割中 ({n} 段)")
        log(f"[分割 第 {attempt} 轮] 共 {n} 段, 每段约 {human_time(seg)}"
            f"  预计 {human_size(size / n)}/段  模式: {mode_name}")

        cleanup_outputs(outdir, base, protect=work_src)
        progress(0.0)
        cmd = build_split_command(work_src, out_pattern if cut_times else out_single,
                                  cut_times, mode, seg, size_limit, has_audio)
        code, err, _ = run_ffmpeg(cmd, duration, progress, cancel_event)

        if cancel_event.is_set():
            cleanup_outputs(outdir, base, protect=work_src)
            raise RuntimeError("用户已取消")

        files = collect_outputs(outdir, base)
        if code != 0 or not files:
            if err:
                log("  FFmpeg 提示: " + err.replace("\n", " | ")[:500])
            if mode_idx < len(modes) - 1:
                mode_idx += 1
                log(f"  → 当前模式不适用于 MP4 封装, 切换为: {modes[mode_idx]}")
                continue
            raise RuntimeError("FFmpeg 分割失败:\n" + (err or f"返回码 {code}"))

        log(f"  已生成 {len(files)} 段, 正在校验 ...")
        worst_ratio = 1.0
        details = []
        for idx, path in enumerate(files, start=1):
            fsize = os.path.getsize(path)
            try:
                fdur = probe_media(path)["duration"]
            except Exception:
                fdur = seg
            details.append((idx, fdur, fsize))
            worst_ratio = max(worst_ratio, fsize / size_limit,
                              fdur / (dur_limit * DUR_TOLERANCE))

        if worst_ratio <= 1.0:
            log("-" * 62)
            log("校验通过, 分割完成:")
            for idx, fdur, fsize in details:
                log(f"  {base}-{idx}.mp4   {human_time(fdur)}   {human_size(fsize)}")
            progress(1.0)
            return files

        new_n = max(n + 1, math.ceil(n * worst_ratio * 1.03))
        log(f"  有分段超出限制 (超出比例 {worst_ratio:.3f}), "
            f"段数由 {n} 调整为 {new_n} 后重试 ...")
        n = new_n

    raise RuntimeError("多次尝试后仍无法满足限制, 请适当放宽时长或体积上限。")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def process_video(src: str, max_minutes: float, max_gb: float,
                  force_reencode: bool, speed: float, speed_mode: str,
                  vad: dict, feat_cache: dict | None,
                  log, progress, status, cancel_event: threading.Event) -> list[str]:
    info = probe_media(src)
    duration: float = info["duration"]
    size: int = info["size"]
    src_fps: float = info["fps"]
    has_audio = info["acodec"] is not None

    dur_limit = max_minutes * 60.0
    size_limit = max_gb * GB
    denoise = bool(vad.get("denoise"))
    do_speed = abs(speed - 1.0) > 1e-6

    outdir = os.path.dirname(os.path.abspath(src))
    if not os.access(outdir, os.W_OK):
        log(f"[警告] 源目录不可写, 输出改为当前目录: {os.getcwd()}")
        outdir = os.getcwd()
    base = sanitize_base(os.path.splitext(os.path.basename(src))[0])

    log(f"源文件   : {os.path.basename(src)}")
    log(f"时长/大小: {human_time(duration)}  /  {human_size(size)}"
        f"   帧率 {src_fps:.3f} fps")
    log(f"平均码率 : {info['bitrate'] * 8 / 1_000_000:.2f} Mbps"
        f"   (视频 {info['vcodec']}, 音频 {info['acodec'] or '无'})")
    log(f"约束条件 : 每段 ≤ {max_minutes:g} 分钟, ≤ {max_gb:g} GB")
    if do_speed:
        log(f"倍速设置 : ×{speed:.1f} "
            f"{'(整段统一)' if speed_mode == 'all' else '(仅无人声部分)'}")
    else:
        log("倍速设置 : 不变速")
    if denoise:
        log("音频降噪 : 已开启 (高通 85 Hz + 自适应频域降噪)")
    log(f"输出目录 : {outdir}")
    log("-" * 62)

    temp_path = os.path.join(outdir, f"~{base}.speed.tmp.mp4")
    work_src, movable = src, False

    try:
        # ---------------- 阶段 1: 倍速 / 降噪 ----------------
        need_stage1 = do_speed or denoise
        if need_stage1:
            free = shutil.disk_usage(outdir).free
            if free < size * 1.1:
                log(f"[警告] 剩余空间 {human_size(free)} 可能不足以存放中间文件。")

        if not do_speed and denoise and has_audio:
            status("音频降噪")
            log("[阶段1/2] 仅降噪 (视频流无损复制) ...")
            progress(0.0)
            code, err = run_denoise_only(src, temp_path, duration, progress,
                                         cancel_event)
            if cancel_event.is_set():
                raise RuntimeError("用户已取消")
            if code == 0 and os.path.isfile(temp_path) and os.path.getsize(temp_path):
                work_src, movable = temp_path, True
                info = probe_media(temp_path)
                log(f"  降噪完成: {human_size(info['size'])}")
                log("-" * 62)
            else:
                log("  降噪失败, 改为原样处理。" +
                    ("  FFmpeg: " + err.replace("\n", " | ")[:300] if err else ""))

        elif do_speed:
            fast_intervals: list[list[float]] | None = None
            if speed_mode == "silence" and has_audio:
                if vad["engine"] == "smart" and HAVE_NUMPY:
                    feat = feat_cache
                    if not (feat and feat.get("path") == os.path.abspath(src)):
                        status("识别人声")
                        log("[阶段1/2] 正在分析音频, 区分人声与噪声 ...")
                        progress(0.0)
                        feat = compute_audio_features(src, duration, progress,
                                                      cancel_event)
                        feat["path"] = os.path.abspath(src)
                    else:
                        log("[阶段1/2] 复用已有的音频分析结果。")
                    res = analyze_speech(feat, vad["snr"], vad["min_speedup"])
                    fast_intervals = res["fast"]
                    log(f"  人声 {len(res['speech'])} 段 / 合计 "
                        f"{human_time(res['speech_total'])}"
                        f"（占 {res['speech_total'] / duration * 100:.1f}%）")
                    log(f"  待加速(静音+噪声) {len(fast_intervals)} 段 / 合计 "
                        f"{human_time(res['fast_total'])}"
                        f"（占 {res['fast_total'] / duration * 100:.1f}%）")
                    log(f"  已保留所有短于 {vad['min_speedup']:.1f} 秒的停顿, "
                        f"人声前后各留 {SPEECH_PAD:.2f} 秒缓冲")
                else:
                    status("检测静音")
                    log(f"[阶段1/2] 音量阈值检测静音 "
                        f"(-30 dB, ≥{vad['min_speedup']:.1f} 秒) ...")
                    progress(0.0)
                    fast_intervals = detect_silence_simple(
                        src, duration, vad["min_speedup"], progress, cancel_event)
                    total = sum(e - s for s, e in fast_intervals)
                    log(f"  检测到 {len(fast_intervals)} 段静音, 合计 "
                        f"{human_time(total)}（占 {total / duration * 100:.1f}%）")
                out_fps = min(src_fps, MAX_OUTPUT_FPS)
                caps = PIECE_CAPS if fast_intervals else ()
                if not fast_intervals:
                    log("  未找到可加速的片段, 跳过倍速处理。")
            else:
                if speed_mode == "silence" and not has_audio:
                    log("[提示] 该视频没有音轨, 视为全程无声, 按整段倍速处理。")
                caps = (1,)
                out_fps = min(src_fps * speed, MAX_OUTPUT_FPS)

            if caps:
                video_bits = max(200_000, info["bitrate"] * 8 - 192_000)
                maxrate = int(max(300_000, video_bits * (out_fps / src_fps) * 1.3))
                ok = False
                for cap in caps:
                    if cancel_event.is_set():
                        raise RuntimeError("用户已取消")
                    if cap == 1:
                        pieces = [(0.0, None, True)]
                    else:
                        pieces = intervals_to_pieces(
                            limit_fast_intervals(fast_intervals or [], cap), duration)
                    if not any(p[2] for p in pieces):
                        log("  没有需要倍速的片段, 跳过倍速处理。")
                        break

                    exp_dur = expected_output_duration(pieces, duration, speed)
                    status("倍速处理")
                    log(f"[阶段1/2] 倍速转码: {len(pieces)} 个片段, "
                        f"输出帧率 {out_fps:.2f} fps, 预计时长 {human_time(exp_dur)}")
                    progress(0.0)
                    graph = build_speed_graph(pieces, speed, out_fps,
                                              has_audio, denoise and has_audio)
                    code, err = run_speed_pass(src, temp_path, graph, out_fps,
                                               exp_dur, maxrate, has_audio,
                                               progress, cancel_event)
                    if cancel_event.is_set():
                        raise RuntimeError("用户已取消")
                    if code == 0 and os.path.isfile(temp_path) \
                            and os.path.getsize(temp_path) > 0:
                        ok = True
                        break
                    if err:
                        log("  FFmpeg 提示: " + err.replace("\n", " | ")[:500])
                    if denoise:
                        log("  → 可能是降噪滤镜不受支持, 关闭降噪后重试 ...")
                        denoise = False
                        graph = build_speed_graph(pieces, speed, out_fps,
                                                  has_audio, False)
                        code, err = run_speed_pass(src, temp_path, graph, out_fps,
                                                   exp_dur, maxrate, has_audio,
                                                   progress, cancel_event)
                        if code == 0 and os.path.isfile(temp_path) \
                                and os.path.getsize(temp_path) > 0:
                            ok = True
                            break
                    if cap != caps[-1]:
                        log("  → 片段过多导致失败, 减少变速片段数量后重试 ...")
                if not ok and caps and any(True for _ in caps):
                    if not os.path.isfile(temp_path):
                        raise RuntimeError(
                            "倍速处理失败, 请尝试降低倍速或改用普通倍速模式。")
                if ok:
                    work_src, movable = temp_path, True
                    info = probe_media(temp_path)
                    log(f"  倍速完成: {human_time(info['duration'])} / "
                        f"{human_size(info['size'])}")
                    log("-" * 62)

        # ---------------- 阶段 2: 分割 ----------------
        return split_media(work_src, base, outdir, info, dur_limit, size_limit,
                           force_reencode, movable, log, progress, status,
                           cancel_event)
    finally:
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                log(f"[提示] 临时文件未能删除, 可手动清理: {temp_path}")


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #

VIDEO_TYPES = [
    ("所有视频文件",
     "*.mp4 *.m4v *.mov *.mkv *.avi *.wmv *.asf *.flv *.f4v *.webm *.ogv "
     "*.mpg *.mpeg *.mpe *.m1v *.m2v *.ts *.m2ts *.mts *.tp *.trp *.vob "
     "*.3gp *.3g2 *.rm *.rmvb *.divx *.dv *.mxf *.amv *.svi *.nsv *.dat"),
    ("MP4 / MOV", "*.mp4 *.m4v *.mov"),
    ("Matroska / WebM", "*.mkv *.webm"),
    ("AVI / WMV / ASF", "*.avi *.wmv *.asf"),
    ("MPEG / TS / VOB", "*.mpg *.mpeg *.ts *.m2ts *.mts *.vob *.dat"),
    ("FLV / RM / 3GP", "*.flv *.f4v *.rm *.rmvb *.3gp *.3g2"),
    ("所有文件", "*.*"),
]


class VideoSplitterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("视频分割工具 — 智能人声倍速 + 按时长/大小平均切分")
        root.geometry("920x900")
        root.minsize(840, 780)

        self.msg_q: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.media_info: dict | None = None
        self.feat_cache: dict | None = None

        self.var_src = tk.StringVar()
        self.var_minutes = tk.StringVar(value="15")
        self.var_gb = tk.StringVar(value="2")
        self.var_mode = tk.StringVar(value="auto")
        self.var_speed_mode = tk.StringVar(value="all")
        self.var_speed = tk.DoubleVar(value=1.0)
        self.var_speed_text = tk.StringVar(value="× 1.0")
        self.var_engine = tk.StringVar(value="smart" if HAVE_NUMPY else "simple")
        self.var_sens = tk.StringVar(value="中（推荐）")
        self.var_min_speedup = tk.StringVar(value=f"{MIN_SPEEDUP_DUR:g}")
        self.var_denoise = tk.BooleanVar(value=False)
        self.var_info = tk.StringVar(value="尚未选择文件")
        self.var_outdir = tk.StringVar(value="输出目录：（选择文件后与源视频同一目录）")
        self.var_plan = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="就绪")

        self._build_ui()
        for var in (self.var_minutes, self.var_gb, self.var_speed_mode,
                    self.var_engine, self.var_sens, self.var_min_speedup):
            var.trace_add("write", lambda *_: self._update_plan())
        self.root.after(100, self._poll_queue)

    # ------------------------------ 界面 ------------------------------ #
    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 5}
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # ① 文件
        box_file = ttk.LabelFrame(main, text="① 选择视频文件", padding=10)
        box_file.grid(row=0, column=0, sticky="ew", **pad)
        box_file.columnconfigure(0, weight=1)
        ttk.Entry(box_file, textvariable=self.var_src, state="readonly")\
            .grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(box_file, text="浏览...", command=self.choose_file)\
            .grid(row=0, column=1)
        ttk.Label(box_file, textvariable=self.var_info, foreground="#0a5")\
            .grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(box_file, textvariable=self.var_outdir, foreground="#666")\
            .grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # ② 分割约束
        box_arg = ttk.LabelFrame(main, text="② 填写分割约束", padding=10)
        box_arg.grid(row=1, column=0, sticky="ew", **pad)
        ttk.Label(box_arg, text="每段最大时长（分钟）：")\
            .grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(box_arg, textvariable=self.var_minutes, width=12)\
            .grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(box_arg, text="每段最大大小（GB）：")\
            .grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(box_arg, textvariable=self.var_gb, width=12)\
            .grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(box_arg, text="（1 GB = 1024 MB）", foreground="#888")\
            .grid(row=1, column=2, sticky="w", padx=(6, 0))
        ttk.Label(box_arg, text="分割方式：")\
            .grid(row=2, column=0, sticky="w", pady=(10, 2))
        frm_mode = ttk.Frame(box_arg)
        frm_mode.grid(row=2, column=1, columnspan=2, sticky="w", pady=(10, 2))
        ttk.Radiobutton(frm_mode, text="自动（优先无损流复制，最快）",
                        variable=self.var_mode, value="auto").pack(side="left")
        ttk.Radiobutton(frm_mode, text="强制重新编码（切点最精准，较慢）",
                        variable=self.var_mode, value="encode")\
            .pack(side="left", padx=(12, 0))

        # ③ 倍速
        box_spd = ttk.LabelFrame(main, text="③ 倍速设置", padding=10)
        box_spd.grid(row=2, column=0, sticky="ew", **pad)
        box_spd.columnconfigure(1, weight=1)

        ttk.Label(box_spd, text="倍速模式：").grid(row=0, column=0, sticky="w")
        frm_sm = ttk.Frame(box_spd)
        frm_sm.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frm_sm, text="普通倍速（整段视频统一倍速）",
                        variable=self.var_speed_mode, value="all").pack(side="left")
        ttk.Radiobutton(frm_sm, text="只对没声音的部分倍速（噪音同样视为没声音）",
                        variable=self.var_speed_mode, value="silence")\
            .pack(side="left", padx=(14, 0))

        ttk.Label(box_spd, text="倍速数值：")\
            .grid(row=1, column=0, sticky="w", pady=(10, 0))
        frm_sc = ttk.Frame(box_spd)
        frm_sc.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        frm_sc.columnconfigure(0, weight=1)
        self.scale_speed = ttk.Scale(frm_sc, from_=SPEED_MIN, to=SPEED_MAX,
                                     orient="horizontal", variable=self.var_speed,
                                     command=self._on_speed_move)
        self.scale_speed.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        ttk.Label(frm_sc, textvariable=self.var_speed_text, width=7,
                  font=("Segoe UI", 11, "bold"), foreground="#c40", anchor="w")\
            .grid(row=0, column=1)
        ttk.Button(frm_sc, text="−", width=3,
                   command=lambda: self._step_speed(-SPEED_STEP))\
            .grid(row=0, column=2, padx=2)
        ttk.Button(frm_sc, text="＋", width=3,
                   command=lambda: self._step_speed(SPEED_STEP))\
            .grid(row=0, column=3, padx=2)
        ttk.Button(frm_sc, text="重置", width=6,
                   command=lambda: self._set_speed(1.0))\
            .grid(row=0, column=4, padx=(6, 0))
        ttk.Label(box_spd, text=f"范围 {SPEED_MIN:g}× ~ {SPEED_MAX:g}×，步进 0.1；"
                                f"1.0 表示不变速（此时全程无损处理，速度最快）",
                  foreground="#888").grid(row=2, column=0, columnspan=2,
                                          sticky="w", pady=(6, 0))

        ttk.Separator(box_spd, orient="horizontal")\
            .grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)

        ttk.Label(box_spd, text="无人声判定：").grid(row=4, column=0, sticky="w")
        frm_eng = ttk.Frame(box_spd)
        frm_eng.grid(row=4, column=1, sticky="w")
        rb_smart = ttk.Radiobutton(
            frm_eng, text="智能人声识别（推荐，可把嗡嗡声/风扇声当作无声）",
            variable=self.var_engine, value="smart")
        rb_smart.pack(side="left")
        ttk.Radiobutton(frm_eng, text="纯音量阈值（快，但怕噪音）",
                        variable=self.var_engine, value="simple")\
            .pack(side="left", padx=(14, 0))
        if not HAVE_NUMPY:
            rb_smart.configure(state="disabled")

        frm_p = ttk.Frame(box_spd)
        frm_p.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(frm_p, text="人声灵敏度：").pack(side="left")
        ttk.Combobox(frm_p, textvariable=self.var_sens, width=20, state="readonly",
                     values=list(SENSITIVITY_DB.keys())).pack(side="left")
        ttk.Label(frm_p, text="     最短加速时长（秒）：").pack(side="left")
        ttk.Entry(frm_p, textvariable=self.var_min_speedup, width=8).pack(side="left")
        ttk.Checkbutton(frm_p, text="顺便消除嗡嗡声（音频降噪）",
                        variable=self.var_denoise).pack(side="left", padx=(20, 0))

        frm_an = ttk.Frame(box_spd)
        frm_an.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.btn_analyze = ttk.Button(frm_an, text="分析人声分布（不处理视频）",
                                     command=self.analyze)
        self.btn_analyze.pack(side="left")
        ttk.Label(frm_an, foreground="#888",
                  text="  只有连续无人声超过“最短加速时长”的片段才会被加速；"
                       "句内停顿常为 0.2~0.5 秒、句间 0.5~1 秒，故建议 ≥1.5 秒")\
            .pack(side="left")

        ttk.Label(main, textvariable=self.var_plan, foreground="#06c",
                  wraplength=860, justify="left")\
            .grid(row=3, column=0, sticky="w", padx=10)

        # ④ 操作
        box_run = ttk.Frame(main)
        box_run.grid(row=4, column=0, sticky="ew", **pad)
        box_run.columnconfigure(2, weight=1)
        self.btn_start = ttk.Button(box_run, text="开始处理", command=self.start)
        self.btn_start.grid(row=0, column=0)
        self.btn_cancel = ttk.Button(box_run, text="取消", command=self.cancel,
                                     state="disabled")
        self.btn_cancel.grid(row=0, column=1, padx=8)
        self.progress = ttk.Progressbar(box_run, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=2, sticky="ew", padx=(8, 8))
        ttk.Label(box_run, textvariable=self.var_status, width=14)\
            .grid(row=0, column=3)

        # 日志
        box_log = ttk.LabelFrame(main, text="运行日志", padding=6)
        box_log.grid(row=5, column=0, sticky="nsew", **pad)
        main.rowconfigure(5, weight=1)
        box_log.columnconfigure(0, weight=1)
        box_log.rowconfigure(0, weight=1)
        self.txt = tk.Text(box_log, height=14, wrap="word", state="disabled",
                           font=("Consolas", 10))
        self.txt.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(box_log, orient="vertical", command=self.txt.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb.set)

        if FFMPEG is None:
            self._append("[错误] 未找到 FFmpeg。请执行: pip install imageio-ffmpeg")
            self.btn_start.configure(state="disabled")
            self.btn_analyze.configure(state="disabled")
        else:
            self._append(f"[就绪] FFmpeg: {FFMPEG}")
            if not HAVE_NUMPY:
                self._append("[提示] 未检测到 numpy, 智能人声识别不可用, "
                             "已退回音量阈值方案。")

    # ------------------------------ 参数读取 ------------------------------ #
    def _vad_params(self) -> dict:
        try:
            min_speedup = float(self.var_min_speedup.get())
        except ValueError:
            min_speedup = MIN_SPEEDUP_DUR
        min_speedup = clamp(min_speedup, 0.2, 60.0)
        return {"engine": self.var_engine.get() if HAVE_NUMPY else "simple",
                "snr": SENSITIVITY_DB.get(self.var_sens.get(), 6.0),
                "min_speedup": min_speedup,
                "denoise": bool(self.var_denoise.get())}

    # ------------------------------ 倍速滑块 ------------------------------ #
    def _on_speed_move(self, raw: str) -> None:
        value = clamp(round(round(float(raw) / SPEED_STEP) * SPEED_STEP, 1),
                      SPEED_MIN, SPEED_MAX)
        if abs(self.var_speed.get() - value) > 1e-9:
            self.var_speed.set(value)
        self.var_speed_text.set(f"× {value:.1f}")
        self._update_plan()

    def _set_speed(self, value: float) -> None:
        value = clamp(round(value, 1), SPEED_MIN, SPEED_MAX)
        self.var_speed.set(value)
        self.var_speed_text.set(f"× {value:.1f}")
        self._update_plan()

    def _step_speed(self, delta: float) -> None:
        self._step = None
        self._set_speed(self.var_speed.get() + delta)

    # ------------------------------ 交互 ------------------------------ #
    def choose_file(self) -> None:
        path = filedialog.askopenfilename(title="请选择要分割的视频文件",
                                          filetypes=VIDEO_TYPES)
        if not path:
            return
        self.var_src.set(path)
        self.var_info.set("正在读取视频信息 ...")
        self.var_outdir.set(f"输出目录：{os.path.dirname(os.path.abspath(path))}")
        self.media_info = None
        self.feat_cache = None
        threading.Thread(target=self._probe_async, args=(path,), daemon=True).start()

    def _probe_async(self, path: str) -> None:
        try:
            self.msg_q.put(("info", probe_media(path)))
        except Exception as exc:
            self.msg_q.put(("info_err", str(exc)))

    def analyze(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        src = self.var_src.get()
        if not src or not os.path.isfile(src) or not self.media_info:
            messagebox.showwarning("提示", "请先选择一个视频文件。")
            return
        if not HAVE_NUMPY:
            messagebox.showwarning("提示", "未安装 numpy, 无法进行人声分析。")
            return
        if self.media_info["acodec"] is None:
            messagebox.showinfo("提示", "该视频没有音轨, 无需分析人声。")
            return

        self.cancel_event.clear()
        self._set_busy(True)
        self.var_status.set("分析中")
        self._append("=" * 62)
        self.worker = threading.Thread(target=self._work_analyze,
                                       args=(src, self.media_info["duration"]),
                                       daemon=True)
        self.worker.start()

    def _work_analyze(self, src: str, duration: float) -> None:
        log = lambda m: self.msg_q.put(("log", m))
        try:
            feat = self.feat_cache
            if not (feat and feat.get("path") == os.path.abspath(src)):
                log("正在解码音频并提取声学特征 ...")
                feat = compute_audio_features(
                    src, duration, lambda f: self.msg_q.put(("prog", f)),
                    self.cancel_event)
                feat["path"] = os.path.abspath(src)
            p = self._vad_params()
            res = analyze_speech(feat, p["snr"], p["min_speedup"])
            log("-" * 62)
            log(f"人声总时长   : {human_time(res['speech_total'])}"
                f"（{res['speech_total'] / duration * 100:.1f}%）"
                f"  共 {len(res['speech'])} 段")
            log(f"无人声总时长 : {human_time(duration - res['speech_total'])}"
                f"（含嗡嗡声等噪声，已判为无人声）")
            log("本视频的间隙长度分布（据此选择“最短加速时长”）:")
            for line in gap_histogram(res["all_gaps"]):
                log(line)
            log(f"当前阈值 {p['min_speedup']:.1f} 秒 → 实际会被加速 "
                f"{len(res['fast'])} 段 / {human_time(res['fast_total'])}")
            self.msg_q.put(("analyzed", feat))
        except Exception as exc:
            self.msg_q.put(("fail", str(exc)))

    def _update_plan(self) -> None:
        if not self.media_info:
            self.var_plan.set("")
            return
        try:
            minutes = float(self.var_minutes.get())
            gb = float(self.var_gb.get())
            if minutes <= 0 or gb <= 0:
                raise ValueError
        except ValueError:
            self.var_plan.set("⚠ 时长和大小必须是大于 0 的数值")
            return

        info = self.media_info
        speed = round(self.var_speed.get(), 1)
        duration, size, fps = info["duration"], info["size"], info["fps"]
        note = ""

        if abs(speed - 1.0) < 1e-9:
            eff_dur, eff_size = duration, size
        elif self.var_speed_mode.get() == "all":
            eff_dur = duration / speed
            out_fps = min(fps * speed, MAX_OUTPUT_FPS)
            eff_size = size * (out_fps / (fps * speed))
            note = (f"倍速后约 {human_time(eff_dur)} / {human_size(eff_size)}"
                    f"（输出 {out_fps:.1f} fps，估算）；")
        else:
            cache = self.feat_cache
            src_abs = os.path.abspath(self.var_src.get()) if self.var_src.get() else ""
            if cache and cache.get("path") == src_abs:
                p = self._vad_params()
                res = analyze_speech(cache, p["snr"], p["min_speedup"])
                pieces = intervals_to_pieces(res["fast"], duration)
                eff_dur = expected_output_duration(pieces, duration, speed)
                eff_size = size * (eff_dur / duration)
                note = (f"依据分析结果：加速 {human_time(res['fast_total'])}，"
                        f"输出约 {human_time(eff_dur)} / {human_size(eff_size)}；")
            else:
                eff_dur, eff_size = duration, size
                note = "点“分析人声分布”可得到精确预估；以下按不变速估算；"

        n = plan_segment_count(eff_dur, eff_size, minutes * 60, gb * GB)
        self.var_plan.set(f"{note}预计分成 {n} 段：每段约 "
                          f"{human_time(eff_dur / n)}、{human_size(eff_size / n)}"
                          f"（最终以处理后实测校验为准）")

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        src = self.var_src.get()
        if not src or not os.path.isfile(src):
            messagebox.showwarning("提示", "请先选择一个视频文件。")
            return
        try:
            minutes = float(self.var_minutes.get())
            gb = float(self.var_gb.get())
            if minutes <= 0 or gb <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "时长和大小必须是大于 0 的数字。")
            return

        self.cancel_event.clear()
        self._set_busy(True)
        self.progress.configure(value=0)
        self.var_status.set("处理中")
        self._append("=" * 62)

        args = (src, minutes, gb, self.var_mode.get() == "encode",
                round(self.var_speed.get(), 1), self.var_speed_mode.get(),
                self._vad_params(), self.feat_cache)
        self.worker = threading.Thread(target=self._work, args=args, daemon=True)
        self.worker.start()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.var_status.set("取消中")
        self._append("[取消] 正在终止 ...")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.btn_start.configure(state=state)
        self.btn_analyze.configure(state="disabled" if busy or not HAVE_NUMPY
                                   else "normal")
        self.btn_cancel.configure(state="normal" if busy else "disabled")

    # ------------------------------ 后台 ------------------------------ #
    def _work(self, src: str, minutes: float, gb: float, force: bool,
              speed: float, speed_mode: str, vad: dict,
              feat_cache: dict | None) -> None:
        try:
            files = process_video(
                src, minutes, gb, force, speed, speed_mode, vad, feat_cache,
                lambda m: self.msg_q.put(("log", m)),
                lambda f: self.msg_q.put(("prog", f)),
                lambda s: self.msg_q.put(("status", s)),
                self.cancel_event)
            self.msg_q.put(("done", files))
        except Exception as exc:
            self.msg_q.put(("fail", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind == "log":
                    self._append(payload)
                elif kind == "prog":
                    self.progress.configure(value=float(payload) * 100)
                elif kind == "status":
                    self.var_status.set(payload)
                elif kind == "info":
                    self.media_info = payload
                    self.var_info.set(
                        f"时长 {human_time(payload['duration'])}   "
                        f"大小 {human_size(payload['size'])}   "
                        f"{payload['fps']:.2f} fps   "
                        f"平均码率 {payload['bitrate'] * 8 / 1_000_000:.2f} Mbps   "
                        f"视频 {payload['vcodec']} / 音频 {payload['acodec'] or '无'}")
                    self._update_plan()
                elif kind == "info_err":
                    self.media_info = None
                    self.var_info.set(f"读取失败：{payload}")
                    self.var_plan.set("")
                elif kind == "analyzed":
                    self.feat_cache = payload
                    self.progress.configure(value=100)
                    self.var_status.set("分析完成")
                    self._set_busy(False)
                    self._update_plan()
                elif kind == "done":
                    self.progress.configure(value=100)
                    self.var_status.set("完成")
                    self._set_busy(False)
                    outdir = os.path.dirname(payload[0]) if payload else ""
                    self._append(f"[完成] 共输出 {len(payload)} 个文件 → {outdir}")
                    messagebox.showinfo(
                        "完成", f"处理完成，共 {len(payload)} 段。\n输出目录：{outdir}")
                elif kind == "fail":
                    self.var_status.set("失败")
                    self._set_busy(False)
                    self._append(f"[失败] {payload}")
                    messagebox.showerror("失败", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _append(self, msg: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    VideoSplitterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()