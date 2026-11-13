# Template Matching & Full-Pipeline Performance — Optimization Suggestions & Estimates

> **Status:** Exploratory performance notes and estimates (not scheduled commitments).  
> **Scope:** Scaling `main.py` (button template scraping), `post_processing.py` (batched CV + HUD OCR + trajectory reconstruction), and large-scale (~1,000-video) batch processing.

---

## 1. Observed Baseline Benchmark (`main.py`)

Measured on laptop hardware (`AMD Ryzen 7 8840HS` 8c/16t, `AMD Radeon 780M` iGPU, ~13.8 GB usable RAM) over 2,000 processed frames (`skip_frames=2`, `1920x1080`):

```text
Processing video: 100%|█████████████████████████████████████████████████████| 2000/2000 [01:08<00:00, 29.11frame/s]
Finished in 68.78 seconds; 0.0344 per frame.

--- Benchmark Breakdown (Effective Wall Time per frame) ---
Video Extraction:   0.01738s (50.5%)
Grayscale Convert:  0.00023s ( 0.7%)
Template Matching:  0.01427s (41.5%)
Other (Overhead):   0.00252s ( 7.3%)
-----------------------------------------------------------
```

### Key Takeaways from the Breakdown
1. **Video Extraction (`17.38 ms`, 50.5%) is the #1 bottleneck**: `VideoReader` (`frame.py:29-119`) wraps a single-threaded `cv2.VideoCapture` decoding 1080p frames on CPU inside `prefetch_generator` (`main.py:121-166`).
2. **Template Matching (`14.27 ms`, 41.5%) is the #2 bottleneck**: `ThreadPoolExecutor` (`main.py:222`) uses `min(32, os.cpu_count() + 4) = 20` threads on 16 logical cores. While OpenCV C++ calls (`cv2.matchTemplate`) release the GIL, pure-Python loop/object overhead in `eval_template.py` and `matcher.py` still contends on the GIL.
3. **Grayscale Conversion (`0.23 ms`, 0.7%) is negligible**: Cropping ROIs before `cv2.cvtColor` would not meaningfully reduce wall time.

---

## 2. Why GPU `cv2.matchTemplate` Is Not Recommended Here

For small UI button ROIs (`px_roi.expanded_bounds` in `eval_template.py:61-63`), uploading frame data/crops to VRAM and launching small CUDA `matchTemplate` kernels is slower than CPU because PCIe transfer and kernel-launch latency exceed compute time. On CPU, those small ROI crops stay hot in L1/L2/L3 cache.

**Optimal hardware split on a desktop (`Ryzen 7 9800X3D + RTX 5090 + 64 GB DDR5-5600`)**:
- **RTX 5090**: Hardware video decoding (**NVDEC**) + batched PyTorch CV/OCR in `post_processing.py`.
- **Ryzen 7 9800X3D**: Cache-resident CPU template matching (96 MB 3D V-Cache + Zen 5 IPC).

---

## 3. Suggested `main.py` Optimizations & Estimated Speedups

For a ~30-minute baseline video (~52,400 processed frames at `29.1 fps`):

### A. Running `main.py` As-Is on `9800X3D + RTX 5090 + 64 GB`
- **Estimate**: **~18–22 min (~40–48 fps, ~1.4–1.6× faster)**
- *Why*: `main.py` does not use the GPU; single-threaded CPU `cv2.VideoCapture` decode and GIL contention remain, so gains come only from 9800X3D single-thread boost and L3 V-Cache.

### B. Suggested Code Changes for `main.py`
1. **Offload Video Decoding to RTX 5090 NVDEC (`17.38 ms` → `~0.5–1.5 ms` effective)**:
   - Replace CPU `cv2.VideoCapture` in `frame.py` with `ffmpeg -hwaccel cuda` (or `PyNvVideoCodec`) and discard skipped frames (`skip_frames=2`) on-GPU/in-filter before copying kept frames to system RAM.
   - NVDEC decodes 1080p60 H.264/HEVC/AV1 at 1,000+ raw fps, hiding video extraction almost completely behind `prefetch_generator`.
2. **Eliminate GIL Contention (`~8–10 ms` on 9800X3D → `~4–6 ms`)**:
   - Switch `ThreadPoolExecutor` (`main.py:222`) to `ProcessPoolExecutor` (leaving 1–2 threads free for the NVDEC feeder/OS, e.g. `max_workers=14` on 16 logical cores).
   - Pre-compute static ROI bounds/slices once at startup instead of constructing `PixelROI` / `ImageSize` objects inside the per-frame `eval_template.py` hot loop.
   - *Estimated total*: **`~6–9 ms/frame` (~110–165 fps, ~3.8–5.7× faster)** → **~5–8 minutes** per 30-min baseline video.
3. **Optional Resolution Scaling for `main.py` (720p)**:
   - Because `cv2.matchTemplate` cost scales roughly quadratically with pixel count (ROI area × template area), running `main.py` at **720p** (with templates scaled proportionally) could cut `Template Matching` by another ~3–5×, bringing `main.py` down to **~2–3 minutes/video**.
   - `post_processing.py` could still read either 720p or 1080p for its sparse action frames if full-resolution card/shop crops are preferred.

---

## 4. `post_processing.py` & HUD OCR Comparative Estimates

Unlike `main.py` (~52,400 frames), `build_run_trajectory` (`post_processing.py:773-990`) deduplicates clicks and only extracts `needed_frames` (`post_processing.py:812-821`)—typically **~250–500 unique frames** for a full 8-Ante run (~100–200× fewer frames).

| Pipeline Component | Current Laptop (`8840HS` CPU) | `9800X3D + RTX 5090 + 64 GB` |
| --- | --- | --- |
| `extract_video_frames` (~300–500 sparse seeks) | ~3–10 s | ~2–5 s |
| `_run_batched_cv` (YOLO + MobileNetV3 in `classifiers/inference.py`) | ~20–60 s | ~0.5–2 s |
| `on_action` / `GameState` simulation loop | < 0.5 s | < 0.2 s |
| **HUD OCR** (`money`, `hands`, `discards`, `round`, `ante`, `reroll_cost`, `current_chips`) | ~1–4 s *(custom tiny CNN/glyph)* or ~15–60 s *(RapidOCR/PaddleOCR)* | **< 0.5–1 s** *(custom tiny CNN)* or **~2–6 s** *(GPU RapidOCR/PaddleOCR)* |
| **Future Temporal `END_ROUND` Model** (~24 rounds @ 15–20 fps) | ~10–25 s | ~1–3 s |
| **Total `post_processing.py`** | **~35–100 s** | **~5–15 s** |

### HUD OCR Design Note
- Because HUD counters use Balatro's fixed pixel font (`m6x11plus.ttf`) at fixed sidebar coordinates on the **same `needed_frames` already decoded**, OCR adds zero extra video-seeking cost.
- Prefer a **batched tiny PyTorch CNN/CRNN or digit glyph matcher** (or GPU-batched `RapidOCR`/`PaddleOCR`) on cropped HUD boxes over `pytesseract` subprocess calls (which add ~1–3 min of process overhead and struggle with colored drop-shadow pixel fonts).
- Only evaluate specific HUD crops on relevant actions (e.g., `current_chips` around `PLAY_HAND`/`END_ROUND`, `reroll_cost` in `SHOP`).

---

## 5. 1,000-Video Batch Processing & Rolling Download Queue Estimates

| Configuration | `main.py` / video | `post_processing.py` + OCR + `END_ROUND` / video | Estimated Total (1,000 videos) |
| --- | --- | --- | --- |
| Current Laptop (`8840HS`) as-is | ~30 min | ~1.5 min | ~525 hours (~22 days) |
| `9800X3D + RTX 5090` as-is (no code changes) | ~18–22 min | ~10–15 s | ~300–370 hours (~12.5–15.5 days) |
| `9800X3D + RTX 5090` + NVDEC + `ProcessPoolExecutor` (1080p) | ~5–8 min | ~10–15 s | **~87–137 hours (~3.6–5.7 days)** |
| Above + 720p `main.py` scaling | ~2–3 min | ~10–15 s | **~35–55 hours (~1.5–2.3 days)** |

### Rolling 50-Video Download Queue Considerations
Keeping ~50 videos buffered on disk (~50–150 GB) and deleting each after processing would have **negligible (<1–2%) compute impact** on an NVMe SSD, provided:
1. **Use an NVMe SSD** (3,000+ MB/s vs. ~5–15 MB/s sustained download + decode), not a spinning HDD or external USB drive.
2. **Download video-only streams** (e.g., `yt-dlp` H.264/HEVC/AV1 video-only): avoids `ffmpeg` audio+video muxing and prevents accidental CPU re-encoding.
3. **Leave 1–2 CPU threads of headroom** (`max_workers=14` on 16 logical cores) or run the downloader at low process priority so NVDEC feeding and downloads never starve workers.
4. **Atomic handoff & rate-limit pacing**: Download to a `.part`/staging path and rename into the ready queue on completion; pace `yt-dlp` requests once the 50-video buffer is full to avoid YouTube/Twitch 403/429 throttling over a multi-day run.
