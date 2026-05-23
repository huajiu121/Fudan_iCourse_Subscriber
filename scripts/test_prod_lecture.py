#!/usr/bin/env python3
"""Production-mode end-to-end test for a single lecture.

Runs the exact same pipeline as ``main.py`` does for one lecture — login,
PPT crawl, image download, dHash dedup, OCR, audio download + ASR,
bucketed prompt assembly, LLM summarization — **without touching the
database**.  Every intermediate artifact is written to ``--out`` so a
human can inspect quality / failure modes without needing to re-decrypt
the production DB.

Use this for:
  - A/B comparing ASR backends (FireRed vs SenseVoice vs Zipformer)
  - Measuring per-stage CPU + throughput on the actual GitHub runner
  - Validating OCR / dedup / bucketer / summary quality on a real lecture

Usage:
  python scripts/test_prod_lecture.py [--course-id C] [--sub-id S]
                                     [--asr {firered,sensevoice,zipformer}]
                                     [--out DIR]
                                     [--skip-audio]
                                     [--skip-summary]

If course-id / sub-id are omitted, picks 机器学习系统 (33974)'s most
recent lecture-with-playback as a STEM default.

Reads credentials from the usual StuId / UISPsw / DASHSCOPE_API_KEY /
DEEPSEEK_API_KEY / GEMINI_API_KEY env vars — same as production.

Outputs to ``test_prod_out/<sub_id>/``:
  meta.json          everything we know about the lecture + course
  images/<page>.jpg  raw PPT images as fetched (kept pages only)
  dedup.json         dHash decisions per page
  ocr/<page>.txt     per-page OCR text + invalid-page classification
  audio.wav          decoded PCM audio (16k mono f32le -> wav header)
  transcript.txt     plain transcript
  segments.json      list of {start_ms, end_ms, text}
  prompt.txt         final bucketed prompt sent to the LLM
  summary.md         LLM output
  metrics.json       per-stage timing + char/page/MB counts + model used
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default STEM target — picked because 机器学习系统 has real PPT slides
# (not just a 板书 camera), so OCR + dedup + summary are all exercised.
DEFAULT_COURSE_ID = "33974"


def _save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pick_lecture(detail: dict, sub_id: Optional[str]) -> dict:
    """Pick a specific sub_id, or the most recent lecture with playback."""
    lectures = detail["lectures"]
    if sub_id:
        for lec in lectures:
            if str(lec["sub_id"]) == str(sub_id):
                return lec
        raise SystemExit(f"sub_id={sub_id} not found in course {detail['title']}")
    playbacks = [l for l in lectures if l.get("has_playback")]
    if not playbacks:
        raise SystemExit(f"course {detail['title']} has no lectures with playback")
    # Lectures come back roughly chronological; take the latest.
    return playbacks[-1]


def _run_ppt(client, course_id: str, sub_id: str, out_dir: str,
             metrics: dict) -> tuple[list[dict], dict[int, bytes]]:
    """Download PPT images + dHash dedup + invalid-page filter + OCR.

    Returns (kept_pages, ocr_text_by_page) where kept_pages is the list of
    page dicts that survived dedup AND OCR classified as 'done' (not
    invalid).  Image bytes are also written to disk for human inspection.
    """
    from src.api import icourse as icourse_mod
    from src.ai.ocr import ocr_image_text
    from src.ai.ppt_dedup import compute_dhash, dedup_dhash, is_invalid_page

    t0 = time.time()
    items = client.get_ppt_list(course_id, sub_id)
    items.sort(key=lambda x: x.get("created_sec") or 0)
    metrics["ppt.list_count"] = len(items)
    metrics["ppt.list_seconds"] = time.time() - t0
    print(f"[PPT] Listed {len(items)} pages in {metrics['ppt.list_seconds']:.1f}s")

    # Parallel download (mimics production's image_pool=20)
    t0 = time.time()
    images: dict[int, bytes] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {
            pool.submit(icourse_mod.fetch_ppt_image, client, item): item
            for item in items
        }
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                img = fut.result()
            except Exception as e:
                print(f"  [WARN] page {item['page_num']} download err: {e}")
                continue
            if img is not None:
                images[item["page_num"]] = img
    metrics["ppt.download_seconds"] = time.time() - t0
    metrics["ppt.downloaded"] = len(images)
    metrics["ppt.download_pic_per_s"] = (
        len(images) / metrics["ppt.download_seconds"]
        if metrics["ppt.download_seconds"] > 0 else 0
    )
    print(
        f"[PPT] Downloaded {len(images)} images in "
        f"{metrics['ppt.download_seconds']:.1f}s "
        f"({metrics['ppt.download_pic_per_s']:.1f} pic/s)"
    )

    # dHash dedup — *before* OCR so we don't waste CPU on near-duplicates
    t0 = time.time()
    dhashes_in_order: list[str | None] = []
    page_at_index: list[int] = []
    for item in items:
        page_num = item["page_num"]
        img = images.get(page_num)
        if img is None:
            continue
        dh = compute_dhash(img)
        dhashes_in_order.append(dh)
        page_at_index.append(page_num)
    dropped_idx = dedup_dhash(dhashes_in_order)
    dropped_pages = {page_at_index[i] for i in dropped_idx}
    metrics["ppt.dedup_seconds"] = time.time() - t0
    metrics["ppt.dedup_dropped"] = len(dropped_pages)
    print(
        f"[PPT] dHash dropped {len(dropped_pages)} dup pages in "
        f"{metrics['ppt.dedup_seconds']:.2f}s"
    )

    # Save raw images for kept pages
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for page_num, img in images.items():
        if page_num in dropped_pages:
            continue
        with open(os.path.join(img_dir, f"{page_num:04d}.jpg"), "wb") as f:
            f.write(img)
    _save_json(os.path.join(out_dir, "dedup.json"), {
        "total": len(items),
        "downloaded": len(images),
        "dropped_as_duplicate": sorted(dropped_pages),
    })

    # OCR every kept page.  Use small pool (4 workers) since RapidOCR is
    # CPU-bound and we want clean throughput numbers — the production
    # ResourceMonitor's dynamic concurrency would muddle the measurement.
    t0 = time.time()
    ocr_dir = os.path.join(out_dir, "ocr")
    os.makedirs(ocr_dir, exist_ok=True)
    kept_pages: list[dict] = []
    ocr_text: dict[int, str] = {}
    invalid_count = 0

    def _ocr_one(item):
        page_num = item["page_num"]
        if page_num in dropped_pages or page_num not in images:
            return None
        try:
            text = ocr_image_text(images[page_num])
        except Exception as e:
            return (page_num, None, "failed", str(e))
        status = "invalid" if is_invalid_page(text) else "done"
        return (page_num, text, status, None)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(_ocr_one, items):
            if result is None:
                continue
            page_num, text, status, err = result
            item_for_meta = next(
                i for i in items if i["page_num"] == page_num
            )
            ocr_text[page_num] = text or ""
            _save_text(
                os.path.join(ocr_dir, f"{page_num:04d}.{status}.txt"),
                text or f"(failed: {err})",
            )
            if status == "done":
                kept_pages.append({
                    "page_num": page_num,
                    "created_sec": item_for_meta.get("created_sec") or 0,
                    "text": text or "",
                })
            elif status == "invalid":
                invalid_count += 1

    metrics["ppt.ocr_seconds"] = time.time() - t0
    metrics["ppt.ocr_done"] = len(kept_pages)
    metrics["ppt.ocr_invalid"] = invalid_count
    metrics["ppt.ocr_page_per_s"] = (
        (len(kept_pages) + invalid_count) / metrics["ppt.ocr_seconds"]
        if metrics["ppt.ocr_seconds"] > 0 else 0
    )
    print(
        f"[PPT] OCR done: {len(kept_pages)} kept, {invalid_count} invalid "
        f"in {metrics['ppt.ocr_seconds']:.1f}s "
        f"({metrics['ppt.ocr_page_per_s']:.2f} page/s)"
    )

    kept_pages.sort(key=lambda p: p["created_sec"])
    return kept_pages, ocr_text


def _run_asr(client, course_id: str, sub_id: str, out_dir: str,
             metrics: dict) -> tuple[str, list[dict]]:
    """Stream the lecture's audio + run ASR.  Returns (transcript, segments)."""
    from src.ai.transcriber import Transcriber

    t0 = time.time()
    video_url = client.get_video_url(course_id, sub_id)
    metrics["asr.video_url_seconds"] = time.time() - t0
    if not video_url:
        raise SystemExit(f"No video URL for {sub_id} — lecture has no playback?")

    print(f"[ASR] Got video URL in {metrics['asr.video_url_seconds']:.1f}s")

    t0 = time.time()
    transcriber = Transcriber()
    metrics["asr.model_load_seconds"] = time.time() - t0

    t0 = time.time()
    transcript, segments = transcriber.transcribe_url(video_url)
    metrics["asr.transcribe_seconds"] = time.time() - t0
    metrics["asr.chars"] = len(transcript)
    metrics["asr.segments"] = len(segments)
    if segments:
        audio_ms = segments[-1]["end_ms"]
        metrics["asr.audio_ms"] = audio_ms
        metrics["asr.realtime_factor"] = (
            audio_ms / 1000 / metrics["asr.transcribe_seconds"]
            if metrics["asr.transcribe_seconds"] > 0 else 0
        )
    print(
        f"[ASR] Transcribed {metrics['asr.chars']} chars / "
        f"{metrics['asr.segments']} segments in "
        f"{metrics['asr.transcribe_seconds']:.0f}s "
        f"({metrics.get('asr.realtime_factor', 0):.1f}× realtime)"
    )

    _save_text(os.path.join(out_dir, "transcript.txt"), transcript)
    _save_json(os.path.join(out_dir, "segments.json"), segments)
    return transcript, segments


def _run_summary(transcript: str, segments: list[dict],
                 kept_pages: list[dict], course_title: str,
                 out_dir: str, metrics: dict) -> str:
    """Assemble bucketed prompt + call LLM."""
    from src.ai import bucketer
    from src.ai.summarizer import Summarizer

    t0 = time.time()
    prompt_text, mode = bucketer.assemble(transcript, segments, kept_pages)
    metrics["summary.bucketer_seconds"] = time.time() - t0
    metrics["summary.prompt_chars"] = len(prompt_text)
    metrics["summary.prompt_mode"] = mode
    print(
        f"[Summary] Prompt assembled ({mode}, {len(prompt_text)} chars) "
        f"in {metrics['summary.bucketer_seconds']:.2f}s"
    )
    _save_text(os.path.join(out_dir, "prompt.txt"), prompt_text)

    summarizer = Summarizer()
    t0 = time.time()
    summary, model_used = summarizer.summarize(course_title, prompt_text)
    metrics["summary.llm_seconds"] = time.time() - t0
    metrics["summary.output_chars"] = len(summary)
    metrics["summary.model_used"] = model_used
    print(
        f"[Summary] {model_used}: {len(prompt_text)} → {len(summary)} chars "
        f"in {metrics['summary.llm_seconds']:.0f}s"
    )
    _save_text(os.path.join(out_dir, "summary.md"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", default=DEFAULT_COURSE_ID)
    parser.add_argument("--sub-id", default=None)
    parser.add_argument("--out", default="test_prod_out")
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="Don't download / transcribe audio. Useful for fast PPT-only iteration."
    )
    parser.add_argument(
        "--skip-summary", action="store_true",
        help="Don't call the LLM. Useful when iterating on ASR/OCR only."
    )
    args = parser.parse_args()

    from src.api.webvpn import WebVPNSession
    from src.api.icourse import ICourseClient

    print("=" * 60)
    print("Production-mode lecture test")
    print("=" * 60)

    print("\n[Login] WebVPN...")
    vpn = WebVPNSession()
    vpn.login()
    print("[Login] iCourse CAS...")
    vpn.authenticate_icourse()
    client = ICourseClient(vpn)

    print(f"\n[Course] Fetching detail for {args.course_id}...")
    detail = client.get_course_detail(args.course_id)
    lecture = _pick_lecture(detail, args.sub_id)
    sub_id = str(lecture["sub_id"])
    print(
        f"[Course] Title: {detail['title']} ({detail.get('teacher','?')})\n"
        f"[Lecture] sub_id={sub_id} title={lecture.get('sub_title')!r} "
        f"date={lecture.get('date')!r}"
    )

    out_dir = os.path.join(args.out, sub_id)
    os.makedirs(out_dir, exist_ok=True)
    metrics: dict = {
        "course_id": args.course_id,
        "course_title": detail["title"],
        "teacher": detail.get("teacher"),
        "sub_id": sub_id,
        "sub_title": lecture.get("sub_title"),
        "date": lecture.get("date"),
        "wall_start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_json(os.path.join(out_dir, "meta.json"), {
        "course": detail,
        "lecture": lecture,
    })

    overall_t0 = time.time()
    kept_pages, _ocr_text = _run_ppt(
        client, args.course_id, sub_id, out_dir, metrics,
    )

    transcript = ""
    segments: list[dict] = []
    if not args.skip_audio:
        transcript, segments = _run_asr(
            client, args.course_id, sub_id, out_dir, metrics,
        )
    else:
        print("\n[ASR] Skipped (--skip-audio).")

    if not args.skip_summary and transcript:
        _run_summary(
            transcript, segments, kept_pages,
            detail["title"], out_dir, metrics,
        )
    else:
        print("\n[Summary] Skipped.")

    metrics["wall_total_seconds"] = time.time() - overall_t0
    metrics["wall_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_json(os.path.join(out_dir, "metrics.json"), metrics)

    print(f"\n{'=' * 60}")
    print(f"All artifacts written to {out_dir}")
    print(f"Total wall time: {metrics['wall_total_seconds']:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
