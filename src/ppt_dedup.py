"""Filter classroom-desktop screenshots and duplicate-text PPTs.

Two filters:
  1. Classroom desktop background detection — match against canonical
     wallpaper text. ≥60% LCS overlap → drop.
  2. Adjacent-frame Jaccard dedup on token sets — ≥85% overlap → drop later.
"""

from __future__ import annotations

import re

CLASSROOM_DESKTOP_TEXT = (
    "上午下午第12节课后请不要关闭设备"
    "避免耽误第34节上课的老师谢谢"
    "触控显示器无线话筒HDMI"
    "多媒体值班室"
    "本教室装有摄录及安全装置"
)

_NORMALIZE_RE = re.compile(r"[\s　]+")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text).lower()


def _longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def is_classroom_desktop(ocr_text: str, threshold: float = 0.6) -> bool:
    norm = _normalize(ocr_text)
    if not norm:
        return False
    canonical = _normalize(CLASSROOM_DESKTOP_TEXT)
    overlap = _longest_common_substring_len(norm, canonical)
    return overlap / max(len(norm), 1) >= threshold


def _tokenize(text: str) -> set[str]:
    chars = re.findall(r"[A-Za-z0-9]+|[一-鿿]", text)
    return set(c.lower() for c in chars if c)


def jaccard(a: str, b: str) -> float:
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def dedup_adjacent(
    pages: list[dict], jaccard_threshold: float = 0.85,
) -> tuple[list[dict], list[dict]]:
    """First page in any duplicate cluster kept; subsequent near-dups dropped."""
    kept = []
    dropped = []
    last_text = None
    for page in pages:
        text = page.get("text", "")
        if last_text is not None and jaccard(text, last_text) >= jaccard_threshold:
            dropped.append(page)
            continue
        kept.append(page)
        last_text = text
    return kept, dropped


def filter_pages(pages: list[dict]) -> tuple[list[dict], dict]:
    """Apply both filters. Returns (kept, stats) with desktop_dropped + jaccard_dropped."""
    after_desktop = []
    desktop_dropped = 0
    for p in pages:
        if is_classroom_desktop(p.get("text", "")):
            desktop_dropped += 1
            continue
        after_desktop.append(p)
    kept, jaccard_drops = dedup_adjacent(after_desktop)
    return kept, {
        "desktop_dropped": desktop_dropped,
        "jaccard_dropped": len(jaccard_drops),
    }
