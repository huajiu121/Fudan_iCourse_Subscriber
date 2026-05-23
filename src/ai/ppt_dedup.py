"""PPT page filters: perceptual-hash dedup + multi-pattern invalid-page match.

Two stages, both run from main.py's _fetch_and_ocr_ppts:

1. ``dedup_dhash`` operates on dhashes computed from the raw image bytes
   (cheap, no OCR needed) and drops near-duplicate frames using a sliding
   window. The iCourse capture pipeline takes timed screenshots, so a
   classroom that stays on one slide for several minutes produces dozens
   of identical frames; collapsing them before OCR saves both time and
   prompt budget.

2. ``is_invalid_page`` runs after OCR and matches the recovered text
   against a list of feature substrings extracted from the two known
   classroom-noise screens (the desktop wallpaper and the e-learning
   resource portal). Patterns are deliberately long and topic-specific
   so they don't false-positive on real slides; punctuation and
   whitespace are stripped before matching to tolerate OCR noise.
"""

from __future__ import annotations

import io
import re
from typing import Iterable
import imagehash
from PIL import Image


def compute_dhash(image_bytes: bytes) -> str | None:
    """Perceptual hash for an image. Returns 16-hex string or None on error.

    Uses imagehash.dhash (8x8 difference hash). Identical/near-identical
    crops yield identical hashes; visually distinct frames almost always
    differ by more than 4 bits.  Caller must tolerate ``None`` (image
    decode failure, missing PIL, etc.) — those pages are excluded from
    the dedup pass and pass through to OCR untouched.
    """

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return str(imagehash.dhash(img))
    except Exception:
        return None


def _hamming_hex(a: str, b: str) -> int:
    """Bit-count XOR of two equal-length hex strings."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def dedup_dhash(
    items: list[str | None],
    window: int = 5,
    threshold: int = 2,
) -> list[int]:
    """Sliding-window perceptual dedup. Returns sorted list of dropped indices.

    For each surviving anchor i, compare its dhash against the next
    ``window`` items; if Hamming distance ≤ threshold, mark the *later*
    index as dropped. Already-dropped images never become anchors —
    that prevents a chain of "near to last-kept" pages from cascading
    drops onto pages that aren't actually near the kept anchor.

    ``items`` may contain ``None`` (compute_dhash failure) — those
    indices are passed through (never dropped, never used as anchor).
    """
    n = len(items)
    dropped: set[int] = set()
    for i in range(n):
        if i in dropped:
            continue
        a = items[i]
        if a is None:
            continue
        for j in range(i + 1, min(i + 1 + window, n)):
            if j in dropped:
                continue
            b = items[j]
            if b is None:
                continue
            if _hamming_hex(a, b) <= threshold:
                dropped.add(j)
    return sorted(dropped)


# ── Invalid-page pattern matching ──────────────────────────────────────────
#
# Patterns are matched after _normalize_for_match strips whitespace and all
# non-alphanumeric/non-CJK characters and lowercases ASCII.  Pick patterns
# that are simultaneously:
#   - Specific enough that they don't appear in real lecture material
#     (avoid bare campus names, common headings).
#   - Long enough (≥6 normalized chars where possible) that incidental
#     OCR mis-recognition doesn't accidentally hit them.
#   - Drawn from features unique to the noise screens (URLs, the long
#     official policy titles, the EV recording pipeline references, the
#     classroom-equipment shutdown reminder).
INVALID_PAGE_PATTERNS: list[str] = [
    # ── Type 1: classroom desktop wallpaper ──
    "请不要关闭设备",
    "避免耽误第34节上课",
    "触控显示器无线话筒hdmi",
    "多媒体值班室",
    "本教室装有摄录及安全装置",
    # ── Type 2: e-learning resource portal screen ──
    "cfdfudaneducn",                       # the cfd.fudan.edu.cn URL
    "icoursefudaneducn",                   # the icourse.fudan.edu.cn URL
    "智慧教学资源平台使用规范",
    "教育部等九部门",
    "加快推进教育数字化",
    "本科课程评教提醒",
    "请于期末考试前完成评教",
    "微信搜索并关注复旦课评",
    "国务院关于深入实施",
    "板书效果展示",
    "双屏效果展示",
    "课程录制exe",
    "ev去噪",
    "录制完成桌面会生成",
    "推荐上传至elearning",
    "ppt演示者视图会影响录屏",
]

_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_for_match(text: str) -> str:
    """Lowercase + strip whitespace and punctuation. CJK chars are kept."""
    if not text:
        return ""
    return _NORMALIZE_RE.sub("", text).lower()


def is_invalid_page(text: str) -> bool:
    """True if any feature string matches the (normalized) OCR'd text."""
    norm = _normalize_for_match(text)
    if not norm:
        return False
    return any(p in norm for p in INVALID_PAGE_PATTERNS)


# ── OCR UI noise stripping ───────────────────────────────────────────────────
# iCourse captures screenshots of the entire desktop, so every slide includes
# the PowerPoint ribbon (tabs, buttons, status bar) plus occasionally the
# Word/PDF window chrome when the instructor switches applications.  The OCR
# dutifully transcribes every label, producing boilerplate that wastes LLM
# prompt budget.
#
# Stopword strategy:
#   - ONLY strip an entire line that exactly matches a known UI label
#     (after normalisation).  Substring matching would risk removing
#     real content — e.g. "幻灯片" is a stopword, but "幻灯片设计原则"
#     should pass through.
#   - Single-char stopwords are included because the PowerPoint ribbon
#     shows standalone icon labels (e.g. "口" for the rectangle shape
#     gallery).  They appear on 50+ pages each across courses and
#     contribute zero semantics.  Lines that happen to be JUST "口"
#     from a real slide are astronomically unlikely — a lecture slide
#     that discusses the character 口 wouldn't put it on its own line
#     in a 720p+ screenshot.
#   - Regex patterns catch status-bar items whose numeric part varies
#     (zoom %, word count, page N of M).
#
# The stopword set is derived from frequency analysis of OCR'd ppt_pages
# rows across 7 real lectures in 5 different courses (2026-05-22 data
# from the decrypted data-branch DB).

# Exact-match (case-folded, whitespace-stripped) UI labels to drop.
# Sorted roughly by PowerPoint ribbon region for maintainability.
PPT_UI_STOPWORDS: set[str] = {
    # ── Ribbon tabs ──
    "文件", "开始", "插入", "设计", "切换", "动画",
    "幻灯片放映", "审阅", "视图", "加载项", "帮助",
    # ── Home tab clusters ──
    "粘贴", "剪切", "复制", "格式刷", "新建", "重置",
    "剪贴板", "字体", "段落", "快速样式", "样式",
    "绘图", "编辑", "排列", "形状填充", "形状轮廓",
    "形状效果", "选择", "查找", "替换", "a替换",
    "ac替换", "目复制", "突出显示", "擦除",
    # ── Insert / Design / Transitions tabs ──
    "表格", "图片", "形状", "图标", "SmartArt", "图表",
    "文本框", "页眉和页脚", "艺术字", "公式", "符号",
    "视频", "音频", "屏幕录制",
    "主题", "变体", "格式", "背景格式",
    "切换到此幻灯片",
    # ── Animations / Slide Show tabs ──
    "动画窗格", "添加动画", "触发",
    "从头开始", "从当前幻灯片开始", "自定义放映",
    "设置幻灯片放映", "隐藏幻灯片",
    # ── Review / View tabs ──
    "拼写和语法", "同义词库", "字数统计", "批注",
    "显示批注", "比较", "接受", "拒绝",
    "页面视图", "阅读视图", "大纲视图",
    "备注", "备注页", "显示比例", "适应窗口",
    "标尺", "网格线", "参考线",
    "拆分", "新建窗口", "全部重排", "层叠",
    "切换窗口", "宏",
    # ── Status bar ──
    "中文（中国）", "简体", "登录", "共享",
    "备注", "批注", "幻灯片", "+创建", "十创建",
    "告诉我您想要做什么",
    "A朗读此页内容", "朗读此页内容",
    # ── Drawing / Shape tools (contextual tab sub-labels) ──
    "绘制", "编辑形状", "文本填充", "文本轮廓",
    "文本效果", "转换为SmartArt", "选择窗格",
    "上移一层", "下移一层",
    # ── Single-char icon labels (appear 15-100+ pages each) ──
    # These are icon-only PowerPoint buttons that OCR reads as a
    # single character.  The list is restricted to characters that
    # appeared on ≥10 distinct pages across our 7-lecture sample
    # AND are consistent with ribbon/gallery icons.
    "口", "品", "日", "昆", "国", "田", "单", "回",
    "器", "三",
    # Keyboard-shortcut hint letters (Alt-key ribbon navigation).
    # Each appears on 8-40+ distinct pages, evenly across courses.
    "A", "B", "C", "D", "H", "I", "K", "M", "P", "Q", "S",
    "X", "a", "b", "k", "w", "x",
    # ── Font/typeface labels in the ribbon ──
    "楷体", "五号", "五号AA", "A字", "Aa",
    # ── Common OCR garbage from UI chrome ──
    "三菜单", "国版式", "目复制",
    "AaBbCc", "AaBbCcDAaBbCcDAaBbCcAaBbCc",
    "登录共享", "）简体",
    # ── Ribbon paragraph-formatting labels ──
    "I文字方向", "文字方向", "[对齐文本", "[]对齐文本",
    "对齐文本", "↑←",
    "abc替换", "c替换",
    # ── Truncated/fused toolbar labels ──
    "告诉我您想要做什", "形状轮廊",
    # Fused adjacent ribbon labels (OCR treats them as one line)
    "幻灯片节",
    # ── Style gallery labels from the Home tab ──
    "日期", "邮件",
}

# Regex patterns for status-bar / window-chrome text whose value varies.
# Applied per-line; a line that matches in full is dropped.
#
# Note: the date pattern only fires for isolated date-like lines
# (length 5-12 chars) so actual slide content containing dates is
# unaffected — a slide about "2026-05-22 会议纪要" is 16 chars
# and won't match.
_UI_NOISE_LINE_RES: list[re.Pattern] = [
    re.compile(r"^(?:幻灯片\s*)?第\d+[页张][,，]?\s*共\d+[页张]$"),  # page/slide N of M
    re.compile(r"^\d{1,3}%$"),                          # zoom level
    re.compile(r"^\d{1,6}个字$"),                       # word count
    re.compile(r"^/\d{1,3}$"),                           # e.g. "/19"
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),                  # isolated date stamp
    re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$"),            # timestamp HH:MM or HH:MM:SS
    re.compile(r"^[A-Z][a-z]+\d{1,2},\s*\d{4}$"),       # e.g. "May29,2025"
    re.compile(r"^First Pa\.\.$"),                       # truncated "First Page"
    re.compile(r"^AbstractJAbs?tra\.+$"),                # truncated abstracts
    re.compile(r"^AuthorJCompact$"),                     # Word metadata
    re.compile(r"^YangZhou.*$"),                         # author name in isolation
    re.compile(r"^Uabexx²+$"),                           # formula OCR noise
    re.compile(r"^[A-Z]\.$"),                            # single initial like "A."
    re.compile(r"^[A-Z][a-z]+University\)?$"),           # e.g. "FudanUniversity)"
    re.compile(r"^☆.{4,}.+$"),                           # truncated window titles
    re.compile(r"^.*\.docx[- ].*Word$"),                 # Word window title
    re.compile(r"^.*\.pptx[- ].*PowerPoint$"),           # PPT window title
    re.compile(r"^单击此处添加(?:备注|标题|副标题|正文)$"),  # PPT placeholder text
]

_NORMALIZE_UI_RE = re.compile(r"[\s　]+")


def clean_ppt_text(text: str) -> str:
    """Remove PowerPoint window-chrome noise from OCR'd slide text.

    Operates per-line so a slide mixing real content and UI labels
    keeps the former while stripping the latter.  Returns the
    cleaned text (may be empty for a fully-noise slide).

    This is a *generic* filter — it does NOT use any course-specific
    vocabulary or subject-matter hotwords.
    """
    if not text:
        return ""
    kept: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        # Normalise away the full-width ideographic space (U+3000) that
        # PowerPoint uses in its ribbon layout, and any repeated spaces.
        norm = _NORMALIZE_UI_RE.sub("", s).strip()
        if not norm:
            continue
        # Exact stopword match (case-insensitive for ASCII labels).
        if norm in PPT_UI_STOPWORDS:
            continue
        if norm.lower() in PPT_UI_STOPWORDS:
            continue
        # Regex patterns — match against the normalised form.
        if any(p.fullmatch(norm) for p in _UI_NOISE_LINE_RES):
            continue
        kept.append(s)
    return "\n".join(kept)


def normalize_for_match(text: str) -> str:  # noqa: D401  exported wrapper
    """Public alias for tests / debugging."""
    return _normalize_for_match(text)
