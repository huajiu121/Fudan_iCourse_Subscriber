"""Download PPT images via WebVPN with retry."""

from __future__ import annotations

import time

from .icourse import ICourseClient
from .webvpn import get_vpn_url


def fetch_ppt_image(client: ICourseClient, item: dict,
                    max_attempts: int = 2, timeout: int = 30) -> bytes | None:
    """Download a single PPT image. Returns bytes or None on persistent failure."""
    url = item["pptimgurl"]
    for attempt in range(1, max_attempts + 1):
        try:
            vpn_url = get_vpn_url(url) if not url.startswith(client.base_url) else url
            resp = client.vpn.get(vpn_url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"[PPTFetcher] download failed (attempt {attempt}/{max_attempts}): "
                  f"{type(e).__name__}: {e}")
            if attempt < max_attempts:
                time.sleep(1)
    return None
