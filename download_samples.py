#!/usr/bin/env python3
"""Download sample historical newspaper scans from Library of Congress."""
from pathlib import Path
import requests

SAMPLES = {
    "evening_star_1907_09_06": "https://www.loc.gov/resource/sn83045462/1907-09-06/ed-1/?sp=1",
    "evening_star_1907_12_07": "https://www.loc.gov/resource/sn83045462/1907-12-07/ed-1/?sp=1",
    "evening_star_1907_12_12": "https://www.loc.gov/resource/sn83045462/1907-12-12/ed-1/?sp=1",
}

def main():
    out = Path("samples")
    out.mkdir(exist_ok=True)
    for stem, page_url in SAMPLES.items():
        # LOC resource metadata may contain JPEG variants; prefer available image URLs.
        try:
            response = requests.get(page_url + "&fo=json", timeout=30,
                                    headers={"User-Agent": "research-ocr-prototype/0.1"})
            response.raise_for_status()
            data = response.json()
            resources = data.get("resources", [])
            images = []
            for resource in resources:
                images.extend(resource.get("image", []) if isinstance(resource.get("image"), list) else [])
            candidates = []
            for image in images:
                if isinstance(image, dict):
                    candidates.extend(image.get("url", []) if isinstance(image.get("url"), list) else [image.get("url")])
                elif isinstance(image, str):
                    candidates.append(image)
            image_url = next((u for u in reversed(candidates) if isinstance(u, str) and u.startswith("http")), None)
            if not image_url:
                print(f"No downloadable image found for {stem}. Download manually: {page_url}")
                continue
            image_response = requests.get(image_url, timeout=90)
            image_response.raise_for_status()
            dest = out / (stem + ".jpg")
            dest.write_bytes(image_response.content)
            print(f"Saved {dest}")
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"Could not download {stem}: {exc}. Manual source: {page_url}")

if __name__ == "__main__":
    main()
