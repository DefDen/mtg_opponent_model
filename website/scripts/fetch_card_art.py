"""Cache Scryfall metadata and bundle images for cards displayed in replays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "HiddenHandResearch/0.1", "Accept": "application/json"}


def fetch(url: str) -> bytes:
    with urlopen(Request(url, headers=HEADERS), timeout=40) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, nargs="*", help="Optional cached Scryfall search pages")
    parser.add_argument("--all-pool", action="store_true", help="Bundle the trained card pool and BLB tokens for arbitrary imports")
    args = parser.parse_args()
    cards = []
    if args.metadata:
        for path in args.metadata:
            cards.extend(json.loads(path.read_text())["data"])
    else:
        url = "https://api.scryfall.com/cards/search?q=%28set%3Ablb%20or%20set%3Atblb%29&unique=cards"
        while url:
            page = json.loads(fetch(url))
            cards.extend(page["data"])
            url = page.get("next_page")
            time.sleep(0.12)
    lookup = {card["name"]: card for card in cards}
    dataset = json.loads((ROOT / "dist/data.json").read_text())
    visible = set()
    for game in dataset["cases"]:
        for frame in game["frames"]:
            visible.update(frame["actual_hand"])
            visible.update(frame["step"]["departed"])
            for player in frame.get("board", {}).values():
                for zone in ("lands", "creatures", "other"):
                    visible.update(player[zone])
            visible.update(card for card, _ in sorted(frame["probabilities"].items(),
                                                    key=lambda pair: (-pair[1], pair[0]))[:20])
            for probabilities in frame.get('methods', {}).values():
                visible.update(card for card, _ in sorted(probabilities.items(),
                               key=lambda pair: (-pair[1], pair[0]))[:20])
    if args.all_pool:
        models = json.loads((ROOT / 'dist/browser-models.json').read_text())
        visible.update(models['cards'])
        visible.update(card['name'] for card in cards if card['set'] == 'tblb')
    missing = visible - lookup.keys()
    if missing and args.all_pool:
        # BLB Limited can include Special Guests whose printing is outside BLB.
        for start in range(0, len(missing), 75):
            batch = sorted(missing)[start:start + 75]
            request = Request('https://api.scryfall.com/cards/collection',
                              data=json.dumps({'identifiers': [{'name': c} for c in batch]}).encode(),
                              headers={**HEADERS, 'Content-Type': 'application/json'})
            with urlopen(request, timeout=40) as response:
                extra = json.loads(response.read())
            lookup.update({card['name']: card for card in extra['data']})
            time.sleep(0.12)
        missing = visible - lookup.keys()
    if missing:
        raise ValueError(f"Missing Scryfall metadata: {sorted(missing)}")
    asset_dir = ROOT / "dist/assets/cards"
    asset_dir.mkdir(parents=True, exist_ok=True)

    def download(name: str) -> tuple[str, dict]:
        card = lookup[name]
        image = card.get("image_uris", card.get("card_faces", [{}])[0].get("image_uris", {}))["normal"]
        filename = card["id"] + ".jpg"
        path = asset_dir / filename
        if not path.exists():
            content = fetch(image)
            if not content.startswith(b"\xff\xd8"):
                raise ValueError(f"Expected a JPEG card image for {name}")
            path.write_bytes(content)
        return name, {"image": f"assets/cards/{filename}", "source_url": card["scryfall_uri"],
                      "image_source_url": image, "artist": card.get("artist", ""),
                      "set": card["set"], "collector_number": card["collector_number"]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        images = dict(pool.map(download, sorted(visible)))
    (ROOT / "dist/card-images.json").write_text(json.dumps(images, indent=2) + "\n", encoding="utf-8")
    print(f"Bundled {len(images)} real card images ({sum(p.stat().st_size for p in asset_dir.glob('*.jpg')):,} bytes)")


if __name__ == "__main__":
    main()
