from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"
DEFAULT_TIMEOUT_SECONDS = 60
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def fetch_cards(api_url: str, timeout_seconds: int) -> dict:
    logger.info("Fetching card data from %s (timeout=%ss)", api_url, timeout_seconds)
    try:
        request = Request(api_url, headers=DEFAULT_HEADERS)
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec: B310
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        logger.exception("Failed to fetch YGOPRODeck data")
        raise RuntimeError(f"Failed to fetch YGOPRODeck data: {exc}") from exc

    data = json.loads(payload)
    if "data" not in data or not isinstance(data["data"], list):
        raise RuntimeError("Unexpected YGOPRODeck response format: missing 'data' list")

    logger.info("Fetched %d cards from YGOPRODeck", len(data["data"]))
    return data


def save_raw_payload(payload: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"ygoprodeck_cards_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    logger.info("Wrote raw payload to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch full card dataset from YGOPRODeck and store raw JSON."
    )
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help="YGOPRODeck API endpoint for card data.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where raw JSON output will be written.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting YGOPRODeck ingestion script")
    payload = fetch_cards(api_url=args.api_url, timeout_seconds=args.timeout_seconds)
    output_path = save_raw_payload(payload=payload, output_dir=Path(args.output_dir))
    logger.info("Saved %d cards to %s", len(payload["data"]), output_path)


if __name__ == "__main__":
    main()
