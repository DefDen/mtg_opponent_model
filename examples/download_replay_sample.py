"""Save complete game rows from a bounded 17Lands replay download."""
import argparse
import csv
import gzip
import io
from pathlib import Path
from urllib.request import Request, urlopen


REPLAY_URL = ("https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/"
              "replay_data_public.BLB.PremierDraft.csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--compressed-bytes", type=int, default=8388608)
    parser.add_argument("--prefix", type=Path,
                        help="Reuse a local compressed prefix instead of downloading")
    parser.add_argument("--output", type=Path,
                        default=Path("data/replay_sample.BLB.PremierDraft.csv.gz"))
    args = parser.parse_args()
    if args.games <= 0 or args.compressed_bytes <= 0:
        parser.error("games and compressed-bytes must be positive")
    request = Request(REPLAY_URL, headers={"Range": f"bytes=0-{args.compressed_bytes - 1}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    try:
        if args.prefix is not None:
            with args.prefix.open("rb") as stream:
                compressed = stream.read(args.compressed_bytes)
        else:
            with urlopen(request, timeout=30) as response:
                if response.status != 206:
                    raise ValueError("Server did not honor the bounded range request")
                compressed = response.read(args.compressed_bytes)
        # A range is an incomplete gzip stream. Read only requested complete
        # rows and write a new valid gzip; never pretend the range is a dataset.
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as source:
                reader = csv.reader(source)
                header = next(reader)
                with gzip.open(temporary, "wt", encoding="utf-8", newline="") as destination:
                    writer = csv.writer(destination)
                    writer.writerow(header)
                    for _ in range(args.games):
                        row = next(reader)
                        if len(row) != len(header):
                            raise ValueError("Incomplete replay row")
                        writer.writerow(row)
        temporary.replace(args.output)
    except (EOFError, StopIteration) as exc:
        raise ValueError("Replay prefix too short; increase --compressed-bytes") from exc
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {args.games} complete game rows to {args.output}")


if __name__ == "__main__":
    main()
