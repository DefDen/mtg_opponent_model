"""Reproducible experiment on a local 17Lands game-data file."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

from mtg_opponent_model.evaluation import evaluate, sample_cases
from mtg_opponent_model.model import CooccurrenceModel
from mtg_opponent_model.preprocessing import CardVocabulary, split_decks
from mtg_opponent_model.seventeenlands import SeventeenLandsLoader


class PopularityModel(CooccurrenceModel):
    """Ignore observations except to exclude already revealed candidates."""

    def rank(self, observed: Iterable[int], k: int | None = None) -> list[tuple[int, float]]:
        seen = set(observed)
        ranked = [item for item in super().rank([]) if item[0] not in seen]
        return ranked if k is None else ranked[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expansion", default="BLB")
    parser.add_argument("--format", default="PremierDraft")
    parser.add_argument("--max-rows", type=int, default=20000,
                        help="Game-row prefix limit; 0 reads the whole file")
    parser.add_argument("--observed-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/historical_results.json"))
    args = parser.parse_args()
    loader = SeventeenLandsLoader(args.path, args.expansion, args.format,
                                  None if args.max_rows == 0 else args.max_rows)
    decks = list(loader.load())
    train, test = split_decks(decks, test_fraction=0.2, seed=args.seed)
    vocabulary = CardVocabulary(card for deck in train for card in deck.cards)
    model = CooccurrenceModel().fit(vocabulary.encode(deck) for deck in train)
    popularity = PopularityModel().fit(vocabulary.encode(deck) for deck in train)
    cases = sample_cases(test, args.observed_count, args.seed)
    with args.path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "source_url": ("https://17lands-public.s3.amazonaws.com/analysis_data/game_data/"
                       f"game_data_public.{args.expansion}.{args.format}.csv.gz"),
        "source_sha256": digest.hexdigest(),
        "expansion": args.expansion, "format": args.format,
        "max_rows": args.max_rows, "seed": args.seed, "test_fraction": 0.2,
        "observed_count": args.observed_count, "basic_lands_included": True,
        "sampling": "file prefix" if args.max_rows else "whole file",
        "loader": asdict(loader.stats),
        "train_decks": len(train), "test_decks": len(test),
        "train_events": len({d.group_id for d in train}),
        "test_events": len({d.group_id for d in test}),
        "evaluation_cases": len(cases), "vocabulary_size": len(vocabulary.cards),
        "deck_size_range": [min(sum(d.cards.values()) for d in decks),
                            max(sum(d.cards.values()) for d in decks)],
        "cooccurrence": evaluate(model, vocabulary, cases),
        "popularity": evaluate(popularity, vocabulary, cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    vocabulary.save(args.output.with_suffix(".vocabulary.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
