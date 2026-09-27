"""Score a news/social text corpus with FinBERT (or Twitter-RoBERTa) into daily features.

Input: a CSV with a timestamp column and a text column - headlines, article titles
or tweets that you supply (no free API serves five years of crypto headlines, so the
corpus is the part you bring). Output: ``data/external/news_sentiment_<symbol>.csv``,
which ``external_features.py`` picks up automatically.

    python scripts/score_news_sentiment.py --input news.csv --symbol btcusdt

Needs ``transformers`` (not in requirements-model-comparison.txt):

    pip install transformers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "finbert": "ProsusAI/finbert",
    "twitter-roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}
POSITIVE = {"positive", "label_2", "bullish"}
NEGATIVE = {"negative", "label_0", "bearish"}


def _load_pipeline(model: str, batch_size: int):
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise SystemExit("transformers is not installed. Run: pip install transformers") from exc
    return pipeline(
        "sentiment-analysis",
        model=MODELS.get(model, model),
        truncation=True,
        max_length=256,
        batch_size=batch_size,
    )


def polarity(label: str, score: float) -> float:
    """Map a classifier label to a signed score in [-1, 1]."""
    lowered = label.strip().lower()
    if lowered in POSITIVE:
        return score
    if lowered in NEGATIVE:
        return -score
    return 0.0


def aggregate(frame: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Aggregate per-document polarity into a time series of sentiment features."""
    grouped = frame.set_index("time").groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame(
        {
            "sent_mean": grouped["polarity"].mean(),
            "sent_pos_ratio": grouped["polarity"].apply(lambda values: float((values > 0).mean()) if len(values) else np.nan),
            "sent_count": grouped["polarity"].size(),
        }
    )
    return out[out["sent_count"] > 0].reset_index()


def score_corpus(
    input_path: Path,
    time_column: str,
    text_column: str,
    model: str,
    batch_size: int,
    freq: str,
) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    missing = {time_column, text_column}.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns in {input_path}: {', '.join(sorted(missing))}")
    frame["time"] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time"])
    texts = frame[text_column].astype(str).str.strip()
    frame = frame[texts.str.len() > 0]
    texts = texts[texts.str.len() > 0].tolist()
    if not texts:
        raise SystemExit(f"No usable text rows in {input_path}")

    classifier = _load_pipeline(model, batch_size)
    frame["polarity"] = [polarity(item["label"], float(item["score"])) for item in classifier(texts)]
    return aggregate(frame[["time", "polarity"]], freq=freq)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a news/social corpus into daily sentiment features.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--symbol", default="", help="e.g. btcusdt; empty writes the shared news_sentiment.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--model", default="finbert", choices=[*MODELS, "custom"])
    parser.add_argument("--model-name", default="", help="HuggingFace id when --model custom")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freq", default="1D")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    model = args.model_name if args.model == "custom" else args.model
    if args.model == "custom" and not model:
        raise SystemExit("--model custom requires --model-name")
    scored = score_corpus(args.input, args.time_column, args.text_column, model, args.batch_size, args.freq)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.symbol.lower()}" if args.symbol else ""
    output_path = args.output_dir / f"news_sentiment{suffix}.csv"
    scored.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{len(scored)} rows -> {output_path}")


if __name__ == "__main__":
    main()
