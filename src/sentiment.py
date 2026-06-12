"""FinBERT sentiment scoring for earnings call transcripts.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text.
It classifies text into positive / negative / neutral sentiment.

BERT models have a hard limit of 512 tokens per input. Earnings call
transcripts are far longer, so we split the token sequence into chunks,
score each chunk independently, and combine the results with a
length-weighted average.

Chunks are scored in batches: shorter chunks are padded to the batch's
longest length and an attention mask tells the model to ignore padding.
On CPU this is several times faster than one chunk at a time.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"

LABELS = ("positive", "negative", "neutral")

# 512 minus the two special tokens ([CLS] at the start, [SEP] at the end)
# that BERT requires around every input.
MAX_CHUNK_TOKENS = 510

BATCH_SIZE = 8


def aggregate(chunk_scores: list[dict]) -> dict[str, float]:
    """Length-weighted average of per-chunk probabilities.

    Longer chunks influence the final score proportionally more, so the
    result matches scoring the whole text in one (hypothetical) pass.
    """
    total = sum(c["tokens"] for c in chunk_scores)
    return {
        label: sum(c[label] * c["tokens"] for c in chunk_scores) / total
        for label in LABELS
    }


class FinbertScorer:
    """Loads FinBERT once and scores arbitrarily long texts."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()  # inference mode: disables dropout etc.

    def _chunks(self, text: str) -> list[list[int]]:
        """Tokenize the full text, then slice into pieces of <= 510 tokens."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [
            token_ids[start : start + MAX_CHUNK_TOKENS]
            for start in range(0, len(token_ids), MAX_CHUNK_TOKENS)
        ]

    def score_chunks(self, text: str, on_progress=None) -> list[dict]:
        """Score every <=510-token chunk of `text` separately.

        Returns one dict per chunk, in document order:
            {'positive': p, 'negative': n, 'neutral': u,
             'tokens': <chunk length>, 'text': <decoded chunk text>}

        Keeping per-chunk scores (instead of only the aggregate) lets a UI
        plot how tone evolves through the call and surface the most
        positive / most negative passages.

        `on_progress(done_chunks, total_chunks)` is called after every
        batch so UIs can show real progress on long transcripts.
        """
        chunks = self._chunks(text)
        if not chunks:
            raise ValueError("Text is empty after tokenization.")

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        results = []
        for b in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[b : b + BATCH_SIZE]
            max_len = max(len(c) for c in batch) + 2  # + [CLS] and [SEP]

            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for row, chunk in enumerate(batch):
                seq = [cls_id, *chunk, sep_id]
                input_ids[row, : len(seq)] = torch.tensor(seq)
                attention_mask[row, : len(seq)] = 1

            with torch.no_grad():  # no gradients needed for inference
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
            probs = torch.softmax(logits, dim=-1)

            for row, chunk in enumerate(batch):
                chunk_result = {
                    self.model.config.id2label[i]: float(probs[row][i])
                    for i in range(self.model.config.num_labels)
                }
                chunk_result["tokens"] = len(chunk)
                chunk_result["text"] = self.tokenizer.decode(chunk)
                results.append(chunk_result)

            if on_progress is not None:
                on_progress(min(b + BATCH_SIZE, len(chunks)), len(chunks))

        return results

    def score(self, text: str, on_progress=None) -> dict[str, float]:
        """Return {'positive': p, 'negative': n, 'neutral': u} summing to 1.

        Each chunk's probabilities are weighted by its token count, so
        longer passages influence the final score proportionally more.
        """
        return aggregate(self.score_chunks(text, on_progress=on_progress))


if __name__ == "__main__":
    sample = (
        "We delivered record revenue this quarter, exceeding guidance across "
        "all segments. Margins expanded significantly and we are raising our "
        "full-year outlook. However, we remain cautious about supply chain "
        "headwinds in the second half."
    )
    scorer = FinbertScorer()
    result = scorer.score(sample)
    for label, prob in sorted(result.items(), key=lambda x: -x[1]):
        print(f"{label:>9}: {prob:.1%}")
