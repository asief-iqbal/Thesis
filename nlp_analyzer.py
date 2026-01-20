import zlib
from dataclasses import dataclass
from typing import List
import regex

_RE_CODE = regex.compile(r"(?i)\b(?:def|class)\b")
_RE_MATH = regex.compile(r"(?:\\frac|\\sum|\\int|∑|∫|√|≤|≥|≈|≠)")
_RE_QUESTION = regex.compile(r"(?i)(?:\?|\b(?:who|what|where|when|why|how)\b)")


@dataclass
class PromptComplexityMetrics:
    token_count: int
    verb_count: int
    interrogative_count: int
    complexity_score: float


class NlpPromptAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _compression_ratio(text: str) -> float:
        if not text:
            return 0.0
        data = text.encode("utf-8", errors="ignore")
        if not data:
            return 0.0
        return len(zlib.compress(data)) / float(len(data))

    def get_feature_vector(self, prompt_text: str) -> List[float]:
        text = prompt_text or ""
        token_norm = len(text) / 512.0
        compression_ratio = self._compression_ratio(text)
        has_code = 1.0 if _RE_CODE.search(text) is not None else 0.0
        has_math = 1.0 if _RE_MATH.search(text) is not None else 0.0
        has_question = 1.0 if _RE_QUESTION.search(text) is not None else 0.0
        return [token_norm, compression_ratio, has_code, has_math, has_question]

    def analyze_complexity(self, prompt_text: str) -> PromptComplexityMetrics:
        features = self.get_feature_vector(prompt_text)
        token_count = int(round(features[0] * 512.0))
        interrogative_count = int(features[4])
        complexity_score = float(sum(features) / max(1, len(features)))
        print(
            f"[NLP Analyzer] Tokens={token_count}, Verbs=0, Questions={interrogative_count} -> Complexity: {complexity_score:.3f}"
        )
        return PromptComplexityMetrics(
            token_count=token_count,
            verb_count=0,
            interrogative_count=interrogative_count,
            complexity_score=complexity_score,
        )
