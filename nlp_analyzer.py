import os
os.environ['NLTK_DATA'] = 'd:\\LLM Research\\nltk_data'
os.environ['SPACY_DATA'] = 'd:\\LLM Research\\spacy_data'

# Fallback NLTK imports
import nltk
from dataclasses import dataclass
from typing import List, Optional

# Try to use spaCy for richer linguistic features
try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False


@dataclass
class PromptComplexityMetrics:
    """Data structure to hold the linguistic analysis of the prompt."""
    token_count: int         # Total number of LLM tokens.
    verb_count: int          # Number of verbs (actions, instructions).
    interrogative_count: int # Number of question words (who, what, why, etc.).
    complexity_score: float  # Combined score used by the RL agent.


class NlpPromptAnalyzer:
    """
    Analyze prompt complexity using a hybrid approach:
    - Prefer spaCy (if available) for POS, noun chunks, sentence segmentation, dependencies.
    - Fallback to NLTK POS tagging if spaCy is unavailable.

    Provides:
    - analyze_complexity(): human-readable metrics + scalar complexity score.
    - get_feature_vector(): normalized features vector for the RL state.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.spacy_nlp: Optional[object] = None
        if _SPACY_AVAILABLE:
            try:
                # Use small English model. If not present, user should install: python -m spacy download en_core_web_sm
                self.spacy_nlp = spacy.load("en_core_web_sm")  # type: ignore
            except Exception:
                self.spacy_nlp = None

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    def get_feature_vector(self, prompt_text: str) -> List[float]:
        """
        Return a normalized feature vector capturing prompt complexity signals.
        Features (in order):
        - llm_token_norm: LLM token count / 200 (clipped to 1.0)
        - type_token_ratio: unique_word_count / word_count
        - question_norm: interrogative count / 2
        - verb_norm: verbs / 5
        - avg_sentence_len_norm: avg sent length / 30
        - noun_chunks_norm: noun chunks / 20
        - dep_span_norm: max dependency span (token-head distance) / 20
        - complexity_score: weighted sum of key normalized features
        """
        # Basic tokenization using NLTK
        words = nltk.word_tokenize(prompt_text)
        pos_tags = nltk.pos_tag(words) if words else []
        verb_count = sum(1 for _, tag in pos_tags if tag.startswith('V'))
        interrogative_count = sum(1 for _, tag in pos_tags if tag in ['WDT', 'WP', 'WP$', 'WRB']) + prompt_text.count('?')

        # LLM tokenizer count (more relevant than whitespace tokens)
        llm_tokens = len(self.tokenizer.encode(prompt_text))

        # spaCy-derived features (if available)
        noun_chunks = 0
        avg_sent_len = 0.0
        dep_span = 0
        if self.spacy_nlp is not None:
            try:
                doc = self.spacy_nlp(prompt_text)  # type: ignore
                sents = list(doc.sents)
                if sents:
                    avg_sent_len = sum(len(s) for s in sents) / max(1, len(sents))
                noun_chunks = sum(1 for _ in doc.noun_chunks)
                # Approximate tree depth with max token-head distance
                dep_span = max((abs(t.i - t.head.i) for t in doc), default=0)
            except Exception:
                pass

        # Lexical diversity
        ttr = self._safe_div(len(set([w.lower() for w in words])), max(1, len(words)))

        # Normalizations
        llm_norm = min(1.0, llm_tokens / 200.0)
        q_norm = min(1.0, interrogative_count / 2.0)
        v_norm = min(1.0, verb_count / 5.0)
        sent_norm = min(1.0, avg_sent_len / 30.0)
        noun_norm = min(1.0, noun_chunks / 20.0)
        dep_norm = min(1.0, dep_span / 20.0)

        # Weighted complexity (can be tuned)
        complexity_score = 0.4*llm_norm + 0.05*q_norm + 0.20*v_norm + 0.10*sent_norm + 0.25*noun_norm + 0.10*dep_norm

        # Feature vector used in RL state
        features = [
            llm_norm,
            ttr,
            q_norm,
            v_norm,
            sent_norm,
            noun_norm,
            dep_norm,
            complexity_score,
        ]
        return features

    def analyze_complexity(self, prompt_text: str) -> PromptComplexityMetrics:
        features = self.get_feature_vector(prompt_text)
        # Recover raw counts for logging
        words = nltk.word_tokenize(prompt_text)
        pos_tags = nltk.pos_tag(words) if words else []
        verb_count = sum(1 for _, tag in pos_tags if tag.startswith('V'))
        interrogative_count = sum(1 for _, tag in pos_tags if tag in ['WDT', 'WP', 'WP$', 'WRB']) + prompt_text.count('?')
        token_count = len(self.tokenizer.encode(prompt_text))
        complexity_score = features[-1]
        print(f"[NLP Analyzer] Tokens={token_count}, Verbs={verb_count}, Questions={interrogative_count} -> Complexity: {complexity_score:.3f}")
        return PromptComplexityMetrics(
            token_count=token_count,
            verb_count=verb_count,
            interrogative_count=interrogative_count,
            complexity_score=complexity_score,
        )
