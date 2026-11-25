"""Core German pronunciation scoring engine using wav2vec2 phoneme recognition + CTC forced alignment + GOP."""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

from german_aussprachen.aligner import align_free_phonemes, forced_align_phonemes
from german_aussprachen.audio import TARGET_SAMPLE_RATE, audio_rms_db, load_audio_bytes, load_audio_file
from german_aussprachen.phonetics import (
    DISPLAY_IPA_MAP,
    PhonemeSpec,
    WordSpec,
    format_ipa_list,
    get_diagnosis_and_tip,
    phonemize_german_text,
)

DEFAULT_MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

# Common unstressed German function words whose long vowels naturally reduce in fluent speech
UNSTRESSED_FUNCTION_WORDS: set[str] = {
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
    "wir", "ihr", "sie", "er", "es", "du",
    "mir", "dir", "ihm", "ihn", "uns",
    "für", "vor", "über", "aber", "oder", "und", "zu", "so", "da", "wie", "wo",
}

RHOTIC_TOKENS: set[str] = {"ɾ", "r", "ʁ", "ɜ", "ɐ"}


@dataclass
class PhonemeAssessment:
    index: int
    word_index: int
    word: str
    expected_ipa: str
    model_token: str
    heard_ipa: str | None
    score: float  # 0-100
    status: str  # "good" | "fair" | "error" | "omitted"
    logit_gop: float
    posterior_gop: float
    confidence: float  # max non-blank probability in segment
    start_s: float
    end_s: float
    duration_ms: float
    is_vowel: bool
    is_long_vowel: bool
    is_short_vowel: bool
    length_issue: str | None  # None | "too_short" | "too_long"
    free_op: str  # "match" | "sub" | "del"
    diagnosis: str | None
    tip: str | None


@dataclass
class WordAssessment:
    index: int
    word: str
    expected_ipa: str
    score: float
    status: str  # "good" | "fair" | "error" | "omitted"
    start_s: float
    end_s: float
    phonemes: list[PhonemeAssessment]


@dataclass
class IssueSummary:
    word: str
    word_index: int
    expected_ipa: str
    heard_ipa: str | None
    status: str
    score: float
    diagnosis: str
    tip: str | None
    start_s: float
    end_s: float


@dataclass
class PronunciationResult:
    reference_text: str
    audio_duration_s: float
    overall_score: float
    accuracy_score: float
    completeness_score: float
    duration_score: float
    phoneme_error_rate: float
    expected_ipa: str
    heard_ipa: str
    words: list[WordAssessment]
    free_diff: list[dict[str, Any]]
    top_issues: list[IssueSummary]
    model_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _logit_gop_to_score(logit_gop: float) -> float:
    """Map logit-margin GOP (<= 0, where 0 means target/acceptable won) to 0-100.

    Empirically calibrated for wav2vec2-xlsr-53-espeak-cv-ft:
      logit_gop =  0.0  -> 100
      logit_gop = -1.0  -> ~92
      logit_gop = -2.3  -> ~77
      logit_gop = -3.8  -> ~49  (clear substitution -> error)
      logit_gop = -5.5  -> ~22
    """
    margin = max(0.0, -logit_gop)
    score = 100.0 / (1.0 + math.exp(0.68 * (margin - 3.6)))
    max_val = 100.0 / (1.0 + math.exp(0.68 * (0.0 - 3.6)))
    return max(0.0, min(100.0, 100.0 * score / max_val))


class GermanPronunciationScorer:
    """End-to-end German pronunciation scorer (singleton-friendly, thread-safe inference)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_lock = threading.Lock()
        self._processor = None
        self._model = None
        self._vocab: dict[str, int] | None = None
        self._id_to_token: dict[int, str] | None = None
        self._blank_id: int = 0
        self._special_ids: set[int] = {0, 1, 2, 3}

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForCTC.from_pretrained(self.model_id)
            model.to(self.device)
            model.eval()

            vocab = processor.tokenizer.get_vocab()
            id_to_token = {v: k for k, v in vocab.items()}
            blank_id = int(processor.tokenizer.pad_token_id or 0)
            special_ids = {
                vocab[t]
                for t in ("<pad>", "<s>", "</s>", "<unk>")
                if t in vocab
            }
            special_ids.add(blank_id)

            self._processor = processor
            self._model = model
            self._vocab = vocab
            self._id_to_token = id_to_token
            self._blank_id = blank_id
            self._special_ids = special_ids

    @property
    def vocab(self) -> dict[str, int]:
        self.ensure_loaded()
        assert self._vocab is not None
        return self._vocab

    def _run_model(self, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Run wav2vec2 forward pass; returns (logits, log_probs) on CPU with shape (T, V)."""
        self.ensure_loaded()
        assert self._processor is not None and self._model is not None

        inputs = self._processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)
        with torch.inference_mode():
            logits = self._model(input_values).logits[0].float().cpu()
        log_probs = torch.log_softmax(logits, dim=-1)
        return logits, log_probs

    def _greedy_decode(self, logits: torch.Tensor) -> list[str]:
        """Greedy CTC collapse into IPA phoneme tokens (no language model)."""
        assert self._id_to_token is not None
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        tokens: list[str] = []
        prev = None
        for pid in pred_ids:
            if pid != prev and pid not in self._special_ids:
                tok = self._id_to_token.get(pid)
                if tok:
                    tokens.append(tok)
            prev = pid
        return tokens

    def score_waveform(
        self,
        audio: np.ndarray,
        reference_text: str,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> PronunciationResult:
        self.ensure_loaded()
        assert self._vocab is not None and self._id_to_token is not None

        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"Expected {TARGET_SAMPLE_RATE}Hz audio, got {sample_rate}Hz")

        audio_duration_s = float(len(audio) / TARGET_SAMPLE_RATE)
        if audio_duration_s < 0.15:
            raise ValueError("Recording is too short (< 0.15s). Please record the full sentence.")
        if audio_rms_db(audio) < -55.0:
            raise ValueError("Recording appears silent. Check your microphone input level.")

        words, phonemes = phonemize_german_text(reference_text, self._vocab)
        logits, log_probs = self._run_model(audio)
        num_frames = logits.shape[0]
        frame_dur_s = audio_duration_s / max(num_frames, 1)

        # 1. Free greedy CTC phoneme decoding + Needleman-Wunsch alignment
        free_tokens = self._greedy_decode(logits)
        free_ops, free_by_expected, per = align_free_phonemes(phonemes, free_tokens)

        # 2. Equivalence-aware CTC forced alignment
        spans = forced_align_phonemes(log_probs, phonemes, blank_id=self._blank_id)

        # Mask out special tokens when finding competing phonemes
        non_special_mask = torch.ones(logits.shape[1], dtype=torch.bool)
        for sid in self._special_ids:
            if 0 <= sid < logits.shape[1]:
                non_special_mask[sid] = False

        # Compute raw per-phoneme acoustic metrics & durations
        raw_metrics: list[dict[str, Any]] = []
        for idx, (spec, span) in enumerate(zip(phonemes, spans)):
            # Measure acoustic duration: from active_start to next phoneme's active_start (capped)
            # because CTC puts a sharp spike on emission and blanks across sustained vowels!
            if idx + 1 < len(spans):
                inter_gap_frames = spans[idx + 1].active_start_frame - span.active_start_frame
                # Cap at 22 frames (~440ms) to avoid counting inter-word pauses as vowel duration
                eff_frames = max(1, min(inter_gap_frames, 22))
            else:
                eff_frames = max(1, min(span.end_frame - span.active_start_frame, 18))
            duration_ms = float(eff_frames * frame_dur_s * 1000.0)

            seg_logits = logits[span.start_frame : span.end_frame]
            seg_log_probs = log_probs[span.start_frame : span.end_frame]
            # Focus on the phoneme's active emission frames (plus +-1 neighbor) for GOP
            g_start = max(span.start_frame, span.active_start_frame - 1)
            g_end = min(span.end_frame, span.active_end_frame + 1)
            gop_logits = logits[g_start:g_end]
            gop_log_probs = log_probs[g_start:g_end]

            p_nonblank = 1.0 - torch.exp(gop_log_probs[:, self._blank_id])
            max_nb_prob = float(p_nonblank.max().item())

            # Select peak non-blank frames inside the phoneme's emission window
            peak_mask = p_nonblank >= max(0.15, 0.45 * max_nb_prob)
            if not bool(peak_mask.any()):
                top_k = min(2, gop_logits.shape[0])
                top_idx = torch.topk(p_nonblank, k=top_k).indices
                peak_logits = gop_logits[top_idx]
                peak_log_probs = gop_log_probs[top_idx]
                peak_weights = torch.ones(top_k, dtype=torch.float32) / top_k
            else:
                peak_logits = gop_logits[peak_mask]
                peak_log_probs = gop_log_probs[peak_mask]
                w = p_nonblank[peak_mask].clamp(min=1e-4)
                peak_weights = w / w.sum()

            acc_ids = sorted(spec.acceptable_ids) if spec.acceptable_ids else [spec.model_token_id]

            # Best acceptable logit vs best non-special logit per peak frame
            acc_frame_logits = peak_logits[:, acc_ids].max(dim=-1).values
            masked_logits = peak_logits.masked_fill(~non_special_mask.unsqueeze(0), -1e9)
            best_any_logits = masked_logits.max(dim=-1).values
            logit_gop = float(((acc_frame_logits - best_any_logits) * peak_weights).sum().item())

            # Posterior GOP
            acc_frame_lp = torch.logsumexp(peak_log_probs[:, acc_ids], dim=-1)
            masked_lp = peak_log_probs.masked_fill(~non_special_mask.unsqueeze(0), -1e9)
            best_any_lp = masked_lp.max(dim=-1).values
            posterior_gop = float(((acc_frame_lp - best_any_lp) * peak_weights).sum().item())

            # Dominant heard non-blank phoneme in peak frames
            weighted_probs = (torch.exp(peak_log_probs) * peak_weights.unsqueeze(-1)).sum(dim=0)
            weighted_probs[~non_special_mask] = -1.0
            heard_id = int(torch.argmax(weighted_probs).item())
            heard_token = self._id_to_token.get(heard_id)

            free_op_obj = free_by_expected.get(idx)
            free_op = free_op_obj.op if free_op_obj else "del"
            free_pred = free_op_obj.predicted_token if free_op_obj else None

            # If free decoding caught a clear substitution and GOP was weak, prefer its heard token
            if free_op == "sub" and free_pred and heard_token in spec.acceptable_tokens and logit_gop < -1.5:
                heard_token = free_pred

            raw_metrics.append(
                {
                    "spec": spec,
                    "span": span,
                    "duration_ms": duration_ms,
                    "logit_gop": logit_gop,
                    "posterior_gop": posterior_gop,
                    "max_nb_prob": max_nb_prob,
                    "heard_token": heard_token,
                    "free_op": free_op,
                    "free_pred": free_pred,
                }
            )

        # 3. Speaker-adaptive vowel duration calibration (for German long vs short vowel contrasts!)
        short_durs = [
            m["duration_ms"]
            for m in raw_metrics
            if m["spec"].is_short_vowel and m["free_op"] != "del" and m["max_nb_prob"] > 0.25
        ]
        long_durs = [
            m["duration_ms"]
            for m in raw_metrics
            if m["spec"].is_long_vowel and m["free_op"] != "del" and m["max_nb_prob"] > 0.25
        ]
        all_v_durs = short_durs + long_durs
        median_vowel_ms = float(np.median(all_v_durs)) if all_v_durs else 110.0
        # Reference thresholds relative to speaker's own rate
        short_ref_ms = float(np.median(short_durs)) if len(short_durs) >= 2 else min(median_vowel_ms, 105.0)
        long_ref_ms = float(np.median(long_durs)) if len(long_durs) >= 2 else max(median_vowel_ms * 1.35, 145.0)

        # 4. Compute final per-phoneme scores, statuses, and diagnoses
        phoneme_assessments: list[PhonemeAssessment] = []
        duration_checks = 0
        duration_passes = 0

        for idx, m in enumerate(raw_metrics):
            spec: PhonemeSpec = m["spec"]
            span = m["span"]
            logit_gop: float = m["logit_gop"]
            posterior_gop: float = m["posterior_gop"]
            max_nb_prob: float = m["max_nb_prob"]
            heard_token: str | None = m["heard_token"]
            free_op: str = m["free_op"]
            duration_ms: float = m["duration_ms"]

            base_score = _logit_gop_to_score(logit_gop)

            # Detect genuine omission (sound dropped / skipped word)
            is_omitted = False
            if free_op == "del":
                if max_nb_prob < 0.25 or logit_gop < -4.0:
                    is_omitted = True
                elif max_nb_prob < 0.45 and logit_gop < -2.2:
                    is_omitted = True
                else:
                    # Mild penalty when free decoding missed it but forced alignment still saw acoustic evidence
                    base_score = min(base_score, max(base_score - 12.0, 45.0))
            elif max_nb_prob < 0.12:
                is_omitted = True

            # German vowel length check (only on stressed content words, not pre-rhotic reductions)
            length_issue: str | None = None
            clean_word_lower = words[spec.word_index].clean_text.lower()
            next_is_rhotic = (
                idx + 1 < len(phonemes)
                and phonemes[idx + 1].word_index == spec.word_index
                and phonemes[idx + 1].model_token in RHOTIC_TOKENS
            )
            check_length = (
                not is_omitted
                and (spec.is_long_vowel or spec.is_short_vowel)
                and clean_word_lower not in UNSTRESSED_FUNCTION_WORDS
                and not (spec.is_long_vowel and next_is_rhotic)
            )
            if check_length:
                duration_checks += 1
                if spec.is_long_vowel and duration_ms < max(72.0, min(98.0, short_ref_ms * 0.95)):
                    length_issue = "too_short"
                    base_score = min(base_score, 54.0)
                elif spec.is_short_vowel and duration_ms > max(168.0, long_ref_ms * 1.15):
                    length_issue = "too_long"
                    base_score = min(base_score, 58.0)
                else:
                    duration_passes += 1

            if is_omitted:
                final_score = min(base_score * 0.25, 18.0)
                status = "omitted"
                heard_disp = None
            else:
                final_score = base_score
                if final_score >= 76.0 and length_issue is None:
                    status = "good"
                elif final_score >= 52.0:
                    status = "fair"
                else:
                    status = "error"
                heard_disp = (
                    DISPLAY_IPA_MAP.get(heard_token, heard_token)
                    if heard_token
                    else None
                )

            diagnosis, tip = get_diagnosis_and_tip(
                spec=spec,
                heard_token=heard_token,
                status=status,
                length_issue=length_issue,
            )

            phoneme_assessments.append(
                PhonemeAssessment(
                    index=idx,
                    word_index=spec.word_index,
                    word=spec.word,
                    expected_ipa=spec.display_ipa,
                    model_token=spec.model_token,
                    heard_ipa=heard_disp,
                    score=round(final_score, 1),
                    status=status,
                    logit_gop=round(logit_gop, 3),
                    posterior_gop=round(posterior_gop, 3),
                    confidence=round(max_nb_prob, 3),
                    start_s=round(span.start_frame * frame_dur_s, 3),
                    end_s=round(span.end_frame * frame_dur_s, 3),
                    duration_ms=round(duration_ms, 1),
                    is_vowel=spec.is_vowel,
                    is_long_vowel=spec.is_long_vowel,
                    is_short_vowel=spec.is_short_vowel,
                    length_issue=length_issue,
                    free_op=free_op,
                    diagnosis=diagnosis,
                    tip=tip,
                )
            )

        # 5. Roll up to word-level assessments
        word_assessments: list[WordAssessment] = []
        for w in words:
            w_ph = [phoneme_assessments[i] for i in w.phoneme_indices]
            if not w_ph:
                continue
            omitted_ratio = sum(1 for p in w_ph if p.status == "omitted") / len(w_ph)
            # Emphasize distinctive German sounds (vowels, ch, r, pf, ts, sch) slightly in word score
            weights = [
                1.35 if (p.is_vowel or p.expected_ipa in {"ç", "x", "pf", "ts", "ʃ"}) else 1.0
                for p in w_ph
            ]
            w_score = sum(p.score * wt for p, wt in zip(w_ph, weights)) / sum(weights)
            # Also pull word score down slightly if there is a glaring single-phoneme error
            min_p = min(p.score for p in w_ph)
            if min_p < 45.0:
                w_score = 0.72 * w_score + 0.28 * min_p

            if omitted_ratio >= 0.6:
                w_status = "omitted"
                w_score = min(w_score, 20.0)
            elif w_score >= 78.0 and min_p >= 52.0:
                w_status = "good"
            elif w_score >= 56.0:
                w_status = "fair"
            else:
                w_status = "error"

            word_assessments.append(
                WordAssessment(
                    index=w.index,
                    word=w.text,
                    expected_ipa=w.display_ipa,
                    score=round(w_score, 1),
                    status=w_status,
                    start_s=w_ph[0].start_s,
                    end_s=w_ph[-1].end_s,
                    phonemes=w_ph,
                )
            )

        # 6. Roll up to sentence-level scores
        non_omitted = [p for p in phoneme_assessments if p.status != "omitted"]
        completeness_score = 100.0 * len(non_omitted) / max(len(phoneme_assessments), 1)
        accuracy_score = (
            sum(p.score for p in phoneme_assessments) / max(len(phoneme_assessments), 1)
        )
        duration_score = (
            100.0 * duration_passes / duration_checks if duration_checks > 0 else 100.0
        )

        # Combine accuracy, completeness, and vowel duration
        # Also apply a clear penalty for explicit phoneme errors / length errors / large insertions
        # so 2 glaring errors in a short sentence clearly drop the score below 90.
        ins_count = sum(1 for op in free_ops if op.op == "ins")
        ins_penalty = max(0.0, (ins_count / max(len(phonemes), 1) - 0.15) * 35.0)
        error_count = sum(1 for p in phoneme_assessments if p.status == "error")
        length_err_count = sum(1 for p in phoneme_assessments if p.length_issue is not None)
        salience_penalty = min(24.0, error_count * 4.0 + length_err_count * 3.0)
        overall_score = max(
            0.0,
            min(
                100.0,
                0.70 * accuracy_score
                + 0.20 * completeness_score
                + 0.10 * duration_score
                - ins_penalty
                - salience_penalty,
            ),
        )

        # 7. Prioritize top actionable learner issues
        issue_candidates = [
            p for p in phoneme_assessments if p.status in ("error", "omitted", "fair") and p.diagnosis
        ]
        # Sort errors & omissions first, then lowest score
        priority_rank = {"error": 0, "omitted": 1, "fair": 2}
        issue_candidates.sort(key=lambda p: (priority_rank.get(p.status, 3), p.score))
        top_issues = [
            IssueSummary(
                word=p.word,
                word_index=p.word_index,
                expected_ipa=p.expected_ipa,
                heard_ipa=p.heard_ipa,
                status=p.status,
                score=p.score,
                diagnosis=p.diagnosis or "",
                tip=p.tip,
                start_s=p.start_s,
                end_s=p.end_s,
            )
            for p in issue_candidates[:8]
        ]

        expected_ipa_str = " | ".join(w.expected_ipa for w in word_assessments)
        heard_ipa_str = format_ipa_list(free_tokens)
        free_diff_dicts = [
            {
                "op": op.op,
                "expected": op.expected_token,
                "predicted": DISPLAY_IPA_MAP.get(op.predicted_token, op.predicted_token)
                if op.predicted_token
                else None,
            }
            for op in free_ops
        ]

        return PronunciationResult(
            reference_text=reference_text.strip(),
            audio_duration_s=round(audio_duration_s, 3),
            overall_score=round(overall_score, 1),
            accuracy_score=round(accuracy_score, 1),
            completeness_score=round(completeness_score, 1),
            duration_score=round(duration_score, 1),
            phoneme_error_rate=round(per, 3),
            expected_ipa=expected_ipa_str,
            heard_ipa=heard_ipa_str,
            words=word_assessments,
            free_diff=free_diff_dicts,
            top_issues=top_issues,
            model_id=self.model_id,
        )

    def score_file(self, audio_path: str | Path, reference_text: str) -> PronunciationResult:
        audio, sr = load_audio_file(audio_path)
        return self.score_waveform(audio, reference_text, sample_rate=sr)

    def score_bytes(
        self, data: bytes, reference_text: str, filename: str = "audio.wav"
    ) -> PronunciationResult:
        audio, sr = load_audio_bytes(data, filename=filename)
        return self.score_waveform(audio, reference_text, sample_rate=sr)


_DEFAULT_SCORER: GermanPronunciationScorer | None = None
_SCORER_LOCK = threading.Lock()


def get_default_scorer() -> GermanPronunciationScorer:
    global _DEFAULT_SCORER
    with _SCORER_LOCK:
        if _DEFAULT_SCORER is None:
            _DEFAULT_SCORER = GermanPronunciationScorer()
        return _DEFAULT_SCORER
