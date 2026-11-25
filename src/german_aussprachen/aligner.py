"""CTC forced alignment and Needleman-Wunsch free-phoneme diff alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio

from german_aussprachen.phonetics import PhonemeSpec


@dataclass(frozen=True)
class AlignedSpan:
    """Frame-level alignment span for a single expected phoneme."""

    phoneme_index: int
    start_frame: int  # inclusive
    end_frame: int  # exclusive (at least start_frame + 1)
    active_start_frame: int  # first frame where CTC emitted the target token
    active_end_frame: int  # exclusive last frame where CTC emitted the target token
    align_score: float


@dataclass(frozen=True)
class FreeDiffOp:
    """Single alignment operation between expected phonemes and free CTC decoded phonemes."""

    op: str  # "match" | "sub" | "del" | "ins"
    expected_index: int | None
    expected_token: str | None
    predicted_token: str | None


def forced_align_phonemes(
    log_probs: torch.Tensor,
    phonemes: list[PhonemeSpec],
    blank_id: int = 0,
) -> list[AlignedSpan]:
    """Run equivalence-aware CTC forced alignment using torchaudio.functional.forced_align.

    Args:
        log_probs: Tensor of shape (T, V) with log-probabilities from the CTC model.
        phonemes: Expected PhonemeSpec list of length N.
        blank_id: Blank token ID (0 for wav2vec2).

    Returns:
        List of AlignedSpan, one per expected phoneme in exact order.
    """
    num_frames = log_probs.shape[0]
    num_targets = len(phonemes)
    if num_targets == 0:
        return []
    if num_frames < num_targets:
        raise ValueError(
            f"Audio too short ({num_frames} frames) for {num_targets} expected phonemes."
        )

    # Build a compact emission matrix of shape (1, T, 1 + num_targets):
    #   column 0: blank log_prob
    #   column i+1: logsumexp over phonemes[i].acceptable_ids (so multilingual IPA variants
    #               like ɔø/oɪ or ɑː/a align cleanly to the actual acoustic segment!)
    # Using unique columns per position also guarantees adjacent identical phonemes
    # (e.g., word-boundary /t/ + /t/) map unambiguously to their exact target index!
    align_emissions = torch.empty((1, num_frames, 1 + num_targets), dtype=torch.float32)
    align_emissions[0, :, 0] = log_probs[:, blank_id]

    for idx, spec in enumerate(phonemes):
        ids = sorted(spec.acceptable_ids) if spec.acceptable_ids else [spec.model_token_id]
        if len(ids) == 1:
            align_emissions[0, :, idx + 1] = log_probs[:, ids[0]]
        else:
            align_emissions[0, :, idx + 1] = torch.logsumexp(log_probs[:, ids], dim=-1)

    targets = torch.arange(1, num_targets + 1, dtype=torch.int32).unsqueeze(0)
    input_lengths = torch.tensor([num_frames], dtype=torch.int32)
    target_lengths = torch.tensor([num_targets], dtype=torch.int32)

    aligned_tokens, frame_scores = torchaudio.functional.forced_align(
        align_emissions,
        targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=0,
    )
    path = aligned_tokens[0].tolist()
    scores = frame_scores[0].tolist()

    # Extract exact active frame ranges for each target token 1..num_targets
    active_ranges: dict[int, list[int]] = {i: [] for i in range(1, num_targets + 1)}
    for t, tok in enumerate(path):
        if tok > 0:
            active_ranges[tok].append(t)

    # Build active spans first
    raw_spans: list[tuple[int, int, float]] = []
    last_frame = 0
    for i in range(1, num_targets + 1):
        frames = active_ranges[i]
        if frames:
            a_start = frames[0]
            a_end = frames[-1] + 1
            a_score = sum(scores[f] for f in frames) / len(frames)
            last_frame = a_end
        else:
            a_start = min(last_frame, num_frames - 1)
            a_end = min(a_start + 1, num_frames)
            a_score = -20.0
        raw_spans.append((a_start, a_end, a_score))

    # Expand each phoneme's window into surrounding blank frames up to the midpoint
    # between neighboring active phonemes, giving clean audio playback & duration estimates.
    spans: list[AlignedSpan] = []
    for idx, (a_start, a_end, a_score) in enumerate(raw_spans):
        if idx == 0:
            w_start = max(0, a_start - 2)
        else:
            prev_end = raw_spans[idx - 1][1]
            # Limit gap absorption so long pauses don't inflate a phoneme's duration
            gap = a_start - prev_end
            w_start = prev_end + min(gap // 2, 4) if gap > 0 else a_start

        if idx == num_targets - 1:
            w_end = min(num_frames, a_end + 3)
        else:
            next_start = raw_spans[idx + 1][0]
            gap = next_start - a_end
            w_end = a_end + min((gap + 1) // 2, 6) if gap > 0 else next_start

        w_start = max(0, min(w_start, a_start))
        w_end = max(a_end, min(num_frames, w_end))

        spans.append(
            AlignedSpan(
                phoneme_index=idx,
                start_frame=w_start,
                end_frame=w_end,
                active_start_frame=a_start,
                active_end_frame=a_end,
                align_score=a_score,
            )
        )

    return spans


def align_free_phonemes(
    expected: list[PhonemeSpec],
    predicted: list[str],
) -> tuple[list[FreeDiffOp], dict[int, FreeDiffOp], float]:
    """Align expected phonemes against freely decoded greedy CTC phonemes via Needleman-Wunsch.

    Returns:
        (ops, expected_index_to_op, phoneme_error_rate)
    """
    n = len(expected)
    m = len(predicted)

    # DP table of edit costs
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = float(i)
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = float(j)
        bt[0][j] = "ins"

    for i in range(1, n + 1):
        spec = expected[i - 1]
        for j in range(1, m + 1):
            pred_tok = predicted[j - 1]
            is_match = pred_tok in spec.acceptable_tokens
            sub_cost = 0.0 if is_match else 1.0

            c_diag = dp[i - 1][j - 1] + sub_cost
            c_del = dp[i - 1][j] + 1.0
            c_ins = dp[i][j - 1] + 1.0

            best = min(c_diag, c_del, c_ins)
            dp[i][j] = best
            if best == c_diag:
                bt[i][j] = "match" if is_match else "sub"
            elif best == c_del:
                bt[i][j] = "del"
            else:
                bt[i][j] = "ins"

    # Backtrack
    ops: list[FreeDiffOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i][j]
        if move in ("match", "sub"):
            ops.append(
                FreeDiffOp(
                    op=move,
                    expected_index=i - 1,
                    expected_token=expected[i - 1].display_ipa,
                    predicted_token=predicted[j - 1],
                )
            )
            i -= 1
            j -= 1
        elif move == "del":
            ops.append(
                FreeDiffOp(
                    op="del",
                    expected_index=i - 1,
                    expected_token=expected[i - 1].display_ipa,
                    predicted_token=None,
                )
            )
            i -= 1
        else:
            ops.append(
                FreeDiffOp(
                    op="ins",
                    expected_index=None,
                    expected_token=None,
                    predicted_token=predicted[j - 1],
                )
            )
            j -= 1

    ops.reverse()
    by_expected: dict[int, FreeDiffOp] = {
        op.expected_index: op for op in ops if op.expected_index is not None
    }
    per = dp[n][m] / max(n, 1)
    return ops, by_expected, per
