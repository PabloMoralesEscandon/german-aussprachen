"""German G2P (espeak-ng), vocabulary normalization, allophone sets, and coaching tips."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Iterable

import espeakng_loader
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.separator import Separator

_ESPEAK_LOCK = threading.Lock()
_ESPEAK_BACKEND: EspeakBackend | None = None

# Known espeak-ng German quirks (lowercase word -> canonical espeak IPA phone list)
GERMAN_G2P_OVERRIDES: dict[str, list[str]] = {
    "vögel": ["f", "øː", "ɡ", "ə", "l"],
    "vögeln": ["f", "øː", "ɡ", "ə", "l", "n"],
}

# Mapping from espeak-ng German phones that are OOV in facebook/wav2vec2-xlsr-53-espeak-cv-ft
# to the closest in-vocabulary model token.
ESPEAK_TO_MODEL_TOKEN: dict[str, str] = {
    "ʏ": "y",
    "ɔʏ": "ɔø",
    "aː": "ɑː",
    "ɡ": "ɡ",
    "g": "ɡ",
    "ɐ": "ɜ",
    "ʀ": "ʁ",
    "χ": "x",
}

# Nicer standard German IPA display symbols for learners
DISPLAY_IPA_MAP: dict[str, str] = {
    "ɑː": "aː",
    "ɔø": "ɔʏ",
    "ɜ": "ɐ",
    "ɾ": "ʁ",
}

# German long vs short vowel categories for duration checking
LONG_VOWELS: set[str] = {"aː", "ɑː", "eː", "ɛː", "iː", "oː", "uː", "yː", "øː"}
SHORT_VOWELS: set[str] = {"a", "ɛ", "ɪ", "ɔ", "ʊ", "ʏ", "œ"}
DIPHTHONGS: set[str] = {"aɪ", "aʊ", "ɔø", "ɔʏ", "ɔɪ", "oɪ"}
ALL_VOWELS: set[str] = LONG_VOWELS | SHORT_VOWELS | DIPHTHONGS | {"ə", "ɜ", "ɐ", "y", "ø", "e", "i", "o", "u"}

# Acceptable allophone / multilingual transcription equivalents in wav2vec2-xlsr-53-espeak-cv-ft.
# IMPORTANT: Only include valid German realizations or cross-language IPA notation variants —
# NEVER include common learner mistakes (e.g. /ʃ/ for /ç/, /u/ for /y/, /k/ for /x/)!
RAW_ACCEPTABLE_EQUIVALENTS: dict[str, set[str]] = {
    # Long vowels (quality accepted without length mark; length is checked via duration)
    "ɑː": {"ɑː", "aː", "a", "ɑ"},
    "aː": {"aː", "ɑː", "a", "ɑ"},
    "eː": {"eː", "e", "ɛː"},
    "ɛː": {"ɛː", "ɛ", "eː", "e"},
    "iː": {"iː", "i"},
    "oː": {"oː", "o"},
    "uː": {"uː", "u"},
    "yː": {"yː", "y"},
    "øː": {"øː", "ø", "œ", "ɜː"},
    # Short vowels
    "a": {"a", "ɑ", "ɑː", "aː"},
    "ɛ": {"ɛ", "e", "ɛː"},
    "ɪ": {"ɪ", "i", "iː"},
    "ɔ": {"ɔ", "o", "ɔː"},
    "ʊ": {"ʊ", "u", "uː"},
    "y": {"y", "yː"},  # short ü (ʏ mapped to y) — never accepts u/ʊ/i/ɪ!
    "œ": {"œ", "ø", "øː", "ɜ"},  # short ö — never accepts o/ɔ! (will verify ɛ empirically)
    "ə": {"ə", "ɜ", "ɐ", "ɛ", "e"},
    "ɜ": {"ɜ", "ɐ", "ə", "ɚ", "ɾ", "r", "ʁ"},
    # Diphthongs
    "aɪ": {"aɪ", "ai", "aɪə"},
    "aʊ": {"aʊ", "au"},
    "ɔø": {"ɔø", "ɔɪ", "oɪ", "øi"},
    # German R variants (uvular, alveolar tap/trill, vocalized)
    "ɾ": {"ɾ", "r", "ʁ", "ɐ", "ɜ"},
    "r": {"r", "ɾ", "ʁ", "ɐ", "ɜ"},
    "ʁ": {"ʁ", "ɾ", "r", "ɐ", "ɜ"},
    # Strict German fricatives & affricates
    "ç": {"ç", "ɕ"},  # ich-Laut: NEVER accepts ʃ, x, k, s!
    "x": {"x", "χ"},  # ach-Laut: NEVER accepts k, h, ç, ʃ!
    "ʃ": {"ʃ", "ʂ"},
    "tʃ": {"tʃ", "tɕ", "tS"},
    "ts": {"ts", "ts.", "tsh"},
    "pf": {"pf", "f"},  # Northern/Central German routinely weakens initial /pf/ -> [f]
    # Aspirated stop variants
    "p": {"p", "pʰ", "ph", "pː"},
    "t": {"t", "tʰ", "th", "t̪", "tː"},
    "k": {"k", "kʰ", "kh", "kː"},
    "b": {"b", "bː"},
    "d": {"d", "dː"},
    "ɡ": {"ɡ", "ɡː"},
    "f": {"f"},
    "v": {"v", "ʋ"},
    "s": {"s", "s̪"},
    "z": {"z", "s"},  # Southern German / Austrian speakers often devoice initial /z/
    "h": {"h"},
    "j": {"j", "ʝ"},
    "l": {"l", "ɫ", "lː", "əl"},
    "m": {"m"},
    "n": {"n", "n̩"},
    "ŋ": {"ŋ"},
}

# Specific (expected_display_or_model, heard_token) learner confusion diagnoses
CONFUSION_DIAGNOSES: dict[tuple[str, str], str] = {
    ("ç", "ʃ"): (
        "Sounded like /ʃ/ ('sh') instead of the soft 'ich-Laut' /ç/. "
        "Keep your tongue tip down behind your lower teeth and raise the middle of your tongue "
        "toward the hard palate (like the 'h' in 'huge')."
    ),
    ("ç", "k"): (
        "Sounded like /k/ instead of /ç/. Keep air flowing continuously through a narrow gap "
        "at the hard palate without stopping the airflow."
    ),
    ("ç", "x"): (
        "Sounded like the back 'ach-Laut' /x/ instead of the front 'ich-Laut' /ç/. "
        "After front vowels (i, e, ä, ö, ü) or consonants, pronounce 'ch' further forward."
    ),
    ("ç", "s"): (
        "Sounded like /s/ instead of /ç/. Pull the tongue slightly back from the teeth ridge "
        "and raise the middle of the tongue."
    ),
    ("ç", "tʃ"): (
        "Sounded like /tʃ/ ('ch' in 'church') instead of the soft 'ich-Laut' /ç/. "
        "Remove the initial /t/ stop and let air flow smoothly over the middle of the tongue."
    ),
    ("x", "k"): (
        "Sounded like a hard /k/ instead of the 'ach-Laut' /x/. Let air scrape smoothly "
        "against the soft palate/uvula instead of blocking it completely."
    ),
    ("x", "h"): (
        "Sounded too breathy (/h/) for the 'ach-Laut' /x/. Narrow the back of the tongue "
        "against the soft palate for audible friction."
    ),
    ("x", "ç"): (
        "Sounded like the front 'ich-Laut' /ç/ instead of the back 'ach-Laut' /x/. "
        "After a, o, u, au, pronounce 'ch' further back in the throat."
    ),
    ("ʏ", "u"): (
        "Sounded like /u/ ('oo') instead of German short 'ü' /ʏ/. "
        "Say 'ee' (/i/) with your tongue forward, then round your lips tightly without pulling your tongue back."
    ),
    ("ʏ", "ʊ"): (
        "Sounded like /ʊ/ (as in 'put') instead of German short 'ü' /ʏ/. "
        "Push your tongue forward toward 'ih' (/ɪ/) while keeping your lips rounded."
    ),
    ("ʏ", "ʌ"): (
        "Sounded like English /ʌ/ (as in 'hut') instead of German short 'ü' /ʏ/. "
        "Raise your tongue forward toward /ɪ/ and round your lips tightly."
    ),
    ("iː", "aɪ"): (
        "German 'ie' is always a long /iː/ ('ee' as in 'meet'), never English /aɪ/ ('my')."
    ),
    ("ʏ", "uː"): (
        "Sounded like /uː/ ('oo') instead of German short 'ü' /ʏ/. "
        "Move your tongue forward as for /i/ while keeping tightly rounded lips, and keep the vowel short."
    ),
    ("ʏ", "i"): (
        "Sounded like /i/ ('ee') — your tongue is forward, now round your lips firmly for 'ü' /ʏ/."
    ),
    ("ʏ", "ɪ"): (
        "Sounded like /ɪ/ ('ih') — keep that tongue position, but round your lips tightly for 'ü' /ʏ/."
    ),
    ("yː", "u"): (
        "Sounded like /u/ ('oo') instead of German long 'ü' /yː/. "
        "Hold an 'ee' (/iː/) tongue position and round your lips tightly."
    ),
    ("yː", "uː"): (
        "Sounded like /uː/ ('oo') instead of German long 'ü' /yː/. "
        "Keep your tongue forward like /iː/ and round your lips."
    ),
    ("yː", "ʊ"): (
        "Sounded like /ʊ/ instead of German long 'ü' /yː/. "
        "Push your tongue forward like /iː/ and round your lips."
    ),
    ("yː", "iː"): (
        "Sounded like /iː/ ('ee') — round your lips tightly while holding the /iː/ tongue position for 'ü'."
    ),
    ("yː", "i"): (
        "Sounded like /i/ ('ee') — round your lips tightly and hold the vowel longer for long 'ü' /yː/."
    ),
    ("œ", "o"): (
        "Sounded like /o/ instead of German short 'ö' /œ/. "
        "Say 'eh' (/ɛ/) with your tongue forward, then round your lips."
    ),
    ("œ", "ɔ"): (
        "Sounded like /ɔ/ instead of German short 'ö' /œ/. "
        "Bring your tongue forward to /ɛ/ while keeping your lips rounded."
    ),
    ("œ", "ɛ"): (
        "Sounded unrounded (/ɛ/) — keep your tongue in the 'eh' position and round your lips for 'ö' /œ/."
    ),
    ("øː", "oː"): (
        "Sounded like /oː/ instead of German long 'ö' /øː/. "
        "Say /eː/ with your tongue forward, then round your lips."
    ),
    ("øː", "o"): (
        "Sounded like /o/ instead of German long 'ö' /øː/. "
        "Bring your tongue forward to /eː/ and round your lips."
    ),
    ("pf", "f"): (
        "The initial /p/ closure in /pf/ was dropped (heard /f/). "
        "Close your lips briefly for /p/ and release directly into /f/."
    ),
    ("ts", "s"): (
        "The initial /t/ burst in German 'z' /ts/ was missed (heard /s/). "
        "German 'z' always starts with a crisp /t/ (like the 'ts' in 'cats')."
    ),
    ("ts", "z"): (
        "German 'z' is pronounced /ts/ (like 'ts' in 'cats'), not English /z/."
    ),
    ("ʃ", "s"): (
        "Sounded like /s/ instead of /ʃ/ ('sh'). Before 'p' or 't' at the start of a syllable "
        "(sp-, st-), German pronounces 's' as /ʃ/."
    ),
    ("v", "w"): (
        "Sounded like English /w/ instead of German /v/. "
        "Touch your upper teeth lightly to your lower lip (like English 'v')."
    ),
}

# General German articulation tips by display IPA symbol
PHONEME_TIPS: dict[str, str] = {
    "ç": "Ich-Laut /ç/: tongue tip behind lower teeth, middle of tongue raised toward the hard palate (like 'h' in 'huge').",
    "x": "Ach-Laut /x/: back of tongue near soft palate/uvula with audible friction (after a, o, u, au).",
    "ʏ": "Short 'ü' /ʏ/: tongue position of /ɪ/ ('bit') + tightly rounded lips; keep it short.",
    "yː": "Long 'ü' /yː/: tongue position of /iː/ ('see') + tightly rounded lips; hold it long.",
    "œ": "Short 'ö' /œ/: tongue position of /ɛ/ ('bet') + rounded lips; keep it short.",
    "øː": "Long 'ö' /øː/: tongue position of /eː/ + rounded lips; hold it long.",
    "aː": "Long 'a' /aː/: open, clear 'ah' held noticeably longer than short 'a' (Staat vs. Stadt).",
    "a": "Short 'a' /a/: crisp, brief open vowel — do not stretch it (Stadt vs. Staat).",
    "ɔʏ": "German 'eu/äu' /ɔʏ/: glide smoothly from rounded 'o' (/ɔ/) toward rounded 'ü' (/ʏ/).",
    "pf": "Affricate /pf/: close lips for /p/ and release straight into /f/ in one smooth motion.",
    "ts": "German 'z' /ts/: crisp /t/ released directly into /s/ (like 'ts' in 'cats').",
    "ʃ": "German 'sch' (and initial sp-/st-) /ʃ/: round and protrude your lips slightly for a rich 'sh'.",
    "ʁ": "German 'r' /ʁ/: soft uvular friction at the back of the mouth (or vocalized /ɐ/ at the end of syllables).",
    "ɐ": "Vocalized '-er' /ɐ/: relaxed open central vowel (similar to a soft 'uh/ah'), not an English curled 'r'.",
    "ŋ": "Velar nasal /ŋ/: like 'ng' in 'sing', without releasing a hard /ɡ/ at the end.",
}


@dataclass(frozen=True)
class PhonemeSpec:
    """Expected phoneme tied to its parent word."""

    word_index: int
    word: str
    espeak_phone: str
    display_ipa: str
    model_token: str
    model_token_id: int
    acceptable_tokens: frozenset[str]
    acceptable_ids: frozenset[int]
    is_vowel: bool
    is_long_vowel: bool
    is_short_vowel: bool


@dataclass(frozen=True)
class WordSpec:
    """Expected word and its constituent phoneme indices."""

    index: int
    text: str
    clean_text: str
    phoneme_indices: list[int]
    display_ipa: str


def _get_espeak_backend() -> EspeakBackend:
    global _ESPEAK_BACKEND
    with _ESPEAK_LOCK:
        if _ESPEAK_BACKEND is None:
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            _ESPEAK_BACKEND = EspeakBackend("de", preserve_punctuation=False, with_stress=False)
        return _ESPEAK_BACKEND


def tokenize_german_words(text: str) -> list[tuple[str, str]]:
    """Split German text into (display_word, clean_word) pairs."""
    raw_words = [w for w in text.strip().split() if w]
    result: list[tuple[str, str]] = []
    for raw in raw_words:
        # Keep German letters (including umlauts & ß) and digits; drop surrounding punctuation
        clean = re.sub(r"^[^\wÄÖÜäöüß]+|[^\wÄÖÜäöüß]+$", "", raw, flags=re.UNICODE)
        if clean:
            result.append((raw, clean))
    return result


def map_espeak_to_vocab(phone: str, vocab: dict[str, int]) -> str:
    """Map an espeak-ng phone to a valid model vocabulary token."""
    if phone in ESPEAK_TO_MODEL_TOKEN and ESPEAK_TO_MODEL_TOKEN[phone] in vocab:
        return ESPEAK_TO_MODEL_TOKEN[phone]
    if phone in vocab:
        return phone
    # Strip length mark or diacritics as fallback
    stripped = phone.replace("ː", "").replace(":", "").replace("̩", "").replace("̯", "")
    if stripped in ESPEAK_TO_MODEL_TOKEN and ESPEAK_TO_MODEL_TOKEN[stripped] in vocab:
        return ESPEAK_TO_MODEL_TOKEN[stripped]
    if stripped in vocab:
        return stripped
    # Try first character that exists in vocab
    for ch in phone:
        if ch in ESPEAK_TO_MODEL_TOKEN and ESPEAK_TO_MODEL_TOKEN[ch] in vocab:
            return ESPEAK_TO_MODEL_TOKEN[ch]
        if ch in vocab:
            return ch
    return "<unk>"


def get_acceptable_tokens(model_token: str, vocab: dict[str, int]) -> frozenset[str]:
    """Return the set of in-vocab tokens considered acceptable for a target model token."""
    raw = RAW_ACCEPTABLE_EQUIVALENTS.get(model_token, {model_token})
    valid = {t for t in raw if t in vocab}
    if model_token in vocab:
        valid.add(model_token)
    return frozenset(valid)


def phonemize_german_text(
    text: str,
    vocab: dict[str, int],
) -> tuple[list[WordSpec], list[PhonemeSpec]]:
    """Convert a German sentence into aligned WordSpec and PhonemeSpec lists."""
    word_pairs = tokenize_german_words(text)
    if not word_pairs:
        raise ValueError("Reference text must contain at least one German word.")

    backend = _get_espeak_backend()
    sep = Separator(phone=" ", word=" | ", syllable="")

    # Identify words that need espeak phonemization vs override
    words_to_phonemize: list[str] = []
    override_map: dict[int, list[str]] = {}
    phonemize_indices: list[int] = []

    for idx, (_display, clean) in enumerate(word_pairs):
        lower = clean.lower()
        if lower in GERMAN_G2P_OVERRIDES:
            override_map[idx] = GERMAN_G2P_OVERRIDES[lower]
        else:
            words_to_phonemize.append(clean)
            phonemize_indices.append(idx)

    espeak_results: dict[int, list[str]] = dict(override_map)
    if words_to_phonemize:
        with _ESPEAK_LOCK:
            raw_outputs = backend.phonemize(words_to_phonemize, separator=sep, strip=True)
        for idx, raw_out in zip(phonemize_indices, raw_outputs):
            phones = [t.strip() for t in raw_out.split() if t.strip() and t.strip() != "|"]
            espeak_results[idx] = phones

    words: list[WordSpec] = []
    phonemes: list[PhonemeSpec] = []

    for w_idx, (display_word, clean_word) in enumerate(word_pairs):
        raw_phones = espeak_results.get(w_idx, [])
        w_phoneme_indices: list[int] = []
        w_display_ipa: list[str] = []

        for ep in raw_phones:
            model_tok = map_espeak_to_vocab(ep, vocab)
            if model_tok == "<unk>" or model_tok not in vocab:
                continue
            disp = DISPLAY_IPA_MAP.get(ep, ep)
            acc_toks = get_acceptable_tokens(model_tok, vocab)
            acc_ids = frozenset(vocab[t] for t in acc_toks if t in vocab)

            is_long = ep in LONG_VOWELS or disp in LONG_VOWELS
            is_short = ep in SHORT_VOWELS or disp in SHORT_VOWELS
            is_vowel = is_long or is_short or ep in ALL_VOWELS or disp in ALL_VOWELS

            p_idx = len(phonemes)
            phonemes.append(
                PhonemeSpec(
                    word_index=w_idx,
                    word=display_word,
                    espeak_phone=ep,
                    display_ipa=disp,
                    model_token=model_tok,
                    model_token_id=vocab[model_tok],
                    acceptable_tokens=acc_toks,
                    acceptable_ids=acc_ids,
                    is_vowel=is_vowel,
                    is_long_vowel=is_long,
                    is_short_vowel=is_short,
                )
            )
            w_phoneme_indices.append(p_idx)
            w_display_ipa.append(disp)

        words.append(
            WordSpec(
                index=w_idx,
                text=display_word,
                clean_text=clean_word,
                phoneme_indices=w_phoneme_indices,
                display_ipa="".join(w_display_ipa),
            )
        )

    if not phonemes:
        raise ValueError(f"Could not extract any German phonemes from text: {text!r}")

    return words, phonemes


def get_diagnosis_and_tip(
    spec: PhonemeSpec,
    heard_token: str | None,
    status: str,
    length_issue: str | None = None,
) -> tuple[str | None, str | None]:
    """Build a learner-friendly diagnosis and articulation tip for a phoneme."""
    tip = PHONEME_TIPS.get(spec.display_ipa) or PHONEME_TIPS.get(spec.espeak_phone)

    if status == "omitted":
        return (
            f"The sound /{spec.display_ipa}/ in '{spec.word}' was dropped or too faint to detect.",
            tip,
        )

    if length_issue == "too_short":
        return (
            f"Vowel /{spec.display_ipa}/ in '{spec.word}' was pronounced too short — hold long German vowels clearly.",
            tip or "Long German vowels (aa, ah, ie, single consonant after vowel) should be held noticeably longer than short vowels.",
        )

    if length_issue == "too_long":
        return (
            f"Vowel /{spec.display_ipa}/ in '{spec.word}' was stretched too long — keep short German vowels crisp.",
            tip or "Short German vowels (usually followed by a double consonant) should be brief and crisp.",
        )

    if status in ("error", "fair") and heard_token:
        heard_disp = DISPLAY_IPA_MAP.get(heard_token, heard_token)
        for key in (
            (spec.display_ipa, heard_token),
            (spec.display_ipa, heard_disp),
            (spec.espeak_phone, heard_token),
            (spec.model_token, heard_token),
        ):
            if key in CONFUSION_DIAGNOSES:
                return CONFUSION_DIAGNOSES[key], tip

        if heard_token not in spec.acceptable_tokens:
            severity_verb = "sounded like" if status == "error" else "drifted toward"
            return (
                f"In '{spec.word}', expected /{spec.display_ipa}/ but it {severity_verb} /{heard_disp}/.",
                tip,
            )
        return (
            f"In '{spec.word}', /{spec.display_ipa}/ was weak or unclear.",
            tip,
        )

    return None, tip


def format_ipa_list(tokens: Iterable[str]) -> str:
    """Format a sequence of IPA tokens as a readable string."""
    return " ".join(DISPLAY_IPA_MAP.get(t, t) for t in tokens)
