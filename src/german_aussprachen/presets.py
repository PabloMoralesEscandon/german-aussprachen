"""Curated German practice sentences and built-in demo audio sample definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PracticePreset:
    id: str
    title: str
    text: str
    english: str
    focus: list[str]
    difficulty: str


@dataclass(frozen=True)
class DemoSample:
    id: str
    preset_id: str
    label: str
    kind: str  # "native" | "mispronounced"
    description: str
    reference_text: str
    filename: str


PRACTICE_PRESETS: list[PracticePreset] = [
    PracticePreset(
        id="huette",
        title="Ich-Laut & Short Ü",
        text="Ich möchte eine Hütte mieten.",
        english="I would like to rent a cabin.",
        focus=["/ç/ (ich-Laut)", "/œ/ (short ö)", "/ʏ/ (short ü)", "/iː/ (long i)"],
        difficulty="Beginner",
    ),
    PracticePreset(
        id="staat_stadt",
        title="Long vs. Short Vowels",
        text="Der Staat baut eine neue Stadt.",
        english="The state is building a new city.",
        focus=["/aː/ vs /a/ (Staat vs. Stadt)", "/ʃt/ (st-)", "/ɔʏ/ (eu)"],
        difficulty="Intermediate",
    ),
    PracticePreset(
        id="buecher",
        title="Umlauts & Vocalized -er",
        text="Bücher über schöne Vögel und Äpfel.",
        english="Books about beautiful birds and apples.",
        focus=["/yː/ (long ü)", "/øː/ (long ö)", "/ɐ/ (-er)", "/pf/"],
        difficulty="Intermediate",
    ),
    PracticePreset(
        id="bach",
        title="Ach-Laut, Z & Pf",
        text="Zwölf große Pferde springen über den Bach.",
        english="Twelve big horses jump over the stream.",
        focus=["/ts/ (z)", "/x/ (ach-Laut)", "/pf/", "/ʃp/ (sp-)"],
        difficulty="Advanced",
    ),
    PracticePreset(
        id="kueche",
        title="Küche vs. Kuchen",
        text="In der Küche backen wir einen warmen Kuchen.",
        english="In the kitchen we bake a warm cake.",
        focus=["/ʏ/ + /ç/ (Küche)", "/uː/ + /x/ (Kuchen)", "/ʁ/"],
        difficulty="Intermediate",
    ),
    PracticePreset(
        id="deutsch",
        title="Everyday Greeting",
        text="Sprechen Sie Deutsch? Ja, ein bisschen!",
        english="Do you speak German? Yes, a little bit!",
        focus=["/ʃp/", "/ç/ (sprechen, bisschen)", "/ɔʏ/ (Deutsch)"],
        difficulty="Beginner",
    ),
]


DEMO_SAMPLES: list[DemoSample] = [
    DemoSample(
        id="huette_native",
        preset_id="huette",
        label="Native German — Hütte",
        kind="native",
        description="Clean native German pronunciation of 'Ich möchte eine Hütte mieten.'",
        reference_text="Ich möchte eine Hütte mieten.",
        filename="huette_native.wav",
    ),
    DemoSample(
        id="huette_isch_hut",
        preset_id="huette",
        label="Learner Error — 'Isch möschte ... Hutte'",
        kind="mispronounced",
        description="Classic mistakes: /ʃ/ ('sh') instead of /ç/ in 'ich/möchte', and /u/ instead of /ʏ/ in 'Hütte'.",
        reference_text="Ich möchte eine Hütte mieten.",
        filename="huette_isch_hut.wav",
    ),
    DemoSample(
        id="huette_english_accent",
        preset_id="huette",
        label="English Accent Reading German",
        kind="mispronounced",
        description="Reading the German sentence with English phonology.",
        reference_text="Ich möchte eine Hütte mieten.",
        filename="huette_english_accent.wav",
    ),
    DemoSample(
        id="huette_dropped_word",
        preset_id="huette",
        label="Incomplete — Dropped 'eine Hütte'",
        kind="mispronounced",
        description="Speaker skips 'eine Hütte' — tests completeness & deletion detection.",
        reference_text="Ich möchte eine Hütte mieten.",
        filename="huette_dropped_word.wav",
    ),
    DemoSample(
        id="staat_native",
        preset_id="staat_stadt",
        label="Native German — Staat vs. Stadt",
        kind="native",
        description="Clear contrast between long /aː/ in 'Staat' and short /a/ in 'Stadt'.",
        reference_text="Der Staat baut eine neue Stadt.",
        filename="staat_native.wav",
    ),
    DemoSample(
        id="staat_swapped",
        preset_id="staat_stadt",
        label="Learner Error — Swapped Staat / Stadt",
        kind="mispronounced",
        description="Says short 'Stadt' where long 'Staat' is expected, and vice versa.",
        reference_text="Der Staat baut eine neue Stadt.",
        filename="staat_swapped.wav",
    ),
    DemoSample(
        id="bach_native",
        preset_id="bach",
        label="Native German — Zwölf Pferde & Bach",
        kind="native",
        description="Clean pronunciation of /ts/, /pf/, /ʃp/, and ach-Laut /x/.",
        reference_text="Zwölf große Pferde springen über den Bach.",
        filename="bach_native.wav",
    ),
    DemoSample(
        id="bach_back_error",
        preset_id="bach",
        label="Learner Error — 'Back' (/k/) instead of 'Bach' (/x/)",
        kind="mispronounced",
        description="Replaces the German ach-Laut /x/ in 'Bach' with hard /k/.",
        reference_text="Zwölf große Pferde springen über den Bach.",
        filename="bach_back_error.wav",
    ),
]


def get_samples_dir() -> Path:
    """Resolve the project's samples directory."""
    # src/german_aussprachen/presets.py -> repo root / samples
    return Path(__file__).resolve().parents[2] / "samples"


def list_presets_dict() -> list[dict]:
    return [asdict(p) for p in PRACTICE_PRESETS]


def list_available_samples() -> list[dict]:
    samples_dir = get_samples_dir()
    out = []
    for s in DEMO_SAMPLES:
        path = samples_dir / s.filename
        if path.exists():
            d = asdict(s)
            d["audio_url"] = f"/samples/{s.filename}"
            out.append(d)
    return out
