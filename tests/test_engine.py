"""End-to-end tests for the German pronunciation assessment engine and FastAPI web API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from german_aussprachen.phonetics import phonemize_german_text
from german_aussprachen.presets import get_samples_dir
from german_aussprachen.scorer import get_default_scorer
from german_aussprachen.web.app import create_app


@pytest.fixture(scope="module")
def scorer():
    s = get_default_scorer()
    s.ensure_loaded()
    return s


@pytest.fixture(scope="module")
def client(scorer):
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_g2p_handles_oov_and_overrides(scorer):
    words, phonemes = phonemize_german_text("Ich möchte schöne Vögel und eine Hütte.", scorer.vocab)
    assert len(words) == 7
    # Vögel override fixes espeak's /v/ -> /f/
    voegel_word = words[3]
    assert voegel_word.clean_text == "Vögel"
    assert phonemes[voegel_word.phoneme_indices[0]].display_ipa == "f"
    # Hütte keeps display IPA ʏ while mapping model token to in-vocab 'y'
    huette_word = words[6]
    huette_phones = [phonemes[i] for i in huette_word.phoneme_indices]
    assert any(p.display_ipa == "ʏ" and p.model_token == "y" for p in huette_phones)


def test_native_vs_mispronounced_huette(scorer):
    samples = get_samples_dir()
    ref = "Ich möchte eine Hütte mieten."

    native = scorer.score_file(samples / "huette_native.wav", ref)
    bad = scorer.score_file(samples / "huette_isch_hut.wav", ref)

    assert native.overall_score >= 95.0
    assert native.top_issues == []

    # Must catch BOTH /ç/ -> /ʃ/ in 'Ich' AND /ʏ/ -> /ʊ/ in 'Hütte'
    assert bad.overall_score < native.overall_score - 8.0
    ich_ch = bad.words[0].phonemes[1]
    assert ich_ch.expected_ipa == "ç"
    assert ich_ch.heard_ipa == "ʃ"
    assert ich_ch.status == "error"

    huette_u = bad.words[3].phonemes[1]
    assert huette_u.expected_ipa == "ʏ"
    assert huette_u.heard_ipa in {"ʊ", "u", "uː"}
    assert huette_u.status == "error"


def test_dropped_words_completeness(scorer):
    samples = get_samples_dir()
    ref = "Ich möchte eine Hütte mieten."
    dropped = scorer.score_file(samples / "huette_dropped_word.wav", ref)

    assert dropped.completeness_score < 70.0
    assert dropped.words[2].status == "omitted"  # eine
    assert dropped.words[3].status == "omitted"  # Hütte


def test_vowel_length_staat_vs_stadt(scorer):
    samples = get_samples_dir()
    ref = "Der Staat baut eine neue Stadt."

    native = scorer.score_file(samples / "staat_native.wav", ref)
    swapped = scorer.score_file(samples / "staat_swapped.wav", ref)

    assert native.overall_score >= 95.0
    assert native.duration_score == 100.0

    assert swapped.duration_score < 60.0
    staat_vowel = next(p for p in swapped.words[1].phonemes if p.is_long_vowel)
    assert staat_vowel.length_issue == "too_short"


def test_ach_laut_bach_vs_back(scorer):
    samples = get_samples_dir()
    ref = "Zwölf große Pferde springen über den Bach."

    native = scorer.score_file(samples / "bach_native.wav", ref)
    err = scorer.score_file(samples / "bach_back_error.wav", ref)

    assert native.overall_score >= 95.0
    bach_x = err.words[-1].phonemes[-1]
    assert bach_x.expected_ipa == "x"
    assert bach_x.heard_ipa == "k"
    assert bach_x.status == "error"


def test_web_api_endpoints(client):
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "German Aussprache Lab" in r_index.text

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] in {"ready", "loading"}

    r_presets = client.get("/api/presets")
    assert r_presets.status_code == 200
    assert len(r_presets.json()["presets"]) >= 5

    r_samples = client.get("/api/samples")
    assert r_samples.status_code == 200
    assert len(r_samples.json()["samples"]) >= 6

    r_sample_score = client.post("/api/analyze-sample/huette_isch_hut")
    assert r_sample_score.status_code == 200
    payload = r_sample_score.json()
    assert payload["overall_score"] < 92.0
    assert len(payload["top_issues"]) >= 2

    # Direct file upload endpoint
    wav_bytes = (get_samples_dir() / "huette_native.wav").read_bytes()
    r_upload = client.post(
        "/api/analyze",
        data={"reference_text": "Ich möchte eine Hütte mieten."},
        files={"audio": ("huette_native.wav", wav_bytes, "audio/wav")},
    )
    assert r_upload.status_code == 200
    assert r_upload.json()["overall_score"] >= 95.0
