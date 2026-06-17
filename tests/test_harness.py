"""Regression harness for tune-matcher scoring and identity logic."""
from __future__ import annotations

import os

import pytest

import tune_core as tc

SONG_DIR = os.path.join(os.path.dirname(__file__), "..", "songs")


class TestTempoFactor:
    def test_within_2_seconds_full_credit(self):
        for delta in (0.0, 1.0, 2.0):
            factor, d, ok, mx = tc.tempo_factor(18.0, 18.0 + delta, False)
            assert factor == 1.0
            assert d == delta
            assert ok == tc.TEMPO_OK_SEC
            assert mx == tc.TEMPO_MAX_SEC

    def test_at_3_seconds_zero_credit(self):
        factor, d, _, _ = tc.tempo_factor(18.0, 21.0, False)
        assert factor == 0.0
        assert d == 3.0

    def test_between_2_and_3_linear_ramp(self):
        factor, d, _, _ = tc.tempo_factor(18.0, 20.5, False)
        assert d == 2.5
        assert 0.0 < factor < 1.0
        assert abs(factor - 0.5) < 0.01

    def test_faster_side_symmetric(self):
        f_slow, _, _, _ = tc.tempo_factor(18.0, 21.0, False)
        f_fast, _, _, _ = tc.tempo_factor(18.0, 15.0, False)
        assert f_slow == f_fast == 0.0


class TestScoreCaps:
    def test_wrong_padyam_caps_at_35(self):
        assert tc.apply_score_caps(97.0, wrong_song=True, identity_blocks=False) == 35.0

    def test_identity_blocks_caps_at_35(self):
        assert tc.apply_score_caps(97.0, wrong_song=False, identity_blocks=True) == 35.0

    def test_high_tune_score_cannot_bypass_wrong_padyam(self):
        base = tc.compute_base_score(0.5)
        final, _ = tc.compute_final_score(base, 1.0)
        assert final > 90
        capped = tc.apply_score_caps(final, wrong_song=True, identity_blocks=False)
        assert capped == tc.WRONG_PADYAM_SCORE_CAP
        assert not tc.would_qualify(
            capped, clear_words=True, wrong_song=True, identity_blocks=False)

    def test_would_not_qualify_on_identity_blocks(self):
        assert not tc.would_qualify(
            97.0, clear_words=True, wrong_song=False, identity_blocks=True)

    def test_pacing_penalty_can_drop_below_pass(self):
        base = tc.compute_base_score(0.5)
        final, _ = tc.compute_final_score(base, 0.0)
        assert final < tc.PASS_THRESHOLD


class TestIdentityBlocksPass:
    def test_same_song_never_blocks(self):
        assert not tc.identity_blocks_pass("a", "a", 1.0, 0.9)

    def test_clear_other_winner_blocks(self):
        assert tc.identity_blocks_pass("a", "b", 1.0, 0.9)

    def test_tie_does_not_block(self):
        assert not tc.identity_blocks_pass("a", "b", 1.0, 0.98)

    def test_final_test_skips_identity_block(self):
        assert not tc.identity_blocks_pass(
            "a", "b", 1.0, 0.5, is_final_test=True)


class TestSessionRegistry:
    def test_congestion_at_40(self):
        sessions = {f"s{i}": 1000.0 for i in range(40)}
        assert tc.server_congested(sessions, 1000.0)
        assert not tc.server_congested(sessions, 1000.0, max_concurrent=41)

    def test_stale_sessions_pruned(self):
        sessions = {}
        tc.touch_active_session(sessions, "old", 100.0, inactivity_sec=600)
        tc.touch_active_session(sessions, "new", 800.0, inactivity_sec=600)
        assert "old" not in sessions
        assert sessions["new"] == 800.0


@pytest.fixture(scope="module")
def available_songs():
    songs = tc.list_songs(SONG_DIR)
    if len(songs) < 18:
        pytest.skip("songs/ folder with 18 .wav files required for audio harness tests")
    return songs


@pytest.fixture(scope="module")
def ref_loader(available_songs):
    return tc.make_ref_loader()


class TestWrongPoemDetection:
    def test_all_references_self_match_not_wrong(self, available_songs, ref_loader):
        false_positives = []
        for name, path in available_songs.items():
            chroma, mfcc, _ = tc.load_user_identity_features(path)
            wrong, best, _, _ = tc.detect_wrong_song(
                name, chroma, mfcc, available_songs, ref_loader)
            if wrong:
                false_positives.append((name, best))
        assert false_positives == [], f"false wrong-poem: {false_positives}"

    def test_cross_poem_detected_when_singing_different_ref(self, available_songs, ref_loader):
        keys = sorted(available_songs.keys())
        src, wrong_key = keys[0], keys[1]
        chroma, mfcc, _ = tc.load_user_identity_features(available_songs[src])
        wrong, best, sel_d, best_d = tc.detect_wrong_song(
            wrong_key, chroma, mfcc, available_songs, ref_loader)
        blocks = tc.identity_blocks_pass(wrong_key, best, sel_d, best_d)
        assert best == src
        assert wrong or blocks, (
            f"expected wrong-poem or identity block for {src} vs selected {wrong_key}"
        )

    def test_song_13_self_match(self, available_songs, ref_loader):
        key = "13_anuvu_gaani_chota_vemana"
        if key not in available_songs:
            pytest.skip("song 13 not in songs/")
        chroma, mfcc, _ = tc.load_user_identity_features(available_songs[key])
        wrong, best, _, _ = tc.detect_wrong_song(
            key, chroma, mfcc, available_songs, ref_loader)
        assert not wrong
        assert best == key
