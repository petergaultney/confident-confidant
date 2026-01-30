import pytest

from cc.transcribe.split.choose_silence_cuts import Silence, _choose_cuts, _parse_silences


class TestParseSilences:
    def test_empty_input(self):
        assert _parse_silences([]) == []

    def test_single_silence(self):
        lines = [
            "[silencedetect @ 0x123] silence_start: 10.5",
            "[silencedetect @ 0x123] silence_end: 11.2 | silence_duration: 0.7",
        ]
        result = _parse_silences(lines)
        assert len(result) == 1
        assert result[0].start == 10.5
        assert result[0].end == 11.2

    def test_multiple_silences(self):
        lines = [
            "silence_start: 100.0",
            "silence_end: 101.5",
            "silence_start: 200.0",
            "silence_end: 202.0",
            "silence_start: 300.0",
            "silence_end: 300.5",
        ]
        result = _parse_silences(lines)
        assert len(result) == 3
        assert result[0] == Silence(start=100.0, end=101.5)
        assert result[1] == Silence(start=200.0, end=202.0)
        assert result[2] == Silence(start=300.0, end=300.5)

    def test_ignores_unmatched_end(self):
        lines = [
            "silence_end: 50.0",  # no matching start
            "silence_start: 100.0",
            "silence_end: 101.0",
        ]
        result = _parse_silences(lines)
        assert len(result) == 1
        assert result[0] == Silence(start=100.0, end=101.0)

    def test_ignores_end_before_start(self):
        lines = [
            "silence_start: 100.0",
            "silence_end: 99.0",  # end before start - should be ignored
        ]
        result = _parse_silences(lines)
        assert len(result) == 0

    def test_handles_integer_timestamps(self):
        lines = [
            "silence_start: 100",
            "silence_end: 200",
        ]
        result = _parse_silences(lines)
        assert len(result) == 1
        assert result[0].start == 100.0
        assert result[0].end == 200.0

    def test_mixed_content_lines(self):
        lines = [
            "some random output",
            "frame=  100 fps=50",
            "silence_start: 10.123456",
            "more random output",
            "silence_end: 10.987654 | silence_duration: 0.864198",
            "even more output",
        ]
        result = _parse_silences(lines)
        assert len(result) == 1
        assert result[0].start == pytest.approx(10.123456)
        assert result[0].end == pytest.approx(10.987654)


class TestSilenceMid:
    def test_mid_calculation(self):
        s = Silence(start=10.0, end=20.0)
        assert s.mid == 15.0

    def test_mid_with_decimals(self):
        s = Silence(start=10.5, end=11.5)
        assert s.mid == 11.0


class TestChooseCuts:
    def test_empty_mids(self):
        result = _choose_cuts(
            mids=[],
            every=1200,
            duration=None,
            window=90,
            start_at=1200,
            stop_before_end=30,
        )
        assert result == []

    def test_single_cut(self):
        mids = [1200.0]
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=3600,
            window=90,
            start_at=1200,
            stop_before_end=30,
        )
        assert len(result) == 1
        cut = result[0]
        assert cut.target == 1200.0
        assert cut.chosen == 1200.0
        assert cut.delta == 0.0

    def test_nearest_silence_chosen(self):
        mids = [1150.0, 1190.0, 1250.0]  # 1190 is closest to target 1200
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=2000,  # short duration to only have one cut
            window=90,
            start_at=1200,
            stop_before_end=30,
        )
        assert len(result) == 1
        assert result[0].chosen == 1190.0

    def test_window_constraint(self):
        # 1000 is outside window of 90s from target 1200
        mids = [1000.0, 1150.0]
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=2000,  # short duration to only have one cut
            window=90,
            start_at=1200,
            stop_before_end=30,
        )
        assert len(result) == 1
        assert result[0].chosen == 1150.0

    def test_no_window(self):
        # With window=None, should pick global nearest
        mids = [500.0]  # far from 1200 but only option
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=3600,
            window=None,
            start_at=1200,
            stop_before_end=30,
        )
        assert len(result) == 1
        assert result[0].chosen == 500.0

    def test_multiple_cuts(self):
        mids = [1200.0, 2400.0, 3600.0]
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=5000,
            window=90,
            start_at=1200,
            stop_before_end=30,
        )
        assert len(result) == 3
        assert result[0].chosen == 1200.0
        assert result[1].chosen == 2400.0
        assert result[2].chosen == 3600.0

    def test_stop_before_end(self):
        # Don't place cut within 30s of end (duration=3600)
        mids = [1200.0, 2400.0, 3580.0]  # 3580 is within 30s of 3600
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=3600,
            window=90,
            start_at=1200,
            stop_before_end=30,
        )
        # Should only get cuts at 1200 and 2400, not at 3600 target
        assert len(result) == 2

    def test_no_reuse_of_mids(self):
        # Only one mid available, but two targets
        mids = [1200.0]
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=3600,
            window=None,
            start_at=1200,
            stop_before_end=30,
        )
        # Can only use the mid once
        assert len(result) == 1

    def test_fallback_to_global_when_window_exhausted(self):
        # Window is 10s, but nearest within window already used
        mids = [1195.0, 2500.0]  # 1195 near target 1200, 2500 far from target 2400
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=4000,
            window=10,
            start_at=1200,
            stop_before_end=30,
        )
        assert len(result) == 2
        assert result[0].chosen == 1195.0  # used for target 1200
        assert result[1].chosen == 2500.0  # fallback to global nearest for target 2400

    def test_inferred_duration(self):
        # When duration is None, infer from last mid + every
        mids = [1200.0]
        result = _choose_cuts(
            mids=mids,
            every=1200,
            duration=None,
            window=90,
            start_at=1200,
            stop_before_end=30,
        )
        # Inferred duration = 1200 + 1200 = 2400
        # last_target = 2400 - 30 = 2370
        # Only target at 1200 fits
        assert len(result) == 1
