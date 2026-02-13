from cc.transcribe.diarize.identify import annotate_transcript, format_identifications


def test_format_identifications_consistent():
    transcript = "CHUNK_0_A: Long utterance about something.\nCHUNK_0_A: Another long one here.\n"
    result = format_identifications(transcript, {0: "Peter", 1: "Peter"})
    assert "CHUNK_0_A → Peter" in result
    assert "CONFLICTING" not in result


def test_format_identifications_conflicting():
    transcript = "CHUNK_2_A: Long utterance about something.\nCHUNK_2_A: Another long one here.\n"
    result = format_identifications(transcript, {0: "Peter", 1: "Eby"})
    assert "CONFLICTING" in result


def test_format_identifications_unknown():
    result = format_identifications("CHUNK_0_A: stuff\n", {}, unknowns={"CHUNK_0_A"})
    assert "CHUNK_0_A → unknown" in result


def test_format_identifications_empty():
    assert format_identifications("no chunks", {}) == ""
