from knowledge_engine.logical_gaps import LogicalGapDetector


def test_logical_gap_detector_instantiates():
    detector = LogicalGapDetector()
    assert detector is not None
