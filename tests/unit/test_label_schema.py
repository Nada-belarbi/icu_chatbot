from config.label_schema import (
    ID_TO_LABEL,
    LABEL_ORDER,
    LABEL_TO_ID,
    REQUIRED_DATASET_COLUMNS,
)


def test_label_order_matches_dataset_contract():
    assert LABEL_ORDER == [
        "information_generale",
        "sensible_reformulable",
        "avis_medical_personnalise",
        "urgence",
        "hors_perimetre",
    ]


def test_label_mapping_is_bijective():
    assert len(LABEL_TO_ID) == len(ID_TO_LABEL) == len(LABEL_ORDER)
    for label, idx in LABEL_TO_ID.items():
        assert ID_TO_LABEL[idx] == label


def test_required_dataset_columns():
    assert REQUIRED_DATASET_COLUMNS == ["text", "label", "topic", "source"]
