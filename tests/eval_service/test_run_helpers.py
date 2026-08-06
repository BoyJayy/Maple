import json

from run import extract_ids, load_baseline, load_dataset


def test_load_dataset_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n')
    assert load_dataset(path) == [{"id": 1}, {"id": 2}]


def test_load_dataset_json_array(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('[{"id": 1}, {"id": 2}]')
    assert load_dataset(path) == [{"id": 1}, {"id": 2}]


def test_extract_ids_flattens_results():
    results = [{"message_ids": ["a", "b"]}, {"message_ids": []}, {}]
    assert extract_ids(results) == ["a", "b"]


def test_load_baseline_missing_file(tmp_path):
    assert load_baseline(tmp_path / "absent.json") == {}


def test_load_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    payload = {"stages": {"final": {"recall": 0.9}}}
    path.write_text(json.dumps(payload))
    assert load_baseline(path) == payload
