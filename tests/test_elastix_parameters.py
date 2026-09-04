"""Tests for elastix ParameterMap read/write/round-trip."""

from __future__ import annotations

from commandants.elastix import ParameterMap
from commandants.elastix import presets


def test_value_formatting():
    pm = ParameterMap()
    pm.set("Transform", "EulerTransform")     # string -> quoted
    pm.set("NumberOfResolutions", 4)          # int -> bare
    pm.set("WriteResultImage", True)          # bool -> "true"
    pm.set("Foo", False)                      # bool -> "false"
    pm.set("Schedule", 8, 4, 2, 1)            # sequence -> space-joined
    text = pm.to_text()
    assert '(Transform "EulerTransform")' in text
    assert "(NumberOfResolutions 4)" in text
    assert '(WriteResultImage "true")' in text
    assert '(Foo "false")' in text
    assert "(Schedule 8 4 2 1)" in text


def test_round_trip_preset():
    pm = presets.parameter_map("affine")
    text = pm.to_text()
    back = ParameterMap.parse(text)
    assert back.to_text() == text
    assert back["Transform"] == ("AffineTransform",)
    assert back["NumberOfResolutions"] == (4,)


def test_parse_ignores_comments_and_blanks():
    text = '// a comment\n\n(Transform "AffineTransform")\n(NumberOfResolutions 3)\n'
    pm = ParameterMap.parse(text)
    assert list(pm.keys()) == ["Transform", "NumberOfResolutions"]
    assert pm["NumberOfResolutions"] == (3,)


def test_write_and_from_file(tmp_path):
    pm = presets.parameter_map("rigid")
    path = tmp_path / "rigid.txt"
    pm.write(path)
    loaded = ParameterMap.from_file(path)
    assert loaded.to_text() == pm.to_text()


def test_dict_construction_and_update():
    pm = ParameterMap({"Metric": "AdvancedMattesMutualInformation", "NumberOfHistogramBins": 32})
    assert pm["Metric"] == ("AdvancedMattesMutualInformation",)
    assert pm["NumberOfHistogramBins"] == (32,)
    pm.update({"Metric": "AdvancedNormalizedCorrelation"})
    assert pm["Metric"] == ("AdvancedNormalizedCorrelation",)


def test_preset_kinds_and_bad_kind():
    import pytest

    for kind in ("translation", "rigid", "affine", "bspline"):
        assert isinstance(presets.parameter_map(kind), ParameterMap)
    with pytest.raises(ValueError):
        presets.parameter_map("nonsense")
