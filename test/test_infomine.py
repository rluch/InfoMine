from pytest import raises

from infomine import InfoMiner


def test_dummy():
    assert 1 == 9000
    #assert Roman('X') == 10


def test_gender_classifier():
    assert 1 != 2


def test_when_invalid_numeral_is_passed_it_raises_value_error():
    with raises(ValueError):
        InfoMiner('@#$')
