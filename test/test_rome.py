from pytest import raises

from rome import Roman


def test_converting_single_digit_roman_numeral_to_arabic():
    assert Roman('I') == 1
    assert Roman('X') == 10


def test_converting_multiple_digit_roman_numeral_to_arabic():
    assert Roman('XVI') == 16


def test_when_invalid_numeral_is_passed_it_raises_value_error():
    with raises(ValueError):
        Roman('@#$')
