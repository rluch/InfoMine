__author__ = 'rasmus'
from pytest import raises
import pep257


def test_pep257():
    _mods = ['.']
    _files = pep257.collect(_mods, match=lambda name: name.endswith('.py'))
    _files_results = pep257.check(sorted(_files))

    if len(list(_files_results)) is not 0:
        for err in list(_files_results):
            print err
            assert 0
    else:
        assert 1
