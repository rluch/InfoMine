__author__ = 'rasmus'

import pep257


def test_pep257():
    _mods = ['InfoMine']
    files = pep257.collect(_mods, match=lambda name: name.endswith('.py'))
    files_results = pep257.check(sorted(files))

    if len(list(files_results)) is not 0:
        for err in list(files_results):
            print err
            assert 0
    else:
        assert 1
