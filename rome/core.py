"""Converting Roman numerals to Arabic ones.

Examples
--------
>>> Roman('V')
5
>>> Roman('X') + 3
13

"""
__version__ = '0.1.0'
__author__ = 'Vladimir Keleshev'
__author_email__ = 'vladimir@keleshev.com'


_numerals = {'I': 1, 'V': 5, 'X': 10}


class Roman(int):

    """Roman numeral."""

    def __new__(class_, roman):
        """Construct new Roman numeral from string `roman`."""
        try:
            return sum(_numerals[digit] for digit in roman)
        except KeyError:
            raise ValueError('Invalid Roman numeral: %r' % roman)
