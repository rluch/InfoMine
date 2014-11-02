import os
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

import infomine


class ToxTestCommand(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        sys.exit(os.system('tox'))


setup(
    name=infomine.__name__,
    version=infomine.__version__,
    author=infomine.__author__,
    author_email=infomine.__author_email__,
    description=infomine.__doc__,
    license='MIT',
    keywords='roman numerals',
    url='http://github.com/rluch/infomine',
    # packages=find_packages(),
    py_modules=['infomine'],
    long_description=open('README.rst').read(),
    install_requires=['docopt>=0.6.0,<0.7.0'],
    cmdclass={'test': ToxTestCommand},
    tests_require=['tox'],
    scripts=['bin/infominer'],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'License :: OSI Approved :: MIT License',
    ],
)
