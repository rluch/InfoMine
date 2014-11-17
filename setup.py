import os
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

import infomine


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        sys.exit(os.system('py.test'))


setup(
    name=infomine.__name__,
    version=infomine.__version__,
    author=infomine.__author__,
    author_email=infomine.__author_email__,
    description=infomine.__doc__,
    license='MIT',
    keywords='infomine infominer',
    url='http://github.com/rluch/infomine',
    packages=find_packages(exclude="test"),
    py_modules=['infomine'],
    long_description=open('README.rst').read(),
    install_requires=['docopt>=0.6.0,<0.7.0'],
    cmdclass={'test': PyTest},
    tests_require=['pytest'],
    scripts=['bin/infominer'],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: MIT License',
    ],
)
