import os
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

import rome


class ToxTestCommand(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        sys.exit(os.system('tox'))


setup(
    name=rome.__name__,
    version=rome.__version__,
    author=rome.__author__,
    author_email=rome.__author_email__,
    description=rome.__doc__,
    license='MIT',
    keywords='roman numerals',
    url='http://github.com/halst/rome',
    # packages=find_packages(),
    py_modules=['rome'],
    long_description=open('README.rst').read(),
    install_requires=['docopt>=0.6.0,<0.7.0'],
    cmdclass={'test': ToxTestCommand},
    tests_require=['tox'],
    scripts=['bin/roman'],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'License :: OSI Approved :: MIT License',
    ],
)
