from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='sklearn-diffmap',
      version='0.1dev0',
      description='A scikit-learn compatible implementation of diffusion maps.',
      author='Cammie King',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='cam@cmking.io',
      )
