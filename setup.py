import os
from glob import glob
from setuptools import setup

exec(open('nmrdata/version.py').read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='nmrgnn-data',
      version=__version__,
      scripts=glob(os.path.join('scripts', '*')),
      description='Chemical shift prediction dataset',
      author='Ziyue Yang, Andrew White',
      author_email='andrew.white@rochester.edu',
      url='https://github.com/ur-whitelab/nmrdata',
      license='MIT',
      packages=['nmrdata', 'nmrdata.data', 'nmrdata.parse'],
      install_requires=[
          'tensorflow >= 2.3',
          'click',
          'numpy>=1.18.5',
          'MDAnalysis >= 1.1.1',
          'importlib_resources'],
      extras_require={
          'parse': ['pdbfixer', 'biopython', 'gsd']
      },
      zip_safe=True,
      entry_points='''
        [console_scripts]
        nmrparse=nmrdata.parse.main:nmrparse
        nmrdata=nmrdata.main:nmrdata
            ''',
      package_data={'nmrdata': ['data/*.pb']},
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: Bio-Informatics",
          "Topic :: Scientific/Engineering :: Artificial Intelligence"
      ]
      )
