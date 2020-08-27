import os
from glob import glob
from setuptools import setup

exec(open('nmrdata/version.py').read())

setup(name='nmrdata',
      version=__version__,
      scripts=glob(os.path.join('scripts', '*')),
      description='Chemical shift predictor',
      author='Ziyue Yang, Andrew White',
      author_email='andrew.white@rochester.edu',
      url='http://thewhitelab.org/Software',
      license='MIT',
      packages=['nmrdata'],
      install_requires=[
          'tensorflow >= 2.3',
          'click',
          'numpy',
          'importlib_resources'],
      extra_requires={
          'parse': ['openmm', 'pdbfixer', 'rdkit', 'biopython', 'gsd']
      },
      dependency_links=[
          'https://github.com/openmm/pdbfixer/archive/master.zip'],
      zip_safe=True,
      entry_points='''
        [console_scripts]
        nmrdata=nmrdata.main:main
            ''',
      package_data={'nmrdata': ['data/*.pb']}
      )
