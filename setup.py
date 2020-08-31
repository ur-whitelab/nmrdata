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
          'tensorflow == 2.3',
          'click',
          'numpy==1.18.5',
          'importlib_resources'],
      extras_require={
          'parse': ['pdbfixer @ https://github.com/openmm/pdbfixer/archive/master.zip', 'biopython', 'gsd']
      },
      zip_safe=True,
      entry_points='''
        [console_scripts]
        parse=nmrdata.parse.main:parse
        nmrdata=nmrdata.main:nmrdata
            ''',
      package_data={'nmrdata': ['data/*.pb']}
      )
