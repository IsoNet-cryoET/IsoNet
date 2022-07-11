#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='IsoNet',
    version=0.2,
    description='IsoNet isotropic reconstruction',
    url='https://github.com/Heng-Z/IsoNet',
    license='MIT',
    packages=['IsoNet'],
    package_dir={
        'IsoNet': '.',
    },
    entry_points={
        "console_scripts": [
            "isonet.py = IsoNet.bin.isonet:main",
        ],
    },
    include_package_data = True,
    install_requires=[
        'tensorflow',
        'mrcfile',
        'PyQt5',
        'matplotlib',
        'fire',
        'scikit-image==0.17.2',
        'tqdm',
    ]
)
