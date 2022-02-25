from setuptools import setup, find_packages

setup(
    name='nerf_tutorial',
    version='0.1.0',
    description='Tutorial scripts for NeRF',
    url='https://github.com/ALBERT-Inc/NeRF-tutorial',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
