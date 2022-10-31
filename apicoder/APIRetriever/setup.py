from setuptools import setup, find_packages

setup(
    name='apiretriever',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    install_requires=open('requirements.txt').read().splitlines(),
    url='https://github.com/microsoft/PyCodeGPT',
    license='Apache 2.0',
    author='MSRA-DKI',
    author_email='daoguang@iscas.ac.cn',
    description='A toolkit for learning and running deep dense retrieval models.'
)
