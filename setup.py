from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'gokart',
]

setup(
    name=m3lookeng,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description=Targeted Sentiment Analysis for medicine names in articles from m3.com,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=Tong Li,
    license=MIT License,
    packages=find_packages(),
    install_requires=install_requires,
    test_suite='test')
