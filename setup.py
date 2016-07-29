from setuptools import setup
from setuptools import find_packages

install_requires = [
    'Theano',
    'Keras',
    'seq2seq'
]

setup(
    name='SocratesBot',
    version='0.8.0',
    description='Chat bot based on sequence 2 sequence learning',
    author='Abhishek Rao',
    author_email='abhishek.rao.comm@gmail.com',
    url='https://github.com/abhishekraok/Socrates',
    license='GNU GPL v3',
    install_requires=install_requires,
    packages=find_packages()
)
