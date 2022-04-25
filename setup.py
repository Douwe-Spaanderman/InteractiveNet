from setuptools import setup, find_packages

setup(
    name='interactivenet',
    version='0.0.1',
    description='Setting up a python package',
    author='Douwe Jan Spaanderman',
    author_email='d.spaanderman@erasmusmc.nl',
    packages=find_packages(include=['interactivenet', 'interactivenet.*']),
    #install_requires=[
    #],
    #extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    #setup_requires=['pytest-runner', 'flake8'],
    #tests_require=['pytest'],
    #entry_points={
    #    'console_scripts': ['my-command=exampleproject.example:main']
    #},
    #package_data={'exampleproject': ['data/schema.json']}
)