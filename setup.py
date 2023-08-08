from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='erpy',
    version='0.1.0',
    description='Evolutionary Robotics Python -- Brain-body co-optimisation framework',
    long_description=readme,
    url='https://github.com/Co-Evolve/erpy',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=required
)
