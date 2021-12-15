from setuptools import setup, find_packages

setup(
    name='bites',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/sschrod/BITES',
    license='BSD 2-Clause "Simplified" License',
    author='sschrod',
    author_email='stefan.schrod@bioinf.med.uni-goettingen.de',
    description='bites: Balanced individual treatment effect for survival data'
)
