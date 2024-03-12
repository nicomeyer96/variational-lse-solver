import setuptools

with open('README.md', 'r') as ff:
    long_description = ff.read()

setuptools.setup(
    name='variational-lse-solver',
    version='0.1',
    description='Comprehensive Library of Variational LSE Solvers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Fraunhofer IIS',
    license='Apache License 2.0',
    platforms='any',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
    ],
    packages=setuptools.find_packages(),
    python_requires='~=3.12',
    install_requires=[
        'pennylane~=0.34',
        'torch~=2.2.1',
        'tqdm~=4.66.2'
    ],
    include_package_data=True
)
