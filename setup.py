from setuptools import setup, find_packages

setup(
    name='AlzPy', 
    version='0.1.0',  
    author='Owen Sweeney',  
    author_email='owensweeney97@gmail.com',  
    description='A library for data processing and analysis.',
    
    url='https://github.com/owensweeney97/AlzPy/tree/main',  
    packages=find_packages(),
    install_requires=[
        'pandas>=1.1.0',
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.23.0',
        'statsmodels>=0.12.0',
        'matplotlib>=3.2.0',
        'patsy>=0.5.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

