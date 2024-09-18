from setuptools import setup, find_packages

setup(
    name='your_library',  # Replace with your library's name
    version='0.1.0',  # Initial release version
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    description='A library for data processing and analysis.',
    long_description=open('README.md').read(),  # Ensure you have a README.md
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_library',  # Replace with your library's URL
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
        'License :: OSI Approved :: MIT License',  # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

