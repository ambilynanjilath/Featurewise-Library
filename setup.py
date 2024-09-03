from setuptools import setup, find_packages

setup(
    name='FeatureWise',  
    version='1.0.1',  
    description='A no-code solution for performing data transformations like imputation, encoding, scaling, and feature creation, with an intuitive interface for interactive DataFrame manipulation and easy CSV export.',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown', 
    author='Ambily Biju', 
    author_email='ambilybiju2408@gmail.com',  
    url='https://github.com/ambilynanjilath/Featurewise-Library.git',  
    packages=find_packages(), 
    package_data={
        '': ['featurewise/featurewise_logo.png'], 
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'run-FeatureWise=FeatureWise_app:main',
        ],
    },
    install_requires=[
        "pandas>=1.0.0",
        "scikit-learn>=0.22.0",
        "numpy",
        "pandas",
        "scipy",
        "streamlit",
        "st_aggrid",
    ],
    python_requires='>=3.8',  
    classifiers=[
        'Development Status :: 5 - Production/Stable',  
        'Intended Audience :: Developers',  
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',  
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='data transformation, imputation, encoding, scaling, feature creation, machine learning, data preprocessing, pandas, scikit-learn, feature engineering, data science, Python ',  
    project_urls={
        'Documentation': 'https://github.com/ambilynanjilath/Featurewise-Library/blob/main/README.md',
        'Source': 'https://github.com/ambilynanjilath/Featurewise-Library',
        'Tracker': 'https://github.com/ambilynanjilath/Featurewise-Library/issues',
    },
)