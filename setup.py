from setuptools import setup, find_packages

setup(
    name='your_app_name',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'Flask==2.0.1',
        'joblib==1.0.1',
        'numpy==1.21.0',
        'pandas==1.3.0',
        'scikit-learn==0.24.2',
        'setuptools==70.3.0',
        'six==1.16.0',
    ],
)
