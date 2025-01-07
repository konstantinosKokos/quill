from setuptools import setup, find_packages

setup(
    name='agda-quill',
    version='0.1.0',
    author='Konstantinos Kogkalidis',
    description='Neural premise selection for Agda.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/konstantinosKokos/quill',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.4',
        'torch-geometric>=2.6',
        'flask>=2.2',
        'requests'
    ],
    python_requires='>=3.11',
    entry_points={
        'console_scripts': [
            'agda-quill=quill.api.,aom:main'
        ]
    }
)
