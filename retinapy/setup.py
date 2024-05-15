import setuptools 

setuptools.setup(
    name='retinapy',
    version='0.0.1.dev9',
    author='Baden Lab members',
    author_email='kevin@kdoran.com',
    description=('A package for working with MEA recordings of retina activity'),
    packages=['retinapy'],
    install_requires=[
        'bidict',
        'configargparse',
        'deprecated',
        'einops',
        'h5py',
        'kaleido',
        'numpy',
        'pandas',
        'pillow',
        'plotly',
        'polars',
        'pyarrow',
        'pyyaml',
        'scinot',
        'scikit-learn',
        'semantic_version',
        'scipy',
        'tensorboard',
        'torch',
        'torchinfo',
        'tqdm',
        'typer']
    ,
    package_dir={'': 'src'},
    include_package_data=True,
    license='BSD-3',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research']
)

