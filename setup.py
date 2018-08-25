
import glob
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


with open('README') as f:
    long_description = ''.join(f.readlines())


setup(
    name='glssm',
    version='0.1',
    description='Gaussian Latent State Space Models',
    author='Ye Liu, Zekun Xu',
    author_email='zekunxu@gmail.com',
    license='Public Domain',
    keywords='Gaussian hidden Markov models; dynamic linear models',
    url='https://github.com/Zekun-Jack-Xu/glssm',
    packages=find_packages(),
    zip_safe=False,
    ext_modules=cythonize(glob.glob('src/*.pyx'), language_level=3),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'Cython',
        'NumPy',
        'pandas',
        'scipy',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ]#,
    #setup_requires=['pytest-runner',],
    #tests_require=['pytest',],
)
