from setuptools import setup, find_packages
import temp_python

setup(
    name='veniq',
    version=temp_python.__version__,
    description=temp_python.__doc__.strip(),
    long_description='Govno',
    url='https://github.com/cqfn/veniq.git',
    download_url='https://github.com/cqfn/veniq.git',
    author=temp_python.__author__,
    author_email='yegor256@gmail.com',
    license=temp_python.__licence__,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'temp_python = temp_python.__main__:main'
        ],
    },
    extras_require={},
    install_requires=open('requirements.txt', 'r').readlines(),
    tests_require=open('requirements.txt', 'r').readlines(),
    classifiers=[
        'Programming Language :: Python',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development',
        'Topic :: Utilities'
    ],
    include_package_data=True,
)
