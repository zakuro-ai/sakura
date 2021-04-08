from setuptools import setup
import json

setup(
    name="sakura-ml",
    version="0.0.1",
    short_description="Sakura provides asynchronous training for DNN.",
    long_description="Sakura provides asynchronous training for DNN.",
    url='https://zakuro.ai',
    packages=[
        "sakura",
        "sakura.ml",
        "sakura.ml.decorators",
    ],
    license='ZakuroAI',
    author='ZakuroAI',
    python_requires='>=3.6',
    install_requires=[l.rsplit() for l in open("requirements.txt", "r")],
    author_email='info@zakuro.ai',
    description='Sakura provides asynchronous training for DNN.',
    platforms="linux_debian_10_x86_64",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)

