from setuptools import setup
from sakura import __version__

deps = [l.rsplit()
        for l in open("requirements.txt", "r") if not l.startswith("--")]
deps_url = [l.rsplit()[0].split("=")[1]
            for l in open("requirements.txt", "r") if l.startswith("--")]

setup(
    name="sakura-ml",
    version=__version__,
    short_description="Sakura provides asynchronous training for DNN.",
    long_description="Sakura provides asynchronous training for DNN.",
    url='https://zakuro.ai',
    packages=[
        "sakura",
        "sakura.ml",
        "sakura.ml.epoch",
    ],
    entry_points={
        "console_scripts": [
            "sakura=sakura:main"
        ]
    },
    license='ZakuroAI',
    author='ZakuroAI',
    python_requires='>=3.8',
    dependency_links=deps_url,
    install_requires=deps,
    author_email='info@zakuro.ai',
    description='Sakura provides asynchronous training for DNN.',
    platforms="linux_debian_10_x86_64",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
