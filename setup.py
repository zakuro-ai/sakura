from setuptools import setup
from sakura import __version__

setup(
    name="sakura-ml",
    version=__version__,
    short_description="Sakura provides asynchronous training for DNN.",
    long_description="".join(open("README.md", "r").readlines()),
    long_description_content_type="text/markdown",
    url='https://zakuro.ai',
    packages=[
        "sakura",
        "sakura.lightning",
    ],
    include_package_data=True,
    package_data={"": ["*.yml"]},
    install_requires=[r.rsplit()[0] for r in open("requirements.txt")],
    license='MIT',
    author='ZakuroAI',
    python_requires='>=3.6',
    author_email='git@zakuro.ai',
    description='Sakura provides asynchronous training for DNN.',
    platforms="linux_debian_10_x86_64",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
