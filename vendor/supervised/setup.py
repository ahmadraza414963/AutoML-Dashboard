from setuptools import setup, find_packages

setup(
    name="supervised",
    version="0.11.5",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.7.1",
)
