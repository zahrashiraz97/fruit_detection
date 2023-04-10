from setuptools import setup, find_packages

setup(
    name="mmwave",
    version="0.0.1",
    description="Python implementation of MMWave radar driver",
    # package_dir={"": "python"},
    # packages=find_packages(),
    packages=["mmwave", "mmwave.cli", "mmwave.utils"],
    tests_requires=["unittest"],
    install_requires=["pyserial==3.5", "usb"],
    python_requires=">=3.6"
)
