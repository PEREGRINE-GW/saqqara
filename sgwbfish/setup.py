from setuptools import setup, find_packages

setup(
    name="sgwbfish",
    version="1.0.0",
    description="Simple Fisher forecasting code for SGWB analysis at LISA",
    author="James Alvey",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "jax"],
    package_data={"sgwbfish": ["data/*"]},
)
