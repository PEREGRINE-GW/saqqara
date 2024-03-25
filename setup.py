# setup.py
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="saqqara",
    version="0.0.2",
    packages=find_packages(),
    author="James Alvey, Uddipta Bhardwaj, Mauro Pieroni",
    author_email="j.b.g.alvey@uva.nl;ubhardwaj.gravity@gmail.com;mauroemail@gmail.com",
    description="saqqara is a simulation-based Inference (SBI) library designed for stochastic gravitational wave background data analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "healpy",
        "chex",
        "gw_response",
    ],
    package_data={"saqqara": ["defaults/*"]},
)
