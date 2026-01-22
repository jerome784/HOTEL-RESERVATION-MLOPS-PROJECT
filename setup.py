from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements=f.read().splitlines()


setup(

    name="hotel_reservation_mlops",
    version="0.1",
    author="Jerome Philip John",
    packages=find_packages(),
    install_requires=requirements
)