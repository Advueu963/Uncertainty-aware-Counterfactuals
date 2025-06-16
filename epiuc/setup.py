from setuptools import setup, find_packages

setup(
    name="epiuc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Add dependencies here if needed
    description="Epiuc package for uncertainty-aware processing",
    author="Santo Thies",
    author_email="S.Thies@campus.lmu.de",
    url="https://github.com/Advueu963/uncertainty-aware-cp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
