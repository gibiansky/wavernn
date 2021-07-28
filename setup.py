import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wavernn",
    version="1.0.0",
    author="Andrew Gibiansky",
    author_email="andrew.gibiansky@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gibiansky/wavernn",
    project_urls={
        "Bug Tracker": "https://github.com/gibiansky/wavernn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    scripts=["scripts/wavernn"],
)
