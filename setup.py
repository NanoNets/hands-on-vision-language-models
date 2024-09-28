from setuptools import setup, find_packages

setup(
    name="hands-on-vision-language-models",
    version="0.1.0",
    packages=find_packages("src/"),
    package_dir={"": "src"},
    install_requires=[
        "torch-snippets",
    ],
    entry_points={
        "console_scripts": [
            "vlm = vlm.cli:cli",
        ],
    },
    author="sizhky",
    author_email="yeshwanth@nanonets.com",
    description="A project for hands-on vision and language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sizhky/hands-on-vision-language-models",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
