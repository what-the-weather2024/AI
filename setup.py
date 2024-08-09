from setuptools import find_packages, setup

setup(
    name="wtw_ai",
    packages=find_packages(
        include=["AI", "AI.*"],
    ),
    python_requires=">=3.6.0",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={},
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
)
