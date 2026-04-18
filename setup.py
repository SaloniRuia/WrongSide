"""
Minimal setup.py so the package can be installed with:
    pip install -e .
This removes the need for sys.path.insert() hacks in main.py.
"""
from setuptools import setup, find_packages

setup(
    name="wrong_way_detection",
    version="1.1.0",
    description="GPS + OSM wrong-way driver detection system",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "folium>=0.15.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "wwd=main:main_cli",
        ]
    },
)
