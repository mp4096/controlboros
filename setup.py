try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="controlboros",
    version="0.1",
    description="A simple framework for simulating control systems.",
    license="BSD",
    author="Mikhail Pak <mikhail.pak@tum.de>",
    packages=["controlboros"],
    install_requires=["numpy", "scipy"]
    )
