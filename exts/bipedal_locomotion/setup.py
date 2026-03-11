"""Installation script for the 'bipedal_locomotion' python package."""

import itertools
import os
import sys

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_PATH = os.path.join(EXTENSION_PATH, "config", "extension.toml")

# Read TOML file with fallback support for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
    with open(EXTENSION_TOML_PATH, "rb") as f:
        EXTENSION_TOML_DATA = tomllib.load(f)
else:
    try:
        import tomli
        with open(EXTENSION_TOML_PATH, "rb") as f:
            EXTENSION_TOML_DATA = tomli.load(f)
    except ImportError:
        try:
            import toml
            with open(EXTENSION_TOML_PATH, "r") as f:
                EXTENSION_TOML_DATA = toml.load(f)
        except ImportError:
            raise ImportError(
                "Failed to import TOML reader. Please install one of: tomli, toml\n"
                "Run: pip install tomli"
            )

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    # "torch==2.4.0",
    # "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    # 5.26.0 introduced a breaking change, so we restricted it for now.
    # See issue https://github.com/tensorflow/tensorboard/issues/6808 for details.
    "protobuf>=3.20.2, < 5.0.0",
    # data collection
    "h5py",
    # basic logger
    "tensorboard",
    # video recording
    "moviepy",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

current_dir = os.path.dirname(os.path.abspath(__file__))
rsl_rl_path = os.path.join(current_dir, "..", "rsl_rl")

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {
    "rsl-rl": [rsl_rl_path],
}

# Add the names with hyphens as aliases for convenience
EXTRAS_REQUIRE["rsl_rl"] = EXTRAS_REQUIRE["rsl-rl"]

# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))

# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))

# Installation operation
setup(
    name="bipedal_locomotion",
    packages=["bipedal_locomotion"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)