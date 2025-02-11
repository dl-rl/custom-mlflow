import os
from typing import List

from setuptools import find_packages, setup

import custom_mlflow

ROOT = os.path.abspath(os.path.dirname(__file__))
GITHUB_REPO_URL = "https://github.com/dl-rl/custom-mlflow"


def get_readme() -> str:
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_install_requires() -> List[str]:
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        return [
            line.strip()
            for line in f.readlines()
            if not (line.startswith("#") or (line.strip() == ""))
        ]


setup(
    name="custom-mlflow",
    version=custom_mlflow.__version__,
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=get_install_requires(),
    maintainer="dibyabiva",
    maintainer_email="ds@dl-rl.com",
    url=GITHUB_REPO_URL,
    project_urls={
        "Documentation": os.path.join(GITHUB_REPO_URL, "blob/main/README.md"),
        "Source Code": GITHUB_REPO_URL,
        "Bug Tracker": os.path.join(GITHUB_REPO_URL, "issues"),
    },
    description="Extend MLflow's functionality",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    classifiers=[],
)
