from setuptools import setup

setup(
    packages=[
        "lwlab",
        "policy",
        "tasks"
    ],
    package_dir={
        "lwlab": "lwlab",
        "policy": "policy",
        "tasks": "tasks"
    }
)
