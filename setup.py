from setuptools import setup

setup(
    packages=[
        "lw_benchhub",
        "policy",
        "lw_benchhub_tasks",
        "lw_benchhub_rl"
    ],
    package_dir={
        "lw_benchhub": "lw_benchhub",
        "lw_benchhub_policy": "lw_benchhub_policy",
        "lw_benchhub_tasks": "lw_benchhub_tasks",
    }
)
