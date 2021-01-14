from setuptools import setup, find_packages


setup(
    name="RL",
    description="Reinforcement learning algorithms",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    url="https://github.com/deepmuseum/Algorithms-for-Reinforcement-Learning",
    author="Ayoub Oumani and Sofian Chaybouti",
    license="UNLICENSED",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
    # data_files=[("data", [''])],
    # Dependencies are pinned only to fix the environment during deployment.
    # There are no known issues with updating these versions in the future.
    install_requires=[],
    use_scm_version=True,
)
