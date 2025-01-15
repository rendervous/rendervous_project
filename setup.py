from setuptools import setup, find_packages

setup(
    name="rendervous",
    version="1.0.0",                       # Semantic versioning
    author="Ludwic Leonard",
    author_email="lleonart1984@gmail.com",
    description="Differentiable Renderer with Vulkan and Pytorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rendervous/rendervous_project",
    packages=find_packages(where="src"),  # Look for packages in "src"
    package_dir={"": "src"},  # Map the root package directory to "src"
    # packages=find_packages(),             # Automatically find packages in your project
    install_requires=[
        "vulky @ git+https://github.com/rendervous/vulky_project.git"
    ],
    include_package_data=True,
    extras_requires={
        "gui": [
            "pywin32; platform_system == 'Windows'",
            "pygobject; platform_system == 'Linux'",
            "pyobjc; platform_system == 'Darwin'",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',              # Specify supported Python versions
)