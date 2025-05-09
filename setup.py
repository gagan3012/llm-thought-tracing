from setuptools import setup, find_packages

setup(
    name="ltr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "baukit>=0.1.2",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "ipython>=7.0.0",
        "pillow>=8.0.0",
    ],
    author="gbhat",
    author_email="example@example.com",
    description="LLM Thought Tracing: Mechanistic Interpretability for Neural Reasoning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VilohitT/llm-thought-tracing",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
