from setuptools import setup, find_packages  
import os  

# Read the long description from README.md  
with open("README.md", "r", encoding="utf-8") as f:  
    long_description = f.read()  

# Get the version from bloombee/__init__.py  
def get_version():  
    init_path = os.path.join("src", "bloombee", "__init__.py")  
    with open(init_path, "r", encoding="utf-8") as f:  
        for line in f:  
            if line.startswith("__version__"):  
                return line.split("=")[1].strip().strip('"')  
    raise RuntimeError("Unable to find version string.")  

setup(  
    name="bloombee",  
    version=get_version(),  
    author="Sophie",  
    author_email="syang127@ucmerced.edu",  
    description="A short description of your project",  
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    url="https://github.com/yottalabsai/bloombee",  
    project_urls={  
        "Bug Tracker": "https://github.com/yottalabsai/bloombee/issues",  
    },  
    classifiers=[  
        "Development Status :: 4 - Beta",  
        "Intended Audience :: Developers",  
        "Intended Audience :: Science/Research",  
        "License :: OSI Approved :: Apache Software License",  
        "Programming Language :: Python :: 3",  
        "Programming Language :: Python :: 3.8",  
        "Programming Language :: Python :: 3.9",  
        "Programming Language :: Python :: 3.10",  
        "Programming Language :: Python :: 3.11",  
        "Topic :: Scientific/Engineering",  
        "Topic :: Scientific/Engineering :: Mathematics",  
        "Topic :: Scientific/Engineering :: Artificial Intelligence",  
        "Topic :: Software Development",  
        "Topic :: Software Development :: Libraries",  
        "Topic :: Software Development :: Libraries :: Python Modules",  
    ],  
    package_dir={"": "src"},  
    packages=find_packages(where="src"),  
    python_requires=">=3.8",  
    install_requires=[  
        "torch>=1.12",  
        "bitsandbytes==0.46.0",  
        "accelerate>=0.27.2",  
        "huggingface-hub>=0.11.1,<1.0.0",  
        "tokenizers>=0.13.3",  
        "transformers==4.43.1",  
        "speedtest-cli==2.1.3",  
        "hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9",  
        "tensor_parallel==1.0.23",  
        "humanfriendly",  
        "async-timeout>=4.0.2",  
        "cpufeature>=0.2.0; platform_machine == 'x86_64'",  
        "packaging>=20.9",  
        "sentencepiece>=0.1.99",  
        "peft==0.8.2",  
        "safetensors>=0.3.1",  
        "Dijkstar>=2.6.0",
        "numpy<2",
        "attrs",
        "nvidia-ml-py"
    ],  
    extras_require={  
        "dev": [  
            "pytest==6.2.5",  
            "pytest-forked",  
            "pytest-asyncio==0.16.0",  
            "black==22.3.0",  
            "isort==5.10.1",  
            "psutil",  
        ],  
    },  
)