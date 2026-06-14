from setuptools import setup, find_packages, Extension
import os
import sys

# ------------------------------------------------------------------ #
# Native C extension  (CPython API, replaces ctypes when available)  #
# ------------------------------------------------------------------ #
ext_modules = []
cengine_src = [
    "_cengine.c",
    "lib/src/rin_core.c",
    "lib/src/thorin_api.c",
    "thorin/backends/neon_kernels.c",
    "thorin/backends/wasm_kernels.c",
    "thorin/backends/detect.c",
]
if all(os.path.isfile(s) for s in cengine_src):
    have_python_h = False
    try:
        from distutils.sysconfig import get_config_var
        inc_py = get_config_var("INCLUDEPY")
        have_python_h = inc_py and os.path.isfile(os.path.join(inc_py, "Python.h"))
    except Exception:
        inc_py = None
    try:
        if not have_python_h and inc_py:
            have_python_h = os.path.isfile(os.path.join(inc_py, "Python.h"))
    except Exception:
        pass
    ext_modules.append(
        Extension(
            "thorin._cengine",
            cengine_src,
            include_dirs=[".", "lib/include", "lib/include/experimental"],
            extra_compile_args=[
                "-O3", "-march=native", "-mtune=znver3",
                "-mavx2", "-mfma",
                "-fopenmp", "-funroll-loops", "-flto", "-ffast-math",
                "-D_GNU_SOURCE", "-DTHOR_BUILD_PYTHON",
            ],
            extra_link_args=["-fopenmp", "-lm", "-lrt", "-flto"],
        )
    )

# ------------------------------------------------------------------ #
# Shared library  (fallback for ctypes binding)                      #
# ------------------------------------------------------------------ #
# Also compile librin.so so the ctypes path still works
# This is built separately via Makefile, not via setup.py

setup(
    name="thorin",
    version="1.0.0",
    description="THOR Neural Runtime - Universal Inference Engine",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="THOR Team",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["numpy"],
    extras_require={
        "pytorch": ["torch>=1.10"],
        "onnx": ["onnx>=1.12"],
        "full": ["torch>=1.10", "onnx>=1.12"],
    },
    entry_points={
        "console_scripts": [
            "thorin=thorin.cli.main:main",
        ],
    },
    scripts=[],  # ensure scripts dir is created
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
)
