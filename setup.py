from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='alias_free_activation_cuda',
    ext_modules=[
        CUDAExtension(
            name='alias_free_activation_cuda',
            sources=['anti_alias_activation.cpp', 'anti_alias_activation_cuda.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
