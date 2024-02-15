from setuptools import setup, find_packages

setup(
  name = 'SAC-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Soft Actor Critic',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/SAC-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'reinforcement learning',
    'soft actor critic'
  ],
  install_requires=[
    'beartype',
    'einops>=0.7.0',
    'einx[torch]>=0.3.0',
    'ema-pytorch',
    'pytorch-custom-utils>=0.0.18',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
