from setuptools import setup

setup(name='seistools',
      version='0.1',
      description='Tools useful for earthquake analysis',
      author='Eric Daub',
      author_email='egdaub@memphis.edu',
      packages=['seistools'],
      install_requires=['numpy', 'scipy', 'statsmodels'])
