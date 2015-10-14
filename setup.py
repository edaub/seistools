from setuptools import setup

setup(name='seistools',
      version='0.1',
      description='Tools useful for earthquake analysis',
      url='http://bitbucket.org/egdaub/seistools',
      author='Eric Daub',
      author_email='egdaub@memphis.edu',
      packages=['seistools'],
      install_requires=['numpy', 'scipy'],
      test_suite='nose.collector',
      tests_require=['nose'])
