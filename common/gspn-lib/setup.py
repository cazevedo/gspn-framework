from setuptools import setup, find_packages

setup(name='gspn_lib',
      version='1.0',
      description='Lib to create, manipulate and simulate generalized stochastic Petri nets.',
      url='github.com/cazevedo/gspn-lib',
      author='Carlos Azevedo',
      author_email='cguerraazevedo@tecnico.ulisboa.pt',
      # install_requires=['numpy', 'sparse', 'json', 'flask', 'pandas', 'graphviz', 'ast'],
      install_requires=['numpy', 'sparse', 'graphviz', 'pathlib'],
      packages=['gspn_lib'],
      zip_safe=False
)