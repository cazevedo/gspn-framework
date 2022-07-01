from setuptools import setup, find_packages

setup(name='gspn_framework',
      version='1.0',
      description='Package to create, manipulate and simulate generalized stochastic Petri nets.',
      url='github.com/cazevedo/gspn-framework',
      author='Carlos Azevedo',
      # install_requires=['numpy', 'sparse', 'json', 'flask', 'pandas', 'graphviz', 'ast'],
      install_requires=['numpy', 'sparse', 'graphviz', 'pandas'],
      author_email='cguerraazevedo@tecnico.ulisboa.pt',
      packages=find_packages(),
      zip_safe=False
)