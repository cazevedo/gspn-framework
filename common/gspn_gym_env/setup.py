from setuptools import setup, find_packages

setup(name='gspn_gym_env',
      version='1.0',
      description='OpenAI Gym that simulates multi-robot systems, modeled as generalized stochastic Petri nets.',
      url='github.com/cazevedo/gspn_gym_env',
      author='Carlos Azevedo',
      author_email='cguerraazevedo@tecnico.ulisboa.pt',
      packages=find_packages(),
      install_requires=['gym', 'numpy']
)