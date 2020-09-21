#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# for your packages to be recognized by python
d = generate_distutils_setup(
 packages=['gspn_framework_package', 'gspn_framework_package_ros'],
 package_dir={'gspn_framework_package': 'common/src/gspn_framework_package',
              'gspn_framework_package_ros': 'ros/src/gspn_framework_package_ros'}
)

setup(**d)
