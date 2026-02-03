from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_onnx_inference'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='F1TENTH Team',
    maintainer_email='your_email@example.com',
    description='ONNX inference node for F1TENTH autonomous racing',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = f1tenth_onnx_inference.inference_node:main',
        ],
    },
)
