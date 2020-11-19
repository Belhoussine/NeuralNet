from setuptools import setup, find_packages

setup(
    name='NeuralNet',
    packages=find_packages(include=['NeuralNet']),
    version='0.1.0',
    description='A Machine Learning library for Neural Networks fully written in python. It supports multiple layers of neurons and offers a variety of activation functions, optimization algorithms, and utility functions.',
    author='Othmane Belhoussine',
    author_email='othmane.belhou@gmail.com',
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    install_requires=['numpy==1.19.2','urllib3==1.25.8'],
    test_suite='tests',
)
