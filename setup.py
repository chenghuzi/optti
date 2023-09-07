from setuptools import setup, find_packages
import platform


def get_simnibs_url(version):
    python_version = platform.python_version()[:3].replace('.', '')
    system = platform.system().lower()
    machine = platform.machine().lower()
    platform_url = f"cp{python_version}-cp{python_version}-{system}_{machine}"

    simnibs_url = f'https://github.com/simnibs/simnibs/releases/download\
/v{version}/simnibs-{version}-{platform_url}.whl'
    return simnibs_url


setup(
    name='optti',
    version='0.1',
    packages=find_packages(
        exclude=['tests', 'tests.*', 'local.*', 'models', 'wandb']),

    install_requires=[
        f'simnibs @ {get_simnibs_url("4.0.1")}',
        'numpy',
        'torch',
        'tqdm',
        'pandas',
        'meshio',
        'matplotlib',
    ],
    package_data={
        'optti': ['params/*'],
    },
    # entry_points={
    #     'console_scripts': [
    #         'eye-ti=eye_ti.main:main',
    #     ],
    # },
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for optimizing TI electrode placement',
    url='https://github.com/chenghuzi/oppti',
)
