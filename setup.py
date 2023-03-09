from setuptools import setup, find_packages

setup(
    name='interactivenet',
    version='1.0.0',
    description='InteractiveNet, a framework for minimally interactive medical image segmentation.',
    url='https://github.com/Douwe-Spaanderman/InteractiveNet',
    author='Biomedical Imaging Group Rotterdam (BIGR)',
    author_email='d.spaanderman@erasmusmc.nl',
    packages=find_packages(include=['interactivenet', 'interactivenet.*']),
    install_requires=[
        "SimpleITK>=2.1.1.2",
        "GeodisTK>=0.1.7",
        "matplotlib>=3.5.1",
        "mlflow>=1.24.0",
        "monai>=1.0.0",
        "nibabel>=3.2.2",
        "numpy>=1.22.3",
        "Pillow>=9.2.0",
        "psutil>=5.9.0",
        "pytorch_lightning>=1.5.10",
        "scikit_image>=0.19.2",
        "scikit_learn>=1.1.1",
        "scipy>=1.8.0",
    ],
    setup_requires=['pytest', 'black', 'setuptools'],
    entry_points={
        'console_scripts': [
            'interactivenet_mimic_interactions=interactivenet.experiment_planning.mimic_annotations:main',
            'interactivenet_generate_dataset_json=interactivenet.experiment_planning.generate_dataset_json:main',
            'interactivenet_fingerprinting=interactivenet.experiment_planning.fingerprinting:main',
            'interactivenet_plan_and_process=interactivenet.experiment_planning.plan_and_process:main',
            'interactivenet_preprocessing=interactivenet.experiment_planning.preprocessing:main'
            ]
    },
    keywords=[
        "deep learning", 
        "medical image analysis",
        "interactive segmentation",
        "medical image segmentation",
        "soft-tissue tumors",
        "interactivenet"
    ]
)