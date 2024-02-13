from setuptools import setup, find_packages

#with open('README.rst', 'r') as fh:
#    _description = fh.read()

setup(
    name='interactivenet',
    version='0.2.1',
    author = 'Biomedical Imaging Group Rotterdam (BIGR)',
    license='Apache License, Version 2.0',
    author_email='d.spaanderman@erasmusmc.nl',
    description='InteractiveNet, a framework for minimally interactive medical image segmentation.',
    #long_description=_description,
    url='https://github.com/Douwe-Spaanderman/InteractiveNet',
    download_url = 'https://github.com/Douwe-Spaanderman/InteractiveNet/archive/refs/tags/v0.2.1.tar.gz',
    packages=find_packages(include=['interactivenet', 'interactivenet.*']),
    python_requires='>3.9.0',
    install_requires=[
        "SimpleITK>=2.1.1.2",
        "GeodisTK>=0.1.7",
        "matplotlib>=3.5.1",
        "mlflow>=1.24.0",
        "monai>=1.0.0,<1.3.0",
        "nibabel>=3.2.2",
        "numpy>=1.22.3",
        "Pillow>=9.2.0",
        "psutil>=5.9.0",
        "pytorch_lightning>=1.5.10,<2.0.0",
        "scikit_image>=0.19.2",
        "scikit_learn>=1.1.1",
        "scipy>=1.8.0",
        "seaborn>=0.12.2",
        "pyradiomics>=3.0.1"
    ],
    setup_requires=['pytest', 'black', 'autoflake', 'setuptools'],
    entry_points={
        'console_scripts': [
            'interactivenet_mimic_interactions=interactivenet.experiment_planning.mimic_annotations:main',
            'interactivenet_generate_dataset_json=interactivenet.experiment_planning.generate_dataset_json:main',
            'interactivenet_fingerprinting=interactivenet.experiment_planning.fingerprinting:main',
            'interactivenet_preprocessing=interactivenet.experiment_planning.preprocessing:main',
            'interactivenet_plan_and_process=interactivenet.experiment_planning.plan_and_process:main',
            'interactivenet_train=interactivenet.training.run:main',
            'interactivenet_postprocessing=interactivenet.training.postprocessing:main',
            'interactivenet_ensemble=interactivenet.test.ensemble:main',
            'interactivenet_inference=interactivenet.test.inference:main',
            'interactivenet_predict=interactivenet.test.predict:main',
            'interactivenet_test=interactivenet.test.run:main',
            'interactivenet_deploy=interactivenet.deploy.save_model:main',
            'interactivenet_available_models=interactivenet.deploy.download_model:print_available_pretrained_models',
            'interactivenet_download_model=interactivenet.deploy.download_model:download_and_install_model',
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
