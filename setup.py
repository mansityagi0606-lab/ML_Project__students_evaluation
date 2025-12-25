from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="ML_Project_students_evaluation",
    version="0.0.1",
    author="Mansi",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
