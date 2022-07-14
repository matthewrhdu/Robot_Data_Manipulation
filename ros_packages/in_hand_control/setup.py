from setuptools import setup

package_name = 'in_hand_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matthewrhdu',
    maintainer_email='matthewrhdu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "camera = in_hand_control.camera:main",
            "camera3d = in_hand_contol.camera3d:main",
            "processor = in_hand_control.processor:main",
            "processor3d = in_hand_control.processor3d:main",
            "controller = in_hand_control.controller:main",
            "robot = in_hand_control.robot:main",
        ],
    },
)
