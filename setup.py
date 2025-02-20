from setuptools import setup, find_packages

# 读取 README 中的内容作为项目的长描述（如果存在 README.md 文件）
def readme():
    try:
        with open('README.md', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''

setup(
    name='Triangle_BBH',  # 项目的名称，建议使用文件夹的名称
    version='0.1.0',  # 项目的版本号，建议遵循语义化版本规则
    description='A collection of Python scripts for Binary Black Hole (BBH) analysis',  # 项目简短描述
    long_description=readme(),  # 项目的长描述 (通常为 README.md 的内容)
    long_description_content_type='text/markdown',  # 长描述的类型 (与 README.md 格式匹配)
    url='https://github.com/yourusername/Triangle_BBH',  # 项目的主页或代码仓库地址
    author='Your Name',  # 作者信息
    author_email='your.email@example.com',  # 作者的电子邮箱
    license='MIT',  # 许可证类型 (可选，示例用 MIT 许可证)
    packages=find_packages(),  # 自动发现所有包含 `__init__.py` 的子包
    # py_modules=['script1', 'script2'],  # 如果是单文件模块，这里列出模块文件（不要写 `.py` 后缀）
    classifiers=[
        'Programming Language :: Python :: 3',  # 支持的 Python 版本
        'License :: OSI Approved :: MIT License',  # 项目许可证
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 指定兼容的 Python 版本
    # install_requires=[  
    #     # 在这里列出你的项目依赖的第三方库（会通�� pip 自动安装）
    #     # 示例：
    #     # 'numpy',
    #     # 'scipy'
    # ],
    # entry_points={  # 可选：用于创建命令行工具
    #     'console_scripts': [
    #         # 示例：将 main.py 中的 main 函数作为命令 `triangle-run` 提供
    #         # 'triangle-run=Triangle_BBH.main:main',
    #     ],
    # },
    include_package_data=False,  # 如果有额外的文件需要包含（如非 Python 文件，可以结合 MANIFEST.in 使用）
    # zip_safe=False,  # 设置为 False 表示项目不能以 zip 形式安装（推荐）
)