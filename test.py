# try:
#     import torchtext
#     print("torchtext 已安装，版本为:", torchtext.__version__)
# except ImportError:
#     print("torchtext 未安装")

# import dill
# print(dill.__version__)

# import os

# def find_file(root_dir, target_file):
#     """
#     在指定目录及其子目录中查找目标文件，并返回其路径。
#     如果找不到文件，返回 None。
#     """
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         if target_file in filenames:
#             return os.path.join(dirpath, target_file)
#     return None

# # 当前目录
# current_dir = os.getcwd()

# # 目标文件名
# target_file = "english.txt"

# # 查找文件
# file_path = find_file(current_dir, target_file)

# if file_path:
#     print(f"文件 '{target_file}' 存在于: {file_path}")
# else:
#     print(f"未找到文件 '{target_file}'")


# print("当前工作目录:", os.getcwd())

# 验证en_core_web_sm是不是正确安装
import spacy
import pkg_resources

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a test sentence.")
print([(w.text, w.pos_) for w in doc])

# 获取模型的版本信息
model_version = pkg_resources.get_distribution("en_core_web_sm").version

print(f"模型版本: {model_version}")

"""
输出为：
[('This', 'PRON'), ('is', 'AUX'), ('a', 'DET'), ('test', 'NOUN'), ('sentence', 'NOUN'), ('.', 'PUNCT')]
This: 词性标注为 PRON，表示这是一个代词（Pronoun）。
is: 词性标注为 AUX，表示这是一个助动词（Auxiliary Verb）。
a: 词性标注为 DET，表示这是一个限定词（Determiner）。
test: 词性标注为 NOUN，表示这是一个名词（Noun）。
sentence: 词性标注为 NOUN，表示这是一个名词（Noun）。
.: 词性标注为 PUNCT，表示这是一个标点符号（Punctuation）。
"""

import spacy

nlp = spacy.load("fr_core_news_sm")
doc = nlp("Ceci est une phrase de test.")
print([(w.text, w.pos_) for w in doc])