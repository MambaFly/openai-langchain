import sys
print(sys.getdefaultencoding())
with open('tests/resume.pdf', 'r', encoding='utf-8') as file:
    content = file.read()
