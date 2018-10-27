import re
pattern = r'([A-Za-z]+)'
regex = re.compile(pattern)


with open('../text.txt') as f:
    for line in f:
        regex.sub(' ', line)
