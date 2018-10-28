import re
pattern = r'[A-Za-z]'

with open ('./edited-text.txt', 'w') as f1:
    with open('../text.txt', 'r') as f:
      for line in f:
            f1.writelines(re.sub(pattern, '', line))
