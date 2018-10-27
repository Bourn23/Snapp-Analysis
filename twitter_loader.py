import pandas as pd

data = pd.read_csv("hashtags.csv", delimiter=',', usecols=['f'])

print(data.head())
"""
w : write,
r : read, 
a : append,
w+ : misaze, 
a+ : misaze """
with open('text.txt','w+') as f:
    for i in data['f']:
        f.write(i + "\n")
    f.close()
