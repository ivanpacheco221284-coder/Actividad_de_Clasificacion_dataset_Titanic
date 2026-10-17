# scripts to download Titanic dataset
import requests
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
r = requests.get(url)
open('Activity5/datos/datos_ini/titanic.csv','wb').write(r.content)
print('Downloaded titanic.csv')
