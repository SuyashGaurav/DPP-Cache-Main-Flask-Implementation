import requests
import pandas as pd
from tqdm import tqdm

path_to_input = '311_dataset.txt'
data = pd.read_csv(path_to_input, sep = ' ')
data.columns = ['Timestamp', 'File_ID', "File_Size"]
DataLength = len(data)

for i in tqdm(range(DataLength)):
    url = "http://192.168.137.150:5000/"
    r = requests.post(url, data={'content': f"{data['File_ID'][i]}"})
    url1 = f"http://192.168.137.150:5000/download/{data['File_ID'][i]}"
    r1 = requests.get(url1, data={'content': f"{data['File_ID'][i]}"})
    fp = open(f"downloaded_files/{data['File_ID'][i]}.txt", "wb")
    fp.write(r1.content)
    fp.close()