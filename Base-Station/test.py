import random, hashlib
import datetime
import pandas as pd
# for i in range(5):
#     val = random.randint(0, 50)
#     f = open(f"cached_file/{val}.txt", "w")
#     file = open(f'AllFiles/{val}.txt', 'r')
#     f.write(file.read())
#     file.close()
#     f.close()
lead_hits = pd.read_csv(f"DPP/experiments/csv_10000/LeadCache_Hits.csv")

