import os
import time
import math
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from loader.load_data import *
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from optimizers.network import *
from optimizers.constrained import constrained_solve
import requests
import matplotlib.pyplot as plt

import sys
import hashlib
#-------------------PLOTS--------------------------
def ma(X, w=None):
    if(not w): w = len(X)
    avgs = []
    for i in range(len(X)):
        numer = np.sum(X[max(0, i-w+1):i+1])
        den = min(w, i+1)
        avgs.append(numer/den)
    return np.array(avgs)

#-------------------------------------other-------------------
from loader.load_data import *
from algorithms.offline_opt import MIN 
from algorithms.LFU import Bipartite_LFU
from algorithms.LRU import Bipartite_LRU
from algorithms.Lead_cache import Lead_cache
from optimizers.constrained import constrained_solve_ftpl
from algorithms.Perturbed_LFU import Perturbed_Bipartite_LFU
from algorithms.Generate_network import generate_network_graph

#--------------------------------------other--------------------

BUF_SIZE = 65536

sha256 = hashlib.sha256()

def env_hash():
    with open("./.env", 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

load_dotenv()

Q = int(os.getenv("Q_INIT"))
past = int(os.getenv("PAST"))
V_0 = int(os.getenv("V_0"))
future = int(os.getenv("FUTURE"))
alpha = float(os.getenv("ALPHA"))
NumSeq = int(os.getenv("NUM_SEQ"))
threshold = int(os.getenv("THRESHOLD"))
train_memory = int(os.getenv("TRAIN_MEMORY"))
use_saved = os.getenv("USE_SAVED")=="True"
cost_constraint = int(os.getenv("COST_CONSTRAINT"))
time_limit = float('inf') if os.getenv("TIME_LIMIT")=='inf' else int(os.getenv("TIME_LIMIT"))
path_to_input = os.getenv("PATH_TO_INPUT")
tag = env_hash()[:10]
print("Experiment Tag:", tag)

cache_constraint = int(alpha*threshold)

path = f"./experiments/csv_{NumSeq}/"
try:
    os.makedirs(path)
except FileExistsError:
    pass

our_path = f"./experiments/{tag}/"
try:
    os.makedirs(our_path)
except FileExistsError:
    pass
copyfile("./.env", our_path+"/.env")

#--------------------------------------other--------------------
hit_rate_ftpl = []
download_rate_ftpl = []
prev_demands_run_others = [[0 for i in range(threshold)]]
X_t_1_ftpl = np.zeros((threshold,))
init_indices_run_others = random.sample(range(threshold), cache_constraint)
X_t_1_ftpl[init_indices_run_others] = 1
gamma_run_others = np.random.normal(0, 1, (threshold,))
    

users = 1
caches = 1
d = 1
# Dropping all file requests with id larger than the threshold to reduce the library size
print("Users=", users, "caches=", caches, "Library_Size=", threshold, "time=", time_limit, "NumSeq=", NumSeq, file=open("parameters.log","w"))

# generates a random network
Adj = generate_network_graph(users, caches, d)

# saves the network 
print(Adj, file=open("network_adjacency_matrix.log", "w"))

# Setting up the arrays to store hits and downloads over multiple runs
LFU_Hits = []
perturbed_LFU_Hits = []
perturbed_LFU_Downloads = []
LRU_Hits = []
LeadCache_Hits = []
LFU_Downloads = []
LRU_Downloads = []
LeadCache_Downloads = []
OPT_Hits=[]
OPT_Downloads = []
LeadCache_Hits_Madow = []
LeadCache_Downloads_Madow = []

#--------------------------------------DPP--------------------

data = pd.read_csv(path_to_input, sep = ' ')
data.columns = ['Timestamp', 'File_ID', "File_Size"]
# DataLength = len(data)
DataLength = DataLength_run_others = 659963
# print(len(data)+10)
# print(int((0+1)*DataLength/NumSeq))


gamma = np.random.normal(0, 1, (threshold,))



queue = []
err = []
objective = []
fetching_cost = []
cache_hit = []
prev_demands = []
best_maximum = []
hit_rate = []
download_rate = []


X_t_1 = np.zeros((threshold,))
init_indices = random.sample(range(threshold), cache_constraint)
X_t_1[init_indices] = 1
cachePath = f"../cached_file/"
if os.path.exists(cachePath):
    shutil.rmtree(cachePath)
    
os.makedirs(cachePath)
for i in range(threshold):
    if X_t_1[i]==1:
        url = f"http://10.196.12.220:5001/download/{i}"
        r = requests.get(url, data={'content': f"{i}"})
        fp = open(f'../TempFilesDpp/{i}.txt', 'wb')
        fp.write(r.content)
        fp.close()
        source_path = f'../TempFilesDpp/{i}.txt'
        destination_path = f'../cached_file/{i}.txt'
        # Copy the file to the destination directory
        shutil.copy(source_path, destination_path)
        os.remove(f'../TempFilesDpp/{i}.txt')

i=0
with tqdm(total=NumSeq) as pbar:
    while i<NumSeq:
        data = pd.read_csv(path_to_input, sep = ' ')
        data.columns = ['Timestamp', 'File_ID', "File_Size"]
        if len(data) >= int((i+1)*DataLength/NumSeq):
            #RUN OTHERS
            df = pd.DataFrame(data[int(i*DataLength_run_others/NumSeq) : int((i+1)*DataLength_run_others/NumSeq)])
            df.sort_values("Timestamp")

            old_id = df.File_ID.unique()
            old_id.sort()
            new_id = dict(zip(old_id, range(len(old_id))))
            df = df.replace({"File_ID": new_id})
            df.sort_values("Timestamp")

            df = df[df.File_ID < threshold]
            df = df.reset_index(drop=True)

            library_size = df['File_ID'].max()+2
            C = math.floor(alpha*library_size)
            v = df['File_ID']
            RawSeq = np.array(v)
            time_run_others = int(np.floor(min(time_limit, len(v)/users)))-1
            df = np.array_split(RawSeq, users)

            # Running the algorithms

            hit_rates_OPT, download_rate_OPT = MIN(df, Adj, time_run_others, library_size, C)
            hit_rates_OPT = pd.DataFrame(hit_rates_OPT)
            download_rate_OPT = pd.DataFrame(download_rate_OPT)

            OPT_Hits.append(np.sum(hit_rates_OPT)/(time_run_others*users))
            OPT_Downloads.append(np.sum(download_rate_OPT)/(time_run_others*caches))

            hit_rates_LFU, download_rate_LFU = Bipartite_LFU(
                df, Adj, time_run_others, library_size, C)
            hit_rates_LFU = pd.DataFrame(hit_rates_LFU)
            download_rate_LFU = pd.DataFrame(download_rate_LFU)

            LFU_Hits.append(np.sum(hit_rates_LFU)/(time_run_others*users))
            LFU_Downloads.append(np.sum(download_rate_LFU)/(time_run_others*caches))

            hit_rates_LRU, download_rate_LRU = Bipartite_LRU(
                df, Adj, time_run_others, library_size, C)
            hit_rates_LRU = pd.DataFrame(hit_rates_LRU)
            download_rate_LRU = pd.DataFrame(download_rate_LRU)

            LRU_Hits.append(np.sum(hit_rates_LRU)/(time_run_others*users))
            LRU_Downloads.append(np.sum(download_rate_LRU)/(time_run_others*caches))

            hit_rates_Perturbed_LFU, download_rate_Perturbed_LFU = Perturbed_Bipartite_LFU(
                df, Adj, time_run_others, library_size, C, d)
            hit_rates_Perturbed_LFU = pd.DataFrame(hit_rates_Perturbed_LFU)
            download_rate_Perturbed_LFU = pd.DataFrame(download_rate_Perturbed_LFU)

            perturbed_LFU_Hits.append(np.sum(hit_rates_Perturbed_LFU)/(time_run_others*users))
            perturbed_LFU_Downloads.append(np.sum(download_rate_Perturbed_LFU)/(time_run_others*caches))

            hit_rates_Lead_cache, download_rate_Lead_cache, hit_rates_Madow, download_rates_Madow = Lead_cache(
                df, Adj, time_run_others, library_size, C, d)
            hit_rates_Lead_cache = pd.DataFrame(hit_rates_Lead_cache)
            download_rate_Lead_cache = pd.DataFrame(download_rate_Lead_cache)

            LeadCache_Hits.append(np.sum(hit_rates_Lead_cache)/(time_run_others*users))
            LeadCache_Downloads.append(np.sum(download_rate_Lead_cache)/(time_run_others*caches))

            hit_rates_Madow = pd.DataFrame(hit_rates_Madow)
            download_rates_Madow = pd.DataFrame(download_rates_Madow)

            LeadCache_Hits_Madow.append(np.sum(hit_rates_Madow)/(time_run_others*users))
            LeadCache_Downloads_Madow.append(np.sum(download_rates_Madow)/(time_run_others*caches))
            
            next_dem, time_run_others = get_demands(i, time_limit,data, DataLength_run_others, NumSeq, threshold)
            
            X_t_ftpl = np.zeros((threshold,))
            X_t_ftpl[init_indices_run_others] = 1
            
            
            X_t_ftpl, obj_ftpl = constrained_solve_ftpl(np.array(prev_demands_run_others).sum(axis=0), X_t_1_ftpl, cache_constraint, gamma_run_others, threshold, i)
            
            hit_rate_ftpl.append(np.dot(X_t_ftpl, next_dem)/time_run_others)
            download_rate_ftpl.append(np.sum(np.logical_and(X_t_ftpl==1, X_t_1_ftpl==0))/time_run_others)
            
            X_t_1_ftpl = X_t_ftpl
            prev_demands_run_others.append(next_dem)

            pd.DataFrame(LFU_Hits).to_csv(path+'LFU_Hits.csv',index=False)
            pd.DataFrame(LFU_Downloads).to_csv(path+'LFU_Downloads.csv',index=False)
            pd.DataFrame(LRU_Hits).to_csv(path+'LRU_Hits.csv',index=False)
            pd.DataFrame(LRU_Downloads).to_csv(path+'LRU_Downloads.csv',index=False)
            pd.DataFrame(perturbed_LFU_Hits).to_csv(path+'Perturbed_LFU_Hits.csv',index=False)
            pd.DataFrame(perturbed_LFU_Downloads).to_csv(path+'Perturbed_LFU_Downloads.csv',index=False) 
            pd.DataFrame(LeadCache_Hits).to_csv(path+'LeadCache_Hits.csv',index=False)
            pd.DataFrame(LeadCache_Downloads).to_csv(path+'LeadCache_Downloads.csv',index=False) 
            pd.DataFrame(LeadCache_Hits_Madow).to_csv(path+'LeadCache_Hits_Madow.csv',index=False)
            pd.DataFrame(LeadCache_Downloads_Madow).to_csv(path+'LeadCache_Downloads_Madow.csv',index=False) 
            pd.DataFrame(OPT_Hits).to_csv(path+'OPT_Hits.csv',index=False)
            pd.DataFrame(OPT_Downloads).to_csv(path+'OPT_Downloads.csv',index=False)
                
            pd.DataFrame(hit_rates_OPT).to_csv(path+'OPT_Hits_Seq.csv',index=False)
            pd.DataFrame(download_rate_OPT).to_csv(path+'OPT_Downloads_Seq.csv',index=False)
            pd.DataFrame(hit_rates_LRU).to_csv(path+'LRU_Hits_Seq.csv',index=False)
            pd.DataFrame(download_rate_LRU).to_csv(path+'LRU_Downloads_Seq.csv',index=False)
            pd.DataFrame(hit_rates_LFU).to_csv(path+'LFU_Hits_Seq.csv',index=False)
            pd.DataFrame(download_rate_LFU).to_csv(path+'LFU_Downloads_Seq.csv',index=False)
            pd.DataFrame(hit_rates_Lead_cache).to_csv(path+'LeadCache_Hits_Seq.csv',index=False)
            pd.DataFrame(download_rate_Lead_cache).to_csv(path+'LeadCache_Downloads_Seq.csv',index=False)
            pd.DataFrame(perturbed_LFU_Hits).to_csv(path+'perturbed_LFU_Hits_Seq.csv',index=False)
            pd.DataFrame(perturbed_LFU_Downloads).to_csv(path+'perturbed_LFU_Downloads_Seq.csv',index=False)

            pd.DataFrame(hit_rate_ftpl).to_csv(path+'hit_rate_ftpl.csv',index=False)
            pd.DataFrame(download_rate_ftpl).to_csv(path+'download_rate_ftpl.csv',index=False)

            #DPP
            V = V_0
            if os.getenv("USE_ROOT_V")=="True": V *= (i+1)**0.5
            next_dem, times = get_demands(i, time_limit, data, DataLength, NumSeq, threshold)
            X_t = np.zeros((threshold,))
            init_indices = random.sample(range(threshold), cache_constraint)
            X_t[init_indices] = 1
            
            if i==past+future:
                model = get_model(prev_demands, past, future, threshold, use_saved)
                print(model.summary())
            elif i>past+future:
                to_train = prev_demands[max(0, i-train_memory):]
                update_weight(model, to_train, past, future)
                pred = predict_demand(model, prev_demands[i-past:])
                pred = np.maximum(pred, np.zeros((pred.size,)))
                pred = np.round(pred)
                np.array(prev_demands).mean(axis=0)
                
                delta_t = get_delta()
                X_t, obj = constrained_solve(pred, cache_constraint, cost_constraint, X_t_1, delta_t, Q, V, threshold)
                objective.append(obj)
                Delta = delta_t*np.linalg.norm(X_t-X_t_1, ord=1)/2
                fetching_cost.append(Delta)
                
                
                e = np.linalg.norm(next_dem-pred, ord=2)/len(pred)
                err.append(e)
                actual_cache_hit = np.dot(next_dem, X_t)
                cache_hit.append(actual_cache_hit)
                
                indices = np.argsort(next_dem)[::-1][:cache_constraint]
                final = np.zeros((threshold,))
                final[indices] = 1
                
                
                best = np.dot(next_dem, final)
                best_maximum.append(best)
                        
                Q = max(Q + Delta - cost_constraint, 0)
                queue.append(Q)
                
            plt.plot(ma(cache_hit))
            plt.title("Cache Hit vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Cache Hit")
            plt.savefig(our_path+"Cache_Hit.jpg")
            plt.clf()
            
            plt.plot(ma(err))
            plt.title("Mean Squared Test Error in Demand Prediction vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("MSE")
            plt.savefig(our_path+"NN-MSE.jpg")
            plt.clf()


            plt.plot(ma(queue))
            plt.title("Q vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Q")
            plt.savefig(our_path+"Q.jpg")
            plt.clf()


            plt.plot(ma(objective))
            plt.title("Constrained Objective Function vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Objective Function")
            plt.savefig(our_path+"Obj.jpg")
            plt.clf()


            plt.plot(ma(fetching_cost))
            plt.title("Fetching Cost vs Timeslot")
            plt.axhline(y=cost_constraint, linewidth=2, label='Cost Constraint')
            plt.xlabel("Timeslot")
            plt.ylabel("Cost")
            plt.savefig(our_path+"Cost.jpg")
            plt.clf()


            plt.plot(ma(cache_hit))
            plt.title("Cache Hit vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Cache Hit")
            plt.savefig(our_path+"Cache_Hit.jpg")
            plt.clf()
            
            hit_rate.append(np.dot(X_t, next_dem)/np.sum(next_dem))
            download_rate.append(np.sum(np.logical_and(X_t==1, X_t_1==0))/np.sum(next_dem))
            
            plt.plot(ma(hit_rate))
            plt.title("Cache Hit Rate vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Cache Hit Rate")
            plt.savefig(our_path+"Cache_Hit_Rate.jpg")
            plt.clf()
            
            plt.plot(ma(download_rate))
            plt.title("Download Rate vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Download Rate")
            plt.savefig(our_path+"Download_Rate.jpg")
            plt.clf()

            pd.DataFrame(hit_rate).to_csv(our_path+'hit_rate.csv',index=False)
            pd.DataFrame(download_rate).to_csv(our_path+'download_rate.csv',index=False)

            max_retries = 5
            retry_delay = 0.01
            for retry in range(max_retries):
                try:
                    for j in range(threshold):#
                        if X_t[j]==1 and X_t_1[j]==0:
                            url = f"http://10.196.12.220:5001/download/{j}"
                            r = requests.get(url, data={'content': f"{j}"})
                            f = open(f'../TempFilesDpp/{j}.txt', 'wb')
                            f.write(r.content)
                            f.close()
                            source_path = f'../TempFilesDpp/{j}.txt'
                            destination_path = f'../cached_file/{j}.txt'
                            shutil.copy(source_path, destination_path)
                            os.remove(f'../TempFilesDpp/{j}.txt')

                        if X_t[j]==0 and X_t_1[j]==1:
                            destination_path = f'../cached_file/{j}.txt'
                            os.remove(destination_path)
                    break
                except PermissionError:
                    time.sleep(retry_delay)
                else:
                    print(f"Unable to access the file after {max_retries} retries.")
            X_t_1 = X_t
            
            prev_demands.append(next_dem)
            pbar.update(1)
            i = i+1

            #---------------------------------------plots-------------------------------
            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(40, 12.5)
            fig.tight_layout()

            lfu_hits = pd.read_csv(f"experiments/csv_{NumSeq}/LFU_Hits.csv")
            lru_hits = pd.read_csv(f"experiments/csv_{NumSeq}/LRU_Hits.csv")
            lead_hits = pd.read_csv(f"experiments/csv_{NumSeq}/LeadCache_Hits.csv")
            dpp_hits = pd.read_csv(f"experiments/{tag}/hit_rate.csv")
            ftpl_hits = pd.read_csv(f"experiments/csv_{NumSeq}/hit_rate_ftpl.csv")

            plt.figure(figsize=(10*1.5, 7*1.5))
            plt.xlabel("Timeslot", fontsize=30)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.ylabel("Average Cache Hit", fontsize=30)
            plt.xlim(left=0, right=i+10)
            plt.ylim(bottom=0, top=1)
            plt.plot(ma(lead_hits), label="LeadCache", marker='d', markevery=50, markersize=10, c='green', linewidth=2)
            plt.plot(ma(lru_hits), label="LRU", marker='o', markevery=50, markersize=10, c='orange', linewidth=2)
            plt.plot(ma(lfu_hits), label="LFU", marker='x', markevery=50, markersize=10, c='blue', linewidth=2)
            plt.plot(ma(dpp_hits), label="DPP-cache", marker='^', markevery=50, markersize=10, c='purple', linewidth=2)
            plt.plot(ma(ftpl_hits), label="FTPL", marker='*', markevery=50, markersize=10, c='brown', linewidth=2)
            plt.legend(loc='lower right', prop={'size': 20})
            plt.grid()
            plt.savefig(f"results/cache_hit.jpg", format='jpg', bbox_inches='tight')
            plt.clf()

            #Download rate
            lfu_downloads = pd.read_csv(f"experiments/csv_{NumSeq}/LFU_Downloads.csv")
            lru_downloads = pd.read_csv(f"experiments/csv_{NumSeq}/LRU_Downloads.csv")
            lead_downloads = pd.read_csv(f"experiments/csv_{NumSeq}/LeadCache_Downloads.csv")
            dpp_downloads = pd.read_csv(f"experiments/{tag}/download_rate.csv")
            ftpl_downloads = pd.read_csv(f"experiments/csv_{NumSeq}/download_rate_ftpl.csv")

            plt.figure(figsize=(10*1.5, 7*1.5))
            plt.xlabel("Timeslot", fontsize=30)
            plt.ylabel("Average Cache Replacement Rate", fontsize=30)
            plt.xlim(left=0, right=i+10)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.ylim(bottom=-0.005, top=1.0)
            plt.plot(ma(lfu_downloads), label="LFU", marker='x', markevery=50, markersize=15, c='blue', linewidth=4)
            plt.plot(ma(lru_downloads), label="LRU", marker='o', markevery=50, markersize=15, c='orange', linewidth=4)
            plt.plot(ma(lead_downloads), label="LeadCache", marker='d', markevery=50, markersize=15, c='green', linewidth=4)
            plt.plot(ma(dpp_downloads), label="DPP-cache", marker='^', markevery=50, markersize=15, c='purple', linewidth=4)
            plt.plot(ma(ftpl_downloads), label="FTPL", marker='*', markevery=50, markersize=15, c='brown', linewidth=4)
            plt.legend(loc='center right', prop={'size': 20})
            plt.grid()
            plt.savefig(f"results/cache_replace_rate.jpg", bbox_inches='tight')
            plt.clf()
            
            plt.close()

        else:
            # print(len(data)+10)
            # print(int((i+1)*DataLength/NumSeq))
            sec = 0.10
            time.sleep(sec)

