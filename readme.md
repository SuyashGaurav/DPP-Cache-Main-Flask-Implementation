# DPP-Cache![Research Paper Presentation](https://github.com/SuyashGaurav/DPP-Cache-main/assets/102952185/cffa3298-1b39-492c-84be-d3c5e0acab86)

## Introduction
This research project was initially initiated by [Shashank P.](https://github.com/shashankp28/DPP-Cache) Later, we (Suyash & Himanshu) Collaborated with Prof. Bharath B N. to optimize the [Drift-Plus-Penalty (DPP) caching algorithm](https://github.com/SuyashGaurav/DPP-Cache-Main-Flask-Implementation/tree/main/Base-Station/DPP), effectively outperforming established caching methods such as LRU, LFU, and FTPL.

Caching involves storing frequently accessed data at base station to reduce the need to fetch it from a central server whenever requested. 
The primary goal is to maximize the time-averaged cache hit rate. A cache hit occurs when the requested file is already present in the local cache. This is subject to constraints, specifically fetching cost and a fixed cache size.

This reposiroty contains datasets and files relevant to **DPP-caching algorithm**. We have also made use of GitHub Repository of **LeadCache** Algorithm for experimental comaprisions Links are given below.
We have demonstrated DPP using Flask, incorporating user, base station, and central server interactions. For connectivity, we have utilized Wi-Fi, enhancing the practicality and relevance of the caching algorithm. When a user demands a file that is already present in the local cache of the base station, it results in a cache hit. If the requested file is not present in the local cache of the base station, it leads to a cache miss, then the base station sends a request to the central server to fetch the missing file.
- ***Dataset and Neural Network Description for DPP-Cache***: The dataset used is “311 Service
Requests Pitt” from Kaggle.2. It consists of public service requests such as “potholes, abandoned
vehicles, building maintenance, etc.,” their description, and the time-stamps of the requests from
Feb. 2017 to Nov. 2018, which is divided into 1000 time slots. To use this data for the caching
scenario, we replace the type of service as a proxy for the files. There are a total of 423 unique
files (services) and around 6 million requests. This leads to an average of 660 requests per slot.
At the beginning of every slot, a neural network is updated based on the demands from the
past 5 time slots and is subsequently used to predict the next set of demands using the past 3
slots demand as input. The neural network consists of three LSTM layers and one dense layer
(number of outputs equals 423). This amounts to a total of 2 million neural coefficients

- ***Kaggle***: https://www.kaggle.com/datasets/yoghurtpatil/311-service-requests-pitt
- ***LeadCache***: https://github.com/AbhishekMITIITM/LeadCache-NeurIPS21
- ***DPP previous work by Shashank P.*** : https://github.com/shashankp28/DPP-Cache

<img width="877" alt="image" src="https://github.com/SuyashGaurav/DPP-Cache-Main-Flask-Implementation/assets/102952185/aa130531-cf5b-4cdc-b113-a16d4d41d97f">

## About DPP
***Notation***:
- T: Total time frame, assumed to be slotted.
- 𝓕: Set of files that can be requested, represented as {1,2,...,F}.
- Θₜ: Vector representing the sum of demands for each file at time slot t.
- C: Total cache capacity.
- ν: Cost constraint.
- V: Scaling factor balancing the trade-off between fetching cost and cache hit.
- Zₜ: Array of 0s and 1s denoting whether a file is in the cache in time slot t.

![Collaboration (1)](https://github.com/SuyashGaurav/DPP-Cache-main/assets/102952185/2baf940e-70ac-478b-a9fd-b73d635ec1a6)

![Collaboration (2)](https://github.com/SuyashGaurav/DPP-Cache-main/assets/102952185/1d0f17ab-985f-46e8-abcd-d59825f5e833)

## Pictorial Representation
![WhatsApp Image 2023-12-30 at 10 17 59 PM](https://github.com/SuyashGaurav/DPP-Cache-Main-Flask-Implementation/assets/102952185/b470f7b0-3a61-4496-aaa5-66766e034323)


  ## Performance Evaluation of DPP-Cache for single base station
   **Observation**: 
 By changing the time slot from 200 to 1000, which also changed the requests per time slot from 3300 to 660, we see the importance of adapting the DPP algorithm in real time. This shows that relying on old data may not accurately predict future trends in fast-changing situations. Giving more importance to recent data helps us use our resources efficiently and makes our system more resilient.

 ### Previous result:
![Collaboration](https://github.com/SuyashGaurav/DPP-Cache-main/assets/102952185/f251e2e2-8c09-45f5-9da2-2261bdcc79b1)

  ### Improved result:
![cache_hit](https://github.com/SuyashGaurav/DPP-Cache-main/assets/102952185/5061081d-dc96-4b2c-b552-d6eb85fd0c43)
## How to run
To run our algorithm follow the below steps:

1. Install python dependencies.
```
pip install -r requirements.txt
```
2. Change environemt variables in the *.env* file. A sample is shown below.
```
Q_INIT = 0                                # Inital Q value.
PAST = 3                                  # Number of previous slots used to predict
V_0 = 500                                 # Coeffecient of O(sqrt(T))
FUTURE = 1                                # Number of future slots to predict
ALPHA = 0.1                               # Percentage of catalogue as cache
NUM_SEQ = 1000                             # Number of sequences
THRESHOLD = 423                           # Number of files in the catalogue
TRAIN_MEMORY = 5                          # Previous slots used to train
USE_SAVED = False                         # Whether to use saved model
RUN_OTHERS = True                         # Whether to run other algorithms
COST_CONSTRAINT = 1                       # Fetching cost Constraint
TIME_LIMIT = inf                          # Maximum requests per slot
PATH_TO_INPUT = Datasets/311_dataset.txt  # Path to request dataset
```
***Note:*** **Keep FUTURE key to be always 1**

## Extension
**Multiple Base Stations**
The project extends the enhanced DPP algorithm to multiple base stations, leveraging the federated average technique. This extension introduces new dimensions, allowing for the optimization of caching strategies in a federated environment. More details can be found on this [GitHub repo](https://github.com/SuyashGaurav/Federated-Average-Main-Live-Demonstration).
