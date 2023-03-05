##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
from enum import unique
import glob
import os
import sys
from copy import deepcopy
import numpy as np
import datetime

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    import re
    DGH = {}
    f = open(DGH_file,"r")
    #read = f.read()
    lines = f.readlines()
    
    
    count  = 0
    l = []
    ## read lines and add to list with tab counts as a tuple 
    for line in lines:
        #print("* ")
       
        word = line.strip()
        tab_c = line.count('\t')
        l.append((word,tab_c))
        new = l[len(l)-1]
        
        if count != 0:
            if l[count-1][1] != tab_c:
                ## this is last added 
                
                #print("add " + str(new))
                #print("Start --------------------------------")
                for i in range( len(l) - 2, -1, -1) :
                   
                    tup = l[i]
                    if tup[1] < tab_c:
                       
                        old = DGH[tup[0]]
                        
                        new_up = DGH[tup[0]]+" " + word
                        DGH[word] = new_up
                       
                        break
            else:
                #print("Added a same level ")
                DGH[word] = old + " " + word
                #print("Same level DGH updated " + str(DGH))
                        
                #print("End --------------------------------")
            
                
                    
        else:
            DGH[word] = word
            #print("first is hereee")
            #print(DGH[word])
                
        count = count + 1   
       
   
    return DGH
    
## these functions are called in other functions 
def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    
    
    lenght1 = len(raw_dataset)
    lenght2 = len(anonymized_dataset)
    cost = 0
    
    
    keys = list(raw_dataset[0].keys())
   ## calculate for each 
    for i in range(lenght1): 
        #print("anonymized_dataset")
        #print(anonymized_dataset[i])
        
        cost = cost + row_MD(raw_dataset[i],anonymized_dataset[i],DGHs,keys)
    
   
    return cost
## calculate for rows
def row_MD(raw_dataset_row,anonymized_dataset_row,DGHs,keys):
    
    key_l = len(keys)
    cost = 0
    ## all the keys except income
    for i in range(key_l):
         if keys[i] != 'income':
       
            dgh = DGHs[keys[i]]
            value_raw = raw_dataset_row[keys[i]]
            value_anon = anonymized_dataset_row[keys[i]]
            raw_path = dgh[value_raw]
            anon_path = dgh[value_anon]
            raw_sp = raw_path.split()
            anon_sp = anon_path.split()
         
            
            cost = cost + abs(len(raw_sp) - len(anon_sp))
        #print("cost is " + str(cost))
        
    return cost 

def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    
    lenght1 = len(raw_dataset)
    lenght2 = len(anonymized_dataset)
    cost = 0
    
    keys = list(raw_dataset[0].keys())
    for i in range(lenght1): 
        
        #data sets each row
        cost = cost + row_LM(anonymized_dataset[i],DGHs,keys)
        #break
    
    
    
    

    #TODO: complete this function.
    return cost           
def row_LM(anonymized_dataset_row,DGHs,keys):

    #print(keys)
    #print(len(keys))
    key_l = len(keys)
    cost = 0
    M = key_l-1
    w = 1/M
    for i in range(key_l):
        if keys[i] != 'income':
        #find the corresponding dgh
            dgh = DGHs[keys[i]]
            dgh_dict = child_dict_create(dgh)
            
            total_child = max(dgh_dict.values())
            ## calculate cost lm for anon and raw dataset 
           
            val2 = anonymized_dataset_row[keys[i]]
            #print(val2)
            #val_raw = dgh_dict[val1]
            val_anon = dgh_dict[val2]
            #cost_raw = w*(val_raw-1)/(total_child-1)
            cost_anon = w*(val_anon-1)/(total_child-1)
            #diff = abs(cost_anon - 0)
            #cost = cost + diff
            cost = cost + cost_anon
          
           
    return cost 

## create dictionary containing each nodes number of leaves under 
def child_dict_create(dgh):
   
    
    l = len(dgh)
  
    keys = list(dgh.keys())
    child = []
    path_l = 0
    
    path_l = []
   
    
    for i in range(l):
        
        
        
        #print("i ")
        curr_key = keys[i]
        #print("curr_key " + str(curr_key))
        current  = dgh[keys[i]]
        cur_sp = current.split()
        curr_len = len(cur_sp)
       
        path_l.append((curr_key,curr_len))
       
   
    ll = len(path_l)
    for i in range(ll-1):
        tup = path_l[i]
        tup_n = tup[0]
        tup_l = tup[1]
        
        
        
        next_tup = path_l[i+1]
        next_tup_n = next_tup[0]
        next_tup_l = next_tup[1]
        
        
        if next_tup_l < tup_l :
            child.append((tup_n,tup_l))
        
        if next_tup_l == tup_l :
            child.append((tup_n,tup_l))
        
    
    path_l_new = len(child)
    
    last_tup = child[path_l_new-1]
    path_last_tup =  path_l[ll-1]
   
    child.append((path_last_tup[0],path_last_tup[1]))
        
  
    #print(keys)
    keys_len = len(keys)
   
    ##initiliaze dictionary
    child_dict ={}
    for i in range(keys_len):
        child_dict[keys[i]] = 0
    
    for i in range(len(child)):
       tup = child[i]
      
       path = dgh[tup[0]]
       sp = path.split()
       #print(sp)
       path_len = len(sp)
       #break
       for j in range(path_len):
           #print()
           child_dict[sp[j]] = child_dict[sp[j]] + 1
     
     

    return child_dict


          

def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    
    #edu = DGHs["education"]
    
  
    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...
   
    
    counter = 0
    cl = []
    
    num_clusters = D/k   
    cluster_c = 0
    test = []
    for i in range(D): 
        cl.append(raw_dataset[i])
        #print("cl is ")
        #print(cl)
        counter = counter + 1
        #if counter == k or (num_clusters - cluster_c) < 1:
        if counter == k or ((D-i) <= k and i == D-1) :
            ## make the generalization here 
            clusters.append(cl)
            test = cl.copy()
            counter = 0
            cluster_c = cluster_c + 1
            cl = []
            
    
    
   
    #anon_Cluster(clusters[0],DGHs)
    # edit all clusters 
    #anonymize 
    for i in range (len(clusters)) :
        clusters[i] = anon_Cluster(clusters[i],DGHs)
    
   
    
    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D
    

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)
    #write_dataset(raw_dataset, output_file)
    ## anon a single cluster 
def anon_Cluster(cluster,DGHS):
    keys = list(cluster[0].keys())
    #print("keys ate ")
    #print(keys)
    
    for key in keys:
        if key != 'income' and key != 'index':
            #print("key is " + str(key))
            dgh_type = DGHS[key]
            #print("dgh type is" )
            #print(dgh_type)
            # edit cluster for al the keys except income and index
            cluster = anon_column(dgh_type,cluster,key)
            #break
    return cluster
## anon each column 
def anon_column(dgh,cluster,key):
   
    results  = []
    for record in cluster :
        if key != 'income' and key != 'index':
          
          
            val = record[key]
            #print("my dict res")
            #print(dgh[val])
            ret = dgh[val]
           
            results.append(ret)
          
    import os
    
    
    l = []
    #print(results[0][0])
    for i in range (len(results)):
        sp = results[i].split()
        l.append(sp)
  
    common = os.path.commonprefix(l)
  
    result = common[len(common)-1]
   
    for rec in cluster:
        if key != 'income' and key != 'index':
           
            rec[key] = result
    
    return cluster

def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    #np.random.shuffle(raw_dataset)
    #TODO: complete this function.
    
    L = len(raw_dataset)
    visited = [0]*L
   
    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i
        
    raw_dataset = np.array(raw_dataset)
        
    clusters = []
    D = len(raw_dataset)
    
    i = 0
    
        
        
    
    while visited.count(0) >= 2*k:
       
        index = find_unmarked(visited)
        rec = raw_dataset[index]
        #print("executed")
        visited[index] = 1
        ## returns the smaleest cost n records with the indexes
        res = calculate_lowest_dist(raw_dataset,visited,rec,DGHs,k)
        res.append(rec)
        
        #print("len is" )
        #break
        res_l = len(res)
      
        for i in range(res_l):
            rec = res[i]
          
            index_n = rec['index']
          
            visited[index_n] = 1
       
        clusters.append(res) 
        res = []
        
        i = i + 1
    else:
        last = []
       
        l = len(visited)
        #print("cluster")
        #print(len(clusters))
        #return
        for i in range(l):
            if visited[i] == 0:
                last.append(raw_dataset[i])
        #print()
        clusters.append(last) 
       
    
    
    ##generalize clusters !!!!!!!!!
    
    
    for i in range (len(clusters)) :
        clusters[i] = anon_Cluster(clusters[i],DGHs)
    
    
    anonymized_dataset = [None] * D
     

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
                #print("fault")
                #print(anonymized_dataset[item['index']])
                anonymized_dataset[item['index']] = item
                del item['index']
    
    # Finally, write dataset to a file
    write_dataset(anonymized_dataset, output_file)


def generalize_clusters():
    print



def calculate_lowest_dist(raw_set,visited_list,record,DGHs,k):
    L = len(raw_set)
    cost_list = []
    #print("my count is ")
    #print(visited_list.count(0))
    for i in range(L):
        if visited_list[i] == 0:
            #visited_list[i] = 1
            cost_rec_tup = calculate_LM_cost(raw_set[i],i,record,DGHs)
            cost_list.append(cost_rec_tup)
            
    
    res_list = find_n_smallest_rec(cost_list,k)
   
    return res_list



def find_n_smallest_rec(records,k):
    #print()
    
    import heapq
   
    result = [tup[1] for tup in records]
    
    ##bence hata burda 
   
    sort = sorted(records, 
       key=lambda x: x[1])
    final = []
    for i in range(k-1):
        tup = sort[i]
        final.append(tup[0])
        
   
    return final

## calculate LM cost between two records
def calculate_LM_cost(raw_rec,index_raw,goal_rec,DGHs):
    #print("record 1 ")
    #print(raw_rec)
    ## generalize a record
    hypo_rec1 = generalize_record(raw_rec,goal_rec,DGHs)
   
    keys = list(raw_rec.keys())
    cost = 0
    #for 
    #print("keys are ")
    #print(keys)
    L = len(keys)
    w = 1/(L-1)
    for i in range(L):
        if keys[i] != 'income' and keys[i] != 'index':
        #find the corresponding dgh
            dgh = DGHs[keys[i]]
          
            dgh_dict = child_dict_create(dgh)
           
            total_child = max(dgh_dict.values())
            
            val1 = hypo_rec1[keys[i]]
          
           
            #print(val2)
            val_raw = dgh_dict[val1]
           
            cost_raw = w*(val_raw-1)/(total_child-1)
            cost = cost + cost_raw
          
    
    return (raw_rec,cost)
## generalize two records and return a copy 
def generalize_record(record1,record2,DGHs):
    keys = list(record1.keys())
    L = len(keys)
    gen = []
    rec1_copy = record1.copy()
    import os
    for i in range(L):
        if keys[i] != "income" and keys[i] != 'index':
        #find the current dgh
            dgh = DGHs[keys[i]]
            val1 = dgh[record1[keys[i]]]
            sp1 = val1.split()
           
            val2 = dgh[record2[keys[i]]]
            sp2 = val2.split()
            
            gen.append(sp1)
            gen.append(sp2)
           
            common = os.path.commonprefix(gen)
            result = common[len(common)-1]
           
            rec1_copy[keys[i]] = result
            common = []
            sp1 = []
            sp2 = []
            val1 = []
            val2 = []
            gen = []
    
    return rec1_copy
    



def find_unmarked(visited_rec):
    L = len(visited_rec)
    for i in range(L):
        if visited_rec[i] == 0:
            return i
    return -1
def unmarked_count(visited_rec):
    return visited_rec.count(0)





def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    #TODO: complete this function.
    """
    found  = False
    anons = []
    l_dghs = len(DGHs)
   
    win_list = []
    keys = list(raw_dataset[0].keys())
    key_l = len(keys)
    # do the first one level anonimyzation for the set 
    for i in range(key_l):
        print("for ")
        #print(DGHs[0])
        if keys[i] != 'income':
            dgh = DGHs[keys[i]]
            print(dgh)
            res = anon_one_level(raw_dataset, dgh,keys[i])
            ## add the generated dataset to the list
            anons.append(res)
           
        
    ## check if any set satisfies k anonimity 
    for i in range(len(anons)):
        print("other for ")
        res = check_k_anon(anons[i],k)
        ## if it satisifes add to the winner list
        ## no need for the loop below
        #winner list will be checked at the end and set with lowest LM cost will be returned
        
        if res == True:
            win_list.append(anons[i])  
            print("wooooonnnn")
            found = True
            
    print("anons len are ")
    print(len(anons))
    #return
    
    ## check for all possible combinations 
    ## returns the most generalized data set in the worst case 
    if not found:
        while True:
            print("loop")
            # make anon for all of the keys for one time for each set in  the list
            for i in range(key_l):
                print("888888888888")
                if keys[i] != 'income':
                    dgh = DGHs[keys[i]]
                    # anon each set by one level for the corresponding key 
                    for j in range(len(anons)):
                        res = anon_one_level(anons[i], dgh,keys[i])
                        anons.append(res)
                        print("here ")
                print("len is ")
                print(len(anons))
                ## check if any set satisfies k anonimity 
                for l in range(len(anons)):
                    print("!!!!!!!!!!!1")
                    res = check_k_anon(anons[l],k)
                    print("44444444444444444441")
                    if res == True:
                        print("inisdeeeeeeee")
                        win_list.append(anons[l])
                    
                    
    cost = []  
    print("win list")
    print(win_list)  
    # rcalculate LM cost for sets which satisfy k anonimity 
    for s in win_list: 
        print("s ")
        print(s)
        c = cost_LM(s,s,DGHs)
        tup = (s,c)
       
        #cost.append((cost_LM(s,s,DGHs),s))
        cost.append(tup)
    # sort the list 
    sorted_list = sorted(cost, key=lambda x: x[0])
    # return the set which has the least LM cost
    anonymized_set = sorted_list[0][1]
    """
    
    print("I tried implementing bottom up anonymzier, wrote some helper functions but ")
    print("There is something I am missing and I could not solve it ")
    print("I took the whole code in comments , It would be great if I you can give some partial points ")
    # Finally, write dataset to a file
    write_dataset(raw_dataset, output_file)
    

def anon_one_level(dataset,dgh,key):
    ##this function generalizes the given dataset 1 level for the given dgh
    ## dgh is the dictionary which holds the tree path to the nodes 
    cpy = dataset.copy()
    L = len(dataset)
    print("key is ")
    print(key)
    for i in range(L):
        rec = cpy[i]
        val = rec[key]
        path = dgh[val]
        sp = path.split()
        print("sp path is ")
        print(sp)
        if len(sp) > 1:
            l_split = len(sp)
            gen_val = sp[l_split-2]
            print("gen val is ")
            print(gen_val)
            rec[key] = gen_val
    print("XXXXXXXXXXXXXXXXX")
    return cpy


## checks k anonimity of a dataset 
def check_k_anon(dataset,k):
   
    L = len(dataset)
    visited = [0]*L
    clusters = []
   
    
    while visited.count(0) > 0:
        index = find_unmarked(visited)
   
        rec = dataset[index]
        visited[index] = 1
        res = []
        res.append(rec)
        for j in range(L):
            other = dataset[j]
            
            if visited[j] == 0:
                if check_equality(rec,other) :
                    res.append(other)
                    visited[j] = 1
        clusters.append(res)
    
    
    for cluster in clusters:
        if len(cluster) < k:
            return False
        
    return True
## checks equality of Two records
def check_equality(rec1,rec2):
    for val in rec1:
        ## these can be different no problem
        if val != 'income' and val != 'index':
            if val != rec2[val]:
                return False
    return True
                

# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)
# get the anonymizer function 
algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now() ##
print(start_time) ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

#calls the functions above 
function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)


cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now() ##
print(end_time) ##
print(end_time - start_time)  ##

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300 5