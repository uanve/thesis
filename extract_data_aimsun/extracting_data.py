# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:25:31 2021

@author: Joan
"""
import os
#cwd = os.getcwd()
#print("Current working directory: {0}".format(cwd))
os.chdir('C:/Users/Joan/OneDrive - Danmarks Tekniske Universitet/04 Forth Semester/thesis/code/aimsun')

import pandas as pd
import sqlite3

con = sqlite3.connect('data/Initial_Outputs.sqlite')



cursor = con.cursor()

cursor.execute("SELECT * FROM MICENT_O")

#cursor.execute("help")

x=cursor.execute("SELECT * FROM MICENT_O where did=3156")

for y in x.fetchall():
    print(y)




print(cursor.fetchall())

con.close()
#how many 
replication = 3156
df = pd.read_sql_query("SELECT count(*) FROM MICENT_O where did={} and not destination = 0 group by ent".format(replication), con) #210 lines per each time interval
df = pd.read_sql_query("SELECT * FROM MICENT_O where did=3156 and ent = 0 and oid=1086  ", con)

#OD demand
#did: replication
#oid: centroid id
#sid: veh type
#ent: time interval
#destination

df = pd.read_sql_query("SELECT oid,destination,sid,ent,sum(flow) FROM MICENT_O "+
                        "where did=3156 and ent = 0 and oid=2084 "+
                        "GROUP BY oid,destination,sid,ent "+
                        "ORDER BY sid", con)


##### Extract data from Simulation sqlite
veh_type = 1
replication = 3156
time_interval = 0

##### OD flows from simulation ############################################################## 
#comment: if a origin centroid has no demand, it doesn't show in sql data (columns unexisting)
#function
def OD_flows(veh_type,replication,time_interval,con):
    return pd.read_sql_query("SELECT oid,destination,flow FROM MICENT_O where did={} and ent={} and sid = {} and not destination = 0".format(replication,time_interval,veh_type), con)

df = OD_flows(veh_type,replication,time_interval,con)

##### Expected flow from simulator ###########################################################
def counts_sim(veh_type,replication,time_interval,con):
    return pd.read_sql_query("SELECT oid,flow FROM MIDETEC where did={} and ent={} and sid = {}".format(replication,time_interval,veh_type), con)

df = counts_sim(veh_type,replication,time_interval,con)

import scipy

"hola {} {}".format("Joan","Ventura")
