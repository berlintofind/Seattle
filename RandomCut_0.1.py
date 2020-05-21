# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:56:19 2020

@author: To find Berlin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy  as gb
#import random
import csv
import time
start = time.clock()

case = 60
gl = 10
total = 48*7*52

consumption=pd.read_csv('AEMO-2012-2013-House+PV data.csv', sep=',')
v = consumption[['Customer']]
consumption = consumption[v.replace(v.stack().value_counts()).eq(1095).all(1)]
consumer = []
consumerlist = []
caplist = []  
for i,j in enumerate(consumption["Customer"].unique()):
    
    consumerlist.append("consumer{:.0f}".format(i+gl))               #consumer.append("consumer{:.0f}".format(i+1)) 
    
for i,j in enumerate(consumerlist):
    consumer.append(consumption.iloc[i*1095:(i+1)*1095,0:53])      #consumer[i] = consumption.iloc[i*1095:(i+1)*1095,0:53] 
    caplist.append(consumption.iat[i*1095+1,1])
consumerlist = consumerlist[0:case]

consumerlist = ['consumer0','consumer1','consumer2','consumer3','consumer4','consumer5','consumer6','consumer7','consumer8','consumer9'] + consumerlist + [ 'consumer1000','consumer2000' ]
d_jlist = pd.read_csv('127case.csv', sep=',')
d_jlist = d_jlist.values

#numbers1  = [random.uniform(0.01,0.02) for x in range(20)]
#numbers2  = [random.uniform(2,3) for x in range(20)]
#numbers3  = [random.uniform(0.1,0.2) for x in range(20)]
#numbers4  = [random.uniform(1,2) for x in range(20)]
#numbers = numbers1 + numbers2+numbers3+numbers4
#numbers = np.asarray(numbers).reshape(4,20)
#np.savetxt('numbers2.csv',numbers,fmt='%.4f',delimiter=',')

case = case+2
numbers = pd.read_csv('numbers137.csv', header=None,sep=',').values
replace_rate = 0.1


class expando(object):

    pass

class marketclear:
    def __init__(self):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self._load_data()
        self._build_model()

    def optimize(self):
#        self.model.setParam( 'OutputFlag', 0 )
        self.model.optimize()  
#        print('Done')
        
    def _load_data(self):
        self.data.interval = np.arange(total)                  #这里只是三个peer的全年 48*365*3，一共127 17520        
        self.data.windowinterval = np.arange(48)
        self.data.timewindow = np.arange(int(len(self.data.interval)/len(self.data.windowinterval)))
        
        self.data.agents = consumerlist  
        self.data.generators = ['consumer0','consumer1','consumer2','consumer3','consumer4']        
        self.data.pureload = [ 'consumer5','consumer6','consumer7','consumer8','consumer9']
        self.data.DGen = [ 'consumer1000' ]
        self.data.DLoad = [ 'consumer2000' ]
                
#        numbers  = [random.uniform(0.01,0.02) for x in range(case+gl)]
        self.data.a_nPositive = dict(zip(self.data.agents,numbers[0]))    
#        numbers  = [random.uniform(2,3) for x in range(case+gl)]
        self.data.b_nPositive = dict(zip(self.data.agents,numbers[1]))         
#        numbers  = [random.uniform(0.1,0.2) for x in range(case+gl)]
        self.data.a_nNegative = dict(zip(self.data.agents,numbers[2]))    
#        numbers  = [random.uniform(1,2) for x in range(case+gl)]
        self.data.b_nNegative = dict(zip(self.data.agents,numbers[3]))  
        
        self.data.com = np.ones([case+gl,case+gl],dtype = np.int)                #Fully Communicated matrix, all 1 except diagonal
        np.fill_diagonal(self.data.com, 0)
        Gzone = len(self.data.generators)
        Lzone = Gzone + len(self.data.pureload)
        self.data.com[0:Gzone,0:Gzone ]=0
        self.data.com[Gzone:Lzone,Gzone:Lzone ]=0
        
        self.data.com = np.triu(self.data.com)       #这里只取上半部分！ 
#        replace_rate = 0
        mask = np.random.choice([0,1],size=(self.data.com.shape),p=((1-replace_rate),replace_rate)).astype(np.bool)  
        #生成一个给定数组范围内的，随机数组
        r1 = np.random.rand(*self.data.com.shape) #生成这个大小的随即数，（0，1)之间,再放大到最大倍数
        self.data.com[mask] = r1[mask]   
        self.data.com += self.data.com.T - np.diag(self.data.com.diagonal())
        
        self.data.com[0:Gzone,0:Gzone ]=0
        self.data.com[Gzone:Lzone,Gzone:Lzone ]=0
        self.data.com[0:Gzone,case+gl-2:case+gl ] = 0
        self.data.com[case+gl-2:case+gl,0:Gzone ] = 0
        self.data.com[Gzone:Lzone,case+gl-1:case+gl ] = 0
        self.data.com[case+gl-1:case+gl,Gzone:Lzone ] = 0
        
        self.data.com[Gzone:,case+gl-2:case+gl-1 ] = 1
        self.data.com[case+gl-2:case+gl-1,Gzone: ] = 1
        self.data.com[Lzone:,case+gl-1:case+gl ] = 1
        self.data.com[case+gl-1:case+gl,Lzone: ] = 1
        self.data.com[case+gl-2:case+gl,case+gl-2:case+gl ] = 0
        self.data.sparsity = (np.sum(self.data.com )  / ((case+gl)*(case+gl-1)))
        
#        self.data.com = self.data.com.T | self.data.com   
#        print ((self.data.com))        
#        np.savetxt('com_randomcut.csv',self.data.com,fmt='%.0f',delimiter=',')
        
        self.data.load = {}
        self.data.budget = { }
        i = 0
        for p in self.data.agents:
            if p in self.data.generators:
                pass  
            elif p in self.data.DLoad:     #这里新增
                pass
            elif p in self.data.DGen:
                pass            
            elif p in self.data.pureload:
                for t1 in self.data.timewindow:
                    cnt1 = 0
                    for t2 in self.data.windowinterval:                          
                        self.data.load[p,t1,t2] = np.cos(2*np.pi*(t2-12)/24)/2-3.51    
                        cnt1+=self.data.load[p,t1,t2]
                    self.data.budget[p,t1] = abs(cnt1)*0.9
            else:
                for t1 in self.data.timewindow:
                    cnt2 = 0
                    for t2 in self.data.windowinterval:
                        self.data.load[p,t1,t2] = d_jlist[t1+i*365][t2]
                        cnt2+=abs(self.data.load[p,t1,t2])
                    self.data.budget[p,t1] = abs(cnt2)*0.9
                i+=1
#        print(self.data.budget,'load')
                    
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()        
        self._build_constraints()
    def _build_variables(self):
        m = self.model
        self.variables.P_n = {}
        self.variables.absP_n = {}
        self.variables.P_nm = {}
        self.variables.a_n = {}
        self.variables.b_n = {}
        
        P_nm = self.variables.P_nm
        timewindow = self.data.timewindow
        windowinterval = self.data.windowinterval
        load = self.data.load
        budget = self.data.budget
                   
        for t1 in timewindow:               #P_n,在范围内, Pn = PV - Consumption + flexiblity kW
            for t2 in windowinterval:
                for p in self.data.agents:
                    if p in self.data.generators:
                        self.variables.P_n[p,t1,t2] = m.addVar(lb = 0 , ub = 200 )
                        self.variables.a_n[p,t1,t2] = self.data.a_nPositive[p]
                        self.variables.b_n[p,t1,t2] = self.data.b_nPositive[p]
                    #此处新增    
                    elif p in self.data.DLoad:                                  #Dummy load and Gen, Big M method, a_n = 0, b_n = Big M
                        self.variables.P_n[p,t1,t2] = m.addVar(lb = -np.inf, ub = 0 )
                        self.variables.a_n[p,t1,t2] = 0
                        self.variables.b_n[p,t1,t2] = -10**9
                    elif p in self.data.DGen:
                        self.variables.P_n[p,t1,t2] = m.addVar(lb = 0 , ub = np.inf )  
                        self.variables.a_n[p,t1,t2] = 0
                        self.variables.b_n[p,t1,t2] =  10**9                         
                    elif load[p,t1,t2] >0:  
                        self.variables.P_n[p,t1,t2] = m.addVar(lb = load[p,t1,t2]*0.8 , ub = load[p,t1,t2]*1.1 )
                        self.variables.a_n[p,t1,t2] = self.data.a_nPositive[p]
                        self.variables.b_n[p,t1,t2] = self.data.b_nPositive[p]                                              
                    elif load[p,t1,t2] <0:
                        self.variables.P_n[p,t1,t2] = m.addVar(lb = load[p,t1,t2]*1.1 , ub = load[p,t1,t2]*0.8)
                        self.variables.a_n[p,t1,t2] = self.data.a_nNegative[p]
                        self.variables.b_n[p,t1,t2] = self.data.b_nNegative[p]                        
                                
        for t1 in timewindow:                 #绝对值
            for t2 in windowinterval:
                for p in self.data.agents:  
                    self.variables.absP_n[p,t1,t2] = m.addVar(lb = 0 , ub = 200 )

                    
        for t in self.data.interval:           #P_nm在范围内, a broder range
            for p1 in self.data.agents:
                for p2 in self.data.agents:
                    if p1 == p2 :
                        P_nm[p1,p2,t] = m.addVar(lb=0, ub=0) 
                    elif p1 in self.data.generators and p2 in self.data.generators:
                        P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                    elif p1 in self.data.pureload and p2 in self.data.pureload:
                        P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                    elif p1 in self.data.DGen:
                        if p2 in self.data.generators:
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                        elif p2 in self.data.DLoad:
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                        else:    
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=200) 
                            
                    elif p1 in self.data.DLoad: 
                        if p2 in self.data.generators:
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                        elif p2 in self.data.pureload:   
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                        elif p2 in self.data.DGen:
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                        else:
                            P_nm[p1,p2,t] = m.addVar(lb=-200, ub=0)                                                 
                        
                        
                    elif p1 in self.data.generators:
                        if p2 in self.data.DGen:
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                        elif p2 in self.data.DLoad:
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                        else:
                            P_nm[p1,p2,t] = m.addVar(lb = 0 , ub =budget[p2,int(t/48)]  )                    #budget[p,int(t/48)]    
                    elif p1 in self.data.pureload:
                        if p2 in self.data.DLoad:
                            P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                        else:                          
                            P_nm[p1,p2,t] = m.addVar(lb =load[p1,int(t/48),t%48]*1.1  , ub = 0)

                    else :
                        if load[p1,int(t/48),t%48]>0:
                            if p2 in self.data.generators:
                                P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                            elif p2 in self.data.DGen:
                                P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                            else:
                                P_nm[p1,p2,t] = m.addVar(lb = 0, ub = load[p1,int(t/48),t%48]*1.1)  
                        elif load[p1,int(t/48),t%48]<0:
                            if p2 in self.data.pureload:
                                P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)
                            elif p2 in self.data.DLoad:
                                P_nm[p1,p2,t] = m.addVar(lb=0, ub=0)                                
                            else:                            
                                P_nm[p1,p2,t] = m.addVar(lb = load[p1,int(t/48),t%48]*1.1 , ub = 0)
                 
        m.update()  
          
        P_nm = [(k,v) for k,v in P_nm.items()]   
        P_nm.sort(key=lambda k: (k[0][2], int(k[0][0][8:]),int(k[0][1][8:])))
#        print(P_nm)

        P_nm2 =[]
        for i in P_nm:
           P_nm2.append(i[1])
#        print(P_nm2)
        
#按照每个小时，放到每一层          
        self.variables.P_nm3 = np.zeros((total, case+gl,case+gl), dtype=np.object)  #一次的时间范围
        P_nm3 = self.variables.P_nm3
        j=0
        while j in range(total):                                             
            i=0
            while i in range(len(P_nm2)):
                l = P_nm2[i:i+((case+gl)*(case+gl))]
                l = np.reshape(l,(case+gl,case+gl))
                P_nm3[j,:,:] = l
                i = i+((case+gl)*(case+gl))
                j = j+1
                
#        print(len(P_nm3))
#        print('==============')
                     
#每小时，n*n 的矩阵，不受timewindow影响
        
    def _build_objective(self):   #Timewindow
#        interval = self.data.interval
        P_n = self.variables.P_n
        agents = self.data.agents
        timewindow = self.data.timewindow
        windowinterval = self.data.windowinterval
        a_n = self.variables.a_n
        b_n = self.variables.b_n
        
        self.model.setObjective(
                gb.quicksum( a_n[p,t1,t2]  *P_n[p,t1,t2]*P_n[p,t1,t2] for t1 in timewindow for t2 in windowinterval for p in agents)
                + gb.quicksum( b_n[p,t1,t2] *P_n[p,t1,t2] for t1 in timewindow for t2 in windowinterval for p in agents),
                gb.GRB.MINIMIZE)
#全部时间的叠加求最小，关联留给限制条件
           
    def _build_constraints(self):
        m = self.model
        com = self.data.com
        P_nm3 = self.variables.P_nm3
        agents = self.data.agents
        
        timewindow = self.data.timewindow              
        windowinterval = self.data.windowinterval      
        budget = self.data.budget
        absP_n = self.variables.absP_n
        P_n = self.variables.P_n

        self.constraints.pnmatch = {}  
        self.constraints.pnmmatch = {}       
        self.constraints.Dailybudget = {}
        self.constraints.absv = {}

        for t1 in timewindow:
            for t2 in windowinterval:
                for i, p in enumerate(agents):
                    if p in self.data.generators:
                        pass
                    elif p in self.data.DGen:
                        pass                    
                    else:
                        self.constraints.absv[t1,t2,p] = m.addGenConstrAbs( absP_n[p,t1,t2], P_n[p,t1,t2])                           

#Daily budget:        
        for t1 in timewindow:         
            for i, p in enumerate(agents):
                if p in self.data.generators:
                    pass
                elif p in self.data.DLoad:      #此处新增！
                    pass
                elif p in self.data.DGen:
                    pass                
                else:
                    self.constraints.Dailybudget[t1,p] = m.addConstr( 
                            gb.quicksum( absP_n[p,t1,t2] for t2 in windowinterval), 
                            gb.GRB.LESS_EQUAL,
                            budget[p,t1]
                            )
#        print('==============')

#P_nm = P_n Constraints  
        for t1 in timewindow:
            for t2 in windowinterval:
                k = int(48*t1+t2)
                for i, p in enumerate(agents):
                    self.constraints.pnmatch[t1,t2,p] = m.addConstr(gb.quicksum(com[i]*P_nm3[k,i,:] ), gb.GRB.EQUAL, self.variables.P_n[p,t1,t2])
#        print('tes')
        
#P_nm = P_mn     
        for t in self.data.interval:
            cnt3 = 0
            for i in range(cnt3,len(P_nm3[t])-1):
                for j in range(cnt3+1, len(P_nm3[t][0])):
                    self.constraints.pnmmatch[t,i,j] = m.addConstr(
                            P_nm3[t][i][j],
                            gb.GRB.EQUAL,
                            -P_nm3[t][j][i]
                            )
                cnt3+=1

start2 = time.clock()
poolmarket = marketclear()
poolmarket.model.setParam( 'OutputFlag', 0 )
start3 = time.clock()
poolmarket.optimize()


#df = pd.DataFrame(index = poolmarket.data.interval,data = {
#        'C0 Level': [poolmarket.variables.P_n['consumer0',t1,t2].x  for t1 in poolmarket.data.timewindow for t2 in poolmarket.data.windowinterval],
#        'C1 Level': [poolmarket.variables.P_n['consumer1',t1,t2].x  for t1 in poolmarket.data.timewindow for t2 in poolmarket.data.windowinterval],
#        'C2 Level': [poolmarket.variables.P_n['consumer2',t1,t2].x  for t1 in poolmarket.data.timewindow for t2 in poolmarket.data.windowinterval],
#
#              
#        })    
#df.plot(title = 'P2P Market',marker = '.')
#plt.xlabel('Hours')
#plt.ylabel('Amount')

end = time.clock()
end2 = time.clock()
end3 = time.clock()

#print ('running time {:.3f}s'.format(end-start))
#print ('running time2 {:.3f}s'.format(end2-start2))
#print ('running time3 {:.3f}s'.format(end3-start3))
#print(poolmarket.model.Runtime)
#print('Optimal solution',poolmarket.model.getObjective().getValue())


p = np.zeros([case+gl,case+gl])
for k,i in enumerate(p):
    for y,j in enumerate(i):        
        for t in poolmarket.data.interval:
            p[k][y] = p[k][y] + poolmarket.data.com[k][y] * abs(poolmarket.variables.P_nm3[t,k,y].x)

#print('=======')
#print(np.round(p, decimals=2)) 
#np.savetxt('RandomCut_'+ str(0.1) +'_P_nm_52weeks.csv',p,fmt='%.18f',delimiter=',')    
            
csvfile = 'RandomCut_'+ str(replace_rate) +'_P_nm_52weeks.csv'            
with open (csvfile,'w') as cs:
    csvwriter = csv.writer(cs, dialect='excel')
    csvwriter.writerow(['running time s',end-start] )
    csvwriter.writerow(['running time2 s',end2-start2])
    csvwriter.writerow(['running time3 s',(end3-start3)])
    csvwriter.writerow(['Run time',poolmarket.model.Runtime])
    csvwriter.writerow(['Sparsity level',poolmarket.data.sparsity])
    csvwriter.writerow(['Optimal Solution',poolmarket.model.getObjective().getValue()])
    csvwriter.writerows(p)













#com = np.ones([8,8],dtype = np.int)                #Fully Communicated matrix, all 1 except diagonal
#np.fill_diagonal(com, 0)
#Gzone = len(poolmarket.data.generators)
#Lzone = Gzone + len(poolmarket.data.pureload)
#com[0:Gzone,0:Gzone ]=0
#com[Gzone:Lzone,Gzone:Lzone ]=0
#replace_rate = 0.1
#
#mask = np.random.choice([0,1],size=(com.shape),p=((1-replace_rate),replace_rate)).astype(np.bool)  
##生成一个给定数组范围内的，随机数组
#r1 = np.random.rand(*com.shape) #生成这个大小的随即数，（0，1)之间,再放大到最大倍数
#com[mask] = r1[mask]


#r1[mask][1]= 2.3
#r1[r1 == r1[mask][1]] = 4.3
#print(r1,'r1')
#print('============')
#print(type(r1[mask][1]),type(r1[mask]))
#print('============')
#print(r1[mask],r1.dtype,len(r1[mask]),'r1mask' )    #随概率改变,和Ture的数量一样,把下标为True的值取出来
#print('============')
#print(com[mask],com[mask].dtype,len(com[mask]),'commask')   #随概率改变





#np.set_printoptions(suppress=True)

#p = np.zeros([20,20])
#for k,i in enumerate(p):
#    for y,j in enumerate(i):        
#        for t in poolmarket.data.interval:
#            p[k][y] = p[k][y] + com[k][y] * abs(poolmarket.variables.P_nm3[t,k,y].x)

#np.savetxt('P_nm_random_abs.csv',p,fmt='%.8f',delimiter=',')

#print(np.round(p, decimals=4)) 

#b = np.argpartition(p, np.argmin(p, axis=0))[:, -6:]
#np.savetxt('b_top6.csv',b,fmt='%.8f',delimiter=',')
#print(b)










