# -*- coding: utf-8 -*-
import numpy as np
from igraph import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import time

import random
from numpy.random import MT19937
from numpy.random import default_rng
from numpy.random import RandomState, SeedSequence
from tqdm import tqdm
from time import sleep
import csv

class Data:
    def __init__(self):
        self.data = None

    def create_grid(self, n_size):
        grid_size = n_size
        grid = np.arange(grid_size*grid_size).reshape((grid_size, grid_size))
        return grid

    def create_pop(self, seed, grid, people_size):
        np.random.seed(seed)
        travelers = np.random.randint(0,(grid.size), size =(people_size))

        return travelers

    def create_commutes(self, seed, grid, mobilityRate, people):
        np.random.seed(seed)

        int_mobility = int(mobilityRate)
        probability = round(mobilityRate-int_mobility,2)

        # Sem populacao
        if   (mobilityRate ==0.0 or people.size == 0):
            commutes = []

        # Mobilidade inteira
        elif (int_mobility==0 and probability !=0.0):
            col_mobility = int_mobility+2
            commutes = np.full((people.size,col_mobility), -1) #crio commutes do tamanho da população pela quantidade de movimentos e inicializo todos com -1
            commutes[:,0] = people
            for i in range(0, people.size):
                prob = random.random()
                if (probability >= prob):
                    commutes[i, col_mobility-1] = random.randint(0,grid.size-1)

        elif (int_mobility>0 and probability == 0.0 ):
            col_mobility = int_mobility+1
            commutes = np.full((people.size,col_mobility), -1) #crio commutes do tamanho da população pela quantidade de movimentos e inicializo todos com -1
            commutes[:,0] = people
            for i in range(1,col_mobility):
                commutes[:,i] = np.random.randint(0,(grid.size-1), size =(people.size))

        elif (int_mobility>0 and probability !=0.0 ):
            col_mobility = int_mobility+2
            commutes = np.full((people.size,col_mobility), -1) #crio commutes do tamanho da população pela quantidade de movimentos e inicializo todos com -1
            commutes[:,0] = people
            for i in range(1,col_mobility-1):
                commutes[:,i] = np.random.randint(0,(grid.size-1), size =(people.size))
            for i in range(0, people.size):
                prob = np.random.random()
                if (probability >= prob):
                    commutes[i, col_mobility-1] = np.random.randint(0,grid.size-1)

        return commutes

    def create_mat_od(self, grid, travelers):
        commutes = np.asarray(travelers)
        matrix_od = np.zeros((grid.size, grid.size))
        if(commutes.size >0):
            for i in range(len(commutes)):
                for j in range(len(commutes[i])-1):
                    if (commutes[i][j] == -1) :
                        break
                    if(commutes[i][j+1] == -1):
                        break
                    matrix_od[commutes[i][j]][commutes[i][j+1]]+=1
        return matrix_od

    def create_list_od(self, travelers ):
        list_od = []
        for i in range(len(travelers)):
            for j in range(len(travelers[i])-1):
                list_od.append(travelers[i][j])
                list_od.append(travelers[i][j+1])
        return list_od

    def create_list_2d(self, values):
        aux = []
        for i in values:
            for j in i:
                aux.append(j)
        return aux

class Graph_index:
    def __init__(self):
        self.g = None

    def create_graph_adj_undirected(self, matrix_od):
        g = Graph.Weighted_Adjacency(matrix_od.tolist(), mode=ADJ_UNDIRECTED)
        return g

    def get_avg_k(self, g):
        k_avg = mean(g.degree())
        return k_avg

    def get_avg_c(self, g):
        c_avg = g.transitivity_avglocal_undirected("zero")
        return c_avg

    def get_diameter(self,g):
        d = g.diameter(directed=False,unconn=True,weights=None)
        return d

    def get_components(self,g):
        c = g.components()
        return c

class Plot_Figures:

    def __init__(self):
      self.g = None

    def plot_indexes(self,valuesY, valuesX, grid_size, mobolity_rate, name,  paint, xlabel, ylabel):
      fig = plt.figure()
      plt.plot(valuesX,valuesY, '.-',label = name, color= paint )
      #plt.xticks(numpy.arange(min(valuesX), max(valuesX)+1, int(grid_size*2)))
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=1, mode="expand", borderaxespad=0.)
      fig.savefig(str(mobolity_rate)+'_'+name+'_'+str(grid_size)+".png", format='png', dpi=800)
      plt.clf()

    def plot_indexes_norm(self,valuesY, valuesX, grid_size, mobolity_rate, name,  paint, xlabel, ylabel):
      fig = plt.figure()

      valuesY_n = (valuesY-np.min(valuesY)) /(np.max(valuesY) -np.min(valuesY) )
      plt.plot(valuesX,valuesY_n, '.-',label = name, color= paint )
      #plt.xticks(np.arange(min(valuesX), max(valuesX)+1, int(grid_size*2)))
      plt.xlabel(xlabel)
      plt.ylabel('norm values between '+str(int(min(valuesY)))+' and '+str(int((np.max(valuesY)))))
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=1, mode="expand", borderaxespad=0.)
      fig.savefig(str(mobolity_rate)+'_'+name+'_'+str(grid_size)+"_norm_.png", format='png', dpi=800)
      plt.clf()


    def plots(self, list_k, list_c, list_d, filename):

      fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, constrained_layout=True)

      x_k = np.arange(0,len(list_k))
      x_c = np.arange(0,len(list_c))
      x_d = np.arange(0,len(list_d))


      ax0.plot(list_k, '.-',label = 'Degree', color='g' )
      ax0.set_title('Degree')
      ax0.set_yticks(list_k, ['1'])

      ax1.plot(list_c, '.-',label = 'Clustering', color='b' )
      ax1.set_title('Clustering')
      ax1.set_yticks(list_c, ['1'])

      ax2.plot(list_d, '.-',label = 'Diameter', color='orange')
      ax2.set_title('Diameter')
      ax2.set_yticks(list_d, ['1'])

      plt.savefig(filename+"_graphs.png", format='png', dpi=800)

      plt.clf()

    def histogram(self, data, m_od, filename):
      plt.hist(data, bins=np.arange(len(m_od)),density=False)
      plt.savefig(filename+'histogram.png',format='png', dpi=800)

    def boxPlot(self, data, grid, valuesX, mobolity_rate, filename):
      plt.boxplot(data)
      plt.ylabel(filename)

      plt.title('Grid: '+grid)
      # save plot
      plt.savefig(str(mobolity_rate)+'_'+grid+'_'+filename+'_boxPlot.png',format='png', dpi=800)
      plt.clf()

class Plot_Graph:

    def __init__(self, g, name):
        layout = g.layout("grid")
        plot(g, layout = layout).save(name+'.png')

if __name__ == '__main__':
    pop = []
    commutes = []

    #data
    size = 4 #n²
    data = Data()
    seed_teste = 2

    grid = data.create_grid(size)

    mobility = np.arange(0,3.1,0.1)
    size_pop = [6]


    pop = data.create_pop(seed_teste, grid, size_pop[0]) #seed, grid, people_size
    commutes = data.create_commutes(seed_teste, grid,1,pop) #seed, grid, commutes, people
    m_od = data.create_mat_od(grid, commutes)

    #igraph

    index = Graph_index()
    g = index.create_graph_adj_undirected(m_od)
    g.vs["label"] = ["0", "1", "2", "3", "4","5","6","7","8","9","10","11","12","13","14","15"]
    k = index.get_avg_k(g)
    c = index.get_avg_c(g)
    d = index.get_diameter(g)
    components = index.get_components(g)
    max_components = max(components.sizes())

    print('data:')
    print(pop)
    print(commutes)
    print(m_od)
    print(g)

    print('indexes')
    print(k)
    print(c)
    print(d)
    print(components)
    print(max_components)

    Plot_Graph(g,'graph')
