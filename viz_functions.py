# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:17:23 2020

@author: rfuchs
"""

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

def plot_2Dcyto(X, y, tn, q1, q2, colors = None):
    ''' Plot a 2D cytogram of dimension q1 vs dimension q2 
    X (n x curve_length x nb_curves ndarray): The curves representing the particules
    '''
    
    if colors == None:
        #colors = ['#96ceb4', 'gold', 'black', 'green', 'grey', 'red', 'purple', 'blue', 'silver']
        colors = ['#96ceb4', 'gold', 'lawngreen', 'black', 'green', 'red', 'purple', 'blue', 'grey']

    fig, ax1 = plt.subplots(1,1, figsize=(12,6))
    for id_, label in enumerate(list(tn['label'])):
        obs = X[y == label]
        ax1.scatter(obs[q1], obs[q2], c = colors[id_], label= label, s = 1)
        ax1.legend(loc= 'upper left', shadow=True, fancybox=True, prop={'size':8})
    
    ax1.set_title('True :' +  q1 + ' vs ' + q2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(q1)
    ax1.set_ylabel(q2)
    ax1.set_xlim(1, 10**6)
    ax1.set_ylim(1, 10**6)
    plt.show()



def plot_2D(preds, tn, q1, q2, loc = 'upper left', title = None, colors = None): # Change name
    ''' Plot 2D cytograms as for manual classification. True vs Pred '''
    
    if colors == None:
        #colors = ['#96ceb4', 'gold', 'black', 'green', 'grey', 'red', 'purple', 'blue', 'silver']
        colors = ['#96ceb4', 'gold', 'lawngreen', 'black', 'green', 'red', 'purple', 'blue', 'grey']
        
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,4))
    for id_, label in enumerate(list(tn['Particle_class'])):
        obs = preds[preds['True FFT Label'] == label]
        ax1.scatter(obs[q1], obs[q2], c = colors[id_], label= label, s=1)
        ax1.legend(loc= loc, shadow=True, fancybox=True, prop={'size':8})
    
    ax1.set_title('True :' +  q1 + ' vs ' + q2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(q1)
    ax1.set_ylabel(q2)
    ax1.set_xlim(1, 10**6)
    ax1.set_ylim(1, 10**6)
    
    
    for id_, label in enumerate(list(tn['Particle_class'])):
        obs = preds[preds['Pred FFT Label'] == label]
        ax2.scatter(obs[q1], obs[q2], c = colors[id_], label= label, alpha=1, s=1)
        ax2.legend(loc= loc, shadow=True, fancybox=True, prop={'size':8})
    ax2.set_title('Pred :' +  q1 + ' vs ' + q2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(q1)
    ax2.set_ylabel(q2)
    ax2.set_xlim(1, 10**6)
    ax2.set_ylim(1, 10**6)
    
    if title != None:
        plt.savefig(title)

def plot_decision_boundaries(preds, tn, q1, q2, loc = 'upper left', title = None, colors = None):
    if colors == None:
        colors = ['#96ceb4', 'gold', 'lawngreen', 'black', 'green', 'red', 'purple', 'blue', 'grey']
        
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,4))
    for id_, label in enumerate(list(tn['Particle_class'])):
        obs = preds[preds['True FFT Label'] == label]
        ax1.fill(obs[q1], obs[q2], c = colors[id_], label = label)
        ax1.legend(loc = loc, shadow=True, fancybox=True, prop={'size':8})
    
    ax1.set_title('True :' +  q1 + ' vs ' + q2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(q1)
    ax1.set_ylabel(q2)
    ax1.set_xlim(1, 10**6)
    ax1.set_ylim(1, 10**6)

    for id_, label in enumerate(list(tn['Particle_class'])):
        obs = preds[preds['Pred FFT Label'] == label]
        ax2.fill(obs[q1], obs[q2], c = colors[id_], label = label)
        ax2.legend(loc = loc, shadow=True, fancybox=True, prop={'size':8})
    
    ax2.set_title('Pred :' +  q1 + ' vs ' + q2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(q1)
    ax2.set_ylabel(q2)
    ax2.set_xlim(1, 10**6)
    ax2.set_ylim(1, 10**6)
   
    if title != None:
        plt.savefig(title)
    
    
def plot_decision_boundaries2(preds, tn, q1, q2, loc = 'upper left', title = None, colors = None):
    if colors == None:
        colors = ['#96ceb4', 'gold', 'lawngreen', 'black', 'green', 'red', 'purple', 'blue', 'grey']
        
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,4))
    for id_, label in enumerate(list(tn['Particle_class'])):
        obs = preds[preds['True FFT Label'] == label]
        if len(obs) == 0:
            continue
        points = obs[[q1, q2]].values
        hull = ConvexHull(points)
    
        cent = np.mean(points, 0)
        pts = []
        for pt in points[hull.simplices]:
            pts.append(pt[0].tolist())
            pts.append(pt[1].tolist())
        
        pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                        p[0] - cent[0]))
        pts = pts[0::2]  # Deleting duplicates
        pts.insert(len(pts), pts[0])
        k = 1.1
        color = 'green'
        poly = Polygon(k*(np.array(pts)- cent) + cent,
                       facecolor=colors[id_], alpha=0.2,\
                       label = label)
        poly.set_capstyle('round')
        ax1.add_patch(poly)
     
        #ax1.fill(obs[q1], obs[q2], c = colors[id_], label = label)
        ax1.legend(loc = loc, shadow=True, fancybox=True, prop={'size':8})
    
    ax1.set_title('True :' +  q1 + ' vs ' + q2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(q1)
    ax1.set_ylabel(q2)
    ax1.set_xlim(1, 10**6)
    ax1.set_ylim(1, 10**6)

    for id_, label in enumerate(list(tn['Particle_class'])):
        obs = preds[preds['Pred FFT Label'] == label]
        if len(obs) == 0:
            continue
        points = obs[[q1, q2]].values
        hull = ConvexHull(points)
    
        cent = np.mean(points, 0)
        pts = []
        for pt in points[hull.simplices]:
            pts.append(pt[0].tolist())
            pts.append(pt[1].tolist())
        
        pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                        p[0] - cent[0]))
        pts = pts[0::2]  # Deleting duplicates
        pts.insert(len(pts), pts[0])
        k = 1.1
        color = 'green'
        poly = Polygon(k*(np.array(pts)- cent) + cent,
                       facecolor=colors[id_], alpha=0.2,\
                       label = label)
        poly.set_capstyle('round')
        ax2.add_patch(poly)
     
        #ax1.fill(obs[q1], obs[q2], c = colors[id_], label = label)
        ax2.legend(loc = loc, shadow=True, fancybox=True, prop={'size':8})
    
    ax2.set_title('Pred :' +  q1 + ' vs ' + q2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(q1)
    ax2.set_ylabel(q2)
    ax2.set_xlim(1, 10**6)
    ax2.set_ylim(1, 10**6)
   
    if title != None:
        plt.savefig(title)