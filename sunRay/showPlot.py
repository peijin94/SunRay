import matplotlib.pyplot as plt
import numpy as np
from sunRay import plasmaFreq as pfreq
from sunRay import densityModel as dm
from sunRay import scattering as scat 
from sunRay.parameters import dev_u # use GPU if available

import torch

def showParameters(Ne_r,omega,epsilon):
    rr = torch.linspace(3,100,300).to(dev_u)
    ne = Ne_r(rr)
    fig = plt.figure(1)
    ax = plt.gca()
    ax.plot(rr.cpu(),ne.cpu())
    ax.set_xlabel('Heliocentric distance [Rs]')
    ax.set_ylabel('Electron Density [cm-3}]')
    ax.set_yscale('log')

    nu_r = scat.nuScattering(rr,omega,epsilon).cpu()
    fig = plt.figure(2)
    ax1 = plt.gca()
    ax1.plot(rr.cpu(),nu_r.cpu())
    ax1.set_xlabel('Heliocentric distance [Rs]')
    ax1.set_ylabel('Scattering nu [cm-3}]')
    ax1.set_yscale('log')

def showResultR(r_vec_cur):
    fig2,axs = plt.subplots(1,3)
    fig2.set_size_inches(18, 6.5)
    for num in range(np.min(np.array([400,r_vec_cur.shape[2]]))-1):
        axs[0].plot(r_vec_cur[:,2,num],r_vec_cur[:,1,num])
        axs[1].plot(r_vec_cur[:,2,num],r_vec_cur[:,0,num])


    axs[0].set_xlabel('Z (Rs)')
    axs[0].set_ylabel('Y (Rs)')
    axs[0].set_aspect('equal')

    axs[1].set_xlabel('Z (Rs)')
    axs[1].set_ylabel('X (Rs)')
    axs[1].set_aspect('equal')
    
    axs[2].plot(r_vec_cur[-1,0,:],r_vec_cur[-1,1,:],'k.',markersize=0.5)
    axs[2].plot(np.sin(np.linspace(0,np.pi*2,300)),
        np.cos(np.linspace(0,np.pi*2,300)),'r')
    axs[2].set_xlabel('X (Rs)')
    axs[2].set_ylabel('Y (Rs)')
    axs[2].set_aspect('equal')
    #axs[4].plot()

def XYDistributionImageHist(x_data,y_data,weights_data=1,
                            x_lim=[-2.5,2.5],y_lim=[-2.5,2.5],
                            bins_data = np.linspace(-2.5,2.5,120),
                            plot_sun =True):

    x = x_data
    y = y_data

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.004


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(5, 5))

    ax_main = plt.axes(rect_scatter)
    ax_main.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    img_2d,xx,yy = np.histogram2d(x_data,y_data,weights=weights_data,bins=bins_data)

    print(np.max(img_2d))

    masked_data  = np.ma.masked_where(img_2d.T<1e-2,img_2d.T)

    ax_main.imshow(masked_data,origin='low',interpolation='nearest',
        extent=[xx[0], xx[-1], yy[0], yy[-1]],cmap='magma_r')
    ax_main.set_xlabel('X [Solar Radius]')
    ax_main.set_ylabel('Y [Solar Radius]')

    
    xc = (np.mean(x*weights_data) / 
                    np.mean(weights_data))
    yc = (np.mean(y*weights_data) / 
                    np.mean(weights_data))
    
    sx=np.sqrt(np.mean(weights_data*(x-xc)**2)/
                np.mean(weights_data))*2.355
    sy=np.sqrt(np.mean(weights_data*(y-yc)**2)/
                np.mean(weights_data))*2.355

    ax_main.plot(xc+sx*np.cos(np.linspace(0,2*np.pi)),yc+sy*np.sin(np.linspace(0,2*np.pi)),color='C9')

    ax_main.plot(np.sin(np.linspace(0,2*np.pi)),np.cos(np.linspace(0,2*np.pi)),'k-')

    # now determine nice limits by hand:
    ax_main.set_xlim([np.min(xx),np.max(xx)])
    ax_main.set_ylim([np.min(yy),np.max(yy)])

    ax_histx.hist(x, bins=bins_data,weights=weights_data,histtype='stepfilled'
                , alpha=0.5, density=True,edgecolor='r',color='r',linewidth=1.5)
    ax_histy.hist(y, bins=bins_data,weights=weights_data,orientation='horizontal'
                , alpha=0.5, density=True,histtype='stepfilled',edgecolor='b',color='b',linewidth=1.5)

    ax_histx.text(0.04, 0.8,'X', fontsize=18,
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax_histx.transAxes)


    ax_histy.text(0.86, 0.94,'Y', fontsize=18,
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax_histy.transAxes)


    ax_histx.set_xlim(ax_main.get_xlim())
#    ax_histx.set_ylim([0,2])
    ax_histy.set_ylim(ax_main.get_ylim())
#    ax_histy.set_xlim([0,2])

    ax_histx.set_yticks([])
    ax_histy.set_xticks([]) 

    ax_histx.set_ylabel('Normalized')
    ax_histy.set_xlabel('Normalized')

    plt.show()


def XYVariation(x_data,y_data,t_data,weights_data=1,
                            x_lim=[-2.5,2.5],y_lim=[-2.5,2.5],
                            bins_data = np.linspace(-2.5,2.5,120),
                            plot_sun =True):

    plt.figure(figsize=(9, 5))
    ax_main = plt.axes([0.1,0.1,0.8,0.2])
    hst = np.histogram(t_data,150,weights=weights_data)
    

    pass