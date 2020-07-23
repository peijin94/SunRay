import matplotlib.pyplot as plt
import numpy as np
from sunRay import plasmaFreq as pfreq
from sunRay import densityModel as dm
from sunRay import scattering as scat 
from sunRay.parameters import dev_u # use GPU if available
import sunRay.statisticalRays as raystat

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

    # plot the major 2-D distribution
    XYDistributionImage(ax_main,x,y,weights_data,bins_data)

    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    ax_histx.hist(x, bins=bins_data,weights=weights_data,histtype='stepfilled'
                , alpha=0.5, density=True,edgecolor='r',color='r',linewidth=1.5)
    ax_histy.hist(y, bins=bins_data,weights=weights_data,orientation='horizontal'
                , alpha=0.5, density=True,histtype='stepfilled',edgecolor='b',color='b',linewidth=1.5)

    ax_histx.text(0.04, 0.8,'X', fontsize=18,
        horizontalalignment='center',  verticalalignment='center',
        transform = ax_histx.transAxes)
    ax_histy.text(0.86, 0.94,'Y', fontsize=18,
        horizontalalignment='center',  verticalalignment='center',
        transform = ax_histy.transAxes)

    ax_histx.set_xlim(ax_main.get_xlim())
    ax_histy.set_ylim(ax_main.get_ylim())

    ax_histx.set_yticks([])
    ax_histy.set_xticks([]) 

    ax_histx.set_ylabel('Normalized')
    ax_histy.set_xlabel('Normalized')

    plt.show()

def XYDistributionImage(ax_main,x,y,weights_data,bins_data):

    ax_main.tick_params(direction='in', top=True, right=True)
    # the scatter plot:
    img_2d,xx,yy = np.histogram2d(x,y,weights=weights_data,bins=bins_data)
    masked_data  = np.ma.masked_where(img_2d.T<1e-2,img_2d.T)

    ax_main.imshow(masked_data,origin='low',interpolation='nearest',
        extent=[xx[0], xx[-1], yy[0], yy[-1]],cmap='magma_r')
    ax_main.set_xlabel('X [Solar Radius]')
    ax_main.set_ylabel('Y [Solar Radius]')

    (xc,yc,sx,sy,err_xc,err_yc,err_sx,err_sy) = raystat.centroidXYFWHM(x,y,weights_data)

    ax_main.plot(xc+sx/2*np.cos(np.linspace(0,2*np.pi)),yc+sy/2*np.sin(np.linspace(0,2*np.pi)),color='C9')
    ax_main.plot(np.sin(np.linspace(0,2*np.pi)),np.cos(np.linspace(0,2*np.pi)),'k-')

    # now determine nice limits by hand:
    ax_main.set_xlim([np.min(xx),np.max(xx)])
    ax_main.set_ylim([np.min(yy),np.max(yy)])



def XYVariationPlot(x_data,y_data,t_data,weights_data,t_step = 0.05,num_t_bins=-1):

    (t_bin_center,flux_all,xc_all,yc_all,sx_all,sy_all,err_xc_all,err_yc_all,
        err_sx_all,err_sy_all) = raystat.variationXYFWHM(x_data,y_data,t_data,weights_data,t_step,num_t_bins)

    (xc,yc,sx,sy,err_xc,err_yc,err_sx,err_sy) = raystat.centroidXYFWHM(x_data,y_data,weights_data)
    
    #------------- use the data FWHM
    #try:
    #    FWHM_range = raystat.FWHM(t_bin_center,flux_all)
    #except:
    #    FWHM_range=[np.nan,np.nan]

    #------------- use the fitted FWHM
    #fit_res = raystat.fit_biGaussian(t_bin_center,flux_all)
    #fitted_flux = raystat.biGaussian(t_bin_center,*fit_res)
    
    
    fit_done=True
    try:
        fit_res = raystat.fit_biGaussian(t_bin_center,flux_all)
        fitted_flux = raystat.biGaussian(t_bin_center,*fit_res)
        FWHM_range = raystat.DecayExpTime(t_bin_center,fitted_flux)
    except:
        print('fit fail')
        fit_done=False
        try:
            FWHM_range = raystat.DecayExpTime(t_bin_center,flux_all)
        except:
            FWHM_range = [0,0]
    
    
    FWHM_range = raystat.FWHM(t_bin_center,flux_all)

    print(FWHM_range[1]-FWHM_range[0])

    plt.figure(figsize=(4.5, 6))
    ax_t = plt.axes([0.1,0.65,0.8,0.3])
    # flux data
    ax_t.step(t_bin_center,flux_all/np.max(flux_all),where='mid',color='k')
    # fitted data
    if fit_done:
        ax_t.plot(t_bin_center,fitted_flux/np.max(fitted_flux),color='C9')
    ax_t.set_ylabel('Normalized')
    ax_t.tick_params(direction='in', labelbottom=False)
    ax_t.set_ylim([0,1.1])
    ax_t.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax_XY = plt.axes([0.1,0.35,0.8,0.3])

    l1=ax_XY.errorbar(t_bin_center,xc_all,err_xc_all,color='r',drawstyle='steps-mid',capsize=1,elinewidth=0.2)
    l2=ax_XY.errorbar(t_bin_center,yc_all,err_yc_all,color='b',drawstyle='steps-mid',capsize=1,elinewidth=0.2)
    ax_XY.tick_params(direction='in', labelbottom=False)
    ax_XY.set_ylabel("X,Y Position [R_s]")
    ax_XY.plot([t_bin_center[0],t_bin_center[-1]],[xc,xc],'r--',linewidth=0.6)
    ax_XY.plot([t_bin_center[0],t_bin_center[-1]],[yc,yc],'b--',linewidth=0.6)
    ax_FWHM = plt.axes([0.1,0.05,0.8,0.3])

    ax_FWHM.errorbar(t_bin_center,sx_all,err_sx_all,color='r',drawstyle='steps-mid',capsize=1,elinewidth=0.2)
    ax_FWHM.errorbar(t_bin_center,sy_all,err_sy_all,color='b',drawstyle='steps-mid',capsize=1,elinewidth=0.2)
    ax_FWHM.tick_params(direction='in')
    ax_FWHM.plot([t_bin_center[0],t_bin_center[-1]],[sx,sx],'r--',linewidth=0.6)
    ax_FWHM.plot([t_bin_center[0],t_bin_center[-1]],[sy,sy],'b--',linewidth=0.6)
    ax_FWHM.set_xlabel('Time [s]')
    ax_FWHM.set_ylabel("X,Y FWHM [R_s]")
    ax_t.legend((l1,l2),("X","Y"))

    ax_t.axvspan(FWHM_range[0], FWHM_range[1], alpha=0.1, color='k')
    ax_FWHM.axvspan(FWHM_range[0], FWHM_range[1], alpha=0.1, color='k')
    ax_XY.axvspan(FWHM_range[0], FWHM_range[1], alpha=0.1, color='k')
    
    return plt.gcf()