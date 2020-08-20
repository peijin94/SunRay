import matplotlib.pyplot as plt
import numpy as np
from sunRay import plasmaFreq as pfreq
from sunRay import densityModel as dm
from sunRay import scattering as scat 
from sunRay.parameters import dev_u,c_r # use GPU if available
import sunRay.statisticalRays as raystat

import torch


from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def showParameters(Ne_r,omega,epsilon):
    rr = torch.linspace(3,100,300).to(dev_u)
    ne = Ne_r(rr)
    fig = plt.figure(1)
    ax = plt.gca()
    ax.plot(rr.cpu(),ne.cpu())
    ax.set_xlabel('Heliocentric distance [$R_s$]')
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
    left, width = 0.12, 0.65
    bottom, height = 0.12, 0.65
    spacing = 0.004

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    fig=plt.figure(figsize=(4, 4))

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

    ax_histx.text(0.04, 0.8,'X', fontsize=12,
        horizontalalignment='center',  verticalalignment='center',
        transform = ax_histx.transAxes)
    ax_histy.text(0.86, 0.94,'Y', fontsize=12,
        horizontalalignment='center',  verticalalignment='center',
        transform = ax_histy.transAxes)

    ax_histx.set_xlim(ax_main.get_xlim())
    ax_histy.set_ylim(ax_main.get_ylim())

    ax_histx.set_yticks([])
    ax_histy.set_xticks([]) 

    ax_histx.set_ylabel('Normalized')
    ax_histy.set_xlabel('Normalized')

    return fig,ax_main
    

def XYDistributionImage(ax_main,x,y,weights_data,bins_data=100):
    ax_main.tick_params(direction='in', top=True, right=True)
    # the scatter plot:
    img_2d,xx,yy = np.histogram2d(x,y,weights=weights_data,bins=bins_data)
    masked_data  = np.ma.masked_where(img_2d.T<1e-2,img_2d.T)

    imOBJ = ax_main.imshow(masked_data,origin='low',interpolation='nearest',
        extent=[xx[0], xx[-1], yy[0], yy[-1]],cmap='magma_r')
    ax_main.set_xlabel(r'X [Solar Radius]')
    ax_main.set_ylabel(r'Y [Solar Radius]')

    (xc,yc,sx,sy,err_xc,err_yc,err_sx,err_sy) = raystat.centroidXYFWHM(x,y,weights_data)

    ax_main.plot(xc+sx/2*np.cos(np.linspace(0,2*np.pi)),yc+sy/2*np.sin(np.linspace(0,2*np.pi)),color='C9')
    ax_main.plot(np.sin(np.linspace(0,2*np.pi)),np.cos(np.linspace(0,2*np.pi)),'k-')

    # now determine nice limits by hand:
    ax_main.set_xlim([np.min(xx),np.max(xx)])
    ax_main.set_ylim([np.min(yy),np.max(yy)])

    return imOBJ

    
def MuVariationPlot(k_vec_stat_avail,t_reach_stat_avail,weights_avial,t_step = 0.005,
                    num_t_bins=-1,num_mu_bins=100):
    
    
    frameratio = 0.75
    fig = plt.figure(figsize=(4.3, 4.3/1.5))
    ax_t = plt.axes([0.12,0.16,0.7/frameratio/1.5,0.7])
    
    (t_bin_center, mu_var_all) = raystat.VariationMu(k_vec_stat_avail,t_reach_stat_avail,
                                     weights_avial,t_step = t_step,
                                    num_t_bins=num_t_bins,num_mu_bins=num_mu_bins)
    
    
    ax_t.imshow(mu_var_all.T, origin='lower',extent=[np.min(t_bin_center),np.max(t_bin_center),0,1], 
                aspect = (np.max(t_bin_center)-np.min(t_bin_center))*frameratio ,
                cmap = 'Blues')
    
    ax_t.set_xlabel(r'Time [s]')
    ax_t.set_ylabel(r"$\mu$")
    
    mu_all = k_vec_stat_avail[2,:]/np.sqrt(np.sum(k_vec_stat_avail**2,axis=0))
    
    ax_col = plt.axes([0.12+0.7/frameratio/1.5,0.16,0.22,0.7])
    ax_col.hist(mu_all, bins=np.linspace(0,1,num_mu_bins+1),weights=weights_avial,orientation='horizontal'
                , alpha=0.5, density=True,histtype='stepfilled',edgecolor='C0',color='C0',linewidth=1.5)
    ax_col.set_ylim([0,1])
    ax_col.set_yticks([])
    ax_col.set_title('Accumulated')
    return fig,ax_t,ax_col


def XYVariationPlot(x_data,y_data,t_data,weights_data,t_step = 0.05,num_t_bins=-1,x_0=0,y_0=0,offset=False,fit_all=True):
    rc('text', usetex=False)
    (t_bin_center,flux_all,xc_all,yc_all,sx_all,sy_all,err_xc_all,err_yc_all,
        err_sx_all,err_sy_all) = raystat.variationXYFWHM(x_data,y_data,
                    t_data,weights_data,t_step,num_t_bins)

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
        #FWHM_range = raystat.DecayExpTime(t_bin_center,fitted_flux)
        FWHM_peak = raystat.DecayExpTime(t_bin_center,fitted_flux)[0]
        FWHM_range = raystat.FWHM(t_bin_center,fitted_flux)
    except:
        print('fit fail')
        fit_done=False
        try:
            FWHM_range = raystat.FWHM(t_bin_center,flux_all)
            FWHM_peak = raystat.DecayExpTime(t_bin_center,flux_all)[0]
        except:
            FWHM_range = [0,0]
            FWHM_peak = 0
    
    #FWHM_range = raystat.FWHM(t_bin_center,flux_all) # force using data

    
    if offset:
        xc_all = xc_all - x_0
        yc_all = yc_all - y_0
        
        xc = xc - x_0
        yc = yc - y_0
        
    if fit_all:
        (FWHM_ab,
            pfit_xc_a,pfit_xc_b,pfit_yc_a,pfit_yc_b,
            pfit_sx_a,pfit_sx_b,pfit_sy_a,pfit_sy_b,offset_xa,
            offset_xb,offset_ya,offset_yb,
              pfit_xc_fwhm,pfit_yc_fwhm,
              pfit_sx_fwhm,pfit_sy_fwhm,offset_x_fwhm,offset_y_fwhm,
                sx_a,sx_b,sy_a,sy_b)=raystat.OffsetSpeedPhase(t_bin_center,flux_all,xc_all,yc_all,
                        sx_all,sy_all,err_xc_all,err_yc_all,
                        err_sx_all,err_sy_all,x0_all=0,y0_all=0,offset=True)


    
    print('Total duration : '+ str(FWHM_range[1]-FWHM_range[0]))

    fig=plt.figure(figsize=(5*0.75, 5))
    
    ax_t = plt.axes([0.15,0.7,0.8,0.25])
    ax_XY = plt.axes([0.15,0.4,0.8,0.3]) 
    ax_FWHM = plt.axes([0.15,0.10,0.8,0.3])

    if fit_all:
        ax_XY.plot(FWHM_ab[0:2],np.polyval(pfit_xc_a,FWHM_ab[0:2]),'darkred',zorder=10)
        ax_XY.plot(FWHM_ab[0:2],np.polyval(pfit_yc_a,FWHM_ab[0:2]),'darkblue',zorder=10)
        ax_XY.plot(FWHM_ab[1:],np.polyval(pfit_xc_b,FWHM_ab[1:]),'darkred',zorder=10)
        ax_XY.plot(FWHM_ab[1:],np.polyval(pfit_yc_b,FWHM_ab[1:]),'darkblue',zorder=10)
        
        ax_FWHM.plot(FWHM_ab[0:2],np.polyval(pfit_sx_a,FWHM_ab[0:2]),'darkred',zorder=10)
        ax_FWHM.plot(FWHM_ab[0:2],np.polyval(pfit_sy_a,FWHM_ab[0:2]),'darkblue',zorder=10)
        ax_FWHM.plot(FWHM_ab[1:],np.polyval(pfit_sx_b,FWHM_ab[1:]),'darkred',zorder=10)
        ax_FWHM.plot(FWHM_ab[1:],np.polyval(pfit_sy_b,FWHM_ab[1:]),'darkblue',zorder=10)
        
        print('Duration R/D : ' + str(np.round(np.diff(FWHM_ab),5)))
        print('Vx R/D : '+str(np.round(pfit_xc_a[0],5))+' , '+str(np.round(pfit_xc_b[0],5)))
        print('Vx R/D (c) : '+str(np.round(pfit_xc_a[0]/c_r,5))+' , '+str(np.round(pfit_xc_b[0]/c_r,5)))
        print('Vy R/D : '+str(np.round(pfit_yc_a[0],5))+' , '+str(np.round(pfit_yc_b[0],5)))
        print('Vy R/D (c) : '+str(np.round(pfit_yc_a[0]/c_r,5))+' , '+str(np.round(pfit_yc_b[0]/c_r,5)))
        print('ERx R/D : '+str(np.round(pfit_sx_a[0],5))+' , '+str(np.round(pfit_sx_b[0],5)))
        print('ERx R/D (Deg) : '+str(np.round(pfit_sx_a[0]*32/60,5))+' , '+str(np.round(pfit_sx_b[0]*32/60,5)))
        print('ERx R/D : '+str(np.round(pfit_sy_a[0],5))+' , '+str(np.round(pfit_sy_b[0],5)))
        print('ERx R/D (Deg) : '+str(np.round(pfit_sy_a[0]*32/60,5))+' , '+str(np.round(pfit_sy_b[0]*32/60,5)))
        print('Size X : '   +str(np.round(np.mean(sx_a),5))+      ' , '+str(np.round(np.mean(sx_b),5)))
        print('Size X (Deg)'+str(np.round(np.mean(sx_a)*32/60,5))+' , '+str(np.round(np.mean(sx_b)*32/60,5)))
        print('Size Y : '   +str(np.round(np.mean(sy_a),5))+      ' , '+str(np.round(np.mean(sy_b),5)))
        print('Size Y (Deg)'+str(np.round(np.mean(sy_a)*32/60,5))+' , '+str(np.round(np.mean(sy_b)*32/60,5)))
        
        print('Offset R/D  x: '+str(np.round(offset_xa,6))+' , '+str(np.round(offset_xb,6)))
        print('Offset R/D  y: '+str(np.round(offset_ya,6))+' , '+str(np.round(offset_yb,6)))
        
    para_collect = {'offset':[offset_xa,offset_xb,offset_ya,offset_yb]}
    
    # flux data
    ax_t.step(t_bin_center,flux_all/np.max(flux_all),where='mid',color='k')
    # fitted data
    if fit_done:
        ax_t.plot(t_bin_center,fitted_flux/np.max(fitted_flux),color='C9')
    ax_t.set_ylabel(r'Normalized')
    ax_t.tick_params(direction='in', labelbottom=False)
    ax_t.set_ylim([0,1.1])
    ax_t.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    
    l1=ax_XY.errorbar(t_bin_center,xc_all,err_xc_all,color='r',drawstyle='steps-mid',capsize=1,elinewidth=0.2,linewidth=0.6)
    l2=ax_XY.errorbar(t_bin_center,yc_all,err_yc_all,color='b',drawstyle='steps-mid',capsize=1,elinewidth=0.2,linewidth=0.6)
    ax_XY.tick_params(direction='in', labelbottom=False)
    
    if offset: # label the offset
        ax_XY.set_ylabel(r"X,Y Offset ($R_s$)")
    else:
        ax_XY.set_ylabel(r"X,Y Position ($R_s$)")
    
    ax_XY.plot([t_bin_center[0],t_bin_center[-1]],[xc,xc],'r--',linewidth=0.6)
    ax_XY.plot([t_bin_center[0],t_bin_center[-1]],[yc,yc],'b--',linewidth=0.6)
    
   
    ax_FWHM.errorbar(t_bin_center,sx_all,err_sx_all,color='r',drawstyle='steps-mid',capsize=1,elinewidth=0.2,linewidth=0.6)
    ax_FWHM.errorbar(t_bin_center,sy_all,err_sy_all,color='b',drawstyle='steps-mid',capsize=1,elinewidth=0.2,linewidth=0.6)
    ax_FWHM.tick_params(direction='in')
    ax_FWHM.plot([t_bin_center[0],t_bin_center[-1]],[sx,sx],'r--',linewidth=0.6)
    ax_FWHM.plot([t_bin_center[0],t_bin_center[-1]],[sy,sy],'b--',linewidth=0.6)
    ax_FWHM.set_xlabel(r'Time (s)')
    ax_FWHM.set_ylabel(r"X,Y FWHM ($R_s$)")
    ax_t.legend((l1,l2),(r"X",r"Y"))
    

    ax_t.axvspan(FWHM_range[0], FWHM_range[1], alpha=0.1, color='k')
    ax_FWHM.axvspan(FWHM_range[0], FWHM_range[1], alpha=0.1, color='k')
    ax_XY.axvspan(FWHM_range[0], FWHM_range[1], alpha=0.1, color='k')
    
    ax_t_ylim = ax_t.get_ylim()
    ax_FWHM_ylim = ax_FWHM.get_ylim()
    ax_XY_ylim = ax_XY.get_ylim()
    
    ax_t.plot([FWHM_peak,FWHM_peak],[-1,10],'k-',linewidth=0.5)
    ax_FWHM.plot([FWHM_peak,FWHM_peak],[-10,10],'k-',linewidth=0.5)
    ax_XY.plot([FWHM_peak,FWHM_peak],[-1,15],'k-',linewidth=0.5)
    
    ax_t.set_ylim(ax_t_ylim)
    ax_FWHM.set_ylim(ax_FWHM_ylim)
    ax_XY.set_ylim(ax_XY_ylim)
    
    return fig,ax_t,ax_XY,para_collect

def MovieVariationXY(x_data,y_data,t_data,weights_data,t_step = 0.2,num_t_bins=-1,
                     save_dir = 'moviedir/',xlim=[-2,2],ylim=[-2,2],
                    x_0=0,y_0=0,title_this=''):
    """
    Generate a movie of the moving source 
    """
    (t_bin_center,flux_all,xc_all,yc_all,sx_all,sy_all,err_xc_all,err_yc_all,
        err_sx_all,err_sy_all) = raystat.variationXYFWHM(x_data,y_data,
                    t_data,weights_data,t_step,num_t_bins)
    (xc,yc,sx,sy,err_xc,err_yc,err_sx,err_sy) = raystat.centroidXYFWHM(x_data,y_data,weights_data)
    
    t_bin_edge = np.zeros(t_bin_center.shape[0]+1)
    t_bin_edge[1:-1] = (t_bin_center[0:-1]+t_bin_center[1:])/2
    t_bin_edge[0]  = t_bin_center[0]  - (t_bin_center[2]-t_bin_center[1])/2
    t_bin_edge[-1] = t_bin_center[-1] + (t_bin_center[2]-t_bin_center[1])/2
    
    t_bins=t_bin_edge
    
    
    idx_cur = 0
    for idx_t_bin in np.arange(len(t_bin_center)):

        idx_in_t_range = np.where((t_data>t_bins[idx_t_bin]) 
                                & (t_data<t_bins[idx_t_bin+1]))
        #print(str(t_bins[idx_t_bin])+" [-] "+str((idx_in_t_range[0].shape)))

        if True:#(idx_in_t_range[0].shape[0])>2:

            x_im_in_t_range = x_data[idx_in_t_range]
            y_im_in_t_range = y_data[idx_in_t_range]
            weights_in_t_range = weights_data[idx_in_t_range]
            
            fig_cur = plt.figure(figsize=(4, 6), dpi= 200)
            ax_t = plt.axes([0.17,0.13,0.78,0.2])
            # flux data
            ax_t.step(t_bin_center,flux_all/np.max(flux_all),where='mid',color='k')
            
            ax_t.plot([t_bin_center[idx_t_bin],t_bin_center[idx_t_bin]],
                     [-1,10],'k-') # show t_bin_center
            
            ax_t.set_ylabel(r'Normalized')
            ax_t.tick_params(direction='in')#, labelbottom=False)
            ax_t.set_ylim([0,1.1])
            ax_t.set_xlabel(r'Time (s)')
            ax_t.set_yticks([0.2, 0.4, 0.6, 0.8, 1])

            ax_main  = plt.axes([0.17,0.4,0.78,0.78*2/3])

            imOBJ=XYDistributionImage(ax_main,x_im_in_t_range,y_im_in_t_range,weights_in_t_range,
                                bins_data = np.linspace(-2.5,2.5,140))
            
            imOBJ.set_clim([0,np.max(flux_all/50)])
            ax_main.plot(x_0,y_0,'+',color='C2',mew=3,markersize=8)
            
            ax_main.set_title(title_this)
            ax_main.set_xlim(xlim)
            ax_main.set_ylim(ylim)
            
            
            fig_cur.savefig(save_dir+ str(idx_cur).rjust(3,'0')+'.png')
            
            plt.close()
            
        idx_cur = idx_cur + 1

    
    
    
    

    return plt.gcf()
