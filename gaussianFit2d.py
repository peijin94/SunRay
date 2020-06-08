import numpy as np
from scipy.ndimage import uniform_filter
smooth = uniform_filter
from scipy.optimize import curve_fit
#from numba import jit, prange


# finding centroids using the weighted mean method
def find_centroid(data):
    height,width = np.shape(data)
    x = np.arange(0,width)
    y = np.arange(0,height)

    X,Y = np.meshgrid(x,y)

    c_x = np.sum(X*data)/np.sum(data)
    c_y = np.sum(Y*data)/np.sum(data)

    return c_x,c_y

# define model 2D Gaussian function and pass independant variables x and y as a list
#@jit(fastmath = True, nopython = False)# accelerate
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                      + c*((y-yo)**2)))
    return g.ravel()

def guass_2d_fit(data_xy,xx,yy):
    xmin, xmax, ymin, ymax = (np.min(xx),  np.max(xx), np.min(yy),  np.max(yy))
    extent = (xmin, xmax, ymin, ymax)

    idx = np.where( np.isnan(data_xy) == True )
    data_xy[idx] = 0

    clipping = 0.7
    # replacing values under a certain level to 0
    idx = np.where( data_xy < clipping*np.max( data_xy ) )
    data_xy[idx] = 0

    x_coord = xx
    y_coord = yy
    # weighted mean centroid
    c_x,c_y = find_centroid( data_xy )
    xcentroid = xmin + (x_coord[1]-x_coord[0])*c_x #in arcsec
    ycentroid = ymin + (y_coord[1]-y_coord[0])*c_y

    # gaussian centroid
    x = np.linspace(xmin, xmax, data_xy.shape[0])
    y = np.linspace(ymin, ymax, data_xy.shape[1])
    x, y = np.meshgrid(x, y)
    data_gauss = data_xy.ravel()

    popt, pcov = curve_fit( twoD_Gaussian, (x, y), data_gauss, p0=(0.9*np.max(data_xy),xcentroid,ycentroid,300,300,0), bounds=([0,np.min(xx),np.min(yy),0,0,0],[1e30,np.max(xx),np.max(yy),3*np.max(xx),3*np.max(yy),2*np.pi+1e-5]) ,ftol=1e-2) ##1600,1300,300

    freq=35
    # shall we use the real beamsize in arcsec
    fwhm = (1.1*3e8/(freq*1e6*2000))*(180/np.pi)*60 # in arcmin

    # assume error in amplitude = 20% of maximum for low brightness bursts
    coeff_err=0.2


    return (xcentroid,ycentroid,popt)            