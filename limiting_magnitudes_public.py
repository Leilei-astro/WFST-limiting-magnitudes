#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:09:56 2023

@author: leilei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 23:15:42 2023

@author: leilei
"""

from scipy import interpolate
import numpy as np    
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.integrate import simps
from scipy.integrate import newton_cotes
import scipy.constants as sciconst
from scipy import integrate

#%% load data, class of limiting magnitudes code (数据加载+积分程序+极限星等程序)

####################################    attension: please correct the path here (注意：在这里修改为你的数据存放路径)
path='/path to your data/' #'/Users/leilei/Documents/works/WFST/data/final/'
####################################    attension: write a moon phase angle value here, the value only can be modifed into one on value from "0", "45", "90", "135", "180". (注意：在这里设置月相角度，只能输入0，45，90，135，180这五个数值之一，比如现在是45度)
moonphase_angle=45
####################################

u,g,r,i,z,w=np.loadtxt(path+"u_trans.txt",skiprows = 2),np.loadtxt(path+"g_trans.txt",skiprows = 2),np.loadtxt(path+"r_trans.txt",skiprows = 2),np.loadtxt(path+"i_trans.txt",skiprows = 2),np.loadtxt(path+"z_trans.txt",skiprows = 2),np.loadtxt(path+"w_trans.txt",skiprows = 2)

optics=np.loadtxt(path+"trans_tot_sys_sci.txt", skiprows = 6)
atmos=np.loadtxt(path+"site4200_sky_trans.txt", skiprows = 0)

johnson_V_eff_zqf=np.loadtxt(path+"filt_trans.txt",skiprows = 3)
johnson_V_eff_zqf=np.vstack((johnson_V_eff_zqf[:,0]/10,johnson_V_eff_zqf[:,1])).T

sky_spec0=np.loadtxt(path+"site4200_sky_flux_0Moon.txt", skiprows = 0)
sky_spec45=np.loadtxt(path+"site4200_sky_flux_45Moon.txt", skiprows = 0)
sky_spec90=np.loadtxt(path+"site4200_sky_flux_90Moon.txt", skiprows = 0)
sky_spec135=np.loadtxt(path+"site4200_sky_flux_135Moon.txt", skiprows = 0)
sky_spec180=np.loadtxt(path+"site4200_sky_flux_180Moon.txt", skiprows = 0)

def moonphase_n(moonphase_angle):
    if moonphase_angle==0:
        n0=0
    if moonphase_angle==45:
        n0=1
    if moonphase_angle==90:
        n0=2
    if moonphase_angle==135:
        n0=3
    if moonphase_angle==180:
        n0=4
    return n0
sky_spec=np.array((sky_spec0,sky_spec45,sky_spec90,sky_spec135,sky_spec180))[moonphase_n(moonphase_angle)]
#%
wave=sky_spec[:7001, 0]

u_wave,g_wave,r_wave,i_wave,z_wave,w_wave=wave,wave,wave,wave,wave,wave#

def interp_filter(band):
    return interpolate.interp1d(band[:,0],band[:,1],kind='cubic')(wave)
u_f,g_f,r_f,i_f,z_f,w_f=interp_filter(u),interp_filter(g),interp_filter(r),interp_filter(i),interp_filter(z),interp_filter(w)#

def interpeff(band):
    return interpolate.interp1d(optics[:,0],optics[:,1],kind='cubic')(wave)*interpolate.interp1d(band[:,0],band[:,1],kind='cubic')(wave)
u_eff,g_eff,r_eff,i_eff,z_eff,w_eff=interpeff(u),interpeff(g),interpeff(r),interpeff(i),interpeff(z),interpeff(w)#

def interp_filter_sky(band):
    return interpolate.interp1d(band[:,0],band[:,1],kind='cubic')(wave)*sky_spec[:7001,1]
uf_sky,gf_sky,rf_sky,if_sky,zf_sky,wf_sky=interp_filter_sky(u),interp_filter_sky(g),interp_filter_sky(r),interp_filter_sky(i),interp_filter_sky(z),interp_filter_sky(w)#

def interpsky(band):
    return interpolate.interp1d(optics[:,0],optics[:,1],kind='cubic')(wave)*interpolate.interp1d(band[:,0],band[:,1],kind='cubic')(wave)*sky_spec[:7001,1]
u_sky,g_sky,r_sky,i_sky,z_sky,w_sky=interpsky(u),interpsky(g),interpsky(r),interpsky(i),interpsky(z),interpsky(w)


#% integration of the night sky spectrum (积分光谱的程序)
def idl_tabulate(x, f, p=5) :
    def newton_cotes_(x, f) :
        if x.shape[0] < 2 :
            return 0
        rn = (x.shape[0] - 1) * (x - x[0]) / (x[-1] - x[0])
        weights = newton_cotes(rn)[0]
        return (x[-1] - x[0]) / (x.shape[0] - 1) * np.dot(weights, f)
    ret = 0
    for idx in np.arange(0, x.shape[0], p - 1) :
        ret += newton_cotes_(x[idx:idx + p], f[idx:idx + p])
    return ret

a=1700
b=4001
v_wave=sky_spec[a:b,0]
v_f=johnson_V_eff_zqf[:,1]
vf_sky=johnson_V_eff_zqf[:,1]*sky_spec[a:b,1]

#% Limiting magnetudes (极限星等程序)
def getlim(exptime, seeing=0.75, bgsky=22.3, airmass=1.2, n_frame=1.0, sig=5.0, nu=0.0):#n_frame 帧数（次数）
    eta=1-nu
    plate_scale = 33.2685    #arcsec/mm
    pixel_scale = 0.332685 #arcsec <--- 10 microns plate
    ron       = 8.0        #e-/pixel/s @ -100 degree   rms
    dn        = 0.005      #e-/pixel at 0.5MHz                  
    
    # input parameters
    eff_area  = np.pi*125**2*0.84   #cm^2   Here we corrected the effective area by considering obscured area and hole area.  (中间1m口径去掉)，np.pi*50**2/(np.pi*125**2)=0.16
    fwhm_seeing_550=seeing #arcsec
    dome_seeing=0.4        #arcsec
    fwhm_tel=0.485063      #arcsec
    fwhm_opt=0.367813      #arcsec
    fwhm_ao_err=0.3        #arcsec
    fwhm_tracking=0.1      #arcsec
    
    lam0=np.array([356.17,476.34, 620.57, 753.07, 870.45, 612.15])
    
    fwhm_seeing=fwhm_seeing_550*(lam0/500.0)**(-0.2)*airmass**0.6
    
    fwhm_tot=1.18*np.sqrt(fwhm_seeing**2+fwhm_tel**2+dome_seeing**2)
    
    pixelnum1d=fwhm_tot/pixel_scale
    pixel_area=pixel_scale**2
    pixelnum_band=np.pi*fwhm_tot**2/4/pixel_area
    
    mag_sky_v=21.73848978217282 # -2.5*np.log10(idl_tabulate(sky_spec[a:b,0], vf_sky/1e4*sciconst.Planck*sciconst.speed_of_light*1e7/(sky_spec[a:b,0]*1e-10), p=5)/idl_tabulate(sky_spec[a:b,0],johnson_V_eff_zqf[:,1], p=5))-21.1
    scaling=10**((-bgsky+mag_sky_v)/2.5)
    
    X=(1-0.96*np.sin(np.arccos(1/airmass))**2)**(-0.5)
    index=10**(-0.172*(X-1)/2.5)*X
    
    counts_sky_filter=np.array([idl_tabulate(u_wave,uf_sky/1e4, p=5),idl_tabulate(g_wave,gf_sky/1e4, p=5),idl_tabulate(r_wave,rf_sky/1e4, p=5),idl_tabulate(i_wave,if_sky/1e4, p=5),idl_tabulate(z_wave,zf_sky/1e4, p=5),idl_tabulate(w_wave,wf_sky/1e4, p=5)])*index*scaling #counts/s/cm^2/arcsec^2,  The  photon counts of integrated spectrum passed the WFST six filters    (积分到WFST滤波片下的天光子数流量，其中由于光谱是模拟的V波段星等为20.78时的光谱（强度），跟台址无关，我们假设WFST的天光V波段星等是22.3等，只要直接做一下这两个星等之间差别导致的scaling=10**((22.3-20.78)/2.5)。)
    counts_sky_alleff=np.array([idl_tabulate(u_wave,u_sky/1e4, p=5),idl_tabulate(g_wave,g_sky/1e4, p=5),idl_tabulate(r_wave,r_sky/1e4, p=5),idl_tabulate(i_wave,i_sky/1e4, p=5),idl_tabulate(z_wave,z_sky/1e4, p=5),idl_tabulate(w_wave,w_sky/1e4, p=5)])*index*scaling#/10**((bgsky-20.78)/2.5)#counts/s/cm^2/arcsec^2,     The photon counts of integrated spectrum passed all of the instruments and gathered in final image, including WFST six filters, optics, reflectors, correctors, ADC and CCD.   (天光通过不止滤波片，通过了光学透镜、反射镜、改正镜、大气消散镜、CCD响应的卷积光子数)
    
    count_zero_point_filter=np.array([idl_tabulate(u_wave,3631/(3.34e4)/(u_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(u_wave)*u_f, p=5),idl_tabulate(g_wave,3631/(3.34e4)/(g_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(g_wave)*g_f, p=5),idl_tabulate(r_wave,3631/(3.34e4)/(r_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(r_wave)*r_f, p=5),idl_tabulate(i_wave,3631/(3.34e4)/(i_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(i_wave)*i_f, p=5),idl_tabulate(z_wave,3631/(3.34e4)/(z_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(z_wave)*z_f, p=5),idl_tabulate(w_wave,3631/(3.34e4)/(w_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(w_wave)*w_f, p=5)])#counts/s/cm^2   The Zero Point, Vega flux passed the WFST six filters.  (Vega星在所有波段流量为3631Jy定义为零点流量，零点流量从频域转换到波长下的流量卷积滤波片的光子数流量)
    
    count_zero_point_alleff=np.array([idl_tabulate(u_wave,3631/(3.34e4)/(u_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(u_wave)*u_eff*(atmos[:7001,1])**airmass, p=5),idl_tabulate(g_wave,3631/(3.34e4)/(g_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(g_wave)*g_eff*(atmos[:7001,1])**airmass, p=5),idl_tabulate(r_wave,3631/(3.34e4)/(r_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(r_wave)*r_eff*(atmos[:7001,1])**airmass, p=5),idl_tabulate(i_wave,3631/(3.34e4)/(i_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(i_wave)*i_eff*(atmos[:7001,1])**airmass, p=5),idl_tabulate(z_wave,3631/(3.34e4)/(z_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(z_wave)*z_eff*(atmos[:7001,1])**airmass, p=5),idl_tabulate(w_wave,3631/(1.0/2.998e-5)/(w_wave)**2/(sciconst.Planck*sciconst.speed_of_light*1e10*1e7)*(w_wave)*w_eff*(atmos[:7001,1])**airmass, p=5)])#counts/s/cm^2  The Zero Point, Vega flux passed all instrements.    (Vega星在所有波段流量为3631Jy定义为零点流量，零点流量从频域转换到波长下的流量卷积通过不止滤波片，通过了光学透镜、反射镜、改正镜、大气消散镜、CCD响应的光子数流量 # 3.34e4 ---> 1.0/2.998e-5)
    
    mag_sky0=-2.5*np.log10(counts_sky_filter/count_zero_point_filter)#  sky magnitudes  (定义好了零点流量，卷积模板得到了WFST天光，用相对流量计算天光对应的星等)
    
    bg = counts_sky_alleff*eff_area*exptime*pixel_area#*seeing #count    sky total counts  (天光在WFST下的总光子数 counts_sky_alleff*eff_area*np.pi*Ra**2#)
    rd = ron**2 #count  read out noise  
    dc = (dn*exptime) #count  dark noise
    
    
    ###################### Method 1: interplote method to calculate Limiting Magnitudes  (假定的一系列不同亮度（sigma）的source来插值得到5sigma极限星等)
    mag = np.arange(10.0,30.01,0.001) #  interplote method to calculate Limiting Magnitudes  (假定的一系列不同亮度的source)
    snra = np.zeros([6,len(mag)]) #  signal to noise ratio  (待记录信噪比数组)
    err = np.zeros([6,len(mag)]) #  error        (待记录误差数组)
    for j in np.arange(0,len(mag)):
        magj = np.ones((6))*mag[j]  #6 bands magnitudes     星等
        count_source=(count_zero_point_alleff*0.61)*10**(-magj/2.5)  #counts/s/cm^2   photons     假设的source的光子数流量
        count_s = (count_source*eff_area*exptime) # count of source
        snr = count_s*eta/np.sqrt(count_s*eta + 2*pixelnum_band*(bg*eta + dc+ rd))  #source的信噪比，其中1*或者2*是指（不扣除背景）【无扣除带来的涨/落的误差叠加】或者（扣除背景）【扣除背景会带来涨和落的误差，需要乘以2】；类似的在红外望远镜当中，由于温度的\delta(T-T0)和\delta(T+T0)两个误差叠加会需要乘以2.
        snra[:,j] = snr*np.sqrt(1.0*n_frame)  #记录信噪比
        err[:,j] = 2.5*np.log10(1+1/snra[:,j])
    maglm = np.zeros(6)
    for ii in np.arange(0,len(maglm)):
        fmi = interpolate.interp1d(snra[ii,:], mag) #某个波段的 信噪比-星等 插值函数
        maglm[ii] = fmi(sig)  #在sig信噪比下插值出对应的极限星等
    
    ##################### Method 2: Analytical Solution of Limiting Magnitudes  
    S=(sig**2+np.sqrt(sig**4+8*n_frame*(sig**2)*pixelnum_band*(eta*bg+dc+rd)))/(2*n_frame*exptime*eta*eff_area) #     解析计算极限星等的结果
    
    return(maglm,mag_sky0,snra,mag,err,count_zero_point_filter,counts_sky_filter,fwhm_tot,pixelnum1d,pixelnum_band,counts_sky_alleff,count_zero_point_alleff) #  Retuen the limiting magnotudes in six band, and sky mag, ....              (返回极限星等和误差等等结果)

#%% print result of 5-sigma Limiting Magnitudes + errors

np.set_printoptions(precision=2, suppress=True)
print('V-band Sky: 22.0 mag, confidence level: 3 sigma')
print('band  ',' u    ','g    ','r    ','i    ','z    ','w    ')
#print(getlim(30,airmass=1.2))
print('sky   ',getlim(30, bgsky=22.0, n_frame=1,airmass=1.2, sig=3.0)[1]) # WFST night sky in ugrizw six bands, V-bandSky=22.0
print('30s   ',getlim(30, bgsky=22.0, n_frame=1,airmass=1.2, sig=3.0)[0]) # limiting magnitudes of six bands in 30 s single exposure mode.
print('3ks   ',getlim(30, bgsky=22.0, n_frame=100,airmass=1.2, sig=3.0)[0])# limiting magnitudes of six bands in 30 s 100 frames exposure mode.
print('V-band Sky: 22.3 mag, confidence level: 3 sigma')
print('band  ',' u    ','g    ','r    ','i    ','z    ','w    ')
#print(getlim(30,airmass=1.2))
print('sky   ',getlim(30, bgsky=22.3, n_frame=1,airmass=1.2, sig=3.0)[1]) # WFST night sky in ugrizw six bands, V-bandSky=22.0
print('30s   ',getlim(30, bgsky=22.3, n_frame=1,airmass=1.2, sig=3.0)[0]) #
print('3ks   ',getlim(30, bgsky=22.3, n_frame=100,airmass=1.2, sig=3.0)[0])#

print('V-band Sky: 22.0 mag, confidence level: 5 sigma')
print('band  ',' u    ','g    ','r    ','i    ','z    ','w    ')
#print(getlim(30,airmass=1.2))
print('sky   ',getlim(30, bgsky=22.0, n_frame=1,airmass=1.2, sig=5.0)[1]) # WFST night sky in ugrizw six bands, V-bandSky=22.0
print('30s   ',getlim(30, bgsky=22.0, n_frame=1,airmass=1.2, sig=5.0)[0]) #
print('3ks   ',getlim(30, bgsky=22.0, n_frame=100,airmass=1.2, sig=5.0)[0])#
print('V-band Sky: 22.3 mag, confidence level: 5 sigma')
print('band  ',' u    ','g    ','r    ','i    ','z    ','w    ')
#print(getlim(30,airmass=1.2))
print('sky   ',getlim(30, bgsky=22.3, n_frame=1,airmass=1.2, sig=5.0)[1]) # WFST night sky in ugrizw six bands, V-bandSky=22.0
print('30s   ',getlim(30, bgsky=22.3, n_frame=1,airmass=1.2, sig=5.0)[0]) #
print('3ks   ',getlim(30, bgsky=22.3, n_frame=100,airmass=1.2, sig=5.0)[0])#

print('V-band Sky: 22.0 mag, confidence level: 10 sigma')
print('band  ',' u    ','g    ','r    ','i    ','z    ','w    ')
#print(getlim(30,airmass=1.2))
print('sky   ',getlim(30, bgsky=22.0, n_frame=1,airmass=1.2, sig=10.0)[1]) # WFST night sky in ugrizw six bands, V-bandSky=22.0
print('30s   ',getlim(30, bgsky=22.0, n_frame=1,airmass=1.2, sig=10.0)[0]) #
print('3ks   ',getlim(30, bgsky=22.0, n_frame=100,airmass=1.2, sig=10.0)[0])#
print('V-band Sky: 22.3 mag, confidence level: 10 sigma')
print('band  ',' u    ','g    ','r    ','i    ','z    ','w    ')
#print(getlim(30,airmass=1.2))
print('sky   ',getlim(30, bgsky=22.3, n_frame=1,airmass=1.2, sig=10.0)[1]) # WFST night sky in ugrizw six bands, V-bandSky=22.0
print('30s   ',getlim(30, bgsky=22.3, n_frame=1,airmass=1.2, sig=10.0)[0]) #
print('3ks   ',getlim(30, bgsky=22.3, n_frame=100,airmass=1.2, sig=10.0)[0])#

