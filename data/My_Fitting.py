# 这里我们来整理一下之前的fitting函数，希望能够方便此后系统的调用，不需要每次都粘贴来粘贴去

import numpy as np
from jaxfit import CurveFit
import jax.numpy as jnp
from tqdm import tqdm
import scipy
from scipy import stats

def fit_extract_h(t, fixed_params, N_free, *params):
    # fixed_params: [[omega_r,omega_i],[,],...]
    # params: 2*N_fix: A,phi + 4*N_free A,phi,omega_r,omega_i

    h=0

    # fixed part
    N_fix=len(fixed_params)
    for i in range(N_fix):
        A=params[0][2*i]
        phi=params[0][2*i+1]
        omega_r=fixed_params[i][0]
        omega_i=fixed_params[i][1]

        h_i=A*jnp.exp(-1.0j*(omega_r+1.0j*omega_i)*t[1]-1.0j*phi)
        h_real=h_i.real
        h_imag=h_i.imag
        h+=(1-t[0])*h_real+t[0]*h_imag

    # free part
    for i in range(N_free):
        A=params[0][2*N_fix+4*i]
        phi=params[0][2*N_fix+4*i+1]
        omega_r=params[0][2*N_fix+4*i+2]
        omega_i=params[0][2*N_fix+4*i+3]

        h_i=A*jnp.exp(-1.0j*(omega_r+1.0j*omega_i)*t[1]-1.0j*phi)
        h_real=h_i.real
        h_imag=h_i.imag
        h+=(1-t[0])*h_real+t[0]*h_imag

    return h

# the loss function
def fit_extract_loss(fixed_params, N_free, *params, time_array, h_array):
    loss=0

    for i in range(len(time_array[0,:])):
        h=fit_extract_h(time_array[:,i],fixed_params,N_free,params)
        loss+=(h-h_array[i])**2

    return loss

# 这就是我们最主要的函数，其设计应该是从给定的时间信号中提取QNM
# 其输入参数与解释如下
# t: 数组，时间：由于不同波形的预处理可能不一样，我们这里直接要求输入处理好了的时间，
#    即从t从0开始到t_end
# h_r,h_i: 显然是波形的实部和虚部，用这种写法的话，输入h还是psi4其实是一样的，要与时间对应
# t0_arr: 这个也从外面给，是外面系列拟合的起始时间点
# N_guess: 最开始初始值的猜测次数，意义不大
# N_free: 我们采用最传统的拟合方式，即考虑有N_free，N_fix mode，暂时不特殊处理一些理论
# N_fix: 如有必要可以固定一些模式的频率来进行提取，需要同时给定omega_fix
# omega_fix: 数组，[[omega_r, omega_i], [omega_r, omega_i], ...] 长度是N_fix
# pert: 拟合的时候每次拟合给初值加的扰动，当然设成0就是老办法了
# N_pert_guess: 扰动的话，会从N_pert_guess里选出loss最低的一个，同时总是保留一个使用未扰动的参数
# max_nfev: 取200000就完了，是允许的拟合次数
def fit_modes(t,h_r,h_i,t0_arr,N_guess,N_free,N_fix,omega_fix,pert,N_pert_guess,max_nfev):
    fixed_length=2*len(t)


    # 首先先创建初值
    jcf=CurveFit(flength=fixed_length)
    loss=np.zeros(N_guess)
    initial_guess=np.zeros([N_guess,2*N_fix+4*N_free])
    guess_fit=[]

    time_array=np.zeros([2,2*len(t)])
    h_array=np.zeros(2*len(t))
    for k in range(len(t)):
        time_array[0,2*k]=0
        time_array[0,2*k+1]=1
        time_array[1,2*k]=t[k]
        time_array[1,2*k+1]=time_array[1,2*k]
        h_array[2*k]=h_r[k]
        h_array[2*k+1]=h_i[k]

    h_peak=(h_r[0]**2+h_i[0]**2)**(1/2)

    print("initial guess:")
    for ig in tqdm(range(N_guess)):
        for k in range(N_fix):
            initial_guess[ig,2*k]=h_peak*scipy.stats.loguniform.rvs(0.1,10)
            initial_guess[ig,2*k+1]=scipy.stats.uniform.rvs(0, 2*np.pi)
        for k in range(N_free):
            initial_guess[ig,2*N_fix+4*k]=h_peak*scipy.stats.loguniform.rvs(0.1,10)
            initial_guess[ig,2*N_fix+4*k+1]=scipy.stats.uniform.rvs(0, 2*np.pi)
            initial_guess[ig,2*N_fix+4*k+2]=scipy.stats.uniform.rvs(-2,4)
            initial_guess[ig,2*N_fix+4*k+3]=scipy.stats.uniform.rvs(-1,0)
        try:
            # 防一手拟合失败
            popt, pcov = jcf.curve_fit(lambda t, *params: fit_extract_h(t,omega_fix,N_free, params), \
            time_array, h_array, p0=initial_guess[ig,:], max_nfev=max_nfev,method="trf")
            guess_fit.append(popt)
        except RuntimeError as e:
            # 如果拟合失败就直接摆烂使用初值
            guess_fit.append(initial_guess[ig,:])

    for ig in range(N_guess):
        loss[ig]=fit_extract_loss(omega_fix, N_free, *guess_fit[ig],time_array=time_array,h_array=h_array)

    initial_guess=guess_fit[np.argmin(loss)]
    print("initial guess index: ",np.argmin(loss))
    print("min loss:",loss[np.argmin(loss)])

    params_fit=[]
    print("fitting for N_free = ",N_free)
    for i in tqdm(range(len(t0_arr))):
        # 创建对应的time_array和h_array
        index0=np.argmin(np.abs(t-t0_arr[i]))
        time_array=np.zeros([2,2*(len(t)-index0)])
        h_array=np.zeros(2*(len(t)-index0))
        for k in range(len(t)-index0):
            time_array[0,2*k]=0
            time_array[0,2*k+1]=1
            time_array[1,2*k]=t[index0+k]
            time_array[1,2*k+1]=time_array[1,2*k]
            h_array[2*k]=h_r[index0+k]
            h_array[2*k+1]=h_i[index0+k]

        n_try=N_pert_guess
        initial_try=np.zeros([n_try,2*N_fix+4*N_free])
        loss_try=np.zeros(n_try)
        try_fit=[]
        for i_try in range(n_try):
            for j in range(len(initial_guess)):
                if(i_try==0):
                    initial_try[i_try,j]=initial_guess[j]
                else:
                    initial_try[i_try,j]=initial_guess[j]*(1.0+scipy.stats.uniform.rvs(-1,2)*pert)
            try:
                popt, pcov = jcf.curve_fit(lambda t, *params: fit_extract_h(t,omega_fix,N_free, params), \
                time_array, h_array, p0=initial_try[i_try,:], max_nfev=max_nfev,method="trf")
                try_fit.append(popt)
            except RuntimeError as e:
                try_fit.append(initial_try[i_try,:])
                            
            loss_try[i_try]=fit_extract_loss(omega_fix,N_free, *try_fit[i_try],time_array=time_array,h_array=h_array)

        initial_guess=try_fit[np.argmin(loss_try)]
        params_fit.append(try_fit[np.argmin(loss_try)])

    return params_fit

# 参数设置尽可能和此前一致
# t: 数组，时间：由于不同波形的预处理可能不一样，我们这里直接要求输入处理好了的时间，
#    即从t从0开始到t_end
# h_r,h_i: 显然是波形的实部和虚部，用这种写法的话，输入h还是psi4其实是一样的，要与时间对应
# t0_arr: 这个也从外面给，是外面系列拟合的起始时间点
# N_free: 我们采用最传统的拟合方式，即考虑有N_free，N_fix mode，暂时不特殊处理一些理论
# N_fix: 如有必要可以固定一些模式的频率来进行提取，需要同时给定omega_fix
# omega_fix: 数组，[[omega_r, omega_i], [omega_r, omega_i], ...] 长度是N_fix
# 由于我们要使用线性拟合-非线性拟合混合的方式，我们拟合参数的自然选择将是
# 线性部分：Ck_r Ck_i 非线性部分：omega_r omega_i
# 虽然原则上线性部分我们可以手工求解出最佳拟合，但我们这里不妨交给jaxfit数值的来做
# 但怎么说呢，这个的问题是我们相当于无法使用jaxfit tracing的特性了,会有一定的降速
# 我们可能还是考虑原来的方案吧,使用微扰模式
# def fit_VarPro(t,h_r,h_i,t0_arr,N_free,N_fix,omega_fix):
#     return

bkj_220r=[[1.0,       0.537583,   -2.990402,  1.503421],
     [-1.899567, -2.128633,  6.626680,   -2.903790],
     [1.015454,  2.147094,   -4.672847,  1.891731],
     [-0.111430, -0.581706,  1.021061,   -0.414517]]

ckj_220r=[[1.0,          0.548651,   -3.141145,  1.636377],
     [-2.238461,    -2.291933,  7.695570,   -3.458474],
     [1.581677,     2.662938,   -6.256090,  2.494264],
     [-0.341455,    -0.930069,  1.688288,   -0.612643]]

bkj_220i=[[1.0,          -2.721789,     2.472860,      -0.750015],
          [-2.533958,    7.181110,      -6.870324,     2.214689],
          [2.102750,     -6.317887,     6.206452,      -1.980749],
          [-0.568636,    1.857404,      -1.820547,     0.554722]]

ckj_220i=[[1.0,          -2.732346,     2.495049,      -0.761581],
          [-2.498341,    7.089542,      -6.781334,     2.181880],
          [2.056918,     -6.149334,     6.010021,      -1.909275],
          [-0.557557,    1.786783,      -1.734461,     0.524997]]

# 330
bkj_330r=[[1.0,         -0.311963,      -1.457057,      0.825692],
          [-1.928277,   -0.026433,      3.139427,       -1.484557],
          [1.044039,    0.545708,       -2.188569,      0.940019],
          [-0.112303,   -0.226402,      0.482482,       -0.204299]]

ckj_330r=[[1.0,         -0.299153,      -1.591595,      0.938987],
          [-2.265230,   0.058508,       3.772084,       -1.852247],
          [1.624332,    0.533096,       -3.007197,      1.285026],
          [-0.357651,   -0.300599,      0.810387,       -0.314715]]

bkj_330i=[[1.0,         -2.813977,      2.666759,       -0.850618],
          [-2.163575,   6.934304,       -7.425335,      2.640936],
          [1.405496,    -5.678573,      6.621826,       -2.345713],
          [-0.241561,   1.555843,       -1.890365,      0.637480]]

ckj_330i=[[1.0,         -2.820763,      2.680557,       -0.857462],
          [-2.130446,   6.825101,       -7.291058,      2.583282],
          [1.394144,    -5.533669,      6.393699,       -2.254239],
          [-0.261229,   1.517744,       -1.810579,      0.608393]]

# 221
bkj_221r=[[1.0,         -2.918987,      2.866252,       -0.944554],
          [-1.850299,   7.321955,       -8.783456,      3.292966],
          [0.944088,    -5.584876,      7.675096,       -3.039132],
          [-0.088458,   1.198758,       -1.973222,      0.838109]]

ckj_221r=[[1.0,         -2.941138,      2.907859,       -0.964407],
          [-2.250169,   8.425183,       -9.852886,      3.660289],
          [1.611393,    -7.869432,      9.999751,       -3.737205],
          [-0.359285,   2.392321,       -3.154979,      1.129776]]

bkj_221i=[[1.0,         -3.074983,      3.182195,       -1.105297],
          [0.366066,    4.296285,       -9.700146,      5.016955],
          [-3.290350,   -0.844265,      9.999863,       -5.818349],
          [1.927196,    -0.401520,      -3.537667,      2.077991]]

ckj_221i=[[1.0,         -3.079686,      3.191889,       -1.110140],
          [0.388928,    4.159242,       -9.474149,      4.904881],
          [-3.119527,   -0.914668,      9.767356,       -5.690517],
          [1.746957,    -0.240680,      -3.505359,     2.049254]]

def sta_cal_A(C_r_list,C_i_list):
    A=(C_r_list**2+C_i_list**2)**(1/2)
    A_ave=np.average(A)
    C_r_std=np.std(C_r_list)
    C_i_std=np.std(C_i_list)
    return (C_r_std**2+C_i_std**2)**(1/2)/np.abs(A_ave)

def sta_cal_ful(C_r_list,C_i_list,omega_r_list,omega_i_list):
    A=(C_r_list**2+C_i_list**2)**(1/2)
    A_ave=np.average(A)   
    omega=(omega_r_list**2+omega_i_list**2)**(1/2)
    omega_ave=np.average(omega)

    C_r_std=np.std(C_r_list)
    C_i_std=np.std(C_i_list)
    omega_r_std=np.std(omega_r_list)
    omega_i_std=np.std(omega_i_list)
    return ((C_r_std**2+C_i_std**2)/A_ave**2+(omega_r_std**2+omega_i_std**2)/omega_ave**2)**(1/2)
