 import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sko.PSO import PSO
from sklearn.gaussian_process.kernels import RBF
import scipy.stats as st
import seaborn as sns
from sklearn.utils import resample
import pandas as pd
import time
import warnings
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中
plt.rcParams['axes.unicode_minus'] = False
start = time.time()




## ---------------------构建target model loss_func函数---------------------------------
def target_func(x):
    sum = 0
    for i in range(len(x)):
        sum += (0.5 * x[i] ** 2 - 10 * np.cos(0.5 * np.pi * x[i]) + 10)
    return sum


def target_loss(x):
    return -target_func(x)


# ---------------------构建source model loss_func函数---------------------------------
# s11、13、23、31和target比较相似
# source_1为改变(x - a)
def source_func_11(x, a=2):
    sum = 0
    for i in range(len(x)):
        sum += (0.5 * (x[i] - a) ** 2 - 10 * np.cos(0.5 * np.pi * x[i]) + 10)
    return sum


def source_loss_11(x):
    return -source_func_11(x)


def source_func_12(x, a=4):
    sum = 0
    for i in range(len(x)):
        sum += (0.5 * (x[i] - a) ** 2 - 10 * np.cos(0.5 * np.pi * x[i]) + 10)
    return sum


def source_loss_12(x):
    return -source_func_12(x)


def source_func_13(x, a=-2):
    sum = 0
    for i in range(len(x)):
        sum += (0.5 * (x[i] - a) ** 2 - 10 * np.cos(0.5 * np.pi * x[i]) + 10)
    return sum


def source_loss_13(x):
    return -source_func_13(x)


# source_2为改变b * (x)
def source_func_21(x, b=-1):
    sum = 0
    for i in range(len(x)):
        sum += (b * (x[i] + 2) ** 2 - 10 * np.cos(0.5 * np.pi * x[i]) + 10)
    return sum


def source_loss_21(x):
    return -source_func_21(x)


def source_func_22(x, b=-0.1):
    sum = 0
    for i in range(len(x)):
        sum += (b * (x[i] - 3) ** 2 - 10 * np.cos(0.5 * np.pi * x[i]) + 10)
    return sum


def source_loss_22(x):
    return -source_func_22(x)


def source_func_23(x, b=0.5):
    sum = 0
    for i in range(len(x)):
        sum += (b * (x[i] - 1) ** 2 - 10 * np.cos(0.5 * np.pi * x[i]) + 10)
    return sum


def source_loss_23(x):
    return -source_func_23(x)


# source_3为改变cos部分
def source_func_31(x, c=0.5, d=11):
    sum = 0
    for i in range(len(x)):
        sum += (0.5 * (x[i]) ** 2 - d * np.cos(0.5 * np.pi * (x[i] - c)) + 10)
    return sum


def source_loss_31(x):
    return -source_func_31(x)


def source_func_32(x, c=1, d=20):
    sum = 0
    for i in range(len(x)):
        sum += (0.1 * (x[i]) ** 2 - d * np.cos(0.5 * np.pi * (x[i] - c)) + 10)
    return sum


def source_loss_32(x):
    return -source_func_32(x)


def source_func_33(x, c=-3, d=-20):
    sum = 0
    for i in range(len(x)):
        sum += (1 * (x[i]) ** 2 - d * np.cos(0.5 * np.pi * (x[i] - c)) + 10)
    return sum


def source_loss_33(x):
    return -source_func_33(x)



## ------------------------------2.构建高斯过程----------------------
kernel = RBF(length_scale=3, length_scale_bounds=(1e-4, 2e5))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                              n_restarts_optimizer=10, random_state=1, )
gp_t = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                n_restarts_optimizer=10, random_state=1, )
gp_s11 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_s12 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_s13 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_s21 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_s22 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_s23 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_s31 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_s32 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_s33 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                  n_restarts_optimizer=10, random_state=1, )
gp_parallelBO = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', normalize_y=True,
                                         n_restarts_optimizer=10, random_state=1, )


## ---------------------------3. 采集函数--------------------------
def acq_EI_PSO(x):  # xi为超参数
    global y_max_parallelBO
    xi = 0.1
    x = x.reshape(-1, x_dim)
    mean_t, std_t = gp_parallelBO.predict(x, return_std=True)
    a = (mean_t - y_max_parallelBO - xi)
    z = a / std_t
    utility_t = (a * st.norm.cdf(z) + std_t * st.norm.pdf(z))
    return -utility_t


def acq_EI_PSO_EI(x):  # xi为超参数
    global y_max_t
    global y_max_s11
    global y_max_s12
    global y_max_s13
    global y_max_s21
    global y_max_s22
    global y_max_s23
    global y_max_s31
    global y_max_s32
    global y_max_s33
    global W

    xi = 0.1
    x = x.reshape(-1, x_dim)
    # target的u

    mean_t, std_t = gp_t.predict(x, return_std=True)
    a = (mean_t - y_max_t - xi)
    z = a / std_t
    utility_t = (a * st.norm.cdf(z) + std_t * st.norm.pdf(z))

    # source_s11的u
    mean, std = gp_s11.predict(x, return_std=True)
    a = (mean - y_max_s11 - xi)
    z = a / std
    utility_s11 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    # source_s12的u
    mean, std = gp_s12.predict(x, return_std=True)
    a = (mean - y_max_s12 - xi)
    z = a / std
    utility_s12 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    # source_s13的u

    mean, std = gp_s13.predict(x, return_std=True)
    a = (mean - y_max_s13 - xi)
    z = a / std
    utility_s13 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    # source_s21的u
    mean, std = gp_s21.predict(x, return_std=True)
    a = (mean - y_max_s21 - xi)
    z = a / std
    utility_s21 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    # source_s22的u

    mean, std = gp_s22.predict(x, return_std=True)
    a = (mean - y_max_s22 - xi)
    z = a / std
    utility_s22 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    # source_s23的u

    mean, std = gp_s23.predict(x, return_std=True)
    a = (mean - y_max_s23 - xi)
    z = a / std
    utility_s23 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    # source_s31的u
    mean, std = gp_s31.predict(x, return_std=True)
    a = (mean - y_max_s31 - xi)
    z = a / std
    utility_s31 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    # source_s32的u
    mean, std = gp_s32.predict(x, return_std=True)
    a = (mean - y_max_s32 - xi)
    z = a / std
    utility_s32 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    # source_s33的u

    mean, std = gp_s33.predict(x, return_std=True)
    a = (mean - y_max_s33 - xi)
    z = a / std
    utility_s33 = (a * st.norm.cdf(z) + std * st.norm.pdf(z))

    utility = (utility_t * W[0] + utility_s11 * W[1] + utility_s12 * W[2] + utility_s13 * W[3] + utility_s21 * W[
        4] + utility_s22 * W[
                   5] + utility_s23 * W[6] + utility_s31 * W[7] + utility_s32 * W[8] + utility_s33 * W[9]) / sum(W)

    return -utility


## ---------------------4.权重设计--------------------

# 计算source model损失L：损失定义为排名错误的对的数量
def L_S(xsamples_t, loss_ysamples_t, gp):
    L = 0
    mean, std = gp.predict(xsamples_t.reshape(-1, x_dim), return_std=True)
    for i in range(len(xsamples_t)):
        for j in range(len(xsamples_t)):
            if (mean[i] < mean[j]) ^ (loss_ysamples_t[i] < loss_ysamples_t[j]):  # 用均值代替
                L += 1
    return L


# 计算target model损失L：损失定义为排名错误的对的数量
# 留一交叉验证法
def L_D(xsamples_t, loss_ysamples_t):
    L = 0
    for i in range(len(xsamples_t)):
        xsamples = np.delete(xsamples_t, [i], axis=0)
        ysamples = np.delete(loss_ysamples_t, [i], axis=0)
        gp.fit(xsamples, ysamples)
        mean_i, std_i = gp.predict(xsamples_t[i].reshape(-1, x_dim), return_std=True)
        for j in range(len(xsamples_t)):
            if (mean_i < loss_ysamples_t[j]) ^ (loss_ysamples_t[i] < loss_ysamples_t[j]):  # 用均值代替
                L += 1
    return L

### -------------------------贝叶斯优化------------------------------
num_test = 10  # 实验次数
max_iter = 50  # 最大采样点数
x_dim =15  # 输入参数的维度
num_parallelBO = 5  # 输入并行BO推荐点数
sum_test = []
xsamples_num_t = 6  # target任务的初始数据数
xsamples_num_s = 500  # source任务的初始数据
max_loss = -0.01  # target_loss的精度要求,通用性方法：所测最大结果的1%，迭代结束
bootstrap_s = 50  # bootstrapSamples循环采样次数
max_ysamples = np.zeros(shape=(max_iter + xsamples_num_t, num_test))
sum_loss = 0
xsamples_t_rand=np.array()   # 随机一些数据

for m in range(num_test):
    # 初始化相关参数
    m_iter = 0  # 记录采样点数
    W_iter = np.empty(shape=(0, 10))
    W = []  # 各个模型权重

    xsamples_t = xsamples_t_rand[m]  # target任务的初始数据点x
    loss_ysamples_t = np.empty(shape=(0, 1))  # target任务的初始数据点loss

    xsamples_init_s = np.empty(shape=(0, x_dim))  # source任务的初始数据点x
    loss_ysamples_s11 = np.empty(shape=(0, 1))  # source_s任务的初始数据点loss
    loss_ysamples_s12 = np.empty(shape=(0, 1))
    loss_ysamples_s13 = np.empty(shape=(0, 1))
    loss_ysamples_s21 = np.empty(shape=(0, 1))
    loss_ysamples_s22 = np.empty(shape=(0, 1))
    loss_ysamples_s23 = np.empty(shape=(0, 1))
    loss_ysamples_s31 = np.empty(shape=(0, 1))
    loss_ysamples_s32 = np.empty(shape=(0, 1))
    loss_ysamples_s33 = np.empty(shape=(0, 1))



    for i in range(xsamples_num_t):
        loss = target_loss(xsamples_t[i])
        loss_ysamples_t = np.append(loss_ysamples_t, np.array([[loss]]), axis=0)
        max_ysamples[(m_iter + i), m] = max(loss_ysamples_t.flatten())
    # print(xsamples_init)
    # print(loss_ysamples)

    # 计算初始的source 数据
    x_rand = np.empty(shape=(xsamples_num_s, x_dim))
    for i in range(xsamples_num_s):
        for j in range(x_dim):
            x_rand[i, j] = np.random.rand() * 20 - 10
    xsamples_init_s = np.vstack((xsamples_init_s, x_rand))


    for i in range(xsamples_num_s):
        loss_ysamples_s11 = np.append(loss_ysamples_s11, np.array([[source_loss_11(xsamples_init_s[i])]]), axis=0)
        loss_ysamples_s12 = np.append(loss_ysamples_s12, np.array([[source_loss_12(xsamples_init_s[i])]]), axis=0)
        loss_ysamples_s13 = np.append(loss_ysamples_s13, np.array([[source_loss_13(xsamples_init_s[i])]]), axis=0)
        loss_ysamples_s21 = np.append(loss_ysamples_s21, np.array([[source_loss_21(xsamples_init_s[i])]]), axis=0)
        loss_ysamples_s22 = np.append(loss_ysamples_s22, np.array([[source_loss_22(xsamples_init_s[i])]]), axis=0)
        loss_ysamples_s23 = np.append(loss_ysamples_s23, np.array([[source_loss_23(xsamples_init_s[i])]]), axis=0)
        loss_ysamples_s31 = np.append(loss_ysamples_s31, np.array([[source_loss_31(xsamples_init_s[i])]]), axis=0)
        loss_ysamples_s32 = np.append(loss_ysamples_s32, np.array([[source_loss_32(xsamples_init_s[i])]]), axis=0)
        loss_ysamples_s33 = np.append(loss_ysamples_s33, np.array([[source_loss_33(xsamples_init_s[i])]]), axis=0)

    gp_s11.fit(xsamples_init_s, loss_ysamples_s11)
    gp_s12.fit(xsamples_init_s, loss_ysamples_s12)
    gp_s13.fit(xsamples_init_s, loss_ysamples_s13)
    gp_s21.fit(xsamples_init_s, loss_ysamples_s21)
    gp_s22.fit(xsamples_init_s, loss_ysamples_s22)
    gp_s23.fit(xsamples_init_s, loss_ysamples_s23)
    gp_s31.fit(xsamples_init_s, loss_ysamples_s31)
    gp_s32.fit(xsamples_init_s, loss_ysamples_s32)
    gp_s33.fit(xsamples_init_s, loss_ysamples_s33)

    y_max_s11 = max(loss_ysamples_s11.flatten())
    y_max_s12 = max(loss_ysamples_s12.flatten())
    y_max_s13 = max(loss_ysamples_s13.flatten())
    y_max_s21 = max(loss_ysamples_s21.flatten())
    y_max_s22 = max(loss_ysamples_s22.flatten())
    y_max_s23 = max(loss_ysamples_s23.flatten())
    y_max_s31 = max(loss_ysamples_s31.flatten())
    y_max_s32 = max(loss_ysamples_s32.flatten())
    y_max_s33 = max(loss_ysamples_s33.flatten())
    while max(loss_ysamples_t.flatten()) < max_loss:
        m_iter += 1  # 记录采点个数
        gp_t.fit(xsamples_t, loss_ysamples_t)
        y_max_t = max(loss_ysamples_t.flatten())

        xsamples_parallel = np.empty(shape=(0, x_dim))  # 并行BO点的初始化
        rec_parallelBO = np.empty(shape=(0, x_dim + 1))  # 记录推荐点的位置和推荐值
        ## 选择paralled并行的推荐点
        W_sum=[0,0,0,0,0,0,0,0,0,0]# 记录总的权重
        for n in range(num_parallelBO):
            xy_parallelBO = np.hstack((xsamples_t, loss_ysamples_t))  # 将x和y合并成一个数组
            if len(xy_parallelBO) < 10:
                n_samples = 2
            if len(xy_parallelBO) >= 10:
                n_samples = len(xy_parallelBO) // 5
            for i in range(n_samples):  # 去除部分target数据
                xy_parallelBO = np.delete(xy_parallelBO, np.random.choice(range(len(xy_parallelBO)), size=1), axis=0)

            gp_parallelBO.fit(xy_parallelBO[:, 0:x_dim], xy_parallelBO[:, -1].reshape(-1, 1))  # 建立并行的GP
            y_max_parallelBO = max(xy_parallelBO[:, -1].reshape(-1, 1).flatten())
            # 计算每个采样的推荐点
            lb = []
            ub = []
            for i in range(x_dim):
                lb.append(-10)
                ub.append(10)
            pso = PSO(func=acq_EI_PSO, dim=x_dim, pop=100, max_iter=40, lb=lb, ub=ub, w=0.4, c1=0.5, c2=0.5)
            pso.run()


            ## 对paralled并行的推荐点邻域进行相似性分析，确定新推荐点，并保留推荐值
            x_dist = np.zeros(shape=(len(xsamples_t), 1))
            for j in range(len(xsamples_t)):  # 计算每个推荐点邻域包含的target数据
                for k in range(x_dim):
                    x_dist[j, 0] += (pso.gbest_x.reshape(1, -1)[0, k] - xsamples_t[j, k]) * (
                                pso.gbest_x.reshape(1, -1)[0, k] - xsamples_t[j, k])


            xy_samples_t = np.hstack((xsamples_t, loss_ysamples_t))
            xy_samples_t_dist = np.hstack((xy_samples_t, x_dist))

            xy_samples_t_dist = xy_samples_t_dist[xy_samples_t_dist[:, -1].argsort()]  # 按照第最后1列进行从小到大排序数据
            xy_samples_t_parallelBO = xy_samples_t_dist[0:int((len(xy_samples_t_dist) // 1.25)),
                                      0:-1]  # 取距离推荐点较近的0.8倍数量的target数据

            ## 计算各模型权重
            I_t = 0
            I_s11 = 0
            I_s12 = 0
            I_s13 = 0
            I_s21 = 0
            I_s22 = 0
            I_s23 = 0
            I_s31 = 0
            I_s32 = 0
            I_s33 = 0
            # 采样bootstrapSamples采样确定
            for s in range(bootstrap_s):  # 确定去除几个样本，等于不放回采样
                if len(xy_samples_t_parallelBO) < 10:
                    n_samples = 2
                if len(xy_samples_t_parallelBO) >= 10:
                    n_samples = int(len(xy_samples_t_parallelBO) // 8)
                bootstrapSamples = xy_samples_t_parallelBO
                for i in range(n_samples):  # 不放回采样数据
                    bootstrapSamples = np.delete(bootstrapSamples,
                                                 np.random.choice(range(len(bootstrapSamples)), size=1),
                                                 axis=0)
                bootstrapSamples_x = bootstrapSamples[:, 0:-1]
                bootstrapSamples_y = bootstrapSamples[:, -1]

                # 计算target的损失
                Loss_t = L_D(bootstrapSamples_x, bootstrapSamples_y)
                # 计算source的损失
                Loss_s11 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s11)
                Loss_s12 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s12)
                Loss_s13 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s13)
                Loss_s21 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s21)
                Loss_s22 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s22)
                Loss_s23 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s23)
                Loss_s31 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s31)
                Loss_s32 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s32)
                Loss_s33 = L_S(bootstrapSamples_x, bootstrapSamples_y, gp_s33)

                Loss = [Loss_t, Loss_s11, Loss_s12, Loss_s13, Loss_s21, Loss_s22, Loss_s23, Loss_s31, Loss_s32,
                        Loss_s33]
                if Loss[0] == min(Loss):  # 此处没有考虑并列第一的情况
                    I_t += 1
                if Loss[1] == min(Loss):
                    I_s11 += 1
                if Loss[2] == min(Loss):
                    I_s12 += 1
                if Loss[3] == min(Loss):
                    I_s13 += 1
                if Loss[4] == min(Loss):
                    I_s21 += 1
                if Loss[5] == min(Loss):
                    I_s22 += 1
                if Loss[6] == min(Loss):
                    I_s23 += 1
                if Loss[7] == min(Loss):
                    I_s31 += 1
                if Loss[8] == min(Loss):
                    I_s32 += 1
                if Loss[9] == min(Loss):
                    I_s33 += 1
            # 各模型权重
            I_sum = I_t + I_s11 + I_s12 + I_s13 + I_s21 + I_s22 + I_s23 + I_s31 + I_s32 + I_s33
            w_t = I_t / I_sum
            w_s11 = I_s11 / I_sum
            w_s12 = I_s12 / I_sum
            w_s13 = I_s13 / I_sum
            w_s21 = I_s21 / I_sum
            w_s22 = I_s22 / I_sum
            w_s23 = I_s23 / I_sum
            w_s31 = I_s31 / I_sum
            w_s32 = I_s32 / I_sum
            w_s33 = I_s33 / I_sum

            W = [w_t, w_s11, w_s12, w_s13, w_s21, w_s22, w_s23, w_s31, w_s32, w_s33]
            for i in range(len(W)):
                W_sum[i] +=W[i]
            ## 计算下一采样点
            lb = []
            ub = []
            for j in range(x_dim):  # 确定pso采样的x范围
                lb.append(xy_samples_t_parallelBO[:, j].min() - 1)
                ub.append(xy_samples_t_parallelBO[:, j].max() + 1)


            pso = PSO(func=acq_EI_PSO_EI, dim=x_dim, pop=100, max_iter=40, lb=lb, ub=ub, w=0.4, c1=0.5, c2=0.5)

            pso.run()
            pso_best = np.hstack((pso.gbest_x.reshape(1, -1), pso.gbest_y.reshape(1, -1)))

            rec_parallelBO = np.append(rec_parallelBO, pso_best, axis=0)
        for i in range(len(W_sum)):
            W_sum[i] =W_sum[i]/sum(W_sum)
        W_iter = np.append(W_iter, np.array([W_sum]), axis=0)  # 统计每次迭代的权重
        rec_parallelBO = rec_parallelBO[rec_parallelBO[:, -1].argsort()]
        xsamples_new = rec_parallelBO[0, 0:-1]

        xsamples_t = np.append(xsamples_t, xsamples_new.reshape(1, -1), axis=0)
        loss_y = target_loss(xsamples_new)
        loss_ysamples_t = np.append(loss_ysamples_t, np.array([[loss_y]]), axis=0)

        max_ysamples[m_iter + xsamples_num_t - 1, m] = max(loss_ysamples_t.flatten())

        if m_iter >= max_iter:
            break
            # plt.figure()
            # plt.show()
    sum_loss += -max(loss_ysamples_t.flatten())
    print(f'超参数值：{xsamples_t[np.argmax(loss_ysamples_t)]}')
    print(f'max y is {max(loss_ysamples_t.flatten())}')
    sum_test.append(m_iter)

ave_loss = sum_loss / num_test
print(f'ave_loss is {ave_loss}')
print(f'ave_iter is {sum(sum_test) / num_test}')

end = time.time()
print('Running time: %s Seconds' % (end - start))

