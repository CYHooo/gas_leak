from django.conf import settings
from django.shortcuts import render
import numpy as np
from django.http import JsonResponse
import json
import os
from pyowm.owm import OWM
import requests
import math
from scipy.ndimage import zoom
# Create your views here.

# def map_detail(request):
#     return render(request, 'map/index.html')

# myapp/views.py
def test(request):
    data_file = os.path.join(settings.BASE_DIR,'map/static/map/data/volcano.json')
    with open(data_file, 'r') as f:
        volcano_data = json.load(f)
    # 将数据传递给模板
    context = {
        'volcano_data': json.dumps(volcano_data)
    }
    return render(request, 'map/test.html', context)



###################################################################################


def get_wind_data(lat=35.488079, lng=129.361921, api_key='040a7c49f2ebab8cc1ac0e74f3807087'):
    apikey = api_key
    owm = OWM(apikey)
    mgr = owm.weather_manager()
    wind = mgr.weather_at_coords(lat,lng).weather.wind()
    
    speed = wind['speed']
    deg = wind['deg']
    return speed,deg

def wind_direction_to_uv(wind_speed, wind_deg):
    # 将风向转换为弧度，并计算 x 和 y 方向的风速分量
    theta = math.radians(wind_deg)
    u = wind_speed * math.sin(theta)  # x 方向风速
    v = wind_speed * math.cos(theta)  # y 方向风速
    return u, v


def get_dispersion_coefficients(x_downwind, stability_class):
    # 定义 Pasquill-Gifford 稳定度分类的系数
    stability_classes = {
        'A': {'a_y': 0.22, 'b_y': 0.5, 'a_z': 0.20, 'b_z': 1.0},
        'B': {'a_y': 0.16, 'b_y': 0.5, 'a_z': 0.12, 'b_z': 1.0},
        'C': {'a_y': 0.11, 'b_y': 0.5, 'a_z': 0.08, 'b_z': 1.0},
        'D': {'a_y': 0.08, 'b_y': 0.5, 'a_z': 0.06, 'b_z': 1.0},
        'E': {'a_y': 0.06, 'b_y': 0.5, 'a_z': 0.03, 'b_z': 1.0},
        'F': {'a_y': 0.04, 'b_y': 0.5, 'a_z': 0.016, 'b_z': 1.0},
    }
    coeffs = stability_classes[stability_class]

    # 避免对零或负值取对数
    x_downwind = np.maximum(x_downwind, 1)

    sigma_y = coeffs['a_y'] * x_downwind ** coeffs['b_y']
    sigma_z = coeffs['a_z'] * x_downwind ** coeffs['b_z']
    return sigma_y, sigma_z


def gaussian_puff(x, y, Q, u, v, t, H=0, sigmax=None, sigmay=None, sigmaz=None):
    """
    gaussian puff model
    
    参数:
    x, y: xy coor position on grid (m)
    Q: gas leak speed (kg/s)
    u: wind speed on x axis (m/s)
    v: wind speed on y axis (m/s)
    t: time (s)
    H: gas leak height (m)
    sigmax, sigmay, sigmaz: Diffusion coefficient (m)
    
    返回:
    c: Concentration Matrix
    """
    # sigma xyz need search table's value
    if sigmax is None:
        sigmax = 0.08 * (x + 0.0001) * (1 + 0.0001 * x)**(-0.5)
    if sigmay is None:
        sigmay = 0.08 * (x + 0.0001) * (1 + 0.0001 * x)**(-0.5)
    if sigmaz is None:
        sigmaz = 0.06 * (x + 0.0001) * (1 + 0.0015 * x)**(-0.5)
    
    # 计算浓度分布
    c = (Q / (2 * np.pi * sigmay * sigmaz * np.sqrt(u**2 + v**2))) * \
        np.exp(-0.5 * (y**2 / sigmay**2)) * \
        (np.exp(-0.5 * ((H)**2 / sigmaz**2)) + \
         np.exp(-0.5 * ((2 * H)**2 / sigmaz**2)))
    # c = (Q / ((8 * np.pi * t)**1.5 * np.sqrt(sigmax*sigmay*sigmaz))) * \
    #     np.exp(-((x - u * t)**2 / (4 * sigmax * t) + y**2 / (4 * sigmay * t) + H**2 / (4 * sigmaz * t)))
    
    return c

def calculate_concentration_field(grid_size, Q, u, v, t, H=0):
    """
    Args:
    grid_size: (int), make grid_size x grid_size concentration metrix
    Q: leak speed (g/s)
    u: X wind speed (m/s)
    v: Y wind speed (m/s)
    t: time (s)
    H: height (m)

    return:
    C: metrix (2D numpy array)
    """
    # 定义坐标范围
    x = np.linspace(0, 300, grid_size)
    y = np.linspace(-150, 150, grid_size)
    X, Y = np.meshgrid(x, y)

    # 计算下风向距离
    x_downwind = X - (u * t)
    y_crosswind = Y - (v * t)

    # 计算扩散参数
    sigma_x = 0.08 * x_downwind * (1 + 0.0001 * x_downwind) ** (-0.5)
    sigma_y = sigma_x  # 假设 sigma_x = sigma_y
    sigma_z = 0.06 * x_downwind * (1 + 0.0015 * x_downwind) ** (-0.5)

    # 避免 sigma 为负或为零
    sigma_x = np.maximum(sigma_x, 1e-3)
    sigma_y = np.maximum(sigma_y, 1e-3)
    sigma_z = np.maximum(sigma_z, 1e-3)

    # 计算浓度场
    C = (Q / (2 * np.pi * sigma_y * sigma_z)) * \
        np.exp(-0.5 * (y_crosswind ** 2) / (sigma_y ** 2)) * \
        (np.exp(-0.5 * ((H) ** 2) / (sigma_z ** 2)) + \
         np.exp(-0.5 * ((H + 2 * H) ** 2) / (sigma_z ** 2)))

    # 将非物理的负值设为零
    C = np.maximum(C, 0)

    return C



def extract_and_resize_concentration(C, center_x, center_y, target_size=128):
    """
    resize concentration metrix
    
    参数:
    C: origin metrix (2D numpy array)
    center_x, center_y: init xy (int)
    u: x wind speed (m/s)
    v: y wind speed (m/s)
    t: time (s)
    target_size:  (int), default 128
    
    返回:
    result: D3 format json data
    """

    half_size = target_size // 2
    x_min = center_x - half_size
    x_max = center_x + half_size
    y_min = center_y - half_size
    y_max = center_y + half_size

    # 初始化一个全零矩阵，大小为 target_size x target_size
    C_extracted = np.zeros((target_size, target_size))

    # 计算在原始矩阵 C 中有效的索引范围
    x_min_valid = max(x_min, 0)
    x_max_valid = min(x_max, C.shape[1])
    y_min_valid = max(y_min, 0)
    y_max_valid = min(y_max, C.shape[0])

    # 计算在目标矩阵中对应的索引范围
    x_start = x_min_valid - x_min
    x_end = x_start + (x_max_valid - x_min_valid)
    y_start = y_min_valid - y_min
    y_end = y_start + (y_max_valid - y_min_valid)

    # 将原始矩阵中的有效区域复制到目标矩阵中
    C_extracted[y_start:y_end, x_start:x_end] = C[y_min_valid:y_max_valid, x_min_valid:x_max_valid]

    # 如果需要调整大小，可以取消下面的注释
    # 这里由于我们已经确保了提取的区域大小为 target_size x target_size，所以不需要调整大小
    # 但如果有需要，可以使用以下代码进行插值调整
    # C_extracted = zoom(C_extracted, (target_size / C_extracted.shape[0], target_size / C_extracted.shape[1]), order=3)

    # 将浓度值归一化到所需范围（例如 0 到 200）
    min_val, max_val = np.min(C_extracted), np.max(C_extracted)
    if max_val > min_val:
        C_normalized = np.interp(C_extracted, (min_val, max_val), (0, 200))
    else:
        C_normalized = np.zeros_like(C_extracted)

    # 将数据转换为整数
    C_normalized = np.round(C_normalized).astype(int)

    # 格式化为符合 D3.js 的数据格式
    result = {
        "width": target_size,
        "height": target_size,
        "value": C_normalized.flatten().tolist()
    }
    
    return result


def contour(request):
    w_speed, w_deg = get_wind_data()
    u, v = wind_direction_to_uv(w_speed, w_deg)
    sigma_y, sigma_z = get_dispersion_coefficients(u, 'D')
    grid_size = 256
    # C = gaussian_puff(x=0, y=0, Q=100,t=10, H=0, sigmax=sigma_y, sigmay=sigma_y)
    C = calculate_concentration_field(grid_size=grid_size, Q=1000, t=10, u=u, v=v, H=0)
    center_x = grid_size // 2
    center_y = grid_size // 2

    result = extract_and_resize_concentration(C, center_x=center_x, center_y=center_y, target_size=128)

    # with open('test.json','w') as f:
    #     json.dump(result,f)
    context = {
        'data': json.dumps(result)
    }

    return render(request, 'map/index.html', context)
