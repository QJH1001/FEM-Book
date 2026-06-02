"""
单元分析模块：计算单元刚度矩阵 Ke
"""
import numpy as np

def compute_element_stiffness_1d(model, e):
    """一维杆单元刚度"""
    node1, node2 = model['IEN'][e]
    L = abs(model['coord'][node2,0] - model['coord'][node1,0])
    E = model['E'][e]
    A = model['A'][e]
    k = E * A / L
    Ke = np.array([[k, -k], [-k, k]])
    return Ke, L

def compute_element_stiffness_2d(model, e):
    """二维桁架单元刚度"""
    node1, node2 = model['IEN'][e]
    x1, y1 = model['coord'][node1]
    x2, y2 = model['coord'][node2]
    
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    c = dx / L
    s = dy / L
    
    E = model['E'][e]
    A = model['A'][e]
    k = E * A / L
    
    # 2D桁架单元刚度矩阵
    T = np.array([[c, s, 0, 0],
                  [0, 0, c, s]])
    Ke_local = np.array([[1, -1], [-1, 1]])
    Ke = k * T.T @ Ke_local @ T
    return Ke, L, c, s

def compute_all_element_stiffness(model):
    """计算所有单元刚度并保存"""
    nel = model['nel']
    model['Ke_list'] = []
    model['elem_info'] = []
    
    for e in range(nel):
        if model['nsd'] == 1:
            Ke, L = compute_element_stiffness_1d(model, e)
            model['elem_info'].append({'L': L})
        else:
            Ke, L, c, s = compute_element_stiffness_2d(model, e)
            model['elem_info'].append({'L': L, 'c': c, 's': s})
        model['Ke_list'].append(Ke)