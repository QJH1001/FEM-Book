"""
求解模块：缩减法处理边界，求解位移与反力
"""
import numpy as np

def solve_displacement(model):
    K = model['K'].copy()
    F = model['force'].copy()
    fixed_dof = model['fixed_dof']
    fixed_val = model['fixed_val']
    total_dof = model['total_dof']
    
    # 自由自由度 & 约束自由度
    all_dof = np.arange(total_dof)
    free_dof = np.setdiff1d(all_dof, fixed_dof)
    
    # 分块矩阵
    Kff = K[np.ix_(free_dof, free_dof)]
    Kfc = K[np.ix_(free_dof, fixed_dof)]
    Ff = F[free_dof]
    
    # 求解未知位移
    d = np.zeros(total_dof)
    d[fixed_dof] = fixed_val
    d[free_dof] = np.linalg.solve(Kff, Ff - Kfc @ fixed_val)
    
    # 计算约束反力
    reaction = K @ d - F
    reaction[free_dof] = 0
    
    # 保存结果
    model['disp'] = d
    model['reaction'] = reaction