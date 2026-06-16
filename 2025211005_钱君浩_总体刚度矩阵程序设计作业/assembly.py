"""
组装模块：直接组装总体刚度矩阵 K
"""
import numpy as np

def assemble_global_stiffness(model):
    """组装全局刚度矩阵"""
    total_dof = model['total_dof']
    K = np.zeros((total_dof, total_dof))
    nel = model['nel']
    LM = model['LM']
    Ke_list = model['Ke_list']
    
    for e in range(nel):
        Ke = Ke_list[e]
        dofs = LM[:, e]
        # 直接组装：累加
        for a in range(len(dofs)):
            for b in range(len(dofs)):
                i = dofs[a]
                j = dofs[b]
                K[i, j] += Ke[a, b]
    
    model['K'] = K