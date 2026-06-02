"""
后处理模块：输出位移、反力、应力、轴力
"""
import numpy as np

def compute_stress_force(model, e):
    """计算单元应力和轴力"""
    ndof = model['ndof']
    Ke = model['Ke_list'][e]
    info = model['elem_info'][e]
    d_e = model['disp'][model['LM'][:, e]]
    
    if model['nsd'] == 1:
        L = info['L']
        E = model['E'][e]
        stress = (E / L) * np.array([-1, 1]) @ d_e
    else:
        L = info['L']
        c, s = info['c'], info['s']
        E = model['E'][e]
        stress = (E / L) * np.array([-c, -s, c, s]) @ d_e
    
    force = stress * model['A'][e]
    return stress, force

def print_results(model):
    print("="*60)
    print(f"【{model['title']}】有限元计算结果")
    print("="*60)
    
    print("\n1. 总体刚度矩阵 K：")
    print(np.round(model['K'], 4))
    
    print("\n2. 节点位移：")
    for i in range(model['total_dof']):
        print(f"自由度 {i+1:2d} : {model['disp'][i]:10.6f}")
    
    print("\n3. 约束反力：")
    for i in model['fixed_dof']:
        print(f"自由度 {i+1:2d} : {model['reaction'][i]:10.6f}")
    
    print("\n4. 单元结果：")
    for e in range(model['nel']):
        stress, force = compute_stress_force(model, e)
        info = model['elem_info'][e]
        print(f"单元 {e+1}: 长度={info['L']:.4f}, 应力={stress:.6f}, 轴力={force:.6f}")

def check_stiffness_properties(model):
    K = model['K']
    print("\n" + "="*60)
    print("刚度矩阵性质校验")
    print("="*60)
    print(f"对称: {np.allclose(K, K.T)}")
    print(f"奇异: {np.isclose(np.linalg.det(K), 0)}")
    print(f"对角元非负: {np.all(np.diag(K) >= -1e-10)}")