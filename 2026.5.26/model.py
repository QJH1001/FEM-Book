"""
前处理模块：读取JSON模型，初始化自由度
"""
import json
import numpy as np

def read_json(json_file):
    """读取JSON输入文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model = {}
    model['title'] = data['Title']
    model['nsd'] = data['nsd']      # 空间维数
    model['ndof'] = data['ndof']    # 单节点自由度
    model['nnp'] = data['nnp']      # 节点总数
    model['nel'] = data['nel']      # 单元总数
    model['nen'] = data['nen']      # 单单元节点数
    model['E'] = np.array(data['E'], dtype=float)
    model['A'] = np.array(data['CArea'], dtype=float)
    
    # 节点坐标
    if model['nsd'] == 1:
        model['coord'] = np.array(data['x'], dtype=float).reshape(-1,1)
    else:
        x = np.array(data['x'], dtype=float)
        y = np.array(data['y'], dtype=float)
        model['coord'] = np.column_stack([x, y])
    
    # 单元连接矩阵IEN（节点号转0索引）
    model['IEN'] = np.array([[i-1 for i in elem] for elem in data['IEN']], dtype=int)
    
    # 边界条件（自由度转0索引）
    model['fixed_dof'] = [d-1 for d in data['fixed_dof']]
    model['fixed_val'] = np.array(data['fixed_value'], dtype=float)
    
    # 载荷
    model['force_dof'] = [d-1 for d in data['force_dof']]
    model['force_val'] = np.array(data['force_value'], dtype=float)
    
    return model

def init_dof(model):
    """初始化总自由度数、LM对号矩阵、载荷向量"""
    ndof_node = model['ndof']
    nnp = model['nnp']
    nel = model['nel']
    nen = model['nen']
    
    # 总自由度数
    model['total_dof'] = ndof_node * nnp
    
    # 全局自由度编号
    model['dof_table'] = np.arange(model['total_dof'], dtype=int).reshape(nnp, ndof_node)
    
    # 对号矩阵 LM(ndof_per_elem, nel)
    ndof_per_elem = ndof_node * nen
    model['LM'] = np.zeros((ndof_per_elem, nel), dtype=int)
    
    for e in range(nel):
        nodes = model['IEN'][e]
        dofs = []
        for node in nodes:
            dofs.extend(model['dof_table'][node])
        model['LM'][:, e] = dofs
    
    # 初始化全局载荷向量
    model['force'] = np.zeros(model['total_dof'])
    model['force'][model['force_dof']] = model['force_val']