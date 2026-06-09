import numpy as np

def truss3d_element_stiffness(x1, y1, z1, x2, y2, z2, E, A):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    
    cx = dx / L
    cy = dy / L
    cz = dz / L
    
    k = E * A / L
    C = np.array([
        [cx**2, cx*cy, cx*cz],
        [cx*cy, cy**2, cy*cz],
        [cx*cz, cy*cz, cz**2]
    ])
    
    Ke = np.zeros((6, 6))
    Ke[0:3, 0:3] = C
    Ke[0:3, 3:6] = -C
    Ke[3:6, 0:3] = -C
    Ke[3:6, 3:6] = C
    Ke = k * Ke
    
    return Ke, L, cx, cy, cz

def truss3d_element_stress(x1, y1, z1, x2, y2, z2, E, d):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    cx = dx / L
    cy = dy / L
    cz = dz / L
    
    S = np.array([-cx, -cy, -cz, cx, cy, cz])
    strain = S @ d / L
    stress = E * strain
    return strain, stress

def example1():
    print("===== 算例1：X轴直杆 =====")
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = 2, 0, 0
    E = 200e9
    A = 0.0001
    d = np.array([0.001, 0, 0, 0.003, 0, 0])
    
    Ke, L, cx, cy, cz = truss3d_element_stiffness(x1,y1,z1,x2,y2,z2,E,A)
    strain, stress = truss3d_element_stress(x1,y1,z1,x2,y2,z2,E,d)
    force = stress * A
    
    print(f"长度 L = {L:.4f} m")
    print(f"方向余弦 cx={cx:.4f}, cy={cy:.4f}, cz={cz:.4f}")
    print("单元刚度矩阵 Ke (6×6):")
    print(np.round(Ke, 2))
    print(f"应变 = {strain:.6f}")
    print(f"应力 = {stress:.2f} Pa")
    print(f"轴力 = {force:.2f} N\n")

def example2():
    print("===== 算例2：空间斜杆 =====")
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = 1, 2, 2
    E = 1e9
    A = 0.01
    d = np.array([0.00, 0.00, 0.00, 0.01, 0.02, 0.02])
    
    Ke, L, cx, cy, cz = truss3d_element_stiffness(x1,y1,z1,x2,y2,z2,E,A)
    strain, stress = truss3d_element_stress(x1,y1,z1,x2,y2,z2,E,d)
    force = stress * A
    
    print(f"长度 L = {L:.4f} m")
    print(f"方向余弦 cx={cx:.4f}, cy={cy:.4f}, cz={cz:.4f}")
    print("刚度矩阵对称：", np.allclose(Ke, Ke.T))
    print(f"应变 = {strain:.6f}")
    print(f"应力 = {stress:.2f} Pa")
    print(f"轴力 = {force:.2f} N\n")

if __name__ == "__main__":
    example1()
    example2()
    print("===== 两个算例全部运行完成 =====")