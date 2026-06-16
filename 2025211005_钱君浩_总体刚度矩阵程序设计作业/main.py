"""
有限元作业：总体刚度矩阵组装与桁架求解
一次性自动运行 算例1（一维杆） + 算例2（二维桁架）
"""
import numpy as np
from model import read_json, init_dof
from element import compute_all_element_stiffness
from assembly import assemble_global_stiffness
from solver import solve_displacement
from postprocess import print_results, check_stiffness_properties

def run_example(json_file):
    """运行单个算例"""
    model = read_json(json_file)
    init_dof(model)
    compute_all_element_stiffness(model)
    assemble_global_stiffness(model)
    solve_displacement(model)
    print_results(model)
    check_stiffness_properties(model)

def main():
    print("\n" + "="*80)
    print("                          开始运行 算例1：一维两单元杆")
    print("="*80)
    run_example("example1.json")

    print("\n\n" + "="*80)
    print("                          开始运行 算例2：二维两杆桁架")
    print("="*80)
    run_example("example2.json")

    print("\n" + "="*80)
    print("                              两个算例全部运行完成！")
    print("="*80)

if __name__ == "__main__":
    main()