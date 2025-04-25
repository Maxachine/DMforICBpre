from DataSelect import SelectAndGenerate
import itertools

if __name__ == '__main__':
    # 第一步，遍历所有六维条件组合，生成训练数据 
    param_grid = {
        'TMB': [True,False],
        'Systemic_therapy_history': [True,False],
        'Albumin': [True,False],
        'NLR': [True,False],
        'Age': [True,False],
        'FCNA': [True,False],
        'HED': [True,False],
        'Platelets': [True,False],
        'HGB': [True,False],
        'BMI': [True,False],
        'PFS_Months': [True,False],
        'OS_Months': [True,False],
        'new': [True,False],
    } 
    # 生成所有可能的组合
    all_combinations = list(itertools.product(*param_grid.values()))

    # 筛选出True的个数恰好为6的组合
    filtered_combinations = [params for params in all_combinations if sum(params) == 6]

    for params in filtered_combinations:
        param_string = "".join(['1' if p else '0' for p in params])
    
        # 生成 output_file 名称
        output_file = f"./test_files/{param_string}.txt"
        SelectAndGenerate(params, output_file, 'test')