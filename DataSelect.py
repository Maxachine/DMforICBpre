import pandas as pd

def extract_data_to_txt(file_path, sheet_name, output_file, headers_selection):
    """
    从指定的 Excel sheet 中提取指定表头的数据并写入文本文件。

    :param file_path: Excel 文件路径
    :param sheet_name: 要读取的 sheet 名称
    :param output_file: 输出文本文件路径
    :param headers_selection: 一个字典，表头为键，布尔值为值，表示是否选中该列
    """
    try:
        # 读取指定的 sheet
        data = pd.read_excel(file_path, sheet_name=sheet_name)

        # 获取需要的表头
        selected_columns = [header for header, is_selected in headers_selection.items() if is_selected]

        # 检查指定的列是否存在
        for column in selected_columns:
            if column not in data.columns:
                raise ValueError(f"列名 '{column}' 不存在于 sheet '{sheet_name}' 中。")

        # 提取指定列的数据
        selected_data = data[selected_columns]

        # 将数据写入文本文件
        selected_data.to_csv(output_file, sep=',', index=False)
        print(f"数据已成功写入 '{output_file}' 文件。")

    except Exception as e:
        print(f"发生错误: {e}")

CancerType = False
headers_selection = {
    "TMB": True,
    "PDL1_TPS(%)": False,
    "Systemic_therapy_history": True,
    "Albumin": True,
    "CancerType_grouped": False,
    "NLR": True,
    "Age": True,
    "FCNA": False,
    "Drug": False,
    "Sex": False,
    "MSI": False,
    "Stage": False,
    "HLA_LOH": False,
    "HED": False,
    "Platelets": False,
    "HGB": False,
    "BMI": False,
    "PFS_Event": False,
    "PFS_Months": False,
    "OS_Event": False,
    "OS_Months": False,
    "CancerType1": CancerType,
    "CancerType2": CancerType,
    "CancerType3": CancerType,
    "CancerType4": CancerType,
    "CancerType5": CancerType,
    "CancerType6": CancerType,
    "CancerType7": CancerType,
    "CancerType8": CancerType,
    "CancerType9": CancerType,
    "CancerType10": CancerType,
    "CancerType11": CancerType,
    "CancerType12": CancerType,
    "CancerType13": CancerType,
    "CancerType14": CancerType,
    "CancerType15": CancerType,
    "CancerType16": CancerType,
    "CancerType17": CancerType,
    "CancerType18": CancerType,
    "new": True,
    "Response": True,
}


# 示例用法
file_path = "AllData.xlsx"          # Excel 文件路径
sheet_name = "Chowell_test"               # 指定的 sheet 名称
output_file = "test_type123.txt"         # 输出文件路径 ./ScoreMatchingDM/

extract_data_to_txt(file_path, sheet_name, output_file, headers_selection)


