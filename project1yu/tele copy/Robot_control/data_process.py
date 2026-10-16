import pandas as pd
import os

def save_dict_to_excel(data, file_dir, sheet_name='Sheet1'):
    """将字典或字典列表保存到 Excel 文件。

    Args:
        data: 字典或字典列表。
        filename: 要保存的文件名。
        sheet_name: 工作表名称。
    """
    df = pd.DataFrame.from_dict(data, orient='index').T
    print(df)
    df.to_excel(file_dir, sheet_name=sheet_name, index=True)  # index=False 去除索引列






if __name__ == '__main__':
    excel_path = './datas'

    # test：
    data = {'name': 'Alice', 'age': 30, 'city': 'New York'}
    save_dict_to_excel(data, os.path.join(excel_path, 'single_dict.xlsx'))

    complex_dict = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [30, 25, 35],
        'city': ['New York', 'Los Angeles', 'Chicago']
    }
    save_dict_to_excel(complex_dict, os.path.join(excel_path, 'complex_dict.xlsx'))