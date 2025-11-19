import json
import argparse

def add_no_think(input_file, output_file):
    """
    给JSONL文件中每行的"instruction"字段添加"/no_think"后缀
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 检查是否存在instruction字段
                if "instruction" in data:
                    # 确保instruction是字符串类型
                    if isinstance(data["instruction"], str):
                        # 避免重复添加/no_think
                        if not data["instruction"].strip().endswith("/no_think"):
                            # 在指令末尾添加/no_think（保留原指令的标点和空格）
                            data["instruction"] = f"{data['instruction'].rstrip()} /no_think"
                
                if "input" in data:
                    # 确保instruction是字符串类型
                    if isinstance(data["input"], str):
                        # 避免重复添加/no_think
                        data["input"] = f"{data['input'].rstrip()}\nPlease provide only the final answer, wrapped in <answer> tags.\nAnswer:"
                data['prompt'] = data['instruction'] + '\n' + data['input']
                del data['instruction']
                del data['input']
                # 写入处理后的行
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"警告：第{line_num}行不是有效的JSON，已跳过")
            except Exception as e:
                print(f"处理第{line_num}行时出错：{str(e)}，已跳过")
    
    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='修改jsonl文件中的instruction字段')
    parser.add_argument('--input', type=str, required=True, help='输入jsonl文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出jsonl文件路径')
    args = parser.parse_args()
    
    add_no_think(args.input, args.output)