import json
import argparse

def modify_instructions(input_file, output_file):
    """
    读取jsonl文件，为每条数据的instruction字段添加指定文本
    在"answer the following question"后添加"with only the result, no extra text"
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # 解析json数据
            data = json.loads(line)
            print("1")
            # 检查并处理instruction字段
            if 'instruction' in data:
                instruction = data['instruction']
                # 查找目标字符串位置
                target = "answer the following question"
                idx = instruction.find(target)
                if idx != -1:
                    # 在目标字符串后插入指定文本
                    new_instruction = (
                        instruction[:idx + len(target)] + 
                        ", put your final answer within \\boxed{}" + 
                        instruction[idx + len(target):]
                    )
                    data['instruction'] = new_instruction
            
            # 写入处理后的行
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"处理完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='修改jsonl文件中的instruction字段')
    parser.add_argument('--input', type=str, required=True, help='输入jsonl文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出jsonl文件路径')
    args = parser.parse_args()
    
    modify_instructions(args.input, args.output)