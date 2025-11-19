
import json

def clean_kto_data(input_file, output_file):


    print(f"清洗数据: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_count = 0
    skipped_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(lines):
            try:
                item = json.loads(line.strip())
                
                # 转换 tag 字符串为 bool
                if 'kto_tag' in item:
                    if item['kto_tag'] == 'true':
                        item['kto_tag'] = True
                    elif item['kto_tag'] == 'false':
                        item['kto_tag'] = False
                
 
                output = item.get('output', '')
                if (isinstance(output, str) and 
                    output.lower().strip() in ['not found', 'unknown', '']):
                    skipped_count += 1
                    continue
                
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                cleaned_count += 1
                
            except json.JSONDecodeError:
                skipped_count += 1
                print(f"跳过第 {i+1} 行 - JSON解析错误")
                continue
    
    print(f"清洗完成!")
    print(f"保留样本: {cleaned_count}")
    print(f"跳过样本: {skipped_count}")

input_file = '/root/autodl-tmp/1105/kto_train_shuffled.jsonl'
output_file = '/root/autodl-tmp/1105/kto_train_cleaned.jsonl'

clean_kto_data(input_file, output_file)