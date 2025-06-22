import json

def process_trajectory_line(line):
    
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"警告: 无法解析行: {line.strip()}. 错误: {e}")
        return None

    if "images" in data and isinstance(data["images"], list) and data["images"]:
        data["images"] = [data["images"][-1]] 
    if "messages" in data and isinstance(data["messages"], list):
        first_image_tag_found_in_trajectory = False
        for i in range(len(data["messages"])):
            message = data["messages"][i]
            if isinstance(message, dict) and "content" in message and isinstance(message["content"], str):
                content = message["content"]
                if first_image_tag_found_in_trajectory:
                    data["messages"][i]["content"] = content.replace("<image>", "")
                else:
                    first_occurrence_index = content.find("<image>")
                    if first_occurrence_index != -1:
                        prefix = content[:first_occurrence_index + len("<image>")]
                        suffix = content[first_occurrence_index + len("<image>"):]
                        suffix_cleaned = suffix.replace("<image>", "")
                        data["messages"][i]["content"] = prefix + suffix_cleaned
                        first_image_tag_found_in_trajectory = True
            else:
                print(f"警告: 'messages' 列表中发现无效的条目格式: {message}")
    
    return json.dumps(data, ensure_ascii=False)


def process_jsonl_file(input_filepath, output_filepath):
    processed_lines = 0
    error_lines = 0
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:
            for line_number, line in enumerate(infile, 1):
                processed_line_json = process_trajectory_line(line)
                if processed_line_json:
                    outfile.write(processed_line_json + '\n')
                    processed_lines +=1
                else:
                    print(f"跳过第 {line_number} 行的处理。")
                    error_lines +=1
            print(f"\n处理完成!")
            print(f"成功处理 {processed_lines} 行。")
            if error_lines > 0:
                print(f"处理失败或跳过 {error_lines} 行。")
            print(f"结果已保存到: {output_filepath}")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_filepath}' 未找到。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    input_filename = "/cluster/home1/wzx/EgoReasoner/data/full/embodied_agent_train_dataset.jsonl"
    output_filename = "/cluster/home1/wzx/EgoReasoner/data/full/embodied_agent_train_dataset_last_image.jsonl" # 建议更改输出文件名以反映更改
    process_jsonl_file(input_filename, output_filename)