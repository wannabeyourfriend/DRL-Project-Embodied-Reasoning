import json
import argparse

def check_json_line_validity(line_number, data_object):
    required_keys = ["action", "messages", "images"]
    errors = []

    for key in required_keys:
        if key not in data_object:
            errors.append(f"行 {line_number}: 错误 - 键 '{key}' 缺失。")
        elif not data_object[key]:
            errors.append(f"行 {line_number}: 错误 - 键 '{key}' 存在但其值为空。")
    
    return errors

def main():
    parser = argparse.ArgumentParser(description="检查 JSONL 文件的每一行是否包含指定的非空键 (action, message, images)。")
    parser.add_argument("filepath", help="要检查的 JSONL 文件的路径。")

    args = parser.parse_args()

    filepath = args.filepath
    total_lines = 0
    valid_lines = 0
    invalid_lines = 0
    all_error_messages = []

    print(f"开始检查文件: {filepath}\n")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line_content in enumerate(f, 1):
                total_lines += 1
                try:
                    data = json.loads(line_content.strip())
                    if not isinstance(data, dict):
                        error_msg = f"行 {i}: 错误 - 内容不是一个有效的 JSON 对象 (字典)。内容: '{line_content.strip()}'"
                        all_error_messages.append(error_msg)
                        invalid_lines += 1
                        continue

                    line_errors = check_json_line_validity(i, data)
                    
                    if not line_errors:
                        valid_lines += 1
                    else:
                        invalid_lines += 1
                        all_error_messages.extend(line_errors)


                except json.JSONDecodeError as e:
                    error_msg = f"行 {i}: 错误 - JSON 解析失败: {e}. 内容: '{line_content.strip()}'"
                    all_error_messages.append(error_msg)
                    invalid_lines += 1
                except Exception as e:
                    error_msg = f"行 {i}: 发生意外错误: {e}. 内容: '{line_content.strip()}'"
                    all_error_messages.append(error_msg)
                    invalid_lines +=1


    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    print("\n--- 检查结果 ---")
    if not all_error_messages:
        print("所有行均符合要求！")
    else:
        print("发现以下问题：")
        for msg in all_error_messages:
            print(msg)
    
    print("\n--- 总结 ---")
    print(f"总共处理行数: {total_lines}")
    print(f"符合要求的行数: {valid_lines}")
    print(f"不符合要求的行数: {invalid_lines}")

if __name__ == "__main__":
    main()