import json
import os
import glob
import argparse

# 目标顶层键的顺序
TARGET_KEY_ORDER = ["messages", "action", "images"]

def reorder_and_modify_json_object(data_dict):
    """
    修改单个JSON对象:
    1. 在 'messages' 列表末尾追加一个空的助手消息。
    2. 按照 TARGET_KEY_ORDER 重新排序顶层键，其他键跟在后面。
    """
    if not isinstance(data_dict, dict):
        print(f"警告: 遇到非字典项: {type(data_dict)}。将原样返回。")
        return data_dict

    # --- 1. 添加空的助手消息 ---
    if "messages" not in data_dict:
        # print(f"信息: 在一个数据对象中未找到 'messages' 键。将创建并添加一个空的助手回答。")
        data_dict["messages"] = []  # 如果 'messages' 不存在，则初始化为空列表
    elif not isinstance(data_dict["messages"], list):
        original_messages_val = data_dict["messages"]
        print(f"警告: 在一个数据对象中的 'messages' 键不是列表 (类型: {type(original_messages_val)})。")
        if isinstance(original_messages_val, dict): # 如果原值是单个消息字典
            print(f"信息: 将原有的单个消息对象放入列表中。")
            data_dict["messages"] = [original_messages_val]
        else:
            print(f"信息: 将用包含空助手回答的新列表覆盖 'messages'。")
            data_dict["messages"] = [] # 回退方案：用空列表覆盖

    # 向 messages 列表追加空助手回答
    data_dict["messages"].append({"role": "assistant", "content": ""})

    # --- 2. 重新排序键 ---
    reordered_data = {}

    # 首先按照 TARGET_KEY_ORDER 添加键
    for key in TARGET_KEY_ORDER:
        if key in data_dict:
            reordered_data[key] = data_dict[key]

    # 然后添加原始 data_dict 中剩余的键，保持它们原有的相对顺序
    for key, value in data_dict.items():
        if key not in reordered_data:  # 只添加尚未通过 TARGET_KEY_ORDER 添加的键
            reordered_data[key] = value

    return reordered_data

def main_script():
    parser = argparse.ArgumentParser(
        description="修改JSONL文件：为每行数据在messages中添加一个空助手回答，并按指定顺序调整顶层键。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "paths", metavar="PATH", type=str, nargs='+',
        help="要处理的JSONL文件路径、包含JSONL文件的目录路径，或glob模式。\n"
             "例如: 'data/file.jsonl', 'logs/*.jsonl', 'my_data_folder/', 'project_data/**/*.jsonl'"
    )
    parser.add_argument(
        "--overwrite", action='store_true',
        help="覆盖原始文件。警告: 此操作不可逆！\n"
             "默认行为是创建带 '_modified' 后缀的新文件，或使用 --output_dir。"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="用于保存修改后文件的目录。原始文件名将被保留。\n"
             "如果未设置此项且未指定 --overwrite，则会在原文件名后添加 '_modified' 后缀。"
    )

    args = parser.parse_args()

    # --- 收集和规范化文件路径 ---
    input_path_patterns = []
    for p_arg in args.paths:
        if os.path.isdir(p_arg) and not any(c in p_arg for c in "*?[]"):
            # 如果用户提供的是一个普通目录路径，则为其创建递归搜索 *.jsonl 的模式
            input_path_patterns.append(os.path.join(p_arg, "**", "*.jsonl"))
        else:
            # 用户提供的是文件路径或包含通配符的glob模式
            input_path_patterns.append(p_arg)

    filepaths_to_process_intermediate = []
    for path_pattern in input_path_patterns:
        filepaths_to_process_intermediate.extend(glob.glob(path_pattern, recursive=True))

    # 过滤以确保只处理实际存在的文件，并且是 .jsonl 后缀
    filepaths_to_process = sorted(list(set(
        os.path.normpath(p) for p in filepaths_to_process_intermediate
        if os.path.isfile(p) and p.endswith(".jsonl")
    )))

    if not filepaths_to_process:
        print("未找到与给定路径匹配的JSONL文件。")
        return

    print(f"找到 {len(filepaths_to_process)} 个JSONL文件待处理。")
    processed_count = 0
    error_count = 0

    for filepath in filepaths_to_process:
        print(f"正在处理: {filepath} ...")

        output_filepath = filepath
        if not args.overwrite:
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True) # 确保输出目录存在
                output_filepath = os.path.join(args.output_dir, os.path.basename(filepath))
            else:
                root, ext = os.path.splitext(filepath)
                output_filepath = f"{root}_modified{ext}"

        # 确保输出文件的目录存在 (主要针对 output_dir 的情况)
        output_file_dir = os.path.dirname(output_filepath)
        if output_file_dir and not os.path.exists(output_file_dir):
            try:
                os.makedirs(output_file_dir, exist_ok=True)
            except OSError as e:
                print(f"  错误: 无法为输出文件 {output_filepath} 创建目录 {output_file_dir}。已跳过。错误: {e}")
                error_count += 1
                continue

        try:
            output_lines = []
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line_num, line in enumerate(infile, 1):
                    stripped_line = line.strip()
                    if not stripped_line: # 保留空行（如果文件中有）
                        output_lines.append("")
                        continue
                    try:
                        data = json.loads(stripped_line)
                        modified_data = reorder_and_modify_json_object(data)
                        # JSONL每行都是一个独立的、紧凑的JSON对象
                        output_lines.append(json.dumps(modified_data, ensure_ascii=False))
                    except json.JSONDecodeError as e_json:
                        print(f"  警告: 解析 {filepath} 文件第 {line_num} 行时发生JSON解码错误。该行将保持原样。错误: {e_json}")
                        output_lines.append(stripped_line) # 出错则保留原始行
                    except Exception as e_obj_proc: # 捕获 reorder_and_modify_json_object 中可能发生的其他错误
                        print(f"  警告: 处理 {filepath} 文件第 {line_num} 行时发生错误。该行将保持原样。错误: {e_obj_proc}")
                        output_lines.append(stripped_line)

            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for out_line in output_lines:
                    outfile.write(out_line + '\n')

            if output_filepath != filepath:
                print(f"  已将修改后的内容保存到: {output_filepath}")
            else:
                print(f"  已修改并覆盖: {filepath}")
            processed_count += 1

        except Exception as e_file_proc: # 捕获文件级处理错误 (例如读取权限问题)
            print(f"  错误: 处理文件 {filepath} 时发生错误: {e_file_proc}")
            error_count += 1
            # import traceback # 如果需要更详细的错误堆栈
            # traceback.print_exc()

    print(f"\n处理完成。成功修改 {processed_count} 个文件。处理中遇到错误的文件数: {error_count}。")
    if not args.overwrite:
        if args.output_dir:
            print(f"修改后的文件保存在或位于: {os.path.abspath(args.output_dir)}")
        else:
            print("修改后的文件在其原目录中以 '_modified' 后缀保存。")

if __name__ == "__main__":
    main_script()