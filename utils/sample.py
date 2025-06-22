import random
import argparse
import os

def sample_lines_from_jsonl(input_filepath, output_filepath, num_samples, random_seed=None):
    if not os.path.exists(input_filepath):
        print(f"错误: 输入文件 '{input_filepath}' 未找到。")
        return

    if input_filepath == output_filepath:
        print("错误: 输入文件和输出文件路径不能相同，以避免数据丢失。")
        return

    if random_seed is not None:
        random.seed(random_seed)
        print(f"使用随机种子: {random_seed}")

    try:
        print(f"正在从 '{input_filepath}' 读取所有行...")
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            all_lines = infile.readlines()
        
        num_total_lines = len(all_lines)
        print(f"文件共有 {num_total_lines} 行。")

        if num_total_lines == 0:
            print("输入文件为空，无法抽样。")
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                pass 
            print(f"已创建空的输出文件 '{output_filepath}'。")
            return

        actual_samples_to_take = min(num_samples, num_total_lines)
        
        if actual_samples_to_take < num_samples:
            print(f"注意: 文件行数 ({num_total_lines}) 少于请求的抽样数 ({num_samples})。将抽取所有 {actual_samples_to_take} 行。")
        else:
            print(f"准备随机抽取 {actual_samples_to_take} 行...")

        sampled_lines = random.sample(all_lines, actual_samples_to_take)
        
        print(f"抽样完成。正在将 {len(sampled_lines)} 行写入 '{output_filepath}'...")
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for line in sampled_lines:
                outfile.write(line) # line 本身包含换行符
        
        print(f"成功！已将 {len(sampled_lines)} 行随机抽样数据保存到 '{output_filepath}'。")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="从 JSONL 文件中随机抽取指定数量的行。")
    parser.add_argument("input_file", help="输入的 JSONL 文件路径。")
    parser.add_argument("output_file", help="输出的 JSONL 文件路径，用于存放抽样结果。")
    parser.add_argument("-n", "--num_samples", type=int, default=100,
                        help="要抽取的行数 (默认为 100)。如果文件行数不足，则抽取所有行。")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="可选的随机种子，用于可复现的抽样。")
    
    args = parser.parse_args()

    sample_lines_from_jsonl(args.input_file, args.output_file, args.num_samples, args.seed)

if __name__ == "__main__":
    test_input_filename = "/cluster/home1/wzx/EgoReasoner/data/full/embodied_agent_train_dataset_1.jsonl"
    test_output_filename = "/cluster/home1/wzx/EgoReasoner/data/full/embodied_agent_train_dataset_1_100.jsonl"
    
    
    main()