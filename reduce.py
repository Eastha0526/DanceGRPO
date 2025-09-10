import os

# 설정
original_file = "./assets/prompts.txt"
new_file = "./assets/prompts_2000.txt"
num_lines_to_keep = 2000

print(f"Reading from '{original_file}'...")

try:
    # 원본 파일을 열어 모든 줄을 읽어옵니다.
    with open(original_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Original file has {len(lines)} lines.")

    # 처음 2000줄만 잘라서 새 파일에 씁니다.
    with open(new_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:num_lines_to_keep])

    print(f"✅ Successfully created '{new_file}' with {num_lines_to_keep} lines.")

except FileNotFoundError:
    print(f"❌ Error: The file '{original_file}' was not found.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")