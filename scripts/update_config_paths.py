#!/usr/bin/env python3
"""
批量更新配置文件中的绝对路径
将 /share/data/ripl/jjt/projects/oracles 替换为 ${paths.project_root}
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OLD_PATH = "/share/data/ripl/jjt/projects/oracles"
PATTERN = re.compile(re.escape(OLD_PATH))

def update_file(filepath: Path):
    """更新单个文件中的路径"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含旧路径
        if OLD_PATH not in content:
            return False
        
        # 替换路径
        new_content = PATTERN.sub('${paths.project_root}', content)
        
        # 如果内容有变化，写回文件
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"错误处理 {filepath}: {e}")
        return False

def main():
    """主函数：遍历所有配置文件并更新"""
    updated_files = []
    
    # 遍历所有 yaml 文件
    for yaml_file in CONFIG_DIR.rglob("*.yaml"):
        # 跳过 paths.yaml 和 paths_local.yaml
        if yaml_file.name in ("paths.yaml", "paths_local.yaml"):
            continue
        
        if update_file(yaml_file):
            updated_files.append(yaml_file)
            print(f"✓ 已更新: {yaml_file.relative_to(PROJECT_ROOT)}")
    
    print(f"\n总共更新了 {len(updated_files)} 个文件")
    
    if updated_files:
        print("\n请检查更改，然后提交:")
        print("  git add config/")
        print("  git commit -m 'Update config paths to use paths.project_root'")

if __name__ == "__main__":
    main()

