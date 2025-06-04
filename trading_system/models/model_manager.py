#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import hashlib
import json
import requests
from pathlib import Path

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import load_config, TradingLogger


class ModelManager:
    """模型管理器类，负责模型的列表、下载、验证和删除"""
    
    def __init__(self, config_path='config/trading_config.yaml'):
        """初始化模型管理器"""
        self.config = load_config(config_path)
        self.logger = TradingLogger('model_manager', log_level=self.config.get('general', {}).get('log_level', 'INFO'))
        
        # 模型存储目录
        self.models_dir = os.path.join(ROOT_DIR, 'models', 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 模型元数据文件
        self.metadata_file = os.path.join(self.models_dir, 'models_metadata.json')
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
    
    def list_models(self):
        """列出所有可用模型"""
        self.logger.info("列出可用模型...")
        
        # 读取模型元数据
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"读取模型元数据失败: {e}")
            return []
        
        # 检查模型文件是否存在
        models = []
        for model_id, info in metadata.items():
            model_path = os.path.join(self.models_dir, info['filename'])
            if os.path.exists(model_path):
                models.append({
                    'id': model_id,
                    'name': info.get('name', model_id),
                    'version': info.get('version', 'unknown'),
                    'type': info.get('type', 'unknown'),
                    'symbols': info.get('symbols', []),
                    'created_at': info.get('created_at', 'unknown'),
                    'file_size': os.path.getsize(model_path),
                    'path': model_path
                })
        
        return models
    
    def download_model(self, model_id, url=None):
        """从远程服务器下载模型"""
        self.logger.info(f"下载模型: {model_id}")
        
        if url is None:
            # 从配置中获取默认URL
            base_url = self.config.get('models', {}).get('remote_url')
            if not base_url:
                self.logger.error("未提供URL且配置中没有设置默认URL")
                return False
            url = f"{base_url}/{model_id}"
        
        try:
            # 下载模型文件
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 解析文件名
            if 'Content-Disposition' in response.headers:
                filename = response.headers['Content-Disposition'].split('filename=')[1].strip('"')
            else:
                filename = f"{model_id}.keras"
            
            # 保存模型文件
            model_path = os.path.join(self.models_dir, filename)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 计算文件哈希值
            file_hash = self._calculate_file_hash(model_path)
            
            # 更新元数据
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
            
            metadata[model_id] = {
                'filename': filename,
                'name': model_id,
                'version': '1.0',
                'type': 'unknown',
                'symbols': [],
                'created_at': 'unknown',
                'downloaded_at': datetime.now().isoformat(),
                'file_hash': file_hash,
                'url': url
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"模型下载成功: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"下载模型失败: {e}")
            return False
    
    def verify_model(self, model_id):
        """验证模型完整性"""
        self.logger.info(f"验证模型完整性: {model_id}")
        
        try:
            # 读取模型元数据
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if model_id not in metadata:
                self.logger.error(f"模型 {model_id} 不存在于元数据中")
                return False
            
            model_info = metadata[model_id]
            model_path = os.path.join(self.models_dir, model_info['filename'])
            
            if not os.path.exists(model_path):
                self.logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 计算文件哈希值
            file_hash = self._calculate_file_hash(model_path)
            
            # 比较哈希值
            if 'file_hash' in model_info and file_hash == model_info['file_hash']:
                self.logger.info(f"模型验证成功: {model_id}")
                return True
            else:
                self.logger.warning(f"模型哈希值不匹配: {model_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"验证模型失败: {e}")
            return False
    
    def delete_model(self, model_id):
        """删除模型"""
        self.logger.info(f"删除模型: {model_id}")
        
        try:
            # 读取模型元数据
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if model_id not in metadata:
                self.logger.error(f"模型 {model_id} 不存在于元数据中")
                return False
            
            model_info = metadata[model_id]
            model_path = os.path.join(self.models_dir, model_info['filename'])
            
            # 删除模型文件
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # 更新元数据
            del metadata[model_id]
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"模型删除成功: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除模型失败: {e}")
            return False
    
    def _calculate_file_hash(self, file_path):
        """计算文件的SHA256哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='加密货币交易系统 - 模型管理工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 列出模型命令
    list_parser = subparsers.add_parser('list', help='列出可用模型')
    list_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    
    # 下载模型命令
    download_parser = subparsers.add_parser('download', help='从远程服务器下载模型')
    download_parser.add_argument('--model-id', type=str, required=True, help='模型ID')
    download_parser.add_argument('--url', type=str, default=None, help='模型下载URL')
    download_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    
    # 验证模型命令
    verify_parser = subparsers.add_parser('verify', help='验证模型完整性')
    verify_parser.add_argument('--model-id', type=str, required=True, help='模型ID')
    verify_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    
    # 删除模型命令
    delete_parser = subparsers.add_parser('delete', help='删除模型')
    delete_parser.add_argument('--model-id', type=str, required=True, help='模型ID')
    delete_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建模型管理器
    manager = ModelManager(args.config if hasattr(args, 'config') else 'config/trading_config.yaml')
    
    if args.command == 'list':
        models = manager.list_models()
        if models:
            print("可用模型列表:")
            for i, model in enumerate(models):
                print(f"{i+1}. {model['id']} ({model['type']}, {model['version']})")
                print(f"   名称: {model['name']}")
                print(f"   交易对: {', '.join(model['symbols']) if model['symbols'] else '未知'}")
                print(f"   创建时间: {model['created_at']}")
                print(f"   文件大小: {model['file_size'] / 1024:.2f} KB")
                print(f"   文件路径: {model['path']}")
                print()
        else:
            print("没有可用模型")
    
    elif args.command == 'download':
        success = manager.download_model(args.model_id, args.url)
        if success:
            print(f"模型 {args.model_id} 下载成功")
        else:
            print(f"模型 {args.model_id} 下载失败")
            sys.exit(1)
    
    elif args.command == 'verify':
        valid = manager.verify_model(args.model_id)
        if valid:
            print(f"模型 {args.model_id} 验证成功")
        else:
            print(f"模型 {args.model_id} 验证失败")
            sys.exit(1)
    
    elif args.command == 'delete':
        success = manager.delete_model(args.model_id)
        if success:
            print(f"模型 {args.model_id} 删除成功")
        else:
            print(f"模型 {args.model_id} 删除失败")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    from datetime import datetime
    main()