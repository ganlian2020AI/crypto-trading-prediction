#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import json
from datetime import datetime

# 尝试导入可选的数据库驱动
try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import TradingLogger


class DatabaseManager:
    """数据库管理类，支持SQLite、MySQL和PostgreSQL"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据库管理器
        
        参数:
            config: 配置信息
        """
        self.config = config
        self.db_config = config.get('database', {})
        self.db_type = self.db_config.get('type', 'sqlite').lower()
        
        # 初始化日志
        log_level = config.get('general', {}).get('log_level', 'INFO')
        log_file = os.path.join(ROOT_DIR, 'logs', 'database.log')
        self.logger = TradingLogger('database', log_level, log_file)
        
        # 连接和游标
        self.conn = None
        self.cursor = None
        
        # 连接到数据库
        self._connect()
    
    def _connect(self) -> bool:
        """
        连接到数据库
        
        返回:
            是否连接成功
        """
        try:
            if self.db_type == 'sqlite':
                # SQLite数据库
                db_path = self.db_config.get('path', 'data/trading_system.db')
                
                # 确保路径是绝对路径
                if not os.path.isabs(db_path):
                    db_path = os.path.join(ROOT_DIR, db_path)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                
                self.conn = sqlite3.connect(db_path)
                self.conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
                self.cursor = self.conn.cursor()
                
                self.logger.info(f"已连接到SQLite数据库: {db_path}")
                return True
                
            elif self.db_type == 'mysql' or self.db_type == 'mariadb':
                # MySQL/MariaDB数据库
                if not MYSQL_AVAILABLE:
                    self.logger.error("未安装PyMySQL，无法连接到MySQL/MariaDB数据库")
                    return False
                
                host = self.db_config.get('host', 'localhost')
                port = self.db_config.get('port', 3306)
                user = self.db_config.get('user', 'root')
                password = self.db_config.get('password', '')
                database = self.db_config.get('database', 'trading_system')
                
                self.conn = pymysql.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database,
                    charset='utf8mb4'
                )
                self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)
                
                self.logger.info(f"已连接到MySQL/MariaDB数据库: {host}:{port}/{database}")
                return True
                
            elif self.db_type == 'postgresql':
                # PostgreSQL数据库
                if not POSTGRESQL_AVAILABLE:
                    self.logger.error("未安装psycopg2，无法连接到PostgreSQL数据库")
                    return False
                
                host = self.db_config.get('host', 'localhost')
                port = self.db_config.get('port', 5432)
                user = self.db_config.get('user', 'postgres')
                password = self.db_config.get('password', '')
                database = self.db_config.get('database', 'trading_system')
                
                self.conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=database
                )
                self.cursor = self.conn.cursor()
                
                self.logger.info(f"已连接到PostgreSQL数据库: {host}:{port}/{database}")
                return True
                
            else:
                self.logger.error(f"不支持的数据库类型: {self.db_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"连接数据库失败: {str(e)}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.logger.info("已关闭数据库连接")
    
    def execute(self, sql: str, params: tuple = None) -> bool:
        """
        执行SQL语句
        
        参数:
            sql: SQL语句
            params: SQL参数
            
        返回:
            是否执行成功
        """
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"执行SQL失败: {str(e)}\nSQL: {sql}\n参数: {params}")
            self.conn.rollback()
            return False
    
    def query(self, sql: str, params: tuple = None) -> List[Dict]:
        """
        查询数据
        
        参数:
            sql: SQL语句
            params: SQL参数
            
        返回:
            查询结果列表
        """
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            
            if self.db_type == 'sqlite':
                # SQLite结果转为字典列表
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            elif self.db_type == 'mysql' or self.db_type == 'mariadb':
                # MySQL/MariaDB已经是字典格式
                return self.cursor.fetchall()
            elif self.db_type == 'postgresql':
                # PostgreSQL结果转为字典列表
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
        except Exception as e:
            self.logger.error(f"查询失败: {str(e)}\nSQL: {sql}\n参数: {params}")
            return []
    
    def create_tables(self):
        """创建必要的数据表"""
        try:
            # 创建价格数据表
            price_table_sql = """
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                source TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
            """
            
            # 根据不同数据库调整SQL语法
            if self.db_type == 'mysql' or self.db_type == 'mariadb':
                price_table_sql = """
                CREATE TABLE IF NOT EXISTS price_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open DOUBLE NOT NULL,
                    high DOUBLE NOT NULL,
                    low DOUBLE NOT NULL,
                    close DOUBLE NOT NULL,
                    volume DOUBLE NOT NULL,
                    source VARCHAR(50),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
                """
            elif self.db_type == 'postgresql':
                price_table_sql = """
                CREATE TABLE IF NOT EXISTS price_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    source VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
                """
            
            self.execute(price_table_sql)
            
            # 创建交易记录表
            trades_table_sql = """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                order_type TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                fee REAL,
                total_value REAL NOT NULL,
                status TEXT NOT NULL,
                trade_id TEXT,
                strategy TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # 根据不同数据库调整SQL语法
            if self.db_type == 'mysql' or self.db_type == 'mariadb':
                trades_table_sql = """
                CREATE TABLE IF NOT EXISTS trades (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    order_type VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    price DOUBLE NOT NULL,
                    quantity DOUBLE NOT NULL,
                    fee DOUBLE,
                    total_value DOUBLE NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    trade_id VARCHAR(100),
                    strategy VARCHAR(50),
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            elif self.db_type == 'postgresql':
                trades_table_sql = """
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    order_type VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    fee DOUBLE PRECISION,
                    total_value DOUBLE PRECISION NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    trade_id VARCHAR(100),
                    strategy VARCHAR(50),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            
            self.execute(trades_table_sql)
            
            # 创建信号表
            signals_table_sql = """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                signal_type INTEGER NOT NULL,
                confidence REAL,
                price REAL,
                executed BOOLEAN DEFAULT 0,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # 根据不同数据库调整SQL语法
            if self.db_type == 'mysql' or self.db_type == 'mariadb':
                signals_table_sql = """
                CREATE TABLE IF NOT EXISTS signals (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    signal_type INT NOT NULL,
                    confidence DOUBLE,
                    price DOUBLE,
                    executed BOOLEAN DEFAULT FALSE,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            elif self.db_type == 'postgresql':
                signals_table_sql = """
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    signal_type INT NOT NULL,
                    confidence DOUBLE PRECISION,
                    price DOUBLE PRECISION,
                    executed BOOLEAN DEFAULT FALSE,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            
            self.execute(signals_table_sql)
            
            self.logger.info("已创建必要的数据表")
            return True
            
        except Exception as e:
            self.logger.error(f"创建数据表失败: {str(e)}")
            return False
    
    def save_price_data(self, df: pd.DataFrame, symbol: str, source: str = 'binance') -> bool:
        """
        保存价格数据到数据库
        
        参数:
            df: 价格数据DataFrame
            symbol: 交易对符号
            source: 数据来源
            
        返回:
            是否保存成功
        """
        try:
            # 确保DataFrame包含所需列
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"DataFrame缺少必需的列: {col}")
                    return False
            
            # 转换timestamp列为datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 准备插入数据
            records = []
            for _, row in df.iterrows():
                record = (
                    symbol,
                    row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    source
                )
                records.append(record)
            
            # 构建插入SQL
            if self.db_type == 'sqlite':
                sql = """
                INSERT OR REPLACE INTO price_data 
                (symbol, timestamp, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
            elif self.db_type == 'mysql' or self.db_type == 'mariadb':
                sql = """
                INSERT INTO price_data 
                (symbol, timestamp, open, high, low, close, volume, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume),
                source = VALUES(source)
                """
            elif self.db_type == 'postgresql':
                sql = """
                INSERT INTO price_data 
                (symbol, timestamp, open, high, low, close, volume, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source
                """
            
            # 批量插入数据
            for record in records:
                self.execute(sql, record)
            
            self.logger.info(f"已保存{len(records)}条{symbol}价格数据到数据库")
            return True
            
        except Exception as e:
            self.logger.error(f"保存价格数据失败: {str(e)}")
            return False
    
    def load_price_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从数据库加载价格数据
        
        参数:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            价格数据DataFrame
        """
        try:
            # 构建查询条件
            conditions = ["symbol = ?"]
            params = [symbol]
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            
            # 构建SQL查询
            sql = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp ASC
            """
            
            # 执行查询
            self.cursor.execute(sql, tuple(params))
            
            # 获取结果
            if self.db_type == 'sqlite':
                # SQLite结果转为字典列表
                columns = [col[0] for col in self.cursor.description]
                rows = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            elif self.db_type == 'mysql' or self.db_type == 'mariadb':
                # MySQL/MariaDB已经是字典格式
                rows = self.cursor.fetchall()
            elif self.db_type == 'postgresql':
                # PostgreSQL结果转为字典列表
                columns = [col[0] for col in self.cursor.description]
                rows = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            # 转换为DataFrame
            df = pd.DataFrame(rows)
            
            if not df.empty:
                # 转换timestamp列为datetime类型
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"已从数据库加载{len(df)}条{symbol}价格数据")
            return df
            
        except Exception as e:
            self.logger.error(f"加载价格数据失败: {str(e)}")
            return pd.DataFrame()
    
    def save_trade(self, trade_data: Dict) -> bool:
        """
        保存交易记录到数据库
        
        参数:
            trade_data: 交易数据字典
            
        返回:
            是否保存成功
        """
        try:
            # 构建插入SQL
            sql = """
            INSERT INTO trades 
            (symbol, timestamp, order_type, side, price, quantity, fee, total_value, status, trade_id, strategy, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # 根据不同数据库调整SQL语法
            if self.db_type == 'mysql' or self.db_type == 'mariadb' or self.db_type == 'postgresql':
                sql = sql.replace('?', '%s')
            
            # 准备参数
            params = (
                trade_data.get('symbol'),
                trade_data.get('timestamp'),
                trade_data.get('order_type'),
                trade_data.get('side'),
                trade_data.get('price'),
                trade_data.get('quantity'),
                trade_data.get('fee'),
                trade_data.get('total_value'),
                trade_data.get('status'),
                trade_data.get('trade_id'),
                trade_data.get('strategy'),
                trade_data.get('notes')
            )
            
            # 执行插入
            success = self.execute(sql, params)
            
            if success:
                self.logger.info(f"已保存交易记录: {trade_data.get('symbol')} {trade_data.get('side')} {trade_data.get('quantity')} @ {trade_data.get('price')}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"保存交易记录失败: {str(e)}")
            return False
    
    def save_signal(self, signal_data: Dict) -> bool:
        """
        保存交易信号到数据库
        
        参数:
            signal_data: 信号数据字典
            
        返回:
            是否保存成功
        """
        try:
            # 构建插入SQL
            sql = """
            INSERT INTO signals 
            (symbol, timestamp, signal_type, confidence, price, executed, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            # 根据不同数据库调整SQL语法
            if self.db_type == 'mysql' or self.db_type == 'mariadb' or self.db_type == 'postgresql':
                sql = sql.replace('?', '%s')
            
            # 准备参数
            params = (
                signal_data.get('symbol'),
                signal_data.get('timestamp'),
                signal_data.get('signal_type'),
                signal_data.get('confidence'),
                signal_data.get('price'),
                signal_data.get('executed', False),
                signal_data.get('notes')
            )
            
            # 执行插入
            success = self.execute(sql, params)
            
            if success:
                self.logger.info(f"已保存交易信号: {signal_data.get('symbol')} 类型:{signal_data.get('signal_type')} 置信度:{signal_data.get('confidence')}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"保存交易信号失败: {str(e)}")
            return False


def main():
    """测试数据库管理器"""
    import yaml
    
    # 加载配置
    config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建数据库管理器
    db_manager = DatabaseManager(config)
    
    # 创建表
    db_manager.create_tables()
    
    # 测试查询
    result = db_manager.query("SELECT * FROM price_data LIMIT 5")
    print(f"查询结果: {result}")
    
    # 关闭连接
    db_manager.close()


if __name__ == "__main__":
    main()