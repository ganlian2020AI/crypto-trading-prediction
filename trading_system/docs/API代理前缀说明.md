# API代理前缀功能说明

## 概述

API代理前缀是一种允许通过第三方服务转发API请求的功能。在某些网络环境下，直接访问交易所API可能会受到限制，通过配置API代理前缀，可以将请求先发送到代理服务器，再由代理服务器转发到实际的API端点。

## 支持的功能

目前，API代理前缀功能支持以下模块：

1. **数据下载模块**：支持通过代理前缀下载币安和Alpha Vantage的历史数据
2. **实盘交易模块**：支持通过代理前缀进行币安交易所的实盘交易

## 配置方法

### 1. 数据下载模块配置

在`trading_config.yaml`中配置：

```yaml
# 数据下载配置
data_download:
  # 其他配置...
  
  # API代理前缀URL（可选）
  # 例如: https://你的域名/api/proxy/
  api_proxy_prefix: ""
  
  # 代理设置
  proxy:
    enabled: false
    url: "http://127.0.0.1:7890"
```

### 2. 实盘交易模块配置

在`trading_config.yaml`中配置：

```yaml
# 实盘交易配置
live_trading:
  exchange: "binance"
  api_key: ""
  api_secret: ""
  testnet: true
  # 其他配置...
  
  # API代理前缀URL（可选）
  # 例如: https://你的域名/api/proxy/
  api_proxy_prefix: ""
  
  # 代理设置
  proxy:
    enabled: false
    url: "http://127.0.0.1:7890"
```

## API代理前缀与HTTP代理的区别

1. **API代理前缀**：
   - 通过URL前缀方式工作，例如`https://你的域名/api/proxy/https://api.binance.com/api/v3/klines`
   - 适用于需要特定域名转发的场景
   - 不需要在本地配置代理服务器
   - 请求路径对服务器可见

2. **HTTP代理**：
   - 通过代理服务器转发所有HTTP/HTTPS请求
   - 需要在本地配置代理服务器
   - 可以同时代理多个域名的请求
   - 通常需要额外的代理软件（如Clash、V2Ray等）

## 使用场景

1. **API访问受限**：在某些地区，直接访问交易所API可能受到限制，可以通过配置API代理前缀解决
2. **网络优化**：通过选择地理位置更接近交易所服务器的代理服务，可以减少网络延迟
3. **安全性考虑**：避免直接暴露本机IP给交易所API

## 注意事项

1. 使用API代理前缀时，请确保代理服务器是可信的，因为所有API请求（包括API密钥）都会通过代理服务器
2. 建议使用HTTPS协议的代理前缀，确保数据传输的安全性
3. 如果同时启用了API代理前缀和HTTP代理，系统会优先使用API代理前缀，然后通过HTTP代理发送请求
4. 测试网络（testnet）也支持API代理前缀功能

## 自建代理服务

如需自建API代理服务，可以使用Nginx等反向代理服务器，配置示例：

```nginx
server {
    listen 443 ssl;
    server_name 你的域名;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location /api/proxy/ {
        resolver 8.8.8.8;
        proxy_ssl_server_name on;
        proxy_pass $scheme://$proxy_host$proxy_uri;
        proxy_set_header Host $proxy_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 从URL中提取实际的目标URL
        set $proxy_uri "";
        set $proxy_host "";
        
        if ($request_uri ~ "^/api/proxy/(https?)://([^/]+)(.*)$") {
            set $proxy_scheme $1;
            set $proxy_host $2;
            set $proxy_uri $3;
        }
    }
} 