from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    # 这里直接用一个字符串模板，你也可以改成 render_template + 模板文件
    html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Flask 示例页面</title>
    </head>
    <body>
        <h1>欢迎来到 Flask 端口 8080 的页面！</h1>
        <p>当前监听端口：8080</p>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    # host='0.0.0.0' 表示接受来自局域网内其他设备的访问，若只在本机测试可改为默认（127.0.0.1）
    app.run(host='0.0.0.0', port=8080, debug=True)
