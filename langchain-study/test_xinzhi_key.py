import requests

url = "https://api.seniverse.com/v3/weather/now.json"

params = {
    "key": "",  # 填写你的私钥
    "location": "北京",  # 你要查询的地区可以用代号，拼音或者汉字，文档在官方下载，这里举例北京
    "language": "zh-Hans",  # 中文简体
    "unit": "c",  # 获取气温
}

response = requests.get(url, params=params)  # 发送get请求
temperature = response.json()  # 接受消息中的json部分
print(temperature['results'][0]['now'])  # 输出接收到的消息进行查看
