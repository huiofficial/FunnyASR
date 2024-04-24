# FunnyASR

现在很多手机和电脑端的语音识别 APP 居然要收费，太过分了有木有！那本项目就基于开源的 funasr 进行本地的端到端语音识别，为大家做一个简单的免费语音识别工具。

后续会构建 macOS app，敬请期待！

## 使用方法

### 安装

```# 安装依赖
pip install -r requirements.txt

pip install torch torchaudio

brew install ffmpeg
```

###  运行
方式 1：启动本地 Gradio 服务
```
 python launch.py
```
方式 2：使用命令行
```
python run.py --in_file test1.m4a
```

## 支持

+ 如果这个项目对你有帮助，请给它一个Star吧！🌟 <a href='https://github.com/huiofficial/FunnyASR'><img src='https://img.shields.io/github/stars/huiofficial/FunnyASR?style=social&label=Star'></a>

如果您有任何问题，欢迎建立 issue 和 pull request，也可以通过我的[邮箱](mailto:tzattack@outlook.com) 与我取得联系。 

如果您觉得我的教程对您有所帮助，且您力所能及的情况下，欢迎扫描下方二维码给我支持  

![付款码](img/pay.png)

+ 超过 10 元可以备注私人微信，我会加您。  
+ 超过 1000 元我会邀请您在上海、扬州共进晚餐。  
+ 超过 10000 元我会飞往您所在的城市，与您共进晚餐。