# 执行模块依赖

本目录包含SETO-Versal交易系统执行模块的各种依赖需求文件。

## 依赖文件列表

- `binance.txt` - 币安交易所适配器所需的依赖
- `test.txt` - 执行模块测试需要的依赖

## 安装依赖

可以使用以下命令安装特定的依赖：

```bash
# 安装币安交易所适配器依赖
pip install -r seto_versal/execution/requirements/binance.txt

# 安装测试依赖
pip install -r seto_versal/execution/requirements/test.txt
```

也可以在项目根目录的requirements.txt中包含对这些文件的引用：

```
# requirements.txt
-r seto_versal/execution/requirements/binance.txt
``` 