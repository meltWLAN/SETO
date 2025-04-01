import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto_versal.market.state import MarketState
from gui_dashboard import DashboardGUI

# 配置日志
logging.basicConfig(level=logging.INFO)

# 创建测试配置
config = {
    'mode': 'paper',  # 使用模拟交易模式
    'symbols': ['000001.SZ'],  # 测试用股票代码
    'data_dir': 'data/market'  # 数据目录
}

def main():
    try:
        # 创建市场状态实例
        market = MarketState(config)
        
        # 创建并运行仪表板
        dashboard = DashboardGUI(market)
        dashboard.run()
        
    except Exception as e:
        logging.error(f"运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 