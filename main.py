#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票预测系统主程序

这是一个基于深度学习的股票价格预测系统，集成了Tushare金融数据平台，
使用LSTM模型进行智能股票价格预测。系统支持模型训练、单股预测、批量预测、
数据可视化和Web界面等功能。

主要功能：
- 模型训练：支持基线LSTM和K线LSTM模型训练
- 单股预测：预测指定股票的未来价格
- 批量预测：同时预测多只股票的价格
- 数据获取：获取并保存股票历史数据
- 系统帮助：查看系统使用说明

作者：AI助手
版本：v1.4
创建时间：2024-11-14
"""

import os
import torch
from stock_data import get_device, save_stock_data
from train_stock_model import train_stock_model
from predict_stock import main_predict, batch_predict_stocks, get_default_model_path
import pandas as pd

# 导入配置文件
from config import DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_SEQUENCE_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_MODEL_TYPE

def clear_screen():
    """清屏函数
    
    清空终端屏幕，提供更好的用户体验。
    支持Windows (cls) 和Unix/Linux/Mac (clear) 系统。
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def display_menu():
    """显示主菜单
    
    显示系统的主要功能选项，包括：
    - 模型训练
    - 单股预测
    - 批量预测
    - 数据获取
    - 系统帮助
    - 退出系统
    
    用户通过输入数字选择对应的功能。
    """
    print("=" * 50)
    print("      股票预测系统 v1.4")
    print("=" * 50)
    print("1. 训练股票预测模型")
    print("2. 预测单只股票价格")
    print("3. 批量预测多只股票价格")
    print("4. 获取并保存股票历史数据")
    print("5. 查看系统帮助")
    print("0. 退出系统")
    print("=" * 50)

def train_model_menu():
    """模型训练菜单"""
    clear_screen()
    print("=" * 50)
    print("      训练股票预测模型")
    print("=" * 50)
    
    # 获取用户输入的股票代码、日期范围和参数
    stock_code = input("请输入股票代码(例如: 000001.SZ): ").strip()
    start_date = input(f"请输入开始日期(格式: YYYYMMDD，默认为{DEFAULT_START_DATE}): ").strip() or DEFAULT_START_DATE
    end_date = input(f"请输入结束日期(格式: YYYYMMDD，默认为{DEFAULT_END_DATE}): ").strip() or DEFAULT_END_DATE
    
    # 模型类型选择
    print("\n请选择模型类型:")
    print("1. LSTM模型(推荐)")
    print("2. 基线LSTM模型")
    print("3. GRU模型")
    model_choice = input("请输入选择(默认为1): ").strip()
    
    model_type_map = {
        '1': 'lstm',
        '2': 'baseline',
        '3': 'gru'
    }
    model_type = model_type_map.get(model_choice, DEFAULT_MODEL_TYPE)
    
    # 可选参数设置
    try:
        hidden_size = int(input(f"请输入隐藏层大小(默认为{DEFAULT_HIDDEN_SIZE}): ").strip() or str(DEFAULT_HIDDEN_SIZE))
        num_layers = int(input(f"请输入LSTM层数(默认为{DEFAULT_NUM_LAYERS}): ").strip() or str(DEFAULT_NUM_LAYERS))
        sequence_length = int(input(f"请输入序列长度(默认为{DEFAULT_SEQUENCE_LENGTH}): ").strip() or str(DEFAULT_SEQUENCE_LENGTH))
        batch_size = int(input(f"请输入批次大小(默认为{DEFAULT_BATCH_SIZE}): ").strip() or str(DEFAULT_BATCH_SIZE))
        learning_rate = float(input(f"请输入学习率(默认为{DEFAULT_LEARNING_RATE}): ").strip() or str(DEFAULT_LEARNING_RATE))
        epochs = int(input(f"请输入训练轮数(默认为{DEFAULT_EPOCHS}): ").strip() or str(DEFAULT_EPOCHS))
        patience = int(input(f"请输入早停耐心值(默认为{DEFAULT_PATIENCE}): ").strip() or str(DEFAULT_PATIENCE))
    except ValueError:
        print("输入参数格式错误，将使用默认参数")
        hidden_size = DEFAULT_HIDDEN_SIZE
        num_layers = DEFAULT_NUM_LAYERS
        sequence_length = DEFAULT_SEQUENCE_LENGTH
        batch_size = DEFAULT_BATCH_SIZE
        learning_rate = DEFAULT_LEARNING_RATE
        epochs = DEFAULT_EPOCHS
        patience = DEFAULT_PATIENCE
    
    print("\n开始训练模型，请稍候...")
    
    try:
        # 调用训练函数
        model_path = train_stock_model(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            sequence_length=sequence_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience
        )
        
        print(f"\n模型训练成功并保存到: {model_path}")
    except Exception as e:
        print(f"\n模型训练失败: {e}")
    
    input("\n按回车键返回主菜单...")

def predict_single_stock_menu():
    """单只股票预测菜单"""
    clear_screen()
    print("=" * 50)
    print("      单只股票价格预测")
    print("=" * 50)
    
    # 获取用户输入的股票代码
    stock_code = input("请输入股票代码(例如: 000001.SZ): ").strip()
    
    # 模型类型选择
    print("\n请选择模型类型:")
    print("1. LSTM模型")
    print("2. 基线LSTM模型")
    print("3. GRU模型")
    model_choice = input("请输入选择(默认为1): ").strip()
    
    model_type_map = {
        '1': 'lstm',
        '2': 'baseline',
        '3': 'gru'
    }
    model_type = model_type_map.get(model_choice, 'lstm')
    
    # 是否使用自定义模型路径
    use_custom_path = input("是否使用自定义模型路径？(y/n，默认为n): ").strip().lower()
    
    model_path = None
    if use_custom_path == 'y':
        model_path = input("请输入模型文件路径: ").strip()
    
    print("\n开始预测，请稍候...")
    
    try:
        # 调用预测函数，正确传递参数
        if model_path:
            # 如果使用自定义路径，将模型路径传递为model_dir参数
            success = main_predict(stock_code, model_type, model_dir=model_path)
        else:
            # 使用默认路径
            success = main_predict(stock_code, model_type)
        
        if not success:
            print("\n预测失败，请检查股票代码或确保模型已训练")
    except Exception as e:
        print(f"\n预测过程中发生错误: {e}")
    
    input("\n按回车键返回主菜单...")

def batch_predict_menu():
    """批量股票预测菜单"""
    clear_screen()
    print("=" * 50)
    print("      批量股票价格预测")
    print("=" * 50)
    
    # 获取用户输入的股票代码列表
    print("请输入股票代码列表，用逗号分隔(例如: 000001.SZ,000002.SZ,000003.SZ): ")
    stock_codes_input = input().strip()
    
    # 解析股票代码列表
    stock_codes = [code.strip() for code in stock_codes_input.split(',') if code.strip()]
    
    if not stock_codes:
        print("未输入有效的股票代码")
        input("\n按回车键返回主菜单...")
        return
    
    # 获取设备
    device = get_device()
    
    # 模型类型选择和路径
    print("\n请选择模型类型:")
    print("1. LSTM模型")
    print("2. 基线LSTM模型")
    print("3. GRU模型")
    model_choice = input("请输入选择(默认为1): ").strip()
    
    model_type_map = {
        '1': 'lstm',
        '2': 'baseline',
        '3': 'gru'
    }
    model_type = model_type_map.get(model_choice, 'lstm')
    
    # 使用第一个股票代码的模型进行批量预测
    # 注意：在实际应用中，可能需要为不同股票使用不同的模型
    reference_stock = stock_codes[0]
    model_path = get_default_model_path(reference_stock, model_type)
    
    print(f"\n使用股票 {reference_stock} 的模型进行批量预测，请稍候...")
    
    try:
        # 调用批量预测函数
        results = batch_predict_stocks(model_path, stock_codes, device)
        
        if results:
            # 导入股票名称获取函数
            from stock_data import get_stock_name
            
            print("\n批量预测结果:")
            print("=" * 90)
            print(f"{'股票名称':<16}{'股票代码':<12}{'最新收盘价':<12}{'预测收盘价':<12}{'预测涨跌幅(%)':<12}")
            print("=" * 90)
            
            # 显示预测结果
            for result in results:
                stock_code = result['股票代码']
                stock_name = get_stock_name(stock_code)
                # 限制股票名称长度，避免表格变形
                stock_name = (stock_name[:14] + '..') if len(stock_name) > 14 else stock_name
                print(f"{stock_name:<16}{stock_code:<12}{result['最新收盘价']:<12.2f}{result['预测收盘价']:<12.2f}{result['预测涨跌幅(%)']:<12.2f}")
        
            print("=" * 90)
            
            # 询问是否保存结果
            save_results = input("\n是否保存预测结果到CSV文件？(y/n，默认为n): ").strip().lower()
            
            if save_results == 'y':
                # 创建结果保存目录
                result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
                os.makedirs(result_dir, exist_ok=True)
                
                # 生成文件名
                import datetime
                current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = os.path.join(result_dir, f'batch_prediction_{current_time}.csv')
                
                # 保存到CSV
                df = pd.DataFrame(results)
                df.to_csv(result_file, index=False, encoding='utf-8-sig')
                
                print(f"预测结果已保存到: {result_file}")
        else:
            print("\n未能获取有效的预测结果")
    except Exception as e:
        print(f"\n批量预测过程中发生错误: {e}")
    
    input("\n按回车键返回主菜单...")

def fetch_stock_data_menu():
    """获取股票历史数据菜单"""
    clear_screen()
    print("=" * 50)
    print("      获取股票历史数据")
    print("=" * 50)
    
    # 获取用户输入的股票代码和日期范围
    stock_code = input("请输入股票代码(例如: 000001.SZ): ").strip()
    start_date = input(f"请输入开始日期(格式: YYYYMMDD，默认为{DEFAULT_START_DATE}): ").strip() or DEFAULT_START_DATE
    end_date = input(f"请输入结束日期(格式: YYYYMMDD，默认为{DEFAULT_END_DATE}): ").strip() or DEFAULT_END_DATE
    
    # 询问保存目录
    save_dir = input("请输入保存目录(默认为data): ").strip() or "data"
    
    print("\n开始获取股票数据，请稍候...")
    
    try:
        # 调用数据获取函数
        success = save_stock_data(stock_code, start_date, end_date, save_dir)
        
        if success:
            print("\n股票数据获取成功")
        else:
            print("\n股票数据获取失败")
    except Exception as e:
        print(f"\n获取股票数据过程中发生错误: {e}")
    
    input("\n按回车键返回主菜单...")

def show_help():
    """显示系统帮助信息"""
    clear_screen()
    print("=" * 50)
    print("      股票预测系统帮助")
    print("=" * 50)
    print("1. 系统介绍")
    print("   本系统使用LSTM/GRU深度学习模型进行股票价格预测，基于Tushare金融数据平台提供的数据。")
    print("\n2. 使用说明")
    print("   - 训练模型：首先需要训练模型，输入股票代码和时间范围。")
    print("   - 预测股票：训练完成后，可以进行单只或多只股票的价格预测。")
    print("   - 获取数据：可以单独获取股票历史数据并保存到本地。")
    print("\n3. 股票代码格式")
    print("   - 深圳证券交易所：000001.SZ")
    print("   - 上海证券交易所：600000.SH")
    print("\n4. 注意事项")
    print("   - 股票预测仅供参考，不构成投资建议。")
    print("   - 首次使用需要确保Tushare API配置正确。")
    print("   - 训练模型需要一定的时间，取决于数据量和计算资源。")
    print("=" * 50)
    
    input("\n按回车键返回主菜单...")

def main():
    """主函数"""
    while True:
        clear_screen()
        display_menu()
        
        # 获取用户选择
        choice = input("请输入您的选择(0-5): ").strip()
        
        # 根据用户选择执行相应功能
        if choice == '0':
            print("感谢使用股票预测系统，再见！")
            break
        elif choice == '1':
            train_model_menu()
        elif choice == '2':
            predict_single_stock_menu()
        elif choice == '3':
            batch_predict_menu()
        elif choice == '4':
            fetch_stock_data_menu()
        elif choice == '5':
            show_help()
        else:
            print("无效的选择，请重新输入！")
            input("按回车键继续...")

if __name__ == "__main__":
    main()