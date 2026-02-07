// 数据可视化功能模块

// 全局图表实例存储
const chartInstances = {};

/**
 * 初始化图表
 * @param {string} chartId - 图表容器的ID
 * @param {string} chartType - 图表类型
 * @param {Object} data - 图表数据
 * @param {Object} options - 图表配置选项
 * @returns {Chart} - Chart.js图表实例
 */
function initChart(chartId, chartType, data, options = {}) {
    const ctx = document.getElementById(chartId).getContext('2d');
    
    // 如果已有该ID的图表实例，先销毁
    if (chartInstances[chartId]) {
        chartInstances[chartId].destroy();
    }
    
    // 默认配置选项
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true,
                    boxWidth: 6
                }
            },
            tooltip: {
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 10,
                cornerRadius: 4,
                titleFont: {
                    size: 14
                },
                bodyFont: {
                    size: 13
                }
            }
        },
        scales: {
            x: {
                display: true,
                grid: {
                    display: false
                },
                ticks: {
                    maxRotation: 45,
                    minRotation: 45
                }
            },
            y: {
                display: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.05)'
                }
            }
        },
        animation: {
            duration: 1000,
            easing: 'easeOutQuart'
        }
    };
    
    // 合并用户选项和默认选项
    const mergedOptions = { ...defaultOptions, ...options };
    
    // 创建图表实例
    const chartInstance = new Chart(ctx, {
        type: chartType,
        data: data,
        options: mergedOptions
    });
    
    // 存储图表实例
    chartInstances[chartId] = chartInstance;
    
    return chartInstance;
}

/**
 * 更新现有图表
 * @param {string} chartId - 图表容器的ID
 * @param {Object} newData - 新的图表数据
 */
function updateChart(chartId, newData) {
    const chart = chartInstances[chartId];
    if (chart) {
        chart.data = newData;
        chart.update();
    }
}

/**
 * 创建股票价格走势图
 * @param {string} chartId - 图表容器的ID
 * @param {Array} dates - 日期数组
 * @param {Array} prices - 价格数组
 * @param {Object} options - 可选配置
 * @param {string} stockCode - 股票代码（可选）
 * @param {string} stockName - 股票名称（可选）
 */
function createStockPriceChart(chartId, dates, prices, options = {}, stockCode = '', stockName = '') {
    // 根据是否有股票名称设置标签和标题
    const chartLabel = stockCode ? (stockName ? `${stockName}(${stockCode}) 收盘价` : `${stockCode} 收盘价`) : '收盘价';
    
    const data = {
        labels: dates,
        datasets: [
            {
                label: chartLabel,
                data: prices.close_prices,
                borderColor: '#165DFF',
                backgroundColor: 'rgba(22, 93, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.2,
                pointRadius: 2,
                pointHoverRadius: 4
            },
            {
                label: '5日均线',
                data: prices.ma5 || [],
                borderColor: '#52C41A',
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                fill: false,
                tension: 0.2,
                pointRadius: 0
            },
            {
                label: '10日均线',
                data: prices.ma10 || [],
                borderColor: '#FAAD14',
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                fill: false,
                tension: 0.2,
                pointRadius: 0
            }
        ].filter(dataset => dataset.data && dataset.data.length > 0) // 过滤掉空数据的数据集
    };
    
    return initChart(chartId, 'line', data, options);
}

/**
 * 创建预测结果图表
 * @param {string} chartId - 图表容器的ID
 * @param {string} stockCode - 股票代码
 * @param {number} latestClose - 最新收盘价
 * @param {number} predictedPrice - 预测价格
 * @param {Object} options - 可选配置
 * @param {string} stockName - 股票名称（可选）
 */
function createPredictionResultChart(chartId, stockCode, latestClose, predictedPrice, options = {}, stockName = '') {
    // 准备数据
    const labels = ['最新价格', '预测价格'];
    const data = [latestClose, predictedPrice];
    
    // 确定颜色
    const colors = predictedPrice > latestClose 
        ? ['#94a3b8', '#52C41A']  // 绿色表示上涨
        : ['#94a3b8', '#F5222D']; // 红色表示下跌
    
    // 根据是否有股票名称设置标签和标题
    const chartLabel = stockName ? `${stockName}(${stockCode})` : stockCode;
    const chartTitle = stockName 
        ? `${stockName}(${stockCode}) 未来五天价格预测结果` 
        : `${stockCode} 未来五天价格预测结果`;
    
    const chartData = {
        labels: labels,
        datasets: [
            {
                label: chartLabel,
                data: data,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 4
            }
        ]
    };
    
    // 自定义配置
    const customOptions = {
        plugins: {
            title: {
                display: true,
                text: chartTitle,
                font: {
                    size: 16
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        return `${context.label}: ${context.raw.toFixed(2)}`;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                ticks: {
                    callback: function(value) {
                        return value.toFixed(2);
                    }
                }
            }
        }
    };
    
    const mergedOptions = { ...options, ...customOptions };
    
    return initChart(chartId, 'bar', chartData, mergedOptions);
}

/**
 * 创建模型训练历史图表
 * @param {string} chartId - 图表容器的ID
 * @param {Array} epochs - 训练轮数数组
 * @param {Object} metrics - 训练指标数据
 * @param {Object} options - 可选配置
 */
function createTrainingHistoryChart(chartId, epochs, metrics, options = {}) {
    const chartData = {
        labels: epochs,
        datasets: [
            {
                label: '训练损失',
                data: metrics.train_loss,
                borderColor: '#165DFF',
                backgroundColor: 'rgba(22, 93, 255, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                pointRadius: 2,
                pointHoverRadius: 4
            },
            {
                label: '验证损失',
                data: metrics.val_loss,
                borderColor: '#F5222D',
                backgroundColor: 'rgba(245, 34, 45, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                pointRadius: 2,
                pointHoverRadius: 4
            }
        ]
    };
    
    // 自定义配置
    const customOptions = {
        plugins: {
            title: {
                display: true,
                text: '模型训练损失曲线',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                ticks: {
                    callback: function(value) {
                        return value.toFixed(4);
                    }
                }
            }
        }
    };
    
    const mergedOptions = { ...options, ...customOptions };
    
    return initChart(chartId, 'line', chartData, mergedOptions);
}

/**
 * 创建批量预测结果对比图表
 * @param {string} chartId - 图表容器的ID
 * @param {Array} stockCodes - 股票代码数组
 * @param {Array} latestPrices - 最新价格数组
 * @param {Array} predictedPrices - 预测价格数组
 * @param {Object} options - 可选配置
 * @param {Array} stockNames - 股票名称数组（可选）
 */
function createBatchPredictionChart(chartId, stockCodes, latestPrices, predictedPrices, options = {}, stockNames = []) {
    // 如果提供了股票名称数组且长度匹配，则使用股票名称和代码的组合作为标签
    let chartLabels = stockCodes;
    if (stockNames && stockNames.length > 0 && stockNames.length === stockCodes.length) {
        chartLabels = stockCodes.map((code, index) => {
            const name = stockNames[index] || '';
            return name ? `${name}(${code})` : code;
        });
    }
    
    const chartData = {
        labels: chartLabels,
        datasets: [
            {
                label: '最新价格',
                data: latestPrices,
                backgroundColor: '#94a3b8',
                borderColor: '#94a3b8',
                borderWidth: 1,
                borderRadius: 4
            },
            {
                label: '预测价格',
                data: predictedPrices,
                backgroundColor: '#165DFF',
                borderColor: '#165DFF',
                borderWidth: 1,
                borderRadius: 4
            }
        ]
    };
    
    // 自定义配置
    const customOptions = {
        plugins: {
            title: {
                display: true,
                text: '批量股票预测结果对比',
                font: {
                    size: 16
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        return `${context.dataset.label}: ${context.raw.toFixed(2)}`;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                ticks: {
                    callback: function(value) {
                        return value.toFixed(2);
                    }
                }
            }
        }
    };
    
    const mergedOptions = { ...options, ...customOptions };
    
    return initChart(chartId, 'bar', chartData, mergedOptions);
}

/**
 * 创建模型性能评估雷达图
 * @param {string} chartId - 图表容器的ID
 * @param {Object} metrics - 评估指标数据
 * @param {Object} options - 可选配置
 */
function createModelPerformanceRadarChart(chartId, metrics, options = {}) {
    // 计算各项指标的最大可能值（用于归一化）
    const maxValues = {
        'MSE': 100,
        'RMSE': 10,
        'MAE': 5,
        'MAPE': 5,
        'R2': 1
    };
    
    // 准备雷达图数据
    const chartData = {
        labels: Object.keys(metrics),
        datasets: [
            {
                label: '模型性能',
                data: Object.keys(metrics).map(key => {
                    // 对于R2，值越大越好；对于其他指标，值越小越好
                    if (key === 'R2') {
                        return metrics[key];
                    } else {
                        // 归一化其他指标（值越小，性能越好）
                        const normalizedValue = 1 - Math.min(metrics[key] / maxValues[key], 1);
                        return normalizedValue;
                    }
                }),
                backgroundColor: 'rgba(22, 93, 255, 0.2)',
                borderColor: '#165DFF',
                borderWidth: 2,
                pointBackgroundColor: '#165DFF',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#165DFF'
            }
        ]
    };
    
    // 自定义配置
    const customOptions = {
        plugins: {
            title: {
                display: true,
                text: '模型性能评估',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            r: {
                angleLines: {
                    display: true
                },
                suggestedMin: 0,
                suggestedMax: 1
            }
        }
    };
    
    const mergedOptions = { ...options, ...customOptions };
    
    return initChart(chartId, 'radar', chartData, mergedOptions);
}

/**
 * 格式化时间戳为可读日期
 * @param {number} timestamp - 时间戳
 * @returns {string} - 格式化后的日期字符串
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

/**
 * 格式化大数字显示
 * @param {number} num - 要格式化的数字
 * @param {number} decimals - 小数位数
 * @returns {string} - 格式化后的数字字符串
 */
function formatNumber(num, decimals = 2) {
    if (isNaN(num)) return '0.00';
    
    // 处理大数字显示
    if (Math.abs(num) >= 1000000000) {
        return (num / 1000000000).toFixed(decimals) + 'B';
    } else if (Math.abs(num) >= 1000000) {
        return (num / 1000000).toFixed(decimals) + 'M';
    } else if (Math.abs(num) >= 1000) {
        return (num / 1000).toFixed(decimals) + 'K';
    } else {
        return num.toFixed(decimals);
    }
}

/**
 * 创建简单的工具提示
 * @param {HTMLElement} element - 要添加工具提示的元素
 * @param {string} text - 工具提示文本
 */
function createTooltip(element, text) {
    // 检查是否已存在工具提示
    if (element.getAttribute('data-tooltip')) return;
    
    element.setAttribute('data-tooltip', text);
    element.classList.add('tooltip-trigger');
    
    // 创建工具提示元素
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = text;
    tooltip.style.cssText = `
        position: absolute;
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s;
        z-index: 1000;
        white-space: nowrap;
    `;
    document.body.appendChild(tooltip);
    
    // 添加事件监听
    element.addEventListener('mouseenter', function(e) {
        const rect = element.getBoundingClientRect();
        tooltip.style.top = `${rect.bottom + window.pageYOffset + 5}px`;
        tooltip.style.left = `${rect.left + window.pageXOffset + rect.width / 2}px`;
        tooltip.style.transform = 'translateX(-50%)';
        tooltip.style.opacity = '1';
    });
    
    element.addEventListener('mouseleave', function() {
        tooltip.style.opacity = '0';
    });
    
    // 清理函数
    return function cleanup() {
        element.removeAttribute('data-tooltip');
        element.classList.remove('tooltip-trigger');
        element.removeEventListener('mouseenter', tooltip.mouseenterHandler);
        element.removeEventListener('mouseleave', tooltip.mouseleaveHandler);
        if (tooltip.parentNode === document.body) {
            document.body.removeChild(tooltip);
        }
    };
}

/**
 * 为元素添加淡入动画效果
 * @param {HTMLElement} element - 要添加动画的元素
 * @param {number} duration - 动画持续时间（毫秒）
 */
function fadeInElement(element, duration = 500) {
    element.style.opacity = '0';
    element.style.transform = 'translateY(10px)';
    element.style.transition = `opacity ${duration}ms ease-out, transform ${duration}ms ease-out`;
    
    // 触发重排
    element.offsetHeight;
    
    element.style.opacity = '1';
    element.style.transform = 'translateY(0)';
}

/**
 * 为元素添加滑入动画效果
 * @param {HTMLElement} element - 要添加动画的元素
 * @param {string} direction - 滑入方向（left, right, top, bottom）
 * @param {number} duration - 动画持续时间（毫秒）
 */
function slideInElement(element, direction = 'left', duration = 500) {
    const transformValues = {
        left: 'translateX(-20px)',
        right: 'translateX(20px)',
        top: 'translateY(-20px)',
        bottom: 'translateY(20px)'
    };
    
    element.style.opacity = '0';
    element.style.transform = transformValues[direction] || 'translateX(-20px)';
    element.style.transition = `opacity ${duration}ms ease-out, transform ${duration}ms ease-out`;
    
    // 触发重排
    element.offsetHeight;
    
    element.style.opacity = '1';
    element.style.transform = 'translate(0)';
}

/**
 * 平滑滚动到指定元素
 * @param {HTMLElement|string} target - 目标元素或元素ID
 * @param {Object} options - 滚动选项
 */
function smoothScrollTo(target, options = {}) {
    const defaultOptions = {
        behavior: 'smooth',
        block: 'start',
        inline: 'nearest'
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    let targetElement;
    if (typeof target === 'string') {
        targetElement = document.getElementById(target);
    } else if (target instanceof HTMLElement) {
        targetElement = target;
    }
    
    if (targetElement) {
        targetElement.scrollIntoView(mergedOptions);
    }
}

/**
 * 导出CSV文件
 * @param {Array} data - 要导出的数据数组
 * @param {string} filename - 文件名
 */
function exportToCSV(data, filename = 'export.csv') {
    if (!data || data.length === 0) return;
    
    // 获取表头（假设第一行是完整的）
    const headers = Object.keys(data[0]);
    
    // 创建CSV内容
    const csvContent = [
        headers.join(','), // 表头
        ...data.map(row => 
            headers.map(header => {
                const value = row[header];
                // 处理包含逗号或引号的值
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            }).join(',')
        )
    ].join('\n');
    
    // 创建Blob对象
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    
    // 创建下载链接
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // 释放URL对象
    URL.revokeObjectURL(url);
}

/**
 * 检查元素是否在视口中
 * @param {HTMLElement} element - 要检查的元素
 * @returns {boolean} - 是否在视口中
 */
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// 导出函数
window.DataVisualization = {
    initChart,
    updateChart,
    createStockPriceChart,
    createPredictionResultChart,
    createTrainingHistoryChart,
    createBatchPredictionChart,
    createModelPerformanceRadarChart,
    formatTimestamp,
    formatNumber,
    createTooltip,
    fadeInElement,
    slideInElement,
    smoothScrollTo,
    exportToCSV,
    isInViewport
};