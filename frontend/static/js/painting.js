/**
 * 图像绘画功能JavaScript代码
 * 实现画布绘画、工具管理、撤销重做、保存导出等功能
 */

class PaintingApp {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.isDrawing = false;
        this.currentTool = 'pen';
        this.currentColor = '#000000';
        this.currentSize = 5;
        this.currentOpacity = 1;
        this.history = [];
        this.historyStep = -1;
        this.startX = 0;
        this.startY = 0;
        this.textInputActive = false;
        this.paintings = [];
        
        this.init();
    }

    init() {
        this.setupCanvas();
        this.setupEventListeners();
        this.setupTools();
        this.loadPaintings();
    }

    setupCanvas() {
        this.canvas = document.getElementById('painting-canvas');
        if (!this.canvas) return;
        
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 800;
        this.canvas.height = 600;
        
        // 设置画布样式
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        // 保存初始状态
        this.saveState();
    }

    setupEventListeners() {
        // 画布鼠标事件
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseUp(e));
        
        // 工具栏事件
        document.getElementById('brush-color').addEventListener('change', (e) => {
            this.currentColor = e.target.value;
        });
        
        document.getElementById('brush-size').addEventListener('input', (e) => {
            this.currentSize = e.target.value;
            document.getElementById('brush-size-value').textContent = e.target.value + 'px';
        });
        
        document.getElementById('brush-opacity').addEventListener('input', (e) => {
            this.currentOpacity = e.target.value;
            document.getElementById('brush-opacity-value').textContent = Math.round(e.target.value * 100) + '%';
        });
        
        document.getElementById('canvas-size').addEventListener('change', (e) => {
            this.handleCanvasSizeChange(e.target.value);
        });
        
        document.getElementById('new-canvas-btn').addEventListener('click', () => {
            this.newCanvas();
        });
        
        document.getElementById('clear-canvas-btn').addEventListener('click', () => {
            this.clearCanvas();
        });
        
        document.getElementById('undo-btn').addEventListener('click', () => {
            this.undo();
        });
        
        document.getElementById('redo-btn').addEventListener('click', () => {
            this.redo();
        });
        
        document.getElementById('save-painting-btn').addEventListener('click', () => {
            this.savePainting();
        });
        
        document.getElementById('export-png-btn').addEventListener('click', () => {
            this.exportPNG();
        });
        
        document.getElementById('add-text-btn').addEventListener('click', () => {
            this.addText();
        });
    }

    setupTools() {
        // 工具按钮事件
        const toolButtons = document.querySelectorAll('.tool-btn');
        toolButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectTool(e.target.id.replace('-tool-btn', ''));
            });
        });
    }

    selectTool(tool) {
        this.currentTool = tool;
        
        // 更新按钮状态
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.classList.remove('active', 'bg-primary', 'text-white');
            btn.classList.add('bg-gray-500', 'text-white');
        });
        
        const activeBtn = document.getElementById(`${tool}-tool-btn`);
        if (activeBtn) {
            activeBtn.classList.remove('bg-gray-500', 'text-white');
            activeBtn.classList.add('active', 'bg-primary', 'text-white');
        }
        
        // 显示/隐藏文字输入
        const textContainer = document.getElementById('text-input-container');
        if (tool === 'text') {
            textContainer.classList.remove('hidden');
            this.textInputActive = true;
        } else {
            textContainer.classList.add('hidden');
            this.textInputActive = false;
        }
        
        // 设置画笔样式
        this.updateBrushStyle();
    }

    updateBrushStyle() {
        this.ctx.globalCompositeOperation = this.currentTool === 'eraser' ? 'destination-out' : 'source-over';
        this.ctx.strokeStyle = this.currentColor;
        this.ctx.globalAlpha = this.currentOpacity;
        this.ctx.lineWidth = this.currentSize;
    }

    handleMouseDown(e) {
        if (this.textInputActive) return;
        
        const rect = this.canvas.getBoundingClientRect();
        this.startX = e.clientX - rect.left;
        this.startY = e.clientY - rect.top;
        
        if (this.currentTool === 'pen' || this.currentTool === 'eraser') {
            this.isDrawing = true;
            this.updateBrushStyle();
            this.ctx.beginPath();
            this.ctx.moveTo(this.startX, this.startY);
        } else if (['line', 'rect', 'circle'].includes(this.currentTool)) {
            this.tempCanvas = document.createElement('canvas');
            this.tempCanvas.width = this.canvas.width;
            this.tempCanvas.height = this.canvas.height;
            this.tempCtx = this.tempCanvas.getContext('2d');
            this.tempCtx.drawImage(this.canvas, 0, 0);
        }
    }

    handleMouseMove(e) {
        if (!this.isDrawing && !['line', 'rect', 'circle'].includes(this.currentTool)) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.tempCanvas, 0, 0);
        
        this.updateBrushStyle();
        
        if (this.isDrawing) {
            // 自由绘画
            this.ctx.lineTo(currentX, currentY);
            this.ctx.stroke();
        } else if (this.currentTool === 'line') {
            // 绘制直线
            this.ctx.beginPath();
            this.ctx.moveTo(this.startX, this.startY);
            this.ctx.lineTo(currentX, currentY);
            this.ctx.stroke();
        } else if (this.currentTool === 'rect') {
            // 绘制矩形
            const width = currentX - this.startX;
            const height = currentY - this.startY;
            this.ctx.strokeRect(this.startX, this.startY, width, height);
        } else if (this.currentTool === 'circle') {
            // 绘制圆形
            const radius = Math.sqrt(Math.pow(currentX - this.startX, 2) + Math.pow(currentY - this.startY, 2));
            this.ctx.beginPath();
            this.ctx.arc(this.startX, this.startY, radius, 0, 2 * Math.PI);
            this.ctx.stroke();
        }
    }

    handleMouseUp(e) {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.ctx.closePath();
            this.saveState();
        } else if (this.tempCanvas) {
            this.saveState();
            this.tempCanvas = null;
            this.tempCtx = null;
        }
    }

    saveState() {
        this.historyStep++;
        if (this.historyStep < this.history.length) {
            this.history.length = this.historyStep;
        }
        this.history.push(this.canvas.toDataURL());
        
        // 限制历史记录数量
        if (this.history.length > 50) {
            this.history.shift();
            this.historyStep--;
        }
        
        this.updateUndoRedoButtons();
    }

    undo() {
        if (this.historyStep > 0) {
            this.historyStep--;
            this.loadState(this.history[this.historyStep]);
            this.updateUndoRedoButtons();
        }
    }

    redo() {
        if (this.historyStep < this.history.length - 1) {
            this.historyStep++;
            this.loadState(this.history[this.historyStep]);
            this.updateUndoRedoButtons();
        }
    }

    loadState(state) {
        const img = new Image();
        img.onload = () => {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(img, 0, 0);
        };
        img.src = state;
    }

    updateUndoRedoButtons() {
        const undoBtn = document.getElementById('undo-btn');
        const redoBtn = document.getElementById('redo-btn');
        
        undoBtn.disabled = this.historyStep <= 0;
        redoBtn.disabled = this.historyStep >= this.history.length - 1;
    }

    clearCanvas() {
        if (confirm('确定要清空画布吗？此操作不可撤销。')) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.saveState();
        }
    }

    newCanvas() {
        if (confirm('确定要创建新画布吗？当前内容将被清空。')) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.history = [];
            this.historyStep = -1;
            this.saveState();
        }
    }

    handleCanvasSizeChange(size) {
        const customContainer = document.getElementById('custom-size-container');
        if (size === 'custom') {
            customContainer.classList.remove('hidden');
        } else {
            customContainer.classList.add('hidden');
            const [width, height] = size.split('x').map(Number);
            this.resizeCanvas(width, height);
        }
    }

    resizeCanvas(width, height) {
        // 保存当前内容
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(this.canvas, 0, 0);
        
        // 调整画布大小
        this.canvas.width = width;
        this.canvas.height = height;
        this.ctx = this.canvas.getContext('2d');
        
        // 恢复内容
        this.ctx.drawImage(tempCanvas, 0, 0);
        
        // 更新样式
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        this.saveState();
    }

    addText() {
        const textContent = document.getElementById('text-content').value;
        const textSize = document.getElementById('text-size').value;
        
        if (!textContent.trim()) {
            alert('请输入文字内容');
            return;
        }
        
        this.ctx.font = `${textSize}px Arial`;
        this.ctx.fillStyle = this.currentColor;
        this.ctx.globalAlpha = this.currentOpacity;
        
        // 在画布中心位置添加文字
        const x = this.canvas.width / 2;
        const y = this.canvas.height / 2;
        
        this.ctx.textAlign = 'center';
        this.ctx.fillText(textContent, x, y);
        
        this.saveState();
        document.getElementById('text-content').value = '';
    }

    async savePainting() {
        try {
            const canvasData = this.canvas.toDataURL('image/png');
            
            const response = await fetch('/api/painting/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    canvas_data: canvasData,
                    timestamp: Date.now()
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                alert('画作保存成功！');
                this.loadPaintings();
            } else {
                alert('保存失败：' + result.message);
            }
        } catch (error) {
            console.error('保存画作时出错：', error);
            alert('保存失败，请稍后重试');
        }
    }

    async exportPNG() {
        try {
            const canvasData = this.canvas.toDataURL('image/png');
            
            const response = await fetch('/api/painting/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    canvas_data: canvasData,
                    filename: `painting_${Date.now()}.png`
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // 触发下载
                const link = document.createElement('a');
                link.href = result.download_url;
                link.download = result.filename;
                link.click();
            } else {
                alert('导出失败：' + result.message);
            }
        } catch (error) {
            console.error('导出画作时出错：', error);
            alert('导出失败，请稍后重试');
        }
    }

    async loadPaintings() {
        try {
            const response = await fetch('/api/painting/list');
            const result = await response.json();
            
            if (result.success) {
                this.paintings = result.paintings;
                this.displayPaintings();
            }
        } catch (error) {
            console.error('加载画作列表时出错：', error);
        }
    }

    displayPaintings() {
        const container = document.getElementById('paintings-list');
        container.innerHTML = '';
        
        this.paintings.forEach((painting, index) => {
            const item = document.createElement('div');
            item.className = 'bg-gray-50 rounded-lg p-3 border border-gray-200 hover:shadow-md transition-shadow duration-300';
            
            item.innerHTML = `
                <img src="${painting.thumbnail}" alt="画作 ${index + 1}" class="w-full h-32 object-cover rounded mb-2">
                <div class="text-sm text-gray-600 mb-2">${new Date(painting.timestamp).toLocaleString()}</div>
                <div class="flex gap-2">
                    <button onclick="paintingApp.loadPainting('${painting.id}')" 
                            class="flex-1 bg-primary hover:bg-primary/90 text-white px-2 py-1 rounded text-xs">
                        加载
                    </button>
                    <button onclick="paintingApp.deletePainting('${painting.id}')" 
                            class="flex-1 bg-red-500 hover:bg-red-600 text-white px-2 py-1 rounded text-xs">
                        删除
                    </button>
                </div>
            `;
            
            container.appendChild(item);
        });
    }

    async loadPainting(id) {
        try {
            const response = await fetch(`/api/painting/load/${id}`);
            const result = await response.json();
            
            if (result.success) {
                const img = new Image();
                img.onload = () => {
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    this.ctx.drawImage(img, 0, 0);
                    this.saveState();
                };
                img.src = result.canvas_data;
            } else {
                alert('加载失败：' + result.message);
            }
        } catch (error) {
            console.error('加载画作时出错：', error);
            alert('加载失败，请稍后重试');
        }
    }

    async deletePainting(id) {
        if (!confirm('确定要删除这幅画作吗？')) return;
        
        try {
            const response = await fetch(`/api/painting/delete/${id}`, {
                method: 'DELETE'
            });
            const result = await response.json();
            
            if (result.success) {
                alert('删除成功！');
                this.loadPaintings();
            } else {
                alert('删除失败：' + result.message);
            }
        } catch (error) {
            console.error('删除画作时出错：', error);
            alert('删除失败，请稍后重试');
        }
    }
}

// 初始化绘画应用
let paintingApp;

document.addEventListener('DOMContentLoaded', function() {
    // 检查绘画section是否存在
    if (document.getElementById('painting-canvas')) {
        paintingApp = new PaintingApp();
    }
});

// 导出绘画应用类
window.PaintingApp = PaintingApp;