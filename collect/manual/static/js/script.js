// 全局变量
let screenshotImg = null;
let isInteracting = false;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartTime = 0;
let isCollecting = false;
let currentTaskDescription = ''; // 当前任务描述
let currentAppName = ''; // 当前应用名称
let currentTaskType = ''; // 当前任务类型
let currentElements = []; // 当前页面的UI元素信息
let hoveredElement = null; // 当前悬停的元素
let elementOverlay = null; // 元素高亮覆盖层

let autoRefreshEnabled = false; // 是否启用自动刷新

// 鼠标位置追踪
let lastMousePosition = { x: 0, y: 0 }; // 记录最后的鼠标位置

async function startDataCollection() {
    // 显示应用信息输入弹窗
    showAppInfoModal();
}

async function endDataCollection() {
    const startBtn = document.getElementById('startBtn');
    const endBtn = document.getElementById('endBtn');
    const nextBtn = document.getElementById('nextBtn');
    const deleteBtn = document.getElementById('deleteBtn');
    const inputBtn = document.getElementById('inputBtn');
    const historyBtn = document.getElementById('historyBtn');
    const autoRefreshBtn = document.getElementById('autoRefreshBtn');
    const collectionInfo = document.getElementById('collectionInfo');
    const tagBtn = document.getElementById('tagBtn'); 

    try {
        // 停止自动刷新
        if (autoRefreshEnabled) {
            stopAutoRefresh();
        }

        await saveCurrentData();

        // 更新UI状态
        startBtn.disabled = false;
        endBtn.disabled = true;
        nextBtn.disabled = true;
        deleteBtn.disabled = true;
        inputBtn.disabled = true;
        historyBtn.disabled = true;
        autoRefreshBtn.disabled = true;      
        isCollecting = false;
        tagBtn.disabled = true;  
        // 隐藏自动刷新状态
        const statusPanel = document.getElementById('autoRefreshStatus');
        statusPanel.style.display = 'none';
        autoRefreshBtn.textContent = '⏰ 自动刷新';

        // 更新状态显示
        const statusDiv = document.querySelector('.collection-status');
        statusDiv.classList.remove('collecting');
        collectionInfo.innerHTML = `✅ 数据收集已结束`;

        // 隐藏操作提示
        const hint = document.getElementById('actionHint');
        if (hint) {
            hint.style.display = 'none';
        }

        updateStatus(`数据收集已结束，自动刷新已关闭`, 'success');

    } catch (error) {
        updateStatus(`结束收集失败: ${error.message}`, 'error');
    }
}

async function nextDataCollection() {
    try {
        // 保存当前数据
        await saveCurrentData();

        // 显示应用信息输入弹窗，为下一条数据输入新的应用信息和任务描述
        showTaskDescriptionModal(true);

    } catch (error) {
        updateStatus(`切换到下一条数据失败: ${error.message}`, 'error');
    }
}

async function deleteDataCollection() {
    try {
        // 删除当前数据
        await deleteCurrentData();

        // 显示任务描述输入弹窗，为下一条数据输入新的任务描述
        showTaskDescriptionModal(true); // 传入true表示这是删除后的下一条数据

    } catch (error) {
        updateStatus(`删除数据失败: ${error.message}`, 'error');
    }
}

async function takeScreenshot() {
    const status = document.getElementById('status');
    const container = document.getElementById('screenshotContainer');

    // 显示加载状态
    status.innerHTML = '<div class="loading">正在获取截图，请稍候...</div>';
    container.innerHTML = '<div class="loading">截图中...</div>';

    try {
        // 直接调用获取截图的API，该API会自动更新截图
        const response = await fetch('/screenshot');

        if (response.ok) {
            const result = await response.json();
            status.innerHTML = '<div class="success">截图成功！可以点击或滑动进行操作</div>';

            // 显示截图并添加事件监听
            container.innerHTML = `
                <img id="screenshotImage" 
                     alt="设备截图" 
                     class="screenshot-img"
                     onerror="this.parentElement.innerHTML='<div class=error>截图加载失败</div>'">
            `;

            // 获取截图元素引用
            screenshotImg = document.getElementById('screenshotImage');

            // 直接设置截图数据
            if (result.image_data) {
                screenshotImg.src = result.image_data;

                // 存储层次结构信息供后续使用
                window.currentHierarchy = result.hierarchy;

                // 解析并保存所有UI元素信息
                if (result.hierarchy) {
                    currentElements = parseUIElements(result.hierarchy);
                    const clickableElements = currentElements.filter(el => el.clickable);
                    console.log(`UI元素信息已加载: ${currentElements.length} 个元素 (其中 ${clickableElements.length} 个可点击)`);
                }
            }

            // 显示操作提示
            const hint = document.getElementById('actionHint');
            if (hint) {
                hint.style.display = 'block';
            }

            // 为截图添加交互功能
            setupScreenshotInteraction();
        } else {
            const error = await response.json();
            throw new Error(error.detail || '截图失败');
        }
    } catch (error) {
        status.innerHTML = `<div class="error">错误: ${error.message}</div>`;
        container.innerHTML = '<div class="error">截图失败，请重试</div>';
    }
}

function setupScreenshotInteraction() {
    screenshotImg = document.getElementById('screenshotImage');
    if (!screenshotImg) {
        console.error('找不到截图元素');
        return;
    }

    console.log('设置截图交互功能...');

    // 确保清除之前的状态
    clearElementHighlight();
    hoveredElement = null;

    // 添加鼠标事件处理
    screenshotImg.addEventListener('mousedown', handleMouseDown);
    screenshotImg.addEventListener('mousemove', handleMouseMove);
    screenshotImg.addEventListener('mouseup', handleMouseUp);
    screenshotImg.addEventListener('mouseleave', handleMouseUp); // 鼠标离开时也要结束拖拽

    // 添加元素高亮的鼠标移动处理
    screenshotImg.addEventListener('mousemove', handleScreenshotMouseMove);
    screenshotImg.addEventListener('mouseleave', () => {
        clearElementHighlight();
        lastMousePosition = { x: -1, y: -1 }; // 重置鼠标位置
    });

    // 禁用图片的默认拖拽行为
    screenshotImg.addEventListener('dragstart', (e) => e.preventDefault());

    // 添加触摸事件支持
    screenshotImg.addEventListener('touchstart', handleTouchStart);
    screenshotImg.addEventListener('touchmove', handleTouchMove);
    screenshotImg.addEventListener('touchend', handleTouchEnd);

    // 禁用右键菜单
    screenshotImg.addEventListener('contextmenu', (e) => e.preventDefault());

    console.log('截图交互功能设置完成');
}

function handleMouseDown(event) {
    if (isInteracting) return;

    isDragging = true;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    dragStartTime = Date.now();

    // 获取相对于图片的坐标
    const rect = screenshotImg.getBoundingClientRect();
    const relativeX = event.clientX - rect.left;
    const relativeY = event.clientY - rect.top;

    // 计算在原始图片上的坐标
    const scaleX = screenshotImg.naturalWidth / screenshotImg.width;
    const scaleY = screenshotImg.naturalHeight / screenshotImg.height;

    dragStartX = Math.round(relativeX * scaleX);
    dragStartY = Math.round(relativeY * scaleY);

    screenshotImg.style.cursor = 'grabbing';
    event.preventDefault();
}

function handleMouseMove(event) {
    if (!isDragging) return;

    // 更新光标样式以显示正在拖拽
    screenshotImg.style.cursor = 'grabbing';
    event.preventDefault();
}

function handleMouseUp(event) {
    if (!isDragging) return;

    isDragging = false;
    screenshotImg.style.cursor = 'crosshair';

    const dragEndTime = Date.now();
    const dragDuration = dragEndTime - dragStartTime;

    // 获取相对于图片的坐标
    const rect = screenshotImg.getBoundingClientRect();
    const relativeX = event.clientX - rect.left;
    const relativeY = event.clientY - rect.top;

    // 计算在原始图片上的坐标
    const scaleX = screenshotImg.naturalWidth / screenshotImg.width;
    const scaleY = screenshotImg.naturalHeight / screenshotImg.height;

    const dragEndX = Math.round(relativeX * scaleX);
    const dragEndY = Math.round(relativeY * scaleY);

    // 计算移动距离
    const deltaX = dragEndX - dragStartX;
    const deltaY = dragEndY - dragStartY;
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

    // 如果移动距离很小或时间很短，认为是点击
    if (distance < 10 || dragDuration < 150) {
        handleClickAction(dragStartX, dragStartY);
    } else {
        // 否则认为是滑动，判断方向
        handleSwipeAction(dragStartX, dragStartY, dragEndX, dragEndY, deltaX, deltaY);
    }

    event.preventDefault();
}

function handleTouchStart(event) {
    if (isInteracting) return;

    const touch = event.touches[0];
    isDragging = true;

    const rect = screenshotImg.getBoundingClientRect();
    const relativeX = touch.clientX - rect.left;
    const relativeY = touch.clientY - rect.top;

    const scaleX = screenshotImg.naturalWidth / screenshotImg.width;
    const scaleY = screenshotImg.naturalHeight / screenshotImg.height;

    dragStartX = Math.round(relativeX * scaleX);
    dragStartY = Math.round(relativeY * scaleY);
    dragStartTime = Date.now();

    event.preventDefault();
}

function handleTouchMove(event) {
    if (!isDragging) return;
    event.preventDefault();
}

function handleTouchEnd(event) {
    if (!isDragging) return;

    isDragging = false;
    const dragEndTime = Date.now();
    const dragDuration = dragEndTime - dragStartTime;

    const touch = event.changedTouches[0];
    const rect = screenshotImg.getBoundingClientRect();
    const relativeX = touch.clientX - rect.left;
    const relativeY = touch.clientY - rect.top;

    const scaleX = screenshotImg.naturalWidth / screenshotImg.width;
    const scaleY = screenshotImg.naturalHeight / screenshotImg.height;

    const dragEndX = Math.round(relativeX * scaleX);
    const dragEndY = Math.round(relativeY * scaleY);

    const deltaX = dragEndX - dragStartX;
    const deltaY = dragEndY - dragStartY;
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

    if (distance < 10 || dragDuration < 150) {
        handleClickAction(dragStartX, dragStartY);
    } else {
        handleSwipeAction(dragStartX, dragStartY, dragEndX, dragEndY, deltaX, deltaY);
    }

    event.preventDefault();
}

async function handleClickAction(x, y) {
    isInteracting = true;

    try {
        // 如果正在自动刷新，暂时停止以避免冲突
        const wasAutoRefreshing = autoRefreshEnabled;
        if (wasAutoRefreshing) {
            console.log('点击操作开始，暂停自动刷新');
            stopAutoRefresh();
        }

        const response = await fetch('/click', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ x, y })
        });

        if (response.ok) {
            const result = await response.json();
            updateStatus(`点击操作完成: (${x}, ${y}) | 总操作数: ${result.action_count || 0}`, 'success');

            // 显示被点击的元素信息
            if (result.clicked_elements && result.clicked_elements.length > 0) {
                displayElementInfo(result.clicked_elements);
            }

            // 操作完成后刷新截图和UI元素信息
            setTimeout(async () => {
                await refreshScreenshot();
                console.log('点击操作后已刷新UI元素信息');

                // 如果之前开启了自动刷新，重新开启
                if (wasAutoRefreshing && isCollecting) {
                    setTimeout(() => {
                        console.log('重新开启自动刷新');
                        startAutoRefresh();
                        const btn = document.getElementById('autoRefreshBtn');
                        const statusPanel = document.getElementById('autoRefreshStatus');
                        btn.textContent = '⏹️ 停止刷新';
                        statusPanel.style.display = 'block';
                    }, 500); // 延迟500ms再开启自动刷新，给操作完成留出时间
                }
            }, 200);
        } else {
            const error = await response.json();
            updateStatus(`点击操作失败: ${error.detail}`, 'error');
        }
    } catch (error) {
        updateStatus(`点击操作错误: ${error.message}`, 'error');
    } finally {
        isInteracting = false;
    }
}

async function handleSwipeAction(startX, startY, endX, endY, deltaX, deltaY) {
    isInteracting = true;

    // 判断滑动方向
    let direction;
    if (Math.abs(deltaX) > Math.abs(deltaY))
        direction = deltaX > 0 ? 'right' : 'left';
    else
        direction = deltaY > 0 ? 'down' : 'up';

    try {
        // 如果正在自动刷新，暂时停止以避免冲突
        const wasAutoRefreshing = autoRefreshEnabled;
        if (wasAutoRefreshing) {
            console.log('滑动操作开始，暂停自动刷新');
            stopAutoRefresh();
        }

        const response = await fetch('/swipe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                startX,
                startY,
                endX,
                endY,
                direction
            })
        });

        if (response.ok) {
            const result = await response.json();
            updateStatus(`滑动操作完成: (${startX}, ${startY}) → (${endX}, ${endY}) [${direction}] | 总操作数: ${result.action_count || 0}`, 'success');

            setTimeout(async () => {
                await refreshScreenshot();
                console.log('滑动操作后已刷新UI元素信息');

                // 如果之前开启了自动刷新，重新开启
                if (wasAutoRefreshing && isCollecting) {
                    setTimeout(() => {
                        console.log('重新开启自动刷新');
                        startAutoRefresh();
                        const btn = document.getElementById('autoRefreshBtn');
                        const statusPanel = document.getElementById('autoRefreshStatus');
                        btn.textContent = '⏹️ 停止刷新';
                        statusPanel.style.display = 'block';
                    }, 500);
                }
            }, 200);
        } else {
            const error = await response.json();
            updateStatus(`滑动操作失败: ${error.detail}`, 'error');
        }
    } catch (error) {
        updateStatus(`滑动操作错误: ${error.message}`, 'error');
    } finally {
        isInteracting = false;
    }
}

function updateStatus(message, type) {
    const status = document.getElementById('status');
    status.innerHTML = `<div class="${type}">${message}</div>`;
}

async function refreshScreenshot() {
    try {
        console.log('开始刷新截图和UI元素信息...');

        const response = await fetch('/screenshot');
        const data = await response.json();

        if (screenshotImg && data.image_data) {
            screenshotImg.src = data.image_data;

            // 存储层次结构信息供后续使用
            window.currentHierarchy = data.hierarchy;

            // 解析并保存所有UI元素信息
            if (data.hierarchy) {
                const oldElementsCount = currentElements.length;
                currentElements = parseUIElements(data.hierarchy);

                // 统计可点击元素数量
                const clickableElements = currentElements.filter(el => el.clickable);
                console.log(`UI元素信息已更新: ${oldElementsCount} -> ${currentElements.length} 个元素 (其中 ${clickableElements.length} 个可点击)`);

                // 清除当前高亮，确保下次鼠标移动时重新计算
                clearElementHighlight();
                hoveredElement = null;

                // 如果鼠标在截图区域内，重新检测鼠标位置的元素
                checkMousePositionAfterRefresh();
            } else {
                console.warn('未获取到层次结构数据');
                currentElements = [];
            }

            console.log('截图和UI元素信息刷新完成');
            return true;
        } else {
            console.error('截图数据不完整');
            return false;
        }

    } catch (error) {
        console.error('刷新截图时出错:', error);
        return false;
    }
}

async function showActionHistory() {
    try {
        const response = await fetch('/action_history');
        const data = await response.json();

        if (response.ok) {
            displayHistoryModal(data.actions, data.total_actions);
        } else {
            updateStatus('获取操作历史失败', 'error');
        }
    } catch (error) {
        updateStatus(`获取操作历史错误: ${error.message}`, 'error');
    }
}

function displayHistoryModal(actions, totalCount) {
    // 创建弹窗
    const modal = document.createElement('div');
    modal.className = 'history-modal';

    const content = document.createElement('div');
    content.className = 'history-content';

    // 创建标题栏
    const header = document.createElement('div');
    header.className = 'history-header';
    header.innerHTML = `
        <h3>操作历史记录 (总计: ${totalCount})</h3>
        <button class="close-btn" onclick="closeHistoryModal()">&times;</button>
    `;

    content.appendChild(header);

    // 创建操作列表
    if (actions.length === 0) {
        content.innerHTML += '<p>暂无操作记录</p>';
    } else {
        actions.reverse().forEach((action, index) => {
            const item = document.createElement('div');
            item.className = 'action-item';

            const timestamp = new Date(action.timestamp).toLocaleString();
            let details = '';

            if (action.type === 'click') {
                details = `点击操作 - 位置: (${action.position.x}, ${action.position.y})`;
            } else if (action.type === 'swipe') {
                details = `滑动操作 - 从 (${action.press_position.x}, ${action.press_position.y}) 到 (${action.release_position.x}, ${action.release_position.y}) [${action.direction}]`;
            }

            item.innerHTML = `
                <div class="action-timestamp">${timestamp}</div>
                <div class="action-details">${details}</div>
            `;

            content.appendChild(item);
        });
    }

    modal.appendChild(content);
    document.body.appendChild(modal);

    // 点击背景关闭弹窗
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeHistoryModal();
        }
    });

    window.currentHistoryModal = modal;
}

function closeHistoryModal() {
    if (window.currentHistoryModal) {
        document.body.removeChild(window.currentHistoryModal);
        window.currentHistoryModal = null;
    }
}

async function saveCurrentData() {
    try {
        updateStatus(`正在保存数据...`, 'loading');

        const response = await fetch('/save_data', {
            method: 'POST'
        });

        if (response.ok) {
            const result = await response.json();
            updateStatus(`第 ${result.data_index} 条数据已保存 (${result.saved_actions} 个操作)`, 'success');
            return result;
        } else {
            const error = await response.json();
            throw new Error(error.detail || '保存数据失败');
        }
    } catch (error) {
        updateStatus(`保存数据失败: ${error.message}`, 'error');
        throw error;
    }
}

async function deleteCurrentData() {
    try {
        updateStatus(`正在删除数据...`, 'loading');

        const response = await fetch('/delete_data', {
            method: 'POST'
        });

        if (response.ok) {
            const result = await response.json();
            updateStatus(`第 ${result.data_index} 条数据已删除`, 'success');
            return result;
        } else {
            const error = await response.json();
            throw new Error(error.detail || '删除数据失败');
        }
    } catch (error) {
        updateStatus(`删除数据失败: ${error.message}`, 'error');
        throw error;
    }
}


function showTaskDescriptionModal(isNextData = false) {
    const modal = document.getElementById('taskDescriptionModal');
    const taskInput = document.getElementById('taskDescription');
    const confirmBtn = document.getElementById('confirmTaskBtn');
    const header = modal.querySelector('.task-modal-header h3');

    // 根据场景修改标题
    if (isNextData) {
        header.textContent = '📝 下一条数据 - 任务描述';
    } else {
        header.textContent = '📝 任务描述';
    }

    // 清空输入框
    taskInput.value = '';
    taskInput.focus();

    // 显示弹窗
    modal.style.display = 'flex';

    // 只绑定确认按钮事件
    confirmBtn.onclick = async () => {
        const description = taskInput.value.trim();
        if (description === '') {
            alert('请输入任务描述才能开始任务！');
            taskInput.focus();
            return;
        }

        currentTaskDescription = description;
        hideTaskDescriptionModal();

        if (isNextData) {
            await continueWithNextDataCollection();
        } else {
            await startDataCollectionWithDescription();
        }
    };
}

function hideTaskDescriptionModal() {
    const modal = document.getElementById('taskDescriptionModal');
    modal.style.display = 'none';
}

async function startDataCollectionWithDescription() {
    const startBtn = document.getElementById('startBtn');
    const endBtn = document.getElementById('endBtn');
    const nextBtn = document.getElementById('nextBtn');
    const deleteBtn = document.getElementById('deleteBtn');
    const inputBtn = document.getElementById('inputBtn');
    const historyBtn = document.getElementById('historyBtn');
    const autoRefreshBtn = document.getElementById('autoRefreshBtn');
    const collectionInfo = document.getElementById('collectionInfo');
    const status = document.getElementById('status');
    const container = document.getElementById('screenshotContainer');
    const tagBtn = document.getElementById('tagBtn');

    try {
        // 重置UI状态
        resetUIState();

        // 发送任务描述到后端
        await sendTaskDescription(currentTaskDescription);

        // 更新UI状态
        startBtn.disabled = true;
        endBtn.disabled = false;
        nextBtn.disabled = false;
        deleteBtn.disabled = false;
        inputBtn.disabled = false;
        historyBtn.disabled = false;
        autoRefreshBtn.disabled = false;
        isCollecting = true;
        tagBtn.disabled = false;
        // 更新状态显示
        const statusDiv = document.querySelector('.collection-status');
        statusDiv.classList.add('collecting');
        collectionInfo.innerHTML = `应用：${currentAppName} | 类型：${currentTaskType}<br/>任务：${currentTaskDescription}`;
        status.innerHTML = '<div class="loading">正在获取初始截图...</div>';
        container.innerHTML = '<div class="loading">截图中...</div>';

        // 自动获取截图
        await takeScreenshot();

        // 自动开启自动刷新功能
        if (!autoRefreshEnabled) {
            startAutoRefresh();
            autoRefreshBtn.textContent = '⏹️ 停止刷新';
            const statusPanel = document.getElementById('autoRefreshStatus');
            statusPanel.style.display = 'block';
            updateStatus('数据收集已开始，自动刷新已开启', 'success');
        }

        // 显示操作提示
        const hint = document.getElementById('actionHint');
        if (hint) {
            hint.style.display = 'block';
        }

    } catch (error) {
        updateStatus(`开始收集失败: ${error.message}`, 'error');
        // 恢复按钮状态
        startBtn.disabled = false;
        endBtn.disabled = true;
        nextBtn.disabled = true;
        deleteBtn.disabled = true;
        autoRefreshBtn.disabled = true;
        isCollecting = false;
    }
}

async function sendTaskDescription(description) {
    try {
        const response = await fetch('/set_task_description', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                description: description,
                app_name: currentAppName,
                task_type: currentTaskType
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '发送任务描述失败');
        }
    } catch (error) {
        console.error('发送任务描述失败:', error);
        throw error;
    }
}

// 重置UI状态函数
function resetUIState() {
    // 停止自动刷新
    if (autoRefreshEnabled) {
        stopAutoRefresh();
    }

    // 清除元素高亮
    clearElementHighlight();

    // 重置全局变量
    hoveredElement = null;
    currentElements = [];

    // 如果存在元素覆盖层，移除它
    if (elementOverlay) {
        elementOverlay.remove();
        elementOverlay = null;
    }

    // 清除之前的鼠标事件监听器（如果有的话）
    if (screenshotImg) {
        // 克隆节点来移除所有事件监听器
        const newImg = screenshotImg.cloneNode(true);
        screenshotImg.parentNode.replaceChild(newImg, screenshotImg);
        screenshotImg = newImg;
    }

    console.log('UI状态已重置');
}

async function continueWithNextDataCollection() {
    const collectionInfo = document.getElementById('collectionInfo');

    try {
        // 重置UI状态
        resetUIState();

        // 发送新的任务描述到后端
        await sendTaskDescription(currentTaskDescription);

        // 更新状态显示
        collectionInfo.innerHTML = `应用：${currentAppName} | 类型：${currentTaskType}<br/>任务：${currentTaskDescription}`;

        // 自动获取新截图
        await takeScreenshot();

        // 自动开启自动刷新功能
        if (!autoRefreshEnabled) {
            startAutoRefresh();
            const autoRefreshBtn = document.getElementById('autoRefreshBtn');
            autoRefreshBtn.textContent = '⏹️ 停止刷新';
            const statusPanel = document.getElementById('autoRefreshStatus');
            statusPanel.style.display = 'block';
        }

        updateStatus(`已切换下一条数据，自动刷新已开启`, 'success');

    } catch (error) {
        updateStatus(`切换到下一条数据失败: ${error.message}`, 'error');
    }
}

// 文本输入功能
function showInputModal() {
    if (!isCollecting) {
        updateStatus('请先开始数据收集', 'error');
        return;
    }

    const modal = document.getElementById('inputModal');
    const inputText = document.getElementById('inputText');

    modal.style.display = 'flex';
    inputText.value = '';
    inputText.focus();

    // 添加键盘快捷键支持
    inputText.onkeydown = function (event) {
        if (event.key === 'Escape') {
            hideInputModal();
        }
    };
}

function hideInputModal() {
    const modal = document.getElementById('inputModal');
    modal.style.display = 'none';
}

async function sendInputText() {
    const inputText = document.getElementById('inputText');
    const text = inputText.value.trim();

    if (!text) {
        updateStatus('请输入文本内容', 'error');
        return;
    }

    if (!isCollecting) {
        updateStatus('请先开始数据收集', 'error');
        hideInputModal();
        return;
    }

    try {
        updateStatus('正在发送文本...', 'info');

        // 如果正在自动刷新，暂时停止以避免冲突
        const wasAutoRefreshing = autoRefreshEnabled;
        if (wasAutoRefreshing) {
            console.log('文本输入操作开始，暂停自动刷新');
            stopAutoRefresh();
        }

        hideInputModal();
        const response = await fetch('/input', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            })
        });
        if (response.ok) {
            const result = await response.json();
            updateStatus(`文本输入完成: "${text}"`, 'success');

            // 操作完成后刷新截图和UI元素信息
            setTimeout(async () => {
                await refreshScreenshot();
                console.log('输入操作后已刷新UI元素信息');

                // 如果之前开启了自动刷新，重新开启
                if (wasAutoRefreshing && isCollecting) {
                    setTimeout(() => {
                        console.log('重新开启自动刷新');
                        startAutoRefresh();
                        const btn = document.getElementById('autoRefreshBtn');
                        const statusPanel = document.getElementById('autoRefreshStatus');
                        btn.textContent = '⏹️ 停止刷新';
                        statusPanel.style.display = 'block';
                    }, 500);
                }
            }, 200);
        } else {
            const error = await response.json();
            updateStatus(`输入操作失败: ${error.detail}`, 'error');
        }

    } catch (error) {
        console.error('文本输入失败:', error);
        updateStatus(`文本输入失败: ${error.message}`, 'error');
    }
}

// 显示元素信息
function displayElementInfo(elements) {
    const elementInfo = document.getElementById('elementInfo');
    const elementDetails = document.getElementById('elementDetails');

    if (!elements || elements.length === 0) {
        elementInfo.style.display = 'none';
        return;
    }

    let html = '';
    elements.forEach((element, index) => {
        html += `
            <div class="element-item">
                <div class="element-property"><strong>元素 ${index + 1}:</strong></div>
                <div class="element-property element-bounds"><strong>位置:</strong> ${element.bounds}</div>
                <div class="element-property"><strong>类型:</strong> ${element.class}</div>
                ${element['resource-id'] ? `<div class="element-property"><strong>ID:</strong> ${element['resource-id']}</div>` : ''}
                ${element.text ? `<div class="element-property element-text"><strong>文本:</strong> ${element.text}</div>` : ''}
                ${element['content-desc'] ? `<div class="element-property"><strong>描述:</strong> ${element['content-desc']}</div>` : ''}
                <div class="element-property"><strong>可点击:</strong> ${element.clickable ? '是' : '否'}</div>
                <div class="element-property"><strong>应用包名:</strong> ${element.package}</div>
            </div>
        `;
    });

    elementDetails.innerHTML = html;
    elementInfo.style.display = 'block';
}

// 解析UI层次结构，提取所有元素的位置信息
function parseUIElements(hierarchyXml) {
    if (!hierarchyXml) return [];

    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(hierarchyXml, 'text/xml');
    const nodes = xmlDoc.querySelectorAll('node');
    const elements = [];

    nodes.forEach(node => {
        const bounds = node.getAttribute('bounds');
        if (bounds) {
            // 解析bounds属性，格式如: [left,top][right,bottom]
            const boundsMatch = bounds.match(/\[(\d+),(\d+)\]\[(\d+),(\d+)\]/);
            if (boundsMatch) {
                const left = parseInt(boundsMatch[1]);
                const top = parseInt(boundsMatch[2]);
                const right = parseInt(boundsMatch[3]);
                const bottom = parseInt(boundsMatch[4]);

                elements.push({
                    bounds: bounds,
                    left: left,
                    top: top,
                    right: right,
                    bottom: bottom,
                    width: right - left,
                    height: bottom - top,
                    class: node.getAttribute('class') || '',
                    'resource-id': node.getAttribute('resource-id') || '',
                    text: node.getAttribute('text') || '',
                    'content-desc': node.getAttribute('content-desc') || '',
                    clickable: node.getAttribute('clickable') === 'true',
                    package: node.getAttribute('package') || ''
                });
            }
        }
    });

    return elements;
}

// 创建元素高亮覆盖层
function createElementOverlay() {
    if (elementOverlay) return elementOverlay;

    if (!screenshotImg || !screenshotImg.parentElement) {
        console.error('截图元素或其父容器不存在');
        return null;
    }

    const overlay = document.createElement('div');
    overlay.id = 'elementOverlay';
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.pointerEvents = 'none';
    overlay.style.zIndex = '10';

    const container = screenshotImg.parentElement;
    container.style.position = 'relative';
    container.appendChild(overlay);

    elementOverlay = overlay;

    // 监听窗口大小变化，重新绘制边框
    window.addEventListener('resize', () => {
        if (hoveredElement) {
            drawElementBorder(hoveredElement);
        }
    });

    console.log('元素高亮覆盖层已创建');
    return overlay;
}

// 在指定位置绘制元素边框
function drawElementBorder(element) {
    if (!screenshotImg || !element) {
        console.warn('绘制元素边框失败：缺少截图或元素信息');
        return;
    }

    const overlay = createElementOverlay();
    if (!overlay) {
        console.error('创建覆盖层失败，无法绘制元素边框');
        return;
    }

    // 获取图片在容器中的实际位置
    const imgRect = screenshotImg.getBoundingClientRect();
    const containerRect = screenshotImg.parentElement.getBoundingClientRect();

    // 计算图片相对于容器的偏移
    const imgOffsetX = imgRect.left - containerRect.left;
    const imgOffsetY = imgRect.top - containerRect.top;

    // 计算缩放比例
    const scaleX = screenshotImg.width / screenshotImg.naturalWidth;
    const scaleY = screenshotImg.height / screenshotImg.naturalHeight;

    // 计算在显示图片上的位置（相对于图片左上角）
    const displayLeft = element.left * scaleX;
    const displayTop = element.top * scaleY;
    const displayWidth = element.width * scaleX;
    const displayHeight = element.height * scaleY;

    // 创建边框元素，位置相对于容器，但要加上图片的偏移
    const border = document.createElement('div');
    border.style.position = 'absolute';
    border.style.left = (imgOffsetX + displayLeft) + 'px';
    border.style.top = (imgOffsetY + displayTop) + 'px';
    border.style.width = displayWidth + 'px';
    border.style.height = displayHeight + 'px';
    border.style.border = '2px solid #ff6b6b';
    border.style.backgroundColor = 'rgba(255, 107, 107, 0.1)';
    border.style.boxSizing = 'border-box';

    // 清除之前的边框
    overlay.innerHTML = '';
    overlay.appendChild(border);
}

// 清除元素高亮
function clearElementHighlight() {
    if (elementOverlay) {
        elementOverlay.innerHTML = '';
    }
    hoveredElement = null;
}

// 根据鼠标位置查找对应的UI元素（只显示可点击的元素）
function findElementAtPosition(x, y) {
    if (!currentElements.length) return null;

    // 计算在原始图片上的坐标
    const scaleX = screenshotImg.naturalWidth / screenshotImg.width;
    const scaleY = screenshotImg.naturalHeight / screenshotImg.height;

    const originalX = x * scaleX;
    const originalY = y * scaleY;

    // 找到包含该点的可点击元素（只显示clickable为true的元素）
    const matchingElements = currentElements.filter(element =>
        element.clickable &&  // 只显示可点击的元素
        originalX >= element.left &&
        originalX <= element.right &&
        originalY >= element.top &&
        originalY <= element.bottom
    );

    if (matchingElements.length === 0) return null;

    // 返回面积最小的可点击元素
    return matchingElements.reduce((smallest, current) => {
        const smallestArea = smallest.width * smallest.height;
        const currentArea = current.width * current.height;
        return currentArea < smallestArea ? current : smallest;
    });
}

// 鼠标移动处理函数
function handleScreenshotMouseMove(event) {
    if (!screenshotImg) {
        console.log('没有截图元素');
        return;
    }

    if (!currentElements.length) {
        console.log('没有UI元素信息，currentElements长度:', currentElements.length);
        return;
    }

    const rect = screenshotImg.getBoundingClientRect();
    const relativeX = event.clientX - rect.left;
    const relativeY = event.clientY - rect.top;

    // 更新鼠标位置记录
    lastMousePosition = { x: relativeX, y: relativeY };

    // 确保鼠标在图片范围内
    if (relativeX < 0 || relativeX > screenshotImg.width ||
        relativeY < 0 || relativeY > screenshotImg.height) {
        if (hoveredElement) {
            clearElementHighlight();
        }
        return;
    }

    const element = findElementAtPosition(relativeX, relativeY);

    if (element !== hoveredElement) {
        hoveredElement = element;

        if (element) {
            drawElementBorder(element);
            console.log('高亮可点击元素:', element.class, element.clickable ? '✓可点击' : '✗不可点击');
        } else {
            clearElementHighlight();
        }
    }
}

// 刷新后检测鼠标位置的元素
function checkMousePositionAfterRefresh() {
    if (!screenshotImg || !currentElements.length) {
        return;
    }

    // 如果有记录的鼠标位置且在有效范围内
    if (lastMousePosition.x >= 0 && lastMousePosition.y >= 0) {
        const rect = screenshotImg.getBoundingClientRect();

        // 确保鼠标位置在图片范围内
        if (lastMousePosition.x >= 0 && lastMousePosition.x <= screenshotImg.width &&
            lastMousePosition.y >= 0 && lastMousePosition.y <= screenshotImg.height) {

            const element = findElementAtPosition(lastMousePosition.x, lastMousePosition.y);

            if (element !== hoveredElement) {
                hoveredElement = element;

                if (element) {
                    drawElementBorder(element);
                    console.log('刷新后重新高亮元素:', element.class, element.clickable ? '✓可点击' : '✗不可点击');
                } else {
                    clearElementHighlight();
                }
            }
        }
    }
}

// 自动刷新功能 - 简化版本，固定0.7秒间隔
function toggleAutoRefresh() {
    if (!isCollecting) {
        updateStatus('请先开始数据收集', 'error');
        return;
    }

    const btn = document.getElementById('autoRefreshBtn');
    const statusPanel = document.getElementById('autoRefreshStatus');

    if (autoRefreshEnabled) {
        // 当前已开启，点击关闭
        stopAutoRefresh();
        btn.textContent = '⏰ 自动刷新';
        statusPanel.style.display = 'none';
        updateStatus('自动刷新已关闭', 'success');
    } else {
        // 当前已关闭，点击开启
        startAutoRefresh();
        btn.textContent = '⏹️ 停止刷新';
        statusPanel.style.display = 'block';
        updateStatus('自动刷新已开启，连续刷新模式', 'success');
    }
}

// 连续自动刷新功能 - 请求完成后立即发下一个请求
async function startAutoRefresh() {
    if (autoRefreshEnabled) return;
    autoRefreshEnabled = true;

    while (autoRefreshEnabled && isCollecting) {
        // 检查是否应该刷新：正在收集数据、没有正在交互
        if (!isInteracting) {
            try {
                console.log('连续自动刷新截图...');
                const success = await refreshScreenshot();
                if (success) {
                    console.log('连续自动刷新完成');
                } else {
                    console.log('连续自动刷新跳过或失败');
                }
            } catch (error) {
                console.error('连续自动刷新失败:', error);
                // 出错时等待一小段时间再继续，避免连续错误
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        } else {
            // 如果不能刷新，等待一小段时间再检查
            if (!isCollecting) console.log('连续刷新等待：未在收集数据');
            if (isInteracting) console.log('连续刷新等待：正在交互');

            await new Promise(resolve => setTimeout(resolve, 100)); // 等待100ms后重新检查
        }
    }
    console.log('连续自动刷新已停止');
}

function stopAutoRefresh() {
    if (!autoRefreshEnabled) return;
    autoRefreshEnabled = false;
}

async function tag() {
  if (!isCollecting) {
    updateStatus('请先开始数据收集', 'error');
    return;
  }

  // 让用户输入要写入 JSON 的标签值（可为数字或字符串）
  const label = window.prompt('请输入标签（如 1、A、Start 等）');
  if (label === null || label === '') return;

  const wasAuto = autoRefreshEnabled;
  if (wasAuto) stopAutoRefresh();

  try {
    const resp = await fetch('/annotate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ label })       // 关键：把标签传给后端
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || '打标签失败');
    }

    updateStatus(`已打标签：${label}（并保存了当前截图）`, 'success');
    await refreshScreenshot();              // 打完刷新一次预览
  } catch (e) {
    updateStatus(`打标签失败：${e.message}`, 'error');
  } finally {
    if (wasAuto && isCollecting) {
      startAutoRefresh();
      // （可选）如果你有“自动刷新按钮文字/状态条”的 UI 联动，这里顺手恢复
      const btn = document.getElementById('autoRefreshBtn');
      const statusPanel = document.getElementById('autoRefreshStatus');
      if (btn) btn.textContent = '⏹️ 停止刷新';
      if (statusPanel) statusPanel.style.display = 'block';
    }
  }
}


// 应用信息输入功能
function showAppInfoModal() {
    const modal = document.getElementById('appInfoModal');
    const appNameInput = document.getElementById('appName');
    const taskTypeInput = document.getElementById('taskType');
    const confirmBtn = document.getElementById('confirmAppInfoBtn');

    // 清空输入框
    appNameInput.value = '';
    taskTypeInput.value = '';
    appNameInput.focus();

    // 显示弹窗
    modal.style.display = 'flex';

    // 绑定确认按钮事件
    confirmBtn.onclick = async () => {
        const appName = appNameInput.value.trim();
        const taskType = taskTypeInput.value.trim();

        if (appName === '') {
            alert('请选择应用名称！');
            appNameInput.focus();
            return;
        }

        if (taskType === '') {
            alert('请输入任务类型！');
            taskTypeInput.focus();
            return;
        }

        // 保存应用信息
        currentAppName = appName;
        currentTaskType = taskType;

        // 隐藏应用信息弹窗
        hideAppInfoModal();

        // 显示任务描述弹窗
        showTaskDescriptionModal();
    };
}

function hideAppInfoModal() {
    const modal = document.getElementById('appInfoModal');
    modal.style.display = 'none';
}