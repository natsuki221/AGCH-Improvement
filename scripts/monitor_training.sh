#!/bin/bash
# scripts/monitor_training.sh
#
# 訓練監控腳本
#
# 對應手冊章節:
# - §13.1 記憶體使用檢查清單
#
# 功能:
# 1. 即時監控 GPU 使用率與 VRAM
# 2. 監控 CPU 與 RAM 使用率
# 3. 支援寫入日誌檔案
#
# 使用方式:
#   ./scripts/monitor_training.sh           # 即時監控
#   ./scripts/monitor_training.sh -l log    # 寫入日誌

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 預設參數
INTERVAL=2
LOG_FILE=""
VRAM_ALERT_THRESHOLD=14.5

# 解析參數
while getopts "l:i:t:h" opt; do
    case $opt in
        l)
            LOG_FILE="$OPTARG"
            ;;
        i)
            INTERVAL="$OPTARG"
            ;;
        t)
            VRAM_ALERT_THRESHOLD="$OPTARG"
            ;;
        h)
            echo "使用方式: $0 [-l logfile] [-i interval] [-t vram_threshold]"
            echo ""
            echo "選項:"
            echo "  -l logfile      將監控資料寫入日誌檔案"
            echo "  -i interval     監控間隔（秒），預設 2"
            echo "  -t threshold    VRAM 警告閾值（GB），預設 14.5"
            echo "  -h              顯示此說明"
            exit 0
            ;;
        \?)
            echo "無效選項: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# 標題
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}       GPU 訓練監控（RTX 5080 16GB）${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "監控間隔: ${INTERVAL}s"
echo "VRAM 警告閾值: ${VRAM_ALERT_THRESHOLD}GB"
if [ -n "$LOG_FILE" ]; then
    echo "日誌檔案: $LOG_FILE"
    echo "timestamp,gpu_util,vram_used,vram_total,temp,power,cpu_percent,ram_used,ram_total" > "$LOG_FILE"
fi
echo ""
echo "按 Ctrl+C 停止監控"
echo ""

# 監控函數
monitor() {
    while true; do
        # 獲取 GPU 資訊
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
        
        if [ -z "$GPU_INFO" ]; then
            echo -e "${RED}無法獲取 GPU 資訊${NC}"
            sleep $INTERVAL
            continue
        fi
        
        # 解析 GPU 資訊
        GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f1 | tr -d ' ')
        VRAM_USED=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
        VRAM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f3 | tr -d ' ')
        GPU_TEMP=$(echo "$GPU_INFO" | cut -d',' -f4 | tr -d ' ')
        GPU_POWER=$(echo "$GPU_INFO" | cut -d',' -f5 | tr -d ' ')
        
        # 計算 VRAM 使用百分比
        VRAM_PERCENT=$(awk "BEGIN {printf \"%.1f\", $VRAM_USED / $VRAM_TOTAL * 100}")
        VRAM_USED_GB=$(awk "BEGIN {printf \"%.2f\", $VRAM_USED / 1024}")
        VRAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $VRAM_TOTAL / 1024}")
        
        # 獲取 CPU 與 RAM 資訊
        CPU_PERCENT=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        RAM_INFO=$(free -m | grep Mem)
        RAM_USED=$(echo "$RAM_INFO" | awk '{print $3}')
        RAM_TOTAL=$(echo "$RAM_INFO" | awk '{print $2}')
        RAM_USED_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_USED / 1024}")
        RAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_TOTAL / 1024}")
        RAM_PERCENT=$(awk "BEGIN {printf \"%.1f\", $RAM_USED / $RAM_TOTAL * 100}")
        
        # 清除上一行
        echo -en "\033[2K\r"
        
        # 顯示狀態
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        
        # GPU 使用率顏色
        if (( $(echo "$GPU_UTIL > 90" | bc -l) )); then
            GPU_COLOR=$GREEN
        elif (( $(echo "$GPU_UTIL > 50" | bc -l) )); then
            GPU_COLOR=$YELLOW
        else
            GPU_COLOR=$RED
        fi
        
        # VRAM 使用率顏色與警告
        VRAM_WARNING=""
        if (( $(echo "$VRAM_USED_GB > $VRAM_ALERT_THRESHOLD" | bc -l) )); then
            VRAM_COLOR=$RED
            VRAM_WARNING=" ⚠️"
        elif (( $(echo "$VRAM_USED_GB > 12" | bc -l) )); then
            VRAM_COLOR=$YELLOW
        else
            VRAM_COLOR=$GREEN
        fi
        
        # 輸出
        echo -e "[${TIMESTAMP}] " \
            "GPU: ${GPU_COLOR}${GPU_UTIL}%${NC} | " \
            "VRAM: ${VRAM_COLOR}${VRAM_USED_GB}GB${NC}/${VRAM_TOTAL_GB}GB (${VRAM_PERCENT}%)${VRAM_WARNING} | " \
            "Temp: ${GPU_TEMP}°C | " \
            "Power: ${GPU_POWER}W | " \
            "CPU: ${CPU_PERCENT}% | " \
            "RAM: ${RAM_USED_GB}GB/${RAM_TOTAL_GB}GB (${RAM_PERCENT}%)"
        
        # 寫入日誌
        if [ -n "$LOG_FILE" ]; then
            echo "$TIMESTAMP,$GPU_UTIL,$VRAM_USED_GB,$VRAM_TOTAL_GB,$GPU_TEMP,$GPU_POWER,$CPU_PERCENT,$RAM_USED_GB,$RAM_TOTAL_GB" >> "$LOG_FILE"
        fi
        
        sleep $INTERVAL
    done
}

# 處理中斷
trap 'echo -e "\n\n${GREEN}監控已停止${NC}"; exit 0' INT

# 開始監控
monitor
