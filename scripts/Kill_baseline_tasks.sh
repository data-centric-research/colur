#!/bin/bash

# 定义要查找并杀掉的进程关键字列表
# keywords=("baseline_code" "core_model" "run_experiment.py")
keywords=("baseline_code", "CRUL", "colearn")

# 定义监控时长（秒），可以通过第一个参数指定，默认是30秒
monitor_duration=${1:-150}
end_time=$((SECONDS + monitor_duration))

# 函数用于杀掉进程并打印结果
kill_processes() {
    local keyword=$1
    echo -e "\e[34m[Info] Searching for processes containing keyword: $keyword\e[0m"
    pids=$(ps aux | grep "$keyword" | grep -v "grep" | awk '{print $2}')
    
    if [ -z "$pids" ]; then
        echo -e "\e[33m[Warning] No processes found for keyword: $keyword\e[0m"
    else
        for pid in $pids; do
            echo -e "\e[34m[Info] Attempting to kill process with PID: $pid (keyword: $keyword)\e[0m"
            kill -9 "$pid" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "\e[32m[✔] Successfully killed process with PID: $pid\e[0m"
            else
                echo -e "\e[31m[✖] Failed to kill process with PID: $pid\e[0m"
            fi
        done
    fi
}

# 初次执行杀掉进程
for keyword in "${keywords[@]}"; do
    kill_processes "$keyword"
done

echo -e "\e[34m[Info] Initial process kill completed. Monitoring for $monitor_duration seconds...\e[0m"

# 监控并杀掉重启的进程
while [ $SECONDS -lt $end_time ]; do
    for keyword in "${keywords[@]}"; do
        kill_processes "$keyword"
    done
    sleep 2 # 每隔2秒检查一次
done

echo -e "\e[34m[Info] Monitoring completed. All matching processes have been processed.\e[0m"