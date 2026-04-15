#!/bin/bash
# 切換區域控制器
# 用法：
#   ./switch_controller.sh rpp
#   ./switch_controller.sh mppi
#   ./switch_controller.sh mpc

set -e

CONTROLLER=${1:-""}
WS="$HOME/masterthesis"
YAML="$WS/src/ammr_navigation/config/nav2_params.yaml"
CONTROLLERS_DIR="$WS/src/ammr_navigation/config/controllers"

if [ -z "$CONTROLLER" ]; then
    echo "用法：$0 [rpp|mppi|mpc]"
    echo ""
    echo "目前控制器："
    grep "plugin:" "$YAML" | grep -A0 "FollowPath" | head -1 || grep "FollowPath" -A1 "$YAML" | grep plugin
    exit 1
fi

CONTROLLER_FILE="$CONTROLLERS_DIR/$CONTROLLER.yaml"

if [ ! -f "$CONTROLLER_FILE" ]; then
    echo "找不到控制器設定：$CONTROLLER_FILE"
    echo "可用控制器：$(ls $CONTROLLERS_DIR | sed 's/.yaml//g' | tr '\n' ' ')"
    exit 1
fi

python3 - "$YAML" "$CONTROLLER_FILE" <<'EOF'
import sys
import re

yaml_path = sys.argv[1]
controller_path = sys.argv[2]

with open(yaml_path) as f:
    content = f.read()

with open(controller_path) as f:
    new_controller = f.read()

# 縮排 controller yaml 內容（加 4 個空格）
indented = '\n'.join('    ' + line if line.strip() else line
                     for line in new_controller.strip().splitlines())

# 替換 FollowPath 區塊
pattern = r'    FollowPath:.*?(?=\n\S|\n    [a-z]|\Z)'
replacement = indented

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open(yaml_path, 'w') as f:
    f.write(new_content)

EOF

echo "✓ 已切換至：$CONTROLLER"
echo ""
echo "現在執行以下指令讓變更生效："
echo "  cd $WS && colcon build --packages-select ammr_navigation && source install/setup.bash"
