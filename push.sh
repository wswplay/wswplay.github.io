#!/usr/bin/env sh
git status
git add .
git status

# 输入注释
read -p "请输入本次提交注释: " MSG
# 打印注释
echo "注释为: $MSG"

# 等待5秒
sleep 5

git commit -m $MSG

echo "完成推送!"
git status

