# 阿里云部署说明

当前版本适合单服务器、一次少量家庭会话的实验部署。家庭端无需邀请码；研究端访问码只供研究人员使用。

## 1. 服务器与网络

- 使用 Linux ECS，准备域名和 HTTPS 证书。
- 安全组对公网只开放 80/443；SSH 仅允许研究人员的固定 IP。
- 不开放 8766。Nginx 监听公网端口，应用只监听 127.0.0.1:8766。
- 当前会话准入、声音样本和研究端实时状态保存在单进程内存中，因此只运行一个应用进程，不要配置多个 Uvicorn worker。

阿里云的[安全组官方说明](https://help.aliyun.com/zh/ecs/user-guide/start-using-security-groups)同样建议公网网站按需开放 80/443，并限制 SSH 来源。

## 2. 安装

```bash
sudo apt update
sudo apt install -y python3 python3-venv build-essential nginx
cd /srv/coregulation-realtime-live
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

编辑 `.env`，至少填写：

```dotenv
DASHSCOPE_API_KEY=你的百炼API密钥
ALIYUN_WORKSPACE_ID=你的业务空间ID
ALIYUN_REGION=cn-beijing
OMNI_MODEL=qwen3.5-omni-flash-realtime
TEXT_CHAT_MODEL=qwen3.7-plus
JUDGMENT_MODEL=qwen3.7-plus
TENCENT_SECRET_ID=你的腾讯云SecretId
TENCENT_SECRET_KEY=你的腾讯云SecretKey
TENCENT_VOICEPRINT_REGION=ap-guangzhou
TENCENT_VOICEPRINT_MINIMUM_SCORE=70
RESEARCH_CONSOLE_ACCESS_TOKEN=单独生成的研究端长随机值
OUTPUT_DIR=/srv/coregulation-realtime-live/data/output
```

`BROWSER_CAPTURE_ACCESS_TOKEN`可以留空。家庭填写信息后，服务器会自动发放短时会话令牌，页面不会要求邀请码。

## 3. 启动前检查

```bash
. .venv/bin/activate
coregulation-poc doctor
coregulation-poc connection-test
```

`connection-test`必须返回 `"ok": true`。当前使用的实时模型和文本模型均是阿里云百炼已发布模型。

## 4. 启动应用

```bash
coregulation-poc web-live \
  --host 127.0.0.1 \
  --port 8766 \
  --enable-closed-loop \
  --enable-voice \
  --window-seconds 10 \
  --assessment-interval-seconds 10 \
  --max-assessments 180
```

不要漏掉 `--enable-closed-loop`，否则只有界面预览，不会调用模型判断状态。建议用 systemd 托管上述单进程命令，并设置失败自动重启。

## 5. Nginx 关键配置

在 HTTPS 站点的 `server` 中加入：

```nginx
client_max_body_size 2m;

location / {
    proxy_pass http://127.0.0.1:8766;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

家庭端必须用 `https://你的域名/`，研究端使用 `https://你的域名/research`。摄像头和麦克风在公网 HTTP 页面通常无法使用，因此不能省略 HTTPS。

## 6. 上线验收

1. 打开家庭端，填写年龄、年级和作业信息，选择角色。
2. 分别录制家长与儿童声音，确认两项都显示完成。
3. 开始活动，研究端应出现该会话。
4. 让双方交谈并完成至少一个 12 秒窗口；研究端应显示状态和说话人区分结果。
5. 结束活动，家庭端应出现单次总结。
6. 检查 `data/output/runs/` 中生成了 `manifest.json`、`events.jsonl`、`metrics.json` 和 `result.json`。

浏览器录制的两段绑定声音由服务器通过腾讯云官方SDK注册为本次会话的临时声纹；绑定原音和声纹ID不写入磁盘。实验结束时，服务器先逐个删除云端家长与儿童声纹，再删除空分组；清理结果写入`events.jsonl`。家庭明确同意后，活动阶段的完整音视频会另存到该匿名实验的`media/`目录，供事后人工复核；它属于受限研究数据，必须按伦理审批控制访问与删除。

腾讯密钥只写在服务器的 `/srv/coregulation-realtime-live/.env` 中，并将该文件权限设为仅服务账号可读：

```bash
chmod 600 /srv/coregulation-realtime-live/.env
```
