# ============================================================
# 安全响应头片段（仅用于不经过后端的静态资源 location）
# ============================================================
# 为什么用 .inc 后缀而非 .conf：nginx.conf 里有
# ``include /etc/nginx/conf.d/*.conf;``，本片段若叫 .conf 会被当成
# 一份独立的顶层配置加载，因缺少 server/http 上下文而直接报错。
# .inc 后缀使其只能被显式 include，语义清晰。
#
# 为什么必须显式 include 到每个静态 location：nginx 的 add_header
# 继承规则是「本层存在任意 add_header 时，完全放弃上层继承」。
# index.html 与 assets 已经各自设了 Cache-Control，若不 include 本片段，
# 它们将**一个安全头都没有**。
#
# 各头的作用（医疗系统含个人健康信息，一个都不能少）：
# - X-Content-Type-Options: nosniff —— 禁止浏览器猜测响应类型，
#   防止上传的文本被当作脚本执行
# - X-Frame-Options: DENY —— 禁止被嵌套，防点击劫持
# - Referrer-Policy —— 跨域跳转不外泄含患者 ID 的完整路径
# - Permissions-Policy —— 后端与前端都用不到摄像头/麦克风/定位，一律拒绝，
#   避免被恶意页面借本域发起授权弹窗
#
# /api/ 的响应由后端中间件设置同样的头（imcs_main.add_security_headers），
# 因此本片段**不要**用在 server 级或 /api/ location 上，否则头会重复。
# ============================================================

add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;
