# visual_recall

让 AstrBot 里的 Agent **根据语境按需回看会话中出现过的历史图片**，**无需引用图片，也不用在消息中附带图片，也不需要开启图片上下文**。

  模型在需要参考之前某张图片时，可调用 recall_image 工具按会话内序号或关键词检索图片并真正看到图片内容。\
  适用场景：用户在群里连发多张图片后，中间插入几轮文字对话，模型想引用前面的图时无需让用户重发。\
  原理：拦截所有 incoming 图片 → 复制到本地存储并建立 SQLite 索引 → 在 Agent 上下文里注入索引摘要（on_agent_begin）→ 工具返回图片触发 runner 自动追加 user 消息。

## 工作原理

1. **捕获留档**：监听全部消息，同步将消息链内图片复制至插件数据目录，同时将会话、发送者、序号、文本、时间等元数据写入 SQLite 索引。AstrBot 事件结束会销毁临时媒体文件，不可异步执行复制，否则文件丢失。

2. **向模型告知存在图片**：Agent 会话启动钩子 `on_agent_begin`，通过 `run_context.messages` 注入系统消息，推送该会话的图片索引列表，告知模型可回看历史图片。> 注：Agent 模式不会触发 `on_llm_request`，因此只能使用 `on_agent_begin` 作为注入点，该步骤是功能生效的关键。

3. **召回渲染图片给模型**：模型需要查阅图片时调用 `recall_image` 工具，返回携带 base64 的 `CallToolResult`。AstrBot 的 `tool_loop_agent_runner.py` 会缓存图片，并追加携带真实图片的 user 消息进上下文，让模型真正读取图片内容，而非仅拿到文件路径文本。

4. **GIF 自动抽帧**：识别到 `mime=image/gif` 时，插件内置的 PIL 工具会按等间隔抽若干帧（默认 8 帧），拼成一张 `cols×rows` 的 contact sheet 静态预览图返回，每帧左上角标 `#K` 表示该原始帧序号。多数多模态模型直接看真 GIF 不稳定（多半只会渲染第 1 帧），静态预览图能让模型一次看清整个动画过程。需要安装 Pillow，未安装时回退原图。需要更精细控制时也可显式调用 `recall_gif_frames` 工具。

## 前置条件

1. **主模型供应商必须支持视觉**。这是硬条件。runner 会检查 provider 的 `modalities`，不含 `image` 时只会保留图片路径文字，模型看不到图像内容。

2. **AstrBot 版本够新**。本插件按 master 源码实现，依赖 `add_llm_tools`（v4.5.1+）、`on_agent_begin` 钩子、以及 runner 里那段「把工具返回的图片回注给 LLM」的逻辑（对应 `tool_image_cache`）。版本过旧时工具能调通，但模型看不到图。

3. **Pillow 需要安装**（本插件根目录已附带 `requirements.txt`）。AstrBot 在「安装 / 加载」插件时会自动执行 `pip install -r requirements.txt`，把 Pillow 一并装上，**无需手动操作**。若自动安装失败（如离线环境、网络受限），请手动执行：
   
   ```bash
   pip install "Pillow>=10.0.0"
   # 或进入插件目录后：
   # pip install -r requirements.txt
   ```
   
   Pillow 缺失时的降级行为：**图片压缩、GIF 自动抽帧、recall_gif_frames 全部失效**。`recall_image` 仍可用，但 GIF 会回退成原文件（多数模型只渲染第 1 帧），静态图按原图返回、不压缩。建议安装后再用。

## 安装

在 AstrBot 控制台「插件市场 → 已安装 → 上传插件」上传即可（上传时 AstrBot 会自动读取 `requirements.txt` 安装 Pillow）。如果报 AstrBot 版本不匹配，说明你的 AstrBot 低于 4.17.0，需要先升级。

目录结构：

```
astrbot_plugin_visual_recall/
├── metadata.yaml        AstrBot 据此识别插件（必需）
├── main.py              插件主体：捕获、注入、清理、调试命令
├── tools.py             recall_image 工具定义
├── storage.py           SQLite 索引
├── _conf_schema.json    配置项
└── requirements.txt     Python 依赖（Pillow），AstrBot 安装时自动 pip 安装
```

## 配置

| 配置项                      | 默认   | 说明                    |
| ------------------------ | ---- | --------------------- |
| `enabled`                | true | 总开关                   |
| `inject_index`           | true | 是否注入图片索引，关掉后模型不会再主动召回 |
| `index_entries`          | 8    | 索引里最多列几张              |
| `max_images_per_call`    | 3    | 单次召回最多返回几张            |
| `max_image_edge`         | 1280 | 召回图片最大边长，超出等比压缩，0 关闭  |
| `gif_as_contact_sheet`   | true | GIF 是否自动抽帧拼成预览图       |
| `gif_frame_count`        | 8    | GIF 抽帧数（1-16）         |
| `gif_max_edge`           | 640  | GIF 预览图每帧的最大边长        |
| `max_store_mb`           | 15   | 单张超过此大小不入索引           |
| `max_images_per_session` | 60   | 每会话最多保留多少张            |
| `keep_days`              | 7    | 记录保留天数                |

## 工具

| 工具                  | 说明                                   |
| ------------------- | ------------------------------------ |
| `recall_image`      | 通用回看本会话历史图片。GIF 会自动抽帧成 contact sheet |
| `recall_gif_frames` | 专门处理 GIF：可指定抽多少帧、一次处理多张 GIF，普通图会拒绝   |

## 指令

| 指令            | 说明             |
| ------------- | -------------- |
| `/vimg`       | 查看索引列表         |
| `/vimg_clear` | 清空当前会话的索引与图片文件 |

## 使用

1.无需引用图片，无需附带图片，自动根据语境按需回看图片

<img title="" src="docs\examples\1.jpg" alt="" width="1379">

2.llm根据序号/时间/备注 索引回看图片

<img title="" src="docs\examples/2.jpg" alt="" width="330" data-align="inline"><img title="" src="docs\examples/3.jpg" alt="" width="282"><img title="" src="docs\examples/4.jpg" alt="" width="479" data-align="left">

1. 在会话里发一张图，附带一句文字（比如「这是柜门」）。
2. 发送 `/vimg`，应当能看到索引列表，例如 `#1 09-03 14:22 这是柜门`。
3. 隔一轮后问关于这张图的问题，例如「刚才那张柜门是什么颜色」。
4. 若模型正确调用了 `recall_image` 并描述了图片内容，说明链路打通。

`/vimg_clear` 可以清空当前会话的索引与图片文件。

## 补充说明

- 只能看到**本插件安装之后**发送的图片，更早的历史图片没有索引。
- 索引按会话隔离（`unified_msg_origin`），群 A 的图不会出现在群 B。
- 关键词检索匹配的是图片发送时**伴随的文字**。如果用户发图时没配文字，只能靠序号定位，这也是注入索引时要带上序号的原因。
- 每次召回都会把图片重新送进上下文，会消耗视觉 token。图片多或体积大时，把 `max_image_edge` 调小一些。
- 图片文件会占用磁盘，靠 `max_images_per_session` 和 `keep_days` 控制，清理每小时最多触发一次。
- **主模型必须支持视觉，否则会失效。**
- **GIF 动图默认会被插件自动抽帧拼成接触印片**（按 `gif_frame_count` 等间隔采样，按 `cols×cols` 网格拼接）。如果你更想让模型直接看原 GIF，把 `gif_as_contact_sheet` 设为 `false`；需要看更多/更少帧时，用 `recall_gif_frames` 工具并传 `frame_count`。抽帧 / 拼图依赖 Pillow。
