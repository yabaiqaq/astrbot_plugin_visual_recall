"""visual_recall —— 让 Agent 按需回看会话中的历史图片。

三段式工作：

1. 捕获：任何消息进来时，把其中的图片复制一份到插件数据目录并记入索引。
   AstrBot 核心为事件创建的临时文件在事件结束后就会被清理，必须自己留档。
2. 告知：Agent 每轮开始前，把本会话的图片索引以 system 消息注入，
   模型才知道「有图可看」以及每张图的序号。
3. 召回：模型判断需要看图时调用 recall_image，工具返回真实图片，
   AstrBot 的 runner 会把图片追加成一条 user 消息，模型于是真正看到它。

依赖：本目录有 __init__.py，因此是正经 Python 包；内部 import 都用
相对路径（from .tools / from .storage），由 AstrBot 加载器把插件当作
astrbot_plugin_visual_recall 包来 import。
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from pathlib import Path

from astrbot import logger  # noqa: E402
from astrbot.api.event import AstrMessageEvent, filter  # noqa: E402
from astrbot.api.star import Context, Star, StarTools, register  # noqa: E402
from astrbot.core.agent.message import Message, TextPart  # noqa: E402
from astrbot.core.agent.run_context import ContextWrapper  # noqa: E402
from astrbot.core.astr_agent_context import AstrAgentContext  # noqa: E402
from astrbot.core.message.components import Image  # noqa: E402

from .storage import ImageStore, guess_mime  # noqa: E402
from .tools import RecallGifFramesTool, RecallImageTool  # noqa: E402

_ALLOWED_SUFFIX = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")


@register(
    "visual_recall",
    "yabaiqaq",
    "让 Agent 根据语境按需回看会话中的历史图片",
    "1.0.0",
)
class VisualRecallPlugin(Star):
    def __init__(self, context: Context, config=None) -> None:
        super().__init__(context)
        self.conf = config or {}
        self._ready = False
        self._init_task = None
        self._last_cleanup = 0.0

        self.data_dir = self._resolve_data_dir()
        self.img_dir = self.data_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.store = ImageStore(self.data_dir / "index.db")

        # 插件加载时通常已在事件循环内；不在的话延迟到首次使用时再建任务
        try:
            self._init_task = asyncio.create_task(self._boot())
        except RuntimeError:
            self._init_task = None

        self.context.add_llm_tools(RecallImageTool(store=self.store, conf=self.conf))
        self.context.add_llm_tools(RecallGifFramesTool(store=self.store, conf=self.conf))

    # ------------------------------------------------------------------ 配置

    def _cfg(self, key: str, default):
        try:
            value = self.conf.get(key, default)  # type: ignore[union-attr]
        except Exception:
            return default
        return default if value is None else value

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("enabled", True))

    @property
    def inject_index(self) -> bool:
        return bool(self._cfg("inject_index", True))

    @property
    def index_entries(self) -> int:
        try:
            return max(1, min(int(self._cfg("index_entries", 8)), 30))
        except (TypeError, ValueError):
            return 8

    def _resolve_data_dir(self) -> Path:
        try:
            return StarTools.get_data_dir("visual_recall")
        except Exception:
            fallback = Path.cwd() / "data" / "plugin_data" / "visual_recall"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    async def _boot(self) -> None:
        # 依赖自检：Pillow 缺失会让 GIF 抽帧 / 图片压缩全部失效，
        # 这里在插件加载时就给出明确告警，避免功能静默退化。
        try:
            import PIL  # noqa: F401

            _pillow_ok = True
        except ImportError:
            _pillow_ok = False
        if not _pillow_ok:
            logger.warning(
                "[visual_recall] 未检测到 Pillow！图片压缩、GIF 自动抽帧、"
                "recall_gif_frames 将不可用。请执行 `pip install \"Pillow>=10.0.0\"` "
                "(或 `pip install -r requirements.txt`) 后重载插件。"
            )
        try:
            await self.store.init()
            self._ready = True
        except Exception as e:
            logger.error(f"[visual_recall] 索引初始化失败：{e}")

    async def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        if self._init_task is None:
            self._init_task = asyncio.create_task(self._boot())
        await self._init_task
        return self._ready

    # ---------------------------------------------------------------- 捕获图片

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def _capture(self, event: AstrMessageEvent) -> None:
        """把消息里的图片留档并记入索引。

        这里刻意不使用 create_task：AstrBot 会在事件结束后清理临时媒体文件，
        异步拖到后面再复制，文件可能已经不存在了。
        """
        if not self.enabled:
            return
        if not await self._ensure_ready():
            return

        try:
            components = event.get_messages() or []
        except Exception:
            return

        images = [c for c in components if isinstance(c, Image)]
        if not images:
            return

        umo = event.unified_msg_origin
        sender = str(event.get_sender_id() or "")
        caption = (event.get_message_str() or "").strip()

        for img in images:
            try:
                await self._persist(umo, sender, caption, img)
            except Exception as e:
                logger.debug(f"[visual_recall] 处理图片时出错：{e}")

        await self._maybe_cleanup()

    async def _persist(
        self, umo: str, sender: str, caption: str, img: Image
    ) -> None:
        try:
            src = await asyncio.wait_for(img.convert_to_file_path(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("[visual_recall] 图片落地超时，已跳过")
            return
        except Exception as e:
            logger.debug(f"[visual_recall] 无法取得图片本地路径：{e}")
            return

        src_path = Path(src)
        try:
            if not src_path.is_file():
                return
            size = src_path.stat().st_size
        except OSError:
            return
        if size == 0:
            return

        max_bytes = 0
        try:
            max_bytes = int(self._cfg("max_store_mb", 15)) * 1024 * 1024
        except (TypeError, ValueError):
            max_bytes = 0
        if max_bytes > 0 and size > max_bytes:
            logger.debug(f"[visual_recall] 图片超过 {max_bytes} 字节，跳过索引")
            return

        suffix = src_path.suffix.lower()
        if suffix not in _ALLOWED_SUFFIX:
            suffix = ".jpg"

        name = f"{self._hash(umo)}_{int(time.time() * 1000)}_{id(img) % 10000}{suffix}"
        dest = self.img_dir / name
        try:
            await asyncio.to_thread(shutil.copyfile, src_path, dest)
        except Exception as e:
            logger.warning(f"[visual_recall] 复制图片失败：{e}")
            return

        seq = await self.store.add(
            umo, sender, str(dest), caption, guess_mime(dest), size
        )
        logger.debug(f"[visual_recall] 已索引 #{seq} <- {dest.name}")

    @staticmethod
    def _hash(umo: str) -> str:
        return hashlib.md5(umo.encode("utf-8", errors="ignore")).hexdigest()[:10]

    # ---------------------------------------------------------------- 注入索引

    @filter.on_agent_begin()
    async def _inject_index(
        self,
        event: AstrMessageEvent,
        run_context: ContextWrapper[AstrAgentContext],
    ) -> None:
        """让模型知道本会话有哪些历史图片可看。

        没有这一段，模型压根不知道有图可召回，也就不会去调用工具。
        """
        if not self.enabled or not self.inject_index:
            return
        if not await self._ensure_ready():
            return

        umo = event.unified_msg_origin
        try:
            records = await self.store.recent(umo, self.index_entries)
        except Exception as e:
            logger.debug(f"[visual_recall] 读取索引失败：{e}")
            return
        if not records:
            return

        try:
            total = await self.store.count(umo)
        except Exception:
            total = len(records)

        lines = [
            "[视觉记忆] 本会话此前出现过图片，需要时可以回看：",
            f"共 {total} 张，最近 {len(records)} 张如下（# 后为序号）：",
        ]
        for r in reversed(records):
            lines.append(f"  #{r.seq}  {r.time_str}  {r.short_caption}")
        lines.append(
            "当用户的问题涉及某张图片的内容时，调用 recall_image 查看它："
            "用 seq 精确定位，或用 query 关键词检索。"
            "与图片无关的问题不要调用。"
        )

        self._append_system(run_context, "\n".join(lines))

    @staticmethod
    def _append_system(run_context: ContextWrapper[AstrAgentContext], text: str) -> None:
        """把索引文本拼进 system 消息，避免打断 user/assistant 的交替结构。"""
        messages = run_context.messages
        part = TextPart(text=text)

        if messages:
            head = messages[0]
            if getattr(head, "role", None) == "system":
                content = head.content
                if isinstance(content, list):
                    content.append(part)
                else:
                    head.content = [TextPart(text=str(content or "")), part]
                return

        messages.insert(0, Message(role="system", content=[part]))

    # -------------------------------------------------------------------- 清理

    async def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < 3600:
            return
        self._last_cleanup = now

        try:
            max_per_session = int(self._cfg("max_images_per_session", 60))
        except (TypeError, ValueError):
            max_per_session = 60
        try:
            max_days = float(self._cfg("keep_days", 7))
        except (TypeError, ValueError):
            max_days = 7

        try:
            removed = await self.store.cleanup(max_per_session, max_days)
        except Exception as e:
            logger.warning(f"[visual_recall] 清理索引失败：{e}")
            return

        for p in removed:
            try:
                await asyncio.to_thread(Path(p).unlink)
            except Exception:
                pass
        if removed:
            logger.info(f"[visual_recall] 已清理 {len(removed)} 张过期图片")

    # -------------------------------------------------------------------- 命令

    @filter.command("vimg")
    async def _cmd_list(self, event: AstrMessageEvent):
        """查看本会话已索引的图片，用于确认插件是否正常工作。"""
        if not await self._ensure_ready():
            yield event.plain_result("索引尚未就绪，请查看日志。")
            return

        umo = event.unified_msg_origin
        records = await self.store.recent(umo, 10)
        if not records:
            yield event.plain_result("本会话还没有索引到任何图片。")
            return

        total = await self.store.count(umo)
        lines = [f"本会话共 {total} 张，最近 {len(records)} 张："]
        for r in reversed(records):
            lines.append(f"#{r.seq}  {r.time_str}  {r.short_caption}")
        yield event.plain_result("\n".join(lines))

    @filter.command("vimg_clear")
    async def _cmd_clear(self, event: AstrMessageEvent):
        """清空本会话的图片索引与文件。"""
        if not await self._ensure_ready():
            yield event.plain_result("索引尚未就绪。")
            return

        umo = event.unified_msg_origin
        try:
            removed = await self.store.clear_session(umo)
        except Exception as e:
            logger.warning(f"[visual_recall] 清空失败：{e}")
            yield event.plain_result("清空失败，请查看日志。")
            return

        for p in removed:
            try:
                await asyncio.to_thread(Path(p).unlink)
            except Exception:
                pass
        yield event.plain_result(f"已清空本会话的 {len(removed)} 张图片记录。")

    async def terminate(self) -> None:
        """插件卸载时收尾。"""
        if self._init_task is not None and not self._init_task.done():
            self._init_task.cancel()
