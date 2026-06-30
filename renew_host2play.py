import os, sys, logging, random, json, time, re, html, tempfile, base64, asyncio
from pathlib import Path
from datetime import datetime

# ★ 修复：CPU 版 PyTorch 不支持 BFloat16 / mixed dtype 矩阵乘法
# 错误形式1: "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"
# 错误形式2: "mixed dtype (CPU): expect parameter to have scalar type of Float"
# 根本原因：Botright recognizer 用了 torch.amp.autocast('cuda')，在无 GPU 的 CPU 上
#           某些 tensor 被 cast 到 BFloat16，但 CPU 线性层权重是 Float32，两者不兼容。
try:
    import torch
    import torch.nn as nn

    class _NoopAutocast:
        """完全禁用 autocast，强制所有计算保持 float32"""
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __call__(self, func): return func

    # 覆盖所有 autocast 入口
    torch.autocast = _NoopAutocast
    if hasattr(torch, 'amp'):
        if hasattr(torch.amp, 'autocast'):
            torch.amp.autocast = _NoopAutocast
        if hasattr(torch.amp, 'GradScaler'):
            # GradScaler 在 CPU 上也会引起问题，替换为 noop
            class _NoopGradScaler:
                def __init__(self, *a, **kw): pass
                def scale(self, loss): return loss
                def step(self, optimizer, *a, **kw): optimizer.step()
                def update(self): pass
                def unscale_(self, optimizer): pass
            torch.amp.GradScaler = _NoopGradScaler
            torch.cuda.amp.GradScaler = _NoopGradScaler

    # 替换 torch.load：加载权重时把所有 BFloat16 tensor 转为 float32
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        result = _orig_torch_load(*args, **kwargs)
        def _cast(obj):
            if isinstance(obj, torch.Tensor):
                return obj.float() if obj.dtype == torch.bfloat16 else obj
            if isinstance(obj, dict):
                return {k: _cast(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                casted = [_cast(v) for v in obj]
                return type(obj)(casted)
            return obj
        return _cast(result)
    torch.load = _patched_torch_load

    # ★ 关键新增：给 nn.Module 注册全局 forward pre-hook
    # 在每次 forward 调用前，把输入 tensor 中的 BFloat16 强制转为 float32
    # 这样即使模型权重已经是 float32，传入的激活值如果被 autocast 转成 BFloat16 也会被纠正
    def _cast_inputs_to_float(module, args):
        new_args = []
        for a in args:
            if isinstance(a, torch.Tensor) and a.dtype == torch.bfloat16:
                new_args.append(a.float())
            else:
                new_args.append(a)
        return tuple(new_args)
    torch.nn.modules.module.register_module_forward_pre_hook(_cast_inputs_to_float)

    # 全局默认 dtype float32
    torch.set_default_dtype(torch.float32)

    logging.getLogger(__name__).info("✅ torch BFloat16 修复已启用（autocast noop + load cast + forward hook + default float32）")
except ImportError:
    pass

# ★ 新增：monkeypatch recognizer 库的 handle_recaptcha
# 原库逻辑：点 VERIFY 提交后，无论是 "Please also check new images"（还有新格子没选）
# 还是 "Incorrect, try again"（答案选错），统统走 load_captcha(reset=True) → 点击
# #recaptcha-reload-button 整题作废重开。但 check_new 的真实含义是"答案没错，只是
# 动态题里又冒出几张新图没勾选"，正确做法应该是原地再跑一次 detect_tiles() 把新格子
# 补上再提交一次，而不是把已经选对的进度全部扔掉重开一轮（reload还会跟外层脚本自己
# 的"全局看门狗"竞态，导致 DOM 中途被两边同时操作，出现 images amount must equal 9
# or 16. Is: 0 这种识图失败）。
try:
    from recognizer.agents.playwright.async_control import AsyncChallenger as _AsyncChallenger
    from playwright.async_api import TimeoutError as _PWTimeoutError

    async def _patched_handle_recaptcha(self):
        if isinstance(loaded_captcha := await self.load_captcha(), str):
            return loaded_captcha

        captcha_frame = self.page.frame_locator("//iframe[contains(@src,'bframe')]")
        label_obj = captcha_frame.locator("//strong")
        if not (prompt := await label_obj.text_content()):
            raise ValueError("reCaptcha Task Text did not load.")

        for _ in range(30):
            recaptcha_tiles = await captcha_frame.locator("[class='rc-imageselect-tile']").all()
            tiles_visibility = [await tile.is_visible() for tile in recaptcha_tiles]
            if len(recaptcha_tiles) in (9, 16) and len(tiles_visibility) in (9, 16):
                break
            await self.page.wait_for_timeout(1000)
        else:
            await self.load_captcha(captcha_frame, reset=True)
            return await self.handle_recaptcha()

        area_captcha = len(recaptcha_tiles) == 16
        result_clicked = await self.detect_tiles(prompt, area_captcha)

        if self.dynamic and not area_captcha:
            while result_clicked:
                await self.page.wait_for_timeout(5000)
                result_clicked = await self.detect_tiles(prompt, area_captcha)
        elif not result_clicked:
            await self.load_captcha(captcha_frame, reset=True)
            return await self.handle_recaptcha()

        try:
            submit_button = captcha_frame.locator("#recaptcha-verify-button")
            await submit_button.click()
        except _PWTimeoutError:
            await self.load_captcha(captcha_frame, reset=True)
            return await self.handle_recaptcha()

        for _ in range(5):
            if captcha_token := await self.check_result():
                return captcha_token
            await self.page.wait_for_timeout(1000)

        # ★ 区分 check_new（漏选，需要补选）vs 真正 incorrect（选错，需要reset）
        check_new_el = captcha_frame.locator(
            ".rc-imageselect-error-dynamic-more, .rc-imageselect-error-select-more"
        )
        try:
            is_check_new = await check_new_el.first.is_visible(timeout=500)
        except Exception:
            is_check_new = False

        if is_check_new:
            # 原地再做一轮识图补选新出现的格子，不reset、不丢已选进度
            await self.detect_tiles(prompt, area_captcha)
            return await self.handle_recaptcha()

        incorrect = self.page.locator("[class='rc-imageselect-incorrect-response']")
        errors = self.page.locator("[class *= 'rc-imageselect-error']")
        if await incorrect.is_visible() or any([await error.is_visible() for error in await errors.all()]):
            await self.load_captcha(captcha_frame, reset=True)

        return await self.handle_recaptcha()

    _AsyncChallenger.handle_recaptcha = _patched_handle_recaptcha
    logging.getLogger(__name__).info("✅ recognizer.handle_recaptcha 已patch：check_new 改为原地补选，不再reset整题")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ★ 立即输出一行，让 GitHub Actions 日志流马上建立（否则长时间无输出导致步骤无法展开）
print("===== renew_host2play.py 启动中 =====", flush=True)

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    log.warning("requests 未安装")

# ==============================================================================
# 配置
# ==============================================================================
_TOKEN1 = os.environ.get("RENEW_TOKEN_1", "")
_TOKEN2 = os.environ.get("RENEW_TOKEN_2", "")

RENEW_URLS = [
    f"https://host2play.gratis/server/renew?i={t}&hl=en"
    for t in [_TOKEN1, _TOKEN2]
    if t
]

if not RENEW_URLS:
    raise SystemExit("❌ 未配置任何 RENEW_TOKEN_*，请在 GitHub Secrets 中添加 RENEW_TOKEN_1 和 RENEW_TOKEN_2")

# 代理：Xray 本地 SOCKS5
PROXY_SERVER = "socks5://127.0.0.1:10808"
log.info("🌐 使用 Xray SOCKS5 代理（127.0.0.1:10808）")

SCREENSHOT_DIR = Path("./screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

MAX_CAPTCHA_ATTEMPTS = 3



# ==============================================================================
# WxPusher 推送（可选）
# ==============================================================================
WXPUSHER_TOKEN = os.environ.get("WXPUSHER_TOKEN", "")
WXPUSHER_UID   = os.environ.get("WXPUSHER_UID", "")

def wxpush(content: str):
    if not WXPUSHER_TOKEN or not WXPUSHER_UID:
        log.warning("📨 WXPUSHER_TOKEN 或 WXPUSHER_UID 未配置，跳过推送")
        return
    import urllib.request
    payload = json.dumps({
        "appToken": WXPUSHER_TOKEN,
        "content":  content,
        "contentType": 1,
        "uids": [WXPUSHER_UID],
    }).encode()
    try:
        req = urllib.request.Request(
            "https://wxpusher.zjiecode.com/api/send/message",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("success"):
                log.info("📨 WxPusher 推送成功")
            else:
                log.warning(f"📨 WxPusher 推送失败: {result}")
    except Exception as e:
        log.warning(f"📨 WxPusher 推送异常: {e}")

# ==============================================================================
# 工具函数
# ==============================================================================

# ==============================================================================
async def take_screenshot(page, name, blocking=False):
    """
    blocking=False（默认）：fire-and-forget，不阻塞主流程，超时3s。
    blocking=True：等待截图完成（用于关键截图）。
    """
    async def _do():
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(SCREENSHOT_DIR / f"{ts}_{name}.png")
            await page.screenshot(path=path, full_page=False, timeout=3000)
            log.info(f"📸 截图: {path}")
        except Exception as e:
            log.debug(f"截图失败（已跳过）: {e}")
    if blocking:
        await _do()
    else:
        asyncio.ensure_future(_do())

async def get_text(page) -> str:
    try:
        return await page.inner_text("body") or ""
    except:
        return ""

async def human_delay(min_s=0.5, max_s=1.2):
    await asyncio.sleep(random.uniform(min_s, max_s))

async def bezier_mouse_move(page, target_x: float, target_y: float, steps: int = 20):
    """
    用贝塞尔曲线模拟人类鼠标轨迹移动到目标坐标。
    控制点随机偏移，让轨迹看起来像真人操作。
    """
    try:
        # 获取当前鼠标大致位置（随机起点）
        start_x = random.uniform(100, 900)
        start_y = random.uniform(100, 600)
        # 随机控制点（贝塞尔曲线弯曲程度）
        cp1x = start_x + random.uniform(-200, 200)
        cp1y = start_y + random.uniform(-150, 150)
        cp2x = target_x + random.uniform(-100, 100)
        cp2y = target_y + random.uniform(-80, 80)

        for i in range(steps + 1):
            t = i / steps
            # 三次贝塞尔公式
            x = ((1-t)**3 * start_x + 3*(1-t)**2*t * cp1x +
                 3*(1-t)*t**2 * cp2x + t**3 * target_x)
            y = ((1-t)**3 * start_y + 3*(1-t)**2*t * cp1y +
                 3*(1-t)*t**2 * cp2y + t**3 * target_y)
            await page.mouse.move(x, y)
            await asyncio.sleep(random.uniform(0.005, 0.025))
    except Exception as e:
        log.debug(f"贝塞尔移动失败（忽略）: {e}")
        try:
            await page.mouse.move(target_x, target_y)
        except:
            pass

async def is_cf_blocked(page) -> bool:
    try:
        body = (await get_text(page)).lower()
        return "verify you are human" in body or ("cloudflare" in body and "security" in body)
    except:
        return False

async def wait_cf_pass(page, timeout=60) -> bool:
    log.info("等待 Cloudflare 验证自动通过...")
    for i in range(timeout):
        if not await is_cf_blocked(page):
            log.info(f"✅ Cloudflare 验证通过（{i}s）")
            return True
        if i % 5 == 0 and i > 0:
            log.info(f"  CF 等待中... {i}s")
        await asyncio.sleep(1)
    log.error(f"Cloudflare 验证超时（{timeout}s）")
    return False

async def navigate(page, url, timeout=60) -> bool:
    log.info(f"导航到: {url}")
    try:
        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
    except Exception as e:
        log.warning(f"goto 超时/异常: {e}，继续等待...")
    if not await is_cf_blocked(page):
        return True
    if await wait_cf_pass(page, timeout=timeout):
        return True
    log.info("CF 未过，刷新重试...")
    try:
        await page.reload(wait_until="domcontentloaded", timeout=30000)
    except:
        pass
    return await wait_cf_pass(page, timeout=30)

async def read_delete_date(page) -> str | None:
    try:
        text = await get_text(page)
        m = re.search(r'Deletes on[:\s]*(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})', text)
        if m:
            return m.group(1).strip()
        m2 = re.search(r'Deletes on[:\s]*(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', text)
        if m2:
            return m2.group(1).strip()
    except Exception as e:
        log.warning(f"读取到期时间失败: {e}")
    return None

# ==============================================================================
# ★ 新增：通用看门狗包装器
# ------------------------------------------------------------------------------
# 背景：CDP/代理偶发异常（如 gzip abort）会让连接处于"半残"状态——既不报错也不
# 返回，导致 page.evaluate() / mouse.click() / bounding_box() 这类没有自带超时的
# 协议级调用直接卡死数分钟（曾观察到卡住 8 分 22 秒），而外层 try/except 完全
# 捕获不到（因为它根本没有抛异常，只是没返回）。
# 用 asyncio.wait_for 强制给这些调用加超时，超时就当作失败处理，绝不无限期等待。
# ==============================================================================
class WatchdogTimeout(Exception):
    """标记一次操作被看门狗强制中断（而非业务逻辑本身的异常）"""
    pass

async def with_watchdog(coro, timeout: float, label: str = ""):
    """
    给任意协程加硬超时。超时后抛出 WatchdogTimeout，调用方据此判断是否需要
    page.reload() / 跳过本次尝试，而不是傻等。
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        log.warning(f"  [看门狗] ⚠️ 操作超时（{timeout}s）: {label or '未命名操作'}，强制放弃等待")
        raise WatchdogTimeout(f"{label} 超过 {timeout}s 未完成") from None

async def safe_reload(page, timeout_ms: int = 20000, label: str = "") -> bool:
    """
    安全地刷新页面：goto 自身的 timeout 参数有时在 CDP 半残状态下也不可靠，
    所以额外用看门狗包一层硬超时，双重保险。
    """
    try:
        await with_watchdog(
            page.reload(wait_until="domcontentloaded", timeout=timeout_ms),
            timeout=(timeout_ms / 1000) + 10,
            label=f"safe_reload {label}",
        )
        return True
    except WatchdogTimeout:
        return False
    except Exception as e:
        log.warning(f"  [safe_reload] {label} 异常: {e}")
        return False

async def close_ads(page):
    """
    关闭各种广告弹窗和遮挡层。
    先尝试点击关闭按钮，再用 JS 强制移除无法关闭的覆盖层。
    """
    # 1. 尝试常见关闭按钮
    close_selectors = [
        "[aria-label='Close']", "[aria-label='close']", "[aria-label='Dismiss']",
        ".close-btn", ".ad-close", ".popup-close", ".modal-close",
        "button:has-text('Close')", "button:has-text('×')", "button:has-text('✕')",
        "a:has-text('Close')",       # Google Vignette/Survey 关闭链接（<a> 标签，非 <button>）
        "button:has-text('OPEN')",   # 购物广告的 OPEN 按钮（点掉就关闭了）
        ".dismiss", "[data-dismiss]", ".overlay-close",
    ]
    for sel in close_selectors:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=400):
                # ★ click() 无自带超时，CDP半残时会无限挂起，加看门狗硬超时
                await with_watchdog(btn.click(timeout=5000), timeout=8, label=f"close_ads click {sel}")
                log.info(f"  [close_ads] 关闭广告: {sel}")
                await asyncio.sleep(0.3)
        except:
            pass

    # 2. JS 强制移除所有高 z-index 的遮挡层（购物广告、iframe 广告等）
    try:
        # ★ page.evaluate() 没有自带超时参数，CDP半残/代理卡顿时会无限期挂起
        # （曾实测卡住超过8分钟），用看门狗硬超时兜底
        removed = await with_watchdog(page.evaluate("""() => {
            let removed = 0;
            const isProtected = (el) => {
                // 保护 recaptcha / swal / cmp / gdpr 相关元素，以及它们的任意祖先
                let cur = el;
                while (cur && cur !== document.body) {
                    const id = cur.id || '';
                    const cls = (cur.className && typeof cur.className === 'string') ? cur.className : '';
                    if (id.includes('recaptcha') || cls.includes('recaptcha') ||
                        cls.includes('swal') || id.includes('swal') ||
                        cls.includes('cmp') || cls.includes('gdpr') ||
                        id.includes('google_vignette')) return true;
                    cur = cur.parentElement;
                }
                return false;
            };
            const els = document.querySelectorAll(
                'div[class*="overlay"], div[class*="popup"], div[class*="modal"], ' +
                'div[class*="ad"], div[class*="banner"], div[class*="promo"], ' +
                'div[class*="sticky"], div[class*="bottom-bar"], div[class*="footer-ad"], ' +
                'iframe[id*="ad"], iframe[name*="ad"], ' +
                '[id*="ad-container"], [id*="ad_container"], [id*="adsense"]'
            );
            const removed_els = [];
            for (const el of els) {
                if (isProtected(el)) continue;
                const style = window.getComputedStyle(el);
                const pos = style.position;
                const z = parseInt(style.zIndex) || 0;
                // ★ 修复：isFloating 必须同时要求 z-index > 0，防止误删 fixed 的 swal/recaptcha 容器
                const isFloating = (pos === 'fixed' || pos === 'sticky') && z > 0;
                if ((z > 100 && (pos === 'fixed' || pos === 'absolute')) || isFloating) {
                    removed_els.push(el.tagName + (el.id ? '#'+el.id : '') + (el.className && typeof el.className === 'string' ? '.'+el.className.trim().split(/[\s]+/).join('.') : ''));
                    el.remove();
                    removed++;
                }
            }
            // 把删了哪些元素也返回出来，方便调试
            return {count: removed, els: removed_els.slice(0, 20)};
        }"""), timeout=8, label="close_ads JS evaluate")
        count = removed.get("count", 0) if isinstance(removed, dict) else removed
        els_info = removed.get("els", []) if isinstance(removed, dict) else []
        if count > 0:
            log.info(f"  [close_ads] JS 强制移除了 {count} 个遮挡层: {els_info}")
        else:
            log.debug("  [close_ads] JS 未移除任何遮挡层")
    except WatchdogTimeout:
        log.warning("  [close_ads] JS evaluate 超时，跳过遮挡层清理")
    except Exception as e:
        log.debug(f"  [close_ads] JS 移除失败: {e}")

    # ★ 补充：专门关闭底部视频广告条（不符合高z-index条件但会拦截点击）
    # 截图确认：host2play 页面底部经常出现第三方视频广告 iframe 条
    bottom_ad_selectors = [
        "div[style*='position: fixed'][style*='bottom']",
        "div[style*='position:fixed'][style*='bottom']",
        "[id*='adngin'], [id*='adthrive'], [id*='mediavine']",
        "[class*='video-ad'], [class*='sticky-ad'], [class*='bottom-ad']",
        "div[data-ad-unit], div[data-ad-slot]",
    ]
    for _ad_sel in bottom_ad_selectors:
        try:
            _ad_el = page.locator(_ad_sel).first
            if await _ad_el.is_visible(timeout=300):
                _handle = await with_watchdog(_ad_el.element_handle(timeout=3000), timeout=6, label="bottom_ad element_handle")
                await with_watchdog(page.evaluate("(el) => el.remove()", _handle), timeout=6, label="bottom_ad remove")
                log.info(f"  [close_ads] 底部广告条已移除: {_ad_sel}")
        except:
            pass

    await asyncio.sleep(0.5)

    # ★ 兜底：检测 Google Vignette / Survey 广告（URL fragment 或覆盖弹窗）
    await dismiss_google_vignette(page)

# ==============================================================================
# ★ 新增：关闭 Google Vignette / Survey 广告
# ==============================================================================
async def dismiss_google_vignette(page):
    """
    检测并关闭 Google Vignette / Survey 广告。

    处理策略（优先级从高到低）：
      1. 直接点击右上角 Close 按钮（<a> 标签，主框架元素，不在 iframe 内）
         → 最优方案：页面状态完全保留，swal2/reCAPTCHA 不受影响
      2. 若 Close 点击失败 且 URL 含 #google_vignette fragment
         → fallback：goto 去掉 fragment（会重载页面，调用方需处理后续状态）
         → 返回 "GOTO_RESET" 哨兵，区别于 True（调用方知道页面已重载）

    为什么优先点 Close 而不是 goto：
      - goto 会重载整个页面，swal2 弹窗和 reCAPTCHA iframe 全消失
      - 直接点 Close 是最轻量的方式，完全不影响页面其他元素
      - 截图确认：Close 是主框架的 <a> 文字链接，Playwright 可直接点击
    """
    try:
        cur_url = page.url
        _has_vignette_fragment = "#google_vignette" in cur_url or "#google_survey" in cur_url

        # ── 优先策略：直接点击 Close 按钮（无论是否有 fragment，先试点击）──
        vignette_close_selectors = [
            "a:has-text('Close')",              # 最常见：右上角纯文字 <a> 链接（截图确认）
            "a[href='#'][class*='close']",    # 带 # href 的关闭链接
            "[id='survey-close']",
            "[class*='survey-close']",
            "a.survey-close",
            "[aria-label='Close survey']",
            "button[aria-label='Close survey']",
        ]
        for sel in vignette_close_selectors:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=400):
                    await with_watchdog(btn.click(timeout=5000), timeout=8, label=f"vignette click {sel}")
                    log.info(f"  [vignette] ✅ 点击 Close 关闭 Vignette: {sel}")
                    await asyncio.sleep(1)
                    return True
            except:
                pass

        # ── Fallback：Close 点击失败，且 URL 含 fragment → goto 重载 ──
        if _has_vignette_fragment:
            clean_url = cur_url.split("#")[0]
            log.warning(f"  [vignette] Close 按钮未找到，fallback goto: {clean_url}")
            await page.goto(clean_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            log.info("  [vignette] ✅ Vignette 已跳过（goto fallback，页面已重载）")
            return "GOTO_RESET"   # 区别于 True：调用方知道页面已重载

        return False
    except Exception as e:
        log.debug(f"  [vignette] dismiss 异常（忽略）: {e}")
        return False


# ==============================================================================
# ★ 新增：关闭 GDPR Cookie 同意弹窗（荷兰语 CMP）
# ==============================================================================
async def close_gdpr_consent(page) -> bool:
    """
    检测并关闭荷兰语 GDPR Cookie 同意弹窗（CMP）。
    弹窗特征：标题含 "Welkom" / "toestemming"，按钮为 "Toestemming" 或 "×"。
    返回 True 表示弹窗已关闭或不存在。
    ★ 修复：点击前先注入 consent cookie，防止点击按钮触发 CMP 回调阻塞页面主线程。
    """
    # ★ 优先：直接写 consent cookie，让 CMP 认为用户已同意
    # 这样即使后面再点按钮，CMP 回调检测到 cookie 存在会直接跳过耗时初始化
    try:
        await page.evaluate("""() => {
            const expires = new Date(Date.now() + 365*24*3600*1000).toUTCString();
            // Quantcast CMP / SourcePoint 常见 consent cookie
            const cookiePairs = [
                ['euconsent-v2', 'consent_given'],
                ['eupubconsent-v2', 'consent_given'],
                ['sp_lit', '1'],
                ['cmapi_cookie_privacy', 'permit 1,2,3'],
                ['CookieConsent', 'true'],
                ['cookieconsent_status', 'allow'],
                ['gdpr_consent', '1'],
            ];
            for (const [name, val] of cookiePairs) {
                document.cookie = `${name}=${val}; expires=${expires}; path=/; domain=.host2play.gratis`;
                document.cookie = `${name}=${val}; expires=${expires}; path=/`;
            }
            // 如果 CMP 把同意状态存在 localStorage 里
            try { localStorage.setItem('CookieConsent', 'true'); } catch(e) {}
            try { localStorage.setItem('gdpr_consent', '1'); } catch(e) {}
        }""")
        log.info("  [GDPR] 已注入 consent cookie（防止点击按钮阻塞主线程）")
        await asyncio.sleep(0.2)
    except Exception as _ce:
        log.debug(f"  [GDPR] cookie 注入失败（忽略）: {_ce}")

    # 先滚动到顶部，确保弹窗按钮在视口内可点击
    try:
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.3)
    except:
        pass

    # 优先尝试"Toestemming"（同意）按钮，因为拒绝可能导致页面功能受限
    # 调试：打印当前所有可见按钮文字，帮助定位新 CMP 按钮
    try:
        visible_btns = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('button'))
                .filter(b => b.offsetParent !== null)
                .map(b => (b.innerText || b.textContent || '').trim())
                .filter(t => t.length > 0 && t.length < 50);
        }""")
        if visible_btns:
            log.info(f"  [GDPR调试] 当前可见按钮: {visible_btns}")
    except Exception as dbg_e:
        log.debug(f"  [GDPR调试] 获取按钮失败: {dbg_e}")

    consent_selectors = [
        "button:has-text('Toestemming')",       # 荷兰语：同意
        "button:has-text('Akkoord')",            # 荷兰语：接受
        "button:has-text('Accepteren')",         # 荷兰语：接受
        "button:has-text('Consent')",            # 英语：同意（locale=en-US 时出现）
        "button:has-text('Accept')",             # 英语兜底
        "button:has-text('Accept all')",
        "button:has-text('Agree')",              # 英语：同意
        "button:has-text('I agree')",
        "button:has-text('Alle akkoord')",
        "[aria-label='Close']",
        "[aria-label='Sluiten']",               # 荷兰语：关闭
        ".cmp-close-button",
        ".sp_choice_type_11",                   # SourcePoint CMP 关闭按钮
        ".message-close-button",
    ]

    for sel in consent_selectors:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=800):
                await btn.click()
                log.info(f"✅ 关闭 GDPR 弹窗，点击: {sel}")
                # ★ 等待按钮从 DOM 中消失，确认弹窗真正关闭（最多等 3s）
                try:
                    await btn.wait_for(state="hidden", timeout=3000)
                    log.info("  弹窗按钮已从 DOM 消失，确认关闭成功")
                except:
                    pass
                await asyncio.sleep(0.5)
                return True
        except:
            pass

    # 尝试关闭按钮 "×"（右上角）
    try:
        close_btn = page.locator(".sp_choice_type_12, .close-button, button.close").first
        if await close_btn.is_visible(timeout=500):
            await close_btn.click()
            log.info("✅ 关闭 GDPR 弹窗（× 按钮）")
            await asyncio.sleep(0.5)
            return True
    except:
        pass

    # 截图中可见的 × 按钮（右上角 aria-label 不明确）
    try:
        for close_text in ["×", "✕", "✖", "Close", "Sluiten"]:
            btn = page.get_by_role("button", name=close_text).first
            if await btn.is_visible(timeout=300):
                await btn.click()
                log.info(f"✅ 关闭 GDPR 弹窗（按钮文本: {close_text}）")
                await asyncio.sleep(0.5)
                return True
    except:
        pass

    log.info("未检测到 GDPR 弹窗（或已关闭）")
    return True  # 不存在弹窗也算"已处理"


async def wait_gdpr_gone(page, timeout=15) -> bool:
    """
    等待 GDPR 弹窗真正消失。
    ★ 修复：改用 JS DOM 可见性检测（按钮是否还在页面上可见），
       不再用 await get_text() 文字检测（会误判，因为 DOM 文字即使弹窗消失也可能残留）。
    """
    for i in range(timeout):
        try:
            still_visible = await page.evaluate("""() => {
                // 1. 检查同意/关闭按钮是否还可见
                var consentTexts = ['Toestemming','Akkoord','Accepteren','Consent',
                                    'Accept','Accept all','Agree','I agree',
                                    'Manage options','Manage preferences'];
                var btns = Array.from(document.querySelectorAll('button'));
                for (var b of btns) {
                    var txt = (b.innerText || b.textContent || '').trim();
                    if (consentTexts.includes(txt) && b.offsetParent !== null) {
                        return true;
                    }
                }
                // 2. 检查常见 CMP 弹窗容器 class/id
                var cmpSelectors = [
                    '.sp-message-container', '.sp_message_iframe',
                    '[id*="sp_message"]', '.fc-dialog-container',
                    '.gdpr-dialog', '.cmp-popup', '.message-container',
                    '[class*="cmp-"]', '[id*="cmp-"]',
                    '[class*="consent"]', '[id*="consent"]',
                    '[class*="cookie-banner"]', '[id*="cookie-banner"]',
                    '[class*="privacy-"]', '[id*="privacy-"]'
                ];
                for (var sel of cmpSelectors) {
                    var el = document.querySelector(sel);
                    if (el && el.offsetParent !== null && el.offsetHeight > 50) {
                        return true;
                    }
                }
                // 3. ★ 通用兜底：检测覆盖全屏的大型 overlay（z-index高且面积>屏幕1/4）
                var allDivs = Array.from(document.querySelectorAll('div'));
                for (var d of allDivs) {
                    var st = window.getComputedStyle(d);
                    var pos = st.position;
                    var z = parseInt(st.zIndex) || 0;
                    var h = d.offsetHeight;
                    var w = d.offsetWidth;
                    if ((pos === 'fixed' || pos === 'absolute') 
                        && z > 50
                        && h > window.innerHeight * 0.3
                        && w > window.innerWidth * 0.3
                        && d.offsetParent !== null) {
                        // 排除 reCAPTCHA 和 SweetAlert2
                        var id = d.id || '';
                        var cls = (typeof d.className === 'string') ? d.className : '';
                        if (!id.includes('recaptcha') && !cls.includes('swal')
                            && !cls.includes('sweet') && !id.includes('swal')) {
                            return true;
                        }
                    }
                }
                return false;
            }""")
            if not still_visible:
                log.info(f"✅ GDPR 弹窗已真正消失（DOM检测, {i}s）")
                return True
            else:
                log.info(f"  GDPR 弹窗仍可见（{i}s），再次尝试关闭...")
                await close_gdpr_consent(page)
        except Exception as e:
            log.warning(f"wait_gdpr_gone 检测异常: {e}")
        await asyncio.sleep(1)
    log.warning(f"⚠️ GDPR 弹窗 {timeout}s 内未消失")
    return False

# ==============================================================================
# reCAPTCHA 辅助
# ==============================================================================
async def find_recaptcha_frame(page, kind: str):
    """查找包含 kind（'anchor' 或 'bframe'）的 reCAPTCHA frame"""
    try:
        for frame in page.frames:
            if "recaptcha" in frame.url and kind in frame.url:
                return frame
    except Exception:
        pass
    return None

async def is_recaptcha_solved(page) -> bool:
    """检查 reCAPTCHA 是否已通过"""
    # ★ 修复：不用 page.evaluate（在 Cloudflare 页面容易挂起），改用 locator
    try:
        resp_loc = page.locator('textarea[name="g-recaptcha-response"]')
        if await resp_loc.count() > 0:
            val = await resp_loc.first.input_value(timeout=1000)
            if val and len(val) > 10:
                return True
    except:
        pass
    anchor = await find_recaptcha_frame(page, "anchor")
    if anchor:
        try:
            val = await anchor.locator("#recaptcha-anchor").get_attribute("aria-checked", timeout=1000)
            if val == "true":
                return True
        except:
            pass
    return False

async def is_image_challenge_present(page) -> bool:
    """检测是否出现了图片挑战（bframe 存在且可见）"""
    bframe = await find_recaptcha_frame(page, "bframe")
    if not bframe:
        return False
    try:
        fl = page.frame_locator("iframe[src*='recaptcha'][src*='bframe']")
        return await fl.locator(".rc-imageselect").is_visible(timeout=1000)
    except:
        return False

async def is_ip_blocked(page) -> bool:
    """检测 IP 是否被 Google 封锁（Try again later）"""
    bframe = await find_recaptcha_frame(page, "bframe")
    if not bframe:
        return False
    try:
        fl = page.frame_locator("iframe[src*='recaptcha'][src*='bframe']")
        header = fl.locator(".rc-doscaptcha-header-text").first
        if await header.is_visible(timeout=500):
            text = await header.inner_text()
            if "try again later" in text.lower():
                return True
    except:
        pass
    return False

# ==============================================================================
# reCAPTCHA：强制切换为英语界面（修复 recognizer 不识别荷兰语标签的问题）
# ==============================================================================
async def force_recaptcha_english(page) -> bool:
    """
    将页面内所有 reCAPTCHA iframe 的语言参数强制改为英语（hl=en）。

    背景：host2play.gratis 使用荷兰语界面，reCAPTCHA 挑战词因此显示为荷兰语
    （如 fietsen / bussen / auto's），而 recognizer 的模型只识别英语标签
    （bicycle / bus / car），导致 "label not yet scheduled" 错误连续失败。

    方案：在点击 checkbox 之前，通过 JS 把所有 reCAPTCHA iframe 的 src 里
    的 hl 参数替换为 en，触发 iframe 重新加载，Google 服务端会按新语言返回
    英语挑战词，recognizer 即可正常识别。
    """
    log.info("强制切换 reCAPTCHA 语言为英语（hl=en）...")
    try:
        result = await page.evaluate("""() => {
            let changed = 0;
            document.querySelectorAll('iframe[src*="recaptcha"]').forEach(iframe => {
                const oldSrc = iframe.src;
                // 提取当前 hl 参数值，方便日志
                const hlMatch = oldSrc.match(/[?&]hl=([^&]+)/);
                const curHl = hlMatch ? hlMatch[1] : '(无hl参数)';
                // 不管是什么语言，统一强制替换/追加为 hl=en
                let newSrc;
                if (/[?&]hl=/.test(oldSrc)) {
                    newSrc = oldSrc.replace(/([?&]hl=)[^&]+/, '$1en');
                } else {
                    newSrc = oldSrc + (oldSrc.includes('?') ? '&' : '?') + 'hl=en';
                }
                // 只要目标不是 en 就切换（包括无 hl 参数的情况）
                if (curHl !== 'en') {
                    iframe.src = newSrc;
                    changed++;
                    console.log('[hl切换] ' + curHl + ' → en');
                }
            });
            return changed;
        }""")
        if result and result > 0:
            log.info(f"✅ 已将 {result} 个 reCAPTCHA iframe 切换为英语，等待重新加载...")
            # ★ 不用固定 sleep，改为主动等待：旧 anchor iframe detach 后新 iframe 出现
            # 固定 3s 不够：有时旧 iframe 还在，_wait_anchor_stable 检测到旧的就返回
            # 导致 _click_checkbox 拿到正在重载的新 iframe，checkbox 一直不 visible
            old_anchor_url = None
            try:
                for f in page.frames:
                    if "recaptcha" in f.url and "anchor" in f.url:
                        old_anchor_url = f.url
                        break
            except:
                pass
            # 等旧 anchor iframe detach（最多 8s）
            for _w in range(16):
                await asyncio.sleep(0.5)
                still_old = any(
                    f.url == old_anchor_url
                    for f in page.frames
                    if not f.is_detached()
                )
                if not still_old:
                    log.info(f"  旧 anchor iframe 已 detach（{(_w+1)*0.5:.1f}s），等待新 iframe...")
                    break
            await asyncio.sleep(1)  # 给新 iframe 时间开始加载
            return True
        else:
            log.info("  未找到需要切换语言的 reCAPTCHA iframe（可能已是英语或尚未加载）")
            return False
    except Exception as e:
        log.warning(f"切换 reCAPTCHA 语言失败: {e}")
        return False


# ==============================================================================
# reCAPTCHA：普通模式（仅点击 checkbox）
# ==============================================================================
async def _cdp_click_in_bframe(page, selector: str, timeout_ms: int = 2000) -> bool:
    """
    用 CDP 真实鼠标事件点击 reCAPTCHA bframe iframe 内的按钮。

    原理：
      1. 用 page.locator("iframe[src*=bframe]").bounding_box() 拿 iframe 在页面的绝对位置
      2. 用 frame_locator + locator.bounding_box() 拿按钮在 iframe 内的相对位置
      3. 两者相加得到按钮在页面视口的绝对坐标
      4. 发 CDP mouseMoved → mousePressed → mouseReleased

    注意：frame_locator 上的 bounding_box() 返回的是 iframe 内坐标，
    必须加上 iframe 自身的页面偏移才能得到正确的绝对坐标。
    """
    try:
        # 1. iframe 在页面上的绝对位置
        iframe_el = page.locator("iframe[src*='recaptcha'][src*='bframe']")
        iframe_box = await iframe_el.bounding_box()
        if not iframe_box:
            return False

        # 2. 按钮在 iframe 内的相对位置
        fl = page.frame_locator("iframe[src*='recaptcha'][src*='bframe']")
        btn = fl.locator(selector)
        if not await btn.is_visible(timeout=timeout_ms):
            return False
        btn_box = await btn.bounding_box()
        if not btn_box:
            return False

        # 3. 换算页面绝对坐标
        abs_x = iframe_box['x'] + btn_box['x'] + btn_box['width'] / 2
        abs_y = iframe_box['y'] + btn_box['y'] + btn_box['height'] / 2

        # 4. CDP 三步序列
        cdp = await page.context.new_cdp_session(page)
        try:
            await cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseMoved", "x": abs_x, "y": abs_y,
                "button": "none", "buttons": 0, "clickCount": 0, "modifiers": 0,
            })
            await asyncio.sleep(0.05)
            await cdp.send("Input.dispatchMouseEvent", {
                "type": "mousePressed", "x": abs_x, "y": abs_y,
                "button": "left", "buttons": 1, "clickCount": 1, "modifiers": 0,
            })
            await asyncio.sleep(0.1)
            await cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseReleased", "x": abs_x, "y": abs_y,
                "button": "left", "buttons": 0, "clickCount": 1, "modifiers": 0,
            })
        finally:
            await cdp.detach()
        return True
    except Exception as _e:
        log.debug(f"  [_cdp_click_in_bframe] {selector} 点击失败: {_e}")
        return False


async def _click_checkbox(page):
    """点击 anchor frame 内的 reCAPTCHA checkbox"""

    # ★ 每次轮询都重新获取 frame_locator，避免 iframe detach/重建后旧引用挂死
    def _fresh_checkbox():
        return page.frame_locator("iframe[src*='recaptcha'][src*='anchor']").locator("#recaptcha-anchor")

    log.info("尝试点击 reCAPTCHA checkbox（第1次）...")
    checkbox_visible = False
    for _wi in range(45):
        # 每次拿新引用，彻底避免旧 frame 引用挂死
        checkbox = _fresh_checkbox()
        try:
            # asyncio.wait_for 兜底：万一 is_visible 内部挂死，2s 强制超时
            visible = await asyncio.wait_for(
                checkbox.is_visible(timeout=800),
                timeout=2.0
            )
            if visible:
                checkbox_visible = True
                log.info(f"  [checkbox] 第{_wi}s visible=True，准备点击")
                break
            else:
                log.debug(f"  [checkbox] 第{_wi}s visible=False")
        except asyncio.TimeoutError:
            log.warning(f"  [checkbox] 第{_wi}s is_visible 超时（2s），iframe 可能正在重建，重新获取引用...")
        except Exception as _e:
            log.debug(f"  [checkbox] 第{_wi}s 异常: {type(_e).__name__}: {_e}")

        # 同时检查 iframe 是否存在、是否 detached
        try:
            frames_info = [(f.url[:80], f.is_detached()) for f in page.frames if "recaptcha" in f.url and "anchor" in f.url]
            if frames_info:
                log.debug(f"  [checkbox] anchor frames: {frames_info}")
            else:
                log.debug(f"  [checkbox] 第{_wi}s 未找到 anchor iframe（共{len(page.frames)}个frame）")
        except Exception:
            pass
        # 边等边检查是否已经直接通过（极少数情况）
        if await is_recaptcha_solved(page):
            log.info("✅ checkbox 等待中检测到已直接通过")
            return
        await asyncio.sleep(1)

    if not checkbox_visible:
        # ★ checkbox 超时前先检查是否还有 CMP/广告遮挡，尝试再关一遍
        log.warning("⚠️ checkbox 45s 内未 visible，检查是否有 CMP 弹窗残留...")
        try:
            still_blocked = await page.evaluate("""() => {
                var consentTexts = ['Toestemming','Akkoord','Accepteren','Consent',
                                    'Accept','Accept all','Agree','I agree',
                                    'Manage options','Manage preferences'];
                var btns = Array.from(document.querySelectorAll('button'));
                for (var b of btns) {
                    var txt = (b.innerText || b.textContent || '').trim();
                    if (consentTexts.includes(txt) && b.offsetParent !== null) return true;
                }
                var allDivs = Array.from(document.querySelectorAll('div'));
                for (var d of allDivs) {
                    var st = window.getComputedStyle(d);
                    if ((st.position === 'fixed' || st.position === 'absolute')
                        && (parseInt(st.zIndex)||0) > 50
                        && d.offsetHeight > window.innerHeight * 0.3
                        && d.offsetWidth  > window.innerWidth  * 0.3
                        && d.offsetParent !== null) {
                        var id = d.id || '';
                        var cls = (typeof d.className === 'string') ? d.className : '';
                        if (!id.includes('recaptcha') && !cls.includes('swal')) return true;
                    }
                }
                return false;
            }""")
            if still_blocked:
                log.warning("  仍有遮挡层，再次执行 close_gdpr_consent + close_ads...")
                await close_gdpr_consent(page)
                await asyncio.sleep(1)
                await close_ads(page)
                await asyncio.sleep(1)
                # 再等 10s 看 checkbox 是否出现
                for _ri in range(10):
                    try:
                        if await checkbox.is_visible(timeout=1000):
                            checkbox_visible = True
                            break
                    except:
                        pass
                    await asyncio.sleep(1)
        except Exception as _ce:
            log.warning(f"  CMP 残留检测异常: {_ce}")
        if not checkbox_visible:
            raise RuntimeError("reCAPTCHA checkbox 超时：页面可能被 CMP/广告遮挡，请检查截图")

    for attempt in range(3):
        try:
            if attempt > 0:
                log.info(f"尝试点击 reCAPTCHA checkbox（第{attempt+1}次）...")

            # ★ 方法1：直接用 frame 对象的 locator.click()
            # 比 CDP 坐标点击更可靠，frame 内部自己处理坐标系
            anchor_frame = await find_recaptcha_frame(page, "anchor")
            if anchor_frame and not anchor_frame.is_detached():
                try:
                    cb = anchor_frame.locator("#recaptcha-anchor")
                    if await asyncio.wait_for(cb.is_visible(timeout=1000), timeout=2.0):
                        log.info(f"  [click] frame.locator('#recaptcha-anchor').click()")
                        await asyncio.wait_for(
                            cb.click(timeout=3000),
                            timeout=5.0
                        )
                        log.info("✅ 已点击 reCAPTCHA checkbox（frame.locator click）")
                        log.info("  等待 reCAPTCHA 响应（solved 或 图片挑战）...")
                        return
                except asyncio.TimeoutError:
                    log.warning(f"  方法1 超时，尝试 CDP 坐标点击...")
                except Exception as _e1:
                    log.warning(f"  方法1 失败: {_e1}，尝试 CDP 坐标点击...")

            # ★ 方法2：CDP 坐标点击（anchor iframe偏移 + 内部坐标）
            # frame_locator().bounding_box() 返回的是页面绝对坐标，可以直接用
            anchor_iframe_el = page.locator("iframe[src*='recaptcha'][src*='anchor']")
            try:
                iframe_box = await asyncio.wait_for(
                    anchor_iframe_el.bounding_box(),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                log.warning(f"第{attempt+1}次：anchor iframe bounding_box 超时，重试...")
                await asyncio.sleep(1)
                continue

            if not iframe_box:
                log.warning(f"第{attempt+1}次：anchor iframe bounding_box 为空，跳过")
                await asyncio.sleep(1)
                continue

            # checkbox 在 anchor iframe 内的相对位置
            fl = page.frame_locator("iframe[src*='recaptcha'][src*='anchor']")
            try:
                cb_box = await asyncio.wait_for(
                    fl.locator("#recaptcha-anchor").bounding_box(),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                cb_box = None

            if cb_box:
                # ★ frame_locator().bounding_box() 返回的已经是页面绝对坐标，不需要加 iframe 偏移
                cx = cb_box['x'] + cb_box['width'] / 2
                cy = cb_box['y'] + cb_box['height'] / 2
                log.info(f"  [方法2] cb 绝对坐标({cx:.0f},{cy:.0f}) iframe=({iframe_box['x']:.0f},{iframe_box['y']:.0f})")
            else:
                # 兜底：点 iframe 中心
                cx = iframe_box['x'] + iframe_box['width'] / 2
                cy = iframe_box['y'] + iframe_box['height'] / 2
                log.warning(f"  [方法2] cb_box 为空，点 iframe 中心 ({cx:.0f},{cy:.0f})")

            _cdp = await asyncio.wait_for(page.context.new_cdp_session(page), timeout=5.0)
            try:
                log.info(f"  [CDP] dispatchMouseEvent checkbox ({cx:.0f},{cy:.0f})")
                await asyncio.wait_for(_cdp.send("Input.dispatchMouseEvent", {
                    "type": "mouseMoved", "x": cx, "y": cy,
                    "button": "none", "buttons": 0, "clickCount": 0, "modifiers": 0,
                }), timeout=3.0)
                await asyncio.sleep(0.05)
                await asyncio.wait_for(_cdp.send("Input.dispatchMouseEvent", {
                    "type": "mousePressed", "x": cx, "y": cy,
                    "button": "left", "buttons": 1, "clickCount": 1, "modifiers": 0,
                }), timeout=3.0)
                await asyncio.sleep(random.uniform(0.08, 0.15))
                await asyncio.wait_for(_cdp.send("Input.dispatchMouseEvent", {
                    "type": "mouseReleased", "x": cx, "y": cy,
                    "button": "left", "buttons": 0, "clickCount": 1, "modifiers": 0,
                }), timeout=3.0)
            finally:
                try:
                    await asyncio.wait_for(_cdp.detach(), timeout=2.0)
                except Exception:
                    pass
            log.info("✅ 已点击 reCAPTCHA checkbox（CDP 方法2）")
            log.info("  等待 reCAPTCHA 响应（solved 或 图片挑战）...")
            return
        except Exception as e:
            log.warning(f"第{attempt+1}次点击失败: {e}")
            if attempt < 2:
                await asyncio.sleep(1)
    raise RuntimeError("reCAPTCHA checkbox 点击全部失败")

async def try_simple_recaptcha(page, wait_secs=20) -> bool:
    """
    普通模式：点击 checkbox 后等待直接变绿勾。
    若通过 → True；若出现图片挑战 → False。
    """
    log.info("【普通模式】点击 checkbox，等待直接通过...")
    try:
        await _click_checkbox(page)
    except Exception as e:
        log.warning(f"点击 checkbox 失败: {e}")
        return False

    for i in range(wait_secs):
        if await is_recaptcha_solved(page):
            log.info(f"✅ 普通模式通过（{i}s）")
            return True
        if await is_image_challenge_present(page):
            log.info(f"  出现图片挑战（{i}s），转 recognizer 模式")
            return False
        await asyncio.sleep(1)

    if await is_recaptcha_solved(page):
        log.info("✅ 普通模式通过（超时后检测）")
        return True
    log.info("普通模式未通过，转 recognizer 模式")
    return False

# ==============================================================================
# ★ 新增：recognizer 图片挑战识别
# ==============================================================================

async def _wait_anchor_stable(page, label="", timeout=20) -> bool:
    """
    等待 reCAPTCHA anchor iframe 真正渲染完毕：
    1. anchor iframe 存在且未 detached
    2. #recaptcha-anchor is_visible
    3. anchor iframe 实际宽度 >= 200px（排除尺寸为零的假可见状态）
    4. checkbox bounding_box 高度 >= 20px（确认内容已渲染）
    """
    for i in range(timeout):
        if i % 3 == 0:
            try:
                await close_ads(page)
            except Exception as _ad_e:
                log.debug(f"  [anchor等待] 广告清除异常（忽略）: {_ad_e}")
        frame = await find_recaptcha_frame(page, "anchor")
        if frame and not frame.is_detached():
            try:
                fl = page.frame_locator("iframe[src*='recaptcha'][src*='anchor']")
                cb = fl.locator("#recaptcha-anchor")
                if not await cb.is_visible(timeout=800):
                    log.debug(f"  [anchor等待] {i}s: checkbox not visible")
                    await asyncio.sleep(1)
                    continue

                # ★ 检查 anchor iframe 实际尺寸，排除尺寸为零的假可见
                anchor_iframe_el = page.locator("iframe[src*='recaptcha'][src*='anchor']")
                try:
                    iframe_box = await asyncio.wait_for(
                        anchor_iframe_el.bounding_box(), timeout=2.0
                    )
                except Exception:
                    iframe_box = None

                if not iframe_box or iframe_box['width'] < 200 or iframe_box['height'] < 30:
                    log.debug(f"  [anchor等待] {i}s: iframe 尺寸太小 {iframe_box}，等待渲染...")
                    await asyncio.sleep(1)
                    continue

                # ★ 检查 checkbox 自身的 bounding_box 高度
                try:
                    cb_box = await asyncio.wait_for(cb.bounding_box(), timeout=2.0)
                except Exception:
                    cb_box = None

                if not cb_box or cb_box['height'] < 20:
                    log.debug(f"  [anchor等待] {i}s: checkbox bounding_box 太小 {cb_box}，等待渲染...")
                    await asyncio.sleep(1)
                    continue

                log.info(f"✅ anchor iframe 已稳定{label}（{i}s）iframe={iframe_box['width']:.0f}x{iframe_box['height']:.0f} cb={cb_box['width']:.0f}x{cb_box['height']:.0f}")
                return True
            except Exception as _e:
                log.debug(f"  [anchor等待] {i}s 异常: {_e}")
        else:
            log.debug(f"  [anchor等待] {i}s: anchor frame 不存在或已 detached")
        await asyncio.sleep(1)
    return False


async def solve_recaptcha(page, url: str = "") -> bool:
    """
    reCAPTCHA 解决策略：
      1. 确认 GDPR 弹窗已消失
      2. 等待 anchor iframe 首次完全稳定（checkbox 可见）
         ※ hl=en 已由 add_init_script 在 iframe 创建时注入，无需事后重载
      3. 直接交给 Botright page.solve_recaptcha() 全程接管（已移除手动点 checkbox 路径）
    """
    # ★ 注册 reCAPTCHA 网络活动监听器（只注册一次，page级别去重）
    # 原理：recognizer 每次"换题"(reload)或"提交答案"(userverify)都会真实
    # 打一次 Google 接口，这是 Botright 在认真干活的客观证据，比"格子数量是否
    # 变化"这种 DOM 层面的弱信号更可靠（4×4一次性识别+提交，绝大部分时间
    # 格子数本来就不变，不能靠它判断是否卡死）
    if not hasattr(page, "_recaptcha_activity"):
        page._recaptcha_activity = {"last_ts": asyncio.get_event_loop().time()}

        def _on_recaptcha_request(request):
            try:
                url = request.url
                if ("google.com" in url or "recaptcha.net" in url) and (
                    "reload" in url or "userverify" in url
                ):
                    page._recaptcha_activity["last_ts"] = asyncio.get_event_loop().time()
            except Exception:
                pass

        page.on("request", _on_recaptcha_request)

    # 步骤1：确认 GDPR 已消失
    log.info("solve_recaptcha: 确认 GDPR 弹窗已消失...")
    if not await wait_gdpr_gone(page, timeout=5):
        log.warning("  GDPR 仍在，再次强制关闭...")
        await close_gdpr_consent(page)
        await asyncio.sleep(2)

    # 步骤2：等待 anchor iframe 首次稳定（最多 20s）
    log.info("等待 reCAPTCHA anchor iframe 首次加载稳定...")
    if not await _wait_anchor_stable(page, label="（初始）", timeout=5):
        log.warning("⚠️ anchor iframe 5s 内未真正渲染，执行 grecaptcha.reset() 重试...")
        await take_screenshot(page, "recaptcha_anchor_timeout")
        try:
            await page.evaluate("grecaptcha.reset()")
            log.info("  grecaptcha.reset() 已执行，再等 5s...")
        except Exception as _re:
            log.warning(f"  reset 失败: {_re}")
        if not await _wait_anchor_stable(page, label="（reset后初始）", timeout=5):
            log.error("reCAPTCHA anchor frame reset 后仍超时，放弃")
            await take_screenshot(page, "recaptcha_anchor_timeout2")
            return False
    await take_screenshot(page, "recaptcha_anchor_stable")

    # 步骤3（已移除）：hl=en 由 add_init_script 在 iframe 创建时直接注入，无需事后重载

    # 步骤4：直接交给 Botright 全程接管（已移除手动点 checkbox 的路径1，
    # 避免污染 checkbox 状态/浪费时间，recognizer 已可识别英语标签）
    await take_screenshot(page, f"recaptcha_before_botright")
    log.info("Botright page.solve_recaptcha() 接管（最多重试3次）...")

    async def _do_botright_attempt(attempt_no: int) -> bool:
        """单次 Botright 解题尝试，带 80s 硬超时。"""
        # ★ 每次尝试前，用 gc 找到所有存活的 nn.Module 实例
        # 把权重转 float32，并注册 forward pre-hook 拦截 BFloat16 激活值
        # 注意：sys.modules 里是"模块文件"不是 nn.Module 实例，必须用 gc
        try:
            import torch, gc
            _hooks = []
            def _cast_bf16_inputs(module, args):
                return tuple(
                    a.float() if isinstance(a, torch.Tensor) and a.dtype == torch.bfloat16 else a
                    for a in args
                )
            cast_count = 0
            for obj in gc.get_objects():
                if isinstance(obj, torch.nn.Module):
                    try:
                        # 把所有参数和 buffer 转为 float32
                        for p in list(obj.parameters(recurse=False)):
                            if p.data.dtype == torch.bfloat16:
                                p.data = p.data.float()
                        for b in list(obj.buffers(recurse=False)):
                            if b.dtype == torch.bfloat16:
                                obj._buffers[
                                    next(k for k, v in obj._buffers.items() if v is b)
                                ] = b.float()
                        # 注册 hook 拦截推理时的 BFloat16 输入
                        h = obj.register_forward_pre_hook(_cast_bf16_inputs)
                        _hooks.append(h)
                        cast_count += 1
                    except Exception:
                        pass
            if cast_count:
                log.info(f"  [尝试{attempt_no}] 已对 {cast_count} 个 nn.Module 注册 float32 hook")
        except Exception as _e:
            log.debug(f"  [尝试{attempt_no}] float32 hook 注册跳过: {_e}")
        # reset 让 Google 重新出题
        try:
            await page.evaluate("grecaptcha.reset()")
            log.info(f"  [尝试{attempt_no}] grecaptcha.reset() 已执行")
            await asyncio.sleep(2)
            await _wait_anchor_stable(page, label=f"（reset后尝试{attempt_no}）", timeout=5)
        except Exception as e:
            log.warning(f"  [尝试{attempt_no}] reset 失败（{e}），继续...")

        # ★ Fix1：reset 后额外等待 bframe 出现，确保 Botright 接管时 reCAPTCHA 已完全初始化
        # 原问题：_wait_anchor_stable 只检测 anchor iframe，但 solve_recaptcha() 需要 bframe
        # 若 bframe 未加载，Botright 会空等长达300s直到超时
        log.info(f"  [尝试{attempt_no}] 等待 reCAPTCHA bframe 加载（最多15s）...")
        _bframe_ready = False
        for _bi in range(15):
            try:
                _bframe_count = await page.locator("iframe[src*='recaptcha'][src*='bframe']").count()
                if _bframe_count > 0:
                    # bframe 存在，再等 1s 让内容渲染完
                    await asyncio.sleep(1)
                    _bframe_ready = True
                    log.info(f"  [尝试{attempt_no}] ✅ bframe 已出现（{_bi}s）")
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        if not _bframe_ready:
            # bframe 15s 内没出现，说明 Google 没有弹图片挑战（可能直接通过或状态异常）
            # 检查是否已通过，若未通过则继续（Botright 会自己点 checkbox 触发弹窗）
            if await is_recaptcha_solved(page):
                log.info(f"  [尝试{attempt_no}] ✅ bframe未出现但 reCAPTCHA 已通过（checkbox直接变绿）")
                return True
            log.warning(f"  [尝试{attempt_no}] ⚠️ bframe 15s 内未出现，让 Botright 自行触发挑战...")

        # 滚动到 bframe 确保 Botright 渲染可见
        try:
            await page.evaluate("""() => {
                const f = Array.from(document.querySelectorAll('iframe'))
                    .find(f => f.src && f.src.includes('recaptcha') && f.src.includes('bframe'));
                if (f) f.scrollIntoView({block: 'center', behavior: 'instant'});
                else window.scrollTo(0, document.body.scrollHeight / 3);
            }""")
            await asyncio.sleep(0.5)
        except Exception as e:
            log.warning(f"  [尝试{attempt_no}] 滚动失败（{e}），继续...")

        async def _get_challenge_status() -> str:
            """
            检测当前挑战状态：
            - 'try_again'  : "Please try again" — Google 刁难，Botright 继续做下一题
            - 'check_new'  : "Please also check the new images" — 有漏选，继续找
            - 'select_all' : "Please select all matching images" — 没选就提交，自动提交兜底
            - '4x4'        : 16格大图，reset 重开
            - '3x3'        : 正常3×3动态挑战
            - 'unknown'    : 无法判断
            """
            try:
                fl = page.frame_locator("iframe[src*='recaptcha'][src*='bframe']")
                # ★ 扩大错误提示选择器覆盖范围，避免漏检
                # 实际观察到红字 "Please select all matching images." 由多种class承载
                err_selectors = [
                    ".rc-imageselect-error-select-more",
                    ".rc-imageselect-incorrect-response",
                    ".rc-imageselect-error-dynamic-more",
                    # 兜底：bframe内任何红色错误提示文本
                    "[class*='error']:visible",
                ]
                for sel in err_selectors:
                    try:
                        err_el = fl.locator(sel)
                        if await err_el.first.is_visible(timeout=300):
                            err_text = (await err_el.first.inner_text()).lower().strip()
                            if not err_text:
                                continue
                            log.debug(f"  [status] 检测到错误文本({sel}): '{err_text}'")
                            if "try again" in err_text:
                                return "try_again"
                            if "check" in err_text and ("new" in err_text or "image" in err_text):
                                return "check_new"
                            # ★ 扩大匹配：只要含 "select" 就认为是没选够
                            if "select" in err_text:
                                return "select_all"
                    except:
                        continue
                if await fl.locator(".rc-imageselect-table-44").is_visible(timeout=400):
                    return "4x4"
                if await fl.locator(".rc-imageselect-table-33").is_visible(timeout=400):
                    return "3x3"
            except:
                pass
            return "unknown"

        async def _count_checked_tiles() -> int:
            """统计当前已勾选的格子数"""
            try:
                fl = page.frame_locator("iframe[src*='recaptcha'][src*='bframe']")
                checked = fl.locator(".rc-imageselect-tile.rc-imageselect-tileselected")
                return await checked.count()
            except:
                return 0

        async def _has_loading_tiles() -> bool:
            """检测是否有格子正在刷新加载中（动态3×3换图期间）"""
            try:
                fl = page.frame_locator("iframe[src*='recaptcha'][src*='bframe']")
                # 动态挑战刷新时格子会有 rc-imageselect-dynamic-selected 或 loading class
                loading = fl.locator(
                    ".rc-imageselect-tile.rc-imageselect-dynamic-selected,"
                    ".rc-imageselect-tile .rc-imageselect-progress"
                )
                if await loading.count() > 0:
                    return True
                # 备用：检测格子内图片是否还在加载（src 为空或 blob 未完成）
                tiles = fl.locator(".rc-imageselect-tile")
                count = await tiles.count()
                for i in range(count):
                    try:
                        tile = tiles.nth(i)
                        # 如果格子有 dynamic-selected class 说明刚被选中、图片还在刷入
                        cls = await tile.get_attribute("class") or ""
                        if "dynamic-selected" in cls:
                            return True
                    except:
                        continue
                return False
            except:
                return False

        async def _click_verify_button() -> bool:
            """主动点击 VERIFY 按钮，点前等待所有格子加载完毕"""
            try:
                fl = page.frame_locator("iframe[src*='recaptcha'][src*='bframe']")
                btn = fl.locator("#recaptcha-verify-button")
                if not await btn.is_visible(timeout=1000):
                    return False

                # ★ 点VERIFY前，等待所有动态刷新的格子加载完（最多等5s）
                for _ in range(10):
                    if not await _has_loading_tiles():
                        break
                    log.info(f"  [自动提交] 等待动态格子加载完毕...")
                    await asyncio.sleep(0.5)

                # 再等一点点，确保图片渲染完成，避免提交时机太早
                await asyncio.sleep(0.8)

                await btn.click()
                log.info(f"  [自动提交] ✅ 主动点击 VERIFY 按钮")
                return True
            except:
                pass
            return False

        try:
            # ★ 不做预点击，直接让 Botright 从干净状态接管整个流程
            # 原因：预点击会改变 checkbox 状态（或触发 Google 静默拒绝导致被重置），
            # Botright 内部有自己的点击逻辑，干净状态下它能正确处理点击→弹挑战→识图全流程
            # ★ 提高 recognizer 内部 retry_times（默认15），4×4 区域挑战识别准确率较低，
            # 容易在单次 attempt 内因递归重试次数耗尽而提前触发 RecursionError，
            # 调大后能把更多时间花在真正识图重试上，而不是被打断重启整轮 Botright 接管
            try:
                page.recaptcha_solver.retry_times = 40
            except Exception as _rt_e:
                log.debug(f"  [尝试{attempt_no}] 设置 retry_times 失败（不影响主流程）: {_rt_e}")

            log.info(f"  [尝试{attempt_no}] 调用 page.solve_recaptcha()（500s超时）...")

            # ★ 新增：快速熔断 —— 监听 stdout，捕获 recognizer 库打印的
            # "[ERROR] Images amount must equal 9 or 16. Is: 0" 这条特征错误。
            # 该错误通常出现在 DOM 中途被刷新/换题（比如手动reload按钮点击与
            # Botright自身任务竞态）导致 recognizer 截图时拿到空白/残留页面，
            # 此后只会反复刷同一条错误，原来要等90s全局看门狗才会强制reload，
            # 现在只要短时间内连续出现2次就立刻提前触发reload，不用死等90秒。
            class _ImgCountWatcher:
                def __init__(self, real):
                    self._real = real
                    self.count = 0
                    self.last_ts = 0.0
                def write(self, s):
                    self._real.write(s)
                    if "Images amount must equal" in s:
                        self.count += 1
                        self.last_ts = asyncio.get_event_loop().time()
                def flush(self):
                    self._real.flush()
            _img_watcher = _ImgCountWatcher(sys.stdout)
            _old_stdout = sys.stdout
            sys.stdout = _img_watcher

            solve_task = asyncio.create_task(page.solve_recaptcha())
            bad_challenge = False
            _do_botright_attempt._4x4_reloads = 0  # 每次新attempt重置4×4换题计数
            # ★ 修复：从90s延长到150s，因为try_again后Botright需要继续做第二轮图片挑战
            # 第一轮约50s + try_again等待约5s + 第二轮约50s = 约105s，90s根本不够
            # ★ 再延长到500s：retry_times调大后单次attempt内部重试更多轮，实测5轮try_again约210s，需要足够窗口
            deadline = asyncio.get_event_loop().time() + 500
            _attempt_start_time = asyncio.get_event_loop().time()  # ★ Fix2：记录attempt开始时间
            last_status = ""
            _last_checked_count = 0
            _checked_stable_since = 0.0
            _last_auto_verify_time = 0.0
            # ★ 新增：全局无进展看门狗。status 检测在 DOM 异常（如图片0张/socket hang up
            # 导致 recognizer 网络拦截失败）时可能在 "3x3"/"unknown" 之间反复跳变，
            # 使依赖"连续两次status相同"的局部watchdog永远不触发，只能干等到500s硬超时。
            # 这里不看 status 是否稳定，只看"checked_count 或 status 有没有发生过任何变化"，
            # 90s 内完全没有任何变化 → 判定整个挑战已死锁，直接 page.reload() 强制刷新页面，
            # 拿到全新的 DOM/iframe，而不是让 Botright 继续对着旧状态空转。
            _global_progress_ts = asyncio.get_event_loop().time()
            _global_last_status = ""
            _global_last_count = -1

            while not solve_task.done():
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    solve_task.cancel()
                    log.warning(f"  [尝试{attempt_no}] ⏰ 500s 超时，reset 重新开局")
                    bad_challenge = True
                    break

                # ★ URL 守卫：广告跳转检测
                # Botright 做题时可能点到广告 iframe 跳到第三方网站
                # 一旦离开 host2play.gratis 立即 cancel，重新导航回来
                # ★ 修复：同时检测 #google_vignette fragment（域名不变，原守卫漏掉）
                try:
                    _cur_url = page.url
                    # ── 情形A：Google Vignette/Survey 广告（URL带fragment，域名未变）──
                    if "#google_vignette" in _cur_url or "#google_survey" in _cur_url:
                        log.warning(f"  [尝试{attempt_no}] ⚠️ solve_recaptcha 期间检测到 Google Vignette 广告！")
                        log.warning(f"  [尝试{attempt_no}]   当前 URL: {_cur_url[:120]}")
                        solve_task.cancel()
                        # ★ 优先点 Close 按钮（保留页面状态，swal2/reCAPTCHA 不丢失）
                        # 若 Close 不可见则 fallback goto（页面重载，返回 VIGNETTE_RESET）
                        _vr = await dismiss_google_vignette(page)
                        if _vr is True:
                            # Close 点击成功，页面状态保留，正常 reset 重试即可
                            log.info(f"  [尝试{attempt_no}] ✅ Vignette Close 点击成功，页面状态保留，继续重试")
                            try:
                                await close_gdpr_consent(page)
                                await wait_gdpr_gone(page, timeout=5)
                            except:
                                pass
                            bad_challenge = True
                            break
                        else:
                            # goto fallback：页面已重载，需要重新点击 Renew server
                            try:
                                await close_gdpr_consent(page)
                                await wait_gdpr_gone(page, timeout=5)
                                await close_ads(page)
                                log.info(f"  [尝试{attempt_no}] ✅ Vignette goto fallback 完成，需要重新点击 Renew server")
                            except Exception as _nav_e:
                                log.warning(f"  [尝试{attempt_no}] Vignette 后清理失败: {_nav_e}")
                            return "VIGNETTE_RESET"
                    # ── 情形B：页面跳离 host2play.gratis（第三方网站）──
                    elif "host2play.gratis" not in _cur_url:
                        log.warning(f"  [尝试{attempt_no}] ⚠️ 检测到页面已跳离 host2play.gratis！")
                        log.warning(f"  [尝试{attempt_no}]   当前 URL: {_cur_url[:120]}")
                        solve_task.cancel()
                        # 关闭所有可能被打开的新标签页
                        try:
                            for _pg in page.context.pages[1:]:
                                await _pg.close()
                        except:
                            pass
                        # 重新导航回续期页面
                        log.info(f"  [尝试{attempt_no}] 重新导航回 {url} ...")
                        try:
                            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                            await asyncio.sleep(2)
                            await close_gdpr_consent(page)
                            await wait_gdpr_gone(page, timeout=5)
                            await close_ads(page)
                        except Exception as _nav_e:
                            log.warning(f"  [尝试{attempt_no}] 重新导航失败: {_nav_e}")
                        bad_challenge = True
                        break
                except Exception as _url_e:
                    log.debug(f"  [URL守卫] 检测异常: {_url_e}")

                status = await _get_challenge_status()
                now = asyncio.get_event_loop().time()

                # ★ 快速熔断：3秒内连续命中2次 "Images amount must equal" 报错，
                # 说明 recognizer 截图拿到的页面是残留/空白DOM，不用等90s全局看门狗，
                # 立刻强制 page.reload() 拿干净DOM。
                if _img_watcher.count >= 2 and (now - _img_watcher.last_ts) < 3:
                    log.warning(
                        f"  [尝试{attempt_no}] ⚡ 快速熔断：{_img_watcher.count}次 'Images amount "
                        f"must equal' 报错（DOM残留/识图失败），立即 page.reload()..."
                    )
                    solve_task.cancel()
                    try:
                        await page.reload(wait_until="domcontentloaded", timeout=20000)
                    except Exception as _reload_e:
                        log.warning(f"  [尝试{attempt_no}] page.reload() 失败: {_reload_e}")
                    bad_challenge = True
                    break

                # ★ 自动提交检测：格子已选 且 状态稳定超过8s 且 距上次自动提交超过15s
                checked_count = await _count_checked_tiles()

                # ★ 全局无进展看门狗（与下方依赖 status 连续相同的局部watchdog互补）
                # 任何 status 变化 或 checked_count 变化都算"有进展"，重置计时
                if status != _global_last_status or checked_count != _global_last_count:
                    _global_progress_ts = now
                    _global_last_status = status
                    _global_last_count = checked_count
                elif now - _global_progress_ts > 90:
                    log.warning(
                        f"  [尝试{attempt_no}] ⚠️ 90s 内 status/格子数完全无变化"
                        f"（当前status={status!r}），疑似 socket hang up / DOM 残留导致死锁，"
                        f"强制 page.reload() 拿干净DOM..."
                    )
                    solve_task.cancel()
                    try:
                        await page.reload(wait_until="domcontentloaded", timeout=20000)
                    except Exception as _reload_e:
                        log.warning(f"  [尝试{attempt_no}] page.reload() 失败: {_reload_e}")
                    bad_challenge = True
                    break

                if checked_count > 0:
                    if checked_count != _last_checked_count:
                        # 格子数变化，重置稳定计时
                        _last_checked_count = checked_count
                        _checked_stable_since = now
                    elif now - _checked_stable_since > 20 and now - _last_auto_verify_time > 30:
                        # 格子数稳定超过8s，Botright 可能卡住了，主动点 VERIFY
                        log.info(f"  [自动提交] 检测到 {checked_count} 个格子已选且稳定8s，Botright可能卡住，主动提交...")
                        await _click_verify_button()
                        _last_auto_verify_time = now
                        _checked_stable_since = now  # 重置，避免连续触发
                else:
                    _last_checked_count = 0
                    _checked_stable_since = now

                if status != last_status:
                    if status == "try_again":
                        # ★ 修复：try_again 后新图已出现，Botright 自己会继续选格子
                        # 绝对不能主动点 VERIFY，否则打断 Botright 正在做的新一轮选图
                        # 只记录日志，完全交给 Botright 和"格子稳定8s"兜底逻辑处理
                        log.info(f"  [尝试{attempt_no}] ⚠️ 'Please try again' — 新图已加载，等待 Botright 继续选图...")
                        # 重置稳定计时，避免上一轮格子计数干扰新一轮
                        _last_checked_count = 0
                        _checked_stable_since = asyncio.get_event_loop().time()
                    elif status == "select_all":
                        # select_all = 漏选格子
                        # ★ 策略：先让 recognizer 扫描当前题目，把漏选的格子补上再提交
                        # 只有当前已选格子为0（完全空白）时才直接 reload 换题
                        log.warning(f"  [尝试{attempt_no}] ⚠️ 'Please select all matching' — 先尝试补选后提交")
                        _cur_checked = await _count_checked_tiles()
                        if _cur_checked > 0:
                            # 已有部分格子被选，让 Botright 自己再补一轮（等2s观察）
                            log.info(f"  [尝试{attempt_no}] 当前已选 {_cur_checked} 格，等2s观察Botright是否补选...")
                            await asyncio.sleep(2)
                            _after_checked = await _count_checked_tiles()
                            if _after_checked != _cur_checked:
                                # Botright 在动，不干预，重置稳定计时
                                log.info(f"  [尝试{attempt_no}] Botright在补选（{_cur_checked}→{_after_checked}格），交给它继续...")
                                _last_checked_count = _after_checked
                                _checked_stable_since = asyncio.get_event_loop().time()
                            else:
                                # Botright 没动，主动点一次 VERIFY 再看结果
                                log.info(f"  [尝试{attempt_no}] Botright未动，主动点 VERIFY 尝试提交（已选{_after_checked}格）...")
                                await _click_verify_button()
                                _last_checked_count = _after_checked
                                _checked_stable_since = asyncio.get_event_loop().time()
                        else:
                            # 完全没有格子被选，直接 reload 换题
                            log.warning(f"  [尝试{attempt_no}] 当前已选0格，直接 reload 换题")
                            if await _cdp_click_in_bframe(page, "#recaptcha-reload-button"):
                                log.info(f"  [尝试{attempt_no}] ✅ reload换题完成（CDP），等Botright识别新题...")
                            else:
                                log.warning(f"  [尝试{attempt_no}] reload按钮不可见或CDP失败")
                            _last_checked_count = 0
                            _checked_stable_since = asyncio.get_event_loop().time()
                    elif status == "check_new":
                        # ★ check_new 现已由 patch 过的 recognizer 自己原地补选处理
                        # （见文件头部 _patched_handle_recaptcha），不再需要外层脚本手动
                        # 用 CDP 点击 #recaptcha-reload-button 强制换题——那样做等于跟
                        # recognizer 内部正在跑的 detect_tiles() 抢同一个DOM，是真正导致
                        # "Images amount must equal 9 or 16. Is: 0" 的原因。这里只做纯观察，
                        # 给 recognizer 留出时间完成一轮截图+识图+点击+再提交。
                        log.info(f"  [尝试{attempt_no}] 🔍 'Please also check new images' — recognizer 正在原地补选，观察中...")
                        _prev_checked = await _count_checked_tiles()
                        await asyncio.sleep(3)
                        _cur = await _count_checked_tiles()
                        if _cur != _prev_checked:
                            log.info(f"  [尝试{attempt_no}] recognizer 补选中（{_prev_checked}→{_cur}格），不干预")
                        else:
                            log.info(f"  [尝试{attempt_no}] 格子数未变，继续交给 recognizer 内部流程处理，不主动reload")
                        _last_checked_count = _cur
                        _checked_stable_since = asyncio.get_event_loop().time()
                    elif status == "4x4":
                        # ★ 4×4 观察 Botright 是否在动（格子数变化），在动就放手不干预
                        # 问题：table-44 class 做题期间一直存在，每轮都返回 4x4
                        # ★ Fix2：reset 后 DOM 可能残留旧的 table-44，12s 冷却内不做判定
                        # ★ Fix3：不再用 CDP 模拟点击 reload 按钮——这会跟 Botright/recognizer
                        # 内部正在进行的操作（提交答案、重新检测tile等）产生竞态冲突，
                        # 实测会把整个挑战打回未勾选 checkbox 并被 Google 判定 expired，
                        # 而 Botright 的 solve_task 还在死等已经不存在的旧 DOM，反而卡死到 300s 超时。
                        # 改为跟 3×3 一致的"长时间无变化才整体重开"策略，不再戳同一个 iframe。
                        _elapsed_since_start = asyncio.get_event_loop().time() - _attempt_start_time
                        if _elapsed_since_start < 12:
                            log.info(f"  [尝试{attempt_no}] 📐 4×4 检测（已运行{_elapsed_since_start:.0f}s），可能是 reset 后 DOM 残留，冷却等待...")
                            await asyncio.sleep(3)
                        else:
                            if last_status != "4x4":
                                _do_botright_attempt._4x4_since = now
                                _do_botright_attempt._4x4_last_count = await _count_checked_tiles()
                            else:
                                _4x4_cur_count = await _count_checked_tiles()
                                _4x4_prev_count = getattr(_do_botright_attempt, '_4x4_last_count', _4x4_cur_count)
                                _net_last_ts = page._recaptcha_activity["last_ts"]
                                _net_idle = now - _net_last_ts
                                if _4x4_cur_count != _4x4_prev_count or _net_idle < 60:
                                    if _4x4_cur_count != _4x4_prev_count:
                                        log.info(f"  [尝试{attempt_no}] 📐 4×4 Botright选格中（{_4x4_prev_count}→{_4x4_cur_count}），不干预")
                                    elif _net_idle < 60:
                                        log.debug(f"  [尝试{attempt_no}] 📐 4×4 格子数未变，但{_net_idle:.0f}s前有reload/userverify请求，仍在工作")
                                    _do_botright_attempt._4x4_last_count = _4x4_cur_count
                                    _do_botright_attempt._4x4_since = now
                                elif now - getattr(_do_botright_attempt, '_4x4_since', now) > 60:
                                    log.warning(f"  [尝试{attempt_no}] ⚠️ 4×4 状态 60s 无进展（格子数不变 且 {_net_idle:.0f}s 内无 reload/userverify 请求），Botright 真卡死，整体 reset 重开")
                                    solve_task.cancel()
                                    bad_challenge = True
                                    break
                            await asyncio.sleep(1)  # ★ 从2s缩短到1s，更快响应状态变化
                    elif status == "3x3":
                        log.info(f"  [尝试{attempt_no}] 🎯 3×3 动态挑战进行中...")
                        # ★ 修复：记录 3x3 状态开始时间，如果格子数长时间不变则判定 Botright 卡死
                        if last_status != "3x3":
                            _3x3_since = now
                            _3x3_last_count = await _count_checked_tiles()
                    last_status = status

                # 3x3 卡死检测：格子数 45s 内无变化，Botright 内部卡死
                if status == "3x3" and last_status == "3x3":
                    _cur_count = await _count_checked_tiles()
                    if _cur_count != getattr(_do_botright_attempt, '_3x3_last_count', _cur_count):
                        _do_botright_attempt._3x3_last_count = _cur_count
                        _do_botright_attempt._3x3_since = now
                    elif now - getattr(_do_botright_attempt, '_3x3_since', now) > 45:
                        log.warning(f"  [尝试{attempt_no}] ⚠️ 3×3 状态 45s 无进展，Botright 卡死，强制 reset")
                        solve_task.cancel()
                        bad_challenge = True
                        break
                try:
                    await asyncio.wait_for(asyncio.shield(solve_task), timeout=3)
                except asyncio.TimeoutError:
                    pass

            sys.stdout = _old_stdout  # ★ 恢复 stdout，监控器只在本次 attempt 内生效

            if bad_challenge:
                try:
                    await solve_task
                except:
                    pass
                # ★ 修复：超时/取消后先检查 reCAPTCHA 是否已实际通过（checkbox变绿勾）
                # 原bug：90s超时直接return False，但第一轮挑战可能已成功只是Botright还在跑第二轮
                if await is_recaptcha_solved(page):
                    log.info(f"  [尝试{attempt_no}] ⚠️ 超时/取消，但 reCAPTCHA 已实际通过，直接返回成功！")
                    return True
                return False

            result = solve_task.result() if not solve_task.cancelled() else False
            log.info(f"  [尝试{attempt_no}] Botright 返回: {result}")
        except asyncio.CancelledError:
            try:
                sys.stdout = _old_stdout
            except NameError:
                pass
            log.warning(f"  [尝试{attempt_no}] solve_recaptcha 被取消")
            result = False
        except RecursionError:
            try:
                sys.stdout = _old_stdout
            except NameError:
                pass
            # Botright 内部 retry 超过 15 次，本局打不过，reset 重开
            log.warning(f"  [尝试{attempt_no}] Botright RecursionError（retry上限），reset 重开")
            result = False
        except Exception as e:
            try:
                sys.stdout = _old_stdout
            except NameError:
                pass
            log.warning(f"  [尝试{attempt_no}] 异常: {type(e).__name__}: {e}")
            result = False


        await asyncio.sleep(2)
        solved = await is_recaptcha_solved(page)
        log.info(f"  [尝试{attempt_no}] is_recaptcha_solved: {solved}")
        if result or solved:
            log.info(f"✅ [尝试{attempt_no}] 路径2 Botright 解决成功！")
            return True
        return False

    for attempt in range(1, MAX_CAPTCHA_ATTEMPTS + 1):
        log.info(f"【路径2-尝试{attempt}/{MAX_CAPTCHA_ATTEMPTS}】")
        result = await _do_botright_attempt(attempt)
        if result is True:
            return True
        # ★ Vignette 重置：页面已回到初始状态，swal2/reCAPTCHA 全消失，
        # 需要 renew_server 重新点击 Renew 按钮，这里直接把哨兵透传上去
        if result == "VIGNETTE_RESET":
            log.warning("  [solve_recaptcha] ⚠️ Vignette 导致页面重置，需要重新点击 Renew server")
            return "VIGNETTE_RESET"
        if attempt < MAX_CAPTCHA_ATTEMPTS:
            # ★ 关键修复：失败后做页面级 reload，而不是只 sleep
            # 原因：AssertionError "Challenge is not visible" 说明 Google 已把整个挑战框关掉
            # 这时 grecaptcha.reset() 救不了——bframe 已经不存在，reset 只重置 checkbox 逻辑状态
            # 必须重新加载整个页面 → 重走 GDPR 关闭 → 重新点 Renew server → 等 swal2 弹窗
            # 才能得到一个干净的新 bframe 让下次 attempt 正常开始
            _wait = random.uniform(3, 5)
            log.info(f"  尝试{attempt}失败，等待{_wait:.0f}s后重新加载页面...")
            await asyncio.sleep(_wait)

            async def _do_reset_before_retry():
                """整段'重试前页面重置'逻辑，被外层看门狗硬超时包裹"""
                log.info(f"  [重试前] 重新导航到续期页面，获取干净的 reCAPTCHA session...")
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(2)
                await close_gdpr_consent(page)
                await wait_gdpr_gone(page, timeout=10)
                await close_ads(page)
                # 重新点击 Renew server 触发 swal2 弹窗
                _re_clicked = False
                for _rs in ["button.btn-primary", "button:has-text('Renew server')", ".btn:has-text('Renew server')"]:
                    try:
                        _rb = page.locator(_rs).first
                        if await _rb.is_visible(timeout=3000):
                            _rbox = await _rb.bounding_box(timeout=5000)
                            if _rbox:
                                # ★ mouse.click 走 CDP，没有自带超时，CDP半残时会无限挂起，加看门狗
                                await with_watchdog(
                                    page.mouse.click(
                                        _rbox["x"] + _rbox["width"] / 2,
                                        _rbox["y"] + _rbox["height"] / 2
                                    ),
                                    timeout=8,
                                    label=f"重试前 mouse.click {_rs}",
                                )
                                log.info(f"  [重试前] ✅ 重新点击 Renew server ({_rs})")
                                _re_clicked = True
                                break
                    except Exception as _rce:
                        log.debug(f"  [重试前] {_rs} 失败: {_rce}")
                if not _re_clicked:
                    log.warning("  [重试前] ⚠️ 找不到 Renew server 按钮，下次 attempt 可能直接失败")
                else:
                    # 等 swal2 弹窗和 reCAPTCHA iframe 重新出现
                    for _wi in range(15):
                        try:
                            if await page.locator(".swal2-container").is_visible(timeout=500):
                                log.info(f"  [重试前] ✅ swal2 弹窗已重新出现（{_wi}s）")
                                break
                        except:
                            pass
                        await asyncio.sleep(1)
                    await close_gdpr_consent(page)
                    await wait_gdpr_gone(page, timeout=5)
                    await close_ads(page)
                    await asyncio.sleep(2)
                    log.info("  [重试前] 页面已重置，下次 attempt 从干净状态开始")

            try:
                # ★ 关键修复：整段重置流程曾因 CDP/代理半残卡死 8 分22秒。
                # 给它整体加一个 60s 硬看门狗——超时就放弃本次重置，
                # 直接进入下一个 attempt（attempt 内部仍会再做一次 goto/检测，
                # 总比卡死一两个小时直到 GitHub Actions 25 分钟超时强制杀进程好）。
                await with_watchdog(_do_reset_before_retry(), timeout=60, label=f"尝试{attempt}重试前页面重置")
            except WatchdogTimeout:
                log.warning(f"  [重试前] ⚠️ 整段重置超过60s仍未完成，疑似CDP/代理半残，放弃本次重置，直接进入下一 attempt")
                # 尝试做一次最后的硬刷新，给下一个 attempt 一个干净起点；
                # 如果连这个都卡住，safe_reload 内部也有看门狗兜底，绝不会无限期挂起
                await safe_reload(page, timeout_ms=15000, label=f"尝试{attempt}超时后兜底刷新")
            except Exception as _reload_e:
                log.warning(f"  [重试前] 页面重置失败（{_reload_e}），继续重试（可能失败）")

    log.error(f"❌ 路径2 全部 {MAX_CAPTCHA_ATTEMPTS} 次尝试均失败")
    if await is_recaptcha_solved(page):
        log.info("✅ 最终检测：reCAPTCHA 已通过")
        return True
    return False

# ==============================================================================
# 核心续期流程
# ==============================================================================
async def renew_server(page, url: str, server_label: str) -> tuple[bool, str | None, str | None]:
    log.info(f"=== 开始续期: {server_label} ===")
    log.debug(f"[renew_server] URL={url}")

    if not await navigate(page, url):
        log.error(f"❌ {server_label}: 无法通过 CF 验证，跳过")
        await take_screenshot(page, f"{server_label}_cf_fail")
        return False, None, None

    # ★ 修复1：页面加载后立即关闭 GDPR Cookie 弹窗
    # 弹窗会遮挡页面内容和 reCAPTCHA anchor iframe，导致后续流程全部失败
    log.info("检测并关闭 GDPR Cookie 同意弹窗...")
    await close_gdpr_consent(page)
    # ★ 修复：用 DOM 可见性检测等待弹窗真正消失，替换原来的文字检测（会误判）
    if not await wait_gdpr_gone(page, timeout=15):
        log.warning("⚠️ GDPR 弹窗未能完全关闭，继续尝试强制关闭...")
        # 最后尝试：按 Escape 键关闭
        try:
            await page.keyboard.press("Escape")
            await asyncio.sleep(1)
        except:
            pass
        # 再等 3 秒
        await wait_gdpr_gone(page, timeout=3)

    # 等待页面内容真正渲染完毕
    log.info("等待页面内容加载完毕...")
    page_ready = False
    for i in range(20):
        body = await get_text(page)
        body_lower = body.lower()
        if ("renew server" in body_lower or "deletes on" in body_lower
                or "expires in" in body_lower):
            log.info(f"✅ 页面内容已加载（{i}s）")
            page_ready = True
            break
        if i % 5 == 0 and i > 0:
            log.info(f"  页面仍在加载... {i}s")
        await asyncio.sleep(1)
    if not page_ready:
        log.warning(f"⚠️ {server_label}: 页面加载超时（20s），尝试继续...")

    await human_delay(0.5, 1)
    await take_screenshot(page, f"{server_label}_01_loaded")

    before_date = await read_delete_date(page)
    log.info(f"续期前 Deletes on: {before_date}")

    body = await get_text(page)
    if "expires in" not in body.lower() and "renew" not in body.lower():
        log.warning(f"⚠️ {server_label}: 页面未找到续期相关内容")
        await take_screenshot(page, f"{server_label}_no_renew")
        return False, before_date, None

    # 页面停留预热
    log.info("页面停留预热 5 秒...")
    await asyncio.sleep(5)

    # 步骤1：点击 "Renew server" 按钮（触发弹窗）
    log.info("步骤1：确认 GDPR 弹窗已消失后，点击 'Renew server' 按钮...")
    if not await wait_gdpr_gone(page, timeout=10):
        log.warning("⚠️ 步骤1前 GDPR 弹窗仍存在，强制关闭中...")
        await close_gdpr_consent(page)
        await asyncio.sleep(2)
    # 关闭可能遮挡按钮的广告弹窗
    log.info("  [步骤1] 清除广告遮挡层...")
    _t0 = asyncio.get_event_loop().time()
    await close_ads(page)
    log.info(f"  [步骤1] close_ads 耗时: {asyncio.get_event_loop().time()-_t0:.1f}s")


    # page health check: is the countdown ticking?
    # The time digits (HH:MM:SS) live in a child text node; use JS to find them
    log.info("  [\u6b65\u9aa41] \u68c0\u6d4b\u9875\u9762\u5012\u8ba1\u65f6\u662f\u5426\u6b63\u5e38\u8d70\u52a8...")
    _page_frozen = False

    async def _read_countdown():
        # span#expireDate contains the ticking HH:MM:SS digits
        val = await page.evaluate(
            """() => {
                var el = document.getElementById("expireDate");
                if (el) return el.textContent.trim();
                return null;
            }"""
        )
        return val

    try:
        _ct1 = await _read_countdown()
        await asyncio.sleep(3)
        _ct2 = await _read_countdown()
        log.info(f"  [\u6b65\u9aa41] \u5012\u8ba1\u65f6\u8bfb\u6570: {_ct1!r} \u2192 {_ct2!r}")
        if _ct1 is None and _ct2 is None:
            log.warning("  [\u6b65\u9aa41] \u26a0\ufe0f \u9875\u9762\u672a\u627e\u5230\u5012\u8ba1\u65f6\uff0c\u53ef\u80fd\u9875\u9762\u672a\u6b63\u5e38\u6e32\u67d3\uff0c\u5c1d\u8bd5\u5237\u65b0...")
            _page_frozen = True
        elif _ct1 is not None and _ct2 is not None and _ct1 == _ct2:
            log.warning(f"  [\u6b65\u9aa41] \u26a0\ufe0f \u9875\u9762\u5012\u8ba1\u65f6\u51bb\u7ed3\uff08{_ct1}\uff09\uff0c\u9875\u9762JS\u5361\u6b7b\uff0c\u5c06\u5237\u65b0\u91cd\u8bd5...")
            _page_frozen = True
        else:
            log.info(f"  [\u6b65\u9aa41] \u2705 \u5012\u8ba1\u65f6\u6b63\u5e38\u8d70\u52a8\uff08{_ct1} \u2192 {_ct2}\uff09")
    except Exception as _e:
        log.warning(f"  [\u6b65\u9aa41] \u5012\u8ba1\u65f6\u68c0\u6d4b\u5f02\u5e38\uff08\u5ffd\u7565\uff0c\u7ee7\u7eed\uff09: {_e}")

    if _page_frozen:
        log.info("  [\u6b65\u9aa41] \u5237\u65b0\u9875\u9762...")
        await page.reload(timeout=30000, wait_until="domcontentloaded")
        await asyncio.sleep(3)
        await close_gdpr_consent(page)
        await wait_gdpr_gone(page, timeout=10)
        await asyncio.sleep(1)
        log.info("  [\u6b65\u9aa41] \u5237\u65b0\u5b8c\u6bd5\uff0c\u91cd\u65b0\u68c0\u6d4b\u5012\u8ba1\u65f6...")
        try:
            _ct3 = await _read_countdown()
            await asyncio.sleep(3)
            _ct4 = await _read_countdown()
            log.info(f"  [\u6b65\u9aa41] \u5237\u65b0\u540e\u5012\u8ba1\u65f6\u8bfb\u6570: {_ct3!r} \u2192 {_ct4!r}")
            if (_ct3 is None and _ct4 is None) or (_ct3 is not None and _ct3 == _ct4):
                log.error("  [\u6b65\u9aa41] \u274c \u5237\u65b0\u540e\u5012\u8ba1\u65f6\u4ecd\u51bb\u7ed3\u6216\u4e0d\u5b58\u5728\uff0c\u9875\u9762\u5f02\u5e38\uff0c\u653e\u5f03\u672c\u6b21\u7eed\u671f")
                await take_screenshot(page, f"{server_label}_frozen_after_reload")
                return False, before_date, None
            else:
                log.info(f"  [\u6b65\u9aa41] \u2705 \u5237\u65b0\u540e\u5012\u8ba1\u65f6\u6062\u590d\u6b63\u5e38\uff08{_ct3} \u2192 {_ct4}\uff09")
        except Exception as _e:
            log.warning(f"  [\u6b65\u9aa41] \u5237\u65b0\u540e\u5012\u8ba1\u65f6\u68c0\u6d4b\u5f02\u5e38\uff08\u7ee7\u7eed\uff09: {_e}")


    clicked_renew_btn = False
    # ★ 点击策略：
    # - JS dispatchEvent / b.click() 会被 Cloudflare __cfRLUnblockHandlers 拦截
    # - Playwright click(force=True) 内部仍调 scroll_into_view_if_needed，被倒计时卡死
    # - page.evaluate 在 Cloudflare JS 繁忙时会挂起，而 asyncio.wait_for 对
    #   Playwright coroutine 的 cancel 不可靠（Playwright 用 pyee/greenlet 自管调度），
    #   导致 wait_for 设了超时却永远不触发，程序挂死直到 Actions timeout(124)
    # - 正确做法：全部改用 Playwright 原生 timeout 参数，走 Playwright 自己的取消机制

    log.info("  点击 Renew server：locator.bounding_box + mouse.click 坐标点击...")
    coords = None

    # 候选选择器，依次尝试
    _renew_selectors = [
        "button.btn-primary",
        "button:has-text('Renew server')",
        ".btn:has-text('Renew server')",
    ]

    # 步骤1：用 Playwright locator（自带可靠超时）拿按钮坐标
    # ★ 不用 scroll_into_view_if_needed：倒计时页面 DOM 每秒更新，
    #   Playwright actionability check 一直等 "element to be stable" 而超时
    # 直接 bounding_box 拿坐标即可，按钮本身在视口内不需要滚动
    for _sel in _renew_selectors:
        try:
            _loc = page.locator(_sel).filter(has_text="Renew server") \
                if "btn-primary" in _sel \
                else page.locator(_sel)
            log.info(f"  [定位] 尝试选择器: {_sel}")
            _box = await _loc.first.bounding_box(timeout=5000)
            if _box and _box['width'] > 0 and _box['height'] > 0:
                coords = {
                    'x': _box['x'] + _box['width'] / 2,
                    'y': _box['y'] + _box['height'] / 2
                }
                log.info(f"  [定位] ✅ 坐标: ({coords['x']:.0f},{coords['y']:.0f})  selector={_sel}")
                break
            else:
                log.warning(f"  [定位] bounding_box 为空，换下一个选择器")
        except Exception as _le:
            log.warning(f"  [定位] {_sel} 失败: {_le}")

    if coords:
        try:
            # 步骤2：用 CDP 直接发原始鼠标事件（fire-and-forget，不等渲染进程 ack）
            # ★ 修复：page.mouse.click() 内部等待 CDP Input.dispatchMouseEvent 的 ack，
            #   Cloudflare 反爬脚本阻塞 JS 主线程验证时渲染进程不 ack，导致永久挂死
            #   改用 cdp_session.send() 发完即返回，不等 ack
            log.info(f"  [点击] CDP dispatchMouseEvent ({coords['x']:.0f},{coords['y']:.0f})")
            _cdp = await page.context.new_cdp_session(page)
            _cx, _cy = coords['x'], coords['y']
            await _cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseMoved", "x": _cx, "y": _cy, "button": "none",
                "buttons": 0, "clickCount": 0, "modifiers": 0
            })
            await asyncio.sleep(0.05)
            await _cdp.send("Input.dispatchMouseEvent", {
                "type": "mousePressed", "x": _cx, "y": _cy, "button": "left",
                "buttons": 1, "clickCount": 1, "modifiers": 0
            })
            await asyncio.sleep(0.08)
            await _cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseReleased", "x": _cx, "y": _cy, "button": "left",
                "buttons": 0, "clickCount": 1, "modifiers": 0
            })
            await _cdp.detach()
            log.info(f"✅ 点击 Renew server CDP ({coords['x']:.0f},{coords['y']:.0f})")
            clicked_renew_btn = True

            # 步骤3：点击后 1.5s 检测弹窗是否出现
            await asyncio.sleep(1.5)
            try:
                renew_triggered = await page.locator(".swal2-container").is_visible(timeout=1000)
            except Exception:
                renew_triggered = False

            if not renew_triggered:
                log.warning("  ⚠️ 点击后 1.5s 弹窗未出现，尝试重新 CDP 点击...")
                try:
                    for _sel2 in _renew_selectors:
                        _loc2 = page.locator(_sel2).filter(has_text="Renew server") \
                            if "btn-primary" in _sel2 \
                            else page.locator(_sel2)
                        _box2 = await _loc2.first.bounding_box(timeout=5000)
                        if _box2 and _box2['width'] > 0:
                            _cx2 = _box2['x'] + _box2['width'] / 2
                            _cy2 = _box2['y'] + _box2['height'] / 2
                            _cdp2 = await page.context.new_cdp_session(page)
                            await _cdp2.send("Input.dispatchMouseEvent", {
                                "type": "mousePressed", "x": _cx2, "y": _cy2,
                                "button": "left", "buttons": 1, "clickCount": 1, "modifiers": 0
                            })
                            await asyncio.sleep(0.08)
                            await _cdp2.send("Input.dispatchMouseEvent", {
                                "type": "mouseReleased", "x": _cx2, "y": _cy2,
                                "button": "left", "buttons": 0, "clickCount": 1, "modifiers": 0
                            })
                            await _cdp2.detach()
                            log.info(f"  重新 CDP 点击 ({_cx2:.0f},{_cy2:.0f})")
                            break
                except Exception as _e2:
                    log.warning(f"  重试 CDP 点击失败: {_e2}")
            else:
                log.info("  ✅ 验证：renew() 已触发，弹窗可见")
        except Exception as _ce:
            log.warning(f"  CDP 点击失败: {_ce}")
    if not clicked_renew_btn:
        # 备用：Playwright locator + tab键聚焦后 Enter（完全绕过稳定性检查）
        log.info("  备用：locator.focus() + keyboard.press('Enter')...")
        try:
            btn = page.locator("button.btn-primary").first
            await btn.focus(timeout=3000)
            await page.keyboard.press("Enter")
            log.info("✅ focus + Enter 点击 Renew server")
            clicked_renew_btn = True
        except Exception as _e:
            log.warning(f"  focus+Enter 失败: {_e}")

    if not clicked_renew_btn:
        log.error(f"❌ {server_label}: 找不到 'Renew server' 按钮")
        await take_screenshot(page, f"{server_label}_no_renew_btn")
        return False, before_date, None

    # ★ 修复：等待 SweetAlert2 弹窗或 reCAPTCHA iframe 出现，替换原来的文字检测
    # 原来检测 "not a robot" / "verify" 文字经常不匹配，导致直接跳过等待
    # ★ 修复2：改用 locator 检测弹窗可见性，避免 page.evaluate 在 Cloudflare 页面挂起
    log.info("等待续期弹窗（SweetAlert2 / reCAPTCHA）出现...")

    # ★ 修复3：点击 Renew 后 Google Vignette 广告可能立刻弹出（URL 变成 #google_vignette）
    # 它会完全遮住页面，导致 swal2-container 永远检测不到，先等 1s 再清一次
    await asyncio.sleep(1)
    if await dismiss_google_vignette(page):
        log.info("  [renew_server] ✅ 点击 Renew 后检测到并关闭了 Google Vignette 广告")
        await asyncio.sleep(1)

    modal_appeared = False
    for i in range(20):
        try:
            # ★ 每轮同步检测 vignette：广告可能在等待期间任意时刻弹出
            if "#google_vignette" in page.url or "#google_survey" in page.url:
                log.warning(f"  [renew_server] 等待弹窗期间检测到 Vignette（第{i}s），关闭中...")
                await dismiss_google_vignette(page)
                await asyncio.sleep(1)
                continue

            if await page.locator(".swal2-container").is_visible(timeout=500):
                log.info(f"✅ 续期弹窗已出现（{i}s）")
                modal_appeared = True
                break
            if await page.locator("iframe[src*='recaptcha']").count() > 0:
                log.info(f"✅ 续期弹窗已出现（reCAPTCHA iframe, {i}s）")
                modal_appeared = True
                break
        except:
            pass
        await asyncio.sleep(1)

    if not modal_appeared:
        log.warning("⚠️ 续期弹窗未检测到（20s超时），可能被 GDPR 弹窗拦截，尝试重新关闭并重新点击...")
        await close_gdpr_consent(page)
        await wait_gdpr_gone(page, timeout=5)
        await asyncio.sleep(1)
        # 重新点击一次（用 bounding_box + mouse.click，绕过 Cloudflare __cfRLUnblockHandlers）
        for sel in ["button.btn-primary", "button:has-text('Renew server')", ".btn:has-text('Renew server')"]:
            try:
                btn = page.locator(sel).filter(has_text="Renew server") if "btn-primary" in sel else page.locator(sel)
                if await btn.first.is_visible(timeout=2000):
                    # ★ 不用 scroll_into_view_if_needed（倒计时页面 actionability check 会卡死）
                    box = await btn.first.bounding_box(timeout=5000)
                    if box:
                        cx = box['x'] + box['width'] / 2
                        cy = box['y'] + box['height'] / 2
                        await page.mouse.click(cx, cy)
                        log.info(f"✅ 重新坐标点击 '{sel}' ({cx:.0f},{cy:.0f})")
                        break
            except:
                pass
        # 再等 10 秒
        for i in range(10):
            try:
                if await page.locator(".swal2-container").is_visible(timeout=500):
                    log.info(f"✅ 重试后续期弹窗已出现（{i}s）")
                    modal_appeared = True
                    break
                if await page.locator("iframe[src*='recaptcha']").count() > 0:
                    log.info(f"✅ 重试后续期弹窗已出现（reCAPTCHA iframe, {i}s）")
                    modal_appeared = True
                    break
            except:
                pass
            await asyncio.sleep(1)

    if not modal_appeared:
        log.warning("⚠️ 续期弹窗始终未出现，继续尝试 reCAPTCHA 流程（可能已在后台加载）...")

    await human_delay(0.5, 1)

    # ★ 修复2：弹窗出现后再次用 DOM 检测确认 GDPR 弹窗已消失
    # 弹窗有时会在 SweetAlert2 弹出后重新出现，遮挡 reCAPTCHA iframe
    gdpr_closed = await wait_gdpr_gone(page, timeout=8)
    if not gdpr_closed:
        log.warning("⚠️ 步骤2前 GDPR 弹窗仍存在，强制关闭...")
        await close_gdpr_consent(page)
        await wait_gdpr_gone(page, timeout=5)

    # 关闭可能遮挡的广告
    await close_ads(page)

    # ★ 额外等待 2 秒，让 reCAPTCHA iframe 在 GDPR 关闭后有时间完成加载
    log.info("等待 reCAPTCHA iframe 初始化（GDPR 关闭后需要时间加载）...")
    await asyncio.sleep(2)

    # 广告/GDPR 清理完毕后再截图，确保弹窗内容可见
    await take_screenshot(page, f"{server_label}_02_modal")
    await asyncio.sleep(2)

    # 步骤2：reCAPTCHA（普通模式优先，图片挑战用 recognizer）
    # ★ 外层循环：处理 Google Vignette 导致的页面重置
    # 若 solve_recaptcha 返回 "VIGNETTE_RESET"，说明做题中途页面被 goto 重载，
    # swal2 弹窗和 reCAPTCHA iframe 全消失，必须重新点击 Renew server 按钮触发弹窗。
    _MAX_VIGNETTE_RETRIES = 3
    recaptcha_ok = False
    for _vignette_round in range(_MAX_VIGNETTE_RETRIES):
        if _vignette_round > 0:
            # 重新触发弹窗：等 GDPR 消失 → close_ads → 找并点击 Renew server 按钮
            log.warning(f"  [Vignette重试 {_vignette_round}/{_MAX_VIGNETTE_RETRIES-1}] 重新点击 Renew server 按钮，触发 swal2 弹窗...")
            if not await wait_gdpr_gone(page, timeout=10):
                await close_gdpr_consent(page)
                await asyncio.sleep(2)
            # ★ goto fallback 后页面需要重新加载，Renew server 按钮最多需要 ~15-20s 才出现
            # 改为轮询等待，每秒检测一次，最多等 25s
            _re_clicked = False
            _renew_btn_selectors = ["button.btn-primary", "button:has-text('Renew server')", ".btn:has-text('Renew server')"]
            log.info(f"  [Vignette重试] 等待 Renew server 按钮出现（最多25s）...")
            for _wait_i in range(25):
                # 每5s清一次广告（防止遮挡）
                if _wait_i > 0 and _wait_i % 5 == 0:
                    await close_ads(page)
                _found_btn = False
                for _rs in _renew_btn_selectors:
                    try:
                        _rb = page.locator(_rs).first
                        if await _rb.is_visible(timeout=800):
                            _rbox = await _rb.bounding_box(timeout=3000)
                            if _rbox:
                                await page.mouse.click(
                                    _rbox["x"] + _rbox["width"] / 2,
                                    _rbox["y"] + _rbox["height"] / 2
                                )
                                log.info(f"  [Vignette重试] ✅ 重新点击 '{_rs}'（等待了{_wait_i}s）")
                                _re_clicked = True
                                _found_btn = True
                                break
                    except Exception as _re:
                        log.debug(f"  [Vignette重试] 选择器 {_rs} 失败: {_re}")
                if _found_btn:
                    break
                if _wait_i < 24:
                    log.debug(f"  [Vignette重试] 按钮未出现（{_wait_i+1}s），继续等待...")
                    await asyncio.sleep(1)
            if not _re_clicked:
                log.error(f"  [Vignette重试] ❌ 等待25s后仍找不到 Renew server 按钮，放弃")
                break
            # 等待 swal2 弹窗重新出现（最多 20s）
            _modal_re = False
            for _i in range(20):
                try:
                    if "#google_vignette" in page.url or "#google_survey" in page.url:
                        await dismiss_google_vignette(page)
                        await asyncio.sleep(1)
                        continue
                    if await page.locator(".swal2-container").is_visible(timeout=500):
                        log.info(f"  [Vignette重试] ✅ swal2 弹窗重新出现（{_i}s）")
                        _modal_re = True
                        break
                    if await page.locator("iframe[src*='recaptcha']").count() > 0:
                        log.info(f"  [Vignette重试] ✅ reCAPTCHA iframe 重新出现（{_i}s）")
                        _modal_re = True
                        break
                except:
                    pass
                await asyncio.sleep(1)
            if not _modal_re:
                log.error(f"  [Vignette重试] ❌ swal2 弹窗未重新出现，放弃")
                break
            await close_gdpr_consent(page)
            await wait_gdpr_gone(page, timeout=5)
            await close_ads(page)
            await asyncio.sleep(2)

        log.info("步骤2：处理 reCAPTCHA（代理IP普通模式优先，图片挑战用 recognizer）...")
        log.info(f"[renew_server] 开始 reCAPTCHA 处理...")
        _captcha_result = await solve_recaptcha(page, url)
        if _captcha_result == "VIGNETTE_RESET":
            log.warning(f"  [renew_server] ⚠️ solve_recaptcha 返回 VIGNETTE_RESET（第{_vignette_round+1}次），重新走步骤1...")
            continue
        recaptcha_ok = bool(_captcha_result)
        break

    log.info(f"[renew_server] reCAPTCHA 结果: {'✅通过' if recaptcha_ok else '❌未通过'}")

    await human_delay(1, 1.5)
    await take_screenshot(page, f"{server_label}_03_after_captcha")

    if not recaptcha_ok:
        log.error(f"❌ {server_label}: reCAPTCHA 未通过，放弃续期")
        try:
            await page.locator("button:has-text('Cancel')").first.click()
        except:
            pass
        return False, before_date, None

    # 步骤3：注入 token 并点击弹窗 "Renew" 确认按钮
    log.info("步骤3：注入 reCAPTCHA token 并点击 'Renew' 确认按钮...")
    clicked_confirm = False

    token_ok = False
    for _ in range(10):
        try:
            token = await page.evaluate("""() => {
                var el = document.querySelector('textarea[name="g-recaptcha-response"]');
                return el ? el.value : '';
            }""")
            if token and len(token) > 10:
                log.info(f"✅ g-recaptcha-response token 已就绪（长度={len(token)}）")
                token_ok = True
                break
        except:
            pass
        await asyncio.sleep(0.5)

    if not token_ok:
        log.warning("⚠️ g-recaptcha-response 为空，尝试从 anchor frame 读取并手动注入...")
        try:
            anchor = await find_recaptcha_frame(page, "anchor")
            if anchor:
                injected = await page.evaluate("""() => {
                    var iframes = document.querySelectorAll('iframe[src*="recaptcha"]');
                    for (var f of iframes) {
                        try {
                            var resp = f.contentDocument
                                ? f.contentDocument.querySelector('#recaptcha-token')
                                : null;
                            if (resp && resp.value) return resp.value;
                        } catch(e) {}
                    }
                    return null;
                }""")
                if injected:
                    await page.evaluate("""(token) => {
                        var el = document.querySelector('textarea[name="g-recaptcha-response"]');
                        if (el) el.value = token;
                    }""", injected)
                    log.info("✅ 已手动注入 reCAPTCHA token")
                    token_ok = True
        except Exception as e:
            log.warning(f"手动注入 token 失败: {e}")

    if not token_ok:
        log.warning("⚠️ token 注入失败，仍尝试点击 Renew（可能失败）")

    await asyncio.sleep(random.uniform(0.5, 1.2))

    # ★ 点确认按钮前，先清掉可能弹出的广告遮挡层
    # 广告在 reCAPTCHA 解决后才弹出，之前的 close_ads 时机太早捕获不到
    log.info("  [步骤3] 清除广告遮挡层（点确认前）...")
    try:
        # 先点 Close 按钮
        for ad_sel in [
            "button:has-text('Close')", "[aria-label='Close']", "[aria-label='close']",
            "button:has-text('×')", "button:has-text('✕')",
        ]:
            try:
                btn = page.locator(ad_sel).first
                if await btn.is_visible(timeout=300):
                    await btn.click()
                    log.info(f"  [步骤3] 关闭广告按钮: {ad_sel}")
                    await asyncio.sleep(0.3)
            except:
                pass
        # 再用 JS 强制移除所有非 swal2/reCAPTCHA 的高 z-index 遮挡层
        removed = await page.evaluate("""() => {
            let removed = 0;
            const all = Array.from(document.querySelectorAll('*'));
            for (const el of all) {
                if (!el.isConnected) continue;
                const s = window.getComputedStyle(el);
                const z = parseInt(s.zIndex) || 0;
                if (z > 1000 && (s.position === 'fixed' || s.position === 'absolute')) {
                    const cls = (el.className || '') + (el.id || '');
                    if (!cls.includes('swal') && !cls.includes('recaptcha') &&
                        !cls.includes('gdpr') && !cls.includes('cmp')) {
                        el.remove();
                        removed++;
                    }
                }
            }
            return removed;
        }""")
        if removed > 0:
            log.info(f"  [步骤3] JS 强制移除 {removed} 个广告遮挡层")
        await asyncio.sleep(0.3)
    except Exception as e:
        log.debug(f"  [步骤3] 广告清除异常（忽略）: {e}")

    # ★ 步骤3 额外检测：reCAPTCHA 解完后 Google Vignette 也可能在此时弹出
    await dismiss_google_vignette(page)

    # ★ 修复：原来 button.btn-primary 排第一，但实际弹窗按钮类是 swal2-confirm swal2-styled
    # btn-primary 匹配到了页面背后某个隐藏元素，导致弹窗没有真正提交
    # 修复：优先用 aria-label="Renew" + swal2-confirm，把 btn-primary 移到最后兜底
    for sel in [
        "button.swal2-confirm[aria-label='Renew']",   # 最精准：aria-label + swal2类
        "button.swal2-confirm.swal2-styled",           # F12 确认的实际类名
        ".swal2-popup button.swal2-confirm",           # 限定在弹窗容器内
        ".swal2-container button.swal2-confirm",
        "button.swal2-confirm",                        # 通用 swal2
        ".swal2-actions button:first-child",           # actions 区第一个按钮（通常是确认）
        "button:has-text('Renew'):not(:has-text('server'))",  # 文字匹配，排除Renew server
        ".swal2-popup button:has-text('Renew')",
        "button.btn-primary",                          # 最后兜底
    ]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=2000):
                # 额外确认按钮在视口内（排除 display:none 的隐藏元素误匹配）
                box = await btn.bounding_box()
                if box and box["width"] > 0 and box["height"] > 0:
                    await btn.scroll_into_view_if_needed()
                    await asyncio.sleep(0.3)
                    await btn.click()
                    log.info(f"✅ [方法1] 点击弹窗 Renew 确认按钮: {sel} (box={box['width']:.0f}x{box['height']:.0f})")
                    clicked_confirm = True
                    break
                else:
                    log.debug(f"  [方法1] {sel} is_visible=True 但 bounding_box 为空，跳过")
        except:
            pass

    if not clicked_confirm:
        log.info("方法1未成功，尝试方法2：JS 定位 swal2-confirm + 真实鼠标点击...")
        coords = await page.evaluate("""() => {
            // 优先找 swal2-confirm（F12确认的实际类名）
            var candidates = [
                document.querySelector('.swal2-confirm[aria-label="Renew"]'),
                document.querySelector('.swal2-confirm.swal2-styled'),
                document.querySelector('.swal2-confirm'),
                document.querySelector('.swal2-actions button'),
            ];
            for (var b of candidates) {
                if (!b) continue;
                var rect = b.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0 && rect.top > 0) {
                    return {
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2,
                        text: (b.innerText || b.textContent || '').trim()
                    };
                }
            }
            // 兜底：找文字含 renew 且不含 server 的可见按钮
            var btns = Array.from(document.querySelectorAll('button'));
            for (var b of btns) {
                var t = (b.innerText || b.textContent || '').trim().toLowerCase();
                if (t.includes('renew') && !t.includes('server')) {
                    var rect = b.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0 && rect.top > 0) {
                        return {
                            x: rect.left + rect.width / 2,
                            y: rect.top + rect.height / 2,
                            text: b.innerText.trim()
                        };
                    }
                }
            }
            return null;
        }""")
        if coords:
            x, y, btn_text = coords['x'], coords['y'], coords['text']
            log.info(f"✅ [方法2] 找到按钮 '{btn_text}'，坐标 ({x:.0f}, {y:.0f})")
            await page.mouse.move(x + random.uniform(-5, 5), y + random.uniform(-5, 5))
            await asyncio.sleep(random.uniform(0.2, 0.5))
            await page.mouse.click(x, y)
            clicked_confirm = True

    if not clicked_confirm:
        log.info("方法3：JS 直接 click() 兜底（优先 swal2-confirm）...")
        result = await page.evaluate("""() => {
            // 优先点 swal2-confirm
            var swal = document.querySelector('.swal2-confirm[aria-label="Renew"], .swal2-confirm.swal2-styled, .swal2-confirm');
            if (swal) { swal.click(); return (swal.innerText || swal.textContent || '').trim(); }
            // 再找文字
            var btns = Array.from(document.querySelectorAll('button'));
            for (var b of btns) {
                var t = (b.innerText || b.textContent || '').trim().toLowerCase();
                if (t.includes('renew') && !t.includes('server')) {
                    b.click();
                    return b.innerText.trim();
                }
            }
            return null;
        }""")
        if result:
            log.info(f"✅ [方法3] JS 直接点击: {result}")
            clicked_confirm = True

    if not clicked_confirm:
        log.error(f"❌ {server_label}: 找不到弹窗 Renew 确认按钮")
        await take_screenshot(page, f"{server_label}_no_confirm_btn")
        return False, before_date, None

    log.info("等待弹窗关闭（确认提交成功）...")
    modal_closed = False
    for i in range(15):
        try:
            swal_visible = await page.locator(".swal2-container").is_visible(timeout=500)
            if not swal_visible:
                log.info(f"✅ 弹窗已关闭（{i}s），续期请求已提交")
                modal_closed = True
                break
        except:
            log.info(f"✅ 弹窗已消失（{i}s），续期请求已提交")
            modal_closed = True
            break
        await asyncio.sleep(1)

    if not modal_closed:
        log.warning("⚠️ 弹窗 15 秒内未关闭，可能提交失败")
        await take_screenshot(page, f"{server_label}_modal_not_closed")

    log.info("等待续期结果...")
    await take_screenshot(page, f"{server_label}_04_after_confirm")

    # 等2s让服务器处理续期，避免立即reload时页面自身还在跳转
    await asyncio.sleep(2)

    log.info("重新加载页面读取续期后到期时间...")
    after_date = None
    for _reload_try in range(3):
        try:
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            await asyncio.sleep(1)
            # 轮询等新日期，最多等8s
            for _i in range(16):
                after_date = await read_delete_date(page)
                if after_date and after_date != before_date:
                    break
                await asyncio.sleep(0.5)
            if after_date:
                break
        except Exception as e:
            log.warning(f"重新加载失败（第{_reload_try+1}次）: {e}")
            await asyncio.sleep(2)

    log.info(f"续期后 Deletes on: {after_date}")
    await take_screenshot(page, f"{server_label}_05_final")

    if before_date and after_date and before_date != after_date:
        log.info(f"✅ {server_label}: 续期成功！{before_date} → {after_date}")
        return True, before_date, after_date
    elif after_date and not before_date:
        log.info(f"✅ {server_label}: 续期操作完成（after={after_date}）")
        return True, before_date, after_date
    else:
        log.warning(f"⚠️ {server_label}: 到期时间未变化（before={before_date}, after={after_date}）")
        return False, before_date, after_date

# ==============================================================================
# 主流程
# ==============================================================================
async def main():
    import botright

    # ★ 连通性检测（Xray SOCKS5 代理）
    import subprocess as _sp

    log.info("检测 Google reCAPTCHA 连通性（via 代理 127.0.0.1:10808）...")
    try:
        _r = _sp.run(
            ["curl", "--socks5", "127.0.0.1:10808",
             "--connect-timeout", "12", "-s", "-o", "/dev/null",
             "-w", "%{http_code}", "https://www.google.com/recaptcha/api.js"],
            capture_output=True, text=True, timeout=18
        )
        _code = _r.stdout.strip()
        if _code in ("200", "301", "302"):
            # 顺便打印代理出口 IP
            try:
                _ri = _sp.run(
                    ["curl", "--socks5", "127.0.0.1:10808",
                     "--connect-timeout", "8", "-s", "https://ifconfig.me"],
                    capture_output=True, text=True, timeout=12
                )
                log.info(f"✅ Google reCAPTCHA 可访问，代理出口 IP: {_ri.stdout.strip() or '?'}")
            except Exception:
                log.info("✅ Google reCAPTCHA 可访问")
        else:
            msg = (f"❌ host2play 续期失败：Google reCAPTCHA 不可访问（http_code={_code}）。"
                   f"请检查代理配置或稍后手动触发重跑。")
            log.error(msg)
            wxpush(msg)
            raise SystemExit("Google reCAPTCHA 不可达，退出")
    except SystemExit:
        raise
    except Exception as _ce:
        msg = f"❌ host2play 续期失败：连通性检测异常 {_ce}"
        log.error(msg)
        wxpush(msg)
        raise SystemExit("Google reCAPTCHA 连通性检测异常，退出")

    log.info(f"启动 Botright（防检测浏览器，内置 reCAPTCHA 解决）... proxy={'已配置' if PROXY_SERVER else '无'}")
    botright_client = await botright.Botright(
        headless=False,
        block_images=False,
        scroll_into_view=True,
    )
    # Botright proxy 格式：直接传字符串，去掉 "socks5://" 前缀
    proxy_str = PROXY_SERVER.replace("socks5://", "") if PROXY_SERVER else None
    browser = await botright_client.new_browser(
        locale="en-US",
        **({"proxy": proxy_str} if proxy_str else {})
    )
    page = await browser.new_page()
    await page.set_viewport_size({"width": 1280, "height": 900})

    # ★ JS 层拦截 Google Vignette / Survey 广告
    # 原理：Vignette 通过修改 location.hash 为 #google_vignette 或 #google_survey 触发。
    # 在每次导航前注入脚本，劫持 hash 赋值，直接丢弃这两个值，广告根本弹不出来。
    # 同时覆盖 history.pushState / replaceState，防止第三方广告 JS 走 History API 打开广告页。
    await page.add_init_script("""
        (() => {
            // 拦截 location.hash 赋值
            const _hashDesc = Object.getOwnPropertyDescriptor(Location.prototype, 'hash');
            if (_hashDesc && _hashDesc.set) {
                Object.defineProperty(Location.prototype, 'hash', {
                    get: _hashDesc.get,
                    set(v) {
                        if (typeof v === 'string' &&
                            (v.includes('google_vignette') || v.includes('google_survey'))) {
                            return;  // 静默丢弃
                        }
                        _hashDesc.set.call(this, v);
                    },
                    configurable: true,
                });
            }
            // 拦截 history.pushState / replaceState（广告有时走这条路）
            ['pushState', 'replaceState'].forEach(method => {
                const orig = history[method].bind(history);
                history[method] = function(state, title, url) {
                    if (typeof url === 'string' &&
                        (url.includes('google_vignette') || url.includes('google_survey'))) {
                        return;
                    }
                    return orig(state, title, url);
                };
            });
        })();
    """)
    log.info("✅ 已注入 Google Vignette/Survey 广告拦截脚本（JS层）")

    # ★ 在 iframe 创建时就把 reCAPTCHA hl 改为 en，省掉事后重载那一轮等待
    # 原理：MutationObserver 监听 iframe 被插入 DOM 的时机，
    # 在浏览器发出请求前直接改 src，Google 服务端第一次就返回英语挑战词。
    await page.add_init_script("""
        (() => {
            function patchRecaptchaHl(iframe) {
                const src = iframe.getAttribute('src') || '';
                if (!src.includes('recaptcha')) return;
                let newSrc;
                if (/[?&]hl=/.test(src)) {
                    if (/[?&]hl=en(&|$)/.test(src)) return;  // 已经是 en，跳过
                    newSrc = src.replace(/([?&]hl=)[^&]+/, '$1en');
                } else {
                    newSrc = src + (src.includes('?') ? '&' : '?') + 'hl=en';
                }
                iframe.setAttribute('src', newSrc);
            }
            // 处理已存在的 iframe（页面预渲染场景）
            document.querySelectorAll('iframe[src*="recaptcha"]').forEach(patchRecaptchaHl);
            // 监听新插入的 iframe
            new MutationObserver(mutations => {
                for (const m of mutations) {
                    for (const node of m.addedNodes) {
                        if (node.nodeType !== 1) continue;
                        if (node.tagName === 'IFRAME') patchRecaptchaHl(node);
                        node.querySelectorAll && node.querySelectorAll('iframe[src*="recaptcha"]').forEach(patchRecaptchaHl);
                    }
                    // 处理 src 被动态修改的情况
                    if (m.type === 'attributes' && m.attributeName === 'src' && m.target.tagName === 'IFRAME') {
                        patchRecaptchaHl(m.target);
                    }
                }
            }).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
        })();
    """)
    log.info("✅ 已注入 reCAPTCHA hl=en 强制英语脚本（iframe 创建时即生效）")

    # ★ 预注入 GDPR consent cookie，让 CMP 一加载就认为用户已同意
    # 目的：避免 CMP 弹窗出现，或即使出现点击按钮时回调不再阻塞主线程
    await page.add_init_script("""
        (() => {
            const expires = new Date(Date.now() + 365*24*3600*1000).toUTCString();
            const cookiePairs = [
                ['euconsent-v2', 'consent_given'],
                ['eupubconsent-v2', 'consent_given'],
                ['sp_lit', '1'],
                ['CookieConsent', 'true'],
                ['cookieconsent_status', 'allow'],
                ['gdpr_consent', '1'],
                ['cmapi_cookie_privacy', 'permit 1,2,3'],
            ];
            for (const [name, val] of cookiePairs) {
                document.cookie = `${name}=${val}; expires=${expires}; path=/; domain=.host2play.gratis`;
                document.cookie = `${name}=${val}; expires=${expires}; path=/`;
            }
            try { localStorage.setItem('CookieConsent', 'true'); } catch(e) {}
            try { localStorage.setItem('gdpr_consent', '1'); } catch(e) {}
        })();
    """)
    log.info("✅ 已预注入 GDPR consent cookie（页面加载时即生效）")

    # ★ 自动关闭广告新 tab：solve_recaptcha 期间广告可能触发 window.open
    # 新页面一旦出现立刻关闭，防止焦点跑走导致验证卡死
    async def _close_popup(popup):
        try:
            log.info(f"  [popup] 检测到新 tab，自动关闭: {popup.url!r}")
            await popup.close()
        except Exception as _e:
            log.debug(f"  [popup] 关闭失败（已关闭？）: {_e}")

    page.on("popup", lambda popup: asyncio.ensure_future(_close_popup(popup)))

    results = []
    try:
        for idx, url in enumerate(RENEW_URLS, 1):
            label = f"server{idx}"
            log.info(f"\n{'='*50}")
            log.info(f"处理第 {idx}/{len(RENEW_URLS)} 个续期链接")
            log.info(f"URL: {url}")
            log.info(f"{'='*50}")

            try:
                ok, before, after = await renew_server(page, url, label)
                results.append((label, ok, before, after))
            except Exception as e:
                log.exception(f"处理 {label} 时发生异常: {e}")
                await take_screenshot(page, f"{label}_exception")
                results.append((label, False, None, None))

            if idx < len(RENEW_URLS):
                log.info("等待 5 秒后处理下一个...")
                await asyncio.sleep(5)

    finally:
        await asyncio.sleep(3)
        await browser.close()
        await botright_client.close()
        log.info("浏览器已关闭")

    lines = ["🔄 host2play.gratis 自动续期报告"]
    all_ok = True
    for label, ok, before, after in results:
        if ok:
            if before and after and before != after:
                lines.append(f"✅ {label}: 续期成功\n   {before} → {after}")
            else:
                lines.append(f"✅ {label}: 续期操作完成（Deletes on: {after}）")
        else:
            all_ok = False
            lines.append(f"❌ {label}: 续期失败（before={before}, after={after}）")

    summary = "\n".join(lines)
    log.info(f"\n{summary}")
    wxpush(summary)

    if not all_ok:
        raise SystemExit("部分服务器续期失败，请查看 Actions 日志和截图")


if __name__ == "__main__":
    asyncio.run(main())
