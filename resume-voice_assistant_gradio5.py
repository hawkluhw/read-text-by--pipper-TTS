#!/usr/bin/env python3
"""
Voice Assistant App - Gradio 5 with Resume Support
新增功能：Stop 后重新点击 Read English 从上次停止的位置继续播放
"""

import os
import threading
import subprocess
import logging
import time
import re
from typing import List, Tuple, Optional

import gradio as gr

# ==================== Configuration ====================
PIPER_BIN = '/home/test/my-venv/venv-3.12/bin/piper'
PIPER_MODEL = '/home/test/.local/share/piper/models/en_US-lessac-high.onnx'
PIPER_MAX_CHARS = 4000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ==================== State Management ====================
class AppState:
    def __init__(self):
        self.lines_cache = {}
        self.cache_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.current_thread: Optional[threading.Thread] = None
        self.thread_lock = threading.Lock()
        self.current_procs = {"piper": None, "paplay": None}
        self.procs_lock = threading.Lock()
        self.now_playing = ""
        self.now_playing_lock = threading.Lock()
        self.now_playing_nonce = 0
        self.nonce_lock = threading.Lock()

        # 新增：恢复播放相关状态
        self.saved_text = ""           # 上次播放的完整文本
        self.saved_sentences = []      # 上次分割后的句子列表
        self.saved_idx = 0             # 上次停止时的句子索引
        self.saved_chunk_select = ""   # 上次播放的 chunk 标识
        self.resume_lock = threading.Lock()

    def reset_stop(self):
        self.stop_event.clear()

    def request_stop(self):
        self.stop_event.set()

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def set_now_playing(self, text: str):
        with self.now_playing_lock:
            self.now_playing = text
        with self.nonce_lock:
            self.now_playing_nonce += 1

    def get_now_playing(self) -> str:
        with self.now_playing_lock:
            text = self.now_playing
        with self.nonce_lock:
            nonce = self.now_playing_nonce
        return f"{text} [{nonce}]"

    def clear_now_playing(self):
        with self.now_playing_lock:
            self.now_playing = ""
        with self.nonce_lock:
            self.now_playing_nonce += 1

    def save_progress(self, text: str, sentences: List[str], idx: int, chunk_select: str):
        """保存播放进度"""
        with self.resume_lock:
            self.saved_text = text
            self.saved_sentences = sentences
            self.saved_idx = idx
            self.saved_chunk_select = chunk_select

    def get_resume_info(self, chunk_select: str) -> Tuple[bool, List[str], int]:
        """
        检查是否可以恢复播放
        返回: (是否可以恢复, 句子列表, 开始索引)
        """
        with self.resume_lock:
            if self.saved_chunk_select == chunk_select and self.saved_idx > 0:
                if self.saved_idx < len(self.saved_sentences):
                    return True, self.saved_sentences, self.saved_idx
            return False, [], 0

    def clear_progress(self):
        """清除保存的进度"""
        with self.resume_lock:
            self.saved_text = ""
            self.saved_sentences = []
            self.saved_idx = 0
            self.saved_chunk_select = ""

    def register_thread(self, thread: threading.Thread) -> bool:
        with self.thread_lock:
            if self.current_thread and self.current_thread.is_alive():
                return False
            self.current_thread = thread
            return True

    def join_thread(self, timeout: float = 2.0):
        with self.thread_lock:
            thread = self.current_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def set_proc(self, name: str, proc: Optional[subprocess.Popen]):
        with self.procs_lock:
            self.current_procs[name] = proc

    def get_proc(self, name: str) -> Optional[subprocess.Popen]:
        with self.procs_lock:
            return self.current_procs.get(name)

    def terminate_processes(self):
        with self.procs_lock:
            procs = dict(self.current_procs)

        for name, proc in procs.items():
            if proc is None:
                continue
            try:
                proc.terminate()
                logging.info("Sent terminate to %s (pid=%d)", name, proc.pid)
            except Exception as e:
                logging.warning("Failed to terminate %s: %s", name, e)

        time.sleep(0.3)

        for name, proc in procs.items():
            if proc is None:
                continue
            try:
                if proc.poll() is None:
                    proc.kill()
                    logging.info("Sent kill to %s (pid=%d)", name, proc.pid)
            except Exception as e:
                logging.warning("Failed to kill %s: %s", name, e)

        with self.procs_lock:
            for key in self.current_procs:
                self.current_procs[key] = None

    def cache_lines(self, path: str, mtime: float, size: int, lines: List[str]):
        with self.cache_lock:
            self.lines_cache[path] = (mtime, size, lines)

    def get_cached_lines(self, path: str, mtime: float, size: int) -> Optional[List[str]]:
        with self.cache_lock:
            cached = self.lines_cache.get(path)
            if cached and cached[0] == mtime and cached[1] == size:
                return cached[2]
            return None


_state = AppState()


# ==================== File Operations ====================
def _cache_key_for_file(file_obj) -> Optional[Tuple[str, float, int]]:
    if file_obj is None:
        return None
    path = getattr(file_obj, "name", None)
    if not path or not os.path.exists(path):
        return None
    try:
        st = os.stat(path)
        return (path, st.st_mtime, st.st_size)
    except OSError:
        return None


def _get_lines_cached(file_obj) -> List[str]:
    key = _cache_key_for_file(file_obj)
    if key is None:
        return []

    path, mtime, size = key
    cached = _state.get_cached_lines(path, mtime, size)
    if cached is not None:
        return cached

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        _state.cache_lines(path, mtime, size, lines)
        return lines
    except UnicodeDecodeError:
        logging.error("File %s is not valid UTF-8", path)
        return []
    except OSError as e:
        logging.error("Failed to read file %s: %s", path, e)
        return []


# ==================== Chunk Management ====================
def get_chunks(file_path, chunk_size) -> Tuple[str, List[str]]:
    if file_path is None or chunk_size is None:
        return "Total lines: 0", []

    try:
        chunk_size = max(1, int(chunk_size))
    except (ValueError, TypeError):
        chunk_size = 1

    lines = _get_lines_cached(file_path)
    total = len(lines)
    if total == 0:
        return "Total lines: 0", []

    n_chunks = (total + chunk_size - 1) // chunk_size
    choices = []
    for i in range(n_chunks):
        start = i * chunk_size + 1
        end = min((i + 1) * chunk_size, total)
        choices.append(f"Chunk {i+1}: Line {start} to {end}")

    return f"Total lines: {total}", choices


def show_chunk(file_path, chunk_choice) -> str:
    if file_path is None or not chunk_choice:
        return ""

    lines = _get_lines_cached(file_path)
    parts = chunk_choice.split(": Line ")
    if len(parts) < 2:
        return ""

    try:
        line_range = parts[1].strip()
        start_s, end_s = line_range.split(" to ")
        start, end = int(start_s), int(end_s)
    except (ValueError, IndexError):
        return ""

    start = max(1, start)
    end = min(len(lines), end)
    if start > end:
        return ""

    return "".join(lines[start - 1:end])


# ==================== Text Processing ====================
def _preprocess_text(text: str) -> str:
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('**'):
            continue
        line = re.sub(r'(?<!\*)\*\*([^\s*].*?[^\s*])\*\*(?!\*)', r'\1', line)
        idx = line.find('** ')
        if idx != -1:
            line = line[:idx]
        filtered_lines.append(line)
    return '\n'.join(filtered_lines)


def _split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    result = []
    i = 0
    current = ""

    while i < len(text):
        if text[i] == '"':
            if current.strip():
                parts = re.split(r'(?<=[\.\?\!]\s)', current)
                for p in parts:
                    p = p.strip()
                    if p:
                        result.append(p)
                current = ""

            end_quote = text.find('"', i + 1)
            if end_quote == -1:
                end_quote = len(text)

            quote_content = text[i+1:end_quote]
            if quote_content.strip():
                result.append(quote_content.strip())

            i = end_quote + 1
        else:
            current += text[i]
            i += 1

    if current.strip():
        parts = re.split(r'(?<=[\.\?\!]\s)', current)
        for p in parts:
            p = p.strip()
            if p:
                result.append(p)

    return [s for s in result if s]


# ==================== TTS Engine ====================
def _run_piper_to_paplay(text: str, rate: str = '22050'):
    if not text or _state.is_stopped():
        return

    piper_cmd = [PIPER_BIN, '--model', PIPER_MODEL, '--output-raw']
    paplay_cmd = ['paplay', '--rate', rate, '--format', 's16le', '--channels', '1', '--raw']

    piper_proc = None
    paplay_proc = None

    try:
        piper_proc = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _state.set_proc('piper', piper_proc)
        logging.info("Started piper (pid=%d)", piper_proc.pid)
    except FileNotFoundError:
        logging.error("Piper binary not found at %s", PIPER_BIN)
        return
    except OSError as e:
        logging.error("Failed to start piper: %s", e)
        return

    try:
        paplay_proc = subprocess.Popen(
            paplay_cmd,
            stdin=piper_proc.stdout
        )
        _state.set_proc('paplay', paplay_proc)
        logging.info("Started paplay (pid=%d)", paplay_proc.pid)
    except FileNotFoundError:
        logging.error("paplay not found, please install pulseaudio-utils")
        _state.terminate_processes()
        return
    except OSError as e:
        logging.error("Failed to start paplay: %s", e)
        _state.terminate_processes()
        return

    try:
        with _state.procs_lock:
            if _state.is_stopped():
                _state.terminate_processes()
                return
            piper_proc.stdin.write(text.encode('utf-8'))
            piper_proc.stdin.close()
    except (BrokenPipeError, OSError) as e:
        logging.warning("Failed to write to piper stdin: %s", e)
        _state.terminate_processes()
        return

    try:
        while not _state.is_stopped():
            pap = _state.get_proc('paplay')
            if pap is None:
                break
            ret = pap.poll()
            if ret is not None:
                break
            time.sleep(0.1)
    finally:
        _state.terminate_processes()


def _tts_worker(text: str, sentences: List[str], start_idx: int, chunk_select: str):
    """
    TTS 后台工作线程
    从 start_idx 开始播放，支持恢复
    """
    try:
        logging.info("Starting playback from sentence %d/%d", start_idx + 1, len(sentences))

        for i in range(start_idx, len(sentences)):
            if _state.is_stopped():
                logging.info("Stopped at sentence %d", i)
                # 保存进度：下次从这句开始
                _state.save_progress(text, sentences, i, chunk_select)
                return

            sentence = sentences[i].strip()
            if not sentence:
                continue

            # 更新当前索引（用于进度保存）
            _state.save_progress(text, sentences, i, chunk_select)

            # 如果句子太长，进一步分块
            if len(sentence) > PIPER_MAX_CHARS:
                sub_chunks = []
                cur = ""
                sub_sentences = re.split(r'(?<=[\.\?\!]\s)', sentence)
                for s in sub_sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(cur) + len(s) + 1 <= PIPER_MAX_CHARS:
                        cur = (cur + " " + s).strip() if cur else s
                    else:
                        if cur:
                            sub_chunks.append(cur)
                        cur = s
                if cur:
                    sub_chunks.append(cur)

                for j, sub in enumerate(sub_chunks):
                    if _state.is_stopped():
                        _state.save_progress(text, sentences, i, chunk_select)
                        return
                    _state.set_now_playing(sub)
                    logging.info(
                        "Sentence %d/%d sub-chunk %d/%d (%d chars): %s",
                        i + 1, len(sentences), j + 1, len(sub_chunks), len(sub),
                        sub[:50] + "..." if len(sub) > 50 else sub
                    )
                    time.sleep(0.1)
                    _run_piper_to_paplay(sub)
                    time.sleep(0.05)
            else:
                _state.set_now_playing(sentence)
                logging.info(
                    "Sentence %d/%d (%d chars): %s",
                    i + 1, len(sentences), len(sentence),
                    sentence[:50] + "..." if len(sentence) > 50 else sentence
                )
                time.sleep(0.1)
                _run_piper_to_paplay(sentence)
                time.sleep(0.05)

        # 播放完成，清除进度
        logging.info("Playback completed, clearing progress")
        _state.clear_progress()

    except Exception:
        logging.exception("Error in TTS worker")
        # 出错时也保存进度
        _state.save_progress(text, sentences, _state.saved_idx, chunk_select)
    finally:
        _state.terminate_processes()
        _state.clear_now_playing()
        logging.info("TTS worker finished")


# ==================== UI Handlers ====================
def stop_tts():
    logging.info("Stop requested")
    _state.request_stop()
    _state.terminate_processes()
    _state.join_thread(timeout=2.0)
    _state.clear_now_playing()
    return _state.get_now_playing()


def read_english_start(file_path, chunk_select, chunk_size) -> Tuple[str, str, str]:
    """
    开始朗读，支持从上次停止位置恢复
    返回: (chunk_text, now_playing, button_text)
    """
    # 先停止当前播放
    _state.request_stop()
    _state.terminate_processes()
    _state.join_thread(timeout=2.0)
    _state.reset_stop()

    if file_path is None or not chunk_select:
        _state.clear_progress()
        return "", "", "Read English"

    lines = _get_lines_cached(file_path)
    parts = chunk_select.split(": Line ")
    if len(parts) < 2:
        _state.clear_progress()
        return "", "", "Read English"

    try:
        line_range = parts[1].strip()
        start_s, end_s = line_range.split(" to ")
        start, end = int(start_s), int(end_s)
    except (ValueError, IndexError):
        _state.clear_progress()
        return "", "", "Read English"

    start = max(1, start)
    end = min(len(lines), end)
    if start > end:
        _state.clear_progress()
        return "", "", "Read English"

    current_text = "".join(lines[start - 1:end])

    # 检查是否可以恢复播放
    can_resume, saved_sentences, saved_idx = _state.get_resume_info(chunk_select)

    if can_resume:
        sentences = saved_sentences
        start_idx = saved_idx
        logging.info("Resuming playback from sentence %d/%d", start_idx + 1, len(sentences))
        button_text = "Resume"
    else:
        # 新播放：预处理并分割
        processed_text = _preprocess_text(current_text)
        sentences = _split_into_sentences(processed_text)
        start_idx = 0
        logging.info("Starting new playback with %d sentences", len(sentences))
        button_text = "Read English"

    if not sentences:
        return current_text, "", button_text

    worker = threading.Thread(
        target=_tts_worker,
        args=(current_text, sentences, start_idx, chunk_select),
        daemon=False
    )
    if not _state.register_thread(worker):
        logging.warning("Previous TTS thread still running")
        return current_text, _state.get_now_playing(), button_text

    logging.info("Starting TTS thread for %d chars", len(current_text))
    worker.start()

    return current_text, _state.get_now_playing(), button_text


def get_now_playing() -> str:
    return _state.get_now_playing()


# ==================== Gradio 5 UI ====================
with gr.Blocks() as app:
    gr.Markdown("# Voice Assistant App (Gradio 5 with Resume)")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload File")
            total_lines = gr.Textbox(label="", lines=1, interactive=False)
            chunk_size_input = gr.Number(
                label="Chunk Size (lines per chunk)",
                value=10,
                precision=0
            )
            chunk_select = gr.Radio(label="Select Chunk", choices=[])

        with gr.Column():
            text_output = gr.Textbox(
                label="File Content",
                lines=15,
                elem_id="file_content_textbox",
                autoscroll=False
            )
            now_playing = gr.Textbox(
                label="Now Playing",
                lines=3,
                interactive=False
            )

    with gr.Row():
        read_button = gr.Button("Read English", variant="primary")
        stop_button = gr.Button("Stop", variant="stop")

    # Gradio 5 Timer
    timer = gr.Timer(value=0.1, active=True)
    timer.tick(fn=get_now_playing, outputs=[now_playing])

    # Event bindings
    def on_input_change(file_path, chunk_size):
        total, choices = get_chunks(file_path, chunk_size)
        return total, gr.update(choices=choices, value=choices[0] if choices else "")

    def on_chunk_change(file_path, chunk_select):
        # 切换 chunk 时清除之前的进度
        _state.clear_progress()
        return show_chunk(file_path, chunk_select)

    file_input.change(
        fn=on_input_change,
        inputs=[file_input, chunk_size_input],
        outputs=[total_lines, chunk_select]
    )
    chunk_size_input.change(
        fn=on_input_change,
        inputs=[file_input, chunk_size_input],
        outputs=[total_lines, chunk_select]
    )

    chunk_select.change(
        fn=on_chunk_change,
        inputs=[file_input, chunk_select],
        outputs=text_output,
        js="""
        () => {
            setTimeout(function() {
                var textareas = document.querySelectorAll('textarea');
                for (var i = 0; i < textareas.length; i++) {
                    var ta = textareas[i];
                    var label = ta.closest('[class*="block"]');
                    if (label && label.textContent.includes('File Content')) {
                        ta.scrollTop = 0;
                        break;
                    }
                }
            }, 100);
            return [];
        }
        """
    )

    read_button.click(
        fn=read_english_start,
        inputs=[file_input, chunk_select, chunk_size_input],
        outputs=[text_output, now_playing, read_button]
    )
    stop_button.click(
        fn=stop_tts,
        inputs=[],
        outputs=[now_playing]
    )

    app.queue()


if __name__ == "__main__":
    app.launch()
