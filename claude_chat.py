"""Use the installed Claude Code CLI with its existing authentication.
No shell interpolation: user text is sent over UTF-8 stdin, never as a command.
"""
import json
import shutil
import subprocess
import tempfile


def ask_claude(prompt):
    executable = shutil.which('claude.exe') or shutil.which('claude')
    if not executable:
        raise RuntimeError('Không tìm thấy Claude Code. Kiểm tra claude --version trong Terminal rồi khởi động lại VS Code.')
    # A separate working directory prevents loading FruitCareAI files as context.
    # Authentication remains managed by Claude Code; all agent tools are disabled.
    with tempfile.TemporaryDirectory(prefix='fruitcare-chat-') as workdir:
        try:
            result = subprocess.run(
                [executable, '-p', '--disallowedTools', '*', '--output-format', 'json'],
                input=prompt, text=True, encoding='utf-8', errors='replace',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=workdir, timeout=180, shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Claude chưa trả lời sau 180 giây. Hãy thử câu hỏi ngắn hơn hoặc kiểm tra Claude Code trong Terminal.') from exc
        except OSError as exc:
            raise RuntimeError('Không khởi chạy được Claude Code. Kiểm tra lệnh claude trong Terminal.') from exc
    output = result.stdout.strip()
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError('Claude Code không trả về JSON hợp lệ. Thử lệnh kiểm tra Claude trong Terminal để xem lỗi đăng nhập hoặc phiên bản.') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('Định dạng kết quả Claude Code không hợp lệ.')
    if result.returncode != 0 or payload.get('is_error'):
        raise RuntimeError('Claude Code báo lỗi. Kiểm tra đăng nhập, kết nối và hạn mức bằng lệnh claude -p trong Terminal.')
    answer = payload.get('result')
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError('Claude chưa trả về nội dung trả lời. Hãy thử lại câu hỏi.')
    return answer.strip()
