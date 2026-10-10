import cv2
import numpy as np
import json
import os
import datetime
import time
import threading
import sys
import subprocess
from insightface.app import FaceAnalysis
from picamera2 import Picamera2
import vosk
import pyaudio
import json as json_module
import serial
from flask import Flask, request, jsonify, render_template, send_from_directory
import tempfile
import wave
from piper import PiperVoice
os.environ['PYAUDIO_ALSA_USE_MMAP'] = '0'

# ---------- 全局配置 ----------
DB_PATH = "face_db.json"
face_db = {}
LOG_FILE = "medication_log.txt"
authenticated_user = None

# ---------- 用户用药方案（已废弃，改为文件存储） ----------
USER_PLAN_PATH = "user_plans.json"
user_plans = {}  # 保留兼容

# ===== 用户方案存储目录 =====
USER_PLANS_DIR = os.path.expanduser("~/SmartMedicineBox/venv/user_plans")
os.makedirs(USER_PLANS_DIR, exist_ok=True)

# ---------- 唤醒状态 ----------
wake_state = True

# ---------- 初始化 Flask 应用 ----------
app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = os.path.expanduser("~/SmartMedicineBox/temp_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import wave
import os
from piper import PiperVoice
import subprocess

PIPER_MODEL_PATH = os.path.expanduser("~/SmartMedicineBox/zh_CN-huayan-medium.onnx")
_piper_voice = None

def get_piper_voice():
    global _piper_voice
    if _piper_voice is None:
        try:
            _piper_voice = PiperVoice.load(PIPER_MODEL_PATH, use_cuda=False)
            print("✅ Piper TTS 模型加载成功")
        except Exception as e:
            print(f"⚠️ Piper TTS 加载失败: {e}")
            _piper_voice = False
    return _piper_voice if _piper_voice is not False else None

def speak(msg, rate=1.0):
    try:
        output_file = "/tmp/speech.wav"
        if not os.path.exists(PIPER_MODEL_PATH):
            print(f"⚠️ Piper 模型文件不存在: {PIPER_MODEL_PATH}")
            os.system(f'espeak-ng -v cmn "{msg}"')
            return

        cmd = [
            "piper",
            "--model", PIPER_MODEL_PATH,
            "--output_file", output_file,
            "--length_scale", str(rate),
            "--sentence_silence", "0.2"
        ]
        proc = subprocess.run(cmd, input=msg.encode('utf-8'), capture_output=True, timeout=10)
        
        if proc.returncode == 0 and os.path.exists(output_file):
            os.system(f"aplay -D plughw:3,0 -q {output_file}")
        else:
            print(f"⚠️ Piper 合成失败 (返回码 {proc.returncode})")
            if proc.stderr:
                print(f"错误信息: {proc.stderr.decode()}")
            os.system(f'espeak-ng -v cmn "{msg}"')
    except subprocess.TimeoutExpired:
        print("⚠️ Piper 合成超时")
        os.system(f'espeak-ng -v cmn "{msg}"')
    except Exception as e:
        print(f"⚠️ Piper 调用异常: {e}")
        os.system(f'espeak-ng -v cmn "{msg}"')

# ---------- ESP32串口配置 ----------
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    time.sleep(2)
    print("✅ ESP32串口已连接")
except Exception as e:
    print(f"⚠️ ESP32串口未找到: {e}")
    ser = None

def send_to_esp32(command):
    if ser is None:
        print("⚠️ ESP32未连接")
        return False
    try:
        ser.write((command + "\n").encode())
        print(f"📤 发送指令: {command}")
        return True
    except Exception as e:
        print(f"❌ 发送失败: {e}")
        return False

def send_dispense_to_esp32(counts):
    if ser is None:
        print("⚠️ ESP32未连接")
        return False
    if len(counts) == 6:
        counts = list(counts) + [0]
    try:
        cmd = "DISPENSE:" + " ".join(str(x) for x in counts) + "\n"
        ser.write(cmd.encode())
        print(f"📤 发送取药指令: {cmd.strip()}")
        return True
    except Exception as e:
        print(f"❌ 发送失败: {e}")
        return False

# ---------- 初始化AI模型 ----------
print("正在加载AI模型...")
app_face = FaceAnalysis(name='buffalo_l')
app_face.prepare(ctx_id=0, det_size=(320, 320))
print("✅ AI模型加载成功！")

# ---------- 数据库操作 ----------
def load_db():
    global face_db
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        for name in raw:
            face_db[name] = [np.array(emb) for emb in raw[name]]
        print(f"✅ 已加载 {len(face_db)} 位用户")
    else:
        print("📭 暂无用户数据")

def save_db():
    to_save = {}
    for name, emb_list in face_db.items():
        to_save[name] = [emb.tolist() for emb in emb_list]
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(to_save, f, ensure_ascii=False, indent=4)
    print("💾 数据库已保存")

# ---------- 用户用药方案（迁移至新文件夹） ----------
def get_user_plan_path(username):
    safe_name = username.replace('/', '_')
    return os.path.join(USER_PLANS_DIR, f"{safe_name}.json")

def load_user_plan_str(username):
    path = get_user_plan_path(username)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def save_user_plan_str(username, plan_str):
    path = get_user_plan_path(username)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(plan_str)

def migrate_old_user_plans():
    if not os.path.exists(USER_PLAN_PATH):
        return
    if os.listdir(USER_PLANS_DIR):
        print("📂 用户方案目录已有文件，跳过迁移")
        return
    print("🔄 正在迁移旧版 user_plans.json ...")
    with open(USER_PLAN_PATH, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    for name, plan_list in old_data.items():
        if len(plan_list) < 7:
            plan_list = list(plan_list) + [0] * (7 - len(plan_list))
        plan_str = " ".join(str(x) for x in plan_list[:7])
        save_user_plan_str(name, plan_str)
        print(f"  已迁移用户: {name} -> {plan_str}")
    print("✅ 迁移完成")
    os.rename(USER_PLAN_PATH, USER_PLAN_PATH + ".bak")

# ---------- 录入人脸（从文件路径） ----------
def register_face(name, image_paths):
    """录入人脸特征（供 Flask 接口调用）"""
    global face_db
    if name not in face_db:
        face_db[name] = []
    count = 0
    for path in image_paths:
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            continue
        with open(path, 'rb') as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法解码图片: {path}")
            continue
        faces = app_face.get(img)
        if len(faces) == 0:
            print(f"未检测到人脸: {os.path.basename(path)}")
            continue
        embedding = faces[0].normed_embedding
        face_db[name].append(embedding)
        count += 1
        print(f"  已处理: {os.path.basename(path)}")
    if count > 0:
        save_db()
        print(f"🎉 {name} 录入成功！当前共有 {len(face_db[name])} 个特征")
        return True
    else:
        print(f"❌ {name} 录入失败，没有有效照片")
        return False

# ---------- 手动录入人脸（终端交互） ----------
def register_face_interactive():
    name = input("请输入姓名: ").strip()
    if not name:
        print("姓名不能为空")
        return
    path = input("请拖入照片路径: ").strip().replace('"', '')
    if not os.path.exists(path):
        print("文件不存在")
        return
    register_face(name, [path])

# ---------- 日志记录 ----------
def log_action(name):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.datetime.now()}, {name}, opened_box\n")
    print(f"📝 已记录 {name} 的开箱")

# ---------- 人脸识别（无界面） ----------
def start_recognition():
    global authenticated_user
    authenticated_user = None
    print("🔐 认证已重置")
    if not face_db:
        speak("数据库为空，请先录入人脸")
        return

    picam2 = None
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        print("🔍 人脸识别中... 按 Ctrl+C 停止")
    except Exception as e:
        print(f"❌ 摄像头启动失败: {e}")
        speak("摄像头启动失败，请重试")
        if picam2:
            picam2.stop()
            picam2.close()
        return

    last_name = None
    last_time = 0
    recognized = False

    try:
        while not recognized:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            faces = app_face.get(frame_bgr)
            if len(faces) == 0:
                if last_name is not None:
                    print("👤 人脸已离开")
                    last_name = None
                continue

            face = faces[0]
            query_emb = face.normed_embedding
            best_name = None
            best_sim = -1
            for name, emb_list in face_db.items():
                for db_emb in emb_list:
                    sim = np.dot(query_emb, db_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(db_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

            if best_sim >= 0.5:
                display_name = best_name
                current_time = time.time()
                if display_name != last_name or (current_time - last_time) > 5:
                    authenticated_user = display_name
                    print(f"✅ 识别成功：{authenticated_user}，相似度：{best_sim:.2f}")
                    log_action(display_name)
                    speak(f"{authenticated_user}，认证成功，请说“取药”获取您的药方")
                    recognized = True
                    last_name = display_name
                    last_time = current_time
            else:
                if last_name is not None:
                    print("❌ 未知人脸")
                    authenticated_user = None
                    last_name = None

    except KeyboardInterrupt:
        print("\n👋 人脸识别已停止")
    except Exception as e:
        print(f"⚠️ 人脸识别异常: {e}")
    finally:
        if picam2:
            picam2.stop()
            picam2.close()
            print("📷 摄像头已释放")
        if not recognized:
            speak("人脸识别未完成，请重试")

# ---------- 处理语音指令 ----------
def handle_voice_command(text):
    global authenticated_user, wake_state
    clean_text = text.replace(" ", "")
    print(f"📢 语音指令（原: {text}，清洗后: {clean_text})")

    wake_keywords = ["健康助手", "健康住手", "住手", "助手"]
    if any(kw in clean_text for kw in wake_keywords):
        if not wake_state:
            wake_state = True
            authenticated_user = None
            print("🔊 唤醒成功，已重新认证")
            speak("我是智能药箱，已为您更新身份，请先进行人脸识别")
        else:
            authenticated_user = None
            print("🔊 已唤醒，重置认证状态")
            speak("已更新，请先进行人脸识别")
        return

    if "关闭" in clean_text or "关箱" in clean_text:
        if wake_state:
            wake_state = False
            authenticated_user = None
            print("🔇 进入休眠状态")
            speak("期待下一次和您见面")
        return

    if not wake_state:
        return

    if "取药" in clean_text or "吃药" in clean_text:
        if authenticated_user is None:
            speak("请先进行人脸识别")
            return
        plan_str = load_user_plan_str(authenticated_user)
        if plan_str is None:
            speak(f"{authenticated_user}，您还没有配置用药方案，请联系管理员")
            return
        try:
            counts = [int(x) for x in plan_str.split()]
            if len(counts) < 7:
                counts = counts + [0] * (7 - len(counts))
            elif len(counts) > 7:
                counts = counts[:7]
            speak(f"{authenticated_user}，正在为您取药")
            send_dispense_to_esp32(counts)
        except Exception as e:
            print(f"解析药方失败: {e}")
            speak("药方数据错误，请重新设置")
        return

    if "人脸识别" in clean_text or "身份认证" in clean_text:
        speak("正在启动人脸识别")
        start_recognition()
        return

    if "录入人脸" in clean_text or ("录入" in clean_text and "人脸" in clean_text):
        speak("请按照提示录入人脸")
        register_face_interactive()
        return

    if "查看日志" in clean_text or "日志" in clean_text:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                print(f.read())
            speak("日志已显示，请查看终端")
        else:
            speak("暂无日志记录")
        return

    if "退出" in clean_text or "关闭系统" in clean_text:
        speak("再见")
        sys.exit(0)

    greetings = ['你好', 'Hello', '在吗']
    if any(word in clean_text for word in greetings):
        speak("你好！我是智能药箱")
        return

    open_orders = ['打开', '开箱', '开门']
    if any(word in clean_text for word in open_orders):
        if authenticated_user:
            speak(f"认证通过，{authenticated_user}，药箱已开启")
        else:
            speak("请先进行人脸识别")
        return

    check_orders = ['查询', '用药']
    if any(word in clean_text for word in check_orders):
        if authenticated_user:
            plan_str = load_user_plan_str(authenticated_user)
            if plan_str:
                speak(f"{authenticated_user}，您的用药方案已准备")
            else:
                speak("您还没有配置用药方案")
        else:
            speak("请先进行人脸识别")
        return

    speak("我没听清，请再说一遍好吗？")

# ---------- 语音识别线程 ----------
def voice_listener():
    try:
        if not os.path.exists("model"):
            print("⚠️ 语音模型不存在")
            return
        model = vosk.Model("model")
        rec = vosk.KaldiRecognizer(model, 16000)

        p = pyaudio.PyAudio()
        device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"设备 {i}: {info['name']} (输入通道: {info['maxInputChannels']})")
            if "seeed" in info["name"].lower() or "reSpeaker" in info["name"].lower():
                device_index = i
                print(f"✅ 找到 ReSpeaker 设备: {info['name']} (索引 {i})")
                break
        if device_index is None:
            print("⚠️ 未找到 ReSpeaker，使用默认设备")
            device_index = p.get_default_input_device_info()["index"]
            print(f"使用默认输入设备索引: {device_index}")

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=4000
        )

        print("🎤 语音监听已启动...")
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json_module.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    handle_voice_command(text)
    except Exception as e:
        print(f"⚠️ 语音监听异常: {e}")

# ============================================================
# ---------- Flask Web 路由 ----------
# ============================================================

# ===== 新增：药方持久化辅助函数（保留但不再使用） =====
PLANS_FILE = "plans.json"
def load_plans_from_file():
    if os.path.exists(PLANS_FILE):
        with open(PLANS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_plans_to_file(plans):
    with open(PLANS_FILE, 'w', encoding='utf-8') as f:
        json.dump(plans, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    return render_template('index.html')

# ===== 修改：上传照片并保存副本到持久目录 =====
@app.route('/upload', methods=['POST'])
def upload_photo():
    try:
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({'success': False, 'message': '姓名不能为空'}), 400
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '未选择文件'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名为空'}), 400

        # 保存临时文件（用于 face 提取）
        temp_path = os.path.join(UPLOAD_FOLDER, f"{name}_{int(time.time())}.jpg")
        file.save(temp_path)

        # 录入人脸特征
        result = register_face(name, [temp_path])

        if result:
            return jsonify({'success': True, 'message': f'{name} 录入成功'})
        else:
            # 录入失败则删除临时文件（可选）
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'success': False, 'message': '人脸录入失败，请检查照片是否清晰'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ===== 新增：获取用户照片列表 =====
@app.route('/get_user_photos', methods=['GET'])
def get_user_photos():
    """返回所有已保存的照片信息，按用户名分组"""
    photos = {}
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 解析用户名：假设格式为 用户名_时间戳.扩展名
            # 可能包含下划线，我们取第一个下划线之前的部分作为用户名
            base = os.path.splitext(fname)[0]
            parts = base.split('_')
            if len(parts) >= 2:
                username = '_'.join(parts[:-1])  # 如果用户名中本来就有下划线，则全部合并
            else:
                username = base  # 兼容旧格式
            if username not in photos:
                photos[username] = []
            photos[username].append(fname)
    # 转换格式：返回数组 {name, files}
    result = [{'name': k, 'photos': v} for k, v in photos.items()]
    return jsonify({'photos': result})

# ===== 新增：提供照片文件（用于前端预览） =====
@app.route('/photo/<filename>')
def get_photo(filename):
    """返回照片文件"""
    return send_from_directory(UPLOAD_FOLDER, filename)

# ===== 原有路由 =====
@app.route('/cmd', methods=['GET'])
def handle_cmd():
    try:
        cmd = request.args.get('c', '')
        if not cmd:
            return "No command", 400
        print(f"📥 收到网页指令: {cmd}")
        if ser is not None:
            ser.write((cmd + "\n").encode())
            print(f"📤 已转发: {cmd}")
            return f"OK: {cmd}"
        else:
            return "ESP32 not connected", 503
    except Exception as e:
        print(f"⚠️ 指令处理失败: {e}")
        return f"Error: {e}", 500

@app.route('/status', methods=['GET'])
def status():
    return "Idle"

@app.route('/get_user_list', methods=['GET'])
def get_user_list():
    return jsonify({'users': list(face_db.keys())})

@app.route('/get_user_plan', methods=['GET'])
def get_user_plan():
    name = request.args.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': '用户名不能为空'}), 400
    plan_str = load_user_plan_str(name)
    if plan_str is None:
        return jsonify({'success': True, 'plan': [0,0,0,0,0,0,0]})
    try:
        counts = [int(x) for x in plan_str.split()]
        if len(counts) < 7:
            counts = counts + [0] * (7 - len(counts))
        elif len(counts) > 7:
            counts = counts[:7]
        return jsonify({'success': True, 'plan': counts})
    except:
        return jsonify({'success': True, 'plan': [0,0,0,0,0,0,0]})

@app.route('/save_user_plan', methods=['POST'])
def save_user_plan():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        plan = data.get('plan', [])
        if not name:
            return jsonify({'success': False, 'message': '用户名不能为空'}), 400
        if not isinstance(plan, list) or len(plan) < 6 or len(plan) > 7:
            return jsonify({'success': False, 'message': '需要6或7个数字'}), 400
        if len(plan) == 6:
            plan = list(plan) + [0]
        elif len(plan) > 7:
            plan = plan[:7]
        plan_str = " ".join(str(x) for x in plan)
        save_user_plan_str(name, plan_str)
        return jsonify({'success': True, 'message': f'{name} 的用药方案已保存'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/esp32/dispense', methods=['GET'])
def esp32_dispense():
    try:
        cmd = request.args.get('cmd', '')
        if not cmd:
            return jsonify({'success': False, 'message': '缺少 cmd 参数'}), 400
        parts = cmd.strip().split(' ')
        counts = [int(p) for p in parts if p.isdigit()]
        if len(counts) < 6 or len(counts) > 7:
            return jsonify({'success': False, 'message': '需要6或7个数字'}), 400
        if len(counts) == 6:
            counts = list(counts) + [0]
        success = send_dispense_to_esp32(counts)
        if success:
            return jsonify({'success': True, 'message': '取药指令已发送'})
        else:
            return jsonify({'success': False, 'message': 'ESP32 发送失败'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/esp32/status', methods=['GET'])
def esp32_status():
    return "Idle"

@app.route('/esp32/reset', methods=['GET'])
def esp32_reset():
    try:
        send_to_esp32("RESET")
        return "Reset OK"
    except Exception as e:
        return f"Reset failed: {e}"

@app.route('/esp32/rotateBase', methods=['GET'])
def esp32_rotate_base():
    try:
        index = request.args.get('index', '0')
        cmd = f"FILL:{index}"
        send_to_esp32(cmd)
        return "Done"
    except Exception as e:
        return f"Failed: {e}"

@app.route('/esp32/saveCurrentPlan', methods=['GET'])
def esp32_save_current_plan():
    try:
        cmd = request.args.get('cmd', '')
        if not cmd:
            return "No command"
        send_to_esp32(f"SAVE_PLAN:{cmd}")
        return "Plan saved"
    except Exception as e:
        return f"Failed: {e}"

# ===== 保留旧版药方 API（可选） =====
@app.route('/api/plans', methods=['GET', 'POST'])
def handle_plans():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        cmd = data.get('cmd')
        if not name or cmd is None:
            return jsonify({'success': False, 'message': '缺少参数'}), 400
        all_plans = load_plans_from_file()
        all_plans[name] = cmd
        save_plans_to_file(all_plans)
        return jsonify({'success': True, 'message': f'药方 {name} 已保存到树莓派'})
    else:
        all_plans = load_plans_from_file()
        return jsonify(all_plans)

@app.route('/api/plans/<name>', methods=['DELETE'])
def delete_plan(name):
    all_plans = load_plans_from_file()
    if name in all_plans:
        del all_plans[name]
        save_plans_to_file(all_plans)
        return jsonify({'success': True, 'message': f'已删除 {name}'})
    return jsonify({'success': False, 'message': '药方不存在'}), 404

# ---------- 启动 Flask ----------
def start_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ---------- 主菜单 ----------
def main():
    migrate_old_user_plans()
    load_db()
    threading.Thread(target=voice_listener, daemon=True).start()
    threading.Thread(target=start_flask, daemon=True).start()
    print("🌐 Flask Web 服务器已启动，访问 http://树莓派IP:5000")
    speak("欢迎您使用智能药箱，请先进行人脸识别。如需取药，请说“取药”。")

    while True:
        print("\n" + "="*30)
        print("   智能药箱 AI 系统（语音控制 + Web 控制台）")
        print("="*30)
        print("语音指令: 健康助手（唤醒）| 关闭（休眠）| 人脸识别 | 录入人脸 | 查看日志 | 取药 | 退出")
        print("="*30)
        print("网页控制台: http://树莓派IP:5000")
        print("="*30)
        print("1. 录入人脸")
        print("2. 启动人脸识别")
        print("3. 查看日志")
        print("4. 测试语音")
        print("5. 退出")
        choice = input("请选择: ")

        if choice == "1":
            register_face_interactive()
        elif choice == "2":
            start_recognition()
        elif choice == "3":
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    print(f.read())
            else:
                print("暂无日志")
        elif choice == "4":
            speak("你好，我是智能药箱，语音测试成功")
        elif choice == "5":
            speak("再见")
            break

if __name__ == "__main__":
    main()