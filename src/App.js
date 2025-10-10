// App.tsx / App.jsx

import React, { useEffect, useRef, useState } from 'react';
import './App.css';

// ===== TensorFlow.js core & backend =====
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl'; // Ưu tiên chạy trên GPU (WebGL)

// ===== Mô hình & bộ phân loại =====
import * as mobilenet from '@tensorflow-models/mobilenet'; // Dùng làm bộ trích xuất đặc trưng (feature extractor)
import * as knnClassifier from '@tensorflow-models/knn-classifier'; // KNN phân loại 'touched' / 'not_touch'

// ===== Âm thanh & thông báo =====
import { Howl } from 'howler'; // Điều khiển phát âm thanh
import { initNotifications, notify } from '@mycv/f8-notification'; // Thông báo trình duyệt (Web Notification)

// ===== Lưu/đọc IndexedDB (dạng key-value) =====
import { set as idbSet, get as idbGet, del as idbDel } from 'idb-keyval';

import soundURL from './assets/sound/Hey-Take_your_hand_off.mp3';

// Tạo đối tượng âm thanh từ file mp3
const sound = new Howl({ src: [soundURL] });

// ===== Các hằng & cấu hình =====
const NOT_TOUCH_LABEL = 'not_touch'; // Nhãn cho trạng thái KHÔNG CHẠM
const TOUCHED_LABEL = 'touched';     // Nhãn cho trạng thái CÓ CHẠM
const TRAINING_TIMES = 50;           // Mỗi lần nhấn Train sẽ thêm 50 mẫu (× batches)
const LOOP_DELAY_MS = 160;           // Độ trễ mục tiêu mỗi vòng dự đoán (~6.25 FPS nếu mỗi vòng ~160ms)
const LS_KEY = 'touch-guard-dataset-v1'; // Key lưu dataset vào IndexedDB
const MUTE_KEY = 'tg-muted';             // Key lưu trạng thái tắt âm vào localStorage
const COMPRESS_TO_CENTROID = true;       // Nén mẫu KNN bằng vector trung bình (centroid) cho mỗi lớp
const AUTO_RESET_ON_EXIT = true;         // Nếu true: không tải lại dataset cũ (xóa khi khởi động)

// ===== Component chính =====
function App() {
  // ==== Refs / States ====
  // useRef: lưu tham chiếu hoặc biến không gây re-render khi thay đổi
  const video = useRef(null);            // Tham chiếu tới thẻ <video> hiển thị webcam
  const classifier = useRef(null);       // KNN classifier instance
  const mobilenetModule = useRef(null);  // MobileNet đã load
  const canPlaySound = useRef(true);     // Chống phát âm thanh liên tục (chỉ phát khi flag true)
  const didInit = useRef(false);         // Đảm bảo init() chỉ chạy 1 lần
  const stopTracksRef = useRef(() => { });// Hàm dừng stream camera hiện tại
  const runningRef = useRef(false);      // Cờ đang chạy vòng dự đoán (run loop)
  const frameCountRef = useRef(0);       // Đếm frame để tính FPS
  const lastFpsTimeRef = useRef(performance.now()); // Thời điểm gần nhất cập nhật FPS
  const trainingLockRef = useRef(false); // Khóa tránh bấm Train lặp (song song)

  // ==== State giao diện & logic ====
  const [touched, setTouched] = useState(false); // Đang phát hiện 'touched' hay không
  const [step, setStep] = useState(1);           // Bước UI: 1 Train not, 2 Train touch, 3 Ready, 4 Running
  const [progress, setProgress] = useState({ not: 0, touch: 0 }); // % tiến độ Train mỗi lớp (UI)
  const [exampleCount, setExampleCount] = useState({ not: 0, touch: 0 }); // Số mẫu đã thu mỗi lớp
  const [confidence, setConfidence] = useState(0.8); // Ngưỡng xác suất để coi là 'touched'
  const [batches, setBatches] = useState(1);         // Hệ số nhân số mẫu khi Train
  const [isTraining, setIsTraining] = useState(false); // Cờ đang train (để disable nút…)

  // Lưu trạng thái mute vào localStorage (khởi tạo)
  const [muted, setMuted] = useState(() => {
    try { return localStorage.getItem(MUTE_KEY) === 'true'; } catch { return false; }
  });

  // Một số trạng thái hiển thị
  const [loading, setLoading] = useState(true); // Đang khởi tạo TF.js + camera + model
  const [backend, setBackend] = useState('');   // Backend đang dùng: 'webgl' hoặc 'cpu'
  const [error, setError] = useState('');       // Thông báo lỗi hiển thị UI
  const [fps, setFps] = useState(0);            // FPS đo được thực tế

  // Danh sách camera & camera đang chọn
  const [devices, setDevices] = useState([]);   // Các videoinput devices
  const [deviceId, setDeviceId] = useState(''); // ID của camera hiện tại

  // Chỉ cho phép chạy run() khi đã có cả 2 lớp mẫu (>0)
  const allowRun = exampleCount.not > 0 && exampleCount.touch > 0;

  // ==== Persist trạng thái mute ====
  useEffect(() => {
    // Lưu muted vào localStorage, đồng thời bật/tắt âm của Howler
    try { localStorage.setItem(MUTE_KEY, String(muted)); } catch { }
    try { sound.mute(muted); } catch { }
  }, [muted]);

  // ==== Hàm khởi tạo toàn bộ (TF.js, camera, model, dataset, notification) ====
  const init = async () => {
    if (didInit.current) return; // Đảm bảo chỉ init một lần
    didInit.current = true;

    setError('');
    setLoading(true);

    try {
      // Ưu tiên backend 'webgl', nếu lỗi fallback sang 'cpu'
      try { await tf.setBackend('webgl'); } catch { await tf.setBackend('cpu'); }
      await tf.ready(); // Chờ TF.js sẵn sàng
      setBackend(tf.getBackend()); // Lưu tên backend đang dùng để hiển thị

      // Yêu cầu quyền camera trước để enumerateDevices có label
      await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const mediaDevices = await navigator.mediaDevices.enumerateDevices();
      const cams = mediaDevices.filter(d => d.kind === 'videoinput'); // Lọc ra video đầu vào
      setDevices(cams);
      setDeviceId(cams[0]?.deviceId || ''); // Chọn camera đầu tiên

      // Mở stream camera & gán vào <video>
      const stream = await setupCamera(cams[0]?.deviceId || undefined);
      // Lưu hàm dừng stream để dùng khi switch camera / unmount
      stopTracksRef.current = () => stream.getTracks().forEach(t => t.stop());

      // Tạo KNN classifier & load MobileNet
      classifier.current = knnClassifier.create();
      mobilenetModule.current = await mobilenet.load();

      // Tải dataset (nếu cấu hình cho phép)
      await tryLoadDataset();

      // Khởi tạo hệ thống thông báo (cooldown 3s)
      initNotifications({ cooldown: 3000 });

      setLoading(false);
    } catch (e) {
      console.error(e);
      setError('Không thể khởi tạo camera hoặc model. Kiểm tra quyền camera / reload trang.');
      setLoading(false);
    }
  };

  // ==== Xin stream camera và gán vào phần tử <video> ====
  const setupCamera = (devId) =>
    new Promise(async (resolve, reject) => {
      try {
        // Nếu có deviceId cụ thể thì chọn đúng camera đó
        const stream = await navigator.mediaDevices.getUserMedia({
          video: devId ? { deviceId: { exact: devId } } : true,
          audio: false
        });
        if (!video.current) return reject(new Error('Video element missing'));
        // Gán stream vào thẻ video
        video.current.srcObject = stream;
        // Khi video đã có data khung hình đầu tiên thì resolve
        video.current.addEventListener('loadeddata', () => resolve(stream), { once: true });
        try { await video.current.play(); } catch { /* một số trình duyệt có thể chặn autoplay */ }
      } catch (e) { reject(e); }
    });

  // ==== Chuyển camera đang dùng sang camera khác ====
  const switchCamera = async (newId) => {
    setDeviceId(newId);
    runningRef.current = false; // Dừng vòng dự đoán nếu đang chạy
    setTouched(false);
    try {
      // Dừng stream cũ
      stopTracksRef.current?.();
      // Mở stream mới
      const stream = await setupCamera(newId);
      stopTracksRef.current = () => stream.getTracks().forEach(t => t.stop());
    } catch (e) {
      console.error(e);
      setError('Không chuyển được camera. Hãy thử camera khác hoặc reload.');
    }
  };

  // ==== Nhấn Train (thu thập mẫu cho 1 nhãn) ====
  const train = async (label) => {
    // Chặn train khi đang chạy loop, đang train, hoặc đang khóa train
    if (runningRef.current || isTraining || trainingLockRef.current) return;

    trainingLockRef.current = true;
    setIsTraining(true);
    try {
      const isNotTouch = label === NOT_TOUCH_LABEL;
      const key = isNotTouch ? 'not' : 'touch';

      // Điều phối bước UI
      if (isNotTouch && step !== 1) setStep(1);
      if (!isNotTouch && step < 2) setStep(2);

      // Tổng số mẫu sẽ thêm = TRAINING_TIMES × batches
      const total = TRAINING_TIMES * batches;
      for (let i = 0; i < total; i++) {
        const percent = Math.round(((i + 1) / total) * 100);
        // Cập nhật tiến độ phần trăm cho UI
        setProgress(s => ({ ...s, [key]: percent }));
        // Thực sự thêm 1 mẫu (embedding + label) vào KNN
        await training(label);
      }

      // Lấy lại số mẫu đã có từ classifier
      const counts = classifier.current.getClassExampleCount();
      const newCounts = {
        not: counts[NOT_TOUCH_LABEL] || exampleCount.not,
        touch: counts[TOUCHED_LABEL] || exampleCount.touch,
      };
      setExampleCount(newCounts);
      // Lưu dataset (có thể dạng centroid nếu bật)
      await trySaveDataset(newCounts);

      // Chuyển bước UI tiếp theo
      if (isNotTouch) setStep(2); else setStep(3);
    } finally {
      setIsTraining(false);
      trainingLockRef.current = false;
    }
  };

  // ==== Thêm 1 mẫu vào KNN: trích xuất embedding từ MobileNet ====
  const training = (label) =>
    new Promise(async (resolve) => {
      // infer(video, true) → lấy vector đặc trưng (không phải logits)
      const embedding = mobilenetModule.current.infer(video.current, true);
      // Thêm vào classifier với nhãn tương ứng
      classifier.current.addExample(embedding, label);
      // Giải phóng bộ nhớ tensor
      embedding.dispose?.();
      // Nghỉ nhẹ để tránh block UI (tạo cảm giác "đang thu")
      await sleep(50);
      resolve();
    });

  // ==== Chạy vòng dự đoán liên tục ====
  const run = async () => {
    if (!allowRun) return; // Chỉ chạy khi đã có cả hai lớp mẫu
    setStep(4);
    runningRef.current = true;
    canPlaySound.current = true; // Cho phép phát âm ở lần phát hiện tiếp theo
    frameCountRef.current = 0;
    lastFpsTimeRef.current = performance.now();

    // Hàm loop đệ quy
    const loop = async () => {
      if (!runningRef.current) return;
      // Nếu classifier trống, quay lại bước 3
      if (!classifier.current || classifier.current.getNumClasses() === 0) {
        runningRef.current = false;
        setStep(3);
        return;
      }

      const t0 = performance.now();
      try {
        // Tạo embedding từ khung hình hiện tại
        const embedding = mobilenetModule.current.infer(video.current, true);
        // Dự đoán nhãn bằng KNN
        const result = await classifier.current.predictClass(embedding);
        // Giải phóng embedding
        embedding.dispose?.();

        // Lấy xác suất (confidence) tương ứng với nhãn dự đoán
        const prob = result.confidences[result.label] ?? 0;
        // Nếu nhãn là 'touched' và prob vượt ngưỡng → kích hoạt cảnh báo
        if (result.label === TOUCHED_LABEL && prob > confidence) {
          // Chặn phát âm liên tục khi âm đang play
          if (canPlaySound.current && !muted) {
            canPlaySound.current = false;
            sound.play(); // phát âm thanh cảnh báo
          }
          // Gửi thông báo
          notify('Bỏ tay ra!', { body: 'Bạn vừa chạm tay vào mặt!' });
          setTouched(true);
        } else {
          setTouched(false);
        }
      } catch (e) {
        console.warn('Loop interrupted:', e);
        runningRef.current = false;
        setStep(3);
        return;
      }

      // ===== Tính FPS (số khung/giây) =====
      frameCountRef.current += 1;
      const now = performance.now();
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }

      // Điều chỉnh delay để gần đạt LOOP_DELAY_MS mỗi vòng
      const elapsed = performance.now() - t0;
      const delay = Math.max(0, LOOP_DELAY_MS - elapsed);
      await sleep(delay);
      loop(); // Gọi lại chính nó
    };

    loop();
  };

  // ==== Tạm dừng vòng chạy ====
  const pause = () => {
    runningRef.current = false;
    setStep(3);
    sound.stop();
    canPlaySound.current = true;
  };

  // ==== Xóa toàn bộ mẫu & reset UI ====
  const resetTraining = async () => {
    runningRef.current = false;
    setTouched(false);
    setStep(1);
    sound.stop();
    canPlaySound.current = true;
    await sleep(50);
    try { classifier.current?.clearAllClasses?.(); } catch { }
    setExampleCount({ not: 0, touch: 0 });
    setProgress({ not: 0, touch: 0 });
    await idbDel(LS_KEY); // Xóa dataset đã lưu
  };

  // ==== Lưu dataset KNN vào IndexedDB (có thể nén centroid) ====
  const trySaveDataset = async (displayCounts) => {
    if (!classifier.current) return;
    const dataset = classifier.current.getClassifierDataset();
    // payload: { label: { data, shape, centroid, n } }
    const payload = {};

    // Duyệt từng nhãn trong dataset
    for (const [label, tensor] of Object.entries(dataset)) {
      const n = tensor.shape[0]; // số mẫu trong lớp này
      if (COMPRESS_TO_CENTROID) {
        // Nén: lấy vector trung bình của tất cả mẫu (1 × dim)
        const mean = tensor.mean(0);
        const data = Array.from(await mean.data());
        payload[label] = {
          data,
          shape: mean.shape, // [dim]
          centroid: true,
          // n hiển thị UI: lấy từ displayCounts nếu có, fallback n thực tế
          n: displayCounts?.[label === NOT_TOUCH_LABEL ? 'not' : 'touch'] ?? n
        };
        mean.dispose();
      } else {
        // Không nén: lưu toàn bộ tensor (tốn dung lượng)
        const data = Array.from(await tensor.data());
        payload[label] = { data, shape: tensor.shape, centroid: false, n };
      }
    }

    // Đảm bảo có key cho cả 2 nhãn (kể cả khi rỗng)
    if (!payload[NOT_TOUCH_LABEL]) {
      payload[NOT_TOUCH_LABEL] = { data: [], shape: [0, 0], centroid: true, n: displayCounts?.not ?? 0 };
    }
    if (!payload[TOUCHED_LABEL]) {
      payload[TOUCHED_LABEL] = { data: [], shape: [0, 0], centroid: true, n: displayCounts?.touch ?? 0 };
    }

    await idbSet(LS_KEY, payload); // Ghi vào IndexedDB
  };

  // ==== Tải dataset KNN từ IndexedDB (nếu không reset auto) ====
  const tryLoadDataset = async () => {
    if (AUTO_RESET_ON_EXIT) {
      // Nếu cấu hình reset khi vào app: xóa dataset & reset UI
      await idbDel(LS_KEY).catch(() => { });
      setExampleCount({ not: 0, touch: 0 });
      setStep(1);
      return;
    }

    const saved = await idbGet(LS_KEY);
    if (!saved) return;

    const tensors = {}; // { label: tf.tensor(...) }
    let uiNot = 0, uiTouch = 0;

    for (const label of Object.keys(saved)) {
      const { data, shape, centroid, n } = saved[label];

      if (data && data.length) {
        if (centroid) {
          // Nếu dữ liệu là centroid: shape = [dim] → tạo tensor [1, dim]
          const dim = shape?.[0] ?? data.length;
          tensors[label] = tf.tensor(data, [1, dim], 'float32');
        } else {
          // Nếu dữ liệu là full: dùng shape gốc
          tensors[label] = tf.tensor(data, shape, 'float32');
        }
      }

      if (label === NOT_TOUCH_LABEL) uiNot = n || 0;
      if (label === TOUCHED_LABEL) uiTouch = n || 0;
    }

    // Nạp vào classifier nếu có dữ liệu
    if (Object.keys(tensors).length > 0) {
      classifier.current.setClassifierDataset(tensors);
    }

    setExampleCount({ not: uiNot, touch: uiTouch });
    if (uiNot > 0 && uiTouch > 0) setStep(3); // Nếu đủ 2 lớp, chuyển sang bước sẵn sàng
  };

  // ==== Tiện ích sleep (await được) ====
  const sleep = (ms = 0) => new Promise(res => setTimeout(res, ms));

  // ==== Mount/Unmount effect ====
  useEffect(() => {
    // Khởi động toàn bộ
    init();
    // Khi âm thanh phát xong: cho phép phát lại lần sau
    sound.on('end', () => { canPlaySound.current = true; });

    // Cleanup khi unmount
    return () => {
      stopTracksRef.current?.(); // Dừng stream camera
      runningRef.current = false;
      sound.stop();
      canPlaySound.current = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cờ UI: step === 4 tức là đang chạy vòng dự đoán
  const isRunning = step === 4;

  // ==== UI ====
  return (
    <div className={`main ${touched ? 'touched' : ''}`}>
      <div className="container">
        {/* Top bar */}
        <header className="topbar">
          <div className="brand">
            <span className="brand-text">Touch Guard <b>Vision</b></span>
          </div>

          <div className="top-actions">
            <div className="chip small">Backend: <b>{backend || '...'}</b></div>
            <div className="chip small">FPS: <b>{fps}</b></div>
          </div>
        </header>

        {/* Hiển thị lỗi nếu có */}
        {error && <div className="alert"><b>Lỗi:</b> {error}</div>}

        {/* Bố cục 2 cột: video bên trái, panel điều khiển bên phải */}
        <div className="layout">
          {/* Camera / HUD */}
          <section className="panel video-panel">
            <div className="video-wrap">
              {/* 4 góc HUD trang trí */}
              <div className="hud-corners" aria-hidden />
              {/* Video webcam */}
              <video ref={video} className="video" autoPlay muted playsInline />
              {/* Trạng thái hiện tại: CLEAR / INITIALIZING... / TOUCH DETECTED */}
              <div className={`status ${touched ? 'danger' : 'safe'}`}>
                <span className="dot" />
                {touched ? 'TOUCH DETECTED' : (loading ? 'INITIALIZING...' : 'CLEAR')}
              </div>

              {/* Nút điều khiển nhanh bên dưới video */}
              <div className="video-actions">
                {runningRef.current ? (
                  <>
                    <button className="btn btn-secondary" onClick={pause}>Stop</button>
                    <button className="btn btn-ghost" onClick={resetTraining} disabled={isTraining}>Reset</button>
                  </>
                ) : (
                  <>
                    <button className="btn btn-primary" onClick={run} disabled={loading || !allowRun || isTraining}>Run</button>
                    <button className="btn btn-ghost" onClick={resetTraining} disabled={isTraining}>Reset</button>
                  </>
                )}
              </div>
            </div>

            {/* Thống kê nhanh: số mẫu & ngưỡng */}
            <div className="stats">
              <div className="stat">
                <div className="stat-label">Không chạm</div>
                <div className="stat-value">{exampleCount.not}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Chạm</div>
                <div className="stat-value">{exampleCount.touch}</div>
              </div>
              <div className="stat stretch">
                <div className="stat-label">Ngưỡng</div>
                <div className="stat-control">
                  <input
                    type="range"
                    min="0.5"
                    max="0.99"
                    step="0.01"
                    value={confidence}
                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                    className="slider"
                    disabled={isTraining}
                  />

                  <span className="stat-badge">{confidence.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </section>

          {/* Panel huấn luyện & cài đặt */}
          <aside className={`panel side-panel ${isTraining ? 'busy' : ''}`}>
            <div className="drawer-title">Huấn luyện</div>

            {/* Bước 1: Train KHÔNG CHẠM */}
            <div className={`step-card ${step >= 1 ? 'active' : ''} ${step > 1 ? 'done' : ''}`}>
              <div className="step-head">
                <span className="step-index">01</span>
                <span className="step-name">KHÔNG CHẠM</span>
              </div>
              <div className="train-row">
                <button
                  className="btn"
                  onClick={() => train(NOT_TOUCH_LABEL)}
                  disabled={loading || step > 1 || isRunning || isTraining}
                >
                  {isTraining && step === 1 ? "Training..." : "Train 1"}
                </button>

              </div>
              <div className="progress"><div className="progress-bar" style={{ width: `${progress.not}%` }} /></div>
            </div>

            {/* Bước 2: Train CÓ CHẠM */}
            <div className={`step-card ${step >= 2 ? 'active' : ''} ${step > 2 ? 'done' : ''}`}>
              <div className="step-head">
                <span className="step-index">02</span>
                <span className="step-name">CÓ CHẠM</span>
              </div>
              <div className="train-row">
                <button
                  className="btn"
                  onClick={() => train(TOUCHED_LABEL)}
                  disabled={loading || step < 2 || step > 2 || isRunning || isTraining}
                >
                  {isTraining && step === 2 ? "Training..." : "Train 2"}
                </button>

              </div>
              <div className="progress"><div className="progress-bar" style={{ width: `${progress.touch}%` }} /></div>
            </div>

            {/* Số batches × TRAINING_TIMES */}
            <div className="batch">
              <span className="batch-label">Batches</span>
              <input
                type="number"
                min="1"
                max="10"
                value={batches}
                onChange={(e) => setBatches(Math.max(1, Math.min(20, parseInt(e.target.value || '1', 10))))}
                className="batch-input"
                disabled={isTraining || isRunning}
              />

              <span className="x">×{TRAINING_TIMES}</span>
            </div>

            {/* Cài đặt: mute & chọn camera */}
            <div className="drawer-title mt">Cài đặt</div>
            <div className="settings">
              <label className="switch">
                <input type="checkbox" checked={muted} onChange={(e) => setMuted(e.target.checked)} />
                <span>Tắt âm</span>
              </label>

              <div className="field">
                <label>Camera</label>
                <select
                  className="select"
                  value={deviceId}
                  onChange={(e) => switchCamera(e.target.value)}
                  disabled={devices.length <= 1 || loading || isTraining || isRunning}
                >
                  {/* Hiển thị danh sách camera: dùng label nếu có, fallback một phần deviceId */}
                  {devices.map(d => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label || `Camera ${d.deviceId.slice(0, 4)}`}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </aside>
        </div>

        {/* Footer thông tin dự án */}
        <footer className="footer">
          <div className="left">
            <span className="foot-brand">TouchGuard Vision</span>
            <span className="foot-dot">•</span>
            <span>TensorFlow.js · MobileNet + KNN</span>
          </div>

          <div className="right">
            <span>©2025, TruongPersonal</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
