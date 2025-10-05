import React, { useEffect, useRef, useState } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { Howl } from 'howler';
import { initNotifications, notify } from '@mycv/f8-notification';
import { set as idbSet, get as idbGet, del as idbDel } from 'idb-keyval';
import soundURL from './assets/sound/Hey-Take_your_hand_off.mp3';

const sound = new Howl({ src: [soundURL] });

const NOT_TOUCH_LABEL = 'not_touch';
const TOUCHED_LABEL = 'touched';
const TRAINING_TIMES = 50;
const LOOP_DELAY_MS = 160;
const LS_KEY = 'touch-guard-dataset-v1';
const MUTE_KEY = 'tg-muted';
const COMPRESS_TO_CENTROID = true;
const AUTO_RESET_ON_EXIT = true;

function App() {
  // ==== Refs / States ====
  const video = useRef(null);
  const classifier = useRef(null);
  const mobilenetModule = useRef(null);
  const canPlaySound = useRef(true);
  const didInit = useRef(false);
  const stopTracksRef = useRef(() => { });
  const runningRef = useRef(false);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(performance.now());
  const trainingLockRef = useRef(false);


  const [touched, setTouched] = useState(false);
  const [step, setStep] = useState(1);
  const [progress, setProgress] = useState({ not: 0, touch: 0 });
  const [exampleCount, setExampleCount] = useState({ not: 0, touch: 0 });
  const [confidence, setConfidence] = useState(0.8);
  const [batches, setBatches] = useState(1);
  const [isTraining, setIsTraining] = useState(false);

  const [muted, setMuted] = useState(() => {
    try { return localStorage.getItem(MUTE_KEY) === 'true'; } catch { return false; }
  });

  const [loading, setLoading] = useState(true);
  const [backend, setBackend] = useState('');
  const [error, setError] = useState('');
  const [fps, setFps] = useState(0);

  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState('');

  const allowRun = exampleCount.not > 0 && exampleCount.touch > 0;

  // ==== Persist mute ====
  useEffect(() => {
    try { localStorage.setItem(MUTE_KEY, String(muted)); } catch { }
    try { sound.mute(muted); } catch { }
  }, [muted]);

  // ==== Init ====
  const init = async () => {
    if (didInit.current) return;
    didInit.current = true;

    setError('');
    setLoading(true);

    try {
      try { await tf.setBackend('webgl'); } catch { await tf.setBackend('cpu'); }
      await tf.ready();
      setBackend(tf.getBackend());

      // Ask cam permission first so enumerateDevices returns labels
      await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const mediaDevices = await navigator.mediaDevices.enumerateDevices();
      const cams = mediaDevices.filter(d => d.kind === 'videoinput');
      setDevices(cams);
      setDeviceId(cams[0]?.deviceId || '');

      const stream = await setupCamera(cams[0]?.deviceId || undefined);
      stopTracksRef.current = () => stream.getTracks().forEach(t => t.stop());

      classifier.current = knnClassifier.create();
      mobilenetModule.current = await mobilenet.load();

      await tryLoadDataset();

      initNotifications({ cooldown: 3000 });
      setLoading(false);
    } catch (e) {
      console.error(e);
      setError('Không thể khởi tạo camera hoặc model. Kiểm tra quyền camera / reload trang.');
      setLoading(false);
    }
  };

  const setupCamera = (devId) =>
    new Promise(async (resolve, reject) => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: devId ? { deviceId: { exact: devId } } : true,
          audio: false
        });
        if (!video.current) return reject(new Error('Video element missing'));
        video.current.srcObject = stream;
        video.current.addEventListener('loadeddata', () => resolve(stream), { once: true });
        try { await video.current.play(); } catch { }
      } catch (e) { reject(e); }
    });

  const switchCamera = async (newId) => {
    setDeviceId(newId);
    runningRef.current = false;
    setTouched(false);
    try {
      stopTracksRef.current?.();
      const stream = await setupCamera(newId);
      stopTracksRef.current = () => stream.getTracks().forEach(t => t.stop());
    } catch (e) {
      console.error(e);
      setError('Không chuyển được camera. Hãy thử camera khác hoặc reload.');
    }
  };

  // ==== Train ====
  const train = async (label) => {
    // chặn khi đang chạy hoặc đang train
    if (runningRef.current || isTraining || trainingLockRef.current) return;

    trainingLockRef.current = true;
    setIsTraining(true);
    try {
      const isNotTouch = label === NOT_TOUCH_LABEL;
      const key = isNotTouch ? 'not' : 'touch';

      if (isNotTouch && step !== 1) setStep(1);
      if (!isNotTouch && step < 2) setStep(2);

      const total = TRAINING_TIMES * batches;
      for (let i = 0; i < total; i++) {
        const percent = Math.round(((i + 1) / total) * 100);
        setProgress(s => ({ ...s, [key]: percent }));
        await training(label);
      }

      const counts = classifier.current.getClassExampleCount();
      const newCounts = {
        not: counts[NOT_TOUCH_LABEL] || exampleCount.not,
        touch: counts[TOUCHED_LABEL] || exampleCount.touch,
      };
      setExampleCount(newCounts);
      await trySaveDataset(newCounts);

      if (isNotTouch) setStep(2); else setStep(3);
    } finally {
      setIsTraining(false);
      trainingLockRef.current = false;
    }
  };


  const training = (label) =>
    new Promise(async (resolve) => {
      const embedding = mobilenetModule.current.infer(video.current, true);
      classifier.current.addExample(embedding, label);
      embedding.dispose?.();
      await sleep(50);
      resolve();
    });

  // ==== Run Loop ====
  const run = async () => {
    if (!allowRun) return;
    setStep(4);
    runningRef.current = true;
    canPlaySound.current = true;
    frameCountRef.current = 0;
    lastFpsTimeRef.current = performance.now();

    const loop = async () => {
      if (!runningRef.current) return;
      if (!classifier.current || classifier.current.getNumClasses() === 0) {
        runningRef.current = false;
        setStep(3);
        return;
      }

      const t0 = performance.now();
      try {
        const embedding = mobilenetModule.current.infer(video.current, true);
        const result = await classifier.current.predictClass(embedding);
        embedding.dispose?.();

        const prob = result.confidences[result.label] ?? 0;
        if (result.label === TOUCHED_LABEL && prob > confidence) {
          if (canPlaySound.current && !muted) {
            canPlaySound.current = false;
            sound.play();
          }
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

      // FPS
      frameCountRef.current += 1;
      const now = performance.now();
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }

      const elapsed = performance.now() - t0;
      const delay = Math.max(0, LOOP_DELAY_MS - elapsed);
      await sleep(delay);
      loop();
    };
    loop();
  };

  const pause = () => {
    runningRef.current = false;
    setStep(3);
    sound.stop();
    canPlaySound.current = true;
  };

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
    await idbDel(LS_KEY);
  };

  // ==== Save / Load Dataset (IndexedDB) ====
  const trySaveDataset = async (displayCounts) => {
    if (!classifier.current) return;
    const dataset = classifier.current.getClassifierDataset();
    const payload = {}; // { label: { data, shape, centroid, n } }

    for (const [label, tensor] of Object.entries(dataset)) {
      const n = tensor.shape[0];
      if (COMPRESS_TO_CENTROID) {
        const mean = tensor.mean(0);
        const data = Array.from(await mean.data());
        payload[label] = {
          data,
          shape: mean.shape,
          centroid: true,
          n: displayCounts?.[label === NOT_TOUCH_LABEL ? 'not' : 'touch'] ?? n
        };
        mean.dispose();
      } else {
        const data = Array.from(await tensor.data());
        payload[label] = { data, shape: tensor.shape, centroid: false, n };
      }
    }

    if (!payload[NOT_TOUCH_LABEL]) {
      payload[NOT_TOUCH_LABEL] = { data: [], shape: [0, 0], centroid: true, n: displayCounts?.not ?? 0 };
    }
    if (!payload[TOUCHED_LABEL]) {
      payload[TOUCHED_LABEL] = { data: [], shape: [0, 0], centroid: true, n: displayCounts?.touch ?? 0 };
    }

    await idbSet(LS_KEY, payload);
  };

  const tryLoadDataset = async () => {
    if (AUTO_RESET_ON_EXIT) {
      await idbDel(LS_KEY).catch(() => { });
      setExampleCount({ not: 0, touch: 0 });
      setStep(1);
      return;
    }

    const saved = await idbGet(LS_KEY);
    if (!saved) return;

    const tensors = {};
    let uiNot = 0, uiTouch = 0;

    for (const label of Object.keys(saved)) {
      const { data, shape, centroid, n } = saved[label];

      if (data && data.length) {
        if (centroid) {
          const dim = shape?.[0] ?? data.length;
          tensors[label] = tf.tensor(data, [1, dim], 'float32');
        } else {
          tensors[label] = tf.tensor(data, shape, 'float32');
        }
      }

      if (label === NOT_TOUCH_LABEL) uiNot = n || 0;
      if (label === TOUCHED_LABEL) uiTouch = n || 0;
    }

    if (Object.keys(tensors).length > 0) {
      classifier.current.setClassifierDataset(tensors);
    }

    setExampleCount({ not: uiNot, touch: uiTouch });
    if (uiNot > 0 && uiTouch > 0) setStep(3);
  };

  const sleep = (ms = 0) => new Promise(res => setTimeout(res, ms));

  useEffect(() => {
    init();
    sound.on('end', () => { canPlaySound.current = true; });

    return () => {
      stopTracksRef.current?.();
      runningRef.current = false;
      sound.stop();
      canPlaySound.current = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isRunning = step === 4;

  // ==== UI ====
  return (
    <div className={`main ${touched ? 'touched' : ''}`}>
      <div className="container">
        {/* Top */}
        <header className="topbar">
          <div className="brand">
            <span className="brand-text">TouchGuard <b>Vision</b></span>
          </div>

          <div className="top-actions">
            <div className="chip small">Backend: <b>{backend || '...'}</b></div>
            <div className="chip small">FPS: <b>{fps}</b></div>
          </div>
        </header>

        {error && <div className="alert"><b>Lỗi:</b> {error}</div>}

        {/* Layout */}
        <div className="layout">
          {/* Camera / HUD */}
          <section className="panel video-panel">
            <div className="video-wrap">
              {/* HUD corners */}
              <div className="hud-corners" aria-hidden />
              <video ref={video} className="video" autoPlay muted playsInline />
              <div className={`status ${touched ? 'danger' : 'safe'}`}>
                <span className="dot" />
                {touched ? 'TOUCH DETECTED' : (loading ? 'INITIALIZING...' : 'CLEAR')}
              </div>

              {/* Inline actions under video */}
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

            {/* Stats compact */}
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

          {/* Training Steps */}
          <aside className={`panel side-panel ${isTraining ? 'busy' : ''}`}>
            <div className="drawer-title">Huấn luyện</div>

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

        <footer className="footer">
          <div className="left">
            <span className="foot-brand">TouchGuard Vision</span>
            <span className="foot-dot">•</span>
            <span>TensorFlow.js · MobileNet + KNN</span>
          </div>

          <div className="right">
            <span>©2025 @TruongPersonal</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
