# Adaptive Attention Detection System
## プログラマー向け統合ガイド

---

## 概要

VR環境でユーザーの注意状態（Glancing/Overview/DeepEngagement）をリアルタイム推定し、DeepEngagement状態でキャプションを表示するトリガーを発火する。

---

## 必要な入力データ

Eye Tracker / Head Trackerから以下の4つの特徴量を毎フレーム取得：

| 特徴量 | 単位 | 取得元 | 説明 |
|--------|------|--------|------|
| `headRotVelocity` | rad/s | Head Tracker | 頭部回転の角速度 |
| `gazeDispersion` | m | Eye Tracker | 直近0.5秒の視線位置の標準偏差 |
| `spatialEntropy` | 0-1 | 計算 | 視線位置分布のエントロピー（正規化） |
| `pupilDelta` | mm | Eye Tracker | 瞳孔径とベースラインの差 |

---

## 基本的な使い方

### 1. 初期化

```csharp
// シーン開始時に自動初期化される（Awakeで実行）
// 手動で呼ぶ場合:
AdaptiveHMMAttentionDetector.Instance.Initialize();
```

### 2. 毎フレームの推論

```csharp
void Update()
{
    // Eye Trackerから特徴量を計算
    float headRotVelocity = CalculateHeadRotVelocity();
    float gazeDispersion = CalculateGazeDispersion();
    float spatialEntropy = CalculateSpatialEntropy();
    float pupilDelta = currentPupil - baselinePupil;
    
    // HMM推論
    var (shouldTrigger, confidence) = AdaptiveHMMAttentionDetector.Instance.Update(
        headRotVelocity,
        gazeDispersion,
        spatialEntropy,
        pupilDelta,
        currentLookingObjectId  // オプション: 見逃し検出用
    );
    
    if (shouldTrigger)
    {
        ShowCaption();
    }
}
```

### 3. キャリブレーション

```csharp
// キャリブレーションフェーズで呼び出し
// Glancing（軽く見る）サンプル収集
detector.AddCalibrationGlancing(new float[] { headRot, disp, entropy, pupil });

// DeepEngagement（じっくり見る）サンプル収集
detector.AddCalibrationEngaged(new float[] { headRot, disp, entropy, pupil });

// 完了（30サンプル以上必要）
bool success = detector.CompleteCalibration();
```

### 4. 暗黙的フィードバック

```csharp
// キャプション表示後にユーザーが読んだ
detector.ReportFeedback(FeedbackType.TruePositive);

// キャプション表示後すぐに視線が逸れた
detector.ReportFeedback(FeedbackType.FalsePositive);

// 長時間注視したがトリガーなし（自動検出も可能）
detector.ReportFeedback(FeedbackType.FalseNegative);
```

---

## 主要API

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `CurrentState` | `AttentionState` | 現在の状態 (Glancing/Overview/DeepEngagement) |
| `DeepEngagementProb` | `float` | DeepEngagement状態の確率 (0-1) |
| `IsCalibrated` | `bool` | キャリブレーション完了フラグ |
| `CurrentAdaptationMode` | `AdaptationMode` | 適応モード (Calibrating/Exploring/Confident) |

### イベント

| イベント | 引数 | タイミング |
|----------|------|----------|
| `OnStateChanged` | `AttentionState` | 状態が変化した時 |
| `OnTrigger` | なし | トリガー条件を満たした時 |
| `OnFeedbackReceived` | `FeedbackType` | フィードバックが記録された時 |
| `OnCalibrationComplete` | なし | キャリブレーション完了時 |

---

## 特徴量の計算方法

### 頭部回転速度

```csharp
private Quaternion lastRotation;

float CalculateHeadRotVelocity()
{
    Quaternion current = Camera.main.transform.rotation;
    float angle = Quaternion.Angle(lastRotation, current);
    float velocity = angle / Time.deltaTime * Mathf.Deg2Rad;
    lastRotation = current;
    return velocity;
}
```

### 視線分散

```csharp
private Queue<Vector3> gazeBuffer = new Queue<Vector3>();
private const int WINDOW_SIZE = 30;  // 0.5秒@60fps

float CalculateGazeDispersion()
{
    gazeBuffer.Enqueue(currentGazeWorldPoint);
    if (gazeBuffer.Count > WINDOW_SIZE) gazeBuffer.Dequeue();
    
    Vector3 mean = gazeBuffer.Average();
    float variance = gazeBuffer.Sum(p => (p - mean).sqrMagnitude) / gazeBuffer.Count;
    return Mathf.Sqrt(variance);
}
```

### 空間エントロピー

```csharp
float CalculateSpatialEntropy()
{
    // 視線位置を5x5グリッドに投影
    int[] histogram = new int[25];
    foreach (var pos in gazeBuffer)
    {
        int gx = Mathf.Clamp((int)((pos.x + 1) * 2.5f), 0, 4);
        int gy = Mathf.Clamp((int)((pos.y + 1) * 2.5f), 0, 4);
        histogram[gy * 5 + gx]++;
    }
    
    // エントロピー計算
    float entropy = 0f;
    int total = gazeBuffer.Count;
    foreach (int count in histogram)
    {
        if (count > 0)
        {
            float p = (float)count / total;
            entropy -= p * Mathf.Log(p, 2);
        }
    }
    
    return entropy / 4.64f;  // 正規化（max = log2(25)）
}
```

---

## 数理モデル

### HMM構造

- **状態**: $S = \{Glancing, Overview, DeepEngagement\}$
- **観測**: $\mathbf{o}_t = [\omega, \sigma, H, \Delta P]^T$

### トリガー条件

$$
\text{Trigger} \Leftrightarrow P(DeepEngagement \mid \mathbf{o}_{1:t}) > 0.7 \quad \text{かつ} \quad 直近30フレーム中80\%以上がDeepEngagement
$$

### オンライン適応

$$
\mu_k^{new} = (1 - \alpha_t) \mu_k^{old} + \alpha_t \bar{o}, \quad \alpha_t = \frac{0.3}{\sqrt{t+1}}
$$

---

## Inspector設定

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| `triggerThreshold` | トリガー確率閾値 | 0.7 |
| `sustainedFrames` | 持続判定フレーム数 | 30 |
| `initialAdaptationRate` | 初期学習率 | 0.3 |
| `minAdaptationRate` | 最小学習率 | 0.02 |
| `captionReadThreshold` | 「読んだ」判定時間 | 0.5秒 |
| `quickLookAwayThreshold` | 「無視した」判定時間 | 0.2秒 |

---

## よくある問題

| 症状 | 対処 |
|------|------|
| トリガーされない | `triggerThreshold` を 0.5 に下げる |
| 誤トリガーが多い | キャリブレーションの `engagedDuration` を延長 |
| 適応が不安定 | `maxDeltaPerUpdate` を 0.2 に下げる |
