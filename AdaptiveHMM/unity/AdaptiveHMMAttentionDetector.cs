/*
 * AdaptiveHMMAttentionDetector.cs
 * ===============================
 * キャリブレーション + オンライン適応機能付きHMM
 * 
 * 機能:
 *   1. 初期キャリブレーション（Glancing/DeepEngagement分布の学習）
 *   2. オンライン適応（暗黙的フィードバックからの学習）
 *   3. リアルタイム注意状態推定
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace FloatingCaption.Attention
{
    /// <summary>
    /// 注意状態の定義
    /// </summary>
    public enum AttentionState
    {
        Glancing = 0,        // 一瞥・通過
        Overview = 1,        // 概観把握
        DeepEngagement = 2   // 深い興味 → トリガー対象
    }

    /// <summary>
    /// 適応モード
    /// </summary>
    public enum AdaptationMode
    {
        Disabled,       // 適応なし
        Calibrating,    // 初期キャリブレーション中
        Exploring,      // 探索中（中速適応）
        Confident       // 十分なサンプル後（低速適応）
    }

    /// <summary>
    /// フィードバックの種類
    /// </summary>
    public enum FeedbackType
    {
        TruePositive,   // 正しくトリガー（キャプションを読んだ）
        FalsePositive,  // 誤トリガー（すぐに視線を逸らした）
        FalseNegative   // 見逃し（長時間注視だがトリガーなし）
    }

    /// <summary>
    /// Adaptive HMM 注意検出器
    /// </summary>
    public class AdaptiveHMMAttentionDetector : MonoBehaviour
    {
        public static AdaptiveHMMAttentionDetector Instance { get; private set; }

        #region Inspector Settings

        [Header("HMM Core Settings")]
        [SerializeField] private string parametersPath = "hmm_params";
        [SerializeField] private float triggerThreshold = 0.7f;
        [SerializeField] private int sustainedFrames = 30;

        [Header("Calibration Settings")]
        [SerializeField] private bool enableCalibration = true;
        [SerializeField] private int minCalibrationSamples = 30;

        [Header("Adaptation Settings")]
        [SerializeField] private bool enableAdaptation = true;
        [SerializeField] private float initialAdaptationRate = 0.3f;
        [SerializeField] private float minAdaptationRate = 0.02f;
        [SerializeField] private float maxDeltaPerUpdate = 0.5f;
        [SerializeField] private float minStd = 0.05f;

        [Header("Feedback Detection")]
        [SerializeField] private float captionReadThreshold = 0.5f;    // キャプション注視判定時間
        [SerializeField] private float quickLookAwayThreshold = 0.2f;  // 誤トリガー判定時間
        [SerializeField] private float longGazeThreshold = 5.0f;       // 見逃し判定時間

        [Header("Debug")]
        [SerializeField] private bool debugMode = false;

        #endregion

        #region HMM Parameters

        private const int N_STATES = 3;
        private const int N_FEATURES = 4;

        // 初期確率
        private float[] pi = { 0.7f, 0.2f, 0.1f };

        // 遷移行列
        private float[,] A = {
            { 0.7f, 0.25f, 0.05f },
            { 0.2f, 0.5f, 0.3f },
            { 0.1f, 0.3f, 0.6f }
        };

        // 観測モデル（適応対象）
        private float[,] means = {
            { 0.5f, 0.8f, 0.3f, 0.0f },    // Glancing
            { 0.2f, 0.5f, 0.4f, 0.1f },    // Overview
            { 0.05f, 0.2f, 0.6f, 0.23f }   // DeepEngagement
        };

        private float[,] stds = {
            { 0.2f, 0.3f, 0.1f, 0.1f },
            { 0.1f, 0.2f, 0.1f, 0.1f },
            { 0.05f, 0.1f, 0.1f, 0.1f }
        };

        // 正規化パラメータ
        private float[] featureMeans = { 0.2f, 0.4f, 0.4f, 0.1f };
        private float[] featureStds = { 0.2f, 0.3f, 0.2f, 0.15f };

        #endregion

        #region State Tracking

        private float[] alpha;
        private Queue<AttentionState> stateHistory;
        private Queue<float[]> recentFeatures;

        // 適応状態
        private AdaptationMode adaptationMode = AdaptationMode.Disabled;
        private int positiveExamples = 0;
        private int negativeExamples = 0;
        private int missedExamples = 0;

        // キャリブレーションデータ
        private List<float[]> calibrationGlancing = new List<float[]>();
        private List<float[]> calibrationEngaged = new List<float[]>();

        // フィードバック検出用
        private float lastTriggerTime = -1f;
        private float currentObjectGazeTime = 0f;
        private int currentObjectId = -1;

        #endregion

        #region Properties

        public AttentionState CurrentState { get; private set; }
        public float[] StateProbs => alpha;
        public float DeepEngagementProb => alpha?[(int)AttentionState.DeepEngagement] ?? 0f;
        public bool IsReady { get; private set; }
        public bool IsCalibrated { get; private set; }
        public AdaptationMode CurrentAdaptationMode => adaptationMode;
        public int TotalFeedbackCount => positiveExamples + negativeExamples + missedExamples;

        #endregion

        #region Events

        public event Action<AttentionState> OnStateChanged;
        public event Action OnTrigger;
        public event Action<FeedbackType> OnFeedbackReceived;
        public event Action OnCalibrationComplete;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else
            {
                Destroy(gameObject);
                return;
            }

            Initialize();
        }

        #endregion

        #region Initialization

        public void Initialize()
        {
            alpha = new float[N_STATES];
            Array.Copy(pi, alpha, N_STATES);

            stateHistory = new Queue<AttentionState>();
            recentFeatures = new Queue<float[]>();

            if (enableAdaptation)
            {
                adaptationMode = enableCalibration ? 
                    AdaptationMode.Calibrating : AdaptationMode.Exploring;
            }

            IsReady = true;
            Debug.Log($"AdaptiveHMM initialized. Mode: {adaptationMode}");
        }

        public void ResetForward()
        {
            Array.Copy(pi, alpha, N_STATES);
            stateHistory.Clear();
        }

        #endregion

        #region Calibration

        /// <summary>
        /// キャリブレーション用サンプルを追加（Glancing状態）
        /// </summary>
        public void AddCalibrationGlancing(float[] features)
        {
            if (adaptationMode != AdaptationMode.Calibrating) return;
            calibrationGlancing.Add((float[])features.Clone());
            
            if (debugMode)
                Debug.Log($"Glancing sample added. Total: {calibrationGlancing.Count}");
        }

        /// <summary>
        /// キャリブレーション用サンプルを追加（DeepEngagement状態）
        /// </summary>
        public void AddCalibrationEngaged(float[] features)
        {
            if (adaptationMode != AdaptationMode.Calibrating) return;
            calibrationEngaged.Add((float[])features.Clone());
            
            if (debugMode)
                Debug.Log($"Engaged sample added. Total: {calibrationEngaged.Count}");
        }

        /// <summary>
        /// キャリブレーションを完了し、パラメータを更新
        /// </summary>
        public bool CompleteCalibration()
        {
            if (calibrationGlancing.Count < minCalibrationSamples ||
                calibrationEngaged.Count < minCalibrationSamples)
            {
                Debug.LogWarning($"Insufficient calibration data. " +
                    $"Glancing: {calibrationGlancing.Count}, Engaged: {calibrationEngaged.Count}");
                return false;
            }

            // Glancing状態の分布更新
            UpdateStateDistribution(0, calibrationGlancing);

            // DeepEngagement状態の分布更新
            UpdateStateDistribution(2, calibrationEngaged);

            // Overview状態は両者の中間として補間
            InterpolateOverviewState();

            // モード遷移
            adaptationMode = AdaptationMode.Exploring;
            IsCalibrated = true;
            
            ResetForward();
            calibrationGlancing.Clear();
            calibrationEngaged.Clear();

            Debug.Log("Calibration complete. Transitioning to Exploring mode.");
            OnCalibrationComplete?.Invoke();

            return true;
        }

        private void UpdateStateDistribution(int stateIdx, List<float[]> samples)
        {
            int n = samples.Count;
            float[] sumMean = new float[N_FEATURES];
            float[] sumVar = new float[N_FEATURES];

            // 平均計算
            foreach (var sample in samples)
            {
                var normalized = NormalizeFeatures(sample);
                for (int f = 0; f < N_FEATURES; f++)
                    sumMean[f] += normalized[f];
            }

            for (int f = 0; f < N_FEATURES; f++)
                means[stateIdx, f] = sumMean[f] / n;

            // 分散計算
            foreach (var sample in samples)
            {
                var normalized = NormalizeFeatures(sample);
                for (int f = 0; f < N_FEATURES; f++)
                {
                    float diff = normalized[f] - means[stateIdx, f];
                    sumVar[f] += diff * diff;
                }
            }

            for (int f = 0; f < N_FEATURES; f++)
                stds[stateIdx, f] = Mathf.Max(Mathf.Sqrt(sumVar[f] / n), minStd);
        }

        private void InterpolateOverviewState()
        {
            for (int f = 0; f < N_FEATURES; f++)
            {
                means[1, f] = (means[0, f] + means[2, f]) / 2f;
                stds[1, f] = (stds[0, f] + stds[2, f]) / 2f;
            }
        }

        #endregion

        #region Real-time Inference

        /// <summary>
        /// 1フレーム分の推論を実行
        /// </summary>
        public (bool shouldTrigger, float confidence) Update(
            float headRotVelocity,
            float gazeDispersion,
            float spatialEntropy,
            float pupilDelta,
            int currentObjectId = -1)
        {
            if (!IsReady) return (false, 0f);

            float[] features = { headRotVelocity, gazeDispersion, spatialEntropy, pupilDelta };

            // 特徴量履歴に追加
            recentFeatures.Enqueue((float[])features.Clone());
            if (recentFeatures.Count > 120) // 2秒分
                recentFeatures.Dequeue();

            // 正規化 & Forward step
            float[] normalized = NormalizeFeatures(features);
            ForwardStep(normalized);

            // 状態更新
            AttentionState newState = GetMostLikelyState();
            if (newState != CurrentState)
            {
                CurrentState = newState;
                OnStateChanged?.Invoke(newState);
            }

            // 履歴更新
            stateHistory.Enqueue(CurrentState);
            if (stateHistory.Count > sustainedFrames)
                stateHistory.Dequeue();

            // オブジェクト追跡（見逃し検出用）
            UpdateObjectTracking(currentObjectId);

            // トリガー判定
            bool shouldTrigger = CheckTriggerCondition();
            float confidence = alpha[(int)AttentionState.DeepEngagement];

            if (shouldTrigger)
            {
                lastTriggerTime = Time.time;
                OnTrigger?.Invoke();
            }

            if (debugMode && Time.frameCount % 60 == 0)
            {
                Debug.Log($"State: {CurrentState}, DE: {confidence:F3}, Mode: {adaptationMode}");
            }

            return (shouldTrigger, confidence);
        }

        private float[] NormalizeFeatures(float[] features)
        {
            float[] normalized = new float[N_FEATURES];
            for (int i = 0; i < N_FEATURES; i++)
            {
                normalized[i] = (features[i] - featureMeans[i]) / (featureStds[i] + 1e-8f);
                if (float.IsNaN(normalized[i]) || float.IsInfinity(normalized[i]))
                    normalized[i] = 0f;
            }
            return normalized;
        }

        private void ForwardStep(float[] features)
        {
            float[] alphaNew = new float[N_STATES];
            float[] emission = new float[N_STATES];

            // 観測確率
            for (int k = 0; k < N_STATES; k++)
                emission[k] = GaussianPDF(features, k) + 1e-10f;

            // Forward更新
            for (int j = 0; j < N_STATES; j++)
            {
                float sum = 0f;
                for (int i = 0; i < N_STATES; i++)
                    sum += alpha[i] * A[i, j];
                alphaNew[j] = emission[j] * sum;
            }

            // 正規化
            float total = alphaNew.Sum();
            if (total > 1e-10f)
            {
                for (int k = 0; k < N_STATES; k++)
                    alpha[k] = alphaNew[k] / total;
            }
        }

        private float GaussianPDF(float[] x, int stateIdx)
        {
            float logProb = 0f;
            for (int i = 0; i < N_FEATURES; i++)
            {
                float diff = x[i] - means[stateIdx, i];
                float std = stds[stateIdx, i] + 1e-8f;
                logProb -= 0.5f * (diff * diff) / (std * std);
                logProb -= Mathf.Log(std);
            }
            logProb -= 0.5f * N_FEATURES * Mathf.Log(2f * Mathf.PI);
            return Mathf.Exp(Mathf.Clamp(logProb, -500f, 0f));
        }

        private AttentionState GetMostLikelyState()
        {
            int maxIdx = 0;
            float maxProb = alpha[0];
            for (int k = 1; k < N_STATES; k++)
            {
                if (alpha[k] > maxProb)
                {
                    maxProb = alpha[k];
                    maxIdx = k;
                }
            }
            return (AttentionState)maxIdx;
        }

        private bool CheckTriggerCondition()
        {
            if (stateHistory.Count < sustainedFrames) return false;

            int deCount = stateHistory.Count(s => s == AttentionState.DeepEngagement);
            float ratio = (float)deCount / sustainedFrames;

            return ratio > 0.8f && alpha[(int)AttentionState.DeepEngagement] > triggerThreshold;
        }

        #endregion

        #region Online Adaptation

        /// <summary>
        /// 暗黙的フィードバックを報告
        /// </summary>
        public void ReportFeedback(FeedbackType feedback)
        {
            if (!enableAdaptation || adaptationMode == AdaptationMode.Calibrating)
                return;

            float rate = GetCurrentAdaptationRate();
            float[][] recentFeaturesArray = recentFeatures.ToArray();

            switch (feedback)
            {
                case FeedbackType.TruePositive:
                    // 正しいトリガー → DeepEngagement分布を強化
                    AdaptStateFromSamples(2, recentFeaturesArray, rate);
                    positiveExamples++;
                    break;

                case FeedbackType.FalsePositive:
                    // 誤トリガー → この特徴量をGlancing側に寄せる
                    AdaptStateFromSamples(0, recentFeaturesArray, rate);
                    negativeExamples++;
                    break;

                case FeedbackType.FalseNegative:
                    // 見逃し → DeepEngagement分布を拡張
                    ExpandStateDistribution(2, recentFeaturesArray, rate);
                    missedExamples++;
                    break;
            }

            UpdateAdaptationMode();
            OnFeedbackReceived?.Invoke(feedback);

            if (debugMode)
                Debug.Log($"Feedback: {feedback}, Rate: {rate:F3}, Total: {TotalFeedbackCount}");
        }

        /// <summary>
        /// キャプション表示後のユーザー反応を検出
        /// </summary>
        public void DetectCaptionFeedback(bool userReadCaption, float gazeOnCaptionTime)
        {
            if (lastTriggerTime < 0) return;

            float timeSinceTrigger = Time.time - lastTriggerTime;

            if (userReadCaption && gazeOnCaptionTime >= captionReadThreshold)
            {
                ReportFeedback(FeedbackType.TruePositive);
            }
            else if (timeSinceTrigger < quickLookAwayThreshold)
            {
                ReportFeedback(FeedbackType.FalsePositive);
            }

            lastTriggerTime = -1f;
        }

        private void UpdateObjectTracking(int objectId)
        {
            if (objectId == currentObjectId && objectId >= 0)
            {
                currentObjectGazeTime += Time.deltaTime;

                // 見逃し検出
                if (currentObjectGazeTime >= longGazeThreshold && 
                    lastTriggerTime < Time.time - longGazeThreshold)
                {
                    ReportFeedback(FeedbackType.FalseNegative);
                    currentObjectGazeTime = 0f; // リセット
                }
            }
            else
            {
                currentObjectId = objectId;
                currentObjectGazeTime = 0f;
            }
        }

        private float GetCurrentAdaptationRate()
        {
            int total = TotalFeedbackCount;

            // 減衰学習率
            float decayedRate = initialAdaptationRate / Mathf.Sqrt(total + 1);
            return Mathf.Max(decayedRate, minAdaptationRate);
        }

        private void UpdateAdaptationMode()
        {
            int total = TotalFeedbackCount;

            if (total < 5)
                adaptationMode = AdaptationMode.Exploring;
            else if (total < 20)
                adaptationMode = AdaptationMode.Exploring;
            else
                adaptationMode = AdaptationMode.Confident;
        }

        private void AdaptStateFromSamples(int stateIdx, float[][] samples, float rate)
        {
            if (samples.Length == 0) return;

            // 直近のサンプルから平均を計算
            float[] newMean = new float[N_FEATURES];
            int count = Mathf.Min(samples.Length, 30); // 直近30サンプル

            for (int i = samples.Length - count; i < samples.Length; i++)
            {
                var normalized = NormalizeFeatures(samples[i]);
                for (int f = 0; f < N_FEATURES; f++)
                    newMean[f] += normalized[f];
            }

            for (int f = 0; f < N_FEATURES; f++)
                newMean[f] /= count;

            // 制限付き更新
            for (int f = 0; f < N_FEATURES; f++)
            {
                float delta = rate * (newMean[f] - means[stateIdx, f]);
                delta = Mathf.Clamp(delta, -maxDeltaPerUpdate, maxDeltaPerUpdate);
                means[stateIdx, f] += delta;
            }

            // Overview状態も調整
            InterpolateOverviewState();
        }

        private void ExpandStateDistribution(int stateIdx, float[][] samples, float rate)
        {
            if (samples.Length == 0) return;

            // 見逃しの場合、標準偏差を少し広げる
            float expansionFactor = 1.0f + rate * 0.5f;

            for (int f = 0; f < N_FEATURES; f++)
            {
                stds[stateIdx, f] *= expansionFactor;
                // 上限設定
                stds[stateIdx, f] = Mathf.Min(stds[stateIdx, f], 1.0f);
            }
        }

        #endregion

        #region Export/Import

        /// <summary>
        /// 適応後のパラメータをJSONで出力
        /// </summary>
        public string ExportParameters()
        {
            var data = new HMMParametersData
            {
                means = To2DArray(means),
                stds = To2DArray(stds),
                featureMeans = featureMeans,
                featureStds = featureStds,
                positiveExamples = positiveExamples,
                negativeExamples = negativeExamples,
                missedExamples = missedExamples
            };

            return JsonUtility.ToJson(data, true);
        }

        private float[][] To2DArray(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[][] result = new float[rows][];
            for (int i = 0; i < rows; i++)
            {
                result[i] = new float[cols];
                for (int j = 0; j < cols; j++)
                    result[i][j] = matrix[i, j];
            }
            return result;
        }

        [Serializable]
        private class HMMParametersData
        {
            public float[][] means;
            public float[][] stds;
            public float[] featureMeans;
            public float[] featureStds;
            public int positiveExamples;
            public int negativeExamples;
            public int missedExamples;
        }

        #endregion

        #region Debug Visualization

        private void OnGUI()
        {
            if (!debugMode || !IsReady) return;

            GUILayout.BeginArea(new Rect(10, 10, 300, 200));
            GUILayout.Label($"Attention State: {CurrentState}");
            GUILayout.Label($"DE Probability: {DeepEngagementProb:F3}");
            GUILayout.Label($"Adaptation Mode: {adaptationMode}");
            GUILayout.Label($"Total Feedback: {TotalFeedbackCount}");
            GUILayout.Label($"  Positive: {positiveExamples}");
            GUILayout.Label($"  Negative: {negativeExamples}");
            GUILayout.Label($"  Missed: {missedExamples}");
            GUILayout.EndArea();
        }

        #endregion
    }
}
