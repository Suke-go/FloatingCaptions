/*
 * HMMAttentionDetector.cs
 * ======================
 * Unity用の軽量HMMリアルタイム推論
 * 
 * 使用法:
 *   1. Pythonで訓練したhmm_params.jsonをResources/に配置
 *   2. HMMAttentionDetector.Instance.Initialize()
 *   3. 毎フレーム: detector.Update(headRotVelocity, gazeDispersion, entropy, pupilDelta)
 */

using System;
using System.Collections.Generic;
using UnityEngine;

namespace FloatingCaption.Attention
{
    /// <summary>
    /// 注意状態の定義
    /// </summary>
    public enum AttentionState
    {
        Glancing = 0,       // 一瞥
        Overview = 1,        // 概観
        DeepEngagement = 2   // 深い興味 → トリガー対象
    }

    /// <summary>
    /// HMMパラメータ（JSONから読み込み）
    /// </summary>
    [Serializable]
    public class HMMParameters
    {
        public int n_states;
        public int n_features;
        public float[] pi;
        public float[][] A;
        public float[][] means;
        public float[][] stds;
        public float[] feature_means;
        public float[] feature_stds;
    }

    /// <summary>
    /// 軽量HMM推論エンジン
    /// </summary>
    public class HMMAttentionDetector : MonoBehaviour
    {
        public static HMMAttentionDetector Instance { get; private set; }

        [Header("HMM Settings")]
        [SerializeField] private string parametersPath = "hmm_params";
        [SerializeField] private float triggerThreshold = 0.6f;
        [SerializeField] private int sustainedFrames = 30; // 0.5秒 @ 60fps

        [Header("Debug")]
        [SerializeField] private bool debugMode = false;

        // HMMパラメータ
        private int nStates;
        private int nFeatures;
        private float[] pi;
        private float[,] A;
        private float[,] means;
        private float[,] stds;
        private float[] featureMeans;
        private float[] featureStds;

        // 状態追跡
        private float[] alpha;
        private Queue<AttentionState> stateHistory;

        // 公開プロパティ
        public AttentionState CurrentState { get; private set; }
        public float[] StateProbs => alpha;
        public float DeepEngagementProb => alpha != null ? alpha[(int)AttentionState.DeepEngagement] : 0f;
        public bool IsReady { get; private set; }

        // イベント
        public event Action<AttentionState> OnStateChanged;
        public event Action OnTrigger;

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
            }
        }

        private void Start()
        {
            Initialize();
        }

        /// <summary>
        /// JSONからパラメータを読み込んで初期化
        /// </summary>
        public void Initialize()
        {
            try
            {
                TextAsset jsonFile = Resources.Load<TextAsset>(parametersPath);
                if (jsonFile == null)
                {
                    Debug.LogError($"HMM parameters not found: {parametersPath}");
                    InitializeDefault();
                    return;
                }

                var jsonData = JsonUtility.FromJson<HMMParametersJson>(jsonFile.text);
                LoadFromJson(jsonData);
                
                Debug.Log($"HMM initialized with {nStates} states, {nFeatures} features");
                IsReady = true;
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to load HMM parameters: {e.Message}");
                InitializeDefault();
            }
        }

        /// <summary>
        /// デフォルトパラメータで初期化（フォールバック）
        /// </summary>
        private void InitializeDefault()
        {
            nStates = 3;
            nFeatures = 4;

            pi = new float[] { 0.7f, 0.2f, 0.1f };

            A = new float[,] {
                { 0.7f, 0.25f, 0.05f },
                { 0.2f, 0.5f, 0.3f },
                { 0.1f, 0.3f, 0.6f }
            };

            means = new float[,] {
                { 0.5f, 0.8f, 0.3f, 0.0f },
                { 0.2f, 0.5f, 0.4f, 0.1f },
                { 0.05f, 0.2f, 0.6f, 0.23f }
            };

            stds = new float[,] {
                { 0.2f, 0.3f, 0.1f, 0.1f },
                { 0.1f, 0.2f, 0.1f, 0.1f },
                { 0.05f, 0.1f, 0.1f, 0.1f }
            };

            featureMeans = new float[] { 0.2f, 0.4f, 0.4f, 0.1f };
            featureStds = new float[] { 0.2f, 0.3f, 0.2f, 0.15f };

            ResetForward();
            stateHistory = new Queue<AttentionState>();
            IsReady = true;

            Debug.Log("HMM initialized with default parameters");
        }

        /// <summary>
        /// Forward変数をリセット
        /// </summary>
        public void ResetForward()
        {
            alpha = new float[nStates];
            Array.Copy(pi, alpha, nStates);
            stateHistory = new Queue<AttentionState>();
        }

        /// <summary>
        /// 1フレーム分の推論を実行
        /// </summary>
        public (bool shouldTrigger, float confidence) Update(
            float headRotVelocity,
            float gazeDispersion,
            float spatialEntropy,
            float pupilDelta)
        {
            if (!IsReady) return (false, 0f);

            // 特徴量ベクトル
            float[] features = new float[] {
                headRotVelocity,
                gazeDispersion,
                spatialEntropy,
                pupilDelta
            };

            // 正規化
            float[] normalized = NormalizeFeatures(features);

            // Forward step
            ForwardStep(normalized);

            // 状態決定
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

            // トリガー判定
            bool shouldTrigger = ShouldTrigger();
            float confidence = alpha[(int)AttentionState.DeepEngagement];

            if (shouldTrigger)
            {
                OnTrigger?.Invoke();
            }

            if (debugMode)
            {
                Debug.Log($"State: {CurrentState}, DE prob: {confidence:F3}, Trigger: {shouldTrigger}");
            }

            return (shouldTrigger, confidence);
        }

        /// <summary>
        /// 特徴量を正規化
        /// </summary>
        private float[] NormalizeFeatures(float[] features)
        {
            float[] normalized = new float[nFeatures];
            for (int i = 0; i < nFeatures; i++)
            {
                normalized[i] = (features[i] - featureMeans[i]) / (featureStds[i] + 1e-8f);
                if (float.IsNaN(normalized[i]) || float.IsInfinity(normalized[i]))
                    normalized[i] = 0f;
            }
            return normalized;
        }

        /// <summary>
        /// ガウス確率密度計算（対角共分散）
        /// </summary>
        private float GaussianPDF(float[] x, int stateIdx)
        {
            float logProb = 0f;
            for (int i = 0; i < nFeatures; i++)
            {
                float diff = x[i] - means[stateIdx, i];
                float std = stds[stateIdx, i] + 1e-8f;
                logProb -= 0.5f * (diff * diff) / (std * std);
                logProb -= Mathf.Log(std);
            }
            logProb -= 0.5f * nFeatures * Mathf.Log(2f * Mathf.PI);
            return Mathf.Exp(Mathf.Clamp(logProb, -500f, 0f));
        }

        /// <summary>
        /// Forward step（1フレーム分）
        /// </summary>
        private void ForwardStep(float[] features)
        {
            float[] alphaNew = new float[nStates];
            float[] emission = new float[nStates];

            // 観測確率計算
            for (int k = 0; k < nStates; k++)
            {
                emission[k] = GaussianPDF(features, k) + 1e-10f;
            }

            // Forward更新
            for (int j = 0; j < nStates; j++)
            {
                float sum = 0f;
                for (int i = 0; i < nStates; i++)
                {
                    sum += alpha[i] * A[i, j];
                }
                alphaNew[j] = emission[j] * sum;
            }

            // 正規化
            float total = 0f;
            for (int k = 0; k < nStates; k++)
                total += alphaNew[k];

            if (total > 1e-10f)
            {
                for (int k = 0; k < nStates; k++)
                    alpha[k] = alphaNew[k] / total;
            }
        }

        /// <summary>
        /// 最も確率の高い状態を取得
        /// </summary>
        private AttentionState GetMostLikelyState()
        {
            int maxIdx = 0;
            float maxProb = alpha[0];
            for (int k = 1; k < nStates; k++)
            {
                if (alpha[k] > maxProb)
                {
                    maxProb = alpha[k];
                    maxIdx = k;
                }
            }
            return (AttentionState)maxIdx;
        }

        /// <summary>
        /// トリガー判定（持続条件付き）
        /// </summary>
        private bool ShouldTrigger()
        {
            if (stateHistory.Count < sustainedFrames)
                return false;

            int deCount = 0;
            foreach (var state in stateHistory)
            {
                if (state == AttentionState.DeepEngagement)
                    deCount++;
            }

            float ratio = (float)deCount / sustainedFrames;
            return ratio > 0.8f && alpha[(int)AttentionState.DeepEngagement] > triggerThreshold;
        }

        // JSON読み込み用の内部クラス
        [Serializable]
        private class HMMParametersJson
        {
            public int n_states;
            public int n_features;
            public float[] pi;
            public float[][] A;
            public float[][] means;
            public float[][] stds;
            public float[] feature_means;
            public float[] feature_stds;
        }

        private void LoadFromJson(HMMParametersJson json)
        {
            nStates = json.n_states;
            nFeatures = json.n_features;
            pi = json.pi;

            A = new float[nStates, nStates];
            means = new float[nStates, nFeatures];
            stds = new float[nStates, nFeatures];

            for (int i = 0; i < nStates; i++)
            {
                for (int j = 0; j < nStates; j++)
                    A[i, j] = json.A[i][j];

                for (int f = 0; f < nFeatures; f++)
                {
                    means[i, f] = json.means[i][f];
                    stds[i, f] = json.stds[i][f];
                }
            }

            featureMeans = json.feature_means;
            featureStds = json.feature_stds;

            ResetForward();
            stateHistory = new Queue<AttentionState>();
        }
    }
}
