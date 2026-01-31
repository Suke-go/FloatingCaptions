/*
 * CalibrationController.cs
 * ========================
 * キャリブレーションフェーズのUI・フロー制御
 * 
 * 使用法:
 *   1. シーンにCalibrationControllerを配置
 *   2. キャリブレーション対象オブジェクトを設定
 *   3. StartCalibration() を呼び出し
 */

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace FloatingCaption.Attention
{
    /// <summary>
    /// キャリブレーションフェーズの制御
    /// </summary>
    public class CalibrationController : MonoBehaviour
    {
        [Header("Calibration Targets")]
        [SerializeField] private Transform[] calibrationTargets;
        [SerializeField] private float glancingDuration = 1.5f;
        [SerializeField] private float engagedDuration = 3.0f;
        [SerializeField] private float transitionDelay = 1.0f;

        [Header("UI Elements")]
        [SerializeField] private GameObject calibrationUI;
        [SerializeField] private TextMeshProUGUI instructionText;
        [SerializeField] private Image progressRing;
        [SerializeField] private Image targetHighlight;

        [Header("Visual Feedback")]
        [SerializeField] private Color glancingColor = new Color(1f, 0.8f, 0f);  // 黄色
        [SerializeField] private Color engagedColor = new Color(0f, 0.8f, 0.3f); // 緑
        [SerializeField] private float highlightScale = 1.5f;

        [Header("Feature Input")]
        [SerializeField] private FeatureExtractor featureExtractor;

        // 状態
        private bool isCalibrating = false;
        private Coroutine calibrationCoroutine;

        // イベント
        public event Action OnCalibrationStarted;
        public event Action OnCalibrationCompleted;
        public event Action<float> OnProgressUpdated;  // 0-1

        #region Public API

        /// <summary>
        /// キャリブレーションを開始
        /// </summary>
        public void StartCalibration()
        {
            if (isCalibrating) return;

            isCalibrating = true;
            calibrationUI?.SetActive(true);
            OnCalibrationStarted?.Invoke();

            calibrationCoroutine = StartCoroutine(CalibrationSequence());
        }

        /// <summary>
        /// キャリブレーションをキャンセル
        /// </summary>
        public void CancelCalibration()
        {
            if (calibrationCoroutine != null)
            {
                StopCoroutine(calibrationCoroutine);
                calibrationCoroutine = null;
            }

            isCalibrating = false;
            calibrationUI?.SetActive(false);
            HideHighlight();
        }

        #endregion

        #region Calibration Sequence

        private IEnumerator CalibrationSequence()
        {
            var detector = AdaptiveHMMAttentionDetector.Instance;
            int totalSteps = calibrationTargets.Length * 2;
            int currentStep = 0;

            // ========================
            // Phase 1: Glancing（軽く見る）
            // ========================
            ShowInstruction("まず、各オブジェクトを<b>軽く</b>見てください");
            yield return new WaitForSeconds(2f);

            foreach (var target in calibrationTargets)
            {
                ShowInstruction("このオブジェクトを軽く見てください");
                ShowHighlight(target, glancingColor);

                float elapsed = 0f;
                while (elapsed < glancingDuration)
                {
                    // 特徴量を収集
                    if (featureExtractor != null)
                    {
                        float[] features = featureExtractor.GetCurrentFeatures();
                        detector.AddCalibrationGlancing(features);
                    }

                    elapsed += Time.deltaTime;
                    UpdateProgress(elapsed / glancingDuration, glancingColor);
                    yield return null;
                }

                currentStep++;
                OnProgressUpdated?.Invoke((float)currentStep / totalSteps);

                HideHighlight();
                yield return new WaitForSeconds(transitionDelay);
            }

            // ========================
            // Phase 2: DeepEngagement（じっくり見る）
            // ========================
            ShowInstruction("次は、各オブジェクトを<b>じっくり</b>見てください");
            yield return new WaitForSeconds(2f);

            foreach (var target in calibrationTargets)
            {
                ShowInstruction("このオブジェクトをじっくり見てください\n（細部まで観察してください）");
                ShowHighlight(target, engagedColor);

                float elapsed = 0f;
                while (elapsed < engagedDuration)
                {
                    // 特徴量を収集
                    if (featureExtractor != null)
                    {
                        float[] features = featureExtractor.GetCurrentFeatures();
                        detector.AddCalibrationEngaged(features);
                    }

                    elapsed += Time.deltaTime;
                    UpdateProgress(elapsed / engagedDuration, engagedColor);
                    yield return null;
                }

                currentStep++;
                OnProgressUpdated?.Invoke((float)currentStep / totalSteps);

                HideHighlight();
                yield return new WaitForSeconds(transitionDelay);
            }

            // ========================
            // 完了処理
            // ========================
            ShowInstruction("キャリブレーション完了！");
            
            bool success = detector.CompleteCalibration();
            if (success)
            {
                yield return new WaitForSeconds(1f);
                ShowInstruction("準備完了です。体験をお楽しみください！");
            }
            else
            {
                ShowInstruction("キャリブレーションに失敗しました。再試行してください。");
            }

            yield return new WaitForSeconds(2f);

            isCalibrating = false;
            calibrationUI?.SetActive(false);
            OnCalibrationCompleted?.Invoke();
        }

        #endregion

        #region UI Helpers

        private void ShowInstruction(string text)
        {
            if (instructionText != null)
                instructionText.text = text;
        }

        private void ShowHighlight(Transform target, Color color)
        {
            if (targetHighlight == null) return;

            // ワールド座標をスクリーン座標に変換
            Vector3 screenPos = Camera.main.WorldToScreenPoint(target.position);
            targetHighlight.transform.position = screenPos;
            targetHighlight.color = color;
            targetHighlight.transform.localScale = Vector3.one * highlightScale;
            targetHighlight.gameObject.SetActive(true);
        }

        private void HideHighlight()
        {
            if (targetHighlight != null)
                targetHighlight.gameObject.SetActive(false);
        }

        private void UpdateProgress(float progress, Color color)
        {
            if (progressRing != null)
            {
                progressRing.fillAmount = progress;
                progressRing.color = color;
            }
        }

        #endregion
    }

    /// <summary>
    /// 特徴量抽出器（Eye Tracker / Head Trackerからの入力）
    /// </summary>
    public class FeatureExtractor : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private Transform headTransform;

        // 内部状態
        private Quaternion lastHeadRotation;
        private Vector3[] gazePositionBuffer = new Vector3[60];  // 1秒分
        private int gazeBufferIndex = 0;
        private float[] pupilBuffer = new float[60];
        private int pupilBufferIndex = 0;
        private float baselinePupil = 3.5f;  // mm

        private void Start()
        {
            if (headTransform == null)
                headTransform = Camera.main.transform;

            lastHeadRotation = headTransform.rotation;
        }

        /// <summary>
        /// 現在のフレームの特徴量を取得
        /// </summary>
        public float[] GetCurrentFeatures()
        {
            float headRotVelocity = CalculateHeadRotVelocity();
            float gazeDispersion = CalculateGazeDispersion();
            float spatialEntropy = CalculateSpatialEntropy();
            float pupilDelta = CalculatePupilDelta();

            return new float[] { headRotVelocity, gazeDispersion, spatialEntropy, pupilDelta };
        }

        /// <summary>
        /// Eye Trackerからの生データを更新（外部から呼び出し）
        /// </summary>
        public void UpdateGazeData(Vector3 worldGazePoint, float leftPupil, float rightPupil)
        {
            gazePositionBuffer[gazeBufferIndex] = worldGazePoint;
            gazeBufferIndex = (gazeBufferIndex + 1) % gazePositionBuffer.Length;

            float avgPupil = (leftPupil + rightPupil) / 2f;
            pupilBuffer[pupilBufferIndex] = avgPupil;
            pupilBufferIndex = (pupilBufferIndex + 1) % pupilBuffer.Length;
        }

        /// <summary>
        /// ベースライン瞳孔径を設定
        /// </summary>
        public void SetBaselinePupil(float baseline)
        {
            baselinePupil = baseline;
        }

        #region Feature Calculation

        private float CalculateHeadRotVelocity()
        {
            Quaternion currentRot = headTransform.rotation;
            float angle = Quaternion.Angle(lastHeadRotation, currentRot);
            float velocity = angle / Time.deltaTime * Mathf.Deg2Rad;
            lastHeadRotation = currentRot;
            return velocity;
        }

        private float CalculateGazeDispersion()
        {
            // 直近30フレームの視線位置の標準偏差
            Vector3 mean = Vector3.zero;
            int count = 0;

            for (int i = 0; i < 30; i++)
            {
                int idx = (gazeBufferIndex - 1 - i + gazePositionBuffer.Length) % gazePositionBuffer.Length;
                if (gazePositionBuffer[idx] != Vector3.zero)
                {
                    mean += gazePositionBuffer[idx];
                    count++;
                }
            }

            if (count < 5) return 0.5f;
            mean /= count;

            float variance = 0f;
            for (int i = 0; i < 30; i++)
            {
                int idx = (gazeBufferIndex - 1 - i + gazePositionBuffer.Length) % gazePositionBuffer.Length;
                if (gazePositionBuffer[idx] != Vector3.zero)
                {
                    variance += (gazePositionBuffer[idx] - mean).sqrMagnitude;
                }
            }

            return Mathf.Sqrt(variance / count);
        }

        private float CalculateSpatialEntropy()
        {
            // 視線位置を2Dグリッドに投影してエントロピー計算
            int[] histogram = new int[25];  // 5x5グリッド
            int count = 0;

            for (int i = 0; i < 30; i++)
            {
                int idx = (gazeBufferIndex - 1 - i + gazePositionBuffer.Length) % gazePositionBuffer.Length;
                Vector3 pos = gazePositionBuffer[idx];
                if (pos == Vector3.zero) continue;

                // 正規化（-1〜1 → 0〜4）
                int gx = Mathf.Clamp(Mathf.FloorToInt((pos.x + 1f) * 2.5f), 0, 4);
                int gy = Mathf.Clamp(Mathf.FloorToInt((pos.y + 1f) * 2.5f), 0, 4);
                histogram[gy * 5 + gx]++;
                count++;
            }

            if (count < 5) return 0.5f;

            // エントロピー計算
            float entropy = 0f;
            for (int i = 0; i < 25; i++)
            {
                if (histogram[i] > 0)
                {
                    float p = (float)histogram[i] / count;
                    entropy -= p * Mathf.Log(p) / Mathf.Log(2);
                }
            }

            // 正規化（max = log2(25) ≈ 4.64）
            return entropy / 4.64f;
        }

        private float CalculatePupilDelta()
        {
            // 直近の平均瞳孔径とベースラインの差
            float sum = 0f;
            int count = 0;

            for (int i = 0; i < 30; i++)
            {
                int idx = (pupilBufferIndex - 1 - i + pupilBuffer.Length) % pupilBuffer.Length;
                if (pupilBuffer[idx] > 0)
                {
                    sum += pupilBuffer[idx];
                    count++;
                }
            }

            if (count < 5) return 0f;
            return (sum / count) - baselinePupil;
        }

        #endregion
    }
}
