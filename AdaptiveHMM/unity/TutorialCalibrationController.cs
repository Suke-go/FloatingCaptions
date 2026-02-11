/*
 * TutorialCalibrationController.cs
 * =================================
 * チュートリアル埋め込み型キャリブレーション
 * 
 * 明示的なキャリブレーション指示を排除し、
 * VR美術館チュートリアルの導線として自然にサンプルを収集する。
 * 
 * フロー:
 *   Target 1: 移動（Glancing収集）→ 注視（Engaged収集）
 *   Target 2: 移動（Glancing収集）→ 注視（Engaged収集）
 *   ...
 *   Target N: 移動（Glancing収集）→ 注視（Engaged収集）
 *   → CompleteCalibration → 自由探索開始
 */

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace FloatingCaption.Attention
{
    /// <summary>
    /// チュートリアル埋め込み型キャリブレーションコントローラー
    /// </summary>
    public class TutorialCalibrationController : MonoBehaviour
    {
        #region Inspector Settings

        [Header("Tutorial Targets")]
        [Tooltip("巡回する展示物のTransform（推奨: 3個以上）")]
        [SerializeField] private Transform[] calibrationTargets;

        [Header("Timing")]
        [Tooltip("各ターゲット間の移動フェーズ時間（秒）")]
        [SerializeField] private float exploreDuration = 5.0f;
        [Tooltip("各ターゲット前の注視フェーズ時間（秒）")]
        [SerializeField] private float focusDuration = 5.0f;
        [Tooltip("フェーズ間のインターバル（秒）")]
        [SerializeField] private float transitionDelay = 0.5f;

        [Header("UI Elements")]
        [SerializeField] private GameObject tutorialUI;
        [SerializeField] private TextMeshProUGUI messageText;
        [SerializeField] private Image progressBar;
        [SerializeField] private GameObject guideArrow;

        [Header("Visual Feedback")]
        [SerializeField] private Color highlightColor = new Color(1f, 0.95f, 0.8f, 0.6f);
        [SerializeField] private float highlightPulseSpeed = 2.0f;

        [Header("References")]
        [SerializeField] private FeatureExtractor featureExtractor;

        #endregion

        #region State

        private bool isRunning = false;
        private Coroutine tutorialCoroutine;
        private int currentTargetIndex = -1;

        // ハイライト用
        private Renderer currentHighlightRenderer;
        private Color originalColor;
        private MaterialPropertyBlock highlightBlock;

        #endregion

        #region Events

        public event Action OnTutorialStarted;
        public event Action OnTutorialCompleted;
        public event Action<int, int> OnTargetChanged;  // (currentIndex, totalCount)
        public event Action<float> OnProgressUpdated;    // 0-1

        #endregion

        #region Public API

        /// <summary>
        /// チュートリアルを開始
        /// </summary>
        public void StartTutorial()
        {
            if (isRunning) return;
            if (calibrationTargets == null || calibrationTargets.Length == 0)
            {
                Debug.LogError("TutorialCalibration: No calibration targets assigned.");
                return;
            }

            isRunning = true;
            highlightBlock = new MaterialPropertyBlock();
            tutorialUI?.SetActive(true);
            OnTutorialStarted?.Invoke();

            // HMMをチュートリアルキャリブレーションモードに
            var detector = AdaptiveHMMAttentionDetector.Instance;
            if (detector != null)
            {
                detector.StartTutorialCalibration();
            }

            tutorialCoroutine = StartCoroutine(TutorialSequence());
        }

        /// <summary>
        /// チュートリアルをスキップ（デバッグ用）
        /// </summary>
        public void SkipTutorial()
        {
            if (tutorialCoroutine != null)
            {
                StopCoroutine(tutorialCoroutine);
                tutorialCoroutine = null;
            }

            ClearHighlight();
            isRunning = false;
            tutorialUI?.SetActive(false);

            // デフォルトパラメータでExploringモードへ
            var detector = AdaptiveHMMAttentionDetector.Instance;
            if (detector != null)
            {
                detector.CompleteTutorialCalibration();
            }
        }

        public bool IsRunning => isRunning;

        #endregion

        #region Tutorial Sequence

        private IEnumerator TutorialSequence()
        {
            var detector = AdaptiveHMMAttentionDetector.Instance;
            int totalTargets = calibrationTargets.Length;

            // --- Welcome ---
            ShowMessage("ようこそ！最初の作品に向かいましょう");
            yield return new WaitForSeconds(2.0f);

            // --- Target Cycling ---
            for (int i = 0; i < totalTargets; i++)
            {
                currentTargetIndex = i;
                OnTargetChanged?.Invoke(i, totalTargets);

                // ===========================
                // 移動フェーズ（Glancing収集）
                // ===========================
                if (i == 0)
                    ShowMessage("周りを見ながら作品に近づいてみましょう");
                else
                    ShowMessage("次の作品に向かいましょう");

                ShowGuideArrow(calibrationTargets[i]);

                float elapsed = 0f;
                while (elapsed < exploreDuration)
                {
                    // Glancingサンプル収集
                    if (featureExtractor != null && detector != null)
                    {
                        float[] features = featureExtractor.GetCurrentFeatures();
                        detector.AddCalibrationGlancing(features);
                    }

                    elapsed += Time.deltaTime;
                    UpdateProgress(elapsed / exploreDuration);
                    yield return null;
                }

                // ===========================
                // 注視フェーズ（Engaged収集）
                // ===========================
                ShowMessage("この作品をご覧ください");
                HideGuideArrow();
                SetHighlight(calibrationTargets[i]);

                elapsed = 0f;
                while (elapsed < focusDuration)
                {
                    // Engagedサンプル収集
                    if (featureExtractor != null && detector != null)
                    {
                        float[] features = featureExtractor.GetCurrentFeatures();
                        detector.AddCalibrationEngaged(features);
                    }

                    elapsed += Time.deltaTime;
                    UpdateProgress(elapsed / focusDuration);

                    // ハイライトパルスアニメーション
                    PulseHighlight(elapsed);

                    yield return null;
                }

                ClearHighlight();

                // 短いインターバル
                yield return new WaitForSeconds(transitionDelay);

                // 全体進捗更新
                OnProgressUpdated?.Invoke((float)(i + 1) / totalTargets);
            }

            // --- 完了 ---
            bool success = false;
            if (detector != null)
            {
                success = detector.CompleteTutorialCalibration();
            }

            if (success)
            {
                ShowMessage("準備完了！自由にお楽しみください");
            }
            else
            {
                ShowMessage("もう少し見回してみましょう…");
                // サンプル不足の場合、オンライン適応に頼る
                Debug.LogWarning("TutorialCalibration: Insufficient samples, falling back to online adaptation.");
            }

            yield return new WaitForSeconds(2.0f);

            // クリーンアップ
            HideGuideArrow();
            isRunning = false;
            tutorialUI?.SetActive(false);
            OnTutorialCompleted?.Invoke();
        }

        #endregion

        #region UI Helpers

        private void ShowMessage(string text)
        {
            if (messageText != null)
                messageText.text = text;
        }

        private void UpdateProgress(float progress)
        {
            if (progressBar != null)
                progressBar.fillAmount = Mathf.Clamp01(progress);
        }

        private void ShowGuideArrow(Transform target)
        {
            if (guideArrow == null) return;
            guideArrow.SetActive(true);
            // アローをターゲット方向に向ける
            StartCoroutine(UpdateGuideArrowDirection(target));
        }

        private void HideGuideArrow()
        {
            if (guideArrow != null)
                guideArrow.SetActive(false);
        }

        private IEnumerator UpdateGuideArrowDirection(Transform target)
        {
            while (guideArrow != null && guideArrow.activeSelf && target != null)
            {
                Vector3 dir = (target.position - Camera.main.transform.position).normalized;
                dir.y = 0; // 水平方向のみ
                if (dir.sqrMagnitude > 0.01f)
                {
                    guideArrow.transform.rotation = Quaternion.LookRotation(dir);
                }
                yield return null;
            }
        }

        #endregion

        #region Highlight

        private void SetHighlight(Transform target)
        {
            if (target == null) return;

            currentHighlightRenderer = target.GetComponentInChildren<Renderer>();
            if (currentHighlightRenderer == null) return;

            currentHighlightRenderer.GetPropertyBlock(highlightBlock);
            originalColor = currentHighlightRenderer.material.color;
        }

        private void PulseHighlight(float time)
        {
            if (currentHighlightRenderer == null) return;

            float pulse = (Mathf.Sin(time * highlightPulseSpeed * Mathf.PI * 2f) + 1f) * 0.5f;
            Color pulsed = Color.Lerp(originalColor, highlightColor, pulse * 0.3f);
            highlightBlock.SetColor("_Color", pulsed);
            currentHighlightRenderer.SetPropertyBlock(highlightBlock);
        }

        private void ClearHighlight()
        {
            if (currentHighlightRenderer == null) return;

            highlightBlock.SetColor("_Color", originalColor);
            currentHighlightRenderer.SetPropertyBlock(highlightBlock);
            currentHighlightRenderer = null;
        }

        #endregion
    }
}
