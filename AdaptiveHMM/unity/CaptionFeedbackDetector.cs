/*
 * CaptionFeedbackDetector.cs
 * ==========================
 * キャプション表示後のユーザー反応を検出し、
 * Adaptive HMMにフィードバックを提供
 */

using System;
using UnityEngine;

namespace FloatingCaption.Attention
{
    /// <summary>
    /// キャプションへのユーザー反応を検出
    /// </summary>
    public class CaptionFeedbackDetector : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private AdaptiveHMMAttentionDetector hmmDetector;
        [SerializeField] private Transform captionTransform;

        [Header("Detection Settings")]
        [SerializeField] private float captionReadThreshold = 0.5f;    // 読んだと判定する時間
        [SerializeField] private float quickLookAwayThreshold = 0.2f;  // 誤トリガー判定時間
        [SerializeField] private float captionHitRadius = 0.5f;        // キャプションの当たり判定

        // 状態
        private bool isCaptionVisible = false;
        private float captionShowTime = -1f;
        private float gazeOnCaptionTime = 0f;
        private bool feedbackReported = false;

        // 視線位置（外部から更新）
        private Vector3 currentGazeWorldPosition;

        #region Public API

        /// <summary>
        /// キャプションが表示された時に呼び出す
        /// </summary>
        public void OnCaptionShown(Transform caption)
        {
            captionTransform = caption;
            isCaptionVisible = true;
            captionShowTime = Time.time;
            gazeOnCaptionTime = 0f;
            feedbackReported = false;
        }

        /// <summary>
        /// キャプションが非表示になった時に呼び出す
        /// </summary>
        public void OnCaptionHidden()
        {
            // フィードバックがまだ報告されていない場合
            if (isCaptionVisible && !feedbackReported)
            {
                EvaluateAndReportFeedback();
            }

            isCaptionVisible = false;
            captionTransform = null;
        }

        /// <summary>
        /// 視線位置を更新（Eye Trackerから毎フレーム呼び出し）
        /// </summary>
        public void UpdateGazePosition(Vector3 worldGazePoint)
        {
            currentGazeWorldPosition = worldGazePoint;
        }

        #endregion

        #region Unity Lifecycle

        private void Start()
        {
            if (hmmDetector == null)
                hmmDetector = AdaptiveHMMAttentionDetector.Instance;

            // HMMのトリガーイベントを購読
            if (hmmDetector != null)
            {
                hmmDetector.OnTrigger += HandleTrigger;
            }
        }

        private void OnDestroy()
        {
            if (hmmDetector != null)
            {
                hmmDetector.OnTrigger -= HandleTrigger;
            }
        }

        private void Update()
        {
            if (!isCaptionVisible || captionTransform == null || feedbackReported)
                return;

            // キャプションを見ているかチェック
            float distance = Vector3.Distance(currentGazeWorldPosition, captionTransform.position);
            bool isLookingAtCaption = distance < captionHitRadius;

            if (isLookingAtCaption)
            {
                gazeOnCaptionTime += Time.deltaTime;

                // 十分に読んだ → True Positive
                if (gazeOnCaptionTime >= captionReadThreshold)
                {
                    ReportFeedback(FeedbackType.TruePositive);
                }
            }
            else
            {
                // すぐに視線を逸らした → False Positive
                float timeSinceShow = Time.time - captionShowTime;
                if (timeSinceShow >= quickLookAwayThreshold && gazeOnCaptionTime < 0.1f)
                {
                    ReportFeedback(FeedbackType.FalsePositive);
                }
            }
        }

        #endregion

        #region Feedback Logic

        private void HandleTrigger()
        {
            // トリガー時にキャプションが表示されることを期待
            // 実際のキャプション表示はCaptionDisplayControllerから呼ばれる
        }

        private void EvaluateAndReportFeedback()
        {
            if (feedbackReported) return;

            if (gazeOnCaptionTime >= captionReadThreshold)
            {
                ReportFeedback(FeedbackType.TruePositive);
            }
            else if (gazeOnCaptionTime < 0.1f)
            {
                ReportFeedback(FeedbackType.FalsePositive);
            }
            // else: 曖昧なケースはフィードバックしない
        }

        private void ReportFeedback(FeedbackType type)
        {
            if (feedbackReported) return;
            feedbackReported = true;

            hmmDetector?.ReportFeedback(type);
            Debug.Log($"Caption feedback reported: {type}, gaze time: {gazeOnCaptionTime:F2}s");
        }

        #endregion
    }
}
