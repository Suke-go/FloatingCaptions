/*
 * CaptionDisplayController.cs
 * ============================
 * HMMトリガーに連動してキャプションを表示・非表示するコントローラー
 * 
 * 使用法:
 *   1. シーンに配置し、captionDataList に表示対象を登録
 *   2. HMMの OnTrigger イベントで自動表示
 *   3. 視線が外れると自動的にフェードアウト
 */

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace FloatingCaption.Attention
{
    /// <summary>
    /// キャプションデータ（Inspector設定用）
    /// </summary>
    [Serializable]
    public class CaptionData
    {
        [Tooltip("対象オブジェクト")]
        public Transform targetObject;

        [Tooltip("キャプションテキスト")]
        [TextArea(2, 5)]
        public string captionText;

        [Tooltip("対象物からのオフセット（ローカル座標）")]
        public Vector3 displayOffset = new Vector3(0f, 0.5f, 0f);
    }

    /// <summary>
    /// キャプション表示コントローラー
    /// </summary>
    public class CaptionDisplayController : MonoBehaviour
    {
        #region Inspector Settings

        [Header("Caption Data")]
        [SerializeField] private List<CaptionData> captionDataList = new List<CaptionData>();

        [Header("Caption Prefab")]
        [Tooltip("WorldSpace CanvasのキャプションPrefab（TextMeshProUGUIを含むこと）")]
        [SerializeField] private GameObject captionPrefab;

        [Header("Display Settings")]
        [SerializeField] private float fadeDuration = 0.3f;
        [SerializeField] private float hideDelay = 2.0f;
        [SerializeField] private float cooldownSeconds = 10.0f;
        [SerializeField] private float gazeHitRadius = 1.5f;

        [Header("Look-at Behavior")]
        [SerializeField] private bool billboardToCamera = true;

        [Header("References")]
        [SerializeField] private AdaptiveHMMAttentionDetector hmmDetector;
        [SerializeField] private CaptionFeedbackDetector feedbackDetector;

        #endregion

        #region State

        // 表示中のキャプションインスタンス
        private Dictionary<Transform, GameObject> activeCaptions = new Dictionary<Transform, GameObject>();
        // クールダウン管理
        private Dictionary<Transform, float> lastShowTime = new Dictionary<Transform, float>();
        // 現在の視線ターゲット
        private Transform currentGazeTarget;
        // 紐づけ用ルックアップ
        private Dictionary<Transform, CaptionData> captionLookup = new Dictionary<Transform, CaptionData>();

        #endregion

        #region Events

        public event Action<CaptionData> OnCaptionShown;
        public event Action<CaptionData> OnCaptionHidden;

        #endregion

        #region Unity Lifecycle

        private void Start()
        {
            // ルックアップ構築
            foreach (var data in captionDataList)
            {
                if (data.targetObject != null)
                    captionLookup[data.targetObject] = data;
            }

            // HMMイベント購読
            if (hmmDetector == null)
                hmmDetector = AdaptiveHMMAttentionDetector.Instance;

            if (hmmDetector != null)
            {
                hmmDetector.OnTrigger += HandleHMMTrigger;
            }
        }

        private void OnDestroy()
        {
            if (hmmDetector != null)
            {
                hmmDetector.OnTrigger -= HandleHMMTrigger;
            }
        }

        private void LateUpdate()
        {
            // ビルボード処理
            if (billboardToCamera)
            {
                foreach (var kvp in activeCaptions)
                {
                    if (kvp.Value != null && kvp.Value.activeSelf)
                    {
                        kvp.Value.transform.LookAt(Camera.main.transform);
                        // テキストが裏返らないように反転
                        kvp.Value.transform.Rotate(0, 180, 0);
                    }
                }
            }
        }

        #endregion

        #region Public API

        /// <summary>
        /// 現在の視線ターゲットを更新（外部から毎フレーム呼び出し）
        /// </summary>
        public void UpdateGazeTarget(Transform target)
        {
            currentGazeTarget = target;
        }

        /// <summary>
        /// 特定オブジェクトのキャプションを強制表示（チュートリアル用）
        /// </summary>
        public void ForceShowCaption(Transform target)
        {
            if (captionLookup.TryGetValue(target, out var data))
            {
                ShowCaption(data);
            }
        }

        /// <summary>
        /// 全キャプションを非表示
        /// </summary>
        public void HideAllCaptions()
        {
            foreach (var kvp in activeCaptions)
            {
                if (kvp.Value != null)
                    StartCoroutine(FadeOutAndDestroy(kvp.Key, kvp.Value));
            }
        }

        #endregion

        #region Trigger Handling

        private void HandleHMMTrigger()
        {
            if (currentGazeTarget == null) return;

            // 対象にキャプションデータがあるか
            if (!captionLookup.TryGetValue(currentGazeTarget, out var data))
                return;

            // クールダウンチェック
            if (lastShowTime.TryGetValue(currentGazeTarget, out float last))
            {
                if (Time.time - last < cooldownSeconds)
                    return;
            }

            // すでに表示中なら何もしない
            if (activeCaptions.ContainsKey(currentGazeTarget))
                return;

            ShowCaption(data);
        }

        #endregion

        #region Show / Hide

        private void ShowCaption(CaptionData data)
        {
            if (captionPrefab == null || data.targetObject == null) return;

            // キャプションインスタンス生成
            Vector3 worldPos = data.targetObject.position + data.targetObject.TransformDirection(data.displayOffset);
            GameObject caption = Instantiate(captionPrefab, worldPos, Quaternion.identity);

            // テキスト設定
            var tmpText = caption.GetComponentInChildren<TextMeshProUGUI>();
            if (tmpText != null)
                tmpText.text = data.captionText;

            // フェードイン
            var canvasGroup = caption.GetComponentInChildren<CanvasGroup>();
            if (canvasGroup == null)
                canvasGroup = caption.AddComponent<CanvasGroup>();
            StartCoroutine(FadeIn(canvasGroup));

            activeCaptions[data.targetObject] = caption;
            lastShowTime[data.targetObject] = Time.time;

            // フィードバック検出器に通知
            if (feedbackDetector != null)
                feedbackDetector.OnCaptionShown(caption.transform);

            OnCaptionShown?.Invoke(data);

            // 自動非表示タイマー開始
            StartCoroutine(AutoHideCoroutine(data.targetObject, caption));
        }

        private IEnumerator AutoHideCoroutine(Transform target, GameObject caption)
        {
            float timeSinceLastGaze = 0f;

            while (caption != null && caption.activeSelf)
            {
                // ユーザーがまだ対象を見ているか
                bool isLooking = (currentGazeTarget == target);

                if (isLooking)
                {
                    timeSinceLastGaze = 0f;
                }
                else
                {
                    timeSinceLastGaze += Time.deltaTime;
                    if (timeSinceLastGaze >= hideDelay)
                    {
                        HideCaption(target, caption);
                        yield break;
                    }
                }

                yield return null;
            }
        }

        private void HideCaption(Transform target, GameObject caption)
        {
            if (caption == null) return;

            // フィードバック検出器に通知
            if (feedbackDetector != null)
                feedbackDetector.OnCaptionHidden();

            OnCaptionHidden?.Invoke(captionLookup.ContainsKey(target) ? captionLookup[target] : null);

            StartCoroutine(FadeOutAndDestroy(target, caption));
        }

        #endregion

        #region Fade Animation

        private IEnumerator FadeIn(CanvasGroup group)
        {
            if (group == null) yield break;

            group.alpha = 0f;
            float elapsed = 0f;

            while (elapsed < fadeDuration)
            {
                elapsed += Time.deltaTime;
                group.alpha = Mathf.Clamp01(elapsed / fadeDuration);
                yield return null;
            }

            group.alpha = 1f;
        }

        private IEnumerator FadeOutAndDestroy(Transform target, GameObject caption)
        {
            var group = caption.GetComponentInChildren<CanvasGroup>();
            if (group != null)
            {
                float elapsed = 0f;
                while (elapsed < fadeDuration)
                {
                    elapsed += Time.deltaTime;
                    group.alpha = 1f - Mathf.Clamp01(elapsed / fadeDuration);
                    yield return null;
                }
            }

            if (activeCaptions.ContainsKey(target))
                activeCaptions.Remove(target);

            Destroy(caption);
        }

        #endregion
    }
}
