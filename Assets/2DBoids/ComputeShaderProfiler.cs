using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

namespace BoidsUnity
{
    public class ComputeShaderProfiler : MonoBehaviour
    {
        [Header("Profiling Settings")]
        [SerializeField] private bool enableProfiler = true;
        [SerializeField] private bool showProfilerOnScreen = true;
        [SerializeField] private KeyCode toggleProfilerKey = KeyCode.P;
        [SerializeField] private KeyCode logResultsKey = KeyCode.L;
        [SerializeField] private KeyCode toggleVisibilityKey = KeyCode.V;
        
        [Header("Profiler Display")]
        [SerializeField] private Vector2 displayPosition = new Vector2(10, 10);
        [SerializeField] private int fontSize = 16;
        [SerializeField] private Color textColor = Color.white;
        [SerializeField] private Color backgroundColor = new Color(0, 0, 0, 0.7f);
        
        private bool _initialized = false;
        
        void Start()
        {
            if (enableProfiler)
            {
                ProfilingUtility.EnableProfiling(showProfilerOnScreen);
            }
            
            _initialized = true;
        }
        
        void Update()
        {
            // Toggle profiler on/off with key press
            if (Input.GetKeyDown(toggleProfilerKey))
            {
                if (ProfilingUtilityExtensions._profilerEnabled)
                {
                    ProfilingUtility.DisableProfiling();
                    Debug.Log("Profiler disabled");
                }
                else
                {
                    ProfilingUtility.EnableProfiling(showProfilerOnScreen);
                    Debug.Log("Profiler enabled");
                }
            }
            
            // Toggle visibility with key press
            if (Input.GetKeyDown(toggleVisibilityKey) && ProfilingUtilityExtensions._profilerEnabled)
            {
                bool isVisible = ProfilingUtilityExtensions._showOnScreen;
                ProfilingUtilityExtensions._showOnScreen = !isVisible;
                Debug.Log(isVisible ? "Profiler display hidden" : "Profiler display visible");
            }
            
            // Log results with key press
            if (Input.GetKeyDown(logResultsKey) && ProfilingUtilityExtensions._profilerEnabled)
            {
                ProfilingUtility.LogTimings();
            }
            
            // Begin frame profiling
            ProfilingUtility.BeginFrame();
        }
        
        void LateUpdate()
        {
            // End frame profiling
            ProfilingUtility.EndFrame();
        }
        
        void OnGUI()
        {
            if (!_initialized || !ProfilingUtilityExtensions._profilerEnabled || !showProfilerOnScreen) return;
            
            ProfilingUtility.OnGUI();
        }
        
        void OnDestroy()
        {
            ProfilingUtility.Cleanup();
        }
    }
    
    // Extension to make the private fields accessible
    public static class ProfilingUtilityExtensions
    {
        public static bool _profilerEnabled
        {
            get { return (bool)(typeof(ProfilingUtility).GetField("_profilerEnabled", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static).GetValue(null) ?? false); }
            set { typeof(ProfilingUtility).GetField("_profilerEnabled", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static).SetValue(null, value); }
        }
        
        public static bool _showOnScreen
        {
            get { return (bool)(typeof(ProfilingUtility).GetField("_showOnScreen", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static).GetValue(null) ?? false); }
            set { typeof(ProfilingUtility).GetField("_showOnScreen", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static).SetValue(null, value); }
        }
    }
}