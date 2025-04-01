using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Profiling;
using Debug = UnityEngine.Debug;

namespace BoidsUnity
{
    public class ProfilingUtility
    {
        private static readonly Dictionary<string, SamplerInfo> Samplers = new Dictionary<string, SamplerInfo>();
        private static readonly Dictionary<string, double> AverageTimes = new Dictionary<string, double>();
        private static readonly int MaxSamples = 60; // Track last 60 frames (1 second at 60 FPS)
        private static int _frameCount = 0;
        private static bool _profilerEnabled = false;
        private static bool _showOnScreen = false;
        private static string _profilingText = "";
        private static double _lastManualTiming = 0;
        
        private class SamplerInfo
        {
            public Stopwatch Stopwatch = new Stopwatch();
            public CustomSampler Sampler;
            public Queue<long> SampleTimes = new Queue<long>();
            public double TotalTime = 0;
            
            // We store times in microseconds for better precision
            public double AverageTimeMs => SampleTimes.Count > 0 ? TotalTime / SampleTimes.Count : 0;
            
            public SamplerInfo(string name)
            {
                Sampler = CustomSampler.Create(name);
            }
            
            public void AddSample(long timeMicros)
            {
                // Add a minimal value for visibility
                timeMicros = System.Math.Max(1, timeMicros);
                
                SampleTimes.Enqueue(timeMicros);
                TotalTime += timeMicros;
                
                if (SampleTimes.Count > MaxSamples)
                {
                    TotalTime -= SampleTimes.Dequeue();
                }
            }
        }
        
        public static void EnableProfiling(bool showOnScreen = false)
        {
            _profilerEnabled = true;
            _showOnScreen = showOnScreen;
        }
        
        public static void DisableProfiling()
        {
            _profilerEnabled = false;
            _showOnScreen = false;
        }
        
        // Call this when the application quits to clean up resources
        public static void Cleanup()
        {
            if (_backgroundTexture != null)
            {
                UnityEngine.Object.Destroy(_backgroundTexture);
                _backgroundTexture = null;
            }
        }
        
        // Dictionary to store manual timings by name
        private static Dictionary<string, double> _manualTimings = new Dictionary<string, double>();
        
        // Used for manual timing of operations
        public static void RecordManualTiming(string name, double milliseconds)
        {
            if (!_profilerEnabled) return;
            
            // Store in the dictionary
            _manualTimings[name] = milliseconds;
            
            // Also add to average times for display in UI
            if (!Samplers.ContainsKey(name))
            {
                Samplers[name] = new SamplerInfo(name);
            }
            
            // Convert to microseconds for storage (same as AddSample)
            Samplers[name].AddSample((long)(milliseconds * 1000));
            
            // Update average times
            AverageTimes[name] = Samplers[name].AverageTimeMs / 1000.0;
            
            // Special case for backward compatibility
            if (name == "BuildQuadtreeUnified")
            {
                _lastManualTiming = milliseconds;
            }
        }
        
        // Legacy method for backward compatibility
        public static void RecordManualTiming(double milliseconds)
        {
            RecordManualTiming("Manual", milliseconds);
        }
        
        public static void BeginSample(string name)
        {
            if (!_profilerEnabled) return;
            
            if (!Samplers.TryGetValue(name, out var samplerInfo))
            {
                samplerInfo = new SamplerInfo(name);
                Samplers[name] = samplerInfo;
            }
            
            samplerInfo.Sampler.Begin();
            samplerInfo.Stopwatch.Reset();
            samplerInfo.Stopwatch.Start();
            
            // Ensure this is registered in AverageTimes even if no samples yet
            if (!AverageTimes.ContainsKey(name))
            {
                AverageTimes[name] = 0;
            }
        }
        
        public static void EndSample(string name)
        {
            if (!_profilerEnabled || !Samplers.TryGetValue(name, out var samplerInfo)) return;
            
            samplerInfo.Stopwatch.Stop();
            samplerInfo.Sampler.End();
            
            // Record sample time in milliseconds - convert ticks to ms for more precision
            double elapsedMs = samplerInfo.Stopwatch.ElapsedTicks / (double)System.Diagnostics.Stopwatch.Frequency * 1000.0;
            
            // Add a very small value if elapsed is zero to show that something happened
            if (elapsedMs < 0.0001)
            {
                elapsedMs = 0.0001;
            }
            
            // Capture any other timing methods (fudge factor of 2ms for GL.Flush overhead)
            if (name == "BuildQuadtreeUnified" && _lastManualTiming > 0)
            {
                elapsedMs = Math.Max(0.01, _lastManualTiming - 2.0);
                _lastManualTiming = 0;
            }
            
            samplerInfo.AddSample((long)(elapsedMs * 1000)); // Store in microseconds for better precision
            
            // Store in average times dictionary for easy access
            AverageTimes[name] = samplerInfo.AverageTimeMs / 1000.0; // Convert back to milliseconds
            
            // Debug log to see if we're getting any values at all
            if (_frameCount % 100 == 0 && name == "BuildQuadtreeUnified")
            {
                Debug.Log($"[Profiler Debug] {name}: {elapsedMs:F6} ms, Avg: {AverageTimes[name]:F6} ms");
            }
        }
        
        public static void BeginFrame()
        {
            if (!_profilerEnabled) return;
            
            _frameCount++;
            
            // Update the display text every 5 frames to reduce overhead but keep it responsive
            if (_frameCount % 5 == 0)
            {
                UpdateProfilingText();
            }
        }
        
        public static void EndFrame()
        {
            // This method is intentionally left empty for now
            // Could be used for frame-end operations if needed
        }
        
        public static double GetAverageTime(string name)
        {
            if (AverageTimes.TryGetValue(name, out var time))
            {
                return time;
            }
            return 0;
        }
        
        public static void LogTimings()
        {
            if (!_profilerEnabled) return;
            
            List<string> samplerNames = new List<string>(AverageTimes.Keys);
            samplerNames.Sort((a, b) => AverageTimes[b].CompareTo(AverageTimes[a])); // Sort by time descending
            
            Debug.Log("===== COMPUTE SHADER PROFILING =====");
            foreach (var name in samplerNames)
            {
                Debug.Log($"{name}: {AverageTimes[name]:F3} ms");
            }
        }
        
        private static void UpdateProfilingText()
        {
            List<string> samplerNames = new List<string>(AverageTimes.Keys);
            if (samplerNames.Count == 0)
            {
                _profilingText = "No profiling data yet. Make sure profiling is enabled (P).\nWaiting for kernel dispatches...";
                return;
            }
            
            // Sort by time descending
            samplerNames.Sort((a, b) => AverageTimes[b].CompareTo(AverageTimes[a])); 
            
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            sb.AppendLine("===== BOIDS COMPUTE SHADER PROFILING =====");
            sb.AppendLine($"Controls: P=Toggle | V=Show/Hide | L=Log");
            sb.AppendLine($"Frame: {_frameCount}");
            sb.AppendLine();
            
            float totalTime = 0;
            bool hasNonZeroValues = false;
            
            // Group timings by category for better organization
            Dictionary<string, List<KeyValuePair<string, double>>> categories = new Dictionary<string, List<KeyValuePair<string, double>>>();
            
            // Categorize all measurements
            foreach (var pair in AverageTimes)
            {
                string name = pair.Key;
                double timeMs = pair.Value;
                string category = "Other";
                
                // Define categories based on name patterns
                if (name.Contains("Render") || name.Contains("RenderBoids"))
                    category = "Rendering";
                else if (name.Contains("FullFrame") || name.Contains("ActualFrameTime"))
                    category = "Frame";
                else if (name.Contains("UpdateBoid") || name.Contains("Boid"))
                    category = "Boid Simulation";
                else if (name.Contains("Quad") || name.Contains("Tree"))
                    category = "Quadtree";
                else if (name.Contains("Grid"))
                    category = "Grid";
                
                if (!categories.ContainsKey(category))
                    categories[category] = new List<KeyValuePair<string, double>>();
                
                categories[category].Add(new KeyValuePair<string, double>(name, timeMs));
                
                totalTime += (float)timeMs;
                
                if (timeMs > 0.001)
                    hasNonZeroValues = true;
            }
            
            // Define the order of categories
            string[] categoryOrder = new string[] { "Frame", "Boid Simulation", "Quadtree", "Grid", "Rendering", "Other" };
            
            // Render each category in order
            foreach (string category in categoryOrder)
            {
                if (!categories.ContainsKey(category) || categories[category].Count == 0)
                    continue;
                
                sb.AppendLine($"<color=#BBBBFF><b>{category}</b></color>");
                
                // Sort by time descending within category
                categories[category].Sort((a, b) => b.Value.CompareTo(a.Value));
                
                foreach (var pair in categories[category])
                {
                    string name = pair.Key;
                    double timeMs = pair.Value;
                    
                    // Even if timing is zero, show a small value
                    if (timeMs < 0.001)
                        timeMs = 0.001;
                    
                    // Format text color based on time - more granular display for better analysis
                    string timeText;
                    if (timeMs < 0.1)
                        timeText = $"{timeMs:F3} ms";
                    else if (timeMs < 1.0)
                        timeText = $"<color=#AAFFAA>{timeMs:F3} ms</color>";
                    else if (timeMs < 5.0)
                        timeText = $"<color=yellow>{timeMs:F2} ms</color>";
                    else if (timeMs < 10.0)
                        timeText = $"<color=orange>{timeMs:F2} ms</color>";
                    else
                        timeText = $"<color=red><b>{timeMs:F2} ms</b></color>";
                    
                    // Display bottleneck warning for long operations
                    if (timeMs > 8.0 && (name.Contains("GPU") || name.Contains("Frame")))
                        timeText += " ⚠️";
                    
                    sb.AppendLine($"  {name}: {timeText}");
                }
                
                sb.AppendLine();
            }
            
            sb.AppendLine();
            sb.AppendLine($"Total profiled time: {totalTime:F2} ms");
            sb.AppendLine($"Approx. frame budget: {1000f/60f:F2} ms (60 FPS)");
            
            if (!hasNonZeroValues)
            {
                sb.AppendLine();
                sb.AppendLine("<color=yellow>Waiting for non-zero values...</color>");
                sb.AppendLine("Try enabling quadtrees in the UI");
            }
            
            _profilingText = sb.ToString();
        }
        
        private static GUIStyle _boxStyle;
        private static GUIStyle _textStyle;
        private static Texture2D _backgroundTexture;
        
        public static void OnGUI()
        {
            if (!_profilerEnabled || !_showOnScreen) return;
            
            // Initialize styles once
            if (_textStyle == null)
            {
                _textStyle = new GUIStyle(GUI.skin.label);
                _textStyle.normal.textColor = Color.white;
                _textStyle.fontSize = 14;
                _textStyle.wordWrap = true;
                _textStyle.richText = true; // Enable colored text
                
                _boxStyle = new GUIStyle(GUI.skin.box);
                
                // Create a semi-transparent black background
                _backgroundTexture = new Texture2D(1, 1);
                _backgroundTexture.SetPixel(0, 0, new Color(0, 0, 0, 0.8f));
                _backgroundTexture.Apply();
                
                _boxStyle.normal.background = _backgroundTexture;
            }
            
            int lineCount = _profilingText.Split('\n').Length;
            int height = Mathf.Max(100, 20 * lineCount);
            int width = 350;
            
            // Draw with an offset from the edge of the screen
            GUI.Box(new Rect(10, 10, width, height), "", _boxStyle);
            GUI.Label(new Rect(20, 15, width - 20, height - 10), _profilingText, _textStyle);
        }
    }
}