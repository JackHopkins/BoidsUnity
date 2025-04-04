using UnityEngine;
using UnityEngine.UI;

namespace BoidsUnity
{
    public class UIManager : MonoBehaviour
    {
        [SerializeField] private Text fpsText;
        [SerializeField] private Text boidText;
        [SerializeField] private Slider numSlider;
        [SerializeField] private Button modeButton;
        
        [Header("Grid Visualization")]
        [SerializeField] private Button toggleGridsButton;
        [SerializeField] private Button toggleRegularGridButton;
        [SerializeField] private Button toggleOfficerGridButton;
        
        private bool useGpu = false;
        private int numBoids = 500;
        private readonly int cpuLimit = 1 << 16;
        private readonly int gpuLimit = 256 * 65535;
        
        // Reference to the main simulation controller
        private Main2D mainController;
        
        public void Initialize(Main2D controller, int initialBoidCount, bool initialUseGpu)
        {
            mainController = controller;
            numBoids = initialBoidCount;
            useGpu = initialUseGpu;
            
            // Set up the UI elements
            boidText.text = "Boids: " + numBoids;
            numSlider.maxValue = Mathf.Log(useGpu ? gpuLimit : cpuLimit, 2);
            
            // Set button color and text based on mode
            UpdateModeButton();
            
            // Set up grid visualization button listeners
            if (toggleGridsButton != null)
            {
                toggleGridsButton.onClick.AddListener(ToggleGrids);
            }
            
            if (toggleRegularGridButton != null)
            {
                toggleRegularGridButton.onClick.AddListener(ToggleRegularGrid);
            }
            
            if (toggleOfficerGridButton != null)
            {
                toggleOfficerGridButton.onClick.AddListener(ToggleOfficerGrid);
            }
        }
        
        public void UpdateFPS()
        {
            fpsText.text = "FPS: " + (int)(1 / Time.smoothDeltaTime);
        }
        
        public void SliderChange(float val)
        {
            numBoids = (int)Mathf.Pow(2, val);
            var limit = useGpu ? gpuLimit : cpuLimit;
            if (numBoids > limit)
            {
                numBoids = limit;
            }
            boidText.text = "Boids: " + numBoids;
            
            // Notify the main controller about the boid count change
            mainController.RestartSimulation(numBoids);
        }
        
        public void ModeChange()
        {
            useGpu = !useGpu;
            UpdateModeButton();
            numSlider.maxValue = Mathf.Log(useGpu ? gpuLimit : cpuLimit, 2);
            
            // Notify the main controller about the GPU mode change
            mainController.SetGpuMode(useGpu);
        }
        
        private void UpdateModeButton()
        {
            modeButton.image.color = useGpu ? Color.green : Color.red;
            modeButton.GetComponentInChildren<Text>().text = useGpu ? "GPU" : "CPU";
        }
        
        public void SwitchTo3D()
        {
            UnityEngine.SceneManagement.SceneManager.LoadScene("Boids3DScene");
        }
        
        // Grid visualization methods
        public void ToggleGrids()
        {
            if (mainController != null)
            {
                mainController.ToggleGridVisualization();
            }
        }
        
        public void ToggleRegularGrid()
        {
            if (mainController != null)
            {
                mainController.ToggleRegularGrid();
            }
        }
        
        public void ToggleOfficerGrid()
        {
            if (mainController != null)
            {
                mainController.ToggleOfficerGrid();
            }
        }
        
        private void OnDestroy()
        {
            // Clean up button listeners
            if (toggleGridsButton != null)
            {
                toggleGridsButton.onClick.RemoveListener(ToggleGrids);
            }
            
            if (toggleRegularGridButton != null)
            {
                toggleRegularGridButton.onClick.RemoveListener(ToggleRegularGrid);
            }
            
            if (toggleOfficerGridButton != null)
            {
                toggleOfficerGridButton.onClick.RemoveListener(ToggleOfficerGrid);
            }
        }
    }
}
