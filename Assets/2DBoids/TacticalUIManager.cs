using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using Unity.Mathematics;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct BattalionData
{
    public int id;
    public int startIndex;
    public int count;
    public int padding1;
    public float targetPosX;
    public float targetPosY;
    public float formationSizeX;
    public float formationSizeY;
    public int formationType;
    public int padding2;
}

public class TacticalUIManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private TacticalBoids2D boidManager;
    [SerializeField] private BattalionManager battalionManager;
    
    [Header("UI Elements")]
    [SerializeField] private Button squareFormationButton;
    [SerializeField] private Button lineFormationButton;
    [SerializeField] private Button columnFormationButton;
    [SerializeField] private Button wedgeFormationButton;
    [SerializeField] private Button scatteredFormationButton;
    
    [SerializeField] private Text selectedBattalionsText;
    [SerializeField] private Text troopCountText;
    
    [Header("Command Panel")]
    [SerializeField] private GameObject commandPanel;
    [SerializeField] private Button attackButton;
    [SerializeField] private Button holdPositionButton;
    [SerializeField] private Button retreatButton;
    
    // Colors for different button states
    private Color activeColor = new Color(0.2f, 0.8f, 0.2f);
    private Color inactiveColor = new Color(0.7f, 0.7f, 0.7f);

    private ComputeBuffer battalionBuffer;
    
    void Start()
    {
        // Ensure we have references to necessary components
        if (boidManager == null)
            boidManager = FindObjectOfType<TacticalBoids2D>();
            
        if (battalionManager == null)
        {
            battalionManager = GetComponent<BattalionManager>();
            if (battalionManager == null)
            {
                battalionManager = gameObject.AddComponent<BattalionManager>();
            }
        }
        
        // Set up button click events
        if (squareFormationButton != null)
            squareFormationButton.onClick.AddListener(() => SetFormation(FormationType.Square));
            
        if (lineFormationButton != null)
            lineFormationButton.onClick.AddListener(() => SetFormation(FormationType.Line));
            
        if (columnFormationButton != null)
            columnFormationButton.onClick.AddListener(() => SetFormation(FormationType.Column));
            
        if (wedgeFormationButton != null)
            wedgeFormationButton.onClick.AddListener(() => SetFormation(FormationType.Wedge));
            
        if (scatteredFormationButton != null)
            scatteredFormationButton.onClick.AddListener(() => SetFormation(FormationType.Scattered));
            
        // Set up command buttons
        if (attackButton != null)
            attackButton.onClick.AddListener(CommandAttack);
            
        if (holdPositionButton != null)
            holdPositionButton.onClick.AddListener(CommandHoldPosition);
            
        if (retreatButton != null)
            retreatButton.onClick.AddListener(CommandRetreat);
            
        // Initialize command panel to inactive
        if (commandPanel != null)
            commandPanel.SetActive(false);
        
        // battalionManager.InitializeBattalions(boidManager.numBoids);
        //
        // // Create battalion buffer with correct size
        // battalionBuffer = new ComputeBuffer(
        //     Mathf.Max(1, battalionManager.battalions.Count), // Ensure at least size 1
        //     Marshal.SizeOf<BattalionData>()
        // );
        //
        // // Update the buffer
        // UpdateBattalionBuffer();
    }
    
    void Update()
    {
        // Update UI based on selection state
        if (battalionManager != null)
        {
            UpdateSelectionInfo();
            
            // Show/hide command panel based on selection
            if (commandPanel != null)
                commandPanel.SetActive(battalionManager.selectedBattalions.Count > 0);
        }
    }
    
    // Update the selection info text
    void UpdateSelectionInfo()
    {
        if (selectedBattalionsText != null)
        {
            selectedBattalionsText.text = $"Selected: {battalionManager.selectedBattalions.Count} battalion(s)";
        }
        
        if (troopCountText != null)
        {
            int totalTroops = 0;
            foreach (Battalion battalion in battalionManager.selectedBattalions)
            {
                totalTroops += battalion.count;
            }
            
            troopCountText.text = $"Troops: {totalTroops}";
        }
        
        // Update formation button visuals based on current formation
        UpdateFormationButtonStates();
    }
    
    // Update the visual state of formation buttons based on current selection
    void UpdateFormationButtonStates()
    {
        // Reset all buttons to inactive
        SetButtonState(squareFormationButton, false);
        SetButtonState(lineFormationButton, false);
        SetButtonState(columnFormationButton, false);
        SetButtonState(wedgeFormationButton, false);
        SetButtonState(scatteredFormationButton, false);
        
        // If we have a selection, highlight the current formation
        if (battalionManager.selectedBattalions.Count > 0)
        {
            // Check if all battalions have the same formation
            bool sameFormation = true;
            FormationType formation = battalionManager.selectedBattalions[0].formationType;
            
            foreach (Battalion battalion in battalionManager.selectedBattalions)
            {
                if (battalion.formationType != formation)
                {
                    sameFormation = false;
                    break;
                }
            }
            
            // If all have the same formation, highlight that button
            if (sameFormation)
            {
                switch (formation)
                {
                    case FormationType.Square:
                        SetButtonState(squareFormationButton, true);
                        break;
                    case FormationType.Line:
                        SetButtonState(lineFormationButton, true);
                        break;
                    case FormationType.Column:
                        SetButtonState(columnFormationButton, true);
                        break;
                    case FormationType.Wedge:
                        SetButtonState(wedgeFormationButton, true);
                        break;
                    case FormationType.Scattered:
                        SetButtonState(scatteredFormationButton, true);
                        break;
                }
            }
        }
    }
    
    // Set the visual state of a button (active/inactive)
    void SetButtonState(Button button, bool active)
    {
        if (button != null)
        {
            ColorBlock colors = button.colors;
            colors.normalColor = active ? activeColor : inactiveColor;
            button.colors = colors;
        }
    }
    
    // Command to set formation for selected battalions
    public void SetFormation(FormationType formation)
    {
        if (battalionManager != null)
        {
            battalionManager.SetFormation(formation);
            UpdateFormationButtonStates();
        }
    }
    
    // Command to attack (move toward a target)
    public void CommandAttack()
    {
        // In a real game, this would need to select a target
        // For now, we'll just put battalions in an attack formation (wedge)
        if (battalionManager != null)
        {
            battalionManager.SetFormation(FormationType.Wedge);
            UpdateFormationButtonStates();
        }
    }
    
    // Command to hold position 
    public void CommandHoldPosition()
    {
        // Hold position in a defensive formation (square)
        if (battalionManager != null)
        {
            battalionManager.SetFormation(FormationType.Square);
            UpdateFormationButtonStates();
        }
    }
    
    // Command to retreat
    public void CommandRetreat()
    {
        // Retreat in a column formation for speed
        if (battalionManager != null)
        {
            battalionManager.SetFormation(FormationType.Column);
            UpdateFormationButtonStates();
            
            // In a real game, we would also set a retreat point behind the current line
        }
    }
}