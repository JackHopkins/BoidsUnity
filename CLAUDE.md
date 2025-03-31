# CLAUDE.md - Development Guidelines

## Build/Run Commands
- Run in Unity Editor: Open project and press Play button
- Build for WebGL: File > Build Settings > Select WebGL > Build
- Build for desktop: File > Build Settings > Select Windows/Mac/Linux > Build

## Testing
- No formal test framework in place
- Manual testing via Unity Play mode
- Performance testing via in-game FPS counter

## Code Style Guidelines
- Import order: Unity namespaces first, then System namespaces
- Use Unity.Mathematics (float2, float3) for vector operations
- ComputeBuffer naming: end with "Buffer" (e.g., boidBuffer)
- Struct fields: Use the smallest appropriate data type with padding for alignment
- Kernel naming: Descriptive verb-noun format (e.g., UpdateBoids, ClearGrid)
- Use SerializeField attributes for inspector-exposed fields
- Header attributes to organize inspector sections
- Clean up resources in OnDestroy() with SafeReleaseBuffer helper function
- Keep boid logic modular with separate methods for behaviors
- Use const/readonly for magic numbers
- Consistent bracing style with opening brace on same line

## Error Handling
- Perform null checks before accessing ComputeBuffers
- Use helper methods like SafeReleaseBuffer to prevent errors
- Check buffer allocations and indices to avoid out-of-bounds access