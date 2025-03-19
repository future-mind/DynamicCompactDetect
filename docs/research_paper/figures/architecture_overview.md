# Architecture Overview Diagram

Since we cannot generate a diagram directly, this file describes what the architecture overview diagram should contain:

The DynamicCompactDetect architecture consists of:

1. **Input**: 640×640×3 RGB image
2. **Backbone (DarknetCSP)**:
   - Stem: Initial 3×3 Conv (stride 2)
   - Four stages (Dark1-Dark4) with CSP blocks
   - Each stage doubles channels and halves resolution
   - Final stage includes SPPF module
3. **Neck (PAN)**:
   - Bidirectional feature fusion
   - Top-down path (from deeper to shallower layers)
   - Bottom-up path (from shallower to deeper layers)
   - CSP blocks at each fusion point
4. **Head (Decoupled)**:
   - Separate classification and regression branches
   - Three detection scales
   - Grid-based detection outputs

Key innovation: RepConv modules are used throughout the network, providing mathematical equivalence between training and inference models but with reduced computation during inference.

Note: This placeholder should be replaced with an actual diagram in SVG or PNG format for the final documentation. 