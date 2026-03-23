"""
Real NIfTI Data BrainBrowser - Actually loads and displays your real data
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64
import nibabel as nib
import numpy as np
import json

def create_real_nifti_viewer(nifti_path: Path, title: str, key: str, color: bool = False, height: int = 820):
    """
    Create a viewer that actually loads and displays real NIfTI data
    """
    
    try:
        # Load the actual NIfTI data
        img = nib.load(str(nifti_path))
        data = img.get_fdata().astype(np.float32)
        
        # Get image info
        shape = data.shape
        voxel_dims = img.header.get_zooms()[:3]
        data_min, data_max = float(data.min()), float(data.max())
        
        # Normalize data to 0-255 for display
        if data_max > data_min:
            data_normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            data_normalized = np.zeros_like(data, dtype=np.uint8)
        
        # Create unique viewer ID
        viewer_id = f"real_{key}_{hash(str(nifti_path)) % 10000}"
        
        # Convert slices to base64 for embedding
        # Sample every few slices to reduce data size
        step = max(1, shape[2] // 50)
        sample_slices = {}
        
        # Axial slices (Z direction)
        axial_slices = []
        for z in range(0, shape[2], step):
            slice_data = data_normalized[:, :, z]
            axial_slices.append(slice_data.tolist())
        
        # Sagittal slices (X direction) 
        sagittal_slices = []
        step_x = max(1, shape[0] // 50)
        for x in range(0, shape[0], step_x):
            slice_data = data_normalized[x, :, :]
            sagittal_slices.append(slice_data.tolist())
            
        # Coronal slices (Y direction)
        coronal_slices = []
        step_y = max(1, shape[1] // 50)
        for y in range(0, shape[1], step_y):
            slice_data = data_normalized[:, y, :]
            coronal_slices.append(slice_data.tolist())
        
        # Create JSON data for JavaScript (ensure all values are JSON serializable)
        volume_data = {
            'axial': axial_slices,
            'sagittal': sagittal_slices, 
            'coronal': coronal_slices,
            'shape': [int(shape[0]), int(shape[1]), int(shape[2])],
            'voxelSize': [float(voxel_dims[0]), float(voxel_dims[1]), float(voxel_dims[2])],
            'dataRange': [float(data_min), float(data_max)],
            'steps': [int(step), int(step_x), int(step_y)]
        }
        
        volume_json = json.dumps(volume_data)
        
    except Exception as e:
        st.error(f"Error loading real NIfTI data: {e}")
        return

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                background: #f8f9fa;
                overflow-y: auto;
            }}
            
            .viewer-container {{
                background: white;
                border-radius: 12px;
                padding: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 24px;
                padding: 20px;
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                border-radius: 8px;
            }}
            
            .header h1 {{
                margin: 0 0 8px 0;
                font-size: 28px;
                font-weight: 700;
            }}
            
            .volume-info {{
                background: #dbeafe;
                border: 1px solid #93c5fd;
                padding: 16px;
                border-radius: 8px;
                margin-bottom: 24px;
                font-size: 14px;
                color: #1e40af;
            }}
            
            .viewer-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 16px;
                margin-bottom: 16px;
            }}

            .view-block {{
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}

            .inline-control {{
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 4px 0;
            }}

            .inline-control input[type=range] {{
                flex: 1;
                cursor: pointer;
            }}

            .inline-control input[type=number] {{
                width: 52px;
                padding: 3px 6px;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                font-size: 13px;
                text-align: center;
            }}

            .slice-max {{
                font-size: 12px;
                color: #6b7280;
                white-space: nowrap;
            }}
            
            .view-panel {{
                background: #000;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                position: relative;
                aspect-ratio: 1;
                overflow: hidden;
            }}
            
            .view-label {{
                position: absolute;
                top: 12px;
                left: 16px;
                color: #00ff00;
                font-weight: bold;
                font-size: 16px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
                z-index: 10;
            }}
            
            .view-canvas {{
                width: 100%;
                height: 100%;
                cursor: crosshair;
                image-rendering: pixelated;
            }}
            
            .controls {{
                background: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 20px;
            }}
            
            .control-section {{
                margin-bottom: 20px;
                padding-bottom: 16px;
                border-bottom: 1px solid #e5e7eb;
            }}
            
            .control-section:last-child {{
                border-bottom: none;
                margin-bottom: 0;
            }}
            
            .control-section h3 {{
                margin: 0 0 12px 0;
                font-size: 16px;
                font-weight: 600;
                color: #374151;
            }}
            
            .control-row {{
                display: flex;
                align-items: center;
                margin-bottom: 12px;
                gap: 12px;
                flex-wrap: wrap;
            }}
            
            .control-label {{
                font-weight: 500;
                min-width: 100px;
                color: #4b5563;
                font-size: 14px;
            }}
            
            .control-input {{
                flex: 1;
                min-width: 200px;
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                font-size: 14px;
                background: white;
            }}
            
            .control-button {{
                background: #4f46e5;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s;
            }}
            
            .control-button:hover {{
                background: #4338ca;
                transform: translateY(-1px);
            }}
            
            .coordinates-display {{
                background: white;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 12px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 13px;
                color: #374151;
                min-height: 80px;
                line-height: 1.5;
            }}
            
            .status {{
                padding: 12px 16px;
                border-radius: 6px;
                margin-bottom: 20px;
                font-size: 14px;
                font-weight: 500;
            }}
            
            .status-success {{
                background: #dcfce7;
                color: #166534;
                border: 1px solid #bbf7d0;
            }}
            
            .status-error {{
                background: #fef2f2;
                color: #dc2626;
                border: 1px solid #fecaca;
            }}

            details.info-panel {{
                margin-top: 20px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                overflow: hidden;
            }}

            details.info-panel summary {{
                background: #f3f4f6;
                padding: 12px 16px;
                cursor: pointer;
                font-weight: 600;
                font-size: 14px;
                color: #374151;
                user-select: none;
                list-style: none;
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            details.info-panel summary::-webkit-details-marker {{ display: none; }}

            details.info-panel summary::before {{
                content: '▶';
                font-size: 11px;
                transition: transform 0.2s;
            }}

            details.info-panel[open] summary::before {{
                transform: rotate(90deg);
            }}

            details.info-panel .panel-body {{
                padding: 16px;
            }}
        </style>
    </head>
    <body>
        <div class="viewer-container">

            <!-- ── Three views, each with its own slice slider ── -->
            <div class="viewer-grid">

                <div class="view-block">
                    <div class="view-panel">
                        <div class="view-label">Axial (Z)</div>
                        <canvas id="axial-{viewer_id}" class="view-canvas"></canvas>
                    </div>
                    <div class="inline-control">
                        <input type="range" id="slice-z-{viewer_id}"
                               min="0" max="{shape[2]-1}" value="{shape[2]//2}"
                               oninput="updateSlice_{viewer_id}('z', this.value)">
                        <input type="number" id="slice-z-num-{viewer_id}"
                               min="0" max="{shape[2]-1}" value="{shape[2]//2}"
                               onchange="updateSlice_{viewer_id}('z', this.value)">
                        <span class="slice-max">/ {shape[2]-1}</span>
                    </div>
                </div>

                <div class="view-block">
                    <div class="view-panel">
                        <div class="view-label">Sagittal (X)</div>
                        <canvas id="sagittal-{viewer_id}" class="view-canvas"></canvas>
                    </div>
                    <div class="inline-control">
                        <input type="range" id="slice-x-{viewer_id}"
                               min="0" max="{shape[0]-1}" value="{shape[0]//2}"
                               oninput="updateSlice_{viewer_id}('x', this.value)">
                        <input type="number" id="slice-x-num-{viewer_id}"
                               min="0" max="{shape[0]-1}" value="{shape[0]//2}"
                               onchange="updateSlice_{viewer_id}('x', this.value)">
                        <span class="slice-max">/ {shape[0]-1}</span>
                    </div>
                </div>

                <div class="view-block">
                    <div class="view-panel">
                        <div class="view-label">Coronal (Y)</div>
                        <canvas id="coronal-{viewer_id}" class="view-canvas"></canvas>
                    </div>
                    <div class="inline-control">
                        <input type="range" id="slice-y-{viewer_id}"
                               min="0" max="{shape[1]-1}" value="{shape[1]//2}"
                               oninput="updateSlice_{viewer_id}('y', this.value)">
                        <input type="number" id="slice-y-num-{viewer_id}"
                               min="0" max="{shape[1]-1}" value="{shape[1]//2}"
                               onchange="updateSlice_{viewer_id}('y', this.value)">
                        <span class="slice-max">/ {shape[1]-1}</span>
                    </div>
                </div>

            </div><!-- end viewer-grid -->

        </div>

        <script>
            // Real volume data loaded from your NIfTI file
            const volumeData_{viewer_id} = {volume_json};
            
            // Current slice positions
            let currentSlices_{viewer_id} = {{
                x: Math.floor(volumeData_{viewer_id}.shape[0] / 2),
                y: Math.floor(volumeData_{viewer_id}.shape[1] / 2),
                z: Math.floor(volumeData_{viewer_id}.shape[2] / 2)
            }};
            
            function interpolateSlice_{viewer_id}(slices, targetIndex, maxIndex) {{
                // Interpolate between available slices
                const step = slices.length > 1 ? maxIndex / (slices.length - 1) : 1;
                const sliceIndex = Math.min(Math.floor(targetIndex / step), slices.length - 1);
                return slices[sliceIndex];
            }}
            
            function drawSlice_{viewer_id}(canvasId, axis, sliceIndex) {{
                const canvas = document.getElementById(canvasId);
                if (!canvas) return;
                
                const ctx = canvas.getContext('2d');
                const data = volumeData_{viewer_id};
                
                let sliceData;
                let width, height;
                
                if (axis === 'z') {{
                    sliceData = interpolateSlice_{viewer_id}(data.axial, sliceIndex, data.shape[2] - 1);
                    // Rotate 90° clockwise: swap width/height
                    width = data.shape[1];
                    height = data.shape[0];
                }} else if (axis === 'x') {{
                    sliceData = interpolateSlice_{viewer_id}(data.sagittal, sliceIndex, data.shape[0] - 1);
                    width = data.shape[1];
                    height = data.shape[2];
                }} else if (axis === 'y') {{
                    sliceData = interpolateSlice_{viewer_id}(data.coronal, sliceIndex, data.shape[1] - 1);
                    width = data.shape[0];
                    height = data.shape[2];
                }}
                
                if (!sliceData) return;
                
                // Set canvas size
                canvas.width = width;
                canvas.height = height;
                
                // Create image data
                const imageData = ctx.createImageData(width, height);
                
                // Fixed display parameters (no controls)
                const contrast = 1.0;
                const brightness = 0;
                
                // Render the slice
                const origW = (axis === 'z') ? data.shape[0] : width;
                for (let y = 0; y < height; y++) {{
                    for (let x = 0; x < width; x++) {{
                        // Axial (Z): rotate 90° clockwise — srcRow=H-1-x, srcCol=y (H=shape[1])
                        let value = (axis === 'z')
                            ? sliceData[data.shape[1] - 1 - x][y]
                            : sliceData[y][x];
                        
                        // Apply contrast and brightness
                        value = value * contrast + brightness;
                        value = Math.max(0, Math.min(255, value));
                        
                        const pixelIdx = (y * width + x) * 4;
                        
                        if ({str(color).lower()}) {{
                            // Color mapping for segmentation
                            const [r, g, b] = valueToColor_{viewer_id}(value);
                            imageData.data[pixelIdx] = r;
                            imageData.data[pixelIdx + 1] = g;
                            imageData.data[pixelIdx + 2] = b;
                        }} else {{
                            // Grayscale
                            imageData.data[pixelIdx] = value;
                            imageData.data[pixelIdx + 1] = value;
                            imageData.data[pixelIdx + 2] = value;
                        }}
                        imageData.data[pixelIdx + 3] = 255;
                    }}
                }}
                
                ctx.putImageData(imageData, 0, 0);
            }}
            
            function valueToColor_{viewer_id}(value) {{
                // Simple colormap for segmentation
                const normalizedValue = value / 255;
                const hue = normalizedValue * 300; // 0 to 300 degrees
                
                const c = 0.8; // chroma
                const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
                const m = 0.3; // lightness adjustment
                
                let r, g, b;
                if (hue < 60) {{ r = c; g = x; b = 0; }}
                else if (hue < 120) {{ r = x; g = c; b = 0; }}
                else if (hue < 180) {{ r = 0; g = c; b = x; }}
                else if (hue < 240) {{ r = 0; g = x; b = c; }}
                else if (hue < 300) {{ r = x; g = 0; b = c; }}
                else {{ r = c; g = 0; b = x; }}
                
                return [
                    Math.round((r + m) * 255),
                    Math.round((g + m) * 255),
                    Math.round((b + m) * 255)
                ];
            }}
            
            function syncControls_{viewer_id}(axis, sliceIndex) {{
                // Sync range slider
                const range = document.getElementById(`slice-${{axis}}-{viewer_id}`);
                if (range) range.value = sliceIndex;
                // Sync number input
                const num = document.getElementById(`slice-${{axis}}-num-{viewer_id}`);
                if (num) num.value = sliceIndex;
            }}

            function updateSlice_{viewer_id}(axis, value) {{
                const sliceIndex = Math.max(0, parseInt(value) || 0);
                currentSlices_{viewer_id}[axis] = sliceIndex;
                syncControls_{viewer_id}(axis, sliceIndex);
                // Redraw only the affected view
                if (axis === 'z') drawSlice_{viewer_id}('axial-{viewer_id}', 'z', sliceIndex);
                else if (axis === 'x') drawSlice_{viewer_id}('sagittal-{viewer_id}', 'x', sliceIndex);
                else if (axis === 'y') drawSlice_{viewer_id}('coronal-{viewer_id}', 'y', sliceIndex);
            }}
            
            function updateDisplay_{viewer_id}() {{
                // Redraw all views
                drawSlice_{viewer_id}('axial-{viewer_id}', 'z', currentSlices_{viewer_id}.z);
                drawSlice_{viewer_id}('sagittal-{viewer_id}', 'x', currentSlices_{viewer_id}.x);
                drawSlice_{viewer_id}('coronal-{viewer_id}', 'y', currentSlices_{viewer_id}.y);
            }}
            
            function setupClickHandlers_{viewer_id}() {{
                const views = [
                    {{canvas: 'axial-{viewer_id}', axis: 'z'}},
                    {{canvas: 'sagittal-{viewer_id}', axis: 'x'}},
                    {{canvas: 'coronal-{viewer_id}', axis: 'y'}}
                ];
                
                views.forEach(view => {{
                    const canvas = document.getElementById(view.canvas);
                    if (!canvas) return;

                    canvas.addEventListener('click', function(e) {{
                        const rect = canvas.getBoundingClientRect();
                        const scaleX = canvas.width / rect.width;
                        const scaleY = canvas.height / rect.height;

                        const px = Math.floor((e.clientX - rect.left) * scaleX);
                        const py = Math.floor((e.clientY - rect.top) * scaleY);

                        let voxelX, voxelY, voxelZ;

                        if (view.axis === 'z') {{
                            // Axial clicked: px=X, py=Y → update sagittal(X) and coronal(Y)
                            voxelX = Math.min(px, volumeData_{viewer_id}.shape[0] - 1);
                            voxelY = Math.min(py, volumeData_{viewer_id}.shape[1] - 1);
                            voxelZ = currentSlices_{viewer_id}.z;
                            updateSlice_{viewer_id}('x', voxelX);
                            updateSlice_{viewer_id}('y', voxelY);
                        }} else if (view.axis === 'x') {{
                            // Sagittal clicked: px=Y, py=Z → update coronal(Y) and axial(Z)
                            voxelX = currentSlices_{viewer_id}.x;
                            voxelY = Math.min(px, volumeData_{viewer_id}.shape[1] - 1);
                            voxelZ = Math.min(py, volumeData_{viewer_id}.shape[2] - 1);
                            updateSlice_{viewer_id}('y', voxelY);
                            updateSlice_{viewer_id}('z', voxelZ);
                        }} else {{
                            // Coronal clicked: px=X, py=Z → update sagittal(X) and axial(Z)
                            voxelX = Math.min(px, volumeData_{viewer_id}.shape[0] - 1);
                            voxelY = currentSlices_{viewer_id}.y;
                            voxelZ = Math.min(py, volumeData_{viewer_id}.shape[2] - 1);
                            updateSlice_{viewer_id}('x', voxelX);
                            updateSlice_{viewer_id}('z', voxelZ);
                        }}

                    }});
                }});
            }}
            
            function getPixelValue_{viewer_id}(canvasId, x, y) {{
                const canvas = document.getElementById(canvasId);
                if (!canvas) return 0;
                
                const ctx = canvas.getContext('2d');
                const imageData = ctx.getImageData(x, y, 1, 1);
                return imageData.data[0]; // Red channel (grayscale)
            }}
            
            // Initialize the viewer
            function initializeViewer_{viewer_id}() {{
                setupClickHandlers_{viewer_id}();
                updateDisplay_{viewer_id}();
            }}
            
            // Start when page loads
            window.addEventListener('load', function() {{
                setTimeout(initializeViewer_{viewer_id}, 100);
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(html_content, height=height)

def real_nifti_viewer(nifti_path: Path, title: str, key: str, color: bool = False):
    """
    Creates a viewer that displays real NIfTI data
    """
    create_real_nifti_viewer(nifti_path, title, f"real_{key}", color, height=820)