# Overlap Threshold Feature

## Overview

The overlap threshold feature prevents objects in synthetic datasets from significantly overlapping with each other or extending outside image boundaries. This improves dataset quality by ensuring all objects are clearly visible and properly contained.

## Problem Solved

**Before:**
- Objects could overlap completely, hiding one behind another
- Objects could extend significantly outside image boundaries
- Used simple margin-based approach that was inflexible

**After:**
- Objects maintain configurable minimum separation
- Objects stay within image boundaries (with configurable tolerance)
- Uses IoU-based overlap detection for precise control

## How It Works

### Algorithm

When placing each object in a synthetic image:

1. **Generate** random position and size
2. **Calculate** bounding box coordinates
3. **Check boundary overlap**: Ensure object doesn't extend too far outside image
4. **Check object overlap**: Calculate IoU with all existing objects
5. **Accept or retry**: If overlaps exceed threshold, try new position (up to max attempts)
6. **Skip if needed**: If placement fails after all attempts, skip object and log debug message

### Key Functions

#### `calculate_bbox_iou(bbox1, bbox2)`
Calculates Intersection over Union between two bounding boxes.

**Parameters:**
- `bbox1`, `bbox2`: Tuples of (cx, cy, w, h) in normalized coordinates

**Returns:**
- Float between 0 (no overlap) and 1 (complete overlap)

#### `calculate_boundary_overlap(bbox, img_bounds)`
Calculates how much of a bounding box extends outside image boundaries.

**Parameters:**
- `bbox`: Tuple of (cx, cy, w, h) in normalized coordinates
- `img_bounds`: Image boundaries (default: (0, 0, 1, 1))

**Returns:**
- Float between 0 (fully inside) and 1 (fully outside)

## Usage

### Python API

```python
from lit_yolo.data import BaseDataModule

# Create dataset with custom overlap threshold
dataset_path = BaseDataModule.create_synthetic_dataset(
    root="./my_dataset",
    num_samples=100,
    overlap_threshold=0.3,  # Default: 0.3 (30% IoU allowed)
)
```

### CLI

```bash
# Create dataset via command line
python -m lit_yolo create dataset \
    --output ./my_dataset \
    --num_samples 100 \
    --overlap_threshold 0.3
```

## Configuration

### overlap_threshold Parameter

Controls maximum allowed overlap (IoU) between objects and with boundaries.

**Values:**
- `0.0` - No overlap allowed (strictest, may place fewer objects)
- `0.1` - Minimal overlap (strict)
- `0.3` - Balanced (default, recommended)
- `0.5` - More lenient (allows denser packing)
- Higher values allow more overlap

**Trade-offs:**
- Lower values: Better visibility, fewer objects per image
- Higher values: More objects per image, some occlusion possible

### max_placement_attempts

Number of attempts to place each object before skipping it.

**Default:** 50
**Range:** 1-1000

Higher values increase chances of placing all objects but take longer.

## Examples

See `examples/overlap_threshold_demo.py` for a comprehensive demonstration.

```bash
python examples/overlap_threshold_demo.py
```

## Implementation Details

### Changes to Code

1. **New functions in `utils.py`:**
   - `calculate_bbox_iou()` - IoU calculation
   - `calculate_boundary_overlap()` - Boundary check

2. **Modified `generate_synthetic_sample()`:**
   - Added overlap checking logic
   - Retry mechanism for placement
   - Debug logging for skipped objects

3. **Updated API:**
   - `BaseDataModule.create_synthetic_dataset()` - Added overlap_threshold parameter
   - CLI wrapper - Exposed parameter for command-line use

### Backward Compatibility

✅ Fully backward compatible:
- Default `overlap_threshold=0.3` provides good balance
- Existing code continues to work without changes
- No breaking changes to existing APIs

## Testing

Comprehensive test suite in `tests/unittests/data/test_overlap.py`:

- Helper function correctness (IoU, boundary overlap)
- Different threshold values (0.0, 0.1, 0.3, 0.5)
- Object placement verification
- Boundary constraint verification

Run tests:
```bash
pytest tests/unittests/data/test_overlap.py -v
```

## Performance

**Impact:** Minimal
- Overlap checking is O(n) per object (n = existing objects)
- Typical datasets: negligible impact
- Large objects + low thresholds: may take slightly longer due to retries

**Optimization:**
- Placement attempts limited to prevent infinite loops
- Early exit when valid placement found
- Debug logging only (no performance impact in production)

## Future Enhancements

Potential improvements:
- Spatial indexing for faster overlap detection
- Smart placement hints based on existing objects
- Anisotropic thresholds (different for x/y directions)
- Gradient-based placement optimization
