import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from collections import deque

def extract_vertex_data(vertices, has_scal=True, has_rgb=False):
    """Extract and convert vertex data from a structured numpy array of vertices."""
    data = []
    
    # Determine the prefix to be used based on whether "scal_" should be included
    prefix = 'scal_' if has_scal else ''
    
    # Iterate over each vertex and extract the necessary attributes
    for vertex in vertices:
        entry = (
            vertex['x'], vertex['y'], vertex['z'],
            vertex['nx'], vertex['ny'], vertex['nz'],
            vertex[f'{prefix}f_dc_0'], vertex[f'{prefix}f_dc_1'], vertex[f'{prefix}f_dc_2'],
            *[vertex[f'{prefix}f_rest_{i}'] for i in range(45)],
            vertex[f'{prefix}opacity'],
            vertex[f'{prefix}scale_0'], vertex[f'{prefix}scale_1'], vertex[f'{prefix}scale_2'],
            vertex[f'{prefix}rot_0'], vertex[f'{prefix}rot_1'], vertex[f'{prefix}rot_2'], vertex[f'{prefix}rot_3']
        )
        
        # If the point cloud contains RGB data, append it to the entry
        if has_rgb:
            entry += (vertex['red'], vertex['green'], vertex['blue'])
        
        data.append(entry)
    
    return data


def define_dtype(has_scal=True, has_rgb=False):
    prefix = 'scal_' if has_scal else ''
    
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        (f'{prefix}f_dc_0', 'f4'), (f'{prefix}f_dc_1', 'f4'), (f'{prefix}f_dc_2', 'f4'),
        *[(f'{prefix}f_rest_{i}', 'f4') for i in range(45)],
        (f'{prefix}opacity', 'f4'),
        (f'{prefix}scale_0', 'f4'), (f'{prefix}scale_1', 'f4'), (f'{prefix}scale_2', 'f4'),
        (f'{prefix}rot_0', 'f4'), (f'{prefix}rot_1', 'f4'), (f'{prefix}rot_2', 'f4'), (f'{prefix}rot_3', 'f4')
    ]
    if has_rgb:
        dtype.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    return dtype


def text_based_detect_format(file_path):
    """Detect if the given file is in '3dgs' or 'cc' format."""
    with open(file_path, 'rb') as file:
        header_bytes = file.read(2048)  # Read the beginning to detect the format

    header = header_bytes.decode('utf-8', errors='ignore')
    if "property float f_dc_0" in header:
        return "3dgs"
    elif "property float scal_f_dc_0" in header:
        return "cc"
    else:
        return None

def convert_3dgs_to_cc_without_rgb(input_path, output_path, plydata):
    """
    Convert a PLY file from the 3DGS format to the CC format without adding RGB data.
    This function only updates the header and directly copies the binary data.
    """
    
    # Get the appropriate dtype for CC format (with "scal_" prefix and without RGB)
    dtype_for_cc = define_dtype(has_scal=True, has_rgb=False)
    
    # Convert the vertices data to CC format
    converted_data = plydata['vertex'].data.astype(dtype_for_cc)
    
    # Write to new PLY file
    new_plydata = PlyData([PlyElement.describe(converted_data, 'vertex')], byte_order='=')
    new_plydata.write(output_path)
    
    print(f"Converted 3DGS file to CC format without RGB and saved to {output_path}.")


def convert_3dgs_to_cc_with_rgb(input_path, output_path, detected_format, vertices=None):
    """Create a new PLY file with the opposite header of the detected format and binary data from the original."""
    
    # If vertices are not provided, read from the input file
    if vertices is None:
        plydata = PlyData.read(input_path)
        vertices = plydata['vertex'].data
    
    # Determine if RGB processing is needed
    assert isinstance(detected_format, str), "detected_format is not a string!"
    process_rgb = (detected_format == "3dgs")
    use_scal = (detected_format == "3dgs")
    
    data = extract_vertex_data(vertices, has_scal=use_scal, has_rgb=process_rgb)
    dtype = define_dtype(has_scal=use_scal, has_rgb=process_rgb)

    all_processed_data = []

    # We process in chunks without tqdm
    for start_idx in range(0, len(vertices), 10000):
        end_idx = start_idx + 10000
        chunk_data = data[start_idx:end_idx]
        all_processed_data.append(chunk_data)

    # Concatenate all processed chunks together
    final_data = np.concatenate(all_processed_data)

    new_plydata = PlyData([PlyElement.describe(final_data, 'vertex')], byte_order='=')
    new_plydata.write(output_path)
    
def convert_cc_to_3dgs(input_path, output_path, vertices=None):
    """Create a new 3DGS PLY file from a CC PLY file."""
    
    plydata = PlyData.read(input_path)
    
    # Check if RGB data exists
    has_rgb = 'red' in plydata['vertex'].data.dtype.names

    # If vertices are not provided, read from the input file
    if vertices is None:
        vertices = plydata['vertex'].data
    
    # Extract vertex data without RGB (since we're converting to 3DGS)
    data = extract_vertex_data(vertices, has_scal=True, has_rgb=has_rgb, strip_rgb=True)
    dtype = define_dtype(has_scal=True, has_rgb=False)  # No RGB for 3DGS

    # Write to new PLY file
    new_plydata = PlyData([PlyElement.describe(data, 'vertex')], byte_order='=')
    new_plydata.write(output_path)


def get_neighbors(voxel_coords):
    """Get the face-touching neighbors of the given voxel coordinates."""
    x, y, z = voxel_coords
    neighbors = [
        (x-1, y, z), (x+1, y, z),
        (x, y-1, z), (x, y+1, z),
        (x, y, z-1), (x, y, z+1)
    ]
    return neighbors

def apply_density_filter(plydata, voxel_size=1.0, threshold_ratio=0.0032):
    """
    Filter the vertices of plydata to keep only those in denser regions and from the largest contiguous voxel cluster.
    
    Parameters:
    - plydata: The PlyData object containing the point cloud.
    - voxel_size: The size of the voxels used for density calculation.
    - threshold_ratio: The ratio used to calculate the dynamic threshold. 
                       Threshold = number of vertices * threshold_ratio
    """
    vertices = plydata['vertex'].data

    # Create a dictionary to count points in each voxel
    voxel_counts = {}
    
    for vertex in vertices:
        # Compute the voxel coordinates for each vertex
        voxel_coords = (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size))
        if voxel_coords in voxel_counts:
            voxel_counts[voxel_coords] += 1
        else:
            voxel_counts[voxel_coords] = 1

    # Calculate dynamic threshold
    threshold = int(len(vertices) * threshold_ratio)

    # Filter out voxels based on density threshold
    dense_voxels = {k: v for k, v in voxel_counts.items() if v >= threshold}

    # Identify the largest contiguous voxel cluster
    visited = set()
    max_cluster = set()
    for voxel in dense_voxels:
        if voxel not in visited:
            current_cluster = set()
            queue = deque([voxel])
            while queue:
                current_voxel = queue.popleft()
                visited.add(current_voxel)
                current_cluster.add(current_voxel)
                for neighbor in get_neighbors(current_voxel):
                    if neighbor in dense_voxels and neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
            if len(current_cluster) > len(max_cluster):
                max_cluster = current_cluster

    # Filter vertices to retain only those from the largest contiguous voxel cluster
    filtered_vertices = [vertex for vertex in vertices if (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size)) in max_cluster]

    # Update the plydata object with the filtered vertices
    new_vertex_element = PlyElement.describe(np.array(filtered_vertices, dtype=vertices.dtype), 'vertex')
    plydata.elements = (new_vertex_element,) + plydata.elements[1:]
        
    print(f"Number of vertices after density filter: {len(filtered_vertices)}")



def main():
    parser = argparse.ArgumentParser(description="Convert between standard 3D Gaussian Splat and 3D Gaussian Splat for Cloud Compare formats.")
    
    # Arguments for input, output, and format
    parser.add_argument("--input", "-i", required=True, help="Path to the point cloud file.")
    parser.add_argument("--output", "-o", required=True, help="Path to save the converted point cloud file.")
    parser.add_argument("--format", "-f", choices=["3dgs", "cc"], required=True, help="Desired output format.")
    
    # Other flags
    parser.add_argument("--rgb", action="store_true", help="Add RGB values to the output file based on f_dc values (only for Cloud Compare format).")
    parser.add_argument("--density_filter", action="store_true", help="Filter the points to keep only regions with higher point density.")
    
    args = parser.parse_args()

    # Check the format of the input file
    detected_format = text_based_detect_format(args.input)
    
    # Diagnose the detected_format
    print(f"detected_format type: {type(detected_format)}")
    print(f"detected_format value: {detected_format}")

    if not detected_format:
        print("The provided file is not a recognized 3D Gaussian Splat point cloud format.")
        return

    plydata = PlyData.read(args.input)
    vertices = plydata['vertex'].data
    print(f"Initial number of vertices: {len(vertices)}")

    if args.density_filter:
        print("Applying density filter...")
        apply_density_filter(plydata)
        print(f"Number of vertices after density filter: {len(plydata['vertex'].data)}")

    # Conversion operations
    if detected_format == "3dgs" and args.format == "cc":
        if args.rgb:
            convert_3dgs_to_cc_with_rgb(args.input, args.output, vertices)
        else:
            convert_3dgs_to_cc_without_rgb(args.input, args.output,  plydata)
    elif detected_format == "cc" and args.format == "3dgs":
        convert_cc_to_3dgs(args.input, args.output, vertices)
    elif detected_format == args.format:
        if args.rgb and detected_format == "3dgs":
            print("Warning: RGB flag is not supported for 3DGS format. Ignoring RGB flag.")
        convert_3dgs_to_cc_without_rgb(args.input, args.output, vertices)
    else:
        print(f"Invalid conversion direction specified for the detected format.")

    print(f"Conversion completed. Output saved to: {args.output}")

if __name__ == "__main__":
    main()


