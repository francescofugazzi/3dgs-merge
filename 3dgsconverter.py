import argparse
import numpy as np
import multiprocessing
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from collections import deque
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors

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

def compute_rgb_from_vertex(plydata, detected_format=None):
    print("Entering compute_rgb_from_vertex function...")
    
    vertices = plydata['vertex'].data
    print("Loaded vertices data from plydata.")

    if detected_format is None:
        # If detected format is not provided, infer it from the plydata
        if 'f_dc_0' in vertices.dtype.names:
            detected_format = '3dgs'
        else:
            detected_format = 'cc'
    
    # Depending on the detected format, use the appropriate field names
    if detected_format == "3dgs":
        f_dc = np.column_stack((vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']))
    else:
        f_dc = np.column_stack((vertices['scal_f_dc_0'], vertices['scal_f_dc_1'], vertices['scal_f_dc_2']))
    
    colors = (f_dc + 1) * 127.5
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    
    return colors

def define_dtype(has_scal=True, has_rgb=False):
    prefix = 'scalar_scal_' if has_scal else ''
    
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
    #print("Entering text_based_detect_format function...")
    
    with open(file_path, 'rb') as file:
        header_bytes = file.read(2048)  # Read the beginning to detect the format
        #print("Read header bytes from the file...")

    header = header_bytes.decode('utf-8', errors='ignore')
    #print("Decoded header to text...")

    if "property float f_dc_0" in header:
        return "3dgs"
    elif "property float scal_f_dc_0" in header or "property float scalar_scal_f_dc_0" in header or "property float scalar_f_dc_0" in header:
        return "cc"
    else:
        return None
    
def copy_data_with_prefix_check(source, target, possible_prefixes):
    """
    Given two structured numpy arrays (source and target), copy the data from source to target.
    If a field exists in source but not in target, this function will attempt to find the field
    in target by adding any of the possible prefixes to the field name.
    """
    for name in source.dtype.names:
        if name in target.dtype.names:
            target[name] = source[name]
        else:
            copied = False
            for prefix in possible_prefixes:
                # If the field starts with the prefix, try the field name without the prefix
                if name.startswith(prefix):
                    stripped_name = name[len(prefix):]
                    if stripped_name in target.dtype.names:
                        target[stripped_name] = source[name]
                        copied = True
                        break
                # If the field doesn't start with any prefix, try adding the prefix
                else:
                    prefixed_name = prefix + name
                    if prefixed_name in target.dtype.names:
                        target[prefixed_name] = source[name]
                        copied = True
                        break
            if not copied:
                print(f"Warning: Field {name} not found in target.")



def convert_3dgs_to_cc(input_path, output_path, plydata, process_rgb=False):
    print("Starting conversion process...")
    vertices = plydata['vertex'].data
    print("Loaded vertices from plydata.")

    print("\nSample of original data (first 10 rows):")
    for i in range(10):
        print(vertices[i])

    if process_rgb:
        print("\nProcessing RGB data...")
        colors = compute_rgb_from_vertex(plydata)
        
        dtype_for_cc = define_dtype(has_scal=True, has_rgb=True)
        extended_vertices = np.zeros(vertices.shape, dtype=dtype_for_cc)

        # Use the helper function to copy the data
        copy_data_with_prefix_check(vertices, extended_vertices, ["", "scal_", "scalar_", "scalar_scal_"])

        extended_vertices['red'] = colors[:, 0]
        extended_vertices['green'] = colors[:, 1]
        extended_vertices['blue'] = colors[:, 2]

        print("\nSample of extended data with RGB (first 10 rows):")
        for i in range(10):
            print(extended_vertices[i])

        print("\nWriting extended data to the output file...")
        new_plydata = PlyData([PlyElement.describe(extended_vertices, 'vertex')], byte_order='=')
        new_plydata.write(output_path)
    else:
        print("\nProcessing data without RGB...")
        dtype_for_cc = define_dtype(has_scal=True, has_rgb=False)
        converted_data = np.zeros(vertices.shape, dtype=dtype_for_cc)

        # Use the helper function to copy the data
        copy_data_with_prefix_check(vertices, converted_data, ["", "scal_", "scalar_", "scalar_scal_"])

        print("\nSample of converted data without RGB (first 10 rows):")
        for i in range(10):
            print(converted_data[i])

        print("\nWriting converted data to the output file...")
        new_plydata = PlyData([PlyElement.describe(converted_data, 'vertex')], byte_order='=')
        new_plydata.write(output_path)
    
    print(f"\nConverted 3DGS file to CC format {'with' if process_rgb else 'without'} RGB and saved to {output_path}.")
    
def convert_cc_to_3dgs(input_path, output_path, vertices):
    print("Converting from CC to 3DGS format...")
    
    # Create a new structured numpy array for 3DGS format
    dtype_3dgs = define_dtype(has_scal=False, has_rgb=False)  # Define 3DGS dtype without any prefix
    converted_data = np.zeros(vertices.shape[0], dtype=dtype_3dgs)
    
    # Use the helper function to copy the data
    copy_data_with_prefix_check(vertices, converted_data, ["", "scal_", "scalar_", "scalar_scal_"])

    print("\nSample of converted data (first 10 rows):")
    for i in range(10):
        print(converted_data[i])

    # Write to new PLY file
    new_plydata = PlyData([PlyElement.describe(converted_data, 'vertex')], byte_order='=')
    new_plydata.write(output_path)

    print(f"\nConverted CC file to 3DGS format and saved to {output_path}.")

def get_neighbors(voxel_coords):
    """Get the face-touching neighbors of the given voxel coordinates."""
    x, y, z = voxel_coords
    neighbors = [
        (x-1, y, z), (x+1, y, z),
        (x, y-1, z), (x, y+1, z),
        (x, y, z-1), (x, y, z+1)
    ]
    return neighbors

def count_voxels_chunk(vertices_chunk, voxel_size):
    voxel_counts = {}
    for vertex in vertices_chunk:
        voxel_coords = (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size))
        if voxel_coords in voxel_counts:
            voxel_counts[voxel_coords] += 1
        else:
            voxel_counts[voxel_coords] = 1
    return voxel_counts

def parallel_voxel_counting(vertices, voxel_size):
    num_processes = cpu_count()
    chunk_size = len(vertices) // num_processes
    chunks = [vertices[i:i + chunk_size] for i in range(0, len(vertices), chunk_size)]

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(count_voxels_chunk, [(chunk, voxel_size) for chunk in chunks])

    # Aggregate results from all processes
    total_voxel_counts = {}
    for result in results:
        for k, v in result.items():
            if k in total_voxel_counts:
                total_voxel_counts[k] += v
            else:
                total_voxel_counts[k] = v

    return total_voxel_counts

def apply_density_filter(plydata, voxel_size=1.0, threshold_percentage=0.32):
    vertices = plydata['vertex'].data

    # Convert threshold_percentage into a ratio
    threshold_ratio = threshold_percentage / 100.0

    # Parallelized voxel counting
    voxel_counts = parallel_voxel_counting(vertices, voxel_size)

    threshold = int(len(vertices) * threshold_ratio)
    dense_voxels = {k: v for k, v in voxel_counts.items() if v >= threshold}

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

    filtered_vertices = [vertex for vertex in vertices if (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size)) in max_cluster]
    new_vertex_element = PlyElement.describe(np.array(filtered_vertices, dtype=vertices.dtype), 'vertex')
    plydata.elements = (new_vertex_element,) + plydata.elements[1:]

def knn_worker(args):
    coords, tree, k = args
    coords = coords.reshape(1, -1)  # Reshape to a 2D array
    distances, _ = tree.kneighbors(coords)
    return np.mean(distances[:, 1:])

def remove_flyers(plydata, k=25, threshold_factor=10.5):
    """
    Remove flyer vertices from the point cloud using the K-Nearest Neighbors method.

    Parameters:
    - plydata: The plydata containing the vertices of the point cloud.
    - k: The number of neighbors for the KNN algorithm.
    - threshold_factor: Multiplier for the threshold to determine the range for outlier removal.

    Returns:
    - plydata with filtered vertices.
    """
    vertices = plydata['vertex'].data

    # Extract the coordinates for calculations
    coords = np.vstack((vertices['x'], vertices['y'], vertices['z'])).T

    # Compute K-Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    
    # Create a pool of workers
    num_cores = max(1, cpu_count() - 1)  # Leave one core free
    with Pool(processes=num_cores) as pool:
        avg_distances = pool.map(knn_worker, [(coord, nbrs, k) for coord in coords])

    # Threshold for removal is based on the mean and standard deviation of these average distances
    threshold = np.mean(avg_distances) + threshold_factor * np.std(avg_distances)

    # Create a mask for points to keep
    mask = np.array(avg_distances) < threshold

    # Create a new PlyElement with the filtered vertices
    new_vertex_element = PlyElement.describe(vertices[mask], 'vertex')

    # Update the plydata elements
    plydata.elements = (new_vertex_element,) + plydata.elements[1:]

    return plydata

def main():
    parser = argparse.ArgumentParser(description="Convert between standard 3D Gaussian Splat and 3D Gaussian Splat for Cloud Compare formats.")
    
    # Arguments for input, output, and format
    parser.add_argument("--input", "-i", required=True, help="Path to the point cloud file.")
    parser.add_argument("--output", "-o", required=True, help="Path to save the converted point cloud file.")
    parser.add_argument("--format", "-f", choices=["3dgs", "cc"], required=True, help="Desired output format.")
    
    # Other flags
    parser.add_argument("--rgb", action="store_true", help="Add RGB values to the output file based on f_dc values (only for Cloud Compare format).")
    parser.add_argument("--density_filter", action="store_true", help="Filter the points to keep only regions with higher point density.")
    parser.add_argument("--remove_flyers", action="store_true", help="Remove flyer points that are distant from the main cloud.")
    
    args = parser.parse_args()

    # Check the format of the input file
    detected_format = text_based_detect_format(args.input)
    
    # Diagnose the detected_format
    #print(f"detected_format type: {type(detected_format)}")
    #print(f"detected_format value: {detected_format}")

    if not detected_format:
        print("The provided file is not a recognized 3D Gaussian Splat point cloud format.")
        return

    plydata = PlyData.read(args.input)
    print(f"Initial number of vertices: {len(plydata['vertex'].data)}")

    if args.density_filter:
        print("Applying density filter...")
        apply_density_filter(plydata)
        print(f"Number of vertices after density filter: {len(plydata['vertex'].data)}")
        
    if args.remove_flyers:
        print("Removing flyers from the point cloud...")
        plydata = remove_flyers(plydata)
        print(f"Number of vertices after flyer removal: {len(plydata['vertex'].data)}")


    # Conversion operations
    if detected_format == "3dgs" and args.format == "cc":
        print("Converting from 3DGS to CC format...")
        convert_3dgs_to_cc(args.input, args.output, plydata, args.rgb)
    elif detected_format == "cc" and args.format == "3dgs":
        print("Converting from CC to 3DGS format...")
        if args.rgb:
            print("Warning: RGB flag is ignored for conversion to 3DGS format.")
        convert_cc_to_3dgs(args.input, args.output, plydata['vertex'].data)
    else:
        print(f"Conversion direction not recognized or not supported.")

if __name__ == "__main__":
    main()
