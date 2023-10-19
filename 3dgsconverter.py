import argparse
import re

# Define constant headers for gs3d and cc formats
GS3D_HEADER = """
ply
format binary_little_endian 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
property float f_rest_2
property float f_rest_3
property float f_rest_4
property float f_rest_5
property float f_rest_6
property float f_rest_7
property float f_rest_8
property float f_rest_9
property float f_rest_10
property float f_rest_11
property float f_rest_12
property float f_rest_13
property float f_rest_14
property float f_rest_15
property float f_rest_16
property float f_rest_17
property float f_rest_18
property float f_rest_19
property float f_rest_20
property float f_rest_21
property float f_rest_22
property float f_rest_23
property float f_rest_24
property float f_rest_25
property float f_rest_26
property float f_rest_27
property float f_rest_28
property float f_rest_29
property float f_rest_30
property float f_rest_31
property float f_rest_32
property float f_rest_33
property float f_rest_34
property float f_rest_35
property float f_rest_36
property float f_rest_37
property float f_rest_38
property float f_rest_39
property float f_rest_40
property float f_rest_41
property float f_rest_42
property float f_rest_43
property float f_rest_44
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
""".strip()

CC_HEADER = """
ply
format binary_little_endian 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float scal_f_dc_0
property float scal_f_dc_1
property float scal_f_dc_2
property float scal_f_rest_0
property float scal_f_rest_1
property float scal_f_rest_2
property float scal_f_rest_3
property float scal_f_rest_4
property float scal_f_rest_5
property float scal_f_rest_6
property float scal_f_rest_7
property float scal_f_rest_8
property float scal_f_rest_9
property float scal_f_rest_10
property float scal_f_rest_11
property float scal_f_rest_12
property float scal_f_rest_13
property float scal_f_rest_14
property float scal_f_rest_15
property float scal_f_rest_16
property float scal_f_rest_17
property float scal_f_rest_18
property float scal_f_rest_19
property float scal_f_rest_20
property float scal_f_rest_21
property float scal_f_rest_22
property float scal_f_rest_23
property float scal_f_rest_24
property float scal_f_rest_25
property float scal_f_rest_26
property float scal_f_rest_27
property float scal_f_rest_28
property float scal_f_rest_29
property float scal_f_rest_30
property float scal_f_rest_31
property float scal_f_rest_32
property float scal_f_rest_33
property float scal_f_rest_34
property float scal_f_rest_35
property float scal_f_rest_36
property float scal_f_rest_37
property float scal_f_rest_38
property float scal_f_rest_39
property float scal_f_rest_40
property float scal_f_rest_41
property float scal_f_rest_42
property float scal_f_rest_43
property float scal_f_rest_44
property float scal_opacity
property float scal_scale_0
property float scal_scale_1
property float scal_scale_2
property float scal_rot_0
property float scal_rot_1
property float scal_rot_2
property float scal_rot_3
end_header
""".strip()

def text_based_detect_format(file_path):
    """Detect if the given file is in 'gs3d' or 'cc' format."""
    with open(file_path, 'rb') as file:
        header_bytes = file.read(2048)  # Read the beginning to detect the format

    header = header_bytes.decode('utf-8', errors='ignore')
    if "property float f_dc_0" in header:
        return "gs3d"
    elif "property float scal_f_dc_0" in header:
        return "cc"
    elif "property float scalar_scal_f_dc_0" in header:
        return "cc"
    else:
        return None

def create_new_ply_with_header(input_path, output_path, detected_format):
    """Create a new PLY file with the opposite header of the detected format and binary data from the original."""
    # Get the end of the header in the original file to skip it
    with open(input_path, 'rb') as file:
        file_content = file.read()
        header_end = re.search(b"end_header\n", file_content).end()
        binary_data = file_content[header_end:]

    # Set the appropriate header based on the detected format
    if detected_format == "gs3d":
        new_header = CC_HEADER
    else:
        new_header = GS3D_HEADER

    # Fill in the number of vertices in the new header
    num_vertices = re.search(b"element vertex (\d+)", file_content).group(1).decode('utf-8')
    new_header = new_header.replace("{num_vertices}", num_vertices)

    # Create the new PLY file with the desired header and original binary data
    with open(output_path, 'wb') as file:
        file.write(new_header.encode('utf-8'))
        file.write(b'\n')  # Newline after the header
        file.write(binary_data)

def main():
    parser = argparse.ArgumentParser(description="Detect point cloud format: standard 3d gaussian splat or 3d gaussian splat for cloud compare and create a new file with the opposite header.")
    parser.add_argument("input_file", help="Path to the point cloud file.")
    parser.add_argument("output_file", help="Path to save the new point cloud file with the opposite header.")
    args = parser.parse_args()
    
    format_display_names = {
        "gs3d": "standard 3D Gaussian Splat",
        "cc": "3d Gaussian Splat for Cloud Compare"
    }

    detected_format = text_based_detect_format(args.input_file)
    
    if detected_format:
        print(f"Detected format of input file: {format_display_names[detected_format]}")
        desired_format = "cc" if detected_format == "gs3d" else "gs3d"
        create_new_ply_with_header(args.input_file, args.output_file, detected_format)
        print(f"Created new PLY file in {format_display_names[desired_format]} format: {args.output_file}")
    else:
        print("The provided file is not a recognized 3d gaussian splat point cloud format.")

if __name__ == "__main__":
    main()