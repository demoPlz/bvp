import trimesh
import os
import glob

def convert_to_collision_obj(input_file, output_file):
    ## load the input mesh
    mesh = trimesh.load_mesh(input_file)
    

    ##assume that mesh is always not watertight and generate convex hull
    ##although the number of faces and vertices reduce it should be fine

    ##fix holes 
    trimesh.repair.fill_holes(mesh)
    ## fix normals and winding
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    mesh =  mesh.convex_hull

  

    mesh.export(output_file)


def batch_convert(input_dir,output_dir,file_pattern="*.STL"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    for file_path in glob.glob(os.path.join(input_dir,file_pattern)):
        filename = os.path.basename(file_path)
        name,_ = os.path.splitext(filename)
        output_path = os.path.join(output_dir,f"{name}.obj")
        print(f"converting {file_path} to {output_path}")
        convert_to_collision_obj(file_path,output_path)

    



if __name__ == "__main__":
    input_file = "/home/perseusdg/Development/colcon_ws/src/ros2_kortex/kortex_description/grippers/gen3_lite_2f/meshes"
    output_file = "collision"

    batch_convert(input_file,output_file)

