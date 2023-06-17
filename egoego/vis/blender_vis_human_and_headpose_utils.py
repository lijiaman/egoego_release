import numpy as np
import json
import os
import math
import argparse

import bpy

if __name__ == "__main__":
    import sys
    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--")+1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment.')
    parser.add_argument('--folder', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='')
    parser.add_argument('--out-folder', type=str, metavar='PATH',
                        help='path to output folder which include rendered img files',
                        default='')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .blend path for 3D scene',
                        default='')
    parser.add_argument('--head-path', type=str, 
                        help='head pose numpy path',
                        default='')
    parser.add_argument("--vis_head_only", action="store_true")
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    # Load the world
    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    scene_name = args.scene.split("/")[-1].replace("_scene.blend", "")
    print("scene name:{0}".format(scene_name))
   
    obj_folder = args.folder
    output_dir = args.out_folder
    print("obj_folder:{0}".format(obj_folder))
    print("output dir:{0}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load head pose numpy 
    head_pose_data = np.load(args.head_path) # T X 7 
    head_trans = head_pose_data[:, :3] # T X 3 
    head_rot_quat = head_pose_data[:, 3:] # T X 4 

    vis_head_only = args.vis_head_only 

    num_frames = head_pose_data.shape[0] 

    for frame_idx in range(num_frames):
        # Create head object.
        head_object = bpy.data.objects.get("coord.001")
        head_object.hide_render = False 

        head_object.scale = (0.10, 0.10, 0.10)

        head_object.rotation_quaternion = (head_rot_quat[frame_idx, 0], head_rot_quat[frame_idx, 1], \
        head_rot_quat[frame_idx, 2], head_rot_quat[frame_idx, 3]) 
       
        head_object.location = (head_trans[frame_idx, 0], head_trans[frame_idx, 1], head_trans[frame_idx, 2])

        # Create fullbody object.
        file_name = "%05d"%(frame_idx)+".obj"
        path_to_file = os.path.join(obj_folder, "%05d"%(frame_idx)+".obj")

        if not vis_head_only:
            # Load human mesh and set material 
            if ".obj" in path_to_file:
                human_new_obj = bpy.ops.import_scene.obj(filepath=path_to_file, split_mode ="OFF")
            elif ".ply" in path_to_file:
                human_new_obj = bpy.ops.import_mesh.ply(filepath=path_to_file)
            # obj_object = bpy.context.selected_objects[0]
            if file_name == "00000.obj":
                human_obj_object = bpy.data.objects[str(file_name.replace(".ply", "").replace(".obj", ""))+".004"]
            else:
                human_obj_object = bpy.data.objects[str(file_name.replace(".ply", "").replace(".obj", ""))]
            # obj_object.scale = (0.3, 0.3, 0.3)
            human_mesh = human_obj_object.data
            for f in human_mesh.polygons:
                f.use_smooth = True
            
            human_obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0))

            # human_obj_object.location.y = -1 

            human_obj_object.active_material = bpy.data.materials.get("blue")

            bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, ("%05d"%frame_idx)+".jpg")
            bpy.ops.render.render(write_still=True)

            bpy.data.objects.remove(human_obj_object, do_unlink=True)    
        else:
            bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, ("%05d"%frame_idx)+".jpg")
            bpy.ops.render.render(write_still=True)

    bpy.ops.wm.quit_blender()
