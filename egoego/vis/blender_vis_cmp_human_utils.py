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
    parser.add_argument('--gt-folder', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='')
    parser.add_argument('--out-folder', type=str, metavar='PATH',
                        help='path to output folder which include rendered img files',
                        default='')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .blend path for 3D scene',
                        default='')
    parser.add_argument('--material-color', type=str, 
                        help='material, decides color',
                        default='blue')
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
    gt_obj_folder = args.gt_folder 
    output_dir = args.out_folder
    print("obj_folder:{0}".format(obj_folder))
    print("output dir:{0}".format(output_dir))

    mat_color = args.material_color 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare ply paths 
    ori_obj_files = os.listdir(obj_folder)
    ori_obj_files.sort()
    obj_files = []
    for tmp_name in ori_obj_files:
        if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            obj_files.append(tmp_name)

    gt_ori_obj_files = os.listdir(gt_obj_folder)
    gt_ori_obj_files.sort()
    gt_obj_files = []
    for tmp_name in gt_ori_obj_files:
        if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            gt_obj_files.append(tmp_name)

    for frame_idx in range(len(obj_files)):
        file_name = obj_files[frame_idx]
        gt_file_name = gt_obj_files[frame_idx]

        # Iterate folder to process all model
        path_to_file = os.path.join(obj_folder, file_name)
        gt_path_to_file = os.path.join(gt_obj_folder, gt_file_name)

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
        
        human_obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
        # obj_object.location.y = 0

        # human_mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
        # human_obj_object.data.materials.append(human_mat)
        # human_mat.use_nodes = True
        # principled_bsdf = human_mat.node_tree.nodes['Principled BSDF']
        # if principled_bsdf is not None:
        #     # principled_bsdf.inputs[0].default_value = (220/255.0, 220/255.0, 220/255.0, 1) # Gray, close to white after rendering 
        #     principled_bsdf.inputs[0].default_value = (10/255.0, 30/255.0, 225/255.0, 1) # Light Blue, used for floor scene 

        # human_obj_object.active_material = human_mat
        if mat_color == "orange":
            human_obj_object.active_material = bpy.data.materials.get("orange")
        elif mat_color == "blue":
            human_obj_object.active_material = bpy.data.materials.get("blue")
        elif mat_color == "purple":
            human_obj_object.active_material = bpy.data.materials.get("purple")
        elif mat_color == "green":
            human_obj_object.active_material = bpy.data.materials.get("green")
       
        # Load GT human mesh and set material 
        if ".obj" in gt_path_to_file:
            gt_human_new_obj = bpy.ops.import_scene.obj(filepath=gt_path_to_file, split_mode ="OFF")
        elif ".ply" in path_to_file:
            gt_human_new_obj = bpy.ops.import_mesh.ply(filepath=gt_path_to_file)
        # obj_object = bpy.context.selected_objects[0]
        gt_human_obj_object = bpy.data.objects[str(gt_file_name.replace(".ply", "").replace(".obj", ""))]
        # obj_object.scale = (0.3, 0.3, 0.3)
        gt_human_mesh = gt_human_obj_object.data
        for f in gt_human_mesh.polygons:
            f.use_smooth = True
        
        gt_human_obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
        # obj_object.location.y = 0

        # gt_human_mat = bpy.data.materials.new(name="MaterialNameGT")  # set new material to variable
        # gt_human_obj_object.data.materials.append(gt_human_mat)
        # gt_human_mat.use_nodes = True
        # gt_principled_bsdf = gt_human_mat.node_tree.nodes['Principled BSDF']
        # if gt_principled_bsdf is not None:
        #     # gt_principled_bsdf.inputs[0].default_value = (220/255.0, 220/255.0, 220/255.0, 1) # Gray, close to white after rendering 
        #     # gt_principled_bsdf.inputs[0].default_value = (10/255.0, 30/255.0, 225/255.0, 1) # Light Blue, used for floor scene 
        #     gt_principled_bsdf.inputs[0].default_value = (11/255.0, 83/255.0, 69/255.0, 1)

        # gt_human_obj_object.active_material = gt_human_mat
        gt_human_obj_object.active_material = bpy.data.materials.get("green")

        # bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, file_name.replace(".ply", ".png"))
        bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, ("%05d"%frame_idx)+".jpg")
        bpy.ops.render.render(write_still=True)

        # Delet materials
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        bpy.data.objects.remove(human_obj_object, do_unlink=True)    
        bpy.data.objects.remove(gt_human_obj_object, do_unlink=True)     

    bpy.ops.wm.quit_blender()
