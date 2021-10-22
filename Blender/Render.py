import bpy
import numpy as np
 
# switch on nodes
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
  
# clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)
  
# create input render layer node
rl = tree.nodes.new('CompositorNodeRLayers')      
rl.location = 185,285
 
# create output node
v = tree.nodes.new('CompositorNodeViewer')   
v.location = 750,210
v.use_alpha = False
 
# Links
links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input
 
# render
bpy.ops.render.render()
 
pixels = bpy.data.images['Viewer Node'].save_render(r"C:\Users\jared\Desktop\ImagesBlender\OutputImage.png")
 
# # get viewer pixels
# pixels = bpy.data.images['Viewer Node'].pixels
# print(len(pixels)) # size is always width * height * 4 (rgba)
 
# # copy buffer to numpy array for faster manipulation
# arr = np.array(pixels[:])

# output_image = bpy.data.images.new('OutputImage', alpha=True, width=1920, height=1080)
# output_image.file_format = 'PNG'
# output_image.alpha_mode = 'STRAIGHT'
# output_image.pixels = arr.ravel()
# output_image.filepath_raw = bpy.path.abspath(r"C:\Users\jared\Desktop\ImagesBlender\OutputImage.png")
# output_image.save()