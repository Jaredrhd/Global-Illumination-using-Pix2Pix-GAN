import bpy
import numpy as np
from random import randint, uniform, seed
from math import pi

seed(0)

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

#------------------------------------------------------------------------------------------------

imagesToCreate = 1000

for idx in range(500, imagesToCreate):

    #how many cubes you want to add
    count = randint(25,50)

    #Remove unused materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    objectsAdded = []

    for c in range(0,count):
        x = randint(-13,13)
        y = randint(-13,13)
        z = randint(2,15)

        xRot = pi * randint(0,360)/180
        yRot = pi * randint(0,360)/180
        zRot = pi * randint(0,360)/180

        redCol = uniform(0,1)
        greenCol = uniform(0,1)
        blueCol = uniform(0,1)


        #Spawn cube
        randomShape = randint(0, 5)

        if randomShape == 0:
            bpy.ops.mesh.primitive_cube_add(location=(x,y,z), rotation=(xRot, yRot, zRot))
        elif randomShape == 1:
            bpy.ops.mesh.primitive_cylinder_add(location=(x,y,z), rotation=(xRot, yRot, zRot))
        elif randomShape == 2:
            bpy.ops.mesh.primitive_cone_add(location=(x,y,z), rotation=(xRot, yRot, zRot))
        elif randomShape == 3:
            bpy.ops.mesh.primitive_uv_sphere_add(location=(x,y,z), rotation=(xRot, yRot, zRot))
        elif randomShape == 4:
            bpy.ops.mesh.primitive_ico_sphere_add(location=(x,y,z), rotation=(xRot, yRot, zRot))
        elif randomShape == 5:
            bpy.ops.mesh.primitive_torus_add(location=(x,y,z), rotation=(xRot, yRot, zRot))


        tempObj = bpy.context.object

        mat = bpy.data.materials.new("Random Col")
        mat.use_nodes = True
        
        # Get the principled BSDF (created by default)
        principled = mat.node_tree.nodes['Principled BSDF']
        principled.inputs['Base Color'].default_value = (redCol, greenCol, blueCol, 1)

        # Assign the material to the object
        tempObj.data.materials.append(mat)

        objectsAdded.append(tempObj)

    #---------------------------------------------------------------
    #Walls
    #---------------------------------------------------------------
    #Set up wall colours
    for i in range(5):
        wallName = 'Wall0{0}'.format(i)
        wall = bpy.data.objects[wallName]
        wall.data.materials.clear()
        wallmat = bpy.data.materials.new("Wall Col")
        wallmat.use_nodes = True
        wallmat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (uniform(0,1), uniform(0,1), uniform(0,1), 1)
        wall.data.materials.append(wallmat)


    #Set up Camera
    scene  = bpy.data.scenes['Scene']
    scene.camera.rotation_mode = "XYZ"

    scene.camera.rotation_euler[0] = pi/2 - pi * randint(0,10)/180    #X
    scene.camera.rotation_euler[2] = pi * randint(0,360)/180    #Z
    
    #Move point light
    bpy.data.objects['Point'].location = (randint(-10, 10), randint(-10, 10), randint(1, 10))
    

    #EEVEE Render
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    links.new(rl.outputs[0], v.inputs[0])  #DL
    bpy.ops.render.render()
    bpy.data.images['Viewer Node'].save_render(r"C:\Users\jared\Desktop\ImagesBlender\{0}.png".format(str(idx).zfill(4) + "_DI"))
    
    links.new(rl.outputs[2], v.inputs[0])  #Depth
    bpy.ops.render.render(write_still=False)
    bpy.data.images['Viewer Node'].save_render(r"C:\Users\jared\Desktop\ImagesBlender\{0}.png".format(str(idx).zfill(4) + "_Depth"))

    links.new(rl.outputs[3], v.inputs[0])  #Normal
    bpy.ops.render.render(write_still=False)
    bpy.data.images['Viewer Node'].save_render(r"C:\Users\jared\Desktop\ImagesBlender\{0}.png".format(str(idx).zfill(4) + "_Normal"))

    links.new(rl.outputs[4], v.inputs[0])  #Shadow 
    bpy.ops.render.render(write_still=False)
    bpy.data.images['Viewer Node'].save_render(r"C:\Users\jared\Desktop\ImagesBlender\{0}.png".format(str(idx).zfill(4) + "_Shadow"))

    links.new(rl.outputs[5], v.inputs[0])  #Diffuse 
    bpy.ops.render.render(write_still=False)
    bpy.data.images['Viewer Node'].save_render(r"C:\Users\jared\Desktop\ImagesBlender\{0}.png".format(str(idx).zfill(4) + "_Diffuse"))

    #CYCLES Render
    bpy.context.scene.render.engine = 'CYCLES'

    links.new(rl.outputs[0], v.inputs[0])  #Path Traced
    bpy.ops.render.render(write_still=False)
    bpy.data.images['Viewer Node'].save_render(r"C:\Users\jared\Desktop\ImagesBlender\{0}.png".format(str(idx).zfill(4) + "_RI"))


    #-----------------------------------------------------------------------------
    #Delete remaining objects in scene
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    for object in objectsAdded:
        object.select_set(True)

    bpy.ops.object.delete()