import numpy as np
from plyfile import PlyData, PlyElement
import pandas as pd
import pdb
import open3d as o3d
import cv2
import os
from PIL import Image
import Imath
import OpenEXR

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="800"/>
            <integer name="height" value="800"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.025"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def render_point(render_path, save_path):
    with open(render_path, 'r') as f:
        files = f.readlines()
        for file in files:
            xml_segments = [xml_head]
            file = file.strip('\n')
            o3d_point = o3d.io.read_point_cloud(file)
            judge = (file.split('_')[-1]).split('.')[0]
            if judge != 'GT':
                o3d_point.translate((-2, 0, 0), relative=True)

            o3d_point.translate((0, 0.2, -0.3), relative=True)
            R = o3d_point.get_rotation_matrix_from_xyz((np.pi/6, 0, np.pi/3))
            o3d_point.rotate(R, center=(0,0,0))

            points_coordinate = np.asarray(o3d_point.points)
            points_color = np.asarray(o3d_point.colors)
            for i in range(points_coordinate.shape[0]):
                xml_segments.append(xml_ball_segment.format(points_coordinate[i,2], points_coordinate[i,0], points_coordinate[i,1], 
                points_color[i,0], points_color[i,1], points_color[i,2]))
            xml_segments.append(xml_tail)
            xml_content = str.join('', xml_segments)

            xml_name = (file.split('/')[-1]).split('.')[0] + '.xml'
            xml_path = save_path + xml_name
            with open(xml_path, 'w') as m:
                m.write(xml_content)
                print(f'{xml_path} | finish!')
        f.close()

def ConvertEXRToJPG(exrfile, jpgfile):
    File = OpenEXR.InputFile(exrfile)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

    rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)

    rgb8 = [Image.frombytes("F", Size, c.tostring()).convert("L") for c in rgb]

    Image.merge("RGB", rgb8).save(jpgfile, "JPEG", quality=95)

def EXR_to_JPG_batch(exr_folder, img_folder):
    exr_files = os.listdir(exr_folder)
    for file in exr_files:
        exr_file = exr_folder + file
        img_path = img_folder + ''
        ConvertEXRToJPG(exr_file, img_path)
        print(f'{exr_file} | finish!')
    print(f'Saved in {img_folder}')


if __name__=='__main__':
    exr_file = '***.exr'
    jpg_file = '***.jpg'
    rend_path = 'a txt file which contain all .ply file'
    save_path = 'folder to save the .xml file'
    render_point(rend_path, save_path)




