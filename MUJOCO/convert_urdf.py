import re
import os
import tempfile
import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path

def generate_mjcf():
    script_dir = Path(__file__).resolve().parent
    absolute_pkg_path = str(script_dir.parent)
    urdf_path = str(script_dir.parent/'urdf'/'quadruped.urdf')
    output_path = str(script_dir.parent/'urdf'/'scene.xml') 

    # 1. Load and clean URDF strings
    with open(urdf_path, 'r') as file:
        urdf_xml = file.read()

    urdf_xml = urdf_xml.replace('package://quins_ros2', absolute_pkg_path)
    urdf_xml = re.sub(r'<xacro:arg.*?>', '', urdf_xml)
    urdf_xml = re.sub(r'(<robot[^>]*>)', r'\1\n<mujoco><compiler fusestatic="false"/></mujoco>', urdf_xml, count=1)

    # 2. Initial MuJoCo compilation to resolve base URDF
    temp_model = mujoco.MjModel.from_xml_string(urdf_xml)
    temp_mjcf = tempfile.NamedTemporaryFile(delete=False, suffix='.xml')
    temp_mjcf.close()
    mujoco.mj_saveLastXML(temp_mjcf.name, temp_model)

    with open(temp_mjcf.name, 'r') as file:
        mjcf_xml = file.read()
    os.remove(temp_mjcf.name)

    # 3. Inject Environment, Scene Lighting, and Procedural Terrain (Obstacles)
    environment_injection = """
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="32" height="32"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    </asset>
    <worldbody>
        <light pos="0 0 5" dir="0 0 -1" directional="true"/>
        <geom name="floor" type="plane" pos="0 0 -2.5" size="100 100 0.1" material="grid" condim="3" friction="1.5 0.005 0.0001" contype="1" conaffinity="1"/>
        
        <!-- Phase 1 Terrain Curriculum: Hurdles and Stairs -->
        <geom name="hurdle_1" type="box" pos="3.0 0 -2.4" size="0.2 2.0 0.1" material="grid" condim="3" contype="1" conaffinity="1"/>
        <geom name="hurdle_2" type="box" pos="5.0 0 -2.3" size="0.2 2.0 0.2" material="grid" condim="3" contype="1" conaffinity="1"/>
        <geom name="stair_1" type="box" pos="7.0 0 -2.4" size="0.5 2.0 0.1" material="grid" condim="3" contype="1" conaffinity="1"/>
        <geom name="stair_2" type="box" pos="8.0 0 -2.2" size="0.5 2.0 0.3" material="grid" condim="3" contype="1" conaffinity="1"/>
    """
    mjcf_xml = mjcf_xml.replace('<worldbody>', environment_injection)

    # 4. Inject Actuators
    JOINT_NAMES = {
        'LF': ['tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint'],
        'RF': ['tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint'],
        'LH': ['bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint'],
        'RH': ['br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint']
    }
    
    actuators_xml = "<actuator>\n"
    for leg, joints in JOINT_NAMES.items():
        for joint in joints:
            actuators_xml += f'    <motor name="{joint}_motor" joint="{joint}" gear="1" ctrllimited="true" ctrlrange="-1500 1500"/>\n'
    actuators_xml += "</actuator>\n"
    
    damping_xml = "<default>\n    <joint damping=\"0.05\" frictionloss=\"0.01\"/>\n    <geom contype=\"1\" conaffinity=\"0\"/>\n</default>\n"
    mjcf_xml = mjcf_xml.replace('<worldbody>', f'{damping_xml}<worldbody>')
    mjcf_xml = mjcf_xml.replace('</worldbody>', f'</worldbody>\n{actuators_xml}')

    # 5. Inject Floating Base and Camera
    root_xml = ET.fromstring(mjcf_xml)
    worldbody = root_xml.find('worldbody')
    if worldbody is not None:
        for body in worldbody.findall('body'):
            if body.find('joint') is None:
                body.set('pos', '0 0 1.5')
                ET.SubElement(body, 'freejoint', name='root_floating_base')
                ET.SubElement(body, 'camera', name='track_cam', mode='trackcom', pos='0 -2 1')
                break
                
    # 6. Save final XML
    tree = ET.ElementTree(root_xml)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Successfully generated native MJCF at: {output_path}")

if __name__ == '__main__':
    generate_mjcf()
