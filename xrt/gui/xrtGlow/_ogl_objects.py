# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:23:24 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import copy
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as scprot


from ._ogl_axes import CoordinateBox
from ._utils import (create_qt_buffer, basis_rotation_q)
from ._utils import (is_oe, is_dcm, is_plate, is_aperture, is_screen)
from ._constants import (ambient, diffuse, specular, shininess)

from ..commons import qt
from ..commons import gl

from ...backends import raycing
from ...backends.raycing import oes as roes
from ...backends.raycing import materials as rmats

try:
    from stl import mesh
    isSTLsupported = True
except ImportError:
    isSTLsupported = False


class Beam3D():

    vertex_source = '''
    #version 410 core

    layout(location = 0) in vec3 position_start;
    layout(location = 4) in vec3 position_end;

    layout(location = 1) in float colorAxis;
    layout(location = 2) in float state;
    layout(location = 3) in float intensity;

    uniform sampler1D hsvTexture;

    uniform float opacity;
    uniform float iMax;
    uniform int isLost;
    uniform vec2 colorMinMax;

    uniform mat4 mPV;
    uniform mat4 modelStart;
    uniform mat4 modelEnd;

    uniform vec4 gridMask;
    uniform vec4 gridProjection;

    out vec4 vs_out_start;
    out vec4 vs_out_end;
    out vec4 vs_out_color;

    float hue;
    float intensity_v;
    vec4 hrgb;

    void main()
    {

     vs_out_start = mPV * (gridMask * (modelStart * vec4(position_start, 1.0))+
                           gridProjection);
     vs_out_end = mPV * (gridMask * (modelEnd * vec4(position_end, 1.0)) +
                         gridProjection);
     if (isLost > 0) {
             hrgb = vec4(0.9, 0., 0., 0.1);}
     else {
         hue = (colorAxis - colorMinMax.x) / (colorMinMax.y - colorMinMax.x);
         intensity_v = opacity*intensity/iMax;
         hrgb = vec4(texture(hsvTexture, hue*0.85).rgb, intensity_v);}

     vs_out_color = hrgb;

    }
    '''

    geometry_source = '''
    #version 410 core

    layout(points) in;
    layout(line_strip, max_vertices = 2) out;

    in vec4 vs_out_start[];
    in vec4 vs_out_end[];
    in vec4 vs_out_color[];

    uniform float pointSize;

    out vec4 gs_out_color;

    void main() {

        gl_Position = vs_out_start[0];
        gl_PointSize = pointSize;
        gs_out_color = vs_out_color[0];
        EmitVertex();

        gl_Position = vs_out_end[0];
        gs_out_color = vs_out_color[0];
        EmitVertex();

        EndPrimitive();
    }
    '''

    compute_source = '''
    #version 430

    layout(local_size_x = 32) in;

    layout(std430, binding = 0) buffer VertexBuffer {
        vec3 position[];
    };

    layout(std430, binding = 1) buffer ColorBuffer {
        float color[];
    };

    layout(std430, binding = 2) buffer StateBuffer {
        float state[];
    };

    layout(std430, binding = 3) buffer IntensityBuffer {
        float intensity[];
    };

    layout(std430, binding = 4) buffer RedBuffer {
        int red[];
    };

    layout(std430, binding = 5) buffer GreenBuffer {
        int green[];
    };

    layout(std430, binding = 6) buffer BlueBuffer {
        int blue[];
    };

    layout(std430, binding = 7) buffer IndexBuffer {
        uint ind_out[];
    };

    uniform sampler1D hsvTexture;

    uniform vec2 numBins;
    uniform vec4 bounds; // Min and max for X,Y as [[xmin, ymin], [xmax, ymax]]
    uniform vec2 colorMinMax;
    uniform float iMax;

    vec3 rgb_color;

    void main(void) {
        uint idx = gl_GlobalInvocationID.x;

        vec2 normalized = (position[idx].xy - bounds.xy) /
            (bounds.zw - bounds.xy);
        vec2 binIndex = vec2(normalized.x * numBins.x,
                             normalized.y * numBins.y);
        uint flatIndex = uint(binIndex.x) + uint(binIndex.y * numBins.x);

        float hue = (color[idx] - colorMinMax.x) /
            (colorMinMax.y - colorMinMax.x);
        if (state[idx] > 0) {
                rgb_color =  intensity[idx] / iMax * 10000. *
                texture(hsvTexture, hue*0.85).rgb;
        } else {
                rgb_color = vec3(0, 0, 0);
        };

        atomicAdd(red[flatIndex], int(rgb_color.x));
        atomicAdd(green[flatIndex], int(rgb_color.y));
        atomicAdd(blue[flatIndex], int(rgb_color.z));
        ind_out[idx] = flatIndex;
    }
    '''

    fragment_source = '''
    #version 410 core

    in vec4 gs_out_color;

    out vec4 fragColor;

    void main()
    {
      fragColor = gs_out_color;
    }
    '''

    vertex_source_point = '''
    #version 410 core

    layout(location = 0) in vec3 position_start;
    layout(location = 1) in float colorAxis;
    layout(location = 2) in float state;
    layout(location = 3) in float intensity;

    uniform sampler1D hsvTexture;

    uniform float opacity;
    uniform float iMax;
    uniform int isLost;
    uniform vec2 colorMinMax;

    uniform mat4 mPV;
    uniform mat4 modelStart;

    uniform float pointSize;
    uniform vec4 gridMask;
    uniform vec4 gridProjection;

    out vec4 vs_out_color;

    float hue;
    vec4 hrgb;
    float intensity_v;

    void main()
    {

     gl_Position = mPV * (gridMask * (modelStart * vec4(position_start, 1.0)) +
                          gridProjection);
     gl_PointSize = pointSize;

//     hue = (colorAxis - colorMinMax.x) / (colorMinMax.y - colorMinMax.x);
//     hrgb = vec4(texture(hsvTexture, hue).rgb, opacity*intensity/iMax);

     if (isLost > 0) {
             hrgb = vec4(0.9, 0., 0., 0.1);}
     else {
         hue = (colorAxis - colorMinMax.x) / (colorMinMax.y - colorMinMax.x);
         intensity_v = opacity*intensity/iMax;
         hrgb = vec4(texture(hsvTexture, hue*0.85).rgb, intensity_v);}

     vs_out_color = hrgb;

    }
    '''

    fragment_source_point = '''
    #version 410 core

    in vec4 vs_out_color;

    out vec4 fragColor;

    void main()
    {

      fragColor = vs_out_color;

    }
    '''


class OEMesh3D():
    """Container for an optical element mesh"""

    vertex_source = '''
    #version 410 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normals;

    out vec4 w_position;  // position of the vertex in world space
    out vec3 varyingNormalDirection;  // surface normal vector in world space
    out vec3 localPos;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;

    uniform mat3 m_3x3_inv_transp;
    //varying vec2 texUV;

    void main()
    {
      localPos = position;
      w_position = model * vec4(position, 1.);
      varyingNormalDirection = normalize(m_3x3_inv_transp * normals);

      //mat4 mvp = projection*view*model;
      gl_Position = projection*view*model * vec4(position, 1.);

    }
    '''

    fragment_source = '''
    #version 410 core

    in vec4 w_position;  // position of the vertex in world space
    in vec3 varyingNormalDirection;  // surface normal vector in world space
    in vec3 localPos;

    //uniform mat4 model;
    //uniform mat4 projection;
    //uniform mat4 view;

    uniform mat4 v_inv;
    uniform vec2 texlimitsx;
    uniform vec2 texlimitsy;
    uniform vec2 texlimitsz;
    uniform sampler2D u_texture;
    uniform float opacity;
    uniform float surfOpacity;
    uniform int isApt;

    out vec4 fragColor;
    float texOpacity;

    vec2 texUV;
    vec4 histColor;

    struct lightSource
    {
      vec4 position;
      vec4 diffuse;
      vec4 specular;
      float constantAttenuation, linearAttenuation, quadraticAttenuation;
      float spotCutoff, spotExponent;
      vec3 spotDirection;
    };

    lightSource light0 = lightSource(
      vec4(0.0,  0.0,  3.0, 0.0),
      vec4(0.6,  0.6,  0.6, 1.0),
      vec4(1.0,  1.0,  1.0, 1.0),
      0.0, 1.0, 0.0,
      90.0, 0.0,
      vec3(0.0, 0.0, -1.0)
    );
    vec4 scene_ambient = vec4(0.5, 0.5, 0.5, 1.0);

    struct material
    {
      vec4 ambient;
      vec4 diffuse;
      vec4 specular;
      float shininess;
    };

    uniform material frontMaterial;

    void main()
    {
      vec3 normalDirection = normalize(varyingNormalDirection);
      vec3 viewDirection = normalize(vec3(v_inv * vec4(0.0, 0.0, 0.0, 1.0) -
                                          w_position));
      vec3 lightDirection;
      float attenuation;

      if (0.0 == light0.position.w) // directional light?
        {
          attenuation = 1.0; // no attenuation
          lightDirection = normalize(vec3(light0.position));
        }
      else // point light or spotlight (or other kind of light)
        {
          vec3 positionToLightSource = -viewDirection;
          //vec3 positionToLightSource = vec3(light0.position - w_position);
          float distance = length(positionToLightSource);
          lightDirection = normalize(positionToLightSource);
          attenuation = 1.0 / (light0.constantAttenuation
                               + light0.linearAttenuation * distance
                               + light0.quadraticAttenuation * distance *
                               distance);

          if (light0.spotCutoff <= 90.0) // spotlight?
        {
          float clampedCosine = max(0.0, dot(-lightDirection,
                                             light0.spotDirection));
          if (clampedCosine < cos(radians(light0.spotCutoff)))
            {
              attenuation = 0.0;
            }
          else
            {
              attenuation = attenuation * pow(clampedCosine,
                                              light0.spotExponent);
            }
        }
        }

      vec3 ambientLighting = vec3(scene_ambient) * vec3(frontMaterial.ambient);

      vec3 diffuseReflection = attenuation
        * vec3(light0.diffuse) * vec3(frontMaterial.diffuse)
        * max(0.0, dot(normalDirection, lightDirection));

      vec3 specularReflection;
      if (dot(normalDirection, lightDirection) < 0.0)
        {
          specularReflection = vec3(0.0, 0.0, 0.0); // no specular reflection
        }
      else // light source on the right side
        {
          specularReflection = attenuation * vec3(light0.specular) *
          vec3(frontMaterial.specular) *
          pow(max(0.0, dot(reflect(-lightDirection, normalDirection),
                           viewDirection)), frontMaterial.shininess);
        }
     texUV = vec2((localPos.x-texlimitsx.x)/(texlimitsx.y-texlimitsx.x),
                 (localPos.y-texlimitsy.x)/(texlimitsy.y-texlimitsy.x));

     texOpacity = surfOpacity;
     if (texUV.x>0 && texUV.x<1 && texUV.y>0 && texUV.y<1 &&
         localPos.z<texlimitsz.y && localPos.z>texlimitsz.x) {
         histColor = texture(u_texture, texUV);
         if (isApt>0) texOpacity = 0.; }
     else
         histColor = vec4(0, 0, 0, 0);
      fragColor = vec4(ambientLighting + diffuseReflection +
                       specularReflection, texOpacity) + histColor*opacity;
    }
    '''

    vertex_source_flat = '''
    #version 410 core

    struct Material {
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        float shininess;
    };

    struct Light {
        vec3 position;
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
    };

    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normals;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;

    uniform vec2 texlimitsx;
    uniform vec2 texlimitsy;
    //uniform vec2 texlimitsz;

    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform Material material;
    uniform Light light;
    uniform float surfOpacity;

    //uniform vec3 lightColor;

    out vec4 color_out;
    out vec2 texUV;


    void main()
    {
        vec4 worldCoord = model * vec4(position, 1.0);
        gl_Position = projection * view * worldCoord;

        vec3 ambient = light.ambient * material.ambient;
        vec3 norm = vec3(model * vec4(normals, 0));
        vec3 lightDir = vec3(0, 0, -1);
        //vec3 lightDir = normalize(lightPos - worldCoord.xyz);

        float diff = max(dot(norm, -lightDir), 0.0);
        vec3 diffuse = light.diffuse * (diff * material.diffuse);

        vec3 viewDir = normalize(viewPos - worldCoord.xyz);
        vec3 reflectDir = reflect(lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0),
                         material.shininess);
        vec3 specular = light.specular * (spec * material.specular);
        //vec3 result = ambient + diffuse + specular;
        vec3 result = ambient + specular;
        color_out = vec4(result, surfOpacity);

        texUV = vec2((position.x-texlimitsx.x)/(texlimitsx.y-texlimitsx.x),
                     (position.y-texlimitsy.x)/(texlimitsy.y-texlimitsy.x));
    }
    '''

    fragment_source_flat = '''
    #version 410 core

    in vec4 color_out;
    in vec2 texUV;

    uniform sampler2D u_texture;
    uniform float opacity;
    vec4 histColor;

    out vec4 fragColor;

    void main()
    {

     if (texUV.x>0 && texUV.x<1 && texUV.y>0 && texUV.y<1)
         histColor = texture(u_texture, texUV);
     else
         histColor = vec4(0, 0, 0, 0);

     //gl_FragColor = vec4(color_out+histColor);
     fragColor = color_out;
    }
    '''

    vertex_contour = '''
    #version 410 core
    layout(location = 0) in vec3 position;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;

    void main()
    {
        gl_Position = projection*view*model*vec4(position, 1.);
    }
    '''

    fragment_contour = '''
    #version 410 core
    uniform vec4 cColor;
    out vec4 fragColor;

    void main()
    {
        fragColor = cColor;
    }
    '''

    vertex_magnet = """
    #version 410 core
    layout (location = 0) in vec3 inPosition;
    layout (location = 1) in vec3 inNormal;
    layout (location = 2) in vec3 instancePosition;
    layout (location = 3) in vec3 instanceColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 scale;
    uniform mat3 m_3x3_inv_transp;

    out vec4 w_position;
    out vec3 varyingNormalDirection;
    out vec3 DiffuseColor;

    void main()
    {
        vec4 scaledPos = vec4(inPosition, 1.0) * scale;
        w_position = model * vec4(scaledPos + vec4(instancePosition, 1.0));

        gl_Position = projection * view * w_position;

        varyingNormalDirection = normalize(m_3x3_inv_transp * inNormal);
        DiffuseColor = instanceColor;


    }
    """

    fragment_magnet = """
    #version 410 core

    in vec4 w_position;
    in vec3 varyingNormalDirection;
    in vec3 DiffuseColor;
    out vec4 fragColor;

    uniform mat4 v_inv;

    struct lightSource
    {
      vec4 position;
      vec4 diffuse;
      vec4 specular;
      float constantAttenuation, linearAttenuation, quadraticAttenuation;
      float spotCutoff, spotExponent;
      vec3 spotDirection;
    };

    lightSource light0 = lightSource(
      vec4(0.0,  0.0,  3.0, 0.0),
      vec4(0.6,  0.6,  0.6, 1.0),
      vec4(1.0,  1.0,  1.0, 1.0),
      0.0, 1.0, 0.0,
      90.0, -0.7,
      vec3(0.0, 0.0, -1.0)
    );

    vec4 scene_ambient = vec4(0.5, 0.5, 0.5, 1.0);

    struct material
    {
      vec4 ambient;
      vec4 diffuse;
      vec4 specular;
      float shininess;
    };

    uniform material frontMaterial;

    void main()
    {

      vec3 normalDirection = normalize(varyingNormalDirection);
      vec3 viewDirection = normalize(vec3(v_inv * vec4(0.0, 0.0, 0.0, 1.0) -
                                          w_position));
      vec3 lightDirection;
      float attenuation;

      if (0.0 == light0.position.w) // directional light?
        {
          attenuation = 1.0; // no attenuation
          lightDirection = normalize(vec3(light0.position));
        }
      else // point light or spotlight (or other kind of light)
        {
          vec3 positionToLightSource = -viewDirection;
          //vec3 positionToLightSource = vec3(light0.position - w_position);
          float distance = length(positionToLightSource);
          lightDirection = normalize(positionToLightSource);
          attenuation = 1.0 / (light0.constantAttenuation
                               + light0.linearAttenuation * distance
                               + light0.quadraticAttenuation * distance *
                               distance);

          if (light0.spotCutoff <= 90.0) // spotlight?
        {
          float clampedCosine = max(0.0, dot(-lightDirection,
                                             light0.spotDirection));
          if (clampedCosine < cos(radians(light0.spotCutoff)))
            {
              attenuation = 0.0;
            }
          else
            {
              attenuation = attenuation * pow(clampedCosine,
                                              light0.spotExponent);
            }
        }
        }

      vec3 ambientLighting = DiffuseColor * vec3(frontMaterial.ambient);

      vec3 diffuseReflection = attenuation
        * vec3(light0.diffuse) * vec3(frontMaterial.diffuse) * DiffuseColor
        * max(0.0, dot(normalDirection, lightDirection));

      vec3 specularReflection;
      if (dot(normalDirection, lightDirection) < 0.0)
        {
          specularReflection = vec3(0.0, 0.0, 0.0); // no specular reflection
        }
      else // light source on the right side
        {
          specularReflection = attenuation * vec3(light0.specular) *
          vec3(frontMaterial.specular) *
          pow(max(0.0, dot(reflect(-lightDirection, normalDirection),
                           viewDirection)), frontMaterial.shininess);
        }

        fragColor = vec4(ambientLighting + diffuseReflection +
                         specularReflection, 1.0);

    }
    """

    def __init__(self, parentOE, parentWidget):
        self.emptyTex = qt.QOpenGLTexture(
                qt.QImage(np.zeros((256, 256, 4)),
                          256, 256, qt.QImage.Format_RGBA8888))
        self.defaultLimits = np.array([[-1.]*3, [1.]*3])
#        print("def shape", self.defaultLimits.shape)
#        texture.save(str(oe.name)+"_beam_hist.png")
#        if hasattr(meshObj, 'beamTexture'):
#            oe.beamTexture.setData(texture)
#        meshObj.beamTexture[nsIndex] = qg.QOpenGLTexture(texture)
#        meshObj.beamLimits[nsIndex] = beamLimits
        self.oe = parentOE
        self.parent = parentWidget
        self.isStl = False
        self.shader = {}
        self.shader_c = {}
        self.vao = {}
        self.vao_c = {}
        self.ibo = {}
        self.beamTexture = {}
        self.beamLimits = {}
        self.transMatrix = {}
        self.arrLengths = {}

        self.vbo_vertices = {}
        self.vbo_normals = {}
        self.vbo_positions = {}
        self.vbo_colors = {}

        self.vbo_contour = {}

        if self.parent is not None:
            self.oeThickness = self.parent.oeThickness
            self.tiles = self.parent.tiles
        else:
            self.oeThickness = 5
            self.tiles = [25, 25]
        self.showLocalAxes = False
        self.isEnabled = False
        self.stencilNum = 0

        self.cube_vertices = np.array([
            # Positions           Normals
            # Front face
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
            0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
            0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,

            # Back face
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
            0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
            0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
            0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,

            # Left face
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,

            # Right face
            0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
            0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
            0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
            0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
            0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
            0.5,  0.5,  0.5,  1.0,  0.0,  0.0,

            # Bottom face
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
            0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
            0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
            0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,

            # Top face
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
            0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
            0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
            0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
        ], dtype=np.float32)

    def update_surface_mesh(self, is2ndXtal=False):
        pass

    def update_transformation_matrix(self):
        for elIndex in range(len(self.transMatrix)):
            self.transMatrix[elIndex] = self.get_loc2glo_transformation_matrix(
                    self.oe, is2ndXtal=bool(elIndex))

    @staticmethod
    def get_loc2glo_transformation_matrix(oe, is2ndXtal=False):
        if is_oe(oe):
            dx, dy, dz = 0, 0, 0
            extraAnglesSign = 1.  # only for pitch and yaw

            if is_dcm(oe):
                if is2ndXtal:
                    try:
                        pitch = -oe.pitch - oe.bragg + oe.cryst2pitch +\
                            oe.cryst2finePitch
                    except (ValueError, TypeError):
                        pitch = 0
                        print("Unresolved value in", oe.name, "pitch")
                    roll = oe.roll + oe.cryst2roll + oe.positionRoll
                    yaw = -oe.yaw
                    dx = -oe.dx
                    dy = oe.cryst2longTransl
                    dz = -oe.cryst2perpTransl
                    extraAnglesSign = -1.
                else:
                    try:
                        pitch = oe.pitch + oe.bragg
                    except (ValueError, TypeError):
                        pitch = 0
                        print("Unresolved value in", oe.name, "pitch")
                    roll = oe.roll + oe.positionRoll + oe.cryst1roll
                    yaw = oe.yaw
                    dx = oe.dx
            else:
                pitch = oe.pitch
                roll = oe.roll + oe.positionRoll
                yaw = oe.yaw

            rotAx = {'x': pitch,
                     'y': roll,
                     'z': yaw}
            extraRotAx = {'x': extraAnglesSign*oe.extraPitch,
                          'y': oe.extraRoll,
                          'z': extraAnglesSign*oe.extraYaw}

            rotSeq = (oe.rotationSequence[slice(1, None, 2)])[::-1]
            extraRotSeq = (oe.extraRotationSequence[slice(1, None, 2)])[::-1]

            try:
                rotation = (scprot.from_euler(
                        rotSeq, [rotAx[i] for i in rotSeq])).as_quat()
                extraRot = (scprot.from_euler(
                        extraRotSeq,
                        [extraRotAx[i] for i in extraRotSeq])).as_quat()
            except ValueError:
                rotation = (scprot.from_euler(
                        rotSeq, [0, 0, 0])).as_quat()
                extraRot = (scprot.from_euler(
                        extraRotSeq, [0, 0, 0])).as_quat()
                print("Unresolved values in", oe.name, "rotation sequence:",
                      [rotAx[i] for i in rotSeq])
            rotation = [rotation[-1], rotation[0], rotation[1], rotation[2]]
            extraRot = [extraRot[-1], extraRot[0], extraRot[1], extraRot[2]]

            # 1. Only for DCM - translate to 2nd crystal position
            m2ndXtalPos = qt.QMatrix4x4()
            m2ndXtalPos.translate(dx, dy, dz)

            # 2. Apply extra rotation
            mExtraRot = qt.QMatrix4x4()
            mExtraRot.rotate(qt.QQuaternion(*extraRot))

            # 3. Apply rotation
            mRotation = qt.QMatrix4x4()
            mRotation.rotate(qt.QQuaternion(*rotation))

            # 4. Only for DCM - flip 2nd crystal
            m2ndXtalRot = qt.QMatrix4x4()
            if is_dcm(oe):
                if is2ndXtal:
                    m2ndXtalRot.rotate(180, 0, 1, 0)

            # 5. Move to position in global coordinates
            mTranslation = qt.QMatrix4x4()
            try:
                mTranslation.translate(*oe.center)
            except TypeError:  # Unresolved 'auto' in the center list
                print("Unresolved values in", oe.name, "center:", oe.center)
            orientation = mTranslation * m2ndXtalRot * mRotation *\
                mExtraRot * m2ndXtalPos
        elif is_screen(oe) or is_aperture(oe):  # Screens, Apertures
            bStart = np.column_stack(([1, 0, 0], [0, 0, 1], [0, -1, 0]))
#            bStart = np.column_stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
#            bStart = np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
            bEnd = np.column_stack((oe.x / np.linalg.norm(oe.x),
                                    oe.y / np.linalg.norm(oe.y),
                                    oe.z / np.linalg.norm(oe.z)))

            rotationQ = basis_rotation_q(bStart, bEnd)

            mRotation = qt.QMatrix4x4()
            mRotation.rotate(qt.QQuaternion(*rotationQ))

            posMatr = qt.QMatrix4x4()
            try:
                posMatr.translate(*oe.center)
            except TypeError:  # Unresolved 'auto' in the center list
                print("Unresolved values in", oe.name, "center:", oe.center)
            orientation = posMatr*mRotation
        else:  # source
            posMatr = qt.QMatrix4x4()
            posMatr.translate(*oe.center)
            orientation = posMatr

        return orientation

    def prepare_surface_mesh(self, nsIndex=0, updateMesh=False,
                             autoSize=False):
        def get_thickness():

            thsrc = self.parent or self
            if thsrc.oeThicknessForce is not None:
                return thsrc.oeThicknessForce
            thickness = thsrc.oeThickness

            if isScreen or isAperture:
                return 0
            if isPlate:
                if self.oe.t is not None:
                    thickness = self.oe.t
                    if hasattr(self.oe, 'zmax'):
                        if self.oe.zmax is not None:
                            thickness += self.oe.zmax
                            if isinstance(self.oe, roes.DoubleParaboloidLens):
                                thickness += self.oe.zmax
                    return thickness
            if hasattr(self.oe, "material"):
                if self.oe.material is not None:
                    thickness = self.oeThickness
                    if hasattr(self.oe.material, "t"):
                        thickness = self.oe.material.t if\
                            self.oe.material.t is not None else thickness
                    elif isinstance(self.oe.material, rmats.Multilayer):
                        if self.oe.material.substrate is not None:
                            if hasattr(self.oe.material.substrate, 't'):
                                if self.oe.material.substrate.t is not None:
                                    thickness = self.oe.material.substrate.t

            return thickness

        gl.glGetError()

        is2ndXtal = False

        if is_oe(self.oe):
            is2ndXtal = bool(nsIndex)

        self.transMatrix[int(is2ndXtal)] =\
            self.get_loc2glo_transformation_matrix(
                self.oe, is2ndXtal=is2ndXtal)

        if nsIndex in self.vao.keys():  # Updating existing
            vao = self.vao[nsIndex]
        else:
            vao = qt.QOpenGLVertexArrayObject()
            vao.create()
            self.vao[nsIndex] = None  # Will be updated after generation

        if hasattr(self.oe, 'stl_mesh') and hasattr(self.oe, 'points'):
            self.isStl = True

            self.vbo_vertices[nsIndex] = create_qt_buffer(
                    self.oe.points.copy())
            self.vbo_normals[nsIndex] = create_qt_buffer(
                    self.oe.normals.copy())
            self.arrLengths[nsIndex] = len(self.oe.points)
            gl.glGetError()

            vao.bind()

            self.vbo_vertices[nsIndex].bind()
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(0)
            self.vbo_vertices[nsIndex].release()

            self.vbo_normals[nsIndex].bind()
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(1)
            self.vbo_normals[nsIndex].release()

            vao.release()

            self.vao[nsIndex] = vao
            self.ibo[nsIndex] = None  # Check if works with glDrawElements
            return

        isPlate = is_plate(self.oe)
        isScreen = is_screen(self.oe)
        isAperture = is_aperture(self.oe)

        thickness = get_thickness()

        self.bBox = np.zeros((3, 2))
        self.bBox[:, 0] = 1e10
        self.bBox[:, 1] = -1e10

        # TODO: Consider plates
        oeShape = self.oe.shape if hasattr(self.oe, 'shape') else 'rect'
        oeDx = self.oe.dx if hasattr(self.oe, 'dx') else 0
        isOeParametric = self.oe.isParametric if hasattr(
                self.oe, 'isParametric') else False

        yDim = 1
        if isScreen:
            if autoSize and hasattr(self.oe, 'footprint') and len(
                    self.oe.footprint) > 0:
                xLimits = self.oe.footprint[nsIndex][:, 0]
            elif self.oe.limPhysX is not None and np.sum(np.abs(
                    self.oe.limPhysX)) > 0:
                xLimits = self.oe.limPhysX if isinstance(
                    self.oe.limPhysX, list) else self.oe.limPhysX.tolist()
            else:
                xLimits = [-10, 10]

            if autoSize and hasattr(self.oe, 'footprint') and len(
                    self.oe.footprint) > 0:
                yLimits = self.oe.footprint[nsIndex][:, 2]
            elif self.oe.limPhysY is not None and np.sum(np.abs(
                    self.oe.limPhysY)) > 0:
                yLimits = self.oe.limPhysY if isinstance(
                    self.oe.limPhysY, list) else self.oe.limPhysY.tolist()
            else:
                yLimits = [-10, 10]
            yDim = 2
        elif isAperture:
            renderStyle = getattr(self.oe, 'renderStyle', 'mask')

            bt = 5.  # TODO: Must be configurable in glow UI
            defaultWidth = 10

            if len(set(self.oe.kind) & {'left', 'right'}) > 1:
                awidth = np.abs(self.oe.opening[self.oe.kind.index('left')] -
                                self.oe.opening[self.oe.kind.index('right')])
                awidth = 0.5*awidth + bt if renderStyle == 'mask' else\
                    max(awidth, defaultWidth)
            else:
                awidth = defaultWidth

            if len(set(self.oe.kind) & {'top', 'bottom'}) > 1:
                aheight = np.abs(self.oe.opening[self.oe.kind.index('top')] -
                                 self.oe.opening[self.oe.kind.index('bottom')])
                aheight = 0.5*aheight if renderStyle == 'mask' else\
                    max(aheight, defaultWidth)
            else:
                aheight = defaultWidth

            if str(nsIndex) == 'left':
                xLimits = [self.oe.opening[self.oe.kind.index('left')] - bt,
                           self.oe.opening[self.oe.kind.index('left')]]
                yLimits = [-aheight, aheight]
            elif str(nsIndex) == 'right':
                xLimits = [self.oe.opening[self.oe.kind.index('right')],
                           self.oe.opening[self.oe.kind.index('right')] + bt]
                yLimits = [-aheight, aheight]
            elif str(nsIndex) == 'bottom':  # Limits are inverted for rendering
                xLimits = [-awidth, awidth]
                yLimits = [-self.oe.opening[self.oe.kind.index('bottom')],
                           -self.oe.opening[self.oe.kind.index('bottom')] + bt]
            elif str(nsIndex) == 'top':
                xLimits = [-awidth, awidth]
                yLimits = [-self.oe.opening[self.oe.kind.index('top')] - bt,
                           -self.oe.opening[self.oe.kind.index('top')]]
            yDim = 2
        elif is2ndXtal:
            xLimits = list(self.oe.limPhysX2)
            yLimits = list(self.oe.limPhysY2)
        else:
            xLimits = list(self.oe.limPhysX)
            yLimits = list(self.oe.limPhysY)

        isClosedSurface = False
        if np.all(np.abs(xLimits) == raycing.maxHalfSizeOfOE):
            isClosedSurface = isinstance(self.oe, roes.SurfaceOfRevolution)
            if autoSize and hasattr(self.oe, 'footprint') and len(
                    self.oe.footprint) > 0:
                xLimits = self.oe.footprint[nsIndex][:, 0]
            elif self.oe.limOptX is not None and not\
                    np.all(np.abs(self.oe.limOptX) == raycing.maxHalfSizeOfOE):
                xLimits = list(self.oe.limOptX)
        if np.all(np.abs(yLimits) == raycing.maxHalfSizeOfOE):
            if autoSize and hasattr(self.oe, 'footprint') and len(
                    self.oe.footprint) > 0:
                yLimits = self.oe.footprint[nsIndex][:, yDim]
            elif self.oe.limOptY is not None and not\
                    np.all(np.abs(self.oe.limOptY) == raycing.maxHalfSizeOfOE):
                yLimits = list(self.oe.limOptY)

        self.xLimits = copy.deepcopy(xLimits)
        self.yLimits = copy.deepcopy(yLimits)

        tiles = self.parent.tiles if self.parent is not None else self.tiles
        localTiles = np.array(tiles)

        if oeShape == 'round':
            rX = np.abs((xLimits[1] - xLimits[0]))*0.5
            rY = np.abs((yLimits[1] - yLimits[0]))*0.5
            cX = (xLimits[1] + xLimits[0])*0.5
            cY = (yLimits[1] + yLimits[0])*0.5
            xLimits = [0, 1.]
            yLimits = [0, 2*np.pi]
            localTiles[1] *= 3

        if isClosedSurface:
            # the limits are in parametric coordinates
            xLimits = yLimits  # s
            yLimits = [0, 2*np.pi]  # phi
            localTiles[1] *= 3

        xGridOe = np.linspace(xLimits[0], xLimits[1],
                              localTiles[0]) + oeDx
        yGridOe = np.linspace(yLimits[0], yLimits[1], localTiles[1])

        xv, yv = np.meshgrid(xGridOe, yGridOe)

        sideL = np.vstack((xv[:, 0], yv[:, 0])).T
        sideR = np.vstack((xv[:, -1], yv[:, -1])).T
        sideF = np.vstack((xv[0, :], yv[0, :])).T
        sideB = np.vstack((xv[-1, :], yv[-1, :])).T

        if oeShape == 'round':
            xv, yv = rX*xv*np.cos(yv)+cX, rY*xv*np.sin(yv)+cY

        xv = xv.flatten()
        yv = yv.flatten()

        if is2ndXtal:
            zExt = '2'
        else:
            zExt = '1' if hasattr(self.oe, 'local_z1') else ''

        if isScreen:
            local_n = lambda x, y: [0, 0, 1]
            local_z = lambda x, y: np.zeros_like(x)
        elif isAperture:
            apThick = 0.1
            local_n = lambda x, y: [0, 0, 1]
            if str(nsIndex) in ['left', 'right'] and renderStyle != 'mask':  # Depth for rendering only
                local_z = lambda x, y: 0 * np.ones_like(x)  # actual thickness
                thickness = -apThick*0.5  # Inverted position of the back side
            else:
                zsurf = apThick*0.5 if renderStyle == 'mask' else 0
                local_z = lambda x, y: zsurf * np.ones_like(x)
                thickness = apThick*0.5  # Inverted position of the back side
        else:
            local_z = getattr(self.oe, 'local_r{}'.format(zExt)) if\
                self.oe.isParametric else getattr(self.oe,
                                                  'local_z{}'.format(zExt))
            local_n = getattr(self.oe, 'local_n{}'.format(zExt))

        xv = np.copy(xv)
        yv = np.copy(yv)
        zv = np.zeros_like(xv)
        if isinstance(self.oe, roes.SurfaceOfRevolution):
            # at z=0 (axis of rotation) phi is undefined, therefore:
            zv -= 100.

        if isOeParametric and not isClosedSurface:
            xv, yv, zv = self.oe.xyz_to_param(xv, yv, zv)

        zv = np.array(local_z(xv, yv))
        nv = np.array(local_n(xv, yv)).T

        if len(nv) == 3:  # flat
            nv = np.ones_like(zv)[:, np.newaxis] * np.array(nv)

        if isOeParametric and not isClosedSurface:
            xv, yv, zv = self.oe.param_to_xyz(xv, yv, zv)

#        zmax = np.max(zv)
#        zmin =
#        self.bBox[:, 1] = yLimit

        if oeShape == 'round':
            xC, yC = rX*sideR[:, 0]*np.cos(sideR[:, -1]) +\
                     cX, rY*sideR[:, 0]*np.sin(sideR[:, -1]) + cY
            zC = np.array(local_z(xC, yC))
            if isOeParametric:
                xC, yC, zC = self.oe.param_to_xyz(xC, yC, zC)

        points = np.vstack((xv, yv, zv)).T
        surfmesh = {}

        triS = Delaunay(points[:, :-1])

        if not isPlate:
            bottomPoints = points.copy()
            bottomPoints[:, 2] = -thickness
            bottomNormals = np.zeros((len(points), 3))
            bottomNormals[:, 2] = -1

        # side: x, y
        zs = []
        xs = []
        ys = []

        for elSide in (sideL, sideR, sideF, sideB):
            tmpx, tmpy, tmpz =\
                elSide[:, 0], elSide[:, -1], np.zeros_like(elSide[:, 0])
            if isOeParametric and not isClosedSurface:
                tmpx, tmpy, tmpz = self.oe.xyz_to_param(tmpx, tmpy, tmpz)
            tmpz = np.array(local_z(tmpx, tmpy))
            if isOeParametric and not isClosedSurface:
                tmpx, tmpy, tmpz = self.oe.param_to_xyz(tmpx, tmpy, tmpz)
            xs.append(tmpx)
            ys.append(tmpy)
            zs.append(tmpz)

        zL = zs[0]
        zR = zs[1]
        zF = zs[2]
        zB = zs[3]

        tL = np.vstack((xs[0], ys[0], np.ones_like(zL)*thickness))
        bottomLine = zL - thickness if isPlate else -np.ones_like(zL)*thickness
        tL = np.hstack((tL, np.vstack((np.flip(xs[0]), np.flip(ys[0]),
                                       -np.ones_like(zL)*thickness)))).T
        normsL = np.zeros((len(zL)*2, 3))
        normsL[:, 0] = -1
        if not (isScreen):  # or isAperture):
            try:
                triLR = Delaunay(tL[:, [1, -1]])  # Works for round elements
                useLR = True
            except:
                useLR = False
        tL[:len(zL), 2] = zL
        tL[len(zL):, 2] = bottomLine

        tR = np.vstack((xs[1], ys[1], zR))
        bottomLine = zR - thickness if isPlate else -np.ones_like(zR)*thickness
        tR = np.hstack((tR, np.vstack((np.flip(xs[1]), np.flip(ys[1]),
                                       bottomLine)))).T
        normsR = np.zeros((len(zR)*2, 3))
        normsR[:, 0] = 1

        tF = np.vstack((xs[2], ys[2], np.ones_like(zF)*thickness))
        bottomLine = zF - thickness if isPlate else -np.ones_like(zF)*thickness
        tF = np.hstack((tF, np.vstack((np.flip(xs[2]), np.flip(ys[2]),
                                       bottomLine)))).T
        normsF = np.zeros((len(zF)*2, 3))
        normsF[:, 1] = -1
        if not (isScreen):  # or isAperture):
            try:
                triFB = Delaunay(tF[:, [0, -1]])
                useFB = True
            except:
                useFB = False
        tF[:len(zF), 2] = zF

        if oeShape == 'round':
            tB = np.vstack((xC, yC, zC))
            bottomLine = zC - thickness if isPlate else\
                -np.ones_like(zC)*thickness
            tB = np.hstack((tB, np.vstack((xC, np.flip(yC), bottomLine)))).T
            normsB = np.vstack((tB[:, 0], tB[:, 1], np.zeros_like(tB[:, 0]))).T
            norms = np.linalg.norm(normsB, axis=1, keepdims=True)
            normsB /= norms
        else:
            tB = np.vstack((xs[3], ys[3], zB))
            bottomLine = zB - thickness if isPlate else\
                -np.ones_like(zB)*thickness
            tB = np.hstack((tB, np.vstack((np.flip(xs[3]), np.flip(ys[3]),
                                           bottomLine)))).T
            normsB = np.zeros((len(zB)*2, 3))
            normsB[:, 1] = 1

        allSurfaces = points
        allNormals = nv
        allIndices = triS.simplices.flatten()
        indArrOffset = len(points)

        # Bottom Surface, use is2ndXtal for plates
        if not (isPlate):  # or isScreen or isAperture):
            allSurfaces = np.vstack((allSurfaces, bottomPoints))
            allNormals = np.vstack((nv, bottomNormals))
            allIndices = np.hstack((allIndices,
                                    triS.simplices.flatten() + indArrOffset))
            indArrOffset += len(points)

        # Side Surface, do not plot for 2ndXtal of Plate
        if not ((isPlate and is2ndXtal) or isScreen):  # or isAperture):
            if oeShape == 'round':  # Side surface
                allSurfaces = np.vstack((allSurfaces, tB))
                allNormals = np.vstack((allNormals, normsB))
                allIndices = np.hstack((allIndices,
                                        triLR.simplices.flatten() +
                                        indArrOffset))
            else:
                if useLR:
                    allSurfaces = np.vstack((allSurfaces, tL, tR))
                    allNormals = np.vstack((allNormals, normsL, normsR))
                    allIndices = np.hstack((allIndices,
                                            triLR.simplices.flatten() +
                                            indArrOffset,
                                            triLR.simplices.flatten() +
                                            indArrOffset+len(tL)))
                    indArrOffset += len(tL)*2
                if useFB:
                    allSurfaces = np.vstack((allSurfaces, tF, tB))
                    allNormals = np.vstack((allNormals, normsF, normsB))
                    allIndices = np.hstack((allIndices,
                                            triFB.simplices.flatten() +
                                            indArrOffset,
                                            triFB.simplices.flatten() +
                                            indArrOffset+len(tF)))

        surfmesh['points'] = allSurfaces.copy()
        surfmesh['normals'] = allNormals.copy()
        surfmesh['indices'] = allIndices

        self.allSurfaces = allSurfaces
        self.allIndices = allIndices

        if oeShape == 'round':
            surfmesh['contour'] = tB
        else:
            surfmesh['contour'] = np.vstack((tL, tF, np.flip(tR, axis=0), tB))
        surfmesh['lentb'] = len(tB)

        self.bBox[:, 0] = np.min(surfmesh['contour'], axis=0)
        self.bBox[:, 1] = np.max(surfmesh['contour'], axis=0)

        if updateMesh:
            oldVBOpoints = self.vbo_vertices[nsIndex] if\
                nsIndex in self.vbo_vertices.keys() else None
            oldVBOnorms = self.vbo_normals[nsIndex] if\
                nsIndex in self.vbo_normals.keys() else None
            oldIBO = self.ibo[nsIndex] if nsIndex in self.ibo.keys() else None
            if oldVBOpoints is not None:
                oldVBOpoints.destroy()
                gl.glGetError()
            if oldVBOnorms is not None:
                oldVBOnorms.destroy()
                gl.glGetError()
            if oldIBO is not None:
                oldIBO.destroy()
                gl.glGetError()
            oldVBOpoints, oldVBOnorms, oldIBO = None, None, None
            # check existence
#            del self.vbo_vertices[nsIndex]
#            del self.vbo_normals[nsIndex]
#            del self.ibo[nsIndex]
            self.vbo_vertices[nsIndex] = None
            self.vbo_normals[nsIndex] = None
            self.ibo[nsIndex] = None
            if vao is not None:
                vao.destroy()
            gl.glGetError()
#            del self.vao[nsIndex]
            self.vao[nsIndex] = None

        self.vbo_vertices[nsIndex] = create_qt_buffer(surfmesh['points'])
        self.vbo_normals[nsIndex] = create_qt_buffer(surfmesh['normals'])
        self.ibo[nsIndex] = create_qt_buffer(surfmesh['indices'], isIndex=True)
        self.arrLengths[nsIndex] = len(surfmesh['indices'])
        gl.glGetError()

        if updateMesh:
            vao = qt.QOpenGLVertexArrayObject()
            vao.create()
            gl.glGetError()

        vao.bind()

        self.vbo_vertices[nsIndex].bind()
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        self.vbo_vertices[nsIndex].release()

        self.vbo_normals[nsIndex].bind()
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)
        self.vbo_normals[nsIndex].release()

        self.ibo[nsIndex].bind()

        vao.release()

        self.vao[nsIndex] = vao

        if isScreen:
            axisGridArray, gridLabels, precisionLabels =\
                CoordinateBox.make_plane([xLimits, yLimits])
            self.grid_vbo = {}
            self.grid_vbo['vertices'] = create_qt_buffer(axisGridArray)
            self.grid_vbo['gridLen'] = len(axisGridArray)
            self.grid_vbo['gridLabels'] = gridLabels
            self.grid_vbo['precisionLabels'] = precisionLabels

#        gridvao = qt.QOpenGLVertexArrayObject()
#        gridvao.create()

    def delete_mesh(self):
        for nsIndex in self.vao.keys():
            vao = self.vao[nsIndex]

            oldVBOpoints = self.vbo_vertices[nsIndex] if\
                nsIndex in self.vbo_vertices.keys() else None
            oldVBOnorms = self.vbo_normals[nsIndex] if\
                nsIndex in self.vbo_normals.keys() else None
            oldIBO = self.ibo[nsIndex] if nsIndex in self.ibo.keys() else None
            oldVBOcolors = self.vbo_colors[nsIndex] if\
                nsIndex in self.vbo_colors.keys() else None
            oldVBOpositions = self.vbo_positions[nsIndex] if\
                nsIndex in self.vbo_positions.keys() else None

            if oldVBOpoints is not None:
                oldVBOpoints.destroy()
                gl.glGetError()
            if oldVBOnorms is not None:
                oldVBOnorms.destroy()
                gl.glGetError()
            if oldIBO is not None:
                oldIBO.destroy()
                gl.glGetError()
            if oldVBOcolors is not None:
                oldVBOcolors.destroy()
                gl.glGetError()
            if oldVBOpositions is not None:
                oldVBOpositions.destroy()
                gl.glGetError()
            oldVBOpoints, oldVBOnorms, oldIBO = None, None, None
            oldVBOcolors, oldVBOpositions = None, None

            self.vbo_vertices[nsIndex] = None
            self.vbo_normals[nsIndex] = None
            self.ibo[nsIndex] = None
            self.vbo_colors[nsIndex] = None
            self.vbo_positions[nsIndex] = None
            if vao is not None:
                vao.destroy()
            gl.glGetError()
            self.vao[nsIndex] = None

    def generate_instance_data(self, num):
        period = self.oe.period if hasattr(self.oe, 'period') else 40  # [mm]
        gap = 10  # [mm]

        instancePositions = np.zeros((int(num*2), 3), dtype=np.float32)
        instanceColors = np.zeros((int(num*2), 3), dtype=np.float32)

        for n in range(int(num)):
            pos_x = 0
            dy = n - 0.5*num if num > 1 else 0
            pos_y = period * dy

            instancePositions[2*n] = (pos_x, pos_y, gap+0.5*self.mag_z_size)
            instancePositions[2*n+1] = (pos_x, pos_y, -gap-0.5*self.mag_z_size)
            isEven = (n % 2) == 0
            instanceColors[2*n] = (1.0, 0.0, 0.0) if isEven else\
                (0.0, 0.0, 1.0)
            instanceColors[2*n+1] = (0.0, 0.0, 1.0) if isEven else\
                (1.0, 0.0, 0.0)

        return instancePositions, instanceColors

    def prepare_magnets(self, updateMesh=False):
        self.transMatrix[0] = self.get_loc2glo_transformation_matrix(
            self.oe, is2ndXtal=False)
        nsIndex = 0  # to unify syntax

        num_poles = int(self.oe.n*2) if hasattr(self.oe, 'n') else 1
        self.mag_z_size = 20

        if updateMesh:
            if self.vbo_positions.get(nsIndex) is not None:
                self.vbo_positions[nsIndex].destroy()
                self.vbo_positions[nsIndex] = None
                gl.glGetError()
            if self.vbo_colors.get(nsIndex) is not None:
                self.vbo_colors[nsIndex].destroy()
                self.vbo_colors[nsIndex] = None
                gl.glGetError()
        else:
            self.vbo_vertices[nsIndex] = create_qt_buffer(
                    self.cube_vertices.reshape(-1, 6)[:, :3].copy())
            self.vbo_normals[nsIndex] = create_qt_buffer(
                    self.cube_vertices.reshape(-1, 6)[:, 3:].copy())

        instancePositions, instanceColors = self.generate_instance_data(
                num_poles)

        self.vbo_positions[nsIndex] = create_qt_buffer(
                instancePositions.copy())
        self.vbo_colors[nsIndex] = create_qt_buffer(instanceColors.copy())

        if not self.vao:
            self.vao[nsIndex] = qt.QOpenGLVertexArrayObject()
            self.vao[nsIndex].create()

        self.vao[nsIndex].bind()

        self.vbo_vertices[nsIndex].bind()
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        self.vbo_vertices[nsIndex].release()
        # Normal attribute
        self.vbo_normals[nsIndex].bind()
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        self.vbo_normals[nsIndex].release()
        # Instance arrays
        self.vbo_positions[nsIndex].bind()
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glVertexAttribDivisor(2, 1)
        self.vbo_positions[nsIndex].release()
        self.vbo_colors[nsIndex].bind()
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glVertexAttribDivisor(3, 1)
        self.vbo_colors[nsIndex].release()
        self.vao[nsIndex].release()
        self.num_poles = num_poles

    def render_surface(self, mMod, mView, mProj, oeIndex=0,
                       isSelected=False, shader=None):

        vao = self.vao[oeIndex]

        beamTexture = self.beamTexture[oeIndex] if len(self.beamTexture) > 0\
            else self.emptyTex  # what if there's no texture?
        beamLimits = self.beamLimits[oeIndex] if len(self.beamLimits) > 0\
            else self.defaultLimits

        xLimits, yLimits, zLimits =\
            beamLimits[:, 0], beamLimits[:, 1], beamLimits[:, 2]

        surfOpacity = 1.0
        if is_screen(self.oe):
            surfOpacity = 0.75
#        elif is_aperture(self.oe):
#            xLimits, yLimits = self.xLimits, self.yLimits
        oeOrientation = self.transMatrix[0] if is_aperture(self.oe) else\
            self.transMatrix[oeIndex]
        arrLen = self.arrLengths[oeIndex]

        shader.bind()
        vao.bind()

        shader.setUniformValue("model", mMod*oeOrientation)
        shader.setUniformValue("view", mView)
        shader.setUniformValue("projection", mProj)

        mvp = mMod*oeOrientation*mView
        shader.setUniformValue("m_3x3_inv_transp", mvp.normalMatrix())
        shader.setUniformValue("v_inv", mView.inverted()[0])

        shader.setUniformValue("texlimitsx", qt.QVector2D(*xLimits))
        shader.setUniformValue("texlimitsy", qt.QVector2D(*yLimits))
        shader.setUniformValue("texlimitsz", qt.QVector2D(*zLimits))

        if is_aperture(self.oe):
            mat = 'Cu'
        elif is_screen(self.oe):
            mat = 'Screen'
        else:
            mat = 'Si'

#        mat = 'Cu' if is_aperture(self.oe) else 'Si'

        ambient_in = ambient['selected'] if isSelected else ambient[mat]
        diffuse_in = diffuse[mat]
        specular_in = specular[mat]
        shininess_in = shininess[mat]

        shader.setUniformValue("frontMaterial.ambient", ambient_in)
        shader.setUniformValue("frontMaterial.diffuse", diffuse_in)
        shader.setUniformValue("frontMaterial.specular", specular_in)
        shader.setUniformValue("frontMaterial.shininess", shininess_in)

        shader.setUniformValue("opacity", float(self.parent.pointOpacity*2))
        shader.setUniformValue("surfOpacity", float(surfOpacity))
        shader.setUniformValue("isApt", 0)

        if beamTexture is not None:
            beamTexture.bind()

        if self.isStl:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, arrLen)
        else:
            gl.glDrawElements(gl.GL_TRIANGLES, arrLen,
                              gl.GL_UNSIGNED_INT, [])

        if beamTexture is not None:
            beamTexture.release()
        shader.release()
        vao.release()

    def render_magnets(self, mMod, mView, mProj, isSelected=False,
                       shader=None):
        if shader is None:
            return
        nsIndex = 0

        shader.bind()
        if not self.vao:
            return

        oeOrientation = self.transMatrix[0]

        self.vao[nsIndex].bind()

        shader.setUniformValue("model", mMod*oeOrientation)
        shader.setUniformValue("view", mView)
        shader.setUniformValue("projection", mProj)
        mModScale = qt.QMatrix4x4()
        mModScale.setToIdentity()
        mag_y = self.oe.period*0.75 if hasattr(self.oe, 'period') else 40
        mModScale.scale(*(np.array([mag_y, mag_y, self.mag_z_size])))
        shader.setUniformValue("scale", mModScale)

        mvp = mMod*mView
        shader.setUniformValue("m_3x3_inv_transp", mvp.normalMatrix())
        shader.setUniformValue("v_inv", mView.inverted()[0])

        mat = 'Si'
        ambient_in = ambient['selected'] if isSelected else ambient[mat]
        diffuse_in = diffuse[mat]
        specular_in = specular[mat]
        shininess_in = shininess[mat]

        shader.setUniformValue("frontMaterial.ambient", ambient_in)
        shader.setUniformValue("frontMaterial.diffuse", diffuse_in)
        shader.setUniformValue("frontMaterial.specular", specular_in)
        shader.setUniformValue("frontMaterial.shininess", shininess_in)

        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 36, self.num_poles*2)
        self.vao[nsIndex].release()
        shader.release()

    def export_stl(self, filename):
        if isSTLsupported and hasattr(self, 'allSurfaces'):
            try:
                triangles = self.allSurfaces[self.allIndices].reshape(-1, 3, 3)
                m = mesh.Mesh(np.zeros(triangles.shape[0],
                                       dtype=mesh.Mesh.dtype))
                m.vectors[:] = triangles
                m.update_normals()
                m.save(filename)
            except Exception as e:
                print(e)
