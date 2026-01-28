# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:27:01 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import os  # analysis:ignore
import numpy as np  # analysis:ignore
from freetype import Face, FT_LOAD_RENDER  # analysis:ignore
from matplotlib import font_manager  # analysis:ignore

from ._utils import (create_qt_buffer, update_qt_buffer, basis_rotation_q)  # analysis:ignore

from ..commons import qt  # analysis:ignore
from ..commons import gl  # analysis:ignore


class CoordinateBox():

    vertex_source = '''
    #version 410 core
    layout(location = 0) in vec3 position;
    uniform mat4 pvm;
    void main()
    {
      gl_Position = pvm * vec4(position, 1.0);
    }
    '''

    fragment_source = '''
    #version 410 core
    uniform float lineOpacity;
    uniform vec3 lineColor;
    out vec4 fragColor;
    void main()
    {
      fragColor = vec4(lineColor, lineOpacity);
    }
    '''

    orig_vertex_source = '''
    #version 410 core

    layout(location = 0) in vec4 position;
    layout(location = 1) in vec3 linecolor;
    //layout(location = 2) in mat4 rotation;

    uniform mat4 pvm;  // projection * view * model

    out vec3 out_color;
    void main()
    {
     out_color = linecolor;
     gl_Position = pvm * position;
     //gl_Position = pvm * (rotation*vec4(position, 1.0));
    }
    '''

    orig_fragment_source = '''
    #version 410 core
    uniform float lineOpacity;
    in vec3 out_color;
    out vec4 fragColor;
    void main()
    {
      fragColor = vec4(out_color, lineOpacity);
    }
    '''

    text_vertex_code = """
    #version 410 core

    layout(location = 0) in vec4 in_pos;

    out vec2 vUV;

    uniform mat4 model;
    //uniform mat4 projection;

    void main()
    {
        vUV         = in_pos.zw;
        gl_Position = model * vec4(in_pos.xy, 0.0, 1.0);
    }
    """

    text_fragment_code = """
    #version 410 core

    in vec2 vUV;

    uniform sampler2D u_texture;
    uniform vec3 textColor;
    uniform float textOpacity;

    out vec4 fragColor;

    void main()
    {
        vec2 uv = vUV.xy;
        float text = texture(u_texture, uv).r;
        fragColor = vec4(textColor, textOpacity) *
            vec4(text, text, text, text);
    }
    """

    def __init__(self, parent):

        self.parent = parent
        self.axPosModifier = np.ones(3)
        self.perspectiveEnabled = True
        self.shader = None
        self.origShader = None
        self.textShader = None
        self.vaoFrame = qt.QOpenGLVertexArrayObject()
        self.vaoFrame.create()

        self.vaoGrid = qt.QOpenGLVertexArrayObject()
        self.vaoGrid.create()

        self.vaoFineGrid = qt.QOpenGLVertexArrayObject()
        self.vaoFineGrid.create()

        self.vaoOrigin = qt.QOpenGLVertexArrayObject()
        self.vaoOrigin.create()

        self.vaoOrigin = qt.QOpenGLVertexArrayObject()
        self.vaoOrigin.create()

        self.vaoText = qt.QOpenGLVertexArrayObject()
        self.vaoText.create()

        self.characters = []
        self.fontSize = 32
        self.fontScale = 4.
        self.fontFile = 'FreeSans.ttf'

        self.z2y = qt.QMatrix4x4()
        self.z2y.rotate(90, 1, 0, 0)
        self.z2x = qt.QMatrix4x4()
        self.z2x.rotate(90, 0, -1, 0)

        self.initialGridLen = 0

#        self.vquad = [
#          # x   y  u  v
#            0, 1, 0, 0,
#            0,  0, 0, 1,
#            1,  0, 1, 1,
#            0, 1, 0, 0,
#            1,  0, 1, 1,
#            1, 1, 1, 0
#        ]

#        self.prepare_grid()

    @staticmethod
    def make_plane(limits):
        # working in local coordinates
        # limits: [[xmin, xmax], [ymin, ymax]]
        gridLabels = []
        precisionLabels = []
        limits = np.array(limits)

        frame = np.array([[limits[0, 0], 0, limits[1, 0]],  # xmin, ymin
                          [limits[0, 1], 0, limits[1, 0]],  # xmax, ymin
                          [limits[0, 1], 0, limits[1, 0]],  # xmax, ymin
                          [limits[0, 1], 0, limits[1, 1]],  # xmax, ymax
                          [limits[0, 1], 0, limits[1, 1]],  # xmax, ymax
                          [limits[0, 0], 0, limits[1, 1]],  # xmin, ymax
                          [limits[0, 0], 0, limits[1, 1]],  # xmin, ymax
                          [limits[0, 0], 0, limits[1, 0]]  # xmin, ymin
                          ])

        axisGridArray = []

        for iAx in [0, 1]:
            # need to convert to model coordinates
            # dx1 will be a vector.
            dx1 = np.abs(limits[iAx][0] - limits[iAx][1]) * 1.1

            order = np.floor(np.log10(dx1))
            m1 = dx1 * 10**-order

            if (m1 >= 1) and (m1 < 2):
                step = 0.2 * 10**order
            elif (m1 >= 2) and (m1 < 4):
                step = 0.5 * 10**order
            else:
                step = 10**order
            if step < 1:
                decimalX = int(np.abs(order)) + 1 if m1 < 4 else\
                    int(np.abs(order))
            else:
                decimalX = 0

            step *= 0.2  # fine step
            gridX = np.arange(np.int32(limits[iAx][0]/step)*step,
                              limits[iAx][1], step)
            gridX = gridX if gridX[0] >= limits[iAx][0] else\
                gridX[1:]
            gridLabels.extend([gridX])
            precisionLabels.extend([np.ones_like(gridX)*decimalX])
            axisGridArray.extend([gridX])

        xPoints = np.array(axisGridArray[0])
        yPoints = np.array(axisGridArray[1])
        col_x = np.vstack((np.ones_like(yPoints)*limits[0][0],
                           np.ones_like(yPoints)*limits[0][1])).flatten('F')

        col_y = np.vstack((yPoints, yPoints)).flatten('F')

        vertices = np.vstack((frame, np.column_stack((
                col_x, np.zeros_like(col_x), col_y))))

        col_y = np.vstack((np.ones_like(xPoints)*limits[1][0],
                           np.ones_like(xPoints)*limits[1][1])).flatten('F')
        col_x = np.vstack((xPoints, xPoints)).flatten('F')
        vertices = np.vstack((vertices, np.column_stack((
                col_x, np.zeros_like(col_x), col_y))))

        return vertices, gridLabels, precisionLabels

    def make_frame(self, limits):
        back = np.array([[-limits[0], limits[1], -limits[2]],
                         [-limits[0], limits[1], limits[2]],
                         [-limits[0], limits[1], limits[2]],
                         [-limits[0], -limits[1], limits[2]],
                         [-limits[0], -limits[1], limits[2]],
                         [-limits[0], -limits[1], -limits[2]],
                         [-limits[0], -limits[1], -limits[2]],
                         [-limits[0], limits[1], -limits[2]]])

        side = np.array([[limits[0], -limits[1], -limits[2]],
                         [-limits[0], -limits[1], -limits[2]],
                         [-limits[0], -limits[1], -limits[2]],
                         [-limits[0], -limits[1], limits[2]],
                         [-limits[0], -limits[1], limits[2]],
                         [limits[0], -limits[1], limits[2]],
                         [limits[0], -limits[1], limits[2]],
                         [limits[0], -limits[1], -limits[2]]])

        bottom = np.array([[limits[0], -limits[1], -limits[2]],
                           [limits[0], limits[1], -limits[2]],
                           [limits[0], limits[1], -limits[2]],
                           [-limits[0], limits[1], -limits[2]],
                           [-limits[0], limits[1], -limits[2]],
                           [-limits[0], -limits[1], -limits[2]],
                           [-limits[0], -limits[1], -limits[2]],
                           [limits[0], -limits[1], -limits[2]]])

        back[:, 0] *= self.axPosModifier[0]
        side[:, 1] *= self.axPosModifier[1]
        bottom[:, 2] *= self.axPosModifier[2]
        self.halfCube = np.float32(np.vstack((back, side, bottom)))

    def make_coarse_grid(self):

        self.gridLabels = []
        self.precisionLabels = []
        #  Calculating regular grids in world coordinates
        limits = np.array([-1, 1])[:, np.newaxis] * np.array(self.parent.aPos)
        #  allLimits -> in model coordinates
        allLimits = limits * self.parent.maxLen / self.parent.scaleVec -\
            self.parent.tVec + self.parent.coordOffset
        axisGridArray = []

        for iAx in range(3):
            m2 = self.parent.aPos[iAx] / 0.9
            dx1 = np.abs(allLimits[:, iAx][0] - allLimits[:, iAx][1]) / m2
            order = np.floor(np.log10(dx1))
            m1 = dx1 * 10**-order

            if (m1 >= 1) and (m1 < 2):
                step = 0.2 * 10**order
            elif (m1 >= 2) and (m1 < 4):
                step = 0.5 * 10**order
            else:
                step = 10**order
            if step < 1:
                decimalX = int(np.abs(order)) + 1 if m1 < 4 else\
                    int(np.abs(order))
            else:
                decimalX = 0

            gridX = np.arange(np.int32(allLimits[:, iAx][0]/step)*step,
                              allLimits[:, iAx][1], step)
            gridX = gridX if gridX[0] >= allLimits[:, iAx][0] else\
                gridX[1:]
            self.gridLabels.extend([gridX])
            self.precisionLabels.extend([np.ones_like(gridX)*decimalX])
            axisGridArray.extend([gridX - self.parent.coordOffset[iAx]])
#            if self.parent.fineGridEnabled:
#                fineStep = step * 0.2
#                fineGrid = np.arange(
#                    np.int32(allLimits[:, iAx][0]/fineStep)*fineStep,
#                    allLimits[:, iAx][1], fineStep)
#                fineGrid = fineGrid if\
#                    fineGrid[0] >= allLimits[:, iAx][0] else fineGrid[1:]
#                fineGridArray.extend([fineGrid - self.parent.coordOffset[iAx]])

        self.axisL, self.axGrid = self.populate_grid(axisGridArray)
        self.gridLen = len(self.axGrid)

#        for iAx in range(3):
#            if not (not self.perspectiveEnabled and
#                    iAx == self.parent.visibleAxes[2]):
#
#                midp = int(len(self.axisL[iAx][0, :])/2)
#                if iAx == self.parent.visibleAxes[1]:  # Side plane,
#                    print(self.axisL[iAx][:, midp], self.parent.visibleAxes[0])
##                    if self.useScalableFont:
##                        tAlign = getAlignment(axisL[iAx][:, midp],
##                                              self.visibleAxes[0])
##                    else:
#                    self.axisL[iAx][self.parent.visibleAxes[2], :] *= 1.05  # depth
#                    self.axisL[iAx][self.parent.visibleAxes[0], :] *= 1.05  # side
#                if iAx == self.parent.visibleAxes[0]:  # Bottom plane, left-right
##                    if self.useScalableFont:
##                        tAlign = getAlignment(axisL[iAx][:, midp],
##                                              self.visibleAxes[2],
##                                              self.visibleAxes[1])
##                    else:
#                    self.axisL[iAx][self.parent.visibleAxes[1], :] *= 1.05  # height
#                    self.axisL[iAx][self.parent.visibleAxes[2], :] *= 1.05  # side
#                if iAx == self.parent.visibleAxes[2]:  # Bottom plane, left-right
##                    if self.useScalableFont:
##                        tAlign = getAlignment(axisL[iAx][:, midp],
##                                              self.visibleAxes[0],
##                                              self.visibleAxes[1])
##                    else:
#                    self.axisL[iAx][self.parent.visibleAxes[1], :] *= 1.05  # height
#                    self.axisL[iAx][self.parent.visibleAxes[0], :] *= 1.05  # side

    def update_grid(self):
        if hasattr(self, "vbo_frame"):
            self.make_frame(self.parent.aPos)
            update_qt_buffer(self.vbo_frame, self.halfCube)

        if hasattr(self, "vbo_grid"):
            self.make_coarse_grid()
#            if self.gridLen < self.initialGridLen:  # initial grid size x10
#            print("Gridlen", self.gridLen)
            update_qt_buffer(self.vbo_grid, self.axGrid)
#            else:
#                print("That'll be a humongous grid")
#                print(self.initialGridLen)
#                self.vbo_grid.destroy()
#                gl.glGetError()
#                self.vaoGrid.destroy()
#                gl.glGetError()
#                self.vaoGrid = qt.QOpenGLVertexArrayObject()
#                self.vaoGrid.create()
#                self.initialGridLen = self.gridLen*10
#                self.vbo_grid = create_qt_buffer(np.tile(self.axGrid, 10).copy())
#                gl.glGetError()
#                self.vaoGrid.bind()
#                self.vbo_grid.bind()
#                gl.glGetError()
#                gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
#                gl.glEnableVertexAttribArray(0)
#                gl.glGetError()
#                self.vbo_grid.release()
#                self.vaoGrid.release()

    def prepare_grid(self):
        self.make_font()

        self.make_frame(self.parent.aPos)

        self.make_coarse_grid()
        self.initialGridLen = self.gridLen*10
#        print("Initial grid", self.initialGridLen)
#        if self.parent.fineGridEnabled:
#            fineGridArray = []
#        print(axisL)
#        if self.parent.fineGridEnabled:
#            tmp, fineAxGrid = self.populateGrid(fineGridArray)
#            self.fineGridLen = len(fineAxGrid)
#            self.vaoFineGrid.bind()
#            self.vbo_fineGrid = self.setVertexBuffer(fineAxGrid, 3, self.shader, "position" )
#            self.vaoFineGrid.release()

#        cLines = np.array([[-self.parent.aPos[0], 0, 0],
#                           [self.parent.aPos[0], 0, 0],
#                           [0, -self.parent.aPos[1], 0],
#                           [0, self.parent.aPos[1], 0],
#                           [0, 0, -self.parent.aPos[2]],
#                           [0, 0, self.parent.aPos[2]]])*0.5
#
#        cLineColors = np.array([[0, 0.5, 1],
#                                [0, 0.5, 1],
#                                [0, 0.9, 0],
#                                [0, 0.9, 0],
#                                [0.8, 0, 0],
#                                [0.8, 0, 0]])

        self.vbo_frame = create_qt_buffer(self.halfCube.copy())
        self.vaoFrame.bind()
        self.vbo_frame.bind()
        gl.glGetError()
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        gl.glGetError()
        self.vbo_frame.release()
        self.vaoFrame.release()

        self.vbo_grid = create_qt_buffer(np.tile(self.axGrid, 10).copy())
        self.vaoGrid.bind()
        self.vbo_grid.bind()
        gl.glGetError()
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        gl.glGetError()
        self.vbo_grid.release()
        self.vaoGrid.release()
        gl.glGetError()

#        self.vaoOrigin.bind()
#        self.vbo_origin = setVertexBuffer(
#                cLines, 3, self.origShader, "position")
#        self.vbo_oc = setVertexBuffer(
#                cLineColors, 3, self.origShader, "linecolor")
#        self.vaoOrigin.release()
        # TODO: Move font init outside
        # x  y  u  v
        vquad = np.array([
            0, 1, 0, 0,
            0,  0, 0, 1,
            1,  0, 1, 1,
            0, 1, 0, 0,
            1,  0, 1, 1,
            1, 1, 1, 0
        ])
        self.vbo_Text = create_qt_buffer(vquad.copy())
        self.vaoText.bind()
        self.vbo_Text.bind()
        gl.glGetError()
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        gl.glGetError()
        self.vbo_Text.release()
        self.vaoText.release()
        gl.glGetError()

    def prepare_arrows(self, z0, z, r, nSegments):
        phi = np.linspace(0, 2*np.pi, nSegments)
        xp = r * np.cos(phi)
        yp = r * np.sin(phi)
        base = np.vstack((xp, yp, np.ones_like(xp)*(z0-z), np.ones_like(xp)))
        coneVertices = np.vstack((np.array([[0, 0, 0, 1], [0, 0, z0, 1]]),
                                  base.T))
        self.arrows = coneVertices.copy()

        for rotation in [self.z2y, self.z2x]:
            m3rot = np.array(rotation.data()).reshape(4, 4)
            self.arrows = np.vstack((self.arrows,
                                     np.matmul(coneVertices, m3rot.T)))
        self.arrowLen = len(coneVertices)
        self.vbo_arrows = create_qt_buffer(self.arrows)
        colorArr = None
        for line in range(3):
            oneColor = np.tile(np.identity(3)[line, :],
                               self.arrowLen)
            colorArr = np.vstack(
                    (colorArr, oneColor)) if colorArr is not None else oneColor
        self.vbo_arr_colors = create_qt_buffer(colorArr)

        vao = qt.QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()
        gl.glGetError()
        self.vbo_arrows.bind()
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        self.vbo_arrows.release()

        self.vbo_arr_colors.bind()
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)
        self.vbo_arr_colors.release()

        vao.release()
        self.vao_arrow = vao

    def populate_grid(self, grids):
        pModel = np.array(self.parent.mView.data()).reshape(4, 4)[:-1, :-1]
#                print(pModel)
#        self.visibleAxes = np.argmax(np.abs(pModel), axis=0)
        self.signs = np.sign(pModel)
#                self.axPosModifier = np.ones(3)
        for iAx in range(3):
            self.axPosModifier[iAx] = (self.signs[iAx][2] if
                                       self.signs[iAx][2] != 0 else 1)
        axisLabelC = []
        axisLabelC.extend([np.vstack(
            (self.parent.modelToWorld(grids, 0),
             np.ones(len(grids[0]))*self.parent.aPos[1]*self.axPosModifier[1],
             np.ones(len(grids[0]))*-self.parent.aPos[2]*self.axPosModifier[2]
             ))])
        axisLabelC.extend([np.vstack(
            (np.ones(len(grids[1]))*self.parent.aPos[0]*self.axPosModifier[0],
             self.parent.modelToWorld(grids, 1),
             np.ones(len(grids[1]))*-self.parent.aPos[2]*self.axPosModifier[2]
             ))])
        zAxis = np.vstack(
            (np.ones(len(grids[2]))*-self.parent.aPos[0]*self.axPosModifier[0],
             np.ones(len(grids[2]))*self.parent.aPos[1]*self.axPosModifier[1],
             self.parent.modelToWorld(grids, 2)))

        xAxisB = np.vstack(
            (self.parent.modelToWorld(grids, 0),
             np.ones(len(grids[0]))*-self.parent.aPos[1]*self.axPosModifier[1],
             np.ones(len(grids[0]))*-self.parent.aPos[2]*self.axPosModifier[2]))
        yAxisB = np.vstack(
            (np.ones(len(grids[1]))*-self.parent.aPos[0]*self.axPosModifier[0],
             self.parent.modelToWorld(grids, 1),
             np.ones(len(grids[1]))*-self.parent.aPos[2]*self.axPosModifier[2]))
        zAxisB = np.vstack(
            (np.ones(len(grids[2]))*-self.parent.aPos[0]*self.axPosModifier[0],
             np.ones(len(grids[2]))*-self.parent.aPos[1]*self.axPosModifier[1],
             self.parent.modelToWorld(grids, 2)))

        xAxisC = np.vstack(
            (self.parent.modelToWorld(grids, 0),
             np.ones(len(grids[0]))*-self.parent.aPos[1]*self.axPosModifier[1],
             np.ones(len(grids[0]))*self.parent.aPos[2]*self.axPosModifier[2]))
        yAxisC = np.vstack(
            (np.ones(len(grids[1]))*-self.parent.aPos[0]*self.axPosModifier[0],
             self.parent.modelToWorld(grids, 1),
             np.ones(len(grids[1]))*self.parent.aPos[2]*self.axPosModifier[2]))
        axisLabelC.extend([np.vstack(
            (np.ones(len(grids[2]))*self.parent.aPos[0]*self.axPosModifier[0],
             np.ones(len(grids[2]))*-self.parent.aPos[1]*self.axPosModifier[1],
             self.parent.modelToWorld(grids, 2)))])

        xLines = np.vstack(
            (axisLabelC[0], xAxisB, xAxisB, xAxisC)).T.flatten().reshape(
            4*xAxisB.shape[1], 3)
        yLines = np.vstack(
            (axisLabelC[1], yAxisB, yAxisB, yAxisC)).T.flatten().reshape(
            4*yAxisB.shape[1], 3)
        zLines = np.vstack(
            (zAxis, zAxisB, zAxisB, axisLabelC[2])).T.flatten().reshape(
            4*zAxisB.shape[1], 3)

        return axisLabelC, np.float32(np.vstack((xLines, yLines, zLines)))

    def render_grid(self, model, view, projection):

        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        self.shader.bind()
        self.shader.setUniformValue("pvm", projection*view*model)
        self.shader.setUniformValue("lineColor", self.parent.lineColor)

        self.vaoFrame.bind()
        self.shader.setUniformValue("lineOpacity", 0.75)
        gl.glLineWidth(min(self.parent.cBoxLineWidth * 2, 1.))
        gl.glDrawArrays(gl.GL_LINES, 0, 24)
        self.vaoFrame.release()

        self.vaoGrid.bind()
        self.shader.setUniformValue("lineOpacity", 0.5)

        gl.glLineWidth(min(self.parent.cBoxLineWidth, 1.))
        gl.glDrawArrays(gl.GL_LINES, 0, self.gridLen)
        self.vaoGrid.release()

#        if self.parent.fineGridEnabled:
#            self.vaoFineGrid.bind()
#            self.shader.setUniformValue("lineOpacity", 0.25)
#            gl.glLineWidth(self.parent.cBoxLineWidth)
#            gl.glDrawArrays(gl.GL_LINES, 0, self.fineGridLen)
#            self.vaoFineGrid.release()
        self.shader.release()

#        self.origShader.bind()
#        self.origShader.setUniformValue("pvm", projection*view*model)
#        self.vaoOrigin.bind()
#        self.origShader.setUniformValue("lineOpacity", 0.85)
#        gl.glLineWidth(self.parent.cBoxLineWidth)
#        gl.glDrawArrays(gl.GL_LINES, 0, 6)
#        self.vaoOrigin.release()
#        self.origShader.release()

        self.textShader.bind()
        self.vaoText.bind()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
#        gl.glEnable(gl.GL_POLYGON_SMOOTH)
#        gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
        vpMat = projection*view
        for iAx in range(3):
            if not (not self.perspectiveEnabled and
                    iAx == self.parent.visibleAxes[2]):

                midp = int(len(self.axisL[iAx][0, :])/2)
                p0 = self.axisL[iAx][:, midp]
                alignment = None
                if iAx == self.parent.visibleAxes[1]:  # Side plane,
                    alignment = self.getAlignment(vpMat, p0,
                                                  self.parent.visibleAxes[0])
                if iAx == self.parent.visibleAxes[0]:  # Bottom plane, L-R
                    alignment = self.getAlignment(vpMat, p0,
                                                  self.parent.visibleAxes[2],
                                                  self.parent.visibleAxes[1])
                if iAx == self.parent.visibleAxes[2]:  # Bottom plane, L-R
                    alignment = self.getAlignment(vpMat, p0,
                                                  self.parent.visibleAxes[0],
                                                  self.parent.visibleAxes[1])

                for tick, tText, pcs in list(zip(self.axisL[iAx].T,
                                                 self.gridLabels[iAx],
                                                 self.precisionLabels[iAx])):
                    valueStr = "{0:.{1}f}".format(tText, int(pcs))
                    tickPos = (vpMat*qt.QVector4D(*tick, 1)).toVector3DAffine()
                    self.render_text(tickPos, valueStr, alignment=alignment,
                                     scale=0.04*self.fontScale,
                                     textColor=self.parent.lineColor)
        self.vaoText.release()
        self.textShader.release()

    def render_text(self, pos, text, alignment, scale, textColor=None):
        tcValue = textColor or qt.QVector3D(1, 1, 1)
        self.textShader.setUniformValue("textColor", tcValue)
        self.textShader.setUniformValue("textOpacity", 0.75)
        char_x = 0
        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
        scaleX = scale/float(pView[2])
        scaleY = scale/float(pView[3])
        coordShift = np.zeros(2, dtype=np.float32)

        aw = []
        ah = []
        axrel = []
        ayrel = []

        for c in text:
            c = ord(c)
            ch = self.characters[c]
            w, h = ch[1][0] * scaleX, ch[1][1] * scaleY
            xrel = char_x + ch[2][0] * scaleX
            yrel = (ch[1][1] - ch[2][1]) * scaleY
            if c == 45:
                yrel = ch[1][0] * scaleY
            char_x += (ch[3] >> 6) * scaleX
            aw.append(w)
            ah.append(h)
            axrel.append(xrel)
            ayrel.append(yrel)

        if alignment is not None:
            if alignment[0] == 'left':
                coordShift[0] = -(axrel[-1]+2*aw[-1])
            else:
                coordShift[0] = 2*aw[-1]

            if alignment[1] == 'top':
                vOffset = 0.5
            elif alignment[1] == 'bottom':
                vOffset = -2
            else:
                vOffset = -1
            coordShift[1] = vOffset*ah[-1]

        for ic, c in enumerate(text):
            c = ord(c)
            ch = self.characters[c]
            if ch[1] == (0, 0):
                continue
            mMod = qt.QMatrix4x4()
            mMod.setToIdentity()
            mMod.translate(pos)
            mMod.translate(axrel[ic]+coordShift[0], ayrel[ic]+coordShift[1], 0)
            mMod.scale(aw[ic], ah[ic], 1)
            ch[0].bind()
            self.textShader.setUniformValue("model", mMod)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            ch[0].release()
        return mMod*qt.QVector4D(1.0, 0.0, 0.0, 1.0)

    def render_local_axes(self, moe, trans, view, proj, shader, isScreen):

        mRotation = qt.QMatrix4x4()

        if moe is not None:
            moe_np = np.array(moe.data()).reshape((4, 4), order=('F'))

            if isScreen:
                bStart = np.column_stack(([1, 0, 0], [0, 0, 1], [0, -1, 0]))
                x = np.matmul(moe_np, np.array([1, 0, 0, 0]))[:-1]
                y = np.matmul(moe_np, np.array([0, -1, 0, 0]))[:-1]
            else:
                bStart = np.column_stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
                x = np.matmul(moe_np, np.array([1, 0, 0, 0]))[:-1]
                y = np.matmul(moe_np, np.array([0, 1, 0, 0]))[:-1]

            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)
            z = np.cross(x, y)
            z = z / np.linalg.norm(z)

            bEnd = np.column_stack((x, y, z))
            rotationQ = basis_rotation_q(bStart, bEnd)

            mRotation.translate(trans)
            mRotation.rotate(qt.QQuaternion(*rotationQ))

        shader.setUniformValue("pvm", proj*view*mRotation)

        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 1, self.arrowLen-1)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, self.arrowLen+1, self.arrowLen-1)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, self.arrowLen*2+1, self.arrowLen-1)

        gl.glDrawArrays(gl.GL_LINES, 0, 2)
        gl.glDrawArrays(gl.GL_LINES, self.arrowLen, 2)
        gl.glDrawArrays(gl.GL_LINES, self.arrowLen*2, 2)

    def get_sans_font(self):
        fallback_fonts = ["Arial", "Helvetica", "DejaVu Sans",
                          "Liberation Sans", "Sans-serif"]

        available_fonts = [font_manager.FontProperties(fname=path).get_name()
                           for path in font_manager.findSystemFonts()]

        for font in fallback_fonts:
            if font in available_fonts:
                return font

        return self.parent.scalableFontType

    def make_font(self):
        try:
            fontName = self.get_sans_font()
            font_path = font_manager.findfont(fontName)
        except Exception:  # TODO: track exceptions
            fontpath = os.path.dirname(__file__)
            font_path = os.path.join(fontpath, self.fontFile)

        face = Face(font_path)
        face.set_pixel_sizes(self.fontSize*8, self.fontSize*8)

        for c in range(128):
            face.load_char(chr(c), FT_LOAD_RENDER)
            glyph = face.glyph
            bitmap = glyph.bitmap
            size = bitmap.width, bitmap.rows
            bearing = glyph.bitmap_left, 2 * bitmap.rows - glyph.bitmap_top
            advance = glyph.advance.x

            qi = qt.QImage(np.array(bitmap.buffer, dtype=np.uint8),
                           int(bitmap.width), int(bitmap.rows),
                           int(bitmap.width),
                           qt.QImage.Format_Grayscale8)
            texObj = qt.QOpenGLTexture(qi)
            texObj.setMinificationFilter(qt.QOpenGLTexture.LinearMipMapLinear)
            texObj.setMagnificationFilter(qt.QOpenGLTexture.Linear)
            texObj.generateMipMaps()
            self.characters.append((texObj, size, bearing, advance))

    def getAlignment(self, pvMatr, point, hDim, vDim=None):
        pointH = np.copy(point)
        pointV = np.copy(point)

        sp0 = pvMatr * qt.QVector4D(*point, 1)
        pointH[hDim] *= 1.1
        spH = pvMatr * qt.QVector4D(*pointH, 1)

        if vDim is None:
            vAlign = 'middle'
        else:
            pointV[vDim] *= 1.1
            spV = pvMatr * qt.QVector4D(*pointV, 1)
            vAlign = 'top' if spV[1] - sp0[1] > 0 else 'bottom'
        hAlign = 'left' if spH[0] - sp0[0] < 0 else 'right'
        return (hAlign, vAlign)